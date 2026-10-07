#include "nyx/recovery/ssa/constants.hpp"

#include <algorithm>
#include <bit>
#include <limits>

namespace nyx::recovery {
namespace {
SsaConstantFoldResult Decline(SsaConstantFoldRefusal reason) {
  SsaConstantFoldResult result;
  result.reason = reason;
  return result;
}

SsaConstantFoldResult Fold(const ir::SsaGraph& original, std::span<const std::uint8_t> executable,
                           const ir::SsaConstantFacts& facts, std::span<const ir::Group> sources,
                           bool read_only, Budget& budget) {
  SsaConstantFoldResult result;

  // Which edits replace an image location. Under a declared bias its value is
  // a number, but a transfer target or load address still has to name a place
  // in the image, so it becomes the location literal rather than the number.
  std::vector<std::uint8_t> located;

  // Per block, which nodes the dispatch condition decides. Writing a value
  // into one leaves the block's two successor records disagreeing.
  std::optional<std::uint32_t> dispatch_slot;
  std::vector<std::uint8_t> dispatch_dependent;
  bool retired_loop = false;
  for (const auto& fact : facts.values) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return Decline(SsaConstantFoldRefusal::resource_limit);
    if (fact.kind != ir::SsaValueKind::node) continue;
    const auto& block = *original.Get(fact.block);
    if (dispatch_slot != fact.block.slot) {
      auto dependent = ir::SsaDispatchDependent(block, fact.block, budget);
      if (!dependent) return Decline(SsaConstantFoldRefusal::resource_limit);
      dispatch_dependent = std::move(*dependent);
      dispatch_slot = fact.block.slot;
      retired_loop = std::any_of(block.control_rewrites.begin(), block.control_rewrites.end(),
                                 [](const ir::ConditionalRewrite& rewrite) {
                                   return rewrite.rule == ir::RewriteRule::bounded_exit;
                                 });
    }

    // A retired loop's nodes still say what its iterations computed, and its
    // record which way its condition went; what one run leaves in them is dead
    // and folding it would only make the record contradict itself.
    if (retired_loop) continue;
    if (fact.index < dispatch_dependent.size() && dispatch_dependent[fact.index]) continue;
    const auto& node = block.nodes[fact.index];
    const auto* descriptor = ir::Descriptor(node.op);
    if (budget.try_consume(
            {static_cast<std::uint64_t>(std::bit_width(block.dead_pure_nodes.size())), 0}) !=
        BudgetDecline::none)
      return Decline(SsaConstantFoldRefusal::resource_limit);
    const bool read = node.op == ir::Op::read;
    if (!descriptor ||
        (read_only ? !read
                   : (descriptor->effect != ir::Effect::pure || !descriptor->produces_value ||
                      !descriptor->arity)) ||
        std::binary_search(block.dead_pure_nodes.begin(), block.dead_pure_nodes.end(), fact.index))
      continue;
    std::optional<ir::SsaRead> removed_read;
    if (read) {
      if (budget.try_consume({block.reads.size(), 0}) != BudgetDecline::none)
        return Decline(SsaConstantFoldRefusal::resource_limit);
      const auto record =
          std::find_if(block.reads.begin(), block.reads.end(),
                       [&](const ir::SsaRead& item) { return item.node == fact.index; });
      if (record == block.reads.end()) return Decline(SsaConstantFoldRefusal::invalid_graph);
      removed_read = *record;
    }

    if (budget.try_consume({1, sizeof(SsaConstantFoldEdit)}) != BudgetDecline::none)
      return Decline(SsaConstantFoldRefusal::resource_limit);
    if (budget.try_consume({64, 1}) != BudgetDecline::none)
      return Decline(SsaConstantFoldRefusal::resource_limit);
    // A register that only ever carries a location is spelled as one too, or
    // a store indexed off it would read as a number plus an offset, which an
    // image store check cannot tell from any other pointer.
    bool place = original.load_bias() && fact.value >= *original.load_bias() && node.width == 64;
    if (place && read) {
      const auto based = ir::SsaImageBasedValue(
          original, {ir::SsaValueKind::node, fact.block, fact.index}, budget);
      if (!based) return Decline(SsaConstantFoldRefusal::resource_limit);
      place = *based;
    } else if (place) {
      place = ir::SsaImageLocation(block.nodes, fact.index, std::nullopt).has_value();
    }

    result.journal.push_back(
        {fact.block, {}, fact.index, node, fact.value, original.revision(), 0, removed_read});
    located.push_back(place);
  }

  if (result.journal.empty()) return result;
  auto candidate = original.Clone(budget);
  if (!candidate) return Decline(SsaConstantFoldRefusal::resource_limit);
  for (std::size_t first = 0; first < result.journal.size();) {
    const auto slot = result.journal[first].original_block.slot;
    const auto handle = candidate->Handle(slot);
    if (!handle) return Decline(SsaConstantFoldRefusal::invalid_graph);
    auto copied = candidate->CopyBlock(*handle, budget);
    if (!copied) return Decline(SsaConstantFoldRefusal::resource_limit);
    auto block = std::move(*copied);
    do {
      const bool place = located[first];
      auto& edit = result.journal[first++];
      block.nodes[edit.node] =
          place ? ir::Node{ir::Op::image_address, 64, {}, edit.value - *original.load_bias()}
                : ir::Node{ir::Op::constant, edit.original.width, {}, edit.value};
      if (edit.removed_read) {
        if (block.reads.size() > std::numeric_limits<std::uint64_t>::max() / 2 ||
            budget.try_consume({2 * block.reads.size(), 0}) != BudgetDecline::none)
          return Decline(SsaConstantFoldRefusal::resource_limit);
        const auto record =
            std::find_if(block.reads.begin(), block.reads.end(),
                         [&](const ir::SsaRead& item) { return item.node == edit.node; });
        if (record == block.reads.end()) return Decline(SsaConstantFoldRefusal::invalid_graph);
        block.reads.erase(record);
      }

      edit.result_block = *handle;
    } while (first < result.journal.size() && result.journal[first].original_block.slot == slot);
    if (!candidate->Replace(*handle, std::move(block)))
      return Decline(SsaConstantFoldRefusal::invalid_graph);
  }

  // A value written in can place a store the graph could not place before;
  // a declared value that store writes no longer holds, so its record goes.
  auto dropped = ir::SsaDropWrittenPathReads(*candidate, budget);
  if (!dropped) return Decline(SsaConstantFoldRefusal::resource_limit);
  result.dropped_path_reads = std::move(*dropped);
  const auto checked_candidate = ir::ValidateSsaWithSources(*candidate, sources, budget);
  if (checked_candidate != ir::SsaDecline::none)
    return Decline(checked_candidate == ir::SsaDecline::resource_limit
                       ? SsaConstantFoldRefusal::resource_limit
                       : SsaConstantFoldRefusal::invalid_graph);
  for (std::size_t slot = 0; slot < candidate->slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return Decline(SsaConstantFoldRefusal::resource_limit);
    if (!executable[slot]) continue;
    const auto handle = candidate->Handle(slot);
    if (!handle) return Decline(SsaConstantFoldRefusal::invalid_graph);
    const auto complete = ir::ValidateSsaDirectSuccessors(*candidate->Get(*handle), budget);
    if (complete != ir::SsaDecline::none)
      return Decline(complete == ir::SsaDecline::resource_limit
                         ? SsaConstantFoldRefusal::resource_limit
                         : SsaConstantFoldRefusal::invalid_graph);
  }

  for (auto& edit : result.journal) edit.to_revision = candidate->revision();
  result.provisional = std::move(*candidate);
  return result;
}
}  // namespace

SsaConstantFoldResult ProposeSsaConstantFold(const ir::SsaGraph& original,
                                             const ir::SsaReachabilityFacts& reachable,
                                             const ir::SsaConstantFacts& facts,
                                             std::span<const ir::Group> sources, Budget& budget,
                                             const ir::SsaBoundedLoopFacts* loops) {
  const auto valid = ir::ValidateSsaWithSources(original, sources, budget);
  if (valid != ir::SsaDecline::none)
    return Decline(valid == ir::SsaDecline::resource_limit ? SsaConstantFoldRefusal::resource_limit
                                                           : SsaConstantFoldRefusal::invalid_graph);
  const auto checked =
      ir::ValidateSsaConstantFacts(original, reachable, facts, sources, budget, loops);
  if (checked != ir::SsaDecline::none)
    return Decline(checked == ir::SsaDecline::resource_limit
                       ? SsaConstantFoldRefusal::resource_limit
                       : SsaConstantFoldRefusal::stale_proof);
  return Fold(original, reachable.reachable, facts, sources, false, budget);
}

SsaConstantFoldResult ProposeSsaSccpFold(const ir::SsaGraph& original,
                                         const ir::SsaSccpFacts& facts,
                                         std::span<const ir::Group> sources, Budget& budget,
                                         const ir::SsaBoundedLoopFacts* loops) {
  const auto valid = ir::ValidateSsaWithSources(original, sources, budget);
  if (valid != ir::SsaDecline::none)
    return Decline(valid == ir::SsaDecline::resource_limit ? SsaConstantFoldRefusal::resource_limit
                                                           : SsaConstantFoldRefusal::invalid_graph);
  const auto checked = ir::ValidateSsaSccpFacts(original, facts, sources, budget, loops);
  if (checked != ir::SsaDecline::none)
    return Decline(checked == ir::SsaDecline::resource_limit
                       ? SsaConstantFoldRefusal::resource_limit
                       : SsaConstantFoldRefusal::stale_proof);
  return Fold(original, facts.executable, facts.constants, sources, false, budget);
}

SsaConstantFoldResult ProposeSsaSccpReadFold(const ir::SsaGraph& original,
                                             const ir::SsaSccpFacts& facts,
                                             std::span<const ir::Group> sources, Budget& budget,
                                             const ir::SsaBoundedLoopFacts* loops) {
  const auto valid = ir::ValidateSsaWithSources(original, sources, budget);
  if (valid != ir::SsaDecline::none)
    return Decline(valid == ir::SsaDecline::resource_limit ? SsaConstantFoldRefusal::resource_limit
                                                           : SsaConstantFoldRefusal::invalid_graph);
  const auto checked = ir::ValidateSsaSccpFacts(original, facts, sources, budget, loops);
  if (checked != ir::SsaDecline::none)
    return Decline(checked == ir::SsaDecline::resource_limit
                       ? SsaConstantFoldRefusal::resource_limit
                       : SsaConstantFoldRefusal::stale_proof);
  return Fold(original, facts.executable, facts.constants, sources, true, budget);
}

}  // namespace nyx::recovery
