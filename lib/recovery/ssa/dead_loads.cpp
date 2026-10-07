#include "nyx/recovery/ssa/dead_loads.hpp"

#include <algorithm>
#include <bit>
#include <limits>

namespace nyx::recovery {
namespace {
using ir::Op;
using ir::ValueId;

SsaDeadLoadResult Decline(SsaDeadLoadRefusal reason) {
  SsaDeadLoadResult result;
  result.reason = reason;
  return result;
}

bool Listed(std::span<const ValueId> sorted, ValueId id) {
  return std::binary_search(sorted.begin(), sorted.end(), id);
}

// Loads another record already accounts for, or whose effect is shared with
// one: they keep whatever that record decided.
bool Accounted(const ir::SsaBlock& block, ValueId id) {
  return Listed(block.disabled_effects, id) ||
         std::any_of(block.paired_load_omissions.begin(), block.paired_load_omissions.end(),
                     [&](const ir::PairedLoadOmission& pair) {
                       return pair.loads[0] == id || pair.loads[1] == id;
                     });
}

// The latest earlier store that executes and writes every byte the load reads.
std::optional<ValueId> WrittenBefore(const ir::SsaBlock& block, ValueId load) {
  const auto& nodes = block.nodes;
  for (ValueId id = load; id-- > 0;) {
    const auto& node = nodes[id];
    if (node.op != Op::store || Listed(block.disabled_effects, id) ||
        std::any_of(block.store_omissions.begin(), block.store_omissions.end(),
                    [&](const ir::StoreOmission& omission) { return omission.store == id; }))
      continue;
    if (ir::SsaSameValue(block, node.inputs[0], nodes[load].inputs[0]) &&
        nodes[node.inputs[1]].width >= nodes[load].width &&
        nodes[load].access.alignment <= node.access.alignment)
      return id;
  }

  return std::nullopt;
}

}  // namespace

SsaDeadLoadResult ProposeSsaDeadLoads(const ir::SsaGraph& original,
                                      std::span<const ir::Group> sources, ir::ImageFacts facts,
                                      ir::ImageAccessContract access, Budget& budget) {
  const auto valid = sources.empty() ? ir::ValidateSsa(original, budget)
                                     : ir::ValidateSsaWithSources(original, sources, budget);
  if (valid != ir::SsaDecline::none)
    return Decline(valid == ir::SsaDecline::resource_limit ? SsaDeadLoadRefusal::resource_limit
                                                           : SsaDeadLoadRefusal::invalid_graph);
  const bool image = access.mapped_readable_lifetime && access.no_runtime_unmapping &&
                     access.ordinary_reads_unobservable;
  auto candidate = original.Clone(budget);
  if (!candidate) return Decline(SsaDeadLoadRefusal::resource_limit);
  SsaDeadLoadResult result;
  for (std::size_t slot = 0; slot < candidate->slots(); ++slot) {
    const auto handle = candidate->Handle(slot);
    if (!handle) continue;
    const auto source = original.Handle(slot);
    const auto& block = *original.Get(*source);
    if (block.opaque) continue;
    std::vector<ir::SsaRetiredLoad> retired;
    const auto live = ir::SsaLiveValues(original, *source, budget);
    if (!live) return Decline(SsaDeadLoadRefusal::resource_limit);
    for (ValueId id = 0; id < block.nodes.size(); ++id) {
      if (budget.try_consume({4 + id, 0}) != BudgetDecline::none)
        return Decline(SsaDeadLoadRefusal::resource_limit);
      const auto& node = block.nodes[id];
      if (node.op != Op::load || !node.width || node.width > 64 || node.width % 8 ||
          Accounted(block, id))
        continue;
      if ((*live)[id]) continue;
      if (const auto store = WrittenBefore(block, id)) {
        retired.push_back({id, ir::SsaRetiredLoadBasis::written_before, *store});
        continue;
      }

      if (!image) continue;
      const auto span = ir::SsaLoadImageSpan(block, id, original.load_bias());
      if (!span) continue;
      if (budget.try_consume({facts.constants.size() + std::bit_width(facts.pointers.size()), 0}) !=
          BudgetDecline::none)
        return Decline(SsaDeadLoadRefusal::resource_limit);
      const auto range = std::find_if(
          facts.constants.begin(), facts.constants.end(), [&](const ir::ConstantImageRange& item) {
            return span->first >= item.address &&
                   item.bytes.size() <= std::numeric_limits<std::uint64_t>::max() - item.address &&
                   span->second <= item.address + item.bytes.size();
          });
      if (range != facts.constants.end()) {
        retired.push_back({id, ir::SsaRetiredLoadBasis::declared_image, 0, range->address,
                           range->bytes.size(), access});
        continue;
      }

      // A relocated slot is image bytes too, whatever value the loader wrote.
      const auto slot =
          std::upper_bound(facts.pointers.begin(), facts.pointers.end(), span->first,
                           [](std::uint64_t address, const ir::RelocatedPointer& item) {
                             return address < item.address;
                           });
      if (slot == facts.pointers.begin()) continue;
      const auto& pointer = *std::prev(slot);
      if (pointer.address > std::numeric_limits<std::uint64_t>::max() - 8 ||
          span->second > pointer.address + 8)
        continue;
      retired.push_back(
          {id, ir::SsaRetiredLoadBasis::declared_image, 0, pointer.address, 8, access});
    }

    if (retired.empty()) continue;
    if (budget.try_consume({retired.size() * 4,
                            retired.size() * (sizeof(SsaDeadLoadEdit) + sizeof(ir::SsaRetiredLoad) +
                                              sizeof(ValueId))}) != BudgetDecline::none)
      return Decline(SsaDeadLoadRefusal::resource_limit);
    auto copy = candidate->CopyBlock(*handle, budget);
    if (!copy) return Decline(SsaDeadLoadRefusal::resource_limit);
    for (const auto& item : retired) {
      copy->disabled_effects.insert(
          std::lower_bound(copy->disabled_effects.begin(), copy->disabled_effects.end(), item.node),
          item.node);
      copy->retired_loads.insert(
          std::lower_bound(
              copy->retired_loads.begin(), copy->retired_loads.end(), item.node,
              [](const ir::SsaRetiredLoad& load, ValueId node) { return load.node < node; }),
          item);
      result.journal.push_back({*source, *handle, item, original.revision(), 0});
    }

    if (!candidate->Replace(*handle, std::move(*copy)))
      return Decline(SsaDeadLoadRefusal::invalid_graph);
  }

  if (result.journal.empty()) return {};
  const auto checked = sources.empty() ? ir::ValidateSsa(*candidate, budget)
                                       : ir::ValidateSsaWithSources(*candidate, sources, budget);
  if (checked != ir::SsaDecline::none)
    return Decline(checked == ir::SsaDecline::resource_limit ? SsaDeadLoadRefusal::resource_limit
                                                             : SsaDeadLoadRefusal::invalid_graph);
  for (auto& edit : result.journal) edit.to_revision = candidate->revision();
  result.provisional = std::move(*candidate);
  return result;
}

}  // namespace nyx::recovery
