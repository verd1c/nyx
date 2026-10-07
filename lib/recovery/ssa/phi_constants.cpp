#include "nyx/recovery/ssa/phi_constants.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>

#include "nyx/ir/fold.hpp"

namespace nyx::recovery {
namespace {
struct Value {
  std::optional<std::uint64_t> number;
  bool from_phi = false;
};

SsaPhiFoldResult Decline(SsaPhiFoldRefusal reason) {
  SsaPhiFoldResult result;
  result.reason = reason;
  return result;
}
}  // namespace

SsaPhiFoldResult ProposeSsaPhiConstantFold(const ir::SsaGraph& original,
                                           const ir::SsaReachabilityFacts& reachable,
                                           const ir::SsaPhiConstantFacts& facts,
                                           std::span<const ir::Group> sources, Budget& budget) {
  const auto valid = ir::ValidateSsaWithSources(original, sources, budget);
  if (valid != ir::SsaDecline::none)
    return Decline(valid == ir::SsaDecline::resource_limit ? SsaPhiFoldRefusal::resource_limit
                                                           : SsaPhiFoldRefusal::invalid_graph);
  const auto checked = ir::ValidateSsaPhiConstantFacts(original, reachable, facts, sources, budget);
  if (checked != ir::SsaDecline::none)
    return Decline(checked == ir::SsaDecline::resource_limit ? SsaPhiFoldRefusal::resource_limit
                                                             : SsaPhiFoldRefusal::stale_proof);
  if (facts.constants.empty()) return {};
  SsaPhiFoldResult result;
  std::size_t fact_index = 0;
  for (std::size_t slot = 0; slot < original.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return Decline(SsaPhiFoldRefusal::resource_limit);
    const auto handle = original.Handle(slot);
    if (!handle || !reachable.reachable[slot]) continue;
    const auto& block = *original.Get(*handle);
    const auto node_work = 2 + std::bit_width(block.dead_pure_nodes.size());
    if (block.nodes.size() > std::numeric_limits<std::size_t>::max() / node_work ||
        block.phis.size() > std::numeric_limits<std::size_t>::max() / sizeof(Value) ||
        block.nodes.size() >
            std::numeric_limits<std::size_t>::max() / (sizeof(Value) + sizeof(std::size_t)) ||
        budget.try_consume({block.nodes.size() * node_work,
                            block.nodes.size() * (sizeof(Value) + sizeof(std::size_t))}) !=
            BudgetDecline::none ||
        budget.try_consume({block.phis.size(), block.phis.size() * sizeof(Value)}) !=
            BudgetDecline::none ||
        budget.try_consume({block.reads.size(), 0}) != BudgetDecline::none)
      return Decline(SsaPhiFoldRefusal::resource_limit);
    std::vector<Value> values(block.nodes.size());
    std::vector<Value> phis(block.phis.size());
    std::vector<std::size_t> read_at(block.nodes.size(), SIZE_MAX);
    for (std::size_t index = 0; index < block.reads.size(); ++index)
      read_at[block.reads[index].node] = index;
    while (fact_index < facts.constants.size() && facts.constants[fact_index].block.slot == slot) {
      const auto& fact = facts.constants[fact_index++];
      phis[fact.phi] = {fact.value, true};
    }

    for (std::size_t id = 0; id < block.nodes.size(); ++id) {
      const auto& node = block.nodes[id];
      auto& value = values[id];
      if (node.op == ir::Op::constant && node.width <= 64) {
        value.number = node.immediate & ir::LowMask(node.width);
      } else if (node.op == ir::Op::read) {
        const auto& read = block.reads[read_at[id]];
        if (read.value.kind == ir::SsaValueKind::phi)
          value = phis[read.value.index];
        else if (read.value.kind == ir::SsaValueKind::node)
          value = values[read.value.index];
      } else if (node.width <= 64) {
        const auto* descriptor = ir::Descriptor(node.op);
        if (!descriptor || descriptor->effect != ir::Effect::pure || !descriptor->produces_value ||
            !descriptor->arity)
          continue;
        std::array<std::uint64_t, 3> operands{};
        bool known = true;
        for (unsigned input = 0; input < descriptor->arity; ++input) {
          const auto& operand = values[node.inputs[input]];
          if (!operand.number)
            known = false;
          else
            operands[input] = *operand.number;
          value.from_phi |= operand.from_phi;
        }

        if (known) {
          value.number = ir::FoldPure(node, std::span(operands.data(), descriptor->arity),
                                      block.nodes[node.inputs[0]].width);
        } else if (node.op == ir::Op::select && values[node.inputs[0]].number) {
          const auto& selected = values[node.inputs[*values[node.inputs[0]].number ? 1 : 2]];
          value.number = selected.number;
          value.from_phi = values[node.inputs[0]].from_phi || selected.from_phi;
        }

        if (!value.number || !value.from_phi ||
            std::binary_search(block.dead_pure_nodes.begin(), block.dead_pure_nodes.end(), id))
          continue;
        if (budget.try_consume({1, sizeof(SsaPhiFoldEdit)}) != BudgetDecline::none)
          return Decline(SsaPhiFoldRefusal::resource_limit);
        result.journal.push_back({*handle,
                                  {},
                                  static_cast<ir::ValueId>(id),
                                  node,
                                  *value.number,
                                  original.revision(),
                                  0});
      }
    }
  }

  if (result.journal.empty()) return result;
  auto candidate = original.Clone(budget);
  if (!candidate) return Decline(SsaPhiFoldRefusal::resource_limit);
  for (std::size_t first = 0; first < result.journal.size();) {
    const auto slot = result.journal[first].original_block.slot;
    const auto handle = candidate->Handle(slot);
    if (!handle) return Decline(SsaPhiFoldRefusal::invalid_graph);
    auto copied = candidate->CopyBlock(*handle, budget);
    if (!copied) return Decline(SsaPhiFoldRefusal::resource_limit);
    auto block = std::move(*copied);
    do {
      auto& edit = result.journal[first++];
      block.nodes[edit.node] = {ir::Op::constant, edit.original.width, {}, edit.value};
      edit.result_block = *handle;
    } while (first < result.journal.size() && result.journal[first].original_block.slot == slot);
    if (!candidate->Replace(*handle, std::move(block)))
      return Decline(SsaPhiFoldRefusal::invalid_graph);
  }

  const auto checked_candidate = ir::ValidateSsaWithSources(*candidate, sources, budget);
  if (checked_candidate != ir::SsaDecline::none)
    return Decline(checked_candidate == ir::SsaDecline::resource_limit
                       ? SsaPhiFoldRefusal::resource_limit
                       : SsaPhiFoldRefusal::invalid_graph);
  for (std::size_t slot = 0; slot < candidate->slots(); ++slot) {
    if (!reachable.reachable[slot]) continue;
    const auto handle = candidate->Handle(slot);
    if (!handle) return Decline(SsaPhiFoldRefusal::invalid_graph);
    const auto complete = ir::ValidateSsaDirectSuccessors(*candidate->Get(*handle), budget);
    if (complete != ir::SsaDecline::none)
      return Decline(complete == ir::SsaDecline::resource_limit ? SsaPhiFoldRefusal::resource_limit
                                                                : SsaPhiFoldRefusal::invalid_graph);
  }

  for (auto& edit : result.journal) edit.to_revision = candidate->revision();
  result.provisional = std::move(*candidate);
  return result;
}

}  // namespace nyx::recovery
