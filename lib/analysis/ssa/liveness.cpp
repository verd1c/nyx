#include "nyx/analysis/ssa/liveness.hpp"

#include <algorithm>
#include <bit>
#include <limits>

namespace nyx::analysis {

SsaLivenessResult ProveSsaNodeLiveness(const ir::SsaGraph& graph, Budget& budget) {
  const auto valid = ir::ValidateSsa(graph, budget);
  if (valid != ir::SsaDecline::none)
    return {{},
            valid == ir::SsaDecline::resource_limit ? SsaLivenessRefusal::resource_limit
                                                    : SsaLivenessRefusal::invalid_graph};
  if (graph.slots() > std::numeric_limits<std::size_t>::max() / sizeof(std::vector<std::uint8_t>) ||
      budget.try_consume({graph.slots(), graph.slots() * sizeof(std::vector<std::uint8_t>)}) !=
          BudgetDecline::none)
    return {{}, SsaLivenessRefusal::resource_limit};
  ir::SsaLivenessFacts facts{graph.arena(), graph.revision(),
                             std::vector<std::vector<std::uint8_t>>(graph.slots())};
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto& block = *graph.Get(*handle);
    const auto node_work = 2 + 4 * std::bit_width(block.constant_loads.size());
    if (block.nodes.size() > std::numeric_limits<std::size_t>::max() / node_work ||
        budget.try_consume({block.nodes.size() * node_work, block.nodes.size()}) !=
            BudgetDecline::none ||
        budget.try_consume({block.boundaries.size(), 0}) != BudgetDecline::none ||
        budget.try_consume({block.edges.size(), 0}) != BudgetDecline::none ||
        budget.try_consume({block.control_rewrites.size(), 0}) != BudgetDecline::none)
      return {{}, SsaLivenessRefusal::resource_limit};
    auto& live = facts.live_nodes[slot];
    live.resize(block.nodes.size());
    const auto mark = [&](ir::ValueId id) {
      if (id < live.size()) live[id] = 1;
    };

    for (std::size_t id = 0; id < block.nodes.size(); ++id)
      if (!ir::SsaEffectFree(block, static_cast<ir::ValueId>(id))) live[id] = 1;
    std::size_t rewrite_index = 0;
    for (std::size_t index = 0; index < block.boundaries.size(); ++index) {
      const auto& boundary = block.boundaries[index];
      if (budget.try_consume({boundary.writes.size(), 0}) != BudgetDecline::none)
        return {{}, SsaLivenessRefusal::resource_limit};
      for (std::size_t write_index = 0; write_index < boundary.writes.size(); ++write_index) {
        if (budget.try_consume({std::uint64_t{0} + std::bit_width(block.dead_storage_writes.size()),
                                0}) != BudgetDecline::none)
          return {{}, SsaLivenessRefusal::resource_limit};
        const auto retired =
            std::lower_bound(block.dead_storage_writes.begin(), block.dead_storage_writes.end(),
                             std::pair{index, write_index},
                             [](const ir::SsaDeadStorageWrite& item, const auto& key) {
                               return item.boundary < key.first ||
                                      (item.boundary == key.first && item.index < key.second);
                             });
        if (retired == block.dead_storage_writes.end() || retired->boundary != index ||
            retired->index != write_index)
          mark(boundary.writes[write_index].value);
      }

      const ir::Transfer* transfer = boundary.transfer ? &*boundary.transfer : nullptr;
      if (rewrite_index < block.control_rewrites.size() &&
          block.control_rewrites[rewrite_index].boundary == index)
        transfer = &block.control_rewrites[rewrite_index++].replacement;
      if (!transfer) continue;
      mark(transfer->target);
      if (transfer->condition) mark(*transfer->condition);
      if (transfer->alternative) mark(*transfer->alternative);
      if (transfer->continuation) mark(*transfer->continuation);
    }

    for (const auto& edge : block.edges)
      if (edge.condition) mark(*edge.condition);
    for (std::size_t id = block.nodes.size(); id-- > 0;) {
      if (!live[id]) continue;
      const auto& node = block.nodes[id];
      if (const auto condition = ir::SsaFoldCondition(block, static_cast<ir::ValueId>(id)))
        mark(*condition);
      if (const auto index = ir::SsaFoldIndex(block, static_cast<ir::ValueId>(id))) mark(*index);
      if (!ir::SsaNeedsInputs(block, static_cast<ir::ValueId>(id))) continue;
      const auto* descriptor = ir::Descriptor(node.op);
      for (unsigned input = 0; input < descriptor->arity; ++input) mark(node.inputs[input]);
    }
  }

  const auto checked = ir::ValidateSsaLivenessFacts(graph, facts, budget);
  if (checked != ir::SsaDecline::none)
    return {{},
            checked == ir::SsaDecline::resource_limit ? SsaLivenessRefusal::resource_limit
                                                      : SsaLivenessRefusal::invalid_graph};
  return {std::move(facts), SsaLivenessRefusal::none};
}

}  // namespace nyx::analysis
