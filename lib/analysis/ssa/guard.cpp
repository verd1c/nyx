#include "nyx/analysis/ssa/guard.hpp"

#include <algorithm>
#include <limits>

#include "nyx/analysis/ssa/reachability.hpp"

namespace nyx::analysis {

SsaGuardResult ProveSsaGuardEdge(const ir::SsaGraph& graph, std::span<const ir::Group> sources,
                                 ir::SsaEntryScope scope, ir::SsaHandle branch,
                                 std::uint32_t edge_index, ir::SsaHandle guarded_block,
                                 Budget& budget, bool through_dispatches) {
  const auto reach = ProveSsaReachability(graph, sources, scope, budget, through_dispatches);
  if (!reach.facts) {
    switch (reach.reason) {
      case SsaReachabilityRefusal::open_entries:
        return {{}, SsaGuardRefusal::open_entries};
      case SsaReachabilityRefusal::incomplete_successors:
        return {{}, SsaGuardRefusal::incomplete_successors};
      case SsaReachabilityRefusal::resource_limit:
        return {{}, SsaGuardRefusal::resource_limit};
      default:
        return {{}, SsaGuardRefusal::invalid_graph};
    }
  }

  const auto* source = graph.Get(branch);
  if (!source || !graph.Get(guarded_block)) return {{}, SsaGuardRefusal::invalid_graph};
  if (edge_index >= source->edges.size()) return {{}, SsaGuardRefusal::not_conditional};
  const auto& guard = source->edges[edge_index];
  if (!guard.target_block || !guard.condition || !guard.when ||
      (guard.kind != ir::SsaEdgeKind::branch && guard.kind != ir::SsaEdgeKind::fallthrough))
    return {{}, SsaGuardRefusal::not_conditional};
  if (!reach.facts->reachable[branch.slot] || !reach.facts->reachable[guarded_block.slot])
    return {{}, SsaGuardRefusal::unreachable_block};

  struct Incoming {
    std::size_t predecessor;
    std::size_t edge;
  };

  const auto count = graph.slots();
  std::size_t edges = 0;
  for (std::size_t slot = 0; slot < count; ++slot) {
    const auto handle = graph.Handle(slot);
    const auto size = handle && reach.facts->reachable[slot] ? graph.Get(*handle)->edges.size() : 0;
    if (size == SIZE_MAX || budget.try_consume({1 + size, 0}) != BudgetDecline::none ||
        edges > SIZE_MAX - size)
      return {{}, SsaGuardRefusal::resource_limit};
    edges += size;
  }

  if (count > SIZE_MAX / (sizeof(std::vector<Incoming>) + sizeof(std::size_t) + 3) ||
      edges > SIZE_MAX / sizeof(Incoming))
    return {{}, SsaGuardRefusal::resource_limit};
  const auto base = count * (sizeof(std::vector<Incoming>) + sizeof(std::size_t) + 3);
  if (edges * sizeof(Incoming) > SIZE_MAX - base) return {{}, SsaGuardRefusal::resource_limit};
  std::uint64_t work = 0;
  const auto charge = [&](std::uint64_t value) {
    if (work > UINT64_MAX - value) return false;
    work += value;
    return true;
  };

  if (count > UINT64_MAX / 9 || edges > UINT64_MAX / 2 || !charge(9 * count) ||
      !charge(2 * edges) || !charge(graph.entries().size()) ||
      budget.try_consume({work, base + edges * sizeof(Incoming)}) != BudgetDecline::none)
    return {{}, SsaGuardRefusal::resource_limit};

  std::vector<std::vector<Incoming>> predecessors(count);
  std::vector<std::size_t> indegree(count);
  std::vector<std::uint8_t> entry(count), must(count), next(count);
  for (const auto handle : graph.entries()) entry[handle.slot] = 1;
  for (std::size_t slot = 0; slot < count; ++slot) {
    const auto handle = graph.Handle(slot);
    if (!handle || !reach.facts->reachable[slot]) continue;
    for (const auto& edge : graph.Get(*handle)->edges)
      if (edge.target_block) ++indegree[edge.target_block->slot];
  }

  for (std::size_t slot = 0; slot < count; ++slot) predecessors[slot].reserve(indegree[slot]);
  for (std::size_t slot = 0; slot < count; ++slot) {
    const auto handle = graph.Handle(slot);
    if (!handle || !reach.facts->reachable[slot]) continue;
    const auto& outgoing = graph.Get(*handle)->edges;
    for (std::size_t index = 0; index < outgoing.size(); ++index)
      if (outgoing[index].target_block)
        predecessors[outgoing[index].target_block->slot].push_back({slot, index});
  }

  for (std::size_t slot = 0; slot < count; ++slot)
    must[slot] = reach.facts->reachable[slot] && !entry[slot];
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t slot = 0; slot < count; ++slot) {
      if (budget.try_consume({1 + predecessors[slot].size(), 0}) != BudgetDecline::none)
        return {{}, SsaGuardRefusal::resource_limit};
      const bool holds =
          reach.facts->reachable[slot] && !entry[slot] && !predecessors[slot].empty() &&
          std::all_of(predecessors[slot].begin(), predecessors[slot].end(), [&](Incoming incoming) {
            return (incoming.predecessor == branch.slot && incoming.edge == edge_index) ||
                   must[incoming.predecessor];
          });
      next[slot] = holds;
      changed |= holds != must[slot];
    }

    must.swap(next);
  }

  if (!must[guarded_block.slot]) return {{}, SsaGuardRefusal::bypass_path};
  ir::SsaGuardEdgeFact fact{graph.arena(), graph.revision(),  scope, branch, edge_index,
                            guarded_block, through_dispatches};
  const auto checked = ir::ValidateSsaGuardEdgeFact(graph, fact, sources, budget);
  if (checked != ir::SsaDecline::none)
    return {{},
            checked == ir::SsaDecline::resource_limit ? SsaGuardRefusal::resource_limit
                                                      : SsaGuardRefusal::invalid_graph};
  return {fact, SsaGuardRefusal::none};
}

SsaPredicateResult ProveSsaBranchPredicate(const ir::SsaGraph& graph,
                                           std::span<const ir::Group> sources, ir::SsaHandle branch,
                                           std::uint32_t edge_index, Budget& budget) {
  const auto* block = graph.Get(branch);
  if (!block) return {{}, SsaPredicateRefusal::invalid_graph};
  if (block->opaque || block->transition || !block->control_rewrites.empty())
    return {{}, SsaPredicateRefusal::recovered_control};
  if (!block->store_omissions.empty() || !block->paired_load_omissions.empty() ||
      !block->destination_nodes.empty() || !block->disabled_effects.empty() ||
      !block->dead_pure_nodes.empty() || !block->frame_accesses.empty() ||
      !block->constant_loads.empty())
    return {{}, SsaPredicateRefusal::transformed_block};
  if (block->boundaries.empty() || edge_index >= block->edges.size() ||
      !block->boundaries.back().transfer || !block->boundaries.back().transfer->condition ||
      !block->edges[edge_index].when)
    return {{}, SsaPredicateRefusal::invalid_graph};
  const ir::SsaBranchPredicateFact fact{graph.arena(),
                                        graph.revision(),
                                        branch,
                                        edge_index,
                                        *block->boundaries.back().transfer->condition,
                                        *block->edges[edge_index].when};
  const auto checked = ir::ValidateSsaBranchPredicateFact(graph, fact, sources, budget);
  if (checked != ir::SsaDecline::none)
    return {{},
            checked == ir::SsaDecline::resource_limit ? SsaPredicateRefusal::resource_limit
                                                      : SsaPredicateRefusal::invalid_graph};
  return {fact, SsaPredicateRefusal::none};
}

}  // namespace nyx::analysis
