#include <algorithm>
#include <limits>

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::ir {

SsaDecline ValidateSsaGuardEdgeFact(const SsaGraph& graph, const SsaGuardEdgeFact& fact,
                                    std::span<const Group> sources, Budget& budget) {
  const auto valid = ValidateSsa(graph, budget);
  if (valid != SsaDecline::none) return valid;
  if (fact.graph_arena != graph.arena() || fact.graph_revision != graph.revision() ||
      fact.entry_scope != SsaEntryScope::closed_population || graph.entries().empty())
    return SsaDecline::invalid_graph;
  const auto* branch = graph.Get(fact.branch);
  if (!branch || !graph.Get(fact.guarded_block) || fact.edge_index >= branch->edges.size())
    return SsaDecline::invalid_graph;
  const auto& edge = branch->edges[fact.edge_index];
  if ((edge.kind != SsaEdgeKind::branch && edge.kind != SsaEdgeKind::fallthrough) ||
      !edge.target_block || !edge.condition || !edge.when)
    return SsaDecline::invalid_graph;
  const auto count = graph.slots();
  if (count > SIZE_MAX / (sizeof(std::size_t) + 1) ||
      budget.try_consume({count, count * (sizeof(std::size_t) + 1)}) != BudgetDecline::none)
    return SsaDecline::resource_limit;
  std::vector<std::size_t> queue;
  queue.reserve(count);
  std::vector<std::uint8_t> seen(count);
  const auto traverse = [&](bool omit_guard) {
    if (graph.entries().size() > UINT64_MAX - count ||
        budget.try_consume({count + graph.entries().size(), 0}) != BudgetDecline::none)
      return SsaDecline::resource_limit;
    std::fill(seen.begin(), seen.end(), 0);
    queue.clear();
    for (const auto entry : graph.entries()) {
      if (seen[entry.slot]) continue;
      seen[entry.slot] = 1;
      queue.push_back(entry.slot);
    }

    for (std::size_t head = 0; head < queue.size(); ++head) {
      const auto handle = graph.Handle(queue[head]);
      if (!handle) return SsaDecline::invalid_graph;
      const auto& block = *graph.Get(*handle);
      if (budget.try_consume({1 + block.edges.size(), 0}) != BudgetDecline::none)
        return SsaDecline::resource_limit;
      for (std::size_t index = 0; index < block.edges.size(); ++index) {
        if (omit_guard && *handle == fact.branch && index == fact.edge_index) continue;
        const auto& outgoing = block.edges[index];
        if (!outgoing.target_block) continue;
        const auto target = outgoing.target_block->slot;
        if (seen[target]) continue;
        seen[target] = 1;
        queue.push_back(target);
      }
    }

    return SsaDecline::none;
  };

  const auto full = traverse(false);
  if (full != SsaDecline::none) return full;
  if (!seen[fact.branch.slot] || !seen[fact.guarded_block.slot]) return SsaDecline::invalid_graph;
  if (budget.try_consume({count, count}) != BudgetDecline::none) return SsaDecline::resource_limit;
  const SsaReachabilityFacts reach{graph.arena(), graph.revision(), fact.entry_scope, seen};
  const auto checked =
      ValidateSsaReachabilityFacts(graph, reach, sources, budget, fact.through_dispatches);
  if (checked != SsaDecline::none) return checked;
  const auto bypass = traverse(true);
  if (bypass != SsaDecline::none) return bypass;
  return seen[fact.guarded_block.slot] ? SsaDecline::invalid_graph : SsaDecline::none;
}

}  // namespace nyx::ir
