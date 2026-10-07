#include <algorithm>
#include <limits>

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::ir {

SsaDecline ValidateSsaDominanceFacts(const SsaGraph& graph, const SsaDominanceFacts& facts,
                                     std::span<const Group> sources, Budget& budget) {
  const auto valid = ValidateSsa(graph, budget);
  if (valid != SsaDecline::none) return valid;
  const auto count = graph.slots();
  if (facts.graph_arena != graph.arena() || facts.graph_revision != graph.revision() ||
      facts.entry_scope != SsaEntryScope::closed_population || facts.reachable.size() != count ||
      (count && count > SIZE_MAX / count) || facts.dominates.size() != count * count)
    return SsaDecline::invalid_graph;
  if (budget.try_consume({count, count}) != BudgetDecline::none) return SsaDecline::resource_limit;
  const SsaReachabilityFacts reach{facts.graph_arena, facts.graph_revision, facts.entry_scope,
                                   facts.reachable};
  const auto checked = ValidateSsaReachabilityFacts(graph, reach, sources, budget);
  if (checked != SsaDecline::none) return checked;
  if (count > SIZE_MAX / (sizeof(std::size_t) + 1) ||
      budget.try_consume({count, count * (sizeof(std::size_t) + 1)}) != BudgetDecline::none)
    return SsaDecline::resource_limit;
  std::vector<std::size_t> queue;
  queue.reserve(count);
  std::vector<std::uint8_t> seen(count);
  const auto traverse = [&](std::optional<std::size_t> blocked) {
    if (graph.entries().size() > UINT64_MAX - count) return false;
    if (budget.try_consume({count + graph.entries().size(), 0}) != BudgetDecline::none)
      return false;
    std::fill(seen.begin(), seen.end(), 0);
    queue.clear();
    for (const auto entry : graph.entries()) {
      if (blocked == entry.slot || seen[entry.slot]) continue;
      seen[entry.slot] = 1;
      queue.push_back(entry.slot);
    }

    for (std::size_t head = 0; head < queue.size(); ++head) {
      const auto handle = graph.Handle(queue[head]);
      if (!handle) return false;
      const auto& block = *graph.Get(*handle);
      if (budget.try_consume({1 + block.edges.size(), 0}) != BudgetDecline::none) return false;
      for (const auto& edge : block.edges) {
        if (!edge.target_block) continue;
        const auto next = edge.target_block->slot;
        if (blocked == next || seen[next]) continue;
        seen[next] = 1;
        queue.push_back(next);
      }
    }

    return true;
  };

  if (!traverse({})) return SsaDecline::resource_limit;
  if (seen != facts.reachable) return SsaDecline::invalid_graph;
  for (std::size_t d = 0; d < count; ++d) {
    if (!facts.reachable[d]) {
      if (budget.try_consume({count, 0}) != BudgetDecline::none) return SsaDecline::resource_limit;
      for (std::size_t b = 0; b < count; ++b)
        if (facts.dominates[d * count + b]) return SsaDecline::invalid_graph;
      continue;
    }

    if (!traverse(d)) return SsaDecline::resource_limit;
    if (budget.try_consume({count, 0}) != BudgetDecline::none) return SsaDecline::resource_limit;
    for (std::size_t b = 0; b < count; ++b) {
      const bool expected = facts.reachable[b] && (d == b || !seen[b]);
      if (facts.dominates[d * count + b] != expected) return SsaDecline::invalid_graph;
    }
  }

  return SsaDecline::none;
}

}  // namespace nyx::ir
