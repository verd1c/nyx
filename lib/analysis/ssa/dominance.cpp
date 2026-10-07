#include "nyx/analysis/ssa/dominance.hpp"

#include <algorithm>
#include <limits>

#include "nyx/analysis/ssa/reachability.hpp"

namespace nyx::analysis {

SsaDominanceResult ProveSsaDominance(const ir::SsaGraph& graph, std::span<const ir::Group> sources,
                                     ir::SsaEntryScope scope, Budget& budget) {
  const auto reach = ProveSsaReachability(graph, sources, scope, budget);
  if (!reach.facts) {
    switch (reach.reason) {
      case SsaReachabilityRefusal::open_entries:
        return {{}, SsaDominanceRefusal::open_entries};
      case SsaReachabilityRefusal::incomplete_successors:
        return {{}, SsaDominanceRefusal::incomplete_successors};
      case SsaReachabilityRefusal::resource_limit:
        return {{}, SsaDominanceRefusal::resource_limit};
      default:
        return {{}, SsaDominanceRefusal::invalid_graph};
    }
  }

  const auto count = graph.slots();
  if (count && count > std::numeric_limits<std::size_t>::max() / count)
    return {{}, SsaDominanceRefusal::resource_limit};
  const auto cells = count * count;
  std::size_t edge_count = 0;
  for (std::size_t slot = 0; slot < count; ++slot) {
    const auto handle = graph.Handle(slot);
    const auto size = handle && reach.facts->reachable[slot] ? graph.Get(*handle)->edges.size() : 0;
    if (size == SIZE_MAX || budget.try_consume({1 + size, 0}) != BudgetDecline::none)
      return {{}, SsaDominanceRefusal::resource_limit};
    if (!handle || !reach.facts->reachable[slot]) continue;
    if (edge_count > SIZE_MAX - size) return {{}, SsaDominanceRefusal::resource_limit};
    edge_count += size;
  }

  if (count > SIZE_MAX / (sizeof(std::vector<std::size_t>) + sizeof(std::size_t)) ||
      edge_count > SIZE_MAX / sizeof(std::size_t))
    return {{}, SsaDominanceRefusal::resource_limit};
  std::size_t bytes = count * (sizeof(std::vector<std::size_t>) + sizeof(std::size_t));
  if (edge_count * sizeof(std::size_t) > SIZE_MAX - bytes)
    return {{}, SsaDominanceRefusal::resource_limit};
  bytes += edge_count * sizeof(std::size_t);
  std::uint64_t work = 0;
  const auto add_work = [&](std::uint64_t amount) {
    if (work > UINT64_MAX - amount) return false;
    work += amount;
    return true;
  };

  if (count > (SIZE_MAX - bytes) / 2 || cells > (SIZE_MAX - bytes - 2 * count) / 2 ||
      count > UINT64_MAX / 4 || edge_count > UINT64_MAX / 2 || cells > UINT64_MAX / 3 ||
      !add_work(4 * count) || !add_work(2 * edge_count) || !add_work(graph.entries().size()) ||
      !add_work(3 * cells) ||
      budget.try_consume({work, bytes + 2 * count + 2 * cells}) != BudgetDecline::none)
    return {{}, SsaDominanceRefusal::resource_limit};
  std::vector<std::vector<std::size_t>> predecessors(count);
  std::vector<std::size_t> indegree(count);
  std::vector<std::uint8_t> entries(count);
  for (const auto entry : graph.entries()) entries[entry.slot] = 1;
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
    for (const auto& edge : graph.Get(*handle)->edges)
      if (edge.target_block) predecessors[edge.target_block->slot].push_back(slot);
  }

  ir::SsaDominanceFacts facts{graph.arena(), graph.revision(), scope, reach.facts->reachable,
                              std::vector<std::uint8_t>(cells)};
  std::vector<std::uint8_t> next(cells);
  for (std::size_t d = 0; d < count; ++d)
    for (std::size_t b = 0; b < count; ++b)
      facts.dominates[d * count + b] =
          facts.reachable[d] && facts.reachable[b] && (entries[b] ? d == b : true);
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t d = 0; d < count; ++d) {
      for (std::size_t b = 0; b < count; ++b) {
        if (budget.try_consume({1 + predecessors[b].size(), 0}) != BudgetDecline::none)
          return {{}, SsaDominanceRefusal::resource_limit};
        bool holds = facts.reachable[d] && facts.reachable[b] && d == b;
        if (facts.reachable[d] && facts.reachable[b] && !entries[b] && d != b &&
            !predecessors[b].empty()) {
          holds = std::all_of(
              predecessors[b].begin(), predecessors[b].end(),
              [&](std::size_t predecessor) { return facts.dominates[d * count + predecessor]; });
        }

        next[d * count + b] = holds;
        changed |= holds != facts.dominates[d * count + b];
      }
    }

    facts.dominates.swap(next);
  }

  const auto checked = ir::ValidateSsaDominanceFacts(graph, facts, sources, budget);
  if (checked != ir::SsaDecline::none)
    return {{},
            checked == ir::SsaDecline::resource_limit ? SsaDominanceRefusal::resource_limit
                                                      : SsaDominanceRefusal::invalid_graph};
  return {std::move(facts), SsaDominanceRefusal::none};
}

}  // namespace nyx::analysis
