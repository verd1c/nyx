#include "nyx/analysis/ssa/reachability.hpp"

#include <limits>

namespace nyx::analysis {

SsaReachabilityResult ProveSsaReachability(const ir::SsaGraph& graph,
                                           std::span<const ir::Group> sources,
                                           ir::SsaEntryScope scope, Budget& budget,
                                           bool through_dispatches) {
  const auto checked = ir::ValidateSsa(graph, budget);
  if (checked != ir::SsaDecline::none)
    return {{},
            checked == ir::SsaDecline::resource_limit ? SsaReachabilityRefusal::resource_limit
                                                      : SsaReachabilityRefusal::invalid_graph};
  if (scope != ir::SsaEntryScope::closed_population)
    return {{}, SsaReachabilityRefusal::open_entries};
  if (graph.entries().empty()) return {{}, SsaReachabilityRefusal::invalid_graph};
  const auto count = graph.slots();
  if (count > std::numeric_limits<std::size_t>::max() / (sizeof(ir::SsaHandle) + 1) ||
      budget.try_consume({count, count * (sizeof(ir::SsaHandle) + 1)}) != BudgetDecline::none)
    return {{}, SsaReachabilityRefusal::resource_limit};
  ir::SsaReachabilityFacts facts{graph.arena(), graph.revision(), scope,
                                 std::vector<std::uint8_t>(count)};
  std::vector<ir::SsaHandle> queue;
  queue.reserve(count);
  for (const auto entry : graph.entries()) {
    facts.reachable[entry.slot] = 1;
    queue.push_back(entry);
  }

  for (std::size_t head = 0; head < queue.size(); ++head) {
    const auto* block = graph.Get(queue[head]);
    if (!block) return {{}, SsaReachabilityRefusal::invalid_graph};
    if (budget.try_consume({1 + block->edges.size(), 0}) != BudgetDecline::none)
      return {{}, SsaReachabilityRefusal::resource_limit};
    const auto complete = through_dispatches ? ir::ValidateSsaDispatchSuccessors(*block, budget)
                                             : ir::ValidateSsaDirectSuccessors(*block, budget);
    if (complete != ir::SsaDecline::none)
      return {{},
              complete == ir::SsaDecline::resource_limit
                  ? SsaReachabilityRefusal::resource_limit
                  : SsaReachabilityRefusal::incomplete_successors};
    const auto bound = ir::ValidateSsaSourceBinding(*block, sources, budget);
    if (bound != ir::SsaDecline::none)
      return {{},
              bound == ir::SsaDecline::resource_limit ? SsaReachabilityRefusal::resource_limit
                                                      : SsaReachabilityRefusal::invalid_graph};
    for (const auto& edge : block->edges) {
      if (!edge.target_block || facts.reachable[edge.target_block->slot]) continue;
      facts.reachable[edge.target_block->slot] = 1;
      queue.push_back(*edge.target_block);
    }
  }

  const auto valid =
      ir::ValidateSsaReachabilityFacts(graph, facts, sources, budget, through_dispatches);
  if (valid != ir::SsaDecline::none)
    return {{},
            valid == ir::SsaDecline::resource_limit ? SsaReachabilityRefusal::resource_limit
                                                    : SsaReachabilityRefusal::invalid_graph};
  return {std::move(facts), SsaReachabilityRefusal::none};
}

}  // namespace nyx::analysis
