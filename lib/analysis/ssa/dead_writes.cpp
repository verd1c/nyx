#include "nyx/analysis/ssa/dead_writes.hpp"

#include <algorithm>
#include <bit>
#include <limits>

namespace nyx::analysis {

SsaDeadWriteResult ProveSsaDeadWrites(const ir::SsaGraph& graph,
                                      const ir::SsaReachabilityFacts& reachable,
                                      std::span<const ir::Group> sources, Budget& budget) {
  const auto shape = ir::ValidateSsaWithSources(graph, sources, budget);
  if (shape != ir::SsaDecline::none)
    return {{},
            shape == ir::SsaDecline::resource_limit ? SsaDeadWriteRefusal::resource_limit
                                                    : SsaDeadWriteRefusal::invalid_graph};
  const auto scope = ir::ValidateSsaReachabilityFacts(graph, reachable, sources, budget);
  if (scope != ir::SsaDecline::none)
    return {{},
            scope == ir::SsaDecline::resource_limit ? SsaDeadWriteRefusal::resource_limit
                                                    : SsaDeadWriteRefusal::stale_reachability};
  const auto liveness = ir::SsaStorageLiveness::Compute(graph, budget);
  if (!liveness) return {{}, SsaDeadWriteRefusal::resource_limit};
  ir::SsaDeadWriteFacts facts{graph.arena(), graph.revision(), {}};
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    const auto handle = graph.Handle(slot);
    if (!handle || !reachable.reachable[slot]) continue;
    const auto& block = *graph.Get(*handle);
    for (std::size_t boundary = 0; boundary < block.boundaries.size(); ++boundary) {
      const auto& writes = block.boundaries[boundary].writes;
      for (std::size_t index = 0; index < writes.size(); ++index) {
        if (budget.try_consume({std::uint64_t{1} + std::bit_width(block.dead_storage_writes.size()),
                                0}) != BudgetDecline::none)
          return {{}, SsaDeadWriteRefusal::resource_limit};
        if (boundary > UINT32_MAX || index > UINT32_MAX) continue;
        const auto found = std::lower_bound(
            block.dead_storage_writes.begin(), block.dead_storage_writes.end(),
            std::pair{boundary, index}, [](const ir::SsaDeadStorageWrite& mark, const auto& key) {
              return mark.boundary < key.first ||
                     (mark.boundary == key.first && mark.index < key.second);
            });
        if (found != block.dead_storage_writes.end() && found->boundary == boundary &&
            found->index == index)
          continue;
        const ir::SsaDeadWriteFact candidate{*handle, static_cast<std::uint32_t>(boundary),
                                             static_cast<std::uint32_t>(index)};
        const auto checked = liveness->CheckDeadWrite(candidate, budget);
        if (checked == ir::SsaDecline::resource_limit)
          return {{}, SsaDeadWriteRefusal::resource_limit};
        if (checked != ir::SsaDecline::none) continue;
        if (budget.try_consume({1, sizeof(ir::SsaDeadWriteFact)}) != BudgetDecline::none)
          return {{}, SsaDeadWriteRefusal::resource_limit};
        facts.writes.push_back(candidate);
      }
    }
  }

  const auto checked = ir::ValidateSsaDeadWriteFacts(graph, reachable, facts, sources, budget);
  if (checked != ir::SsaDecline::none)
    return {{},
            checked == ir::SsaDecline::resource_limit ? SsaDeadWriteRefusal::resource_limit
                                                      : SsaDeadWriteRefusal::invalid_graph};
  return {std::move(facts), SsaDeadWriteRefusal::none};
}

}  // namespace nyx::analysis
