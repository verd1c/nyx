#include "nyx/analysis/ssa/copy.hpp"

#include <algorithm>
#include <bit>

namespace nyx::analysis {

SsaCopyResult ProveSsaPredecessorCopies(const ir::SsaGraph& graph,
                                        const ir::SsaReachabilityFacts& reachable,
                                        std::span<const ir::Group> sources, Budget& budget) {
  const auto shape = ir::ValidateSsaWithSources(graph, sources, budget);
  if (shape != ir::SsaDecline::none)
    return {{},
            shape == ir::SsaDecline::resource_limit ? SsaCopyRefusal::resource_limit
                                                    : SsaCopyRefusal::invalid_graph};
  const auto valid = ir::ValidateSsaReachabilityFacts(graph, reachable, sources, budget);
  if (valid != ir::SsaDecline::none)
    return {{},
            valid == ir::SsaDecline::resource_limit ? SsaCopyRefusal::resource_limit
                                                    : SsaCopyRefusal::stale_reachability};
  ir::SsaPredecessorCopyFacts facts{graph.arena(), graph.revision(), {}};
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return {{}, SsaCopyRefusal::resource_limit};
    const auto handle = graph.Handle(slot);
    if (!handle || !reachable.reachable[slot]) continue;
    const auto& block = *graph.Get(*handle);
    if (budget.try_consume({block.reads.size() + block.phis.size(), 0}) != BudgetDecline::none)
      return {{}, SsaCopyRefusal::resource_limit};
    for (const auto& read : block.reads) {
      if (read.value.kind != ir::SsaValueKind::phi || read.value.index != read.phi ||
          read.phi >= block.phis.size())
        continue;
      const auto& phi = block.phis[read.phi];
      if (phi.external_entry || phi.incoming.size() != 1) continue;
      const auto source = phi.incoming[0].value;
      const auto* predecessor = graph.Get(phi.incoming[0].predecessor);
      const auto* descriptor = predecessor && source.index < predecessor->nodes.size()
                                   ? ir::Descriptor(predecessor->nodes[source.index].op)
                                   : nullptr;
      if (budget.try_consume(
              {std::uint64_t{1} +
                   (predecessor ? std::bit_width(predecessor->dead_pure_nodes.size()) : 0),
               0}) != BudgetDecline::none)
        return {{}, SsaCopyRefusal::resource_limit};
      if (!predecessor || predecessor->opaque || phi.incoming[0].predecessor == *handle ||
          source.kind != ir::SsaValueKind::node || source.block != phi.incoming[0].predecessor ||
          !descriptor || !descriptor->produces_value ||
          std::binary_search(predecessor->dead_pure_nodes.begin(),
                             predecessor->dead_pure_nodes.end(), source.index))
        continue;
      if (budget.try_consume({1, sizeof(ir::SsaPredecessorCopy)}) != BudgetDecline::none)
        return {{}, SsaCopyRefusal::resource_limit};
      facts.copies.push_back({*handle, read.node, source});
    }
  }

  const auto checked =
      ir::ValidateSsaPredecessorCopyFacts(graph, reachable, facts, sources, budget);
  if (checked != ir::SsaDecline::none)
    return {{},
            checked == ir::SsaDecline::resource_limit ? SsaCopyRefusal::resource_limit
                                                      : SsaCopyRefusal::invalid_graph};
  return {std::move(facts), SsaCopyRefusal::none};
}

}  // namespace nyx::analysis
