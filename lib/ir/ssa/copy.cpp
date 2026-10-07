#include <algorithm>
#include <bit>

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::ir {

SsaDecline ValidateSsaPredecessorCopyFacts(const SsaGraph& graph,
                                           const SsaReachabilityFacts& reachable,
                                           const SsaPredecessorCopyFacts& facts,
                                           std::span<const Group> sources, Budget& budget) {
  const auto shape = ValidateSsaWithSources(graph, sources, budget);
  if (shape != SsaDecline::none) return shape;
  const auto valid = ValidateSsaReachabilityFacts(graph, reachable, sources, budget);
  if (valid != SsaDecline::none) return valid;
  if (facts.graph_arena != graph.arena() || facts.graph_revision != graph.revision())
    return SsaDecline::invalid_graph;
  std::size_t claimed = 0;
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return SsaDecline::resource_limit;
    const auto handle = graph.Handle(slot);
    if (!handle || !reachable.reachable[slot]) continue;
    const auto& block = *graph.Get(*handle);
    for (const auto& read : block.reads) {
      if (budget.try_consume({1, 0}) != BudgetDecline::none) return SsaDecline::resource_limit;
      if (read.phi >= block.phis.size() ||
          read.value != SsaValue{SsaValueKind::phi, *handle, read.phi})
        continue;
      const auto& phi = block.phis[read.phi];
      if (phi.external_entry || phi.incoming.size() != 1) continue;
      const auto input = phi.incoming[0];
      const auto* predecessor = graph.Get(input.predecessor);
      if (budget.try_consume(
              {std::uint64_t{0} +
                   (predecessor ? std::bit_width(predecessor->dead_pure_nodes.size()) : 0),
               0}) != BudgetDecline::none)
        return SsaDecline::resource_limit;
      const auto* descriptor = predecessor && input.value.index < predecessor->nodes.size()
                                   ? Descriptor(predecessor->nodes[input.value.index].op)
                                   : nullptr;
      if (!predecessor || predecessor->opaque || input.predecessor == *handle ||
          input.value.kind != SsaValueKind::node || input.value.block != input.predecessor ||
          !descriptor || !descriptor->produces_value ||
          std::binary_search(predecessor->dead_pure_nodes.begin(),
                             predecessor->dead_pure_nodes.end(), input.value.index))
        continue;
      if (claimed >= facts.copies.size() ||
          facts.copies[claimed] != SsaPredecessorCopy{*handle, read.node, input.value})
        return SsaDecline::invalid_graph;
      ++claimed;
    }
  }

  return claimed == facts.copies.size() ? SsaDecline::none : SsaDecline::invalid_graph;
}

}  // namespace nyx::ir
