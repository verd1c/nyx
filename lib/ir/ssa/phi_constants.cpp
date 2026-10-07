#include "nyx/ir/fold.hpp"
#include "nyx/ir/ssa/graph.hpp"

namespace nyx::ir {

SsaDecline ValidateSsaPhiConstantFacts(const SsaGraph& graph, const SsaReachabilityFacts& reachable,
                                       const SsaPhiConstantFacts& facts,
                                       std::span<const Group> sources, Budget& budget) {
  const auto valid = ValidateSsa(graph, budget);
  if (valid != SsaDecline::none) return valid;
  if (facts.graph_arena != graph.arena() || facts.graph_revision != graph.revision())
    return SsaDecline::invalid_graph;
  const auto scope = ValidateSsaReachabilityFacts(graph, reachable, sources, budget);
  if (scope != SsaDecline::none) return scope;
  std::size_t claimed = 0;
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return SsaDecline::resource_limit;
    const auto handle = graph.Handle(slot);
    if (!handle || !reachable.reachable[slot]) continue;
    const auto& block = *graph.Get(*handle);
    for (std::size_t phi_index = 0; phi_index < block.phis.size(); ++phi_index) {
      const auto& phi = block.phis[phi_index];
      if (budget.try_consume({1 + phi.incoming.size(), 0}) != BudgetDecline::none)
        return SsaDecline::resource_limit;
      if (phi.external_entry || phi.width > 64 || phi.incoming.empty()) continue;
      std::optional<std::uint64_t> agreed;
      bool observed = false, conflict = false;
      for (const auto& input : phi.incoming) {
        if (!reachable.reachable[input.predecessor.slot]) continue;
        observed = true;
        const auto* predecessor = graph.Get(input.predecessor);
        if (!predecessor || input.value.kind != SsaValueKind::node ||
            input.value.index >= predecessor->nodes.size() ||
            predecessor->nodes[input.value.index].op != Op::constant) {
          conflict = true;
          break;
        }

        const auto value = predecessor->nodes[input.value.index].immediate & LowMask(phi.width);
        if (agreed && *agreed != value) {
          conflict = true;
          break;
        }

        agreed = value;
      }

      if (!observed || conflict || !agreed) continue;
      if (claimed >= facts.constants.size() ||
          facts.constants[claimed] !=
              SsaPhiConstant{*handle, static_cast<std::uint32_t>(phi_index), *agreed})
        return SsaDecline::invalid_graph;
      ++claimed;
    }
  }

  return claimed == facts.constants.size() ? SsaDecline::none : SsaDecline::invalid_graph;
}

}  // namespace nyx::ir
