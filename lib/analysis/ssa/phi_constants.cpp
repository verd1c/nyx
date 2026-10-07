#include "nyx/analysis/ssa/phi_constants.hpp"

#include "nyx/ir/fold.hpp"

namespace nyx::analysis {

SsaPhiConstantResult ProveSsaPhiConstants(const ir::SsaGraph& graph,
                                          const ir::SsaReachabilityFacts& reachable,
                                          std::span<const ir::Group> sources, Budget& budget) {
  const auto valid = ir::ValidateSsa(graph, budget);
  if (valid != ir::SsaDecline::none)
    return {{},
            valid == ir::SsaDecline::resource_limit ? SsaPhiConstantRefusal::resource_limit
                                                    : SsaPhiConstantRefusal::invalid_graph};
  const auto scope = ir::ValidateSsaReachabilityFacts(graph, reachable, sources, budget);
  if (scope != ir::SsaDecline::none)
    return {{},
            scope == ir::SsaDecline::resource_limit ? SsaPhiConstantRefusal::resource_limit
                                                    : SsaPhiConstantRefusal::stale_reachability};
  ir::SsaPhiConstantFacts facts{graph.arena(), graph.revision(), {}};
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return {{}, SsaPhiConstantRefusal::resource_limit};
    const auto handle = graph.Handle(slot);
    if (!handle || !reachable.reachable[slot]) continue;
    const auto& block = *graph.Get(*handle);
    for (std::size_t index = 0; index < block.phis.size(); ++index) {
      const auto& phi = block.phis[index];
      if (budget.try_consume({1 + phi.incoming.size(), 0}) != BudgetDecline::none)
        return {{}, SsaPhiConstantRefusal::resource_limit};
      if (phi.external_entry || phi.width > 64 || phi.incoming.empty()) continue;
      std::optional<std::uint64_t> agreed;
      bool observed = false, conflict = false;
      for (const auto& input : phi.incoming) {
        if (!reachable.reachable[input.predecessor.slot]) continue;
        observed = true;
        const auto* predecessor = graph.Get(input.predecessor);
        if (!predecessor || input.value.kind != ir::SsaValueKind::node ||
            input.value.index >= predecessor->nodes.size() ||
            predecessor->nodes[input.value.index].op != ir::Op::constant) {
          conflict = true;
          break;
        }

        const auto value = predecessor->nodes[input.value.index].immediate & ir::LowMask(phi.width);
        if (agreed && *agreed != value) {
          conflict = true;
          break;
        }

        agreed = value;
      }

      if (!observed || conflict || !agreed) continue;
      if (budget.try_consume({1, sizeof(ir::SsaPhiConstant)}) != BudgetDecline::none)
        return {{}, SsaPhiConstantRefusal::resource_limit};
      facts.constants.push_back({*handle, static_cast<std::uint32_t>(index), *agreed});
    }
  }

  const auto checked = ir::ValidateSsaPhiConstantFacts(graph, reachable, facts, sources, budget);
  if (checked != ir::SsaDecline::none)
    return {{},
            checked == ir::SsaDecline::resource_limit ? SsaPhiConstantRefusal::resource_limit
                                                      : SsaPhiConstantRefusal::invalid_graph};
  return {std::move(facts), SsaPhiConstantRefusal::none};
}

}  // namespace nyx::analysis
