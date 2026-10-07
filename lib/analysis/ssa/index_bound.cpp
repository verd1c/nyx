#include "nyx/analysis/ssa/index_bound.hpp"

#include <algorithm>

#include "nyx/analysis/ssa/guard.hpp"

namespace nyx::analysis {

SsaIndexBoundResult ProveSsaDirectIndexBound(const ir::SsaGraph& graph,
                                             std::span<const ir::Group> sources,
                                             ir::SsaEntryScope scope, ir::SsaHandle branch,
                                             std::uint32_t edge_index, ir::SsaHandle guarded_block,
                                             Budget& budget) {
  const auto guarded =
      ProveSsaGuardEdge(graph, sources, scope, branch, edge_index, guarded_block, budget, true);
  if (!guarded.fact)
    return {{},
            guarded.reason == SsaGuardRefusal::resource_limit
                ? SsaIndexBoundRefusal::resource_limit
                : SsaIndexBoundRefusal::guard_unproved};
  const auto predicate = ProveSsaBranchPredicate(graph, sources, branch, edge_index, budget);
  if (!predicate.fact)
    return {{},
            predicate.reason == SsaPredicateRefusal::resource_limit
                ? SsaIndexBoundRefusal::resource_limit
                : SsaIndexBoundRefusal::predicate_unbound};
  const auto* source = graph.Get(branch);
  const auto* target = graph.Get(guarded_block);
  if (!source || !target || edge_index >= source->edges.size())
    return {{}, SsaIndexBoundRefusal::invalid_graph};
  if (source->edges[edge_index].target_block != guarded_block ||
      predicate.fact->condition >= source->nodes.size())
    return {{}, SsaIndexBoundRefusal::unsupported_condition};
  if (budget.try_consume({3 * ir::kSsaCoreSteps, 0}) != BudgetDecline::none)
    return {{}, SsaIndexBoundRefusal::resource_limit};
  const auto bound =
      ir::SsaGuardEdgeBound(source->nodes, predicate.fact->condition, predicate.fact->when);
  if (!bound) return {{}, SsaIndexBoundRefusal::unsupported_condition};
  if (budget.try_consume({source->reads.size() + 2 * source->exits.size() + target->phis.size() +
                              (source->exits.size() + 4) * ir::kSsaCoreSteps,
                          0}) != BudgetDecline::none)
    return {{}, SsaIndexBoundRefusal::resource_limit};
  // The bounded value reaches the guarded block in a register: the one it was
  // read from, or the one the guard computed it into.
  const auto& compared = source->nodes[bound->value];
  std::optional<ir::StorageId> storage;
  if (compared.op == ir::Op::read) {
    storage = compared.storage;
  } else {
    for (const auto& exit : source->exits) {
      if (exit.value.kind != ir::SsaValueKind::node || exit.value.block != branch ||
          exit.value.index >= source->nodes.size() ||
          ir::SsaUnsignedCore(source->nodes, exit.value.index) != bound->value)
        continue;
      if (storage) return {{}, SsaIndexBoundRefusal::changed_index};
      storage = exit.storage;
    }
  }

  const auto* exit =
      storage ? ir::SsaExitHolding(*source, branch, *storage, bound->value) : nullptr;
  const auto phi =
      std::find_if(target->phis.begin(), target->phis.end(),
                   [&](const ir::SsaPhi& item) { return storage && item.storage == *storage; });
  if (!exit || phi == target->phis.end() || phi->external_entry || phi->incoming.size() != 1 ||
      phi->incoming[0].predecessor != branch || phi->incoming[0].value != exit->value)
    return {{}, SsaIndexBoundRefusal::changed_index};
  const ir::SsaIndexBoundFact fact{*guarded.fact, *predicate.fact, *storage,
                                   bound->exclusive_upper};
  const auto checked = ir::ValidateSsaIndexBoundFact(graph, fact, sources, budget);
  if (checked != ir::SsaDecline::none)
    return {{},
            checked == ir::SsaDecline::resource_limit ? SsaIndexBoundRefusal::resource_limit
                                                      : SsaIndexBoundRefusal::invalid_graph};
  return {fact, SsaIndexBoundRefusal::none};
}

}  // namespace nyx::analysis
