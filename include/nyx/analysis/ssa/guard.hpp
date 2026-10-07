#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::analysis {

enum class SsaGuardRefusal {
  none,
  open_entries,
  incomplete_successors,
  invalid_graph,
  resource_limit,
  not_conditional,
  unreachable_block,
  bypass_path
};

struct SsaGuardResult {
  std::optional<ir::SsaGuardEdgeFact> fact;
  SsaGuardRefusal reason = SsaGuardRefusal::none;
};

enum class SsaPredicateRefusal {
  none,
  recovered_control,
  transformed_block,
  invalid_graph,
  resource_limit
};

struct SsaPredicateResult {
  std::optional<ir::SsaBranchPredicateFact> fact;
  SsaPredicateRefusal reason = SsaPredicateRefusal::none;
};

// Prove that every entry path to guarded_block traverses one conditional edge.
// `through_dispatches` takes the edges of table dispatches no fold proves yet
// as their successors. A switch's own dispatch is such a block, so without it
// the guard that would let it be folded can never be proved. The fact serves
// only to propose that fold; see ir::ValidateSsaDispatchSuccessors.
[[nodiscard]] SsaGuardResult ProveSsaGuardEdge(const ir::SsaGraph&,
                                               std::span<const ir::Group> decoded_sources,
                                               ir::SsaEntryScope, ir::SsaHandle branch,
                                               std::uint32_t edge_index,
                                               ir::SsaHandle guarded_block, Budget&,
                                               bool through_dispatches = false);
[[nodiscard]] SsaPredicateResult ProveSsaBranchPredicate(const ir::SsaGraph&,
                                                         std::span<const ir::Group> decoded_sources,
                                                         ir::SsaHandle branch,
                                                         std::uint32_t edge_index, Budget&);

}  // namespace nyx::analysis
