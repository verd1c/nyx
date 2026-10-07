#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::analysis {

enum class SsaReachabilityRefusal {
  none,
  open_entries,
  incomplete_successors,
  invalid_graph,
  resource_limit
};

struct SsaReachabilityResult {
  std::optional<ir::SsaReachabilityFacts> facts;
  SsaReachabilityRefusal reason = SsaReachabilityRefusal::none;
};

// `through_dispatches` takes an unfolded table dispatch's edges as its
// successors (ir::ValidateSsaDispatchSuccessors). Such facts validate only
// under the same flag, so no consumer that rechecks them strictly accepts them.
[[nodiscard]] SsaReachabilityResult ProveSsaReachability(const ir::SsaGraph&,
                                                         std::span<const ir::Group> decoded_sources,
                                                         ir::SsaEntryScope, Budget&,
                                                         bool through_dispatches = false);

}  // namespace nyx::analysis
