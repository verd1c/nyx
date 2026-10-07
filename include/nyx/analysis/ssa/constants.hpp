#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::analysis {

enum class SsaConstantRefusal { none, invalid_graph, stale_reachability, resource_limit };

struct SsaConstantResult {
  std::optional<ir::SsaConstantFacts> facts;
  SsaConstantRefusal reason = SsaConstantRefusal::none;
};

// Facts are relative to the supplied SSA semantic graph. The caller must
// justify any earlier graph edits against their source groups separately.
[[nodiscard]] SsaConstantResult ProveSsaConstants(const ir::SsaGraph&,
                                                  const ir::SsaReachabilityFacts&,
                                                  std::span<const ir::Group> decoded_sources,
                                                  Budget&,
                                                  const ir::SsaBoundedLoopFacts* loops = nullptr);

}  // namespace nyx::analysis
