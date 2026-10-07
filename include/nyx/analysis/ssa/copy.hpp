#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::analysis {

enum class SsaCopyRefusal { none, invalid_graph, stale_reachability, resource_limit };

struct SsaCopyResult {
  std::optional<ir::SsaPredecessorCopyFacts> facts;
  SsaCopyRefusal reason = SsaCopyRefusal::none;
};

[[nodiscard]] SsaCopyResult ProveSsaPredecessorCopies(const ir::SsaGraph&,
                                                      const ir::SsaReachabilityFacts&,
                                                      std::span<const ir::Group> decoded_sources,
                                                      Budget&);

}  // namespace nyx::analysis
