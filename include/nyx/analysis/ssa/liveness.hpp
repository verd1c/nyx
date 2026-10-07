#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::analysis {

enum class SsaLivenessRefusal { none, invalid_graph, resource_limit };

struct SsaLivenessResult {
  std::optional<ir::SsaLivenessFacts> facts;
  SsaLivenessRefusal reason = SsaLivenessRefusal::none;
};

[[nodiscard]] SsaLivenessResult ProveSsaNodeLiveness(const ir::SsaGraph&, Budget&);

}  // namespace nyx::analysis
