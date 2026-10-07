#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::analysis {

enum class SsaDeadWriteRefusal { none, invalid_graph, stale_reachability, resource_limit };

struct SsaDeadWriteResult {
  std::optional<ir::SsaDeadWriteFacts> facts;
  SsaDeadWriteRefusal reason = SsaDeadWriteRefusal::none;
};

[[nodiscard]] SsaDeadWriteResult ProveSsaDeadWrites(const ir::SsaGraph&,
                                                    const ir::SsaReachabilityFacts&,
                                                    std::span<const ir::Group> decoded_sources,
                                                    Budget&);

}  // namespace nyx::analysis
