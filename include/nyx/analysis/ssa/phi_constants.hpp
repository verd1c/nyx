#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::analysis {

enum class SsaPhiConstantRefusal { none, invalid_graph, stale_reachability, resource_limit };

struct SsaPhiConstantResult {
  std::optional<ir::SsaPhiConstantFacts> facts;
  SsaPhiConstantRefusal reason = SsaPhiConstantRefusal::none;
};

[[nodiscard]] SsaPhiConstantResult ProveSsaPhiConstants(const ir::SsaGraph&,
                                                        const ir::SsaReachabilityFacts&,
                                                        std::span<const ir::Group> decoded_sources,
                                                        Budget&);

}  // namespace nyx::analysis
