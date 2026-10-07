#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::analysis {

enum class SsaBoundedLoopRefusal { none, invalid_graph, resource_limit };

struct SsaBoundedLoopResult {
  std::optional<ir::SsaBoundedLoopFacts> facts;
  SsaBoundedLoopRefusal reason = SsaBoundedLoopRefusal::none;
};

// What each block that branches only to itself leaves in its successor's phis.
// A lattice cannot answer this: the second iteration disagrees with the first
// and every loop-carried value goes overdefined. Running the block does answer
// it, where its entry values are constants and its effects are ones the run
// accounts for.
[[nodiscard]] SsaBoundedLoopResult ProveSsaBoundedLoops(const ir::SsaGraph&,
                                                        std::span<const ir::Group> decoded_sources,
                                                        Budget&);

}  // namespace nyx::analysis
