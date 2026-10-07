#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::recovery {

enum class SsaLoopExitRefusal { none, invalid_graph, stale_proof, resource_limit };

// Why one proved loop stays: control this rule does not rewrite, an effect a
// single run would not repeat, or something it changes that is read after it.
enum class SsaLoopExitDecline { unsupported_control, unsupported_effect, live_state };

struct SsaLoopExitEdit {
  ir::SsaHandle block;
  ir::SsaHandle exit;
  std::uint32_t iterations;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
};

struct SsaLoopExitRefused {
  ir::SsaHandle block;
  SsaLoopExitDecline reason;
};

struct SsaLoopExitResult {
  std::optional<ir::SsaGraph> provisional;
  std::vector<SsaLoopExitEdit> journal;
  std::vector<SsaLoopExitRefused> refused;
  SsaLoopExitRefusal reason = SsaLoopExitRefusal::none;
};

// A loop a bounded-loop fact shows leaving after a known number of runs, that
// touches nothing but storage and whose changes nobody reads after it, is
// rewritten to go where it leaves: running it once changes nothing observable
// that running it to the end would not. A loop with any other effect, such as
// an image checksum that reads memory, is real work and stays.
[[nodiscard]] SsaLoopExitResult ProposeSsaLoopExits(const ir::SsaGraph&,
                                                    const ir::SsaBoundedLoopFacts&,
                                                    std::span<const ir::Group> decoded_sources,
                                                    Budget&);

}  // namespace nyx::recovery
