#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::recovery {

enum class SsaLeafCallRefusal { none, invalid_graph, resource_limit };

// Why one call stays: its callee has no body the graph carries, the body is
// not a leaf that only computes, or something it writes is read after it.
enum class SsaLeafCallDecline { no_body, not_leaf, live_result };

struct SsaLeafCallEdit {
  ir::SsaHandle block;
  std::uint64_t callee;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
};

struct SsaLeafCallRefused {
  ir::SsaHandle block;
  std::uint64_t callee;
  SsaLeafCallDecline reason;
};

struct SsaLeafCallResult {
  std::optional<ir::SsaGraph> provisional;
  std::vector<SsaLeafCallEdit> journal;
  std::vector<SsaLeafCallRefused> refused;
  SsaLeafCallRefusal reason = SsaLeafCallRefusal::none;
};

// A call whose callee's body the graph carries, that only computes and writes
// registers and returns where it was called from, and whose every write is
// dead where it returns, changes nothing anyone observes: it goes straight to
// its continuation. The obfuscator's getters (a table's address, a pair of
// constants) are such calls once what they return has been folded.
[[nodiscard]] SsaLeafCallResult ProposeSsaLeafCalls(const ir::SsaGraph&,
                                                    std::span<const ir::Group> decoded_sources,
                                                    Budget&);

}  // namespace nyx::recovery
