#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::recovery {

enum class SsaResolvedCallRefusal { none, invalid_graph, resource_limit };

struct SsaResolvedCallEdit {
  ir::SsaHandle block;
  std::uint64_t callee;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
};

struct SsaResolvedCallResult {
  std::optional<ir::SsaGraph> provisional;
  std::vector<SsaResolvedCallEdit> journal;
  SsaResolvedCallRefusal reason = SsaResolvedCallRefusal::none;
};

// A call through a computed target the graph settles on one image location
// (for example a pointer the obfuscator decoded from a table and parked in a
// frame slot) calls that location directly. What computed the pointer then serves
// nothing, and the passes that retire dead work can take it away.
[[nodiscard]] SsaResolvedCallResult ProposeSsaResolvedCalls(
    const ir::SsaGraph&, std::span<const ir::Group> decoded_sources, Budget&);

}  // namespace nyx::recovery
