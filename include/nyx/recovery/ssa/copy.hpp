#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::recovery {

enum class SsaCopyRefusal { none, invalid_graph, stale_proof, resource_limit };

struct SsaCopyEdit {
  ir::SsaHandle original_block;
  ir::SsaHandle result_block;
  ir::ValueId read;
  ir::SsaValue source;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
};

struct SsaCopyProposal {
  std::optional<ir::SsaGraph> provisional;
  std::vector<SsaCopyEdit> journal;
  SsaCopyRefusal reason = SsaCopyRefusal::none;
};

[[nodiscard]] SsaCopyProposal ProposeSsaPredecessorCopies(
    const ir::SsaGraph&, const ir::SsaReachabilityFacts&, const ir::SsaPredecessorCopyFacts&,
    std::span<const ir::Group> decoded_sources, Budget&);

}  // namespace nyx::recovery
