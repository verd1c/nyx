#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::recovery {

enum class SsaDeadWriteRefusal { none, invalid_graph, stale_proof, resource_limit };

struct SsaDeadWriteEdit {
  ir::SsaHandle original_block;
  ir::SsaHandle result_block;
  std::uint32_t boundary;
  std::uint32_t index;
  ir::Write original;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
};

struct SsaDeadWriteRejected {
  ir::SsaHandle block;
  std::uint32_t boundary;
  std::uint32_t index;
};

struct SsaDeadWriteProposal {
  std::optional<ir::SsaGraph> provisional;
  std::vector<SsaDeadWriteEdit> journal;
  std::vector<SsaDeadWriteRejected> refused;
  SsaDeadWriteRefusal reason = SsaDeadWriteRefusal::none;
};

[[nodiscard]] SsaDeadWriteProposal ProposeSsaDeadWrites(const ir::SsaGraph&,
                                                        const ir::SsaReachabilityFacts&,
                                                        const ir::SsaDeadWriteFacts&,
                                                        std::span<const ir::Group> decoded_sources,
                                                        Budget&);

}  // namespace nyx::recovery
