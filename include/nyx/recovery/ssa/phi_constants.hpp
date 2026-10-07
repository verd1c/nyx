#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::recovery {

enum class SsaPhiFoldRefusal { none, invalid_graph, stale_proof, resource_limit };

struct SsaPhiFoldEdit {
  ir::SsaHandle original_block;
  ir::SsaHandle result_block;
  ir::ValueId node;
  ir::Node original;
  std::uint64_t value;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
};

struct SsaPhiFoldResult {
  std::optional<ir::SsaGraph> provisional;
  std::vector<SsaPhiFoldEdit> journal;
  SsaPhiFoldRefusal reason = SsaPhiFoldRefusal::none;
};

[[nodiscard]] SsaPhiFoldResult ProposeSsaPhiConstantFold(const ir::SsaGraph&,
                                                         const ir::SsaReachabilityFacts&,
                                                         const ir::SsaPhiConstantFacts&,
                                                         std::span<const ir::Group> decoded_sources,
                                                         Budget&);

}  // namespace nyx::recovery
