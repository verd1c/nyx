#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::recovery {

enum class SsaBranchRetirementRefusal { none, invalid_graph, stale_proof, resource_limit };

struct SsaBranchRetirementEdit {
  ir::SsaHandle original_block;
  ir::SsaHandle result_block;
  std::uint32_t removed_edge;
  ir::SsaEdge removed;
  ir::SsaEdge retained;
  std::uint64_t from_revision;
  std::uint64_t to_revision;

  // A dispatcher's conditional is already a rewrite, so deciding it replaces
  // that rewrite in place rather than adding a second one to its boundary.
  bool dispatcher = false;
};

struct SsaBranchRetirementPhiEdit {
  ir::SsaHandle original_block;
  ir::SsaHandle result_block;
  ir::SsaHandle predecessor;
  std::uint32_t phi;
  bool frame;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
};

struct SsaBranchRetirementResult {
  std::optional<ir::SsaGraph> provisional;
  std::vector<SsaBranchRetirementEdit> journal;
  std::vector<SsaBranchRetirementPhiEdit> phi_journal;
  SsaBranchRetirementRefusal reason = SsaBranchRetirementRefusal::none;
};

[[nodiscard]] SsaBranchRetirementResult ProposeSsaBranchRetirement(
    const ir::SsaGraph&, const ir::SsaSccpFacts&, std::span<const ir::Group> decoded_sources,
    Budget&, const ir::SsaBoundedLoopFacts* loops = nullptr);

}  // namespace nyx::recovery
