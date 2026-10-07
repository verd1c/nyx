#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::recovery {

enum class SsaConstantFoldRefusal { none, invalid_graph, stale_proof, resource_limit };

struct SsaConstantFoldEdit {
  ir::SsaHandle original_block;
  ir::SsaHandle result_block;
  ir::ValueId node;
  ir::Node original;
  std::uint64_t value;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
  std::optional<ir::SsaRead> removed_read;
};

struct SsaConstantFoldResult {
  std::optional<ir::SsaGraph> provisional;
  std::vector<SsaConstantFoldEdit> journal;

  // Declared reads a store the folds let the graph place writes, with their
  // records removed from the candidate, named in the candidate.
  std::vector<ir::SsaDroppedPathRead> dropped_path_reads;
  SsaConstantFoldRefusal reason = SsaConstantFoldRefusal::none;
};

[[nodiscard]] SsaConstantFoldResult ProposeSsaConstantFold(
    const ir::SsaGraph&, const ir::SsaReachabilityFacts&, const ir::SsaConstantFacts&,
    std::span<const ir::Group> decoded_sources, Budget&,
    const ir::SsaBoundedLoopFacts* loops = nullptr);
[[nodiscard]] SsaConstantFoldResult ProposeSsaSccpFold(
    const ir::SsaGraph&, const ir::SsaSccpFacts&, std::span<const ir::Group> decoded_sources,
    Budget&, const ir::SsaBoundedLoopFacts* loops = nullptr);
[[nodiscard]] SsaConstantFoldResult ProposeSsaSccpReadFold(
    const ir::SsaGraph&, const ir::SsaSccpFacts&, std::span<const ir::Group> decoded_sources,
    Budget&, const ir::SsaBoundedLoopFacts* loops = nullptr);

}  // namespace nyx::recovery
