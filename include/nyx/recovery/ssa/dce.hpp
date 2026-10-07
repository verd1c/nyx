#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::recovery {

enum class SsaDceRefusal { none, stale_proof, invalid_graph, resource_limit };
enum class SsaDceEditKind { pruned_storage_phi, pruned_frame_phi, removed_block };

struct SsaDceEdit {
  SsaDceEditKind kind;
  ir::SsaHandle original_block;
  std::optional<ir::SsaHandle> result_block;
  std::uint32_t index;
  ir::SsaHandle predecessor;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
};

struct SsaDceResult {
  std::optional<ir::SsaGraph> provisional;
  std::vector<SsaDceEdit> journal;
  ir::SsaEntryScope entry_scope = ir::SsaEntryScope::discovered_only;
  SsaDceRefusal reason = SsaDceRefusal::none;
};

// Deletes only blocks unreachable from the declared complete entry set. A
// reachable unresolved successor or return that may re-enter the population
// refuses the entire candidate.
[[nodiscard]] SsaDceResult ProposeUnreachableBlocks(const ir::SsaGraph&,
                                                    const ir::SsaReachabilityFacts&,
                                                    std::span<const ir::Group> decoded_sources,
                                                    Budget&);

}  // namespace nyx::recovery
