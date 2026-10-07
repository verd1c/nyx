#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::analysis {

enum class SsaIndexBoundRefusal {
  none,
  guard_unproved,
  predicate_unbound,
  unsupported_condition,
  changed_index,
  invalid_graph,
  resource_limit
};

struct SsaIndexBoundResult {
  std::optional<ir::SsaIndexBoundFact> fact;
  SsaIndexBoundRefusal reason = SsaIndexBoundRefusal::none;
};

// This first range proof covers one direct true arm with no intermediate block.
[[nodiscard]] SsaIndexBoundResult ProveSsaDirectIndexBound(
    const ir::SsaGraph&, std::span<const ir::Group> decoded_sources, ir::SsaEntryScope,
    ir::SsaHandle branch, std::uint32_t edge_index, ir::SsaHandle guarded_block, Budget&);

}  // namespace nyx::analysis
