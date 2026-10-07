#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::analysis {

enum class SsaDominanceRefusal {
  none,
  open_entries,
  incomplete_successors,
  invalid_graph,
  resource_limit
};

struct SsaDominanceResult {
  std::optional<ir::SsaDominanceFacts> facts;
  SsaDominanceRefusal reason = SsaDominanceRefusal::none;
};

[[nodiscard]] SsaDominanceResult ProveSsaDominance(const ir::SsaGraph&,
                                                   std::span<const ir::Group> decoded_sources,
                                                   ir::SsaEntryScope, Budget&);

}  // namespace nyx::analysis
