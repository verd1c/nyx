#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::analysis {

enum class SsaSccpRefusal {
  none,
  invalid_graph,
  open_entries,
  incomplete_successors,
  resource_limit
};

struct SsaSccpResult {
  std::optional<ir::SsaSccpFacts> facts;
  SsaSccpRefusal reason = SsaSccpRefusal::none;
};

[[nodiscard]] SsaSccpResult ProveSsaSccp(const ir::SsaGraph&, ir::SsaEntryScope,
                                         std::span<const ir::Group> decoded_sources, Budget&,
                                         const ir::SsaBoundedLoopFacts* loops = nullptr);

}  // namespace nyx::analysis
