#pragma once

#include "nyx/analysis/entry_relations.hpp"
#include "nyx/analysis/unflatten.hpp"
#include "nyx/ir/recovered_path.hpp"
#include "nyx/ir/ssa/graph.hpp"

namespace nyx::analysis {

struct SsaResult {
  std::optional<ir::SsaGraph> graph;
  ir::SsaDecline reason = ir::SsaDecline::none;
};

// Paths are indexed by region candidate. Only paths selected by transitions
// are required. All other blocks retain their original normalized SSA.
// `observed` becomes the graph's contract. A call ordinarily leaves every
// register fresh; under a declared contract, a register it lists as preserved
// keeps its value across the call.
// A direct call into an interior block follows its actual target with the
// instruction's writes, without a callee summary or a synthetic return edge.
[[nodiscard]] SsaResult BuildSsa(const Regions&, const Unflattening&,
                                 std::span<const std::optional<ir::RecoveredPath>>, Budget&,
                                 const EntryRelations* = nullptr, ImageFacts = {},
                                 ir::SsaObservability observed = {}, bool entries_closed = false);

}  // namespace nyx::analysis
