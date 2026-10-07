#pragma once

#include "nyx/analysis/control.hpp"
#include "nyx/ir/path.hpp"
#include "nyx/ir/recovered_path.hpp"

namespace nyx::analysis {

enum class ExpectedMatch { not_applicable, always, never, unknown };
enum class PathEdgeRole { fallthrough, branch, callee, return_, potential_return };

struct PathControlEdge {
  PathEdgeRole role;
  TargetKind target_kind = TargetKind::unknown;
  std::uint64_t target_address = 0;

  // Null only for synthetic fallthrough; all other references use Path SSA IDs.
  std::optional<ir::ValueId> target_value;
  std::optional<ir::ValueId> condition;
  std::optional<bool> when;
  std::optional<bool> known_condition;

  // Set when the destination or selected route rests on declared image bytes
  // or relocated pointer values, including facts folded before this analysis.
  bool constant_image_dependency = false;
};

// A transfer whose destination is an entry of a constant table, indexed by a
// value this path's own guard bounds. Unlike an edge, this is a complete
// successor set: the bound is what makes it complete, so the transfer can go
// nowhere else. The original conditional structure is not recovered here,
// only the set of places this one transfer can reach.
struct BoundedDispatch {
  ir::ValueId index;
  std::uint64_t bound;
  std::uint32_t guard_boundary;
  std::vector<std::uint64_t> destinations;
  bool constant_image_dependency = false;
};

struct BoundaryControl {
  std::uint32_t boundary;
  std::uint64_t source_address;
  std::optional<ir::TransferKind> transfer_kind;

  // Preserve the effective target expression even when its top-level select is
  // described as two guarded edges.
  std::optional<ir::ValueId> transfer_target;
  std::optional<std::uint64_t> expected_image_successor;
  ExpectedMatch expected_match = ExpectedMatch::not_applicable;
  std::array<PathControlEdge, 3> edges{};
  unsigned edge_count = 0;
  bool callee_return_unknown = false;

  // Expression dependencies in the effective basis only. Image-folded reads
  // are tracked by the fold journal; absence here never licenses load deletion.
  std::vector<ir::ValueId> load_dependencies;
  std::optional<BoundedDispatch> dispatch;
};

struct PathControlFacts {
  std::uint64_t path_revision;
  std::uint32_t total_boundaries;
  std::vector<BoundaryControl> boundaries;
  std::optional<std::uint32_t> proved_divergence;
  std::uint64_t path_identity = 0;
};

struct PathControlResult {
  std::optional<PathControlFacts> facts;
  ControlDecline reason = ControlDecline::none;
};

// The caller's complete list of values folded from declared image bytes or
// relocated pointer slots, taken from the final fold journal. An empty list
// attests that the final journal has no such values. IDs name basis nodes;
// analysis validates their order, shape and basis revision, but cannot prove
// the caller supplied every dependent value.
struct PathImageDependencies {
  std::uint64_t basis_revision;
  std::span<const ir::ValueId> nodes;
};

// Each record assumes successful execution of its instruction and every earlier
// actual-successor check. Unknown checks do not prove reachability. No records
// follow a proved mismatch; the supplied Path and its residual exits stay intact.
// Image targets are load_bias + offset modulo64; absolute runtime targets cannot
// be compared to image successors without placement and remain unknown.
// Use the overload with an explicit dependency overlay after image-value folding.
// This shortcut is for paths whose nodes have not been replaced from image facts.
[[nodiscard]] PathControlResult AnalyzePathControl(const ir::Path&, Budget&, ir::BlockLimits = {},
                                                   ImageFacts = {});
[[nodiscard]] PathControlResult AnalyzePathControl(const ir::Path&, PathImageDependencies, Budget&,
                                                   ir::BlockLimits = {}, ImageFacts = {});
[[nodiscard]] PathControlResult AnalyzePathControl(const ir::RecoveredPath&, PathImageDependencies,
                                                   Budget&, ir::BlockLimits = {}, ImageFacts = {});

}  // namespace nyx::analysis
