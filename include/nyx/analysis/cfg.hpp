#pragma once

#include "nyx/analysis/control.hpp"

namespace nyx::analysis {

std::uint64_t NextCfgIdentity();

enum class OpaqueReason { none, unsupported, invalid_encoding, not_decoded };

// Supplied by the target decoder. Analysis checks its structural use, while
// source-byte verification remains the caller's responsibility. A trap never
// completes, so under the successful-execution scope it has no successor, as
// an access that always faults would not; a handler's choices lie outside it.
enum class OpaqueControl { unknown, normal_fallthrough, trap };

struct SourceRecord {
  std::uint64_t address;
  std::vector<std::uint8_t> bytes;
  std::optional<ir::Group> semantics;
  OpaqueReason opaque_reason = OpaqueReason::none;
  OpaqueControl opaque_control = OpaqueControl::unknown;
};

enum class CfgEdgeKind {
  fallthrough,
  branch,
  callee,
  return_,
  potential_return,
  opaque_unknown,
  trap
};

enum class TargetResolution {
  block_entry,
  opaque_entry,
  mid_instruction,
  outside_population,
  absolute_runtime,
  unknown
};

struct CfgTarget {
  TargetKind kind = TargetKind::unknown;
  std::uint64_t address = 0;
  std::optional<ir::ValueId> value;
};

struct CfgEdge {
  CfgEdgeKind kind;
  CfgTarget target;
  TargetResolution resolution = TargetResolution::unknown;
  std::optional<std::uint32_t> target_source;
  std::optional<std::uint32_t> target_block;
  std::optional<ir::ValueId> condition;
  std::optional<bool> when;
  bool constant_image_dependency = false;
};

// A jump whose destination is an entry of a constant table, indexed by a value
// the guard block bounds on every entry to this one. Its edges are then a
// complete successor set rather than one known target among unknown ones: the
// bound is what makes them complete. The destinations are those edges; this
// records what their completeness rests on, and which index they run over.
struct CfgDispatch {
  std::uint32_t guard_block;
  ir::ValueId index;
  std::uint64_t bound;
};

struct CfgBlock {
  std::uint32_t first_source;
  std::uint32_t source_count;
  std::optional<ir::Block> ssa;
  std::optional<ControlFacts> control;
  std::optional<CfgDispatch> dispatch;
  std::vector<CfgEdge> edges;
};

// What the image-fact refutation was able to look at. The scan covers the
// supplied population only: a declaration it does not refute stays a
// declaration, because writers elsewhere in the image are never examined.
struct ImageFactScan {
  std::uint64_t scanned_blocks = 0;

  // Blocks with no modeled semantics. Their effects, a store among them, were
  // not examined at all.
  std::uint64_t unscanned_blocks = 0;

  // Stores to an image-derived address the scan could not place. Each could
  // reach a declared location.
  std::uint64_t unresolved_writes = 0;
};

// The caller supplies the population; it is neither discovered nor checked for
// reachability. Unknown
// transfers may enter block interiors; this partition assumes only known entries.
class Cfg {
 public:
  Cfg(std::vector<SourceRecord> sources, std::vector<std::uint64_t> entries,
      std::vector<CfgBlock> blocks, std::uint32_t passes,
      std::vector<ir::RefutedConstantSpan> refuted_constants = {},
      std::vector<std::size_t> refuted_pointers = {}, ImageFactScan image_fact_scan = {})
      : sources_(std::move(sources)),
        entries_(std::move(entries)),
        blocks_(std::move(blocks)),
        passes_(passes),
        refuted_constants_(std::move(refuted_constants)),
        refuted_pointers_(std::move(refuted_pointers)),
        image_fact_scan_(image_fact_scan) {}

  std::span<const SourceRecord> sources() const { return sources_; }

  std::span<const std::uint64_t> entries() const { return entries_; }

  std::span<const CfgBlock> blocks() const { return blocks_; }

  std::uint32_t passes() const { return passes_; }

  std::uint64_t generation() const { return 0; }

  std::uint64_t identity() const { return identity_; }

  // Indices into the ImageFacts passed to BuildCfg. The CFG borrows no image
  // bytes; callers can derive a reduced view while their original bytes live.
  // A constant range loses only the bytes a placed store reaches: each span
  // names its range, the bytes, and the store, sorted by range then address.
  std::span<const ir::RefutedConstantSpan> refuted_constant_spans() const {
    return refuted_constants_;
  }

  std::span<const std::size_t> refuted_pointer_indices() const { return refuted_pointers_; }

  const ImageFactScan& image_fact_scan() const { return image_fact_scan_; }

 private:
  std::uint64_t identity_ = NextCfgIdentity();
  std::vector<SourceRecord> sources_;
  std::vector<std::uint64_t> entries_;
  std::vector<CfgBlock> blocks_;
  std::uint32_t passes_;
  std::vector<ir::RefutedConstantSpan> refuted_constants_;
  std::vector<std::size_t> refuted_pointers_;
  ImageFactScan image_fact_scan_;
};

struct CfgLimits {
  std::uint32_t max_sources = 65536;
  std::uint32_t max_blocks = 16384;
  std::uint32_t max_passes = 64;
  std::uint64_t max_source_bytes = 16ULL * 1024 * 1024;
  std::uint32_t max_dispatch_targets = 4096;
  ir::BlockLimits block{};
};

enum class CfgDecline { none, invalid_source, invalid_entry, invalid_ir, resource_limit };

struct CfgResult {
  std::optional<Cfg> cfg;
  CfgDecline reason = CfgDecline::none;
};

// Image facts let an indirect transfer resolve, and a resolved destination is a
// newly known entry: the partition loop already re-runs until no entry appears,
// so discovery through a jump table is the same fixpoint as any other.
[[nodiscard]] CfgResult BuildCfg(std::span<const SourceRecord>,
                                 std::span<const std::uint64_t> selected_image_entries, Budget&,
                                 CfgLimits = {}, ImageFacts = {});

}  // namespace nyx::analysis
