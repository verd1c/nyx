#pragma once

#include "nyx/ir/path.hpp"
#include "nyx/ir/storage_relation.hpp"

namespace nyx::recovery {

enum class AddressBase { absolute, load_bias, ssa_value };

struct AffineAddress {
  AddressBase base = AddressBase::absolute;
  ir::ValueId value = 0;
  std::uint64_t offset = 0;
};

enum class MemoryProofScope { straightline_entry, successful_itinerary_prefix };

struct ForwardingFact {
  ir::ValueId store;
  ir::ValueId load;
  ir::ValueId value;
  AffineAddress address;
  unsigned width;
  ir::ByteOrder byte_order;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
  MemoryProofScope scope = MemoryProofScope::straightline_entry;

  // The load was told apart from some store it passed, or matched to its own,
  // only through a supplied entry relation. The fact holds only where that does.
  bool entry_relation = false;
};

enum class MemoryUse { operand, final_write, target, condition, alternative, continuation };

struct MemoryEdit {
  std::uint32_t fact;
  MemoryUse use;

  // Node ID for operand edits; boundary index for writes and terminal fields.
  std::uint32_t owner;
  std::uint32_t slot;
  ir::ValueId original;
  ir::ValueId replacement;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
};

struct MemoryRecoveryLimits {
  ir::BlockLimits block{};
  std::uint32_t max_stores = 4096;
  std::uint32_t max_edits = 196608;
};

enum class MemoryRecoveryDecline {
  none,
  invalid_ir,
  unsupported_control,
  resource_limit,
  revision_overflow
};

struct MemoryRecoveryResult {
  std::optional<ir::Block> block;
  std::vector<ForwardingFact> facts;
  std::vector<MemoryEdit> journal;
  MemoryRecoveryDecline reason = MemoryRecoveryDecline::none;
};

struct PathMemoryRecoveryResult {
  std::optional<ir::Path> path;
  std::vector<ForwardingFact> facts;
  std::vector<MemoryEdit> journal;
  MemoryRecoveryDecline reason = MemoryRecoveryDecline::none;
};

// Ordinary atomic_scalar_reference memory only. Every access remains executed:
// facts describe values after successful loads, never permission to remove them.
// No callee effects, concurrent writers or file-snapshot constants are inferred.
[[nodiscard]] MemoryRecoveryResult ForwardMemoryValues(const ir::Block&, Budget&,
                                                       MemoryRecoveryLimits = {});

// `entry` relates storage values at the path's entry; a read of related storage
// is then placed against the same base as its root. An invalid list declines.
[[nodiscard]] PathMemoryRecoveryResult ForwardMemoryValues(
    const ir::Path&, Budget&, MemoryRecoveryLimits = {},
    std::span<const ir::StorageRelation> entry = {});

}  // namespace nyx::recovery
