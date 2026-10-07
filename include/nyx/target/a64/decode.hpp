#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "nyx/ir/group.hpp"
#include "nyx/support/budget.hpp"

namespace nyx::a64 {

inline constexpr ir::StorageId kSp = 31;
inline constexpr ir::StorageId kN = 32;
inline constexpr ir::StorageId kZ = 33;
inline constexpr ir::StorageId kC = 34;
inline constexpr ir::StorageId kV = 35;

// The thread pointer. Its value is the runtime's, never a constant of the image.
inline constexpr ir::StorageId kTpidrEl0 = 36;

// Full 128-bit SIMD registers; partial scalar/vector aliases are unsupported.
inline constexpr ir::StorageId kQ0 = 37;

// The AAPCS64 callee-saved general registers, X19-X29, and SP, sorted. This
// describes the calling convention; it is no evidence that a callee obeys it.
inline constexpr std::array<ir::StorageId, 12> kCalleeSaved = {19, 20, 21, 22, 23, 24,
                                                               25, 26, 27, 28, 29, kSp};

enum class DecodeDecline {
  none,
  unsupported,
  invalid_encoding,
  work_limit,
  byte_limit,
  invalid_location
};

// This experimental profile models ordinary scalar accesses as all-or-nothing,
// with SP alignment checking disabled. W pairs/LDPSW use one 64-bit access;
// X pairs and Q loads/stores use two ordered 64-bit accesses, and SIMD&FP
// pairs access each element in turn. QEMU writes a SIMD&FP pair's first
// register before its second access; here a fault on that access leaves both.
// LDAR/STLR (including byte/halfword forms) retain aligned accesses; unaligned addresses decline
// because the selected QEMU backend does not have one fault rule for them. This profile has no
// acquire/release ordering or concurrent observer. Destination/writeback registers commit on
// success. These QEMU-reference choices are not architectural fault authority. Adds the selected
// QEMU scalar exclusive-monitor behavior to the ordinary scalar profile. It does not model
// concurrent observers or acquire/release ordering.
enum class MemoryProfile { none, concrete_atomic_scalar, concrete_exclusive_scalar };

struct Options {
  MemoryProfile memory_profile = MemoryProfile::none;
};

struct DecodeResult {
  std::optional<ir::Group> group;
  DecodeDecline reason = DecodeDecline::none;
};

// Source locations must be four-byte aligned and contain all four bytes without wrap.
// Base control transfers model GPR/NZCV with BTI and GCS inactive; PSTATE.BTYPE,
// target fetch and callee execution are outside this instruction-effect subset.
DecodeResult Decode(std::uint64_t address, std::array<std::uint8_t, 4> bytes, Budget& budget,
                    Options options = {});

// LDAXR, STXR and STLXR have a normal PC+4 successor. The ordinary
// scalar profile leaves their effects opaque.
bool OpaqueNormalFallthrough(std::array<std::uint8_t, 4> bytes);

// BRK always raises a Breakpoint Instruction exception; it never completes.
bool OpaqueTrap(std::array<std::uint8_t, 4> bytes);

}  // namespace nyx::a64
