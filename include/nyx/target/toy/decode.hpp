#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>

#include "nyx/ir/group.hpp"
#include "nyx/support/budget.hpp"

// A deliberately dissimilar second target for the generic stages: big-endian
// immediates and memory, one- to ten-byte instructions, sixteen 64-bit
// registers with no flags, SP in r13 and the link in r14. It exists to show
// that nothing generic assumes AArch64. It is a made-up architecture.
namespace nyx::toy {

inline constexpr unsigned kRegisters = 16;
inline constexpr ir::StorageId kSp = 13;
inline constexpr ir::StorageId kLink = 14;

// The toy calling convention's preserved registers, SP included, sorted.
inline constexpr std::array<ir::StorageId, 6> kCalleeSaved = {8, 9, 10, 11, 12, kSp};

// `truncated`: fewer bytes than the opcode's length. `invalid_location`: the
// instruction would end past the top of the address space.
enum class DecodeDecline {
  none,
  unsupported,
  invalid_encoding,
  truncated,
  invalid_location,
  work_limit,
  byte_limit
};

struct DecodeResult {
  std::optional<ir::Group> group;

  // Bytes the instruction occupies, never more than were supplied: set when the
  // whole instruction is present, so a sweep can step over a trap it declined
  // to lift, and zero for a truncated, misplaced or unknown instruction.
  unsigned length = 0;
  DecodeDecline reason = DecodeDecline::none;
};

// Encodings, first byte the opcode, registers as nibbles (d, s, t), immediates
// big-endian and relative to the instruction's own address:
//   00 trap (1)            01 ret (1)             02 nop (1)
//   10-15 add/sub/and/or/xor/mul d,s,t  [op][ds][t0] (3)
//   16 not d,s (2)         17 mov d,s (2)         18 shl d,s,#imm6 (3)
//   20 movi d,#imm64 (10)  21 movi16 d,#imm16 (4)  22 addi d,s,#simm16 (4)
//   23 lea d,rel16 (4)     30 ld d,[s+simm16] (4)  31 st [s+simm16],t  [31][ts][imm] (4)
//   32 ldx d,[s+t<<3] (3)  33 ldl d,rel16 (4)      40 seqz d,s (2)
//   50 beqz s,rel16 (4)    51 bnez s,rel16 (4)     52 bgeu s,#imm8,rel16 (5)
//   53 jmp rel16 (3)       54 call rel16 (3)       55 jr s (2)
//   57 callr s (2)
//   56 bgtu s,#imm8,rel16 (5)
[[nodiscard]] DecodeResult Decode(std::uint64_t address, std::span<const std::uint8_t> bytes,
                                  Budget& budget);

// The trap opcode never completes; it is otherwise unlifted.
[[nodiscard]] bool OpaqueTrap(std::span<const std::uint8_t> bytes);

}  // namespace nyx::toy
