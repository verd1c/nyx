#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::ir {

struct PrivateFrameContract {
  StorageId sp_storage;

  // Half-open interval relative to entry SP, declared fresh, mapped, writable,
  // externally unreachable and unobservable to asynchronous actors. The
  // function forms addresses into it only by arithmetic on SP, never by
  // recovering one through comparisons or control flow.
  std::int64_t begin;
  std::int64_t end;
  bool fresh_mapped_writable = false;
  bool no_external_aliases = false;
  bool no_async_observers = false;
  unsigned sp_alignment = 1;

  // Stronger than a calling convention: a callee may otherwise inspect the
  // caller's frame even if it preserves the named registers. A callee that
  // cannot touch the frame also hands back no address into it.
  bool callees_cannot_touch = false;
  bool callees_preserve_sp = false;
};

struct PrivateFrameAccess {
  SsaHandle block;
  ValueId node;
  std::int64_t offset;
  std::uint32_t size;
  bool store;
};

struct PrivateFrameSlot {
  std::int64_t offset;
  std::uint32_t size;
  std::vector<PrivateFrameAccess> accesses;
};

struct PrivateFrameFacts {
  std::uint64_t arena;
  std::uint64_t revision;
  PrivateFrameContract contract;
  std::vector<PrivateFrameSlot> slots;
  bool entry_relations = false;
  bool relation_declared_abi = false;
  bool relation_return_leaves = false;
  bool relation_constant_image = false;
  bool relation_declared_opaque_control = false;
};

// What a frame proposal rests on, copied into its result so the journal can be
// rechecked and printed after the proof object that produced it is gone.
struct PrivateFrameBasis {
  PrivateFrameContract contract;
  bool entry_relations = false;
  bool relation_declared_abi = false;
  bool relation_return_leaves = false;
  bool relation_constant_image = false;
  bool relation_declared_opaque_control = false;
};

inline PrivateFrameBasis BasisOf(const PrivateFrameFacts& facts) {
  return {facts.contract,
          facts.entry_relations,
          facts.relation_declared_abi,
          facts.relation_return_leaves,
          facts.relation_constant_image,
          facts.relation_declared_opaque_control};
}

// What one value can be relative to the entry stack pointer: exactly entry
// SP plus `offset`, a frame address at an offset not known, or a value that
// is not derived from the stack pointer at all (with `offset` its value when
// it is a literal). Values follow SSA through every block: SP writes, frame
// addresses held in registers or promoted slots, and storage a call
// preserves.
struct FrameAddress {
  enum class Kind : std::uint8_t { none, other, exact, any };
  Kind kind = Kind::none;
  std::int64_t offset = 0;
  bool literal = false;
  friend bool operator==(const FrameAddress&, const FrameAddress&) = default;
};

enum class FrameScanDecline {
  none,
  invalid,
  missing_contract,
  unknown_call,
  opaque_effect,
  unknown_continuation,
  escaped_address,
  resource_limit
};

// Every access of a graph that can reach the private region. Under the
// contract nothing outside the function holds a frame address, so an access
// whose address does not derive from the stack pointer cannot reach it, as
// long as no frame address escapes: into memory, a call's arguments, a
// return's results, or a computed transfer. That is refused.
struct FrameScan {
  // Accesses at a known offset wholly inside the region, in graph order.
  std::vector<PrivateFrameAccess> exact;

  // [begin, end) byte ranges of the region that an access reaches without
  // lying wholly inside it.
  std::vector<std::pair<std::int64_t, std::int64_t>> blocked;

  // Some access has a frame address at an offset not known.
  bool dynamic = false;
  FrameScanDecline reason = FrameScanDecline::none;
};

// Caller validates the graph first.
FrameScan ScanPrivateFrame(const SsaGraph&, const PrivateFrameContract&, Budget&);

// The slots a scan proves private: every access at one offset and size, none
// of another shape or an unknown offset overlapping it, each a plain load or
// store aligned by the entry SP. Empty when any access has an unknown offset.
std::optional<std::vector<PrivateFrameSlot>> PrivateFrameSlots(const FrameScan&, const SsaGraph&,
                                                               const PrivateFrameContract&,
                                                               Budget&);

enum class PrivateFrameFactDecline { none, invalid, resource_limit };

// Independently checks the claimed slots against a fresh scan of the bound
// SSA graph before a recovery pass may consume them.
PrivateFrameFactDecline ValidatePrivateFrameFacts(const PrivateFrameFacts&, const SsaGraph&,
                                                  Budget&);

}  // namespace nyx::ir
