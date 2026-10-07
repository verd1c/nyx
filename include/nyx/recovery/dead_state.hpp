#pragma once

#include "nyx/ir/frame.hpp"

namespace nyx::recovery {

enum class DeadStateRefusal {
  none,
  stale_proof,
  invalid_graph,
  slot_used,
  unsupported_effect,
  resource_limit
};

struct RefusedDeadSlot {
  std::int64_t offset;
  std::uint32_t size;
  DeadStateRefusal reason;
};

struct DeadStateEdit {
  ir::SsaHandle original_block;
  ir::SsaHandle result_block;
  ir::ValueId node;
  ir::Op operation;
  std::int64_t slot_offset;
  std::uint32_t slot_size;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
};

struct DeadStateResult {
  // A candidate until an independent whole-function comparison accepts it.
  std::optional<ir::SsaGraph> provisional;

  // Every journal entry rests on this contract and the relations it used.
  std::optional<ir::PrivateFrameBasis> basis;
  std::vector<DeadStateEdit> journal;
  std::vector<RefusedDeadSlot> refused;
  DeadStateRefusal reason = DeadStateRefusal::none;
};

[[nodiscard]] DeadStateResult ProposeDeadPrivateState(const ir::SsaGraph&,
                                                      const ir::PrivateFrameFacts&, Budget&);

}  // namespace nyx::recovery
