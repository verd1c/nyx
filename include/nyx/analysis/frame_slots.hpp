#pragma once

#include "nyx/ir/frame.hpp"

namespace nyx::analysis {

using ir::PrivateFrameAccess;
using ir::PrivateFrameContract;
using ir::PrivateFrameFacts;
using ir::PrivateFrameSlot;
using ir::ValidatePrivateFrameFacts;

// A frame access at an offset not known (`unknown_offset`) could reach any
// slot, so none is proved; slots only some access overlaps, misaligns or
// reaches with an exclusive access are left out without declining the rest.
enum class PrivateFrameDecline {
  none,
  invalid_graph,
  missing_contract,
  unknown_call,
  opaque_effect,
  unknown_continuation,
  escaped_address,
  unknown_offset,
  resource_limit
};

struct PrivateFrameResult {
  std::optional<PrivateFrameFacts> facts;
  PrivateFrameDecline reason = PrivateFrameDecline::none;
};

[[nodiscard]] PrivateFrameResult ProvePrivateFrameSlots(const ir::SsaGraph&, PrivateFrameContract,
                                                        Budget&);

}  // namespace nyx::analysis
