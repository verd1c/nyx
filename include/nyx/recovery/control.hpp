#pragma once

#include "nyx/ir/recovered_path.hpp"

namespace nyx::recovery {

struct ControlRecoveryLimits {
  ir::BlockLimits path{};
  std::uint32_t max_edits = 4096;
};

enum class ControlRecoveryDecline { none, invalid_ir, resource_limit, revision_overflow };

struct ControlRecoveryResult {
  std::optional<ir::RecoveredPath> path;
  ControlRecoveryDecline reason = ControlRecoveryDecline::none;
};

// Two rewrites, one batch, one revision. A literal width-one predicate decides
// its conditional and leaves the jump it always took. A computed jump whose
// destination rests on a single unresolved choice, both arms of which settle on
// a place in the image under the declared facts, becomes the conditional that
// flattening replaced by a table index; the reads that resolved it stay in the
// path and still execute, and the record names them. Nothing else authorizes a
// rewrite. The owned basis keeps every original source, node, effect and
// itinerary successor check intact.
[[nodiscard]] ControlRecoveryResult RecoverControl(ir::Path&&, const ir::ImageFacts&, Budget&,
                                                   ControlRecoveryLimits = {});

}  // namespace nyx::recovery
