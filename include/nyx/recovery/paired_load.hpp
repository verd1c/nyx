#pragma once

#include "nyx/ir/recovered_path.hpp"
#include "nyx/recovery/memory.hpp"

namespace nyx::recovery {

enum class PairedLoadDecline { none, invalid_ir, resource_limit, revision_overflow };

struct PairedLoadResult {
  std::optional<ir::RecoveredPath> path;
  PairedLoadDecline reason = PairedLoadDecline::none;
};

// Forwarding facts seed candidates; each pair is rechecked against the final
// basis. Execution retains the source instruction's register/writeback effects
// and checks both omitted reads in a stable normal-memory mapping.
[[nodiscard]] PairedLoadResult OmitForwardedPairLoads(ir::RecoveredPath&& input,
                                                      std::span<const ForwardingFact> forwarding,
                                                      ir::StorageId base_storage, Budget& budget,
                                                      ir::ImageFacts facts = {},
                                                      std::uint32_t max_pairs = 4096);

}  // namespace nyx::recovery
