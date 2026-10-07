#pragma once

#include "nyx/ir/recovered_path.hpp"

namespace nyx::recovery {

enum class StoreCleanupDecline { none, invalid_ir, resource_limit, revision_overflow };

struct StoreCleanupResult {
  std::optional<ir::RecoveredPath> path;
  StoreCleanupDecline reason = StoreCleanupDecline::none;
};

// Proposes exact-width overwrites within 16 source boundaries through one
// entry-storage base, with no intervening memory access or transfer.
// The recovered executor checks the omitted access is mapped and writable at
// runtime; the result is scoped to successful, resource-sufficient execution
// in its single-threaded atomic-scalar model.
[[nodiscard]] StoreCleanupResult OmitOverwrittenStores(ir::RecoveredPath&& input,
                                                       ir::StorageId base_storage, Budget& budget,
                                                       ir::ImageFacts facts = {},
                                                       std::uint32_t max_omissions = 4096);

}  // namespace nyx::recovery
