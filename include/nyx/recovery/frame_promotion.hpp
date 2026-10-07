#pragma once

#include "nyx/ir/frame.hpp"

namespace nyx::recovery {

enum class PromotionRefusal {
  none,
  stale_proof,
  invalid_graph,
  existing_omission,
  already_promoted,
  unsupported_access,
  incompatible_access,
  resource_limit
};

struct RefusedPromotionSlot {
  std::int64_t offset;
  std::uint32_t size;
  PromotionRefusal reason;
};

struct PromotionEdit {
  ir::SsaHandle original_block;
  ir::SsaHandle result_block;
  ir::ValueId node;
  std::int64_t slot_offset;
  std::uint32_t slot_size;
  std::optional<ir::SsaValue> replacement;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
};

struct PromotionResult {
  // A candidate until an independent whole-function comparison accepts it.
  std::optional<ir::SsaGraph> provisional;

  // Every journal entry rests on this contract and the relations it used.
  std::optional<ir::PrivateFrameBasis> basis;
  std::vector<PromotionEdit> journal;
  std::vector<RefusedPromotionSlot> refused;
  PromotionRefusal reason = PromotionRefusal::none;
};

[[nodiscard]] PromotionResult ProposeFramePromotion(const ir::SsaGraph&,
                                                    const ir::PrivateFrameFacts&, Budget&);

}  // namespace nyx::recovery
