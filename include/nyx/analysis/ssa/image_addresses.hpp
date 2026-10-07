#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::analysis {

enum class SsaImageAddressRefusal { none, invalid_graph, resource_limit, not_selected_image };

struct RefusedSsaImageAddress {
  ir::SsaHandle block;
  ir::ValueId load;
  SsaImageAddressRefusal reason;
};

struct SsaImageAddressResult {
  std::optional<ir::SsaImageAddressFacts> facts;
  std::vector<RefusedSsaImageAddress> refused;
  SsaImageAddressRefusal reason = SsaImageAddressRefusal::none;
};

// Prove every ordinary load whose address has exactly two image locations
// selected by one SSA condition. Other load addresses remain unproved.
[[nodiscard]] SsaImageAddressResult ProveSsaSelectedImageAddresses(const ir::SsaGraph&, Budget&);

}  // namespace nyx::analysis
