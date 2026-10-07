#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::analysis {

enum class SsaTableAddressRefusal {
  none,
  index_unbounded,
  source_unbound,
  unsupported_address,
  index_changed,
  address_overflow,
  invalid_graph,
  resource_limit
};

struct SsaTableAddressResult {
  std::optional<ir::SsaBoundedTableAddressFact> fact;
  SsaTableAddressRefusal reason = SsaTableAddressRefusal::none;
};

// Prove the address range of a direct indexed load; no image access is removed.
[[nodiscard]] SsaTableAddressResult ProveSsaBoundedTableAddress(
    const ir::SsaGraph&, std::span<const ir::Group> decoded_sources, const ir::SsaIndexBoundFact&,
    ir::ValueId load, Budget&);

}  // namespace nyx::analysis
