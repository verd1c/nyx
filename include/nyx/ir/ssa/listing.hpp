#pragma once

#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "nyx/ir/ssa/graph.hpp"
#include "nyx/support/budget.hpp"

namespace nyx::ir {

struct SsaStorageName {
  StorageId storage;
  std::string_view name;
};

// What a target lends a listing: a name per storage, and which storage a
// callee or the caller may still observe when control leaves through a call
// or a return. Both sets are the declared calling convention's; the listing
// says it relies on them.
struct SsaListingTarget {
  std::span<const SsaStorageName> names;            // sorted by storage
  std::span<const StorageId> observable_at_call;    // sorted
  std::span<const StorageId> observable_at_return;  // sorted
  // The decoded groups the graph was built from, when the caller has them.
  // They say which register a call reads its target from after folding has
  // turned that read into the value it held.
  std::span<const Group> sources = {};
};

struct SsaListingResult {
  std::optional<std::string> text;
  SsaDecline reason = SsaDecline::none;
};

// A human-readable view of a validated graph; no pass consumes it. Blocks appear in reverse
// postorder from the entries, leaving out an edge whose condition is a literal that never selects
// it and any block only such edges reach. Each shows only the effects that still execute and the
// register writes a later block, a call or the caller can observe; a pure value used once is
// written inline. Omitted and retired accesses are hidden even though execution still checks that
// their address is mapped. The printed SSA text remains the audit record.
[[nodiscard]] SsaListingResult ListSsa(const SsaGraph&, const SsaListingTarget&, Budget&);

}  // namespace nyx::ir
