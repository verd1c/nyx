#pragma once

#include "nyx/ir/group.hpp"

namespace nyx::ir {

// On entry to a region, `storage` holds `root` plus `offset`, modulo 2^64. The producer
// is responsible for showing it holds and on which arrivals; a pass
// handed one takes it as a precondition of its region and reports what used it.
// A list of them names each storage at most once, sorted, with every root the
// smallest storage of its class and never itself related.
struct StorageRelation {
  StorageId storage;
  StorageId root;
  std::uint64_t offset;
  bool operator==(const StorageRelation&) const = default;
};

}  // namespace nyx::ir
