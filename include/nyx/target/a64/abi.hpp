#pragma once

#include <span>

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::a64 {

// The storage AAPCS64 lets a callee observe at a call: arguments X0-X8 and
// Q0-Q7, the platform register X18, preserved X19-X29 and Q8-Q15 (a callee
// may save them to memory), SP, and the link register it returns through.
// Sorted.
std::span<const ir::StorageId> ObservedAtCall();

// The storage the caller observes at a return: results X0-X8 and Q0-Q7, X18,
// and what it expects preserved. Sorted. Flags and X9-X17 are in neither set.
std::span<const ir::StorageId> ObservedAtReturn();

// The declared AAPCS64 contract for an SSA graph, without the fault declaration:
// a call preserves X19-X29 and SP.
ir::SsaObservability AbiObservability();

}  // namespace nyx::a64
