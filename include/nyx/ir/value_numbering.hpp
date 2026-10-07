#pragma once

#include <span>
#include <vector>

#include "nyx/ir/group.hpp"
#include "nyx/support/budget.hpp"

namespace nyx::ir {

// Identifiers for values rather than for nodes.
//
// `same_value[i]` is the lowest identifier known to denote the same value as
// node `i`; `unwrapped[i]` is `i` with any zero-extend and narrow pair that
// normalization wrapped around it removed. Two nodes agree exactly when their
// unwrapped representatives agree, which is what an analysis needs to know
// before treating two identifiers as one value.
//
// Only pure value-producing nodes are numbered. A repeated load is a distinct
// definition even at one address, and a register read is relative to its own
// instruction's entry state, so neither is ever merged with another node.
[[nodiscard]] bool NumberValues(std::span<const Node> nodes, Budget&,
                                std::vector<ValueId>& same_value, std::vector<ValueId>& unwrapped);

// A zero-extension immediately narrowed back to its own width is the value
// itself. Normalization inserts that pair around every W-form operand.
[[nodiscard]] ValueId Unwrap(ValueId, std::span<const Node> nodes);

}  // namespace nyx::ir
