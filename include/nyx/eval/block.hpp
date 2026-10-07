#pragma once

#include "nyx/eval/concrete.hpp"
#include "nyx/ir/block.hpp"

namespace nyx::eval {

struct BlockExecutionResult {
  Outcome outcome = Outcome::invalid_group;
  std::uint64_t source_address = 0;
  std::size_t completed_boundaries = 0;
  std::vector<ExecutionResult> trace;
};

// Each instruction publishes independently. A later decline preserves its
// completed prefix; a modeled fault may additionally publish ordered effects
// from the failing instruction. Trace operation IDs are source-local IDs.
// This evaluates supplied IR without fetching source bytes. Machine replay must
// establish immutable code separately; writable code aliases are not admitted.
BlockExecutionResult ExecuteBlock(const ir::Block&, State&, Memory&, Budget&, Limits = {},
                                  ExecutionContext = {});

}  // namespace nyx::eval
