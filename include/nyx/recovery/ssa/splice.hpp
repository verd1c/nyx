#pragma once

#include <span>

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::recovery {

// Inserts effect-free definitions before node `before` of one block of a
// private provisional graph, as part of that node's instruction. Every value
// identifier at or after the insertion moves together, in every record of
// every block that can name it. The inserted nodes may use values before
// `before` and each other, in order. False, with the graph partly edited,
// when the budget, the block size limit or a malformed node refuses it.
[[nodiscard]] bool SpliceBefore(ir::SsaGraph&, ir::SsaHandle, ir::ValueId before,
                                std::span<const ir::Node> inserted, std::uint32_t max_nodes,
                                Budget&);

}  // namespace nyx::recovery
