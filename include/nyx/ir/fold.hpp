#pragma once

#include <cstdint>
#include <optional>
#include <span>

#include "nyx/ir/block.hpp"

namespace nyx::ir {

// The value of a pure operation over literal operands, each already truncated to
// `operand_width`, truncated in turn to the node's width. Empty for effects, for
// widths over 64 bits and for operations with no literal meaning, so no caller
// folds what this cannot evaluate.
[[nodiscard]] std::optional<std::uint64_t> FoldPure(const Node& node,
                                                    std::span<const std::uint64_t> operands,
                                                    unsigned operand_width);

// All-ones in the low `width` bits, for widths 1 through 64.
[[nodiscard]] constexpr std::uint64_t LowMask(unsigned width) {
  return width >= 64 ? UINT64_MAX : (UINT64_C(1) << width) - 1;
}

}  // namespace nyx::ir
