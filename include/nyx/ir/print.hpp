#pragma once

#include <optional>
#include <string>

#include "nyx/ir/block.hpp"
#include "nyx/ir/path.hpp"
#include "nyx/ir/recovered_path.hpp"
#include "nyx/support/budget.hpp"

namespace nyx::ir {

enum class PrintDecline { none, invalid_group, work_limit, byte_limit, resource_limit };

struct PrintResult {
  std::optional<std::string> json;
  PrintDecline reason = PrintDecline::none;
};

// Output-only expression/effect schema preserving the declared memory model.
// Serialization is not execution validation; storage existence, general operand
// widths and execution limits belong to the evaluator.
[[nodiscard]] PrintResult PrintJson(const Group& group, Budget& budget);
[[nodiscard]] PrintResult PrintJson(const Block& block, Budget& budget);
[[nodiscard]] PrintResult PrintJson(const Path& path, Budget& budget);

// The declared bytes a dispatch rewrite rests on; without them its record
// cannot be rechecked, and an unvalidated path is not printed.
[[nodiscard]] PrintResult PrintJson(const RecoveredPath& path, Budget& budget, ImageFacts = {});

}  // namespace nyx::ir
