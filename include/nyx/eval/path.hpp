#pragma once

#include "nyx/eval/concrete.hpp"
#include "nyx/ir/path.hpp"
#include "nyx/ir/recovered_path.hpp"

namespace nyx::eval {

enum class PathStop { completed, diverged, declined, fault };

struct PathExecutionResult {
  Outcome outcome = Outcome::invalid_group;
  PathStop stop = PathStop::declined;
  std::uint64_t source_address = 0;
  std::size_t completed_boundaries = 0;
  std::vector<ExecutionResult> trace;
  std::optional<std::uint64_t> runtime_next;
};

// A differing actual successor commits its instruction and stops the itinerary.
// Fault/decline retains prior completed instructions; modeled faults may publish
// ordered effects of the failing instruction. Machine replay requires immutable
// code separately. Placement must be explicit, including for fallthrough guards.
PathExecutionResult ExecutePath(const ir::Path&, State&, Memory&, Budget&, Limits = {},
                                ExecutionContext = {});

// Executes validated control replacements, retaining every basis node and effect.
// Trace transfer kinds describe the effective IR (a folded branch is a jump, a
// recovered dispatch a conditional); original source transfers remain available
// through the immutable basis. The declared facts validate the replacements that
// rest on them; execution itself reads the supplied memory, never those bytes,
// so a path whose rewrites need a declaration this caller does not make is
// refused rather than executed.
PathExecutionResult ExecuteRecoveredPath(const ir::RecoveredPath&, State&, Memory&, Budget&,
                                         Limits = {}, ExecutionContext = {}, ir::ImageFacts = {});

}  // namespace nyx::eval
