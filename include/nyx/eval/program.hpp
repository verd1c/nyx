#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "nyx/eval/concrete.hpp"

namespace nyx::eval {

enum class ProgramStatus {
  exit,
  unresolved,
  step_limit,
  resource_limit,
  fault,
  invalid_program,
  invalid_state,
  unsupported
};

struct ProgramLimits {
  std::uint32_t max_groups = 4096;
  std::uint32_t max_steps = 100000;
  Limits instruction_limits{};
};

struct ProgramStep {
  std::uint64_t runtime_pc;
  ExecutionResult execution;
};

struct ProgramResult {
  ProgramStatus status;
  std::uint64_t runtime_pc;

  // A modeled-fault step counts as a publication of its ordered prefix, even
  // though the instruction did not complete. Its trace records the actual fault.
  std::uint64_t committed_steps = 0;
  std::vector<ProgramStep> trace;
};

// Executes only explicitly supplied groups, without discovering code or modeling
// missing callees. Each completed instruction remains committed on later decline.
// Runtime placement and fallthrough use modulo-64 address arithmetic; individual
// source and runtime byte extents must not wrap. Exit addresses stop before dispatch.
// Code ranges may overlap data mappings only when the overlapping bytes match
// and the mapping is not writable.
[[nodiscard]] ProgramResult Run(std::span<const ir::Group> groups, State& state, Memory& memory,
                                std::uint64_t runtime_entry,
                                std::span<const std::uint64_t> runtime_exit_addresses,
                                Budget& budget, ProgramLimits limits = {},
                                ExecutionContext context = {});

}  // namespace nyx::eval
