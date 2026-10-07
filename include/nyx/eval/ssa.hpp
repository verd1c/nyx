#pragma once

#include <functional>

#include "nyx/eval/concrete.hpp"
#include "nyx/ir/ssa/graph.hpp"

namespace nyx::eval {

enum class SsaStop { returned, trap, unresolved, fault, declined, step_limit };

struct SsaExecutionResult {
  Outcome outcome = Outcome::invalid_group;
  SsaStop stop = SsaStop::declined;
  std::uint64_t runtime_pc = 0;
  std::size_t completed_blocks = 0;
  std::optional<Fault> fault;
  std::vector<ir::SsaHandle> visited_blocks;
  std::vector<ExecutionResult> trace;
};

// The callback models one declared external callee and reports where it
// returned. A continuation edge is followed only when that address agrees.
struct SsaCalleeOutcome {
  Outcome outcome;
  std::optional<std::uint64_t> return_pc;
};

using SsaCallee = std::function<SsaCalleeOutcome(std::uint64_t, State&, Memory&, Budget&)>;

[[nodiscard]] SsaExecutionResult ExecuteSsa(const ir::SsaGraph&, std::span<const ir::Group> sources,
                                            ir::SsaHandle entry, State&, Memory&, Budget&,
                                            ExecutionContext, SsaCallee = {},
                                            std::size_t max_blocks = 4096, Limits = {});

}  // namespace nyx::eval
