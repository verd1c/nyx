#pragma once

#include "nyx/analysis/ssa/bounded_loop.hpp"
#include "nyx/analysis/ssa/reachability.hpp"
#include "nyx/analysis/ssa/sccp.hpp"

namespace nyx::analysis {

// One function's immutable decoded source population. A changed graph revision
// or an explicit source reset invalidates all dependent facts.
class SsaProofCache {
 public:
  struct Counts {
    std::uint64_t reachability_computations = 0;
    std::uint64_t sccp_computations = 0;
    std::uint64_t bounded_loop_computations = 0;
    std::uint64_t invalidations = 0;
  };

  SsaProofCache(std::span<const ir::Group> sources, ir::SsaEntryScope scope)
      : sources_(sources), scope_(scope) {}

  SsaProofCache(const SsaProofCache&) = delete;
  SsaProofCache& operator=(const SsaProofCache&) = delete;
  SsaProofCache(SsaProofCache&&) = delete;
  SsaProofCache& operator=(SsaProofCache&&) = delete;
  void ResetSources(std::span<const ir::Group> sources);
  const SsaReachabilityResult& Reachability(const ir::SsaGraph&, Budget&);
  const SsaSccpResult& Sccp(const ir::SsaGraph&, Budget&);

  // What each self-looping block leaves its successor, or null where none
  // was proved. The constant lattices take it as a further source of values,
  // so it is computed before them and shares their invalidation.
  const ir::SsaBoundedLoopFacts* Loops(const ir::SsaGraph&, Budget&);

  // Why the last Loops() found nothing, for the stage that prints it.
  SsaBoundedLoopRefusal loops_reason() const {
    return loops_ ? loops_->reason : SsaBoundedLoopRefusal::none;
  }

  Counts counts() const { return counts_; }

 private:
  void Bind(const ir::SsaGraph&);

  std::span<const ir::Group> sources_;
  ir::SsaEntryScope scope_;
  std::uint64_t arena_ = 0;
  std::uint64_t revision_ = 0;
  bool bound_ = false;
  std::optional<SsaReachabilityResult> reachability_;
  std::optional<SsaSccpResult> sccp_;
  std::optional<SsaBoundedLoopResult> loops_;
  SsaReachabilityResult reachability_limit_{{}, SsaReachabilityRefusal::resource_limit};
  SsaSccpResult sccp_limit_{{}, SsaSccpRefusal::resource_limit};
  Counts counts_;
};

}  // namespace nyx::analysis
