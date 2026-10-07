#include "nyx/analysis/ssa/cache.hpp"

namespace nyx::analysis {

void SsaProofCache::ResetSources(std::span<const ir::Group> sources) {
  sources_ = sources;
  reachability_.reset();
  sccp_.reset();
  loops_.reset();
  bound_ = false;
  ++counts_.invalidations;
}

void SsaProofCache::Bind(const ir::SsaGraph& graph) {
  if (bound_ && arena_ == graph.arena() && revision_ == graph.revision()) return;
  if (bound_) ++counts_.invalidations;
  reachability_.reset();
  sccp_.reset();
  loops_.reset();
  arena_ = graph.arena();
  revision_ = graph.revision();
  bound_ = true;
}

const SsaReachabilityResult& SsaProofCache::Reachability(const ir::SsaGraph& graph,
                                                         Budget& budget) {
  Bind(graph);
  if (reachability_) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return reachability_limit_;
    return *reachability_;
  }

  ++counts_.reachability_computations;
  auto proved = ProveSsaReachability(graph, sources_, scope_, budget);
  if (proved.reason == SsaReachabilityRefusal::resource_limit) return reachability_limit_;
  reachability_ = std::move(proved);
  return *reachability_;
}

const SsaSccpResult& SsaProofCache::Sccp(const ir::SsaGraph& graph, Budget& budget) {
  const auto* loops = Loops(graph, budget);
  if (sccp_) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return sccp_limit_;
    return *sccp_;
  }

  ++counts_.sccp_computations;
  auto proved = ProveSsaSccp(graph, scope_, sources_, budget, loops);
  if (proved.reason == SsaSccpRefusal::resource_limit) return sccp_limit_;
  sccp_ = std::move(proved);
  return *sccp_;
}

const ir::SsaBoundedLoopFacts* SsaProofCache::Loops(const ir::SsaGraph& graph, Budget& budget) {
  Bind(graph);
  if (!loops_) {
    ++counts_.bounded_loop_computations;
    auto proved = ProveSsaBoundedLoops(graph, sources_, budget);

    // A refusal is cached as "nothing proved": the lattices are correct
    // without these values and only weaker, so a retry per query is waste.
    loops_ = std::move(proved);
  }

  return loops_->facts && !loops_->facts->loops.empty() ? &*loops_->facts : nullptr;
}

}  // namespace nyx::analysis
