#include "nyx/recovery/ssa/pure_dce.hpp"

#include <algorithm>
#include <bit>
#include <limits>

namespace nyx::recovery {
namespace {
SsaPureDceResult Decline(SsaPureDceRefusal reason) {
  SsaPureDceResult result;
  result.reason = reason;
  return result;
}
}  // namespace

SsaPureDceResult ProposeDeadPureNodesImpl(const ir::SsaGraph& original,
                                          const ir::SsaLivenessFacts& facts,
                                          std::optional<std::span<const ir::Group>> sources,
                                          Budget& budget) {
  const auto valid = sources ? ir::ValidateSsaWithSources(original, *sources, budget)
                             : ir::ValidateSsa(original, budget);
  if (valid != ir::SsaDecline::none)
    return Decline(valid == ir::SsaDecline::resource_limit ? SsaPureDceRefusal::resource_limit
                                                           : SsaPureDceRefusal::invalid_graph);
  const auto checked = ir::ValidateSsaLivenessFacts(original, facts, budget);
  if (checked != ir::SsaDecline::none)
    return Decline(checked == ir::SsaDecline::resource_limit ? SsaPureDceRefusal::resource_limit
                                                             : SsaPureDceRefusal::stale_proof);
  std::size_t edits = 0;
  for (std::size_t slot = 0; slot < original.slots(); ++slot) {
    const auto handle = original.Handle(slot);
    if (!handle) continue;
    const auto& block = *original.Get(*handle);
    const auto node_work = 1 + std::bit_width(block.constant_loads.size());
    if (block.nodes.size() > std::numeric_limits<std::size_t>::max() / node_work ||
        budget.try_consume({block.nodes.size() * node_work, 0}) != BudgetDecline::none ||
        budget.try_consume({block.dead_pure_nodes.size(), 0}) != BudgetDecline::none)
      return Decline(SsaPureDceRefusal::resource_limit);
    std::size_t old = 0;
    for (std::size_t id = 0; id < block.nodes.size(); ++id) {
      while (old < block.dead_pure_nodes.size() && block.dead_pure_nodes[old] < id) ++old;
      if (!facts.live_nodes[slot][id] && ir::SsaEffectFree(block, static_cast<ir::ValueId>(id)) &&
          (old == block.dead_pure_nodes.size() || block.dead_pure_nodes[old] != id))
        ++edits;
    }
  }

  if (!edits) return {};
  auto candidate = original.Clone(budget);
  if (!candidate) return Decline(SsaPureDceRefusal::resource_limit);
  SsaPureDceResult result;
  for (std::size_t slot = 0; slot < candidate->slots(); ++slot) {
    const auto handle = candidate->Handle(slot);
    if (!handle) continue;
    const auto& source = *original.Get(*original.Handle(slot));
    const auto& live = facts.live_nodes[slot];
    const auto node_work = 1 + std::bit_width(source.constant_loads.size());
    if (source.nodes.size() > std::numeric_limits<std::size_t>::max() / node_work ||
        budget.try_consume({source.nodes.size() * node_work, 0}) != BudgetDecline::none ||
        budget.try_consume({source.dead_pure_nodes.size(), 0}) != BudgetDecline::none)
      return Decline(SsaPureDceRefusal::resource_limit);
    std::vector<ir::ValueId> added;
    std::size_t old = 0;
    for (std::size_t id = 0; id < source.nodes.size(); ++id) {
      while (old < source.dead_pure_nodes.size() && source.dead_pure_nodes[old] < id) ++old;
      if (live[id] || !ir::SsaEffectFree(source, static_cast<ir::ValueId>(id)) ||
          (old < source.dead_pure_nodes.size() && source.dead_pure_nodes[old] == id))
        continue;
      if (budget.try_consume({1, sizeof(ir::ValueId) + sizeof(SsaPureDceEdit)}) !=
          BudgetDecline::none)
        return Decline(SsaPureDceRefusal::resource_limit);
      added.push_back(static_cast<ir::ValueId>(id));
      result.journal.push_back(
          {*original.Handle(slot), *handle, static_cast<ir::ValueId>(id), original.revision(), 0});
    }

    if (added.empty()) continue;
    auto copied = candidate->CopyBlock(*handle, budget);
    if (!copied) return Decline(SsaPureDceRefusal::resource_limit);
    auto block = std::move(*copied);
    if (added.size() > std::numeric_limits<std::size_t>::max() / sizeof(ir::ValueId) ||
        block.dead_pure_nodes.size() >
            (std::numeric_limits<std::size_t>::max() / sizeof(ir::ValueId)) - added.size() ||
        budget.try_consume({block.dead_pure_nodes.size() + added.size(),
                            (block.dead_pure_nodes.size() + added.size()) * sizeof(ir::ValueId)}) !=
            BudgetDecline::none)
      return Decline(SsaPureDceRefusal::resource_limit);
    std::vector<ir::ValueId> merged;
    merged.reserve(block.dead_pure_nodes.size() + added.size());
    std::merge(block.dead_pure_nodes.begin(), block.dead_pure_nodes.end(), added.begin(),
               added.end(), std::back_inserter(merged));
    block.dead_pure_nodes = std::move(merged);
    if (!candidate->Replace(*handle, std::move(block)))
      return Decline(SsaPureDceRefusal::invalid_graph);
  }

  const auto validated = sources ? ir::ValidateSsaWithSources(*candidate, *sources, budget)
                                 : ir::ValidateSsa(*candidate, budget);
  if (validated != ir::SsaDecline::none)
    return Decline(validated == ir::SsaDecline::resource_limit ? SsaPureDceRefusal::resource_limit
                                                               : SsaPureDceRefusal::invalid_graph);
  for (auto& edit : result.journal) edit.to_revision = candidate->revision();
  result.provisional = std::move(*candidate);
  return result;
}

SsaPureDceResult ProposeDeadPureNodes(const ir::SsaGraph& original,
                                      const ir::SsaLivenessFacts& facts, Budget& budget) {
  return ProposeDeadPureNodesImpl(original, facts, std::nullopt, budget);
}

SsaPureDceResult ProposeDeadPureNodes(const ir::SsaGraph& original,
                                      const ir::SsaLivenessFacts& facts,
                                      std::span<const ir::Group> sources, Budget& budget) {
  return ProposeDeadPureNodesImpl(original, facts, sources, budget);
}

}  // namespace nyx::recovery
