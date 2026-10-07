#include "nyx/recovery/ssa/dead_writes.hpp"

#include <algorithm>
#include <bit>

namespace nyx::recovery {
namespace {
SsaDeadWriteProposal Decline(SsaDeadWriteRefusal reason) {
  SsaDeadWriteProposal result;
  result.reason = reason;
  return result;
}
}  // namespace

SsaDeadWriteProposal ProposeSsaDeadWrites(const ir::SsaGraph& original,
                                          const ir::SsaReachabilityFacts& reachable,
                                          const ir::SsaDeadWriteFacts& facts,
                                          std::span<const ir::Group> sources, Budget& budget) {
  const auto checked = ir::ValidateSsaDeadWriteFacts(original, reachable, facts, sources, budget);
  if (checked != ir::SsaDecline::none)
    return Decline(checked == ir::SsaDecline::resource_limit ? SsaDeadWriteRefusal::resource_limit
                                                             : SsaDeadWriteRefusal::stale_proof);
  if (facts.writes.empty()) return {};

  // Facts proved on one graph stay dead together, so the batch normally
  // validates at once; the one-at-a-time fallback isolates any that do not.
  auto batch = original.Clone(budget);
  if (!batch) return Decline(SsaDeadWriteRefusal::resource_limit);
  {
    SsaDeadWriteProposal result;
    for (const auto& fact : facts.writes) {
      const auto handle = batch->Handle(fact.block.slot);
      if (!handle) return Decline(SsaDeadWriteRefusal::invalid_graph);
      auto changed = batch->CopyBlock(*handle, budget);
      if (!changed) return Decline(SsaDeadWriteRefusal::resource_limit);
      auto block = std::move(*changed);
      if (fact.boundary >= block.boundaries.size() ||
          fact.index >= block.boundaries[fact.boundary].writes.size())
        return Decline(SsaDeadWriteRefusal::invalid_graph);
      const auto work = std::uint64_t{1} + std::bit_width(block.dead_storage_writes.size()) +
                        block.dead_storage_writes.size();
      if (budget.try_consume({work, sizeof(ir::SsaDeadStorageWrite) + sizeof(SsaDeadWriteEdit)}) !=
          BudgetDecline::none)
        return Decline(SsaDeadWriteRefusal::resource_limit);
      const auto at =
          std::lower_bound(block.dead_storage_writes.begin(), block.dead_storage_writes.end(),
                           std::pair{fact.boundary, fact.index},
                           [](const ir::SsaDeadStorageWrite& mark, const auto& key) {
                             return mark.boundary < key.first ||
                                    (mark.boundary == key.first && mark.index < key.second);
                           });
      block.dead_storage_writes.insert(at, {fact.boundary, fact.index, true});
      result.journal.push_back({fact.block, *handle, fact.boundary, fact.index,
                                block.boundaries[fact.boundary].writes[fact.index],
                                original.revision(), 0});
      if (!batch->Replace(*handle, std::move(block)))
        return Decline(SsaDeadWriteRefusal::invalid_graph);
    }

    const auto valid = ir::ValidateSsaWithSources(*batch, sources, budget);
    if (valid == ir::SsaDecline::resource_limit)
      return Decline(SsaDeadWriteRefusal::resource_limit);
    if (valid == ir::SsaDecline::none) {
      for (auto& edit : result.journal) edit.to_revision = batch->revision();
      result.provisional = std::move(*batch);
      return result;
    }
  }

  auto candidate = original.Clone(budget);
  if (!candidate) return Decline(SsaDeadWriteRefusal::resource_limit);
  SsaDeadWriteProposal result;
  for (const auto& fact : facts.writes) {
    const auto handle = candidate->Handle(fact.block.slot);
    if (!handle) return Decline(SsaDeadWriteRefusal::invalid_graph);
    auto copied = candidate->CopyBlock(*handle, budget);
    if (!copied) return Decline(SsaDeadWriteRefusal::resource_limit);
    auto saved = std::move(*copied);
    auto changed = candidate->CopyBlock(*handle, budget);
    if (!changed) return Decline(SsaDeadWriteRefusal::resource_limit);
    auto block = std::move(*changed);
    if (fact.boundary >= block.boundaries.size() ||
        fact.index >= block.boundaries[fact.boundary].writes.size())
      return Decline(SsaDeadWriteRefusal::invalid_graph);
    const auto original_write = block.boundaries[fact.boundary].writes[fact.index];
    const auto work = std::uint64_t{1} + std::bit_width(block.dead_storage_writes.size()) +
                      block.dead_storage_writes.size();
    if (budget.try_consume({work, sizeof(ir::SsaDeadStorageWrite) + sizeof(SsaDeadWriteEdit)}) !=
        BudgetDecline::none)
      return Decline(SsaDeadWriteRefusal::resource_limit);
    const auto at =
        std::lower_bound(block.dead_storage_writes.begin(), block.dead_storage_writes.end(),
                         std::pair{fact.boundary, fact.index},
                         [](const ir::SsaDeadStorageWrite& mark, const auto& key) {
                           return mark.boundary < key.first ||
                                  (mark.boundary == key.first && mark.index < key.second);
                         });
    block.dead_storage_writes.insert(at, {fact.boundary, fact.index, true});
    if (!candidate->Replace(*handle, std::move(block)))
      return Decline(SsaDeadWriteRefusal::invalid_graph);
    const auto valid = ir::ValidateSsaWithSources(*candidate, sources, budget);
    if (valid == ir::SsaDecline::resource_limit)
      return Decline(SsaDeadWriteRefusal::resource_limit);
    if (valid != ir::SsaDecline::none) {
      if (!candidate->Replace(*handle, std::move(saved)))
        return Decline(SsaDeadWriteRefusal::invalid_graph);
      if (budget.try_consume({1, sizeof(SsaDeadWriteRejected)}) != BudgetDecline::none)
        return Decline(SsaDeadWriteRefusal::resource_limit);
      result.refused.push_back({fact.block, fact.boundary, fact.index});
      continue;
    }

    result.journal.push_back(
        {fact.block, *handle, fact.boundary, fact.index, original_write, original.revision(), 0});
  }

  if (result.journal.empty()) return result;
  for (auto& edit : result.journal) edit.to_revision = candidate->revision();
  result.provisional = std::move(*candidate);
  return result;
}

}  // namespace nyx::recovery
