#include "nyx/recovery/ssa/dce.hpp"

#include <algorithm>
#include <limits>

namespace nyx::recovery {
namespace {
SsaDceResult Decline(SsaDceRefusal reason, ir::SsaEntryScope scope) {
  SsaDceResult result;
  result.reason = reason;
  result.entry_scope = scope;
  return result;
}
}  // namespace

SsaDceResult ProposeUnreachableBlocks(const ir::SsaGraph& original,
                                      const ir::SsaReachabilityFacts& facts,
                                      std::span<const ir::Group> sources, Budget& budget) {
  const auto scope = facts.entry_scope;
  const auto checked = ir::ValidateSsaWithSources(original, sources, budget);
  if (checked != ir::SsaDecline::none)
    return Decline(checked == ir::SsaDecline::resource_limit ? SsaDceRefusal::resource_limit
                                                             : SsaDceRefusal::invalid_graph,
                   scope);
  const auto proof = ir::ValidateSsaReachabilityFacts(original, facts, sources, budget);
  if (proof != ir::SsaDecline::none)
    return Decline(proof == ir::SsaDecline::resource_limit ? SsaDceRefusal::resource_limit
                                                           : SsaDceRefusal::stale_proof,
                   scope);
  const auto count = original.slots();
  if (count > std::numeric_limits<std::size_t>::max() / 3 ||
      budget.try_consume({3 * count, 0}) != BudgetDecline::none)
    return Decline(SsaDceRefusal::resource_limit, scope);
  const auto& reachable = facts.reachable;
  std::size_t dead_count = 0;
  for (std::size_t slot = 0; slot < count; ++slot)
    if (original.Handle(slot) && !reachable[slot]) ++dead_count;
  SsaDceResult result;
  result.entry_scope = scope;
  if (!dead_count) return result;

  // Erase checks references by scanning every surviving slot. Charge the
  // worst case for each deletion before the first graph mutation.
  std::uint64_t erase_scan = original.entries().size() + count;
  const auto add_scan = [&](std::uint64_t work) {
    if (erase_scan > std::numeric_limits<std::uint64_t>::max() - work) return false;
    erase_scan += work;
    return true;
  };

  for (std::size_t slot = 0; slot < count; ++slot) {
    const auto handle = original.Handle(slot);
    if (!handle) continue;
    const auto& block = *original.Get(*handle);
    if (budget.try_consume({1 + block.phis.size() + block.frame_phis.size(), 0}) !=
            BudgetDecline::none ||
        !add_scan(block.edges.size()) || !add_scan(block.reads.size()) ||
        !add_scan(block.phis.size()) || !add_scan(block.frame_phis.size()))
      return Decline(SsaDceRefusal::resource_limit, scope);
    for (const auto& phi : block.phis)
      if (!add_scan(phi.incoming.size())) return Decline(SsaDceRefusal::resource_limit, scope);
    for (const auto& phi : block.frame_phis)
      if (!add_scan(phi.incoming.size())) return Decline(SsaDceRefusal::resource_limit, scope);
  }

  if (dead_count > std::numeric_limits<std::uint64_t>::max() / erase_scan ||
      budget.try_consume({dead_count * erase_scan, 0}) != BudgetDecline::none)
    return Decline(SsaDceRefusal::resource_limit, scope);
  auto candidate = original.Clone(budget);
  if (!candidate) return Decline(SsaDceRefusal::resource_limit, scope);
  const auto append = [&](SsaDceEditKind kind, ir::SsaHandle block,
                          std::optional<ir::SsaHandle> replacement, std::uint32_t index,
                          ir::SsaHandle predecessor) {
    if (budget.try_consume({1, sizeof(SsaDceEdit)}) != BudgetDecline::none) return false;
    result.journal.push_back(
        {kind, block, replacement, index, predecessor, original.revision(), 0});
    return true;
  };

  for (std::size_t slot = 0; slot < count; ++slot) {
    const auto handle = candidate->Handle(slot);
    if (!handle || !reachable[slot]) continue;
    const auto* current = candidate->Get(*handle);
    bool prune = false;
    for (const auto& phi : current->phis) {
      if (budget.try_consume({1 + phi.incoming.size(), 0}) != BudgetDecline::none)
        return Decline(SsaDceRefusal::resource_limit, scope);
      for (const auto& input : phi.incoming) prune |= !reachable[input.predecessor.slot];
    }

    for (const auto& phi : current->frame_phis) {
      if (budget.try_consume({1 + phi.incoming.size(), 0}) != BudgetDecline::none)
        return Decline(SsaDceRefusal::resource_limit, scope);
      for (const auto& input : phi.incoming) prune |= !reachable[input.predecessor.slot];
    }

    if (!prune) continue;
    auto copied = candidate->CopyBlock(*handle, budget);
    if (!copied) return Decline(SsaDceRefusal::resource_limit, scope);
    auto block = std::move(*copied);
    const ir::SsaHandle original_handle{original.arena(), handle->slot, handle->generation};
    for (std::size_t index = 0; index < block.phis.size(); ++index) {
      auto& incoming = block.phis[index].incoming;
      if (incoming.size() > std::numeric_limits<std::size_t>::max() / 2 ||
          budget.try_consume({2 * incoming.size() + 1, 0}) != BudgetDecline::none)
        return Decline(SsaDceRefusal::resource_limit, scope);
      for (const auto& input : incoming) {
        if (reachable[input.predecessor.slot]) continue;
        const ir::SsaHandle predecessor{original.arena(), input.predecessor.slot,
                                        input.predecessor.generation};
        if (!append(SsaDceEditKind::pruned_storage_phi, original_handle, *handle,
                    static_cast<std::uint32_t>(index), predecessor))
          return Decline(SsaDceRefusal::resource_limit, scope);
      }

      std::erase_if(incoming,
                    [&](const auto& input) { return !reachable[input.predecessor.slot]; });
    }

    for (std::size_t index = 0; index < block.frame_phis.size(); ++index) {
      auto& incoming = block.frame_phis[index].incoming;
      if (incoming.size() > std::numeric_limits<std::size_t>::max() / 2 ||
          budget.try_consume({2 * incoming.size() + 1, 0}) != BudgetDecline::none)
        return Decline(SsaDceRefusal::resource_limit, scope);
      for (const auto& input : incoming) {
        if (reachable[input.predecessor.slot]) continue;
        const ir::SsaHandle predecessor{original.arena(), input.predecessor.slot,
                                        input.predecessor.generation};
        if (!append(SsaDceEditKind::pruned_frame_phi, original_handle, *handle,
                    static_cast<std::uint32_t>(index), predecessor))
          return Decline(SsaDceRefusal::resource_limit, scope);
      }

      std::erase_if(incoming,
                    [&](const auto& input) { return !reachable[input.predecessor.slot]; });
    }

    if (!candidate->Replace(*handle, std::move(block)))
      return Decline(SsaDceRefusal::invalid_graph, scope);
  }

  // Clear references inside the unreachable subgraph before erasing any of
  // its handles. No intermediate graph is published or validated.
  for (std::size_t slot = 0; slot < count; ++slot) {
    const auto handle = candidate->Handle(slot);
    if (!handle || reachable[slot]) continue;
    auto copied = candidate->CopyBlock(*handle, budget);
    if (!copied) return Decline(SsaDceRefusal::resource_limit, scope);
    auto block = std::move(*copied);
    block.edges.clear();
    for (auto& phi : block.phis) phi.incoming.clear();
    for (auto& phi : block.frame_phis) phi.incoming.clear();

    // A copy record names the predecessor it copies from, which may be
    // another block about to go.
    for (auto& read : block.reads) {
      read.predecessor_copy.reset();
      read.copy_closed_entries = false;
    }

    if (!candidate->Replace(*handle, std::move(block)))
      return Decline(SsaDceRefusal::invalid_graph, scope);
  }

  for (std::size_t slot = 0; slot < count; ++slot) {
    const auto handle = candidate->Handle(slot);
    if (!handle || reachable[slot]) continue;
    const ir::SsaHandle source{original.arena(), handle->slot, handle->generation};
    if (!append(SsaDceEditKind::removed_block, source, {}, 0, {}))
      return Decline(SsaDceRefusal::resource_limit, scope);
    if (!candidate->Erase(*handle)) return Decline(SsaDceRefusal::invalid_graph, scope);
  }

  const auto valid = ir::ValidateSsaWithSources(*candidate, sources, budget);
  if (valid != ir::SsaDecline::none)
    return Decline(valid == ir::SsaDecline::resource_limit ? SsaDceRefusal::resource_limit
                                                           : SsaDceRefusal::invalid_graph,
                   scope);
  for (auto& edit : result.journal) edit.to_revision = candidate->revision();
  result.provisional = std::move(*candidate);
  return result;
}

}  // namespace nyx::recovery
