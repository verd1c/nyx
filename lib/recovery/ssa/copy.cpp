#include "nyx/recovery/ssa/copy.hpp"

#include <algorithm>

namespace nyx::recovery {

SsaCopyProposal ProposeSsaPredecessorCopies(const ir::SsaGraph& original,
                                            const ir::SsaReachabilityFacts& reachable,
                                            const ir::SsaPredecessorCopyFacts& facts,
                                            std::span<const ir::Group> sources, Budget& budget) {
  const auto valid =
      ir::ValidateSsaPredecessorCopyFacts(original, reachable, facts, sources, budget);
  if (valid != ir::SsaDecline::none)
    return {{},
            {},
            valid == ir::SsaDecline::resource_limit ? SsaCopyRefusal::resource_limit
                                                    : SsaCopyRefusal::stale_proof};
  SsaCopyProposal result;
  for (const auto& fact : facts.copies) {
    const auto* block = original.Get(fact.block);
    if (!block) return {{}, {}, SsaCopyRefusal::invalid_graph};
    if (budget.try_consume({block->reads.size(), 0}) != BudgetDecline::none)
      return {{}, {}, SsaCopyRefusal::resource_limit};
    const auto read = std::find_if(block->reads.begin(), block->reads.end(),
                                   [&](const ir::SsaRead& item) { return item.node == fact.read; });
    if (read == block->reads.end()) return {{}, {}, SsaCopyRefusal::invalid_graph};
    if (read->predecessor_copy) continue;
    if (budget.try_consume({1, sizeof(SsaCopyEdit)}) != BudgetDecline::none)
      return {{}, {}, SsaCopyRefusal::resource_limit};
    result.journal.push_back({fact.block, {}, fact.read, fact.source, original.revision(), 0});
  }

  if (result.journal.empty()) return result;
  auto candidate = original.Clone(budget);
  if (!candidate) return {{}, {}, SsaCopyRefusal::resource_limit};
  for (std::size_t first = 0; first < result.journal.size();) {
    const auto slot = result.journal[first].original_block.slot;
    const auto handle = candidate->Handle(slot);
    if (!handle) return {{}, {}, SsaCopyRefusal::invalid_graph};
    auto copied = candidate->CopyBlock(*handle, budget);
    if (!copied) return {{}, {}, SsaCopyRefusal::resource_limit};
    auto block = std::move(*copied);
    do {
      auto& edit = result.journal[first++];
      if (budget.try_consume({block.reads.size(), 0}) != BudgetDecline::none)
        return {{}, {}, SsaCopyRefusal::resource_limit};
      auto read = std::find_if(block.reads.begin(), block.reads.end(),
                               [&](const ir::SsaRead& item) { return item.node == edit.read; });
      if (read == block.reads.end()) return {{}, {}, SsaCopyRefusal::invalid_graph};
      read->predecessor_copy = edit.source;
      read->predecessor_copy->block.arena = candidate->arena();
      read->copy_closed_entries = true;
      edit.result_block = *handle;
      edit.source.block.arena = candidate->arena();
    } while (first < result.journal.size() && result.journal[first].original_block.slot == slot);
    if (!candidate->Replace(*handle, std::move(block)))
      return {{}, {}, SsaCopyRefusal::invalid_graph};
  }

  const auto checked = ir::ValidateSsaWithSources(*candidate, sources, budget);
  if (checked != ir::SsaDecline::none)
    return {{},
            {},
            checked == ir::SsaDecline::resource_limit ? SsaCopyRefusal::resource_limit
                                                      : SsaCopyRefusal::invalid_graph};
  for (auto& edit : result.journal) edit.to_revision = candidate->revision();
  result.provisional = std::move(*candidate);
  return result;
}

}  // namespace nyx::recovery
