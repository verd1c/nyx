#include "nyx/recovery/ssa/branch_retirement.hpp"

#include <algorithm>
#include <limits>

namespace nyx::recovery {
namespace {
SsaBranchRetirementResult Decline(SsaBranchRetirementRefusal reason) {
  SsaBranchRetirementResult result;
  result.reason = reason;
  return result;
}
}  // namespace

SsaBranchRetirementResult ProposeSsaBranchRetirement(const ir::SsaGraph& original,
                                                     const ir::SsaSccpFacts& facts,
                                                     std::span<const ir::Group> sources,
                                                     Budget& budget,
                                                     const ir::SsaBoundedLoopFacts* loops) {
  const auto valid = ir::ValidateSsaWithSources(original, sources, budget);
  if (valid != ir::SsaDecline::none)
    return Decline(valid == ir::SsaDecline::resource_limit
                       ? SsaBranchRetirementRefusal::resource_limit
                       : SsaBranchRetirementRefusal::invalid_graph);
  const auto checked = ir::ValidateSsaSccpFacts(original, facts, sources, budget, loops);
  if (checked != ir::SsaDecline::none)
    return Decline(checked == ir::SsaDecline::resource_limit
                       ? SsaBranchRetirementRefusal::resource_limit
                       : SsaBranchRetirementRefusal::stale_proof);
  SsaBranchRetirementResult result;
  if (budget.try_consume({original.slots(), original.slots()}) != BudgetDecline::none)
    return Decline(SsaBranchRetirementRefusal::resource_limit);
  std::vector<std::uint8_t> table_guards(original.slots());
  for (std::size_t slot = 0; slot < original.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return Decline(SsaBranchRetirementRefusal::resource_limit);
    const auto handle = original.Handle(slot);
    if (!handle) continue;
    const auto& block = *original.Get(*handle);
    if (budget.try_consume({block.constant_loads.size(), 0}) != BudgetDecline::none)
      return Decline(SsaBranchRetirementRefusal::resource_limit);
    for (const auto& fold : block.constant_loads)
      if (fold.kind == ir::SsaConstantKind::bounded_table && fold.table_guard)
        table_guards[fold.table_guard->slot] = 1;
  }

  for (std::size_t slot = 0; slot < original.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return Decline(SsaBranchRetirementRefusal::resource_limit);
    const auto handle = original.Handle(slot);
    if (!handle || !facts.executable[slot] || table_guards[slot]) continue;
    const auto& block = *original.Get(*handle);
    if (block.path_revision == UINT64_MAX || block.boundaries.empty() ||
        !block.boundaries.back().transfer || block.edges.size() != 2)
      continue;
    // A dispatcher's conditional is itself a rewrite over an indirect jump,
    // so its condition is the rewrite's, not the decoded transfer's.
    const ir::ConditionalRewrite* dispatch = nullptr;
    if (block.control_rewrites.size() == 1 &&
        block.control_rewrites.front().rule == ir::RewriteRule::dispatch_branch &&
        block.control_rewrites.front().boundary == block.boundaries.size() - 1 &&
        block.boundaries.back().transfer->kind == ir::TransferKind::jump) {
      dispatch = &block.control_rewrites.front();
    } else if (block.control_rewrites.size() ||
               block.boundaries.back().transfer->kind != ir::TransferKind::conditional) {
      continue;
    }

    const auto condition =
        dispatch ? dispatch->condition : *block.boundaries.back().transfer->condition;
    if (condition >= block.nodes.size()) continue;
    const auto& node = block.nodes[condition];
    if (node.op != ir::Op::constant || node.width != 1) continue;
    if (budget.try_consume({block.edges.size(), 0}) != BudgetDecline::none)
      return Decline(SsaBranchRetirementRefusal::resource_limit);
    std::size_t selected = SIZE_MAX, removed = SIZE_MAX;
    for (std::size_t index = 0; index < block.edges.size(); ++index) {
      if (facts.edges[slot][index])
        selected = index;
      else
        removed = index;
    }

    if (selected == SIZE_MAX || removed == SIZE_MAX || !block.edges[selected].condition ||
        !block.edges[removed].condition ||
        *block.edges[selected].when != bool(node.immediate & 1) ||
        *block.edges[removed].when == bool(node.immediate & 1))
      continue;
    if (result.journal.size() == result.journal.capacity()) {
      const auto next = result.journal.size() + 1;
      if (next > SIZE_MAX / sizeof(SsaBranchRetirementEdit) ||
          budget.try_consume({next, next * sizeof(SsaBranchRetirementEdit)}) != BudgetDecline::none)
        return Decline(SsaBranchRetirementRefusal::resource_limit);
      result.journal.reserve(next);
    } else if (budget.try_consume({1, 0}) != BudgetDecline::none) {
      return Decline(SsaBranchRetirementRefusal::resource_limit);
    }

    result.journal.push_back({*handle,
                              {},
                              static_cast<std::uint32_t>(removed),
                              block.edges[removed],
                              block.edges[selected],
                              original.revision(),
                              0,
                              dispatch != nullptr});
  }

  if (result.journal.empty()) return result;
  auto candidate = original.Clone(budget);
  if (!candidate) return Decline(SsaBranchRetirementRefusal::resource_limit);
  for (auto& edit : result.journal) {
    const auto handle = candidate->Handle(edit.original_block.slot);
    if (!handle) return Decline(SsaBranchRetirementRefusal::invalid_graph);
    auto copied = candidate->CopyBlock(*handle, budget);
    if (!copied) return Decline(SsaBranchRetirementRefusal::resource_limit);
    auto block = std::move(*copied);
    if (budget.try_consume({1, sizeof(ir::ConditionalRewrite)}) != BudgetDecline::none)
      return Decline(SsaBranchRetirementRefusal::resource_limit);
    if (edit.dispatcher) {
      // The rewrite that restored the conditional now states which way it
      // goes. Both destination nodes stay: they are why it is that one.
      if (block.control_rewrites.size() != 1 ||
          block.control_rewrites.front().rule != ir::RewriteRule::dispatch_branch)
        return Decline(SsaBranchRetirementRefusal::invalid_graph);
      auto& rewrite = block.control_rewrites.front();
      if (rewrite.condition >= block.nodes.size() || !rewrite.replacement.alternative)
        return Decline(SsaBranchRetirementRefusal::invalid_graph);
      const auto value = bool(block.nodes[rewrite.condition].immediate & 1);
      const auto selected = value ? rewrite.replacement.target : *rewrite.replacement.alternative;
      rewrite.rule = ir::RewriteRule::decided_dispatch;
      rewrite.replacement = {ir::TransferKind::jump, selected, {}, {}, {}};
      rewrite.condition_value = value;

      // Both ends, so the pair names the revisions this edit spans rather
      // than reaching back to when the dispatch was first restored.
      rewrite.from_revision = block.path_revision;
      rewrite.to_revision = block.path_revision + 1;
    } else {
      const auto& transfer = *block.boundaries.back().transfer;
      const auto condition = *transfer.condition;
      const auto value = bool(block.nodes[condition].immediate & 1);
      const auto selected = value ? transfer.target : *transfer.alternative;
      block.control_rewrites.push_back({ir::RewriteRule::folded_condition,
                                        static_cast<std::uint32_t>(block.boundaries.size() - 1),
                                        transfer,
                                        {ir::TransferKind::jump, selected, {}, {}, {}},
                                        condition,
                                        value,
                                        0,
                                        0,
                                        {},
                                        block.path_revision,
                                        block.path_revision + 1});
    }

    ++block.path_revision;
    const auto retained = block.edges[1 - edit.removed_edge];
    block.edges = {retained};
    block.edges.front().condition.reset();
    block.edges.front().when.reset();
    if (!candidate->Replace(*handle, std::move(block)))
      return Decline(SsaBranchRetirementRefusal::invalid_graph);
    edit.result_block = *handle;
  }

  for (const auto& edit : result.journal) {
    if (!edit.removed.target_block) return Decline(SsaBranchRetirementRefusal::invalid_graph);
    const auto predecessor = candidate->Handle(edit.original_block.slot);
    const auto target = candidate->Handle(edit.removed.target_block->slot);
    if (!predecessor || !target) return Decline(SsaBranchRetirementRefusal::invalid_graph);
    const auto& source_block = *candidate->Get(*predecessor);
    if (budget.try_consume({source_block.edges.size(), 0}) != BudgetDecline::none)
      return Decline(SsaBranchRetirementRefusal::resource_limit);
    if (std::any_of(source_block.edges.begin(), source_block.edges.end(),
                    [&](const auto& edge) { return edge.target_block == *target; }))
      continue;
    auto copied = candidate->CopyBlock(*target, budget);
    if (!copied) return Decline(SsaBranchRetirementRefusal::resource_limit);
    auto block = std::move(*copied);
    const auto prune = [&](auto& phis, bool frame) {
      for (std::size_t index = 0; index < phis.size(); ++index) {
        auto& incoming = phis[index].incoming;
        if (incoming.size() > (UINT64_MAX - 1) / 2 ||
            budget.try_consume({1 + 2 * incoming.size(), 0}) != BudgetDecline::none)
          return false;
        const auto found = std::find_if(incoming.begin(), incoming.end(), [&](const auto& input) {
          return input.predecessor == *predecessor;
        });
        if (found == incoming.end()) continue;
        if (result.phi_journal.size() == result.phi_journal.capacity()) {
          const auto next = result.phi_journal.size() + 1;
          if (next > SIZE_MAX / sizeof(SsaBranchRetirementPhiEdit) ||
              budget.try_consume({next, next * sizeof(SsaBranchRetirementPhiEdit)}) !=
                  BudgetDecline::none)
            return false;
          result.phi_journal.reserve(next);
        } else if (budget.try_consume({1, 0}) != BudgetDecline::none) {
          return false;
        }

        result.phi_journal.push_back({*edit.removed.target_block, *target, edit.original_block,
                                      static_cast<std::uint32_t>(index), frame, original.revision(),
                                      0});
        incoming.erase(found);
      }

      return true;
    };

    if (!prune(block.phis, false) || !prune(block.frame_phis, true))
      return Decline(SsaBranchRetirementRefusal::resource_limit);
    // A read recorded as copying the one incoming value copied it from this
    // predecessor, and with the edge gone the phi has no incoming to copy.
    for (auto& read : block.reads) {
      if (!read.predecessor_copy || read.phi >= block.phis.size() ||
          !block.phis[read.phi].incoming.empty())
        continue;
      read.predecessor_copy.reset();
      read.copy_closed_entries = false;
    }

    if (!candidate->Replace(*target, std::move(block)))
      return Decline(SsaBranchRetirementRefusal::invalid_graph);
  }

  const auto candidate_valid = ir::ValidateSsaWithSources(*candidate, sources, budget);
  if (candidate_valid != ir::SsaDecline::none)
    return Decline(candidate_valid == ir::SsaDecline::resource_limit
                       ? SsaBranchRetirementRefusal::resource_limit
                       : SsaBranchRetirementRefusal::invalid_graph);
  for (std::size_t slot = 0; slot < candidate->slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return Decline(SsaBranchRetirementRefusal::resource_limit);
    if (!facts.executable[slot]) continue;
    const auto handle = candidate->Handle(slot);
    if (!handle) return Decline(SsaBranchRetirementRefusal::invalid_graph);
    const auto complete = ir::ValidateSsaDirectSuccessors(*candidate->Get(*handle), budget);
    if (complete != ir::SsaDecline::none)
      return Decline(complete == ir::SsaDecline::resource_limit
                         ? SsaBranchRetirementRefusal::resource_limit
                         : SsaBranchRetirementRefusal::invalid_graph);
  }

  for (auto& edit : result.journal) edit.to_revision = candidate->revision();
  for (auto& edit : result.phi_journal) edit.to_revision = candidate->revision();
  result.provisional = std::move(*candidate);
  return result;
}

}  // namespace nyx::recovery
