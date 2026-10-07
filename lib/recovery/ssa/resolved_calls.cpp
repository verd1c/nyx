#include "nyx/recovery/ssa/resolved_calls.hpp"

#include <algorithm>

namespace nyx::recovery {
namespace {

SsaResolvedCallResult Decline(SsaResolvedCallRefusal reason) {
  SsaResolvedCallResult result;
  result.reason = reason;
  return result;
}

SsaResolvedCallRefusal From(ir::SsaDecline decline) {
  return decline == ir::SsaDecline::resource_limit ? SsaResolvedCallRefusal::resource_limit
                                                   : SsaResolvedCallRefusal::invalid_graph;
}

}  // namespace

SsaResolvedCallResult ProposeSsaResolvedCalls(const ir::SsaGraph& original,
                                              std::span<const ir::Group> sources, Budget& budget) {
  const auto valid = ir::ValidateSsaWithSources(original, sources, budget);
  if (valid != ir::SsaDecline::none) return Decline(From(valid));
  SsaResolvedCallResult result;

  struct Planned {
    ir::SsaHandle block;
    std::uint64_t callee;
  };

  std::vector<Planned> planned;
  for (std::size_t slot = 0; slot < original.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return Decline(SsaResolvedCallRefusal::resource_limit);
    const auto handle = original.Handle(slot);
    if (!handle) continue;
    const auto& block = *original.Get(*handle);
    if (block.opaque || !block.control_rewrites.empty() || block.boundaries.empty() ||
        !block.boundaries.back().transfer ||
        block.boundaries.back().transfer->kind != ir::TransferKind::call)
      continue;
    const auto& transfer = *block.boundaries.back().transfer;

    // A target already a literal names its callee; there is nothing to resolve.
    if (transfer.target >= block.nodes.size() ||
        block.nodes[transfer.target].op == ir::Op::image_address)
      continue;
    const auto edge =
        std::find_if(block.edges.begin(), block.edges.end(),
                     [](const ir::SsaEdge& item) { return item.kind == ir::SsaEdgeKind::callee; });
    if (edge == block.edges.end() || edge->target_kind != ir::SsaTargetKind::unknown) continue;
    const auto callee = ir::SsaSettledTarget(original, block, transfer.target, budget);
    if (!callee) continue;

    // A call to a leaf is for retiring outright once what it writes is dead.
    if (ir::SsaLeafCalleeReads(original, block, budget)) continue;
    planned.push_back({*handle, *callee});
  }

  if (planned.empty()) return result;

  auto candidate = original.Clone(budget);
  if (!candidate) return Decline(SsaResolvedCallRefusal::resource_limit);
  for (const auto& plan : planned) {
    const auto here = candidate->Handle(plan.block.slot);
    if (!here) return Decline(SsaResolvedCallRefusal::invalid_graph);
    auto copied = candidate->CopyBlock(*here, budget);
    if (!copied || budget.try_consume({1, sizeof(ir::ConditionalRewrite) + sizeof(ir::Node) +
                                              sizeof(SsaResolvedCallEdit)}) != BudgetDecline::none)
      return Decline(SsaResolvedCallRefusal::resource_limit);
    auto block = std::move(*copied);
    const auto transfer = *block.boundaries.back().transfer;
    const auto destination =
        static_cast<ir::ValueId>(block.nodes.size() + block.destination_nodes.size());
    block.destination_nodes.push_back({ir::Op::image_address, 64, {}, plan.callee});
    const auto from = block.path_revision;
    block.control_rewrites.push_back(
        {ir::RewriteRule::resolved_call,
         static_cast<std::uint32_t>(block.boundaries.size() - 1),
         transfer,
         {ir::TransferKind::call, destination, {}, {}, transfer.continuation},
         transfer.target,
         false,
         plan.callee,
         0,
         {},
         from,
         from + 1});
    ++block.path_revision;
    for (auto& edge : block.edges) {
      if (edge.kind != ir::SsaEdgeKind::callee) continue;
      edge.target_kind = ir::SsaTargetKind::image_location;
      edge.address = plan.callee;
    }

    if (!candidate->Replace(*here, std::move(block)))
      return Decline(SsaResolvedCallRefusal::invalid_graph);
    result.journal.push_back({plan.block, plan.callee, original.revision(), 0});
  }

  const auto checked = ir::ValidateSsaWithSources(*candidate, sources, budget);
  if (checked != ir::SsaDecline::none) return Decline(From(checked));
  std::vector<std::uint32_t> rewritten;
  for (const auto& plan : planned) rewritten.push_back(plan.block.slot);
  const auto evidence = ir::ValidateSsaRetiredCalls(*candidate, rewritten, budget);
  if (evidence != ir::SsaDecline::none) return Decline(From(evidence));
  for (const auto& edit : result.journal) {
    const auto complete = ir::ValidateSsaDirectSuccessors(
        *candidate->Get(*candidate->Handle(edit.block.slot)), budget);
    if (complete != ir::SsaDecline::none) return Decline(From(complete));
  }

  for (auto& edit : result.journal) edit.to_revision = candidate->revision();
  result.provisional = std::move(*candidate);
  return result;
}

}  // namespace nyx::recovery
