#include "nyx/recovery/ssa/leaf_calls.hpp"

#include <algorithm>

namespace nyx::recovery {
namespace {

SsaLeafCallResult Decline(SsaLeafCallRefusal reason) {
  SsaLeafCallResult result;
  result.reason = reason;
  return result;
}

SsaLeafCallRefusal From(ir::SsaDecline decline) {
  return decline == ir::SsaDecline::resource_limit ? SsaLeafCallRefusal::resource_limit
                                                   : SsaLeafCallRefusal::invalid_graph;
}

// The storages a body writes, and the one it returns through, when it runs
// straight through to a return, touching neither memory nor a monitor and
// never writing that register.
struct Leaf {
  std::vector<ir::StorageId> written;
  ir::StorageId link = 0;
};

std::optional<Leaf> AsLeaf(const ir::SsaCalleeBody& body) {
  if (body.groups.empty() || body.groups.size() > ir::kMaxCalleeBodyGroups) return std::nullopt;
  Leaf leaf;
  auto address = body.address;
  for (std::size_t index = 0; index < body.groups.size(); ++index) {
    const auto& group = body.groups[index];
    if (group.source_address() != address || group.bytes().empty()) return std::nullopt;
    address += group.bytes().size();
    for (const auto& node : group.nodes()) {
      if (node.op == ir::Op::write) {
        leaf.written.push_back(node.storage);
        continue;
      }

      const auto* descriptor = ir::Descriptor(node.op);
      if (!descriptor || (descriptor->effect != ir::Effect::pure &&
                          descriptor->effect != ir::Effect::storage_read))
        return std::nullopt;
    }

    for (const auto& write : group.writes()) leaf.written.push_back(write.storage);
    const auto& transfer = group.transfer();
    if (index + 1 < body.groups.size()) {
      if (transfer) return std::nullopt;
      continue;
    }

    if (!transfer || transfer->kind != ir::TransferKind::return_ ||
        transfer->target >= group.nodes().size() ||
        group.nodes()[transfer->target].op != ir::Op::read)
      return std::nullopt;
    leaf.link = group.nodes()[transfer->target].storage;
  }

  if (std::find(leaf.written.begin(), leaf.written.end(), leaf.link) != leaf.written.end())
    return std::nullopt;
  return leaf;
}

}  // namespace

SsaLeafCallResult ProposeSsaLeafCalls(const ir::SsaGraph& original,
                                      std::span<const ir::Group> sources, Budget& budget) {
  const auto valid = ir::ValidateSsaWithSources(original, sources, budget);
  if (valid != ir::SsaDecline::none) return Decline(From(valid));
  if (original.callee_bodies().empty()) return {};
  const auto liveness = ir::SsaStorageLiveness::Compute(original, budget);
  if (!liveness) return Decline(SsaLeafCallRefusal::resource_limit);

  SsaLeafCallResult result;

  struct Planned {
    ir::SsaHandle block;
    std::uint64_t callee;
  };

  std::vector<Planned> planned;
  for (std::size_t slot = 0; slot < original.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return Decline(SsaLeafCallRefusal::resource_limit);
    const auto handle = original.Handle(slot);
    if (!handle) continue;
    const auto& block = *original.Get(*handle);
    if (block.opaque || !block.control_rewrites.empty() || block.boundaries.empty() ||
        !block.boundaries.back().transfer ||
        block.boundaries.back().transfer->kind != ir::TransferKind::call ||
        !block.boundaries.back().transfer->continuation)
      continue;
    const ir::SsaEdge* back = nullptr;
    for (const auto& edge : block.edges)
      if (edge.kind == ir::SsaEdgeKind::potential_return) back = &edge;
    if (!back || !back->target_block) continue;
    const auto callee = ir::SsaCallTarget(original, block, budget);
    if (!callee) continue;
    const auto bodies = original.callee_bodies();
    const auto body =
        std::find_if(bodies.begin(), bodies.end(),
                     [&](const ir::SsaCalleeBody& item) { return item.address == *callee; });
    if (budget.try_consume({bodies.size() + block.nodes.size(), sizeof(SsaLeafCallRefused)}) !=
        BudgetDecline::none)
      return Decline(SsaLeafCallRefusal::resource_limit);
    if (body == bodies.end()) {
      result.refused.push_back({*handle, *callee, SsaLeafCallDecline::no_body});
      continue;
    }

    const auto leaf = AsLeaf(*body);

    // The call must leave its continuation in the register the body returns
    // through, or the body's return goes somewhere else.
    const auto& transfer = *block.boundaries.back().transfer;
    const auto& boundary = block.boundaries.back();
    bool linked = false;
    if (leaf) {
      for (const auto& write : boundary.writes)
        if (write.storage == leaf->link)
          linked = ir::SsaCopySource(block.nodes, write.value) ==
                   ir::SsaCopySource(block.nodes, *transfer.continuation);
    }

    if (!leaf || !linked) {
      result.refused.push_back({*handle, *callee, SsaLeafCallDecline::not_leaf});
      continue;
    }

    if (std::any_of(leaf->written.begin(), leaf->written.end(), [&](ir::StorageId storage) {
          return liveness->LiveIn(back->target_block->slot, storage);
        })) {
      result.refused.push_back({*handle, *callee, SsaLeafCallDecline::live_result});
      continue;
    }

    planned.push_back({*handle, *callee});
  }

  if (planned.empty()) return result;

  auto candidate = original.Clone(budget);
  if (!candidate) return Decline(SsaLeafCallRefusal::resource_limit);
  for (const auto& plan : planned) {
    const auto here = candidate->Handle(plan.block.slot);
    if (!here) return Decline(SsaLeafCallRefusal::invalid_graph);
    auto copied = candidate->CopyBlock(*here, budget);
    if (!copied || budget.try_consume({1, sizeof(ir::ConditionalRewrite) +
                                              sizeof(SsaLeafCallEdit)}) != BudgetDecline::none)
      return Decline(SsaLeafCallRefusal::resource_limit);
    auto block = std::move(*copied);
    const auto& transfer = *block.boundaries.back().transfer;
    const auto from = block.path_revision;
    block.control_rewrites.push_back({ir::RewriteRule::retired_call,
                                      static_cast<std::uint32_t>(block.boundaries.size() - 1),
                                      transfer,
                                      {ir::TransferKind::jump, *transfer.continuation, {}, {}, {}},
                                      transfer.target,
                                      false,
                                      plan.callee,
                                      0,
                                      {},
                                      from,
                                      from + 1});
    ++block.path_revision;
    auto back = *std::find_if(block.edges.begin(), block.edges.end(), [](const ir::SsaEdge& edge) {
      return edge.kind == ir::SsaEdgeKind::potential_return;
    });
    back.kind = ir::SsaEdgeKind::branch;
    block.edges = {back};
    if (!candidate->Replace(*here, std::move(block)))
      return Decline(SsaLeafCallRefusal::invalid_graph);
    result.journal.push_back({plan.block, plan.callee, original.revision(), 0});
  }

  const auto checked = ir::ValidateSsaWithSources(*candidate, sources, budget);
  if (checked != ir::SsaDecline::none) return Decline(From(checked));
  std::vector<std::uint32_t> rewritten;
  for (const auto& plan : planned) rewritten.push_back(plan.block.slot);
  const auto evidence = ir::ValidateSsaRetiredCalls(*candidate, rewritten, budget);
  if (evidence != ir::SsaDecline::none) return Decline(From(evidence));
  const auto unobserved = ir::ValidateSsaRetiredWork(*candidate, budget);
  if (unobserved != ir::SsaDecline::none) return Decline(From(unobserved));
  for (auto& edit : result.journal) edit.to_revision = candidate->revision();
  result.provisional = std::move(*candidate);
  return result;
}

}  // namespace nyx::recovery
