#include "nyx/recovery/dead_state.hpp"

#include <algorithm>
#include <bit>
#include <limits>

namespace nyx::recovery {
namespace {
bool IsDisabled(const ir::SsaBlock& block, ir::ValueId id) {
  return std::binary_search(block.disabled_effects.begin(), block.disabled_effects.end(), id);
}

bool ValidAccess(const ir::SsaGraph& graph, const ir::PrivateFrameAccess& access) {
  const auto* block = graph.Get(access.block);
  if (!block || access.node >= block->nodes.size()) return false;
  const auto& node = block->nodes[access.node];
  return node.op == (access.store ? ir::Op::store : ir::Op::load) && node.width == access.size * 8;
}

bool LoadIsLive(const ir::SsaGraph& graph, const ir::PrivateFrameSlot& slot, Budget& budget,
                bool& exhausted) {
  for (std::size_t index = 0; index < graph.slots(); ++index) {
    const auto handle = graph.Handle(index);
    if (!handle) continue;
    const auto& block = *graph.Get(*handle);
    const auto node_work = 4 + std::bit_width(block.disabled_effects.size());
    if (block.nodes.size() > std::numeric_limits<std::size_t>::max() / node_work ||
        budget.try_consume({block.nodes.size() * node_work, block.nodes.size() * 2}) !=
            BudgetDecline::none) {
      exhausted = true;
      return true;
    }

    if (budget.try_consume({slot.accesses.size(), 0}) != BudgetDecline::none) {
      exhausted = true;
      return true;
    }

    std::vector<std::uint8_t> candidate(block.nodes.size()), live(block.nodes.size());
    for (const auto& access : slot.accesses)
      if (access.block == *handle) candidate[access.node] = 1;
    const auto mark = [&](ir::ValueId id) {
      if (id < live.size()) live[id] = 1;
    };

    for (std::size_t id = 0; id < block.nodes.size(); ++id) {
      if (candidate[id] || IsDisabled(block, static_cast<ir::ValueId>(id))) continue;
      const auto* descriptor = ir::Descriptor(block.nodes[id].op);
      if (descriptor && descriptor->effect != ir::Effect::pure &&
          descriptor->effect != ir::Effect::storage_read)
        mark(static_cast<ir::ValueId>(id));
    }

    if (budget.try_consume({block.frame_accesses.size(), 0}) != BudgetDecline::none ||
        budget.try_consume({block.control_rewrites.size(), 0}) != BudgetDecline::none ||
        budget.try_consume({block.boundaries.size(), 0}) != BudgetDecline::none ||
        budget.try_consume({block.edges.size(), 0}) != BudgetDecline::none) {
      exhausted = true;
      return true;
    }

    for (const auto& access : block.frame_accesses) {
      if (block.nodes[access.node].op == ir::Op::store) mark(block.nodes[access.node].inputs[1]);
    }

    std::size_t rewrite_index = 0;
    for (std::size_t index = 0; index < block.boundaries.size(); ++index) {
      const auto& boundary = block.boundaries[index];
      if (budget.try_consume({boundary.writes.size(), 0}) != BudgetDecline::none) {
        exhausted = true;
        return true;
      }

      for (const auto& write : boundary.writes) mark(write.value);
      const ir::Transfer* transfer = boundary.transfer ? &*boundary.transfer : nullptr;
      if (rewrite_index < block.control_rewrites.size() &&
          block.control_rewrites[rewrite_index].boundary == index)
        transfer = &block.control_rewrites[rewrite_index++].replacement;
      if (transfer) {
        mark(transfer->target);
        if (transfer->condition) mark(*transfer->condition);
        if (transfer->alternative) mark(*transfer->alternative);
        if (transfer->continuation) mark(*transfer->continuation);
      }
    }

    for (const auto& edge : block.edges)
      if (edge.condition) mark(*edge.condition);
    for (std::size_t n = block.nodes.size(); n-- > 0;) {
      if (!live[n]) continue;
      const auto* descriptor = ir::Descriptor(block.nodes[n].op);
      if (!descriptor) continue;
      for (unsigned input = 0; input < descriptor->arity; ++input)
        mark(block.nodes[n].inputs[input]);
    }

    if (budget.try_consume({slot.accesses.size(), 0}) != BudgetDecline::none) {
      exhausted = true;
      return true;
    }

    for (const auto& access : slot.accesses) {
      if (access.block == *handle && !access.store && live[access.node]) return true;
    }
  }

  return false;
}
}  // namespace

DeadStateResult ProposeDeadPrivateState(const ir::SsaGraph& original,
                                        const ir::PrivateFrameFacts& facts, Budget& budget) {
  DeadStateResult result;
  const auto decline = [](DeadStateRefusal reason) {
    DeadStateResult refused;
    refused.reason = reason;
    return refused;
  };

  const auto proof = ir::ValidatePrivateFrameFacts(facts, original, budget);
  if (proof != ir::PrivateFrameFactDecline::none)
    return decline(proof == ir::PrivateFrameFactDecline::resource_limit
                       ? DeadStateRefusal::resource_limit
                       : DeadStateRefusal::stale_proof);
  const auto validation = ir::ValidateSsa(original, budget);
  if (validation != ir::SsaDecline::none) {
    return decline(validation == ir::SsaDecline::resource_limit ? DeadStateRefusal::resource_limit
                                                                : DeadStateRefusal::invalid_graph);
  }

  struct Removal {
    ir::PrivateFrameAccess access;
    std::int64_t offset;
    std::uint32_t size;
  };

  std::vector<Removal> removals;
  for (const auto& slot : facts.slots) {
    if (budget.try_consume({1, sizeof(RefusedDeadSlot)}) != BudgetDecline::none)
      return decline(DeadStateRefusal::resource_limit);
    bool valid = true;
    for (const auto& access : slot.accesses) {
      if (access.offset != slot.offset || access.size != slot.size ||
          !ValidAccess(original, access))
        valid = false;
    }

    if (!valid) {
      result.refused.push_back({slot.offset, slot.size, DeadStateRefusal::invalid_graph});
      continue;
    }

    bool exhausted = false;
    if (LoadIsLive(original, slot, budget, exhausted)) {
      if (exhausted) {
        return decline(DeadStateRefusal::resource_limit);
      }

      result.refused.push_back({slot.offset, slot.size, DeadStateRefusal::slot_used});
      continue;
    }

    if (budget.try_consume({slot.accesses.size(), slot.accesses.size() * sizeof(Removal)}) !=
        BudgetDecline::none) {
      return decline(DeadStateRefusal::resource_limit);
    }

    for (const auto& access : slot.accesses) {
      const auto* block = original.Get(access.block);
      if (!IsDisabled(*block, access.node)) removals.push_back({access, slot.offset, slot.size});
    }
  }

  if (removals.empty()) return result;
  if (budget.try_consume({removals.size(), removals.size() * sizeof(DeadStateEdit)}) !=
      BudgetDecline::none) {
    return decline(DeadStateRefusal::resource_limit);
  }

  auto candidate = original.Clone(budget);
  if (!candidate) {
    return decline(DeadStateRefusal::resource_limit);
  }

  std::sort(removals.begin(), removals.end(), [](const Removal& a, const Removal& b) {
    if (a.access.block.slot != b.access.block.slot)
      return a.access.block.slot < b.access.block.slot;
    return a.access.node < b.access.node;
  });
  for (const auto& removal : removals) {
    const ir::SsaHandle target{candidate->arena(), removal.access.block.slot,
                               removal.access.block.generation};
    const auto* original_block = candidate->Get(target);
    if (!original_block) {
      return decline(DeadStateRefusal::invalid_graph);
    }

    auto copied = candidate->CopyBlock(target, budget);
    if (!copied) {
      return decline(DeadStateRefusal::resource_limit);
    }

    auto block = std::move(*copied);
    auto at = std::lower_bound(block.disabled_effects.begin(), block.disabled_effects.end(),
                               removal.access.node);
    if (at != block.disabled_effects.end() && *at == removal.access.node) continue;
    block.disabled_effects.insert(at, removal.access.node);
    result.journal.push_back({removal.access.block, target, removal.access.node,
                              block.nodes[removal.access.node].op, removal.offset, removal.size,
                              original.revision(), 0});
    if (!candidate->Replace(target, std::move(block))) {
      return decline(DeadStateRefusal::invalid_graph);
    }
  }

  const auto checked = ir::ValidateSsa(*candidate, budget);
  if (checked != ir::SsaDecline::none) {
    return decline(checked == ir::SsaDecline::resource_limit ? DeadStateRefusal::resource_limit
                                                             : DeadStateRefusal::invalid_graph);
  }

  for (auto& edit : result.journal) edit.to_revision = candidate->revision();
  result.provisional = std::move(*candidate);
  result.basis = ir::BasisOf(facts);
  return result;
}

}  // namespace nyx::recovery
