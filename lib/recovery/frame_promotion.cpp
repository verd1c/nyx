#include "nyx/recovery/frame_promotion.hpp"

#include <algorithm>
#include <bit>
#include <limits>

namespace nyx::recovery {
namespace {
bool IsDisabled(const ir::SsaBlock& block, ir::ValueId id) {
  return std::binary_search(block.disabled_effects.begin(), block.disabled_effects.end(), id);
}

bool HasOldOmission(const ir::SsaBlock& block, ir::ValueId id) {
  for (const auto& omission : block.store_omissions)
    if (omission.store == id) return true;
  for (const auto& omission : block.paired_load_omissions)
    if (omission.loads[0] == id || omission.loads[1] == id) return true;
  return false;
}
}  // namespace

PromotionResult ProposeFramePromotion(const ir::SsaGraph& original,
                                      const ir::PrivateFrameFacts& facts, Budget& budget) {
  PromotionResult result;
  const auto decline = [](PromotionRefusal reason) {
    PromotionResult refused;
    refused.reason = reason;
    return refused;
  };

  const auto proof = ir::ValidatePrivateFrameFacts(facts, original, budget);
  if (proof != ir::PrivateFrameFactDecline::none)
    return decline(proof == ir::PrivateFrameFactDecline::resource_limit
                       ? PromotionRefusal::resource_limit
                       : PromotionRefusal::stale_proof);
  if (!facts.contract.fresh_mapped_writable || !facts.contract.no_external_aliases ||
      !facts.contract.no_async_observers) {
    return decline(PromotionRefusal::invalid_graph);
  }

  const auto validation = ir::ValidateSsa(original, budget);
  if (validation != ir::SsaDecline::none) {
    return decline(validation == ir::SsaDecline::resource_limit ? PromotionRefusal::resource_limit
                                                                : PromotionRefusal::invalid_graph);
  }

  std::vector<const ir::PrivateFrameSlot*> selected;
  std::size_t access_count = 0;
  for (const auto& slot : facts.slots) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return decline(PromotionRefusal::resource_limit);
    if (slot.accesses.empty()) continue;
    bool already = false, invalid = false, old_omission = false, incompatible = false;
    std::optional<ir::ByteOrder> order;
    for (std::size_t i = 0; i < original.slots(); ++i) {
      const auto handle = original.Handle(i);
      if (!handle) continue;
      const auto& block = *original.Get(*handle);
      if (budget.try_consume({1, 0}) != BudgetDecline::none ||
          budget.try_consume({block.frame_phis.size(), 0}) != BudgetDecline::none)
        return decline(PromotionRefusal::resource_limit);
      for (const auto& phi : block.frame_phis)
        already |= phi.offset == slot.offset && phi.size == slot.size;
    }

    for (const auto& access : slot.accesses) {
      const auto* block = original.Get(access.block);
      if (budget.try_consume({1, 0}) != BudgetDecline::none)
        return decline(PromotionRefusal::resource_limit);
      if (!block || access.node >= block->nodes.size() || access.offset != slot.offset ||
          access.size != slot.size) {
        invalid = true;
        continue;
      }

      const auto& node = block->nodes[access.node];
      if (budget.try_consume(
              {static_cast<std::uint64_t>(std::bit_width(block->disabled_effects.size())), 0}) !=
              BudgetDecline::none ||
          budget.try_consume({block->store_omissions.size(), 0}) != BudgetDecline::none ||
          budget.try_consume({block->paired_load_omissions.size(), 0}) != BudgetDecline::none)
        return decline(PromotionRefusal::resource_limit);
      if (node.op != (access.store ? ir::Op::store : ir::Op::load) || node.width != slot.size * 8 ||
          IsDisabled(*block, access.node))
        invalid = true;
      if (order && *order != node.access.byte_order) incompatible = true;
      order = node.access.byte_order;
      old_omission |= HasOldOmission(*block, access.node);
    }

    if (already || invalid || old_omission || incompatible) {
      if (budget.try_consume({1, sizeof(RefusedPromotionSlot)}) != BudgetDecline::none)
        return decline(PromotionRefusal::resource_limit);
      result.refused.push_back({slot.offset, slot.size,
                                already        ? PromotionRefusal::already_promoted
                                : old_omission ? PromotionRefusal::existing_omission
                                : incompatible ? PromotionRefusal::incompatible_access
                                               : PromotionRefusal::unsupported_access});
      continue;
    }

    if (budget.try_consume({1, sizeof(const ir::PrivateFrameSlot*)}) != BudgetDecline::none)
      return decline(PromotionRefusal::resource_limit);
    selected.push_back(&slot);
    if (slot.accesses.size() > std::numeric_limits<std::size_t>::max() - access_count)
      return decline(PromotionRefusal::resource_limit);
    access_count += slot.accesses.size();
  }

  if (selected.empty()) return result;
  for (std::size_t slot = 0; slot < original.slots(); ++slot)
    for (std::size_t selected_slot = 0; selected_slot < selected.size(); ++selected_slot)
      if (budget.try_consume({3, sizeof(ir::SsaFramePhi) + sizeof(ir::SsaValue)}) !=
          BudgetDecline::none)
        return decline(PromotionRefusal::resource_limit);
  for (std::size_t access = 0; access < access_count; ++access)
    if (budget.try_consume({4, sizeof(ir::SsaFrameAccess) + sizeof(PromotionEdit) +
                                   sizeof(ir::ValueId)}) != BudgetDecline::none)
      return decline(PromotionRefusal::resource_limit);
  auto candidate = original.Clone(budget);
  if (!candidate) return decline(PromotionRefusal::resource_limit);
  for (std::size_t i = 0; i < candidate->slots(); ++i) {
    const auto handle = candidate->Handle(i);
    if (!handle) continue;
    auto copied = candidate->CopyBlock(*handle, budget);
    if (!copied) return decline(PromotionRefusal::resource_limit);
    auto block = std::move(*copied);
    if (budget.try_consume({candidate->entries().size(), 0}) != BudgetDecline::none ||
        budget.try_consume({access_count, 0}) != BudgetDecline::none)
      return decline(PromotionRefusal::resource_limit);
    const bool entry = std::find(candidate->entries().begin(), candidate->entries().end(),
                                 *handle) != candidate->entries().end();
    const auto old_size = block.frame_phis.size();
    for (const auto* slot : selected) {
      block.frame_phis.push_back({slot->offset, slot->size, entry, {}});
      const auto index = static_cast<std::uint32_t>(block.frame_phis.size() - 1);
      block.frame_exits.push_back({ir::SsaValueKind::frame_phi, *handle, index});
    }

    if (budget.try_consume(
            {block.nodes.size(), block.nodes.size() * sizeof(std::optional<std::uint32_t>)}) !=
        BudgetDecline::none) {
      return decline(PromotionRefusal::resource_limit);
    }

    std::vector<std::optional<std::uint32_t>> access_to_phi(block.nodes.size());
    for (std::size_t selected_index = 0; selected_index < selected.size(); ++selected_index) {
      for (const auto& access : selected[selected_index]->accesses) {
        if (access.block.slot != handle->slot) continue;
        if (access_to_phi[access.node]) return decline(PromotionRefusal::invalid_graph);
        access_to_phi[access.node] = static_cast<std::uint32_t>(old_size + selected_index);
      }
    }

    if (budget.try_consume({block.frame_phis.size(),
                            block.frame_phis.size() * sizeof(ir::SsaValue)}) != BudgetDecline::none)
      return decline(PromotionRefusal::resource_limit);
    std::vector<ir::SsaValue> current;
    current.reserve(block.frame_phis.size());
    for (std::size_t phi = 0; phi < block.frame_phis.size(); ++phi)
      current.push_back({ir::SsaValueKind::frame_phi, *handle, static_cast<std::uint32_t>(phi)});
    for (std::uint32_t id = 0; id < block.nodes.size(); ++id) {
      if (!access_to_phi[id]) continue;
      const auto phi = *access_to_phi[id];
      const auto& node = block.nodes[id];
      std::optional<ir::SsaValue> replacement;
      if (node.op == ir::Op::load)
        replacement = current[phi];
      else
        current[phi] = {ir::SsaValueKind::node, *handle, node.inputs[1]};
      block.frame_accesses.push_back({id, phi, replacement});
      if (budget.try_consume(
              {static_cast<std::uint64_t>(std::bit_width(block.disabled_effects.size())), 0}) !=
          BudgetDecline::none)
        return decline(PromotionRefusal::resource_limit);
      const auto at =
          std::lower_bound(block.disabled_effects.begin(), block.disabled_effects.end(), id);
      if (budget.try_consume({static_cast<std::uint64_t>(block.disabled_effects.end() - at), 0}) !=
          BudgetDecline::none)
        return decline(PromotionRefusal::resource_limit);
      block.disabled_effects.insert(at, id);
      const auto* slot = selected[phi - old_size];
      const ir::SsaHandle original_handle{original.arena(), handle->slot, handle->generation};
      result.journal.push_back({original_handle, *handle, id, slot->offset, slot->size, replacement,
                                original.revision(), 0});
    }

    for (std::size_t phi = old_size; phi < current.size(); ++phi)
      block.frame_exits[phi] = current[phi];
    const auto sort_depth = std::bit_width(block.frame_accesses.size());
    if (sort_depth &&
        block.frame_accesses.size() > std::numeric_limits<std::size_t>::max() / sort_depth)
      return decline(PromotionRefusal::resource_limit);
    if (budget.try_consume({block.frame_accesses.size() * sort_depth, 0}) != BudgetDecline::none)
      return decline(PromotionRefusal::resource_limit);
    std::sort(block.frame_accesses.begin(), block.frame_accesses.end(),
              [](const auto& a, const auto& b) { return a.node < b.node; });
    if (!candidate->Replace(*handle, std::move(block)))
      return decline(PromotionRefusal::invalid_graph);
  }

  for (std::size_t i = 0; i < candidate->slots(); ++i) {
    const auto source = candidate->Handle(i);
    if (!source) continue;
    auto copied = candidate->CopyBlock(*source, budget);
    if (!copied) return decline(PromotionRefusal::resource_limit);
    const auto& block = *copied;
    for (const auto& edge : block.edges) {
      if (!edge.target_block) continue;
      auto copied_target = candidate->CopyBlock(*edge.target_block, budget);
      if (!copied_target) return decline(PromotionRefusal::resource_limit);
      auto target = std::move(*copied_target);
      const auto old_size = target.frame_phis.size() - selected.size();
      for (std::size_t phi = old_size; phi < target.frame_phis.size(); ++phi) {
        auto& incoming = target.frame_phis[phi].incoming;
        if (budget.try_consume({incoming.size(), 0}) != BudgetDecline::none)
          return decline(PromotionRefusal::resource_limit);
        if (std::none_of(incoming.begin(), incoming.end(),
                         [&](const auto& input) { return input.predecessor == *source; }))
          incoming.push_back({*source, block.frame_exits[phi]});
      }

      if (!candidate->Replace(*edge.target_block, std::move(target)))
        return decline(PromotionRefusal::invalid_graph);
    }
  }

  const auto checked = ir::ValidateSsa(*candidate, budget);
  if (checked != ir::SsaDecline::none) {
    return decline(checked == ir::SsaDecline::resource_limit ? PromotionRefusal::resource_limit
                                                             : PromotionRefusal::invalid_graph);
  }

  for (auto& edit : result.journal) edit.to_revision = candidate->revision();
  result.provisional = std::move(*candidate);
  result.basis = ir::BasisOf(facts);
  return result;
}

}  // namespace nyx::recovery
