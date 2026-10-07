#include "nyx/recovery/ssa/splice.hpp"

#include <algorithm>
#include <limits>

namespace nyx::recovery {

bool SpliceBefore(ir::SsaGraph& graph, ir::SsaHandle target, ir::ValueId before,
                  std::span<const ir::Node> inserted, std::uint32_t max_nodes, Budget& budget) {
  const auto* source = graph.Get(target);
  if (!source || before >= source->nodes.size() || inserted.empty() ||
      inserted.size() > max_nodes || source->nodes.size() > max_nodes - inserted.size() ||
      source->nodes.size() > UINT32_MAX - inserted.size() ||
      source->destination_nodes.size() > UINT32_MAX - source->nodes.size() - inserted.size())
    return false;
  const auto count = static_cast<ir::ValueId>(inserted.size());
  for (std::size_t i = 0; i < inserted.size(); ++i) {
    const auto* descriptor = ir::Descriptor(inserted[i].op);
    if (!descriptor || descriptor->effect != ir::Effect::pure || before > UINT32_MAX - i ||
        budget.try_consume({1, 0}) != BudgetDecline::none)
      return false;
    for (unsigned input = 0; input < descriptor->arity; ++input)
      if (inserted[i].inputs[input] >= before + i) return false;
  }

  const auto shift = [&](ir::ValueId& id) {
    if (id >= before) id += count;
  };

  const auto shift_transfer = [&](ir::Transfer& transfer) {
    shift(transfer.target);
    if (transfer.condition) shift(*transfer.condition);
    if (transfer.alternative) shift(*transfer.alternative);
    if (transfer.continuation) shift(*transfer.continuation);
  };

  const auto charge = [&](std::size_t count) {
    return budget.try_consume({count, 0}) == BudgetDecline::none;
  };

  const auto moves = [&](const ir::SsaValue& value) {
    return value.kind == ir::SsaValueKind::node && value.block == target && value.index >= before;
  };

  // Whether another block names a value that moves; only those are copied.
  const auto names = [&](const ir::SsaBlock& block) {
    for (const auto& phi : block.phis)
      for (const auto& input : phi.incoming)
        if (moves(input.value)) return true;
    for (const auto& phi : block.frame_phis)
      for (const auto& input : phi.incoming)
        if (moves(input.value)) return true;
    for (const auto& read : block.reads)
      if (moves(read.value) || (read.predecessor_copy && moves(*read.predecessor_copy)))
        return true;
    for (const auto& exit : block.exits)
      if (moves(exit.value)) return true;
    for (const auto& access : block.frame_accesses)
      if (access.replacement && moves(*access.replacement)) return true;
    return std::any_of(block.frame_exits.begin(), block.frame_exits.end(), moves);
  };

  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return false;
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto& current = *graph.Get(*handle);
    if (!charge(current.phis.size()) || !charge(current.frame_phis.size()) ||
        !charge(current.reads.size() * 2) || !charge(current.exits.size()) ||
        !charge(current.frame_accesses.size() * 2) || !charge(current.frame_exits.size()))
      return false;
    for (const auto& phi : current.phis)
      if (!charge(phi.incoming.size())) return false;
    for (const auto& phi : current.frame_phis)
      if (!charge(phi.incoming.size())) return false;
    if (*handle != target && !names(current)) continue;
    auto copied = graph.CopyBlock(*handle, budget);
    if (!copied) return false;
    auto block = std::move(*copied);
    bool changed = *handle == target;
    const auto shift_value = [&](ir::SsaValue& value) {
      if (value.kind == ir::SsaValueKind::node && value.block == target && value.index >= before) {
        value.index += count;
        changed = true;
      }
    };

    for (auto& phi : block.phis)
      for (auto& input : phi.incoming) shift_value(input.value);
    for (auto& phi : block.frame_phis)
      for (auto& input : phi.incoming) shift_value(input.value);
    for (auto& read : block.reads) {
      shift_value(read.value);
      if (read.predecessor_copy) shift_value(*read.predecessor_copy);
    }

    for (auto& exit : block.exits) shift_value(exit.value);
    for (auto& access : block.frame_accesses)
      if (access.replacement) shift_value(*access.replacement);
    for (auto& exit : block.frame_exits) shift_value(exit);
    if (*handle != target) {
      if (changed && !graph.Replace(*handle, std::move(block))) return false;
      continue;
    }

    if (!charge(block.boundaries.size()) || !charge(block.control_rewrites.size() * 4) ||
        !charge(block.store_omissions.size() * 2) ||
        !charge(block.paired_load_omissions.size() * 2) || !charge(block.disabled_effects.size()) ||
        !charge(block.dead_pure_nodes.size()) || !charge(block.constant_loads.size() * 3) ||
        !charge(block.path_reads.size()) || !charge(block.retired_loads.size() * 2) ||
        !charge(block.edges.size()))
      return false;
    for (const auto& boundary : block.boundaries)
      if (!charge(boundary.writes.size()) || (boundary.transfer && !charge(4))) return false;
    for (const auto& rewrite : block.control_rewrites)
      if (!charge(rewrite.witness.size())) return false;
    for (const auto& omission : block.paired_load_omissions)
      if (!charge(omission.stores.size() + omission.loads.size())) return false;
    if (budget.try_consume({block.nodes.size() * 4 + inserted.size(),
                            (block.nodes.size() + inserted.size()) * sizeof(ir::Node)}) !=
        BudgetDecline::none)
      return false;
    block.nodes.insert(block.nodes.begin() + before, inserted.begin(), inserted.end());
    for (std::size_t id = 0; id < block.nodes.size(); ++id) {
      if (id >= before && id < before + count) continue;
      auto& node = block.nodes[id];
      const auto* descriptor = ir::Descriptor(node.op);
      if (!descriptor) return false;
      for (unsigned input = 0; input < descriptor->arity; ++input) shift(node.inputs[input]);
    }

    for (auto& boundary : block.boundaries) {
      const auto end = boundary.first_node + boundary.node_count;
      if (boundary.first_node > before)
        shift(boundary.first_node);
      else if (boundary.first_node <= before && before < end)
        boundary.node_count += count;
      for (auto& write : boundary.writes) shift(write.value);
      if (boundary.transfer) shift_transfer(*boundary.transfer);
    }

    for (auto& rewrite : block.control_rewrites) {
      shift_transfer(rewrite.original);
      shift_transfer(rewrite.replacement);
      shift(rewrite.condition);
      for (auto& read : rewrite.witness) shift(read.node);
    }

    for (auto& omission : block.store_omissions) {
      shift(omission.store);
      shift(omission.overwriter);
    }

    for (auto& omission : block.paired_load_omissions) {
      for (auto& id : omission.stores) shift(id);
      for (auto& id : omission.loads) shift(id);
    }

    for (auto& id : block.disabled_effects) shift(id);
    for (auto& id : block.dead_pure_nodes) shift(id);
    for (auto& read : block.reads) shift(read.node);
    for (auto& access : block.frame_accesses) shift(access.node);
    for (auto& read : block.path_reads) shift(read.node);
    for (auto& retired : block.retired_loads) {
      shift(retired.node);
      if (retired.basis == ir::SsaRetiredLoadBasis::written_before) shift(retired.store);
    }

    for (auto& fold : block.constant_loads) {
      shift(fold.node);
      if (fold.condition) shift(*fold.condition);
      if (fold.kind == ir::SsaConstantKind::bounded_table) shift(fold.table_index);
    }

    for (auto& edge : block.edges)
      if (edge.condition) shift(*edge.condition);
    if (!graph.Replace(*handle, std::move(block))) return false;
  }

  return true;
}

}  // namespace nyx::recovery
