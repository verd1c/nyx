#include <algorithm>
#include <limits>

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::ir {

std::optional<SsaTableAddress> SsaTableLoadAddress(std::span<const Node> nodes, ValueId load,
                                                   std::optional<std::uint64_t> bias) {
  if (load >= nodes.size() || nodes[load].op != Op::load || nodes[load].inputs[0] >= nodes.size())
    return {};
  const auto& address = nodes[nodes[load].inputs[0]];
  if (address.op != Op::add || address.width != 64 || address.inputs[0] >= nodes.size() ||
      address.inputs[1] >= nodes.size())
    return {};
  const auto base = SsaImageLocation(nodes, address.inputs[0], bias);
  if (!base) return {};
  const auto scaled_id = address.inputs[1];
  const auto& scaled = nodes[scaled_id];
  if (scaled.op == Op::read) {
    if (scaled.width != 64) return {};
    return SsaTableAddress{base->first, base->second, scaled_id, 1};
  }

  if ((scaled.op != Op::mul && scaled.op != Op::shl) || scaled.width != 64 ||
      scaled.inputs[0] >= nodes.size() || scaled.inputs[1] >= nodes.size())
    return {};
  const auto& index = nodes[scaled.inputs[0]];
  const auto& stride = nodes[scaled.inputs[1]];
  if (index.op != Op::read || index.width != 64 || stride.op != Op::constant ||
      stride.width != 64 || (scaled.op == Op::mul && !stride.immediate) ||
      (scaled.op == Op::shl && stride.immediate >= 64))
    return {};
  return SsaTableAddress{
      base->first, base->second, scaled.inputs[0],
      scaled.op == Op::shl ? std::uint64_t{1} << stride.immediate : stride.immediate};
}

SsaDecline ValidateSsaBoundedTableAddressFact(const SsaGraph& graph,
                                              const SsaBoundedTableAddressFact& fact,
                                              std::span<const Group> sources, Budget& budget) {
  const auto bound = ValidateSsaIndexBoundFact(graph, fact.index_bound, sources, budget);
  if (bound != SsaDecline::none) return bound;
  const auto handle = fact.index_bound.guard.guarded_block;
  const auto source = ValidateSsaBoundedTableBlockBinding(graph, handle, sources, budget);
  if (source != SsaDecline::none) return source;
  const auto* block = graph.Get(handle);
  if (fact.load >= block->nodes.size()) return SsaDecline::invalid_graph;
  const auto& load = block->nodes[fact.load];
  if (load.op != Op::load || !load.width || load.width > 64 || load.width % 8 ||
      load.inputs[0] >= block->nodes.size())
    return SsaDecline::invalid_graph;
  if (budget.try_consume({64, 0}) != BudgetDecline::none) return SsaDecline::resource_limit;
  const auto shape = SsaTableLoadAddress(block->nodes, fact.load, graph.load_bias());
  if (!shape || block->nodes[shape->index].storage != fact.index_bound.storage ||
      shape->base != fact.base || shape->stride != fact.stride || shape->placed != fact.placed)
    return SsaDecline::invalid_graph;
  if (budget.try_consume({block->phis.size() + block->reads.size(), 0}) != BudgetDecline::none)
    return SsaDecline::resource_limit;
  const auto phi = std::find_if(block->phis.begin(), block->phis.end(), [&](const SsaPhi& item) {
    return item.storage == fact.index_bound.storage;
  });
  const auto read = std::find_if(block->reads.begin(), block->reads.end(),
                                 [&](const SsaRead& item) { return item.node == shape->index; });
  if (phi == block->phis.end() || read == block->reads.end() ||
      read->value != SsaValue{SsaValueKind::phi, handle,
                              static_cast<std::uint32_t>(phi - block->phis.begin())})
    return SsaDecline::invalid_graph;
  const auto maximum = fact.index_bound.exclusive_upper - 1;
  if (maximum > UINT64_MAX / fact.stride) return SsaDecline::invalid_graph;
  const auto offset = maximum * fact.stride;
  if (fact.base > UINT64_MAX - offset) return SsaDecline::invalid_graph;
  const auto last = fact.base + offset;
  if (last > UINT64_MAX - (load.width / 8 - 1)) return SsaDecline::invalid_graph;
  return SsaDecline::none;
}

}  // namespace nyx::ir
