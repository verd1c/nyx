#include "nyx/analysis/ssa/table_address.hpp"

#include <algorithm>

namespace nyx::analysis {

SsaTableAddressResult ProveSsaBoundedTableAddress(const ir::SsaGraph& graph,
                                                  std::span<const ir::Group> sources,
                                                  const ir::SsaIndexBoundFact& bound,
                                                  ir::ValueId load_id, Budget& budget) {
  const auto index = ir::ValidateSsaIndexBoundFact(graph, bound, sources, budget);
  if (index != ir::SsaDecline::none)
    return {{},
            index == ir::SsaDecline::resource_limit ? SsaTableAddressRefusal::resource_limit
                                                    : SsaTableAddressRefusal::index_unbounded};
  const auto handle = bound.guard.guarded_block;
  const auto source = ir::ValidateSsaBoundedTableBlockBinding(graph, handle, sources, budget);
  if (source != ir::SsaDecline::none)
    return {{},
            source == ir::SsaDecline::resource_limit ? SsaTableAddressRefusal::resource_limit
                                                     : SsaTableAddressRefusal::source_unbound};
  const auto* block = graph.Get(handle);
  if (load_id >= block->nodes.size()) return {{}, SsaTableAddressRefusal::unsupported_address};
  const auto& load = block->nodes[load_id];
  if (load.op != ir::Op::load || !load.width || load.width > 64 || load.width % 8 ||
      load.inputs[0] >= block->nodes.size())
    return {{}, SsaTableAddressRefusal::unsupported_address};
  if (budget.try_consume({64, 0}) != BudgetDecline::none)
    return {{}, SsaTableAddressRefusal::resource_limit};
  const auto shape = ir::SsaTableLoadAddress(block->nodes, load_id, graph.load_bias());
  if (!shape) return {{}, SsaTableAddressRefusal::unsupported_address};
  const auto scale = shape->stride;
  if (block->nodes[shape->index].storage != bound.storage)
    return {{}, SsaTableAddressRefusal::index_changed};
  if (budget.try_consume({block->phis.size() + block->reads.size(), 0}) != BudgetDecline::none)
    return {{}, SsaTableAddressRefusal::resource_limit};
  const auto phi =
      std::find_if(block->phis.begin(), block->phis.end(),
                   [&](const ir::SsaPhi& item) { return item.storage == bound.storage; });
  const auto read =
      std::find_if(block->reads.begin(), block->reads.end(),
                   [&](const ir::SsaRead& item) { return item.node == shape->index; });
  if (phi == block->phis.end() || read == block->reads.end() ||
      read->value != ir::SsaValue{ir::SsaValueKind::phi, handle,
                                  static_cast<std::uint32_t>(phi - block->phis.begin())})
    return {{}, SsaTableAddressRefusal::index_changed};
  const auto maximum = bound.exclusive_upper - 1;
  if (maximum > UINT64_MAX / scale || shape->base > UINT64_MAX - maximum * scale ||
      shape->base + maximum * scale > UINT64_MAX - (load.width / 8 - 1))
    return {{}, SsaTableAddressRefusal::address_overflow};
  const ir::SsaBoundedTableAddressFact fact{bound, load_id, shape->base, scale, shape->placed};
  const auto checked = ir::ValidateSsaBoundedTableAddressFact(graph, fact, sources, budget);
  if (checked != ir::SsaDecline::none)
    return {{},
            checked == ir::SsaDecline::resource_limit ? SsaTableAddressRefusal::resource_limit
                                                      : SsaTableAddressRefusal::invalid_graph};
  return {fact, SsaTableAddressRefusal::none};
}

}  // namespace nyx::analysis
