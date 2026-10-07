#include <algorithm>

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::ir {
namespace {
bool SameNode(const Node& a, const Node& b) {
  return a.op == b.op && a.width == b.width && a.inputs == b.inputs && a.immediate == b.immediate &&
         a.storage == b.storage && a.access.byte_order == b.access.byte_order &&
         a.access.alignment == b.access.alignment &&
         a.access.decline_on_unaligned == b.access.decline_on_unaligned;
}

bool SameTransfer(const std::optional<Transfer>& a, const std::optional<Transfer>& b) {
  return a.has_value() == b.has_value() &&
         (!a || (a->kind == b->kind && a->target == b->target && a->condition == b->condition &&
                 a->alternative == b->alternative && a->continuation == b->continuation));
}
}  // namespace

SsaDecline ValidateBlockBinding(const SsaGraph& graph, SsaHandle handle,
                                std::span<const Group> sources, Budget& budget,
                                bool allow_bounded_folds) {
  const auto valid = allow_bounded_folds ? ValidateSsaWithSources(graph, sources, budget)
                                         : ValidateSsa(graph, budget);
  if (valid != SsaDecline::none) return valid;
  const auto* block = graph.Get(handle);

  // Promoted frame slots passing through a block leave its nodes as decoded;
  // only a frame access, a load or store the promotion replaced, changes them.
  if (!block || block->opaque || block->transition || !block->control_rewrites.empty() ||
      !block->store_omissions.empty() || !block->paired_load_omissions.empty() ||
      !block->destination_nodes.empty() || !block->dead_pure_nodes.empty() ||
      !block->frame_accesses.empty() || block->boundaries.empty() ||
      block->original_sources.size() != block->boundaries.size() ||
      block->source_groups.size() != block->boundaries.size() ||
      block->source_bytes.size() != block->boundaries.size())
    return SsaDecline::invalid_graph;
  if (!allow_bounded_folds && (!block->constant_loads.empty() || !block->disabled_effects.empty()))
    return SsaDecline::invalid_graph;
  if (allow_bounded_folds) {
    if (block->constant_loads.size() != block->disabled_effects.size())
      return SsaDecline::invalid_graph;
    if (budget.try_consume({block->constant_loads.size(), 0}) != BudgetDecline::none)
      return SsaDecline::resource_limit;
    for (std::size_t i = 0; i < block->constant_loads.size(); ++i)
      if (block->constant_loads[i].kind != SsaConstantKind::bounded_table ||
          !block->constant_loads[i].skip_access ||
          block->constant_loads[i].node != block->disabled_effects[i])
        return SsaDecline::invalid_graph;
  }

  const auto first = block->original_sources.front();
  if (first > sources.size() || block->original_sources.size() > sources.size() - first)
    return SsaDecline::invalid_graph;
  for (std::size_t i = 0; i < block->original_sources.size(); ++i) {
    const auto& source = sources[first + i];
    const auto& bytes = block->source_bytes[i];
    if (budget.try_consume({1 + bytes.size(), 0}) != BudgetDecline::none)
      return SsaDecline::resource_limit;
    if (block->original_sources[i] != first + i ||
        block->source_groups[i] != source.source_address() ||
        bytes.size() != source.bytes().size() ||
        !std::equal(bytes.begin(), bytes.end(), source.bytes().begin()))
      return SsaDecline::invalid_graph;
  }

  const auto normalized = Normalize(sources.subspan(first, block->original_sources.size()), budget);
  if (!normalized.block)
    return normalized.reason == BlockDecline::resource_limit ? SsaDecline::resource_limit
                                                             : SsaDecline::invalid_graph;
  const auto& expected = *normalized.block;
  if (block->nodes.size() != expected.nodes().size() ||
      block->boundaries.size() != expected.boundaries().size())
    return SsaDecline::invalid_graph;
  if (budget.try_consume({block->nodes.size() + block->boundaries.size(), 0}) !=
      BudgetDecline::none)
    return SsaDecline::resource_limit;
  for (std::size_t i = 0; i < block->nodes.size(); ++i)
    if (!SameNode(block->nodes[i], expected.nodes()[i])) return SsaDecline::invalid_graph;
  for (std::size_t i = 0; i < block->boundaries.size(); ++i) {
    const auto& actual = block->boundaries[i];
    const auto& original = expected.boundaries()[i];
    if (budget.try_consume({actual.writes.size(), 0}) != BudgetDecline::none)
      return SsaDecline::resource_limit;
    if (actual.first_node != original.first_node || actual.node_count != original.node_count ||
        actual.writes.size() != original.writes.size() ||
        !SameTransfer(actual.transfer, original.transfer))
      return SsaDecline::invalid_graph;
    for (std::size_t w = 0; w < actual.writes.size(); ++w)
      if (actual.writes[w].storage != original.writes[w].storage ||
          actual.writes[w].value != original.writes[w].value)
        return SsaDecline::invalid_graph;
  }

  return SsaDecline::none;
}

SsaDecline ValidateSsaDirectBlockBinding(const SsaGraph& graph, SsaHandle handle,
                                         std::span<const Group> sources, Budget& budget) {
  return ValidateBlockBinding(graph, handle, sources, budget, false);
}

SsaDecline ValidateSsaBoundedTableBlockBinding(const SsaGraph& graph, SsaHandle handle,
                                               std::span<const Group> sources, Budget& budget) {
  return ValidateBlockBinding(graph, handle, sources, budget, true);
}

SsaDecline ValidateSsaWithSources(const SsaGraph& graph, std::span<const Group>, Budget& budget) {
  return ValidateSsa(graph, budget);
}

SsaDecline ValidateSsaBranchPredicateFact(const SsaGraph& graph, const SsaBranchPredicateFact& fact,
                                          std::span<const Group> sources, Budget& budget) {
  if (fact.graph_arena != graph.arena() || fact.graph_revision != graph.revision())
    return SsaDecline::invalid_graph;
  const auto bound = ValidateSsaDirectBlockBinding(graph, fact.branch, sources, budget);
  if (bound != SsaDecline::none) return bound;
  const auto* block = graph.Get(fact.branch);
  if (fact.edge_index >= block->edges.size()) return SsaDecline::invalid_graph;
  const auto& transfer = block->boundaries.back().transfer;
  const auto& edge = block->edges[fact.edge_index];
  if (!transfer || transfer->kind != TransferKind::conditional || !transfer->condition ||
      *transfer->condition != fact.condition || edge.condition != fact.condition ||
      edge.when != fact.when || !edge.target_block ||
      (edge.kind != SsaEdgeKind::branch && edge.kind != SsaEdgeKind::fallthrough))
    return SsaDecline::invalid_graph;
  return ValidateSsaDirectSuccessors(*block, budget);
}
}  // namespace nyx::ir
