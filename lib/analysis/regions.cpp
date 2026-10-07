#include "nyx/analysis/regions.hpp"

#include <algorithm>
#include <bit>

namespace nyx::analysis {
namespace {

bool Charge(Budget& budget, std::uint64_t count, std::uint64_t size = 0) {
  return (size == 0 || count <= UINT64_MAX / size) &&
         budget.try_consume({count, count * size}) == BudgetDecline::none;
}

bool SameNode(const ir::Node& a, const ir::Node& b) {
  return a.op == b.op && a.width == b.width && a.inputs == b.inputs && a.immediate == b.immediate &&
         a.storage == b.storage && a.access.byte_order == b.access.byte_order &&
         a.access.alignment == b.access.alignment &&
         a.access.decline_on_unaligned == b.access.decline_on_unaligned;
}

bool SameTransfer(const std::optional<ir::Transfer>& a, const std::optional<ir::Transfer>& b) {
  return a.has_value() == b.has_value() &&
         (!a || (a->kind == b->kind && a->target == b->target && a->condition == b->condition &&
                 a->alternative == b->alternative && a->continuation == b->continuation));
}

bool SameGroup(const ir::Group& a, const ir::Group& b) {
  return a.source_address() == b.source_address() && a.memory_model() == b.memory_model() &&
         SameTransfer(a.transfer(), b.transfer()) &&
         std::equal(a.bytes().begin(), a.bytes().end(), b.bytes().begin(), b.bytes().end()) &&
         std::equal(a.nodes().begin(), a.nodes().end(), b.nodes().begin(), b.nodes().end(),
                    SameNode) &&
         std::equal(a.writes().begin(), a.writes().end(), b.writes().begin(), b.writes().end(),
                    [](const auto& x, const auto& y) {
                      return x.storage == y.storage && x.value == y.value;
                    });
}

// CFG annotations are selection hints. This checks their structural provenance
// only; it does not re-check the SSA against the original instructions.
RegionDecline CheckGraph(const Cfg& graph, Budget& budget, RegionLimits limits) {
  const auto sources = graph.sources();
  const auto blocks = graph.blocks();
  if (sources.empty() || blocks.empty() || graph.entries().empty())
    return RegionDecline::invalid_graph;
  if (sources.size() > limits.max_graph_sources || blocks.size() > limits.max_graph_blocks ||
      graph.entries().size() > limits.max_graph_sources)
    return RegionDecline::resource_limit;
  if (!Charge(budget, sources.size(), sizeof(std::uint32_t))) return RegionDecline::resource_limit;
  std::vector<std::uint32_t> owners(sources.size());
  std::size_t next = 0;
  for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
    const auto& block = blocks[bi];
    if (!Charge(budget, 1)) return RegionDecline::resource_limit;
    const std::size_t max_edges = block.dispatch ? limits.max_graph_block_edges : 3;
    if (block.first_source != next || block.source_count == 0 ||
        block.source_count > sources.size() - next || block.edges.empty() ||
        block.edges.size() > max_edges)
      return RegionDecline::invalid_graph;
    if (block.ssa && block.ssa->sources().size() != block.source_count)
      return RegionDecline::invalid_graph;
    if (!block.ssa && (block.source_count != 1 || block.control))
      return RegionDecline::invalid_graph;
    for (std::uint32_t i = 0; i < block.source_count; ++i) {
      const auto& source = sources[next];
      if (!Charge(budget, source.bytes.size() + 1)) return RegionDecline::resource_limit;
      if (source.bytes.empty() || source.bytes.size() - 1 > UINT64_MAX - source.address ||
          (next != 0 &&
           (source.address <= sources[next - 1].address ||
            source.address - sources[next - 1].address < sources[next - 1].bytes.size())))
        return RegionDecline::invalid_graph;
      if (source.semantics.has_value() != block.ssa.has_value())
        return RegionDecline::invalid_graph;
      if (source.semantics) {
        const auto& group = *source.semantics;
        if (!Charge(budget, group.nodes().size()) || !Charge(budget, group.writes().size()))
          return RegionDecline::resource_limit;
        if (source.opaque_reason != OpaqueReason::none ||
            source.opaque_control != OpaqueControl::unknown ||
            group.source_address() != source.address ||
            !std::equal(source.bytes.begin(), source.bytes.end(), group.bytes().begin(),
                        group.bytes().end()) ||
            !SameGroup(group, block.ssa->sources()[i]))
          return RegionDecline::invalid_graph;
        if (i != 0 && sources[next - 1].address + sources[next - 1].bytes.size() != source.address)
          return RegionDecline::invalid_graph;
        if (i + 1 != block.source_count && group.transfer()) return RegionDecline::invalid_graph;
      } else {
        if (source.opaque_reason != OpaqueReason::unsupported &&
            source.opaque_reason != OpaqueReason::invalid_encoding &&
            source.opaque_reason != OpaqueReason::not_decoded)
          return RegionDecline::invalid_graph;
        if (source.opaque_control != OpaqueControl::unknown &&
            (source.opaque_reason != OpaqueReason::unsupported ||
             source.bytes.size() > UINT64_MAX - source.address))
          return RegionDecline::invalid_graph;
      }

      owners[next++] = static_cast<std::uint32_t>(bi);
    }

    const auto& terminal = sources[next - 1];
    const auto transfer =
        terminal.semantics ? terminal.semantics->transfer() : std::optional<ir::Transfer>{};
    if (transfer.has_value() != block.control.has_value()) return RegionDecline::invalid_graph;
    if (block.control && (block.control->kind != transfer->kind ||
                          block.control->terminal_source != terminal.address ||
                          block.control->block_revision != block.ssa->revision()))
      return RegionDecline::invalid_graph;
    if (!block.ssa) {
      if (block.edges.size() != 1) return RegionDecline::invalid_graph;
      const auto& edge = block.edges[0];
      if (terminal.opaque_control == OpaqueControl::normal_fallthrough) {
        if (edge.kind != CfgEdgeKind::fallthrough ||
            edge.target.kind != TargetKind::image_location ||
            edge.target.address != terminal.address + terminal.bytes.size())
          return RegionDecline::invalid_graph;
      } else if (edge.kind != (terminal.opaque_control == OpaqueControl::trap
                                   ? CfgEdgeKind::trap
                                   : CfgEdgeKind::opaque_unknown) ||
                 edge.target.kind != TargetKind::unknown || edge.target_block ||
                 edge.constant_image_dependency) {
        return RegionDecline::invalid_graph;
      }
    }

    if (block.ssa && !transfer &&
        (block.edges.size() != 1 || block.edges[0].kind != CfgEdgeKind::fallthrough ||
         block.edges[0].target.kind != TargetKind::image_location ||
         block.edges[0].target.address != terminal.address + terminal.bytes.size()))
      return RegionDecline::invalid_graph;
    if (block.dispatch) {
      const auto& dispatch = *block.dispatch;
      if (!transfer || transfer->kind != ir::TransferKind::jump || !block.ssa ||
          !block.ssa->boundaries().back().transfer || dispatch.guard_block >= blocks.size() ||
          dispatch.guard_block == bi || dispatch.index >= block.ssa->nodes().size() ||
          block.edges.size() - 1 > dispatch.bound)
        return RegionDecline::invalid_graph;
      // The named guard must be a way in, or it bounds nothing that arrives here.
      const auto& guard = blocks[dispatch.guard_block];
      if (!Charge(budget, guard.edges.size())) return RegionDecline::resource_limit;
      if (std::none_of(guard.edges.begin(), guard.edges.end(),
                       [bi](const CfgEdge& edge) { return edge.target_block == bi; }))
        return RegionDecline::invalid_graph;
      const auto target = block.ssa->boundaries().back().transfer->target;
      for (std::size_t i = 0; i < block.edges.size(); ++i) {
        const auto& edge = block.edges[i];
        if (edge.condition || edge.when || edge.target.kind != TargetKind::image_location ||
            edge.target.value != target ||
            (i != 0 && edge.target.address <= block.edges[i - 1].target.address))
          return RegionDecline::invalid_graph;
      }
    }

    if (transfer) {
      const auto kind = transfer->kind;
      if (kind == ir::TransferKind::jump || kind == ir::TransferKind::conditional) {
        if (!block.dispatch && (block.edges.size() > 2 ||
                                (kind == ir::TransferKind::conditional && block.edges.size() != 2)))
          return RegionDecline::invalid_graph;
        for (const auto& edge : block.edges)
          if (edge.kind != CfgEdgeKind::branch) return RegionDecline::invalid_graph;
        if (!block.dispatch && block.edges.size() == 2 &&
            (!block.edges[0].condition || block.edges[0].condition != block.edges[1].condition ||
             !block.edges[0].when || !block.edges[1].when ||
             block.edges[0].when == block.edges[1].when))
          return RegionDecline::invalid_graph;
      } else if (kind == ir::TransferKind::call) {
        if (block.edges.size() < 2 || block.edges.back().kind != CfgEdgeKind::potential_return)
          return RegionDecline::invalid_graph;
        for (std::size_t i = 0; i + 1 < block.edges.size(); ++i)
          if (block.edges[i].kind != CfgEdgeKind::callee) return RegionDecline::invalid_graph;
      } else if (kind == ir::TransferKind::return_) {
        if (block.edges.size() > 2) return RegionDecline::invalid_graph;
        for (const auto& edge : block.edges)
          if (edge.kind != CfgEdgeKind::return_) return RegionDecline::invalid_graph;
      } else {
        return RegionDecline::invalid_graph;
      }
    }
  }

  if (next != sources.size()) return RegionDecline::invalid_graph;
  for (std::size_t i = 0; i < graph.entries().size(); ++i) {
    if (!Charge(budget, std::bit_width(sources.size()) + 1)) return RegionDecline::resource_limit;
    const auto address = graph.entries()[i];
    if (i != 0 && graph.entries()[i - 1] >= address) return RegionDecline::invalid_graph;
    const auto source = std::lower_bound(
        sources.begin(), sources.end(), address,
        [](const SourceRecord& value, std::uint64_t entry) { return value.address < entry; });
    if (source == sources.end() || source->address != address ||
        blocks[owners[static_cast<std::size_t>(source - sources.begin())]].first_source !=
            static_cast<std::size_t>(source - sources.begin()))
      return RegionDecline::invalid_graph;
  }

  for (const auto& block : blocks) {
    for (const auto& edge : block.edges) {
      if (!Charge(budget, std::bit_width(sources.size()) + 1)) return RegionDecline::resource_limit;
      if (edge.condition.has_value() != edge.when.has_value()) return RegionDecline::invalid_graph;
      if (edge.condition && (!block.ssa || *edge.condition >= block.ssa->nodes().size() ||
                             block.ssa->nodes()[*edge.condition].width != 1))
        return RegionDecline::invalid_graph;
      if (edge.target.value && (!block.ssa || *edge.target.value >= block.ssa->nodes().size()))
        return RegionDecline::invalid_graph;
      TargetResolution expected = TargetResolution::unknown;
      std::optional<std::uint32_t> source_id, block_id;
      if (edge.target.kind == TargetKind::image_location) {
        const auto after = std::upper_bound(sources.begin(), sources.end(), edge.target.address,
                                            [](std::uint64_t address, const SourceRecord& source) {
                                              return address < source.address;
                                            });
        expected = TargetResolution::outside_population;
        if (after != sources.begin()) {
          const auto index = static_cast<std::size_t>(after - sources.begin() - 1);
          if (edge.target.address - sources[index].address < sources[index].bytes.size()) {
            source_id = static_cast<std::uint32_t>(index);
            if (edge.target.address != sources[index].address)
              expected = TargetResolution::mid_instruction;
            else {
              block_id = owners[index];
              if (blocks[*block_id].first_source != index) return RegionDecline::invalid_graph;
              expected = sources[index].semantics ? TargetResolution::block_entry
                                                  : TargetResolution::opaque_entry;
            }
          }
        }
      } else if (edge.target.kind == TargetKind::absolute_runtime)
        expected = TargetResolution::absolute_runtime;
      else if (edge.target.kind != TargetKind::unknown)
        return RegionDecline::invalid_graph;
      if (edge.resolution != expected || edge.target_source != source_id ||
          edge.target_block != block_id)
        return RegionDecline::invalid_graph;
    }
  }

  return RegionDecline::none;
}

}  // namespace

RegionsResult BuildRegions(Cfg&& graph, Budget& budget, RegionLimits limits) {
  const auto decline = [](RegionDecline reason) { return RegionsResult{{}, reason}; };
  if (const auto reason = CheckGraph(graph, budget, limits); reason != RegionDecline::none)
    return decline(reason);
  if (limits.max_blocks_per_region == 0 || limits.max_sources_per_region == 0)
    return decline(RegionDecline::resource_limit);
  const auto blocks = graph.blocks();
  const auto capacity = std::min<std::uint64_t>(limits.max_candidates, blocks.size() * 2);
  if (!Charge(budget, capacity, sizeof(Region))) return decline(RegionDecline::resource_limit);
  std::vector<Region> candidates;
  candidates.reserve(static_cast<std::size_t>(capacity));
  std::uint64_t occurrences = 0;
  for (std::uint32_t start = 0; start < blocks.size(); ++start) {
    if (!Charge(budget, 1)) return decline(RegionDecline::resource_limit);
    if (!blocks[start].ssa) continue;
    bool second_variant = false;
    for (unsigned variant = 0; variant < 2; ++variant) {
      if (variant == 1 && !second_variant) break;
      if (candidates.size() == capacity) return decline(RegionDecline::resource_limit);
      Region region{start, {}, {}, {}, RegionStop::unresolved, {}};
      const auto block_capacity =
          std::min<std::size_t>(limits.max_blocks_per_region, blocks.size());
      const auto source_capacity =
          std::min<std::size_t>(limits.max_sources_per_region, graph.sources().size());
      if (!Charge(budget, block_capacity * 2, sizeof(std::uint32_t)) ||
          !Charge(budget, source_capacity, sizeof(std::uint32_t)))
        return decline(RegionDecline::resource_limit);
      region.block_ids.reserve(block_capacity);
      region.transition_edges.reserve(block_capacity);
      region.source_ids.reserve(source_capacity);
      auto current = start;
      bool forked = false;
      for (;;) {
        const auto& block = blocks[current];
        if (block.source_count > source_capacity - region.source_ids.size()) {
          region.stop = RegionStop::source_limit;
          break;
        }

        if (block.source_count > limits.max_total_source_occurrences - occurrences ||
            !Charge(budget, block.source_count + 1))
          return decline(RegionDecline::resource_limit);
        occurrences += block.source_count;
        region.block_ids.push_back(current);
        for (std::uint32_t i = 0; i < block.source_count; ++i)
          region.source_ids.push_back(block.first_source + i);
        if (!Charge(budget, block.edges.size())) return decline(RegionDecline::resource_limit);
        if (block.control && block.control->kind == ir::TransferKind::call) {
          region.stop = RegionStop::call;
          break;
        }

        if (block.control && block.control->kind == ir::TransferKind::return_) {
          region.stop = RegionStop::return_;
          break;
        }

        if (block.dispatch) {
          region.stop = RegionStop::dispatch;
          break;
        }

        std::uint32_t edge_index = 0;
        if (block.edges.size() == 2) {
          if (forked) {
            region.stop = RegionStop::second_fork;
            break;
          }

          forked = true;
          second_variant = true;
          edge_index = variant;
        }

        const auto& edge = block.edges[edge_index];
        region.stopped_edge = edge_index;
        if (edge.resolution == TargetResolution::opaque_entry) {
          region.stop = RegionStop::opaque;
          break;
        }

        if ((edge.kind != CfgEdgeKind::branch && edge.kind != CfgEdgeKind::fallthrough) ||
            edge.target.kind != TargetKind::image_location ||
            edge.resolution != TargetResolution::block_entry) {
          region.stop = RegionStop::unresolved;
          break;
        }

        const auto target = *edge.target_block;
        if (!Charge(budget, region.block_ids.size())) return decline(RegionDecline::resource_limit);
        if (std::find(region.block_ids.begin(), region.block_ids.end(), target) !=
            region.block_ids.end()) {
          region.stop = RegionStop::cycle;
          break;
        }

        if (region.block_ids.size() == block_capacity) {
          region.stop = RegionStop::block_limit;
          break;
        }

        if (blocks[target].source_count > source_capacity - region.source_ids.size()) {
          region.stop = RegionStop::source_limit;
          break;
        }

        region.transition_edges.push_back(edge_index);
        region.stopped_edge.reset();
        current = target;
      }

      candidates.push_back(std::move(region));
    }
  }

  return {Regions(std::move(graph), std::move(candidates)), RegionDecline::none};
}

ir::PathResult NormalizeRegion(const Regions& regions, std::size_t candidate, Budget& budget,
                               ir::BlockLimits limits) {
  const auto invalid = [] { return ir::PathResult{{}, ir::BlockDecline::invalid_source}; };
  const auto resource = [] { return ir::PathResult{{}, ir::BlockDecline::resource_limit}; };
  if (candidate >= regions.candidates().size()) return invalid();
  const auto& selection = regions.candidates()[candidate];
  const auto blocks = regions.graph().blocks();
  const auto sources = regions.graph().sources();
  if (selection.block_ids.empty() || selection.block_ids.front() != selection.entry_block ||
      selection.transition_edges.size() != selection.block_ids.size() - 1 ||
      selection.source_ids.empty())
    return invalid();
  if (selection.source_ids.size() > limits.max_groups) return resource();
  std::size_t position = 0;
  for (std::size_t i = 0; i < selection.block_ids.size(); ++i) {
    if (!Charge(budget, 1)) return resource();
    const auto id = selection.block_ids[i];
    if (id >= blocks.size()) return invalid();
    const auto& block = blocks[id];
    if (!block.ssa || block.source_count == 0 || block.first_source > sources.size() ||
        block.source_count > sources.size() - block.first_source ||
        block.source_count > selection.source_ids.size() - position)
      return invalid();
    if (!Charge(budget, block.source_count)) return resource();
    for (std::uint32_t j = 0; j < block.source_count; ++j)
      if (selection.source_ids[position++] != block.first_source + j) return invalid();
    if (i + 1 < selection.block_ids.size()) {
      const auto edge_id = selection.transition_edges[i];
      if (edge_id >= block.edges.size()) return invalid();
      const auto& edge = block.edges[edge_id];
      const auto target = selection.block_ids[i + 1];
      if (target >= blocks.size() || edge.target_block != target ||
          edge.target_source != blocks[target].first_source ||
          edge.resolution != TargetResolution::block_entry ||
          edge.target.kind != TargetKind::image_location ||
          (edge.kind != CfgEdgeKind::branch && edge.kind != CfgEdgeKind::fallthrough) ||
          blocks[target].first_source >= sources.size() ||
          edge.target.address != sources[blocks[target].first_source].address)
        return invalid();
    }
  }

  if (position != selection.source_ids.size()) return invalid();
  if (!Charge(budget, position, sizeof(ir::Group))) return resource();
  for (const auto id : selection.source_ids) {
    if (id >= sources.size() || !sources[id].semantics) return invalid();
    const auto& group = *sources[id].semantics;
    if (!Charge(budget, group.bytes().size(), 1) ||
        !Charge(budget, group.nodes().size(), sizeof(ir::Node)) ||
        !Charge(budget, group.writes().size(), sizeof(ir::Write)))
      return resource();
    if (group.source_address() != sources[id].address ||
        !std::equal(group.bytes().begin(), group.bytes().end(), sources[id].bytes.begin(),
                    sources[id].bytes.end()))
      return invalid();
  }

  std::vector<ir::Group> originals;
  originals.reserve(position);
  for (const auto id : selection.source_ids) originals.push_back(*sources[id].semantics);
  return ir::NormalizePath(originals, budget, limits);
}

}  // namespace nyx::analysis
