#include "nyx/analysis/cfg.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <iterator>
#include <limits>

#include "nyx/ir/value_numbering.hpp"

namespace nyx::analysis {
std::uint64_t NextCfgIdentity() {
  static std::atomic<std::uint64_t> next{1};
  return next.fetch_add(1, std::memory_order_relaxed);
}

namespace {

CfgResult Decline(CfgDecline reason) { return {{}, reason}; }

bool Charge(Budget& budget, std::uint64_t count, std::uint64_t size) {
  return count <= UINT64_MAX / size &&
         budget.try_consume({count, count * size}) == BudgetDecline::none;
}

bool ChargeGroup(const ir::Group& group, Budget& budget) {
  return Charge(budget, group.bytes().size(), 1) &&
         Charge(budget, group.nodes().size(), sizeof(ir::Node)) &&
         Charge(budget, group.writes().size(), sizeof(ir::Write));
}

std::optional<std::uint32_t> Containing(std::span<const SourceRecord> sources,
                                        std::uint64_t address) {
  const auto after = std::upper_bound(
      sources.begin(), sources.end(), address,
      [](std::uint64_t value, const SourceRecord& source) { return value < source.address; });
  if (after == sources.begin()) return {};
  const auto& candidate = *std::prev(after);
  if (address - candidate.address >= candidate.bytes.size()) return {};
  return static_cast<std::uint32_t>(std::prev(after) - sources.begin());
}

CfgEdgeKind Kind(EdgeRole role) {
  switch (role) {
    case EdgeRole::branch:
      return CfgEdgeKind::branch;
    case EdgeRole::callee:
      return CfgEdgeKind::callee;
    case EdgeRole::return_:
      return CfgEdgeKind::return_;
    case EdgeRole::potential_return:
      return CfgEdgeKind::potential_return;
  }

  return CfgEdgeKind::opaque_unknown;
}

void Resolve(CfgEdge& edge, std::span<const SourceRecord> sources) {
  if (edge.target.kind == TargetKind::unknown) return;
  if (edge.target.kind == TargetKind::absolute_runtime) {
    edge.resolution = TargetResolution::absolute_runtime;
    return;
  }

  edge.target_source = Containing(sources, edge.target.address);
  if (!edge.target_source) {
    edge.resolution = TargetResolution::outside_population;
  } else if (sources[*edge.target_source].address != edge.target.address) {
    edge.resolution = TargetResolution::mid_instruction;
  } else {
    edge.resolution = sources[*edge.target_source].semantics ? TargetResolution::block_entry
                                                             : TargetResolution::opaque_entry;
  }
}

// A block whose only way in is a guard's false arm inherits that guard's bound
// on every entry, so a table dispatch there has one complete successor set for
// the block rather than one per path. The bound reaches the jump through a
// register: the predecessor's last write to it is the value the guard bounds,
// and the block reads that register at entry.
struct BlockDispatch {
  CfgDispatch facts;
  std::vector<std::uint64_t> destinations;
  bool constant_image_dependency;
};

std::optional<BlockDispatch> FindDispatch(const CfgBlock& block, const CfgBlock& predecessor,
                                          std::uint32_t guard_block, const CfgEdge& entering,
                                          const ImageFacts& facts, std::uint32_t max_targets,
                                          Budget& budget) {
  if (entering.when != false || !entering.condition || !predecessor.ssa || !block.ssa) return {};
  if (!block.control || block.control->kind != ir::TransferKind::jump) return {};
  const auto bound = BoundFromGuard(predecessor.ssa->nodes(), *entering.condition);
  if (!bound) return {};

  // Which register carries the bounded value out of the predecessor. A W write
  // publishes the value zero-extended, which preserves the bound, so that one
  // widening is followed; a sign extension would not be, and is not.
  const auto core = [nodes = predecessor.ssa->nodes()](ir::ValueId id) {
    for (unsigned step = 0; step < 8; ++step) {
      const auto next = ir::Unwrap(id, nodes);
      if (next != id) {
        id = next;
        continue;
      }

      const auto& node = nodes[id];
      if (node.op != ir::Op::zext || nodes[node.inputs[0]].width >= node.width) break;
      id = node.inputs[0];
    }

    return id;
  };

  std::optional<ir::StorageId> carried;
  const auto guarded = core(bound->first);
  for (const auto& boundary : predecessor.ssa->boundaries()) {
    for (const auto& write : boundary.writes) {
      if (budget.try_consume({1, 0}) != BudgetDecline::none) return {};
      if (core(write.value) == guarded)
        carried = write.storage;
      else if (carried && write.storage == *carried)
        carried.reset();
    }
  }

  if (!carried) return {};

  // One read of that register, or the bound names no single value here.
  std::optional<ir::ValueId> index;
  const auto nodes = block.ssa->nodes();
  for (std::size_t id = 0; id < nodes.size(); ++id) {
    if (nodes[id].op != ir::Op::read || nodes[id].storage != *carried) continue;
    if (index) return {};
    index = static_cast<ir::ValueId>(id);
  }

  if (!index) return {};
  const auto& boundaries = block.ssa->boundaries();
  if (boundaries.empty() || !boundaries.back().transfer) return {};
  auto destinations = EnumerateBoundedTarget(nodes, boundaries.back().transfer->target, *index,
                                             bound->second, facts, budget);
  if (destinations.destinations.empty()) return {};

  // Index order is the table; an edge set is a set.
  std::sort(destinations.destinations.begin(), destinations.destinations.end());
  destinations.destinations.erase(
      std::unique(destinations.destinations.begin(), destinations.destinations.end()),
      destinations.destinations.end());
  if (destinations.destinations.size() > max_targets) return {};
  return BlockDispatch{{guard_block, *index, bound->second},
                       std::move(destinations.destinations),
                       destinations.constant_image_dependency};
}

}  // namespace

CfgResult BuildCfg(std::span<const SourceRecord> input,
                   std::span<const std::uint64_t> selected_entries, Budget& budget,
                   CfgLimits limits, ImageFacts facts) {
  if (input.empty()) return Decline(CfgDecline::invalid_source);
  if (selected_entries.empty()) return Decline(CfgDecline::invalid_entry);
  if (input.size() > limits.max_sources || selected_entries.size() > limits.max_sources ||
      limits.max_passes == 0)
    return Decline(CfgDecline::resource_limit);
  const std::uint64_t search_work = std::bit_width(input.size()) + 1;
  if (!Charge(budget, input.size(), sizeof(SourceRecord)) ||
      !Charge(budget, input.size(), sizeof(std::uint8_t) + sizeof(std::uint32_t)) ||
      !Charge(budget, selected_entries.size(), sizeof(std::uint64_t)) ||
      budget.try_consume({input.size() * search_work +
                              selected_entries.size() *
                                  (std::bit_width(selected_entries.size()) + 1 + search_work),
                          0}) != BudgetDecline::none)
    return Decline(CfgDecline::resource_limit);
  std::uint64_t total_bytes = 0;
  for (const auto& source : input) {
    if (source.bytes.empty() || source.bytes.size() - 1 > UINT64_MAX - source.address) {
      return Decline(CfgDecline::invalid_source);
    }

    if (source.bytes.size() > limits.max_source_bytes - total_bytes)
      return Decline(CfgDecline::resource_limit);
    total_bytes += source.bytes.size();
    if (!Charge(budget, source.bytes.size(), 1)) return Decline(CfgDecline::resource_limit);
    if (source.semantics) {
      const auto& group = *source.semantics;
      if (group.nodes().size() > limits.block.max_nodes ||
          group.writes().size() > limits.block.max_storage) {
        return Decline(CfgDecline::resource_limit);
      }

      if (!ChargeGroup(group, budget)) return Decline(CfgDecline::resource_limit);
      if (source.opaque_reason != OpaqueReason::none ||
          source.opaque_control != OpaqueControl::unknown ||
          group.source_address() != source.address ||
          !std::equal(source.bytes.begin(), source.bytes.end(), group.bytes().begin(),
                      group.bytes().end())) {
        return Decline(CfgDecline::invalid_source);
      }
    } else if (source.opaque_reason != OpaqueReason::unsupported &&
               source.opaque_reason != OpaqueReason::invalid_encoding &&
               source.opaque_reason != OpaqueReason::not_decoded) {
      return Decline(CfgDecline::invalid_source);
    } else if (source.opaque_control != OpaqueControl::unknown &&
               (source.opaque_reason != OpaqueReason::unsupported ||
                source.bytes.size() > UINT64_MAX - source.address)) {
      return Decline(CfgDecline::invalid_source);
    }
  }

  std::vector<SourceRecord> sources(input.begin(), input.end());
  std::sort(sources.begin(), sources.end(),
            [](const auto& a, const auto& b) { return a.address < b.address; });
  for (std::size_t i = 1; i < sources.size(); ++i) {
    if (sources[i - 1].bytes.size() > sources[i].address - sources[i - 1].address) {
      return Decline(CfgDecline::invalid_source);
    }
  }

  std::vector<std::uint64_t> entries(selected_entries.begin(), selected_entries.end());
  std::sort(entries.begin(), entries.end());
  std::vector<std::uint8_t> leaders(sources.size());
  std::vector<std::uint32_t> source_blocks(sources.size());
  leaders[0] = 1;
  for (std::size_t i = 0; i < entries.size(); ++i) {
    const auto source = Containing(sources, entries[i]);
    if (!source || sources[*source].address != entries[i] || (i && entries[i - 1] == entries[i])) {
      return Decline(CfgDecline::invalid_entry);
    }

    leaders[*source] = 1;
  }

  for (std::size_t i = 0; i < sources.size(); ++i) {
    if (!sources[i].semantics) leaders[i] = 1;
    if (i && (!sources[i - 1].semantics || sources[i - 1].semantics->transfer() ||
              sources[i - 1].bytes.size() != sources[i].address - sources[i - 1].address))
      leaders[i] = 1;
  }

  if (!Charge(budget, facts.constants.size(), sizeof(ConstantImageRange) + sizeof(std::size_t)) ||
      !Charge(budget, facts.pointers.size(), sizeof(RelocatedPointer))) {
    return Decline(CfgDecline::resource_limit);
  }

  std::vector<ConstantImageRange> active_constants(facts.constants.begin(), facts.constants.end());
  std::vector<RelocatedPointer> active_pointers(facts.pointers.begin(), facts.pointers.end());
  std::vector<std::size_t> active_constant_indices(active_constants.size());
  for (std::size_t i = 0; i < active_constant_indices.size(); ++i) active_constant_indices[i] = i;
  const auto active_facts = [&]() -> ImageFacts {
    return {active_constants, active_pointers, facts.page_aligned_placement, facts.load_bias};
  };

  std::vector<ir::RefutedConstantSpan> refuted_constants;
  std::vector<std::size_t> refuted_pointers;
  ImageFactScan scan;

  for (std::uint32_t pass = 0; pass < limits.max_passes; ++pass) {
    if (budget.try_consume({sources.size(), 0}) != BudgetDecline::none)
      return Decline(CfgDecline::resource_limit);
    const auto count = static_cast<std::size_t>(std::count(leaders.begin(), leaders.end(), 1));
    if (count > limits.max_blocks || !Charge(budget, count, sizeof(CfgBlock))) {
      return Decline(CfgDecline::resource_limit);
    }

    std::vector<CfgBlock> blocks;
    blocks.reserve(count);
    for (std::size_t first = 0; first < sources.size();) {
      auto end = first + 1;
      while (end < sources.size() && !leaders[end]) ++end;
      CfgBlock block{static_cast<std::uint32_t>(first),
                     static_cast<std::uint32_t>(end - first),
                     {},
                     {},
                     {},
                     {}};
      if (!Charge(budget, 3, sizeof(CfgEdge))) return Decline(CfgDecline::resource_limit);
      block.edges.reserve(3);
      if (!sources[first].semantics) {
        if (sources[first].opaque_control == OpaqueControl::normal_fallthrough) {
          block.edges.push_back({CfgEdgeKind::fallthrough,
                                 {TargetKind::image_location,
                                  sources[first].address + sources[first].bytes.size(),
                                  {}},
                                 TargetResolution::unknown,
                                 {},
                                 {},
                                 {},
                                 {}});
        } else if (sources[first].opaque_control == OpaqueControl::trap) {
          block.edges.push_back({CfgEdgeKind::trap, {}, TargetResolution::unknown, {}, {}, {}, {}});
        } else {
          block.edges.push_back(
              {CfgEdgeKind::opaque_unknown, {}, TargetResolution::unknown, {}, {}, {}, {}});
        }
      } else {
        if (end - first > limits.block.max_groups ||
            !Charge(budget, end - first, sizeof(ir::Group))) {
          return Decline(CfgDecline::resource_limit);
        }

        for (auto i = first; i < end; ++i) {
          if (!ChargeGroup(*sources[i].semantics, budget))
            return Decline(CfgDecline::resource_limit);
        }

        std::vector<ir::Group> groups;
        groups.reserve(end - first);
        for (auto i = first; i < end; ++i) groups.push_back(*sources[i].semantics);
        auto normalized = ir::Normalize(groups, budget, limits.block);
        if (!normalized.block)
          return Decline(normalized.reason == ir::BlockDecline::resource_limit
                             ? CfgDecline::resource_limit
                             : CfgDecline::invalid_ir);
        block.ssa = std::move(normalized.block);
        if (!groups.back().transfer()) {
          const auto& last = sources[end - 1];
          block.edges.push_back({CfgEdgeKind::fallthrough,
                                 {TargetKind::image_location, last.address + last.bytes.size(), {}},
                                 TargetResolution::unknown,
                                 {},
                                 {},
                                 {},
                                 {}});
        }
      }

      std::fill(source_blocks.begin() + first, source_blocks.begin() + end,
                static_cast<std::uint32_t>(blocks.size()));
      blocks.push_back(std::move(block));
      first = end;
    }

    // The supplied population, including blocks later in address order, can
    // contradict a declaration used by an earlier transfer. Scan the complete
    // normalized snapshot before resolving any transfer or dispatch.
    if (!active_constants.empty() || !active_pointers.empty()) {
      std::vector<ir::ImageWrite> writes;

      // The last pass sees the final fact set, so its counts are the ones to
      // publish rather than a sum over passes.
      scan = {};
      for (const auto& block : blocks) {
        if (!block.ssa) {
          ++scan.unscanned_blocks;
          continue;
        }

        ++scan.scanned_blocks;
        auto found = KnownImageWrites(*block.ssa, active_facts(), budget);
        if (!found || !Charge(budget, found->writes.size(), sizeof(ir::ImageWrite))) {
          return Decline(CfgDecline::resource_limit);
        }

        if (found->unresolved > std::numeric_limits<std::uint64_t>::max() - scan.unresolved_writes)
          return Decline(CfgDecline::resource_limit);
        scan.unresolved_writes += found->unresolved;
        writes.insert(writes.end(), found->writes.begin(), found->writes.end());
      }

      const auto refutations = ir::RefuteImageFacts(active_facts(), writes, budget);
      if (!refutations) return Decline(CfgDecline::resource_limit);

      // A store withdraws the declared bytes it writes, and only those: the
      // rest of the range keeps its claim, as the pieces between them.
      const auto spans = ir::RefuteConstantBytes(active_facts(), writes, budget);
      if (!spans) return Decline(CfgDecline::resource_limit);
      if (!spans->empty()) {
        auto pieces = ir::RetainImageFacts(active_facts(), *spans, {}, budget);
        if (!pieces || !Charge(budget, spans->size() + pieces->constants.size(),
                               sizeof(ir::RefutedConstantSpan) + sizeof(std::size_t)))
          return Decline(CfgDecline::resource_limit);
        for (auto span : *spans) {
          span.index = active_constant_indices[span.index];
          refuted_constants.push_back(span);
        }

        std::vector<std::size_t> kept_indices;
        kept_indices.reserve(pieces->origins.size());
        for (const auto origin : pieces->origins)
          kept_indices.push_back(active_constant_indices[origin]);
        active_constants = std::move(pieces->constants);
        active_constant_indices = std::move(kept_indices);
      }

      for (const auto& conflict : refutations->pointers) {
        active_pointers[conflict.index].value_stable = false;
        if (!Charge(budget, 1, sizeof(std::size_t))) return Decline(CfgDecline::resource_limit);
        refuted_pointers.push_back(conflict.index);
      }
    }

    for (auto& block : blocks) {
      if (block.ssa && block.ssa->boundaries().back().transfer) {
        auto analyzed = AnalyzeControl(*block.ssa, budget, limits.block, active_facts());
        if (!analyzed.facts)
          return Decline(analyzed.reason == ControlDecline::resource_limit
                             ? CfgDecline::resource_limit
                             : CfgDecline::invalid_ir);
        block.control = std::move(analyzed.facts);
        for (unsigned i = 0; i < block.control->edge_count; ++i) {
          const auto& edge = block.control->edges[i];
          block.edges.push_back({Kind(edge.role),
                                 {edge.target.kind, edge.target.address, edge.target.value},
                                 TargetResolution::unknown,
                                 {},
                                 {},
                                 edge.condition,
                                 edge.when,
                                 edge.target.constant_image_dependency});
        }
      }

      for (auto& edge : block.edges) {
        if (budget.try_consume({search_work, 0}) != BudgetDecline::none)
          return Decline(CfgDecline::resource_limit);
        Resolve(edge, sources);
      }
    }

    // A dispatch is resolved on the complete snapshot too, because it needs the
    // predecessor's block, and installing its destinations here lets the same
    // discovery loop below pick them up as it would any other known successor.
    for (std::size_t id = 0; id < blocks.size(); ++id) {
      auto& block = blocks[id];
      if (block.edges.size() != 1 || block.edges[0].kind != CfgEdgeKind::branch ||
          block.edges[0].target.kind != TargetKind::unknown)
        continue;
      std::optional<std::uint32_t> guard;
      const CfgEdge* entering = nullptr;
      bool sole = true;
      for (std::size_t other = 0; other < blocks.size() && sole; ++other) {
        for (const auto& edge : blocks[other].edges) {
          if (budget.try_consume({1, 0}) != BudgetDecline::none)
            return Decline(CfgDecline::resource_limit);
          if (!edge.target_source || source_blocks[*edge.target_source] != id) continue;
          if (guard) {
            sole = false;
            break;
          }

          guard = static_cast<std::uint32_t>(other);
          entering = &edge;
        }
      }

      if (!sole || !guard) continue;
      auto dispatch = FindDispatch(block, blocks[*guard], *guard, *entering, active_facts(),
                                   limits.max_dispatch_targets, budget);
      if (!dispatch) continue;
      const auto value = block.edges[0].target.value;
      block.edges.clear();
      if (!Charge(budget, dispatch->destinations.size(), sizeof(CfgEdge))) {
        return Decline(CfgDecline::resource_limit);
      }

      block.edges.reserve(dispatch->destinations.size());
      for (const auto destination : dispatch->destinations) {
        if (budget.try_consume({search_work, 0}) != BudgetDecline::none) {
          return Decline(CfgDecline::resource_limit);
        }

        block.edges.push_back({CfgEdgeKind::branch,
                               {TargetKind::image_location, destination, value},
                               TargetResolution::unknown,
                               {},
                               {},
                               {},
                               {},
                               dispatch->constant_image_dependency});
        Resolve(block.edges.back(), sources);
      }

      block.dispatch = dispatch->facts;
    }

    // Discover on the complete pass snapshot. Mutating leaders while building
    // would silently grow the reserved block table and mix fact generations.
    bool changed = false;
    for (const auto& block : blocks) {
      for (const auto& edge : block.edges) {
        if ((edge.resolution == TargetResolution::block_entry ||
             edge.resolution == TargetResolution::opaque_entry) &&
            !leaders[*edge.target_source]) {
          leaders[*edge.target_source] = 1;
          changed = true;
        }
      }
    }

    if (changed) continue;
    for (auto& block : blocks) {
      for (auto& edge : block.edges) {
        if (edge.resolution == TargetResolution::block_entry ||
            edge.resolution == TargetResolution::opaque_entry) {
          edge.target_block = source_blocks[*edge.target_source];
        }
      }
    }

    const auto sort_work = [&](std::size_t size) {
      return budget.try_consume({size * (std::bit_width(size) + 1), 0}) == BudgetDecline::none;
    };

    if (!sort_work(refuted_constants.size()) || !sort_work(refuted_pointers.size())) {
      return Decline(CfgDecline::resource_limit);
    }

    std::stable_sort(refuted_constants.begin(), refuted_constants.end(),
                     [](const ir::RefutedConstantSpan& a, const ir::RefutedConstantSpan& b) {
                       return std::pair{a.index, a.address} < std::pair{b.index, b.address};
                     });
    std::sort(refuted_pointers.begin(), refuted_pointers.end());
    return {Cfg(std::move(sources), std::move(entries), std::move(blocks), pass + 1,
                std::move(refuted_constants), std::move(refuted_pointers), scan),
            CfgDecline::none};
  }

  return Decline(CfgDecline::resource_limit);
}

}  // namespace nyx::analysis
