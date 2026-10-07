#include "nyx/analysis/ssa/build.hpp"

#include <algorithm>
#include <bit>
#include <map>

namespace nyx::analysis {
namespace {
ir::SsaEdgeKind Kind(CfgEdgeKind kind) {
  switch (kind) {
    case CfgEdgeKind::fallthrough:
      return ir::SsaEdgeKind::fallthrough;
    case CfgEdgeKind::branch:
      return ir::SsaEdgeKind::branch;
    case CfgEdgeKind::callee:
      return ir::SsaEdgeKind::callee;
    case CfgEdgeKind::return_:
      return ir::SsaEdgeKind::return_;
    case CfgEdgeKind::potential_return:
      return ir::SsaEdgeKind::potential_return;
    case CfgEdgeKind::opaque_unknown:
      return ir::SsaEdgeKind::opaque_unknown;
    case CfgEdgeKind::trap:
      return ir::SsaEdgeKind::trap;
  }

  return ir::SsaEdgeKind::opaque_unknown;
}

ir::SsaTargetKind Target(TargetKind kind) {
  switch (kind) {
    case TargetKind::image_location:
      return ir::SsaTargetKind::image_location;
    case TargetKind::absolute_runtime:
      return ir::SsaTargetKind::absolute_runtime;
    case TargetKind::unknown:
      return ir::SsaTargetKind::unknown;
  }

  return ir::SsaTargetKind::unknown;
}

bool SameGroup(const ir::Group& a, const ir::Group& b) {
  if (a.source_address() != b.source_address() || a.memory_model() != b.memory_model() ||
      !std::equal(a.bytes().begin(), a.bytes().end(), b.bytes().begin(), b.bytes().end()) ||
      a.nodes().size() != b.nodes().size() || a.writes().size() != b.writes().size())
    return false;
  for (std::size_t i = 0; i < a.nodes().size(); ++i) {
    const auto& x = a.nodes()[i];
    const auto& y = b.nodes()[i];
    if (x.op != y.op || x.width != y.width || x.inputs != y.inputs || x.immediate != y.immediate ||
        x.storage != y.storage || x.access.byte_order != y.access.byte_order ||
        x.access.alignment != y.access.alignment ||
        x.access.decline_on_unaligned != y.access.decline_on_unaligned)
      return false;
  }

  for (std::size_t i = 0; i < a.writes().size(); ++i)
    if (a.writes()[i].storage != b.writes()[i].storage ||
        a.writes()[i].value != b.writes()[i].value)
      return false;
  const auto x = a.transfer(), y = b.transfer();
  if (x.has_value() != y.has_value()) return false;
  return !x || (x->kind == y->kind && x->target == y->target && x->condition == y->condition &&
                x->alternative == y->alternative && x->continuation == y->continuation);
}

// The declared image values a block's loads read travel with it, so its
// successor rule does not need the run's facts. False means the budget ran out.
bool RecordPathReads(ir::SsaBlock& block, ImageFacts facts, Budget& budget) {
  for (ir::ValueId id = 0; id < block.nodes.size(); ++id) {
    if (block.nodes[id].op != ir::Op::load) continue;

    // Both lookups binary-search the sorted slots and scan the few ranges.
    // The address walk visits at most 64 nodes.
    if (budget.try_consume(
            {80 + facts.constants.size() +
                 4 * static_cast<std::uint64_t>(std::bit_width(facts.pointers.size())),
             sizeof(ir::SsaPathRead)}) != BudgetDecline::none)
      return false;
    if (const auto read = ir::ResolveSsaPathRead(block.nodes, id, facts))
      block.path_reads.push_back(*read);
  }

  return true;
}

}  // namespace

SsaResult BuildSsa(const Regions& regions, const Unflattening& recovered,
                   std::span<const std::optional<ir::RecoveredPath>> paths, Budget& budget,
                   const EntryRelations* relations, ImageFacts facts, ir::SsaObservability observed,
                   bool entries_closed) {
  const auto invalid = [] { return SsaResult{{}, ir::SsaDecline::invalid_graph}; };
  const auto limited = [] { return SsaResult{{}, ir::SsaDecline::resource_limit}; };
  const auto cfg = regions.graph().blocks();
  const auto sources = regions.graph().sources();
  const auto candidates = regions.candidates();
  if (paths.size() != candidates.size() || cfg.size() > UINT32_MAX ||
      recovered.source_identity != regions.graph().identity())
    return invalid();
  if (relations && (relations->blocks.size() != cfg.size() ||
                    relations->source_identity != regions.graph().identity()))
    return invalid();
  if (relations) {
    const auto replay = ProveEntryRelations(
        regions.graph(), budget,
        {relations->calling_convention, relations->preserved, relations->return_leaves});
    if (!replay.relations)
      return replay.reason == RelationDecline::resource_limit ? limited() : invalid();
    if (budget.try_consume({relations->preserved.size(), 0}) != BudgetDecline::none)
      return limited();
    if (replay.relations->preserved != relations->preserved ||
        replay.relations->calling_convention != relations->calling_convention ||
        replay.relations->return_leaves != relations->return_leaves ||
        replay.relations->constant_image != relations->constant_image ||
        replay.relations->declared_opaque_control != relations->declared_opaque_control)
      return invalid();
    for (std::size_t id = 0; id < cfg.size(); ++id) {
      if (budget.try_consume({1 + relations->blocks[id].size(), 0}) != BudgetDecline::none)
        return limited();
      if (replay.relations->blocks[id] != relations->blocks[id]) return invalid();
    }
  }

  std::map<ir::StorageId, unsigned> storage;
  ir::SsaGraph graph;

  // What survives a call comes only from the declared callee contract, which
  // holds per call regardless of whether entry relations can be proved.
  // Relations proved under a different contract would be inconsistent with it.
  if (relations && observed.declared && !relations->preserved.empty() &&
      relations->preserved != observed.preserved)
    return invalid();
  graph.SetObservability(std::move(observed));
  std::vector<std::optional<ir::SsaHandle>> handles(cfg.size());
  for (const auto& stitched : recovered.graph.blocks) {
    if (stitched.block >= cfg.size() || handles[stitched.block] ||
        cfg[stitched.block].first_source >= sources.size() ||
        stitched.address != sources[cfg[stitched.block].first_source].address)
      return invalid();
    ir::SsaBlock block{};
    block.original_block = stitched.block;
    block.address = stitched.address;
    block.transition = stitched.transition;
    if (relations) {
      const auto& related = relations->blocks[stitched.block];
      if (budget.try_consume({related.size(), related.size() * sizeof(ir::StorageRelation)}) !=
          BudgetDecline::none)
        return limited();
      block.entry_relations = related;
      block.relation_declared_abi = relations->calling_convention;
      block.relation_return_leaves = relations->return_leaves;
      block.relation_constant_image = relations->constant_image;
      block.relation_declared_opaque_control = relations->declared_opaque_control;
    }

    if (stitched.transition) {
      if (*stitched.transition >= recovered.transitions.size()) return invalid();
      const auto& transition = recovered.transitions[*stitched.transition];
      if (transition.entry_block != stitched.block || transition.candidate >= paths.size() ||
          !paths[transition.candidate])
        return invalid();
      if (paths[transition.candidate]->revision() != transition.path_revision ||
          paths[transition.candidate]->identity() != transition.path_identity)
        return invalid();
      const auto checked_path =
          ir::ValidateRecoveredPath(*paths[transition.candidate], budget, {}, facts);
      if (checked_path != ir::BlockDecline::none)
        return checked_path == ir::BlockDecline::resource_limit ? limited() : invalid();
      const auto& path = paths[transition.candidate]->basis();
      block.path_revision = paths[transition.candidate]->revision();
      const auto& recovered_path = *paths[transition.candidate];
      const auto& region = candidates[transition.candidate];
      if (path.sources().size() != region.source_ids.size()) return invalid();
      for (std::size_t i = 0; i < region.source_ids.size(); ++i) {
        if (budget.try_consume({1 + path.sources()[i].bytes().size() +
                                    path.sources()[i].nodes().size() +
                                    path.sources()[i].writes().size(),
                                0}) != BudgetDecline::none)
          return limited();
        if (region.source_ids[i] >= sources.size() || !sources[region.source_ids[i]].semantics ||
            path.sources()[i].bytes().size() != sources[region.source_ids[i]].bytes.size() ||
            !std::equal(path.sources()[i].bytes().begin(), path.sources()[i].bytes().end(),
                        sources[region.source_ids[i]].bytes.begin()) ||
            !SameGroup(path.sources()[i], *sources[region.source_ids[i]].semantics))
          return invalid();
      }

      if (budget.try_consume(
              {path.nodes().size() + path.boundaries().size() + path.sources().size() +
                   recovered_path.rewrites().size() + recovered_path.omissions().size() +
                   recovered_path.paired_load_omissions().size() +
                   recovered_path.destinations().size() + 1,
               sizeof(ir::SsaBlock) + path.nodes().size() * sizeof(ir::Node) +
                   path.boundaries().size() * sizeof(ir::Boundary) +
                   path.sources().size() * (sizeof(std::uint64_t) + sizeof(std::uint32_t) +
                                            sizeof(std::vector<std::uint8_t>)) +
                   recovered_path.rewrites().size() * sizeof(ir::ConditionalRewrite) +
                   recovered_path.omissions().size() * sizeof(ir::StoreOmission) +
                   recovered_path.paired_load_omissions().size() * sizeof(ir::PairedLoadOmission) +
                   recovered_path.destinations().size() * sizeof(ir::Node)}) != BudgetDecline::none)
        return limited();
      block.control_rewrites.assign(recovered_path.rewrites().begin(),
                                    recovered_path.rewrites().end());
      block.store_omissions.assign(recovered_path.omissions().begin(),
                                   recovered_path.omissions().end());
      block.paired_load_omissions.assign(recovered_path.paired_load_omissions().begin(),
                                         recovered_path.paired_load_omissions().end());
      block.destination_nodes.assign(recovered_path.destinations().begin(),
                                     recovered_path.destinations().end());
      block.original_sources = region.source_ids;
      block.nodes.assign(path.nodes().begin(), path.nodes().end());
      if (!RecordPathReads(block, facts, budget)) return limited();
      block.boundaries.assign(path.boundaries().begin(), path.boundaries().end());
      for (const auto& source : path.sources()) {
        if (budget.try_consume({source.bytes().size(), source.bytes().size()}) !=
            BudgetDecline::none)
          return limited();
        block.source_groups.push_back(source.source_address());
        block.source_bytes.emplace_back(source.bytes().begin(), source.bytes().end());
      }
    } else {
      const auto& original = cfg[stitched.block];
      if (original.first_source > sources.size() ||
          original.source_count > sources.size() - original.first_source)
        return invalid();
      const auto nodes = original.ssa ? original.ssa->nodes().size() : 0;
      const auto boundaries = original.ssa ? original.ssa->boundaries().size() : 0;
      if (budget.try_consume(
              {nodes + boundaries + original.source_count + 1,
               sizeof(ir::SsaBlock) + nodes * sizeof(ir::Node) + boundaries * sizeof(ir::Boundary) +
                   original.source_count * (sizeof(std::uint64_t) + sizeof(std::uint32_t) +
                                            sizeof(std::vector<std::uint8_t>))}) !=
          BudgetDecline::none)
        return limited();
      for (std::uint32_t i = 0; i < original.source_count; ++i) {
        const auto& bytes = sources[original.first_source + i].bytes;
        if (budget.try_consume({bytes.size(), bytes.size()}) != BudgetDecline::none)
          return limited();
        block.source_groups.push_back(sources[original.first_source + i].address);
        block.source_bytes.push_back(bytes);
      }

      for (std::uint32_t i = 0; i < original.source_count; ++i)
        block.original_sources.push_back(original.first_source + i);
      if (original.ssa) {
        block.nodes.assign(original.ssa->nodes().begin(), original.ssa->nodes().end());
        block.boundaries.assign(original.ssa->boundaries().begin(),
                                original.ssa->boundaries().end());
        if (!RecordPathReads(block, facts, budget)) return limited();
      } else {
        block.opaque = true;
      }
    }

    for (const auto& node : block.nodes) {
      if (node.op != ir::Op::read && node.op != ir::Op::write) continue;
      const auto [it, inserted] = storage.emplace(node.storage, node.width);
      if (!inserted && it->second != node.width) return invalid();
    }

    for (const auto& boundary : block.boundaries) {
      for (const auto& write : boundary.writes) {
        if (write.value >= block.nodes.size()) return invalid();
        const auto [it, inserted] = storage.emplace(write.storage, block.nodes[write.value].width);
        if (!inserted && it->second != block.nodes[write.value].width) return invalid();
      }
    }

    handles[stitched.block] = graph.Add(std::move(block));
  }

  std::vector<ir::SsaHandle> entries;
  for (const auto id : recovered.graph.entries) {
    if (id >= handles.size() || !handles[id]) return invalid();
    entries.push_back(*handles[id]);
  }

  graph.SetEntries(std::move(entries));
  if (storage.size() > UINT32_MAX) return limited();
  for (const auto& stitched : recovered.graph.blocks) {
    auto copied = graph.CopyBlock(*handles[stitched.block], budget);
    if (!copied) return limited();
    auto block = std::move(*copied);
    const bool external = std::find(graph.entries().begin(), graph.entries().end(),
                                    *handles[stitched.block]) != graph.entries().end();
    // A call to a listed entry, recursion say, re-enters the population where
    // every value is already unknown, which is what closed entries rest on;
    // it is a call like any other. One into another block would enter the
    // population somewhere no declaration covers, so its instructions must be
    // followed with the actual incoming state instead of a callee summary.
    const auto listed = [&](std::uint32_t id) {
      return std::find(recovered.graph.entries.begin(), recovered.graph.entries.end(), id) !=
             recovered.graph.entries.end();
    };

    const bool internal_call =
        std::any_of(stitched.edges.begin(), stitched.edges.end(), [&](const RecoveredEdge& edge) {
          return edge.kind == CfgEdgeKind::callee && edge.target_block &&
                 !listed(*edge.target_block);
        });
    if (internal_call) {
      // Follow the actual transfer with the instruction's writes intact. The
      // continuation is metadata, not a second successor of this instruction;
      // any later return must be represented where it executes.
      if (block.transition || block.boundaries.empty() || !block.boundaries.back().transfer)
        return invalid();
      if (budget.try_consume({2 * stitched.edges.size(), 0}) != BudgetDecline::none)
        return limited();
      const auto& transfer = *block.boundaries.back().transfer;
      if (transfer.kind != ir::TransferKind::call ||
          std::count_if(stitched.edges.begin(), stitched.edges.end(),
                        [](const auto& edge) { return edge.kind == CfgEdgeKind::callee; }) != 1)
        return invalid();
      for (const auto& edge : stitched.edges) {
        if (edge.kind == CfgEdgeKind::potential_return) continue;
        if (edge.kind != CfgEdgeKind::callee || !edge.target_block || edge.condition || edge.when ||
            edge.target.kind != TargetKind::image_location)
          return invalid();
      }
    }

    const bool call =
        !internal_call &&
        std::any_of(stitched.edges.begin(), stitched.edges.end(), [](const RecoveredEdge& edge) {
          return edge.kind == CfgEdgeKind::potential_return;
        });
    const bool clobber = block.opaque || call;
    const auto preserved = [&](ir::StorageId id) {
      const auto& contract = graph.observability();
      return !block.opaque && contract.declared &&
             std::binary_search(contract.preserved.begin(), contract.preserved.end(), id);
    };

    if (budget.try_consume({storage.size() * 3 + stitched.edges.size(),
                            storage.size() * (sizeof(ir::SsaPhi) + sizeof(ir::SsaExitValue) + 1) +
                                stitched.edges.size() * sizeof(ir::SsaEdge)}) !=
        BudgetDecline::none)
      return limited();
    for (const auto& [id, width] : storage) {
      const bool fresh = clobber && !preserved(id);
      block.phis.push_back({id, width, external, {}});
      block.clobbers.push_back(fresh);
      const auto index = static_cast<std::uint32_t>(block.phis.size() - 1);
      block.exits.push_back({id,
                             {fresh ? ir::SsaValueKind::clobber : ir::SsaValueKind::phi,
                              *handles[stitched.block], index}});
    }

    std::vector<ir::SsaValue> current;
    for (std::size_t i = 0; i < block.phis.size(); ++i)
      current.push_back(
          {ir::SsaValueKind::phi, *handles[stitched.block], static_cast<std::uint32_t>(i)});
    for (const auto& boundary : block.boundaries) {
      for (std::uint32_t id = boundary.first_node; id < boundary.first_node + boundary.node_count;
           ++id) {
        const auto& node = block.nodes[id];
        if (node.op != ir::Op::read) continue;
        const auto found = storage.find(node.storage);
        if (found == storage.end()) return invalid();
        const auto index = static_cast<std::uint32_t>(std::distance(storage.begin(), found));
        block.reads.push_back({id, index, current[index]});
      }

      for (std::uint32_t id = boundary.first_node; id < boundary.first_node + boundary.node_count;
           ++id) {
        const auto& node = block.nodes[id];
        if (node.op != ir::Op::write) continue;
        const auto found = storage.find(node.storage);
        if (found == storage.end()) return invalid();
        const auto index = static_cast<std::size_t>(std::distance(storage.begin(), found));
        current[index] = {ir::SsaValueKind::node, *handles[stitched.block], node.inputs[0]};
      }

      for (const auto& write : boundary.writes) {
        const auto found = storage.find(write.storage);
        if (found == storage.end()) return invalid();
        const auto index = static_cast<std::size_t>(std::distance(storage.begin(), found));
        current[index] = {ir::SsaValueKind::node, *handles[stitched.block], write.value};
      }
    }

    for (std::size_t i = 0; i < current.size(); ++i)
      if (!block.clobbers[i]) block.exits[i].value = current[i];
    for (const auto& edge : stitched.edges) {
      if (internal_call && edge.kind == CfgEdgeKind::potential_return) continue;
      ir::SsaEdge copied{
          Kind(edge.kind),
          Target(edge.target.kind),
          edge.target.address,
          {},
          edge.condition,
          edge.when,
          {edge.assumptions.constant_image, edge.assumptions.callee_returns_to_continuation,
           edge.assumptions.return_leaves, edge.assumptions.unresolved_target,
           edge.assumptions.declared_opaque_control, edge.assumptions.entry_relations,
           edge.assumptions.declared_abi, edge.assumptions.declared_return_leaves,
           edge.assumptions.declared_noreturn}};
      if (internal_call) copied.kind = ir::SsaEdgeKind::branch;
      if (edge.target_block && (internal_call || edge.kind != CfgEdgeKind::callee)) {
        if (*edge.target_block >= handles.size() || !handles[*edge.target_block]) return invalid();
        copied.target_block = *handles[*edge.target_block];
      }

      block.edges.push_back(std::move(copied));
    }

    if (!graph.Replace(*handles[stitched.block], std::move(block))) return invalid();
  }

  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    const auto source = graph.Handle(slot);
    if (!source) continue;
    auto copied = graph.CopyBlock(*source, budget);
    if (!copied) return limited();
    const auto& block = *copied;
    for (const auto& edge : block.edges) {
      if (!edge.target_block) continue;
      auto copied_target = graph.CopyBlock(*edge.target_block, budget);
      if (!copied_target) return limited();
      auto target = std::move(*copied_target);
      for (std::size_t i = 0; i < target.phis.size(); ++i) {
        auto& incoming = target.phis[i].incoming;
        if (std::none_of(incoming.begin(), incoming.end(), [&](const ir::SsaPhiInput& input) {
              return input.predecessor == *source;
            }))
          incoming.push_back({*source, block.exits[i].value});
      }

      if (!graph.Replace(*edge.target_block, std::move(target))) return invalid();
    }
  }

  // A path read located through the declared bias names a location the graph
  // can check only under that same bias.
  graph.SetLoadBias(facts.load_bias);

  // The stores a path read is checked against are placed as the checker
  // places them, which is further when every way in is listed.
  graph.SetEntriesClosed(entries_closed);
  if (!ir::SsaDropWrittenPathReads(graph, budget, &facts)) return limited();
  const auto reason = ir::ValidateSsa(graph, budget);
  if (reason != ir::SsaDecline::none) return {{}, reason};
  return {std::move(graph), ir::SsaDecline::none};
}

}  // namespace nyx::analysis
