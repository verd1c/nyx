#include <algorithm>

#include <gtest/gtest.h>

#include "nyx/analysis/ssa/build.hpp"
#include "nyx/analysis/ssa/reachability.hpp"
#include "nyx/ir/ssa/print.hpp"

namespace nyx::analysis {
namespace {
SourceRecord Record(ir::Group group) {
  return {group.source_address(),
          {group.bytes().begin(), group.bytes().end()},
          std::move(group),
          OpaqueReason::none};
}

struct Fixture {
  Regions regions;
  Unflattening recovered;
};

std::optional<Fixture> MakeFixture(Budget& budget) {
  std::vector<SourceRecord> sources{
      Record(ir::Group(0x100, {1, 2, 3, 4},
                       {{ir::Op::read, 1, {}, 0, 0},
                        {ir::Op::image_address, 64, {}, 0x104},
                        {ir::Op::image_address, 64, {}, 0x108}},
                       {}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::conditional, 1, 0, 2, {}})),
      Record(ir::Group(0x104, {5, 6, 7, 8},
                       {{ir::Op::constant, 64, {}, 11}, {ir::Op::image_address, 64, {}, 0x10c}},
                       {{8, 0}}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 1, {}, {}, {}})),
      Record(ir::Group(0x108, {9, 10, 11, 12},
                       {{ir::Op::constant, 64, {}, 22}, {ir::Op::image_address, 64, {}, 0x10c}},
                       {{8, 0}}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 1, {}, {}, {}})),
      Record(ir::Group(0x10c, {13, 14, 15, 16}, {{ir::Op::read, 64, {}, 0, 8}}, {},
                       ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}}))};
  const std::uint64_t entries[] = {0x100};
  auto cfg = BuildCfg(sources, entries, budget);
  if (!cfg.cfg) return {};
  Regions regions(std::move(*cfg.cfg), {});
  auto recovered = Unflatten(regions, {}, budget);
  if (!recovered.unflattening) return {};
  return Fixture{std::move(regions), std::move(*recovered.unflattening)};
}

ir::SsaHandle ByAddress(const ir::SsaGraph& graph, std::uint64_t address) {
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    const auto handle = graph.Handle(slot);
    if (handle && graph.Get(*handle)->address == address) return *handle;
  }

  return {};
}

TEST(SSA, JoinsDefinitionsAndRetainsUnknownContinuation) {
  Budget budget({1000000, 10000000});
  auto fixture = MakeFixture(budget);
  ASSERT_TRUE(fixture);
  const std::vector<std::optional<ir::RecoveredPath>> paths;
  auto built = BuildSsa(fixture->regions, fixture->recovered, paths, budget);
  ASSERT_TRUE(built.graph) << static_cast<int>(built.reason);
  auto& graph = *built.graph;
  const auto join_handle = ByAddress(graph, 0x10c);
  const auto left_handle = ByAddress(graph, 0x104);
  const auto right_handle = ByAddress(graph, 0x108);
  const auto* join = graph.Get(join_handle);
  ASSERT_NE(join, nullptr);
  ASSERT_EQ(join->reads.size(), 1);
  const auto& phi = join->phis[join->reads[0].phi];
  EXPECT_EQ(phi.storage, 8U);
  ASSERT_EQ(phi.incoming.size(), 2);
  EXPECT_EQ(phi.incoming[0].predecessor, left_handle);
  EXPECT_EQ(phi.incoming[1].predecessor, right_handle);
  EXPECT_EQ(phi.incoming[0].value.kind, ir::SsaValueKind::node);
  EXPECT_EQ(phi.incoming[1].value.kind, ir::SsaValueKind::node);
  ASSERT_EQ(join->edges.size(), 1);
  EXPECT_FALSE(join->edges[0].target_block);
  EXPECT_TRUE(join->edges[0].assumptions.unresolved_target);
  EXPECT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  auto printed = ir::PrintSsa(graph, budget);
  ASSERT_TRUE(printed.text);
  EXPECT_NE(printed.text->find("phi "), std::string::npos);
  EXPECT_NE(printed.text->find("unresolved"), std::string::npos);
  EXPECT_EQ(*printed.text, *ir::PrintSsa(graph, budget).text);

  // The audit record says whether every way in was declared: stores through
  // carried registers are placed on it.
  EXPECT_EQ(printed.text->find("entries closed"), std::string::npos);
  graph.SetEntriesClosed(true);
  const auto closed = ir::PrintSsa(graph, budget);
  ASSERT_TRUE(closed.text);
  EXPECT_NE(closed.text->find("\nentries closed\n"), std::string::npos);
  auto cloned = graph.Clone(budget);
  ASSERT_TRUE(cloned);
  EXPECT_NE(cloned->arena(), graph.arena());
  EXPECT_EQ(cloned->Get(join_handle), nullptr);
  EXPECT_EQ(ir::ValidateSsa(*cloned, budget), ir::SsaDecline::none);

  ASSERT_TRUE(graph.Update(join_handle, [](auto& block) {
    block.phis[block.reads[0].phi].incoming[0].value =
        block.phis[block.reads[0].phi].incoming[1].value;
  }));
  EXPECT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::invalid_graph);
  EXPECT_EQ(ir::PrintSsa(graph, budget).reason, ir::SsaDecline::invalid_graph);
}

TEST(SSA, RefusesMissingTransitionPathAndInvalidatesDeletedHandle) {
  Budget budget({1000000, 10000000});
  auto fixture = MakeFixture(budget);
  ASSERT_TRUE(fixture);
  fixture->recovered.transitions.push_back({0, 0, 0, {1}, {}});
  fixture->recovered.graph.blocks[0].transition = 0;
  const std::vector<std::optional<ir::RecoveredPath>> paths;
  EXPECT_EQ(BuildSsa(fixture->regions, fixture->recovered, paths, budget).reason,
            ir::SsaDecline::invalid_graph);

  ir::SsaGraph graph;
  const auto detached = graph.Add({});
  ASSERT_NE(graph.Get(detached), nullptr);
  EXPECT_TRUE(graph.Erase(detached));
  EXPECT_EQ(graph.Get(detached), nullptr);
  EXPECT_FALSE(graph.Erase(detached));
}

TEST(SSA, RefusesGraphFactsFromAnotherPopulation) {
  Budget budget({1000000, 10000000});
  auto first = MakeFixture(budget);
  auto other = MakeFixture(budget);
  ASSERT_TRUE(first);
  ASSERT_TRUE(other);
  const std::vector<std::optional<ir::RecoveredPath>> paths;
  EXPECT_EQ(BuildSsa(first->regions, other->recovered, paths, budget).reason,
            ir::SsaDecline::invalid_graph);
  EntryRelations relations;
  relations.source_identity = other->regions.graph().identity();
  relations.blocks.resize(first->regions.graph().blocks().size());
  EXPECT_EQ(BuildSsa(first->regions, first->recovered, paths, budget, &relations).reason,
            ir::SsaDecline::invalid_graph);
  relations.source_identity = first->regions.graph().identity();
  relations.blocks.back().push_back({8, 0, 16});
  EXPECT_EQ(BuildSsa(first->regions, first->recovered, paths, budget, &relations).reason,
            ir::SsaDecline::invalid_graph);
}

TEST(SSA, BridgeBindsTheRecoveredPathAndOriginalSourceSemantics) {
  Budget budget({1000000, 10000000});
  auto fixture = MakeFixture(budget);
  ASSERT_TRUE(fixture);
  Regions regions(fixture->regions.graph(), {{0, {0}, {0}, {}, RegionStop::dispatch, {}}});
  auto recovered = fixture->recovered;
  recovered.transitions.push_back({0, 0, 0, {}, {}});
  recovered.graph.blocks[0].transition = 0;
  const auto original = *regions.graph().sources()[0].semantics;
  auto basis = ir::NormalizePath(std::span(&original, 1), budget);
  ASSERT_TRUE(basis.path);
  std::vector<std::optional<ir::RecoveredPath>> paths;
  paths.emplace_back(ir::RecoveredPath(std::move(*basis.path), {}, 0));
  recovered.transitions[0].path_revision = paths[0]->revision();
  recovered.transitions[0].path_identity = paths[0]->identity();
  EXPECT_TRUE(BuildSsa(regions, recovered, paths, budget).graph);

  auto replacement = ir::NormalizePath(std::span(&original, 1), budget);
  ASSERT_TRUE(replacement.path);
  paths[0].emplace(std::move(*replacement.path), std::vector<ir::ConditionalRewrite>{}, 0);
  EXPECT_EQ(BuildSsa(regions, recovered, paths, budget).reason, ir::SsaDecline::invalid_graph);

  auto nodes = std::vector<ir::Node>(original.nodes().begin(), original.nodes().end());
  nodes[0].storage = 9;
  ir::Group changed(original.source_address(),
                    std::vector<std::uint8_t>(original.bytes().begin(), original.bytes().end()),
                    std::move(nodes), {}, original.memory_model(), original.transfer());
  auto changed_basis = ir::NormalizePath(std::span(&changed, 1), budget);
  ASSERT_TRUE(changed_basis.path);
  paths[0].emplace(std::move(*changed_basis.path), std::vector<ir::ConditionalRewrite>{}, 0);
  recovered.transitions[0].path_identity = paths[0]->identity();
  EXPECT_EQ(BuildSsa(regions, recovered, paths, budget).reason, ir::SsaDecline::invalid_graph);
}

TEST(SSA, ReadAfterBoundaryWriteUsesTheNewDefinition) {
  Budget budget({1000000, 10000000});
  ir::SsaGraph graph;
  ir::SsaBlock block{};
  block.original_block = 0;
  block.address = 0x100;
  block.source_groups = {0x100, 0x104};
  block.original_sources = {0, 1};
  block.nodes = {{ir::Op::constant, 64, {}, 7}, {ir::Op::read, 64, {}, 0, 8}};
  block.boundaries = {{0, 1, {{8, 0}}, {}}, {1, 1, {}, {}}};
  block.phis = {{8, 64, true, {}}};
  block.clobbers = {0};
  const auto handle = graph.Add(std::move(block));
  ASSERT_TRUE(graph.Update(handle, [&](auto& inserted) {
    inserted.reads = {{1, 0, {ir::SsaValueKind::node, handle, 0}}};
    inserted.exits = {{8, {ir::SsaValueKind::node, handle, 0}}};
  }));
  graph.SetEntries({handle});
  EXPECT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  auto clone = graph.Clone(budget);
  ASSERT_TRUE(clone);
  EXPECT_EQ(ir::ValidateSsa(*clone, budget), ir::SsaDecline::none);
  ASSERT_TRUE(graph.Update(
      handle, [&](auto& changed) { changed.reads[0].value = {ir::SsaValueKind::phi, handle, 0}; }));
  EXPECT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::invalid_graph);
}

TEST(SSA, KeepsEntryRelationProofAndDeclarationsAcrossEdits) {
  Budget budget({1000000, 10000000});
  ir::SsaGraph graph;
  ir::SsaBlock block{};
  block.original_block = 0;
  block.address = 0x100;
  block.entry_relations = {{31, 9, 16}};
  block.relation_declared_abi = true;
  block.relation_return_leaves = true;
  block.relation_constant_image = true;
  block.relation_declared_opaque_control = true;
  const auto handle = graph.Add(std::move(block));
  graph.SetEntries({handle});
  EXPECT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  const auto revision = graph.revision();
  EXPECT_FALSE(graph.Update(handle, [](auto& changed) { changed.entry_relations.clear(); }));
  EXPECT_FALSE(graph.Update(handle, [](auto& changed) { changed.entry_relations[0].offset = 8; }));
  EXPECT_FALSE(graph.Update(handle, [](auto& changed) { changed.relation_declared_abi = false; }));
  EXPECT_FALSE(graph.Update(handle, [](auto& changed) { changed.relation_return_leaves = false; }));
  EXPECT_FALSE(
      graph.Update(handle, [](auto& changed) { changed.relation_constant_image = false; }));
  EXPECT_FALSE(graph.Update(
      handle, [](auto& changed) { changed.relation_declared_opaque_control = false; }));
  EXPECT_EQ(graph.revision(), revision);
  auto cloned = graph.Clone(budget);
  ASSERT_TRUE(cloned);
  ASSERT_TRUE(cloned->Get(cloned->entries()[0]));
  EXPECT_FALSE(cloned->Update(cloned->entries()[0],
                              [](auto& changed) { changed.relation_return_leaves = false; }));
  EXPECT_EQ(ir::ValidateSsa(*cloned, budget), ir::SsaDecline::none);
}

TEST(SSA, FramePhiAcceptsBackedge) {
  Budget budget({1000000, 10000000});
  ir::SsaGraph graph;
  ir::SsaBlock block{};
  block.original_block = 0;
  block.address = 0x100;
  const auto handle = graph.Add(std::move(block));
  ASSERT_TRUE(graph.Update(handle, [&](auto& inserted) {
    inserted.frame_phis = {{-8, 8, true, {{handle, {ir::SsaValueKind::frame_phi, handle, 0}}}}};
    inserted.frame_exits = {{ir::SsaValueKind::frame_phi, handle, 0}};
    inserted.edges = {
        {ir::SsaEdgeKind::branch, ir::SsaTargetKind::image_location, 0x100, handle, {}, {}, {}}};
  }));
  graph.SetEntries({handle});
  EXPECT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
}

TEST(SSA, StorageWriteReachesTheSuccessorPhi) {
  Budget budget({1000000, 10000000});
  std::vector<SourceRecord> sources{
      Record(ir::Group(0x200, {1, 2, 3, 4},
                       {{ir::Op::constant, 64, {}, 42},
                        {ir::Op::write, 64, {0}, 0, 8},
                        {ir::Op::image_address, 64, {}, 0x204}},
                       {}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 2, {}, {}, {}})),
      Record(ir::Group(0x204, {5, 6, 7, 8}, {{ir::Op::read, 64, {}, 0, 8}}, {},
                       ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}}))};
  const std::uint64_t entries[] = {0x200};
  auto cfg = BuildCfg(sources, entries, budget);
  ASSERT_TRUE(cfg.cfg);
  Regions regions(std::move(*cfg.cfg), {});
  auto recovered = Unflatten(regions, {}, budget);
  ASSERT_TRUE(recovered.unflattening);
  const std::vector<std::optional<ir::RecoveredPath>> paths;
  auto built = BuildSsa(regions, *recovered.unflattening, paths, budget);
  ASSERT_TRUE(built.graph) << static_cast<int>(built.reason);
  const auto source = ByAddress(*built.graph, 0x200);
  const auto target = ByAddress(*built.graph, 0x204);
  const auto* first = built.graph->Get(source);
  const auto* second = built.graph->Get(target);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  const auto exit = std::find_if(first->exits.begin(), first->exits.end(),
                                 [](const auto& item) { return item.storage == 8; });
  ASSERT_NE(exit, first->exits.end());
  EXPECT_EQ(exit->value, (ir::SsaValue{ir::SsaValueKind::node, source, 0}));
  const auto phi = std::find_if(second->phis.begin(), second->phis.end(),
                                [](const auto& item) { return item.storage == 8; });
  ASSERT_NE(phi, second->phis.end());
  ASSERT_EQ(phi->incoming.size(), 1);
  EXPECT_EQ(phi->incoming[0].value, exit->value);
}

// `bl callee` at 0x100 continuing at 0x104, which returns. The entry reads x8
// so it carries a phi a call edge could wrongly feed.
std::optional<Fixture> CallFixture(Budget& budget, std::uint64_t callee) {
  std::vector<SourceRecord> sources{
      Record(ir::Group(0x100, {1, 2, 3, 4},
                       {{ir::Op::read, 64, {}, 0, 8},
                        {ir::Op::image_address, 64, {}, callee},
                        {ir::Op::image_address, 64, {}, 0x104}},
                       {}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::call, 1, {}, {}, 2})),
      Record(ir::Group(0x104, {5, 6, 7, 8}, {{ir::Op::read, 64, {}, 0, 30}}, {},
                       ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}})),
      Record(ir::Group(0x108, {9, 10, 11, 12}, {{ir::Op::read, 64, {}, 0, 30}}, {},
                       ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}}))};
  const std::uint64_t entries[] = {0x100};
  auto cfg = BuildCfg(sources, entries, budget);
  if (!cfg.cfg) return {};
  Regions regions(std::move(*cfg.cfg), {});

  // Returns leave the population, so only the call decides whether it closes.
  auto recovered = Unflatten(regions, {}, budget, {{}, true});
  if (!recovered.unflattening) return {};
  return Fixture{std::move(regions), std::move(*recovered.unflattening)};
}

TEST(SSA, RecursiveCallToTheListedEntryIsAnOrdinaryCall) {
  Budget budget({1000000, 10000000});
  auto fixture = CallFixture(budget, 0x100);
  ASSERT_TRUE(fixture);
  const auto& stitched = fixture->recovered.graph.blocks;
  ASSERT_TRUE(std::any_of(stitched.begin(), stitched.end(), [](const auto& block) {
    return std::any_of(block.edges.begin(), block.edges.end(), [](const RecoveredEdge& edge) {
      return edge.kind == CfgEdgeKind::callee && edge.target_block;
    });
  }));
  const std::vector<std::optional<ir::RecoveredPath>> paths;
  auto built = BuildSsa(fixture->regions, fixture->recovered, paths, budget);
  ASSERT_TRUE(built.graph) << static_cast<int>(built.reason);
  const auto entry = ByAddress(*built.graph, 0x100);
  const auto* block = built.graph->Get(entry);
  ASSERT_NE(block, nullptr);
  const auto callee = std::find_if(block->edges.begin(), block->edges.end(), [](const auto& edge) {
    return edge.kind == ir::SsaEdgeKind::callee;
  });
  ASSERT_NE(callee, block->edges.end());
  EXPECT_EQ(callee->address, 0x100);
  EXPECT_FALSE(callee->target_block);
  for (const auto& phi : block->phis) EXPECT_TRUE(phi.incoming.empty());
  EXPECT_EQ(ir::ValidateSsa(*built.graph, budget), ir::SsaDecline::none);

  // It leaves and comes back through its continuation as any unknown callee
  // does, so the call block is complete and the population closes.
  EXPECT_EQ(fixture->recovered.reachable_unknown_callees, 1U);
  const auto back = std::find_if(block->edges.begin(), block->edges.end(), [](const auto& edge) {
    return edge.kind == ir::SsaEdgeKind::potential_return;
  });
  ASSERT_NE(back, block->edges.end());
  EXPECT_EQ(back->address, 0x104U);
  EXPECT_TRUE(back->assumptions.callee_returns_to_continuation);
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(*block, budget), ir::SsaDecline::none);
  std::vector<ir::Group> decoded;
  for (const auto& source : fixture->regions.graph().sources()) {
    ASSERT_TRUE(source.semantics);
    decoded.push_back(*source.semantics);
  }

  const auto reachable =
      ProveSsaReachability(*built.graph, decoded, ir::SsaEntryScope::closed_population, budget);
  EXPECT_TRUE(reachable.facts) << static_cast<int>(reachable.reason);
}

TEST(SSA, InteriorCallFollowsTheCalleeWithoutInventingAReturn) {
  Budget budget({1000000, 10000000});
  auto fixture = CallFixture(budget, 0x108);
  ASSERT_TRUE(fixture);
  const std::vector<std::optional<ir::RecoveredPath>> paths;
  auto built = BuildSsa(fixture->regions, fixture->recovered, paths, budget);
  ASSERT_TRUE(built.graph) << static_cast<int>(built.reason);
  const auto entry = built.graph->entries()[0];
  const auto* block = built.graph->Get(entry);
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->edges.size(), 1);
  EXPECT_EQ(block->edges[0].kind, ir::SsaEdgeKind::branch);
  EXPECT_EQ(block->edges[0].target_block, ByAddress(*built.graph, 0x108));
  EXPECT_EQ(block->boundaries.back().transfer->kind, ir::TransferKind::call);
  EXPECT_TRUE(std::all_of(block->clobbers.begin(), block->clobbers.end(),
                          [](auto clobber) { return !clobber; }));
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(*block, budget), ir::SsaDecline::none);
  const auto* continuation = built.graph->Get(ByAddress(*built.graph, 0x104));
  ASSERT_NE(continuation, nullptr);
  for (const auto& phi : continuation->phis) EXPECT_TRUE(phi.incoming.empty());

  auto forged = *block;
  forged.edges[0].address = 0x104;
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(forged, budget), ir::SsaDecline::invalid_graph);
  forged = *block;
  forged.edges.push_back({ir::SsaEdgeKind::potential_return,
                          ir::SsaTargetKind::image_location,
                          0x104,
                          ByAddress(*built.graph, 0x104),
                          {},
                          {},
                          {}});
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(forged, budget), ir::SsaDecline::invalid_graph);
  forged = *block;
  forged.nodes[forged.boundaries.back().transfer->target] = {ir::Op::read, 64, {}, 0, 8};
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(forged, budget), ir::SsaDecline::invalid_graph);
}

// Under a declared bias of zero the number 0x18 names image location 0x18,
// which a declared constant range covers.
TEST(SSA, PathReadThroughTheDeclaredBiasValidates) {
  Budget budget({1000000, 10000000});
  std::vector<SourceRecord> sources{Record(ir::Group(
      0x100, {1, 2, 3, 4},
      {{ir::Op::constant, 64, {}, 0x18}, {ir::Op::load, 64, {0}}, {ir::Op::read, 64, {}, 0, 30}},
      {{8, 1}}, ir::MemoryModel::atomic_scalar_reference,
      ir::Transfer{ir::TransferKind::return_, 2, {}, {}, {}}))};
  const std::uint8_t bytes[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
  const ir::ConstantImageRange range{0x18, bytes};
  const ImageFacts facts{std::span(&range, 1), {}, false, 0};
  const std::uint64_t entries[] = {0x100};
  auto cfg = BuildCfg(sources, entries, budget, {}, facts);
  ASSERT_TRUE(cfg.cfg);
  Regions regions(std::move(*cfg.cfg), {});
  auto recovered = Unflatten(regions, {}, budget);
  ASSERT_TRUE(recovered.unflattening);
  const std::vector<std::optional<ir::RecoveredPath>> paths;
  auto built = BuildSsa(regions, *recovered.unflattening, paths, budget, nullptr, facts);
  ASSERT_TRUE(built.graph) << static_cast<int>(built.reason);
  EXPECT_EQ(built.graph->load_bias(), std::optional<std::uint64_t>(0));
  const auto* block = built.graph->Get(ByAddress(*built.graph, 0x100));
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->path_reads.size(), 1);
  EXPECT_EQ(block->path_reads[0].node, 1U);
  EXPECT_EQ(block->path_reads[0].address, 0x18U);
  EXPECT_FALSE(block->path_reads[0].relocated);
  EXPECT_EQ(block->path_reads[0].value, 0x8877665544332211U);
}
}  // namespace
}  // namespace nyx::analysis
