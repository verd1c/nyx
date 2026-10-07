#include <gtest/gtest.h>

#include "nyx/analysis/ssa/liveness.hpp"
#include "nyx/recovery/constant_load.hpp"
#include "nyx/recovery/ssa/pure_dce.hpp"

namespace nyx::recovery {
namespace {
Budget Plenty() { return Budget({1000000, 10000000}); }

ir::SsaGraph Graph() {
  ir::SsaGraph graph;
  ir::SsaBlock block{};
  block.address = 0x100;
  block.source_groups = {0x100, 0x104};
  block.source_bytes = {{0x1f, 0x00, 0x01, 0x8b}, {0xc0, 0x03, 0x5f, 0xd6}};
  block.original_sources = {0, 1};
  block.nodes = {{ir::Op::read, 64, {}, 0, 0},
                 {ir::Op::read, 64, {}, 0, 1},
                 {ir::Op::add, 64, {0, 1}},
                 {ir::Op::read, 64, {}, 0, 30}};
  block.boundaries = {{0, 3, {}, {}},
                      {3, 1, {}, ir::Transfer{ir::TransferKind::return_, 3, {}, {}, {}}}};
  block.phis = {{0, 64, true, {}}, {1, 64, true, {}}, {30, 64, true, {}}};
  block.clobbers = {0, 0, 0};
  block.edges = {{ir::SsaEdgeKind::return_, ir::SsaTargetKind::unknown, 0, {}, {}, {}, {}}};
  const auto entry = graph.Add(std::move(block));
  graph.Update(entry, [&](auto& inserted) {
    inserted.reads = {{0, 0, {ir::SsaValueKind::phi, entry, 0}},
                      {1, 1, {ir::SsaValueKind::phi, entry, 1}},
                      {3, 2, {ir::SsaValueKind::phi, entry, 2}}};
    inserted.exits = {{0, {ir::SsaValueKind::phi, entry, 0}},
                      {1, {ir::SsaValueKind::phi, entry, 1}},
                      {30, {ir::SsaValueKind::phi, entry, 2}}};
  });
  graph.SetEntries({entry});
  return graph;
}

TEST(SsaPureDce, RetiresOnlyDeadPureNodeAndJournalsRevision) {
  auto graph = Graph();
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  const auto proof = analysis::ProveSsaNodeLiveness(graph, budget);
  ASSERT_TRUE(proof.facts);
  const auto revision = graph.revision();
  auto proposed = ProposeDeadPureNodes(graph, *proof.facts, budget);
  ASSERT_EQ(proposed.reason, SsaPureDceRefusal::none);
  ASSERT_TRUE(proposed.provisional);
  ASSERT_EQ(proposed.journal.size(), 1U);
  EXPECT_EQ(proposed.journal[0].node, 2U);
  EXPECT_EQ(proposed.journal[0].from_revision, revision);
  EXPECT_EQ(proposed.journal[0].to_revision, proposed.provisional->revision());
  EXPECT_EQ(graph.revision(), revision);
  const auto entry = proposed.provisional->entries()[0];
  EXPECT_EQ(proposed.provisional->Get(entry)->dead_pure_nodes, (std::vector<ir::ValueId>{2}));
  EXPECT_EQ(ir::ValidateSsa(*proposed.provisional, budget), ir::SsaDecline::none);
  auto stale = ProposeDeadPureNodes(*proposed.provisional, *proof.facts, budget);
  EXPECT_EQ(stale.reason, SsaPureDceRefusal::stale_proof);
  EXPECT_TRUE(stale.journal.empty());
}

TEST(SsaPureDce, RefusesForgedProofThatDropsEffectOrLiveInput) {
  auto graph = Graph();
  auto budget = Plenty();
  auto proof = analysis::ProveSsaNodeLiveness(graph, budget);
  ASSERT_TRUE(proof.facts);
  auto forged = *proof.facts;
  forged.live_nodes[0][3] = 0;
  auto dropped_return = ProposeDeadPureNodes(graph, forged, budget);
  EXPECT_EQ(dropped_return.reason, SsaPureDceRefusal::stale_proof);
  EXPECT_TRUE(dropped_return.journal.empty());
  forged = *proof.facts;
  forged.live_nodes[0][0] = 0;
  auto dropped_read = ProposeDeadPureNodes(graph, forged, budget);
  EXPECT_EQ(dropped_read.reason, SsaPureDceRefusal::stale_proof);
  EXPECT_TRUE(dropped_read.journal.empty());
  ASSERT_TRUE(graph.Update(graph.entries()[0],
                           [](auto& block) { block.nodes[2] = {ir::Op::load, 64, {0}}; }));
  auto effect = analysis::ProveSsaNodeLiveness(graph, budget);
  ASSERT_TRUE(effect.facts);
  EXPECT_EQ(effect.facts->live_nodes[0][2], 1);
  auto no_edit = ProposeDeadPureNodes(graph, *effect.facts, budget);
  EXPECT_FALSE(no_edit.provisional);
  EXPECT_TRUE(no_edit.journal.empty());
}

TEST(SsaPureDce, EffectiveTransferRetiresOriginalExpression) {
  auto graph = Graph();
  auto budget = Plenty();
  ASSERT_TRUE(graph.Update(graph.entries()[0], [](auto& block) {
    block.transition = 0;
    block.boundaries[1].transfer->target = 2;
    ir::ConditionalRewrite rewrite{};
    rewrite.boundary = 1;
    rewrite.original = *block.boundaries[1].transfer;
    rewrite.replacement = {ir::TransferKind::return_, 3, {}, {}, {}};
    block.control_rewrites.push_back(std::move(rewrite));
  }));
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  auto proof = analysis::ProveSsaNodeLiveness(graph, budget);
  ASSERT_TRUE(proof.facts);
  EXPECT_EQ(proof.facts->live_nodes[0][2], 0);
  auto result = ProposeDeadPureNodes(graph, *proof.facts, budget);
  ASSERT_TRUE(result.provisional);
  EXPECT_EQ(result.journal.size(), 1U);
  EXPECT_EQ(ir::ValidateSsa(*result.provisional, budget), ir::SsaDecline::none);

  auto forged = Graph();
  ASSERT_TRUE(forged.Update(forged.entries()[0], [](auto& block) {
    block.boundaries[1].transfer->target = 2;
    block.dead_pure_nodes = {2};
  }));
  EXPECT_EQ(ir::ValidateSsa(forged, budget), ir::SsaDecline::invalid_graph);
}

TEST(SsaPureDce, RetiresUnusedProvedNonfaultingImageLoad) {
  auto graph = Graph();
  ASSERT_TRUE(graph.Update(graph.entries()[0], [](auto& block) {
    block.nodes[2] = {ir::Op::image_address, 64, {}, 0x2000};
    block.nodes.insert(block.nodes.begin() + 3, {ir::Op::load, 64, {2}});
    block.boundaries[0].node_count = 4;
    block.boundaries[1].first_node = 4;
    block.boundaries[1].transfer->target = 4;
    block.reads[2].node = 4;
  }));
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  constexpr std::array<std::uint8_t, 8> bytes{1, 2, 3, 4, 5, 6, 7, 8};
  const ir::ConstantImageRange range{0x2000, bytes};
  const ir::ImageFacts facts{std::span(&range, 1), {}, true, {}};
  auto folded = ProposeConstantImageLoads(graph, facts, {true, true, true}, budget);
  ASSERT_TRUE(folded.provisional);
  const auto proof = analysis::ProveSsaNodeLiveness(*folded.provisional, budget);
  ASSERT_TRUE(proof.facts);
  EXPECT_EQ(proof.facts->live_nodes[0][3], 0);
  auto retired = ProposeDeadPureNodes(*folded.provisional, *proof.facts, budget);
  ASSERT_TRUE(retired.provisional);
  ASSERT_EQ(retired.journal.size(), 2);
  const auto* block = retired.provisional->Get(retired.provisional->entries()[0]);
  EXPECT_EQ(block->dead_pure_nodes, (std::vector<ir::ValueId>{2, 3}));
  EXPECT_EQ(ir::ValidateSsa(*retired.provisional, budget), ir::SsaDecline::none);
  ASSERT_TRUE(retired.provisional->Update(retired.provisional->entries()[0], [](auto& wrong) {
    wrong.constant_loads[0].skip_access = false;
  }));
  EXPECT_EQ(ir::ValidateSsa(*retired.provisional, budget), ir::SsaDecline::invalid_graph);

  auto live = Graph();
  ASSERT_TRUE(live.Update(live.entries()[0], [](auto& block) {
    block.nodes[2] = {ir::Op::image_address, 64, {}, 0x2000};
    block.nodes.insert(block.nodes.begin() + 3, {ir::Op::load, 64, {2}});
    block.boundaries[0].node_count = 4;
    block.boundaries[0].writes.push_back({0, 3});
    block.boundaries[1].first_node = 4;
    block.boundaries[1].transfer->target = 4;
    block.reads[2].node = 4;
    block.exits[0].value = {ir::SsaValueKind::node, block.exits[0].value.block, 3};
  }));
  ASSERT_EQ(ir::ValidateSsa(live, budget), ir::SsaDecline::none);
  auto live_folded = ProposeConstantImageLoads(live, facts, {true, true, true}, budget);
  ASSERT_TRUE(live_folded.provisional);
  const auto live_proof = analysis::ProveSsaNodeLiveness(*live_folded.provisional, budget);
  ASSERT_TRUE(live_proof.facts);
  EXPECT_EQ(live_proof.facts->live_nodes[0][2], 0);
  EXPECT_EQ(live_proof.facts->live_nodes[0][3], 1);
  auto live_retired = ProposeDeadPureNodes(*live_folded.provisional, *live_proof.facts, budget);
  ASSERT_TRUE(live_retired.provisional);
  ASSERT_EQ(live_retired.journal.size(), 1);
  EXPECT_EQ(live_retired.journal[0].node, 2U);
  EXPECT_EQ(ir::ValidateSsa(*live_retired.provisional, budget), ir::SsaDecline::none);
}

TEST(SsaPureDce, BudgetCutsPublishNoPartialCandidate) {
  auto graph = Graph();
  auto proof_budget = Plenty();
  auto proof = analysis::ProveSsaNodeLiveness(graph, proof_budget);
  ASSERT_TRUE(proof.facts);
  auto full = Plenty();
  ASSERT_TRUE(ProposeDeadPureNodes(graph, *proof.facts, full).provisional);
  const auto used = full.used();
  ASSERT_LT(used.work, 10000U);
  for (std::uint64_t cut = 0; cut < used.work; ++cut) {
    Budget limited({cut, used.bytes});
    auto result = ProposeDeadPureNodes(graph, *proof.facts, limited);
    EXPECT_EQ(result.reason, SsaPureDceRefusal::resource_limit) << cut;
    EXPECT_FALSE(result.provisional) << cut;
    EXPECT_TRUE(result.journal.empty()) << cut;
  }
}
}  // namespace
}  // namespace nyx::recovery
