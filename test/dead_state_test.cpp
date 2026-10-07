#include "nyx/recovery/dead_state.hpp"

#include <gtest/gtest.h>

#include "nyx/analysis/frame_slots.hpp"
#include "nyx/recovery/frame_promotion.hpp"

namespace nyx::recovery {
namespace {
Budget Plenty() { return Budget({1000000, 10000000}); }

ir::PrivateFrameContract Contract() { return {31, -32, 0, true, true, true, 16, false, false}; }

ir::SsaGraph Graph(bool use_load = false) {
  ir::SsaGraph graph;
  ir::SsaBlock block{};
  block.original_block = 0;
  block.address = 0x100;
  block.source_groups = {0x100};
  block.original_sources = {0};
  block.nodes = {{ir::Op::read, 64, {}, 0, 31}, {ir::Op::constant, 64, {}, 16},
                 {ir::Op::sub, 64, {0, 1}},     {ir::Op::constant, 64, {}, 42},
                 {ir::Op::store, 64, {2, 3}},   {ir::Op::load, 64, {2}}};
  block.nodes[4].access.alignment = 8;
  block.nodes[4].access.decline_on_unaligned = true;
  block.nodes[5].access.alignment = 8;
  block.nodes[5].access.decline_on_unaligned = true;
  block.boundaries = {
      {0, 6, use_load ? std::vector<ir::Write>{{0, 5}} : std::vector<ir::Write>{}, {}}};
  block.phis = use_load ? std::vector<ir::SsaPhi>{{0, 64, true, {}}, {31, 64, true, {}}}
                        : std::vector<ir::SsaPhi>{{31, 64, true, {}}};
  block.clobbers = use_load ? std::vector<std::uint8_t>{0, 0} : std::vector<std::uint8_t>{0};
  block.reads = {{0, use_load ? 1U : 0U, {}}};
  const auto handle = graph.Add(std::move(block));
  graph.Update(handle, [&](auto& inserted) {
    inserted.reads[0].value = {ir::SsaValueKind::phi, handle, use_load ? 1U : 0U};
    if (use_load) {
      inserted.exits = {{0, {ir::SsaValueKind::node, handle, 5}},
                        {31, {ir::SsaValueKind::phi, handle, 1}}};
    } else {
      inserted.exits = {{31, {ir::SsaValueKind::phi, handle, 0}}};
    }

    inserted.edges = {{ir::SsaEdgeKind::return_, ir::SsaTargetKind::unknown, 0, {}, {}, {}, {}}};
  });
  graph.SetEntries({handle});
  return graph;
}

TEST(DeadState, ProposesOnlyPrivateUnusedAccessesAndJournalsRevision) {
  auto original = Graph();
  auto budget = Plenty();
  const auto proof = analysis::ProvePrivateFrameSlots(original, Contract(), budget);
  ASSERT_TRUE(proof.facts);
  const auto before = original.revision();
  auto result = ProposeDeadPrivateState(original, *proof.facts, budget);
  ASSERT_TRUE(result.provisional) << static_cast<int>(result.reason);
  EXPECT_EQ(result.journal.size(), 2);
  EXPECT_EQ(original.revision(), before);
  EXPECT_TRUE(original.Get(original.entries()[0])->disabled_effects.empty());
  const auto* changed = result.provisional->Get(result.provisional->entries()[0]);
  ASSERT_NE(changed, nullptr);
  EXPECT_EQ(changed->disabled_effects, (std::vector<ir::ValueId>{4, 5}));
  EXPECT_NE(result.provisional->arena(), original.arena());
  ASSERT_TRUE(result.basis);
  EXPECT_EQ(result.basis->contract.begin, proof.facts->contract.begin);
  EXPECT_EQ(result.basis->contract.end, proof.facts->contract.end);
  EXPECT_EQ(result.basis->entry_relations, proof.facts->entry_relations);
  for (const auto& edit : result.journal) {
    EXPECT_EQ(edit.from_revision, before);
    EXPECT_EQ(edit.to_revision, result.provisional->revision());
  }

  EXPECT_EQ(ir::ValidateSsa(*result.provisional, budget), ir::SsaDecline::none);
  auto full = Plenty();
  ASSERT_TRUE(ProposeDeadPrivateState(original, *proof.facts, full).provisional);
  const auto used = full.used();
  ASSERT_LT(used.work, 10000U);
  for (std::uint64_t cut = 0; cut < used.work; ++cut) {
    Budget limited({cut, used.bytes});
    const auto refused = ProposeDeadPrivateState(original, *proof.facts, limited);
    EXPECT_EQ(refused.reason, DeadStateRefusal::resource_limit) << cut;
    EXPECT_FALSE(refused.provisional) << cut;
    EXPECT_TRUE(refused.journal.empty()) << cut;
  }
}

TEST(DeadState, RefusesLiveLoadAndStaleOrCorruptProof) {
  auto live = Graph(true);
  auto budget = Plenty();
  const auto proof = analysis::ProvePrivateFrameSlots(live, Contract(), budget);
  ASSERT_TRUE(proof.facts);
  auto used = ProposeDeadPrivateState(live, *proof.facts, budget);
  EXPECT_FALSE(used.provisional);
  ASSERT_EQ(used.refused.size(), 1);
  EXPECT_EQ(used.refused[0].reason, DeadStateRefusal::slot_used);
  EXPECT_TRUE(used.journal.empty());

  auto corrupt = *proof.facts;
  corrupt.slots[0].accesses[0].node = 3;
  auto bad = ProposeDeadPrivateState(live, corrupt, budget);
  EXPECT_FALSE(bad.provisional);
  EXPECT_EQ(bad.reason, DeadStateRefusal::stale_proof);
  EXPECT_TRUE(bad.refused.empty());

  ASSERT_TRUE(live.Update(live.entries()[0], [](auto&) {}));
  auto stale = ProposeDeadPrivateState(live, *proof.facts, budget);
  EXPECT_EQ(stale.reason, DeadStateRefusal::stale_proof);
  EXPECT_FALSE(stale.provisional);
}

TEST(DeadState, RefusesForgedSameRevisionPrivateAccess) {
  auto graph = Graph();
  auto budget = Plenty();
  auto proof = analysis::ProvePrivateFrameSlots(graph, Contract(), budget);
  ASSERT_TRUE(proof.facts);
  ASSERT_TRUE(graph.Update(graph.entries()[0], [](auto& block) {
    block.nodes[2] = {ir::Op::image_address, 64, {}, 0x3000};
  }));
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  auto forged = *proof.facts;
  forged.revision = graph.revision();
  auto result = ProposeDeadPrivateState(graph, forged, budget);
  EXPECT_EQ(result.reason, DeadStateRefusal::stale_proof);
  EXPECT_FALSE(result.provisional);
  EXPECT_TRUE(result.journal.empty());
}

TEST(DeadState, RefusesForgedSameRevisionCallTransfer) {
  auto graph = Graph();
  auto budget = Plenty();
  auto proof = analysis::ProvePrivateFrameSlots(graph, Contract(), budget);
  ASSERT_TRUE(proof.facts);
  ASSERT_TRUE(graph.Update(graph.entries()[0], [](auto& block) {
    block.boundaries[0].transfer = ir::Transfer{ir::TransferKind::call, 3, {}, {}, 3};
  }));
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  auto forged = *proof.facts;
  forged.revision = graph.revision();
  const auto result = ProposeDeadPrivateState(graph, forged, budget);
  EXPECT_EQ(result.reason, DeadStateRefusal::stale_proof);
  EXPECT_FALSE(result.provisional);
}

TEST(DeadState, KeepsLoadUsedByRecoveredTransfer) {
  auto graph = Graph();
  ASSERT_TRUE(graph.Update(graph.entries()[0], [](auto& block) {
    block.transition = 0;
    const ir::Transfer original{ir::TransferKind::jump, 3, {}, {}, {}};
    block.boundaries[0].transfer = original;
    ir::ConditionalRewrite rewrite{};
    rewrite.boundary = 0;
    rewrite.original = original;
    rewrite.replacement = {ir::TransferKind::jump, 5, {}, {}, {}};
    block.control_rewrites.push_back(rewrite);
  }));
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  const auto proof = analysis::ProvePrivateFrameSlots(graph, Contract(), budget);
  ASSERT_TRUE(proof.facts) << static_cast<int>(proof.reason);
  const auto result = ProposeDeadPrivateState(graph, *proof.facts, budget);
  EXPECT_FALSE(result.provisional);
  ASSERT_EQ(result.refused.size(), 1);
  EXPECT_EQ(result.refused[0].reason, DeadStateRefusal::slot_used);
}

TEST(DeadState, KeepsLoadFeedingPromotedStore) {
  auto graph = Graph(true);
  ASSERT_TRUE(graph.Update(graph.entries()[0], [](auto& block) {
    block.nodes.push_back({ir::Op::constant, 64, {}, 24});
    block.nodes.push_back({ir::Op::sub, 64, {0, 6}});
    block.nodes.push_back({ir::Op::store, 64, {7, 5}});
    block.nodes.push_back({ir::Op::load, 64, {7}});
    block.nodes[8].access.alignment = 8;
    block.nodes[9].access.alignment = 8;
    block.boundaries[0].node_count = 10;
    block.boundaries[0].writes[0].value = 9;
    block.exits[0].value.index = 9;
  }));
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  const auto proof = analysis::ProvePrivateFrameSlots(graph, Contract(), budget);
  ASSERT_TRUE(proof.facts) << static_cast<int>(proof.reason);
  auto promoted = ProposeFramePromotion(graph, *proof.facts, budget);
  ASSERT_TRUE(promoted.provisional) << static_cast<int>(promoted.reason);
  const auto renewed = analysis::ProvePrivateFrameSlots(*promoted.provisional, Contract(), budget);
  ASSERT_TRUE(renewed.facts) << static_cast<int>(renewed.reason);
  const auto result = ProposeDeadPrivateState(*promoted.provisional, *renewed.facts, budget);
  EXPECT_FALSE(result.provisional);
  ASSERT_EQ(result.refused.size(), 2);
  EXPECT_EQ(result.refused[0].reason, DeadStateRefusal::slot_used);
  EXPECT_EQ(result.refused[1].reason, DeadStateRefusal::slot_used);
}
}  // namespace
}  // namespace nyx::recovery
