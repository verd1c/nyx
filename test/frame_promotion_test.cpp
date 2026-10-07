#include "nyx/recovery/frame_promotion.hpp"

#include <gtest/gtest.h>

#include "nyx/analysis/cfg.hpp"
#include "nyx/analysis/frame_slots.hpp"
#include "nyx/analysis/regions.hpp"
#include "nyx/analysis/ssa/build.hpp"
#include "nyx/analysis/unflatten.hpp"

namespace nyx::recovery {
namespace {
analysis::SourceRecord Record(ir::Group group) {
  return {group.source_address(),
          {group.bytes().begin(), group.bytes().end()},
          std::move(group),
          analysis::OpaqueReason::none};
}

std::optional<ir::SsaGraph> Graph(Budget& budget) {
  std::vector<analysis::SourceRecord> sources{
      Record(ir::Group(0x100, {1, 2, 3, 4},
                       {{ir::Op::read, 1, {}, 0, 1},
                        {ir::Op::image_address, 64, {}, 0x104},
                        {ir::Op::image_address, 64, {}, 0x108}},
                       {}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::conditional, 1, 0, 2, {}})),
      Record(ir::Group(0x104, {5, 6, 7, 8},
                       {{ir::Op::read, 64, {}, 0, 31},
                        {ir::Op::constant, 64, {}, 16},
                        {ir::Op::sub, 64, {0, 1}},
                        {ir::Op::constant, 64, {}, 11},
                        {ir::Op::store, 64, {2, 3}},
                        {ir::Op::image_address, 64, {}, 0x10c}},
                       {}, ir::MemoryModel::atomic_scalar_reference,
                       ir::Transfer{ir::TransferKind::jump, 5, {}, {}, {}})),
      Record(ir::Group(0x108, {9, 10, 11, 12},
                       {{ir::Op::read, 64, {}, 0, 31},
                        {ir::Op::constant, 64, {}, 16},
                        {ir::Op::sub, 64, {0, 1}},
                        {ir::Op::constant, 64, {}, 22},
                        {ir::Op::store, 64, {2, 3}},
                        {ir::Op::image_address, 64, {}, 0x10c}},
                       {}, ir::MemoryModel::atomic_scalar_reference,
                       ir::Transfer{ir::TransferKind::jump, 5, {}, {}, {}})),
      Record(ir::Group(0x10c, {13, 14, 15, 16},
                       {{ir::Op::read, 64, {}, 0, 31},
                        {ir::Op::constant, 64, {}, 16},
                        {ir::Op::sub, 64, {0, 1}},
                        {ir::Op::load, 64, {2}},
                        {ir::Op::read, 64, {}, 0, 30}},
                       {{0, 3}}, ir::MemoryModel::atomic_scalar_reference,
                       ir::Transfer{ir::TransferKind::return_, 4, {}, {}, {}}))};
  const std::uint64_t entries[] = {0x100};
  auto cfg = analysis::BuildCfg(sources, entries, budget);
  if (!cfg.cfg) return {};
  analysis::Regions regions(std::move(*cfg.cfg), {});
  auto flattened = analysis::Unflatten(regions, {}, budget);
  if (!flattened.unflattening) return {};
  const std::vector<std::optional<ir::RecoveredPath>> paths;
  auto built = analysis::BuildSsa(regions, *flattened.unflattening, paths, budget);
  return std::move(built.graph);
}

ir::SsaHandle ByAddress(const ir::SsaGraph& graph, std::uint64_t address) {
  for (std::size_t i = 0; i < graph.slots(); ++i) {
    const auto handle = graph.Handle(i);
    if (handle && graph.Get(*handle)->address == address) return *handle;
  }

  return {};
}

TEST(FramePromotion, MakesJoinPhiForPrivateSlot) {
  Budget budget({1000000, 10000000});
  auto graph = Graph(budget);
  ASSERT_TRUE(graph);
  const ir::PrivateFrameContract contract{31, -32, 0, true, true, true, 16, false, false};
  auto proof = analysis::ProvePrivateFrameSlots(*graph, contract, budget);
  ASSERT_TRUE(proof.facts) << static_cast<int>(proof.reason);
  auto promoted = ProposeFramePromotion(*graph, *proof.facts, budget);
  ASSERT_TRUE(promoted.provisional) << static_cast<int>(promoted.reason);
  ASSERT_EQ(promoted.journal.size(), 3);
  ASSERT_TRUE(promoted.basis);
  EXPECT_EQ(promoted.basis->contract.begin, -32);
  EXPECT_EQ(promoted.basis->contract.sp_alignment, 16U);
  auto& after = *promoted.provisional;
  const auto join_handle = ByAddress(after, 0x10c);
  const auto* join = after.Get(join_handle);
  ASSERT_NE(join, nullptr);
  ASSERT_EQ(join->frame_phis.size(), 1);
  ASSERT_EQ(join->frame_phis[0].incoming.size(), 2);
  EXPECT_EQ(join->frame_phis[0].incoming[0].value.kind, ir::SsaValueKind::node);
  EXPECT_EQ(join->frame_phis[0].incoming[1].value.kind, ir::SsaValueKind::node);
  ASSERT_EQ(join->frame_accesses.size(), 1);
  ASSERT_TRUE(join->frame_accesses[0].replacement);
  EXPECT_EQ(join->frame_accesses[0].replacement->kind, ir::SsaValueKind::frame_phi);
  EXPECT_EQ(join->disabled_effects, (std::vector<ir::ValueId>{3}));
  EXPECT_TRUE(graph->Get(ByAddress(*graph, 0x10c))->frame_phis.empty());
  EXPECT_EQ(ir::ValidateSsa(after, budget), ir::SsaDecline::none);

  Budget full({1000000, 10000000});
  ASSERT_TRUE(ProposeFramePromotion(*graph, *proof.facts, full).provisional);
  const auto used = full.used();
  ASSERT_LT(used.work, 10000U);
  for (std::uint64_t cut = 0; cut < used.work; ++cut) {
    Budget limited({cut, used.bytes});
    const auto refused = ProposeFramePromotion(*graph, *proof.facts, limited);
    EXPECT_EQ(refused.reason, PromotionRefusal::resource_limit) << cut;
    EXPECT_FALSE(refused.provisional) << cut;
    EXPECT_TRUE(refused.journal.empty()) << cut;
  }

  ASSERT_TRUE(after.Update(join_handle, [](auto& wrong) {
    wrong.frame_phis[0].incoming[0].value = wrong.frame_phis[0].incoming[1].value;
  }));
  EXPECT_EQ(ir::ValidateSsa(after, budget), ir::SsaDecline::invalid_graph);
}

TEST(FramePromotion, RefusesStaleFacts) {
  Budget budget({1000000, 10000000});
  auto graph = Graph(budget);
  ASSERT_TRUE(graph);
  const ir::PrivateFrameContract contract{31, -32, 0, true, true, true, 16, false, false};
  auto proof = analysis::ProvePrivateFrameSlots(*graph, contract, budget);
  ASSERT_TRUE(proof.facts);
  ASSERT_TRUE(graph->Update(graph->entries()[0], [](auto&) {}));
  auto result = ProposeFramePromotion(*graph, *proof.facts, budget);
  EXPECT_EQ(result.reason, PromotionRefusal::stale_proof);
  EXPECT_FALSE(result.provisional);
}

TEST(FramePromotion, RefusesForgedSameRevisionAccess) {
  Budget budget({1000000, 10000000});
  auto graph = Graph(budget);
  ASSERT_TRUE(graph);
  const ir::PrivateFrameContract contract{31, -32, 0, true, true, true, 16, false, false};
  auto proof = analysis::ProvePrivateFrameSlots(*graph, contract, budget);
  ASSERT_TRUE(proof.facts);
  ASSERT_TRUE(graph->Update(ByAddress(*graph, 0x104), [](auto& block) {
    block.nodes[2] = {ir::Op::image_address, 64, {}, 0x3000};
  }));
  ASSERT_EQ(ir::ValidateSsa(*graph, budget), ir::SsaDecline::none);
  auto forged = *proof.facts;
  forged.revision = graph->revision();
  auto result = ProposeFramePromotion(*graph, forged, budget);
  EXPECT_EQ(result.reason, PromotionRefusal::stale_proof);
  EXPECT_FALSE(result.provisional);
  EXPECT_TRUE(result.journal.empty());
}

TEST(FramePromotion, RefusesMismatchedByteOrders) {
  Budget budget({1000000, 10000000});
  auto graph = Graph(budget);
  ASSERT_TRUE(graph);
  ASSERT_TRUE(graph->Update(ByAddress(*graph, 0x108), [](auto& branch) {
    branch.nodes[4].access.byte_order = ir::ByteOrder::big;
  }));
  const ir::PrivateFrameContract contract{31, -32, 0, true, true, true, 16, false, false};
  auto proof = analysis::ProvePrivateFrameSlots(*graph, contract, budget);
  ASSERT_TRUE(proof.facts);
  auto result = ProposeFramePromotion(*graph, *proof.facts, budget);
  EXPECT_FALSE(result.provisional);
  ASSERT_EQ(result.refused.size(), 1);
  EXPECT_EQ(result.refused[0].reason, PromotionRefusal::incompatible_access);
}
}  // namespace
}  // namespace nyx::recovery
