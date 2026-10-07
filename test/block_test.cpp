#include "nyx/ir/block.hpp"

#include <limits>

#include <gtest/gtest.h>

namespace nyx::ir {
namespace {
Budget Plenty() { return Budget({UINT64_MAX, UINT64_MAX}); }

TEST(Block, ForwardsRegisterDefinitionsWithoutMergingSourceBoundaries) {
  const std::vector<Group> groups{
      {0x100,
       {1, 2},
       {{Op::read, 64, {}, 0, 77}, {Op::constant, 64, {}, 9}, {Op::add, 64, {0, 1}}},
       {{88, 2}}},
      {0x102,
       {3},
       {{Op::read, 64, {}, 0, 88}, {Op::read, 64, {}, 0, 77}, {Op::sub, 64, {0, 1}}},
       {{77, 2}}}};
  auto budget = Plenty();
  const auto result = Normalize(groups, budget);
  ASSERT_TRUE(result.block);
  EXPECT_EQ(result.block->nodes().size(), 4);
  ASSERT_EQ(result.block->boundaries().size(), 2);
  EXPECT_EQ(result.block->boundaries()[0].node_count, 3);
  EXPECT_EQ(result.block->boundaries()[1].first_node, 3);
  EXPECT_EQ(result.block->nodes()[3].inputs[0], 2);
  EXPECT_EQ(result.block->nodes()[3].inputs[1], 0);
  EXPECT_EQ(result.block->sources()[0].nodes().size(), 3);
  EXPECT_EQ(result.block->sources()[1].nodes().size(), 3);
  EXPECT_EQ(result.block->origins()[3].boundary, 1);
  EXPECT_EQ(result.block->origins()[3].operation, 2);
}

TEST(Block, OrderedWriteDoesNotChangeTheCurrentInstructionEntryRead) {
  const std::vector<Group> groups{
      {0,
       {1},
       {{Op::constant, 64, {}, 42}, {Op::write, 64, {0}, 0, 77}, {Op::read, 64, {}, 0, 77}},
       {{88, 2}}},
      {1,
       {2},
       {{Op::read, 64, {}, 0, 77}, {Op::read, 64, {}, 0, 88}, {Op::add, 64, {0, 1}}},
       {{99, 2}}}};
  auto budget = Plenty();
  const auto result = Normalize(groups, budget);
  ASSERT_TRUE(result.block);
  ASSERT_EQ(result.block->nodes().size(), 4);
  EXPECT_EQ(result.block->nodes()[2].op, Op::read);
  EXPECT_EQ(result.block->nodes()[3].inputs[0], 0);
  EXPECT_EQ(result.block->nodes()[3].inputs[1], 2);
}

TEST(Block, IdenticalLoadAddressesDoNotShareMemoryDefinitions) {
  const std::vector<Group> groups{{0,
                                   {1},
                                   {{Op::constant, 64, {}, 0x1000}, {Op::load, 64, {0}}},
                                   {{77, 1}},
                                   MemoryModel::atomic_scalar_reference},
                                  {1,
                                   {2},
                                   {{Op::constant, 64, {}, 0x1000}, {Op::load, 64, {0}}},
                                   {{88, 1}},
                                   MemoryModel::atomic_scalar_reference}};
  auto budget = Plenty();
  const auto result = Normalize(groups, budget);
  ASSERT_TRUE(result.block);
  EXPECT_EQ(result.block->nodes().size(), 4);
  EXPECT_EQ(result.block->boundaries()[0].writes[0].value, 1);
  EXPECT_EQ(result.block->boundaries()[1].writes[0].value, 3);
}

TEST(Block, RejectsNonlinearRegionsMalformedWidthsAndEffectReferences) {
  const std::vector<std::vector<Group>> invalid{
      {},
      {{0, {}, {}, {}}},
      {{0, {1}, {}, {}}, {2, {2}, {}, {}}},
      {{0,
        {1},
        {{Op::constant, 64}},
        {},
        MemoryModel::unspecified,
        Transfer{TransferKind::call, 0, {}, {}, 0}},
       {1, {2}, {}, {}}},
      {{0, {1}, {{Op::constant, 32}, {Op::constant, 64}, {Op::add, 64, {0, 1}}}, {}}},
      {{0, {1}, {{Op::constant, 64}, {Op::write, 64, {0}, 0, 77}, {Op::add, 64, {0, 1}}}, {}}},
      {{0, {1}, {{Op::read, 64, {}, 0, 77}}, {}}, {1, {2}, {{Op::read, 32, {}, 0, 77}}, {}}}};
  for (const auto& groups : invalid) {
    auto budget = Plenty();
    const auto result = Normalize(groups, budget);
    EXPECT_FALSE(result.block);
    EXPECT_NE(result.reason, BlockDecline::none);
  }
}

TEST(Block, TerminalTransferUsesGlobalDefinitionsAndRetainsCallBoundary) {
  const std::vector<Group> groups{{0, {1}, {{Op::constant, 64, {}, 42}}, {{77, 0}}},
                                  {1,
                                   {2},
                                   {{Op::read, 64, {}, 0, 77}, {Op::image_address, 64, {}, 2}},
                                   {{77, 1}},
                                   MemoryModel::unspecified,
                                   Transfer{TransferKind::call, 0, {}, {}, 1}}};
  auto budget = Plenty();
  const auto result = Normalize(groups, budget);
  ASSERT_TRUE(result.block);
  ASSERT_TRUE(result.block->boundaries()[1].transfer);
  EXPECT_EQ(result.block->boundaries()[1].transfer->target, 0);
  EXPECT_EQ(result.block->boundaries()[1].transfer->continuation, 1);
  EXPECT_EQ(result.block->boundaries()[1].writes[0].value, 1);
}

TEST(Block, RejectsDroppedEffectsAndCrossBoundaryForwardReferences) {
  const std::vector<Group> groups{
      {0, {1}, {{Op::constant, 64, {}, 42}, {Op::write, 64, {0}, 0, 77}}, {}},
      {1, {2}, {{Op::read, 64, {}, 0, 77}}, {{88, 0}}}};
  auto budget = Plenty();
  const auto normalized = Normalize(groups, budget);
  ASSERT_TRUE(normalized.block);
  const auto& original = *normalized.block;
  for (unsigned mutation = 0; mutation < 3; ++mutation) {
    auto nodes = std::vector<Node>(original.nodes().begin(), original.nodes().end());
    auto boundaries =
        std::vector<Boundary>(original.boundaries().begin(), original.boundaries().end());
    auto origins = std::vector<Origin>(original.origins().begin(), original.origins().end());
    if (mutation == 0) nodes[1].op = Op::bit_not;
    if (mutation == 1) boundaries[0].first_node = 1;
    if (mutation == 2) boundaries[1].writes[0].value = 9;
    Block changed(groups, std::move(nodes), std::move(origins), std::move(boundaries));
    EXPECT_EQ(Validate(changed, budget), BlockDecline::invalid_ir);
  }
}

TEST(Block, SparseStorageSlotsPreserveEntryReadsAndBoundActualDistinctCells) {
  const std::vector<Group> groups{
      {0,
       {1},
       {{Op::constant, 64, {}, 42},
        {Op::write, 64, {0}, 0, UINT32_MAX},
        {Op::read, 64, {}, 0, UINT32_MAX}},
       {{0, 2}}},
      {1,
       {2},
       {{Op::read, 64, {}, 0, UINT32_MAX}, {Op::read, 64, {}, 0, 0}, {Op::add, 64, {0, 1}}},
       {{UINT32_MAX, 2}}}};
  BlockLimits exact_limits;
  exact_limits.max_storage = 2;
  Budget exact({10000, 10000});
  const auto result = Normalize(groups, exact, exact_limits);
  ASSERT_TRUE(result.block);
  ASSERT_EQ(result.block->nodes().size(), 4);
  EXPECT_EQ(result.block->nodes()[2].op, Op::read);
  EXPECT_EQ(result.block->nodes()[2].storage, UINT32_MAX);
  EXPECT_EQ(result.block->nodes()[3].inputs[0], 0);
  EXPECT_EQ(result.block->nodes()[3].inputs[1], 2);
  EXPECT_EQ(result.block->boundaries()[0].writes[0].storage, 0);
  BlockLimits huge_limits;
  huge_limits.max_storage = UINT32_MAX;
  Budget huge({10000, 10000});
  const auto unchanged = Normalize(groups, huge, huge_limits);
  ASSERT_TRUE(unchanged.block);
  EXPECT_EQ(huge.used().work, exact.used().work);
  EXPECT_EQ(unchanged.block->nodes().size(), result.block->nodes().size());
  EXPECT_EQ(unchanged.block->nodes()[3].inputs, result.block->nodes()[3].inputs);
  BlockLimits short_limits;
  short_limits.max_storage = 1;
  auto short_budget = Plenty();
  const auto declined = Normalize(groups, short_budget, short_limits);
  EXPECT_FALSE(declined.block);
  EXPECT_EQ(declined.reason, BlockDecline::resource_limit);
}

TEST(Block, SparseSlotPrepassRejectsInconsistentCellWidths) {
  const std::vector<Group> groups{
      {0, {1}, {{Op::constant, 64}, {Op::write, 64, {0}, 0, UINT32_MAX}}, {}},
      {1, {2}, {{Op::read, 32, {}, 0, UINT32_MAX}}, {}}};
  auto budget = Plenty();
  const auto result = Normalize(groups, budget);
  EXPECT_FALSE(result.block);
  EXPECT_EQ(result.reason, BlockDecline::invalid_ir);
  BlockLimits zero;
  zero.max_storage = 0;
  const std::vector<Group> pure{{0, {1}, {{Op::constant, 64}}, {}}};
  EXPECT_TRUE(Normalize(pure, budget, zero).block);
}

TEST(Block, NormalizationPublishesNothingAtEveryResourceCut) {
  const std::vector<Group> groups{
      {0,
       {1},
       {{Op::read, 64, {}, 0, 77}, {Op::constant, 64, {}, 3}, {Op::add, 64, {0, 1}}},
       {{77, 2}}},
      {1, {2}, {{Op::read, 64, {}, 0, 77}}, {{88, 0}}}};
  auto full = Plenty();
  ASSERT_TRUE(Normalize(groups, full).block);
  for (bool bytes : {false, true}) {
    const auto maximum = bytes ? full.used().bytes : full.used().work;
    for (std::uint64_t cut = 0; cut < maximum; ++cut) {
      Budget budget({bytes ? full.used().work : cut, bytes ? cut : full.used().bytes});
      const auto result = Normalize(groups, budget);
      ASSERT_FALSE(result.block) << cut;
      EXPECT_EQ(result.reason, BlockDecline::resource_limit);
    }
  }
}
}  // namespace
}  // namespace nyx::ir
