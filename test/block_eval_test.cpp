#include <limits>

#include <gtest/gtest.h>

#include "nyx/eval/block.hpp"

namespace nyx::eval {
namespace {

Budget Plenty() {
  return Budget(
      {std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max()});
}

State Initial() {
  auto budget = Plenty();
  State state;
  for (const auto& [id, value] : {std::pair{100U, 7U}, {200U, 11U}}) {
    auto bits = BitVector::from_u64(64, value, 64, budget);
    state.cells.push_back({id, std::move(*bits)});
  }

  return state;
}

Memory DataMemory() {
  auto budget = Plenty();
  const std::array<std::uint8_t, 8> bytes{};
  const RegionInput regions[] = {{0x4000, bytes}};
  return std::move(*Memory::Create(regions, budget).memory);
}

TEST(BlockEval, CrossBoundaryValuesMatchInstructionSnapshots) {
  const std::vector<ir::Group> sources{
      {0x100,
       {1, 2},
       {{ir::Op::read, 64, {}, 0, 100}, {ir::Op::read, 64, {}, 0, 200}},
       {{100, 1}, {200, 0}}},
      {0x102,
       {3},
       {{ir::Op::read, 64, {}, 0, 100}, {ir::Op::read, 64, {}, 0, 200}, {ir::Op::add, 64, {0, 1}}},
       {{100, 2}}},
      {0x103, {4, 5, 6}, {{ir::Op::read, 64, {}, 0, 100}}, {{200, 0}}}};
  auto budget = Plenty();
  const auto normalized = ir::Normalize(sources, budget);
  ASSERT_TRUE(normalized.block);
  EXPECT_EQ(normalized.block->boundaries()[2].node_count, 0);
  auto state = Initial();
  auto reference = Initial();
  auto memory = DataMemory();
  for (const auto& source : sources)
    ASSERT_EQ(Execute(source, reference, budget), Outcome::completed);
  const auto result = ExecuteBlock(*normalized.block, state, memory, budget);
  EXPECT_EQ(result.outcome, Outcome::completed);
  EXPECT_EQ(result.completed_boundaries, 3);
  EXPECT_EQ(result.source_address, 0x103);
  ASSERT_EQ(result.trace.size(), 3);
  for (std::size_t i = 0; i < state.cells.size(); ++i)
    EXPECT_EQ(state.cells[i].value, reference.cells[i].value);
  EXPECT_EQ(state.cells[0].value.word(0), 18);
  EXPECT_EQ(state.cells[1].value.word(0), 18);
}

TEST(BlockEval, ExplicitWritesKeepCurrentEntryReadsAndNextBoundaryPublication) {
  const std::vector<ir::Group> sources{
      {0x100,
       {1},
       {{ir::Op::constant, 64, {}, 42},
        {ir::Op::write, 64, {0}, 0, 100},
        {ir::Op::read, 64, {}, 0, 100}},
       {{200, 2}}},
      {0x101,
       {2},
       {{ir::Op::read, 64, {}, 0, 100}, {ir::Op::read, 64, {}, 0, 200}, {ir::Op::add, 64, {0, 1}}},
       {{100, 2}}}};
  auto budget = Plenty();
  auto normalized = ir::Normalize(sources, budget);
  ASSERT_TRUE(normalized.block);
  auto state = Initial();
  auto memory = DataMemory();
  const auto result = ExecuteBlock(*normalized.block, state, memory, budget);
  EXPECT_EQ(result.outcome, Outcome::completed);
  EXPECT_EQ(state.cells[0].value.word(0), 49);
  EXPECT_EQ(state.cells[1].value.word(0), 7);
}

std::vector<ir::Group> MemorySources(bool fault) {
  return {
      {0x100, {1}, {{ir::Op::constant, 64, {}, 99}, {ir::Op::constant, 64, {}, 42}}, {{100, 1}}},
      {0x101,
       {2, 3},
       {{ir::Op::read, 64, {}, 0, 100},
        {ir::Op::constant, 64, {}, 0x4000},
        {ir::Op::store, 64, {1, 0}},
        {ir::Op::write, 64, {0}, 0, 200},
        {ir::Op::constant, 64, {}, fault ? 0x5000U : 0x4000U},
        {ir::Op::load, 64, {4}}},
       {{100, 5}},
       ir::MemoryModel::atomic_scalar_reference}};
}

TEST(BlockEval, FaultRetainsInstructionPrefixAndReportsSourceLocalOperations) {
  const auto sources = MemorySources(true);
  auto budget = Plenty();
  const auto normalized = ir::Normalize(sources, budget);
  ASSERT_TRUE(normalized.block);
  ASSERT_NE(normalized.block->origins().back().operation, normalized.block->nodes().size() - 1);
  auto state = Initial();
  auto memory = DataMemory();
  const auto result = ExecuteBlock(*normalized.block, state, memory, budget);
  EXPECT_EQ(result.outcome, Outcome::fault);
  EXPECT_EQ(result.completed_boundaries, 1);
  EXPECT_EQ(result.source_address, 0x101);
  ASSERT_EQ(result.trace.size(), 2);
  const auto& step = result.trace.back();
  ASSERT_TRUE(step.fault);
  EXPECT_EQ(step.fault->operation, 5);
  EXPECT_EQ(step.fault->address, 0x5000);
  EXPECT_FALSE(step.transfer);
  ASSERT_EQ(step.events.size(), 2);
  EXPECT_EQ(step.events[0].operation, 2);
  EXPECT_TRUE(step.events[0].completed);
  EXPECT_EQ(step.events[1].operation, 5);
  EXPECT_FALSE(step.events[1].completed);
  EXPECT_EQ(state.cells[0].value.word(0), 42);
  EXPECT_EQ(state.cells[1].value.word(0), 42);
  EXPECT_EQ(memory.Regions()[0].bytes[0], 42);
}

TEST(BlockEval, LoadsObserveEarlierBoundaryStoresWithoutReusingStaleMemoryValues) {
  const std::vector<ir::Group> sources{
      {0x100,
       {1},
       {{ir::Op::constant, 64, {}, 0x4000}, {ir::Op::load, 64, {0}}},
       {{100, 1}},
       ir::MemoryModel::atomic_scalar_reference},
      {0x101,
       {2},
       {{ir::Op::constant, 64, {}, 0x4000},
        {ir::Op::constant, 64, {}, 73},
        {ir::Op::store, 64, {0, 1}}},
       {},
       ir::MemoryModel::atomic_scalar_reference},
      {0x102,
       {3},
       {{ir::Op::constant, 64, {}, 0x4000}, {ir::Op::load, 64, {0}}},
       {{200, 1}},
       ir::MemoryModel::atomic_scalar_reference}};
  auto budget = Plenty();
  const auto normalized = ir::Normalize(sources, budget);
  ASSERT_TRUE(normalized.block);
  auto state = Initial();
  auto memory = DataMemory();
  const auto result = ExecuteBlock(*normalized.block, state, memory, budget);
  ASSERT_EQ(result.outcome, Outcome::completed);
  ASSERT_EQ(result.trace.size(), 3);
  EXPECT_EQ(state.cells[0].value.word(0), 0);
  EXPECT_EQ(state.cells[1].value.word(0), 73);
  for (const auto& step : result.trace) ASSERT_EQ(step.events.size(), 1);
  EXPECT_EQ(result.trace[0].events[0].bytes[0], 0);
  EXPECT_EQ(result.trace[2].events[0].bytes[0], 73);
}

TEST(BlockEval, MissingPlacementPreservesPriorBoundaryButNotCurrentWrites) {
  const std::vector<ir::Group> sources{{0x100, {1}, {{ir::Op::constant, 64, {}, 42}}, {{100, 0}}},
                                       {0x101,
                                        {2},
                                        {{ir::Op::constant, 64, {}, 99},
                                         {ir::Op::write, 64, {0}, 0, 200},
                                         {ir::Op::image_address, 64, {}, 0x123}},
                                        {{100, 2}}}};
  auto budget = Plenty();
  const auto normalized = ir::Normalize(sources, budget);
  ASSERT_TRUE(normalized.block);
  auto state = Initial();
  auto memory = DataMemory();
  const auto result = ExecuteBlock(*normalized.block, state, memory, budget);
  EXPECT_EQ(result.outcome, Outcome::unsupported);
  EXPECT_EQ(result.completed_boundaries, 1);
  EXPECT_EQ(result.source_address, 0x101);
  ASSERT_EQ(result.trace.size(), 2);
  EXPECT_TRUE(result.trace.back().events.empty());
  EXPECT_FALSE(result.trace.back().transfer);
  EXPECT_EQ(state.cells[0].value.word(0), 42);
  EXPECT_EQ(state.cells[1].value.word(0), 11);
}

TEST(BlockEval, TerminalTransferUsesCrossBoundaryValues) {
  const std::vector<ir::Group> sources{
      {0x100, {1}, {{ir::Op::image_address, 64, {}, 0x800}}, {{100, 0}}},
      {0x101,
       {2},
       {{ir::Op::read, 64, {}, 0, 100}, {ir::Op::image_address, 64, {}, 0x102}},
       {{200, 1}},
       ir::MemoryModel::unspecified,
       ir::Transfer{ir::TransferKind::call, 0, {}, {}, 1}}};
  auto budget = Plenty();
  const auto normalized = ir::Normalize(sources, budget);
  ASSERT_TRUE(normalized.block);
  auto state = Initial();
  auto memory = DataMemory();
  const auto result = ExecuteBlock(*normalized.block, state, memory, budget, {}, {0x1000});
  ASSERT_EQ(result.outcome, Outcome::completed);
  EXPECT_EQ(result.completed_boundaries, 2);
  ASSERT_EQ(result.trace.size(), 2);
  ASSERT_TRUE(result.trace[1].transfer);
  EXPECT_EQ(result.trace[1].transfer->kind, ir::TransferKind::call);
  EXPECT_EQ(result.trace[1].transfer->target, 0x1800);
  EXPECT_EQ(result.trace[1].transfer->continuation, 0x1102);
  EXPECT_EQ(state.cells[1].value.word(0), 0x1102);
}

TEST(BlockEval, InvalidLaterIrIsRejectedBeforeAnyPublication) {
  const auto sources = MemorySources(false);
  auto budget = Plenty();
  const auto normalized = ir::Normalize(sources, budget);
  ASSERT_TRUE(normalized.block);
  const auto& valid = *normalized.block;
  std::vector<ir::Node> nodes(valid.nodes().begin(), valid.nodes().end());
  nodes.back().inputs[0] = static_cast<ir::ValueId>(nodes.size());
  const ir::Block malformed(sources, std::move(nodes),
                            {valid.origins().begin(), valid.origins().end()},
                            {valid.boundaries().begin(), valid.boundaries().end()});
  auto state = Initial();
  auto memory = DataMemory();
  const auto result = ExecuteBlock(malformed, state, memory, budget);
  EXPECT_EQ(result.outcome, Outcome::invalid_group);
  EXPECT_EQ(result.completed_boundaries, 0);
  EXPECT_TRUE(result.trace.empty());
  EXPECT_EQ(state.cells[0].value.word(0), 7);
  EXPECT_EQ(state.cells[1].value.word(0), 11);
  EXPECT_EQ(memory.Regions()[0].bytes[0], 0);
}

TEST(BlockEval, EveryResourceCutRetainsExactlyTheCompletedPrefix) {
  const auto sources = MemorySources(false);
  auto preparation = Plenty();
  const auto normalized = ir::Normalize(sources, preparation);
  ASSERT_TRUE(normalized.block);
  auto full_state = Initial();
  auto full_memory = DataMemory();
  auto full_budget = Plenty();
  Limits limits;
  limits.max_width = 64;
  ASSERT_EQ(ExecuteBlock(*normalized.block, full_state, full_memory, full_budget, limits).outcome,
            Outcome::completed);
  const auto used = full_budget.used();
  bool saw_prefix = false;
  for (bool cut_bytes : {false, true}) {
    const auto total = cut_bytes ? used.bytes : used.work;
    for (std::uint64_t cutoff = 0; cutoff < total; ++cutoff) {
      SCOPED_TRACE(cutoff);
      auto state = Initial();
      auto memory = DataMemory();
      Budget budget({cut_bytes ? used.work : cutoff, cut_bytes ? cutoff : used.bytes});
      const auto result = ExecuteBlock(*normalized.block, state, memory, budget, limits);
      ASSERT_EQ(result.outcome, Outcome::resource_limit);
      ASSERT_LE(result.completed_boundaries, 1);
      saw_prefix |= result.completed_boundaries == 1;
      EXPECT_EQ(state.cells[0].value.word(0), result.completed_boundaries ? 42 : 7);
      EXPECT_EQ(state.cells[1].value.word(0), 11);
      EXPECT_EQ(memory.Regions()[0].bytes[0], 0);
      if (!result.trace.empty()) {
        EXPECT_EQ(result.trace.back().outcome, Outcome::resource_limit);
        EXPECT_TRUE(result.trace.back().events.empty());
        EXPECT_FALSE(result.trace.back().fault);
        EXPECT_FALSE(result.trace.back().transfer);
        EXPECT_EQ(result.trace.size(), result.completed_boundaries + 1);
      }
    }
  }

  EXPECT_TRUE(saw_prefix);
}

}  // namespace
}  // namespace nyx::eval
