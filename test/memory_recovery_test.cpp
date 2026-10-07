#include <gtest/gtest.h>

#include "nyx/eval/block.hpp"
#include "nyx/eval/path.hpp"
#include "nyx/ir/print.hpp"
#include "nyx/recovery/mba.hpp"
#include "nyx/recovery/memory.hpp"

namespace nyx::recovery {
namespace {
using ir::Node;
using ir::Op;

Budget Plenty() { return Budget({100000000, 100000000}); }

ir::Block Block(std::vector<Node> nodes, std::vector<ir::Write> writes = {},
                std::optional<ir::Transfer> transfer = {}) {
  const std::vector<ir::Group> sources{
      ir::Group(0x1000, {1, 2, 3, 4}, std::move(nodes), std::move(writes),
                ir::MemoryModel::atomic_scalar_reference, transfer)};
  auto budget = Plenty();
  auto normalized = ir::Normalize(sources, budget);
  EXPECT_TRUE(normalized.block);
  return std::move(*normalized.block);
}

eval::State State() {
  auto budget = Plenty();
  eval::State state;
  for (const auto& [id, width, initial] :
       {std::tuple{UINT32_MAX, 64U, UINT64_C(0x4000)}, std::tuple{0U, 64U, UINT64_C(0x4000)},
        std::tuple{7U, 32U, UINT64_C(99)}, std::tuple{42U, 32U, UINT64_C(11)},
        std::tuple{30U, 64U, UINT64_C(0)}}) {
    auto value = BitVector::from_u64(width, initial, 64, budget);
    state.cells.push_back({id, std::move(*value)});
  }

  return state;
}

void Equivalent(const ir::Block& before, const ir::Block& after, bool readable = true) {
  auto budget = Plenty();
  const std::array<std::uint8_t, 4096> data{};
  const eval::RegionInput region[] = {{0x4000, data, readable, true}};
  auto memory_a = eval::Memory::Create(region, budget),
       memory_b = eval::Memory::Create(region, budget);
  auto a = State(), b = State();
  const auto original = eval::ExecuteBlock(before, a, *memory_a.memory, budget, {}, {0});
  const auto transformed = eval::ExecuteBlock(after, b, *memory_b.memory, budget, {}, {0});
  ASSERT_EQ(original.outcome, transformed.outcome);
  ASSERT_EQ(original.trace.size(), transformed.trace.size());
  for (std::size_t i = 0; i < a.cells.size(); ++i) EXPECT_EQ(a.cells[i].value, b.cells[i].value);
  EXPECT_EQ(memory_a.memory->Regions()[0].bytes, memory_b.memory->Regions()[0].bytes);
  for (std::size_t i = 0; i < original.trace.size(); ++i) {
    const auto& x = original.trace[i];
    const auto& y = transformed.trace[i];
    ASSERT_EQ(x.events.size(), y.events.size());
    for (std::size_t e = 0; e < x.events.size(); ++e) {
      EXPECT_EQ(x.events[e].operation, y.events[e].operation);
      EXPECT_EQ(x.events[e].address, y.events[e].address);
      EXPECT_EQ(x.events[e].bytes, y.events[e].bytes);
      EXPECT_EQ(x.events[e].write, y.events[e].write);
      EXPECT_EQ(x.events[e].completed, y.events[e].completed);
      EXPECT_EQ(x.events[e].conditional, y.events[e].conditional);
      EXPECT_EQ(x.events[e].performed, y.events[e].performed);
    }

    ASSERT_EQ(x.fault.has_value(), y.fault.has_value());
    if (x.fault) {
      EXPECT_EQ(x.fault->operation, y.fault->operation);
      EXPECT_EQ(x.fault->kind, y.fault->kind);
      EXPECT_EQ(x.fault->address, y.fault->address);
    }

    ASSERT_EQ(x.transfer.has_value(), y.transfer.has_value());
    if (x.transfer) {
      EXPECT_EQ(x.transfer->kind, y.transfer->kind);
      EXPECT_EQ(x.transfer->target, y.transfer->target);
      EXPECT_EQ(x.transfer->condition, y.transfer->condition);
      EXPECT_EQ(x.transfer->continuation, y.transfer->continuation);
    }
  }
}

TEST(MemoryRecovery, RedirectsSuccessfulUsersButRetainsLoadsStoresAndFaults) {
  const auto block = Block({{Op::read, 64, {}, 0, UINT32_MAX},
                            {Op::constant, 64, {}, 84},
                            {Op::add, 64, {0, 1}},
                            {Op::constant, 32, {}, 123},
                            {Op::store, 32, {2, 3}},
                            {Op::load, 32, {2}},
                            {Op::add, 32, {5, 3}},
                            {Op::write, 32, {5}, 0, 42}},
                           {{7, 5}});
  auto budget = Plenty();
  const auto result = ForwardMemoryValues(block, budget);
  ASSERT_TRUE(result.block);
  ASSERT_EQ(result.facts.size(), 1);
  EXPECT_EQ(result.facts[0].store, 4);
  EXPECT_EQ(result.facts[0].load, 5);
  EXPECT_EQ(result.facts[0].value, 3);
  EXPECT_EQ(result.facts[0].address.base, AddressBase::ssa_value);
  EXPECT_EQ(result.facts[0].address.offset, 84);
  EXPECT_EQ(result.facts[0].scope, MemoryProofScope::straightline_entry);
  ASSERT_EQ(result.journal.size(), 3);
  EXPECT_EQ(result.block->nodes()[5].op, Op::load);
  EXPECT_EQ(result.block->nodes()[4].op, Op::store);
  EXPECT_EQ(result.block->nodes()[6].inputs[0], 3);
  EXPECT_EQ(result.block->nodes()[7].inputs[0], 3);
  EXPECT_EQ(result.block->boundaries()[0].writes[0].value, 3);
  EXPECT_EQ(result.block->revision(), 1);
  EXPECT_EQ(block.nodes()[6].inputs[0], 5);
  Equivalent(block, *result.block);
  Equivalent(block, *result.block, false);
}

TEST(MemoryRecovery, SameBaseDisjointStoresRetainTheStateDefinition) {
  for (std::uint64_t other : {UINT64_C(2000), UINT64_C(1992)}) {
    const auto block = Block({{Op::read, 64, {}, 0, UINT32_MAX},
                              {Op::constant, 64, {}, 84},
                              {Op::add, 64, {0, 1}},
                              {Op::constant, 32, {}, 647},
                              {Op::store, 32, {2, 3}},
                              {Op::constant, 64, {}, other},
                              {Op::add, 64, {0, 5}},
                              {Op::constant, 64, {}, 999},
                              {Op::store, 64, {6, 7}},
                              {Op::load, 32, {2}}},
                             {{7, 9}});
    auto budget = Plenty();
    const auto result = ForwardMemoryValues(block, budget);
    ASSERT_TRUE(result.block);
    ASSERT_EQ(result.facts.size(), 1);
    EXPECT_EQ(result.facts[0].store, 4);
    Equivalent(block, *result.block);
  }
}

TEST(MemoryRecovery, UnknownBaseAndPartialOverlapInvalidateEarlierStores) {
  for (unsigned width : {8U, 16U, 32U, 64U}) {
    for (bool unknown : {false, true}) {
      const auto block = Block({{Op::read, 64, {}, 0, UINT32_MAX},
                                {Op::constant, 32, {}, 0x12345678},
                                {Op::store, 32, {0, 1}},
                                {Op::read, 64, {}, 0, 0},
                                {Op::constant, 64, {}, 1},
                                {Op::add, 64, {0, 4}},
                                {Op::constant, width, {}, 0},
                                {Op::store, width, {unknown ? 3U : 5U, 6}},
                                {Op::load, 32, {0}}},
                               {{7, 8}});
      auto budget = Plenty();
      const auto result = ForwardMemoryValues(block, budget);
      ASSERT_TRUE(result.block);
      EXPECT_TRUE(result.facts.empty());
      EXPECT_TRUE(result.journal.empty());
      EXPECT_EQ(result.block->revision(), 0);
      Equivalent(block, *result.block);
    }
  }
}

TEST(MemoryRecovery, ConditionalExclusiveWriteKillsEarlierStoreFact) {
  const std::vector<ir::Group> groups{
      {0x1000,
       {1, 2, 3, 4},
       {{Op::constant, 64, {}, 0x4000}, {Op::constant, 32, {}, 42}, {Op::store, 32, {0, 1}}},
       {},
       ir::MemoryModel::atomic_scalar_reference},
      {0x1004,
       {5, 6, 7, 8},
       {{Op::constant, 64, {}, 0x4000},
        {Op::exclusive_load, 32, {0}, 0, 0, {ir::ByteOrder::little, 4, true}}},
       {},
       ir::MemoryModel::qemu_exclusive_scalar_reference},
      {0x1008,
       {9, 10, 11, 12},
       {{Op::constant, 64, {}, 0x4000},
        {Op::constant, 32, {}, 55},
        {Op::exclusive_store, 32, {0, 1}, 0, 0, {ir::ByteOrder::little, 4, true}}},
       {},
       ir::MemoryModel::qemu_exclusive_scalar_reference},
      {0x100c,
       {13, 14, 15, 16},
       {{Op::constant, 64, {}, 0x4000}, {Op::load, 32, {0}}},
       {{7, 1}},
       ir::MemoryModel::atomic_scalar_reference}};
  auto budget = Plenty();
  const auto normalized = ir::Normalize(groups, budget);
  ASSERT_TRUE(normalized.block);
  const auto forwarded = ForwardMemoryValues(*normalized.block, budget);
  ASSERT_TRUE(forwarded.block);
  EXPECT_TRUE(forwarded.facts.empty());
  EXPECT_TRUE(forwarded.journal.empty());
  Equivalent(*normalized.block, *forwarded.block);
  auto state = State();
  const std::array<std::uint8_t, 4> bytes{};
  const eval::RegionInput region[] = {{0x4000, bytes}};
  auto memory = std::move(*eval::Memory::Create(region, budget).memory);
  ASSERT_EQ(eval::ExecuteBlock(*normalized.block, state, memory, budget).outcome,
            eval::Outcome::completed);
  EXPECT_EQ(state.cells[2].value.word(0), 55);
  EXPECT_EQ(memory.Regions()[0].bytes[0], 55);
}

TEST(MemoryRecovery, RequiresMatchingWidthAndByteOrder) {
  for (bool endian : {false, true}) {
    std::vector<Node> nodes{{Op::constant, 64, {}, 0x4000},
                            {Op::constant, 64, {}, 0x12345678},
                            {Op::store, 64, {0, 1}},
                            {Op::load, endian ? 64U : 32U, {0}}};
    if (endian) nodes[3].access.byte_order = ir::ByteOrder::big;
    const auto block = Block(std::move(nodes));
    auto budget = Plenty();
    const auto result = ForwardMemoryValues(block, budget);
    ASSERT_TRUE(result.block);
    EXPECT_TRUE(result.facts.empty());
    EXPECT_TRUE(result.journal.empty());
    Equivalent(block, *result.block);
  }
}

TEST(MemoryRecovery, CircularIntervalsDoNotMistakeWrappedOverlapForDisjointness) {
  for (bool overlap : {false, true}) {
    const auto block = Block({{Op::read, 64, {}, 0, UINT32_MAX},
                              {Op::constant, 64, {}, overlap ? UINT64_MAX - 1 : UINT64_MAX - 7},
                              {Op::add, 64, {0, 1}},
                              {Op::constant, 32, {}, 1},
                              {Op::store, 32, {2, 3}},
                              {Op::store, 32, {0, 3}},
                              {Op::load, 32, {2}}},
                             {{7, 6}});
    auto budget = Plenty();
    const auto result = ForwardMemoryValues(block, budget);
    ASSERT_TRUE(result.block);
    EXPECT_EQ(result.facts.size(), overlap ? 0 : 1);
  }
}

TEST(MemoryRecovery, ImageBiasAndAbsoluteAddressMayAliasRatherThanBeingDisjoint) {
  const auto block = Block({{Op::image_address, 64, {}, 0x4000},
                            {Op::constant, 64, {}, 0x4000},
                            {Op::constant, 32, {}, 1},
                            {Op::store, 32, {0, 2}},
                            {Op::constant, 32, {}, 2},
                            {Op::store, 32, {1, 4}},
                            {Op::load, 32, {0}}},
                           {{7, 6}});
  auto budget = Plenty();
  const auto result = ForwardMemoryValues(block, budget);
  ASSERT_TRUE(result.block);
  EXPECT_TRUE(result.facts.empty());
  Equivalent(block, *result.block);
}

TEST(MemoryRecovery, LatestExactStoreAndForwardedStoreValuesAreJournaled) {
  const auto block = Block({{Op::constant, 64, {}, 0x4000},
                            {Op::constant, 32, {}, 1},
                            {Op::store, 32, {0, 1}},
                            {Op::constant, 32, {}, 2},
                            {Op::store, 32, {0, 3}},
                            {Op::load, 32, {0}},
                            {Op::constant, 64, {}, 0x4008},
                            {Op::store, 32, {6, 5}},
                            {Op::load, 32, {6}}},
                           {{7, 8}});
  auto budget = Plenty();
  const auto result = ForwardMemoryValues(block, budget);
  ASSERT_TRUE(result.block);
  ASSERT_EQ(result.facts.size(), 2);
  EXPECT_EQ(result.facts[0].store, 4);
  EXPECT_EQ(result.facts[1].store, 7);
  EXPECT_EQ(result.facts[1].value, 3);
  ASSERT_EQ(result.journal.size(), 2);
  EXPECT_EQ(result.journal[0].use, MemoryUse::operand);
  EXPECT_EQ(result.journal[0].owner, 7);
  EXPECT_EQ(result.journal[1].use, MemoryUse::final_write);
  Equivalent(block, *result.block);
}

TEST(MemoryRecovery, RedirectsTerminalCallOperandsWithoutInferringCalleeEffects) {
  const auto block = Block({{Op::constant, 64, {}, 0x4000},
                            {Op::constant, 64, {}, 0x1234},
                            {Op::store, 64, {0, 1}},
                            {Op::load, 64, {0}}},
                           {{30, 3}}, ir::Transfer{ir::TransferKind::call, 3, {}, {}, 3});
  auto budget = Plenty();
  const auto result = ForwardMemoryValues(block, budget);
  ASSERT_TRUE(result.block);
  ASSERT_EQ(result.journal.size(), 3);
  EXPECT_EQ(result.block->boundaries()[0].transfer->target, 1);
  EXPECT_EQ(result.block->boundaries()[0].transfer->continuation, 1);
  EXPECT_EQ(result.block->nodes()[3].op, Op::load);
  Equivalent(block, *result.block);
}

TEST(MemoryRecovery, EveryResourceCutReturnsNoPartialFactsOrEdits) {
  const auto block = Block({{Op::constant, 64, {}, 0x4000},
                            {Op::constant, 32, {}, 123},
                            {Op::store, 32, {0, 1}},
                            {Op::load, 32, {0}}},
                           {{7, 3}});
  auto full = Plenty();
  ASSERT_TRUE(ForwardMemoryValues(block, full).block);
  for (bool bytes : {false, true}) {
    const auto bound = bytes ? full.used().bytes : full.used().work;
    for (std::uint64_t cut = 0; cut < bound; ++cut) {
      Budget budget({bytes ? full.used().work : cut, bytes ? cut : full.used().bytes});
      const auto result = ForwardMemoryValues(block, budget);
      ASSERT_FALSE(result.block) << cut;
      EXPECT_EQ(result.reason, MemoryRecoveryDecline::resource_limit);
      EXPECT_TRUE(result.facts.empty());
      EXPECT_TRUE(result.journal.empty());
      EXPECT_EQ(block.boundaries()[0].writes[0].value, 3);
    }
  }

  for (bool stores : {false, true}) {
    auto budget = Plenty();
    MemoryRecoveryLimits limits;
    if (stores)
      limits.max_stores = 0;
    else
      limits.max_edits = 0;
    const auto result = ForwardMemoryValues(block, budget, limits);
    EXPECT_FALSE(result.block);
    EXPECT_EQ(result.reason, MemoryRecoveryDecline::resource_limit);
  }
}

TEST(MemoryRecovery, PathFactsRequireTheSuccessfulItineraryAndKeepDivergenceEffects) {
  for (bool taken : {false, true}) {
    const std::vector<ir::Group> sources{
        ir::Group(0x1000, {1},
                  {{Op::constant, 64, {}, 0x4000},
                   {Op::constant, 32, {}, 42},
                   {Op::store, 32, {0, 1}},
                   {Op::constant, 1, {}, taken},
                   {Op::image_address, 64, {}, 0x2000},
                   {Op::image_address, 64, {}, 0x3000}},
                  {}, ir::MemoryModel::atomic_scalar_reference,
                  ir::Transfer{ir::TransferKind::conditional, 4, 3, 5, {}}),
        ir::Group(0x2000, {2}, {{Op::constant, 64, {}, 0x4000}, {Op::load, 32, {0}}}, {{7, 1}},
                  ir::MemoryModel::atomic_scalar_reference)};
    auto budget = Plenty();
    const auto original = ir::NormalizePath(sources, budget);
    ASSERT_TRUE(original.path);
    const auto result = ForwardMemoryValues(*original.path, budget);
    ASSERT_TRUE(result.path);
    ASSERT_EQ(result.facts.size(), 1);
    EXPECT_EQ(result.facts[0].scope, MemoryProofScope::successful_itinerary_prefix);
    EXPECT_EQ(result.path->nodes().back().op, Op::load);
    const std::array<std::uint8_t, 4096> bytes{};
    const eval::RegionInput region[] = {{0x4000, bytes}};
    auto first_memory = eval::Memory::Create(region, budget),
         second_memory = eval::Memory::Create(region, budget);
    auto first_state = State(), second_state = State();
    const auto first =
        eval::ExecutePath(*original.path, first_state, *first_memory.memory, budget, {}, {0});
    const auto second =
        eval::ExecutePath(*result.path, second_state, *second_memory.memory, budget, {}, {0});
    EXPECT_EQ(first.stop, taken ? eval::PathStop::completed : eval::PathStop::diverged);
    EXPECT_EQ(first.stop, second.stop);
    EXPECT_EQ(first.runtime_next, second.runtime_next);
    EXPECT_EQ(first.completed_boundaries, taken ? 2 : 1);
    EXPECT_EQ(first.completed_boundaries, second.completed_boundaries);
    ASSERT_EQ(first.trace.size(), taken ? 2 : 1);
    ASSERT_EQ(first.trace.size(), second.trace.size());
    for (std::size_t i = 0; i < first_state.cells.size(); ++i)
      EXPECT_EQ(first_state.cells[i].value, second_state.cells[i].value);
    EXPECT_EQ(first_memory.memory->Regions()[0].bytes, second_memory.memory->Regions()[0].bytes);
    EXPECT_EQ(first_memory.memory->Regions()[0].bytes[0], 42);
    EXPECT_EQ(second_state.cells[2].value.word(0), taken ? 42 : 99);
  }
}

TEST(MemoryRecovery, InvalidModelAndRevisionOverflowReturnNoArtifact) {
  const auto block = Block({{Op::constant, 64, {}, 0x4000}, {Op::load, 32, {0}}});
  for (bool revision : {false, true}) {
    std::vector<ir::Group> sources(block.sources().begin(), block.sources().end());
    if (!revision)
      sources[0] =
          ir::Group(0x1000, {1, 2, 3, 4}, {block.nodes().begin(), block.nodes().end()}, {});
    const ir::Block input(std::move(sources), {block.nodes().begin(), block.nodes().end()},
                          {block.origins().begin(), block.origins().end()},
                          {block.boundaries().begin(), block.boundaries().end()},
                          revision ? UINT64_MAX : 0);
    auto budget = Plenty();
    const auto result = ForwardMemoryValues(input, budget);
    EXPECT_FALSE(result.block);
    EXPECT_TRUE(result.facts.empty());
    EXPECT_TRUE(result.journal.empty());
    EXPECT_EQ(result.reason, revision ? MemoryRecoveryDecline::revision_overflow
                                      : MemoryRecoveryDecline::invalid_ir);
  }
}

TEST(MemoryRecovery, RepeatedMemoryAndArithmeticPassesKeepAStablePathAndRevisionChain) {
  const std::vector<ir::Group> groups{
      {0x1000,
       {1},
       {{Op::constant, 64, {}, 0x500},
        {Op::constant, 32, {}, 7},
        {Op::store, 32, {0, 1}},
        {Op::load, 32, {0}},
        {Op::image_address, 64, {}, 0x2000}},
       {{77, 3}},
       ir::MemoryModel::atomic_scalar_reference,
       ir::Transfer{ir::TransferKind::jump, 4, {}, {}, {}}},
      {0x2000,
       {2},
       {{Op::read, 32, {}, 0, 77}, {Op::constant, 32, {}, 5}, {Op::add, 32, {0, 1}}},
       {{99, 2}}}};
  auto budget = Plenty();
  const auto normalized = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(normalized.path);
  const auto forwarded = ForwardMemoryValues(*normalized.path, budget);
  ASSERT_TRUE(forwarded.path);
  ASSERT_FALSE(forwarded.journal.empty());
  EXPECT_EQ(forwarded.path->revision(), 1);
  const auto folded = SimplifyMba(*forwarded.path, budget);
  ASSERT_TRUE(folded.path);
  ASSERT_FALSE(folded.journal.empty());
  EXPECT_EQ(folded.path->revision(), 2);
  EXPECT_EQ(folded.path->nodes().back().op, Op::constant);
  EXPECT_EQ(folded.path->nodes().back().immediate, 12);
  const auto again = ForwardMemoryValues(*folded.path, budget);
  ASSERT_TRUE(again.path);
  EXPECT_TRUE(again.journal.empty());
  EXPECT_EQ(again.path->revision(), 2);
  const auto stable = SimplifyMba(*again.path, budget);
  ASSERT_TRUE(stable.path);
  EXPECT_TRUE(stable.journal.empty());
  EXPECT_EQ(stable.path->revision(), 2);
  const auto before = ir::PrintJson(*folded.path, budget);
  const auto after = ir::PrintJson(*stable.path, budget);
  ASSERT_TRUE(before.json);
  ASSERT_TRUE(after.json);
  EXPECT_EQ(before.json, after.json);
}

TEST(MemoryRecovery, APathCarriesMemoryFactsToACallTargetButNotPastACalleeBody) {
  auto budget = Plenty();

  // Continuing at the link address would skip an unmodelled callee, so no fact
  // may cross that boundary.
  const ir::Path elided(
      {ir::Group(0x1000, {1},
                 {{Op::image_address, 64, {}, 0x2000}, {Op::image_address, 64, {}, 0x1001}}, {},
                 ir::MemoryModel::unspecified, ir::Transfer{ir::TransferKind::call, 0, {}, {}, 1}),
       ir::Group(0x1001, {2}, {}, {})},
      {}, {}, {});
  const auto skipped = ForwardMemoryValues(elided, budget);
  EXPECT_FALSE(skipped.path);
  EXPECT_EQ(skipped.reason, MemoryRecoveryDecline::unsupported_control);
  EXPECT_TRUE(skipped.facts.empty());
  EXPECT_TRUE(skipped.journal.empty());

  // Continuing at the call's own target executes nothing in between, so the two
  // sources are one straight-line sequence exactly as across a jump.
  const std::vector<ir::Group> groups{
      ir::Group(0x1000, {1},
                {{Op::image_address, 64, {}, 0x2000}, {Op::image_address, 64, {}, 0x1001}}, {},
                ir::MemoryModel::unspecified, ir::Transfer{ir::TransferKind::call, 0, {}, {}, 1}),
      ir::Group(0x2000, {2}, {}, {})};
  const auto normalized = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(normalized.path);
  const auto carried = ForwardMemoryValues(*normalized.path, budget);
  EXPECT_EQ(carried.reason, MemoryRecoveryDecline::none);
  EXPECT_TRUE(carried.path);
}

}  // namespace
}  // namespace nyx::recovery
