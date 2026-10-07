#include <limits>

#include <gtest/gtest.h>

#include "nyx/eval/concrete.hpp"

namespace nyx::eval {
namespace {

Budget Plenty() {
  return Budget(
      {std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max()});
}

State Initial(unsigned width = 64) {
  auto budget = Plenty();
  State state;
  auto first = BitVector::from_u64(width, 7, 4096, budget);
  auto second = BitVector::from_u64(width, 11, 4096, budget);
  state.cells.push_back({100, std::move(*first)});
  state.cells.push_back({200, std::move(*second)});
  return state;
}

TEST(Eval, WritesCommitTogetherFromSnapshot) {
  // The synthetic target has arbitrary storage IDs and source length; swapping
  // cells catches accidentally sequential register effects.
  const ir::Group group(0x123, {1, 2, 3},
                        {{ir::Op::read, 64, {}, 0, 100}, {ir::Op::read, 64, {}, 0, 200}},
                        {{100, 1}, {200, 0}});
  auto state = Initial();
  auto budget = Plenty();
  EXPECT_EQ(Execute(group, state, budget), Outcome::completed);
  EXPECT_EQ(state.cells[0].value.word(0), 11);
  EXPECT_EQ(state.cells[1].value.word(0), 7);
}

TEST(Eval, WideComparisonsOrderLimbsFromTheMostSignificantEnd) {
  // The lifted AArch64 subset only produces 1, 32 and 64-bit comparisons, so no
  // differential reaches a multi-limb compare, but the IL admits any width up to
  // the block limit. A limb scan starting at the least significant end gets
  // every wide case backwards whenever the low limbs disagree the other way.
  auto budget = Plenty();
  constexpr unsigned kWidth = 128;
  const std::vector<std::uint64_t> above{0, 1};           // 2^64
  const std::vector<std::uint64_t> below{UINT64_MAX, 0};  // 2^64 - 1
  for (const bool swap : {false, true}) {
    for (const auto op : {ir::Op::unsigned_less, ir::Op::signed_less}) {
      State state;
      auto left = BitVector::from_words(kWidth, swap ? below : above, 4096, budget);
      auto right = BitVector::from_words(kWidth, swap ? above : below, 4096, budget);
      ASSERT_TRUE(left);
      ASSERT_TRUE(right);
      auto result = BitVector::from_u64(1, 0, 4096, budget);
      ASSERT_TRUE(result);
      state.cells.push_back({100, std::move(*left)});
      state.cells.push_back({200, std::move(*right)});
      state.cells.push_back({300, std::move(*result)});
      const ir::Group group(
          0x400, {1, 2, 3, 4},
          {{ir::Op::read, kWidth, {}, 0, 100}, {ir::Op::read, kWidth, {}, 0, 200}, {op, 1, {0, 1}}},
          {{300, 2}});
      EXPECT_EQ(Execute(group, state, budget), Outcome::completed);

      // Both values are positive at this width, so the two orders agree.
      EXPECT_EQ(state.cells.back().value.word(0), swap ? 1u : 0u);
    }
  }
}

TEST(Eval, InvalidGroupsDoNotChangeState) {
  const std::vector<ir::Group> groups{
      {0, {}, {{ir::Op::add, 64, {0, 0}}}, {{100, 0}}},
      {0, {}, {{ir::Op::constant, 64, {}, 42}}, {{100, 0}, {100, 0}}},
      {0, {}, {{ir::Op::constant, 64, {}, 42}}, {{100, 1}}},
      {0, {}, {{ir::Op::constant, 0, {}, 42}}, {{100, 0}}},
      {0, {}, {{static_cast<ir::Op>(999), 64}}, {}},
      {0, {}, {{ir::Op::constant, 64}, {ir::Op::extract, 64, {0}, 1}}, {{100, 1}}},
      {0,
       {},
       {{ir::Op::constant, 32}, {ir::Op::constant, 64}, {ir::Op::add, 64, {0, 1}}},
       {{100, 2}}}};
  for (const auto& group : groups) {
    auto state = Initial();
    auto budget = Plenty();
    EXPECT_EQ(Execute(group, state, budget), Outcome::invalid_group);
    EXPECT_EQ(state.cells[0].value.word(0), 7);
    EXPECT_EQ(state.cells[1].value.word(0), 11);
  }
}

TEST(Eval, EveryExhaustionPointLeavesStateIntact) {
  const ir::Group group(0, {}, {{ir::Op::constant, 64, {}, 42}, {ir::Op::constant, 64, {}, 99}},
                        {{100, 0}, {200, 1}});
  auto full_state = Initial();
  auto full_budget = Plenty();
  ASSERT_EQ(Execute(group, full_state, full_budget), Outcome::completed);
  const auto used = full_budget.used();
  for (std::uint64_t bytes = 0; bytes < used.bytes; ++bytes) {
    auto state = Initial();
    Budget budget({used.work, bytes});
    EXPECT_EQ(Execute(group, state, budget), Outcome::resource_limit);
    EXPECT_EQ(state.cells[0].value.word(0), 7);
    EXPECT_EQ(state.cells[1].value.word(0), 11);
  }

  for (std::uint64_t work : {std::uint64_t{0}, used.work / 2, used.work - 1}) {
    auto state = Initial();
    Budget budget({work, used.bytes});
    EXPECT_EQ(Execute(group, state, budget), Outcome::resource_limit);
    EXPECT_EQ(state.cells[0].value.word(0), 7);
  }
}

TEST(Eval, ArbitraryWidthsSignednessAndLargeShiftCounts) {
  auto state = Initial(129);
  auto budget = Plenty();
  const ir::Group group(0, {},
                        {{ir::Op::constant, 129, {}, 0},
                         {ir::Op::bit_not, 129, {0}},
                         {ir::Op::constant, 129, {}, 1},
                         {ir::Op::signed_less, 1, {1, 2}},
                         {ir::Op::unsigned_less, 1, {1, 2}},
                         {ir::Op::zext, 129, {3}},
                         {ir::Op::select, 129, {4, 1, 2}}},
                        {{100, 5}, {200, 6}});
  ASSERT_EQ(Execute(group, state, budget), Outcome::completed);
  EXPECT_EQ(state.cells[0].value.word(0), 1);
  EXPECT_EQ(state.cells[1].value.word(0), 1);

  const ir::Group shift(0, {},
                        {{ir::Op::constant, 129, {}, 0},
                         {ir::Op::bit_not, 129, {0}},
                         {ir::Op::lshr, 129, {1, 1}},
                         {ir::Op::ashr, 129, {1, 1}}},
                        {{100, 2}, {200, 3}});
  ASSERT_EQ(Execute(shift, state, budget), Outcome::completed);
  EXPECT_EQ(state.cells[0].value.word(0), 0);
  EXPECT_EQ(state.cells[0].value.word(2), 0);
  EXPECT_EQ(state.cells[1].value.word(0), ~std::uint64_t{0});
  EXPECT_EQ(state.cells[1].value.word(2), 1);
}

TEST(Eval, DivisionHighProductsAndBitCountsAtNarrowWidths) {
  struct Example {
    ir::Op op;
    unsigned width;
    std::uint64_t a, b, expected;
  };

  const Example examples[] = {
      {ir::Op::udiv, 8, 0xff, 0x10, 0x0f},
      {ir::Op::udiv, 8, 0xff, 0, 0},
      {ir::Op::sdiv, 8, 0x80, 0xff, 0x80},  // -128 / -1 wraps back
      {ir::Op::sdiv, 8, 0x81, 0x02, 0xc1},  // -127 / 2 truncates to -63
      {ir::Op::sdiv, 8, 0x81, 0, 0},
      {ir::Op::umulh, 8, 0xff, 0xff, 0xfe},
      {ir::Op::smulh, 8, 0xff, 0xff, 0x00},  // -1 * -1 = 1
      {ir::Op::smulh, 8, 0x80, 0x02, 0xff},  // -256 has an all-ones high byte
      {ir::Op::umulh, 1, 1, 1, 0},
      {ir::Op::clz, 8, 0x01, 0, 7},
      {ir::Op::clz, 8, 0, 0, 8},
      {ir::Op::clz, 1, 0, 0, 1},
      {ir::Op::rbit, 8, 0x01, 0, 0x80},
      {ir::Op::rbit, 12, 0x00f, 0, 0xf00},
  };

  for (const auto& example : examples) {
    const bool unary = example.op == ir::Op::clz || example.op == ir::Op::rbit;
    const ir::Group group(0, {},
                          {{ir::Op::constant, example.width, {}, example.a},
                           {ir::Op::constant, example.width, {}, example.b},
                           {example.op, example.width, {0, unary ? 0U : 1U}},
                           {ir::Op::zext, 64, {2}}},
                          {{100, 3}});
    auto state = Initial();
    auto budget = Plenty();
    ASSERT_EQ(Execute(group, state, budget), Outcome::completed)
        << ir::Descriptor(example.op)->name;
    EXPECT_EQ(state.cells[0].value.word(0), example.expected)
        << ir::Descriptor(example.op)->name << ' ' << example.width << ' ' << example.a;
  }

  // These are defined only up to 64 bits; a wider one is malformed, not a
  // resource failure.
  for (const auto op :
       {ir::Op::udiv, ir::Op::sdiv, ir::Op::umulh, ir::Op::smulh, ir::Op::clz, ir::Op::rbit}) {
    auto state = Initial(65);
    auto budget = Plenty();
    const ir::Group group(0, {}, {{ir::Op::read, 65, {}, 0, 100}, {op, 65, {0, 0}}}, {{100, 1}});
    EXPECT_EQ(Execute(group, state, budget), Outcome::invalid_group) << ir::Descriptor(op)->name;
    EXPECT_EQ(state.cells[0].value.word(0), 7);
  }
}

TEST(Eval, MissingWrongWidthAndDuplicateStorageDecline) {
  const ir::Group missing(0, {}, {{ir::Op::read, 64, {}, 0, 999}}, {});
  const ir::Group wrong(0, {}, {{ir::Op::read, 32, {}, 0, 100}}, {});
  auto state = Initial();
  auto budget = Plenty();
  EXPECT_EQ(Execute(missing, state, budget), Outcome::invalid_state);
  EXPECT_EQ(Execute(wrong, state, budget), Outcome::invalid_state);
  state.cells[1].id = 100;
  EXPECT_EQ(Execute(ir::Group(0, {}, {}, {}), state, budget), Outcome::invalid_state);
}

TEST(Eval, MovedOutStorageIsNotAnImplicitZero) {
  auto state = Initial();
  auto retained = std::move(state.cells[0].value);
  const ir::Group group(0, {}, {{ir::Op::read, 64, {}, 0, 100}}, {{100, 0}});
  auto budget = Plenty();
  EXPECT_EQ(Execute(group, state, budget), Outcome::invalid_state);
  EXPECT_EQ(retained.word(0), 7);
}

Memory MappedMemory(unsigned size = 16) {
  std::vector<std::uint8_t> bytes(size, 0);
  const RegionInput region{0x1000, bytes};
  auto budget = Plenty();
  auto result = Memory::Create(std::span(&region, 1), budget);
  return std::move(*result.memory);
}

TEST(Eval, ExclusiveScalarReservationSurvivesGroupsAndControlsConditionalWrite) {
  const ir::Group load(0, {},
                       {{ir::Op::constant, 64, {}, 0x1000},
                        {ir::Op::exclusive_load, 32, {0}, 0, 0, {ir::ByteOrder::little, 4, true}}},
                       {{100, 1}}, ir::MemoryModel::qemu_exclusive_scalar_reference);
  const ir::Group store(
      4, {},
      {{ir::Op::constant, 64, {}, 0x1000},
       {ir::Op::constant, 32, {}, 55},
       {ir::Op::exclusive_store, 32, {0, 1}, 0, 0, {ir::ByteOrder::little, 4, true}}},
      {{200, 2}}, ir::MemoryModel::qemu_exclusive_scalar_reference);
  const ir::Group plain(8, {},
                        {{ir::Op::constant, 64, {}, 0x1000},
                         {ir::Op::constant, 32, {}, 77},
                         {ir::Op::store, 32, {0, 1}}},
                        {}, ir::MemoryModel::atomic_scalar_reference);
  const ir::Group same(8, {},
                       {{ir::Op::constant, 64, {}, 0x1000},
                        {ir::Op::constant, 32, {}, 42},
                        {ir::Op::store, 32, {0, 1}}},
                       {}, ir::MemoryModel::atomic_scalar_reference);
  const ir::Group clear(8, {}, {{ir::Op::exclusive_clear, 1}}, {},
                        ir::MemoryModel::qemu_exclusive_scalar_reference);
  auto initial = [] {
    auto budget = Plenty();
    State state;
    for (auto id : {100U, 200U}) {
      auto value = BitVector::from_u64(32, 0, 4096, budget);
      state.cells.push_back({id, std::move(*value)});
    }

    return state;
  };

  auto mapped = [] {
    auto budget = Plenty();
    const std::array<std::uint8_t, 4> bytes{42, 0, 0, 0};
    const RegionInput region{0x1000, bytes};
    return std::move(*Memory::Create(std::span(&region, 1), budget).memory);
  };

  for (unsigned scenario = 0; scenario < 4; ++scenario) {
    auto state = initial();
    auto memory = mapped();
    auto budget = Plenty();
    const auto loaded = Execute(load, state, memory, budget);
    ASSERT_EQ(loaded.outcome, Outcome::completed);
    ASSERT_TRUE(state.exclusive);
    EXPECT_EQ(state.cells[0].value.word(0), 42);
    if (scenario == 1) {
      ASSERT_EQ(Execute(plain, state, memory, budget).outcome, Outcome::completed);
    }

    if (scenario == 2) {
      ASSERT_EQ(Execute(same, state, memory, budget).outcome, Outcome::completed);
    }

    if (scenario == 3) {
      ASSERT_EQ(Execute(clear, state, memory, budget).outcome, Outcome::completed);
    }

    const auto stored = Execute(store, state, memory, budget);
    ASSERT_EQ(stored.outcome, Outcome::completed);
    const bool success = scenario == 0 || scenario == 2;
    EXPECT_EQ(state.cells[1].value.word(0), success ? 0U : 1U);
    EXPECT_EQ(memory.Regions()[0].bytes[0], success ? 55U : scenario == 1 ? 77U : 42U);
    EXPECT_FALSE(state.exclusive);
    if (scenario == 3)
      EXPECT_TRUE(stored.events.empty());
    else {
      ASSERT_EQ(stored.events.size(), 1);
      EXPECT_TRUE(stored.events[0].conditional);
      EXPECT_EQ(stored.events[0].performed, success);
    }

    const auto repeated = Execute(store, state, memory, budget);
    EXPECT_EQ(repeated.outcome, Outcome::completed);
    EXPECT_EQ(state.cells[1].value.word(0), 1);
    EXPECT_TRUE(repeated.events.empty());
  }
}

TEST(Eval, ExclusiveStoreWrongAddressSkipsAccessAndMatchingFaultRetainsReservation) {
  auto budget = Plenty();
  State state;
  for (auto id : {100U, 200U}) {
    auto value = BitVector::from_u64(32, 0, 4096, budget);
    state.cells.push_back({id, std::move(*value)});
  }

  const ir::Group load(0, {},
                       {{ir::Op::constant, 64, {}, 0x1000},
                        {ir::Op::exclusive_load, 32, {0}, 0, 0, {ir::ByteOrder::little, 4, true}}},
                       {{100, 1}}, ir::MemoryModel::qemu_exclusive_scalar_reference);
  const ir::Group wrong(
      4, {},
      {{ir::Op::constant, 64, {}, 0x2000},
       {ir::Op::constant, 32, {}, 9},
       {ir::Op::exclusive_store, 32, {0, 1}, 0, 0, {ir::ByteOrder::little, 4, true}}},
      {{200, 2}}, ir::MemoryModel::qemu_exclusive_scalar_reference);
  const ir::Group matching(
      4, {},
      {{ir::Op::constant, 64, {}, 0x1000},
       {ir::Op::constant, 32, {}, 9},
       {ir::Op::exclusive_store, 32, {0, 1}, 0, 0, {ir::ByteOrder::little, 4, true}}},
      {{200, 2}}, ir::MemoryModel::qemu_exclusive_scalar_reference);
  const std::array<std::uint8_t, 4> bytes{42, 0, 0, 0};
  const RegionInput region{0x1000, bytes, true, false};
  auto memory = std::move(*Memory::Create(std::span(&region, 1), budget).memory);
  ASSERT_EQ(Execute(load, state, memory, budget).outcome, Outcome::completed);
  const auto skipped = Execute(wrong, state, memory, budget);
  EXPECT_EQ(skipped.outcome, Outcome::completed);
  EXPECT_TRUE(skipped.events.empty());
  EXPECT_EQ(state.cells[1].value.word(0), 1);
  EXPECT_FALSE(state.exclusive);
  ASSERT_EQ(Execute(load, state, memory, budget).outcome, Outcome::completed);
  const auto fault = Execute(matching, state, memory, budget);
  ASSERT_EQ(fault.outcome, Outcome::fault);
  ASSERT_TRUE(fault.fault);
  EXPECT_EQ(fault.fault->kind, FaultKind::permission);
  ASSERT_TRUE(state.exclusive);
  EXPECT_EQ(state.exclusive->address, 0x1000);
  EXPECT_EQ(state.cells[1].value.word(0), 1);
  EXPECT_EQ(memory.Regions()[0].bytes[0], 42);
}

TEST(Eval, ExclusiveStoreResourceFailureRollsBackWriteAndMonitor) {
  const ir::Group store(
      0, {},
      {{ir::Op::constant, 64, {}, 0x1000},
       {ir::Op::constant, 32, {}, 55},
       {ir::Op::exclusive_store, 32, {0, 1}, 0, 0, {ir::ByteOrder::little, 4, true}}},
      {{200, 2}}, ir::MemoryModel::qemu_exclusive_scalar_reference);
  const auto initial = [](const Memory& memory) {
    auto budget = Plenty();
    State state;
    state.cells.push_back({200, std::move(*BitVector::from_u64(32, 7, 4096, budget))});
    state.exclusive = ExclusiveReservation{0x1000, 4, {42, 0, 0, 0}, memory.identity()};
    return state;
  };

  const auto mapped = [] {
    auto budget = Plenty();
    const std::array<std::uint8_t, 4> bytes{42, 0, 0, 0};
    const RegionInput region{0x1000, bytes};
    return std::move(*Memory::Create(std::span(&region, 1), budget).memory);
  };

  auto full = Plenty();
  auto completed_memory = mapped();
  auto completed_state = initial(completed_memory);
  ASSERT_EQ(Execute(store, completed_state, completed_memory, full).outcome, Outcome::completed);
  const auto used = full.used();
  for (const auto limits :
       {Resources{used.work - 1, used.bytes}, Resources{used.work, used.bytes - 1}}) {
    auto memory = mapped();
    auto state = initial(memory);
    Budget budget(limits);
    const auto result = Execute(store, state, memory, budget);
    EXPECT_EQ(result.outcome, Outcome::resource_limit);
    EXPECT_TRUE(result.events.empty());
    ASSERT_TRUE(state.exclusive);
    EXPECT_EQ(state.exclusive->bytes[0], 42);
    EXPECT_EQ(state.cells[0].value.word(0), 7);
    EXPECT_EQ(memory.Regions()[0].bytes[0], 42);
  }
}

TEST(Eval, ExclusiveReservationBelongsToItsMemorySnapshotAcrossMoves) {
  auto budget = Plenty();
  State state;
  state.cells.push_back({200, std::move(*BitVector::from_u64(32, 7, 4096, budget))});
  const std::array<std::uint8_t, 4> bytes{42, 0, 0, 0};
  const RegionInput region{0x1000, bytes};
  auto original = std::move(*Memory::Create(std::span(&region, 1), budget).memory);
  auto unrelated = std::move(*Memory::Create(std::span(&region, 1), budget).memory);
  const ir::Group load(0, {},
                       {{ir::Op::constant, 64, {}, 0x1000},
                        {ir::Op::exclusive_load, 32, {0}, 0, 0, {ir::ByteOrder::little, 4, true}}},
                       {}, ir::MemoryModel::qemu_exclusive_scalar_reference);
  const ir::Group store(
      4, {},
      {{ir::Op::constant, 64, {}, 0x1000},
       {ir::Op::constant, 32, {}, 55},
       {ir::Op::exclusive_store, 32, {0, 1}, 0, 0, {ir::ByteOrder::little, 4, true}}},
      {{200, 2}}, ir::MemoryModel::qemu_exclusive_scalar_reference);
  ASSERT_EQ(Execute(load, state, original, budget).outcome, Outcome::completed);
  EXPECT_EQ(Execute(store, state, unrelated, budget).outcome, Outcome::invalid_state);
  ASSERT_TRUE(state.exclusive);
  auto moved = std::move(original);
  EXPECT_EQ(Execute(store, state, moved, budget).outcome, Outcome::completed);
  EXPECT_EQ(state.cells[0].value.word(0), 0);
  EXPECT_EQ(moved.Regions()[0].bytes[0], 55);
}

TEST(Eval, ExclusiveProfileDeclinesTaggedAddressAndInvalidReservation) {
  auto budget = Plenty();
  const std::array<std::uint8_t, 4> bytes{42, 0, 0, 0};
  const RegionInput region{0x1000, bytes};
  auto memory = std::move(*Memory::Create(std::span(&region, 1), budget).memory);
  const ir::Group tagged(
      0, {},
      {{ir::Op::constant, 64, {}, UINT64_C(0x7f00000000001000)},
       {ir::Op::exclusive_load, 32, {0}, 0, 0, {ir::ByteOrder::little, 4, true}}},
      {}, ir::MemoryModel::qemu_exclusive_scalar_reference);
  State state;
  EXPECT_EQ(Execute(tagged, state, memory, budget).outcome, Outcome::unsupported);
  const ir::Group unaligned(
      0, {},
      {{ir::Op::constant, 64, {}, 0x1001},
       {ir::Op::exclusive_load, 32, {0}, 0, 0, {ir::ByteOrder::little, 4, true}}},
      {}, ir::MemoryModel::qemu_exclusive_scalar_reference);
  EXPECT_EQ(Execute(unaligned, state, memory, budget).outcome, Outcome::unsupported);
  const ir::Group ordinary(
      0, {}, {{ir::Op::constant, 64, {}, UINT64_C(0x7f00000000001000)}, {ir::Op::load, 32, {0}}},
      {}, ir::MemoryModel::atomic_scalar_reference);
  EXPECT_EQ(Execute(ordinary, state, memory, budget).outcome, Outcome::unsupported);
  EXPECT_FALSE(state.exclusive);
  state.exclusive = ExclusiveReservation{0x1001, 4, {42, 0, 0, 0}, memory.identity()};
  EXPECT_EQ(Execute(tagged, state, memory, budget).outcome, Outcome::invalid_state);
  EXPECT_EQ(memory.Regions()[0].bytes[0], 42);
}

TEST(Eval, OrderedStoresForwardAndEndiannessIsExplicit) {
  const ir::Group group(0, {},
                        {{ir::Op::constant, 64, {}, 0x1000},
                         {ir::Op::constant, 32, {}, 0x12345678},
                         {ir::Op::store, 32, {0, 1}, 0, 0, {ir::ByteOrder::big, 1}},
                         {ir::Op::load, 32, {0}, 0, 0, {ir::ByteOrder::little, 1}},
                         {ir::Op::zext, 64, {3}}},
                        {{100, 4}}, ir::MemoryModel::atomic_scalar_reference);
  auto state = Initial();
  auto memory = MappedMemory();
  auto budget = Plenty();
  const auto result = Execute(group, state, memory, budget);
  ASSERT_EQ(result.outcome, Outcome::completed);
  EXPECT_EQ(state.cells[0].value.word(0), 0x78563412);
  ASSERT_EQ(result.events.size(), 2);
  EXPECT_TRUE(result.events[0].write);
  EXPECT_FALSE(result.events[1].write);
  EXPECT_EQ(result.events[0].bytes, result.events[1].bytes);
  EXPECT_EQ(memory.Regions()[0].bytes[0], 0x12);
}

TEST(Eval, ModeledFaultPublishesOrderedPrefixNotFinalWrites) {
  const ir::Group group(0, {},
                        {{ir::Op::constant, 64, {}, 0x1000},
                         {ir::Op::constant, 64, {}, 0x1008},
                         {ir::Op::constant, 64, {}, 42},
                         {ir::Op::store, 64, {0, 2}},
                         {ir::Op::write, 64, {2}, 0, 100},
                         {ir::Op::store, 64, {1, 2}}},
                        {{200, 2}}, ir::MemoryModel::atomic_scalar_reference);
  auto state = Initial();
  auto memory = MappedMemory(8);
  auto budget = Plenty();
  const auto result = Execute(group, state, memory, budget);
  ASSERT_EQ(result.outcome, Outcome::fault);
  ASSERT_TRUE(result.fault);
  EXPECT_EQ(result.fault->address, 0x1008);
  EXPECT_EQ(result.fault->operation, 5);
  EXPECT_EQ(result.fault->kind, FaultKind::unmapped);
  EXPECT_EQ(memory.Regions()[0].bytes[0], 42);
  EXPECT_EQ(state.cells[0].value.word(0), 42);
  EXPECT_EQ(state.cells[1].value.word(0), 11);
  ASSERT_EQ(result.events.size(), 2);
  EXPECT_TRUE(result.events[0].completed);
  EXPECT_FALSE(result.events[1].completed);
}

TEST(Eval, LoadsBeforeWritesLeaveDestinationsUnchangedOnModelFault) {
  const ir::Group group(0, {},
                        {{ir::Op::constant, 64, {}, 0x1000},
                         {ir::Op::constant, 64, {}, 0x1008},
                         {ir::Op::load, 64, {0}},
                         {ir::Op::load, 64, {1}}},
                        {{100, 2}, {200, 3}}, ir::MemoryModel::atomic_scalar_reference);
  auto state = Initial();
  auto memory = MappedMemory(8);
  auto budget = Plenty();
  const auto result = Execute(group, state, memory, budget);
  EXPECT_EQ(result.outcome, Outcome::fault);
  EXPECT_EQ(state.cells[0].value.word(0), 7);
  EXPECT_EQ(state.cells[1].value.word(0), 11);
}

TEST(Eval, ResourceFailureDiscardsMemoryRegistersAndEvents) {
  const ir::Group group(0, {},
                        {{ir::Op::constant, 64, {}, 0x1000},
                         {ir::Op::constant, 64, {}, 42},
                         {ir::Op::store, 64, {0, 1}},
                         {ir::Op::write, 64, {1}, 0, 100},
                         {ir::Op::load, 64, {0}}},
                        {{200, 4}}, ir::MemoryModel::atomic_scalar_reference);
  auto full = Plenty();
  auto state = Initial();
  auto memory = MappedMemory();
  ASSERT_EQ(Execute(group, state, memory, full).outcome, Outcome::completed);
  for (std::uint64_t bytes = 0; bytes < full.used().bytes; ++bytes) {
    auto original = Initial();
    auto clean_memory = MappedMemory();
    Budget budget({full.used().work, bytes});
    auto result = Execute(group, original, clean_memory, budget);
    ASSERT_EQ(result.outcome, Outcome::resource_limit) << bytes;
    EXPECT_TRUE(result.events.empty());
    EXPECT_FALSE(result.fault);
    EXPECT_EQ(original.cells[0].value.word(0), 7);
    EXPECT_EQ(original.cells[1].value.word(0), 11);
    EXPECT_EQ(clean_memory.Regions()[0].bytes[0], 0);
  }
}

TEST(Eval, EffectOnlyValuesAndLateMalformedNodesCannotExecute) {
  for (bool late : {false, true}) {
    const ir::Group group(0, {},
                          {{ir::Op::constant, 64, {}, 0x1000},
                           {ir::Op::constant, 64, {}, 42},
                           {ir::Op::store, 64, {0, 1}},
                           late ? ir::Node{ir::Op::load, 64, {0}, 0, 0, {ir::ByteOrder::little, 3}}
                                : ir::Node{ir::Op::add, 64, {2, 1}}},
                          {}, ir::MemoryModel::atomic_scalar_reference);
    auto state = Initial();
    auto memory = MappedMemory();
    auto budget = Plenty();
    EXPECT_EQ(Execute(group, state, memory, budget).outcome, Outcome::invalid_group);
    EXPECT_EQ(memory.Regions()[0].bytes[0], 0);
  }
}

TEST(Eval, AlignmentAndPermissionsAreDistinctFromUnsupportedContext) {
  const ir::Group group(0, {},
                        {{ir::Op::constant, 64, {}, 0x1001},
                         {ir::Op::load, 64, {0}, 0, 0, {ir::ByteOrder::little, 8}}},
                        {{100, 1}}, ir::MemoryModel::atomic_scalar_reference);
  auto state = Initial();
  auto budget = Plenty();
  EXPECT_EQ(Execute(group, state, budget), Outcome::unsupported);
  auto memory = MappedMemory();
  const auto result = Execute(group, state, memory, budget);
  ASSERT_EQ(result.outcome, Outcome::fault);
  EXPECT_EQ(result.fault->kind, FaultKind::alignment);
  EXPECT_EQ(state.cells[0].value.word(0), 7);
}

TEST(Eval, ImageLocationsRequirePlacementAndPreserveLowBiasBits) {
  const ir::Group group(0x1234, {},
                        {{ir::Op::image_address, 64, {}, 0x1234},
                         {ir::Op::constant, 64, {}, ~std::uint64_t{0xfff}},
                         {ir::Op::bit_and, 64, {0, 1}}},
                        {{100, 0}, {200, 2}});
  auto state = Initial();
  auto budget = Plenty();
  EXPECT_EQ(Execute(group, state, budget), Outcome::unsupported);
  EXPECT_EQ(state.cells[0].value.word(0), 7);
  ASSERT_EQ(Execute(group, state, budget, {}, ExecutionContext{0x1fff}), Outcome::completed);
  EXPECT_EQ(state.cells[0].value.word(0), 0x3233);
  EXPECT_EQ(state.cells[1].value.word(0), 0x3000);
}

TEST(Eval, MemoryFaultModelCannotBeImplicitlyAssumed) {
  const ir::Group group(0, {}, {{ir::Op::constant, 64, {}, 0x1000}, {ir::Op::load, 64, {0}}},
                        {{100, 1}});
  auto state = Initial();
  auto memory = MappedMemory();
  auto budget = Plenty();
  const auto result = Execute(group, state, memory, budget);
  EXPECT_EQ(result.outcome, Outcome::unsupported);
  EXPECT_TRUE(result.events.empty());
  EXPECT_EQ(state.cells[0].value.word(0), 7);
}

TEST(Eval, OrderedWritesDoNotChangeEntrySnapshotReads) {
  const ir::Group group(0, {},
                        {{ir::Op::constant, 64, {}, 42},
                         {ir::Op::write, 64, {0}, 0, 100},
                         {ir::Op::read, 64, {}, 0, 100}},
                        {{200, 2}});
  auto state = Initial();
  auto budget = Plenty();
  ASSERT_EQ(Execute(group, state, budget), Outcome::completed);
  EXPECT_EQ(state.cells[0].value.word(0), 42);
  EXPECT_EQ(state.cells[1].value.word(0), 7);
}

TEST(Eval, ConditionalTransferPreservesConditionDestinationRelation) {
  for (bool condition : {false, true}) {
    const ir::Group group(0x1234, {},
                          {{ir::Op::constant, 1, {}, condition},
                           {ir::Op::image_address, 64, {}, 0x1250},
                           {ir::Op::image_address, 64, {}, 0x1238}},
                          {}, ir::MemoryModel::unspecified,
                          ir::Transfer{ir::TransferKind::conditional, 1, 0, 2, {}});
    auto state = Initial();
    auto budget = Plenty();
    const auto result = ExecuteDetailed(group, state, budget, {}, {0x1fff});
    ASSERT_EQ(result.outcome, Outcome::completed);
    ASSERT_TRUE(result.transfer);
    EXPECT_EQ(result.transfer->kind, ir::TransferKind::conditional);
    EXPECT_EQ(result.transfer->condition, condition);
    EXPECT_EQ(result.transfer->target, condition ? 0x324f : 0x3237);
    EXPECT_FALSE(result.transfer->continuation);
  }
}

TEST(Eval, CallResolvesEntryTargetBeforeLinkWriteAndStopsAtBoundary) {
  const ir::Group group(0, {}, {{ir::Op::read, 64, {}, 0, 100}, {ir::Op::image_address, 64, {}, 4}},
                        {{100, 1}}, ir::MemoryModel::unspecified,
                        ir::Transfer{ir::TransferKind::call, 0, {}, {}, 1});
  auto state = Initial();
  auto budget = Plenty();
  EXPECT_EQ(Execute(group, state, budget, {}, {0x1000}), Outcome::unsupported);
  EXPECT_EQ(state.cells[0].value.word(0), 7);
  const auto result = ExecuteDetailed(group, state, budget, {}, {0x1000});
  ASSERT_EQ(result.outcome, Outcome::completed);
  ASSERT_TRUE(result.transfer);
  EXPECT_EQ(result.transfer->target, 7);
  EXPECT_EQ(result.transfer->continuation, 0x1004);
  EXPECT_EQ(state.cells[0].value.word(0), 0x1004);
  EXPECT_EQ(state.cells[1].value.word(0), 11);
}

TEST(Eval, JumpAndReturnReportValuesWithoutInventingFetchOrReturnBehavior) {
  for (const auto kind : {ir::TransferKind::jump, ir::TransferKind::return_}) {
    const ir::Group group(0, {}, {{ir::Op::read, 64, {}, 0, 200}}, {}, ir::MemoryModel::unspecified,
                          ir::Transfer{kind, 0, {}, {}, {}});
    auto state = Initial();
    auto budget = Plenty();
    const auto result = ExecuteDetailed(group, state, budget);
    ASSERT_EQ(result.outcome, Outcome::completed);
    ASSERT_TRUE(result.transfer);
    EXPECT_EQ(result.transfer->kind, kind);
    EXPECT_EQ(result.transfer->target, 11);
    EXPECT_FALSE(result.transfer->condition);
    EXPECT_FALSE(result.transfer->continuation);
  }
}

TEST(Eval, InvalidTerminalEffectsCannotPublishEarlierWrites) {
  const std::vector<ir::Transfer> invalid{{ir::TransferKind::jump, 4, {}, {}, {}},
                                          {ir::TransferKind::jump, 1, {}, {}, {}},
                                          {ir::TransferKind::jump, 3, {}, {}, {}},
                                          {ir::TransferKind::jump, 0, 1, {}, {}},
                                          {ir::TransferKind::conditional, 0, {}, 0, {}},
                                          {ir::TransferKind::conditional, 0, 0, 0, {}},
                                          {ir::TransferKind::conditional, 0, 1, {}, {}},
                                          {ir::TransferKind::conditional, 0, 1, 1, {}},
                                          {ir::TransferKind::call, 0, {}, {}, {}},
                                          {ir::TransferKind::call, 0, {}, {}, 1},
                                          {ir::TransferKind::return_, 0, {}, {}, 0},
                                          {static_cast<ir::TransferKind>(999), 0, {}, {}, {}}};
  for (const auto transfer : invalid) {
    const ir::Group group(0, {},
                          {{ir::Op::constant, 64, {}, 0x1000},
                           {ir::Op::constant, 1, {}, 1},
                           {ir::Op::write, 64, {0}, 0, 100},
                           {ir::Op::store, 64, {0, 0}}},
                          {}, ir::MemoryModel::atomic_scalar_reference, transfer);
    auto state = Initial();
    auto memory = MappedMemory();
    auto budget = Plenty();
    const auto result = Execute(group, state, memory, budget);
    EXPECT_EQ(result.outcome, Outcome::invalid_group);
    EXPECT_FALSE(result.transfer);
    EXPECT_TRUE(result.events.empty());
    EXPECT_EQ(state.cells[0].value.word(0), 7);
    EXPECT_EQ(memory.Regions()[0].bytes[0], 0);
  }
}

TEST(Eval, TransferPublicationRollsBackAtEveryResourceCut) {
  const ir::Group group(0, {}, {{ir::Op::read, 64, {}, 0, 100}, {ir::Op::constant, 64, {}, 0x1234}},
                        {{100, 1}}, ir::MemoryModel::unspecified,
                        ir::Transfer{ir::TransferKind::call, 0, {}, {}, 1});
  auto state = Initial();
  auto full = Plenty();
  ASSERT_EQ(ExecuteDetailed(group, state, full).outcome, Outcome::completed);
  for (bool bytes : {false, true}) {
    const auto bound = bytes ? full.used().bytes : full.used().work;
    for (std::uint64_t cut = 0; cut < bound; ++cut) {
      auto original = Initial();
      Budget budget({bytes ? full.used().work : cut, bytes ? cut : full.used().bytes});
      const auto result = ExecuteDetailed(group, original, budget);
      ASSERT_EQ(result.outcome, Outcome::resource_limit) << cut;
      EXPECT_FALSE(result.transfer);
      EXPECT_EQ(original.cells[0].value.word(0), 7);
    }
  }
}

TEST(Eval, FaultPreventsTransferAndFinalWrites) {
  const ir::Group group(0, {}, {{ir::Op::constant, 64, {}, 0x1008}, {ir::Op::load, 64, {0}}},
                        {{100, 0}}, ir::MemoryModel::atomic_scalar_reference,
                        ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}});
  auto state = Initial();
  auto memory = MappedMemory(8);
  auto budget = Plenty();
  const auto result = Execute(group, state, memory, budget);
  EXPECT_EQ(result.outcome, Outcome::fault);
  EXPECT_FALSE(result.transfer);
  EXPECT_EQ(state.cells[0].value.word(0), 7);
}

TEST(Eval, MissingLatePlacementCannotPublishPrefixOrControl) {
  const ir::Group group(0, {},
                        {{ir::Op::constant, 64, {}, 0x1000},
                         {ir::Op::constant, 64, {}, 42},
                         {ir::Op::store, 64, {0, 1}},
                         {ir::Op::write, 64, {1}, 0, 100},
                         {ir::Op::image_address, 64, {}, 0x1004}},
                        {}, ir::MemoryModel::atomic_scalar_reference,
                        ir::Transfer{ir::TransferKind::jump, 4, {}, {}, {}});
  auto state = Initial();
  auto memory = MappedMemory();
  auto budget = Plenty();
  auto result = Execute(group, state, memory, budget);
  EXPECT_EQ(result.outcome, Outcome::unsupported);
  EXPECT_FALSE(result.transfer);
  EXPECT_FALSE(result.fault);
  EXPECT_TRUE(result.events.empty());
  EXPECT_EQ(state.cells[0].value.word(0), 7);
  EXPECT_EQ(memory.Regions()[0].bytes[0], 0);
}

TEST(Eval, CombinedMemoryControlRollsBackEveryResourceCut) {
  const ir::Group group(0, {},
                        {{ir::Op::constant, 64, {}, 0x1000},
                         {ir::Op::constant, 64, {}, 42},
                         {ir::Op::store, 64, {0, 1}},
                         {ir::Op::write, 64, {1}, 0, 100},
                         {ir::Op::load, 64, {0}}},
                        {{200, 4}}, ir::MemoryModel::atomic_scalar_reference,
                        ir::Transfer{ir::TransferKind::call, 4, {}, {}, 0});
  auto state = Initial();
  auto memory = MappedMemory();
  auto full = Plenty();
  auto result = Execute(group, state, memory, full);
  ASSERT_EQ(result.outcome, Outcome::completed);
  ASSERT_TRUE(result.transfer);
  EXPECT_EQ(result.transfer->target, 42);
  for (bool bytes : {false, true}) {
    auto limit = bytes ? full.used().bytes : full.used().work;
    for (std::uint64_t cut = 0; cut < limit; ++cut) {
      auto initial = Initial();
      auto clean = MappedMemory();
      Budget budget({bytes ? full.used().work : cut, bytes ? cut : full.used().bytes});
      auto declined = Execute(group, initial, clean, budget);
      ASSERT_EQ(declined.outcome, Outcome::resource_limit) << cut;
      EXPECT_FALSE(declined.transfer);
      EXPECT_FALSE(declined.fault);
      EXPECT_TRUE(declined.events.empty());
      EXPECT_EQ(initial.cells[0].value.word(0), 7);
      EXPECT_EQ(initial.cells[1].value.word(0), 11);
      EXPECT_EQ(clean.Regions()[0].bytes[0], 0);
    }
  }
}
}  // namespace
}  // namespace nyx::eval
