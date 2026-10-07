#include "nyx/recovery/image.hpp"

#include <algorithm>
#include <array>
#include <tuple>

#include <gtest/gtest.h>

#include "nyx/eval/path.hpp"

namespace nyx::recovery {
namespace {

using ir::Op;

Budget Plenty() { return Budget({UINT64_MAX, UINT64_MAX}); }

// Two 16-bit table entries at image 0x2010, as a dispatcher indexes them.
constexpr std::array<std::uint8_t, 4> kTable = {0x10, 0x00, 0x20, 0x00};

// adrp-style page base, table offset, index 1, halfword read, scale by four and
// add to an adr base: the lifted shape of a real dispatcher, index fixed.
ir::Path Dispatch(ir::ByteOrder order = ir::ByteOrder::little,
                  std::uint64_t mask = ~UINT64_C(0xfff)) {
  const std::vector<ir::Group> groups{
      ir::Group(0x100, {1, 2, 3, 4},
                {{Op::image_address, 64, {}, 0x2468},
                 {Op::constant, 64, {}, mask},
                 {Op::bit_and, 64, {0, 1}},
                 {Op::constant, 64, {}, 0x10},
                 {Op::add, 64, {2, 3}},
                 {Op::constant, 64, {}, 2},
                 {Op::add, 64, {4, 5}},
                 {Op::load, 16, {6}, 0, 0, {order, 1}},
                 {Op::zext, 64, {7}},
                 {Op::constant, 64, {}, 2},
                 {Op::shl, 64, {8, 9}},
                 {Op::image_address, 64, {}, 0x100},
                 {Op::add, 64, {11, 10}}},
                {}, ir::MemoryModel::atomic_scalar_reference,
                ir::Transfer{ir::TransferKind::jump, 12, {}, {}, {}})};
  auto budget = Plenty();
  auto path = ir::NormalizePath(groups, budget);
  EXPECT_TRUE(path.path);
  return std::move(*path.path);
}

const ir::Node& Target(const ir::Path& path) {
  return path.nodes()[path.boundaries().back().transfer->target];
}

// The same dispatcher, preceded by a store the path places at `written`.
ir::Path Writing(std::uint64_t written, unsigned width = 16) {
  const std::vector<ir::Group> groups{
      ir::Group(0xfc, {9, 9, 9, 9},
                {{Op::image_address, 64, {}, written},
                 {Op::constant, width, {}, 0},
                 {Op::store, width, {0, 1}}},
                {}, ir::MemoryModel::atomic_scalar_reference,
                ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}}),
      ir::Group(0x100, {1, 2, 3, 4},
                {{Op::image_address, 64, {}, 0x2468},
                 {Op::constant, 64, {}, ~UINT64_C(0xfff)},
                 {Op::bit_and, 64, {0, 1}},
                 {Op::constant, 64, {}, 0x10},
                 {Op::add, 64, {2, 3}},
                 {Op::constant, 64, {}, 2},
                 {Op::add, 64, {4, 5}},
                 {Op::load, 16, {6}, 0, 0, {ir::ByteOrder::little, 1}},
                 {Op::zext, 64, {7}},
                 {Op::constant, 64, {}, 2},
                 {Op::shl, 64, {8, 9}},
                 {Op::image_address, 64, {}, 0x100},
                 {Op::add, 64, {11, 10}}},
                {}, ir::MemoryModel::atomic_scalar_reference,
                ir::Transfer{ir::TransferKind::jump, 12, {}, {}, {}})};
  auto budget = Plenty();
  auto path = ir::NormalizePath(groups, budget);
  EXPECT_TRUE(path.path);
  return std::move(*path.path);
}

TEST(ImageValues, ConditionalExclusiveWriteRefutesDeclaredImageBytes) {
  const auto dispatch = Dispatch();
  const std::vector<ir::Group> groups{
      ir::Group(0xfc, {9, 9, 9, 9},
                {{Op::image_address, 64, {}, 0x2010},
                 {Op::constant, 32, {}, 0},
                 {Op::exclusive_store, 32, {0, 1}, 0, 0, {ir::ByteOrder::little, 4, true}},
                 {Op::image_address, 64, {}, 0x100}},
                {}, ir::MemoryModel::qemu_exclusive_scalar_reference,
                ir::Transfer{ir::TransferKind::jump, 3, {}, {}, {}}),
      dispatch.sources()[0]};
  auto budget = Plenty();
  auto normalized = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(normalized.path);
  const std::array<ir::ConstantImageRange, 1> ranges{ir::ConstantImageRange{0x2010, kTable}};
  const auto folded = FoldImageValues(*normalized.path, {ranges, {}, true}, budget);
  ASSERT_TRUE(folded.path);
  ASSERT_EQ(folded.contradicted.size(), 1);
  EXPECT_EQ(folded.contradicted[0].store, 2);
  EXPECT_EQ(folded.contradicted[0].width, 32);
  EXPECT_TRUE(folded.constants.empty());
  EXPECT_NE(Target(*folded.path).op, Op::image_address);
}

TEST(ImageValues, ADeclaredTableAndPagePlacementLeaveTheJumpALiteralLocation) {
  const std::array<ir::ConstantImageRange, 1> ranges{ir::ConstantImageRange{0x2010, kTable}};
  auto budget = Plenty();
  const auto input = Dispatch();
  const auto result = FoldImageValues(input, {ranges, {}, true}, budget);
  ASSERT_TRUE(result.path);
  EXPECT_EQ(result.path->revision(), input.revision() + 1);
  EXPECT_EQ(Target(*result.path).op, Op::image_address);
  EXPECT_EQ(Target(*result.path).immediate, 0x100 + 0x20 * 4);

  // The read is still there and still executes; only what consumed it changed.
  const auto loads = std::count_if(result.path->nodes().begin(), result.path->nodes().end(),
                                   [](const ir::Node& node) { return node.op == Op::load; });
  EXPECT_EQ(loads, 1);
  std::vector<std::pair<ImageRule, bool>> rules;
  for (const auto& edit : result.journal) {
    rules.emplace_back(edit.rule, edit.constant_bytes);
    EXPECT_FALSE(edit.relocated_slot);

    // Every edit here consumes the page base, so every one rests on placement.
    EXPECT_TRUE(edit.page_placement);
    EXPECT_EQ(result.path->nodes()[edit.node].op, edit.replacement.op);
  }

  EXPECT_EQ(rules, (std::vector<std::pair<ImageRule, bool>>{{ImageRule::page_base, false},
                                                            {ImageRule::image_offset, false},
                                                            {ImageRule::image_offset, false},
                                                            {ImageRule::constant_fold, true},
                                                            {ImageRule::constant_fold, true},
                                                            {ImageRule::image_offset, true}}));
}

TEST(ImageValues, EachDeclarationIsNecessaryForWhatRestsOnIt) {
  const std::array<ir::ConstantImageRange, 1> ranges{ir::ConstantImageRange{0x2010, kTable}};
  const auto fold = [&](ir::ImageFacts facts) {
    auto budget = Plenty();
    auto result = FoldImageValues(Dispatch(), facts, budget);
    EXPECT_TRUE(result.path);
    return result;
  };

  // Without placement the page base is unknown, so nothing downstream moves.
  const auto unplaced = fold({ranges, {}, false});
  EXPECT_NE(Target(*unplaced.path).op, Op::image_address);
  EXPECT_TRUE(std::none_of(unplaced.journal.begin(), unplaced.journal.end(),
                           [](const ImageEdit& edit) { return edit.constant_bytes; }));
  // Without the bytes the address resolves but the read gives nothing.
  const auto unread = fold({{}, {}, true});
  EXPECT_NE(Target(*unread.path).op, Op::image_address);
  EXPECT_EQ(unread.journal.size(), 3);

  // A relocated slot reaching into the read means the file bytes are not the
  // run's, even though they are declared.
  const std::array<ir::RelocatedPointer, 1> slot{ir::RelocatedPointer{0x200c, 0x9000}};
  EXPECT_NE(Target(*fold({ranges, slot, true}).path).op, Op::image_address);
}

TEST(ImageValues, OnlyAPageGranuleMaskIsAPageBase) {
  const std::array<ir::ConstantImageRange, 1> ranges{ir::ConstantImageRange{0x2010, kTable}};
  auto budget = Plenty();

  // Clearing thirteen low bits is coarser than the page a loader aligns to.
  const auto coarse = FoldImageValues(Dispatch(ir::ByteOrder::little, ~UINT64_C(0x1fff)),
                                      {ranges, {}, true}, budget);
  ASSERT_TRUE(coarse.path);
  EXPECT_TRUE(coarse.journal.empty());
  EXPECT_EQ(coarse.path->revision(), Dispatch().revision());
}

TEST(ImageValues, ByteOrderIsTheLoadsOwn) {
  const std::array<ir::ConstantImageRange, 1> ranges{ir::ConstantImageRange{0x2010, kTable}};
  auto budget = Plenty();
  const auto result = FoldImageValues(Dispatch(ir::ByteOrder::big), {ranges, {}, true}, budget);
  ASSERT_TRUE(result.path);
  EXPECT_EQ(Target(*result.path).immediate, 0x100 + 0x2000 * 4);
}

TEST(ImageValues, AWholeSlotReadIsTheLocationTheLoaderWrote) {
  const std::vector<ir::Group> groups{
      ir::Group(0x100, {1, 2, 3, 4},
                {{Op::image_address, 64, {}, 0x3000},
                 {Op::load, 64, {0}},
                 {Op::constant, 64, {}, 8},
                 {Op::add, 64, {1, 2}},
                 {Op::image_address, 64, {}, 0x3100},
                 {Op::sub, 64, {3, 4}}},
                {{9, 5}}, ir::MemoryModel::atomic_scalar_reference,
                ir::Transfer{ir::TransferKind::jump, 3, {}, {}, {}})};
  auto budget = Plenty();
  const auto path = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(path.path);
  const std::array<ir::RelocatedPointer, 1> slot{ir::RelocatedPointer{0x3000, 0x7000}};
  const ir::ImageFacts facts{{}, slot, false, {}};
  const auto result = FoldImageValues(*path.path, facts, budget);
  ASSERT_TRUE(result.path);
  EXPECT_EQ(result.RetainedPointers(facts).size(), 1);
  EXPECT_EQ(Target(*result.path).op, Op::image_address);
  EXPECT_EQ(Target(*result.path).immediate, 0x7008);

  // The distance between two image locations does not depend on the bias.
  const auto& distance = result.path->nodes()[result.path->boundaries().back().writes[0].value];
  EXPECT_EQ(distance.op, Op::constant);
  EXPECT_EQ(distance.immediate, 0x7008 - 0x3100);
  ASSERT_EQ(result.journal.size(), 2);
  EXPECT_TRUE(result.journal[0].relocated_slot);
  EXPECT_TRUE(result.journal[1].relocated_slot);

  // No page base on this path, so nothing here depends on placement.
  EXPECT_FALSE(result.journal[0].page_placement);
  EXPECT_FALSE(result.journal[1].page_placement);
}

TEST(ImageValues, AStoreIntoARelocatedSlotDropsItsLoaderValue) {
  const std::vector<ir::Group> groups{
      ir::Group(
          0xfc, {9, 9, 9, 9},
          {{Op::image_address, 64, {}, 0x3004}, {Op::constant, 16, {}, 0}, {Op::store, 16, {0, 1}}},
          {}, ir::MemoryModel::atomic_scalar_reference),
      ir::Group(0x100, {1, 2, 3, 4},
                {{Op::image_address, 64, {}, 0x3000},
                 {Op::load, 64, {0}},
                 {Op::constant, 64, {}, 8},
                 {Op::add, 64, {1, 2}}},
                {}, ir::MemoryModel::atomic_scalar_reference,
                ir::Transfer{ir::TransferKind::jump, 3, {}, {}, {}})};
  auto budget = Plenty();
  auto input = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(input.path);
  const std::array<ir::RelocatedPointer, 1> slots{ir::RelocatedPointer{0x3000, 0x7000}};
  const ir::ImageFacts facts{{}, slots, false, {}};
  const auto result = FoldImageValues(*input.path, facts, budget);
  ASSERT_TRUE(result.path);
  ASSERT_EQ(result.contradicted_pointers.size(), 1);
  EXPECT_EQ(result.contradicted_pointers[0].slot, 0x3000);
  EXPECT_EQ(result.contradicted_pointers[0].address, 0x3004);
  EXPECT_EQ(result.contradicted_pointers[0].width, 16);
  ASSERT_EQ(result.RetainedPointers(facts).size(), 1);
  EXPECT_FALSE(result.RetainedPointers(facts)[0].value_stable);
  EXPECT_NE(Target(*result.path).op, Op::image_address);
  EXPECT_TRUE(std::none_of(result.journal.begin(), result.journal.end(),
                           [](const ImageEdit& edit) { return edit.relocated_slot; }));
}

TEST(ImageValues, PointerOverlapAtTheEndOfImageAddressSpace) {
  for (const auto [slot, written, width] :
       {std::tuple{UINT64_MAX, UINT64_MAX, 8U}, std::tuple{UINT64_MAX, UINT64_MAX - 1, 16U},
        std::tuple{UINT64_MAX - 6, UINT64_C(0), 8U}}) {
    const std::vector<ir::Group> groups{
        ir::Group(0xfc, {9, 9, 9, 9},
                  {{Op::image_address, 64, {}, written},
                   {Op::constant, width, {}, 0},
                   {Op::store, width, {0, 1}}},
                  {}, ir::MemoryModel::atomic_scalar_reference),
        ir::Group(0x100, {1, 2, 3, 4},
                  {{Op::image_address, 64, {}, slot},
                   {Op::load, 64, {0}},
                   {Op::constant, 64, {}, 8},
                   {Op::add, 64, {1, 2}}},
                  {}, ir::MemoryModel::atomic_scalar_reference,
                  ir::Transfer{ir::TransferKind::jump, 3, {}, {}, {}})};
    auto budget = Plenty();
    auto input = ir::NormalizePath(groups, budget);
    ASSERT_TRUE(input.path);
    const std::array<ir::RelocatedPointer, 1> pointers{ir::RelocatedPointer{slot, 0x7000}};
    const ir::ImageFacts facts{{}, pointers, false, {}};
    const auto result = FoldImageValues(*input.path, facts, budget);
    ASSERT_TRUE(result.path);
    ASSERT_EQ(result.contradicted_pointers.size(), 1) << written;
    EXPECT_EQ(result.contradicted_pointers[0].slot, slot);
    ASSERT_EQ(result.RetainedPointers(facts).size(), 1);
    EXPECT_FALSE(result.RetainedPointers(facts)[0].value_stable);
    EXPECT_NE(Target(*result.path).op, Op::image_address);
  }

  const std::vector<ir::Group> groups{
      ir::Group(0xfc, {9, 9, 9, 9},
                {{Op::image_address, 64, {}, 0}, {Op::constant, 8, {}, 0}, {Op::store, 8, {0, 1}}},
                {}, ir::MemoryModel::atomic_scalar_reference),
      ir::Group(0x100, {1, 2, 3, 4},
                {{Op::image_address, 64, {}, UINT64_MAX - 6},
                 {Op::load, 64, {0}},
                 {Op::constant, 64, {}, 8},
                 {Op::add, 64, {1, 2}}},
                {}, ir::MemoryModel::atomic_scalar_reference,
                ir::Transfer{ir::TransferKind::jump, 3, {}, {}, {}})};
  auto budget = Plenty();
  auto input = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(input.path);
  const std::array<ir::RelocatedPointer, 2> pointers{ir::RelocatedPointer{UINT64_MAX - 6, 0x7000},
                                                     ir::RelocatedPointer{UINT64_MAX - 5, 0x8000}};
  const ir::ImageFacts facts{{}, pointers, false, {}};
  const auto result = FoldImageValues(*input.path, facts, budget);
  ASSERT_TRUE(result.path);
  EXPECT_EQ(result.contradicted_pointers.size(), 2);
  ASSERT_EQ(result.RetainedPointers(facts).size(), 2);
  EXPECT_FALSE(result.RetainedPointers(facts)[0].value_stable);
  EXPECT_FALSE(result.RetainedPointers(facts)[1].value_stable);
}

TEST(ImageValues, RetractedPointerStillBlocksOverlappingFileBytes) {
  const std::vector<ir::Group> groups{
      ir::Group(
          0xfc, {9, 9, 9, 9},
          {{Op::image_address, 64, {}, 0x3004}, {Op::constant, 8, {}, 0}, {Op::store, 8, {0, 1}}},
          {}, ir::MemoryModel::atomic_scalar_reference),
      ir::Group(0x100, {1, 2, 3, 4},
                {{Op::image_address, 64, {}, 0x3000},
                 {Op::load, 32, {0}},
                 {Op::zext, 64, {1}},
                 {Op::constant, 64, {}, 8},
                 {Op::add, 64, {2, 3}}},
                {}, ir::MemoryModel::atomic_scalar_reference,
                ir::Transfer{ir::TransferKind::jump, 4, {}, {}, {}})};
  auto budget = Plenty();
  auto input = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(input.path);
  const std::array<std::uint8_t, 4> file_bytes{0x10, 0, 0, 0};
  const std::array<ir::ConstantImageRange, 1> ranges{ir::ConstantImageRange{0x3000, file_bytes}};
  const std::array<ir::RelocatedPointer, 1> pointers{ir::RelocatedPointer{0x3000, 0x7000}};
  const ir::ImageFacts facts{ranges, pointers, false, {}};
  const auto result = FoldImageValues(*input.path, facts, budget);
  ASSERT_TRUE(result.path);
  EXPECT_TRUE(result.contradicted.empty());
  ASSERT_EQ(result.contradicted_pointers.size(), 1);
  const ir::ImageFacts held{result.constants, result.RetainedPointers(facts), false, {}};
  EXPECT_FALSE(ir::ReadConstant(held, 0x3000, 32, ir::ByteOrder::little));
  EXPECT_FALSE(ir::ReadRelocated(held, 0x3000, 64));
  EXPECT_EQ(Target(*result.path).op, Op::add);
}

TEST(ImageValues, LoaderWrittenSlotWrappingToZeroBlocksFileBytes) {
  const std::array<std::uint8_t, 1> bytes{0x42};
  const std::array<ir::ConstantImageRange, 1> ranges{ir::ConstantImageRange{0, bytes}};
  const std::array<ir::RelocatedPointer, 1> wrapped{
      ir::RelocatedPointer{UINT64_MAX - 6, 0x7000, false}};
  EXPECT_FALSE(ir::ReadConstant({ranges, wrapped, false}, 0, 8, ir::ByteOrder::little));
  const std::array<ir::RelocatedPointer, 1> adjacent{
      ir::RelocatedPointer{UINT64_MAX - 7, 0x7000, false}};
  EXPECT_EQ(ir::ReadConstant({ranges, adjacent, false}, 0, 8, ir::ByteOrder::little), 0x42);
}

TEST(ImageValues, ALiteralLessALocationIsNeitherALocationNorANumber) {
  // c - (bias + a) carries a negative bias: no placement-independent value.
  const std::vector<ir::Group> groups{ir::Group(
      0x100, {1, 2, 3, 4},
      {{Op::constant, 64, {}, 0x9000}, {Op::image_address, 64, {}, 0x3000}, {Op::sub, 64, {0, 1}}},
      {{9, 2}}, ir::MemoryModel::unspecified)};
  auto budget = Plenty();
  const auto path = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(path.path);
  const auto result = FoldImageValues(*path.path, {{}, {}, true}, budget);
  ASSERT_TRUE(result.path);
  EXPECT_TRUE(result.journal.empty());
  EXPECT_EQ(result.path->nodes()[result.path->boundaries().back().writes[0].value].op, Op::sub);
}

// The folded path must reach what the original reaches, reading the same bytes,
// at every placement the declarations admit; at one they exclude it need not.
TEST(ImageValues, FoldedAndOriginalAgreeAtEveryAdmittedPlacement) {
  const std::array<ir::ConstantImageRange, 1> ranges{ir::ConstantImageRange{0x2010, kTable}};
  auto budget = Plenty();
  const auto original = Dispatch();
  const auto folded = FoldImageValues(original, {ranges, {}, true}, budget);
  ASSERT_TRUE(folded.path);
  const auto run = [&](const ir::Path& path, std::uint64_t bias) {
    eval::State state;

    // Map the table where this placement puts it, and the page base the
    // original computes, so both executions can complete.
    std::array<std::uint8_t, 0x20> bytes{};
    std::copy(kTable.begin(), kTable.end(), bytes.begin() + 0x10);
    const eval::RegionInput regions[] = {
        {((bias + 0x2468) & ~UINT64_C(0xfff)), bytes, true, false}};
    auto memory = std::move(*eval::Memory::Create(regions, budget).memory);
    return eval::ExecutePath(path, state, memory, budget, {}, {bias});
  };

  for (const std::uint64_t bias : {UINT64_C(0), UINT64_C(0x10000), UINT64_C(0x7fff0000)}) {
    const auto before = run(original, bias), after = run(*folded.path, bias);
    ASSERT_EQ(before.outcome, eval::Outcome::completed) << bias;
    ASSERT_EQ(after.outcome, eval::Outcome::completed) << bias;
    EXPECT_EQ(before.runtime_next, after.runtime_next) << bias;
    EXPECT_EQ(after.runtime_next, bias + 0x180) << bias;
    ASSERT_EQ(before.trace.size(), after.trace.size());
    EXPECT_EQ(before.trace[0].events.size(), after.trace[0].events.size())
        << "the read still happens";
  }

  // An unaligned placement moves the original's page base but not the fold's:
  // the fold is only valid under the declared alignment.
  const auto before = run(original, 0x1800), after = run(*folded.path, 0x1800);
  EXPECT_NE(before.runtime_next, after.runtime_next);
}

TEST(ImageValues, DeclinesPublishNothing) {
  auto budget = Plenty();
  const auto input = Dispatch();
  ir::Path last(std::vector<ir::Group>(input.sources().begin(), input.sources().end()),
                std::vector<ir::Node>(input.nodes().begin(), input.nodes().end()),
                std::vector<ir::Origin>(input.origins().begin(), input.origins().end()),
                std::vector<ir::Boundary>(input.boundaries().begin(), input.boundaries().end()),
                UINT64_MAX);
  const auto overflow = FoldImageValues(last, {}, budget);
  EXPECT_FALSE(overflow.path);
  EXPECT_EQ(overflow.reason, ImageDecline::revision_overflow);
  Budget tight({4, UINT64_MAX});
  const auto starved = FoldImageValues(input, {}, tight);
  EXPECT_FALSE(starved.path);
  EXPECT_TRUE(starved.journal.empty());
  EXPECT_EQ(starved.reason, ImageDecline::resource_limit);
}

TEST(ImageValues, APathThatWritesADeclaredRangeFoldsNothingThroughIt) {
  const std::array<ir::ConstantImageRange, 1> ranges{ir::ConstantImageRange{0x2010, kTable}};
  const ir::ImageFacts facts{ranges, {}, true, {}};

  // The negative control first: the same store, two bytes below the range, is
  // no contradiction, and everything still folds.
  auto budget = Plenty();
  const auto clear = FoldImageValues(Writing(0x200e), facts, budget);
  ASSERT_TRUE(clear.path);
  EXPECT_TRUE(clear.contradicted.empty());
  EXPECT_EQ(Target(*clear.path).op, Op::image_address);
  EXPECT_EQ(Target(*clear.path).immediate, 0x100 + 0x20 * 4);
  const auto rested = std::count_if(clear.journal.begin(), clear.journal.end(),
                                    [](const ImageEdit& edit) { return edit.constant_bytes; });
  EXPECT_EQ(rested, 3);

  // Now a store the path places inside the range it reads. The declaration says
  // the bytes keep their file value and this path says otherwise; the range is
  // dropped, so the read has no value and the jump keeps its expression.
  for (const auto written : {UINT64_C(0x2010), UINT64_C(0x2012), UINT64_C(0x200f)}) {
    const auto result = FoldImageValues(Writing(written), facts, budget);
    ASSERT_TRUE(result.path) << written;
    ASSERT_EQ(result.contradicted.size(), 1) << written;
    EXPECT_EQ(result.contradicted[0].range, 0x2010);
    EXPECT_EQ(result.contradicted[0].address, written);
    EXPECT_EQ(result.contradicted[0].width, 16);
    EXPECT_EQ(result.path->nodes()[result.contradicted[0].store].op, Op::store);
    EXPECT_NE(Target(*result.path).op, Op::image_address);

    // Only what rested on those bytes is gone; the placement folds remain, and
    // the store and the read both still execute.
    EXPECT_TRUE(std::none_of(result.journal.begin(), result.journal.end(),
                             [](const ImageEdit& edit) { return edit.constant_bytes; }));
    EXPECT_FALSE(result.journal.empty());
    EXPECT_EQ(std::count_if(result.path->nodes().begin(), result.path->nodes().end(),
                            [](const ir::Node& node) { return node.op == Op::load; }),
              1);
    EXPECT_EQ(std::count_if(result.path->nodes().begin(), result.path->nodes().end(),
                            [](const ir::Node& node) { return node.op == Op::store; }),
              1);
  }

  // A store whose address the path cannot place is not a contradiction: the
  // declaration is exactly what covers the stores this analysis cannot see.
  const std::vector<ir::Group> opaque{
      ir::Group(0xfc, {9, 9, 9, 9},
                {{Op::read, 64, {}, 0, 3}, {Op::constant, 16, {}, 0}, {Op::store, 16, {0, 1}}}, {},
                ir::MemoryModel::atomic_scalar_reference,
                ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}}),
      ir::Group(0x100, {1, 2, 3, 4},
                {{Op::image_address, 64, {}, 0x2012},
                 {Op::load, 16, {0}},
                 {Op::zext, 64, {1}},
                 {Op::constant, 64, {}, 2},
                 {Op::shl, 64, {2, 3}},
                 {Op::image_address, 64, {}, 0x100},
                 {Op::add, 64, {5, 4}}},
                {}, ir::MemoryModel::atomic_scalar_reference,
                ir::Transfer{ir::TransferKind::jump, 6, {}, {}, {}})};
  auto normalized = ir::NormalizePath(opaque, budget);
  ASSERT_TRUE(normalized.path);
  const auto unplaced = FoldImageValues(*normalized.path, facts, budget);
  ASSERT_TRUE(unplaced.path);
  EXPECT_TRUE(unplaced.contradicted.empty());
  EXPECT_EQ(Target(*unplaced.path).op, Op::image_address);
}

TEST(ImageValues, EveryBudgetCutOnAContradictedPathPublishesNothing) {
  const std::array<ir::ConstantImageRange, 1> ranges{ir::ConstantImageRange{0x2010, kTable}};
  const ir::ImageFacts facts{ranges, {}, true, {}};
  const auto input = Writing(0x2010);
  auto budget = Plenty();
  ASSERT_EQ(FoldImageValues(input, facts, budget).contradicted.size(), 1);
  const auto required = budget.used();
  for (std::uint64_t work = 0; work < required.work; ++work) {
    Budget limited({work, UINT64_MAX});
    const auto result = FoldImageValues(input, facts, limited);
    ASSERT_FALSE(result.path) << work;
    ASSERT_EQ(result.reason, ImageDecline::resource_limit);
    ASSERT_TRUE(result.contradicted.empty());
  }

  Budget exact(required);
  EXPECT_EQ(FoldImageValues(input, facts, exact).contradicted.size(), 1);
}

}  // namespace
}  // namespace nyx::recovery
