#include <array>
#include <tuple>

#include <gtest/gtest.h>

#include "nyx/analysis/control.hpp"

namespace nyx::analysis {
namespace {
using ir::Node;
using ir::Op;
using ir::Transfer;
using ir::TransferKind;

Budget Plenty() { return Budget({10000000, 10000000}); }

ir::Block Make(std::vector<Node> nodes, std::optional<Transfer> transfer,
               ir::MemoryModel model = ir::MemoryModel::unspecified) {
  const std::vector<ir::Group> sources{
      ir::Group(0x1000, {1, 2, 3, 4}, std::move(nodes), {}, model, transfer)};
  auto budget = Plenty();
  auto result = ir::Normalize(sources, budget);
  EXPECT_TRUE(result.block);
  return std::move(*result.block);
}

TEST(ControlAnalysis, EachLiftedFormOfUnsignedAboveBoundsTheFalseArm) {
  using Bound = std::optional<std::pair<ir::ValueId, std::uint64_t>>;

  // AArch64 HI: not carry and not zero of x - 3.
  const std::vector<Node> flags{
      {Op::read, 64, {}, 0, 5}, {Op::constant, 64, {}, 3}, {Op::unsigned_less, 1, {0, 1}},
      {Op::sub, 64, {0, 1}},    {Op::constant, 64, {}, 0}, {Op::equal, 1, {3, 4}},
      {Op::bit_not, 1, {2}},    {Op::bit_not, 1, {5}},     {Op::bit_and, 1, {6, 7}}};
  EXPECT_EQ(BoundFromGuard(flags, 8), (Bound{{0, 3}}));

  // 3 <u x, and not (x <u 4): both are x >u 3.
  const std::vector<Node> above{
      {Op::read, 64, {}, 0, 5},  {Op::constant, 64, {}, 3},      {Op::unsigned_less, 1, {1, 0}},
      {Op::constant, 64, {}, 4}, {Op::unsigned_less, 1, {0, 3}}, {Op::bit_not, 1, {4}},
      {Op::constant, 64, {}, 0}, {Op::unsigned_less, 1, {0, 6}}, {Op::bit_not, 1, {7}}};
  EXPECT_EQ(BoundFromGuard(above, 2), (Bound{{0, 3}}));
  EXPECT_EQ(BoundFromGuard(above, 5), (Bound{{0, 3}}));

  // x <u 4 bounds its true arm, not its false one; not (x <u 0) bounds nothing.
  EXPECT_FALSE(BoundFromGuard(above, 4));
  EXPECT_FALSE(BoundFromGuard(above, 8));
}

TEST(ControlAnalysis, ImageWriteScanSeparatesPlacedStoresFromLostImageAddresses) {
  // A store to a literal image location is placed; a store through an image
  // address the abstraction loses is counted; a stack store is neither, or the
  // count would be every spill in the program.
  const auto block = Make({{Op::image_address, 64, {}, 0x2000},
                           {Op::constant, 64, {}, 8},
                           {Op::add, 64, {0, 1}},
                           {Op::read, 64, {}, 0, 0},
                           {Op::store, 64, {2, 3}},    // placed: 0x2008
                           {Op::bit_xor, 64, {2, 3}},  // image provenance, no location
                           {Op::store, 64, {5, 3}},    // unresolved
                           {Op::read, 64, {}, 0, 31},
                           {Op::constant, 64, {}, 16},
                           {Op::sub, 64, {7, 8}},
                           {Op::store, 64, {9, 3}}},  // stack: neither
                          Transfer{TransferKind::return_, 3, {}, {}, {}},
                          ir::MemoryModel::atomic_scalar_reference);
  auto budget = Plenty();
  const auto scan = KnownImageWrites(block, {}, budget);
  ASSERT_TRUE(scan);
  ASSERT_EQ(scan->writes.size(), 1U);
  EXPECT_EQ(scan->writes[0].address, 0x2008U);
  EXPECT_EQ(scan->unresolved, 1U);
}

TEST(ControlAnalysis, ImageProvenanceStopsAtALoadedValue) {
  // What an image address points at is data. A store through a loaded value is
  // not evidence that the image was written, or every indirect store would be.
  const auto block = Make({{Op::image_address, 64, {}, 0x2000},
                           {Op::load, 64, {0}},
                           {Op::read, 64, {}, 0, 0},
                           {Op::store, 64, {1, 2}}},
                          Transfer{TransferKind::return_, 2, {}, {}, {}},
                          ir::MemoryModel::atomic_scalar_reference);
  auto budget = Plenty();
  const auto scan = KnownImageWrites(block, {}, budget);
  ASSERT_TRUE(scan);
  EXPECT_TRUE(scan->writes.empty());
  EXPECT_EQ(scan->unresolved, 0U);
}

TEST(ControlAnalysis, ImageWriteScanReadsANumberAsTheLocationTheBiasPutsThere) {
  // Under a bias of 0x10000, the number 0x12000 is @0x2000, and a slot read at
  // that number holds what the slot holds: here @0x3000.
  const auto block = Make({{Op::constant, 64, {}, 0x12000},
                           {Op::read, 64, {}, 0, 0},
                           {Op::store, 64, {0, 1}},
                           {Op::load, 64, {0}},
                           {Op::store, 64, {3, 1}},
                           {Op::constant, 64, {}, 0x3000},
                           {Op::store, 64, {5, 1}}},  // below the bias: no place
                          Transfer{TransferKind::return_, 1, {}, {}, {}},
                          ir::MemoryModel::atomic_scalar_reference);
  const ir::RelocatedPointer slot{0x2000, 0x3000, true};
  auto budget = Plenty();
  const auto scan = KnownImageWrites(block, {{}, std::span(&slot, 1), false, 0x10000}, budget);
  ASSERT_TRUE(scan);
  ASSERT_EQ(scan->writes.size(), 2U);
  EXPECT_EQ(scan->writes[0].address, 0x2000U);
  EXPECT_EQ(scan->writes[1].address, 0x3000U);

  // Without a bias a number names no place.
  const auto unplaced = KnownImageWrites(block, {{}, std::span(&slot, 1), false, {}}, budget);
  ASSERT_TRUE(unplaced);
  EXPECT_TRUE(unplaced->writes.empty());
}

TEST(ControlAnalysis, AStorePlacedOnReadOnlyBytesRefutesThem) {
  // Under a bias of zero a null store is a store to @0. The loader maps it
  // without write permission, but a store the graph places there is evidence
  // against the declaration all the same.
  const auto block = Make(
      {{Op::constant, 64, {}, 0}, {Op::read, 64, {}, 0, 0}, {Op::store, 64, {0, 1}}},
      Transfer{TransferKind::return_, 1, {}, {}, {}}, ir::MemoryModel::atomic_scalar_reference);
  constexpr std::array<std::uint8_t, 16> bytes{};
  auto budget = Plenty();
  for (const bool read_only : {true, false}) {
    const ir::ConstantImageRange range{0, bytes, read_only};
    const ImageFacts facts{std::span(&range, 1), {}, false, 0};
    const auto scan = KnownImageWrites(block, facts, budget);
    ASSERT_TRUE(scan);
    ASSERT_EQ(scan->writes.size(), 1U);
    const auto refuted = ir::RefuteImageFacts(facts, scan->writes, budget);
    ASSERT_TRUE(refuted);
    EXPECT_EQ(refuted->constants.size(), 1U) << read_only;
  }
}

TEST(ControlAnalysis, ImageAffineOffsetsStayDistinctFromAbsoluteRuntimeValues) {
  const auto block = Make({{Op::image_address, 64, {}, 0x2000},
                           {Op::constant, 64, {}, UINT64_MAX - 3},
                           {Op::add, 64, {0, 1}}},
                          Transfer{TransferKind::jump, 2, {}, {}, {}});
  auto budget = Plenty();
  const auto result = AnalyzeControl(block, budget);
  ASSERT_TRUE(result.facts);
  ASSERT_EQ(result.facts->edge_count, 1);
  EXPECT_EQ(result.facts->edges[0].target.kind, TargetKind::image_location);
  EXPECT_EQ(result.facts->edges[0].target.address, 0x1ffc);
  EXPECT_EQ(result.facts->edges[0].target.value, 2);
  EXPECT_EQ(result.facts->edges[0].role, EdgeRole::branch);
  EXPECT_FALSE(result.facts->edges[0].condition);
  const auto absolute =
      Make({{Op::constant, 64, {}, 0x2000}}, Transfer{TransferKind::jump, 0, {}, {}, {}});
  const auto second = AnalyzeControl(absolute, budget);
  ASSERT_TRUE(second.facts);
  EXPECT_EQ(second.facts->edges[0].target.kind, TargetKind::absolute_runtime);
  EXPECT_EQ(second.facts->edges[0].target.address, 0x2000);
}

TEST(ControlAnalysis, AffineSubtractionCancelsOnlyMatchingBiasCoefficients) {
  for (bool reverse : {false, true}) {
    const auto block = Make({{Op::image_address, 64, {}, 0x2000},
                             {Op::constant, 64, {}, 0x10},
                             {Op::sub, 64, {reverse ? 1U : 0U, reverse ? 0U : 1U}}},
                            Transfer{TransferKind::jump, 2, {}, {}, {}});
    auto budget = Plenty();
    const auto result = AnalyzeControl(block, budget);
    ASSERT_TRUE(result.facts);
    EXPECT_EQ(result.facts->edges[0].target.kind,
              reverse ? TargetKind::unknown : TargetKind::image_location);
    if (!reverse) {
      EXPECT_EQ(result.facts->edges[0].target.address, 0x1ff0);
    }
  }

  const auto difference = Make({{Op::image_address, 64, {}, 0x2000},
                                {Op::image_address, 64, {}, 0x2100},
                                {Op::sub, 64, {0, 1}}},
                               Transfer{TransferKind::jump, 2, {}, {}, {}});
  auto budget = Plenty();
  const auto result = AnalyzeControl(difference, budget);
  ASSERT_TRUE(result.facts);
  EXPECT_EQ(result.facts->edges[0].target.kind, TargetKind::absolute_runtime);
  EXPECT_EQ(result.facts->edges[0].target.address, UINT64_MAX - 255);
}

TEST(ControlAnalysis, PageMaskCannotAssumeLowLoadBiasBitsAreZero) {
  const auto block = Make({{Op::image_address, 64, {}, 0x1234},
                           {Op::constant, 64, {}, ~UINT64_C(4095)},
                           {Op::bit_and, 64, {0, 1}},
                           {Op::constant, 64, {}, 0x68},
                           {Op::add, 64, {2, 3}}},
                          Transfer{TransferKind::jump, 4, {}, {}, {}});
  auto budget = Plenty();
  const auto result = AnalyzeControl(block, budget);
  ASSERT_TRUE(result.facts);
  EXPECT_EQ(result.facts->edges[0].target.kind, TargetKind::unknown);
  EXPECT_EQ(result.facts->edges[0].target.value, 4);
  constexpr std::uint64_t bias = 0x1800;
  EXPECT_NE(((bias + 0x1234) & ~UINT64_C(4095)) + 0x68, bias + 0x1068);
}

// A four-entry table of 16-bit offsets, as a dispatcher would index.
constexpr std::array<std::uint8_t, 8> kTable = {0x10, 0x00, 0x20, 0x00, 0x30, 0x00, 0x40, 0x00};

ir::Block TableDispatch(std::uint64_t index, unsigned width = 16,
                        ir::ByteOrder order = ir::ByteOrder::little) {
  return Make({{Op::image_address, 64, {}, 0x5000},
               {Op::constant, 64, {}, index * 2},
               {Op::add, 64, {0, 1}},
               {Op::load, width, {2}, 0, 0, {order, 1}},
               {Op::zext, 64, {3}},
               {Op::image_address, 64, {}, 0x9000},
               {Op::add, 64, {5, 4}}},
              Transfer{TransferKind::jump, 6, {}, {}, {}},
              ir::MemoryModel::atomic_scalar_reference);
}

TEST(ControlAnalysis, ConstantImageBytesResolveATableTargetAndDeclareTheDependency) {
  const std::array<ConstantImageRange, 1> ranges{ConstantImageRange{0x5000, kTable}};
  for (std::uint64_t index = 0; index < 4; ++index) {
    auto budget = Plenty();
    const auto block = TableDispatch(index);

    // Without the bytes there is nothing to read and the target stays unknown.
    const auto blind = AnalyzeControl(block, budget);
    ASSERT_TRUE(blind.facts);
    EXPECT_EQ(blind.facts->edges[0].target.kind, TargetKind::unknown);
    EXPECT_FALSE(blind.facts->edges[0].target.constant_image_dependency);

    const auto result = AnalyzeControl(block, budget, {}, ImageFacts{ranges, {}, false, {}});
    ASSERT_TRUE(result.facts);
    EXPECT_EQ(result.facts->edges[0].target.kind, TargetKind::image_location);
    EXPECT_EQ(result.facts->edges[0].target.address, 0x9000 + 0x10 * (index + 1));
    EXPECT_TRUE(result.facts->edges[0].target.constant_image_dependency);
  }
}

TEST(ControlAnalysis, ConstantImageReadsNeverLeaveTheirDeclaredRange) {
  const std::array<ConstantImageRange, 1> ranges{ConstantImageRange{0x5000, kTable}};

  // The last entry ends exactly at the range end; one past it must decline, as
  // must a width the range cannot hold and an address below the range.
  for (const auto [index, width, resolves] :
       std::initializer_list<std::tuple<std::uint64_t, unsigned, bool>>{
           {3, 16, true}, {4, 16, false}, {3, 32, false}, {0, 64, true}, {1, 64, false}}) {
    auto budget = Plenty();
    const auto result =
        AnalyzeControl(TableDispatch(index, width), budget, {}, ImageFacts{ranges, {}, false, {}});
    ASSERT_TRUE(result.facts);
    EXPECT_EQ(result.facts->edges[0].target.kind != TargetKind::unknown, resolves)
        << "index " << index << " width " << width;
  }
}

TEST(ControlAnalysis, ConstantImageReadsHonourTheDeclaredByteOrder) {
  const std::array<ConstantImageRange, 1> ranges{ConstantImageRange{0x5000, kTable}};
  auto budget = Plenty();
  const auto result = AnalyzeControl(TableDispatch(1, 16, ir::ByteOrder::big), budget, {},
                                     ImageFacts{ranges, {}, false, {}});
  ASSERT_TRUE(result.facts);

  // Bytes 20 00 are 0x0020 little-endian and 0x2000 big-endian.
  EXPECT_EQ(result.facts->edges[0].target.address, 0x9000 + 0x2000);
}

// Flattening hides an original conditional branch behind the table index, so
// the destination rests on one choice rather than on none.
ir::Block ChosenIndex(std::vector<Node> extra, ir::ValueId chosen) {
  std::vector<Node> nodes{{Op::read, 1, {}, 0, 100},
                          {Op::constant, 64, {}, 0},
                          {Op::constant, 64, {}, 2},
                          {Op::constant, 64, {}, 4},
                          {Op::constant, 64, {}, 6}};
  nodes.insert(nodes.end(), extra.begin(), extra.end());
  const auto base = static_cast<ir::ValueId>(nodes.size());
  nodes.push_back({Op::image_address, 64, {}, 0x5000});
  nodes.push_back({Op::add, 64, {base, chosen}});
  nodes.push_back({Op::load, 16, {static_cast<ir::ValueId>(base + 1)}});
  nodes.push_back({Op::zext, 64, {static_cast<ir::ValueId>(base + 2)}});
  nodes.push_back({Op::image_address, 64, {}, 0x9000});
  nodes.push_back(
      {Op::add, 64, {static_cast<ir::ValueId>(base + 4), static_cast<ir::ValueId>(base + 3)}});
  const auto last = static_cast<ir::ValueId>(nodes.size() - 1);
  return Make(std::move(nodes), Transfer{TransferKind::jump, last, {}, {}, {}},
              ir::MemoryModel::atomic_scalar_reference);
}

TEST(ControlAnalysis, OneUnresolvedChoiceGivesTwoGuardedDestinations) {
  const std::array<ConstantImageRange, 1> ranges{ConstantImageRange{0x5000, kTable}};

  // select(read, 2, 6) picks table entry 1 or entry 3.
  const auto block = ChosenIndex({{Op::select, 64, {0, 2, 4}}}, 5);
  auto budget = Plenty();
  const auto result = AnalyzeControl(block, budget, {}, ImageFacts{ranges, {}, false, {}});
  ASSERT_TRUE(result.facts);
  ASSERT_EQ(result.facts->edge_count, 2);
  const auto& taken = result.facts->edges[0];
  const auto& other = result.facts->edges[1];
  EXPECT_EQ(taken.target.kind, TargetKind::image_location);
  EXPECT_EQ(other.target.kind, TargetKind::image_location);
  EXPECT_EQ(taken.target.address, 0x9000 + 0x20);
  EXPECT_EQ(other.target.address, 0x9000 + 0x40);
  EXPECT_EQ(taken.when, true);
  EXPECT_EQ(other.when, false);
  EXPECT_EQ(taken.condition, other.condition);
  EXPECT_TRUE(taken.target.constant_image_dependency);

  // Both edges describe one transfer, so both name the same target expression.
  EXPECT_EQ(taken.target.value, other.target.value);
}

TEST(ControlAnalysis, TwoUnresolvedChoicesStayOneUnknownDestination) {
  const std::array<ConstantImageRange, 1> ranges{ConstantImageRange{0x5000, kTable}};

  // Enumerating a product of arms is a different claim from reading a branch.
  const auto block = ChosenIndex({{Op::read, 1, {}, 0, 101},
                                  {Op::select, 64, {0, 2, 4}},
                                  {Op::select, 64, {5, 1, 3}},
                                  {Op::add, 64, {6, 7}}},
                                 8);
  auto budget = Plenty();
  const auto result = AnalyzeControl(block, budget, {}, ImageFacts{ranges, {}, false, {}});
  ASSERT_TRUE(result.facts);
  ASSERT_EQ(result.facts->edge_count, 1);
  EXPECT_EQ(result.facts->edges[0].target.kind, TargetKind::unknown);
}

TEST(ControlAnalysis, AChoiceWhoseArmsDoNotBothSettleStaysUnknown) {
  const std::array<ConstantImageRange, 1> ranges{ConstantImageRange{0x5000, kTable}};

  // One arm indexes past the declared bytes, so neither destination is claimed.
  const auto block = ChosenIndex({{Op::constant, 64, {}, 64}, {Op::select, 64, {0, 2, 5}}}, 6);
  auto budget = Plenty();
  const auto result = AnalyzeControl(block, budget, {}, ImageFacts{ranges, {}, false, {}});
  ASSERT_TRUE(result.facts);
  ASSERT_EQ(result.facts->edge_count, 1);
  EXPECT_EQ(result.facts->edges[0].target.kind, TargetKind::unknown);
}

// A jump through a slot the loader fills from a relative relocation: the file
// bytes there are not the value the run sees.
ir::Block PointerJump(std::uint64_t slot, unsigned width = 64) {
  std::vector<Node> nodes{{Op::image_address, 64, {}, slot}, {Op::load, width, {0}}};
  if (width != 64) nodes.push_back({Op::zext, 64, {1}});
  const auto last = static_cast<ir::ValueId>(nodes.size() - 1);
  return Make(std::move(nodes), Transfer{TransferKind::jump, last, {}, {}, {}},
              ir::MemoryModel::atomic_scalar_reference);
}

TEST(ControlAnalysis, ARelocatedSlotResolvesToAnImageLocationNotToItsFileBytes) {
  const std::array<RelocatedPointer, 2> slots{RelocatedPointer{0x4000, 0x7000},
                                              RelocatedPointer{0x4008, 0x8000}};
  auto budget = Plenty();

  // Without the relocation the destination is whatever happens to be readable,
  // which is why an analysis that only reads bytes must not answer here.
  const auto blind = AnalyzeControl(PointerJump(0x4008), budget);
  ASSERT_TRUE(blind.facts);
  EXPECT_EQ(blind.facts->edges[0].target.kind, TargetKind::unknown);

  const auto result = AnalyzeControl(PointerJump(0x4008), budget, {}, ImageFacts{{}, slots, false});
  ASSERT_TRUE(result.facts);
  EXPECT_EQ(result.facts->edges[0].target.kind, TargetKind::image_location);
  EXPECT_EQ(result.facts->edges[0].target.address, 0x8000);
  EXPECT_TRUE(result.facts->edges[0].target.constant_image_dependency);
}

TEST(ControlAnalysis, ARelocatedSlotIsOnlyTheWholeSlotAtItsOwnAddress) {
  const std::array<RelocatedPointer, 1> slots{RelocatedPointer{0x4000, 0x7000}};
  const ImageFacts facts{{}, slots, false, {}};

  // A different slot, a partial read, and a read starting mid-slot are all
  // something other than the pointer the loader wrote.
  for (const auto [slot, width] : std::initializer_list<std::pair<std::uint64_t, unsigned>>{
           {0x4008, 64}, {0x4000, 32}, {0x4004, 64}}) {
    auto budget = Plenty();
    const auto result = AnalyzeControl(PointerJump(slot, width), budget, {}, facts);
    ASSERT_TRUE(result.facts);
    EXPECT_EQ(result.facts->edges[0].target.kind, TargetKind::unknown)
        << "slot " << std::hex << slot << " width " << std::dec << width;
  }
}

TEST(ControlAnalysis, PageMaskAcceptsOnlyAMaskThatClearsAtMostOnePage) {
  // Declaring page-aligned placement licenses clearing the low twelve bits and
  // no more: a wider mask depends on where the image actually landed.
  for (const auto [mask, resolves] :
       std::initializer_list<std::pair<std::uint64_t, bool>>{{~UINT64_C(0xfff), true},
                                                             {~UINT64_C(0x7ff), true},
                                                             {~UINT64_C(1), true},
                                                             {~UINT64_C(0x1fff), false},
                                                             {~UINT64_C(0xffffffff), false},
                                                             {~UINT64_C(0), false},
                                                             {0xfffffffffffff0ff, false}}) {
    const auto block = Make({{Op::image_address, 64, {}, 0x1234},
                             {Op::constant, 64, {}, mask},
                             {Op::bit_and, 64, {0, 1}}},
                            Transfer{TransferKind::jump, 2, {}, {}, {}});
    auto budget = Plenty();
    ImageFacts facts;
    facts.page_aligned_placement = true;
    const auto result = AnalyzeControl(block, budget, {}, facts);
    ASSERT_TRUE(result.facts);
    EXPECT_EQ(result.facts->edges[0].target.kind == TargetKind::image_location, resolves)
        << "mask " << std::hex << mask;
  }
}

TEST(ControlAnalysis, ConditionalEdgesKeepBothRelationsEvenAtTheSameDestination) {
  for (bool same : {false, true}) {
    const auto block = Make({{Op::read, 1, {}, 0, 100},
                             {Op::image_address, 64, {}, 0x2000},
                             {Op::image_address, 64, {}, same ? 0x2000U : 0x3000U}},
                            Transfer{TransferKind::conditional, 1, 0, 2, {}});
    auto budget = Plenty();
    const auto result = AnalyzeControl(block, budget);
    ASSERT_TRUE(result.facts);
    ASSERT_EQ(result.facts->edge_count, 2);
    EXPECT_EQ(result.facts->edges[0].condition, 0);
    EXPECT_EQ(result.facts->edges[1].condition, 0);
    EXPECT_EQ(result.facts->edges[0].when, true);
    EXPECT_EQ(result.facts->edges[1].when, false);
    EXPECT_EQ(result.facts->edges[0].target.address, 0x2000);
    EXPECT_EQ(result.facts->edges[1].target.address, same ? 0x2000 : 0x3000);
  }
}

TEST(ControlAnalysis, TargetSelectKeepsPolarityAndKnownConditionUsesOnlyLowBit) {
  for (unsigned condition = 0; condition < 3; ++condition) {
    const Node predicate = condition == 2 ? Node{Op::read, 1, {}, 0, 100}
                                          : Node{Op::constant, 1, {}, UINT64_MAX - 1 + condition};
    const auto block = Make({predicate,
                             {Op::image_address, 64, {}, 0x2000},
                             {Op::constant, 64, {}, 0x3000},
                             {Op::select, 64, {0, 1, 2}}},
                            Transfer{TransferKind::jump, 3, {}, {}, {}});
    auto budget = Plenty();
    const auto result = AnalyzeControl(block, budget);
    ASSERT_TRUE(result.facts);
    if (condition == 2) {
      ASSERT_EQ(result.facts->edge_count, 2);
      EXPECT_EQ(result.facts->edges[0].when, true);
      EXPECT_EQ(result.facts->edges[0].target.kind, TargetKind::image_location);
      EXPECT_EQ(result.facts->edges[1].when, false);
      EXPECT_EQ(result.facts->edges[1].target.kind, TargetKind::absolute_runtime);
    } else {
      ASSERT_EQ(result.facts->edge_count, 1);
      EXPECT_FALSE(result.facts->edges[0].condition);
      EXPECT_EQ(result.facts->edges[0].target.address, condition ? 0x2000 : 0x3000);
    }
  }
}

TEST(ControlAnalysis, CallsExposePotentialContinuationWithoutAssumingReturn) {
  const auto original = Make({{Op::read, 64, {}, 0, 30}, {Op::image_address, 64, {}, 0x1004}},
                             Transfer{TransferKind::call, 0, {}, {}, 1});
  const ir::Block block({original.sources().begin(), original.sources().end()},
                        {original.nodes().begin(), original.nodes().end()},
                        {original.origins().begin(), original.origins().end()},
                        {original.boundaries().begin(), original.boundaries().end()}, 17);
  auto budget = Plenty();
  const auto result = AnalyzeControl(block, budget);
  ASSERT_TRUE(result.facts);
  EXPECT_EQ(result.facts->block_revision, 17);
  EXPECT_EQ(result.facts->terminal_source, 0x1000);
  EXPECT_EQ(result.facts->proof, ControlProof::all_inputs_successful_terminal);
  EXPECT_TRUE(result.facts->callee_return_unknown);
  ASSERT_EQ(result.facts->edge_count, 2);
  EXPECT_EQ(result.facts->edges[0].role, EdgeRole::callee);
  EXPECT_EQ(result.facts->edges[0].target.kind, TargetKind::unknown);
  EXPECT_EQ(result.facts->edges[1].role, EdgeRole::potential_return);
  EXPECT_EQ(result.facts->edges[1].target.address, 0x1004);
  const auto returning =
      Make({{Op::read, 64, {}, 0, 30}}, Transfer{TransferKind::return_, 0, {}, {}, {}});
  const auto ret = AnalyzeControl(returning, budget);
  ASSERT_TRUE(ret.facts);
  ASSERT_EQ(ret.facts->edge_count, 1);
  EXPECT_EQ(ret.facts->edges[0].role, EdgeRole::return_);
  EXPECT_EQ(ret.facts->edges[0].target.kind, TargetKind::unknown);
}

TEST(ControlAnalysis, LoadTargetsAndNestedSelectorsRemainExplicitlyUnknown) {
  const auto block =
      Make({{Op::constant, 64, {}, 0x4000},
            {Op::load, 64, {0}},
            {Op::read, 1, {}, 0, 100},
            {Op::image_address, 64, {}, 0x2000},
            {Op::select, 64, {2, 1, 3}},
            {Op::constant, 64, {}, 4},
            {Op::add, 64, {4, 5}}},
           Transfer{TransferKind::jump, 6, {}, {}, {}}, ir::MemoryModel::atomic_scalar_reference);
  auto budget = Plenty();
  const auto result = AnalyzeControl(block, budget);
  ASSERT_TRUE(result.facts);
  ASSERT_EQ(result.facts->edge_count, 1);
  EXPECT_EQ(result.facts->edges[0].target.kind, TargetKind::unknown);
  EXPECT_EQ(result.facts->edges[0].target.value, 6);
}

TEST(ControlAnalysis, WidthTruncationAndDoubleBiasDoNotBecomeImageLocations) {
  for (bool narrow : {false, true}) {
    std::vector<Node> nodes{{Op::image_address, 64, {}, 0x2000}};
    if (narrow) {
      nodes.push_back({Op::extract, 32, {0}, 0});
      nodes.push_back({Op::zext, 64, {1}});
    } else {
      nodes.push_back({Op::image_address, 64, {}, 0x1000});
      nodes.push_back({Op::add, 64, {0, 1}});
    }

    const auto block = Make(std::move(nodes), Transfer{TransferKind::jump, 2, {}, {}, {}});
    auto budget = Plenty();
    const auto result = AnalyzeControl(block, budget);
    ASSERT_TRUE(result.facts);
    EXPECT_EQ(result.facts->edges[0].target.kind, TargetKind::unknown);
  }
}

TEST(ControlAnalysis, InvalidIrAndMissingTransferHaveNoFacts) {
  const auto block = Make({{Op::constant, 64, {}, 1}}, {});
  auto budget = Plenty();
  const auto result = AnalyzeControl(block, budget);
  EXPECT_FALSE(result.facts);
  EXPECT_EQ(result.reason, ControlDecline::no_transfer);
  const ir::Block invalid({}, {}, {}, {});
  const auto rejected = AnalyzeControl(invalid, budget);
  EXPECT_FALSE(rejected.facts);
  EXPECT_EQ(rejected.reason, ControlDecline::invalid_ir);
}

TEST(ControlAnalysis, EveryBudgetCutPublishesNoPartialFacts) {
  const auto block = Make({{Op::read, 1, {}, 0, 100},
                           {Op::image_address, 64, {}, 0x2000},
                           {Op::constant, 64, {}, 0x3000},
                           {Op::select, 64, {0, 1, 2}},
                           {Op::image_address, 64, {}, 0x1004}},
                          Transfer{TransferKind::call, 3, {}, {}, 4});
  auto full = Plenty();
  const auto complete = AnalyzeControl(block, full);
  ASSERT_TRUE(complete.facts);
  EXPECT_EQ(complete.facts->edge_count, 3);
  for (bool bytes : {false, true}) {
    const auto bound = bytes ? full.used().bytes : full.used().work;
    for (std::uint64_t cut = 0; cut < bound; ++cut) {
      Budget budget({bytes ? full.used().work : cut, bytes ? cut : full.used().bytes});
      const auto result = AnalyzeControl(block, budget);
      EXPECT_FALSE(result.facts);
      EXPECT_EQ(result.reason, ControlDecline::resource_limit);
    }
  }
}

}  // namespace
}  // namespace nyx::analysis
