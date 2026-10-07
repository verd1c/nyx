#include "nyx/analysis/path_control.hpp"

#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "nyx/recovery/control.hpp"
#include "nyx/recovery/image.hpp"

namespace nyx::analysis {
namespace {

Budget Plenty() { return Budget({UINT64_MAX, UINT64_MAX}); }

ir::Group Plain(std::uint64_t address) {
  return ir::Group(address, {1, 2, 3, 4}, {{ir::Op::constant, 64, {}, 7}}, {});
}

ir::Group Jump(std::uint64_t address, std::uint64_t target, bool image = true) {
  return ir::Group(
      address, {1, 2, 3, 4}, {{image ? ir::Op::image_address : ir::Op::constant, 64, {}, target}},
      {}, ir::MemoryModel::unspecified, ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}});
}

ir::Group Branch(std::uint64_t address, std::uint64_t yes, std::uint64_t no,
                 std::optional<bool> known = {}) {
  return ir::Group(address, {1, 2, 3, 4},
                   {{known ? ir::Op::constant : ir::Op::read, 1, {}, known.value_or(false), 1},
                    {ir::Op::image_address, 64, {}, yes},
                    {ir::Op::image_address, 64, {}, no}},
                   {}, ir::MemoryModel::unspecified,
                   ir::Transfer{ir::TransferKind::conditional, 1, 0, 2, {}});
}

ir::Path Path(std::vector<ir::Group> groups) {
  auto budget = Plenty();
  auto result = ir::NormalizePath(groups, budget);
  EXPECT_TRUE(result.path);
  return std::move(*result.path);
}

std::vector<ir::ValueId> ImageDependentNodes(const recovery::ImagePathResult& folded) {
  std::vector<ir::ValueId> nodes;
  for (const auto& edit : folded.journal)
    if (edit.constant_bytes || edit.relocated_slot) nodes.push_back(edit.node);
  return nodes;
}

TEST(PathControl, ImageFoldedTargetKeepsItsDeclaredByteDependency) {
  const auto source = ir::Group(0x100, {1, 2, 3, 4},
                                {{ir::Op::image_address, 64, {}, 0x5000},
                                 {ir::Op::load, 16, {0}},
                                 {ir::Op::zext, 64, {1}},
                                 {ir::Op::image_address, 64, {}, 0x9000},
                                 {ir::Op::add, 64, {3, 2}}},
                                {}, ir::MemoryModel::atomic_scalar_reference,
                                ir::Transfer{ir::TransferKind::jump, 4, {}, {}, {}});
  for (auto [byte, expected] : {std::pair{0x10U, 0x9010ULL}, std::pair{0x20U, 0x9020ULL}}) {
    const std::array<std::uint8_t, 2> contents{static_cast<std::uint8_t>(byte), 0};
    const std::array<ConstantImageRange, 1> ranges{ConstantImageRange{0x5000, contents}};
    const ImageFacts facts{ranges, {}, false, {}};
    auto budget = Plenty();
    auto folded = recovery::FoldImageValues(Path({source}), facts, budget);
    ASSERT_TRUE(folded.path);
    const auto revision = folded.path->revision();
    const auto dependent = ImageDependentNodes(folded);
    ASSERT_FALSE(dependent.empty());
    const auto path_facts =
        AnalyzePathControl(*folded.path, {revision, dependent}, budget, {}, facts);
    ASSERT_TRUE(path_facts.facts);
    EXPECT_TRUE(path_facts.facts->boundaries[0].edges[0].constant_image_dependency);
    auto recovered = recovery::RecoverControl(std::move(*folded.path), facts, budget);
    ASSERT_TRUE(recovered.path);
    const auto result =
        AnalyzePathControl(*recovered.path, {revision, dependent}, budget, {}, facts);
    ASSERT_TRUE(result.facts);
    ASSERT_EQ(result.facts->boundaries[0].edge_count, 1);
    EXPECT_EQ(result.facts->boundaries[0].edges[0].target_address, expected);
    EXPECT_TRUE(result.facts->boundaries[0].edges[0].constant_image_dependency);
    const auto stale =
        AnalyzePathControl(*recovered.path, {revision + 1, dependent}, budget, {}, facts);
    EXPECT_FALSE(stale.facts);
    EXPECT_EQ(stale.reason, ControlDecline::invalid_ir);
    auto unqualified = AnalyzePathControl(*recovered.path, {revision, {}}, budget, {}, facts);
    ASSERT_TRUE(unqualified.facts);
    EXPECT_FALSE(unqualified.facts->boundaries[0].edges[0].constant_image_dependency);
  }

  auto budget = Plenty();
  auto folded = recovery::FoldImageValues(Path({source}), {}, budget);
  ASSERT_TRUE(folded.path);
  EXPECT_TRUE(ImageDependentNodes(folded).empty());
  auto recovered = recovery::RecoverControl(std::move(*folded.path), {}, budget);
  ASSERT_TRUE(recovered.path);
  const auto absent =
      AnalyzePathControl(*recovered.path, {recovered.path->basis().revision(), {}}, budget);
  ASSERT_TRUE(absent.facts);
  EXPECT_EQ(absent.facts->boundaries[0].edges[0].target_kind, TargetKind::unknown);
}

TEST(PathControl, FoldedConditionKeepsItsDeclaredByteDependency) {
  const auto source = ir::Group(0x100, {1, 2, 3, 4},
                                {{ir::Op::image_address, 64, {}, 0x5000},
                                 {ir::Op::load, 8, {0}},
                                 {ir::Op::constant, 8, {}, 0},
                                 {ir::Op::equal, 1, {1, 2}},
                                 {ir::Op::image_address, 64, {}, 0x200},
                                 {ir::Op::image_address, 64, {}, 0x300}},
                                {}, ir::MemoryModel::atomic_scalar_reference,
                                ir::Transfer{ir::TransferKind::conditional, 4, 3, 5, {}});
  for (auto [byte, expected] : {std::pair{0U, 0x200ULL}, std::pair{1U, 0x300ULL}}) {
    const std::array<std::uint8_t, 1> contents{static_cast<std::uint8_t>(byte)};
    const std::array<ConstantImageRange, 1> ranges{ConstantImageRange{0x5000, contents}};
    const ImageFacts facts{ranges, {}, false, {}};
    auto budget = Plenty();
    auto folded = recovery::FoldImageValues(Path({source}), facts, budget);
    ASSERT_TRUE(folded.path);
    const auto revision = folded.path->revision();
    const auto dependent = ImageDependentNodes(folded);
    ASSERT_FALSE(dependent.empty());
    auto recovered = recovery::RecoverControl(std::move(*folded.path), facts, budget);
    ASSERT_TRUE(recovered.path);
    ASSERT_EQ(recovered.path->rewrites().size(), 1);
    EXPECT_EQ(recovered.path->rewrites()[0].rule, ir::RewriteRule::folded_condition);
    const auto result =
        AnalyzePathControl(*recovered.path, {revision, dependent}, budget, {}, facts);
    ASSERT_TRUE(result.facts);
    ASSERT_EQ(result.facts->boundaries[0].edge_count, 1);
    EXPECT_EQ(result.facts->boundaries[0].edges[0].target_address, expected);
    EXPECT_TRUE(result.facts->boundaries[0].edges[0].constant_image_dependency);
  }
}

TEST(PathControl, UnknownGuardKeepsItsFoldedByteDependency) {
  const auto source = ir::Group(0x100, {1, 2, 3, 4},
                                {{ir::Op::image_address, 64, {}, 0x5000},
                                 {ir::Op::load, 8, {0}},
                                 {ir::Op::constant, 8, {}, 1},
                                 {ir::Op::add, 8, {1, 2}},
                                 {ir::Op::read, 8, {}, 0, 1},
                                 {ir::Op::equal, 1, {3, 4}},
                                 {ir::Op::image_address, 64, {}, 0x200},
                                 {ir::Op::image_address, 64, {}, 0x300}},
                                {}, ir::MemoryModel::atomic_scalar_reference,
                                ir::Transfer{ir::TransferKind::conditional, 6, 5, 7, {}});
  const std::array<std::uint8_t, 1> contents{7};
  const std::array<ConstantImageRange, 1> ranges{ConstantImageRange{0x5000, contents}};
  const ImageFacts facts{ranges, {}, false, {}};
  auto budget = Plenty();
  auto folded = recovery::FoldImageValues(Path({source}), facts, budget);
  ASSERT_TRUE(folded.path);
  const auto revision = folded.path->revision();
  const auto dependent = ImageDependentNodes(folded);
  ASSERT_EQ(dependent.size(), 1);
  EXPECT_EQ(folded.path->nodes()[dependent[0]].op, ir::Op::constant);
  auto recovered = recovery::RecoverControl(std::move(*folded.path), facts, budget);
  ASSERT_TRUE(recovered.path);
  const auto result = AnalyzePathControl(*recovered.path, {revision, dependent}, budget, {}, facts);
  ASSERT_TRUE(result.facts);
  ASSERT_EQ(result.facts->boundaries[0].edge_count, 2);
  for (const auto& edge : result.facts->boundaries[0].edges) {
    if (!edge.condition) continue;
    EXPECT_FALSE(edge.known_condition);
    EXPECT_TRUE(edge.constant_image_dependency);
  }
}

TEST(PathControl, WidePureValuesKeepFoldedDependencyWhenTargetStaysUnknown) {
  const auto source = ir::Group(0x100, {1, 2, 3, 4},
                                {{ir::Op::image_address, 64, {}, 0x5000},
                                 {ir::Op::load, 16, {0}},
                                 {ir::Op::constant, 16, {}, 1},
                                 {ir::Op::add, 16, {1, 2}},
                                 {ir::Op::zext, 128, {3}},
                                 {ir::Op::extract, 64, {4}},
                                 {ir::Op::image_address, 64, {}, 0x9000},
                                 {ir::Op::add, 64, {6, 5}}},
                                {}, ir::MemoryModel::atomic_scalar_reference,
                                ir::Transfer{ir::TransferKind::jump, 7, {}, {}, {}});
  const std::array<std::uint8_t, 2> contents{7, 0};
  const std::array<ConstantImageRange, 1> ranges{ConstantImageRange{0x5000, contents}};
  const ImageFacts facts{ranges, {}, false, {}};
  auto budget = Plenty();
  auto folded = recovery::FoldImageValues(Path({source}), facts, budget);
  ASSERT_TRUE(folded.path);
  const auto revision = folded.path->revision();
  const auto dependent = ImageDependentNodes(folded);
  ASSERT_EQ(dependent.size(), 1);
  const ir::RecoveredPath recovered(std::move(*folded.path), {}, revision);
  const auto result = AnalyzePathControl(recovered, {revision, dependent}, budget, {}, facts);
  ASSERT_TRUE(result.facts);
  ASSERT_EQ(result.facts->boundaries[0].edge_count, 1);
  EXPECT_EQ(result.facts->boundaries[0].edges[0].target_kind, TargetKind::unknown);
  EXPECT_TRUE(result.facts->boundaries[0].edges[0].constant_image_dependency);
}

TEST(PathControl, SplitReevaluationKeepsFoldedPointerDependency) {
  const auto source = ir::Group(0x100, {1, 2, 3, 4},
                                {{ir::Op::image_address, 64, {}, 0x5000},
                                 {ir::Op::load, 64, {0}},
                                 {ir::Op::constant, 64, {}, 0},
                                 {ir::Op::add, 64, {1, 2}},
                                 {ir::Op::read, 1, {}, 0, 1},
                                 {ir::Op::constant, 64, {}, 0x10},
                                 {ir::Op::constant, 64, {}, 0x20},
                                 {ir::Op::select, 64, {4, 5, 6}},
                                 {ir::Op::add, 64, {3, 7}}},
                                {}, ir::MemoryModel::atomic_scalar_reference,
                                ir::Transfer{ir::TransferKind::jump, 8, {}, {}, {}});
  const std::array<ir::RelocatedPointer, 1> slots{ir::RelocatedPointer{0x5000, 0x9000}};
  const ImageFacts facts{{}, slots, false, {}};
  auto budget = Plenty();
  auto folded = recovery::FoldImageValues(Path({source}), facts, budget);
  ASSERT_TRUE(folded.path);
  const auto revision = folded.path->revision();
  const auto dependent = ImageDependentNodes(folded);
  ASSERT_EQ(dependent.size(), 1);
  EXPECT_EQ(folded.path->nodes()[dependent[0]].op, ir::Op::image_address);
  EXPECT_EQ(folded.path->nodes()[dependent[0]].immediate, 0x9000);
  const ir::RecoveredPath unsplit(*folded.path, {}, revision);
  const auto direct = AnalyzePathControl(unsplit, {revision, dependent}, budget, {}, facts);
  ASSERT_TRUE(direct.facts);
  ASSERT_EQ(direct.facts->boundaries[0].edge_count, 2);
  EXPECT_TRUE(direct.facts->boundaries[0].edges[0].constant_image_dependency);
  EXPECT_TRUE(direct.facts->boundaries[0].edges[1].constant_image_dependency);
  auto recovered = recovery::RecoverControl(std::move(*folded.path), facts, budget);
  ASSERT_TRUE(recovered.path);
  ASSERT_EQ(recovered.path->rewrites().size(), 1);
  EXPECT_EQ(recovered.path->rewrites()[0].rule, ir::RewriteRule::dispatch_branch);
  EXPECT_EQ(dependent[0], recovered.path->basis()
                              .nodes()[recovered.path->basis().boundaries()[0].transfer->target]
                              .inputs[0]);
  const auto result = AnalyzePathControl(*recovered.path, {revision, dependent}, budget, {}, facts);
  ASSERT_TRUE(result.facts);
  ASSERT_EQ(result.facts->boundaries[0].edge_count, 2);
  EXPECT_EQ(result.facts->boundaries[0].edges[0].target_address, 0x9010);
  EXPECT_EQ(result.facts->boundaries[0].edges[1].target_address, 0x9020);
  EXPECT_TRUE(result.facts->boundaries[0].edges[0].constant_image_dependency);
  EXPECT_TRUE(result.facts->boundaries[0].edges[1].constant_image_dependency);
}

// A four-entry table of 16-bit offsets, indexed by a value a guard bounds.
constexpr std::array<std::uint8_t, 8> kEntries = {0x10, 0x00, 0x20, 0x00, 0x30, 0x00, 0x40, 0x00};

// The lifted shape of "unsigned above a literal": carry set and zero clear
// after subtracting it. Taking the other arm is what bounds the index.
ir::Group Guard(std::uint64_t address, std::uint64_t limit, std::uint64_t taken,
                std::uint64_t fallthrough) {
  return ir::Group(address, {1, 2, 3, 4},
                   {{ir::Op::read, 32, {}, 0, 7},
                    {ir::Op::constant, 32, {}, limit},
                    {ir::Op::unsigned_less, 1, {0, 1}},
                    {ir::Op::bit_not, 1, {2}},
                    {ir::Op::sub, 32, {0, 1}},
                    {ir::Op::constant, 32, {}, 0},
                    {ir::Op::equal, 1, {4, 5}},
                    {ir::Op::bit_not, 1, {6}},
                    {ir::Op::bit_and, 1, {3, 7}},
                    {ir::Op::image_address, 64, {}, taken},
                    {ir::Op::image_address, 64, {}, fallthrough}},
                   {}, ir::MemoryModel::unspecified,
                   ir::Transfer{ir::TransferKind::conditional, 9, 8, 10, {}});
}

ir::Group TableJump(std::uint64_t address) {
  return ir::Group(address, {1, 2, 3, 4},
                   {{ir::Op::read, 32, {}, 0, 7},
                    {ir::Op::zext, 64, {0}},
                    {ir::Op::constant, 64, {}, 1},
                    {ir::Op::shl, 64, {1, 2}},
                    {ir::Op::image_address, 64, {}, 0x5000},
                    {ir::Op::add, 64, {4, 3}},
                    {ir::Op::load, 16, {5}},
                    {ir::Op::zext, 64, {6}},
                    {ir::Op::image_address, 64, {}, 0x9000},
                    {ir::Op::add, 64, {8, 7}}},
                   {}, ir::MemoryModel::atomic_scalar_reference,
                   ir::Transfer{ir::TransferKind::jump, 9, {}, {}, {}});
}

ir::Group OffsetJump(std::uint64_t address) {
  return ir::Group(address, {1, 2, 3, 4},
                   {{ir::Op::read, 32, {}, 0, 7},
                    {ir::Op::zext, 64, {0}},
                    {ir::Op::image_address, 64, {}, 0x5000},
                    {ir::Op::load, 64, {2}},
                    {ir::Op::constant, 64, {}, 0},
                    {ir::Op::add, 64, {3, 4}},
                    {ir::Op::add, 64, {5, 1}}},
                   {}, ir::MemoryModel::atomic_scalar_reference,
                   ir::Transfer{ir::TransferKind::jump, 6, {}, {}, {}});
}

ir::Group ImageBoundGuard() {
  return ir::Group(0x100, {1, 2, 3, 4},
                   {{ir::Op::read, 32, {}, 0, 7},
                    {ir::Op::image_address, 64, {}, 0x5000},
                    {ir::Op::load, 8, {1}},
                    {ir::Op::zext, 32, {2}},
                    {ir::Op::unsigned_less, 1, {0, 3}},
                    {ir::Op::bit_not, 1, {4}},
                    {ir::Op::sub, 32, {0, 3}},
                    {ir::Op::constant, 32, {}, 0},
                    {ir::Op::equal, 1, {6, 7}},
                    {ir::Op::bit_not, 1, {8}},
                    {ir::Op::bit_and, 1, {5, 9}},
                    {ir::Op::image_address, 64, {}, 0x900},
                    {ir::Op::image_address, 64, {}, 0x104}},
                   {}, ir::MemoryModel::atomic_scalar_reference,
                   ir::Transfer{ir::TransferKind::conditional, 11, 10, 12, {}});
}

ir::Group ArithmeticJump(std::uint64_t address) {
  return ir::Group(address, {1, 2, 3, 4},
                   {{ir::Op::read, 32, {}, 0, 7},
                    {ir::Op::zext, 64, {0}},
                    {ir::Op::image_address, 64, {}, 0x9000},
                    {ir::Op::add, 64, {2, 1}}},
                   {}, ir::MemoryModel::unspecified,
                   ir::Transfer{ir::TransferKind::jump, 3, {}, {}, {}});
}

TEST(PathControl, BoundedSetKeepsItsFoldedGuardDependency) {
  const std::array<std::uint8_t, 1> contents{3};
  const std::array<ConstantImageRange, 1> ranges{ConstantImageRange{0x5000, contents}};
  const ImageFacts facts{ranges, {}, false, {}};
  auto budget = Plenty();
  auto folded =
      recovery::FoldImageValues(Path({ImageBoundGuard(), ArithmeticJump(0x104)}), facts, budget);
  ASSERT_TRUE(folded.path);
  const auto revision = folded.path->revision();
  const auto dependent = ImageDependentNodes(folded);
  ASSERT_EQ(dependent.size(), 1);
  const ir::RecoveredPath recovered(std::move(*folded.path), {}, revision);
  const auto result = AnalyzePathControl(recovered, {revision, dependent}, budget, {}, facts);
  ASSERT_TRUE(result.facts);
  ASSERT_EQ(result.facts->boundaries.size(), 2);
  const auto& jump = result.facts->boundaries[1];
  ASSERT_TRUE(jump.dispatch);
  EXPECT_EQ(jump.dispatch->destinations,
            (std::vector<std::uint64_t>{0x9000, 0x9001, 0x9002, 0x9003}));
  EXPECT_TRUE(jump.dispatch->constant_image_dependency);
}

TEST(PathControl, BoundedReevaluationKeepsFoldedPointerDependency) {
  const std::array<ir::RelocatedPointer, 1> slots{ir::RelocatedPointer{0x5000, 0x9000}};
  const ImageFacts facts{{}, slots, false, {}};
  auto budget = Plenty();
  auto folded = recovery::FoldImageValues(Path({Guard(0x100, 3, 0x900, 0x104), OffsetJump(0x104)}),
                                          facts, budget);
  ASSERT_TRUE(folded.path);
  const auto revision = folded.path->revision();
  const auto dependent = ImageDependentNodes(folded);
  ASSERT_EQ(dependent.size(), 1);
  const ir::RecoveredPath recovered(std::move(*folded.path), {}, revision);
  const auto result = AnalyzePathControl(recovered, {revision, dependent}, budget, {}, facts);
  ASSERT_TRUE(result.facts);
  ASSERT_EQ(result.facts->boundaries.size(), 2);
  const auto& jump = result.facts->boundaries[1];
  ASSERT_TRUE(jump.dispatch);
  EXPECT_EQ(jump.dispatch->destinations,
            (std::vector<std::uint64_t>{0x9000, 0x9001, 0x9002, 0x9003}));
  EXPECT_TRUE(jump.dispatch->constant_image_dependency);
}

TEST(PathControl, AGuardedTableDispatchReportsEveryDestinationItCanReach) {
  const std::array<ConstantImageRange, 1> ranges{ConstantImageRange{0x5000, kEntries}};
  auto path = Path({Guard(0x100, 3, 0x900, 0x104), TableJump(0x104)});
  auto budget = Plenty();
  const auto result = AnalyzePathControl(path, budget, {}, ImageFacts{ranges, {}, false});
  ASSERT_TRUE(result.facts);
  const auto& jump = result.facts->boundaries[1];

  // The destination itself stays unknown: no single successor was proved.
  EXPECT_EQ(jump.edges[0].target_kind, TargetKind::unknown);
  ASSERT_TRUE(jump.dispatch);
  EXPECT_EQ(jump.dispatch->bound, 3);
  EXPECT_EQ(jump.dispatch->guard_boundary, 0);
  EXPECT_TRUE(jump.dispatch->constant_image_dependency);
  EXPECT_EQ(jump.dispatch->destinations,
            (std::vector<std::uint64_t>{0x9010, 0x9020, 0x9030, 0x9040}));
}

TEST(PathControl, NoBoundMeansNoCompleteSuccessorSet) {
  const std::array<ConstantImageRange, 1> ranges{ConstantImageRange{0x5000, kEntries}};

  // Same dispatch, but the path never passes a guard that bounds the index.
  auto path = Path({Plain(0x100), TableJump(0x104)});
  auto budget = Plenty();
  const auto result = AnalyzePathControl(path, budget, {}, ImageFacts{ranges, {}, false});
  ASSERT_TRUE(result.facts);
  EXPECT_FALSE(result.facts->boundaries[1].dispatch);
}

TEST(PathControl, ABoundReachingPastTheTableClaimsNothing) {
  const std::array<ConstantImageRange, 1> ranges{ConstantImageRange{0x5000, kEntries}};

  // The guard admits index 4, which is past the declared bytes, so the set
  // cannot be completed and no partial set is reported.
  auto path = Path({Guard(0x100, 4, 0x900, 0x104), TableJump(0x104)});
  auto budget = Plenty();
  const auto result = AnalyzePathControl(path, budget, {}, ImageFacts{ranges, {}, false});
  ASSERT_TRUE(result.facts);
  EXPECT_FALSE(result.facts->boundaries[1].dispatch);
}

TEST(PathControl, TakingTheGuardsOtherArmProvesNoBound) {
  const std::array<ConstantImageRange, 1> ranges{ConstantImageRange{0x5000, kEntries}};

  // The itinerary follows the arm taken when the index is above the limit, so
  // that comparison bounds nothing about the path that reaches the table.
  auto path = Path({Guard(0x100, 3, 0x104, 0x900), TableJump(0x104)});
  auto budget = Plenty();
  const auto result = AnalyzePathControl(path, budget, {}, ImageFacts{ranges, {}, false});
  ASSERT_TRUE(result.facts);
  EXPECT_FALSE(result.facts->boundaries[1].dispatch);
}

// The index the guard reads, chosen by a flag between two literal states.
ir::Group ChooseIndex(std::uint64_t address, std::uint64_t first, std::uint64_t second) {
  return ir::Group(address, {1, 2, 3, 4},
                   {{ir::Op::read, 1, {}, 0, 3},
                    {ir::Op::constant, 32, {}, first},
                    {ir::Op::constant, 32, {}, second},
                    {ir::Op::select, 32, {0, 1, 2}}},
                   {{7, 3}});
}

TEST(PathControl, AGuardBothChoicesSatisfyIsDecidedAndTheJumpSplits) {
  const std::array<ConstantImageRange, 1> ranges{ConstantImageRange{0x5000, kEntries}};
  auto path = Path({ChooseIndex(0x100, 1, 2), Guard(0x104, 3, 0x900, 0x108), TableJump(0x108)});
  auto budget = Plenty();
  const auto result = AnalyzePathControl(path, budget, {}, ImageFacts{ranges, {}, false});
  ASSERT_TRUE(result.facts);
  const auto& guard = result.facts->boundaries[1];

  // Neither 1 nor 2 is above 3, so the out-of-range arm is never taken.
  EXPECT_EQ(guard.expected_match, ExpectedMatch::always);
  ASSERT_EQ(guard.edge_count, 2);
  EXPECT_EQ(guard.edges[0].known_condition, false);
  EXPECT_EQ(guard.edges[1].known_condition, false);
  EXPECT_TRUE(guard.edges[0].constant_image_dependency == false);
  const auto& jump = result.facts->boundaries[2];
  ASSERT_EQ(jump.edge_count, 2);
  EXPECT_EQ(jump.edges[0].target_address, 0x9020);
  EXPECT_EQ(jump.edges[1].target_address, 0x9030);
}

TEST(PathControl, AGuardTheChoicesDisagreeOnStaysUndecided) {
  const std::array<ConstantImageRange, 1> ranges{ConstantImageRange{0x5000, kEntries}};

  // One arm is in range and one is not, so which edge is followed depends on
  // the flag; nothing about the route may be claimed.
  auto path = Path({ChooseIndex(0x100, 1, 7), Guard(0x104, 3, 0x900, 0x108), TableJump(0x108)});
  auto budget = Plenty();
  const auto result = AnalyzePathControl(path, budget, {}, ImageFacts{ranges, {}, false});
  ASSERT_TRUE(result.facts);
  const auto& guard = result.facts->boundaries[1];
  EXPECT_EQ(guard.expected_match, ExpectedMatch::unknown);
  EXPECT_FALSE(guard.edges[0].known_condition);
}

TEST(PathControl, FallthroughAndDirectTransfersRetainSourceAndRevision) {
  auto path = Path({Plain(0x100), Jump(0x104, 0x200), Plain(0x200)});
  auto budget = Plenty();
  auto result = AnalyzePathControl(path, budget);
  ASSERT_TRUE(result.facts);
  const auto& facts = *result.facts;
  EXPECT_EQ(facts.path_revision, path.revision());
  EXPECT_EQ(facts.total_boundaries, 3);
  ASSERT_EQ(facts.boundaries.size(), 3);
  EXPECT_FALSE(facts.proved_divergence);
  EXPECT_EQ(facts.boundaries[0].edges[0].role, PathEdgeRole::fallthrough);
  EXPECT_FALSE(facts.boundaries[0].edges[0].target_value);
  EXPECT_EQ(facts.boundaries[0].expected_match, ExpectedMatch::always);
  EXPECT_EQ(facts.boundaries[1].boundary, 1);
  EXPECT_EQ(facts.boundaries[1].source_address, 0x104);
  EXPECT_EQ(facts.boundaries[1].expected_match, ExpectedMatch::always);
  EXPECT_EQ(facts.boundaries[1].transfer_target, path.boundaries()[1].transfer->target);
  EXPECT_EQ(facts.boundaries[2].expected_match, ExpectedMatch::not_applicable);
}

TEST(PathControl, KnownGuardUsesOnlyFeasibleArmAndRetainsBothPolarities) {
  for (const bool known : {false, true}) {
    auto path = Path({Branch(0x100, 0x200, 0x300, known), Plain(0x200)});
    auto budget = Plenty();
    const auto result = AnalyzePathControl(path, budget);
    ASSERT_TRUE(result.facts);
    const auto& first = result.facts->boundaries[0];
    ASSERT_EQ(first.edge_count, 2);
    EXPECT_EQ(first.edges[0].when, true);
    EXPECT_EQ(first.edges[1].when, false);
    EXPECT_EQ(first.edges[0].known_condition, known);
    EXPECT_EQ(first.edges[1].known_condition, known);
    EXPECT_EQ(first.expected_match, known ? ExpectedMatch::always : ExpectedMatch::never);
    EXPECT_EQ(result.facts->boundaries.size(), known ? 2 : 1);
    EXPECT_EQ(result.facts->proved_divergence.has_value(), !known);
  }
}

TEST(PathControl, UnknownGuardRequiresBothArmsForAlwaysOrNever) {
  for (unsigned scenario = 0; scenario < 3; ++scenario) {
    const auto yes = scenario == 2 ? 0x300 : 0x200;
    const auto no = scenario == 0 ? 0x200 : 0x300;
    auto path = Path({Branch(0x100, yes, no), Plain(0x200)});
    auto budget = Plenty();
    const auto result = AnalyzePathControl(path, budget);
    ASSERT_TRUE(result.facts);
    const auto& first = result.facts->boundaries[0];
    EXPECT_EQ(first.expected_match, scenario == 0   ? ExpectedMatch::always
                                    : scenario == 1 ? ExpectedMatch::unknown
                                                    : ExpectedMatch::never);
    EXPECT_EQ(first.edge_count, 2);
    EXPECT_FALSE(first.edges[0].known_condition);
    EXPECT_EQ(first.edges[0].condition, first.edges[1].condition);
    EXPECT_NE(first.edges[0].when, first.edges[1].when);
  }
}

TEST(PathControl, AbsoluteRuntimeTargetCannotBeComparedWithoutPlacement) {
  for (const auto target : {UINT64_C(0x200), UINT64_C(0x400)}) {
    auto path = Path({Jump(0x100, target, false), Plain(0x200)});
    auto budget = Plenty();
    const auto result = AnalyzePathControl(path, budget);
    ASSERT_TRUE(result.facts);
    EXPECT_EQ(result.facts->boundaries[0].edges[0].target_kind, TargetKind::absolute_runtime);
    EXPECT_EQ(result.facts->boundaries[0].expected_match, ExpectedMatch::unknown);
    EXPECT_EQ(result.facts->boundaries.size(), 2);
  }
}

TEST(PathControl, KeepsOriginalSelectAndCrossBoundaryLoadDependencies) {
  auto producer = ir::Group(0x100, {1, 2, 3, 4},
                            {{ir::Op::constant, 64, {}, 0x800},
                             {ir::Op::load, 64, {0}},
                             {ir::Op::constant, 64, {}, 0x900},
                             {ir::Op::load, 64, {2}}},
                            {{8, 1}}, ir::MemoryModel::atomic_scalar_reference);
  auto select = ir::Group(0x104, {1, 2, 3, 4},
                          {{ir::Op::read, 1, {}, 0, 1},
                           {ir::Op::read, 64, {}, 0, 8},
                           {ir::Op::image_address, 64, {}, 0x200},
                           {ir::Op::select, 64, {0, 1, 2}}},
                          {}, ir::MemoryModel::unspecified,
                          ir::Transfer{ir::TransferKind::jump, 3, {}, {}, {}});
  auto path = Path({std::move(producer), std::move(select), Plain(0x200)});
  auto budget = Plenty();
  const auto result = AnalyzePathControl(path, budget);
  ASSERT_TRUE(result.facts);
  const auto& control = result.facts->boundaries[1];
  ASSERT_EQ(control.edge_count, 2);
  EXPECT_EQ(control.transfer_target, path.boundaries()[1].transfer->target);
  EXPECT_EQ(path.nodes()[*control.transfer_target].op, ir::Op::select);
  EXPECT_EQ(control.edges[0].target_kind, TargetKind::unknown);
  EXPECT_EQ(control.edges[1].target_kind, TargetKind::image_location);
  EXPECT_EQ(control.expected_match, ExpectedMatch::unknown);
  EXPECT_EQ(control.load_dependencies, (std::vector<ir::ValueId>{1}));
  EXPECT_EQ(path.nodes()[3].op, ir::Op::load);
  EXPECT_EQ(path.sources()[0].nodes().size(), 4);
}

TEST(PathControl, TerminalCallKeepsCalleeAndPotentialReturnDistinct) {
  auto call = ir::Group(
      0x100, {1, 2, 3, 4}, {{ir::Op::read, 64, {}, 0, 9}, {ir::Op::image_address, 64, {}, 0x104}},
      {}, ir::MemoryModel::unspecified, ir::Transfer{ir::TransferKind::call, 0, {}, {}, 1});
  auto path = Path({std::move(call)});
  auto budget = Plenty();
  const auto result = AnalyzePathControl(path, budget);
  ASSERT_TRUE(result.facts);
  const auto& control = result.facts->boundaries[0];
  EXPECT_TRUE(control.callee_return_unknown);
  ASSERT_EQ(control.edge_count, 2);
  EXPECT_EQ(control.edges[0].role, PathEdgeRole::callee);
  EXPECT_EQ(control.edges[0].target_kind, TargetKind::unknown);
  EXPECT_EQ(control.edges[1].role, PathEdgeRole::potential_return);
  EXPECT_EQ(control.edges[1].target_address, 0x104);
  EXPECT_EQ(control.expected_match, ExpectedMatch::not_applicable);
}

TEST(PathControl, KnownFalseSelectRetainsInactiveLoadProvenanceWithoutUsingItsTarget) {
  auto select = ir::Group(0x100, {1, 2, 3, 4},
                          {{ir::Op::constant, 1, {}, 0},
                           {ir::Op::constant, 64, {}, 0x800},
                           {ir::Op::load, 64, {1}},
                           {ir::Op::image_address, 64, {}, 0x200},
                           {ir::Op::select, 64, {0, 2, 3}}},
                          {}, ir::MemoryModel::atomic_scalar_reference,
                          ir::Transfer{ir::TransferKind::jump, 4, {}, {}, {}});
  auto path = Path({std::move(select), Plain(0x200)});
  auto budget = Plenty();
  const auto result = AnalyzePathControl(path, budget);
  ASSERT_TRUE(result.facts);
  const auto& control = result.facts->boundaries[0];
  EXPECT_EQ(control.transfer_target, 4);
  EXPECT_EQ(control.expected_match, ExpectedMatch::always);
  ASSERT_EQ(control.edge_count, 2);
  EXPECT_EQ(control.edges[0].target_kind, TargetKind::unknown);
  EXPECT_EQ(control.edges[0].known_condition, false);
  EXPECT_EQ(control.edges[0].when, true);
  EXPECT_EQ(control.edges[1].when, false);
  EXPECT_EQ(control.load_dependencies, (std::vector<ir::ValueId>{2}));
}

TEST(PathControl, ProvedDivergenceSuppressesAllLaterSuffixFacts) {
  auto path = Path({Jump(0x100, 0x999), Jump(0x200, 0x300), Plain(0x300)});
  auto budget = Plenty();
  const auto result = AnalyzePathControl(path, budget);
  ASSERT_TRUE(result.facts);
  EXPECT_EQ(result.facts->total_boundaries, 3);
  EXPECT_EQ(result.facts->boundaries.size(), 1);
  EXPECT_EQ(result.facts->proved_divergence, 0);
  EXPECT_EQ(path.sources().size(), 3);
  auto origins = std::vector<ir::Origin>(path.origins().begin(), path.origins().end());
  origins.back().operation = UINT32_MAX;
  auto malformed = ir::Path({path.sources().begin(), path.sources().end()},
                            {path.nodes().begin(), path.nodes().end()}, std::move(origins),
                            {path.boundaries().begin(), path.boundaries().end()});
  const auto invalid = AnalyzePathControl(malformed, budget);
  EXPECT_FALSE(invalid.facts);
  EXPECT_EQ(invalid.reason, ControlDecline::invalid_ir);
}

TEST(PathControl, ModularImageTargetsAndFallthroughDoNotRequireZeroBias) {
  auto path = Path({Plain(UINT64_MAX - 3), Jump(0, 0x100), Plain(0x100)});
  auto budget = Plenty();
  const auto result = AnalyzePathControl(path, budget);
  ASSERT_TRUE(result.facts);
  EXPECT_EQ(result.facts->boundaries[0].edges[0].target_address, 0);
  EXPECT_EQ(result.facts->boundaries[0].expected_match, ExpectedMatch::always);
  EXPECT_EQ(result.facts->boundaries[1].expected_match, ExpectedMatch::always);
}

TEST(PathControl, EveryResourceCutDeclinesWithoutPartialFacts) {
  auto indirect = ir::Group(0x100, {1, 2, 3, 4},
                            {{ir::Op::constant, 64, {}, 0x800},
                             {ir::Op::load, 64, {0}},
                             {ir::Op::read, 1, {}, 0, 1},
                             {ir::Op::image_address, 64, {}, 0x200}},
                            {}, ir::MemoryModel::atomic_scalar_reference,
                            ir::Transfer{ir::TransferKind::conditional, 1, 2, 3, {}});
  auto path = Path({std::move(indirect), Plain(0x200)});
  auto budget = Plenty();
  ASSERT_TRUE(AnalyzePathControl(path, budget).facts);
  const auto required = budget.used();
  for (std::uint64_t work = 0; work < required.work; ++work) {
    Budget limited({work, UINT64_MAX});
    const auto result = AnalyzePathControl(path, limited);
    EXPECT_FALSE(result.facts);
    EXPECT_EQ(result.reason, ControlDecline::resource_limit);
  }

  for (std::uint64_t bytes = 0; bytes < required.bytes; ++bytes) {
    Budget limited({UINT64_MAX, bytes});
    const auto result = AnalyzePathControl(path, limited);
    EXPECT_FALSE(result.facts);
    EXPECT_EQ(result.reason, ControlDecline::resource_limit);
  }

  Budget exact(required);
  EXPECT_TRUE(AnalyzePathControl(path, exact).facts);
}

TEST(PathControl, ImageDependencyOverlayChecksShapeAndBudget) {
  auto basis = Path({Branch(0x100, 0x200, 0x300)});
  const ir::RecoveredPath recovered(basis, {}, basis.revision());
  const auto check_bad = [&](std::span<const ir::ValueId> nodes, std::uint64_t revision) {
    auto budget = Plenty();
    const auto result = AnalyzePathControl(recovered, {revision, nodes}, budget);
    EXPECT_FALSE(result.facts);
    EXPECT_EQ(result.reason, ControlDecline::invalid_ir);
  };

  const std::array<ir::ValueId, 1> nonliteral{0}, out_of_range{99};
  const std::array<ir::ValueId, 2> duplicate{1, 1}, unordered{2, 1}, valid{1, 2};
  check_bad(nonliteral, basis.revision());
  check_bad(out_of_range, basis.revision());
  check_bad(duplicate, basis.revision());
  check_bad(unordered, basis.revision());
  check_bad(valid, basis.revision() + 1);
  auto budget = Plenty();
  ASSERT_TRUE(AnalyzePathControl(recovered, {basis.revision(), valid}, budget).facts);
  const auto required = budget.used();
  Budget short_work({required.work - 1, required.bytes});
  const auto work_result = AnalyzePathControl(recovered, {basis.revision(), valid}, short_work);
  EXPECT_FALSE(work_result.facts);
  EXPECT_EQ(work_result.reason, ControlDecline::resource_limit);
  Budget short_bytes({required.work, required.bytes - 1});
  const auto byte_result = AnalyzePathControl(recovered, {basis.revision(), valid}, short_bytes);
  EXPECT_FALSE(byte_result.facts);
  EXPECT_EQ(byte_result.reason, ControlDecline::resource_limit);
  Budget exact(required);
  EXPECT_TRUE(AnalyzePathControl(recovered, {basis.revision(), valid}, exact).facts);
}

TEST(PathControl, RecoveredFactsUseEffectiveJumpAndRevisionWithoutChangingSourceBasis) {
  auto basis = Path({Branch(0x100, 0x200, 0x300, false), Plain(0x200)});
  const auto original = *basis.boundaries()[0].transfer;
  ir::ConditionalRewrite witness{ir::RewriteRule::folded_condition,
                                 0,
                                 original,
                                 {ir::TransferKind::jump, *original.alternative, {}, {}, {}},
                                 *original.condition,
                                 false,
                                 0,
                                 0,
                                 {},
                                 0,
                                 1};
  const ir::RecoveredPath path(basis, {witness}, 1);
  auto budget = Plenty();
  const auto result = AnalyzePathControl(path, {basis.revision(), {}}, budget);
  ASSERT_TRUE(result.facts);
  EXPECT_EQ(result.facts->path_revision, 1);
  ASSERT_EQ(result.facts->boundaries.size(), 1);
  const auto& fact = result.facts->boundaries[0];
  EXPECT_EQ(fact.transfer_kind, ir::TransferKind::jump);
  EXPECT_EQ(fact.edge_count, 1);
  EXPECT_EQ(fact.edges[0].target_address, 0x300);
  EXPECT_FALSE(fact.edges[0].condition);
  EXPECT_EQ(fact.expected_match, ExpectedMatch::never);
  EXPECT_EQ(result.facts->proved_divergence, 0);
  EXPECT_EQ(path.basis().boundaries()[0].transfer->kind, ir::TransferKind::conditional);
  witness.replacement.target = original.target;
  const auto bad =
      AnalyzePathControl(ir::RecoveredPath(basis, {witness}, 1), {basis.revision(), {}}, budget);
  EXPECT_FALSE(bad.facts);
  EXPECT_EQ(bad.reason, ControlDecline::invalid_ir);
}

}  // namespace
}  // namespace nyx::analysis
