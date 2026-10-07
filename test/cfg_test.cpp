#include "nyx/analysis/cfg.hpp"

#include <array>

#include <gtest/gtest.h>

#include "nyx/analysis/regions.hpp"
#include "nyx/analysis/unflatten.hpp"

namespace nyx::analysis {
namespace {

Budget Plenty() { return Budget({UINT64_MAX, UINT64_MAX}); }

SourceRecord Record(ir::Group group) {
  return {group.source_address(),
          {group.bytes().begin(), group.bytes().end()},
          std::move(group),
          OpaqueReason::none};
}

SourceRecord Plain(std::uint64_t address) {
  return Record(ir::Group(address, {1, 2, 3, 4}, {{ir::Op::constant, 64, {}, 7}}, {}));
}

SourceRecord Jump(std::uint64_t address, std::uint64_t target, bool image = true) {
  return Record(ir::Group(
      address, {5, 6, 7, 8}, {{image ? ir::Op::image_address : ir::Op::constant, 64, {}, target}},
      {}, ir::MemoryModel::unspecified, ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}}));
}

// A four-entry table of 16-bit offsets, as a dispatcher would index.
constexpr std::array<std::uint8_t, 8> kEntries = {0x10, 0x00, 0x20, 0x00, 0x30, 0x00, 0x40, 0x00};

// A narrow load published zero-extended into a register, the way a W-form
// write reaches the rest of the function.
SourceRecord Carry(std::uint64_t address, ir::StorageId storage) {
  return Record(ir::Group(
      address, {1, 2, 3, 4},
      {{ir::Op::image_address, 64, {}, 0x4000}, {ir::Op::load, 32, {0}}, {ir::Op::zext, 64, {1}}},
      {{storage, 2}}, ir::MemoryModel::atomic_scalar_reference));
}

// The lifted shape of "unsigned above a literal": carry set and zero clear
// after subtracting it. Falling through is what bounds the register.
SourceRecord Guard(std::uint64_t address, ir::StorageId storage, std::uint64_t limit,
                   std::uint64_t above, std::uint64_t below) {
  return Record(ir::Group(address, {1, 2, 3, 4},
                          {{ir::Op::read, 64, {}, 0, storage},
                           {ir::Op::extract, 32, {0}, 0},
                           {ir::Op::constant, 32, {}, limit},
                           {ir::Op::unsigned_less, 1, {1, 2}},
                           {ir::Op::bit_not, 1, {3}},
                           {ir::Op::sub, 32, {1, 2}},
                           {ir::Op::constant, 32, {}, 0},
                           {ir::Op::equal, 1, {5, 6}},
                           {ir::Op::bit_not, 1, {7}},
                           {ir::Op::bit_and, 1, {4, 8}},
                           {ir::Op::image_address, 64, {}, above},
                           {ir::Op::image_address, 64, {}, below}},
                          {}, ir::MemoryModel::unspecified,
                          ir::Transfer{ir::TransferKind::conditional, 10, 9, 11, {}}));
}

SourceRecord TableJump(std::uint64_t address, ir::StorageId storage) {
  return Record(ir::Group(address, {1, 2, 3, 4},
                          {{ir::Op::read, 64, {}, 0, storage},
                           {ir::Op::extract, 32, {0}, 0},
                           {ir::Op::zext, 64, {1}},
                           {ir::Op::constant, 64, {}, 1},
                           {ir::Op::shl, 64, {2, 3}},
                           {ir::Op::image_address, 64, {}, 0x5000},
                           {ir::Op::add, 64, {5, 4}},
                           {ir::Op::load, 16, {6}},
                           {ir::Op::zext, 64, {7}},
                           {ir::Op::image_address, 64, {}, 0x9000},
                           {ir::Op::add, 64, {9, 8}}},
                          {}, ir::MemoryModel::atomic_scalar_reference,
                          ir::Transfer{ir::TransferKind::jump, 10, {}, {}, {}}));
}

ImageFacts WithTable() {
  static const std::array<ConstantImageRange, 1> ranges{ConstantImageRange{0x5000, kEntries}};
  return ImageFacts{ranges, {}, false, {}};
}

SourceRecord PointerJump(std::uint64_t address) {
  return Record(ir::Group(address, {1, 2, 3, 4},
                          {{ir::Op::image_address, 64, {}, 0x3000}, {ir::Op::load, 64, {0}}}, {},
                          ir::MemoryModel::atomic_scalar_reference,
                          ir::Transfer{ir::TransferKind::jump, 1, {}, {}, {}}));
}

SourceRecord PointerStore(std::uint64_t address, std::uint64_t next) {
  return Record(ir::Group(address, {1, 2, 3, 4},
                          {{ir::Op::image_address, 64, {}, 0x3000},
                           {ir::Op::constant, 64, {}, 7},
                           {ir::Op::store, 64, {0, 1}},
                           {ir::Op::image_address, 64, {}, next}},
                          {}, ir::MemoryModel::atomic_scalar_reference,
                          ir::Transfer{ir::TransferKind::jump, 3, {}, {}, {}}));
}

TEST(Cfg, AStoreInAnotherBlockRefutesAResolvedPointerBeforeEdgesAreBuilt) {
  const std::vector<SourceRecord> sources{PointerJump(0x100), PointerStore(0x200, 0x100),
                                          Jump(0x300, 0x300)};
  const std::uint64_t entries[] = {0x100, 0x200};
  const std::array<RelocatedPointer, 1> pointers{RelocatedPointer{0x3000, 0x300}};
  auto budget = Plenty();
  auto result = BuildCfg(sources, entries, budget, {}, ImageFacts{{}, pointers, false});
  ASSERT_TRUE(result.cfg);
  EXPECT_EQ(result.cfg->refuted_pointer_indices().size(), 1);
  const auto retained =
      ir::RetainImageFacts(ImageFacts{{}, pointers, false}, result.cfg->refuted_constant_spans(),
                           result.cfg->refuted_pointer_indices(), budget);
  ASSERT_TRUE(retained);
  ASSERT_EQ(retained->pointers.size(), 1);
  EXPECT_FALSE(retained->pointers[0].value_stable);
  EXPECT_FALSE(ir::ReadRelocated(retained->view(), 0x3000, 64));
  ASSERT_EQ(result.cfg->blocks()[0].edges.size(), 1);
  EXPECT_EQ(result.cfg->blocks()[0].edges[0].target.kind, TargetKind::unknown);
}

TEST(Cfg, AResolvedPointerEdgeKeepsItsDeclaredImageDependency) {
  const std::vector<SourceRecord> sources{PointerJump(0x100), Jump(0x300, 0x300)};
  const std::uint64_t entries[] = {0x100};
  const std::array<RelocatedPointer, 1> pointers{RelocatedPointer{0x3000, 0x300}};
  auto budget = Plenty();
  auto result = BuildCfg(sources, entries, budget, {}, ImageFacts{{}, pointers, false});
  ASSERT_TRUE(result.cfg);
  ASSERT_EQ(result.cfg->blocks()[0].edges.size(), 1);
  EXPECT_EQ(result.cfg->blocks()[0].edges[0].target.kind, TargetKind::image_location);
  EXPECT_TRUE(result.cfg->blocks()[0].edges[0].constant_image_dependency);
  Regions regions(std::move(*result.cfg), {});
  const auto stitched = Unflatten(regions, {}, budget);
  ASSERT_TRUE(stitched.unflattening);
  ASSERT_FALSE(stitched.unflattening->graph.blocks.empty());
  ASSERT_EQ(stitched.unflattening->graph.blocks[0].edges.size(), 1);
  EXPECT_TRUE(stitched.unflattening->graph.blocks[0].edges[0].assumptions.constant_image);
}

TEST(Cfg, AStoreInAnotherBlockRefutesAConstantTargetBeforeEdgesAreBuilt) {
  const std::vector<SourceRecord> sources{
      Record(ir::Group(0x100, {1, 2, 3, 4},
                       {{ir::Op::image_address, 64, {}, 0x5000},
                        {ir::Op::load, 16, {0}},
                        {ir::Op::zext, 64, {1}},
                        {ir::Op::image_address, 64, {}, 0x9000},
                        {ir::Op::add, 64, {3, 2}}},
                       {}, ir::MemoryModel::atomic_scalar_reference,
                       ir::Transfer{ir::TransferKind::jump, 4, {}, {}, {}})),
      Record(ir::Group(0x200, {1, 2, 3, 4},
                       {{ir::Op::image_address, 64, {}, 0x5000},
                        {ir::Op::constant, 16, {}, 7},
                        {ir::Op::store, 16, {0, 1}},
                        {ir::Op::image_address, 64, {}, 0x100}},
                       {}, ir::MemoryModel::atomic_scalar_reference,
                       ir::Transfer{ir::TransferKind::jump, 3, {}, {}, {}})),
      Jump(0x9010, 0x9010)};
  const std::uint64_t entries[] = {0x100, 0x200};
  auto budget = Plenty();
  const auto result = BuildCfg(sources, entries, budget, {}, WithTable());
  ASSERT_TRUE(result.cfg);

  // The store withdraws the two bytes it writes, which the load reads; the
  // rest of the table stays declared.
  ASSERT_EQ(result.cfg->refuted_constant_spans().size(), 1);
  EXPECT_EQ(result.cfg->refuted_constant_spans()[0].address, 0x5000U);
  EXPECT_EQ(result.cfg->refuted_constant_spans()[0].bytes, 2U);
  const auto retained = ir::RetainImageFacts(WithTable(), result.cfg->refuted_constant_spans(),
                                             result.cfg->refuted_pointer_indices(), budget);
  ASSERT_TRUE(retained);
  ASSERT_EQ(retained->constants.size(), 1U);
  EXPECT_EQ(retained->constants[0].address, 0x5002U);
  EXPECT_EQ(retained->constants[0].bytes.size(), WithTable().constants[0].bytes.size() - 2);
  ASSERT_EQ(result.cfg->blocks()[0].edges.size(), 1);
  EXPECT_EQ(result.cfg->blocks()[0].edges[0].target.kind, TargetKind::unknown);
}

TEST(Cfg, AGuardedTableDispatchBecomesEveryEdgeItCanTake) {
  const std::vector<SourceRecord> sources{Carry(0x100, 8), Guard(0x104, 8, 3, 0x900, 0x108),
                                          TableJump(0x108, 8)};
  const std::uint64_t entries[] = {0x100};
  auto budget = Plenty();
  const auto result = BuildCfg(sources, entries, budget, {}, WithTable());
  ASSERT_TRUE(result.cfg);
  ASSERT_EQ(result.cfg->blocks().size(), 2);
  const auto& jump = result.cfg->blocks()[1];
  ASSERT_TRUE(jump.dispatch);
  EXPECT_EQ(jump.dispatch->guard_block, 0);
  EXPECT_EQ(jump.dispatch->bound, 3);
  ASSERT_EQ(jump.edges.size(), 4);
  std::vector<std::uint64_t> destinations;
  for (const auto& edge : jump.edges) {
    EXPECT_EQ(edge.kind, CfgEdgeKind::branch);
    EXPECT_EQ(edge.target.kind, TargetKind::image_location);
    EXPECT_TRUE(edge.constant_image_dependency);
    EXPECT_FALSE(edge.condition);
    destinations.push_back(edge.target.address);
  }

  EXPECT_EQ(destinations, (std::vector<std::uint64_t>{0x9010, 0x9020, 0x9030, 0x9040}));

  // The index the edges run over is the block's own entry read of the register.
  ASSERT_LT(jump.dispatch->index, jump.ssa->nodes().size());
  EXPECT_EQ(jump.ssa->nodes()[jump.dispatch->index].op, ir::Op::read);
  EXPECT_EQ(jump.ssa->nodes()[jump.dispatch->index].storage, 8);
  EXPECT_FALSE(result.cfg->blocks()[0].dispatch);
}

TEST(Cfg, ASecondWayInLeavesTheDispatchUnbounded) {
  // The guard bounds the register on its own arm, but nothing bounds it on the
  // other entry, so the table's entries are not all this jump can reach.
  const std::vector<SourceRecord> sources{Carry(0x100, 8), Guard(0x104, 8, 3, 0x900, 0x108),
                                          TableJump(0x108, 8), Jump(0x10c, 0x108)};
  const std::uint64_t entries[] = {0x100};
  auto budget = Plenty();
  const auto result = BuildCfg(sources, entries, budget, {}, WithTable());
  ASSERT_TRUE(result.cfg);
  ASSERT_EQ(result.cfg->blocks().size(), 3);
  EXPECT_FALSE(result.cfg->blocks()[1].dispatch);
  ASSERT_EQ(result.cfg->blocks()[1].edges.size(), 1);
  EXPECT_EQ(result.cfg->blocks()[1].edges[0].target.kind, TargetKind::unknown);
}

TEST(Cfg, ABoundOnOneRegisterSaysNothingAboutAnother) {
  const std::vector<SourceRecord> sources{Carry(0x100, 9), Guard(0x104, 9, 3, 0x900, 0x108),
                                          TableJump(0x108, 8)};
  const std::uint64_t entries[] = {0x100};
  auto budget = Plenty();
  const auto result = BuildCfg(sources, entries, budget, {}, WithTable());
  ASSERT_TRUE(result.cfg);
  ASSERT_EQ(result.cfg->blocks().size(), 2);
  EXPECT_FALSE(result.cfg->blocks()[1].dispatch);
}

TEST(Cfg, WithoutTheTableBytesNoDestinationSetIsClaimed) {
  const std::vector<SourceRecord> sources{Carry(0x100, 8), Guard(0x104, 8, 3, 0x900, 0x108),
                                          TableJump(0x108, 8)};
  const std::uint64_t entries[] = {0x100};
  auto budget = Plenty();
  const auto result = BuildCfg(sources, entries, budget);
  ASSERT_TRUE(result.cfg);
  ASSERT_EQ(result.cfg->blocks().size(), 2);
  EXPECT_FALSE(result.cfg->blocks()[1].dispatch);
}

TEST(Cfg, MoreDestinationsThanTheLimitAdmitsPublishesNone) {
  const std::vector<SourceRecord> sources{Carry(0x100, 8), Guard(0x104, 8, 3, 0x900, 0x108),
                                          TableJump(0x108, 8)};
  const std::uint64_t entries[] = {0x100};
  auto budget = Plenty();
  CfgLimits limits;
  limits.max_dispatch_targets = 3;
  const auto result = BuildCfg(sources, entries, budget, limits, WithTable());
  ASSERT_TRUE(result.cfg);
  EXPECT_FALSE(result.cfg->blocks()[1].dispatch);
}

TEST(Cfg, AccountsForSortedPopulationIncludingOpaqueInstructionsAndGaps) {
  const std::vector<SourceRecord> sources{
      Plain(0x200),
      {0x108, {0xff, 0xff, 0xff, 0xff}, {}, OpaqueReason::unsupported},
      Plain(0x104),
      Plain(0x100),
      Plain(0x10c)};
  const std::uint64_t entries[] = {0x100};
  auto budget = Plenty();
  const auto result = BuildCfg(sources, entries, budget);
  ASSERT_TRUE(result.cfg);
  EXPECT_EQ(result.cfg->generation(), 0);
  ASSERT_EQ(result.cfg->sources().size(), 5);
  ASSERT_EQ(result.cfg->blocks().size(), 4);
  EXPECT_EQ(result.cfg->blocks()[0].source_count, 2);
  EXPECT_EQ(result.cfg->blocks()[0].edges[0].kind, CfgEdgeKind::fallthrough);
  EXPECT_EQ(result.cfg->blocks()[0].edges[0].resolution, TargetResolution::opaque_entry);
  EXPECT_EQ(result.cfg->blocks()[0].edges[0].target_block, 1);
  const auto& opaque = result.cfg->blocks()[1];
  EXPECT_FALSE(opaque.ssa);
  EXPECT_FALSE(opaque.control);
  EXPECT_EQ(opaque.source_count, 1);
  ASSERT_EQ(opaque.edges.size(), 1);
  EXPECT_EQ(opaque.edges[0].kind, CfgEdgeKind::opaque_unknown);
  EXPECT_EQ(opaque.edges[0].resolution, TargetResolution::unknown);
  EXPECT_FALSE(opaque.edges[0].target_block);
  EXPECT_EQ(result.cfg->blocks()[2].edges[0].resolution, TargetResolution::outside_population);
  std::size_t accounted = 0;
  for (const auto& block : result.cfg->blocks()) accounted += block.source_count;
  EXPECT_EQ(accounted, sources.size());
}

TEST(Cfg, DeclaredOpaqueFallthroughConnectsControlButNotSemantics) {
  const std::vector<SourceRecord> sources{Plain(0x100),
                                          {0x104,
                                           {0x0a, 0xfc, 0x5f, 0x88},
                                           {},
                                           OpaqueReason::unsupported,
                                           OpaqueControl::normal_fallthrough},
                                          Plain(0x108)};
  const std::uint64_t entries[] = {0x100};
  auto budget = Plenty();
  auto result = BuildCfg(sources, entries, budget);
  ASSERT_TRUE(result.cfg);
  ASSERT_EQ(result.cfg->blocks().size(), 3);
  const auto& opaque = result.cfg->blocks()[1];
  EXPECT_FALSE(opaque.ssa);
  EXPECT_FALSE(opaque.control);
  ASSERT_EQ(opaque.edges.size(), 1);
  EXPECT_EQ(opaque.edges[0].kind, CfgEdgeKind::fallthrough);
  EXPECT_EQ(opaque.edges[0].resolution, TargetResolution::block_entry);
  EXPECT_EQ(opaque.edges[0].target_block, 2);
  auto regions = BuildRegions(std::move(*result.cfg), budget);
  ASSERT_TRUE(regions.regions);
  EXPECT_EQ(regions.regions->candidates()[0].stop, RegionStop::opaque);
  EXPECT_EQ(regions.regions->candidates()[0].source_ids, (std::vector<std::uint32_t>{0}));
  const std::vector<std::optional<PathControlFacts>> control(regions.regions->candidates().size());
  const auto unflattened = Unflatten(*regions.regions, control, budget);
  ASSERT_TRUE(unflattened.unflattening);
  const auto& stitched = unflattened.unflattening->graph.blocks;
  ASSERT_EQ(stitched.size(), 3);
  ASSERT_EQ(stitched[1].edges.size(), 1);
  EXPECT_TRUE(stitched[1].edges[0].assumptions.declared_opaque_control);
  EXPECT_FALSE(stitched[1].edges[0].assumptions.unresolved_target);
}

TEST(Cfg, DeclaredOpaqueFallthroughOutsidePopulationRemainsUnresolved) {
  const std::vector<SourceRecord> sources{{0x100,
                                           {0x0a, 0xfc, 0x5f, 0x88},
                                           {},
                                           OpaqueReason::unsupported,
                                           OpaqueControl::normal_fallthrough}};
  const std::uint64_t entries[] = {0x100};
  auto budget = Plenty();
  auto result = BuildCfg(sources, entries, budget);
  ASSERT_TRUE(result.cfg);
  ASSERT_EQ(result.cfg->blocks()[0].edges.size(), 1);
  EXPECT_EQ(result.cfg->blocks()[0].edges[0].resolution, TargetResolution::outside_population);
  EXPECT_FALSE(result.cfg->blocks()[0].edges[0].target_block);
  EXPECT_TRUE(BuildRegions(std::move(*result.cfg), budget).regions);
}

TEST(Cfg, CallerOpaqueControlDeclarationRemainsAnExplicitAssumption) {
  // The generic CFG does not decode target bytes, just as it does not re-lift
  // supplied IR. A caller with false metadata must not receive a silent proof.
  const std::vector<SourceRecord> sources{{0x100,
                                           {0xff, 0xff, 0xff, 0xff},
                                           {},
                                           OpaqueReason::unsupported,
                                           OpaqueControl::normal_fallthrough},
                                          Plain(0x104)};
  const std::uint64_t entries[] = {0x100};
  auto budget = Plenty();
  auto graph = BuildCfg(sources, entries, budget);
  ASSERT_TRUE(graph.cfg);
  auto regions = BuildRegions(std::move(*graph.cfg), budget);
  ASSERT_TRUE(regions.regions);
  const std::vector<std::optional<PathControlFacts>> control(regions.regions->candidates().size());
  const auto unflattened = Unflatten(*regions.regions, control, budget);
  ASSERT_TRUE(unflattened.unflattening);
  const auto& edge = unflattened.unflattening->graph.blocks[0].edges[0];
  EXPECT_TRUE(edge.assumptions.declared_opaque_control);
}

TEST(Cfg, OpaqueControlRequiresUnsupportedSourceAndNonoverflowingSuccessor) {
  const std::uint64_t entries[] = {0x100};
  auto budget = Plenty();
  for (auto reason : {OpaqueReason::invalid_encoding, OpaqueReason::not_decoded}) {
    const std::vector<SourceRecord> sources{
        {0x100, {1, 2, 3, 4}, {}, reason, OpaqueControl::normal_fallthrough}};
    EXPECT_EQ(BuildCfg(sources, entries, budget).reason, CfgDecline::invalid_source);
  }

  const std::uint64_t high[] = {UINT64_MAX - 3};
  const std::vector<SourceRecord> overflow{{UINT64_MAX - 3,
                                            {1, 2, 3, 4},
                                            {},
                                            OpaqueReason::unsupported,
                                            OpaqueControl::normal_fallthrough}};
  EXPECT_EQ(BuildCfg(overflow, high, budget).reason, CfgDecline::invalid_source);
}

TEST(Cfg, KnownComputedTargetSplitsBlockAndRebuildsRevisionBoundFacts) {
  const std::vector<SourceRecord> sources{
      Plain(0x100),
      Record(ir::Group(0x104, {1, 2, 3, 4}, {{ir::Op::image_address, 64, {}, 0x104}}, {{77, 0}})),
      Record(ir::Group(0x108, {1, 2, 3, 4}, {{ir::Op::read, 64, {}, 0, 77}}, {},
                       ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}}))};
  const std::uint64_t entries[] = {0x100};
  auto budget = Plenty();
  const auto result = BuildCfg(sources, entries, budget);
  ASSERT_TRUE(result.cfg);
  EXPECT_EQ(result.cfg->passes(), 2);
  ASSERT_EQ(result.cfg->blocks().size(), 2);
  const auto& loop = result.cfg->blocks()[1];
  ASSERT_TRUE(loop.control);
  ASSERT_TRUE(loop.ssa);
  EXPECT_EQ(loop.control->terminal_source, 0x108);
  EXPECT_EQ(loop.control->block_revision, loop.ssa->revision());
  EXPECT_EQ(loop.first_source, 1);
  EXPECT_EQ(loop.source_count, 2);
  EXPECT_EQ(loop.edges[0].target.value, 0);
  EXPECT_EQ(loop.edges[0].target.address, 0x104);
  EXPECT_EQ(loop.edges[0].target_block, 1);
  EXPECT_EQ(loop.ssa->nodes()[0].op, ir::Op::image_address);
}

TEST(Cfg, NewInteriorEntryInvalidatesPriorRegisterForwardingFacts) {
  const std::vector<SourceRecord> sources{
      Record(ir::Group(0x100, {1, 2, 3, 4}, {{ir::Op::image_address, 64, {}, 0x108}}, {{77, 0}})),
      Plain(0x104),
      Record(ir::Group(0x108, {1, 2, 3, 4}, {{ir::Op::read, 64, {}, 0, 77}}, {},
                       ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}}))};
  const std::uint64_t entries[] = {0x100};
  auto budget = Plenty();
  const auto result = BuildCfg(sources, entries, budget);
  ASSERT_TRUE(result.cfg);
  EXPECT_EQ(result.cfg->passes(), 2);
  ASSERT_EQ(result.cfg->blocks().size(), 2);
  const auto& terminal = result.cfg->blocks()[1];
  ASSERT_TRUE(terminal.ssa);
  EXPECT_EQ(terminal.ssa->nodes()[0].op, ir::Op::read);
  EXPECT_EQ(terminal.edges[0].target.kind, TargetKind::unknown);
  EXPECT_EQ(terminal.edges[0].resolution, TargetResolution::unknown);
  EXPECT_FALSE(terminal.edges[0].target_block);
}

TEST(Cfg, ConditionalEdgesPreserveTheConditionDestinationRelation) {
  const std::vector<SourceRecord> sources{
      Record(ir::Group(0x100, {1, 2, 3, 4},
                       {{ir::Op::read, 1, {}, 0, 33},
                        {ir::Op::image_address, 64, {}, 0x104},
                        {ir::Op::image_address, 64, {}, 0x108}},
                       {}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::conditional, 1, 0, 2, {}})),
      Plain(0x104),
      {0x108, {0xff, 0xff, 0xff, 0xff}, {}, OpaqueReason::invalid_encoding}};
  const std::uint64_t entries[] = {0x100};
  auto budget = Plenty();
  const auto result = BuildCfg(sources, entries, budget);
  ASSERT_TRUE(result.cfg);
  const auto& edges = result.cfg->blocks()[0].edges;
  ASSERT_EQ(edges.size(), 2);
  EXPECT_EQ(edges[0].condition, 0);
  EXPECT_EQ(edges[1].condition, 0);
  EXPECT_EQ(edges[0].when, true);
  EXPECT_EQ(edges[1].when, false);
  EXPECT_EQ(edges[0].target.address, 0x104);
  EXPECT_EQ(edges[1].target.address, 0x108);
  EXPECT_EQ(edges[0].resolution, TargetResolution::block_entry);
  EXPECT_EQ(edges[1].resolution, TargetResolution::opaque_entry);
}

TEST(Cfg, CallsKeepPotentialReturnMetadataSeparateFromCalleeExecution) {
  const std::vector<SourceRecord> sources{
      Record(ir::Group(
          0x100, {1, 2, 3, 4},
          {{ir::Op::image_address, 64, {}, 0x200}, {ir::Op::image_address, 64, {}, 0x104}},
          {{99, 1}}, ir::MemoryModel::unspecified,
          ir::Transfer{ir::TransferKind::call, 0, {}, {}, 1})),
      Plain(0x104),
      {0x200, {0xff}, {}, OpaqueReason::not_decoded}};
  const std::uint64_t entries[] = {0x100};
  auto budget = Plenty();
  const auto result = BuildCfg(sources, entries, budget);
  ASSERT_TRUE(result.cfg);
  const auto& call = result.cfg->blocks()[0];
  ASSERT_TRUE(call.control);
  EXPECT_TRUE(call.control->callee_return_unknown);
  ASSERT_EQ(call.edges.size(), 2);
  EXPECT_EQ(call.edges[0].kind, CfgEdgeKind::callee);
  EXPECT_EQ(call.edges[0].target.address, 0x200);
  EXPECT_EQ(call.edges[0].resolution, TargetResolution::opaque_entry);
  EXPECT_EQ(call.edges[1].kind, CfgEdgeKind::potential_return);
  EXPECT_EQ(call.edges[1].target.address, 0x104);
  EXPECT_EQ(call.edges[1].resolution, TargetResolution::block_entry);
}

TEST(Cfg, DistinguishesMidInstructionExternalAndAbsoluteRuntimeTargets) {
  const std::vector<SourceRecord> sources{Jump(0x100, 0x102), Jump(0x104, 0x108, false),
                                          Jump(0x108, 0x999)};
  const std::uint64_t entries[] = {0x100};
  auto budget = Plenty();
  const auto result = BuildCfg(sources, entries, budget);
  ASSERT_TRUE(result.cfg);
  ASSERT_EQ(result.cfg->blocks().size(), 3);
  EXPECT_EQ(result.cfg->blocks()[0].edges[0].resolution, TargetResolution::mid_instruction);
  EXPECT_EQ(result.cfg->blocks()[0].edges[0].target_source, 0);
  EXPECT_FALSE(result.cfg->blocks()[0].edges[0].target_block);
  EXPECT_EQ(result.cfg->blocks()[1].edges[0].resolution, TargetResolution::absolute_runtime);
  EXPECT_FALSE(result.cfg->blocks()[1].edges[0].target_source);
  EXPECT_EQ(result.cfg->blocks()[2].edges[0].resolution, TargetResolution::outside_population);
}

TEST(Cfg, FallthroughWrapsModulo64ButSourceByteExtentsCannotWrap) {
  const std::vector<SourceRecord> sources{Plain(0), Record(ir::Group(UINT64_MAX, {1}, {}, {}))};
  const std::uint64_t entries[] = {UINT64_MAX};
  auto budget = Plenty();
  const auto result = BuildCfg(sources, entries, budget);
  ASSERT_TRUE(result.cfg);
  ASSERT_EQ(result.cfg->blocks().size(), 2);
  const auto& edge = result.cfg->blocks()[1].edges[0];
  EXPECT_EQ(edge.kind, CfgEdgeKind::fallthrough);
  EXPECT_EQ(edge.target.address, 0);
  EXPECT_FALSE(edge.target.value);
  EXPECT_EQ(edge.resolution, TargetResolution::block_entry);
  EXPECT_EQ(edge.target_block, 0);
}

TEST(Cfg, SelectedEntriesPartitionWithoutInventingReachability) {
  const std::vector<SourceRecord> sources{Plain(0x100), Plain(0x104), Plain(0x108)};
  const std::uint64_t entries[] = {0x108, 0x100};
  auto budget = Plenty();
  const auto result = BuildCfg(sources, entries, budget);
  ASSERT_TRUE(result.cfg);
  ASSERT_EQ(result.cfg->entries().size(), 2);
  EXPECT_EQ(result.cfg->entries()[0], 0x100);
  EXPECT_EQ(result.cfg->entries()[1], 0x108);
  ASSERT_EQ(result.cfg->blocks().size(), 2);
  EXPECT_EQ(result.cfg->blocks()[0].source_count, 2);
  EXPECT_EQ(result.cfg->blocks()[1].source_count, 1);
}

TEST(Cfg, RejectsOverlapsWrappedBytesAndSourceSemanticsMismatch) {
  const std::uint64_t entries[] = {0x100};
  std::vector<std::vector<SourceRecord>> invalid{
      {},
      {Plain(0x100), Plain(0x102)},
      {Plain(0x100), Plain(0x100)},
      {{0x100, {}, {}, OpaqueReason::unsupported}},
      {{UINT64_MAX, {1, 2}, {}, OpaqueReason::unsupported}},
      {{0x100, {1}, {}, OpaqueReason::none}},
      {{0x100, {1}, {}, static_cast<OpaqueReason>(99)}}};
  auto wrong_bytes = Plain(0x100);
  wrong_bytes.bytes[0] ^= 1;
  invalid.push_back({wrong_bytes});
  auto wrong_address = Plain(0x100);
  wrong_address.address = 0x104;
  invalid.push_back({wrong_address});
  auto wrong_reason = Plain(0x100);
  wrong_reason.opaque_reason = OpaqueReason::unsupported;
  invalid.push_back({wrong_reason});
  for (const auto& sources : invalid) {
    auto budget = Plenty();
    const auto result = BuildCfg(sources, entries, budget);
    EXPECT_FALSE(result.cfg);
    EXPECT_EQ(result.reason, CfgDecline::invalid_source);
  }

  const std::vector<SourceRecord> sources{Plain(0x100)};
  for (const std::vector<std::uint64_t>& bad :
       {std::vector<std::uint64_t>{}, {0x101}, {0x500}, {0x100, 0x100}}) {
    auto budget = Plenty();
    const auto result = BuildCfg(sources, bad, budget);
    EXPECT_FALSE(result.cfg);
    EXPECT_EQ(result.reason, CfgDecline::invalid_entry);
  }
}

TEST(Cfg, DeclinesMalformedSemanticsAndExhaustedDiscoveryLimits) {
  const std::uint64_t entries[] = {0x100};
  const std::vector<SourceRecord> malformed{
      Record(ir::Group(0x100, {1}, {{ir::Op::add, 64, {0, 0}}}, {}))};
  auto budget = Plenty();
  EXPECT_EQ(BuildCfg(malformed, entries, budget).reason, CfgDecline::invalid_ir);
  const std::vector<SourceRecord> sources{Plain(0x100), Plain(0x104), Jump(0x108, 0x104)};
  for (unsigned choice = 0; choice < 4; ++choice) {
    CfgLimits limits;
    if (choice == 0) limits.max_passes = 1;
    if (choice == 1) limits.max_blocks = 1;
    if (choice == 2) limits.max_sources = 2;
    if (choice == 3) limits.max_source_bytes = 11;
    auto bounded = Plenty();
    const auto result = BuildCfg(sources, entries, bounded, limits);
    EXPECT_FALSE(result.cfg);
    EXPECT_EQ(result.reason, CfgDecline::resource_limit);
  }
}

TEST(Cfg, EveryBudgetCutPublishesNoPartialGraph) {
  const std::vector<SourceRecord> sources{Plain(0x100), Jump(0x104, 0x104)};
  const std::uint64_t entries[] = {0x100};
  CfgLimits limits;
  limits.block.max_storage = 4;
  auto full = Plenty();
  ASSERT_TRUE(BuildCfg(sources, entries, full, limits).cfg);
  for (bool bytes : {false, true}) {
    const auto maximum = bytes ? full.used().bytes : full.used().work;
    for (std::uint64_t cut = 0; cut < maximum; ++cut) {
      Budget budget({bytes ? full.used().work : cut, bytes ? cut : full.used().bytes});
      const auto result = BuildCfg(sources, entries, budget, limits);
      ASSERT_FALSE(result.cfg) << cut;
      EXPECT_EQ(result.reason, CfgDecline::resource_limit) << cut;
    }
  }

  Budget exact(full.used());
  EXPECT_TRUE(BuildCfg(sources, entries, exact, limits).cfg);
}

}  // namespace
}  // namespace nyx::analysis
