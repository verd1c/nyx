#include "nyx/analysis/regions.hpp"

#include <array>

#include <gtest/gtest.h>

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
  return Record(ir::Group(address, {1, 2, 3, 4}, {{ir::Op::constant, 64, {}, 9}}, {{7, 0}}));
}

SourceRecord Jump(std::uint64_t address, std::uint64_t target) {
  return Record(ir::Group(address, {1, 2, 3, 4}, {{ir::Op::image_address, 64, {}, target}}, {},
                          ir::MemoryModel::unspecified,
                          ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}}));
}

SourceRecord Branch(std::uint64_t address, std::uint64_t yes, std::uint64_t no,
                    bool unknown = false) {
  return Record(ir::Group(address, {1, 2, 3, 4},
                          {{ir::Op::read, 1, {}, 0, 3},
                           {ir::Op::image_address, 64, {}, yes},
                           {unknown ? ir::Op::read : ir::Op::image_address, 64, {}, no, 4}},
                          {}, ir::MemoryModel::unspecified,
                          ir::Transfer{ir::TransferKind::conditional, 1, 0, 2, {}}));
}

Cfg Graph(std::vector<SourceRecord> sources) {
  const std::uint64_t entries[] = {sources.front().address};
  auto budget = Plenty();
  auto graph = BuildCfg(sources, entries, budget);
  EXPECT_TRUE(graph.cfg);
  return std::move(*graph.cfg);
}

TEST(Regions, OwnsPopulationAndBuildsCompletePredecessorPaths) {
  auto budget = Plenty();
  auto result = BuildRegions(Graph({Plain(0x100),
                                    Jump(0x104, 0x200),
                                    Plain(0x200),
                                    Jump(0x204, 0x900),
                                    {0x300, {0xff}, {}, OpaqueReason::unsupported}}),
                             budget);
  ASSERT_TRUE(result.regions);
  const auto& regions = *result.regions;
  ASSERT_EQ(regions.candidates().size(), 2);
  const auto& first = regions.candidates()[0];
  EXPECT_EQ(first.block_ids, (std::vector<std::uint32_t>{0, 1}));
  EXPECT_EQ(first.source_ids, (std::vector<std::uint32_t>{0, 1, 2, 3}));
  EXPECT_EQ(first.transition_edges, (std::vector<std::uint32_t>{0}));
  EXPECT_EQ(first.stop, RegionStop::unresolved);
  EXPECT_EQ(regions.graph().sources().size(), 5);
  auto path = NormalizeRegion(regions, 0, budget);
  ASSERT_TRUE(path.path);
  EXPECT_EQ(path.path->sources().size(), 4);
  EXPECT_EQ(path.path->sources()[2].source_address(), 0x200);
  ASSERT_TRUE(path.path->boundaries()[1].transfer);
  EXPECT_EQ(path.path->expected_successor(1), 0x200);
  EXPECT_EQ(regions.candidates()[1].source_ids, (std::vector<std::uint32_t>{2, 3}));
}

TEST(Regions, ConditionalKnownAndUnknownArmsRemainSeparate) {
  auto budget = Plenty();
  auto result = BuildRegions(Graph({Branch(0x100, 0x200, 0, true), Jump(0x200, 0x900)}), budget);
  ASSERT_TRUE(result.regions);
  const auto candidates = result.regions->candidates();
  ASSERT_EQ(candidates.size(), 3);
  EXPECT_EQ(candidates[0].block_ids.size(), 2);
  EXPECT_EQ(candidates[0].transition_edges[0], 0);
  EXPECT_EQ(candidates[1].block_ids.size(), 1);
  EXPECT_EQ(candidates[1].stopped_edge, 1);
  EXPECT_EQ(candidates[1].stop, RegionStop::unresolved);
  const auto& edges = result.regions->graph().blocks()[0].edges;
  EXPECT_NE(edges[0].when, edges[1].when);
  EXPECT_EQ(edges[1].resolution, TargetResolution::unknown);
}

TEST(Regions, EqualDestinationArmsKeepTheirPolarityAndSecondForkStops) {
  auto budget = Plenty();
  auto result = BuildRegions(
      Graph({Branch(0x100, 0x200, 0x200), Branch(0x200, 0x300, 0x300), Jump(0x300, 0x900)}),
      budget);
  ASSERT_TRUE(result.regions);
  const auto candidates = result.regions->candidates();
  ASSERT_EQ(candidates.size(), 5);
  EXPECT_EQ(candidates[0].source_ids, candidates[1].source_ids);
  EXPECT_EQ(candidates[0].transition_edges[0], 0);
  EXPECT_EQ(candidates[1].transition_edges[0], 1);
  EXPECT_EQ(candidates[0].stop, RegionStop::second_fork);
  EXPECT_FALSE(candidates[0].stopped_edge);
  EXPECT_EQ(candidates[2].block_ids.size(), 2);
}

TEST(Regions, CallsReturnsOpaqueAndCyclesAreExplicitStops) {
  auto call = Record(ir::Group(
      0x300, {1, 2, 3, 4},
      {{ir::Op::image_address, 64, {}, 0x400}, {ir::Op::image_address, 64, {}, 0x304}}, {},
      ir::MemoryModel::unspecified, ir::Transfer{ir::TransferKind::call, 0, {}, {}, 1}));
  auto ret = Record(ir::Group(0x400, {1, 2, 3, 4}, {{ir::Op::read, 64, {}, 0, 30}}, {},
                              ir::MemoryModel::unspecified,
                              ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}}));
  auto budget = Plenty();
  auto result = BuildRegions(Graph({Jump(0x100, 0x100),
                                    Jump(0x200, 0x500),
                                    std::move(call),
                                    Plain(0x304),
                                    std::move(ret),
                                    {0x500, {0xff}, {}, OpaqueReason::unsupported}}),
                             budget);
  ASSERT_TRUE(result.regions);
  const auto candidates = result.regions->candidates();
  ASSERT_EQ(candidates.size(), 5);
  EXPECT_EQ(candidates[0].stop, RegionStop::cycle);
  EXPECT_EQ(candidates[1].stop, RegionStop::opaque);
  EXPECT_EQ(candidates[2].stop, RegionStop::call);
  EXPECT_EQ(candidates[2].block_ids.size(), 1);
  EXPECT_EQ(candidates[4].stop, RegionStop::return_);
  const auto& edges = result.regions->graph().blocks()[2].edges;
  EXPECT_EQ(edges.back().kind, CfgEdgeKind::potential_return);
}

TEST(Regions, SharedSuccessorHasIndependentEntryScopedDefinitions) {
  auto budget = Plenty();
  auto successor = Record(ir::Group(0x300, {1, 2, 3, 4}, {{ir::Op::read, 64, {}, 0, 7}}, {{8, 0}}));
  auto result = BuildRegions(Graph({Plain(0x100), Jump(0x104, 0x300), Plain(0x200),
                                    Jump(0x204, 0x300), std::move(successor)}),
                             budget);
  ASSERT_TRUE(result.regions);
  ASSERT_EQ(result.regions->candidates().size(), 3);
  auto first = NormalizeRegion(*result.regions, 0, budget);
  auto independent = NormalizeRegion(*result.regions, 2, budget);
  ASSERT_TRUE(first.path);
  ASSERT_TRUE(independent.path);
  EXPECT_EQ(first.path->sources().size(), 3);
  ASSERT_EQ(independent.path->nodes().size(), 1);
  EXPECT_EQ(independent.path->nodes()[0].op, ir::Op::read);
  EXPECT_EQ(result.regions->graph().blocks()[2].ssa->nodes()[0].op, ir::Op::read);
}

TEST(Regions, CapsReportStopsWithoutTruncatingMachineBlocks) {
  auto original = Graph({Plain(0x100), Jump(0x104, 0x200), Jump(0x200, 0x300), Jump(0x300, 0x900)});
  auto budget = Plenty();
  RegionLimits limits;
  limits.max_sources_per_region = 1;
  auto copy = original;
  auto result = BuildRegions(std::move(copy), budget, limits);
  ASSERT_TRUE(result.regions);
  ASSERT_EQ(result.regions->candidates().size(), 3);
  EXPECT_EQ(result.regions->candidates()[0].stop, RegionStop::source_limit);
  EXPECT_TRUE(result.regions->candidates()[0].source_ids.empty());
  limits.max_sources_per_region = 512;
  limits.max_blocks_per_region = 1;
  result = BuildRegions(std::move(original), budget, limits);
  ASSERT_TRUE(result.regions);
  EXPECT_EQ(result.regions->candidates()[0].source_ids.size(), 2);
  EXPECT_EQ(result.regions->candidates()[0].stop, RegionStop::block_limit);
}

// A guarded table dispatch, built the same way the CFG tests build one: a
// narrow load published zero-extended, a bound on it, then a jump through the
// table it indexes.
constexpr std::array<std::uint8_t, 8> kEntries = {0x10, 0x00, 0x20, 0x00, 0x30, 0x00, 0x40, 0x00};

Cfg DispatchGraph() {
  std::vector<SourceRecord> sources{
      Record(ir::Group(0x100, {1, 2, 3, 4},
                       {{ir::Op::image_address, 64, {}, 0x4000},
                        {ir::Op::load, 32, {0}},
                        {ir::Op::zext, 64, {1}}},
                       {{8, 2}}, ir::MemoryModel::atomic_scalar_reference)),
      Record(ir::Group(0x104, {1, 2, 3, 4},
                       {{ir::Op::read, 64, {}, 0, 8},
                        {ir::Op::extract, 32, {0}, 0},
                        {ir::Op::constant, 32, {}, 3},
                        {ir::Op::unsigned_less, 1, {1, 2}},
                        {ir::Op::bit_not, 1, {3}},
                        {ir::Op::sub, 32, {1, 2}},
                        {ir::Op::constant, 32, {}, 0},
                        {ir::Op::equal, 1, {5, 6}},
                        {ir::Op::bit_not, 1, {7}},
                        {ir::Op::bit_and, 1, {4, 8}},
                        {ir::Op::image_address, 64, {}, 0x900},
                        {ir::Op::image_address, 64, {}, 0x108}},
                       {}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::conditional, 10, 9, 11, {}})),
      Record(ir::Group(0x108, {1, 2, 3, 4},
                       {{ir::Op::read, 64, {}, 0, 8},
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
                       ir::Transfer{ir::TransferKind::jump, 10, {}, {}, {}})),
      // An unrelated block, so that naming a guard with no edge here is a
      // distinguishable forgery rather than an out-of-range one.
      Jump(0x10c, 0x900)};
  const std::uint64_t entries[] = {0x100};
  const std::array<ConstantImageRange, 1> ranges{ConstantImageRange{0x5000, kEntries}};
  auto budget = Plenty();
  auto graph = BuildCfg(sources, entries, budget, {}, ImageFacts{ranges, {}, false});
  EXPECT_TRUE(graph.cfg);
  return std::move(*graph.cfg);
}

TEST(Regions, AFanOutWiderThanAForkEndsTheRegionWhereItBegins) {
  auto graph = DispatchGraph();
  ASSERT_TRUE(graph.blocks()[1].dispatch);
  auto budget = Plenty();
  auto result = BuildRegions(std::move(graph), budget);
  ASSERT_TRUE(result.regions);

  // The guard forks; the arm that bounds the index walks into the dispatch and
  // stops there rather than picking one destination as if the rest were not
  // successors. No edge is named, because none of them is the one not taken.
  unsigned reached = 0;
  for (const auto& candidate : result.regions->candidates()) {
    if (candidate.block_ids.back() != 1) continue;
    ++reached;
    EXPECT_EQ(candidate.stop, RegionStop::dispatch);
    EXPECT_FALSE(candidate.stopped_edge);
  }

  // The guard's bounded arm and the dispatch's own entry both end there.
  EXPECT_EQ(reached, 2);

  // The guard's other arm leaves the population without reaching the table.
  EXPECT_EQ(result.regions->candidates()[0].block_ids, (std::vector<std::uint32_t>{0}));
  EXPECT_EQ(result.regions->candidates()[0].stop, RegionStop::unresolved);
}

TEST(Regions, RejectsDispatchEdgesThatAreNotTheSetTheRecordDescribes) {
  const auto forged = [](auto mutate) {
    auto original = DispatchGraph();
    auto blocks = std::vector<CfgBlock>(original.blocks().begin(), original.blocks().end());
    mutate(blocks[1]);
    auto budget = Plenty();
    auto result = BuildRegions(
        Cfg({original.sources().begin(), original.sources().end()}, {0x100}, std::move(blocks), 1),
        budget);
    EXPECT_FALSE(result.regions);
    return result.reason;
  };

  // A destination out of canonical order, or repeated, is not a set.
  EXPECT_EQ(forged([](CfgBlock& block) { std::swap(block.edges[0], block.edges[1]); }),
            RegionDecline::invalid_graph);
  // An edge describing a different expression than the transfer's own target.
  EXPECT_EQ(forged([](CfgBlock& block) { block.edges[2].target.value = 0; }),
            RegionDecline::invalid_graph);
  // More destinations than the bound admits values.
  EXPECT_EQ(forged([](CfgBlock& block) { block.dispatch->bound = 2; }),
            RegionDecline::invalid_graph);
  // A guard that is the dispatch itself, or one with no edge here, bounds
  // nothing that arrives at this block.
  EXPECT_EQ(forged([](CfgBlock& block) { block.dispatch->guard_block = 1; }),
            RegionDecline::invalid_graph);
  EXPECT_EQ(forged([](CfgBlock& block) { block.dispatch->guard_block = 2; }),
            RegionDecline::invalid_graph);
  // An index that is not a value of this block.
  EXPECT_EQ(forged([](CfgBlock& block) {
              block.dispatch->index = static_cast<ir::ValueId>(block.ssa->nodes().size());
            }),
            RegionDecline::invalid_graph);
  // A wide fan-out with no record of what makes it complete.
  EXPECT_EQ(forged([](CfgBlock& block) { block.dispatch.reset(); }), RegionDecline::invalid_graph);
}

TEST(Regions, RejectsForgedGraphAndCandidateMappings) {
  auto original = Graph({Jump(0x100, 0x200), Plain(0x200)});
  auto blocks = std::vector<CfgBlock>(original.blocks().begin(), original.blocks().end());
  blocks[0].edges[0].target_block = 100;
  auto budget = Plenty();
  auto result = BuildRegions(
      Cfg({original.sources().begin(), original.sources().end()}, {0x100}, std::move(blocks), 1),
      budget);
  EXPECT_FALSE(result.regions);
  EXPECT_EQ(result.reason, RegionDecline::invalid_graph);
  Regions forged(std::move(original), {{0, {0, 100}, {0, 1}, {0}, RegionStop::unresolved, {}}});
  EXPECT_FALSE(NormalizeRegion(forged, 0, budget).path);
  EXPECT_FALSE(NormalizeRegion(forged, 100, budget).path);
}

TEST(Regions, EveryResourceCutDeclinesTransactionallyAndExactBudgetSucceeds) {
  const auto original =
      Graph({Branch(0x100, 0x200, 0x300), Jump(0x200, 0x400), Plain(0x300), Plain(0x400)});
  auto budget = Plenty();
  auto copy = original;
  auto full = BuildRegions(std::move(copy), budget);
  ASSERT_TRUE(full.regions);
  const auto required = budget.used();
  for (std::uint64_t work = 0; work < required.work; ++work) {
    Budget limited({work, UINT64_MAX});
    copy = original;
    auto result = BuildRegions(std::move(copy), limited);
    EXPECT_FALSE(result.regions);
    EXPECT_EQ(result.reason, RegionDecline::resource_limit);
  }

  for (std::uint64_t bytes = 0; bytes < required.bytes; ++bytes) {
    Budget limited({UINT64_MAX, bytes});
    copy = original;
    auto result = BuildRegions(std::move(copy), limited);
    EXPECT_FALSE(result.regions);
    EXPECT_EQ(result.reason, RegionDecline::resource_limit);
  }

  Budget exact(required);
  copy = original;
  EXPECT_TRUE(BuildRegions(std::move(copy), exact).regions);
  RegionLimits limits;
  limits.max_candidates = 1;
  copy = original;
  EXPECT_EQ(BuildRegions(std::move(copy), budget, limits).reason, RegionDecline::resource_limit);
  limits.max_candidates = 100;
  limits.max_total_source_occurrences = 1;
  copy = original;
  EXPECT_EQ(BuildRegions(std::move(copy), budget, limits).reason, RegionDecline::resource_limit);
}

TEST(Regions, NormalizationChargesAllCopiesBeforePublishingAPath) {
  auto budget = Plenty();
  auto result = BuildRegions(Graph({Plain(0x100), Jump(0x104, 0x200), Plain(0x200)}), budget);
  ASSERT_TRUE(result.regions);
  auto measure = Plenty();
  ASSERT_TRUE(NormalizeRegion(*result.regions, 0, measure).path);
  const auto required = measure.used();
  for (std::uint64_t work = 0; work < required.work; ++work) {
    Budget limited({work, UINT64_MAX});
    const auto path = NormalizeRegion(*result.regions, 0, limited);
    EXPECT_FALSE(path.path);
    EXPECT_EQ(path.reason, ir::BlockDecline::resource_limit);
  }

  for (std::uint64_t bytes = 0; bytes < required.bytes; ++bytes) {
    Budget limited({UINT64_MAX, bytes});
    const auto path = NormalizeRegion(*result.regions, 0, limited);
    EXPECT_FALSE(path.path);
    EXPECT_EQ(path.reason, ir::BlockDecline::resource_limit);
  }

  Budget exact(required);
  EXPECT_TRUE(NormalizeRegion(*result.regions, 0, exact).path);
}

TEST(Regions, AbsoluteAndMidInstructionTargetsAreNeverFollowed) {
  auto absolute = Record(ir::Group(0x100, {1, 2, 3, 4}, {{ir::Op::constant, 64, {}, 0x300}}, {},
                                   ir::MemoryModel::unspecified,
                                   ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}}));
  auto budget = Plenty();
  auto result =
      BuildRegions(Graph({std::move(absolute), Jump(0x200, 0x301), Plain(0x300)}), budget);
  ASSERT_TRUE(result.regions);
  const auto candidates = result.regions->candidates();
  ASSERT_EQ(candidates.size(), 3);
  EXPECT_EQ(candidates[0].source_ids.size(), 1);
  EXPECT_EQ(candidates[1].source_ids.size(), 1);
  EXPECT_EQ(candidates[0].stop, RegionStop::unresolved);
  EXPECT_EQ(candidates[1].stop, RegionStop::unresolved);
  EXPECT_EQ(result.regions->graph().blocks()[0].edges[0].resolution,
            TargetResolution::absolute_runtime);
  EXPECT_EQ(result.regions->graph().blocks()[1].edges[0].resolution,
            TargetResolution::mid_instruction);
}

}  // namespace
}  // namespace nyx::analysis
