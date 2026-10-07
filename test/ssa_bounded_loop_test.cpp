#include <algorithm>

#include <gtest/gtest.h>

#include "nyx/analysis/ssa/bounded_loop.hpp"
#include "nyx/analysis/ssa/build.hpp"
#include "nyx/analysis/ssa/constants.hpp"
#include "nyx/analysis/ssa/reachability.hpp"
#include "nyx/analysis/ssa/sccp.hpp"

namespace nyx::analysis {
namespace {
Budget Plenty() { return Budget({10000000, 100000000}); }

struct Fixture {
  ir::SsaGraph graph;
  std::vector<ir::Group> sources;
};

// x8 = 0; do { x8 = x8 + 1; } while (x8 != trip); return x8 + 1
// The loop block branches to itself, so its phi for x8 disagrees with itself
// on every iteration and no lattice settles it. Running it does.
struct Options {
  std::uint64_t trip = 3;

  // Enter the loop block architecturally as well as from its predecessor.
  bool loop_is_entry = false;

  // Give the loop a second way in besides its one predecessor.
  bool second_predecessor = false;

  // Give the loop body a load nothing resolves.
  bool opaque_load = false;
};

std::optional<Fixture> Counter(Budget& budget, Options options) {
  std::vector<ir::Group> sources;
  sources.emplace_back(
      0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
      std::vector<ir::Node>{{ir::Op::constant, 64, {}, 0}, {ir::Op::image_address, 64, {}, 0x104}},
      std::vector<ir::Write>{{8, 0}}, ir::MemoryModel::unspecified,
      ir::Transfer{ir::TransferKind::jump, 1, {}, {}, {}});
  std::vector<ir::Node> body{{ir::Op::read, 64, {}, 0, 8},
                             {ir::Op::constant, 64, {}, 1},
                             {ir::Op::add, 64, {0, 1}},
                             {ir::Op::constant, 64, {}, options.trip},
                             {ir::Op::equal, 1, {0, 3}},
                             {ir::Op::image_address, 64, {}, 0x108},
                             {ir::Op::image_address, 64, {}, 0x104}};
  if (options.opaque_load) {
    body.push_back({ir::Op::read, 64, {}, 0, 9});
    body.push_back({ir::Op::load, 64, {7}});
  }

  sources.emplace_back(
      0x104, std::vector<std::uint8_t>{5, 6, 7, 8}, std::move(body), std::vector<ir::Write>{{8, 2}},
      options.opaque_load ? ir::MemoryModel::atomic_scalar_reference : ir::MemoryModel::unspecified,
      ir::Transfer{ir::TransferKind::conditional, 5, 4, 6, {}});
  sources.emplace_back(0x108, std::vector<std::uint8_t>{9, 10, 11, 12},
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 8},
                                             {ir::Op::constant, 64, {}, 1},
                                             {ir::Op::add, 64, {0, 1}},
                                             {ir::Op::read, 64, {}, 0, 30}},
                       std::vector<ir::Write>{{0, 2}}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::return_, 3, {}, {}, {}});
  if (options.second_predecessor)
    sources.emplace_back(0x10c, std::vector<std::uint8_t>{17, 18, 19, 20},
                         std::vector<ir::Node>{{ir::Op::constant, 64, {}, 1},
                                               {ir::Op::image_address, 64, {}, 0x104}},
                         std::vector<ir::Write>{{8, 0}}, ir::MemoryModel::unspecified,
                         ir::Transfer{ir::TransferKind::jump, 1, {}, {}, {}});
  std::vector<SourceRecord> records;
  for (const auto& source : sources)
    records.push_back(
        {source.source_address(), {source.bytes().begin(), source.bytes().end()}, source});
  std::vector<std::uint64_t> entries{0x100};
  if (options.loop_is_entry) entries.push_back(0x104);
  if (options.second_predecessor) entries.push_back(0x10c);
  auto cfg = BuildCfg(records, entries, budget);
  if (!cfg.cfg) return {};
  Regions regions(std::move(*cfg.cfg), {});
  auto recovered = Unflatten(regions, {}, budget, {{}, true});
  if (!recovered.unflattening) return {};
  const std::vector<std::optional<ir::RecoveredPath>> paths;
  auto built = BuildSsa(regions, *recovered.unflattening, paths, budget);
  if (!built.graph) return {};
  return Fixture{std::move(*built.graph), std::move(sources)};
}

std::optional<ir::SsaHandle> At(const ir::SsaGraph& graph, std::uint64_t address) {
  for (std::size_t slot = 0; slot < graph.slots(); ++slot)
    if (const auto handle = graph.Handle(slot); handle && graph.Get(*handle)->address == address)
      return handle;
  return {};
}

TEST(SsaBoundedLoop, RunsASelfLoopToItsExitValues) {
  auto budget = Plenty();
  auto fixture = Counter(budget, {});
  ASSERT_TRUE(fixture);
  auto proved = ProveSsaBoundedLoops(fixture->graph, fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  ASSERT_EQ(proved.facts->loops.size(), 1u);
  const auto& loop = proved.facts->loops[0];
  EXPECT_EQ(loop.loop, At(fixture->graph, 0x104));
  EXPECT_EQ(loop.successor, At(fixture->graph, 0x108));

  // x8 is 0, 1, 2 and then 3, which is the value the test compares.
  EXPECT_EQ(loop.iterations, 4u);
  ASSERT_FALSE(loop.exits.empty());
  const auto x8 =
      std::find_if(loop.exits.begin(), loop.exits.end(), [&](const ir::SsaBoundedLoopExit& exit) {
        return exit.kind == ir::SsaValueKind::phi &&
               fixture->graph.Get(loop.successor)->phis[exit.index].storage == 8;
      });
  ASSERT_NE(x8, loop.exits.end());
  EXPECT_EQ(x8->value, 4u);
}

TEST(SsaBoundedLoop, AWrongIterationCountIsRejected) {
  auto budget = Plenty();
  auto fixture = Counter(budget, {});
  ASSERT_TRUE(fixture);
  auto proved = ProveSsaBoundedLoops(fixture->graph, fixture->sources, budget);
  ASSERT_TRUE(proved.facts);
  auto tampered = *proved.facts;
  tampered.loops[0].iterations = 3;
  EXPECT_EQ(ir::ValidateSsaBoundedLoopFacts(fixture->graph, tampered, fixture->sources, budget),
            ir::SsaDecline::invalid_graph);
}

TEST(SsaBoundedLoop, AWrongExitValueIsRejected) {
  auto budget = Plenty();
  auto fixture = Counter(budget, {});
  ASSERT_TRUE(fixture);
  auto proved = ProveSsaBoundedLoops(fixture->graph, fixture->sources, budget);
  ASSERT_TRUE(proved.facts);
  ASSERT_FALSE(proved.facts->loops[0].exits.empty());
  auto tampered = *proved.facts;
  tampered.loops[0].exits[0].value ^= 1;
  EXPECT_EQ(ir::ValidateSsaBoundedLoopFacts(fixture->graph, tampered, fixture->sources, budget),
            ir::SsaDecline::invalid_graph);
}

TEST(SsaBoundedLoop, AnOmittedExitValueIsRejected) {
  auto budget = Plenty();
  auto fixture = Counter(budget, {});
  ASSERT_TRUE(fixture);
  auto proved = ProveSsaBoundedLoops(fixture->graph, fixture->sources, budget);
  ASSERT_TRUE(proved.facts);
  auto tampered = *proved.facts;
  tampered.loops[0].exits.pop_back();
  EXPECT_EQ(ir::ValidateSsaBoundedLoopFacts(fixture->graph, tampered, fixture->sources, budget),
            ir::SsaDecline::invalid_graph);
}

TEST(SsaBoundedLoop, ALoopPastTheIterationCapProvesNothing) {
  auto budget = Plenty();
  auto fixture = Counter(budget, {ir::kMaxBoundedLoopIterations + 1});
  ASSERT_TRUE(fixture);
  auto proved = ProveSsaBoundedLoops(fixture->graph, fixture->sources, budget);
  ASSERT_TRUE(proved.facts);
  EXPECT_TRUE(proved.facts->loops.empty());
}

// The point of the fact: the constant lattice reaches the value only with it.
TEST(SsaBoundedLoop, TheConstantLatticeNeedsTheRunToSettleTheExit) {
  auto budget = Plenty();
  auto fixture = Counter(budget, {});
  ASSERT_TRUE(fixture);
  auto reachable = ProveSsaReachability(fixture->graph, fixture->sources,
                                        ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts);
  const auto successor = At(fixture->graph, 0x108);
  ASSERT_TRUE(successor);
  const auto settled = [&](const ir::SsaConstantFacts& facts) {
    return std::any_of(facts.values.begin(), facts.values.end(),
                       [&](const ir::SsaConstantValue& value) {
                         return value.block == *successor && value.kind == ir::SsaValueKind::phi &&
                                value.value == 4;
                       });
  };

  auto without = ProveSsaConstants(fixture->graph, *reachable.facts, fixture->sources, budget);
  ASSERT_TRUE(without.facts);
  EXPECT_FALSE(settled(*without.facts));

  auto loops = ProveSsaBoundedLoops(fixture->graph, fixture->sources, budget);
  ASSERT_TRUE(loops.facts);
  auto with =
      ProveSsaConstants(fixture->graph, *reachable.facts, fixture->sources, budget, &*loops.facts);
  ASSERT_TRUE(with.facts) << static_cast<int>(with.reason);
  EXPECT_TRUE(settled(*with.facts));
}

// A fact the loop facts do not license still has to satisfy the ordinary
// rule, so passing loop facts cannot wave a constant through.
TEST(SsaBoundedLoop, LoopFactsDoNotLicenseAnUnrelatedConstant) {
  auto budget = Plenty();
  auto fixture = Counter(budget, {});
  ASSERT_TRUE(fixture);
  auto reachable = ProveSsaReachability(fixture->graph, fixture->sources,
                                        ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts);
  auto loops = ProveSsaBoundedLoops(fixture->graph, fixture->sources, budget);
  ASSERT_TRUE(loops.facts);
  auto proved =
      ProveSsaConstants(fixture->graph, *reachable.facts, fixture->sources, budget, &*loops.facts);
  ASSERT_TRUE(proved.facts);
  auto tampered = *proved.facts;
  for (auto& value : tampered.values) value.value ^= 1;
  EXPECT_EQ(ir::ValidateSsaConstantFacts(fixture->graph, *reachable.facts, tampered,
                                         fixture->sources, budget, &*loops.facts),
            ir::SsaDecline::invalid_graph);
}

// An entry is reached with storage from outside the graph, so what its one
// other predecessor publishes is not what it runs from, and a value read off
// that predecessor would be wrong for every other way in.
TEST(SsaBoundedLoop, ALoopThatIsAlsoAnEntryProvesNothing) {
  auto budget = Plenty();
  auto fixture = Counter(budget, {3, true});
  ASSERT_TRUE(fixture);
  const auto loop = At(fixture->graph, 0x104);
  ASSERT_TRUE(loop);
  ASSERT_TRUE(std::any_of(fixture->graph.Get(*loop)->phis.begin(),
                          fixture->graph.Get(*loop)->phis.end(),
                          [](const ir::SsaPhi& phi) { return phi.external_entry; }));
  auto proved = ProveSsaBoundedLoops(fixture->graph, fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  EXPECT_TRUE(proved.facts->loops.empty());
}

// An access the state does not model can carry a value between iterations,
// and then every value below it is a guess.
TEST(SsaBoundedLoop, AnAccessNothingModelsStopsTheRun) {
  auto budget = Plenty();
  auto fixture = Counter(budget, {3, false, false, true});
  ASSERT_TRUE(fixture);
  const auto loop = At(fixture->graph, 0x104);
  ASSERT_TRUE(loop);
  const auto& block = *fixture->graph.Get(*loop);
  ASSERT_TRUE(std::any_of(block.nodes.begin(), block.nodes.end(),
                          [](const ir::Node& node) { return node.op == ir::Op::load; }));
  auto proved = ProveSsaBoundedLoops(fixture->graph, fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  EXPECT_TRUE(proved.facts->loops.empty());
}

TEST(SsaBoundedLoop, TheSccpLatticeAlsoTakesTheRun) {
  auto budget = Plenty();
  auto fixture = Counter(budget, {});
  ASSERT_TRUE(fixture);
  const auto successor = At(fixture->graph, 0x108);
  ASSERT_TRUE(successor);
  const auto settled = [&](const ir::SsaSccpFacts& facts) {
    return std::any_of(facts.constants.values.begin(), facts.constants.values.end(),
                       [&](const ir::SsaConstantValue& value) {
                         return value.block == *successor && value.kind == ir::SsaValueKind::phi &&
                                value.value == 4;
                       });
  };

  auto without =
      ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population, fixture->sources, budget);
  ASSERT_TRUE(without.facts);
  EXPECT_FALSE(settled(*without.facts));
  auto loops = ProveSsaBoundedLoops(fixture->graph, fixture->sources, budget);
  ASSERT_TRUE(loops.facts);
  auto with = ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population, fixture->sources,
                           budget, &*loops.facts);
  ASSERT_TRUE(with.facts) << static_cast<int>(with.reason);
  EXPECT_TRUE(settled(*with.facts));
}

// Running out of budget must be reported as such, never as "this is not a
// loop".
TEST(SsaBoundedLoop, ATightBudgetDeclinesRatherThanDisagrees) {
  auto setup = Plenty();
  auto fixture = Counter(setup, {});
  ASSERT_TRUE(fixture);
  auto proved = ProveSsaBoundedLoops(fixture->graph, fixture->sources, setup);
  ASSERT_TRUE(proved.facts);
  ASSERT_EQ(proved.facts->loops.size(), 1u);

  // Sweep every budget, including those past the first refusal, to reach the
  // window where the run itself runs out after the earlier checks pass.
  bool saw_limit = false, saw_none = false;
  for (std::uint64_t work = 1; work <= 4000; work += 1) {
    Budget tight({work, 1 << 20});
    const auto checked =
        ir::ValidateSsaBoundedLoopFacts(fixture->graph, *proved.facts, fixture->sources, tight);
    if (checked == ir::SsaDecline::resource_limit)
      saw_limit = true;
    else {
      saw_none = true;
      EXPECT_EQ(checked, ir::SsaDecline::none) << "at work budget " << work;
    }
  }

  for (std::uint64_t bytes = 0; bytes <= 4000; bytes += 1) {
    Budget tight({1 << 20, bytes});
    const auto checked =
        ir::ValidateSsaBoundedLoopFacts(fixture->graph, *proved.facts, fixture->sources, tight);
    if (checked != ir::SsaDecline::resource_limit) {
      EXPECT_EQ(checked, ir::SsaDecline::none) << "at byte budget " << bytes;
    }
  }

  EXPECT_TRUE(saw_limit);
  EXPECT_TRUE(saw_none);
}

// The entry values come from the one other predecessor the shape test found.
// With two, there is no such thing, and taking either would be a run of a
// state the block need not be in.
TEST(SsaBoundedLoop, ASecondPredecessorProvesNothing) {
  auto budget = Plenty();
  auto fixture = Counter(budget, {3, false, true});
  ASSERT_TRUE(fixture);
  const auto loop = At(fixture->graph, 0x104);
  ASSERT_TRUE(loop);
  std::size_t predecessors = 0;
  for (std::size_t slot = 0; slot < fixture->graph.slots(); ++slot) {
    const auto handle = fixture->graph.Handle(slot);
    if (!handle) continue;
    for (const auto& edge : fixture->graph.Get(*handle)->edges)
      if (edge.target_block == loop) ++predecessors;
  }

  ASSERT_EQ(predecessors, 3u);
  auto proved = ProveSsaBoundedLoops(fixture->graph, fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  EXPECT_TRUE(proved.facts->loops.empty());
}

}  // namespace
}  // namespace nyx::analysis
