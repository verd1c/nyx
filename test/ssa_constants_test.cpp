#include <algorithm>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "nyx/analysis/ssa/build.hpp"
#include "nyx/analysis/ssa/cache.hpp"
#include "nyx/analysis/ssa/constants.hpp"
#include "nyx/analysis/ssa/copy.hpp"
#include "nyx/analysis/ssa/phi_constants.hpp"
#include "nyx/analysis/ssa/reachability.hpp"
#include "nyx/analysis/ssa/sccp.hpp"
#include "nyx/recovery/constant_load.hpp"
#include "nyx/recovery/ssa/branch_retirement.hpp"
#include "nyx/recovery/ssa/constants.hpp"
#include "nyx/recovery/ssa/copy.hpp"
#include "nyx/recovery/ssa/dce.hpp"
#include "nyx/recovery/ssa/simplify.hpp"

namespace nyx::analysis {
namespace {
Budget Plenty() { return Budget({1000000, 10000000}); }

struct Fixture {
  ir::SsaGraph graph;
  std::vector<ir::Group> sources;
};

std::optional<Fixture> Join(Budget& budget, std::uint64_t right_value = 11,
                            bool return_leaves = true, std::optional<bool> entry_condition = {}) {
  std::vector<ir::Group> sources;
  sources.emplace_back(
      0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
      std::vector<ir::Node>{entry_condition
                                ? ir::Node{ir::Op::constant, 1, {}, *entry_condition ? 1U : 0U}
                                : ir::Node{ir::Op::read, 1, {}, 0, 32},
                            {ir::Op::image_address, 64, {}, 0x108},
                            {ir::Op::image_address, 64, {}, 0x104}},
      std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
      ir::Transfer{ir::TransferKind::conditional, 1, 0, 2, {}});
  sources.emplace_back(0x104, std::vector<std::uint8_t>{5, 6, 7, 8},
                       std::vector<ir::Node>{{ir::Op::constant, 64, {}, 10},
                                             {ir::Op::constant, 64, {}, 1},
                                             {ir::Op::add, 64, {0, 1}},
                                             {ir::Op::image_address, 64, {}, 0x10c}},
                       std::vector<ir::Write>{{8, 2}}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 3, {}, {}, {}});
  sources.emplace_back(0x108, std::vector<std::uint8_t>{9, 10, 11, 12},
                       std::vector<ir::Node>{{ir::Op::constant, 64, {}, right_value},
                                             {ir::Op::image_address, 64, {}, 0x10c}},
                       std::vector<ir::Write>{{8, 0}}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 1, {}, {}, {}});
  sources.emplace_back(0x10c, std::vector<std::uint8_t>{13, 14, 15, 16},
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 8},
                                             {ir::Op::constant, 64, {}, 1},
                                             {ir::Op::add, 64, {0, 1}},
                                             {ir::Op::read, 64, {}, 0, 30}},
                       std::vector<ir::Write>{{0, 2}}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::return_, 3, {}, {}, {}});
  std::vector<SourceRecord> records;
  for (const auto& source : sources)
    records.push_back(
        {source.source_address(), {source.bytes().begin(), source.bytes().end()}, source});
  constexpr std::uint64_t entry = 0x100;
  auto cfg = BuildCfg(records, std::span(&entry, 1), budget);
  if (!cfg.cfg) return {};
  Regions regions(std::move(*cfg.cfg), {});
  auto recovered = Unflatten(regions, {}, budget, {{}, return_leaves});
  if (!recovered.unflattening) return {};
  const std::vector<std::optional<ir::RecoveredPath>> paths;
  auto built = BuildSsa(regions, *recovered.unflattening, paths, budget);
  if (!built.graph) return {};
  return Fixture{std::move(*built.graph), std::move(sources)};
}

std::optional<Fixture> Loop(Budget& budget) {
  std::vector<ir::Group> sources;
  sources.emplace_back(
      0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
      std::vector<ir::Node>{{ir::Op::constant, 64, {}, 11}, {ir::Op::image_address, 64, {}, 0x104}},
      std::vector<ir::Write>{{8, 0}}, ir::MemoryModel::unspecified,
      ir::Transfer{ir::TransferKind::jump, 1, {}, {}, {}});
  sources.emplace_back(0x104, std::vector<std::uint8_t>{5, 6, 7, 8},
                       std::vector<ir::Node>{{ir::Op::read, 1, {}, 0, 32},
                                             {ir::Op::image_address, 64, {}, 0x108},
                                             {ir::Op::image_address, 64, {}, 0x10c}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::conditional, 1, 0, 2, {}});
  sources.emplace_back(0x108, std::vector<std::uint8_t>{9, 10, 11, 12},
                       std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x104}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}});
  sources.emplace_back(0x10c, std::vector<std::uint8_t>{13, 14, 15, 16},
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 8},
                                             {ir::Op::constant, 64, {}, 1},
                                             {ir::Op::add, 64, {0, 1}},
                                             {ir::Op::read, 64, {}, 0, 30}},
                       std::vector<ir::Write>{{0, 2}}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::return_, 3, {}, {}, {}});
  std::vector<SourceRecord> records;
  for (const auto& source : sources)
    records.push_back(
        {source.source_address(), {source.bytes().begin(), source.bytes().end()}, source});
  constexpr std::uint64_t entry = 0x100;
  auto cfg = BuildCfg(records, std::span(&entry, 1), budget);
  if (!cfg.cfg) return {};
  Regions regions(std::move(*cfg.cfg), {});
  auto recovered = Unflatten(regions, {}, budget, {{}, true});
  if (!recovered.unflattening) return {};
  const std::vector<std::optional<ir::RecoveredPath>> paths;
  auto built = BuildSsa(regions, *recovered.unflattening, paths, budget);
  if (!built.graph) return {};
  return Fixture{std::move(*built.graph), std::move(sources)};
}

// `bl` to a target outside the population, then a use of x0. What the callee
// left is otherwise fresh state; a declared result says what it left.
std::optional<Fixture> Calling(Budget& budget) {
  std::vector<ir::Group> sources;
  sources.emplace_back(0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
                       std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x900},
                                             {ir::Op::image_address, 64, {}, 0x104}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::call, 0, {}, {}, 1});
  sources.emplace_back(0x104, std::vector<std::uint8_t>{5, 6, 7, 8},
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 0},
                                             {ir::Op::constant, 64, {}, 0x60},
                                             {ir::Op::add, 64, {0, 1}},
                                             {ir::Op::read, 64, {}, 0, 30}},
                       std::vector<ir::Write>{{0, 2}}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::return_, 3, {}, {}, {}});
  std::vector<SourceRecord> records;
  for (const auto& source : sources)
    records.push_back(
        {source.source_address(), {source.bytes().begin(), source.bytes().end()}, source});
  constexpr std::uint64_t entry = 0x100;
  auto cfg = BuildCfg(records, std::span(&entry, 1), budget);
  if (!cfg.cfg) return {};
  Regions regions(std::move(*cfg.cfg), {});
  auto recovered = Unflatten(regions, {}, budget, {{}, true});
  if (!recovered.unflattening) return {};
  const std::vector<std::optional<ir::RecoveredPath>> paths;
  auto built = BuildSsa(regions, *recovered.unflattening, paths, budget);
  if (!built.graph) return {};
  return Fixture{std::move(*built.graph), std::move(sources)};
}

// A conditional whose false arm ends in `br x16`: the one block whose
// successors nothing enumerates sits behind the guard.
std::optional<Fixture> GuardedIndirect(Budget& budget, std::optional<bool> condition) {
  std::vector<ir::Group> sources;
  sources.emplace_back(
      0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
      std::vector<ir::Node>{condition ? ir::Node{ir::Op::constant, 1, {}, *condition ? 1U : 0U}
                                      : ir::Node{ir::Op::read, 1, {}, 0, 32},
                            {ir::Op::image_address, 64, {}, 0x108},
                            {ir::Op::image_address, 64, {}, 0x104}},
      std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
      ir::Transfer{ir::TransferKind::conditional, 1, 0, 2, {}});
  sources.emplace_back(0x104, std::vector<std::uint8_t>{5, 6, 7, 8},
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 16}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}});
  sources.emplace_back(0x108, std::vector<std::uint8_t>{9, 10, 11, 12},
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 30}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}});
  std::vector<SourceRecord> records;
  for (const auto& source : sources)
    records.push_back(
        {source.source_address(), {source.bytes().begin(), source.bytes().end()}, source});
  constexpr std::uint64_t entry = 0x100;
  auto cfg = BuildCfg(records, std::span(&entry, 1), budget);
  if (!cfg.cfg) return {};
  Regions regions(std::move(*cfg.cfg), {});
  auto recovered = Unflatten(regions, {}, budget, {{}, true});
  if (!recovered.unflattening) return {};
  const std::vector<std::optional<ir::RecoveredPath>> paths;
  auto built = BuildSsa(regions, *recovered.unflattening, paths, budget);
  if (!built.graph) return {};
  return Fixture{std::move(*built.graph), std::move(sources)};
}

// A branch the entry's literal condition always takes to 0x108, and the arm
// it never takes: 0x104 writes x1, then 0x10c reads it. With
// `arm_reads_entry` 0x104 writes x1 from the x0 the entry set; otherwise
// from a literal of its own.
std::optional<Fixture> DeadArm(Budget& budget, bool arm_reads_entry) {
  std::vector<ir::Group> sources;
  sources.emplace_back(0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
                       std::vector<ir::Node>{{ir::Op::constant, 1, {}, 1},
                                             {ir::Op::image_address, 64, {}, 0x108},
                                             {ir::Op::image_address, 64, {}, 0x104},
                                             {ir::Op::constant, 64, {}, 5}},
                       std::vector<ir::Write>{{0, 3}}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::conditional, 1, 0, 2, {}});
  sources.emplace_back(
      0x104, std::vector<std::uint8_t>{5, 6, 7, 8},
      std::vector<ir::Node>{arm_reads_entry ? ir::Node{ir::Op::read, 64, {}, 0, 0}
                                            : ir::Node{ir::Op::constant, 64, {}, 7},
                            {ir::Op::image_address, 64, {}, 0x10c}},
      std::vector<ir::Write>{{1, 0}}, ir::MemoryModel::unspecified,
      ir::Transfer{ir::TransferKind::jump, 1, {}, {}, {}});
  sources.emplace_back(0x108, std::vector<std::uint8_t>{9, 10, 11, 12},
                       std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x110}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}});
  sources.emplace_back(
      0x10c, std::vector<std::uint8_t>{13, 14, 15, 16},
      std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 1}, {ir::Op::image_address, 64, {}, 0x110}},
      std::vector<ir::Write>{{2, 0}}, ir::MemoryModel::unspecified,
      ir::Transfer{ir::TransferKind::jump, 1, {}, {}, {}});
  sources.emplace_back(0x110, std::vector<std::uint8_t>{17, 18, 19, 20},
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 30}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}});
  std::vector<SourceRecord> records;
  for (const auto& source : sources)
    records.push_back(
        {source.source_address(), {source.bytes().begin(), source.bytes().end()}, source});
  constexpr std::uint64_t entry = 0x100;
  auto cfg = BuildCfg(records, std::span(&entry, 1), budget);
  if (!cfg.cfg) return {};
  Regions regions(std::move(*cfg.cfg), {});
  auto recovered = Unflatten(regions, {}, budget, {{}, true});
  if (!recovered.unflattening) return {};
  const std::vector<std::optional<ir::RecoveredPath>> paths;
  auto built = BuildSsa(regions, *recovered.unflattening, paths, budget);
  if (!built.graph) return {};
  return Fixture{std::move(*built.graph), std::move(sources)};
}

// The graph with every single-predecessor read recorded as a copy.
std::optional<ir::SsaGraph> WithCopies(const Fixture& fixture, Budget& budget) {
  auto reachable = ProveSsaReachability(fixture.graph, fixture.sources,
                                        ir::SsaEntryScope::closed_population, budget);
  if (!reachable.facts) return {};
  auto proved = ProveSsaPredecessorCopies(fixture.graph, *reachable.facts, fixture.sources, budget);
  if (!proved.facts) return {};
  return recovery::ProposeSsaPredecessorCopies(fixture.graph, *reachable.facts, *proved.facts,
                                               fixture.sources, budget)
      .provisional;
}

// `br` through a slot at 0x800 the loader fills with 0x108, the return block.
// With `relocated` the run declares that slot; without, nothing says what it
// holds.
std::optional<Fixture> SlotJump(Budget& budget, bool relocated) {
  std::vector<ir::Group> sources;
  sources.emplace_back(
      0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
      std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x800}, {ir::Op::load, 64, {0}}},
      std::vector<ir::Write>{}, ir::MemoryModel::atomic_scalar_reference,
      ir::Transfer{ir::TransferKind::jump, 1, {}, {}, {}});
  sources.emplace_back(0x108, std::vector<std::uint8_t>{5, 6, 7, 8},
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 30}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}});
  std::vector<SourceRecord> records;
  for (const auto& source : sources)
    records.push_back(
        {source.source_address(), {source.bytes().begin(), source.bytes().end()}, source});
  const ir::RelocatedPointer slot{0x800, 0x108};
  const ImageFacts facts{
      {}, relocated ? std::span(&slot, 1) : std::span<const ir::RelocatedPointer>{}, false, {}};
  constexpr std::uint64_t entry = 0x100;
  auto cfg = BuildCfg(records, std::span(&entry, 1), budget, {}, facts);
  if (!cfg.cfg) return {};
  Regions regions(std::move(*cfg.cfg), {});
  auto recovered = Unflatten(regions, {}, budget, {{}, true});
  if (!recovered.unflattening) return {};
  const std::vector<std::optional<ir::RecoveredPath>> paths(regions.candidates().size());
  auto built = BuildSsa(regions, *recovered.unflattening, paths, budget, nullptr, facts);
  if (!built.graph) return {};
  return Fixture{std::move(*built.graph), std::move(sources)};
}

const ir::SsaBlock* At(const ir::SsaGraph& graph, std::uint64_t address) {
  for (std::size_t slot = 0; slot < graph.slots(); ++slot)
    if (const auto handle = graph.Handle(slot); handle && graph.Get(*handle)->address == address)
      return graph.Get(*handle);
  return nullptr;
}

TEST(SsaConstants, SccpAsksCompletenessOnlyOfExecutableBlocks) {
  auto budget = Plenty();
  auto fixture = GuardedIndirect(budget, true);
  ASSERT_TRUE(fixture);
  std::optional<ir::SsaHandle> indirect;
  for (std::size_t slot = 0; slot < fixture->graph.slots(); ++slot) {
    const auto handle = fixture->graph.Handle(slot);
    if (handle && fixture->graph.Get(*handle)->address == 0x104) indirect = handle;
  }

  ASSERT_TRUE(indirect);
  EXPECT_NE(ir::ValidateSsaDirectSuccessors(*fixture->graph.Get(*indirect), budget),
            ir::SsaDecline::none);
  EXPECT_EQ(ProveSsaReachability(fixture->graph, fixture->sources,
                                 ir::SsaEntryScope::closed_population, budget)
                .reason,
            SsaReachabilityRefusal::incomplete_successors);
  auto proved =
      ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population, fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  EXPECT_EQ(proved.facts->executable[indirect->slot], 0);
  EXPECT_EQ(std::count(proved.facts->executable.begin(), proved.facts->executable.end(), 1), 2);
  EXPECT_EQ(ir::ValidateSsaSccpFacts(fixture->graph, *proved.facts, fixture->sources, budget),
            ir::SsaDecline::none);
  EXPECT_EQ(
      ProveSsaSccp(fixture->graph, ir::SsaEntryScope::discovered_only, fixture->sources, budget)
          .reason,
      SsaSccpRefusal::open_entries);

  // Once the guard is not decided the jump can run, and what it reaches is
  // outside anything the proof enumerated.
  auto unknown = GuardedIndirect(budget, {});
  ASSERT_TRUE(unknown);
  const auto refused =
      ProveSsaSccp(unknown->graph, ir::SsaEntryScope::closed_population, unknown->sources, budget);
  EXPECT_FALSE(refused.facts);
  EXPECT_EQ(refused.reason, SsaSccpRefusal::incomplete_successors);
}

TEST(SsaConstants, ADeclaredCallResultIsWhatTheCalleeLeft) {
  auto budget = Plenty();
  auto fixture = Calling(budget);
  ASSERT_TRUE(fixture);
  const auto proved = [&](Budget& used) {
    auto reachable = ProveSsaReachability(fixture->graph, fixture->sources,
                                          ir::SsaEntryScope::closed_population, used);
    EXPECT_TRUE(reachable.facts);
    if (!reachable.facts) return std::optional<ir::SsaConstantFacts>{};
    auto result = ProveSsaConstants(fixture->graph, *reachable.facts, fixture->sources, used);
    return result.facts;
  };

  // Without a declaration the callee's effect on x0 is fresh state, so the
  // sum that reads it is not constant.
  auto before = proved(budget);
  ASSERT_TRUE(before);
  const auto sums = [&](const ir::SsaConstantFacts& facts) {
    return std::count_if(facts.values.begin(), facts.values.end(),
                         [](const ir::SsaConstantValue& value) { return value.value == 0x960; });
  };

  EXPECT_EQ(sums(*before), 0);

  fixture->graph.SetCallResults({{0x900, 0, 0x900}});
  auto budget2 = Plenty();
  auto after = proved(budget2);
  ASSERT_TRUE(after);

  // x0 is 0x900 on return, so x0 + 0x60 is a constant the graph can carry, and
  // the independent recheck agrees because it reads the same declaration.
  EXPECT_EQ(sums(*after), 1);

  // A declaration naming another callee says nothing about this one.
  fixture->graph.SetCallResults({{0x901, 0, 0x900}});
  auto budget3 = Plenty();
  auto elsewhere = proved(budget3);
  ASSERT_TRUE(elsewhere);
  EXPECT_EQ(sums(*elsewhere), 0);
}

TEST(SsaConstants, PropagatesAComputedPredecessorValueAcrossTheJoin) {
  auto budget = Plenty();
  auto fixture = Join(budget);
  ASSERT_TRUE(fixture);
  auto reachable = ProveSsaReachability(fixture->graph, fixture->sources,
                                        ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts);
  auto simple = ProveSsaPhiConstants(fixture->graph, *reachable.facts, fixture->sources, budget);
  ASSERT_TRUE(simple.facts);
  EXPECT_TRUE(simple.facts->constants.empty());
  auto proved = ProveSsaConstants(fixture->graph, *reachable.facts, fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  const auto join = std::find_if(proved.facts->values.begin(), proved.facts->values.end(),
                                 [&](const ir::SsaConstantValue& fact) {
                                   const auto* block = fixture->graph.Get(fact.block);
                                   return block && block->address == 0x10c &&
                                          fact.kind == ir::SsaValueKind::phi &&
                                          block->phis[fact.index].storage == 8;
                                 });
  ASSERT_NE(join, proved.facts->values.end());
  EXPECT_EQ(join->value, 11U);
  auto result = recovery::ProposeSsaConstantFold(fixture->graph, *reachable.facts, *proved.facts,
                                                 fixture->sources, budget);
  ASSERT_TRUE(result.provisional) << static_cast<int>(result.reason);
  EXPECT_EQ(result.journal.size(), 2U);
  EXPECT_TRUE(std::any_of(result.journal.begin(), result.journal.end(), [&](const auto& edit) {
    return fixture->graph.Get(edit.original_block)->address == 0x10c && edit.value == 12;
  }));
  EXPECT_EQ(ir::ValidateSsaWithSources(*result.provisional, fixture->sources, budget),
            ir::SsaDecline::none);
  auto forged = *proved.facts;
  const auto index = static_cast<std::size_t>(join - proved.facts->values.begin());
  forged.values[index].value = 13;
  EXPECT_EQ(ir::ValidateSsaConstantFacts(fixture->graph, *reachable.facts, forged, fixture->sources,
                                         budget),
            ir::SsaDecline::invalid_graph);
  EXPECT_EQ(recovery::ProposeSsaConstantFold(fixture->graph, *reachable.facts, forged,
                                             fixture->sources, budget)
                .reason,
            recovery::SsaConstantFoldRefusal::stale_proof);
  EXPECT_EQ(recovery::ProposeSsaConstantFold(*result.provisional, *reachable.facts, *proved.facts,
                                             fixture->sources, budget)
                .reason,
            recovery::SsaConstantFoldRefusal::stale_proof);
}

TEST(SsaConstants, SccpSelectsAProvedEdgeAndRejectsForgedCertificates) {
  auto budget = Plenty();
  auto fixture = Join(budget, 22, true, true);
  ASSERT_TRUE(fixture);
  auto reachable = ProveSsaReachability(fixture->graph, fixture->sources,
                                        ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts);
  auto all_edges = ProveSsaConstants(fixture->graph, *reachable.facts, fixture->sources, budget);
  ASSERT_TRUE(all_edges.facts);
  const auto is_join_phi = [&](const ir::SsaConstantValue& fact) {
    const auto* block = fixture->graph.Get(fact.block);
    return block && block->address == 0x10c && fact.kind == ir::SsaValueKind::phi &&
           block->phis[fact.index].storage == 8;
  };

  EXPECT_FALSE(
      std::any_of(all_edges.facts->values.begin(), all_edges.facts->values.end(), is_join_phi));
  auto proved =
      ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population, fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  const auto join = std::find_if(proved.facts->constants.values.begin(),
                                 proved.facts->constants.values.end(), is_join_phi);
  ASSERT_NE(join, proved.facts->constants.values.end());
  EXPECT_EQ(join->value, 22U);
  EXPECT_EQ(ir::ValidateSsaSccpFacts(fixture->graph, *proved.facts, fixture->sources, budget),
            ir::SsaDecline::none);
  auto folded =
      recovery::ProposeSsaSccpFold(fixture->graph, *proved.facts, fixture->sources, budget);
  ASSERT_TRUE(folded.provisional) << static_cast<int>(folded.reason);
  EXPECT_FALSE(std::any_of(folded.journal.begin(), folded.journal.end(),
                           [](const auto& edit) { return edit.original.op == ir::Op::read; }));
  auto read_folded =
      recovery::ProposeSsaSccpReadFold(fixture->graph, *proved.facts, fixture->sources, budget);
  ASSERT_TRUE(read_folded.provisional) << static_cast<int>(read_folded.reason);
  EXPECT_TRUE(
      std::any_of(read_folded.journal.begin(), read_folded.journal.end(), [](const auto& edit) {
        return edit.original.op == ir::Op::read && edit.removed_read.has_value();
      }));
  EXPECT_TRUE(std::any_of(folded.journal.begin(), folded.journal.end(), [&](const auto& edit) {
    return fixture->graph.Get(edit.original_block)->address == 0x10c && edit.value == 23;
  }));
  auto wrong_value = *proved.facts;
  auto wrong_join = std::find_if(wrong_value.constants.values.begin(),
                                 wrong_value.constants.values.end(), is_join_phi);
  wrong_join->value = 23;
  EXPECT_EQ(ir::ValidateSsaSccpFacts(fixture->graph, wrong_value, fixture->sources, budget),
            ir::SsaDecline::invalid_graph);
  auto missing_edge = *proved.facts;
  const auto entry = fixture->graph.entries().front();
  std::fill(missing_edge.edges[entry.slot].begin(), missing_edge.edges[entry.slot].end(), 0);
  EXPECT_EQ(ir::ValidateSsaSccpFacts(fixture->graph, missing_edge, fixture->sources, budget),
            ir::SsaDecline::invalid_graph);
  EXPECT_EQ(
      recovery::ProposeSsaSccpFold(fixture->graph, missing_edge, fixture->sources, budget).reason,
      recovery::SsaConstantFoldRefusal::stale_proof);
  EXPECT_EQ(
      recovery::ProposeSsaSccpFold(*folded.provisional, *proved.facts, fixture->sources, budget)
          .reason,
      recovery::SsaConstantFoldRefusal::stale_proof);
}

TEST(SsaConstants, RetiresOnlyAProvedConditionalEdge) {
  auto budget = Plenty();
  auto fixture = Join(budget, 22, true, true);
  ASSERT_TRUE(fixture);
  auto proved =
      ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population, fixture->sources, budget);
  ASSERT_TRUE(proved.facts);
  auto retired =
      recovery::ProposeSsaBranchRetirement(fixture->graph, *proved.facts, fixture->sources, budget);
  ASSERT_TRUE(retired.provisional) << static_cast<int>(retired.reason);
  ASSERT_EQ(retired.journal.size(), 1U);
  EXPECT_EQ(retired.journal[0].removed.when, false);
  const auto* entry = retired.provisional->Get(retired.provisional->entries().front());
  ASSERT_NE(entry, nullptr);
  ASSERT_EQ(entry->edges.size(), 1U);
  EXPECT_FALSE(entry->edges[0].condition);
  ASSERT_EQ(entry->control_rewrites.size(), 1U);
  EXPECT_EQ(entry->control_rewrites[0].rule, ir::RewriteRule::folded_condition);
  EXPECT_EQ(ir::ValidateSsaWithSources(*retired.provisional, fixture->sources, budget),
            ir::SsaDecline::none);
  auto after = ProveSsaReachability(*retired.provisional, fixture->sources,
                                    ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(after.facts) << static_cast<int>(after.reason);
  EXPECT_EQ(std::count(after.facts->reachable.begin(), after.facts->reachable.end(), 1), 3);
  auto cleaned = recovery::ProposeUnreachableBlocks(*retired.provisional, *after.facts,
                                                    fixture->sources, budget);
  ASSERT_TRUE(cleaned.provisional) << static_cast<int>(cleaned.reason);
  EXPECT_EQ(std::count_if(cleaned.journal.begin(), cleaned.journal.end(),
                          [](const auto& edit) {
                            return edit.kind == recovery::SsaDceEditKind::removed_block;
                          }),
            1);
  EXPECT_EQ(ir::ValidateSsaWithSources(*cleaned.provisional, fixture->sources, budget),
            ir::SsaDecline::none);
  auto wrong_rewrite = retired.provisional->Clone(budget);
  ASSERT_TRUE(wrong_rewrite);
  const auto wrong_entry = wrong_rewrite->entries().front();
  ASSERT_TRUE(wrong_rewrite->Update(
      wrong_entry, [](auto& block) { block.control_rewrites.front().condition_value = false; }));
  EXPECT_EQ(ir::ValidateSsa(*wrong_rewrite, budget), ir::SsaDecline::invalid_graph);
  auto wrong_target = retired.provisional->Clone(budget);
  ASSERT_TRUE(wrong_target);
  const auto wrong_target_entry = wrong_target->entries().front();
  ASSERT_TRUE(wrong_target->Update(wrong_target_entry,
                                   [](auto& block) { block.edges.front().address += 4; }));
  EXPECT_EQ(ir::ValidateSsa(*wrong_target, budget), ir::SsaDecline::invalid_graph);
  auto open = *proved.facts;
  open.entry_scope = ir::SsaEntryScope::discovered_only;
  EXPECT_EQ(
      recovery::ProposeSsaBranchRetirement(fixture->graph, open, fixture->sources, budget).reason,
      recovery::SsaBranchRetirementRefusal::stale_proof);
  auto unknown = Join(budget, 22);
  ASSERT_TRUE(unknown);
  auto unknown_proved =
      ProveSsaSccp(unknown->graph, ir::SsaEntryScope::closed_population, unknown->sources, budget);
  ASSERT_TRUE(unknown_proved.facts);
  auto unchanged = recovery::ProposeSsaBranchRetirement(unknown->graph, *unknown_proved.facts,
                                                        unknown->sources, budget);
  EXPECT_EQ(unchanged.reason, recovery::SsaBranchRetirementRefusal::none);
  EXPECT_FALSE(unchanged.provisional);
  EXPECT_TRUE(unchanged.journal.empty());
}

// The arm the branch never takes reads the x0 its one predecessor set, and
// that read is recorded as a copy of it. Retiring the edge leaves the phi with
// nothing incoming, so the copy record has nothing left to name.
TEST(SsaConstants, RetiringAnEdgeDropsTheCopyItsTargetReadAcrossIt) {
  auto budget = Plenty();
  auto fixture = DeadArm(budget, true);
  ASSERT_TRUE(fixture);
  auto copied = WithCopies(*fixture, budget);
  ASSERT_TRUE(copied);
  const auto* arm = At(*copied, 0x104);
  ASSERT_NE(arm, nullptr);
  ASSERT_EQ(arm->reads.size(), 1U);
  ASSERT_TRUE(arm->reads[0].predecessor_copy);
  auto proved =
      ProveSsaSccp(*copied, ir::SsaEntryScope::closed_population, fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  auto retired =
      recovery::ProposeSsaBranchRetirement(*copied, *proved.facts, fixture->sources, budget);
  ASSERT_EQ(retired.reason, recovery::SsaBranchRetirementRefusal::none);
  ASSERT_TRUE(retired.provisional);
  EXPECT_EQ(ir::ValidateSsaWithSources(*retired.provisional, fixture->sources, budget),
            ir::SsaDecline::none);
  arm = At(*retired.provisional, 0x104);
  ASSERT_NE(arm, nullptr);
  EXPECT_FALSE(arm->reads[0].predecessor_copy);
  EXPECT_FALSE(arm->reads[0].copy_closed_entries);
}

// Both blocks of the arm go unreachable together, and the second one's read
// is recorded as a copy of what the first computed. That record must go with
// them, or erasing the first leaves the second naming a block that is gone.
TEST(SsaConstants, UnreachableBlocksGoEvenWhenOneCopiesFromAnother) {
  auto budget = Plenty();
  auto fixture = DeadArm(budget, false);
  ASSERT_TRUE(fixture);
  auto copied = WithCopies(*fixture, budget);
  ASSERT_TRUE(copied);
  const auto* second = At(*copied, 0x10c);
  ASSERT_NE(second, nullptr);
  ASSERT_EQ(second->reads.size(), 1U);
  ASSERT_TRUE(second->reads[0].predecessor_copy);
  EXPECT_EQ(copied->Get(second->reads[0].predecessor_copy->block)->address, 0x104U);
  auto proved =
      ProveSsaSccp(*copied, ir::SsaEntryScope::closed_population, fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  auto retired =
      recovery::ProposeSsaBranchRetirement(*copied, *proved.facts, fixture->sources, budget);
  ASSERT_TRUE(retired.provisional) << static_cast<int>(retired.reason);
  auto after = ProveSsaReachability(*retired.provisional, fixture->sources,
                                    ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(after.facts) << static_cast<int>(after.reason);
  EXPECT_EQ(std::count(after.facts->reachable.begin(), after.facts->reachable.end(), 1), 3);
  auto cleaned = recovery::ProposeUnreachableBlocks(*retired.provisional, *after.facts,
                                                    fixture->sources, budget);
  ASSERT_EQ(cleaned.reason, recovery::SsaDceRefusal::none);
  ASSERT_TRUE(cleaned.provisional);
  EXPECT_EQ(std::count_if(cleaned.journal.begin(), cleaned.journal.end(),
                          [](const auto& edit) {
                            return edit.kind == recovery::SsaDceEditKind::removed_block;
                          }),
            2);
  EXPECT_FALSE(At(*cleaned.provisional, 0x104));
  EXPECT_FALSE(At(*cleaned.provisional, 0x10c));
  EXPECT_EQ(ir::ValidateSsaWithSources(*cleaned.provisional, fixture->sources, budget),
            ir::SsaDecline::none);
}

// A plain jump through a declared slot has the one successor the slot's
// declared value names, so the block is complete and the population closed.
// That successor rests on the declaration, which the edge must carry, and on
// the recorded read: without it the loaded target is unknown again.
TEST(SsaConstants, AJumpThroughADeclaredSlotReachesWhatTheSlotHolds) {
  auto budget = Plenty();
  auto fixture = SlotJump(budget, true);
  ASSERT_TRUE(fixture);
  const auto* jump = At(fixture->graph, 0x100);
  ASSERT_NE(jump, nullptr);
  ASSERT_EQ(jump->path_reads.size(), 1U);
  ASSERT_EQ(jump->edges.size(), 1U);
  EXPECT_EQ(jump->edges[0].address, 0x108U);
  EXPECT_TRUE(jump->edges[0].assumptions.constant_image);
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(*jump, budget), ir::SsaDecline::none);
  auto reached = ProveSsaReachability(fixture->graph, fixture->sources,
                                      ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reached.facts) << static_cast<int>(reached.reason);
  EXPECT_EQ(std::count(reached.facts->reachable.begin(), reached.facts->reachable.end(), 1), 2);
  auto bare = *jump;
  bare.edges[0].assumptions.constant_image = false;
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(bare, budget), ir::SsaDecline::invalid_graph);
  auto unread = *jump;
  unread.path_reads.clear();
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(unread, budget), ir::SsaDecline::invalid_graph);
  auto undeclared = SlotJump(budget, false);
  ASSERT_TRUE(undeclared);
  jump = At(undeclared->graph, 0x100);
  ASSERT_NE(jump, nullptr);
  EXPECT_TRUE(jump->path_reads.empty());
  EXPECT_EQ(ProveSsaReachability(undeclared->graph, undeclared->sources,
                                 ir::SsaEntryScope::closed_population, budget)
                .reason,
            SsaReachabilityRefusal::incomplete_successors);
}

TEST(SsaConstants, BranchRetirementBudgetCutsPublishNoPartialGraphOrJournal) {
  auto setup = Plenty();
  auto fixture = Join(setup, 22, true, true);
  ASSERT_TRUE(fixture);
  auto proved =
      ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population, fixture->sources, setup);
  ASSERT_TRUE(proved.facts);
  auto full = Plenty();
  auto retired =
      recovery::ProposeSsaBranchRetirement(fixture->graph, *proved.facts, fixture->sources, full);
  ASSERT_TRUE(retired.provisional);
  const auto used = full.used();
  ASSERT_LT(used.work, 30000U);
  for (std::uint64_t cut = 0; cut < used.work; ++cut) {
    Budget limited({cut, used.bytes});
    const auto result = recovery::ProposeSsaBranchRetirement(fixture->graph, *proved.facts,
                                                             fixture->sources, limited);
    EXPECT_EQ(result.reason, recovery::SsaBranchRetirementRefusal::resource_limit) << cut;
    EXPECT_FALSE(result.provisional) << cut;
    EXPECT_TRUE(result.journal.empty()) << cut;
    EXPECT_TRUE(result.phi_journal.empty()) << cut;
  }

  for (std::uint64_t cut = 0; cut < used.bytes; ++cut) {
    Budget limited({used.work, cut});
    const auto result = recovery::ProposeSsaBranchRetirement(fixture->graph, *proved.facts,
                                                             fixture->sources, limited);
    EXPECT_EQ(result.reason, recovery::SsaBranchRetirementRefusal::resource_limit) << cut;
    EXPECT_FALSE(result.provisional) << cut;
    EXPECT_TRUE(result.journal.empty()) << cut;
    EXPECT_TRUE(result.phi_journal.empty()) << cut;
  }
}

TEST(SsaConstants, SccpKeepsBothUnknownConditionalEdges) {
  auto budget = Plenty();
  auto fixture = Join(budget, 22);
  ASSERT_TRUE(fixture);
  auto proved =
      ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population, fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  const auto entry = fixture->graph.entries().front();
  ASSERT_EQ(proved.facts->edges[entry.slot].size(), 2U);
  EXPECT_EQ(proved.facts->edges[entry.slot][0], 1);
  EXPECT_EQ(proved.facts->edges[entry.slot][1], 1);
  EXPECT_FALSE(std::any_of(proved.facts->constants.values.begin(),
                           proved.facts->constants.values.end(), [&](const auto& fact) {
                             const auto* block = fixture->graph.Get(fact.block);
                             return block && block->address == 0x10c &&
                                    fact.kind == ir::SsaValueKind::phi &&
                                    block->phis[fact.index].storage == 8;
                           }));
}

TEST(SsaConstants, SccpSelectsTheFalseArmAndRespectsBudgetCuts) {
  auto setup = Plenty();
  auto fixture = Join(setup, 22, true, false);
  ASSERT_TRUE(fixture);
  auto full = Plenty();
  auto proved =
      ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population, fixture->sources, full);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  const auto join = std::find_if(proved.facts->constants.values.begin(),
                                 proved.facts->constants.values.end(), [&](const auto& fact) {
                                   const auto* block = fixture->graph.Get(fact.block);
                                   return block && block->address == 0x10c &&
                                          fact.kind == ir::SsaValueKind::phi &&
                                          block->phis[fact.index].storage == 8;
                                 });
  ASSERT_NE(join, proved.facts->constants.values.end());
  EXPECT_EQ(join->value, 11U);
  const auto used = full.used();
  ASSERT_LT(used.work, 30000U);
  for (std::uint64_t cut = 0; cut < used.work; ++cut) {
    Budget limited({cut, used.bytes});
    const auto result = ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population,
                                     fixture->sources, limited);
    EXPECT_EQ(result.reason, SsaSccpRefusal::resource_limit) << cut;
    EXPECT_FALSE(result.facts) << cut;
  }

  for (std::uint64_t cut = 0; cut < used.bytes; ++cut) {
    Budget limited({used.work, cut});
    const auto result = ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population,
                                     fixture->sources, limited);
    EXPECT_EQ(result.reason, SsaSccpRefusal::resource_limit) << cut;
    EXPECT_FALSE(result.facts) << cut;
  }

  auto full_check = Plenty();
  ASSERT_EQ(ir::ValidateSsaSccpFacts(fixture->graph, *proved.facts, fixture->sources, full_check),
            ir::SsaDecline::none);
  const auto check_used = full_check.used();
  for (std::uint64_t cut = 0; cut < check_used.work; ++cut) {
    Budget limited({cut, check_used.bytes});
    EXPECT_EQ(ir::ValidateSsaSccpFacts(fixture->graph, *proved.facts, fixture->sources, limited),
              ir::SsaDecline::resource_limit)
        << cut;
  }

  for (std::uint64_t cut = 0; cut < check_used.bytes; ++cut) {
    Budget limited({check_used.work, cut});
    EXPECT_EQ(ir::ValidateSsaSccpFacts(fixture->graph, *proved.facts, fixture->sources, limited),
              ir::SsaDecline::resource_limit)
        << cut;
  }

  auto full_fold = Plenty();
  auto folded =
      recovery::ProposeSsaSccpFold(fixture->graph, *proved.facts, fixture->sources, full_fold);
  ASSERT_TRUE(folded.provisional) << static_cast<int>(folded.reason);
  const auto fold_used = full_fold.used();
  for (std::uint64_t cut = 0; cut < fold_used.work; ++cut) {
    Budget limited({cut, fold_used.bytes});
    const auto result =
        recovery::ProposeSsaSccpFold(fixture->graph, *proved.facts, fixture->sources, limited);
    EXPECT_EQ(result.reason, recovery::SsaConstantFoldRefusal::resource_limit) << cut;
    EXPECT_FALSE(result.provisional) << cut;
    EXPECT_TRUE(result.journal.empty()) << cut;
  }

  for (std::uint64_t cut = 0; cut < fold_used.bytes; ++cut) {
    Budget limited({fold_used.work, cut});
    const auto result =
        recovery::ProposeSsaSccpFold(fixture->graph, *proved.facts, fixture->sources, limited);
    EXPECT_EQ(result.reason, recovery::SsaConstantFoldRefusal::resource_limit) << cut;
    EXPECT_FALSE(result.provisional) << cut;
    EXPECT_TRUE(result.journal.empty()) << cut;
  }

  auto read_budget = Plenty();
  auto read_folded = recovery::ProposeSsaSccpReadFold(fixture->graph, *proved.facts,
                                                      fixture->sources, read_budget);
  ASSERT_TRUE(read_folded.provisional) << static_cast<int>(read_folded.reason);
  ASSERT_FALSE(read_folded.journal.empty());
  const auto read_used = read_budget.used();
  for (std::uint64_t cut = 0; cut < read_used.work; ++cut) {
    Budget limited({cut, read_used.bytes});
    const auto result =
        recovery::ProposeSsaSccpReadFold(fixture->graph, *proved.facts, fixture->sources, limited);
    EXPECT_EQ(result.reason, recovery::SsaConstantFoldRefusal::resource_limit) << cut;
    EXPECT_FALSE(result.provisional) << cut;
    EXPECT_TRUE(result.journal.empty()) << cut;
  }

  for (std::uint64_t cut = 0; cut < read_used.bytes; ++cut) {
    Budget limited({read_used.work, cut});
    const auto result =
        recovery::ProposeSsaSccpReadFold(fixture->graph, *proved.facts, fixture->sources, limited);
    EXPECT_EQ(result.reason, recovery::SsaConstantFoldRefusal::resource_limit) << cut;
    EXPECT_FALSE(result.provisional) << cut;
    EXPECT_TRUE(result.journal.empty()) << cut;
  }
}

TEST(SsaConstants, LeavesADisagreeingJoinUnknownAndRefusesOpenScope) {
  auto budget = Plenty();
  auto fixture = Join(budget, 22);
  ASSERT_TRUE(fixture);
  auto reachable = ProveSsaReachability(fixture->graph, fixture->sources,
                                        ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts);
  auto proved = ProveSsaConstants(fixture->graph, *reachable.facts, fixture->sources, budget);
  ASSERT_TRUE(proved.facts);
  EXPECT_FALSE(
      std::any_of(proved.facts->values.begin(), proved.facts->values.end(), [&](const auto& fact) {
        const auto* block = fixture->graph.Get(fact.block);
        return block && block->address == 0x10c && fact.kind == ir::SsaValueKind::phi &&
               block->phis[fact.index].storage == 8;
      }));
  auto open = *reachable.facts;
  open.entry_scope = ir::SsaEntryScope::discovered_only;
  EXPECT_EQ(ProveSsaConstants(fixture->graph, open, fixture->sources, budget).reason,
            SsaConstantRefusal::stale_reachability);
  auto no_return = Join(budget, 11, false);
  ASSERT_TRUE(no_return);
  EXPECT_FALSE(ProveSsaReachability(no_return->graph, no_return->sources,
                                    ir::SsaEntryScope::closed_population, budget)
                   .facts);
}

TEST(SsaConstants, AcceptsValidReadsInEitherRecordOrder) {
  auto budget = Plenty();
  auto fixture = Join(budget);
  ASSERT_TRUE(fixture);
  std::optional<ir::SsaHandle> join;
  for (std::size_t slot = 0; slot < fixture->graph.slots(); ++slot) {
    const auto handle = fixture->graph.Handle(slot);
    if (handle && fixture->graph.Get(*handle)->address == 0x10c) join = handle;
  }

  ASSERT_TRUE(join);
  ASSERT_TRUE(fixture->graph.Update(
      *join, [](auto& block) { std::reverse(block.reads.begin(), block.reads.end()); }));
  ASSERT_EQ(ir::ValidateSsa(fixture->graph, budget), ir::SsaDecline::none);
  auto reachable = ProveSsaReachability(fixture->graph, fixture->sources,
                                        ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts);
  auto proved = ProveSsaConstants(fixture->graph, *reachable.facts, fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  ASSERT_EQ(ir::ValidateSsaConstantFacts(fixture->graph, *reachable.facts, *proved.facts,
                                         fixture->sources, budget),
            ir::SsaDecline::none);
  auto folded = recovery::ProposeSsaConstantFold(fixture->graph, *reachable.facts, *proved.facts,
                                                 fixture->sources, budget);
  ASSERT_TRUE(folded.provisional) << static_cast<int>(folded.reason);
  EXPECT_EQ(folded.journal.size(), 2U);
}

TEST(SsaConstants, ProvesAConstantCarriedAroundAReachableLoop) {
  auto budget = Plenty();
  auto fixture = Loop(budget);
  ASSERT_TRUE(fixture);
  auto reachable = ProveSsaReachability(fixture->graph, fixture->sources,
                                        ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts);
  auto proved = ProveSsaConstants(fixture->graph, *reachable.facts, fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  const auto carried =
      std::find_if(proved.facts->values.begin(), proved.facts->values.end(), [&](const auto& fact) {
        const auto* block = fixture->graph.Get(fact.block);
        return block && block->address == 0x104 && fact.kind == ir::SsaValueKind::phi &&
               block->phis[fact.index].storage == 8;
      });
  ASSERT_NE(carried, proved.facts->values.end());
  EXPECT_EQ(carried->value, 11U);
  auto folded = recovery::ProposeSsaConstantFold(fixture->graph, *reachable.facts, *proved.facts,
                                                 fixture->sources, budget);
  ASSERT_TRUE(folded.provisional) << static_cast<int>(folded.reason);
  EXPECT_TRUE(std::any_of(folded.journal.begin(), folded.journal.end(), [&](const auto& edit) {
    return fixture->graph.Get(edit.original_block)->address == 0x10c && edit.value == 12;
  }));
}

TEST(SsaConstants, ProofCacheInvalidatesOnGraphAndSourceRevisions) {
  auto budget = Plenty();
  auto fixture = Join(budget, 22, true, true);
  ASSERT_TRUE(fixture);
  SsaProofCache cache(fixture->sources, ir::SsaEntryScope::closed_population);
  ASSERT_TRUE(cache.Reachability(fixture->graph, budget).facts);
  ASSERT_TRUE(cache.Reachability(fixture->graph, budget).facts);
  ASSERT_TRUE(cache.Sccp(fixture->graph, budget).facts);
  ASSERT_TRUE(cache.Sccp(fixture->graph, budget).facts);
  EXPECT_EQ(cache.counts().reachability_computations, 1U);
  EXPECT_EQ(cache.counts().sccp_computations, 1U);

  const auto& proved = cache.Sccp(fixture->graph, budget);
  auto retired =
      recovery::ProposeSsaBranchRetirement(fixture->graph, *proved.facts, fixture->sources, budget);
  ASSERT_TRUE(retired.provisional);
  ASSERT_TRUE(cache.Reachability(*retired.provisional, budget).facts);
  ASSERT_TRUE(cache.Sccp(*retired.provisional, budget).facts);
  EXPECT_EQ(cache.counts().reachability_computations, 2U);
  EXPECT_EQ(cache.counts().sccp_computations, 2U);
  EXPECT_EQ(cache.counts().invalidations, 1U);

  auto equivalent = retired.provisional->Clone(budget);
  ASSERT_TRUE(equivalent);
  ASSERT_EQ(equivalent->revision(), retired.provisional->revision());
  ASSERT_NE(equivalent->arena(), retired.provisional->arena());
  ASSERT_TRUE(cache.Reachability(*equivalent, budget).facts);
  EXPECT_EQ(cache.counts().reachability_computations, 3U);
  EXPECT_EQ(cache.counts().invalidations, 2U);

  cache.ResetSources(fixture->sources);
  ASSERT_TRUE(cache.Reachability(*equivalent, budget).facts);
  EXPECT_EQ(cache.counts().reachability_computations, 4U);
  EXPECT_EQ(cache.counts().invalidations, 3U);

  auto changed_sources = fixture->sources;
  changed_sources[0] = ir::Group(
      0x100, {99, 2, 3, 4},
      std::vector<ir::Node>(fixture->sources[0].nodes().begin(), fixture->sources[0].nodes().end()),
      std::vector<ir::Write>(fixture->sources[0].writes().begin(),
                             fixture->sources[0].writes().end()),
      fixture->sources[0].memory_model(), fixture->sources[0].transfer());
  cache.ResetSources(changed_sources);
  const auto& declined = cache.Reachability(*equivalent, budget);
  EXPECT_FALSE(declined.facts);
  EXPECT_EQ(declined.reason, SsaReachabilityRefusal::invalid_graph);
  EXPECT_EQ(cache.counts().reachability_computations, 5U);
  cache.ResetSources(fixture->sources);
  EXPECT_TRUE(cache.Reachability(*equivalent, budget).facts);
  EXPECT_EQ(cache.counts().reachability_computations, 6U);

  auto discovered_sources = fixture->sources;
  discovered_sources.emplace_back(0x200, std::vector<std::uint8_t>{17, 18, 19, 20},
                                  std::vector<ir::Node>{{ir::Op::constant, 64, {}, 1}},
                                  std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                                  std::nullopt);
  cache.ResetSources(discovered_sources);
  EXPECT_TRUE(cache.Reachability(*equivalent, budget).facts);
  EXPECT_EQ(cache.counts().reachability_computations, 7U);
}

TEST(SsaConstants, ProofCacheInvalidatesAfterAnEffectEdit) {
  auto budget = Plenty();
  auto fixture = Join(budget, 22, true, true);
  ASSERT_TRUE(fixture);
  SsaProofCache cache(fixture->sources, ir::SsaEntryScope::closed_population);
  ASSERT_TRUE(cache.Reachability(fixture->graph, budget).facts);
  const auto& proved = cache.Sccp(fixture->graph, budget);
  ASSERT_TRUE(proved.facts);
  auto edited =
      recovery::ProposeSsaSccpReadFold(fixture->graph, *proved.facts, fixture->sources, budget);
  ASSERT_TRUE(edited.provisional);
  ASSERT_TRUE(cache.Reachability(*edited.provisional, budget).facts);
  EXPECT_EQ(cache.counts().reachability_computations, 2U);
  EXPECT_EQ(cache.counts().invalidations, 1U);
  EXPECT_EQ(ir::ValidateSsaWithSources(*edited.provisional, fixture->sources, budget),
            ir::SsaDecline::none);
}

TEST(SsaConstants, MovedGraphsCannotShareACacheIdentityWithTheirSource) {
  static_assert(!std::is_move_constructible_v<SsaProofCache>);
  auto budget = Plenty();
  auto fixture = Join(budget);
  ASSERT_TRUE(fixture);
  SsaProofCache cache(fixture->sources, ir::SsaEntryScope::closed_population);
  ASSERT_TRUE(cache.Reachability(fixture->graph, budget).facts);
  const auto old_arena = fixture->graph.arena();
  ir::SsaGraph moved(std::move(fixture->graph));
  EXPECT_EQ(moved.arena(), old_arena);
  EXPECT_NE(fixture->graph.arena(), moved.arena());
  EXPECT_TRUE(cache.Reachability(moved, budget).facts);
  const auto& from_source = cache.Reachability(fixture->graph, budget);
  EXPECT_FALSE(from_source.facts);
  EXPECT_EQ(from_source.reason, SsaReachabilityRefusal::invalid_graph);

  ir::SsaGraph assigned;
  assigned = std::move(moved);
  EXPECT_EQ(assigned.arena(), old_arena);
  EXPECT_NE(moved.arena(), assigned.arena());
  EXPECT_TRUE(cache.Reachability(assigned, budget).facts);
  const auto& from_second_source = cache.Reachability(moved, budget);
  EXPECT_FALSE(from_second_source.facts);
  EXPECT_EQ(from_second_source.reason, SsaReachabilityRefusal::invalid_graph);
}

TEST(SsaConstants, ProofCacheDoesNotReuseResourceLimitAsAProof) {
  auto setup = Plenty();
  auto fixture = Join(setup);
  ASSERT_TRUE(fixture);
  SsaProofCache cache(fixture->sources, ir::SsaEntryScope::closed_population);
  Budget exhausted({0, 0});
  const auto& limited = cache.Reachability(fixture->graph, exhausted);
  EXPECT_FALSE(limited.facts);
  EXPECT_EQ(limited.reason, SsaReachabilityRefusal::resource_limit);
  auto budget = Plenty();
  EXPECT_TRUE(cache.Reachability(fixture->graph, budget).facts);
  EXPECT_EQ(cache.counts().reachability_computations, 2U);
  Budget hit_limit({0, 0});
  const auto& hit = cache.Reachability(fixture->graph, hit_limit);
  EXPECT_FALSE(hit.facts);
  EXPECT_EQ(hit.reason, SsaReachabilityRefusal::resource_limit);
  EXPECT_EQ(cache.counts().reachability_computations, 2U);
}

TEST(SsaConstants, BudgetCutsPublishNoPartialProofOrJournal) {
  auto setup = Plenty();
  auto fixture = Join(setup);
  ASSERT_TRUE(fixture);
  auto reachable = ProveSsaReachability(fixture->graph, fixture->sources,
                                        ir::SsaEntryScope::closed_population, setup);
  ASSERT_TRUE(reachable.facts);
  auto full_proof = Plenty();
  auto proved = ProveSsaConstants(fixture->graph, *reachable.facts, fixture->sources, full_proof);
  ASSERT_TRUE(proved.facts);
  const auto proof_used = full_proof.used();
  ASSERT_LT(proof_used.work, 20000U);
  for (std::uint64_t cut = 0; cut < proof_used.work; ++cut) {
    Budget limited({cut, proof_used.bytes});
    const auto result =
        ProveSsaConstants(fixture->graph, *reachable.facts, fixture->sources, limited);
    EXPECT_EQ(result.reason, SsaConstantRefusal::resource_limit) << cut;
    EXPECT_FALSE(result.facts) << cut;
  }

  for (const auto cut : {std::uint64_t{0}, proof_used.bytes / 2, proof_used.bytes - 1}) {
    Budget limited({proof_used.work, cut});
    const auto result =
        ProveSsaConstants(fixture->graph, *reachable.facts, fixture->sources, limited);
    EXPECT_EQ(result.reason, SsaConstantRefusal::resource_limit) << cut;
    EXPECT_FALSE(result.facts) << cut;
  }

  auto full_check = Plenty();
  ASSERT_EQ(ir::ValidateSsaConstantFacts(fixture->graph, *reachable.facts, *proved.facts,
                                         fixture->sources, full_check),
            ir::SsaDecline::none);
  const auto check_used = full_check.used();
  ASSERT_LT(check_used.work, 20000U);
  for (std::uint64_t cut = 0; cut < check_used.work; ++cut) {
    Budget limited({cut, check_used.bytes});
    EXPECT_EQ(ir::ValidateSsaConstantFacts(fixture->graph, *reachable.facts, *proved.facts,
                                           fixture->sources, limited),
              ir::SsaDecline::resource_limit)
        << cut;
  }

  for (const auto cut : {std::uint64_t{0}, check_used.bytes / 2, check_used.bytes - 1}) {
    Budget limited({check_used.work, cut});
    EXPECT_EQ(ir::ValidateSsaConstantFacts(fixture->graph, *reachable.facts, *proved.facts,
                                           fixture->sources, limited),
              ir::SsaDecline::resource_limit)
        << cut;
  }

  auto full_recovery = Plenty();
  ASSERT_TRUE(recovery::ProposeSsaConstantFold(fixture->graph, *reachable.facts, *proved.facts,
                                               fixture->sources, full_recovery)
                  .provisional);
  const auto recovery_used = full_recovery.used();
  ASSERT_LT(recovery_used.work, 20000U);
  for (std::uint64_t cut = 0; cut < recovery_used.work; ++cut) {
    Budget limited({cut, recovery_used.bytes});
    const auto result = recovery::ProposeSsaConstantFold(fixture->graph, *reachable.facts,
                                                         *proved.facts, fixture->sources, limited);
    EXPECT_EQ(result.reason, recovery::SsaConstantFoldRefusal::resource_limit) << cut;
    EXPECT_FALSE(result.provisional) << cut;
    EXPECT_TRUE(result.journal.empty()) << cut;
  }

  for (const auto cut : {std::uint64_t{0}, recovery_used.bytes / 2, recovery_used.bytes - 1}) {
    Budget limited({recovery_used.work, cut});
    const auto result = recovery::ProposeSsaConstantFold(fixture->graph, *reachable.facts,
                                                         *proved.facts, fixture->sources, limited);
    EXPECT_EQ(result.reason, recovery::SsaConstantFoldRefusal::resource_limit) << cut;
    EXPECT_FALSE(result.provisional) << cut;
    EXPECT_TRUE(result.journal.empty()) << cut;
  }
}

std::optional<Fixture> PhiGuardedIndirect(Budget& budget) {
  std::vector<ir::Group> sources;
  sources.emplace_back(
      0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
      std::vector<ir::Node>{{ir::Op::constant, 1, {}, 1}, {ir::Op::image_address, 64, {}, 0x104}},
      std::vector<ir::Write>{{32, 0}}, ir::MemoryModel::unspecified,
      ir::Transfer{ir::TransferKind::jump, 1, {}, {}, {}});
  sources.emplace_back(0x104, std::vector<std::uint8_t>{5, 6, 7, 8},
                       std::vector<ir::Node>{{ir::Op::read, 1, {}, 0, 32},
                                             {ir::Op::image_address, 64, {}, 0x10c},
                                             {ir::Op::image_address, 64, {}, 0x108}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::conditional, 1, 0, 2, {}});
  sources.emplace_back(0x108, std::vector<std::uint8_t>{9, 10, 11, 12},
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 16}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}});
  sources.emplace_back(0x10c, std::vector<std::uint8_t>{13, 14, 15, 16},
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 30}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}});
  std::vector<SourceRecord> records;
  for (const auto& source : sources)
    records.push_back(
        {source.source_address(), {source.bytes().begin(), source.bytes().end()}, source});
  constexpr std::uint64_t entry = 0x100;
  auto cfg = BuildCfg(records, std::span(&entry, 1), budget);
  if (!cfg.cfg) return {};
  Regions regions(std::move(*cfg.cfg), {});
  auto recovered = Unflatten(regions, {}, budget, {{}, true});
  if (!recovered.unflattening) return {};
  const std::vector<std::optional<ir::RecoveredPath>> paths;
  auto built = BuildSsa(regions, *recovered.unflattening, paths, budget);
  if (!built.graph) return {};
  return Fixture{std::move(*built.graph), std::move(sources)};
}

TEST(SsaConstants, SccpFactsCannotClaimAnIncompleteBlockRuns) {
  auto budget = Plenty();
  auto fixture = GuardedIndirect(budget, true);
  ASSERT_TRUE(fixture);
  auto proved =
      ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population, fixture->sources, budget);
  ASSERT_TRUE(proved.facts);
  auto forged = *proved.facts;
  const auto entry = fixture->graph.entries().front();
  for (auto& e : forged.edges[entry.slot]) e = 1;
  for (std::size_t slot = 0; slot < fixture->graph.slots(); ++slot)
    if (fixture->graph.Handle(slot)) forged.executable[slot] = 1;
  EXPECT_NE(ir::ValidateSsaSccpFacts(fixture->graph, forged, fixture->sources, budget),
            ir::SsaDecline::none);
  auto forged2 = *proved.facts;
  for (std::size_t slot = 0; slot < fixture->graph.slots(); ++slot)
    if (fixture->graph.Handle(slot)) forged2.executable[slot] = 1;
  EXPECT_NE(ir::ValidateSsaSccpFacts(fixture->graph, forged2, fixture->sources, budget),
            ir::SsaDecline::none);
  auto forged3 = *proved.facts;
  forged3.executable[entry.slot] = 0;
  EXPECT_NE(ir::ValidateSsaSccpFacts(fixture->graph, forged3, fixture->sources, budget),
            ir::SsaDecline::none);
}

TEST(SsaConstants, ARetiredGuardTakesItsUnresolvedArmWithIt) {
  auto budget = Plenty();
  auto fixture = GuardedIndirect(budget, true);
  ASSERT_TRUE(fixture);
  auto proved =
      ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population, fixture->sources, budget);
  ASSERT_TRUE(proved.facts);
  auto retired =
      recovery::ProposeSsaBranchRetirement(fixture->graph, *proved.facts, fixture->sources, budget);
  ASSERT_TRUE(retired.provisional) << static_cast<int>(retired.reason);
  auto after = ProveSsaReachability(*retired.provisional, fixture->sources,
                                    ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(after.facts) << static_cast<int>(after.reason);
  auto cleaned = recovery::ProposeUnreachableBlocks(*retired.provisional, *after.facts,
                                                    fixture->sources, budget);
  ASSERT_TRUE(cleaned.provisional) << static_cast<int>(cleaned.reason);
  EXPECT_FALSE(At(*cleaned.provisional, 0x104));
}

TEST(SsaConstants, AGuardDecidedThroughAPhiRetiresItsUnresolvedArm) {
  auto budget = Plenty();
  auto fixture = PhiGuardedIndirect(budget);
  ASSERT_TRUE(fixture);
  EXPECT_EQ(ProveSsaReachability(fixture->graph, fixture->sources,
                                 ir::SsaEntryScope::closed_population, budget)
                .reason,
            SsaReachabilityRefusal::incomplete_successors);
  auto proved =
      ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population, fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  const auto* dead = At(fixture->graph, 0x108);
  ASSERT_TRUE(dead);
  auto read =
      recovery::ProposeSsaSccpReadFold(fixture->graph, *proved.facts, fixture->sources, budget);
  ASSERT_TRUE(read.provisional) << static_cast<int>(read.reason);
  auto again = ProveSsaSccp(*read.provisional, ir::SsaEntryScope::closed_population,
                            fixture->sources, budget);
  ASSERT_TRUE(again.facts) << static_cast<int>(again.reason);
  auto retired = recovery::ProposeSsaBranchRetirement(*read.provisional, *again.facts,
                                                      fixture->sources, budget);
  ASSERT_TRUE(retired.provisional) << static_cast<int>(retired.reason);
  auto after = ProveSsaReachability(*retired.provisional, fixture->sources,
                                    ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(after.facts) << static_cast<int>(after.reason);
}

TEST(SsaConstants, SccpBudgetCutsOnAnIncompleteGraphPublishNothing) {
  auto setup = Plenty();
  auto fixture = GuardedIndirect(setup, true);
  ASSERT_TRUE(fixture);
  auto full = Plenty();
  auto proved =
      ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population, fixture->sources, full);
  ASSERT_TRUE(proved.facts);
  const auto used = full.used();
  for (std::uint64_t cut = 0; cut < used.work; ++cut) {
    Budget limited({cut, used.bytes});
    const auto result = ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population,
                                     fixture->sources, limited);
    EXPECT_EQ(result.reason, SsaSccpRefusal::resource_limit) << cut;
    EXPECT_FALSE(result.facts) << cut;
  }

  auto unknown = GuardedIndirect(setup, {});
  ASSERT_TRUE(unknown);
  auto ufull = Plenty();
  auto refused =
      ProveSsaSccp(unknown->graph, ir::SsaEntryScope::closed_population, unknown->sources, ufull);
  ASSERT_EQ(refused.reason, SsaSccpRefusal::incomplete_successors);
  const auto uused = ufull.used();
  for (std::uint64_t cut = 0; cut < uused.work; ++cut) {
    Budget limited({cut, uused.bytes});
    const auto result = ProveSsaSccp(unknown->graph, ir::SsaEntryScope::closed_population,
                                     unknown->sources, limited);
    EXPECT_EQ(result.reason, SsaSccpRefusal::resource_limit) << cut;
  }
}

std::optional<Fixture> Assemble(std::vector<ir::Group> sources, Budget& budget) {
  std::vector<SourceRecord> records;
  for (const auto& source : sources)
    records.push_back(
        {source.source_address(), {source.bytes().begin(), source.bytes().end()}, source});
  constexpr std::uint64_t entry = 0x100;
  auto cfg = BuildCfg(records, std::span(&entry, 1), budget);
  if (!cfg.cfg) return {};
  Regions regions(std::move(*cfg.cfg), {});
  auto recovered = Unflatten(regions, {}, budget, {{}, true});
  if (!recovered.unflattening) return {};
  const std::vector<std::optional<ir::RecoveredPath>> paths;
  auto built = BuildSsa(regions, *recovered.unflattening, paths, budget);
  if (!built.graph) return {};
  return Fixture{std::move(*built.graph), std::move(sources)};
}

// The page base of an FNV loop's hidden constants: computed once in the entry
// block, carried round the loop in x8, and offset in the loop block.
std::optional<Fixture> PagedLoop(Budget& budget) {
  std::vector<ir::Group> sources;
  sources.emplace_back(0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
                       std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x2000},
                                             {ir::Op::image_address, 64, {}, 0x104}},
                       std::vector<ir::Write>{{8, 0}}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 1, {}, {}, {}});
  sources.emplace_back(0x104, std::vector<std::uint8_t>{5, 6, 7, 8},
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 8},
                                             {ir::Op::constant, 64, {}, 0xd88},
                                             {ir::Op::add, 64, {0, 1}},
                                             {ir::Op::read, 1, {}, 0, 32},
                                             {ir::Op::image_address, 64, {}, 0x108},
                                             {ir::Op::image_address, 64, {}, 0x10c}},
                       std::vector<ir::Write>{{9, 2}}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::conditional, 4, 3, 5, {}});
  sources.emplace_back(0x108, std::vector<std::uint8_t>{9, 10, 11, 12},
                       std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x104}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}});
  sources.emplace_back(0x10c, std::vector<std::uint8_t>{13, 14, 15, 16},
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 30}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}});
  return Assemble(std::move(sources), budget);
}

TEST(SsaConstants, APageBaseCarriedRoundALoopSettlesUnderADeclaredBias) {
  constexpr std::uint64_t bias = 0x7f0000000000;
  auto budget = Plenty();
  auto fixture = PagedLoop(budget);
  ASSERT_TRUE(fixture);
  const auto* loop = At(fixture->graph, 0x104);
  ASSERT_TRUE(loop);
  const auto offset = [&](const ir::SsaSccpFacts& facts) {
    return std::find_if(facts.constants.values.begin(), facts.constants.values.end(),
                        [&](const ir::SsaConstantValue& fact) {
                          const auto* block = fixture->graph.Get(fact.block);
                          return block && block->address == 0x104 &&
                                 fact.kind == ir::SsaValueKind::node &&
                                 block->nodes[fact.index].op == ir::Op::add;
                        });
  };

  // Without a placement the page is a place, and the sum is not a number.
  auto unplaced =
      ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population, fixture->sources, budget);
  ASSERT_TRUE(unplaced.facts) << static_cast<int>(unplaced.reason);
  EXPECT_EQ(offset(*unplaced.facts), unplaced.facts->constants.values.end());

  // Nor may a certificate say otherwise of the location itself.
  const auto entry = fixture->graph.entries().front();
  const auto& entry_nodes = fixture->graph.Get(entry)->nodes;
  const auto page = std::find_if(entry_nodes.begin(), entry_nodes.end(), [](const ir::Node& node) {
    return node.op == ir::Op::image_address && node.immediate == 0x2000;
  });
  ASSERT_NE(page, entry_nodes.end());
  const ir::SsaConstantValue claim{entry, ir::SsaValueKind::node,
                                   static_cast<std::uint32_t>(page - entry_nodes.begin()),
                                   bias + 0x2000};
  const auto forge = [&](ir::SsaSccpFacts facts, std::uint64_t value) {
    auto forged = claim;
    forged.value = value;
    auto& values = facts.constants.values;
    values.erase(std::remove_if(values.begin(), values.end(),
                                [&](const auto& fact) {
                                  return fact.block == forged.block && fact.kind == forged.kind &&
                                         fact.index == forged.index;
                                }),
                 values.end());
    values.insert(std::find_if(values.begin(), values.end(),
                               [&](const auto& fact) {
                                 return fact.block.slot > forged.block.slot ||
                                        (fact.block.slot == forged.block.slot &&
                                         fact.kind == ir::SsaValueKind::node &&
                                         fact.index > forged.index);
                               }),
                  forged);
    return facts;
  };

  EXPECT_EQ(ir::ValidateSsaSccpFacts(fixture->graph, forge(*unplaced.facts, claim.value),
                                     fixture->sources, budget),
            ir::SsaDecline::invalid_graph);
  fixture->graph.SetLoadBias(bias);
  auto placed =
      ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population, fixture->sources, budget);
  ASSERT_TRUE(placed.facts) << static_cast<int>(placed.reason);
  const auto sum = offset(*placed.facts);
  ASSERT_NE(sum, placed.facts->constants.values.end());
  EXPECT_EQ(sum->value, bias + 0x2d88);
  EXPECT_EQ(ir::ValidateSsaSccpFacts(fixture->graph, *placed.facts, fixture->sources, budget),
            ir::SsaDecline::none);
  EXPECT_EQ(ir::ValidateSsaSccpFacts(fixture->graph, forge(*placed.facts, claim.value),
                                     fixture->sources, budget),
            ir::SsaDecline::none);
  EXPECT_EQ(ir::ValidateSsaSccpFacts(fixture->graph, forge(*placed.facts, claim.value + 8),
                                     fixture->sources, budget),
            ir::SsaDecline::invalid_graph);
}

// Both arms of an unknown condition named as image arithmetic, as `adr` and a
// displacement off the pc produce them.
std::optional<Fixture> ComputedArms(Budget& budget) {
  std::vector<ir::Group> sources;
  sources.emplace_back(0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
                       std::vector<ir::Node>{{ir::Op::read, 1, {}, 0, 32},
                                             {ir::Op::image_address, 64, {}, 0x100},
                                             {ir::Op::constant, 64, {}, 8},
                                             {ir::Op::add, 64, {1, 2}},
                                             {ir::Op::constant, 64, {}, 4},
                                             {ir::Op::add, 64, {1, 4}}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::conditional, 3, 0, 5, {}});
  for (const std::uint64_t address : {0x104, 0x108})
    sources.emplace_back(address, std::vector<std::uint8_t>{5, 6, 7, 8},
                         std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 30}},
                         std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                         ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}});
  return Assemble(std::move(sources), budget);
}

TEST(SsaConstants, AFoldedBranchTargetStaysAnImageLocation) {
  constexpr std::uint64_t bias = 0x7f0000000000;
  auto budget = Plenty();
  auto fixture = ComputedArms(budget);
  ASSERT_TRUE(fixture);
  fixture->graph.SetLoadBias(bias);
  auto proved =
      ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population, fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  auto folded =
      recovery::ProposeSsaSccpFold(fixture->graph, *proved.facts, fixture->sources, budget);
  ASSERT_TRUE(folded.provisional) << static_cast<int>(folded.reason);
  ASSERT_EQ(folded.journal.size(), 2U);
  const auto* entry = folded.provisional->Get(folded.provisional->entries().front());
  ASSERT_TRUE(entry);
  std::vector<std::uint64_t> targets;
  for (const auto& edit : folded.journal) {
    const auto& node = entry->nodes[edit.node];
    EXPECT_EQ(node.op, ir::Op::image_address);
    EXPECT_EQ(node.immediate, edit.value - bias);
    targets.push_back(node.immediate);
  }

  std::sort(targets.begin(), targets.end());
  EXPECT_EQ(targets, (std::vector<std::uint64_t>{0x104, 0x108}));
  EXPECT_EQ(ir::ValidateSsaWithSources(*folded.provisional, fixture->sources, budget),
            ir::SsaDecline::none);
}

TEST(SsaConstants, ARegisterCarryingOnlyALocationFoldsToThatLocation) {
  constexpr std::uint64_t bias = 0x7f0000000000;
  auto budget = Plenty();

  // x8 = @0x2000 + 0x10, x9 = a number past the bias; the next block stores
  // x9 at x8 + x1.
  std::vector<ir::Group> sources;
  sources.emplace_back(0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
                       std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x2000},
                                             {ir::Op::constant, 64, {}, 0x10},
                                             {ir::Op::add, 64, {0, 1}},
                                             {ir::Op::constant, 64, {}, bias + 0x3000},
                                             {ir::Op::image_address, 64, {}, 0x104}},
                       std::vector<ir::Write>{{8, 2}, {9, 3}}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 4, {}, {}, {}});
  sources.emplace_back(0x104, std::vector<std::uint8_t>{5, 6, 7, 8},
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 8},
                                             {ir::Op::read, 64, {}, 0, 9},
                                             {ir::Op::read, 64, {}, 0, 1},
                                             {ir::Op::add, 64, {0, 2}},
                                             {ir::Op::store, 64, {3, 1}},
                                             {ir::Op::read, 64, {}, 0, 30}},
                       std::vector<ir::Write>{}, ir::MemoryModel::atomic_scalar_reference,
                       ir::Transfer{ir::TransferKind::return_, 5, {}, {}, {}});
  auto fixture = Assemble(std::move(sources), budget);
  ASSERT_TRUE(fixture);
  fixture->graph.SetLoadBias(bias);
  auto proved =
      ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population, fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  auto folded =
      recovery::ProposeSsaSccpReadFold(fixture->graph, *proved.facts, fixture->sources, budget);
  ASSERT_TRUE(folded.provisional) << static_cast<int>(folded.reason);
  EXPECT_EQ(ir::ValidateSsaWithSources(*folded.provisional, fixture->sources, budget),
            ir::SsaDecline::none);
  const ir::SsaBlock* body = nullptr;
  for (std::size_t slot = 0; slot < folded.provisional->slots(); ++slot)
    if (const auto handle = folded.provisional->Handle(slot);
        handle && folded.provisional->Get(*handle)->address == 0x104)
      body = folded.provisional->Get(*handle);
  ASSERT_TRUE(body);
  EXPECT_EQ(body->nodes[0].op, ir::Op::image_address);
  EXPECT_EQ(body->nodes[0].immediate, 0x2010U);

  // A number is still a number, even past the bias.
  EXPECT_EQ(body->nodes[1].op, ir::Op::constant);
  EXPECT_EQ(body->nodes[1].immediate, bias + 0x3000);

  // So the indexed store is image-derived and may write @0x2010 onwards.
  EXPECT_EQ(ir::SsaConflictingImageStore(*folded.provisional, 0x2018, 8, true, budget), true);
  EXPECT_EQ(ir::SsaConflictingImageStore(*folded.provisional, 0x2018, 8, true, budget, true),
            false);
}

TEST(SsaConstants, AStoreThroughACarriedLocationRefusesTheFoldBeforeAndAfterTheReadFold) {
  constexpr std::uint64_t bias = 0x7f0000000000;
  auto budget = Plenty();

  // x8 = @0x2010 carried into the next block, which stores at x8 + x1 and then
  // reads @0x2018, declared but writable.
  std::vector<ir::Group> sources;
  sources.emplace_back(0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
                       std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x2000},
                                             {ir::Op::constant, 64, {}, 0x10},
                                             {ir::Op::add, 64, {0, 1}},
                                             {ir::Op::image_address, 64, {}, 0x104}},
                       std::vector<ir::Write>{{8, 2}}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 3, {}, {}, {}});
  sources.emplace_back(0x104, std::vector<std::uint8_t>{5, 6, 7, 8},
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 8},
                                             {ir::Op::read, 64, {}, 0, 1},
                                             {ir::Op::add, 64, {0, 1}},
                                             {ir::Op::store, 64, {2, 1}},
                                             {ir::Op::image_address, 64, {}, 0x2018},
                                             {ir::Op::load, 64, {4}},
                                             {ir::Op::read, 64, {}, 0, 30}},
                       std::vector<ir::Write>{{0, 5}}, ir::MemoryModel::atomic_scalar_reference,
                       ir::Transfer{ir::TransferKind::return_, 6, {}, {}, {}});
  auto fixture = Assemble(std::move(sources), budget);
  ASSERT_TRUE(fixture);
  fixture->graph.SetLoadBias(bias);
  constexpr std::array<std::uint8_t, 8> bytes{7};
  const ir::ConstantImageRange range{0x2018, bytes};
  const ir::ImageFacts facts{std::span(&range, 1), {}, true, bias};
  const ir::ImageAccessContract access{true, true, true};
  const auto refused = [&](const ir::SsaGraph& graph) {
    auto result = recovery::ProposeConstantImageLoads(graph, facts, access, budget);
    return !result.provisional && result.refused.size() == 1 &&
           result.refused[0].reason == recovery::ConstantLoadRefusal::conflicting_store;
  };

  EXPECT_TRUE(refused(fixture->graph));

  // The pipeline's order: image loads, then the read fold on what they leave.
  // A load fold the store check accepted here would be one the read fold's
  // candidate then contradicts, and the read fold would decline.
  auto loads = recovery::ProposeConstantImageLoads(fixture->graph, facts, access, budget);
  const auto& after_loads = loads.provisional ? *loads.provisional : fixture->graph;
  auto proved =
      ProveSsaSccp(after_loads, ir::SsaEntryScope::closed_population, fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  auto folded =
      recovery::ProposeSsaSccpReadFold(after_loads, *proved.facts, fixture->sources, budget);
  ASSERT_TRUE(folded.provisional) << static_cast<int>(folded.reason);
  EXPECT_TRUE(refused(*folded.provisional));

  // Were the fold there already, the read fold could not go ahead.
  auto forced = std::move(fixture->graph);
  for (std::size_t slot = 0; slot < forced.slots(); ++slot) {
    const auto handle = forced.Handle(slot);
    if (!handle || forced.Get(*handle)->address != 0x104) continue;
    forced.Update(*handle, [](ir::SsaBlock& block) {
      block.disabled_effects = {5};
      block.constant_loads = {{5, ir::SsaConstantKind::literal, 7, 0x2018}};
      block.constant_loads[0].declared_bytes = {7};
      block.constant_loads[0].access = {true, true, true};
      block.constant_loads[0].page_aligned_placement = true;
      block.constant_loads[0].value_stable = true;
      block.constant_loads[0].skip_access = true;
    });
  }

  EXPECT_EQ(ir::ValidateSsa(forced, budget), ir::SsaDecline::invalid_graph);
}

TEST(SsaConstants, AStoreThroughACarriedLocationIsPlacedWhereEveryWayInAgrees) {
  constexpr std::uint64_t bias = 0x7f0000000000;
  auto budget = Plenty();

  // x8 = @0x2010 carried into the next block, which stores 8 bytes at x8 and
  // then reads @0x2018 and @0x2010.
  std::vector<ir::Group> sources;
  sources.emplace_back(0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
                       std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x2000},
                                             {ir::Op::constant, 64, {}, 0x10},
                                             {ir::Op::add, 64, {0, 1}},
                                             {ir::Op::image_address, 64, {}, 0x104}},
                       std::vector<ir::Write>{{8, 2}}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 3, {}, {}, {}});
  sources.emplace_back(0x104, std::vector<std::uint8_t>{5, 6, 7, 8},
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 8},
                                             {ir::Op::read, 64, {}, 0, 1},
                                             {ir::Op::store, 64, {0, 1}},
                                             {ir::Op::image_address, 64, {}, 0x2018},
                                             {ir::Op::load, 64, {3}},
                                             {ir::Op::image_address, 64, {}, 0x2010},
                                             {ir::Op::load, 64, {5}},
                                             {ir::Op::add, 64, {4, 6}},
                                             {ir::Op::read, 64, {}, 0, 30}},
                       std::vector<ir::Write>{{0, 7}}, ir::MemoryModel::atomic_scalar_reference,
                       ir::Transfer{ir::TransferKind::return_, 8, {}, {}, {}});
  auto fixture = Assemble(std::move(sources), budget);
  ASSERT_TRUE(fixture);
  fixture->graph.SetLoadBias(bias);
  constexpr std::array<std::uint8_t, 16> bytes{7};
  const ir::ConstantImageRange range{0x2010, bytes};
  const ir::ImageFacts facts{std::span(&range, 1), {}, true, bias};
  const ir::ImageAccessContract access{true, true, true};

  // Without every way in listed, the block may be entered holding anything,
  // so the store could be anywhere writable.
  {
    auto open = recovery::ProposeConstantImageLoads(fixture->graph, facts, access, budget);
    EXPECT_TRUE(open.journal.empty());
    EXPECT_EQ(open.refused.size(), 2U);
  }

  fixture->graph.SetEntriesClosed(true);
  const auto check = [&](const ir::SsaGraph& graph) {
    auto result = recovery::ProposeConstantImageLoads(graph, facts, access, budget);
    ASSERT_EQ(result.journal.size(), 1U);
    EXPECT_EQ(result.journal[0].fold.source_address, 0x2018U);
    ASSERT_EQ(result.refused.size(), 1U);
    EXPECT_EQ(result.refused[0].reason, recovery::ConstantLoadRefusal::conflicting_store);
  };

  check(fixture->graph);
  auto proved =
      ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population, fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  auto folded =
      recovery::ProposeSsaSccpReadFold(fixture->graph, *proved.facts, fixture->sources, budget);
  ASSERT_TRUE(folded.provisional) << static_cast<int>(folded.reason);
  check(*folded.provisional);
}

TEST(SsaConstants, AStoreThroughAPointerALoopMovesOnIsAnImageStore) {
  auto budget = Plenty();

  // x8 = @0x3000; loop: store [x8]; x8 += 8. Either spelling of the loop:
  // advanced in the storing block, or in a latch after it.
  for (const bool latch : {false, true}) {
    std::vector<ir::Group> sources;
    sources.emplace_back(0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
                         std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x3000},
                                               {ir::Op::image_address, 64, {}, 0x104}},
                         std::vector<ir::Write>{{8, 0}}, ir::MemoryModel::atomic_scalar_reference,
                         ir::Transfer{ir::TransferKind::jump, 1, {}, {}, {}});
    if (!latch) {
      sources.emplace_back(0x104, std::vector<std::uint8_t>{5, 6, 7, 8},
                           std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 8},
                                                 {ir::Op::constant, 64, {}, 1},
                                                 {ir::Op::store, 64, {0, 1}},
                                                 {ir::Op::constant, 64, {}, 8},
                                                 {ir::Op::add, 64, {0, 3}},
                                                 {ir::Op::read, 1, {}, 0, 32},
                                                 {ir::Op::image_address, 64, {}, 0x104},
                                                 {ir::Op::image_address, 64, {}, 0x108}},
                           std::vector<ir::Write>{{8, 4}}, ir::MemoryModel::atomic_scalar_reference,
                           ir::Transfer{ir::TransferKind::conditional, 6, 5, 7, {}});
    } else {
      sources.emplace_back(0x104, std::vector<std::uint8_t>{5, 6, 7, 8},
                           std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 8},
                                                 {ir::Op::constant, 64, {}, 1},
                                                 {ir::Op::store, 64, {0, 1}},
                                                 {ir::Op::read, 1, {}, 0, 32},
                                                 {ir::Op::image_address, 64, {}, 0x10c},
                                                 {ir::Op::image_address, 64, {}, 0x108}},
                           std::vector<ir::Write>{}, ir::MemoryModel::atomic_scalar_reference,
                           ir::Transfer{ir::TransferKind::conditional, 4, 3, 5, {}});
      sources.emplace_back(0x10c, std::vector<std::uint8_t>{9, 9, 9, 9},
                           std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 8},
                                                 {ir::Op::constant, 64, {}, 8},
                                                 {ir::Op::add, 64, {0, 1}},
                                                 {ir::Op::image_address, 64, {}, 0x104}},
                           std::vector<ir::Write>{{8, 2}}, ir::MemoryModel::unspecified,
                           ir::Transfer{ir::TransferKind::jump, 3, {}, {}, {}});
    }

    sources.emplace_back(0x108, std::vector<std::uint8_t>{9, 10, 11, 12},
                         std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 30}},
                         std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                         ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}});
    auto fixture = Assemble(std::move(sources), budget);
    ASSERT_TRUE(fixture);
    for (const bool closed : {false, true}) {
      fixture->graph.SetEntriesClosed(closed);

      // The store may be at @0x3000 or any step past it; nothing names which.
      EXPECT_EQ(ir::SsaConflictingImageStore(fixture->graph, 0x3000, 8, true, budget), true)
          << latch << closed;
      EXPECT_EQ(ir::SsaConflictingImageStore(fixture->graph, 0x3008, 8, true, budget), true)
          << latch << closed;
      EXPECT_EQ(ir::SsaConflictingImageStore(fixture->graph, 0x3008, 8, true, budget, true), false)
          << latch << closed;
    }
  }
}

TEST(SsaConstants, APathReadAStoreInTheGraphWritesIsNotRecorded) {
  auto budget = Plenty();

  // x9 = @0x2000 carried into the next block, which stores through it and
  // then jumps to what @0x2000 holds. The jump reads a declared word, so the
  // CFG resolves it; the store is one only the whole graph places.
  std::vector<ir::Group> sources;
  sources.emplace_back(0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
                       std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x2000},
                                             {ir::Op::image_address, 64, {}, 0x104}},
                       std::vector<ir::Write>{{9, 0}}, ir::MemoryModel::atomic_scalar_reference,
                       ir::Transfer{ir::TransferKind::jump, 1, {}, {}, {}});
  sources.emplace_back(0x104, std::vector<std::uint8_t>{5, 6, 7, 8},
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 9},
                                             {ir::Op::read, 64, {}, 0, 1},
                                             {ir::Op::store, 64, {0, 1}},
                                             {ir::Op::image_address, 64, {}, 0x2000},
                                             {ir::Op::load, 64, {3}},
                                             {ir::Op::image_address, 64, {}, 0},
                                             {ir::Op::add, 64, {5, 4}}},
                       std::vector<ir::Write>{}, ir::MemoryModel::atomic_scalar_reference,
                       ir::Transfer{ir::TransferKind::jump, 6, {}, {}, {}});
  sources.emplace_back(0x108, std::vector<std::uint8_t>{9, 10, 11, 12},
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 30}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}});
  constexpr std::array<std::uint8_t, 8> word{0x08, 0x01};
  const ir::ConstantImageRange range{0x2000, word};
  const ImageFacts facts{std::span(&range, 1), {}, false, {}};
  std::vector<SourceRecord> records;
  for (const auto& source : sources)
    records.push_back(
        {source.source_address(), {source.bytes().begin(), source.bytes().end()}, source});
  constexpr std::uint64_t entry = 0x100;
  auto cfg = BuildCfg(records, std::span(&entry, 1), budget, {}, facts);
  ASSERT_TRUE(cfg.cfg);
  Regions regions(std::move(*cfg.cfg), {});
  auto recovered = Unflatten(regions, {}, budget, {{}, true});
  ASSERT_TRUE(recovered.unflattening);
  const std::vector<std::optional<ir::RecoveredPath>> paths;

  // Only with every way in listed does the graph say where x9 points.
  auto open = BuildSsa(regions, *recovered.unflattening, paths, budget, nullptr, facts);
  ASSERT_TRUE(open.graph) << static_cast<int>(open.reason);
  auto built = BuildSsa(regions, *recovered.unflattening, paths, budget, nullptr, facts, {}, true);
  ASSERT_TRUE(built.graph) << static_cast<int>(built.reason);
  for (std::size_t slot = 0; slot < open.graph->slots(); ++slot) {
    const auto at = open.graph->Handle(slot);
    if (at && open.graph->Get(*at)->address == 0x104) {
      EXPECT_EQ(open.graph->Get(*at)->path_reads.size(), 1U);
    }
  }

  const ir::SsaBlock* jump = nullptr;
  ir::SsaHandle handle{};
  for (std::size_t slot = 0; slot < built.graph->slots(); ++slot)
    if (const auto at = built.graph->Handle(slot); at && built.graph->Get(*at)->address == 0x104) {
      jump = built.graph->Get(*at);
      handle = *at;
    }
  ASSERT_TRUE(jump);
  EXPECT_TRUE(jump->path_reads.empty());

  // The checker refuses the record the builder left out.
  ASSERT_TRUE(built.graph->Update(
      handle, [](ir::SsaBlock& block) { block.path_reads = {{4, 0x2000, false, 0x108}}; }));
  EXPECT_EQ(ir::ValidateSsa(*built.graph, budget), ir::SsaDecline::invalid_graph);
}

TEST(SsaConstants, AFoldThatPlacesAStoreOnAPathReadDropsTheRecord) {
  constexpr std::uint64_t bias = 0x7f0000000000;
  auto budget = Plenty();

  // x9 = @0x2000 carried into the next block, which stores through it and
  // then returns what @0x2000 holds. Nobody knows where x9 points until the
  // read fold writes the location in.
  std::vector<ir::Group> sources;
  sources.emplace_back(0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
                       std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x2000},
                                             {ir::Op::image_address, 64, {}, 0x104}},
                       std::vector<ir::Write>{{9, 0}}, ir::MemoryModel::atomic_scalar_reference,
                       ir::Transfer{ir::TransferKind::jump, 1, {}, {}, {}});
  sources.emplace_back(0x104, std::vector<std::uint8_t>{5, 6, 7, 8},
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 9},
                                             {ir::Op::read, 64, {}, 0, 1},
                                             {ir::Op::store, 64, {0, 1}},
                                             {ir::Op::image_address, 64, {}, 0x2000},
                                             {ir::Op::load, 64, {3}},
                                             {ir::Op::read, 64, {}, 0, 30}},
                       std::vector<ir::Write>{{0, 4}}, ir::MemoryModel::atomic_scalar_reference,
                       ir::Transfer{ir::TransferKind::return_, 5, {}, {}, {}});
  constexpr std::array<std::uint8_t, 8> word{0x08, 0x01};
  const ir::ConstantImageRange range{0x2000, word};
  const ImageFacts facts{std::span(&range, 1), {}, false, bias};
  std::vector<SourceRecord> records;
  for (const auto& source : sources)
    records.push_back(
        {source.source_address(), {source.bytes().begin(), source.bytes().end()}, source});
  constexpr std::uint64_t entry = 0x100;
  auto cfg = BuildCfg(records, std::span(&entry, 1), budget, {}, facts);
  ASSERT_TRUE(cfg.cfg);
  Regions regions(std::move(*cfg.cfg), {});
  auto recovered = Unflatten(regions, {}, budget, {{}, true});
  ASSERT_TRUE(recovered.unflattening);
  const std::vector<std::optional<ir::RecoveredPath>> paths;
  auto built = BuildSsa(regions, *recovered.unflattening, paths, budget, nullptr, facts);
  ASSERT_TRUE(built.graph) << static_cast<int>(built.reason);
  const auto path_reads = [](const ir::SsaGraph& graph) {
    std::size_t count = 0;
    for (std::size_t slot = 0; slot < graph.slots(); ++slot)
      if (const auto at = graph.Handle(slot)) count += graph.Get(*at)->path_reads.size();
    return count;
  };

  ASSERT_EQ(path_reads(*built.graph), 1U);
  auto proved = ProveSsaSccp(*built.graph, ir::SsaEntryScope::closed_population, sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  auto folded = recovery::ProposeSsaSccpReadFold(*built.graph, *proved.facts, sources, budget);
  ASSERT_TRUE(folded.provisional) << static_cast<int>(folded.reason);
  EXPECT_EQ(path_reads(*folded.provisional), 0U);

  // The fold says what it dropped.
  ASSERT_EQ(folded.dropped_path_reads.size(), 1U);
  EXPECT_EQ(folded.dropped_path_reads[0].node, 4U);
  EXPECT_EQ(folded.dropped_path_reads[0].address, 0x2000U);
  EXPECT_EQ(folded.provisional->Get(folded.dropped_path_reads[0].block)->address, 0x104U);
}

TEST(SsaConstants, APointerLoadedFromADeclaredSlotAndCarriedOnIsAnImageStore) {
  // x8 = *@0x2000, a relocated slot holding @0x3000; the next block stores a
  // byte at x8 + x1 and then reads @0x3000.
  for (const bool closed : {false, true}) {
    std::vector<ir::Group> sources;
    sources.emplace_back(0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
                         std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x2000},
                                               {ir::Op::load, 64, {0}},
                                               {ir::Op::image_address, 64, {}, 0x104}},
                         std::vector<ir::Write>{{8, 1}}, ir::MemoryModel::atomic_scalar_reference,
                         ir::Transfer{ir::TransferKind::jump, 2, {}, {}, {}});
    sources.emplace_back(0x104, std::vector<std::uint8_t>{5, 6, 7, 8},
                         std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 8},
                                               {ir::Op::read, 64, {}, 0, 1},
                                               {ir::Op::add, 64, {0, 1}},
                                               {ir::Op::constant, 8, {}, 0x5a},
                                               {ir::Op::store, 8, {2, 3}},
                                               {ir::Op::image_address, 64, {}, 0x3000},
                                               {ir::Op::load, 64, {5}},
                                               {ir::Op::read, 64, {}, 0, 30}},
                         std::vector<ir::Write>{{0, 6}}, ir::MemoryModel::atomic_scalar_reference,
                         ir::Transfer{ir::TransferKind::return_, 7, {}, {}, {}});
    auto budget = Plenty();
    auto fixture = Assemble(std::move(sources), budget);
    ASSERT_TRUE(fixture);
    fixture->graph.SetEntriesClosed(closed);
    constexpr std::array<std::uint8_t, 8> seven{7};
    const ir::ConstantImageRange range{0x3000, seven};
    const ir::RelocatedPointer pointer{0x2000, 0x3000, true};
    const ir::ImageFacts facts{std::span(&range, 1), std::span(&pointer, 1), true, {}};
    auto loads =
        recovery::ProposeConstantImageLoads(fixture->graph, facts, {true, true, true}, budget);
    const auto* body = [&]() -> const ir::SsaBlock* {
      for (std::size_t slot = 0; slot < fixture->graph.slots(); ++slot)
        if (const auto at = fixture->graph.Handle(slot);
            at && fixture->graph.Get(*at)->address == 0x104)
          return fixture->graph.Get(*at);
      return nullptr;
    }();
    ASSERT_TRUE(body);

    // The read of @0x3000 is refused; the slot itself may still fold.
    EXPECT_FALSE(std::any_of(loads.journal.begin(), loads.journal.end(), [](const auto& edit) {
      return edit.fold.source_address == 0x3000;
    })) << closed;
    EXPECT_TRUE(std::any_of(loads.refused.begin(), loads.refused.end(), [](const auto& item) {
      return item.node == 6 && item.reason == recovery::ConstantLoadRefusal::conflicting_store;
    })) << closed;

    // The checker, which has only the graph, refuses the @0x3000 fold once the
    // slot's fold says what x8 holds.
    if (!loads.provisional) continue;
    auto forced = std::move(*loads.provisional);
    for (std::size_t slot = 0; slot < forced.slots(); ++slot) {
      const auto at = forced.Handle(slot);
      if (!at || forced.Get(*at)->address != 0x104) continue;
      forced.Update(*at, [](ir::SsaBlock& block) {
        block.disabled_effects.push_back(6);
        std::sort(block.disabled_effects.begin(), block.disabled_effects.end());
        ir::SsaConstantLoad fold{6, ir::SsaConstantKind::literal, 7, 0x3000};
        fold.declared_bytes = {7};
        fold.access = {true, true, true};
        fold.page_aligned_placement = true;
        fold.value_stable = true;
        fold.skip_access = true;
        block.constant_loads.push_back(fold);
      });
    }

    EXPECT_EQ(ir::ValidateSsa(forced, budget), ir::SsaDecline::invalid_graph) << closed;
  }
}

TEST(SsaConstants, SimplifyDropsAPathReadItsRewriteLetsAStoreReach) {
  // store [@0x3000 | 0]; then read @0x3000. Under page-aligned placement the
  // store address is a number the bias cannot move, so the read of @0x3000
  // stands. Simplify rewrites `x | 0` to `@0x3000`, placing the store on it.
  auto budget = Plenty();
  std::vector<ir::Group> sources;
  sources.emplace_back(0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
                       std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x3000},
                                             {ir::Op::constant, 64, {}, 0},
                                             {ir::Op::bit_or, 64, {0, 1}},
                                             {ir::Op::constant, 8, {}, 0x5a},
                                             {ir::Op::store, 8, {2, 3}},
                                             {ir::Op::image_address, 64, {}, 0x3000},
                                             {ir::Op::load, 64, {5}},
                                             {ir::Op::read, 64, {}, 0, 30}},
                       std::vector<ir::Write>{{0, 6}}, ir::MemoryModel::atomic_scalar_reference,
                       ir::Transfer{ir::TransferKind::return_, 7, {}, {}, {}});
  constexpr std::array<std::uint8_t, 8> seven{7};
  const ir::ConstantImageRange range{0x3000, seven};
  const ImageFacts facts{std::span(&range, 1), {}, true, {}};
  std::vector<SourceRecord> records;
  for (const auto& source : sources)
    records.push_back(
        {source.source_address(), {source.bytes().begin(), source.bytes().end()}, source});
  constexpr std::uint64_t entry = 0x100;
  auto cfg = BuildCfg(records, std::span(&entry, 1), budget, {}, facts);
  ASSERT_TRUE(cfg.cfg);
  Regions regions(std::move(*cfg.cfg), {});
  auto recovered = Unflatten(regions, {}, budget, {{}, true});
  ASSERT_TRUE(recovered.unflattening);
  const std::vector<std::optional<ir::RecoveredPath>> paths;
  auto built = BuildSsa(regions, *recovered.unflattening, paths, budget, nullptr, facts);
  ASSERT_TRUE(built.graph) << static_cast<int>(built.reason);
  const auto path_reads = [](const ir::SsaGraph& graph) {
    std::size_t count = 0;
    for (std::size_t slot = 0; slot < graph.slots(); ++slot)
      if (const auto at = graph.Handle(slot)) count += graph.Get(*at)->path_reads.size();
    return count;
  };

  ASSERT_EQ(path_reads(*built.graph), 1U);

  // The store is not placed before the rewrite, so the graph is valid.
  EXPECT_EQ(ir::ValidateSsa(*built.graph, budget), ir::SsaDecline::none);
  auto simplified = recovery::ProposeSsaSimplify(*built.graph, {}, true, budget);
  ASSERT_EQ(simplified.reason, recovery::SsaSimplifyRefusal::none);
  ASSERT_TRUE(simplified.provisional);
  EXPECT_EQ(ir::ValidateSsa(*simplified.provisional, budget), ir::SsaDecline::none);
  EXPECT_EQ(path_reads(*simplified.provisional), 0U);
  ASSERT_EQ(simplified.dropped_path_reads.size(), 1U);
  EXPECT_EQ(simplified.dropped_path_reads[0].address, 0x3000U);
}

TEST(SsaConstants, AnEarlierReachQueryWithOtherFactsChangesNothing) {
  // Same shape as the loaded-pointer test; a query first made with facts that
  // do not declare the slot must not decide what the real facts refuse.
  for (const bool primed : {false, true}) {
    std::vector<ir::Group> sources;
    sources.emplace_back(0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
                         std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x2000},
                                               {ir::Op::load, 64, {0}},
                                               {ir::Op::image_address, 64, {}, 0x104}},
                         std::vector<ir::Write>{{8, 1}}, ir::MemoryModel::atomic_scalar_reference,
                         ir::Transfer{ir::TransferKind::jump, 2, {}, {}, {}});
    sources.emplace_back(0x104, std::vector<std::uint8_t>{5, 6, 7, 8},
                         std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 8},
                                               {ir::Op::read, 64, {}, 0, 1},
                                               {ir::Op::add, 64, {0, 1}},
                                               {ir::Op::constant, 8, {}, 0x5a},
                                               {ir::Op::store, 8, {2, 3}},
                                               {ir::Op::image_address, 64, {}, 0x3000},
                                               {ir::Op::load, 64, {5}},
                                               {ir::Op::read, 64, {}, 0, 30}},
                         std::vector<ir::Write>{{0, 6}}, ir::MemoryModel::atomic_scalar_reference,
                         ir::Transfer{ir::TransferKind::return_, 7, {}, {}, {}});
    auto budget = Plenty();
    auto fixture = Assemble(std::move(sources), budget);
    ASSERT_TRUE(fixture);
    constexpr std::array<std::uint8_t, 8> seven{7};
    const ir::ConstantImageRange range{0x3000, seven};
    const ir::RelocatedPointer pointer{0x2000, 0x3000, true};
    if (primed) {
      const ir::ImageFacts none{std::span(&range, 1), {}, true, {}};
      ASSERT_TRUE(ir::SsaImageReaching(fixture->graph, budget, &none));
    }

    const ir::ImageFacts facts{std::span(&range, 1), std::span(&pointer, 1), true, {}};
    auto loads =
        recovery::ProposeConstantImageLoads(fixture->graph, facts, {true, true, true}, budget);
    EXPECT_FALSE(std::any_of(loads.journal.begin(), loads.journal.end(), [](const auto& edit) {
      return edit.fold.source_address == 0x3000;
    })) << primed;
  }
}
}  // namespace
}  // namespace nyx::analysis
