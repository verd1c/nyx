#include <gtest/gtest.h>

#include "nyx/analysis/ssa/build.hpp"
#include "nyx/analysis/ssa/phi_constants.hpp"
#include "nyx/analysis/ssa/reachability.hpp"
#include "nyx/recovery/ssa/phi_constants.hpp"

namespace nyx::analysis {
namespace {
Budget Plenty() { return Budget({1000000, 10000000}); }

struct Fixture {
  ir::SsaGraph graph;
  std::vector<ir::Group> sources;
};

std::optional<Fixture> Join(Budget& budget, std::uint64_t right_value = 11,
                            bool return_leaves = true) {
  std::vector<ir::Group> sources;
  sources.emplace_back(0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
                       std::vector<ir::Node>{{ir::Op::read, 1, {}, 0, 32},
                                             {ir::Op::image_address, 64, {}, 0x108},
                                             {ir::Op::image_address, 64, {}, 0x104}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::conditional, 1, 0, 2, {}});
  sources.emplace_back(
      0x104, std::vector<std::uint8_t>{5, 6, 7, 8},
      std::vector<ir::Node>{{ir::Op::constant, 64, {}, 11}, {ir::Op::image_address, 64, {}, 0x10c}},
      std::vector<ir::Write>{{8, 0}}, ir::MemoryModel::unspecified,
      ir::Transfer{ir::TransferKind::jump, 1, {}, {}, {}});
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

TEST(SsaPhiConstants, ProvesOnlyAgreementFromReachablePredecessors) {
  auto budget = Plenty();
  auto fixture = Join(budget);
  ASSERT_TRUE(fixture);
  auto reachable = ProveSsaReachability(fixture->graph, fixture->sources,
                                        ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts) << static_cast<int>(reachable.reason);
  auto proved = ProveSsaPhiConstants(fixture->graph, *reachable.facts, fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  ASSERT_EQ(proved.facts->constants.size(), 1U);
  const auto fact = proved.facts->constants[0];
  ASSERT_TRUE(fixture->graph.Get(fact.block));
  EXPECT_EQ(fixture->graph.Get(fact.block)->address, 0x10cU);
  EXPECT_EQ(fixture->graph.Get(fact.block)->phis[fact.phi].storage, 8U);
  EXPECT_EQ(fact.value, 11U);
  EXPECT_EQ(ir::ValidateSsaPhiConstantFacts(fixture->graph, *reachable.facts, *proved.facts,
                                            fixture->sources, budget),
            ir::SsaDecline::none);
  auto forged = *proved.facts;
  forged.constants[0].value = 12;
  EXPECT_EQ(ir::ValidateSsaPhiConstantFacts(fixture->graph, *reachable.facts, forged,
                                            fixture->sources, budget),
            ir::SsaDecline::invalid_graph);
}

TEST(SsaPhiConstants, RefusesDisagreementAndStaleOrOpenScope) {
  auto budget = Plenty();
  auto fixture = Join(budget, 22);
  ASSERT_TRUE(fixture);
  auto reachable = ProveSsaReachability(fixture->graph, fixture->sources,
                                        ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts);
  auto proved = ProveSsaPhiConstants(fixture->graph, *reachable.facts, fixture->sources, budget);
  ASSERT_TRUE(proved.facts);
  EXPECT_TRUE(proved.facts->constants.empty());
  auto open = *reachable.facts;
  open.entry_scope = ir::SsaEntryScope::discovered_only;
  EXPECT_EQ(ProveSsaPhiConstants(fixture->graph, open, fixture->sources, budget).reason,
            SsaPhiConstantRefusal::stale_reachability);
  ASSERT_TRUE(fixture->graph.Update(fixture->graph.entries()[0], [](auto&) {}));
  EXPECT_EQ(ProveSsaPhiConstants(fixture->graph, *reachable.facts, fixture->sources, budget).reason,
            SsaPhiConstantRefusal::stale_reachability);
  auto no_return = Join(budget, 11, false);
  ASSERT_TRUE(no_return);
  EXPECT_FALSE(ProveSsaReachability(no_return->graph, no_return->sources,
                                    ir::SsaEntryScope::closed_population, budget)
                   .facts);
}

TEST(SsaPhiConstants, RewritesTheJoinExpressionAndRejectsForgedOrStaleFacts) {
  auto budget = Plenty();
  auto fixture = Join(budget);
  ASSERT_TRUE(fixture);
  auto reachable = ProveSsaReachability(fixture->graph, fixture->sources,
                                        ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts);
  auto proved = ProveSsaPhiConstants(fixture->graph, *reachable.facts, fixture->sources, budget);
  ASSERT_TRUE(proved.facts);
  const auto original_revision = fixture->graph.revision();
  auto result = recovery::ProposeSsaPhiConstantFold(fixture->graph, *reachable.facts, *proved.facts,
                                                    fixture->sources, budget);
  ASSERT_EQ(result.reason, recovery::SsaPhiFoldRefusal::none);
  ASSERT_TRUE(result.provisional);
  ASSERT_EQ(result.journal.size(), 1U);
  EXPECT_EQ(result.journal[0].original.op, ir::Op::add);
  EXPECT_EQ(result.journal[0].value, 12U);
  EXPECT_EQ(result.journal[0].from_revision, original_revision);
  EXPECT_EQ(result.journal[0].to_revision, result.provisional->revision());
  const auto* rewritten = result.provisional->Get(result.journal[0].result_block);
  ASSERT_NE(rewritten, nullptr);
  EXPECT_EQ(rewritten->address, 0x10cU);
  EXPECT_EQ(rewritten->nodes[result.journal[0].node].op, ir::Op::constant);
  EXPECT_EQ(rewritten->nodes[result.journal[0].node].immediate, 12U);
  EXPECT_EQ(ir::ValidateSsa(*result.provisional, budget), ir::SsaDecline::none);
  EXPECT_EQ(fixture->graph.revision(), original_revision);

  auto forged = *proved.facts;
  forged.constants[0].value = 13;
  auto rejected = recovery::ProposeSsaPhiConstantFold(fixture->graph, *reachable.facts, forged,
                                                      fixture->sources, budget);
  EXPECT_EQ(rejected.reason, recovery::SsaPhiFoldRefusal::stale_proof);
  EXPECT_TRUE(rejected.journal.empty());
  rejected = recovery::ProposeSsaPhiConstantFold(*result.provisional, *reachable.facts,
                                                 *proved.facts, fixture->sources, budget);
  EXPECT_EQ(rejected.reason, recovery::SsaPhiFoldRefusal::stale_proof);
  EXPECT_TRUE(rejected.journal.empty());

  auto disagree = Join(budget, 22);
  ASSERT_TRUE(disagree);
  auto reach2 = ProveSsaReachability(disagree->graph, disagree->sources,
                                     ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reach2.facts);
  auto no_fact = ProveSsaPhiConstants(disagree->graph, *reach2.facts, disagree->sources, budget);
  ASSERT_TRUE(no_fact.facts);
  auto unchanged = recovery::ProposeSsaPhiConstantFold(disagree->graph, *reach2.facts,
                                                       *no_fact.facts, disagree->sources, budget);
  EXPECT_FALSE(unchanged.provisional);
  EXPECT_TRUE(unchanged.journal.empty());
}

TEST(SsaPhiConstants, RefusesMalformedPredecessorBeforeIndexingReachability) {
  auto budget = Plenty();
  auto fixture = Join(budget);
  ASSERT_TRUE(fixture);
  auto reachable = ProveSsaReachability(fixture->graph, fixture->sources,
                                        ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts);
  auto proved = ProveSsaPhiConstants(fixture->graph, *reachable.facts, fixture->sources, budget);
  ASSERT_TRUE(proved.facts);
  ASSERT_EQ(proved.facts->constants.size(), 1U);
  const auto join = proved.facts->constants[0].block;
  const auto invalid_slot = static_cast<std::uint32_t>(fixture->graph.slots() + 1);
  ASSERT_TRUE(fixture->graph.Update(join, [&](auto& block) {
    block.phis[proved.facts->constants[0].phi].incoming[0].predecessor.slot = invalid_slot;
  }));
  reachable.facts->graph_revision = fixture->graph.revision();
  proved.facts->graph_revision = fixture->graph.revision();
  EXPECT_EQ(ir::ValidateSsa(fixture->graph, budget), ir::SsaDecline::invalid_graph);
  EXPECT_EQ(ProveSsaPhiConstants(fixture->graph, *reachable.facts, fixture->sources, budget).reason,
            SsaPhiConstantRefusal::invalid_graph);
  EXPECT_EQ(ir::ValidateSsaPhiConstantFacts(fixture->graph, *reachable.facts, *proved.facts,
                                            fixture->sources, budget),
            ir::SsaDecline::invalid_graph);
}

TEST(SsaPhiConstants, BudgetCutsPublishNoPartialProofOrEdit) {
  auto setup = Plenty();
  auto fixture = Join(setup);
  ASSERT_TRUE(fixture);
  auto reachable = ProveSsaReachability(fixture->graph, fixture->sources,
                                        ir::SsaEntryScope::closed_population, setup);
  ASSERT_TRUE(reachable.facts);
  auto producer_budget = Plenty();
  auto proved =
      ProveSsaPhiConstants(fixture->graph, *reachable.facts, fixture->sources, producer_budget);
  ASSERT_TRUE(proved.facts);
  const auto producer_used = producer_budget.used();
  ASSERT_LT(producer_used.work, 10000U);
  for (std::uint64_t cut = 0; cut < producer_used.work; ++cut) {
    Budget limited({cut, producer_used.bytes});
    const auto result =
        ProveSsaPhiConstants(fixture->graph, *reachable.facts, fixture->sources, limited);
    EXPECT_EQ(result.reason, SsaPhiConstantRefusal::resource_limit) << cut;
    EXPECT_FALSE(result.facts) << cut;
  }

  for (const auto cut : {std::uint64_t{0}, producer_used.bytes / 2, producer_used.bytes - 1}) {
    Budget limited({producer_used.work, cut});
    const auto result =
        ProveSsaPhiConstants(fixture->graph, *reachable.facts, fixture->sources, limited);
    EXPECT_EQ(result.reason, SsaPhiConstantRefusal::resource_limit) << cut;
    EXPECT_FALSE(result.facts) << cut;
  }

  auto checker_budget = Plenty();
  ASSERT_EQ(ir::ValidateSsaPhiConstantFacts(fixture->graph, *reachable.facts, *proved.facts,
                                            fixture->sources, checker_budget),
            ir::SsaDecline::none);
  const auto checker_used = checker_budget.used();
  ASSERT_LT(checker_used.work, 10000U);
  for (std::uint64_t cut = 0; cut < checker_used.work; ++cut) {
    Budget limited({cut, checker_used.bytes});
    EXPECT_EQ(ir::ValidateSsaPhiConstantFacts(fixture->graph, *reachable.facts, *proved.facts,
                                              fixture->sources, limited),
              ir::SsaDecline::resource_limit)
        << cut;
  }

  for (const auto cut : {std::uint64_t{0}, checker_used.bytes / 2, checker_used.bytes - 1}) {
    Budget limited({checker_used.work, cut});
    EXPECT_EQ(ir::ValidateSsaPhiConstantFacts(fixture->graph, *reachable.facts, *proved.facts,
                                              fixture->sources, limited),
              ir::SsaDecline::resource_limit)
        << cut;
  }

  auto recovery_budget = Plenty();
  ASSERT_TRUE(recovery::ProposeSsaPhiConstantFold(fixture->graph, *reachable.facts, *proved.facts,
                                                  fixture->sources, recovery_budget)
                  .provisional);
  const auto recovery_used = recovery_budget.used();
  ASSERT_LT(recovery_used.work, 10000U);
  for (std::uint64_t cut = 0; cut < recovery_used.work; ++cut) {
    Budget limited({cut, recovery_used.bytes});
    const auto result = recovery::ProposeSsaPhiConstantFold(
        fixture->graph, *reachable.facts, *proved.facts, fixture->sources, limited);
    EXPECT_EQ(result.reason, recovery::SsaPhiFoldRefusal::resource_limit) << cut;
    EXPECT_FALSE(result.provisional) << cut;
    EXPECT_TRUE(result.journal.empty()) << cut;
  }

  for (const auto cut : {std::uint64_t{0}, recovery_used.bytes / 2, recovery_used.bytes - 1}) {
    Budget limited({recovery_used.work, cut});
    const auto result = recovery::ProposeSsaPhiConstantFold(
        fixture->graph, *reachable.facts, *proved.facts, fixture->sources, limited);
    EXPECT_EQ(result.reason, recovery::SsaPhiFoldRefusal::resource_limit) << cut;
    EXPECT_FALSE(result.provisional) << cut;
    EXPECT_TRUE(result.journal.empty()) << cut;
  }
}
}  // namespace
}  // namespace nyx::analysis
