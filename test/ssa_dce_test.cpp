#include <gtest/gtest.h>

#include "nyx/analysis/ssa/reachability.hpp"
#include "nyx/passes/pipeline.hpp"
#include "nyx/recovery/ssa/dce.hpp"

namespace nyx::recovery {
namespace {
Budget Plenty() { return Budget({1000000, 10000000}); }

std::vector<std::uint8_t> Bytes(std::uint32_t word) {
  return {static_cast<std::uint8_t>(word), static_cast<std::uint8_t>(word >> 8),
          static_cast<std::uint8_t>(word >> 16), static_cast<std::uint8_t>(word >> 24)};
}

std::vector<ir::Group> Sources() {
  std::vector<ir::Group> sources;
  sources.emplace_back(0x100, Bytes(0xd65f03c0),
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 30}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}});
  sources.emplace_back(0x110, Bytes(0x17fffffc),
                       std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x100}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}});
  sources.emplace_back(0x120, Bytes(0x14000000),
                       std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x120}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}});
  return sources;
}

ir::SsaGraph Graph() {
  ir::SsaGraph graph;
  ir::SsaBlock entry{};
  entry.address = 0x100;
  entry.source_groups = {0x100};
  entry.source_bytes = {Bytes(0xd65f03c0)};
  entry.original_sources = {0};
  entry.nodes = {{ir::Op::read, 64, {}, 0, 30}};
  entry.boundaries = {{0, 1, {}, ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}}}};
  entry.phis = {{30, 64, true, {}}};
  entry.clobbers = {0};
  entry.edges = {{ir::SsaEdgeKind::return_,
                  ir::SsaTargetKind::unknown,
                  0,
                  {},
                  {},
                  {},
                  {.return_leaves = true}}};
  const auto start = graph.Add(std::move(entry));
  ir::SsaBlock incoming{};
  incoming.address = 0x110;
  incoming.source_groups = {0x110};
  incoming.source_bytes = {Bytes(0x17fffffc)};
  incoming.original_sources = {1};
  incoming.nodes = {{ir::Op::image_address, 64, {}, 0x100}};
  incoming.boundaries = {{0, 1, {}, ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}}}};
  incoming.phis = {{30, 64, false, {}}};
  incoming.clobbers = {0};
  const auto dead = graph.Add(std::move(incoming));
  ir::SsaBlock loop{};
  loop.address = 0x120;
  loop.source_groups = {0x120};
  loop.source_bytes = {Bytes(0x14000000)};
  loop.original_sources = {2};
  loop.nodes = {{ir::Op::image_address, 64, {}, 0x120}};
  loop.boundaries = {{0, 1, {}, ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}}}};
  const auto isolated = graph.Add(std::move(loop));
  graph.Update(start, [&](auto& block) {
    block.phis[0].incoming = {{dead, {ir::SsaValueKind::phi, dead, 0}}};
    block.reads = {{0, 0, {ir::SsaValueKind::phi, start, 0}}};
    block.exits = {{30, {ir::SsaValueKind::phi, start, 0}}};
  });
  graph.Update(dead, [&](auto& block) {
    block.exits = {{30, {ir::SsaValueKind::phi, dead, 0}}};
    block.edges = {
        {ir::SsaEdgeKind::branch, ir::SsaTargetKind::image_location, 0x100, start, {}, {}, {}}};
  });
  graph.Update(isolated, [&](auto& block) {
    block.edges = {
        {ir::SsaEdgeKind::branch, ir::SsaTargetKind::image_location, 0x120, isolated, {}, {}, {}}};
  });
  graph.SetEntries({start});
  return graph;
}

TEST(SsaDce, DeletesDisconnectedPredecessorAndUnreachableLoop) {
  auto original = Graph();
  const auto sources = Sources();
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(original, budget), ir::SsaDecline::none);
  auto proof = analysis::ProveSsaReachability(original, sources,
                                              ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(proof.facts) << static_cast<int>(proof.reason);
  const auto revision = original.revision();
  auto result = ProposeUnreachableBlocks(original, *proof.facts, sources, budget);
  ASSERT_EQ(result.reason, SsaDceRefusal::none);
  ASSERT_TRUE(result.provisional);
  EXPECT_EQ(result.entry_scope, ir::SsaEntryScope::closed_population);
  EXPECT_EQ(result.journal.size(), 3U);
  EXPECT_EQ(original.revision(), revision);
  EXPECT_EQ(result.provisional->slots(), 3U);
  EXPECT_FALSE(result.provisional->Handle(1));
  EXPECT_FALSE(result.provisional->Handle(2));
  EXPECT_FALSE(result.provisional->Get(original.entries()[0]));
  const auto entry = result.provisional->entries()[0];
  ASSERT_TRUE(result.provisional->Get(entry));
  EXPECT_TRUE(result.provisional->Get(entry)->phis[0].incoming.empty());
  EXPECT_EQ(ir::ValidateSsa(*result.provisional, budget), ir::SsaDecline::none);
  for (const auto& edit : result.journal) {
    EXPECT_EQ(edit.from_revision, revision);
    EXPECT_EQ(edit.to_revision, result.provisional->revision());
  }

  auto new_proof = analysis::ProveSsaReachability(*result.provisional, sources,
                                                  ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(new_proof.facts);
  auto second = ProposeUnreachableBlocks(*result.provisional, *new_proof.facts, sources, budget);
  EXPECT_FALSE(second.provisional);
  EXPECT_TRUE(second.journal.empty());
  auto stale = ProposeUnreachableBlocks(*result.provisional, *proof.facts, sources, budget);
  EXPECT_EQ(stale.reason, SsaDceRefusal::stale_proof);
  EXPECT_TRUE(stale.journal.empty());
  auto corrupt = *proof.facts;
  corrupt.reachable[0] = 0;
  auto invalid = ProposeUnreachableBlocks(original, corrupt, sources, budget);
  EXPECT_EQ(invalid.reason, SsaDceRefusal::stale_proof);
  EXPECT_TRUE(invalid.journal.empty());
}

TEST(SsaDce, CliPublishesBlockAndPhiJournalOnlyWithReturnScope) {
  auto graph = Graph();
  const auto sources = Sources();
  auto budget = Plenty();
  passes::SsaDeclarations declared;
  declared.closed_entries = true;
  declared.return_leaves = true;
  const auto published = passes::RunSsaPipeline(graph, declared, sources, {}, budget).result;
  ASSERT_TRUE(published);
  ASSERT_TRUE(published->provisional);
  EXPECT_EQ(ir::ValidateSsaWithSources(*published->provisional, sources, budget),
            ir::SsaDecline::none);
  EXPECT_EQ(published->provisional->entries().size(), 1U);
  EXPECT_FALSE(published->provisional->Handle(1));
  EXPECT_NE(published->body.find("\"pass\":\"unreachable_blocks\",\"outcome\":\"proposed\""),
            std::string::npos);
  EXPECT_NE(published->body.find("\"kind\":\"pruned_storage_phi\""), std::string::npos);
  EXPECT_NE(published->body.find("\"kind\":\"removed_block\""), std::string::npos);
  EXPECT_NE(published->head.find("\"return_leaves\":true"), std::string::npos);

  // Only the pruned phi changes executed code; removed blocks are not witnessed.
  const auto count = [&](std::string_view text) {
    std::size_t found = 0;
    for (auto at = published->body.find(text); at != std::string::npos;
         at = published->body.find(text, at + 1))
      ++found;
    return found;
  };

  ASSERT_EQ(published->stages[0].outcome, passes::SsaStageOutcome::proposed);
  EXPECT_EQ(published->stages[0].executable_edits, count("\"kind\":\"pruned_storage_phi\""));
  EXPECT_EQ(count("\"kind\":\"removed_block\""), 2U);
  EXPECT_NE(
      published->text.find("ssa revision " + std::to_string(published->provisional->revision())),
      std::string::npos);
  declared.return_leaves = false;
  auto refused_budget = Plenty();
  const auto refused = passes::RunSsaPipeline(graph, declared, sources, {}, refused_budget).result;
  ASSERT_TRUE(refused);
  EXPECT_NE(refused->body.find("\"pass\":\"unreachable_blocks\",\"outcome\":\"not_run\","
                               "\"reason\":\"no_return_leaves_declaration\""),
            std::string::npos);
  EXPECT_EQ(refused->body.find("\"kind\":\"removed_block\""), std::string::npos);
}

TEST(SsaDce, RefusesOpenEntriesUnknownSuccessorsAndReenteringReturns) {
  auto graph = Graph();
  const auto sources = Sources();
  auto budget = Plenty();
  auto open =
      analysis::ProveSsaReachability(graph, sources, ir::SsaEntryScope::discovered_only, budget);
  EXPECT_EQ(open.reason, analysis::SsaReachabilityRefusal::open_entries);
  EXPECT_FALSE(open.facts);
  ASSERT_TRUE(graph.Update(graph.entries()[0], [](auto& block) {
    block.edges = {{ir::SsaEdgeKind::opaque_unknown,
                    ir::SsaTargetKind::unknown,
                    0,
                    {},
                    {},
                    {},
                    {.unresolved_target = true}}};
  }));
  auto unknown =
      analysis::ProveSsaReachability(graph, sources, ir::SsaEntryScope::closed_population, budget);
  EXPECT_EQ(unknown.reason, analysis::SsaReachabilityRefusal::incomplete_successors);
  EXPECT_FALSE(unknown.facts);
  ASSERT_TRUE(graph.Update(graph.entries()[0], [](auto& block) {
    block.edges = {{ir::SsaEdgeKind::return_, ir::SsaTargetKind::unknown, 0, {}, {}, {}, {}}};
  }));
  auto ret =
      analysis::ProveSsaReachability(graph, sources, ir::SsaEntryScope::closed_population, budget);
  EXPECT_EQ(ret.reason, analysis::SsaReachabilityRefusal::incomplete_successors);
  EXPECT_FALSE(ret.facts);
  ASSERT_TRUE(graph.Update(graph.entries()[0], [](auto& block) { block.edges.clear(); }));
  EXPECT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  auto missing =
      analysis::ProveSsaReachability(graph, sources, ir::SsaEntryScope::closed_population, budget);
  EXPECT_EQ(missing.reason, analysis::SsaReachabilityRefusal::incomplete_successors);
  EXPECT_FALSE(missing.facts);
}

TEST(SsaDce, BudgetRefusalHasNoPartialJournal) {
  auto graph = Graph();
  const auto sources = Sources();
  auto proof_budget = Plenty();
  auto proof = analysis::ProveSsaReachability(graph, sources, ir::SsaEntryScope::closed_population,
                                              proof_budget);
  ASSERT_TRUE(proof.facts);
  auto full = Plenty();
  ASSERT_TRUE(ProposeUnreachableBlocks(graph, *proof.facts, sources, full).provisional);
  const auto used = full.used();
  ASSERT_LT(used.work, 10000U);
  for (std::uint64_t cut = 0; cut < used.work; ++cut) {
    Budget limited({cut, used.bytes});
    auto result = ProposeUnreachableBlocks(graph, *proof.facts, sources, limited);
    EXPECT_EQ(result.reason, SsaDceRefusal::resource_limit) << cut;
    EXPECT_FALSE(result.provisional) << cut;
    EXPECT_TRUE(result.journal.empty()) << cut;
  }
}

TEST(SsaDce, RefusesForgedReachabilityAndChangedSourceBytes) {
  auto graph = Graph();
  const auto sources = Sources();
  auto budget = Plenty();
  auto changed_target = *graph.Get(*graph.Handle(1));
  changed_target.nodes[0].immediate = 0x120;
  EXPECT_EQ(ir::ValidateSsaSourceBinding(changed_target, sources, budget),
            ir::SsaDecline::invalid_graph);
  auto malformed = *graph.Get(graph.entries()[0]);
  malformed.source_groups.clear();
  EXPECT_EQ(ir::ValidateSsaSourceBinding(malformed, sources, budget),
            ir::SsaDecline::invalid_graph);
  ASSERT_TRUE(graph.Update(graph.entries()[0], [](auto& block) { block.edges.clear(); }));
  ir::SsaReachabilityFacts forged{
      graph.arena(), graph.revision(), ir::SsaEntryScope::closed_population, {1, 0, 0}};
  auto missing = ProposeUnreachableBlocks(graph, forged, sources, budget);
  EXPECT_EQ(missing.reason, SsaDceRefusal::stale_proof);
  EXPECT_FALSE(missing.provisional);
  EXPECT_TRUE(missing.journal.empty());

  graph = Graph();
  ASSERT_TRUE(graph.Update(graph.entries()[0],
                           [](auto& block) { block.source_bytes = {Bytes(0x14000004)}; }));
  auto changed =
      analysis::ProveSsaReachability(graph, sources, ir::SsaEntryScope::closed_population, budget);
  EXPECT_EQ(changed.reason, analysis::SsaReachabilityRefusal::invalid_graph);
  EXPECT_FALSE(changed.facts);
  forged = {graph.arena(), graph.revision(), ir::SsaEntryScope::closed_population, {1, 0, 0}};
  auto refused = ProposeUnreachableBlocks(graph, forged, sources, budget);
  EXPECT_EQ(refused.reason, SsaDceRefusal::stale_proof);
  EXPECT_FALSE(refused.provisional);
}

}  // namespace
}  // namespace nyx::recovery
