#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <gtest/gtest.h>

#include "nyx/analysis/cfg.hpp"
#include "nyx/analysis/regions.hpp"
#include "nyx/analysis/ssa/build.hpp"
#include "nyx/analysis/ssa/copy.hpp"
#include "nyx/analysis/ssa/reachability.hpp"
#include "nyx/analysis/unflatten.hpp"
#include "nyx/eval/ssa.hpp"
#include "nyx/recovery/ssa/copy.hpp"
#include "nyx/target/a64/decode.hpp"

namespace nyx {
namespace {

Budget Plenty() { return Budget({10000000, 10000000}); }

struct Fixture {
  ir::SsaGraph graph;
  std::vector<ir::Group> sources;
};

std::optional<Fixture> DecodedWords(std::span<const std::pair<std::uint64_t, std::uint32_t>> words,
                                    Budget& budget) {
  std::vector<analysis::SourceRecord> records;
  std::vector<ir::Group> sources;
  for (const auto [address, word] : words) {
    const std::array<std::uint8_t, 4> bytes{
        static_cast<std::uint8_t>(word), static_cast<std::uint8_t>(word >> 8),
        static_cast<std::uint8_t>(word >> 16), static_cast<std::uint8_t>(word >> 24)};
    auto decoded =
        a64::Decode(address, bytes, budget, {a64::MemoryProfile::concrete_atomic_scalar});
    if (!decoded.group) return {};
    sources.push_back(*decoded.group);
    records.push_back({address,
                       {bytes.begin(), bytes.end()},
                       std::move(*decoded.group),
                       analysis::OpaqueReason::none});
  }

  constexpr std::array<std::uint64_t, 1> entries{0x100};
  auto cfg = analysis::BuildCfg(records, entries, budget);
  if (!cfg.cfg) return {};
  analysis::Regions regions(std::move(*cfg.cfg), {});
  auto unflattened = analysis::Unflatten(regions, {}, budget, {{}, true});
  if (!unflattened.unflattening) return {};
  const std::vector<std::optional<ir::RecoveredPath>> paths(regions.candidates().size());
  auto built = analysis::BuildSsa(regions, *unflattened.unflattening, paths, budget);
  if (!built.graph) return {};
  return Fixture{std::move(*built.graph), std::move(sources)};
}

std::optional<Fixture> DecodedCopy(Budget& budget) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 4> words{
      {{0x100, 0x8b030008}, {0x104, 0x14000002}, {0x10c, 0x8b020101}, {0x110, 0xd65f03c0}}};
  return DecodedWords(words, budget);
}

std::optional<Fixture> DecodedJoin(Budget& budget) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 8> words{{{0x100, 0xb40000a0},
                                                                          {0x104, 0xd2800148},
                                                                          {0x108, 0x91000508},
                                                                          {0x10c, 0x14000004},
                                                                          {0x114, 0xd2800168},
                                                                          {0x118, 0x14000001},
                                                                          {0x11c, 0x91000500},
                                                                          {0x120, 0xd65f03c0}}};
  return DecodedWords(words, budget);
}

TEST(SsaCopy, ProvesAndJournalsSinglePredecessorRead) {
  auto budget = Plenty();
  auto fixture = DecodedCopy(budget);
  ASSERT_TRUE(fixture);
  auto reachable = analysis::ProveSsaReachability(fixture->graph, fixture->sources,
                                                  ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts) << static_cast<int>(reachable.reason);
  auto proved = analysis::ProveSsaPredecessorCopies(fixture->graph, *reachable.facts,
                                                    fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  ASSERT_EQ(proved.facts->copies.size(), 1U);
  const auto fact = proved.facts->copies[0];
  EXPECT_EQ(fixture->graph.Get(fact.block)->address, 0x10cU);
  EXPECT_EQ(fact.source.kind, ir::SsaValueKind::node);
  auto proposal = recovery::ProposeSsaPredecessorCopies(fixture->graph, *reachable.facts,
                                                        *proved.facts, fixture->sources, budget);
  ASSERT_TRUE(proposal.provisional) << static_cast<int>(proposal.reason);
  ASSERT_EQ(proposal.journal.size(), 1U);
  EXPECT_EQ(proposal.journal[0].from_revision, fixture->graph.revision());
  EXPECT_EQ(proposal.journal[0].to_revision, proposal.provisional->revision());
  const auto* block = proposal.provisional->Get(proposal.journal[0].result_block);
  ASSERT_TRUE(block);
  const auto read = std::find_if(block->reads.begin(), block->reads.end(),
                                 [&](const ir::SsaRead& item) { return item.node == fact.read; });
  ASSERT_NE(read, block->reads.end());
  ASSERT_TRUE(read->predecessor_copy);
  EXPECT_TRUE(read->copy_closed_entries);
  EXPECT_EQ(ir::ValidateSsaWithSources(*proposal.provisional, fixture->sources, budget),
            ir::SsaDecline::none);

  const std::array<std::uint8_t, 4096> bytes{};
  const std::array<eval::RegionInput, 1> regions{{{0x300000000, bytes}}};
  for (const std::uint64_t value : {0ULL, 1ULL, 0xffffffffffffffffULL}) {
    auto mapped = eval::Memory::Create(regions, budget);
    ASSERT_TRUE(mapped.memory);
    eval::State state;
    for (unsigned index = 0; index < 36; ++index) {
      const auto number = index == 0    ? value
                          : index == 2  ? 7ULL
                          : index == 30 ? 0x200000100ULL
                          : index == 31 ? 0x300000800ULL
                                        : index + 3ULL;
      auto bits = BitVector::from_u64(index < 32 ? 64 : 1, number, 64, budget);
      ASSERT_TRUE(bits);
      state.cells.push_back({index, std::move(*bits)});
    }

    auto run = eval::ExecuteSsa(*proposal.provisional, fixture->sources,
                                proposal.provisional->entries()[0], state, *mapped.memory, budget,
                                {0x1ffffff00});
    EXPECT_EQ(run.outcome, eval::Outcome::completed);
    EXPECT_EQ(run.stop, eval::SsaStop::returned);
    EXPECT_EQ(state.cells[1].value.word(0), value + 13);
  }
}

TEST(SsaCopy, RefusesOpenStaleAndForgedFactsOrOverlay) {
  auto budget = Plenty();
  auto fixture = DecodedCopy(budget);
  ASSERT_TRUE(fixture);
  auto reachable = analysis::ProveSsaReachability(fixture->graph, fixture->sources,
                                                  ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts);
  auto proved = analysis::ProveSsaPredecessorCopies(fixture->graph, *reachable.facts,
                                                    fixture->sources, budget);
  ASSERT_TRUE(proved.facts);
  auto open = *reachable.facts;
  open.entry_scope = ir::SsaEntryScope::discovered_only;
  EXPECT_EQ(
      analysis::ProveSsaPredecessorCopies(fixture->graph, open, fixture->sources, budget).reason,
      analysis::SsaCopyRefusal::stale_reachability);
  auto forged = *proved.facts;
  ++forged.copies[0].source.index;
  EXPECT_EQ(ir::ValidateSsaPredecessorCopyFacts(fixture->graph, *reachable.facts, forged,
                                                fixture->sources, budget),
            ir::SsaDecline::invalid_graph);
  EXPECT_EQ(recovery::ProposeSsaPredecessorCopies(fixture->graph, *reachable.facts, forged,
                                                  fixture->sources, budget)
                .reason,
            recovery::SsaCopyRefusal::stale_proof);
  auto proposal = recovery::ProposeSsaPredecessorCopies(fixture->graph, *reachable.facts,
                                                        *proved.facts, fixture->sources, budget);
  ASSERT_TRUE(proposal.provisional);
  const auto block = proposal.journal[0].result_block;
  ASSERT_TRUE(proposal.provisional->Update(block, [&](auto& changed) {
    const auto read = std::find_if(
        changed.reads.begin(), changed.reads.end(),
        [&](const ir::SsaRead& item) { return item.node == proposal.journal[0].read; });
    read->predecessor_copy->index++;
  }));
  EXPECT_EQ(ir::ValidateSsa(*proposal.provisional, budget), ir::SsaDecline::invalid_graph);
  EXPECT_EQ(recovery::ProposeSsaPredecessorCopies(fixture->graph, open, *proved.facts,
                                                  fixture->sources, budget)
                .reason,
            recovery::SsaCopyRefusal::stale_proof);
  ASSERT_TRUE(fixture->graph.Update(fixture->graph.entries()[0], [](auto&) {}));
  EXPECT_EQ(recovery::ProposeSsaPredecessorCopies(fixture->graph, *reachable.facts, *proved.facts,
                                                  fixture->sources, budget)
                .reason,
            recovery::SsaCopyRefusal::stale_proof);
}

TEST(SsaCopy, LeavesJoinPhiAndRejectsMalformedValueAndBudgetCuts) {
  auto budget = Plenty();
  auto joined = DecodedJoin(budget);
  ASSERT_TRUE(joined);
  auto reachable = analysis::ProveSsaReachability(joined->graph, joined->sources,
                                                  ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts);
  auto facts =
      analysis::ProveSsaPredecessorCopies(joined->graph, *reachable.facts, joined->sources, budget);
  ASSERT_TRUE(facts.facts);
  std::optional<ir::SsaHandle> join;
  for (std::size_t slot = 0; slot < joined->graph.slots(); ++slot) {
    const auto handle = joined->graph.Handle(slot);
    if (handle && joined->graph.Get(*handle)->address == 0x11c) join = handle;
  }

  ASSERT_TRUE(join);
  EXPECT_TRUE(std::none_of(facts.facts->copies.begin(), facts.facts->copies.end(),
                           [&](const auto& copy) { return copy.block == *join; }));
  const auto* join_block = joined->graph.Get(*join);
  const auto joined_read =
      std::find_if(join_block->reads.begin(), join_block->reads.end(),
                   [&](const auto& read) { return join_block->phis[read.phi].storage == 8; });
  ASSERT_NE(joined_read, join_block->reads.end());
  ASSERT_EQ(join_block->phis[joined_read->phi].incoming.size(), 2U);
  ASSERT_TRUE(joined->graph.Update(*join, [&](auto& block) {
    const auto read = std::find_if(block.reads.begin(), block.reads.end(), [&](const auto& item) {
      return item.node == joined_read->node;
    });
    read->predecessor_copy = block.phis[read->phi].incoming[0].value;
    read->copy_closed_entries = true;
  }));
  EXPECT_EQ(ir::ValidateSsa(joined->graph, budget), ir::SsaDecline::invalid_graph);

  auto straight = DecodedCopy(budget);
  ASSERT_TRUE(straight);
  auto straight_reachable = analysis::ProveSsaReachability(
      straight->graph, straight->sources, ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(straight_reachable.facts);
  auto straight_facts = analysis::ProveSsaPredecessorCopies(
      straight->graph, *straight_reachable.facts, straight->sources, budget);
  ASSERT_TRUE(straight_facts.facts);
  Budget no_work({0, 10000000});
  EXPECT_EQ(analysis::ProveSsaPredecessorCopies(straight->graph, *straight_reachable.facts,
                                                straight->sources, no_work)
                .reason,
            analysis::SsaCopyRefusal::resource_limit);
  Budget no_bytes({10000000, 0});
  auto limited =
      recovery::ProposeSsaPredecessorCopies(straight->graph, *straight_reachable.facts,
                                            *straight_facts.facts, straight->sources, no_bytes);
  EXPECT_EQ(limited.reason, recovery::SsaCopyRefusal::resource_limit);
  EXPECT_FALSE(limited.provisional);
  EXPECT_TRUE(limited.journal.empty());

  auto proposal = recovery::ProposeSsaPredecessorCopies(
      straight->graph, *straight_reachable.facts, *straight_facts.facts, straight->sources, budget);
  ASSERT_TRUE(proposal.provisional);
  const auto source = proposal.journal[0].source;
  ASSERT_TRUE(proposal.provisional->Update(
      source.block, [&](auto& block) { block.nodes[source.index].op = static_cast<ir::Op>(255); }));
  EXPECT_EQ(ir::ValidateSsa(*proposal.provisional, budget), ir::SsaDecline::invalid_graph);
}

int RunProbe(std::uint64_t left, std::uint64_t right, unsigned mode) {
  auto budget = Plenty();
  auto fixture = DecodedCopy(budget);
  if (!fixture) return 2;
  auto reachable = analysis::ProveSsaReachability(fixture->graph, fixture->sources,
                                                  ir::SsaEntryScope::closed_population, budget);
  if (!reachable.facts) return 2;
  auto proved = analysis::ProveSsaPredecessorCopies(fixture->graph, *reachable.facts,
                                                    fixture->sources, budget);
  if (!proved.facts || proved.facts->copies.size() != 1) return 2;
  if (mode == 1) reachable.facts->entry_scope = ir::SsaEntryScope::discovered_only;
  if (mode == 2) ++proved.facts->copies[0].source.index;
  auto proposal = recovery::ProposeSsaPredecessorCopies(fixture->graph, *reachable.facts,
                                                        *proved.facts, fixture->sources, budget);
  if (mode == 1 || mode == 2) {
    if (proposal.provisional || proposal.reason != recovery::SsaCopyRefusal::stale_proof) return 2;
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  if (!proposal.provisional || proposal.journal.size() != 1) return 2;
  if (mode == 3) {
    const auto handle = proposal.journal[0].result_block;
    if (!proposal.provisional->Update(handle,
                                      [&](auto& block) {
                                        const auto read = std::find_if(
                                            block.reads.begin(), block.reads.end(),
                                            [&](const ir::SsaRead& item) {
                                              return item.node == proposal.journal[0].read;
                                            });
                                        read->predecessor_copy->index++;
                                      }) ||
        ir::ValidateSsa(*proposal.provisional, budget) != ir::SsaDecline::invalid_graph)
      return 2;
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  if (mode == 4) {
    const auto source = proposal.journal[0].source;
    if (!proposal.provisional->Update(
            source.block, [&](auto& block) { block.nodes[source.index].op = ir::Op::sub; }) ||
        ir::ValidateSsaWithSources(*proposal.provisional, fixture->sources, budget) !=
            ir::SsaDecline::none)
      return 2;
  }

  const std::array<std::uint8_t, 4096> bytes{};
  const std::array<eval::RegionInput, 1> regions{{{0x300000000, bytes}}};
  auto mapped = eval::Memory::Create(regions, budget);
  if (!mapped.memory) return 2;
  eval::State state;
  for (unsigned index = 0; index < 36; ++index) {
    const auto value = index == 0    ? left
                       : index == 2  ? right
                       : index == 30 ? 0x200000100ULL
                       : index == 31 ? 0x300000800ULL
                       : index >= 32 ? (index == 32 || index == 34)
                                     : index + 3ULL;
    auto bits = BitVector::from_u64(index < 32 ? 64 : 1, value, 64, budget);
    if (!bits) return 2;
    state.cells.push_back({index, std::move(*bits)});
  }

  const auto run =
      eval::ExecuteSsa(*proposal.provisional, fixture->sources, proposal.provisional->entries()[0],
                       state, *mapped.memory, budget, {0x1ffffff00});
  if (run.outcome != eval::Outcome::completed || run.stop != eval::SsaStop::returned) return 2;
  std::cout << "{\"edits\":" << proposal.journal.size() << ",\"registers\":[";
  for (unsigned index = 0; index < 31; ++index) {
    if (index) std::cout << ',';
    std::cout << state.cells[index].value.word(0);
  }

  std::uint64_t nzcv = 0;
  for (unsigned index = 32; index < 36; ++index)
    nzcv |= state.cells[index].value.word(0) << (63 - index);
  std::cout << "],\"nzcv\":" << nzcv << ",\"sp\":" << state.cells[31].value.word(0)
            << ",\"pc\":" << run.runtime_pc << ",\"memory_unchanged\":"
            << (mapped.memory->Regions()[0].bytes == std::vector<std::uint8_t>(4096) ? "true"
                                                                                     : "false")
            << "}\n";
  return 0;
}

}  // namespace
}  // namespace nyx

int main(int argc, char** argv) {
  if (argc == 5 && std::string_view(argv[1]) == "--probe") {
    char* end = nullptr;
    errno = 0;
    const auto left = std::strtoull(argv[2], &end, 0);
    if (errno || !end || *end) return 2;
    errno = 0;
    const auto right = std::strtoull(argv[3], &end, 0);
    if (errno || !end || *end) return 2;
    errno = 0;
    const auto mode = std::strtoul(argv[4], &end, 0);
    if (errno || !end || *end || mode > 4) return 2;
    return nyx::RunProbe(left, right, mode);
  }

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
