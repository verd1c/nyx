#include <array>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <gtest/gtest.h>

#include "nyx/analysis/cfg.hpp"
#include "nyx/analysis/regions.hpp"
#include "nyx/analysis/ssa/build.hpp"
#include "nyx/analysis/ssa/dead_writes.hpp"
#include "nyx/analysis/ssa/liveness.hpp"
#include "nyx/analysis/ssa/reachability.hpp"
#include "nyx/analysis/unflatten.hpp"
#include "nyx/eval/ssa.hpp"
#include "nyx/recovery/ssa/dead_writes.hpp"
#include "nyx/target/a64/abi.hpp"
#include "nyx/target/a64/decode.hpp"

namespace nyx {
namespace {

Budget Plenty() { return Budget({10000000, 10000000}); }

struct Fixture {
  ir::SsaGraph graph;
  std::vector<ir::Group> sources;
};

std::optional<Fixture> Decoded(std::span<const std::pair<std::uint64_t, std::uint32_t>> words,
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

std::optional<Fixture> Overwritten(Budget& budget) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 5> words{{{0x100, 0xd2800168},
                                                                          {0x104, 0x14000003},
                                                                          {0x110, 0xd28002c8},
                                                                          {0x114, 0xaa0803e0},
                                                                          {0x118, 0xd65f03c0}}};
  return Decoded(words, budget);
}

TEST(SsaDeadWrites, RemovesCrossBlockWriteAndItsPureValue) {
  auto budget = Plenty();
  auto fixture = Overwritten(budget);
  ASSERT_TRUE(fixture);
  auto reachable = analysis::ProveSsaReachability(fixture->graph, fixture->sources,
                                                  ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts) << static_cast<int>(reachable.reason);
  auto proved =
      analysis::ProveSsaDeadWrites(fixture->graph, *reachable.facts, fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  ASSERT_EQ(proved.facts->writes.size(), 1U);
  const auto fact = proved.facts->writes[0];
  EXPECT_EQ(fixture->graph.Get(fact.block)->address, 0x100U);
  auto proposal = recovery::ProposeSsaDeadWrites(fixture->graph, *reachable.facts, *proved.facts,
                                                 fixture->sources, budget);
  ASSERT_TRUE(proposal.provisional) << static_cast<int>(proposal.reason);
  ASSERT_EQ(proposal.journal.size(), 1U);
  EXPECT_EQ(proposal.journal[0].from_revision, fixture->graph.revision());
  EXPECT_EQ(proposal.journal[0].to_revision, proposal.provisional->revision());
  EXPECT_EQ(ir::ValidateSsaWithSources(*proposal.provisional, fixture->sources, budget),
            ir::SsaDecline::none);
  auto live = analysis::ProveSsaNodeLiveness(*proposal.provisional, budget);
  ASSERT_TRUE(live.facts) << static_cast<int>(live.reason);
  const auto* block = proposal.provisional->Get(proposal.journal[0].result_block);
  ASSERT_TRUE(block);
  EXPECT_EQ(live.facts->live_nodes[fact.block.slot]
                                  [block->boundaries[fact.boundary].writes[fact.index].value],
            0U);

  const std::array<std::uint8_t, 4096> bytes{};
  const std::array<eval::RegionInput, 1> regions{{{0x300000000, bytes}}};
  auto memory = eval::Memory::Create(regions, budget);
  ASSERT_TRUE(memory.memory);
  eval::State state;
  for (unsigned index = 0; index < 36; ++index) {
    const auto value = index == 30 ? 0x200000100ULL : index == 31 ? 0x300000800ULL : index + 3ULL;
    auto bits = BitVector::from_u64(index < 32 ? 64 : 1, value, 64, budget);
    ASSERT_TRUE(bits);
    state.cells.push_back({index, std::move(*bits)});
  }

  auto run =
      eval::ExecuteSsa(*proposal.provisional, fixture->sources, proposal.provisional->entries()[0],
                       state, *memory.memory, budget, {0x1ffffff00});
  EXPECT_EQ(run.outcome, eval::Outcome::completed);
  EXPECT_EQ(run.stop, eval::SsaStop::returned);
  EXPECT_EQ(state.cells[0].value.word(0), 22U);
  EXPECT_EQ(state.cells[8].value.word(0), 22U);
}

TEST(SsaDeadWrites, RefusesObservedWriteAndForgedOrStaleProof) {
  auto budget = Plenty();
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 5> words{{{0x100, 0xd2800168},
                                                                          {0x104, 0x14000003},
                                                                          {0x110, 0xaa0803e0},
                                                                          {0x114, 0xd28002c8},
                                                                          {0x118, 0xd65f03c0}}};
  auto fixture = Decoded(words, budget);
  ASSERT_TRUE(fixture);
  auto reachable = analysis::ProveSsaReachability(fixture->graph, fixture->sources,
                                                  ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts);
  auto proved =
      analysis::ProveSsaDeadWrites(fixture->graph, *reachable.facts, fixture->sources, budget);
  ASSERT_TRUE(proved.facts);
  EXPECT_TRUE(proved.facts->writes.empty());

  auto positive = Overwritten(budget);
  ASSERT_TRUE(positive);
  auto positive_reachable = analysis::ProveSsaReachability(
      positive->graph, positive->sources, ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(positive_reachable.facts);
  auto positive_facts = analysis::ProveSsaDeadWrites(positive->graph, *positive_reachable.facts,
                                                     positive->sources, budget);
  ASSERT_TRUE(positive_facts.facts);
  ASSERT_EQ(positive_facts.facts->writes.size(), 1U);
  auto forged = *positive_facts.facts;
  forged.writes[0].boundary = 99;
  EXPECT_EQ(recovery::ProposeSsaDeadWrites(positive->graph, *positive_reachable.facts, forged,
                                           positive->sources, budget)
                .reason,
            recovery::SsaDeadWriteRefusal::stale_proof);
  auto open = *positive_reachable.facts;
  open.entry_scope = ir::SsaEntryScope::discovered_only;
  EXPECT_EQ(recovery::ProposeSsaDeadWrites(positive->graph, open, *positive_facts.facts,
                                           positive->sources, budget)
                .reason,
            recovery::SsaDeadWriteRefusal::stale_proof);
  ASSERT_TRUE(positive->graph.Update(positive->graph.entries()[0], [](auto&) {}));
  EXPECT_EQ(recovery::ProposeSsaDeadWrites(positive->graph, *positive_reachable.facts,
                                           *positive_facts.facts, positive->sources, budget)
                .reason,
            recovery::SsaDeadWriteRefusal::stale_proof);
}

TEST(SsaDeadWrites, RefusesPotentialFaultBeforeOverwrite) {
  auto budget = Plenty();
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 4> words{
      {{0x100, 0xd2800168}, {0x104, 0xf9400020}, {0x108, 0xd28002c8}, {0x10c, 0xd65f03c0}}};
  auto fixture = Decoded(words, budget);
  ASSERT_TRUE(fixture);
  auto reachable = analysis::ProveSsaReachability(fixture->graph, fixture->sources,
                                                  ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts);
  auto proved =
      analysis::ProveSsaDeadWrites(fixture->graph, *reachable.facts, fixture->sources, budget);
  ASSERT_TRUE(proved.facts);
  EXPECT_TRUE(proved.facts->writes.empty());
  const auto block = fixture->graph.entries()[0];
  ASSERT_TRUE(fixture->graph.Update(
      block, [](auto& changed) { changed.dead_storage_writes.push_back({0, 0, true}); }));
  EXPECT_EQ(ir::ValidateSsa(fixture->graph, budget), ir::SsaDecline::invalid_graph);
}

TEST(SsaDeadWrites, RefusesPathThatCanLoopBeforeOverwrite) {
  auto budget = Plenty();
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 2> words{
      {{0x100, 0xd2800168}, {0x104, 0x14000000}}};
  auto fixture = Decoded(words, budget);
  ASSERT_TRUE(fixture);
  auto reachable = analysis::ProveSsaReachability(fixture->graph, fixture->sources,
                                                  ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts);
  auto proved =
      analysis::ProveSsaDeadWrites(fixture->graph, *reachable.facts, fixture->sources, budget);
  ASSERT_TRUE(proved.facts);
  EXPECT_TRUE(proved.facts->writes.empty());
}

// Facts proved on `fixture` under an optional observability contract.
std::optional<ir::SsaDeadWriteFacts> Proved(Fixture& fixture,
                                            std::optional<ir::SsaObservability> observed,
                                            Budget& budget) {
  if (observed) fixture.graph.SetObservability(std::move(*observed));
  auto reachable = analysis::ProveSsaReachability(fixture.graph, fixture.sources,
                                                  ir::SsaEntryScope::closed_population, budget);
  if (!reachable.facts) return {};
  return analysis::ProveSsaDeadWrites(fixture.graph, *reachable.facts, fixture.sources, budget)
      .facts;
}

ir::SsaObservability Abi(bool faults_terminal) {
  auto observed = a64::AbiObservability();
  observed.faults_terminal = faults_terminal;
  return observed;
}

TEST(SsaDeadWrites, DeclaredAbiRetiresOnlyWritesAReturnCannotObserve) {
  // mov x9, #11; mov x0, #22; ret
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 3> words{
      {{0x100, 0xd2800169}, {0x104, 0xd28002c0}, {0x108, 0xd65f03c0}}};
  auto budget = Plenty();
  auto undeclared = Decoded(words, budget);
  ASSERT_TRUE(undeclared);
  const auto none = Proved(*undeclared, std::nullopt, budget);
  ASSERT_TRUE(none);
  EXPECT_TRUE(none->writes.empty());
  auto declared = Decoded(words, budget);
  ASSERT_TRUE(declared);
  const auto facts = Proved(*declared, Abi(false), budget);
  ASSERT_TRUE(facts);
  ASSERT_EQ(facts->writes.size(), 1U);
  const auto& block = *declared->graph.Get(facts->writes[0].block);
  EXPECT_EQ(block.boundaries[facts->writes[0].boundary].writes[facts->writes[0].index].storage, 9U);

  // The checker refuses a mark on X0, which the caller reads.
  unsigned checked = 0;
  for (std::uint32_t boundary = 0; boundary < block.boundaries.size(); ++boundary) {
    for (std::uint32_t index = 0; index < block.boundaries[boundary].writes.size(); ++index) {
      if (block.boundaries[boundary].writes[index].storage != 0) continue;
      EXPECT_EQ(ir::CheckSsaDeadStorageWrite(declared->graph,
                                             {facts->writes[0].block, boundary, index}, budget),
                ir::SsaDecline::invalid_graph);
      ++checked;
    }
  }

  EXPECT_EQ(checked, 1U);
}

TEST(SsaDeadWrites, APossibleFaultObservesRegistersUnlessDeclaredTerminal) {
  // mov x9, #11; ldr x0, [x1]; ret
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 3> words{
      {{0x100, 0xd2800169}, {0x104, 0xf9400020}, {0x108, 0xd65f03c0}}};
  auto budget = Plenty();
  auto observed = Decoded(words, budget);
  ASSERT_TRUE(observed);
  const auto kept = Proved(*observed, Abi(false), budget);
  ASSERT_TRUE(kept);
  EXPECT_TRUE(kept->writes.empty());
  auto terminal = Decoded(words, budget);
  ASSERT_TRUE(terminal);
  const auto retired = Proved(*terminal, Abi(true), budget);
  ASSERT_TRUE(retired);
  EXPECT_EQ(retired->writes.size(), 1U);
}

TEST(SsaDeadWrites, UnderTheContractARunThatNeverLeavesObservesNothing) {
  // mov x8, #11; b .
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 2> words{
      {{0x100, 0xd2800168}, {0x104, 0x14000000}}};
  auto budget = Plenty();
  auto fixture = Decoded(words, budget);
  ASSERT_TRUE(fixture);
  const auto facts = Proved(*fixture, Abi(false), budget);
  ASSERT_TRUE(facts);
  EXPECT_EQ(facts->writes.size(), 1U);
}

int RunProbe(std::uint64_t initial, unsigned mode) {
  auto budget = Plenty();
  if (mode == 5) {
    constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 4> words{
        {{0x100, 0xd2800168}, {0x104, 0xf9400020}, {0x108, 0xd28002c8}, {0x10c, 0xd65f03c0}}};
    auto faulting = Decoded(words, budget);
    if (!faulting) return 2;
    auto reachable = analysis::ProveSsaReachability(faulting->graph, faulting->sources,
                                                    ir::SsaEntryScope::closed_population, budget);
    if (!reachable.facts) return 2;
    auto proved =
        analysis::ProveSsaDeadWrites(faulting->graph, *reachable.facts, faulting->sources, budget);
    if (!proved.facts || !proved.facts->writes.empty()) return 2;
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  auto fixture = Overwritten(budget);
  if (!fixture) return 2;
  auto reachable = analysis::ProveSsaReachability(fixture->graph, fixture->sources,
                                                  ir::SsaEntryScope::closed_population, budget);
  if (!reachable.facts) return 2;
  auto proved =
      analysis::ProveSsaDeadWrites(fixture->graph, *reachable.facts, fixture->sources, budget);
  if (!proved.facts || proved.facts->writes.size() != 1) return 2;
  if (mode == 1) reachable.facts->entry_scope = ir::SsaEntryScope::discovered_only;
  if (mode == 2) proved.facts->writes[0].boundary = 99;
  auto proposal = recovery::ProposeSsaDeadWrites(fixture->graph, *reachable.facts, *proved.facts,
                                                 fixture->sources, budget);
  if (mode == 1 || mode == 2) {
    if (proposal.provisional || proposal.reason != recovery::SsaDeadWriteRefusal::stale_proof)
      return 2;
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  if (!proposal.provisional || proposal.journal.size() != 1) return 2;
  if (mode == 3) {
    ir::SsaHandle second{};
    for (std::size_t slot = 0; slot < proposal.provisional->slots(); ++slot) {
      const auto handle = proposal.provisional->Handle(slot);
      if (handle && proposal.provisional->Get(*handle)->address == 0x110) second = *handle;
    }

    if (!second.arena ||
        !proposal.provisional->Update(
            second, [](auto& block) { block.dead_storage_writes.push_back({1, 0, true}); }) ||
        ir::ValidateSsa(*proposal.provisional, budget) != ir::SsaDecline::invalid_graph)
      return 2;
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  if (mode == 4) {
    ir::SsaHandle second{};
    for (std::size_t slot = 0; slot < proposal.provisional->slots(); ++slot) {
      const auto handle = proposal.provisional->Handle(slot);
      if (handle && proposal.provisional->Get(*handle)->address == 0x110) second = *handle;
    }

    bool changed = false;
    if (!second.arena ||
        !proposal.provisional->Update(second,
                                      [&](auto& block) {
                                        for (auto& node : block.nodes)
                                          if (node.op == ir::Op::constant && node.immediate == 22) {
                                            node.immediate = 23;
                                            changed = true;
                                          }
                                      }) ||
        !changed ||
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
    const auto value = index == 8    ? initial
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
  if (argc == 4 && std::string_view(argv[1]) == "--probe") {
    char* end = nullptr;
    errno = 0;
    const auto initial = std::strtoull(argv[2], &end, 0);
    if (errno || !end || *end) return 2;
    errno = 0;
    const auto mode = std::strtoul(argv[3], &end, 0);
    if (errno || !end || *end || mode > 5) return 2;
    return nyx::RunProbe(initial, mode);
  }

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
