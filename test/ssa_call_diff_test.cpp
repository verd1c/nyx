#include <array>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <gtest/gtest.h>

#include "nyx/analysis/cfg.hpp"
#include "nyx/analysis/regions.hpp"
#include "nyx/analysis/ssa/build.hpp"
#include "nyx/analysis/unflatten.hpp"
#include "nyx/eval/ssa.hpp"
#include "nyx/recovery/mba.hpp"
#include "nyx/target/a64/decode.hpp"

namespace nyx {
namespace {

constexpr std::uint64_t kCode = 0x200000000;
constexpr std::uint64_t kLanding = kCode + 0x100;
constexpr std::uint64_t kBias = kCode - 0x100;
constexpr std::uint64_t kCallee = kBias + 0x300;
constexpr std::uint64_t kContinuation = kBias + 0x118;

Budget Plenty() { return Budget({10000000, 10000000}); }

struct Fixture {
  ir::SsaGraph graph;
  std::vector<ir::Group> sources;
};

std::optional<Fixture> DecodedCall(Budget& budget) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 9> words{{{0x100, 0xaa1e03f3},
                                                                          {0x104, 0xca010002},
                                                                          {0x108, 0x8a010003},
                                                                          {0x10c, 0x8b030063},
                                                                          {0x110, 0x8b030040},
                                                                          {0x114, 0x9400007b},
                                                                          {0x118, 0x91000400},
                                                                          {0x11c, 0xaa1303fe},
                                                                          {0x120, 0xd65f03c0}}};
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

TEST(SsaCallDiff, RewritesBeforeExternalCallAndChecksContinuation) {
  auto budget = Plenty();
  auto fixture = DecodedCall(budget);
  ASSERT_TRUE(fixture);
  auto proposal = recovery::ProposeLinearMba(fixture->graph, budget);
  ASSERT_TRUE(proposal.provisional) << static_cast<int>(proposal.reason);
  EXPECT_FALSE(proposal.journal.empty());
  EXPECT_EQ(ir::ValidateSsaWithSources(*proposal.provisional, fixture->sources, budget),
            ir::SsaDecline::none);
}

int RunProbe(std::uint64_t left, std::uint64_t right, unsigned mode) {
  auto budget = Plenty();
  auto fixture = DecodedCall(budget);
  if (!fixture) return 2;
  auto proposal = recovery::ProposeLinearMba(fixture->graph, budget);
  if (!proposal.provisional || proposal.journal.empty()) return 2;
  if (mode == 1) {
    bool changed = false;
    for (std::size_t slot = 0; slot < proposal.provisional->slots(); ++slot) {
      const auto handle = proposal.provisional->Handle(slot);
      if (!handle) continue;
      const auto* block = proposal.provisional->Get(*handle);
      if (block->address != 0x118) continue;
      if (!proposal.provisional->Update(*handle, [&](auto& edited) {
            for (auto& node : edited.nodes)
              if (node.op == ir::Op::constant && node.immediate == 1) {
                node.immediate = 2;
                changed = true;
              }
          }))
        return 2;
    }

    if (!changed || ir::ValidateSsaWithSources(*proposal.provisional, fixture->sources, budget) !=
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
                       : index == 1  ? right
                       : index == 4  ? 0x300000400ULL
                       : index == 30 ? kLanding
                       : index == 31 ? 0x300000800ULL
                       : index >= 32 ? (index == 32 || index == 34)
                                     : index + 3ULL;
    auto bits = BitVector::from_u64(index < 32 ? 64 : 1, value, 64, budget);
    if (!bits) return 2;
    state.cells.push_back({index, std::move(*bits)});
  }

  unsigned calls = 0;
  const auto callback = [&](std::uint64_t target, eval::State& live, eval::Memory& memory,
                            Budget& work) -> eval::SsaCalleeOutcome {
    ++calls;
    if (target != kCallee || live.cells[30].value.word(0) != kContinuation)
      return {eval::Outcome::unsupported, {}};
    auto result =
        BitVector::from_u64(64, live.cells[0].value.word(0) + (mode == 2 ? 8 : 7), 64, work);
    if (!result) return {eval::Outcome::resource_limit, {}};
    live.cells[0].value = std::move(*result);
    if (mode != 5) {
      std::array<std::uint8_t, 8> bytes{};
      for (unsigned index = 0; index < bytes.size(); ++index)
        bytes[index] = static_cast<std::uint8_t>(live.cells[0].value.word(0) >> (8 * index));
      auto transaction = memory.Begin(work);
      if (transaction.Write(live.cells[4].value.word(0), bytes).status != eval::MemoryStatus::ok)
        return {eval::Outcome::unsupported, {}};
      transaction.Commit();
    }

    return {eval::Outcome::completed, kContinuation + (mode == 3 ? 4 : 0)};
  };

  const auto run = eval::ExecuteSsa(
      *proposal.provisional, fixture->sources, proposal.provisional->entries()[0], state,
      *mapped.memory, budget, {kBias}, mode == 4 ? eval::SsaCallee{} : eval::SsaCallee{callback});
  if (mode == 3 || mode == 4) {
    if (run.outcome != eval::Outcome::unsupported || run.stop != eval::SsaStop::unresolved)
      return 2;
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  if (run.outcome != eval::Outcome::completed || run.stop != eval::SsaStop::returned || calls != 1)
    return 2;
  std::cout << "{\"edits\":" << proposal.journal.size() << ",\"calls\":" << calls
            << ",\"registers\":[";
  for (unsigned index = 0; index < 31; ++index) {
    if (index) std::cout << ',';
    std::cout << state.cells[index].value.word(0);
  }

  std::uint64_t nzcv = 0;
  for (unsigned index = 32; index < 36; ++index)
    nzcv |= state.cells[index].value.word(0) << (63 - index);
  std::cout << "],\"nzcv\":" << nzcv << ",\"sp\":" << state.cells[31].value.word(0)
            << ",\"pc\":" << run.runtime_pc << ",\"memory_hex\":\"";
  constexpr char hex[] = "0123456789abcdef";
  for (const auto byte : mapped.memory->Regions()[0].bytes)
    std::cout << hex[byte >> 4] << hex[byte & 15];
  std::cout << "\"}\n";
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
    if (errno || !end || *end || mode > 5) return 2;
    return nyx::RunProbe(left, right, mode);
  }

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
