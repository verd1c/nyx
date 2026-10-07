#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <gtest/gtest.h>

#include "nyx/analysis/cfg.hpp"
#include "nyx/analysis/frame_slots.hpp"
#include "nyx/analysis/regions.hpp"
#include "nyx/analysis/ssa/build.hpp"
#include "nyx/analysis/ssa/constants.hpp"
#include "nyx/analysis/ssa/image_addresses.hpp"
#include "nyx/analysis/ssa/index_bound.hpp"
#include "nyx/analysis/ssa/liveness.hpp"
#include "nyx/analysis/ssa/phi_constants.hpp"
#include "nyx/analysis/ssa/reachability.hpp"
#include "nyx/analysis/ssa/sccp.hpp"
#include "nyx/analysis/ssa/table_address.hpp"
#include "nyx/analysis/unflatten.hpp"
#include "nyx/eval/ssa.hpp"
#include "nyx/recovery/constant_load.hpp"
#include "nyx/recovery/dead_state.hpp"
#include "nyx/recovery/frame_promotion.hpp"
#include "nyx/recovery/mba.hpp"
#include "nyx/recovery/ssa/branch_retirement.hpp"
#include "nyx/recovery/ssa/constants.hpp"
#include "nyx/recovery/ssa/dce.hpp"
#include "nyx/recovery/ssa/phi_constants.hpp"
#include "nyx/recovery/ssa/pure_dce.hpp"
#include "nyx/recovery/ssa/simplify.hpp"
#include "nyx/target/a64/decode.hpp"

namespace nyx::eval {
namespace {

constexpr std::uint64_t kCode = 0x200000000;
constexpr std::uint64_t kLanding = kCode + 0x100;
constexpr std::uint64_t kBias = kCode - 0x100;

std::vector<std::uint8_t> Bytes(std::uint32_t word) {
  return {static_cast<std::uint8_t>(word), static_cast<std::uint8_t>(word >> 8),
          static_cast<std::uint8_t>(word >> 16), static_cast<std::uint8_t>(word >> 24)};
}

struct Fixture {
  ir::SsaGraph graph;
  ir::SsaHandle entry;
  std::vector<ir::Group> sources;
};

Fixture BranchingFunction() {
  ir::SsaGraph graph;
  ir::SsaBlock entry{};
  entry.address = 0x100;
  entry.source_groups = {0x100};
  entry.source_bytes = {Bytes(0xb4000080)};
  entry.original_sources = {0};
  entry.nodes = {{ir::Op::read, 64, {}, 0, 0},
                 {ir::Op::constant, 64, {}, 0},
                 {ir::Op::equal, 1, {0, 1}},
                 {ir::Op::image_address, 64, {}, 0x110},
                 {ir::Op::image_address, 64, {}, 0x104}};
  entry.boundaries = {{0, 5, {}, ir::Transfer{ir::TransferKind::conditional, 3, 2, 4, {}}}};
  entry.phis = {{0, 64, true, {}}, {30, 64, true, {}}};
  entry.clobbers = {0, 0};
  const auto start = graph.Add(std::move(entry));

  const auto arm = [&](std::uint64_t address, std::uint64_t value, std::uint32_t source) {
    ir::SsaBlock block{};
    block.address = address;
    block.source_groups = {address, address + 4};
    block.source_bytes = {Bytes(0xd2800000 | value << 5), Bytes(0xd65f03c0)};
    block.original_sources = {source, source + 1};
    block.nodes = {{ir::Op::constant, 64, {}, value}, {ir::Op::read, 64, {}, 0, 30}};
    block.boundaries = {{0, 1, {{0, 0}}, {}},
                        {1, 1, {}, ir::Transfer{ir::TransferKind::return_, 1, {}, {}, {}}}};
    block.phis = {{0, 64, false, {}}, {30, 64, false, {}}};
    block.clobbers = {0, 0};
    return graph.Add(std::move(block));
  };

  const auto left = arm(0x110, 11, 3);
  const auto right = arm(0x104, 22, 1);
  graph.Update(start, [&](auto& block) {
    block.reads = {{0, 0, {ir::SsaValueKind::phi, start, 0}}};
    block.exits = {{0, {ir::SsaValueKind::phi, start, 0}}, {30, {ir::SsaValueKind::phi, start, 1}}};
    block.edges = {
        {ir::SsaEdgeKind::branch, ir::SsaTargetKind::image_location, 0x110, left, 2, true, {}},
        {ir::SsaEdgeKind::branch, ir::SsaTargetKind::image_location, 0x104, right, 2, false, {}}};
  });
  for (const auto handle : {left, right}) {
    graph.Update(handle, [&](auto& block) {
      block.phis[0].incoming = {{start, {ir::SsaValueKind::phi, start, 0}}};
      block.phis[1].incoming = {{start, {ir::SsaValueKind::phi, start, 1}}};
      block.reads = {{1, 1, {ir::SsaValueKind::phi, handle, 1}}};
      block.exits = {{0, {ir::SsaValueKind::node, handle, 0}},
                     {30, {ir::SsaValueKind::phi, handle, 1}}};
      block.edges = {{ir::SsaEdgeKind::return_, ir::SsaTargetKind::unknown, 0, {}, {}, {}, {}}};
    });
  }

  graph.SetEntries({start});
  std::vector<ir::Group> sources;
  sources.emplace_back(0x100, Bytes(0xb4000080),
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 0},
                                             {ir::Op::constant, 64, {}, 0},
                                             {ir::Op::equal, 1, {0, 1}},
                                             {ir::Op::image_address, 64, {}, 0x110},
                                             {ir::Op::image_address, 64, {}, 0x104}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::conditional, 3, 2, 4, {}});
  for (const auto [address, value] : {std::pair{0x104U, 22U}, std::pair{0x110U, 11U}}) {
    sources.emplace_back(address, Bytes(0xd2800000 | value << 5),
                         std::vector<ir::Node>{{ir::Op::constant, 64, {}, value}},
                         std::vector<ir::Write>{{0, 0}});
    sources.emplace_back(address + 4, Bytes(0xd65f03c0),
                         std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 30}},
                         std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                         ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}});
  }

  return {std::move(graph), start, std::move(sources)};
}

std::optional<Fixture> DecodedWords(std::span<const std::pair<std::uint64_t, std::uint32_t>> words,
                                    Budget& budget, std::uint64_t entry_address = 0x100) {
  std::vector<analysis::SourceRecord> records;
  std::vector<ir::Group> groups;
  for (const auto [address, word] : words) {
    const std::array<std::uint8_t, 4> bytes{
        static_cast<std::uint8_t>(word), static_cast<std::uint8_t>(word >> 8),
        static_cast<std::uint8_t>(word >> 16), static_cast<std::uint8_t>(word >> 24)};
    auto decoded =
        a64::Decode(address, bytes, budget, {a64::MemoryProfile::concrete_atomic_scalar});
    if (!decoded.group) return {};
    groups.push_back(*decoded.group);
    records.push_back({address,
                       {bytes.begin(), bytes.end()},
                       std::move(*decoded.group),
                       analysis::OpaqueReason::none});
  }

  const std::array<std::uint64_t, 1> entries{entry_address};
  auto cfg = analysis::BuildCfg(records, entries, budget);
  if (!cfg.cfg) return {};
  analysis::Regions regions(std::move(*cfg.cfg), {});
  auto unflattened = analysis::Unflatten(regions, {}, budget, {{}, true});
  if (!unflattened.unflattening) return {};
  const std::vector<std::optional<ir::RecoveredPath>> paths(regions.candidates().size());
  auto built = analysis::BuildSsa(regions, *unflattened.unflattening, paths, budget);
  if (!built.graph || built.graph->entries().size() != 1) return {};
  const auto entry = built.graph->entries()[0];
  return Fixture{std::move(*built.graph), entry, std::move(groups)};
}

std::optional<Fixture> DecodedFunction(Budget& budget) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 5> words{{{0x100, 0xb4000080},
                                                                          {0x104, 0xd28002c0},
                                                                          {0x108, 0xd65f03c0},
                                                                          {0x110, 0xd2800160},
                                                                          {0x114, 0xd65f03c0}}};
  return DecodedWords(words, budget);
}

std::optional<Fixture> DecodedPhiConstantFunction(Budget& budget) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 7> words{{{0x100, 0xb4000080},
                                                                          {0x104, 0xd2800168},
                                                                          {0x108, 0x14000004},
                                                                          {0x110, 0xd2800168},
                                                                          {0x114, 0x14000001},
                                                                          {0x118, 0x91000500},
                                                                          {0x11c, 0xd65f03c0}}};
  return DecodedWords(words, budget);
}

std::optional<Fixture> DecodedComputedPhiFunction(Budget& budget) {
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

std::optional<Fixture> DecodedSccpFunction(Budget& budget) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 8> words{{{0x100, 0xb40000bf},
                                                                          {0x104, 0xd2800148},
                                                                          {0x108, 0x91000508},
                                                                          {0x10c, 0x14000004},
                                                                          {0x114, 0xd28002c8},
                                                                          {0x118, 0x14000001},
                                                                          {0x11c, 0x91000500},
                                                                          {0x120, 0xd65f03c0}}};
  return DecodedWords(words, budget);
}

std::optional<Fixture> DecodedReadConditionFunction(Budget& budget) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 8> words{{{0x100, 0xd2800000},
                                                                          {0x104, 0x14000002},
                                                                          {0x10c, 0xaa0003e1},
                                                                          {0x110, 0xb4000060},
                                                                          {0x114, 0xd2800160},
                                                                          {0x118, 0xd65f03c0},
                                                                          {0x11c, 0xd28002c0},
                                                                          {0x120, 0xd65f03c0}}};
  return DecodedWords(words, budget);
}

std::optional<Fixture> DecodedMbaFunction(Budget& budget) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 5> words{{{0x100, 0xca010002},
                                                                          {0x104, 0x8a010003},
                                                                          {0x108, 0x8b030063},
                                                                          {0x10c, 0x8b030040},
                                                                          {0x110, 0xd65f03c0}}};
  return DecodedWords(words, budget);
}

std::optional<Fixture> DecodedCanonicalMbaFunction(Budget& budget) {
  constexpr std::array<std::uint32_t, 15> code{{0xaa2003e3, 0xaa2103e4, 0xaa2203e5, 0x8a040066,
                                                0x8a0200c6, 0x8a010067, 0x8a0500e7, 0x8a040008,
                                                0x8a050108, 0x8a010009, 0x8a020129, 0xaa0700c6,
                                                0xaa090108, 0xaa0800c0, 0xd65f03c0}};
  std::array<std::pair<std::uint64_t, std::uint32_t>, code.size()> words{};
  for (std::size_t i = 0; i < words.size(); ++i) words[i] = {0x100 + i * 4, code[i]};
  return DecodedWords(words, budget);
}

std::optional<Fixture> DecodedAffineMbaFunction(Budget& budget) {
  constexpr std::array<std::uint32_t, 11> code{{0xca010003, 0xca020024, 0x8a010005, 0x8a020026,
                                                0x8b0500a7, 0x8b0600c8, 0x8b040069, 0x8b07012a,
                                                0x8b08014b, 0xcb010160, 0xd65f03c0}};
  std::array<std::pair<std::uint64_t, std::uint32_t>, code.size()> words{};
  for (std::size_t i = 0; i < words.size(); ++i) words[i] = {0x100 + i * 4, code[i]};
  return DecodedWords(words, budget);
}

// 2*x0 + x1 - x2 through XOR/AND/BIC terms, so the rewrite needs a doubled
// and a subtracted free value.
std::optional<Fixture> DecodedSignedAffineMbaFunction(Budget& budget) {
  constexpr std::array<std::uint32_t, 10> code{{0xca010003, 0x8a010004, 0xca020005, 0x8a200046,
                                                0x8b040067, 0x8b0400e7, 0x8b0500e7, 0xcb0600e7,
                                                0xcb0600e0, 0xd65f03c0}};
  std::array<std::pair<std::uint64_t, std::uint32_t>, code.size()> words{};
  for (std::size_t i = 0; i < words.size(); ++i) words[i] = {0x100 + i * 4, code[i]};
  return DecodedWords(words, budget);
}

std::optional<Fixture> DecodedPureDceFunction(Budget& budget) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 3> words{
      {{0x100, 0x8b010000}, {0x104, 0x8b01001f}, {0x108, 0xd65f03c0}}};
  return DecodedWords(words, budget);
}

std::optional<Fixture> DecodedFrameDeadFunction(Budget& budget, bool escape) {
  if (escape) {
    constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 3> words{
        {{0x100, 0xd10043e1}, {0x104, 0xaa0103e0}, {0x108, 0xd65f03c0}}};
    return DecodedWords(words, budget);
  }

  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 3> words{
      {{0x100, 0xf81f03e0}, {0x104, 0xf85f03ff}, {0x108, 0xd65f03c0}}};
  return DecodedWords(words, budget);
}

std::optional<Fixture> DecodedFramePromotionFunction(Budget& budget) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 8> words{{{0x100, 0xb4000080},
                                                                          {0x104, 0xd28002c1},
                                                                          {0x108, 0xf81f03e1},
                                                                          {0x10c, 0x14000003},
                                                                          {0x110, 0xd2800161},
                                                                          {0x114, 0xf81f03e1},
                                                                          {0x118, 0xf85f03e0},
                                                                          {0x11c, 0xd65f03c0}}};
  return DecodedWords(words, budget);
}

std::optional<Fixture> DecodedLiteralLoadFunction(Budget& budget, bool live) {
  const std::array<std::pair<std::uint64_t, std::uint32_t>, 2> words{
      {{0x100, live ? 0x58008000U : 0x5800801fU}, {0x104, 0xd65f03c0}}};
  return DecodedWords(words, budget);
}

std::optional<Fixture> DecodedSelectedLoadFunction(Budget& budget) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 5> words{{{0x100, 0xf100001f},
                                                                          {0x104, 0x9a9f17e8},
                                                                          {0x108, 0x10007fc9},
                                                                          {0x10c, 0xf8687921},
                                                                          {0x110, 0xd65f03c0}}};
  return DecodedWords(words, budget);
}

std::optional<Fixture> DecodedBoundedTableFunction(Budget& budget) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 6> words{{{0x100, 0xf100101f},
                                                                          {0x104, 0x54000082},
                                                                          {0x108, 0x1000ffc1},
                                                                          {0x10c, 0xf8607820},
                                                                          {0x110, 0xd65f03c0},
                                                                          {0x114, 0xd65f03c0}}};
  return DecodedWords(words, budget);
}

std::optional<Fixture> DecodedBoundedTableAndReadFunction(Budget& budget) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 9> words{{{0xfc, 0xd28000e2},
                                                                          {0x100, 0xf100101f},
                                                                          {0x104, 0x540000a2},
                                                                          {0x108, 0x1000ffc1},
                                                                          {0x10c, 0xf8607820},
                                                                          {0x110, 0x14000004},
                                                                          {0x118, 0xd65f03c0},
                                                                          {0x120, 0xaa0203e3},
                                                                          {0x124, 0xd65f03c0}}};
  return DecodedWords(words, budget, 0xfc);
}

std::optional<Fixture> DecodedTwoBoundedTableFunction(Budget& budget) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 7> words{{{0x100, 0xf100101f},
                                                                          {0x104, 0x540000a2},
                                                                          {0x108, 0x1000ffc1},
                                                                          {0x10c, 0xf8607823},
                                                                          {0x110, 0xf8607822},
                                                                          {0x114, 0xd65f03c0},
                                                                          {0x118, 0xd65f03c0}}};
  return DecodedWords(words, budget);
}

TEST(SsaEval, DecodedBoundedTableAddressUsesFalseArmAndShift) {
  Budget budget({10000000, 10000000});
  auto built = DecodedBoundedTableFunction(budget);
  ASSERT_TRUE(built);
  ASSERT_EQ(ir::ValidateSsa(built->graph, budget), ir::SsaDecline::none);
  std::optional<ir::SsaHandle> table;
  std::optional<ir::SsaHandle> guard;
  ir::ValueId load_id = 0;
  for (std::size_t slot = 0; slot < built->graph.slots(); ++slot) {
    const auto handle = built->graph.Handle(slot);
    if (!handle) continue;
    const auto* block = built->graph.Get(*handle);
    if (block->address == 0x100) guard = handle;
    if (block->address == 0x108) {
      table = handle;
      for (ir::ValueId id = 0; id < block->nodes.size(); ++id)
        if (block->nodes[id].op == ir::Op::load) load_id = id;
    }
  }

  ASSERT_TRUE(guard);
  ASSERT_TRUE(table);
  const auto* guard_block = built->graph.Get(*guard);
  std::optional<std::uint32_t> edge_index;
  for (std::uint32_t index = 0; index < guard_block->edges.size(); ++index)
    if (guard_block->edges[index].target_block == table) edge_index = index;
  ASSERT_TRUE(edge_index);
  const auto bound = analysis::ProveSsaDirectIndexBound(built->graph, built->sources,
                                                        ir::SsaEntryScope::closed_population,
                                                        *guard, *edge_index, *table, budget);
  ASSERT_TRUE(bound.fact) << static_cast<int>(bound.reason);
  EXPECT_EQ(bound.fact->exclusive_upper, 4U);
  const auto address = analysis::ProveSsaBoundedTableAddress(built->graph, built->sources,
                                                             *bound.fact, load_id, budget);
  ASSERT_TRUE(address.fact) << static_cast<int>(address.reason);
  EXPECT_EQ(address.fact->base, 0x2100U);
  EXPECT_EQ(address.fact->stride, 8U);
  constexpr std::array<std::uint8_t, 32> rows{11, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0,
                                              33, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0};
  const ir::ConstantImageRange range{0x2100, rows};
  const ir::ImageFacts facts{std::span(&range, 1), {}, true, {}};
  auto folded = recovery::ProposeBoundedTableLoads(built->graph, *address.fact, built->sources,
                                                   facts, {true, true, true}, budget);
  ASSERT_TRUE(folded.provisional) << static_cast<int>(folded.reason);
  ASSERT_EQ(folded.journal.size(), 1U);
  EXPECT_EQ(ir::ValidateSsa(*folded.provisional, budget), ir::SsaDecline::none);
}

TEST(SsaEval, FoldsAnUnrelatedReadAfterBoundedTableRecovery) {
  Budget budget({10000000, 10000000});
  auto built = DecodedBoundedTableAndReadFunction(budget);
  ASSERT_TRUE(built);
  std::optional<ir::SsaHandle> guard, table;
  std::optional<ir::ValueId> load;
  for (std::size_t slot = 0; slot < built->graph.slots(); ++slot) {
    const auto handle = built->graph.Handle(slot);
    if (!handle) continue;
    const auto* block = built->graph.Get(*handle);
    if (block->address == 0xfc) guard = handle;
    if (block->address == 0x108) {
      table = handle;
      for (ir::ValueId id = 0; id < block->nodes.size(); ++id)
        if (block->nodes[id].op == ir::Op::load) load = id;
    }
  }

  ASSERT_TRUE(guard);
  ASSERT_TRUE(table);
  ASSERT_TRUE(load);
  std::optional<std::uint32_t> edge;
  const auto* guard_block = built->graph.Get(*guard);
  for (std::uint32_t index = 0; index < guard_block->edges.size(); ++index)
    if (guard_block->edges[index].target_block == table) edge = index;
  ASSERT_TRUE(edge);
  auto bound = analysis::ProveSsaDirectIndexBound(built->graph, built->sources,
                                                  ir::SsaEntryScope::closed_population, *guard,
                                                  *edge, *table, budget);
  ASSERT_TRUE(bound.fact) << static_cast<int>(bound.reason);
  auto address = analysis::ProveSsaBoundedTableAddress(built->graph, built->sources, *bound.fact,
                                                       *load, budget);
  ASSERT_TRUE(address.fact) << static_cast<int>(address.reason);
  constexpr std::array<std::uint8_t, 32> rows{11, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0,
                                              33, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0};
  const ir::ConstantImageRange range{0x2100, rows};
  const ir::ImageFacts image{std::span(&range, 1), {}, true, {}};
  auto table_fold = recovery::ProposeBoundedTableLoads(built->graph, *address.fact, built->sources,
                                                       image, {true, true, true}, budget);
  ASSERT_TRUE(table_fold.provisional) << static_cast<int>(table_fold.reason);
  auto proved = analysis::ProveSsaSccp(
      *table_fold.provisional, ir::SsaEntryScope::closed_population, built->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  auto reads = recovery::ProposeSsaSccpReadFold(*table_fold.provisional, *proved.facts,
                                                built->sources, budget);
  ASSERT_TRUE(reads.provisional) << static_cast<int>(reads.reason);
  EXPECT_TRUE(std::any_of(reads.journal.begin(), reads.journal.end(), [&](const auto& edit) {
    const auto* block = table_fold.provisional->Get(edit.original_block);
    return block && block->address == 0x120 && edit.original.op == ir::Op::read && edit.value == 7;
  }));
  const auto retained_table = reads.provisional->Handle(table->slot);
  ASSERT_TRUE(retained_table);
  EXPECT_TRUE(std::any_of(
      reads.provisional->Get(*retained_table)->constant_loads.begin(),
      reads.provisional->Get(*retained_table)->constant_loads.end(),
      [](const auto& fold) { return fold.kind == ir::SsaConstantKind::bounded_table; }));
  EXPECT_EQ(ir::ValidateSsaWithSources(*reads.provisional, built->sources, budget),
            ir::SsaDecline::none);
}

TEST(SsaEval, FollowsBothSsaEdgesAndRefutesSwappedArms) {
  auto fixture = BranchingFunction();
  Budget budget({10000000, 10000000});
  ASSERT_EQ(ir::ValidateSsa(fixture.graph, budget), ir::SsaDecline::none);
  const std::array<std::uint8_t, 4096> bytes{};
  const std::array<RegionInput, 1> regions{{{0x300000000, bytes}}};
  for (const auto [input, expected] : {std::pair{0U, 11U}, std::pair{1U, 22U}}) {
    auto mapped = Memory::Create(regions, budget);
    ASSERT_TRUE(mapped.memory);
    State state;
    auto value = BitVector::from_u64(64, input, 64, budget);
    ASSERT_TRUE(value);
    state.cells.push_back({0, std::move(*value)});
    auto lr = BitVector::from_u64(64, kLanding, 64, budget);
    ASSERT_TRUE(lr);
    state.cells.push_back({30, std::move(*lr)});
    auto run = ExecuteSsa(fixture.graph, fixture.sources, fixture.entry, state, *mapped.memory,
                          budget, {kBias});
    EXPECT_EQ(run.outcome, Outcome::completed);
    EXPECT_EQ(run.stop, SsaStop::returned);
    EXPECT_EQ(run.runtime_pc, kLanding);
    EXPECT_EQ(run.completed_blocks, 2U);
    EXPECT_EQ(state.cells[0].value.word(0), expected);
  }

  ASSERT_TRUE(fixture.graph.Update(fixture.entry, [](auto& block) {
    std::swap(block.edges[0].address, block.edges[1].address);
    std::swap(block.edges[0].target_block, block.edges[1].target_block);
  }));
  ASSERT_EQ(ir::ValidateSsa(fixture.graph, budget), ir::SsaDecline::none);
  auto mapped = Memory::Create(regions, budget);
  ASSERT_TRUE(mapped.memory);
  State state;
  auto zero = BitVector::from_u64(64, 0, 64, budget);
  ASSERT_TRUE(zero);
  state.cells.push_back({0, std::move(*zero)});
  auto lr = BitVector::from_u64(64, kLanding, 64, budget);
  ASSERT_TRUE(lr);
  state.cells.push_back({30, std::move(*lr)});
  const auto wrong = ExecuteSsa(fixture.graph, fixture.sources, fixture.entry, state,
                                *mapped.memory, budget, {kBias});
  EXPECT_EQ(wrong.outcome, Outcome::unsupported);
  EXPECT_EQ(wrong.stop, SsaStop::unresolved);
}

TEST(SsaEval, PromotedFrameAccessesRequireMappedSlot) {
  Budget budget({10000000, 10000000});
  ir::SsaGraph graph;
  ir::SsaBlock block{};
  block.address = 0x100;
  block.source_groups = {0x100};
  block.source_bytes = {Bytes(0xd65f03c0)};
  block.original_sources = {0};
  block.nodes = {{ir::Op::read, 64, {}, 0, 31}, {ir::Op::constant, 64, {}, 16},
                 {ir::Op::sub, 64, {0, 1}},     {ir::Op::constant, 64, {}, 42},
                 {ir::Op::store, 64, {2, 3}},   {ir::Op::load, 64, {2}},
                 {ir::Op::read, 64, {}, 0, 30}};
  block.nodes[4].access.alignment = 8;
  block.nodes[5].access.alignment = 8;
  block.boundaries = {{0, 7, {{0, 5}}, ir::Transfer{ir::TransferKind::return_, 6, {}, {}, {}}}};
  block.phis = {{0, 64, true, {}}, {30, 64, true, {}}, {31, 64, true, {}}};
  block.clobbers = {0, 0, 0};
  block.disabled_effects = {4, 5};
  block.frame_phis = {{-16, 8, true, {}}};
  const auto handle = graph.Add(std::move(block));
  graph.Update(handle, [&](auto& inserted) {
    inserted.reads = {{0, 2, {ir::SsaValueKind::phi, handle, 2}},
                      {6, 1, {ir::SsaValueKind::phi, handle, 1}}};
    inserted.exits = {{0, {ir::SsaValueKind::node, handle, 5}},
                      {30, {ir::SsaValueKind::phi, handle, 1}},
                      {31, {ir::SsaValueKind::phi, handle, 2}}};
    inserted.frame_accesses = {{4, 0, {}}, {5, 0, ir::SsaValue{ir::SsaValueKind::node, handle, 3}}};
    inserted.frame_exits = {{ir::SsaValueKind::node, handle, 3}};
    inserted.edges = {{ir::SsaEdgeKind::return_, ir::SsaTargetKind::unknown, 0, {}, {}, {}, {}}};
  });
  graph.SetEntries({handle});
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  std::vector<ir::Group> sources;
  sources.emplace_back(0x100, Bytes(0xd65f03c0), graph.Get(handle)->nodes,
                       std::vector<ir::Write>{{0, 5}}, ir::MemoryModel::atomic_scalar_reference);
  const std::array<std::uint8_t, 4096> bytes{};
  const std::array<RegionInput, 1> regions{{{0x300000000, bytes}}};
  const auto execute = [&](std::uint64_t sp, bool mapped_slot) {
    auto memory = Memory::Create(regions, budget);
    EXPECT_TRUE(memory.memory);
    State state;
    const std::array<std::pair<unsigned, std::uint64_t>, 3> cells{
        {{0, 0}, {30, kLanding}, {31, sp}}};
    for (const auto [storage, value] : cells) {
      auto bits = BitVector::from_u64(64, value, 64, budget);
      EXPECT_TRUE(bits);
      state.cells.push_back({storage, std::move(*bits)});
    }

    const auto run = ExecuteSsa(graph, sources, handle, state, *memory.memory, budget, {kBias});
    EXPECT_EQ(run.outcome, mapped_slot ? Outcome::completed : Outcome::unsupported);
    if (mapped_slot) {
      EXPECT_EQ(run.stop, SsaStop::returned);
      EXPECT_EQ(state.cells[0].value.word(0), 42U);
      EXPECT_EQ(memory.memory->Regions()[0].bytes, std::vector<std::uint8_t>(4096));
    }
  };

  execute(0x300000080, true);
  execute(0x400000080, false);
}

TEST(SsaEval, RefusesStaleSourceBytesAndInternalDiversion) {
  Budget budget({10000000, 10000000});
  auto fixture = BranchingFunction();
  const std::array<std::uint8_t, 4096> bytes{};
  const std::array<RegionInput, 1> regions{{{0x300000000, bytes}}};
  const auto run = [&](const Fixture& candidate) {
    auto memory = Memory::Create(regions, budget);
    EXPECT_TRUE(memory.memory);
    State state;
    for (const auto [storage, value] :
         std::array<std::pair<unsigned, std::uint64_t>, 2>{{{0, 0}, {30, kLanding}}}) {
      auto bits = BitVector::from_u64(64, value, 64, budget);
      EXPECT_TRUE(bits);
      state.cells.push_back({storage, std::move(*bits)});
    }

    return ExecuteSsa(candidate.graph, candidate.sources, candidate.entry, state, *memory.memory,
                      budget, {kBias});
  };

  auto changed = fixture.sources[0];
  fixture.sources[0] =
      ir::Group(0x100, Bytes(0xb40000a0),
                std::vector<ir::Node>(changed.nodes().begin(), changed.nodes().end()), {},
                ir::MemoryModel::unspecified, changed.transfer());
  EXPECT_EQ(run(fixture).outcome, Outcome::invalid_group);
  fixture = BranchingFunction();
  const auto left = *fixture.graph.Get(fixture.entry)->edges[0].target_block;
  ASSERT_TRUE(fixture.graph.Update(left, [](auto& block) {
    block.boundaries[0].transfer = ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}};
  }));
  ASSERT_EQ(ir::ValidateSsa(fixture.graph, budget), ir::SsaDecline::none);
  const auto diverged = run(fixture);
  EXPECT_EQ(diverged.outcome, Outcome::unsupported);
  EXPECT_EQ(diverged.stop, SsaStop::unresolved);
  EXPECT_EQ(diverged.completed_blocks, 1U);
}

TEST(SsaEval, ProposesMbaOnDecodedFunction) {
  Budget budget({10000000, 10000000});
  auto fixture = DecodedMbaFunction(budget);
  ASSERT_TRUE(fixture);
  auto proposal = recovery::ProposeLinearMba(fixture->graph, budget);
  ASSERT_EQ(proposal.reason, recovery::MbaDecline::none);
  ASSERT_TRUE(proposal.provisional);
  ASSERT_FALSE(proposal.journal.empty());
  EXPECT_EQ(ir::ValidateSsa(*proposal.provisional, budget), ir::SsaDecline::none);
}

TEST(SsaEval, FoldsTheConstantSharedByBothDecodedArms) {
  Budget budget({10000000, 10000000});
  auto fixture = DecodedPhiConstantFunction(budget);
  ASSERT_TRUE(fixture);
  auto reachable = analysis::ProveSsaReachability(fixture->graph, fixture->sources,
                                                  ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts) << static_cast<int>(reachable.reason);
  auto facts =
      analysis::ProveSsaPhiConstants(fixture->graph, *reachable.facts, fixture->sources, budget);
  ASSERT_TRUE(facts.facts) << static_cast<int>(facts.reason);
  ASSERT_EQ(facts.facts->constants.size(), 1U);
  auto proposed = recovery::ProposeSsaPhiConstantFold(fixture->graph, *reachable.facts,
                                                      *facts.facts, fixture->sources, budget);
  ASSERT_TRUE(proposed.provisional) << static_cast<int>(proposed.reason);
  ASSERT_EQ(proposed.journal.size(), 1U);
  EXPECT_EQ(proposed.journal[0].value, 12U);
  EXPECT_EQ(ir::ValidateSsaWithSources(*proposed.provisional, fixture->sources, budget),
            ir::SsaDecline::none);
}

TEST(SsaEval, PropagatesComputedConstantThroughDecodedJoin) {
  Budget budget({10000000, 10000000});
  auto fixture = DecodedComputedPhiFunction(budget);
  ASSERT_TRUE(fixture);
  auto reachable = analysis::ProveSsaReachability(fixture->graph, fixture->sources,
                                                  ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reachable.facts) << static_cast<int>(reachable.reason);
  auto simple =
      analysis::ProveSsaPhiConstants(fixture->graph, *reachable.facts, fixture->sources, budget);
  ASSERT_TRUE(simple.facts);
  EXPECT_TRUE(simple.facts->constants.empty());
  auto facts =
      analysis::ProveSsaConstants(fixture->graph, *reachable.facts, fixture->sources, budget);
  ASSERT_TRUE(facts.facts) << static_cast<int>(facts.reason);
  auto proposed = recovery::ProposeSsaConstantFold(fixture->graph, *reachable.facts, *facts.facts,
                                                   fixture->sources, budget);
  ASSERT_TRUE(proposed.provisional) << static_cast<int>(proposed.reason);
  ASSERT_EQ(proposed.journal.size(), 2U);
  EXPECT_EQ(ir::ValidateSsaWithSources(*proposed.provisional, fixture->sources, budget),
            ir::SsaDecline::none);
}

TEST(SsaEval, FoldsAJoinReachedThroughAConstantDecodedBranch) {
  Budget budget({10000000, 10000000});
  auto fixture = DecodedSccpFunction(budget);
  ASSERT_TRUE(fixture);
  auto proved = analysis::ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population,
                                       fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  auto proposed =
      recovery::ProposeSsaSccpFold(fixture->graph, *proved.facts, fixture->sources, budget);
  ASSERT_TRUE(proposed.provisional) << static_cast<int>(proposed.reason);
  EXPECT_TRUE(std::any_of(proposed.journal.begin(), proposed.journal.end(),
                          [](const auto& edit) { return edit.value == 23; }));
}

TEST(SsaEval, SccpFoldsAProvedStorageReadUsedByAConditional) {
  Budget budget({10000000, 10000000});
  auto fixture = DecodedReadConditionFunction(budget);
  ASSERT_TRUE(fixture);
  auto proved = analysis::ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population,
                                       fixture->sources, budget);
  ASSERT_TRUE(proved.facts) << static_cast<int>(proved.reason);
  auto folded =
      recovery::ProposeSsaSccpReadFold(fixture->graph, *proved.facts, fixture->sources, budget);
  ASSERT_TRUE(folded.provisional) << static_cast<int>(folded.reason);
  const auto read =
      std::find_if(folded.journal.begin(), folded.journal.end(), [](const auto& edit) {
        return edit.original.op == ir::Op::read && edit.original.storage == 0 && edit.value == 0;
      });
  ASSERT_NE(read, folded.journal.end());
  ASSERT_TRUE(read->removed_read);
  const auto* changed = folded.provisional->Get(read->result_block);
  ASSERT_NE(changed, nullptr);
  EXPECT_EQ(changed->nodes[read->node].op, ir::Op::constant);
  EXPECT_FALSE(std::any_of(changed->reads.begin(), changed->reads.end(),
                           [&](const auto& item) { return item.node == read->node; }));
  EXPECT_EQ(ir::ValidateSsaWithSources(*folded.provisional, fixture->sources, budget),
            ir::SsaDecline::none);
  auto forged = *proved.facts;
  const auto claim = std::find_if(
      forged.constants.values.begin(), forged.constants.values.end(), [&](const auto& fact) {
        return fact.kind == ir::SsaValueKind::node && fact.block == read->original_block &&
               fact.index == read->node;
      });
  ASSERT_NE(claim, forged.constants.values.end());
  claim->value = 1;
  EXPECT_EQ(
      recovery::ProposeSsaSccpReadFold(fixture->graph, forged, fixture->sources, budget).reason,
      recovery::SsaConstantFoldRefusal::stale_proof);
}

TEST(SsaEval, ExternalCallMustReportProvedContinuation) {
  Budget budget({10000000, 10000000});
  ir::SsaGraph graph;
  ir::SsaBlock call{};
  call.address = 0x100;
  call.source_groups = {0x100};
  call.source_bytes = {Bytes(0x94000080)};
  call.original_sources = {0};
  call.nodes = {{ir::Op::image_address, 64, {}, 0x300}, {ir::Op::image_address, 64, {}, 0x104}};
  call.boundaries = {{0, 2, {}, ir::Transfer{ir::TransferKind::call, 0, {}, {}, 1}}};
  call.phis = {{30, 64, true, {}}};

  // A call returns with its registers fresh: whatever the callee left.
  call.clobbers = {1};
  const auto start = graph.Add(std::move(call));
  ir::SsaBlock ending{};
  ending.address = 0x104;
  ending.source_groups = {0x104};
  ending.source_bytes = {Bytes(0xd65f03c0)};
  ending.original_sources = {1};
  ending.nodes = {{ir::Op::read, 64, {}, 0, 30}};
  ending.boundaries = {{0, 1, {}, ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}}}};
  ending.phis = {{30, 64, false, {}}};
  ending.clobbers = {0};
  ending.edges = {{ir::SsaEdgeKind::return_, ir::SsaTargetKind::unknown, 0, {}, {}, {}, {}}};
  const auto finish = graph.Add(std::move(ending));
  graph.Update(start, [&](auto& block) {
    block.exits = {{30, {ir::SsaValueKind::clobber, start, 0}}};
    block.edges = {
        {ir::SsaEdgeKind::callee, ir::SsaTargetKind::image_location, 0x300, {}, {}, {}, {}},
        {ir::SsaEdgeKind::potential_return,
         ir::SsaTargetKind::image_location,
         0x104,
         finish,
         {},
         {},
         {.callee_returns_to_continuation = true}}};
  });
  graph.Update(finish, [&](auto& block) {
    block.phis[0].incoming = {{start, {ir::SsaValueKind::clobber, start, 0}}};
    block.reads = {{0, 0, {ir::SsaValueKind::phi, finish, 0}}};
    block.exits = {{30, {ir::SsaValueKind::phi, finish, 0}}};
  });
  graph.SetEntries({start});
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  const std::vector<ir::Group> sources{
      {0x100,
       Bytes(0x94000080),
       {{ir::Op::image_address, 64, {}, 0x300}, {ir::Op::image_address, 64, {}, 0x104}},
       {},
       ir::MemoryModel::unspecified,
       ir::Transfer{ir::TransferKind::call, 0, {}, {}, 1}},
      {0x104,
       Bytes(0xd65f03c0),
       {{ir::Op::read, 64, {}, 0, 30}},
       {},
       ir::MemoryModel::unspecified,
       ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}}}};
  const std::array<std::uint8_t, 4096> bytes{};
  const std::array<RegionInput, 1> regions{{{0x300000000, bytes}}};
  const auto run = [&](std::uint64_t reported) {
    auto memory = Memory::Create(regions, budget);
    EXPECT_TRUE(memory.memory);
    State state;
    auto lr = BitVector::from_u64(64, kLanding, 64, budget);
    EXPECT_TRUE(lr);
    state.cells.push_back({30, std::move(*lr)});
    return ExecuteSsa(graph, sources, start, state, *memory.memory, budget, {kBias},
                      [reported](std::uint64_t target, State&, Memory&, Budget&) {
                        EXPECT_EQ(target, kBias + 0x300);
                        return SsaCalleeOutcome{Outcome::completed, reported};
                      });
  };

  const auto accepted = run(kBias + 0x104);
  EXPECT_EQ(accepted.outcome, Outcome::completed);
  EXPECT_EQ(accepted.stop, SsaStop::returned);
  EXPECT_EQ(accepted.runtime_pc, kLanding);
  const auto refused = run(kBias + 0x108);
  EXPECT_EQ(refused.outcome, Outcome::unsupported);
  EXPECT_EQ(refused.stop, SsaStop::unresolved);
}

// A call hands back fresh registers except what the declared contract says
// callees preserve. Only that claim may carry a register across one, so a
// graph that keeps any other, or keeps one with nothing declared, is invalid.
TEST(SsaEval, ACallKeepsOnlyWhatTheDeclaredContractPreserves) {
  const auto build = [](bool keep_link) {
    ir::SsaGraph graph;
    ir::SsaBlock call{};
    call.address = 0x100;
    call.source_groups = {0x100};
    call.source_bytes = {Bytes(0x94000080)};
    call.original_sources = {0};
    call.nodes = {{ir::Op::image_address, 64, {}, 0x300}, {ir::Op::image_address, 64, {}, 0x104}};
    call.boundaries = {{0, 2, {}, ir::Transfer{ir::TransferKind::call, 0, {}, {}, 1}}};
    call.phis = {{19, 64, true, {}}, {30, 64, true, {}}};
    call.clobbers = {0, std::uint8_t(keep_link ? 0 : 1)};
    const auto start = graph.Add(std::move(call));
    ir::SsaBlock ending{};
    ending.address = 0x104;
    ending.source_groups = {0x104};
    ending.source_bytes = {Bytes(0xd65f03c0)};
    ending.original_sources = {1};
    ending.nodes = {{ir::Op::read, 64, {}, 0, 30}};
    ending.boundaries = {{0, 1, {}, ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}}}};
    ending.phis = {{19, 64, false, {}}, {30, 64, false, {}}};
    ending.clobbers = {0, 0};
    ending.edges = {{ir::SsaEdgeKind::return_, ir::SsaTargetKind::unknown, 0, {}, {}, {}, {}}};
    const auto finish = graph.Add(std::move(ending));
    const ir::SsaValue link{keep_link ? ir::SsaValueKind::phi : ir::SsaValueKind::clobber, start,
                            1};
    graph.Update(start, [&](auto& block) {
      block.exits = {{19, {ir::SsaValueKind::phi, start, 0}}, {30, link}};
      block.edges = {
          {ir::SsaEdgeKind::callee, ir::SsaTargetKind::image_location, 0x300, {}, {}, {}, {}},
          {ir::SsaEdgeKind::potential_return,
           ir::SsaTargetKind::image_location,
           0x104,
           finish,
           {},
           {},
           {.callee_returns_to_continuation = true}}};
    });
    graph.Update(finish, [&](auto& block) {
      block.phis[0].incoming = {{start, {ir::SsaValueKind::phi, start, 0}}};
      block.phis[1].incoming = {{start, link}};
      block.reads = {{0, 1, {ir::SsaValueKind::phi, finish, 1}}};
      block.exits = {{19, {ir::SsaValueKind::phi, finish, 0}},
                     {30, {ir::SsaValueKind::phi, finish, 1}}};
    });
    graph.SetEntries({start});
    return graph;
  };

  const auto contract = [](std::vector<ir::StorageId> preserved) {
    ir::SsaObservability observed;
    observed.declared = true;
    observed.preserved = std::move(preserved);
    return observed;
  };

  Budget budget({10000000, 10000000});
  auto graph = build(false);
  EXPECT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::invalid_graph);
  graph.SetObservability(contract({20}));
  EXPECT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::invalid_graph);
  graph.SetObservability(contract({19, 20}));
  EXPECT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);

  // The link register is the call's own write, never the caller's to keep.
  auto linked = build(true);
  linked.SetObservability(contract({19, 20}));
  EXPECT_EQ(ir::ValidateSsa(linked, budget), ir::SsaDecline::invalid_graph);

  // Run, the graph reads x19 after the call as it was before; a callee that
  // changed it makes the run something else, which declines.
  const std::vector<ir::Group> sources{
      {0x100,
       Bytes(0x94000080),
       {{ir::Op::image_address, 64, {}, 0x300}, {ir::Op::image_address, 64, {}, 0x104}},
       {},
       ir::MemoryModel::unspecified,
       ir::Transfer{ir::TransferKind::call, 0, {}, {}, 1}},
      {0x104,
       Bytes(0xd65f03c0),
       {{ir::Op::read, 64, {}, 0, 30}},
       {},
       ir::MemoryModel::unspecified,
       ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}}}};
  const std::array<std::uint8_t, 4096> bytes{};
  const std::array<RegionInput, 1> regions{{{0x300000000, bytes}}};
  const auto run = [&](bool keep) {
    auto memory = Memory::Create(regions, budget);
    EXPECT_TRUE(memory.memory);
    State state;
    for (const auto [id, value] :
         {std::pair<ir::StorageId, std::uint64_t>{19, 0x1919}, {30, kLanding}}) {
      auto cell = BitVector::from_u64(64, value, 64, budget);
      EXPECT_TRUE(cell);
      state.cells.push_back({id, std::move(*cell)});
    }

    return ExecuteSsa(graph, sources, graph.entries()[0], state, *memory.memory, budget, {kBias},
                      [keep](std::uint64_t, State& inside, Memory&, Budget& spent) {
                        if (!keep)
                          for (auto& cell : inside.cells)
                            if (cell.id == 19)
                              cell.value = std::move(*BitVector::from_u64(64, 0, 64, spent));
                        return SsaCalleeOutcome{Outcome::completed, kBias + 0x104};
                      });
  };

  const auto kept = run(true);
  EXPECT_EQ(kept.outcome, Outcome::completed);
  EXPECT_EQ(kept.stop, SsaStop::returned);
  const auto broken = run(false);
  EXPECT_EQ(broken.outcome, Outcome::unsupported);
  EXPECT_EQ(broken.stop, SsaStop::declined);
}

TEST(SsaEval, TransitionEdgeMustMatchExecutedTransfer) {
  Budget budget({10000000, 10000000});
  auto fixture = BranchingFunction();
  ASSERT_TRUE(fixture.graph.Update(fixture.entry, [](auto& block) { block.transition = 0; }));
  ASSERT_EQ(ir::ValidateSsa(fixture.graph, budget), ir::SsaDecline::none);
  const std::array<std::uint8_t, 4096> bytes{};
  const std::array<RegionInput, 1> regions{{{0x300000000, bytes}}};
  const auto run = [&]() {
    auto memory = Memory::Create(regions, budget);
    EXPECT_TRUE(memory.memory);
    State state;
    for (const auto [storage, value] :
         std::array<std::pair<unsigned, std::uint64_t>, 2>{{{0, 0}, {30, kLanding}}}) {
      auto bits = BitVector::from_u64(64, value, 64, budget);
      EXPECT_TRUE(bits);
      state.cells.push_back({storage, std::move(*bits)});
    }

    return ExecuteSsa(fixture.graph, fixture.sources, fixture.entry, state, *memory.memory, budget,
                      {kBias});
  };

  EXPECT_EQ(run().outcome, Outcome::completed);
  ASSERT_TRUE(fixture.graph.Update(fixture.entry, [](auto& block) {
    std::swap(block.edges[0].address, block.edges[1].address);
    std::swap(block.edges[0].target_block, block.edges[1].target_block);
  }));
  ASSERT_EQ(ir::ValidateSsa(fixture.graph, budget), ir::SsaDecline::none);
  const auto wrong = run();
  EXPECT_EQ(wrong.outcome, Outcome::unsupported);
  EXPECT_EQ(wrong.stop, SsaStop::unresolved);
}

}  // namespace

int RunSsaProbe(std::uint64_t input, unsigned mutation, bool prune_dead = false) {
  Budget budget({10000000, 10000000});
  auto built = DecodedFunction(budget);
  if (!built) return 2;
  auto fixture = std::move(*built);
  std::size_t removed = 0;
  if (prune_dead) {
    for (std::size_t slot = 0; slot < fixture.graph.slots(); ++slot) {
      const auto handle = fixture.graph.Handle(slot);
      if (!handle || !fixture.graph.Update(*handle, [](auto& block) {
            for (auto& edge : block.edges)
              if (edge.kind == ir::SsaEdgeKind::return_) edge.assumptions.return_leaves = true;
          }))
        return 2;
    }

    ir::SsaBlock dead{};
    dead.address = 0x120;
    dead.source_groups = {0x120};
    dead.source_bytes = {Bytes(0x14000000)};
    dead.original_sources = {static_cast<std::uint32_t>(fixture.sources.size())};
    dead.nodes = {{ir::Op::image_address, 64, {}, 0x120}};
    dead.boundaries = {{0, 1, {}, ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}}}};
    const auto handle = fixture.graph.Add(std::move(dead));
    if (!fixture.graph.Update(handle, [&](auto& block) {
          block.edges = {{ir::SsaEdgeKind::branch,
                          ir::SsaTargetKind::image_location,
                          0x120,
                          handle,
                          {},
                          {},
                          {}}};
        }))
      return 2;
    fixture.sources.emplace_back(0x120, Bytes(0x14000000),
                                 std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x120}},
                                 std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                                 ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}});
    if (mutation == 1) {
      const auto omitted = fixture.graph.Get(fixture.entry)->edges.back().target_block;
      if (!omitted ||
          !fixture.graph.Update(fixture.entry, [](auto& block) { block.edges.pop_back(); }) ||
          !fixture.graph.Update(*omitted, [](auto& block) {
            for (auto& phi : block.phis) phi.incoming.clear();
            for (auto& phi : block.frame_phis) phi.incoming.clear();
          }))
        return 2;
    } else if (mutation == 2) {
      const auto original_arm = fixture.graph.Get(fixture.entry)->edges.front().target_block;
      if (!original_arm ||
          !fixture.graph.Update(fixture.entry,
                                [&](auto& block) {
                                  const auto target = block.boundaries[0].transfer->target;
                                  const auto offset = block.nodes[target].inputs[1];
                                  block.nodes[offset].immediate = 0x20;
                                  block.edges.front().address = 0x120;
                                  block.edges.front().target_block = handle;
                                }) ||
          !fixture.graph.Update(*original_arm, [](auto& block) {
            for (auto& phi : block.phis) phi.incoming.clear();
            for (auto& phi : block.frame_phis) phi.incoming.clear();
          }))
        return 2;
    }

    auto proof = analysis::ProveSsaReachability(fixture.graph, fixture.sources,
                                                ir::SsaEntryScope::closed_population, budget);
    if (!proof.facts) {
      if ((mutation == 1 &&
           proof.reason == analysis::SsaReachabilityRefusal::incomplete_successors) ||
          (mutation == 2 && proof.reason == analysis::SsaReachabilityRefusal::invalid_graph)) {
        std::cout << "{\"declined\":true}\n";
        return 0;
      }

      return 2;
    }

    auto result =
        recovery::ProposeUnreachableBlocks(fixture.graph, *proof.facts, fixture.sources, budget);
    if (!result.provisional || result.reason != recovery::SsaDceRefusal::none) return 2;
    for (const auto& edit : result.journal)
      if (edit.kind == recovery::SsaDceEditKind::removed_block) ++removed;
    if (removed != 1) return 2;
    fixture.graph = std::move(*result.provisional);
    fixture.entry = fixture.graph.entries()[0];
  }

  if (mutation == 1 && !prune_dead && !fixture.graph.Update(fixture.entry, [](auto& block) {
        std::swap(block.edges[0].address, block.edges[1].address);
        std::swap(block.edges[0].target_block, block.edges[1].target_block);
      }))
    return 2;
  const std::array<std::uint8_t, 4096> bytes{};
  const std::array<RegionInput, 1> regions{{{0x300000000, bytes}}};
  auto mapped = Memory::Create(regions, budget);
  if (!mapped.memory) return 2;
  State state;
  for (unsigned index = 0; index < 36; ++index) {
    const auto value = index == 0                   ? input
                       : index == 30                ? kLanding
                       : index == 31                ? 0x300000800ULL
                       : index == 32 || index == 34 ? 1ULL
                       : index >= 32                ? 0ULL
                                                    : index + 3ULL;
    auto bits = BitVector::from_u64(index < 32 ? 64 : 1, value, 64, budget);
    if (!bits) return 2;
    state.cells.push_back({index, std::move(*bits)});
  }

  const auto run = ExecuteSsa(fixture.graph, fixture.sources, fixture.entry, state, *mapped.memory,
                              budget, {kBias});
  if (run.outcome == Outcome::unsupported && run.stop == SsaStop::unresolved) {
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  if (run.outcome != Outcome::completed || run.stop != SsaStop::returned ||
      mapped.memory->Regions().size() != 1)
    return 2;
  std::cout << "{\"removed_blocks\":" << removed << ",\"registers\":[";
  for (unsigned index = 0; index < 31; ++index) {
    if (index) std::cout << ',';
    std::cout << state.cells[index].value.word(0);
  }

  std::cout << "],\"nzcv\":";
  std::uint64_t nzcv = 0;
  for (unsigned index = 32; index < 36; ++index)
    nzcv |= state.cells[index].value.word(0) << (63 - index);
  std::cout << nzcv << ",\"sp\":" << state.cells[31].value.word(0) << ",\"pc\":" << run.runtime_pc
            << ",\"memory_unchanged\":"
            << (mapped.memory->Regions()[0].bytes == std::vector<std::uint8_t>(4096) ? "true"
                                                                                     : "false")
            << "}\n";
  return 0;
}

int RunInternalCallProbe(std::uint64_t input, unsigned mode) {
  Budget budget({10000000, 10000000});
  const std::array<std::pair<std::uint64_t, std::uint32_t>, 7> words{
      {{0x100, 0xaa1e03e9},
       {0x104, 0x94000003},
       {0x108, 0xd2800c60},
       {0x10c, 0xd65f0120},
       {0x110, 0x91000400},
       {0x114, 0xaa1e03e1},
       {0x118, mode == 1 ? 0xd65f03c0U : 0xd65f0120U}}};
  auto built = DecodedWords(words, budget);
  if (!built) return 2;
  if (mode == 2 && !built->graph.Update(built->entry, [](auto& block) {
        // A wrong link write keeps the callee edge intact; the independent
        // execution must catch the state difference at the destination.
        auto& boundary = block.boundaries.back();
        for (auto& write : boundary.writes)
          if (write.storage == 30) write.value = boundary.transfer->target;
      }))
    return 2;
  const std::array<std::uint8_t, 4096> bytes{};
  const std::array<RegionInput, 1> regions{{{0x300000000, bytes}}};
  auto mapped = Memory::Create(regions, budget);
  if (!mapped.memory) return 2;
  State state;
  for (unsigned index = 0; index < 36; ++index) {
    const auto value = index == 0                   ? input
                       : index == 30                ? kLanding
                       : index == 31                ? 0x300000800ULL
                       : index == 32 || index == 34 ? 1ULL
                       : index >= 32                ? 0ULL
                                                    : index + 3ULL;
    auto bits = BitVector::from_u64(index < 32 ? 64 : 1, value, 64, budget);
    if (!bits) return 2;
    state.cells.push_back({index, std::move(*bits)});
  }

  const auto run = ExecuteSsa(built->graph, built->sources, built->entry, state, *mapped.memory,
                              budget, {kBias});
  if (mode == 2 && run.outcome == Outcome::invalid_group) {
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  if (mode == 1) {
    if (run.outcome != Outcome::unsupported || run.stop != SsaStop::unresolved ||
        run.runtime_pc != kBias + 0x108)
      return 2;
    std::cout << "{\"declined\":true,\"pc\":" << run.runtime_pc << "}\n";
    return 0;
  }

  if (run.outcome != Outcome::completed || run.stop != SsaStop::returned) return 2;
  std::cout << "{\"registers\":[";
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

TEST(SsaExecution, FollowsInternalCallWithLinkStateAndRefusesUnresolvedInternalReturn) {
  EXPECT_EQ(RunInternalCallProbe(17, 0), 0);
  EXPECT_EQ(RunInternalCallProbe(17, 1), 0);
}

int RunWideCopyProbe(std::uint64_t high, unsigned mode) {
  Budget budget({10000000, 10000000});
  const std::array<std::pair<std::uint64_t, std::uint32_t>, 3> words{
      {{0x100, 0x3dc00000}, {0x104, 0x3d800020}, {0x108, 0xd65f03c0}}};
  auto built = DecodedWords(words, budget);
  if (!built) return 2;
  auto simplified = recovery::ProposeSsaSimplify(built->graph, {}, false, budget);
  if (simplified.reason != recovery::SsaSimplifyRefusal::none) return 2;
  if (simplified.provisional) {
    built->graph = std::move(*simplified.provisional);
    built->entry = built->graph.entries()[0];
  }

  if (mode == 1 && !built->graph.Update(built->entry, [](auto& block) {
        for (auto& node : block.nodes)
          if (node.op == ir::Op::extract && node.immediate == 64)
            node = {ir::Op::constant, 64, {}, 0};
      }))
    return 2;
  std::array<std::uint8_t, 4096> bytes{};
  for (unsigned i = 0; i < 8; ++i) {
    bytes[i] = static_cast<std::uint8_t>(0x123456789abcdef0ULL >> (i * 8));
    bytes[i + 8] = static_cast<std::uint8_t>(high >> (i * 8));
  }

  const std::array<RegionInput, 1> regions{{{0x300000000, bytes}}};
  auto mapped = Memory::Create(regions, budget);
  if (!mapped.memory) return 2;
  State state;
  for (unsigned index = 0; index < 36; ++index) {
    const auto value = index == 0                   ? 0x300000000ULL
                       : index == 1                 ? 0x300000020ULL
                       : index == 30                ? kLanding
                       : index == 31                ? 0x300000800ULL
                       : index == 32 || index == 34 ? 1ULL
                       : index >= 32                ? 0ULL
                                                    : index + 3ULL;
    auto bits = BitVector::from_u64(index < 32 ? 64 : 1, value, 128, budget);
    if (!bits) return 2;
    state.cells.push_back({index, std::move(*bits)});
  }

  auto vector = BitVector::from_u64(128, 0, 128, budget);
  if (!vector) return 2;
  state.cells.push_back({37, std::move(*vector)});
  const auto run = ExecuteSsa(built->graph, built->sources, built->entry, state, *mapped.memory,
                              budget, {kBias});
  if (run.outcome != Outcome::completed || run.stop != SsaStop::returned) return 2;
  std::cout << "{\"registers\":[";
  for (unsigned index = 0; index < 31; ++index) {
    if (index) std::cout << ',';
    std::cout << state.cells[index].value.word(0);
  }

  std::uint64_t nzcv = 0;
  for (unsigned index = 32; index < 36; ++index)
    nzcv |= state.cells[index].value.word(0) << (63 - index);
  std::cout << "],\"nzcv\":" << nzcv << ",\"sp\":" << state.cells[31].value.word(0)
            << ",\"pc\":" << run.runtime_pc << ",\"memory\":[";
  const auto& after = mapped.memory->Regions()[0].bytes;
  if (mode == 0) {
    auto expected = bytes;
    std::copy_n(bytes.begin(), 16, expected.begin() + 32);
    if (!std::equal(expected.begin(), expected.end(), after.begin(), after.end())) return 2;
  }

  for (std::size_t i = 0; i < after.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << unsigned(after[i]);
  }

  std::cout << "]}\n";
  return 0;
}

TEST(SsaExecution, WideCopySurvivesAlgebraicSimplification) {
  EXPECT_EQ(RunWideCopyProbe(0xfedcba9876543210ULL, 0), 0);
}

int RunMbaProbe(std::uint64_t left, std::uint64_t right, bool mutate) {
  Budget budget({10000000, 10000000});
  auto built = DecodedMbaFunction(budget);
  if (!built) return 2;
  auto proposal = recovery::ProposeLinearMba(built->graph, budget);
  if (!proposal.provisional || proposal.journal.empty()) return 2;
  if (mutate) {
    const auto edit = proposal.journal.back();
    if (!proposal.provisional->Update(
            edit.result_block, [&](auto& block) { block.nodes[edit.edit.node].op = ir::Op::sub; }))
      return 2;
  }

  const std::array<std::uint8_t, 4096> bytes{};
  const std::array<RegionInput, 1> regions{{{0x300000000, bytes}}};
  auto mapped = Memory::Create(regions, budget);
  if (!mapped.memory) return 2;
  State state;
  for (unsigned index = 0; index < 36; ++index) {
    const auto value = index == 0                   ? left
                       : index == 1                 ? right
                       : index == 30                ? kLanding
                       : index == 31                ? 0x300000800ULL
                       : index == 32 || index == 34 ? 1ULL
                       : index >= 32                ? 0ULL
                                                    : index + 3ULL;
    auto bits = BitVector::from_u64(index < 32 ? 64 : 1, value, 64, budget);
    if (!bits) return 2;
    state.cells.push_back({index, std::move(*bits)});
  }

  const auto run =
      ExecuteSsa(*proposal.provisional, built->sources, proposal.provisional->entries()[0], state,
                 *mapped.memory, budget, {kBias});
  if (run.outcome != Outcome::completed || run.stop != SsaStop::returned ||
      mapped.memory->Regions().size() != 1)
    return 2;
  std::cout << "{\"edits\":" << proposal.journal.size() << ",\"registers\":[";
  for (unsigned index = 0; index < 31; ++index) {
    if (index) std::cout << ',';
    std::cout << state.cells[index].value.word(0);
  }

  std::cout << "],\"nzcv\":";
  std::uint64_t nzcv = 0;
  for (unsigned index = 32; index < 36; ++index)
    nzcv |= state.cells[index].value.word(0) << (63 - index);
  std::cout << nzcv << ",\"sp\":" << state.cells[31].value.word(0) << ",\"pc\":" << run.runtime_pc
            << ",\"memory_unchanged\":"
            << (mapped.memory->Regions()[0].bytes == std::vector<std::uint8_t>(4096) ? "true"
                                                                                     : "false")
            << "}\n";
  return 0;
}

enum class CanonicalProbe { boolean, affine, signed_affine };

// Mutation 1 flips the result operation; 2 flips the first inserted node, which
// for the signed fixture is the doubled free value.
int RunCanonicalMbaProbe(std::uint64_t left, std::uint64_t right, std::uint64_t third,
                         unsigned mutate, CanonicalProbe form) {
  Budget budget({10000000, 10000000});
  auto built = form == CanonicalProbe::boolean  ? DecodedCanonicalMbaFunction(budget)
               : form == CanonicalProbe::affine ? DecodedAffineMbaFunction(budget)
                                                : DecodedSignedAffineMbaFunction(budget);
  if (!built) return 2;
  auto proposal = recovery::ProposeCanonicalLinearMba(built->graph, budget);
  if (!proposal.provisional || proposal.journal.size() != 1) return 2;
  const auto edit = proposal.journal[0];
  const auto expected = form == CanonicalProbe::boolean  ? ir::Op::bit_xor
                        : form == CanonicalProbe::affine ? ir::Op::add
                                                         : ir::Op::sub;
  const auto flipped = [](ir::Op op) {
    return op == ir::Op::bit_xor ? ir::Op::bit_or : op == ir::Op::add ? ir::Op::sub : ir::Op::add;
  };

  if (edit.replacement.op != expected ||
      edit.inserted.size() != (form == CanonicalProbe::signed_affine ? 2U : 1U) ||
      (form == CanonicalProbe::signed_affine &&
       (edit.inserted[0].op != ir::Op::add ||
        edit.inserted[0].inputs[0] != edit.inserted[0].inputs[1])))
    return 2;
  const auto mutated = mutate == 2 ? edit.result_node - edit.inserted.size() : edit.result_node;
  if (mutate && !proposal.provisional->Update(edit.result_block, [&](auto& block) {
        block.nodes[mutated].op = flipped(block.nodes[mutated].op);
      }))
    return 2;
  const std::array<std::uint8_t, 4096> bytes{};
  const std::array<RegionInput, 1> regions{{{0x300000000, bytes}}};
  auto mapped = Memory::Create(regions, budget);
  if (!mapped.memory) return 2;
  State state;
  for (unsigned index = 0; index < 36; ++index) {
    const auto value = index == 0                   ? left
                       : index == 1                 ? right
                       : index == 2                 ? third
                       : index == 30                ? kLanding
                       : index == 31                ? 0x300000800ULL
                       : index == 32 || index == 34 ? 1ULL
                       : index >= 32                ? 0ULL
                                                    : index + 3ULL;
    auto bits = BitVector::from_u64(index < 32 ? 64 : 1, value, 64, budget);
    if (!bits) return 2;
    state.cells.push_back({index, std::move(*bits)});
  }

  const auto run =
      ExecuteSsa(*proposal.provisional, built->sources, proposal.provisional->entries()[0], state,
                 *mapped.memory, budget, {kBias});
  if (run.outcome != Outcome::completed || run.stop != SsaStop::returned ||
      mapped.memory->Regions().size() != 1)
    return 2;
  std::cout << "{\"edits\":" << proposal.journal.size() << ",\"registers\":[";
  for (unsigned index = 0; index < 31; ++index) {
    if (index) std::cout << ',';
    std::cout << state.cells[index].value.word(0);
  }

  std::cout << "],\"nzcv\":";
  std::uint64_t nzcv = 0;
  for (unsigned index = 32; index < 36; ++index)
    nzcv |= state.cells[index].value.word(0) << (63 - index);
  std::cout << nzcv << ",\"sp\":" << state.cells[31].value.word(0) << ",\"pc\":" << run.runtime_pc
            << ",\"memory_unchanged\":"
            << (mapped.memory->Regions()[0].bytes == std::vector<std::uint8_t>(4096) ? "true"
                                                                                     : "false")
            << "}\n";
  return 0;
}

int RunPureDceProbe(std::uint64_t left, std::uint64_t right, bool mutate) {
  Budget budget({10000000, 10000000});
  auto built = DecodedPureDceFunction(budget);
  if (!built) return 2;
  auto proof = analysis::ProveSsaNodeLiveness(built->graph, budget);
  if (!proof.facts) return 2;
  if (mutate) {
    const auto* entry = built->graph.Get(built->entry);
    if (!entry || entry->boundaries.empty() || entry->boundaries[0].writes.empty()) return 2;
    proof.facts->live_nodes[built->entry.slot][entry->boundaries[0].writes[0].value] = 0;
    const auto refused = recovery::ProposeDeadPureNodes(built->graph, *proof.facts, budget);
    if (refused.reason != recovery::SsaPureDceRefusal::stale_proof || refused.provisional ||
        !refused.journal.empty())
      return 2;
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  auto proposed = recovery::ProposeDeadPureNodes(built->graph, *proof.facts, budget);
  if (!proposed.provisional || proposed.journal.empty()) return 2;
  const std::array<std::uint8_t, 4096> bytes{};
  const std::array<RegionInput, 1> regions{{{0x300000000, bytes}}};
  auto mapped = Memory::Create(regions, budget);
  if (!mapped.memory) return 2;
  State state;
  for (unsigned index = 0; index < 36; ++index) {
    const auto value = index == 0                   ? left
                       : index == 1                 ? right
                       : index == 30                ? kLanding
                       : index == 31                ? 0x300000800ULL
                       : index == 32 || index == 34 ? 1ULL
                       : index >= 32                ? 0ULL
                                                    : index + 3ULL;
    auto bits = BitVector::from_u64(index < 32 ? 64 : 1, value, 64, budget);
    if (!bits) return 2;
    state.cells.push_back({index, std::move(*bits)});
  }

  const auto run =
      ExecuteSsa(*proposed.provisional, built->sources, proposed.provisional->entries()[0], state,
                 *mapped.memory, budget, {kBias});
  if (run.outcome != Outcome::completed || run.stop != SsaStop::returned ||
      mapped.memory->Regions().size() != 1)
    return 2;
  std::cout << "{\"edits\":" << proposed.journal.size() << ",\"registers\":[";
  for (unsigned index = 0; index < 31; ++index) {
    if (index) std::cout << ',';
    std::cout << state.cells[index].value.word(0);
  }

  std::cout << "],\"nzcv\":";
  std::uint64_t nzcv = 0;
  for (unsigned index = 32; index < 36; ++index)
    nzcv |= state.cells[index].value.word(0) << (63 - index);
  std::cout << nzcv << ",\"sp\":" << state.cells[31].value.word(0) << ",\"pc\":" << run.runtime_pc
            << ",\"memory_unchanged\":"
            << (mapped.memory->Regions()[0].bytes == std::vector<std::uint8_t>(4096) ? "true"
                                                                                     : "false")
            << "}\n";
  return 0;
}

int RunFrameDeadProbe(std::uint64_t input, unsigned mode) {
  Budget budget({10000000, 10000000});
  auto built = DecodedFrameDeadFunction(budget, mode == 1);
  if (!built) return 2;
  const ir::PrivateFrameContract contract{31, -32, 0, true, true, true, 16, false, false};
  auto proof = analysis::ProvePrivateFrameSlots(built->graph, contract, budget);
  if (mode == 1) {
    if (proof.facts || proof.reason != analysis::PrivateFrameDecline::escaped_address) return 2;
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  if (!proof.facts || proof.facts->slots.size() != 1 || proof.facts->slots[0].offset != -16 ||
      proof.facts->slots[0].size != 8 || proof.facts->slots[0].accesses.size() != 2)
    return 2;
  if (mode == 2) {
    if (!built->graph.Update(built->entry, [](auto& block) {
          const auto store =
              std::find_if(block.nodes.begin(), block.nodes.end(),
                           [](const auto& node) { return node.op == ir::Op::store; });
          if (store == block.nodes.end()) return;
          block.nodes[store->inputs[0]] = {ir::Op::image_address, 64, {}, 0x3000};
        }))
      return 2;
    proof.facts->revision = built->graph.revision();
    const auto refused = recovery::ProposeDeadPrivateState(built->graph, *proof.facts, budget);
    if (refused.reason != recovery::DeadStateRefusal::stale_proof || refused.provisional ||
        !refused.journal.empty())
      return 2;
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  auto proposed = recovery::ProposeDeadPrivateState(built->graph, *proof.facts, budget);
  if (!proposed.provisional || proposed.journal.size() != 2 || !proposed.basis) return 2;
  const std::array<std::uint8_t, 4096> bytes{};
  const std::array<RegionInput, 1> regions{{{0x300000000, bytes}}};
  auto mapped = Memory::Create(regions, budget);
  if (!mapped.memory) return 2;
  State state;
  for (unsigned index = 0; index < 36; ++index) {
    const auto value = index == 0                   ? input
                       : index == 30                ? kLanding
                       : index == 31                ? 0x300000800ULL
                       : index == 32 || index == 34 ? 1ULL
                       : index >= 32                ? 0ULL
                                                    : index + 3ULL;
    auto bits = BitVector::from_u64(index < 32 ? 64 : 1, value, 64, budget);
    if (!bits) return 2;
    state.cells.push_back({index, std::move(*bits)});
  }

  const auto run =
      ExecuteSsa(*proposed.provisional, built->sources, proposed.provisional->entries()[0], state,
                 *mapped.memory, budget, {kBias});
  if (run.outcome != Outcome::completed || run.stop != SsaStop::returned ||
      mapped.memory->Regions().size() != 1)
    return 2;
  std::cout << "{\"edits\":" << proposed.journal.size()
            << ",\"slot_offset\":" << proof.facts->slots[0].offset
            << ",\"slot_size\":" << proof.facts->slots[0].size << ",\"registers\":[";
  for (unsigned index = 0; index < 31; ++index) {
    if (index) std::cout << ',';
    std::cout << state.cells[index].value.word(0);
  }

  std::cout << "],\"nzcv\":";
  std::uint64_t nzcv = 0;
  for (unsigned index = 32; index < 36; ++index)
    nzcv |= state.cells[index].value.word(0) << (63 - index);
  std::cout << nzcv << ",\"sp\":" << state.cells[31].value.word(0) << ",\"pc\":" << run.runtime_pc
            << ",\"memory_unchanged\":"
            << (mapped.memory->Regions()[0].bytes == std::vector<std::uint8_t>(4096) ? "true"
                                                                                     : "false")
            << "}\n";
  return 0;
}

int RunFramePromotionProbe(std::uint64_t input, unsigned mode) {
  Budget budget({10000000, 10000000});
  auto built = DecodedFramePromotionFunction(budget);
  if (!built) return 2;
  const ir::PrivateFrameContract contract{31, -32, 0, true, true, true, 16, false, false};
  const auto proof = analysis::ProvePrivateFrameSlots(built->graph, contract, budget);
  if (!proof.facts || proof.facts->slots.size() != 1) return 2;
  auto promoted = recovery::ProposeFramePromotion(built->graph, *proof.facts, budget);
  if (!promoted.provisional || promoted.journal.size() != 3) return 2;
  ir::SsaHandle join{};
  for (std::size_t slot = 0; slot < promoted.provisional->slots(); ++slot) {
    const auto handle = promoted.provisional->Handle(slot);
    if (handle && promoted.provisional->Get(*handle)->address == 0x118) join = *handle;
  }

  const auto* joined = promoted.provisional->Get(join);
  if (!joined || joined->frame_phis.size() != 1 || joined->frame_phis[0].incoming.size() != 2)
    return 2;
  if (mode == 1) {
    if (!promoted.provisional->Update(join, [](auto& block) {
          block.frame_phis[0].incoming[0].value = block.frame_phis[0].incoming[1].value;
        }))
      return 2;
    if (ir::ValidateSsa(*promoted.provisional, budget) != ir::SsaDecline::invalid_graph) return 2;
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  if (mode == 2) {
    for (std::size_t slot = 0; slot < promoted.provisional->slots(); ++slot) {
      const auto handle = promoted.provisional->Handle(slot);
      if (!handle) continue;
      const auto* block = promoted.provisional->Get(*handle);
      if (block->address != 0x104 && block->address != 0x110) continue;
      if (!promoted.provisional->Update(*handle, [](auto& changed) {
            const auto literal =
                std::find_if(changed.nodes.begin(), changed.nodes.end(), [](const auto& node) {
                  return node.op == ir::Op::constant &&
                         (node.immediate == 11 || node.immediate == 22);
                });
            if (literal != changed.nodes.end()) ++literal->immediate;
          }))
        return 2;
    }

    if (ir::ValidateSsa(*promoted.provisional, budget) != ir::SsaDecline::none) return 2;
  }

  const std::array<std::uint8_t, 4096> bytes{};
  const std::array<RegionInput, 1> regions{{{0x300000000, bytes}}};
  auto mapped = Memory::Create(regions, budget);
  if (!mapped.memory) return 2;
  State state;
  for (unsigned index = 0; index < 36; ++index) {
    const auto value = index == 0                   ? input
                       : index == 30                ? kLanding
                       : index == 31                ? 0x300000800ULL
                       : index == 32 || index == 34 ? 1ULL
                       : index >= 32                ? 0ULL
                                                    : index + 3ULL;
    auto bits = BitVector::from_u64(index < 32 ? 64 : 1, value, 64, budget);
    if (!bits) return 2;
    state.cells.push_back({index, std::move(*bits)});
  }

  const auto run =
      ExecuteSsa(*promoted.provisional, built->sources, promoted.provisional->entries()[0], state,
                 *mapped.memory, budget, {kBias});
  if (run.outcome != Outcome::completed || run.stop != SsaStop::returned ||
      mapped.memory->Regions().size() != 1)
    return 2;
  std::cout << "{\"edits\":" << promoted.journal.size()
            << ",\"join_inputs\":" << joined->frame_phis[0].incoming.size() << ",\"registers\":[";
  for (unsigned index = 0; index < 31; ++index) {
    if (index) std::cout << ',';
    std::cout << state.cells[index].value.word(0);
  }

  std::cout << "],\"nzcv\":";
  std::uint64_t nzcv = 0;
  for (unsigned index = 32; index < 36; ++index)
    nzcv |= state.cells[index].value.word(0) << (63 - index);
  std::cout << nzcv << ",\"sp\":" << state.cells[31].value.word(0) << ",\"pc\":" << run.runtime_pc
            << ",\"memory_unchanged\":"
            << (mapped.memory->Regions()[0].bytes == std::vector<std::uint8_t>(4096) ? "true"
                                                                                     : "false")
            << "}\n";
  return 0;
}

int RunLiteralLoadProbe(std::uint64_t input, bool live, unsigned mode) {
  Budget budget({10000000, 10000000});
  auto built = DecodedLiteralLoadFunction(budget, live);
  if (!built) return 2;
  constexpr std::array<std::uint8_t, 8> table{0x78, 0x56, 0x34, 0x12, 0, 0, 0, 0};
  const ir::ConstantImageRange range{0x1100, table};
  const ir::ImageFacts facts{std::span(&range, 1), {}, true, {}};
  const ir::ImageAccessContract access{mode != 1, true, true};
  auto folded = recovery::ProposeConstantImageLoads(built->graph, facts, access, budget);
  if (mode == 1) {
    if (folded.provisional || folded.refused.size() != 1 ||
        folded.refused[0].reason != recovery::ConstantLoadRefusal::not_nonfaulting)
      return 2;
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  if (!folded.provisional || folded.journal.size() != 1 || !folded.journal[0].fold.skip_access)
    return 2;
  auto candidate = std::move(*folded.provisional);
  ir::SsaHandle load_block{};
  ir::ValueId load_node = 0;
  for (std::size_t slot = 0; slot < candidate.slots(); ++slot) {
    const auto handle = candidate.Handle(slot);
    if (!handle) continue;
    const auto* block = candidate.Get(*handle);
    if (block->constant_loads.empty()) continue;
    load_block = *handle;
    load_node = block->constant_loads[0].node;
  }

  if (!candidate.Get(load_block)) return 2;
  if (mode == 3) {
    if (!live) return 2;
    auto proof = analysis::ProveSsaNodeLiveness(candidate, budget);
    if (!proof.facts || !proof.facts->live_nodes[load_block.slot][load_node]) return 2;
    proof.facts->live_nodes[load_block.slot][load_node] = 0;
    const auto refused = recovery::ProposeDeadPureNodes(candidate, *proof.facts, budget);
    if (refused.reason != recovery::SsaPureDceRefusal::stale_proof || refused.provisional) return 2;
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  if (mode == 2) {
    if (!live ||
        !candidate.Update(load_block,
                          [](auto& block) {
                            ++block.constant_loads[0].value;
                            ++block.constant_loads[0].declared_bytes[0];
                          }) ||
        ir::ValidateSsa(candidate, budget) != ir::SsaDecline::none)
      return 2;
  }

  std::size_t dead_nodes = 0;
  const auto proof = analysis::ProveSsaNodeLiveness(candidate, budget);
  if (!proof.facts || bool(proof.facts->live_nodes[load_block.slot][load_node]) != live) return 2;
  auto retired = recovery::ProposeDeadPureNodes(candidate, *proof.facts, budget);
  if (!retired.provisional || retired.journal.size() != (live ? 1U : 2U)) return 2;
  dead_nodes = retired.journal.size();
  candidate = std::move(*retired.provisional);
  std::array<std::uint8_t, 4096> code{};
  const auto brk = Bytes(0xd4200000);
  for (std::size_t offset = 0; offset < code.size(); offset += 4)
    std::copy(brk.begin(), brk.end(), code.begin() + offset);
  const auto instruction = Bytes(live ? 0x58008000U : 0x5800801fU);
  const auto ret = Bytes(0xd65f03c0);
  std::copy(instruction.begin(), instruction.end(), code.begin());
  std::copy(ret.begin(), ret.end(), code.begin() + 4);
  std::array<std::uint8_t, 4096> data{};
  std::copy(table.begin(), table.end(), data.begin());
  const std::array<std::uint8_t, 4096> stack{};
  const std::array<RegionInput, 3> regions{{{kCode, code, true, false},
                                            {kCode + 4096, data, true, false},
                                            {0x300000000, stack, true, true}}};
  auto mapped = Memory::Create(regions, budget);
  if (!mapped.memory) return 2;
  State state;
  for (unsigned index = 0; index < 36; ++index) {
    const auto value = index == 0                   ? input
                       : index == 30                ? kLanding
                       : index == 31                ? 0x300000800ULL
                       : index == 32 || index == 34 ? 1ULL
                       : index >= 32                ? 0ULL
                                                    : index + 3ULL;
    auto bits = BitVector::from_u64(index < 32 ? 64 : 1, value, 64, budget);
    if (!bits) return 2;
    state.cells.push_back({index, std::move(*bits)});
  }

  const auto run = ExecuteSsa(candidate, built->sources, candidate.entries()[0], state,
                              *mapped.memory, budget, {kBias});
  if (run.outcome != Outcome::completed || run.stop != SsaStop::returned ||
      mapped.memory->Regions().size() != 3)
    return 2;
  std::cout << "{\"retired_access\":true,\"dead_nodes\":" << dead_nodes << ",\"registers\":[";
  for (unsigned index = 0; index < 31; ++index) {
    if (index) std::cout << ',';
    std::cout << state.cells[index].value.word(0);
  }

  std::cout << "],\"nzcv\":";
  std::uint64_t nzcv = 0;
  for (unsigned index = 32; index < 36; ++index)
    nzcv |= state.cells[index].value.word(0) << (63 - index);
  std::cout << nzcv << ",\"sp\":" << state.cells[31].value.word(0) << ",\"pc\":" << run.runtime_pc
            << ",\"memory_unchanged\":"
            << (mapped.memory->Regions()[0].bytes ==
                            std::vector<std::uint8_t>(code.begin(), code.end()) &&
                        mapped.memory->Regions()[1].bytes ==
                            std::vector<std::uint8_t>(data.begin(), data.end()) &&
                        mapped.memory->Regions()[2].bytes == std::vector<std::uint8_t>(4096)
                    ? "true"
                    : "false")
            << "}\n";
  return 0;
}

int RunSelectedLoadProbe(std::uint64_t input, unsigned mode) {
  Budget budget({10000000, 10000000});
  auto built = DecodedSelectedLoadFunction(budget);
  if (!built) return 2;
  auto addresses = analysis::ProveSsaSelectedImageAddresses(built->graph, budget);
  if (!addresses.facts || addresses.facts->selected.size() != 1) return 2;
  constexpr std::array<std::uint8_t, 16> table{11, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0};
  const ir::ConstantImageRange range{0x1100, table};
  const ir::ImageFacts facts{std::span(&range, 1), {}, true, {}};
  if (mode == 3) {
    ++addresses.facts->selected[0].when_true;
    const auto refused = recovery::ProposeSelectedImageLoads(built->graph, *addresses.facts, facts,
                                                             {true, true, true}, budget);
    if (refused.provisional || refused.reason != recovery::ConstantLoadRefusal::invalid_graph)
      return 2;
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  const ir::ImageAccessContract access{mode != 1, true, true};
  auto folded =
      recovery::ProposeSelectedImageLoads(built->graph, *addresses.facts, facts, access, budget);
  if (mode == 1) {
    if (folded.provisional || folded.refused.size() != 1 ||
        folded.refused[0].reason != recovery::ConstantLoadRefusal::not_nonfaulting)
      return 2;
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  if (!folded.provisional || folded.journal.size() != 1) return 2;
  auto candidate = std::move(*folded.provisional);
  const auto load_block = candidate.entries()[0];
  const auto load = folded.journal[0].fold.node;
  if (mode == 4) {
    auto live = analysis::ProveSsaNodeLiveness(candidate, budget);
    if (!live.facts || !live.facts->live_nodes[load_block.slot][*folded.journal[0].fold.condition])
      return 2;
    live.facts->live_nodes[load_block.slot][*folded.journal[0].fold.condition] = 0;
    const auto refused = recovery::ProposeDeadPureNodes(candidate, *live.facts, budget);
    if (refused.provisional || refused.reason != recovery::SsaPureDceRefusal::stale_proof) return 2;
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  if (mode == 2 && (!candidate.Update(load_block, [](auto& block) {
        auto& fold = block.constant_loads[0];
        ++fold.value;
        ++fold.declared_bytes[0];
        ++fold.alternative_value;
        ++fold.alternative_declared_bytes[0];
      }) || ir::ValidateSsa(candidate, budget) != ir::SsaDecline::none))
    return 2;
  const auto live = analysis::ProveSsaNodeLiveness(candidate, budget);
  if (!live.facts || !live.facts->live_nodes[load_block.slot][load]) return 2;
  auto retired = recovery::ProposeDeadPureNodes(candidate, *live.facts, budget);
  if (!retired.provisional || retired.journal.size() != 3) return 2;
  candidate = std::move(*retired.provisional);
  std::array<std::uint8_t, 4096> code{};
  const auto brk = Bytes(0xd4200000);
  for (std::size_t offset = 0; offset < code.size(); offset += 4)
    std::copy(brk.begin(), brk.end(), code.begin() + offset);
  constexpr std::array<std::uint32_t, 5> words{0xf100001f, 0x9a9f17e8, 0x10007fc9, 0xf8687921,
                                               0xd65f03c0};
  for (std::size_t index = 0; index < words.size(); ++index) {
    const auto bytes = Bytes(words[index]);
    std::copy(bytes.begin(), bytes.end(), code.begin() + index * 4);
  }

  std::array<std::uint8_t, 4096> data{};
  std::copy(table.begin(), table.end(), data.begin());
  const std::array<std::uint8_t, 4096> stack{};
  const std::array<RegionInput, 3> regions{{{kCode, code, true, false},
                                            {kCode + 4096, data, true, false},
                                            {0x300000000, stack, true, true}}};
  auto mapped = Memory::Create(regions, budget);
  if (!mapped.memory) return 2;
  State state;
  for (unsigned index = 0; index < 36; ++index) {
    const auto value = index == 0                   ? input
                       : index == 30                ? kLanding
                       : index == 31                ? 0x300000800ULL
                       : index == 32 || index == 34 ? 1ULL
                       : index >= 32                ? 0ULL
                                                    : index + 3ULL;
    auto bits = BitVector::from_u64(index < 32 ? 64 : 1, value, 64, budget);
    if (!bits) return 2;
    state.cells.push_back({index, std::move(*bits)});
  }

  const auto run = ExecuteSsa(candidate, built->sources, candidate.entries()[0], state,
                              *mapped.memory, budget, {kBias});
  if (run.outcome != Outcome::completed || run.stop != SsaStop::returned ||
      mapped.memory->Regions().size() != 3)
    return 2;
  std::cout << "{\"retired_access\":true,\"dead_nodes\":3,\"registers\":[";
  for (unsigned index = 0; index < 31; ++index) {
    if (index) std::cout << ',';
    std::cout << state.cells[index].value.word(0);
  }

  std::uint64_t nzcv = 0;
  for (unsigned index = 32; index < 36; ++index)
    nzcv |= state.cells[index].value.word(0) << (63 - index);
  std::cout << "],\"nzcv\":" << nzcv << ",\"sp\":" << state.cells[31].value.word(0)
            << ",\"pc\":" << run.runtime_pc << ",\"memory_unchanged\":"
            << (mapped.memory->Regions()[0].bytes ==
                            std::vector<std::uint8_t>(code.begin(), code.end()) &&
                        mapped.memory->Regions()[1].bytes ==
                            std::vector<std::uint8_t>(data.begin(), data.end()) &&
                        mapped.memory->Regions()[2].bytes == std::vector<std::uint8_t>(4096)
                    ? "true"
                    : "false")
            << "}\n";
  return 0;
}

int RunBoundedTableProbe(std::uint64_t input, unsigned mode) {
  Budget budget({10000000, 10000000});
  auto built = DecodedBoundedTableFunction(budget);
  if (!built) return 2;
  std::optional<ir::SsaHandle> guard;
  std::optional<ir::SsaHandle> table;
  std::optional<ir::ValueId> load;
  for (std::size_t slot = 0; slot < built->graph.slots(); ++slot) {
    const auto handle = built->graph.Handle(slot);
    if (!handle) continue;
    const auto* block = built->graph.Get(*handle);
    if (block->address == 0x100) guard = handle;
    if (block->address == 0x108) {
      table = handle;
      for (ir::ValueId id = 0; id < block->nodes.size(); ++id)
        if (block->nodes[id].op == ir::Op::load) load = id;
    }
  }

  if (!guard || !table || !load) return 2;
  const auto* guard_block = built->graph.Get(*guard);
  std::optional<std::uint32_t> edge;
  for (std::uint32_t index = 0; index < guard_block->edges.size(); ++index)
    if (guard_block->edges[index].target_block == table) edge = index;
  if (!edge) return 2;
  const auto bound = analysis::ProveSsaDirectIndexBound(built->graph, built->sources,
                                                        ir::SsaEntryScope::closed_population,
                                                        *guard, *edge, *table, budget);
  if (!bound.fact) return 2;
  const auto address = analysis::ProveSsaBoundedTableAddress(built->graph, built->sources,
                                                             *bound.fact, *load, budget);
  if (!address.fact) return 2;
  constexpr std::array<std::uint8_t, 32> rows{11, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0,
                                              33, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0};
  const ir::ConstantImageRange range{0x2100, rows};
  const ir::ImageFacts facts{std::span(&range, 1), {}, true, {}};
  auto folded = recovery::ProposeBoundedTableLoads(built->graph, *address.fact, built->sources,
                                                   facts, {mode != 1, true, true}, budget);
  if (mode == 1) {
    if (folded.provisional || folded.reason != recovery::ConstantLoadRefusal::not_nonfaulting)
      return 2;
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  if (!folded.provisional || folded.journal.size() != 1) return 2;
  auto candidate = std::move(*folded.provisional);
  const ir::SsaHandle table_handle{candidate.arena(), table->slot, table->generation};
  if (mode == 2) {
    if (!candidate.Update(table_handle,
                          [](auto& block) {
                            for (auto& row : block.constant_loads[0].table_bytes) ++row[0];
                          }) ||
        ir::ValidateSsa(candidate, budget) != ir::SsaDecline::none)
      return 2;
  }

  if (mode == 3) {
    // A guard edited to let one more index through leaves a fold whose rows
    // no longer cover it, which the graph's own recheck refuses.
    const ir::SsaHandle guard_handle{candidate.arena(), guard->slot, guard->generation};
    if (!candidate.Update(guard_handle, [](auto& block) { block.nodes[1].immediate = 5; }) ||
        ir::ValidateSsa(candidate, budget) != ir::SsaDecline::invalid_graph)
      return 2;
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  const auto live = analysis::ProveSsaNodeLiveness(candidate, budget);
  if (!live.facts) return 2;
  auto retired = recovery::ProposeDeadPureNodes(candidate, *live.facts, built->sources, budget);
  if (!retired.provisional || retired.journal.empty()) return 2;
  const auto dead_nodes = retired.journal.size();
  candidate = std::move(*retired.provisional);
  std::array<std::uint8_t, 4096> code{};
  const auto brk = Bytes(0xd4200000);
  for (std::size_t offset = 0; offset < code.size(); offset += 4)
    std::copy(brk.begin(), brk.end(), code.begin() + offset);
  constexpr std::array<std::uint32_t, 6> words{0xf100101f, 0x54000082, 0x1000ffc1,
                                               0xf8607820, 0xd65f03c0, 0xd65f03c0};
  for (std::size_t index = 0; index < words.size(); ++index) {
    const auto bytes = Bytes(words[index]);
    std::copy(bytes.begin(), bytes.end(), code.begin() + index * 4);
  }

  std::array<std::uint8_t, 4096> data{};
  std::copy(rows.begin(), rows.end(), data.begin());
  const std::array<std::uint8_t, 4096> stack{};
  const std::array<RegionInput, 3> regions{{{kCode, code, true, false},
                                            {kCode + 8192, data, true, false},
                                            {0x300000000, stack, true, true}}};
  auto mapped = Memory::Create(regions, budget);
  if (!mapped.memory) return 2;
  State state;
  for (unsigned index = 0; index < 36; ++index) {
    const auto value = index == 0    ? input
                       : index == 30 ? kLanding
                       : index == 31 ? 0x300000800ULL
                       : index >= 32 ? 0ULL
                                     : index + 3ULL;
    auto bits = BitVector::from_u64(index < 32 ? 64 : 1, value, 64, budget);
    if (!bits) return 2;
    state.cells.push_back({index, std::move(*bits)});
  }

  const auto run = ExecuteSsa(candidate, built->sources, candidate.entries()[0], state,
                              *mapped.memory, budget, {kBias});
  if (run.outcome != Outcome::completed || run.stop != SsaStop::returned ||
      mapped.memory->Regions().size() != 3)
    return 2;
  std::cout << "{\"retired_access\":true,\"dead_nodes\":" << dead_nodes << ",\"registers\":[";
  for (unsigned index = 0; index < 31; ++index) {
    if (index) std::cout << ',';
    std::cout << state.cells[index].value.word(0);
  }

  std::uint64_t nzcv = 0;
  for (unsigned index = 32; index < 36; ++index)
    nzcv |= state.cells[index].value.word(0) << (63 - index);
  std::cout << "],\"nzcv\":" << nzcv << ",\"sp\":" << state.cells[31].value.word(0)
            << ",\"pc\":" << run.runtime_pc << ",\"memory_unchanged\":"
            << (mapped.memory->Regions()[0].bytes ==
                            std::vector<std::uint8_t>(code.begin(), code.end()) &&
                        mapped.memory->Regions()[1].bytes ==
                            std::vector<std::uint8_t>(data.begin(), data.end()) &&
                        mapped.memory->Regions()[2].bytes == std::vector<std::uint8_t>(4096)
                    ? "true"
                    : "false")
            << "}\n";
  return 0;
}

int RunTwoBoundedTableProbe(std::uint64_t input, unsigned mode) {
  Budget budget({10000000, 10000000});
  auto built = DecodedTwoBoundedTableFunction(budget);
  if (!built) return 2;
  std::optional<ir::SsaHandle> guard;
  std::optional<ir::SsaHandle> table;
  std::vector<ir::ValueId> loads;
  for (std::size_t slot = 0; slot < built->graph.slots(); ++slot) {
    const auto handle = built->graph.Handle(slot);
    if (!handle) continue;
    const auto* block = built->graph.Get(*handle);
    if (block->address == 0x100) guard = handle;
    if (block->address == 0x108) {
      table = handle;
      for (ir::ValueId id = 0; id < block->nodes.size(); ++id)
        if (block->nodes[id].op == ir::Op::load) loads.push_back(id);
    }
  }

  if (!guard || !table || loads.size() != 2) return 2;
  std::optional<std::uint32_t> edge;
  const auto* guard_block = built->graph.Get(*guard);
  for (std::uint32_t index = 0; index < guard_block->edges.size(); ++index)
    if (guard_block->edges[index].target_block == table) edge = index;
  if (!edge) return 2;
  constexpr std::array<std::uint8_t, 32> rows{11, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0,
                                              33, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0};
  const ir::ConstantImageRange range{0x2100, rows};
  const ir::ImageFacts facts{std::span(&range, 1), {}, true, {}};
  auto candidate = std::move(built->graph);
  for (std::size_t load_index = 0; load_index < loads.size(); ++load_index) {
    const auto load = loads[load_index];
    const auto current_guard = candidate.Handle(guard->slot);
    const auto current_table = candidate.Handle(table->slot);
    if (!current_guard || !current_table) return 2;
    const auto bound = analysis::ProveSsaDirectIndexBound(
        candidate, built->sources, ir::SsaEntryScope::closed_population, *current_guard, *edge,
        *current_table, budget);
    if (!bound.fact) return 2;
    const auto address =
        analysis::ProveSsaBoundedTableAddress(candidate, built->sources, *bound.fact, load, budget);
    if (!address.fact) return 2;
    auto folded =
        recovery::ProposeBoundedTableLoads(candidate, *address.fact, built->sources, facts,
                                           {mode != 1 || load_index == 0, true, true}, budget);
    if (mode == 1 && load_index == 1) {
      if (folded.provisional || folded.reason != recovery::ConstantLoadRefusal::not_nonfaulting)
        return 2;
      std::cout << "{\"declined\":true}\n";
      return 0;
    }

    if (!folded.provisional || folded.journal.size() != 1) return 2;
    candidate = std::move(*folded.provisional);
  }

  const auto current_table = candidate.Handle(table->slot);
  if (!current_table) return 2;
  if ((mode == 2 || mode == 3) && (!candidate.Update(*current_table, [mode](auto& block) {
        auto& fold = mode == 2 ? block.constant_loads.back() : block.constant_loads.front();
        for (auto& row : fold.table_bytes) ++row[0];
      }) || ir::ValidateSsaWithSources(candidate, built->sources, budget) != ir::SsaDecline::none))
    return 2;
  const auto live = analysis::ProveSsaNodeLiveness(candidate, budget);
  if (!live.facts) return 2;
  auto retired = recovery::ProposeDeadPureNodes(candidate, *live.facts, built->sources, budget);
  if (!retired.provisional || retired.journal.empty()) return 2;
  candidate = std::move(*retired.provisional);
  std::array<std::uint8_t, 4096> code{};
  const auto brk = Bytes(0xd4200000);
  for (std::size_t offset = 0; offset < code.size(); offset += 4)
    std::copy(brk.begin(), brk.end(), code.begin() + offset);
  constexpr std::array<std::uint32_t, 7> words{0xf100101f, 0x540000a2, 0x1000ffc1, 0xf8607823,
                                               0xf8607822, 0xd65f03c0, 0xd65f03c0};
  for (std::size_t index = 0; index < words.size(); ++index) {
    const auto bytes = Bytes(words[index]);
    std::copy(bytes.begin(), bytes.end(), code.begin() + index * 4);
  }

  std::array<std::uint8_t, 4096> data{};
  std::copy(rows.begin(), rows.end(), data.begin());
  const std::array<std::uint8_t, 4096> stack{};
  const std::array<RegionInput, 3> regions{{{kCode, code, true, false},
                                            {kCode + 8192, data, true, false},
                                            {0x300000000, stack, true, true}}};
  auto mapped = Memory::Create(regions, budget);
  if (!mapped.memory) return 2;
  State state;
  for (unsigned index = 0; index < 36; ++index) {
    const auto value = index == 0    ? input
                       : index == 30 ? kLanding
                       : index == 31 ? 0x300000800ULL
                       : index >= 32 ? 0ULL
                                     : index + 3ULL;
    auto bits = BitVector::from_u64(index < 32 ? 64 : 1, value, 64, budget);
    if (!bits) return 2;
    state.cells.push_back({index, std::move(*bits)});
  }

  const auto run = ExecuteSsa(candidate, built->sources, candidate.entries()[0], state,
                              *mapped.memory, budget, {kBias});
  if (run.outcome != Outcome::completed || run.stop != SsaStop::returned ||
      mapped.memory->Regions().size() != 3)
    return 2;
  std::cout << "{\"folds\":2,\"dead_nodes\":" << retired.journal.size() << ",\"registers\":[";
  for (unsigned index = 0; index < 31; ++index) {
    if (index) std::cout << ',';
    std::cout << state.cells[index].value.word(0);
  }

  std::uint64_t nzcv = 0;
  for (unsigned index = 32; index < 36; ++index)
    nzcv |= state.cells[index].value.word(0) << (63 - index);
  std::cout << "],\"nzcv\":" << nzcv << ",\"sp\":" << state.cells[31].value.word(0)
            << ",\"pc\":" << run.runtime_pc << ",\"memory_unchanged\":"
            << (mapped.memory->Regions()[0].bytes ==
                            std::vector<std::uint8_t>(code.begin(), code.end()) &&
                        mapped.memory->Regions()[1].bytes ==
                            std::vector<std::uint8_t>(data.begin(), data.end()) &&
                        mapped.memory->Regions()[2].bytes == std::vector<std::uint8_t>(4096)
                    ? "true"
                    : "false")
            << "}\n";
  return 0;
}

int RunPhiConstantProbe(std::uint64_t input, unsigned mode, bool computed = false,
                        bool sccp = false) {
  Budget budget({10000000, 10000000});
  auto built = sccp       ? DecodedSccpFunction(budget)
               : computed ? DecodedComputedPhiFunction(budget)
                          : DecodedPhiConstantFunction(budget);
  if (!built) return 2;
  auto reachable = analysis::ProveSsaReachability(built->graph, built->sources,
                                                  ir::SsaEntryScope::closed_population, budget);
  if (!reachable.facts) return 2;
  std::optional<ir::SsaGraph> candidate;
  ir::SsaHandle changed_block{};
  ir::ValueId changed_node = 0;
  std::size_t edits = 0;
  std::size_t retired_edges = 0, removed_blocks = 0;
  if (sccp) {
    auto facts = analysis::ProveSsaSccp(built->graph, ir::SsaEntryScope::closed_population,
                                        built->sources, budget);
    if (!facts.facts) return 2;
    if (mode == 1) {
      auto open = *facts.facts;
      open.entry_scope = ir::SsaEntryScope::discovered_only;
      const auto refused = recovery::ProposeSsaSccpFold(built->graph, open, built->sources, budget);
      if (refused.provisional || refused.reason != recovery::SsaConstantFoldRefusal::stale_proof)
        return 2;
      std::cout << "{\"declined\":true}\n";
      return 0;
    }

    auto folded = recovery::ProposeSsaSccpFold(built->graph, *facts.facts, built->sources, budget);
    if (!folded.provisional) return 2;
    edits = folded.journal.size();
    const auto join = std::find_if(folded.journal.begin(), folded.journal.end(),
                                   [](const auto& edit) { return edit.value == 23; });
    if (join == folded.journal.end()) return 2;
    changed_block = join->result_block;
    changed_node = join->node;
    candidate = std::move(folded.provisional);
    auto after_facts = analysis::ProveSsaSccp(*candidate, ir::SsaEntryScope::closed_population,
                                              built->sources, budget);
    if (!after_facts.facts) return 2;
    if (mode == 3) {
      auto forged = *after_facts.facts;
      const auto entry = candidate->entries().front();
      for (auto& edge : forged.edges[entry.slot]) edge ^= 1;
      const auto refused =
          recovery::ProposeSsaBranchRetirement(*candidate, forged, built->sources, budget);
      if (refused.provisional ||
          refused.reason != recovery::SsaBranchRetirementRefusal::stale_proof)
        return 2;
      std::cout << "{\"declined\":true}\n";
      return 0;
    }

    auto retired = recovery::ProposeSsaBranchRetirement(*candidate, *after_facts.facts,
                                                        built->sources, budget);
    if (!retired.provisional || retired.journal.size() != 1) return 2;
    retired_edges = retired.journal.size();
    candidate = std::move(retired.provisional);
    auto final_reachable = analysis::ProveSsaReachability(
        *candidate, built->sources, ir::SsaEntryScope::closed_population, budget);
    if (!final_reachable.facts) return 2;
    auto cleaned = recovery::ProposeUnreachableBlocks(*candidate, *final_reachable.facts,
                                                      built->sources, budget);
    if (!cleaned.provisional) return 2;
    removed_blocks = std::count_if(
        cleaned.journal.begin(), cleaned.journal.end(),
        [](const auto& edit) { return edit.kind == recovery::SsaDceEditKind::removed_block; });
    if (removed_blocks != 1) return 2;
    candidate = std::move(cleaned.provisional);
    const auto remapped = candidate->Handle(changed_block.slot);
    if (!remapped) return 2;
    changed_block = *remapped;
  } else if (computed) {
    auto facts =
        analysis::ProveSsaConstants(built->graph, *reachable.facts, built->sources, budget);
    if (!facts.facts) return 2;
    if (mode == 1) {
      auto open = *reachable.facts;
      open.entry_scope = ir::SsaEntryScope::discovered_only;
      const auto refused = recovery::ProposeSsaConstantFold(built->graph, open, *facts.facts,
                                                            built->sources, budget);
      if (refused.provisional || refused.reason != recovery::SsaConstantFoldRefusal::stale_proof)
        return 2;
      std::cout << "{\"declined\":true}\n";
      return 0;
    }

    auto folded = recovery::ProposeSsaConstantFold(built->graph, *reachable.facts, *facts.facts,
                                                   built->sources, budget);
    if (!folded.provisional || folded.journal.size() != 2) return 2;
    edits = folded.journal.size();
    const auto join = std::find_if(folded.journal.begin(), folded.journal.end(),
                                   [](const auto& edit) { return edit.value == 12; });
    if (join == folded.journal.end()) return 2;
    changed_block = join->result_block;
    changed_node = join->node;
    candidate = std::move(folded.provisional);
  } else {
    auto facts =
        analysis::ProveSsaPhiConstants(built->graph, *reachable.facts, built->sources, budget);
    if (!facts.facts || facts.facts->constants.size() != 1) return 2;
    if (mode == 1) {
      auto open = *reachable.facts;
      open.entry_scope = ir::SsaEntryScope::discovered_only;
      const auto refused = recovery::ProposeSsaPhiConstantFold(built->graph, open, *facts.facts,
                                                               built->sources, budget);
      if (refused.provisional || refused.reason != recovery::SsaPhiFoldRefusal::stale_proof)
        return 2;
      std::cout << "{\"declined\":true}\n";
      return 0;
    }

    auto folded = recovery::ProposeSsaPhiConstantFold(built->graph, *reachable.facts, *facts.facts,
                                                      built->sources, budget);
    if (!folded.provisional || folded.journal.size() != 1) return 2;
    edits = folded.journal.size();
    changed_block = folded.journal[0].result_block;
    changed_node = folded.journal[0].node;
    candidate = std::move(folded.provisional);
  }

  if (mode == 2 && (!candidate->Update(changed_block, [&](auto& block) {
        block.nodes[changed_node].immediate = 13;
      }) || ir::ValidateSsa(*candidate, budget) != ir::SsaDecline::none))
    return 2;
  std::array<std::uint8_t, 4096> code{};
  const auto brk = Bytes(0xd4200000);
  for (std::size_t offset = 0; offset < code.size(); offset += 4)
    std::copy(brk.begin(), brk.end(), code.begin() + offset);
  const std::vector<std::pair<std::size_t, std::uint32_t>> words =
      sccp ? std::vector<std::pair<std::size_t, std::uint32_t>>{{0, 0xb40000bf},  {4, 0xd2800148},
                                                                {8, 0x91000508},  {12, 0x14000004},
                                                                {20, 0xd28002c8}, {24, 0x14000001},
                                                                {28, 0x91000500}, {32, 0xd65f03c0}}
      : computed
          ? std::vector<std::pair<std::size_t, std::uint32_t>>{{0, 0xb40000a0},  {4, 0xd2800148},
                                                               {8, 0x91000508},  {12, 0x14000004},
                                                               {20, 0xd2800168}, {24, 0x14000001},
                                                               {28, 0x91000500}, {32, 0xd65f03c0}}
          : std::vector<std::pair<std::size_t, std::uint32_t>>{
                {0, 0xb4000080},  {4, 0xd2800168},  {8, 0x14000004}, {16, 0xd2800168},
                {20, 0x14000001}, {24, 0x91000500}, {28, 0xd65f03c0}};
  for (const auto [offset, word] : words) {
    const auto bytes = Bytes(word);
    std::copy(bytes.begin(), bytes.end(), code.begin() + offset);
  }

  const std::array<std::uint8_t, 4096> stack{};
  const std::array<RegionInput, 2> regions{
      {{kCode, code, true, false}, {0x300000000, stack, true, true}}};
  auto mapped = Memory::Create(regions, budget);
  if (!mapped.memory) return 2;
  State state;
  for (unsigned index = 0; index < 36; ++index) {
    const auto value = index == 0    ? input
                       : index == 30 ? kLanding
                       : index == 31 ? 0x300000800ULL
                       : index >= 32 ? 0ULL
                                     : index + 3ULL;
    auto bits = BitVector::from_u64(index < 32 ? 64 : 1, value, 64, budget);
    if (!bits) return 2;
    state.cells.push_back({index, std::move(*bits)});
  }

  const auto run = ExecuteSsa(*candidate, built->sources, candidate->entries()[0], state,
                              *mapped.memory, budget, {kBias});
  if (run.outcome != Outcome::completed || run.stop != SsaStop::returned ||
      mapped.memory->Regions().size() != 2)
    return 2;
  std::cout << "{\"edits\":" << edits << ",\"retired_edges\":" << retired_edges
            << ",\"removed_blocks\":" << removed_blocks << ",\"registers\":[";
  for (unsigned index = 0; index < 31; ++index) {
    if (index) std::cout << ',';
    std::cout << state.cells[index].value.word(0);
  }

  std::uint64_t nzcv = 0;
  for (unsigned index = 32; index < 36; ++index)
    nzcv |= state.cells[index].value.word(0) << (63 - index);
  std::cout << "],\"nzcv\":" << nzcv << ",\"sp\":" << state.cells[31].value.word(0)
            << ",\"pc\":" << run.runtime_pc << ",\"memory_unchanged\":"
            << (mapped.memory->Regions()[0].bytes ==
                            std::vector<std::uint8_t>(code.begin(), code.end()) &&
                        mapped.memory->Regions()[1].bytes == std::vector<std::uint8_t>(4096)
                    ? "true"
                    : "false")
            << "}\n";
  return 0;
}

int RunReadConditionProbe(std::uint64_t input, unsigned mode) {
  Budget budget({10000000, 10000000});
  auto fixture = DecodedReadConditionFunction(budget);
  if (!fixture) return 2;
  auto proved = analysis::ProveSsaSccp(fixture->graph, ir::SsaEntryScope::closed_population,
                                       fixture->sources, budget);
  if (!proved.facts) return 2;
  if (mode == 1) {
    auto open = *proved.facts;
    open.entry_scope = ir::SsaEntryScope::discovered_only;
    const auto refused =
        recovery::ProposeSsaSccpReadFold(fixture->graph, open, fixture->sources, budget);
    if (refused.provisional || refused.reason != recovery::SsaConstantFoldRefusal::stale_proof)
      return 2;
    std::cout << "{\"declined\":true}\n";
    return 0;
  }

  auto folded =
      recovery::ProposeSsaSccpReadFold(fixture->graph, *proved.facts, fixture->sources, budget);
  if (!folded.provisional) return 2;
  const auto read =
      std::find_if(folded.journal.begin(), folded.journal.end(), [](const auto& edit) {
        return edit.original.op == ir::Op::read && edit.original.storage == 0 && edit.value == 0;
      });
  if (read == folded.journal.end() || !read->removed_read) return 2;
  if (mode == 2 && (!folded.provisional->Update(read->result_block, [&](auto& edited) {
        edited.nodes[read->node].immediate = 1;
      }) || ir::ValidateSsa(*folded.provisional, budget) != ir::SsaDecline::none))
    return 2;
  std::array<std::uint8_t, 4096> code{};
  const auto brk = Bytes(0xd4200000);
  for (std::size_t offset = 0; offset < code.size(); offset += 4)
    std::copy(brk.begin(), brk.end(), code.begin() + offset);
  for (const auto [offset, word] :
       std::array<std::pair<std::size_t, std::uint32_t>, 8>{{{0, 0xd2800000},
                                                             {4, 0x14000002},
                                                             {12, 0xaa0003e1},
                                                             {16, 0xb4000060},
                                                             {20, 0xd2800160},
                                                             {24, 0xd65f03c0},
                                                             {28, 0xd28002c0},
                                                             {32, 0xd65f03c0}}}) {
    const auto bytes = Bytes(word);
    std::copy(bytes.begin(), bytes.end(), code.begin() + offset);
  }

  const std::array<std::uint8_t, 4096> stack{};
  const std::array<RegionInput, 2> regions{
      {{kCode, code, true, false}, {0x300000000, stack, true, true}}};
  auto mapped = Memory::Create(regions, budget);
  if (!mapped.memory) return 2;
  State state;
  for (unsigned index = 0; index < 36; ++index) {
    const auto value = index == 0    ? input
                       : index == 30 ? kLanding
                       : index == 31 ? 0x300000800ULL
                       : index >= 32 ? 0ULL
                                     : index + 3ULL;
    auto bits = BitVector::from_u64(index < 32 ? 64 : 1, value, 64, budget);
    if (!bits) return 2;
    state.cells.push_back({index, std::move(*bits)});
  }

  const auto run =
      ExecuteSsa(*folded.provisional, fixture->sources, folded.provisional->entries()[0], state,
                 *mapped.memory, budget, {kBias});
  if (run.outcome != Outcome::completed || run.stop != SsaStop::returned) return 2;
  std::cout << "{\"read_folds\":1,\"registers\":[";
  for (unsigned index = 0; index < 31; ++index) {
    if (index) std::cout << ',';
    std::cout << state.cells[index].value.word(0);
  }

  std::uint64_t nzcv = 0;
  for (unsigned index = 32; index < 36; ++index)
    nzcv |= state.cells[index].value.word(0) << (63 - index);
  std::cout << "],\"nzcv\":" << nzcv << ",\"sp\":" << state.cells[31].value.word(0)
            << ",\"pc\":" << run.runtime_pc << ",\"memory_unchanged\":"
            << (mapped.memory->Regions()[0].bytes ==
                            std::vector<std::uint8_t>(code.begin(), code.end()) &&
                        mapped.memory->Regions()[1].bytes == std::vector<std::uint8_t>(4096)
                    ? "true"
                    : "false")
            << "}\n";
  return 0;
}

}  // namespace nyx::eval

int main(int argc, char** argv) {
  if (argc == 4 && std::string_view(argv[1]) == "--wide-copy-probe") {
    errno = 0;
    char* end = nullptr;
    const auto input = std::strtoull(argv[2], &end, 0);
    if (errno || !end || *end ||
        (std::string_view(argv[3]) != "0" && std::string_view(argv[3]) != "1"))
      return 2;
    return nyx::eval::RunWideCopyProbe(input, static_cast<unsigned>(argv[3][0] - '0'));
  }

  if (argc == 4 && std::string_view(argv[1]) == "--internal-call-probe") {
    errno = 0;
    char* end = nullptr;
    const auto input = std::strtoull(argv[2], &end, 0);
    if (errno || !end || *end ||
        (std::string_view(argv[3]) != "0" && std::string_view(argv[3]) != "1" &&
         std::string_view(argv[3]) != "2"))
      return 2;
    return nyx::eval::RunInternalCallProbe(input, static_cast<unsigned>(argv[3][0] - '0'));
  }

  if (argc == 4 && std::string_view(argv[1]) == "--read-condition-probe") {
    errno = 0;
    char* end = nullptr;
    const auto input = std::strtoull(argv[2], &end, 0);
    if (errno || !end || *end ||
        (std::string_view(argv[3]) != "0" && std::string_view(argv[3]) != "1" &&
         std::string_view(argv[3]) != "2"))
      return 2;
    return nyx::eval::RunReadConditionProbe(input, std::strtoul(argv[3], nullptr, 10));
  }

  if (argc == 4 && std::string_view(argv[1]) == "--sccp-probe") {
    errno = 0;
    char* end = nullptr;
    const auto input = std::strtoull(argv[2], &end, 0);
    if (errno || !end || *end ||
        (std::string_view(argv[3]) != "0" && std::string_view(argv[3]) != "1" &&
         std::string_view(argv[3]) != "2" && std::string_view(argv[3]) != "3"))
      return 2;
    return nyx::eval::RunPhiConstantProbe(input, std::strtoul(argv[3], nullptr, 10), false, true);
  }

  if (argc == 4 && std::string_view(argv[1]) == "--computed-phi-probe") {
    errno = 0;
    char* end = nullptr;
    const auto input = std::strtoull(argv[2], &end, 0);
    if (errno || !end || *end ||
        (std::string_view(argv[3]) != "0" && std::string_view(argv[3]) != "1" &&
         std::string_view(argv[3]) != "2"))
      return 2;
    return nyx::eval::RunPhiConstantProbe(input, std::strtoul(argv[3], nullptr, 10), true);
  }

  if (argc == 4 && std::string_view(argv[1]) == "--phi-constant-probe") {
    errno = 0;
    char* end = nullptr;
    const auto input = std::strtoull(argv[2], &end, 0);
    if (errno || !end || *end ||
        (std::string_view(argv[3]) != "0" && std::string_view(argv[3]) != "1" &&
         std::string_view(argv[3]) != "2"))
      return 2;
    return nyx::eval::RunPhiConstantProbe(input, std::strtoul(argv[3], nullptr, 10));
  }

  if (argc == 4 && std::string_view(argv[1]) == "--two-bounded-table-probe") {
    errno = 0;
    char* end = nullptr;
    const auto input = std::strtoull(argv[2], &end, 0);
    if (errno || !end || *end ||
        (std::string_view(argv[3]) != "0" && std::string_view(argv[3]) != "1" &&
         std::string_view(argv[3]) != "2" && std::string_view(argv[3]) != "3"))
      return 2;
    return nyx::eval::RunTwoBoundedTableProbe(input, std::strtoul(argv[3], nullptr, 10));
  }

  if (argc == 4 && std::string_view(argv[1]) == "--bounded-table-probe") {
    errno = 0;
    char* end = nullptr;
    const auto input = std::strtoull(argv[2], &end, 0);
    if (errno || !end || *end ||
        (std::string_view(argv[3]) != "0" && std::string_view(argv[3]) != "1" &&
         std::string_view(argv[3]) != "2" && std::string_view(argv[3]) != "3"))
      return 2;
    return nyx::eval::RunBoundedTableProbe(input, std::strtoul(argv[3], nullptr, 10));
  }

  if (argc == 4 && std::string_view(argv[1]) == "--selected-load-probe") {
    errno = 0;
    char* end = nullptr;
    const auto input = std::strtoull(argv[2], &end, 0);
    if (errno || !end || *end ||
        (std::string_view(argv[3]) != "0" && std::string_view(argv[3]) != "1" &&
         std::string_view(argv[3]) != "2" && std::string_view(argv[3]) != "3" &&
         std::string_view(argv[3]) != "4"))
      return 2;
    return nyx::eval::RunSelectedLoadProbe(input, std::strtoul(argv[3], nullptr, 10));
  }

  if (argc == 5 && std::string_view(argv[1]) == "--literal-load-probe") {
    errno = 0;
    char* end = nullptr;
    const auto input = std::strtoull(argv[2], &end, 0);
    if (errno || !end || *end ||
        (std::string_view(argv[3]) != "0" && std::string_view(argv[3]) != "1") ||
        (std::string_view(argv[4]) != "0" && std::string_view(argv[4]) != "1" &&
         std::string_view(argv[4]) != "2" && std::string_view(argv[4]) != "3"))
      return 2;
    return nyx::eval::RunLiteralLoadProbe(input, std::string_view(argv[3]) == "1",
                                          std::string_view(argv[4]) == "3"   ? 3
                                          : std::string_view(argv[4]) == "2" ? 2
                                          : std::string_view(argv[4]) == "1" ? 1
                                                                             : 0);
  }

  if (argc == 4 && std::string_view(argv[1]) == "--frame-promotion-probe") {
    errno = 0;
    char* end = nullptr;
    const auto input = std::strtoull(argv[2], &end, 0);
    if (errno || !end || *end ||
        (std::string_view(argv[3]) != "0" && std::string_view(argv[3]) != "1" &&
         std::string_view(argv[3]) != "2"))
      return 2;
    return nyx::eval::RunFramePromotionProbe(input, std::string_view(argv[3]) == "2"   ? 2
                                                    : std::string_view(argv[3]) == "1" ? 1
                                                                                       : 0);
  }

  if (argc == 4 && std::string_view(argv[1]) == "--frame-dead-probe") {
    errno = 0;
    char* end = nullptr;
    const auto input = std::strtoull(argv[2], &end, 0);
    if (errno || !end || *end ||
        (std::string_view(argv[3]) != "0" && std::string_view(argv[3]) != "1" &&
         std::string_view(argv[3]) != "2"))
      return 2;
    return nyx::eval::RunFrameDeadProbe(input, std::string_view(argv[3]) == "2"   ? 2
                                               : std::string_view(argv[3]) == "1" ? 1
                                                                                  : 0);
  }

  if (argc == 5 && std::string_view(argv[1]) == "--pure-dce-probe") {
    errno = 0;
    char* end = nullptr;
    const auto left = std::strtoull(argv[2], &end, 0);
    if (errno || !end || *end) return 2;
    errno = 0;
    const auto right = std::strtoull(argv[3], &end, 0);
    if (errno || !end || *end ||
        (std::string_view(argv[4]) != "0" && std::string_view(argv[4]) != "1"))
      return 2;
    return nyx::eval::RunPureDceProbe(left, right, std::string_view(argv[4]) == "1");
  }

  if (argc == 4 && std::string_view(argv[1]) == "--dce-probe") {
    errno = 0;
    char* end = nullptr;
    const auto input = std::strtoull(argv[2], &end, 0);
    if (errno || !end || *end ||
        (std::string_view(argv[3]) != "0" && std::string_view(argv[3]) != "1" &&
         std::string_view(argv[3]) != "2"))
      return 2;
    return nyx::eval::RunSsaProbe(input,
                                  std::string_view(argv[3]) == "2"   ? 2
                                  : std::string_view(argv[3]) == "1" ? 1
                                                                     : 0,
                                  true);
  }

  if (argc == 5 && std::string_view(argv[1]) == "--mba-probe") {
    errno = 0;
    char* end = nullptr;
    const auto left = std::strtoull(argv[2], &end, 0);
    if (errno || !end || *end) return 2;
    errno = 0;
    const auto right = std::strtoull(argv[3], &end, 0);
    if (errno || !end || *end ||
        (std::string_view(argv[4]) != "0" && std::string_view(argv[4]) != "1"))
      return 2;
    return nyx::eval::RunMbaProbe(left, right, std::string_view(argv[4]) == "1");
  }

  if (argc == 6 && (std::string_view(argv[1]) == "--canonical-mba-probe" ||
                    std::string_view(argv[1]) == "--affine-mba-probe" ||
                    std::string_view(argv[1]) == "--signed-affine-mba-probe")) {
    std::array<std::uint64_t, 3> values{};
    for (unsigned i = 0; i < values.size(); ++i) {
      errno = 0;
      char* end = nullptr;
      values[i] = std::strtoull(argv[i + 2], &end, 0);
      if (errno || !end || *end) return 2;
    }

    const std::string_view mode = argv[1], mutate = argv[5];
    if (mutate != "0" && mutate != "1" && (mutate != "2" || mode != "--signed-affine-mba-probe"))
      return 2;
    using nyx::eval::CanonicalProbe;
    return nyx::eval::RunCanonicalMbaProbe(
        values[0], values[1], values[2], static_cast<unsigned>(mutate[0] - '0'),
        mode == "--canonical-mba-probe" ? CanonicalProbe::boolean
        : mode == "--affine-mba-probe"  ? CanonicalProbe::affine
                                        : CanonicalProbe::signed_affine);
  }

  if (argc == 4 && std::string_view(argv[1]) == "--probe") {
    errno = 0;
    char* end = nullptr;
    const auto input = std::strtoull(argv[2], &end, 0);
    if (errno || !end || *end ||
        (std::string_view(argv[3]) != "0" && std::string_view(argv[3]) != "1"))
      return 2;
    return nyx::eval::RunSsaProbe(input, std::string_view(argv[3]) == "1");
  }

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
