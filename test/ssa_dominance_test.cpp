#include <gtest/gtest.h>

#include "nyx/analysis/ssa/dominance.hpp"

namespace nyx::analysis {
namespace {
Budget Plenty() { return Budget({1000000, 10000000}); }

std::vector<std::uint8_t> Bytes(std::uint32_t word) {
  return {static_cast<std::uint8_t>(word), static_cast<std::uint8_t>(word >> 8),
          static_cast<std::uint8_t>(word >> 16), static_cast<std::uint8_t>(word >> 24)};
}

std::vector<ir::Group> Sources(bool join) {
  std::vector<ir::Group> sources;
  sources.emplace_back(
      0x100, Bytes(join ? 0x14000008 : 0x14000004),
      std::vector<ir::Node>{{ir::Op::image_address, 64, {}, join ? 0x120U : 0x110U}},
      std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
      ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}});
  sources.emplace_back(0x110, Bytes(0x14000004),
                       std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x120}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}});
  sources.emplace_back(0x120, Bytes(0xd65f03c0),
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 30}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}});
  return sources;
}

ir::SsaGraph Graph(bool join) {
  ir::SsaGraph graph;
  for (std::uint32_t slot = 0; slot < 3; ++slot) {
    ir::SsaBlock block{};
    block.address = 0x100 + slot * 0x10;
    block.source_groups = {block.address};
    block.source_bytes = {Bytes(slot == 2            ? 0xd65f03c0
                                : slot == 0 && !join ? 0x14000004
                                : slot == 0          ? 0x14000008
                                                     : 0x14000004)};
    block.original_sources = {slot};
    block.nodes = slot == 2
                      ? std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 30}}
                      : std::vector<ir::Node>{
                            {ir::Op::image_address, 64, {}, slot == 0 && !join ? 0x110U : 0x120U}};
    block.boundaries = {
        {0,
         1,
         {},
         ir::Transfer{
             slot == 2 ? ir::TransferKind::return_ : ir::TransferKind::jump, 0, {}, {}, {}}}};
    block.phis = {{30, 64, slot == 0 || (join && slot == 1), {}}};
    block.clobbers = {0};
    graph.Add(std::move(block));
  }

  const auto a = *graph.Handle(0), b = *graph.Handle(1), c = *graph.Handle(2);
  graph.Update(a, [&](auto& block) {
    block.exits = {{30, {ir::SsaValueKind::phi, a, 0}}};
    block.edges = {{ir::SsaEdgeKind::branch,
                    ir::SsaTargetKind::image_location,
                    join ? 0x120U : 0x110U,
                    join ? c : b,
                    {},
                    {},
                    {}}};
  });
  graph.Update(b, [&](auto& block) {
    if (!join) block.phis[0].incoming = {{a, {ir::SsaValueKind::phi, a, 0}}};
    block.exits = {{30, {ir::SsaValueKind::phi, b, 0}}};
    block.edges = {
        {ir::SsaEdgeKind::branch, ir::SsaTargetKind::image_location, 0x120, c, {}, {}, {}}};
  });
  graph.Update(c, [&](auto& block) {
    block.phis[0].incoming = join
                                 ? std::vector<ir::SsaPhiInput>{{a, {ir::SsaValueKind::phi, a, 0}},
                                                                {b, {ir::SsaValueKind::phi, b, 0}}}
                                 : std::vector<ir::SsaPhiInput>{{b, {ir::SsaValueKind::phi, b, 0}}};
    block.edges = {{ir::SsaEdgeKind::return_,
                    ir::SsaTargetKind::unknown,
                    0,
                    {},
                    {},
                    {},
                    {.return_leaves = true}}};
    block.reads = {{0, 0, {ir::SsaValueKind::phi, c, 0}}};
    block.exits = {{30, {ir::SsaValueKind::phi, c, 0}}};
  });
  graph.SetEntries(join ? std::vector{a, b} : std::vector{a});
  return graph;
}

bool Dominates(const ir::SsaDominanceFacts& facts, std::size_t dominator, std::size_t block) {
  return facts.dominates[dominator * facts.reachable.size() + block];
}

TEST(SsaDominance, ChainAndJoinHaveDifferentDominators) {
  auto budget = Plenty();
  auto chain = Graph(false);
  const auto chain_sources = Sources(false);
  ASSERT_EQ(ir::ValidateSsa(chain, budget), ir::SsaDecline::none);
  auto linear =
      ProveSsaDominance(chain, chain_sources, ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(linear.facts) << static_cast<int>(linear.reason);
  EXPECT_TRUE(Dominates(*linear.facts, 0, 2));
  EXPECT_TRUE(Dominates(*linear.facts, 1, 2));
  EXPECT_FALSE(Dominates(*linear.facts, 2, 1));
  EXPECT_EQ(ir::ValidateSsaDominanceFacts(chain, *linear.facts, chain_sources, budget),
            ir::SsaDecline::none);

  auto join = Graph(true);
  const auto join_sources = Sources(true);
  auto fork = ProveSsaDominance(join, join_sources, ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(fork.facts) << static_cast<int>(fork.reason);
  EXPECT_FALSE(Dominates(*fork.facts, 0, 2));
  EXPECT_FALSE(Dominates(*fork.facts, 1, 2));
  EXPECT_TRUE(Dominates(*fork.facts, 2, 2));
}

TEST(SsaDominance, RefusesFalseDominanceStaleProofAndOpenGraph) {
  auto graph = Graph(true);
  const auto sources = Sources(true);
  auto budget = Plenty();
  auto proof = ProveSsaDominance(graph, sources, ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(proof.facts);
  auto forged = *proof.facts;
  forged.dominates[0 * 3 + 2] = 1;
  EXPECT_EQ(ir::ValidateSsaDominanceFacts(graph, forged, sources, budget),
            ir::SsaDecline::invalid_graph);
  ASSERT_TRUE(graph.Update(*graph.Handle(0), [](auto& block) {
    block.nodes.push_back({ir::Op::constant, 64, {}, 1});
    block.boundaries[0].node_count = block.nodes.size();
  }));
  EXPECT_EQ(ir::ValidateSsaDominanceFacts(graph, *proof.facts, sources, budget),
            ir::SsaDecline::invalid_graph);
  auto open = ProveSsaDominance(graph, sources, ir::SsaEntryScope::discovered_only, budget);
  EXPECT_EQ(open.reason, SsaDominanceRefusal::open_entries);
  ASSERT_TRUE(graph.Update(*graph.Handle(0), [](auto& block) {
    block.edges = {{ir::SsaEdgeKind::opaque_unknown,
                    ir::SsaTargetKind::unknown,
                    0,
                    {},
                    {},
                    {},
                    {.unresolved_target = true}}};
  }));
  ASSERT_TRUE(graph.Update(*graph.Handle(2), [](auto& block) {
    block.phis[0].incoming.erase(block.phis[0].incoming.begin());
  }));
  auto incomplete = ProveSsaDominance(graph, sources, ir::SsaEntryScope::closed_population, budget);
  EXPECT_EQ(incomplete.reason, SsaDominanceRefusal::incomplete_successors);
}

TEST(SsaDominance, ConvergesOnReachableLoopAndChecksReachability) {
  auto graph = Graph(false);
  auto sources = Sources(false);
  sources[1] = ir::Group(0x110, Bytes(0x17fffffc),
                         std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x100}},
                         std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                         ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}});
  const auto a = *graph.Handle(0);
  ASSERT_TRUE(graph.Update(a, [&](auto& block) {
    block.phis[0].incoming = {{*graph.Handle(1), {ir::SsaValueKind::phi, *graph.Handle(1), 0}}};
  }));
  ASSERT_TRUE(graph.Update(*graph.Handle(1), [&](auto& block) {
    block.source_bytes[0] = Bytes(0x17fffffc);
    block.nodes[0].immediate = 0x100;
    block.edges[0].address = 0x100;
    block.edges[0].target_block = a;
  }));
  ASSERT_TRUE(graph.Update(*graph.Handle(2), [](auto& block) { block.phis[0].incoming.clear(); }));
  auto budget = Plenty();
  auto proof = ProveSsaDominance(graph, sources, ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(proof.facts) << static_cast<int>(proof.reason);
  EXPECT_EQ(proof.facts->reachable, (std::vector<std::uint8_t>{1, 1, 0}));
  EXPECT_TRUE(Dominates(*proof.facts, 0, 1));
  EXPECT_FALSE(Dominates(*proof.facts, 1, 0));
  EXPECT_FALSE(Dominates(*proof.facts, 2, 2));
  auto forged = *proof.facts;
  forged.reachable[2] = 1;
  EXPECT_EQ(ir::ValidateSsaDominanceFacts(graph, forged, sources, budget),
            ir::SsaDecline::invalid_graph);
}

TEST(SsaDominance, BudgetRefusalPublishesNoFact) {
  auto graph = Graph(false);
  const auto sources = Sources(false);
  auto full = Plenty();
  ASSERT_TRUE(ProveSsaDominance(graph, sources, ir::SsaEntryScope::closed_population, full).facts);
  const auto used = full.used();
  ASSERT_LT(used.work, 100000U);
  for (std::uint64_t cut = 0; cut < used.work; ++cut) {
    Budget limited({cut, used.bytes});
    const auto result =
        ProveSsaDominance(graph, sources, ir::SsaEntryScope::closed_population, limited);
    EXPECT_FALSE(result.facts) << cut;
    EXPECT_EQ(result.reason, SsaDominanceRefusal::resource_limit) << cut;
  }
}
}  // namespace
}  // namespace nyx::analysis
