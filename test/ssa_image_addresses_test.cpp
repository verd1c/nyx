#include <gtest/gtest.h>

#include "nyx/analysis/ssa/image_addresses.hpp"

namespace nyx::analysis {
namespace {
Budget Plenty() { return Budget({1000000, 10000000}); }

ir::SsaGraph Graph() {
  ir::SsaGraph graph;
  ir::SsaBlock block{};
  block.original_block = 0;
  block.address = 0x1000;
  block.source_groups = {0x1000};
  block.original_sources = {0};
  block.nodes = {{ir::Op::read, 64, {}, 0, 0},
                 {ir::Op::constant, 64, {}, 0},
                 {ir::Op::equal, 1, {0, 1}},
                 {ir::Op::constant, 64, {}, 0},
                 {ir::Op::constant, 64, {}, 1},
                 {ir::Op::select, 64, {2, 4, 3}},
                 {ir::Op::constant, 64, {}, 8},
                 {ir::Op::mul, 64, {5, 6}},
                 {ir::Op::image_address, 64, {}, 0x2000},
                 {ir::Op::add, 64, {8, 7}},
                 {ir::Op::load, 64, {9}}};
  block.boundaries = {{0, 11, {{1, 10}}, {}}};
  block.phis = {{0, 64, true, {}}, {1, 64, true, {}}};
  block.clobbers = {0, 0};
  const auto handle = graph.Add(std::move(block));
  graph.Update(handle, [&](auto& inserted) {
    inserted.reads = {{0, 0, {ir::SsaValueKind::phi, handle, 0}}};
    inserted.exits = {{0, {ir::SsaValueKind::phi, handle, 0}},
                      {1, {ir::SsaValueKind::node, handle, 10}}};
    inserted.edges = {{ir::SsaEdgeKind::return_, ir::SsaTargetKind::unknown, 0, {}, {}, {}, {}}};
  });
  graph.SetEntries({handle});
  return graph;
}

TEST(SsaImageAddresses, ProvesTwoLocationsAndRejectsForgedArmOrRevision) {
  auto graph = Graph();
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  auto result = ProveSsaSelectedImageAddresses(graph, budget);
  ASSERT_TRUE(result.facts);
  ASSERT_EQ(result.facts->selected.size(), 1);
  const auto& fact = result.facts->selected[0];
  EXPECT_EQ(fact.block, graph.entries()[0]);
  EXPECT_EQ(fact.load, 10U);
  EXPECT_EQ(fact.condition, 2U);
  EXPECT_EQ(fact.when_true, 0x2008U);
  EXPECT_EQ(fact.when_false, 0x2000U);
  EXPECT_EQ(ir::ValidateSsaImageAddressFacts(graph, *result.facts, budget), ir::SsaDecline::none);

  auto forged = *result.facts;
  ++forged.selected[0].when_true;
  EXPECT_EQ(ir::ValidateSsaImageAddressFacts(graph, forged, budget), ir::SsaDecline::invalid_graph);
  ASSERT_TRUE(graph.Update(graph.entries()[0], [](auto& block) { block.nodes[6].immediate = 16; }));
  EXPECT_EQ(ir::ValidateSsaImageAddressFacts(graph, *result.facts, budget),
            ir::SsaDecline::invalid_graph);
}

TEST(SsaImageAddresses, RechecksTheAddressByValueUnderItsCondition) {
  auto graph = Graph();
  auto budget = Plenty();

  // The address keeps its value but selects on what the condition negates,
  // with the arms swapped, as simplification leaves it. Fixing the condition
  // fixes what it negates too.
  ASSERT_TRUE(graph.Update(graph.entries()[0], [](auto& block) {
    block.nodes[1] = {ir::Op::constant, 64, {}, 0};
    block.nodes[2] = {ir::Op::equal, 1, {0, 1}};
    block.nodes[3] = {ir::Op::bit_not, 1, {2}};
    block.nodes[5] = {ir::Op::select, 64, {2, 1, 4}};
  }));
  auto checked = Plenty();
  ASSERT_EQ(ir::ValidateSsa(graph, checked), ir::SsaDecline::none);
  const auto& block = *graph.Get(graph.entries()[0]);
  EXPECT_EQ(ir::CheckSsaSelectedImageAddress(block, 10, 3, 0x2008, 0x2000, budget),
            ir::SsaDecline::none);
  EXPECT_EQ(ir::CheckSsaSelectedImageAddress(block, 10, 3, 0x2000, 0x2008, budget),
            ir::SsaDecline::invalid_graph);
  const auto implied = ir::SsaImpliedBits(block.nodes, 3, true);
  ASSERT_EQ(implied.size(), 2U);
  EXPECT_EQ(implied[1], (std::pair<ir::ValueId, bool>{2, false}));
}

TEST(SsaImageAddresses, RefusesUnknownIndexOrMissingImageBase) {
  for (const auto change : {0, 1, 2}) {
    auto graph = Graph();
    ASSERT_TRUE(graph.Update(graph.entries()[0], [&](auto& block) {
      if (change == 0)
        block.nodes[5] = {ir::Op::add, 64, {0, 4}};
      else if (change == 1)
        block.nodes[8] = {ir::Op::constant, 64, {}, 0x2000};
      else
        block.nodes[7] = {ir::Op::bit_and, 64, {5, 6}};
    }));
    auto budget = Plenty();
    auto result = ProveSsaSelectedImageAddresses(graph, budget);
    ASSERT_TRUE(result.facts);
    EXPECT_TRUE(result.facts->selected.empty());
    ASSERT_EQ(result.refused.size(), 1);
    EXPECT_EQ(result.refused[0].load, 10U);
    EXPECT_EQ(result.refused[0].reason, SsaImageAddressRefusal::not_selected_image);
  }
}

TEST(SsaImageAddresses, BudgetCutPublishesNoFacts) {
  auto graph = Graph();
  auto full = Plenty();
  ASSERT_TRUE(ProveSsaSelectedImageAddresses(graph, full).facts);
  const auto used = full.used();
  Budget cut({used.work - 1, used.bytes});
  const auto refused = ProveSsaSelectedImageAddresses(graph, cut);
  EXPECT_FALSE(refused.facts);
  EXPECT_TRUE(refused.refused.empty());
  EXPECT_EQ(refused.reason, SsaImageAddressRefusal::resource_limit);
}
}  // namespace
}  // namespace nyx::analysis
