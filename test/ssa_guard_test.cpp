#include <set>

#include <gtest/gtest.h>

#include "nyx/analysis/ssa/constants.hpp"
#include "nyx/analysis/ssa/guard.hpp"
#include "nyx/analysis/ssa/index_bound.hpp"
#include "nyx/analysis/ssa/liveness.hpp"
#include "nyx/analysis/ssa/reachability.hpp"
#include "nyx/analysis/ssa/sccp.hpp"
#include "nyx/analysis/ssa/table_address.hpp"
#include "nyx/eval/ssa.hpp"
#include "nyx/ir/ssa/print.hpp"
#include "nyx/recovery/constant_load.hpp"
#include "nyx/recovery/mba.hpp"
#include "nyx/recovery/ssa/constants.hpp"
#include "nyx/recovery/ssa/pure_dce.hpp"
#include "nyx/recovery/ssa/simplify.hpp"

namespace nyx::analysis {
namespace {
Budget Plenty() { return Budget({1000000, 10000000}); }

std::vector<std::uint8_t> Bytes(std::uint32_t word) {
  return {static_cast<std::uint8_t>(word), static_cast<std::uint8_t>(word >> 8),
          static_cast<std::uint8_t>(word >> 16), static_cast<std::uint8_t>(word >> 24)};
}

struct Fixture {
  ir::SsaGraph graph;
  std::vector<ir::Group> sources;
  ir::SsaHandle branch;
  ir::SsaHandle table;
  ir::SsaHandle fallback;
};

Fixture GuardedGraph(bool bypass = false) {
  Fixture fixture;
  const std::vector<ir::Node> guard_nodes{{ir::Op::read, 64, {}, 0, 0},
                                          {ir::Op::constant, 64, {}, 4},
                                          {ir::Op::unsigned_less, 1, {0, 1}},
                                          {ir::Op::image_address, 64, {}, 0x110},
                                          {ir::Op::image_address, 64, {}, 0x104}};
  const ir::Transfer guard_transfer{ir::TransferKind::conditional, 3, 2, 4, {}};
  fixture.sources.emplace_back(0x100, Bytes(0x54000083), guard_nodes, std::vector<ir::Write>{},
                               ir::MemoryModel::unspecified, guard_transfer);
  fixture.sources.emplace_back(0x110, Bytes(0xd65f03c0),
                               std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 30}},
                               std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                               ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}});
  fixture.sources.emplace_back(
      0x104, Bytes(bypass ? 0x14000003 : 0xd65f03c0),
      std::vector<ir::Node>{bypass ? ir::Node{ir::Op::image_address, 64, {}, 0x110}
                                   : ir::Node{ir::Op::read, 64, {}, 0, 30}},
      std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
      ir::Transfer{bypass ? ir::TransferKind::jump : ir::TransferKind::return_, 0, {}, {}, {}});

  ir::SsaBlock branch{};
  branch.address = 0x100;
  branch.nodes = guard_nodes;
  branch.source_groups = {0x100};
  branch.source_bytes = {Bytes(0x54000083)};
  branch.original_sources = {0};
  branch.boundaries = {{0, 5, {}, guard_transfer}};
  branch.phis = {{0, 64, true, {}}, {30, 64, true, {}}};
  branch.clobbers = {0, 0};
  fixture.branch = fixture.graph.Add(std::move(branch));

  for (unsigned index = 0; index < 2; ++index) {
    const bool table = index == 0;
    ir::SsaBlock block{};
    block.address = table ? 0x110 : 0x104;
    block.nodes = {{!table && bypass ? ir::Op::image_address : ir::Op::read,
                    64,
                    {},
                    !table && bypass ? 0x110U : 0U,
                    30}};
    block.source_groups = {block.address};
    block.source_bytes = {Bytes(!table && bypass ? 0x14000003 : 0xd65f03c0)};
    block.original_sources = {table ? 1U : 2U};
    block.boundaries = {
        {0,
         1,
         {},
         ir::Transfer{!table && bypass ? ir::TransferKind::jump : ir::TransferKind::return_,
                      0,
                      {},
                      {},
                      {}}}};
    block.phis = {{0, 64, false, {}}, {30, 64, false, {}}};
    block.clobbers = {0, 0};
    const auto handle = fixture.graph.Add(std::move(block));
    if (table)
      fixture.table = handle;
    else
      fixture.fallback = handle;
  }

  const auto a = fixture.branch, b = fixture.table, c = fixture.fallback;
  fixture.graph.Update(a, [&](auto& block) {
    block.reads = {{0, 0, {ir::SsaValueKind::phi, a, 0}}};
    block.exits = {{0, {ir::SsaValueKind::phi, a, 0}}, {30, {ir::SsaValueKind::phi, a, 1}}};
    block.edges = {
        {ir::SsaEdgeKind::branch, ir::SsaTargetKind::image_location, 0x110, b, 2, true, {}},
        {ir::SsaEdgeKind::fallthrough, ir::SsaTargetKind::image_location, 0x104, c, 2, false, {}}};
  });
  fixture.graph.Update(b, [&](auto& block) {
    for (std::size_t index = 0; index < 2; ++index) {
      block.phis[index].incoming = {
          {a, {ir::SsaValueKind::phi, a, static_cast<std::uint32_t>(index)}}};
      if (bypass)
        block.phis[index].incoming.push_back(
            {c, {ir::SsaValueKind::phi, c, static_cast<std::uint32_t>(index)}});
    }

    block.reads = {{0, 1, {ir::SsaValueKind::phi, b, 1}}};
    block.exits = {{0, {ir::SsaValueKind::phi, b, 0}}, {30, {ir::SsaValueKind::phi, b, 1}}};
    block.edges = {{ir::SsaEdgeKind::return_,
                    ir::SsaTargetKind::unknown,
                    0,
                    {},
                    {},
                    {},
                    {.return_leaves = true}}};
  });
  fixture.graph.Update(c, [&](auto& block) {
    for (std::size_t index = 0; index < 2; ++index)
      block.phis[index].incoming = {
          {a, {ir::SsaValueKind::phi, a, static_cast<std::uint32_t>(index)}}};
    if (!bypass) block.reads = {{0, 1, {ir::SsaValueKind::phi, c, 1}}};
    block.exits = {{0, {ir::SsaValueKind::phi, c, 0}}, {30, {ir::SsaValueKind::phi, c, 1}}};
    block.edges = bypass ? std::vector<ir::SsaEdge>{{ir::SsaEdgeKind::branch,
                                                     ir::SsaTargetKind::image_location,
                                                     0x110,
                                                     b,
                                                     {},
                                                     {},
                                                     {}}}
                         : std::vector<ir::SsaEdge>{{ir::SsaEdgeKind::return_,
                                                     ir::SsaTargetKind::unknown,
                                                     0,
                                                     {},
                                                     {},
                                                     {},
                                                     {.return_leaves = true}}};
  });
  fixture.graph.SetEntries({a});
  return fixture;
}

Fixture GuardedTableGraph() {
  auto fixture = GuardedGraph();
  const std::vector<ir::Node> nodes{
      {ir::Op::read, 64, {}, 0, 0}, {ir::Op::constant, 64, {}, 8},
      {ir::Op::mul, 64, {0, 1}},    {ir::Op::image_address, 64, {}, 0x2000},
      {ir::Op::add, 64, {3, 2}},    {ir::Op::load, 64, {4}},
      {ir::Op::read, 64, {}, 0, 30}};
  fixture.sources[1] = ir::Group(0x110, Bytes(0xd65f03c0), nodes, std::vector<ir::Write>{{0, 5}},
                                 ir::MemoryModel::atomic_scalar_reference,
                                 ir::Transfer{ir::TransferKind::return_, 6, {}, {}, {}});
  const auto table = fixture.table;
  fixture.graph.Update(table, [&](auto& block) {
    block.nodes = nodes;
    block.boundaries = {{0, 7, {{0, 5}}, ir::Transfer{ir::TransferKind::return_, 6, {}, {}, {}}}};
    block.reads = {{0, 0, {ir::SsaValueKind::phi, table, 0}},
                   {6, 1, {ir::SsaValueKind::phi, table, 1}}};
    block.exits[0].value = {ir::SsaValueKind::node, table, 5};
  });
  return fixture;
}

TEST(SsaGuard, ProvesExactConditionalEdgeAndRejectsForgedFact) {
  auto fixture = GuardedGraph();
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(fixture.graph, budget), ir::SsaDecline::none);
  const auto truth =
      ProveSsaGuardEdge(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                        fixture.branch, 0, fixture.table, budget);
  ASSERT_TRUE(truth.fact) << static_cast<int>(truth.reason);
  EXPECT_EQ(truth.fact->edge_index, 0U);
  EXPECT_EQ(ir::ValidateSsaGuardEdgeFact(fixture.graph, *truth.fact, fixture.sources, budget),
            ir::SsaDecline::none);
  const auto falsity =
      ProveSsaGuardEdge(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                        fixture.branch, 1, fixture.fallback, budget);
  ASSERT_TRUE(falsity.fact);
  EXPECT_EQ(falsity.fact->edge_index, 1U);

  auto forged = *truth.fact;
  forged.edge_index = 1;
  EXPECT_EQ(ir::ValidateSsaGuardEdgeFact(fixture.graph, forged, fixture.sources, budget),
            ir::SsaDecline::invalid_graph);
  ASSERT_TRUE(
      fixture.graph.Update(fixture.branch, [](auto& block) { block.nodes[1].immediate = 5; }));
  EXPECT_EQ(ir::ValidateSsaGuardEdgeFact(fixture.graph, *truth.fact, fixture.sources, budget),
            ir::SsaDecline::invalid_graph);
}

TEST(SsaGuard, RefusesAlternateEntryAndFalseArmBypass) {
  auto fixture = GuardedGraph();
  auto budget = Plenty();
  ASSERT_TRUE(fixture.graph.Update(fixture.table, [](auto& block) {
    for (auto& phi : block.phis) phi.external_entry = true;
  }));
  fixture.graph.SetEntries({fixture.branch, fixture.table});
  const auto alternate =
      ProveSsaGuardEdge(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                        fixture.branch, 0, fixture.table, budget);
  EXPECT_FALSE(alternate.fact);
  EXPECT_EQ(alternate.reason, SsaGuardRefusal::bypass_path);

  auto bypass = GuardedGraph(true);
  ASSERT_EQ(ir::ValidateSsa(bypass.graph, budget), ir::SsaDecline::none);
  const auto false_arm =
      ProveSsaGuardEdge(bypass.graph, bypass.sources, ir::SsaEntryScope::closed_population,
                        bypass.branch, 0, bypass.table, budget);
  EXPECT_FALSE(false_arm.fact);
  EXPECT_EQ(false_arm.reason, SsaGuardRefusal::bypass_path);
}

TEST(SsaGuard, ReachableLoopStillRequiresItsEnteringEdge) {
  auto fixture = GuardedGraph();
  fixture.sources[1] = ir::Group(0x110, Bytes(0x14000000),
                                 std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x110}},
                                 std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                                 ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}});
  const auto table = fixture.table;
  ASSERT_TRUE(fixture.graph.Update(table, [&](auto& block) {
    block.nodes = {{ir::Op::image_address, 64, {}, 0x110}};
    block.source_bytes[0] = Bytes(0x14000000);
    block.boundaries[0].transfer = ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}};
    block.reads.clear();
    for (std::size_t index = 0; index < 2; ++index)
      block.phis[index].incoming.push_back(
          {table, {ir::SsaValueKind::phi, table, static_cast<std::uint32_t>(index)}});
    block.edges = {
        {ir::SsaEdgeKind::branch, ir::SsaTargetKind::image_location, 0x110, table, {}, {}, {}}};
  }));
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(fixture.graph, budget), ir::SsaDecline::none);
  const auto proof =
      ProveSsaGuardEdge(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                        fixture.branch, 0, fixture.table, budget);
  ASSERT_TRUE(proof.fact) << static_cast<int>(proof.reason);
  EXPECT_EQ(ir::ValidateSsaGuardEdgeFact(fixture.graph, *proof.fact, fixture.sources, budget),
            ir::SsaDecline::none);
}

TEST(SsaGuard, RefusesOpenIncompleteAndNonconditionalEdges) {
  auto fixture = GuardedGraph();
  auto budget = Plenty();
  EXPECT_EQ(ProveSsaGuardEdge(fixture.graph, fixture.sources, ir::SsaEntryScope::discovered_only,
                              fixture.branch, 0, fixture.table, budget)
                .reason,
            SsaGuardRefusal::open_entries);
  EXPECT_EQ(ProveSsaGuardEdge(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                              fixture.table, 0, fixture.table, budget)
                .reason,
            SsaGuardRefusal::not_conditional);
  ASSERT_TRUE(fixture.graph.Update(fixture.fallback, [](auto& block) {
    block.edges = {{ir::SsaEdgeKind::opaque_unknown,
                    ir::SsaTargetKind::unknown,
                    0,
                    {},
                    {},
                    {},
                    {.unresolved_target = true}}};
  }));
  EXPECT_EQ(ProveSsaGuardEdge(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                              fixture.branch, 0, fixture.table, budget)
                .reason,
            SsaGuardRefusal::incomplete_successors);
}

TEST(SsaGuard, BudgetRefusalPublishesNoFact) {
  auto fixture = GuardedGraph();
  auto full = Plenty();
  ASSERT_TRUE(ProveSsaGuardEdge(fixture.graph, fixture.sources,
                                ir::SsaEntryScope::closed_population, fixture.branch, 0,
                                fixture.table, full)
                  .fact);
  const auto used = full.used();
  ASSERT_LT(used.work, 100000U);
  for (std::uint64_t cut = 0; cut < used.work; ++cut) {
    Budget limited({cut, used.bytes});
    const auto result =
        ProveSsaGuardEdge(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                          fixture.branch, 0, fixture.table, limited);
    EXPECT_FALSE(result.fact) << cut;
    EXPECT_EQ(result.reason, SsaGuardRefusal::resource_limit) << cut;
  }
}

TEST(SsaGuard, PredicateBindingRejectsChangedComputationAndRecoveredControl) {
  auto fixture = GuardedGraph();
  auto budget = Plenty();
  const auto proof =
      ProveSsaBranchPredicate(fixture.graph, fixture.sources, fixture.branch, 0, budget);
  ASSERT_TRUE(proof.fact) << static_cast<int>(proof.reason);
  EXPECT_EQ(proof.fact->condition, 2U);
  EXPECT_TRUE(proof.fact->when);
  EXPECT_EQ(ir::ValidateSsaBranchPredicateFact(fixture.graph, *proof.fact, fixture.sources, budget),
            ir::SsaDecline::none);
  auto forged = *proof.fact;
  forged.edge_index = 1;
  EXPECT_EQ(ir::ValidateSsaBranchPredicateFact(fixture.graph, forged, fixture.sources, budget),
            ir::SsaDecline::invalid_graph);
  ASSERT_TRUE(
      fixture.graph.Update(fixture.branch, [](auto& block) { block.nodes[2].op = ir::Op::equal; }));
  EXPECT_FALSE(
      ProveSsaBranchPredicate(fixture.graph, fixture.sources, fixture.branch, 0, budget).fact);
  EXPECT_EQ(ir::ValidateSsaBranchPredicateFact(fixture.graph, *proof.fact, fixture.sources, budget),
            ir::SsaDecline::invalid_graph);
  auto altered_source = GuardedGraph();
  altered_source.sources[0] =
      ir::Group(0x100, Bytes(0x54000083),
                std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 0},
                                      {ir::Op::constant, 64, {}, 4},
                                      {ir::Op::equal, 1, {0, 1}},
                                      {ir::Op::image_address, 64, {}, 0x110},
                                      {ir::Op::image_address, 64, {}, 0x104}},
                std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                ir::Transfer{ir::TransferKind::conditional, 3, 2, 4, {}});
  EXPECT_FALSE(ProveSsaBranchPredicate(altered_source.graph, altered_source.sources,
                                       altered_source.branch, 0, budget)
                   .fact);
  auto fresh = GuardedGraph();
  ASSERT_TRUE(fresh.graph.Update(fresh.branch, [](auto& block) { block.transition = 0; }));
  EXPECT_EQ(ProveSsaBranchPredicate(fresh.graph, fresh.sources, fresh.branch, 0, budget).reason,
            SsaPredicateRefusal::recovered_control);

  auto overlaid = GuardedGraph();
  const auto original = overlaid.sources[0];
  std::vector<ir::Node> nodes(original.nodes().begin(), original.nodes().end());
  nodes.push_back({ir::Op::constant, 64, {}, 9});
  overlaid.sources[0] = ir::Group(0x100, Bytes(0x54000083), nodes, std::vector<ir::Write>{},
                                  ir::MemoryModel::unspecified,
                                  ir::Transfer{ir::TransferKind::conditional, 3, 2, 4, {}});
  ASSERT_TRUE(overlaid.graph.Update(overlaid.branch, [&](auto& block) {
    block.nodes = nodes;
    block.boundaries[0].node_count = 6;
    block.dead_pure_nodes = {5};
  }));
  EXPECT_EQ(ir::ValidateSsa(overlaid.graph, budget), ir::SsaDecline::none);
  EXPECT_EQ(
      ProveSsaBranchPredicate(overlaid.graph, overlaid.sources, overlaid.branch, 0, budget).reason,
      SsaPredicateRefusal::transformed_block);
}

// A promoted frame slot passing through the guard leaves its computation as
// decoded, so the guard still binds to its source.
TEST(SsaGuard, PredicateBindingAdmitsPromotedSlotsPassingThrough) {
  auto fixture = GuardedGraph();
  auto budget = Plenty();
  const auto a = fixture.branch;
  ASSERT_TRUE(fixture.graph.Update(a, [&](auto& block) {
    block.frame_phis = {{-8, 8, true, {}}};
    block.frame_exits = {{ir::SsaValueKind::frame_phi, a, 0}};
  }));
  for (const auto handle : {fixture.table, fixture.fallback})
    ASSERT_TRUE(fixture.graph.Update(handle, [&](auto& block) {
      block.frame_phis = {{-8, 8, false, {{a, {ir::SsaValueKind::frame_phi, a, 0}}}}};
      block.frame_exits = {{ir::SsaValueKind::frame_phi, handle, 0}};
    }));
  ASSERT_EQ(ir::ValidateSsa(fixture.graph, budget), ir::SsaDecline::none);
  const auto proof =
      ProveSsaBranchPredicate(fixture.graph, fixture.sources, fixture.branch, 0, budget);
  ASSERT_TRUE(proof.fact) << static_cast<int>(proof.reason);
  EXPECT_EQ(ir::ValidateSsaBranchPredicateFact(fixture.graph, *proof.fact, fixture.sources, budget),
            ir::SsaDecline::none);
  EXPECT_TRUE(ProveSsaDirectIndexBound(fixture.graph, fixture.sources,
                                       ir::SsaEntryScope::closed_population, fixture.branch, 0,
                                       fixture.table, budget)
                  .fact);
}

TEST(SsaGuard, PredicateBindingBudgetRefusesBeforePublishing) {
  auto fixture = GuardedGraph();
  auto full = Plenty();
  ASSERT_TRUE(
      ProveSsaBranchPredicate(fixture.graph, fixture.sources, fixture.branch, 0, full).fact);
  const auto used = full.used();
  ASSERT_LT(used.work, 100000U);
  for (std::uint64_t cut = 0; cut < used.work; ++cut) {
    Budget limited({cut, used.bytes});
    const auto result =
        ProveSsaBranchPredicate(fixture.graph, fixture.sources, fixture.branch, 0, limited);
    EXPECT_FALSE(result.fact) << cut;
    EXPECT_EQ(result.reason, SsaPredicateRefusal::resource_limit) << cut;
  }
}

TEST(SsaGuard, DirectIndexBoundTracksTheComparedSsaValue) {
  auto fixture = GuardedGraph();
  auto budget = Plenty();
  const auto result =
      ProveSsaDirectIndexBound(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                               fixture.branch, 0, fixture.table, budget);
  ASSERT_TRUE(result.fact) << static_cast<int>(result.reason);
  EXPECT_EQ(result.fact->storage, 0U);
  EXPECT_EQ(result.fact->exclusive_upper, 4U);
  EXPECT_EQ(ir::ValidateSsaIndexBoundFact(fixture.graph, *result.fact, fixture.sources, budget),
            ir::SsaDecline::none);
  auto forged = *result.fact;
  forged.exclusive_upper = 5;
  EXPECT_EQ(ir::ValidateSsaIndexBoundFact(fixture.graph, forged, fixture.sources, budget),
            ir::SsaDecline::invalid_graph);
  forged = *result.fact;
  forged.storage = 30;
  EXPECT_EQ(ir::ValidateSsaIndexBoundFact(fixture.graph, forged, fixture.sources, budget),
            ir::SsaDecline::invalid_graph);
  EXPECT_FALSE(ProveSsaDirectIndexBound(fixture.graph, fixture.sources,
                                        ir::SsaEntryScope::closed_population, fixture.branch, 1,
                                        fixture.fallback, budget)
                   .fact);
}

TEST(SsaGuard, DirectIndexBoundRefusesWriteAfterComparison) {
  auto fixture = GuardedGraph();
  const auto branch = fixture.branch;
  std::vector<ir::Node> nodes(fixture.sources[0].nodes().begin(), fixture.sources[0].nodes().end());
  nodes.push_back({ir::Op::constant, 64, {}, 99});
  nodes.push_back({ir::Op::write, 64, {5}, 0, 0});
  fixture.sources[0] = ir::Group(0x100, Bytes(0x54000083), nodes, std::vector<ir::Write>{},
                                 ir::MemoryModel::unspecified,
                                 ir::Transfer{ir::TransferKind::conditional, 3, 2, 4, {}});
  ASSERT_TRUE(fixture.graph.Update(branch, [&](auto& block) {
    block.nodes = nodes;
    block.boundaries[0].node_count = 7;
    block.exits[0].value = {ir::SsaValueKind::node, branch, 5};
  }));
  for (const auto target : {fixture.table, fixture.fallback})
    ASSERT_TRUE(fixture.graph.Update(target, [&](auto& block) {
      block.phis[0].incoming[0].value = {ir::SsaValueKind::node, branch, 5};
    }));
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(fixture.graph, budget), ir::SsaDecline::none);
  EXPECT_TRUE(ProveSsaGuardEdge(fixture.graph, fixture.sources,
                                ir::SsaEntryScope::closed_population, branch, 0, fixture.table,
                                budget)
                  .fact);
  EXPECT_TRUE(ProveSsaBranchPredicate(fixture.graph, fixture.sources, branch, 0, budget).fact);
  EXPECT_EQ(
      ProveSsaDirectIndexBound(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                               branch, 0, fixture.table, budget)
          .reason,
      SsaIndexBoundRefusal::changed_index);
}

TEST(SsaGuard, DirectIndexBoundBudgetRefusesBeforePublishing) {
  auto fixture = GuardedGraph();
  auto full = Plenty();
  ASSERT_TRUE(ProveSsaDirectIndexBound(fixture.graph, fixture.sources,
                                       ir::SsaEntryScope::closed_population, fixture.branch, 0,
                                       fixture.table, full)
                  .fact);
  const auto used = full.used();
  ASSERT_LT(used.work, 100000U);
  for (std::uint64_t cut = 0; cut < used.work; ++cut) {
    Budget limited({cut, used.bytes});
    const auto result = ProveSsaDirectIndexBound(fixture.graph, fixture.sources,
                                                 ir::SsaEntryScope::closed_population,
                                                 fixture.branch, 0, fixture.table, limited);
    EXPECT_FALSE(result.fact) << cut;
    EXPECT_EQ(result.reason, SsaIndexBoundRefusal::resource_limit) << cut;
  }
}

TEST(SsaGuard, BoundedTableAddressTracksIndexAndRefusesMutation) {
  auto fixture = GuardedTableGraph();
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(fixture.graph, budget), ir::SsaDecline::none);
  const auto bound =
      ProveSsaDirectIndexBound(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                               fixture.branch, 0, fixture.table, budget);
  ASSERT_TRUE(bound.fact) << static_cast<int>(bound.reason);
  const auto address =
      ProveSsaBoundedTableAddress(fixture.graph, fixture.sources, *bound.fact, 5, budget);
  ASSERT_TRUE(address.fact) << static_cast<int>(address.reason);
  EXPECT_EQ(address.fact->base, 0x2000U);
  EXPECT_EQ(address.fact->stride, 8U);
  EXPECT_EQ(
      ir::ValidateSsaBoundedTableAddressFact(fixture.graph, *address.fact, fixture.sources, budget),
      ir::SsaDecline::none);
  auto forged = *address.fact;
  forged.stride = 16;
  EXPECT_EQ(ir::ValidateSsaBoundedTableAddressFact(fixture.graph, forged, fixture.sources, budget),
            ir::SsaDecline::invalid_graph);
  ASSERT_TRUE(
      fixture.graph.Update(fixture.table, [](auto& block) { block.nodes[3].immediate = 0x3000; }));
  EXPECT_EQ(
      ir::ValidateSsaBoundedTableAddressFact(fixture.graph, *address.fact, fixture.sources, budget),
      ir::SsaDecline::invalid_graph);
}

TEST(SsaGuard, BoundedTableAddressRefusesOverflowAndWrongIndex) {
  auto overflow = GuardedTableGraph();
  const auto table = overflow.table;
  std::vector<ir::Node> nodes(overflow.sources[1].nodes().begin(),
                              overflow.sources[1].nodes().end());
  nodes[3].immediate = UINT64_MAX - 7;
  overflow.sources[1] = ir::Group(0x110, Bytes(0xd65f03c0), nodes, std::vector<ir::Write>{{0, 5}},
                                  ir::MemoryModel::atomic_scalar_reference,
                                  ir::Transfer{ir::TransferKind::return_, 6, {}, {}, {}});
  ASSERT_TRUE(overflow.graph.Update(table, [&](auto& block) { block.nodes = nodes; }));
  auto budget = Plenty();
  const auto bound = ProveSsaDirectIndexBound(overflow.graph, overflow.sources,
                                              ir::SsaEntryScope::closed_population, overflow.branch,
                                              0, table, budget);
  ASSERT_TRUE(bound.fact);
  EXPECT_EQ(
      ProveSsaBoundedTableAddress(overflow.graph, overflow.sources, *bound.fact, 5, budget).reason,
      SsaTableAddressRefusal::address_overflow);

  auto changed = GuardedTableGraph();
  nodes.assign(changed.sources[1].nodes().begin(), changed.sources[1].nodes().end());
  nodes[0].storage = 30;
  nodes[6].storage = 0;
  changed.sources[1] = ir::Group(0x110, Bytes(0xd65f03c0), nodes, std::vector<ir::Write>{{0, 5}},
                                 ir::MemoryModel::atomic_scalar_reference,
                                 ir::Transfer{ir::TransferKind::return_, 6, {}, {}, {}});
  const auto changed_table = changed.table;
  ASSERT_TRUE(changed.graph.Update(changed_table, [&](auto& block) {
    block.nodes = nodes;
    block.reads = {{0, 1, {ir::SsaValueKind::phi, changed_table, 1}},
                   {6, 0, {ir::SsaValueKind::phi, changed_table, 0}}};
  }));
  ASSERT_EQ(ir::ValidateSsa(changed.graph, budget), ir::SsaDecline::none);
  const auto changed_bound =
      ProveSsaDirectIndexBound(changed.graph, changed.sources, ir::SsaEntryScope::closed_population,
                               changed.branch, 0, changed_table, budget);
  ASSERT_TRUE(changed_bound.fact);
  EXPECT_EQ(
      ProveSsaBoundedTableAddress(changed.graph, changed.sources, *changed_bound.fact, 5, budget)
          .reason,
      SsaTableAddressRefusal::index_changed);
}

TEST(SsaGuard, BoundedTableAddressBudgetRefusesBeforePublishing) {
  auto fixture = GuardedTableGraph();
  auto preliminary = Plenty();
  const auto bound =
      ProveSsaDirectIndexBound(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                               fixture.branch, 0, fixture.table, preliminary);
  ASSERT_TRUE(bound.fact);
  auto full = Plenty();
  ASSERT_TRUE(
      ProveSsaBoundedTableAddress(fixture.graph, fixture.sources, *bound.fact, 5, full).fact);
  const auto used = full.used();
  ASSERT_LT(used.work, 100000U);
  for (std::uint64_t cut = 0; cut < used.work; ++cut) {
    Budget limited({cut, used.bytes});
    const auto result =
        ProveSsaBoundedTableAddress(fixture.graph, fixture.sources, *bound.fact, 5, limited);
    EXPECT_FALSE(result.fact) << cut;
    EXPECT_EQ(result.reason, SsaTableAddressRefusal::resource_limit) << cut;
  }
}

// The guarded block is a switch's dispatch: it loads an offset from the
// table and jumps past 0x300 by it, and the CFG builder gave it one edge per
// target it enumerated. No fold proves those edges yet.
Fixture GuardedDispatchGraph(std::vector<std::uint64_t> targets = {0x300, 0x304, 0x308}) {
  auto fixture = GuardedGraph();
  const std::vector<ir::Node> nodes{{ir::Op::read, 64, {}, 0, 0},
                                    {ir::Op::constant, 64, {}, 8},
                                    {ir::Op::mul, 64, {0, 1}},
                                    {ir::Op::image_address, 64, {}, 0x2000},
                                    {ir::Op::add, 64, {3, 2}},
                                    {ir::Op::load, 64, {4}},
                                    {ir::Op::image_address, 64, {}, 0x300},
                                    {ir::Op::add, 64, {6, 5}}};
  const ir::Transfer jump{ir::TransferKind::jump, 7, {}, {}, {}};
  fixture.sources[1] = ir::Group(0x110, Bytes(0xd61f0200), nodes, std::vector<ir::Write>{},
                                 ir::MemoryModel::atomic_scalar_reference, jump);
  std::vector<ir::SsaHandle> returns;
  for (const auto address : targets) {
    const auto source = static_cast<std::uint32_t>(fixture.sources.size());
    fixture.sources.emplace_back(address, Bytes(0xd65f03c0),
                                 std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 30}},
                                 std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                                 ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}});
    ir::SsaBlock block{};
    block.address = address;
    block.nodes = {{ir::Op::read, 64, {}, 0, 30}};
    block.source_groups = {address};
    block.source_bytes = {Bytes(0xd65f03c0)};
    block.original_sources = {source};
    block.boundaries = {{0, 1, {}, ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}}}};
    block.phis = {{0, 64, false, {}}, {30, 64, false, {}}};
    block.clobbers = {0, 0};
    returns.push_back(fixture.graph.Add(std::move(block)));
  }

  const auto table = fixture.table;
  fixture.graph.Update(table, [&](auto& block) {
    block.nodes = nodes;
    block.source_bytes = {Bytes(0xd61f0200)};
    block.boundaries = {{0, 8, {}, jump}};
    block.reads = {{0, 0, {ir::SsaValueKind::phi, table, 0}}};
    block.edges.clear();
    for (std::size_t index = 0; index < targets.size(); ++index)
      block.edges.push_back({ir::SsaEdgeKind::branch,
                             ir::SsaTargetKind::image_location,
                             targets[index],
                             returns[index],
                             {},
                             {},
                             {.constant_image = true}});
  });
  for (const auto handle : returns)
    fixture.graph.Update(handle, [&](auto& block) {
      for (std::size_t index = 0; index < 2; ++index)
        block.phis[index].incoming = {
            {table, {ir::SsaValueKind::phi, table, static_cast<std::uint32_t>(index)}}};
      block.reads = {{0, 1, {ir::SsaValueKind::phi, handle, 1}}};
      block.exits = {{0, {ir::SsaValueKind::phi, handle, 0}},
                     {30, {ir::SsaValueKind::phi, handle, 1}}};
      block.edges = {{ir::SsaEdgeKind::return_,
                      ir::SsaTargetKind::unknown,
                      0,
                      {},
                      {},
                      {},
                      {.return_leaves = true}}};
    });
  return fixture;
}

// Row offsets 0, 4, 8, 4: the rows send the jump to 0x300, 0x304 and 0x308.
constexpr std::array<std::uint8_t, 32> kDispatchRows{
    0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0};

recovery::ConstantLoadResult FoldDispatch(const Fixture& fixture,
                                          std::span<const std::uint8_t> rows, Budget& budget) {
  const auto bound =
      ProveSsaDirectIndexBound(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                               fixture.branch, 0, fixture.table, budget);
  EXPECT_TRUE(bound.fact) << static_cast<int>(bound.reason);
  if (!bound.fact) return {};
  const auto address =
      ProveSsaBoundedTableAddress(fixture.graph, fixture.sources, *bound.fact, 5, budget);
  EXPECT_TRUE(address.fact) << static_cast<int>(address.reason);
  if (!address.fact) return {};
  const ir::ConstantImageRange range{0x2000, rows};
  const ir::ImageFacts image{std::span(&range, 1), {}, true, {}};
  return recovery::ProposeBoundedTableLoads(fixture.graph, *address.fact, fixture.sources, image,
                                            {true, true, true}, budget);
}

TEST(SsaGuard, AnUnfoldedDispatchIsIncompleteButItsGuardIsProvable) {
  auto fixture = GuardedDispatchGraph();
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(fixture.graph, budget), ir::SsaDecline::none);
  const auto& table = *fixture.graph.Get(fixture.table);
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(table, budget), ir::SsaDecline::invalid_graph);
  EXPECT_EQ(ir::ValidateSsaDispatchSuccessors(table, budget), ir::SsaDecline::none);
  EXPECT_EQ(ProveSsaReachability(fixture.graph, fixture.sources,
                                 ir::SsaEntryScope::closed_population, budget)
                .reason,
            SsaReachabilityRefusal::incomplete_successors);
  const auto through = ProveSsaReachability(fixture.graph, fixture.sources,
                                            ir::SsaEntryScope::closed_population, budget, true);
  ASSERT_TRUE(through.facts);

  // Facts proved through a dispatch do not pass for ordinary reachability.
  EXPECT_EQ(
      ir::ValidateSsaReachabilityFacts(fixture.graph, *through.facts, fixture.sources, budget),
      ir::SsaDecline::invalid_graph);
  EXPECT_EQ(ProveSsaGuardEdge(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                              fixture.branch, 0, fixture.table, budget)
                .reason,
            SsaGuardRefusal::incomplete_successors);
  const auto guard =
      ProveSsaGuardEdge(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                        fixture.branch, 0, fixture.table, budget, true);
  ASSERT_TRUE(guard.fact);
  EXPECT_EQ(ir::ValidateSsaGuardEdgeFact(fixture.graph, *guard.fact, fixture.sources, budget),
            ir::SsaDecline::none);
  auto strict = *guard.fact;
  strict.through_dispatches = false;
  EXPECT_EQ(ir::ValidateSsaGuardEdgeFact(fixture.graph, strict, fixture.sources, budget),
            ir::SsaDecline::invalid_graph);

  // A block that jumps several ways without the dispatch's edge shape is not one.
  auto conditional = table;
  conditional.edges[1].condition = 0;
  conditional.edges[1].when = true;
  EXPECT_EQ(ir::ValidateSsaDispatchSuccessors(conditional, budget), ir::SsaDecline::invalid_graph);
  auto repeated = table;
  repeated.edges[1].address = repeated.edges[0].address;
  EXPECT_EQ(ir::ValidateSsaDispatchSuccessors(repeated, budget), ir::SsaDecline::invalid_graph);
}

TEST(SsaGuard, ATableFoldCompletesTheDispatchItFeeds) {
  auto fixture = GuardedDispatchGraph();
  auto budget = Plenty();
  auto folded = FoldDispatch(fixture, kDispatchRows, budget);
  ASSERT_TRUE(folded.provisional) << static_cast<int>(folded.reason);
  auto& graph = *folded.provisional;
  const ir::SsaHandle table{graph.arena(), fixture.table.slot, fixture.table.generation};
  EXPECT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(*graph.Get(table), budget), ir::SsaDecline::none);
  const auto reach =
      ProveSsaReachability(graph, fixture.sources, ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reach.facts) << static_cast<int>(reach.reason);
  for (std::size_t slot = 0; slot < graph.slots(); ++slot)
    EXPECT_EQ(reach.facts->reachable[slot], 1U) << slot;

  // The rows rest on declared bytes, and so do the edges they prove.
  auto undeclared = *graph.Get(table);
  undeclared.edges[1].assumptions.constant_image = false;
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(undeclared, budget), ir::SsaDecline::invalid_graph);

  auto full = Plenty();
  ASSERT_EQ(ir::ValidateSsaDirectSuccessors(*graph.Get(table), full), ir::SsaDecline::none);
  const auto used = full.used();
  ASSERT_LT(used.work, 100000U);
  for (std::uint64_t cut = 0; cut < used.work; ++cut) {
    Budget limited({cut, used.bytes});
    EXPECT_EQ(ir::ValidateSsaDirectSuccessors(*graph.Get(table), limited),
              ir::SsaDecline::resource_limit)
        << cut;
  }
}

TEST(SsaGuard, ADispatchIsCompleteOnlyWhenItsEdgesAreWhatTheRowsReach) {
  auto budget = Plenty();

  // A row that sends the jump somewhere no edge goes.
  {
    auto fixture = GuardedDispatchGraph();
    auto rows = kDispatchRows;
    rows[24] = 12;
    auto folded = FoldDispatch(fixture, rows, budget);
    EXPECT_FALSE(folded.provisional);
    EXPECT_EQ(folded.reason, recovery::ConstantLoadRefusal::invalid_graph);
  }

  // An edge no row sends the jump along.
  {
    auto fixture = GuardedDispatchGraph({0x300, 0x304, 0x308, 0x30c});
    auto folded = FoldDispatch(fixture, kDispatchRows, budget);
    EXPECT_FALSE(folded.provisional);
    EXPECT_EQ(folded.reason, recovery::ConstantLoadRefusal::invalid_graph);
  }

  auto fixture = GuardedDispatchGraph();
  auto folded = FoldDispatch(fixture, kDispatchRows, budget);
  ASSERT_TRUE(folded.provisional);
  const auto& graph = *folded.provisional;
  const ir::SsaHandle handle{graph.arena(), fixture.table.slot, fixture.table.generation};
  const auto& table = *graph.Get(handle);

  // Forged rows are read as the fold declares them.
  auto forged = table;
  forged.constant_loads[0].table_bytes[3][0] = 12;
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(forged, budget), ir::SsaDecline::invalid_graph);
  forged = table;
  forged.constant_loads[0].table_bytes[1][0] = 0;
  forged.constant_loads[0].table_bytes[3][0] = 0;
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(forged, budget), ir::SsaDecline::invalid_graph);

  // The index the rows are taken under must be what the load is addressed by.
  forged = table;
  forged.constant_loads[0].table_index = 6;
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(forged, budget), ir::SsaDecline::invalid_graph);
  forged = table;
  forged.constant_loads[0].node = 7;
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(forged, budget), ir::SsaDecline::invalid_graph);
  forged = table;
  forged.constant_loads[0].kind = ir::SsaConstantKind::literal;
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(forged, budget), ir::SsaDecline::invalid_graph);
  auto clone = graph.Clone(budget);
  ASSERT_TRUE(clone);
  const ir::SsaHandle cloned{clone->arena(), handle.slot, handle.generation};
  ASSERT_TRUE(
      clone->Update(cloned, [](auto& block) { block.constant_loads[0].table_bytes[3][0] = 12; }));
  EXPECT_EQ(ir::ValidateSsa(*clone, budget), ir::SsaDecline::invalid_graph);
}

// A fold is kept only where every block that can run is complete. Another
// dispatch no fold proves yet might jump anywhere, the table block included.
TEST(SsaGuard, ATableFoldIsRefusedBesideAnUnfoldedDispatch) {
  auto fixture = GuardedDispatchGraph();
  auto budget = Plenty();
  auto folded = FoldDispatch(fixture, kDispatchRows, budget);
  ASSERT_TRUE(folded.provisional);
  const auto& graph = *folded.provisional;
  std::vector<ir::SsaHandle> returns;
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    const auto handle = graph.Handle(slot);
    if (handle && graph.Get(*handle)->address >= 0x300) returns.push_back(*handle);
  }

  ASSERT_EQ(returns.size(), 3U);
  const auto table = ir::SsaHandle{graph.arena(), fixture.table.slot, fixture.table.generation};
  const auto redirect = [&](bool keep_fold) {
    auto clone = graph.Clone(budget);
    EXPECT_TRUE(clone);
    const auto from = ir::SsaHandle{clone->arena(), returns[2].slot, returns[2].generation};
    EXPECT_TRUE(clone->Update(from, [&](auto& block) {
      block.nodes = {{ir::Op::read, 64, {}, 0, 0}};
      block.boundaries = {{0, 1, {}, ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}}}};
      block.reads = {{0, 0, {ir::SsaValueKind::phi, from, 0}}};
      block.edges.clear();
      for (unsigned index = 0; index < 2; ++index)
        block.edges.push_back(
            {ir::SsaEdgeKind::branch,
             ir::SsaTargetKind::image_location,
             graph.Get(returns[index])->address,
             ir::SsaHandle{clone->arena(), returns[index].slot, returns[index].generation},
             {},
             {},
             {.constant_image = true}});
    }));
    for (unsigned index = 0; index < 2; ++index) {
      const auto to = ir::SsaHandle{clone->arena(), returns[index].slot, returns[index].generation};
      EXPECT_TRUE(clone->Update(to, [&](auto& block) {
        for (std::uint32_t phi = 0; phi < 2; ++phi)
          block.phis[phi].incoming.push_back({from, {ir::SsaValueKind::phi, from, phi}});
      }));
    }

    if (!keep_fold) {
      const auto cloned = ir::SsaHandle{clone->arena(), table.slot, table.generation};
      EXPECT_TRUE(clone->Update(cloned, [](auto& block) {
        block.constant_loads.clear();
        block.disabled_effects.clear();
      }));
    }

    return ir::ValidateSsa(*clone, budget);
  };

  EXPECT_EQ(redirect(false), ir::SsaDecline::none);
  EXPECT_EQ(redirect(true), ir::SsaDecline::invalid_graph);
}

// Several facts fold in one candidate. One refused on its own is listed and
// left out; a candidate refused as a whole publishes nothing.
TEST(SsaGuard, TableFoldsAreProposedTogetherAndRefusedTogether) {
  auto fixture = GuardedDispatchGraph();
  auto budget = Plenty();
  const auto bound =
      ProveSsaDirectIndexBound(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                               fixture.branch, 0, fixture.table, budget);
  ASSERT_TRUE(bound.fact);
  const auto address =
      ProveSsaBoundedTableAddress(fixture.graph, fixture.sources, *bound.fact, 5, budget);
  ASSERT_TRUE(address.fact);
  const ir::ConstantImageRange range{0x2000, kDispatchRows};
  const ir::ImageFacts image{std::span(&range, 1), {}, true, {}};
  const std::array<ir::SsaBoundedTableAddressFact, 2> twice{*address.fact, *address.fact};
  auto result = recovery::ProposeBoundedTableLoads(fixture.graph, twice, fixture.sources, image,
                                                   {true, true, true}, budget);
  ASSERT_TRUE(result.provisional);
  EXPECT_EQ(result.journal.size(), 1U);
  ASSERT_EQ(result.refused.size(), 1U);
  EXPECT_EQ(result.refused[0].reason, recovery::ConstantLoadRefusal::existing_omission);

  auto rows = kDispatchRows;
  rows[24] = 12;
  const ir::ConstantImageRange stray{0x2000, rows};
  const ir::ImageFacts strayed{std::span(&stray, 1), {}, true, {}};
  const auto refused = recovery::ProposeBoundedTableLoads(fixture.graph, twice, fixture.sources,
                                                          strayed, {true, true, true}, budget);
  EXPECT_FALSE(refused.provisional);
  EXPECT_TRUE(refused.journal.empty());
  EXPECT_EQ(refused.reason, recovery::ConstantLoadRefusal::invalid_graph);

  auto full = Plenty();
  ASSERT_TRUE(recovery::ProposeBoundedTableLoads(fixture.graph, twice, fixture.sources, image,
                                                 {true, true, true}, full)
                  .provisional);
  const auto used = full.used();
  ASSERT_LT(used.work, 1000000U);
  for (std::uint64_t cut = 0; cut < used.work; cut += 7) {
    Budget limited({cut, used.bytes});
    const auto cut_result = recovery::ProposeBoundedTableLoads(
        fixture.graph, twice, fixture.sources, image, {true, true, true}, limited);
    EXPECT_FALSE(cut_result.provisional) << cut;
    EXPECT_TRUE(cut_result.journal.empty()) << cut;
    EXPECT_EQ(cut_result.reason, recovery::ConstantLoadRefusal::resource_limit) << cut;
  }
}

TEST(SsaGuard, ADispatchFoldBudgetRefusesBeforePublishing) {
  auto fixture = GuardedDispatchGraph();
  auto full = Plenty();
  ASSERT_TRUE(FoldDispatch(fixture, kDispatchRows, full).provisional);
  const auto used = full.used();
  ASSERT_LT(used.work, 1000000U);
  for (std::uint64_t cut = 0; cut < used.work; ++cut) {
    Budget limited({cut, used.bytes});
    const auto bound = ProveSsaDirectIndexBound(fixture.graph, fixture.sources,
                                                ir::SsaEntryScope::closed_population,
                                                fixture.branch, 0, fixture.table, limited);
    if (!bound.fact) {
      EXPECT_EQ(bound.reason, SsaIndexBoundRefusal::resource_limit) << cut;
      continue;
    }

    const auto address =
        ProveSsaBoundedTableAddress(fixture.graph, fixture.sources, *bound.fact, 5, limited);
    if (!address.fact) {
      EXPECT_EQ(address.reason, SsaTableAddressRefusal::resource_limit) << cut;
      continue;
    }

    const ir::ConstantImageRange range{0x2000, kDispatchRows};
    const ir::ImageFacts image{std::span(&range, 1), {}, true, {}};
    const auto result = recovery::ProposeBoundedTableLoads(
        fixture.graph, *address.fact, fixture.sources, image, {true, true, true}, limited);
    EXPECT_FALSE(result.provisional) << cut;
    EXPECT_TRUE(result.journal.empty()) << cut;
    EXPECT_EQ(result.reason, recovery::ConstantLoadRefusal::resource_limit) << cut;
  }
}

// The shapes a guard edge bounds a value by, and near misses of each.
TEST(SsaGuard, GuardEdgeBoundReadsEachComparisonShape) {
  using ir::Node;
  using ir::Op;

  // 0 x, 1 N=5, 2 x<N, 3 !(x<N), 4 x-N, 5 0, 6 x-N==0, 7 !(x-N==0), 8 x>N
  const std::vector<Node> wide{
      {Op::read, 64, {}, 0, 0}, {Op::constant, 64, {}, 5}, {Op::unsigned_less, 1, {0, 1}},
      {Op::bit_not, 1, {2}},    {Op::sub, 64, {0, 1}},     {Op::constant, 64, {}, 0},
      {Op::equal, 1, {4, 5}},   {Op::bit_not, 1, {6}},     {Op::bit_and, 1, {3, 7}}};
  const auto bound = [](std::span<const Node> nodes, ir::ValueId condition, bool when) {
    const auto result = ir::SsaGuardEdgeBound(nodes, condition, when);
    return result ? std::optional(std::pair{result->value, result->exclusive_upper}) : std::nullopt;
  };

  using Bound = std::optional<std::pair<ir::ValueId, std::uint64_t>>;
  EXPECT_EQ(bound(wide, 2, true), (Bound{{0, 5}}));
  EXPECT_EQ(bound(wide, 3, false), (Bound{{0, 5}}));
  EXPECT_EQ(bound(wide, 8, false), (Bound{{0, 6}}));

  // The other arm of each leaves the value unbounded above.
  EXPECT_EQ(bound(wide, 2, false), Bound{});
  EXPECT_EQ(bound(wide, 3, true), Bound{});
  EXPECT_EQ(bound(wide, 8, true), Bound{});
  auto changed = wide;
  changed[4].inputs = {1, 0};  // N - x
  EXPECT_EQ(bound(changed, 8, false), Bound{});
  changed = wide;
  changed.push_back({Op::constant, 64, {}, 6});
  changed[4].inputs = {0, 9};  // x - 6 against x < 5
  EXPECT_EQ(bound(changed, 8, false), Bound{});
  changed = wide;
  changed[5].immediate = 1;
  EXPECT_EQ(bound(changed, 8, false), Bound{});

  // A 32-bit compare bounds the number it compares, which is y here: the low
  // half of zext(y) is y.
  // 0 r, 1 y=low(r), 2 zext(y), 3 low(zext(y)), 4 N=3, 5 x<N, 6 x-N, 7 0, 8 ==, 9..11 x>N
  const std::vector<Node> narrow{
      {Op::read, 64, {}, 0, 0},  {Op::extract, 32, {0}, 0}, {Op::zext, 64, {1}},
      {Op::extract, 32, {2}, 0}, {Op::constant, 32, {}, 3}, {Op::unsigned_less, 1, {3, 4}},
      {Op::sub, 32, {3, 4}},     {Op::constant, 32, {}, 0}, {Op::equal, 1, {6, 7}},
      {Op::bit_not, 1, {5}},     {Op::bit_not, 1, {8}},     {Op::bit_and, 1, {9, 10}}};
  EXPECT_EQ(bound(narrow, 11, false), (Bound{{1, 4}}));
  EXPECT_EQ(bound(narrow, 5, true), (Bound{{1, 3}}));

  // The low half of a number that may not fit in it is a number of its own,
  // which only a register holding exactly it carries (SsaExitHolding).
  changed = narrow;
  changed[2].op = Op::bit_or;
  changed[2].inputs = {0, 0};
  changed[1].width = 64;
  EXPECT_EQ(bound(changed, 11, false), (Bound{{3, 4}}));
  changed = narrow;
  changed[3].inputs = {0};
  EXPECT_EQ(bound(changed, 11, false), (Bound{{3, 4}}));

  // x == N itself, as a simplifier leaves x - N == 0, and a copy of x in it.
  changed = narrow;
  changed[6] = {Op::zext, 32, {3}};
  changed[8].inputs = {6, 4};
  EXPECT_EQ(bound(changed, 11, false), (Bound{{1, 4}}));
  changed[8].inputs = {6, 7};  // x == 0
  EXPECT_EQ(bound(changed, 11, false), Bound{});
  changed = narrow;
  changed[4].immediate = 0x100000003;  // a 32-bit literal cannot hold this
  EXPECT_EQ(bound(changed, 5, true), Bound{});
}

// A switch as AArch64 code has it: the guard computes the index into a W
// register, compares it as x > 3 on the flags, and falls into the dispatch,
// which loads a byte at a PC page plus an offset plus the index and jumps
// past 0x300 by it.
struct A64Switch {
  ir::SsaGraph graph;
  std::vector<ir::Group> sources;
  ir::SsaHandle guard;
  ir::SsaHandle table;
};

A64Switch A64SwitchGraph() {
  A64Switch fixture;
  const std::vector<ir::Node> guard_nodes{{ir::Op::read, 64, {}, 0, 1},
                                          {ir::Op::extract, 32, {0}, 0},
                                          {ir::Op::constant, 32, {}, 1},
                                          {ir::Op::sub, 32, {1, 2}},
                                          {ir::Op::zext, 64, {3}},
                                          {ir::Op::extract, 32, {4}, 0},
                                          {ir::Op::constant, 32, {}, 3},
                                          {ir::Op::sub, 32, {5, 6}},
                                          {ir::Op::constant, 32, {}, 0},
                                          {ir::Op::equal, 1, {7, 8}},
                                          {ir::Op::unsigned_less, 1, {5, 6}},
                                          {ir::Op::bit_not, 1, {10}},
                                          {ir::Op::bit_not, 1, {9}},
                                          {ir::Op::bit_and, 1, {11, 12}},
                                          {ir::Op::image_address, 64, {}, 0x104},
                                          {ir::Op::image_address, 64, {}, 0x110}};
  const ir::Transfer guard_transfer{ir::TransferKind::conditional, 14, 13, 15, {}};
  const std::vector<ir::Node> table_nodes{{ir::Op::read, 64, {}, 0, 0},
                                          {ir::Op::image_address, 64, {}, 0x110},
                                          {ir::Op::constant, 64, {}, ~std::uint64_t{0xfff}},
                                          {ir::Op::bit_and, 64, {1, 2}},
                                          {ir::Op::constant, 64, {}, 0x2000},
                                          {ir::Op::add, 64, {3, 4}},
                                          {ir::Op::add, 64, {5, 0}},
                                          {ir::Op::load, 8, {6}},
                                          {ir::Op::zext, 64, {7}},
                                          {ir::Op::image_address, 64, {}, 0x300},
                                          {ir::Op::add, 64, {9, 8}}};
  const ir::Transfer jump{ir::TransferKind::jump, 10, {}, {}, {}};
  fixture.sources.emplace_back(0x100, Bytes(0x54000088), guard_nodes,
                               std::vector<ir::Write>{{0, 4}}, ir::MemoryModel::unspecified,
                               guard_transfer);
  fixture.sources.emplace_back(0x104, Bytes(0xd65f03c0),
                               std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 30}},
                               std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                               ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}});
  fixture.sources.emplace_back(0x110, Bytes(0xd61f0200), table_nodes, std::vector<ir::Write>{},
                               ir::MemoryModel::atomic_scalar_reference, jump);
  const std::vector<std::uint64_t> targets{0x300, 0x304, 0x308};
  for (const auto address : targets)
    fixture.sources.emplace_back(address, Bytes(0xd65f03c0),
                                 std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 30}},
                                 std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                                 ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}});
  constexpr std::array<ir::StorageId, 3> storages{0, 1, 30};
  const auto block = [&](std::uint32_t source, std::vector<ir::Node> nodes, bool entry) {
    const auto& group = fixture.sources[source];
    ir::SsaBlock result{};
    result.address = group.source_address();
    result.nodes = std::move(nodes);
    result.source_groups = {group.source_address()};
    result.source_bytes = {std::vector<std::uint8_t>(group.bytes().begin(), group.bytes().end())};
    result.original_sources = {source};
    result.boundaries = {{0, static_cast<std::uint32_t>(result.nodes.size()),
                          std::vector<ir::Write>(group.writes().begin(), group.writes().end()),
                          group.transfer()}};
    for (const auto storage : storages) result.phis.push_back({storage, 64, entry, {}});
    result.clobbers = {0, 0, 0};
    return fixture.graph.Add(std::move(result));
  };

  const auto ret = std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 30}};
  fixture.guard = block(0, guard_nodes, true);
  const auto fallback = block(1, ret, false);
  fixture.table = block(2, table_nodes, false);
  std::vector<ir::SsaHandle> returns;
  for (std::uint32_t index = 0; index < targets.size(); ++index)
    returns.push_back(block(3 + index, ret, false));
  const auto phi = [](ir::SsaHandle handle, std::uint32_t index) {
    return ir::SsaValue{ir::SsaValueKind::phi, handle, index};
  };

  const auto passes = [&](ir::SsaHandle handle, ir::SsaHandle from) {
    fixture.graph.Update(handle, [&](auto& item) {
      for (std::uint32_t index = 0; index < 3; ++index)
        item.phis[index].incoming = {{from, phi(from, index)}};
      if (from == fixture.guard)
        item.phis[0].incoming[0].value = {ir::SsaValueKind::node, fixture.guard, 4};
      item.exits = {{0, phi(handle, 0)}, {1, phi(handle, 1)}, {30, phi(handle, 2)}};
    });
  };

  const auto g = fixture.guard, t = fixture.table;
  fixture.graph.Update(g, [&](auto& item) {
    item.reads = {{0, 1, phi(g, 1)}};
    item.exits = {{0, {ir::SsaValueKind::node, g, 4}}, {1, phi(g, 1)}, {30, phi(g, 2)}};
    item.edges = {
        {ir::SsaEdgeKind::branch, ir::SsaTargetKind::image_location, 0x104, fallback, 13, true, {}},
        {ir::SsaEdgeKind::fallthrough, ir::SsaTargetKind::image_location, 0x110, t, 13, false, {}}};
  });
  passes(fallback, g);
  passes(t, g);
  fixture.graph.Update(fallback, [&](auto& item) {
    item.reads = {{0, 2, phi(fallback, 2)}};
    item.edges = {{ir::SsaEdgeKind::return_,
                   ir::SsaTargetKind::unknown,
                   0,
                   {},
                   {},
                   {},
                   {.return_leaves = true}}};
  });
  fixture.graph.Update(t, [&](auto& item) {
    item.reads = {{0, 0, phi(t, 0)}};
    for (std::size_t index = 0; index < targets.size(); ++index)
      item.edges.push_back({ir::SsaEdgeKind::branch,
                            ir::SsaTargetKind::image_location,
                            targets[index],
                            returns[index],
                            {},
                            {},
                            {.constant_image = true}});
  });
  for (const auto handle : returns) {
    passes(handle, t);
    fixture.graph.Update(handle, [&](auto& item) {
      item.reads = {{0, 2, phi(handle, 2)}};
      item.edges = {{ir::SsaEdgeKind::return_,
                     ir::SsaTargetKind::unknown,
                     0,
                     {},
                     {},
                     {},
                     {.return_leaves = true}}};
    });
  }

  fixture.graph.SetEntries({g});
  return fixture;
}

TEST(SsaGuard, AnAArch64ShapedSwitchFoldsAndCompletes) {
  auto fixture = A64SwitchGraph();
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(fixture.graph, budget), ir::SsaDecline::none);
  const auto bound =
      ProveSsaDirectIndexBound(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                               fixture.guard, 1, fixture.table, budget);
  ASSERT_TRUE(bound.fact) << static_cast<int>(bound.reason);
  EXPECT_EQ(bound.fact->storage, 0U);
  EXPECT_EQ(bound.fact->exclusive_upper, 4U);

  // The edge the flags test takes is x > 3, which bounds nothing.
  EXPECT_EQ(ProveSsaDirectIndexBound(
                fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population, fixture.guard,
                0, *fixture.graph.Get(fixture.guard)->edges[0].target_block, budget)
                .reason,
            SsaIndexBoundRefusal::unsupported_condition);
  const auto address =
      ProveSsaBoundedTableAddress(fixture.graph, fixture.sources, *bound.fact, 7, budget);
  ASSERT_TRUE(address.fact) << static_cast<int>(address.reason);
  EXPECT_EQ(address.fact->base, 0x2000U);
  EXPECT_EQ(address.fact->stride, 1U);
  EXPECT_TRUE(address.fact->placed);
  constexpr std::array<std::uint8_t, 4> rows{0, 4, 8, 4};
  const ir::ConstantImageRange range{0x2000, rows};

  // The base is a PC page, so it names the table only under page-aligned placement.
  const ir::ImageFacts unplaced{std::span(&range, 1), {}, false, {}};
  const auto refused = recovery::ProposeBoundedTableLoads(
      fixture.graph, *address.fact, fixture.sources, unplaced, {true, true, true}, budget);
  EXPECT_FALSE(refused.provisional);
  EXPECT_EQ(refused.reason, recovery::ConstantLoadRefusal::no_invariant);

  const ir::ImageFacts placed{std::span(&range, 1), {}, true, {}};
  auto folded = recovery::ProposeBoundedTableLoads(fixture.graph, *address.fact, fixture.sources,
                                                   placed, {true, true, true}, budget);
  ASSERT_TRUE(folded.provisional) << static_cast<int>(folded.reason);
  auto& graph = *folded.provisional;
  const ir::SsaHandle table{graph.arena(), fixture.table.slot, fixture.table.generation};
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(*graph.Get(table), budget), ir::SsaDecline::none);
  EXPECT_TRUE(
      ProveSsaReachability(graph, fixture.sources, ir::SsaEntryScope::closed_population, budget)
          .facts);
  auto clone = graph.Clone(budget);
  ASSERT_TRUE(clone);
  ASSERT_TRUE(
      clone->Update(ir::SsaHandle{clone->arena(), table.slot, table.generation},
                    [](auto& block) { block.constant_loads[0].page_aligned_placement = false; }));
  EXPECT_EQ(ir::ValidateSsa(*clone, budget), ir::SsaDecline::invalid_graph);

  // The guard's limit is the table's length; one more row is one the guard lets through.
  clone = graph.Clone(budget);
  ASSERT_TRUE(clone);
  ASSERT_TRUE(
      clone->Update(ir::SsaHandle{clone->arena(), table.slot, table.generation},
                    [](auto& block) { block.constant_loads[0].table_bytes.push_back({4}); }));
  EXPECT_EQ(ir::ValidateSsa(*clone, budget), ir::SsaDecline::invalid_graph);

  // The index must leave the guard holding the compared value.
  clone = graph.Clone(budget);
  ASSERT_TRUE(clone);
  const ir::SsaHandle guard{clone->arena(), fixture.guard.slot, fixture.guard.generation};
  ASSERT_TRUE(clone->Update(
      guard, [&](auto& block) { block.exits[0].value = {ir::SsaValueKind::phi, guard, 0}; }));
  ASSERT_TRUE(clone->Update(
      ir::SsaHandle{clone->arena(), table.slot, table.generation},
      [&](auto& block) { block.phis[0].incoming[0].value = {ir::SsaValueKind::phi, guard, 0}; }));
  EXPECT_EQ(ir::ValidateSsa(*clone, budget), ir::SsaDecline::invalid_graph);
}

std::optional<ir::SsaGraph> FoldA64Switch(const A64Switch& fixture, Budget& budget) {
  const auto bound =
      ProveSsaDirectIndexBound(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                               fixture.guard, 1, fixture.table, budget);
  if (!bound.fact) return {};
  const auto address =
      ProveSsaBoundedTableAddress(fixture.graph, fixture.sources, *bound.fact, 7, budget);
  if (!address.fact) return {};
  constexpr std::array<std::uint8_t, 4> rows{0, 4, 8, 4};
  const ir::ConstantImageRange range{0x2000, rows};
  const ir::ImageFacts placed{std::span(&range, 1), {}, true, {}};
  auto folded = recovery::ProposeBoundedTableLoads(fixture.graph, *address.fact, fixture.sources,
                                                   placed, {true, true, true}, budget);
  return std::move(folded.provisional);
}

// The fold still bounds its index and still completes its dispatch.
void ExpectFoldHolds(const ir::SsaGraph& graph, const A64Switch& fixture, Budget& budget) {
  EXPECT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  const auto* table = graph.Get({graph.arena(), fixture.table.slot, fixture.table.generation});
  ASSERT_TRUE(table);
  ASSERT_EQ(table->constant_loads.size(), 1U);
  EXPECT_EQ(table->constant_loads[0].kind, ir::SsaConstantKind::bounded_table);
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(*table, budget), ir::SsaDecline::none);
}

// Later passes rewrite the guard and the table block; the fold stays, since
// its bound is read off the graph as it stands.
TEST(SsaGuard, ATableFoldSurvivesLaterPassesInItsBlocks) {
  auto fixture = A64SwitchGraph();

  // Declared, the load bias turns the page arithmetic into numbers to fold.
  fixture.graph.SetLoadBias(0);
  auto budget = Plenty();
  auto folded = FoldA64Switch(fixture, budget);
  ASSERT_TRUE(folded);
  const auto in_fold_blocks = [&](const auto& journal) {
    std::set<std::uint32_t> slots;
    for (const auto& edit : journal)
      if (edit.original_block.slot == fixture.guard.slot ||
          edit.original_block.slot == fixture.table.slot)
        slots.insert(edit.original_block.slot);
    return slots;
  };

  std::set<std::uint32_t> touched;

  auto simplified = recovery::ProposeSsaSimplify(*folded, fixture.sources, true, budget);
  ASSERT_TRUE(simplified.provisional) << static_cast<int>(simplified.reason);
  touched.merge(in_fold_blocks(simplified.journal));
  ExpectFoldHolds(*simplified.provisional, fixture, budget);
  const auto live = ProveSsaNodeLiveness(*simplified.provisional, budget);
  ASSERT_TRUE(live.facts);
  auto retired =
      recovery::ProposeDeadPureNodes(*simplified.provisional, *live.facts, fixture.sources, budget);
  ASSERT_TRUE(retired.provisional) << static_cast<int>(retired.reason);
  touched.merge(in_fold_blocks(retired.journal));
  ExpectFoldHolds(*retired.provisional, fixture, budget);

  const auto reach =
      ProveSsaReachability(*folded, fixture.sources, ir::SsaEntryScope::closed_population, budget);
  ASSERT_TRUE(reach.facts);
  const auto constants = ProveSsaConstants(*folded, *reach.facts, fixture.sources, budget);
  ASSERT_TRUE(constants.facts);
  auto propagated = recovery::ProposeSsaConstantFold(*folded, *reach.facts, *constants.facts,
                                                     fixture.sources, budget);
  ASSERT_TRUE(propagated.provisional) << static_cast<int>(propagated.reason);
  EXPECT_FALSE(in_fold_blocks(propagated.journal).empty());
  touched.merge(in_fold_blocks(propagated.journal));
  ExpectFoldHolds(*propagated.provisional, fixture, budget);

  const auto sccp =
      ProveSsaSccp(*folded, ir::SsaEntryScope::closed_population, fixture.sources, budget);
  ASSERT_TRUE(sccp.facts) << static_cast<int>(sccp.reason);
  auto swept = recovery::ProposeSsaSccpFold(*folded, *sccp.facts, fixture.sources, budget);
  ASSERT_TRUE(swept.provisional) << static_cast<int>(swept.reason);
  EXPECT_FALSE(in_fold_blocks(swept.journal).empty());
  ExpectFoldHolds(*swept.provisional, fixture, budget);

  EXPECT_EQ(touched, (std::set<std::uint32_t>{fixture.guard.slot, fixture.table.slot}));
}

// An edit after which the bound no longer derives refuses the graph rather
// than keeping a fold that no longer holds.
TEST(SsaGuard, AnEditThatBreaksTheBoundIsRefused) {
  auto fixture = A64SwitchGraph();
  auto budget = Plenty();
  auto folded = FoldA64Switch(fixture, budget);
  ASSERT_TRUE(folded);
  ExpectFoldHolds(*folded, fixture, budget);
  const auto edit = [&](ir::SsaHandle handle, auto change) {
    auto clone = folded->Clone(budget);
    EXPECT_TRUE(clone);
    EXPECT_TRUE(clone->Update({clone->arena(), handle.slot, handle.generation}, change));
    return ir::ValidateSsa(*clone, budget);
  };

  // The guard lets 4 through, which the four rows do not cover.
  EXPECT_EQ(edit(fixture.guard, [](auto& block) { block.nodes[6].immediate = 4; }),
            ir::SsaDecline::invalid_graph);
  // The guard compares a different number from the one it hands on.
  EXPECT_EQ(edit(fixture.guard,
                 [](auto& block) {
                   block.nodes[5].inputs = {0};
                   block.nodes[5].width = 32;
                 }),
            ir::SsaDecline::invalid_graph);
  // The table is addressed from elsewhere.
  EXPECT_EQ(edit(fixture.table, [](auto& block) { block.nodes[4].immediate = 0x2008; }),
            ir::SsaDecline::invalid_graph);
  // The index read is no longer the phi the guard feeds.
  EXPECT_EQ(edit(fixture.table, [](auto& block) { block.nodes[6].inputs = {5, 5}; }),
            ir::SsaDecline::invalid_graph);
}

TEST(SsaGuard, BoundedTableFoldJournalsDeclaredBytes) {
  auto fixture = GuardedTableGraph();
  auto budget = Plenty();
  const auto bound =
      ProveSsaDirectIndexBound(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                               fixture.branch, 0, fixture.table, budget);
  ASSERT_TRUE(bound.fact);
  const auto address =
      ProveSsaBoundedTableAddress(fixture.graph, fixture.sources, *bound.fact, 5, budget);
  ASSERT_TRUE(address.fact);
  constexpr std::array<std::uint8_t, 32> bytes{11, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0,
                                               33, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0};
  const ir::ConstantImageRange range{0x2000, bytes};
  const ir::ImageFacts image{std::span(&range, 1), {}, true, {}};
  const ir::ImageAccessContract access{true, true, true};
  auto result = recovery::ProposeBoundedTableLoads(fixture.graph, *address.fact, fixture.sources,
                                                   image, access, budget);
  ASSERT_TRUE(result.provisional) << static_cast<int>(result.reason);
  ASSERT_EQ(result.journal.size(), 1U);
  EXPECT_TRUE(result.journal[0].table_address);
  EXPECT_EQ(result.journal[0].fold.table_bytes.size(), 4U);
  EXPECT_EQ(result.journal[0].fold.table_bytes[2][0], 33U);
  EXPECT_TRUE(fixture.graph.Get(fixture.table)->constant_loads.empty());
  EXPECT_EQ(ir::ValidateSsa(*result.provisional, budget), ir::SsaDecline::none);
  EXPECT_TRUE(ir::PrintSsa(*result.provisional, budget).text);
  EXPECT_TRUE(ir::PrintSsa(*result.provisional, fixture.sources, budget).text);
  EXPECT_NE(recovery::ProposeLinearMba(*result.provisional, budget).reason,
            recovery::MbaDecline::invalid_ir);

  const ir::RelocatedPointer pointer{0x2010, 0x3000};
  const ir::ImageFacts relocated{std::span(&range, 1), std::span(&pointer, 1), true, {}};
  const auto refused = recovery::ProposeBoundedTableLoads(
      fixture.graph, *address.fact, fixture.sources, relocated, access, budget);
  EXPECT_FALSE(refused.provisional);
  EXPECT_EQ(refused.reason, recovery::ConstantLoadRefusal::no_invariant);

  constexpr std::array<std::uint8_t, 8> conflicting_bytes{99, 0, 0, 0, 0, 0, 0, 0};
  const std::array<ir::ConstantImageRange, 2> conflicting{
      {{0x2000, conflicting_bytes}, {0x2000, bytes}}};
  const ir::ImageFacts overlapping{conflicting, {}, true, {}};
  const auto overlap = recovery::ProposeBoundedTableLoads(
      fixture.graph, *address.fact, fixture.sources, overlapping, access, budget);
  EXPECT_FALSE(overlap.provisional);
  EXPECT_EQ(overlap.reason, recovery::ConstantLoadRefusal::no_invariant);
}

TEST(SsaGuard, BoundedTableFoldMatchesOriginalAndMutationRefutes) {
  auto fixture = GuardedTableGraph();
  auto budget = Plenty();
  const auto bound =
      ProveSsaDirectIndexBound(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                               fixture.branch, 0, fixture.table, budget);
  ASSERT_TRUE(bound.fact);
  const auto address =
      ProveSsaBoundedTableAddress(fixture.graph, fixture.sources, *bound.fact, 5, budget);
  ASSERT_TRUE(address.fact);
  constexpr std::array<std::uint8_t, 32> bytes{11, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0,
                                               33, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0};
  const ir::ConstantImageRange range{0x2000, bytes};
  const ir::ImageFacts image{std::span(&range, 1), {}, true, {}};
  const ir::ImageAccessContract access{true, true, true};
  auto proposed = recovery::ProposeBoundedTableLoads(fixture.graph, *address.fact, fixture.sources,
                                                     image, access, budget);
  ASSERT_TRUE(proposed.provisional);
  constexpr std::uint64_t bias = 0x100000000;
  const std::array<eval::RegionInput, 1> regions{{{bias + 0x2000, bytes}}};
  const auto run = [&](const ir::SsaGraph& graph, std::uint64_t input) {
    auto mapped = eval::Memory::Create(regions, budget);
    EXPECT_TRUE(mapped.memory);
    eval::State state;
    auto x0 = BitVector::from_u64(64, input, 64, budget);
    auto lr = BitVector::from_u64(64, bias + 0x3000, 64, budget);
    EXPECT_TRUE(x0 && lr);
    state.cells.push_back({0, std::move(*x0)});
    state.cells.push_back({30, std::move(*lr)});
    const auto entry = graph.entries()[0];
    const auto result =
        eval::ExecuteSsa(graph, fixture.sources, entry, state, *mapped.memory, budget, {bias});
    EXPECT_EQ(result.outcome, eval::Outcome::completed);
    EXPECT_EQ(result.stop, eval::SsaStop::returned);
    return state.cells[0].value.word(0);
  };

  for (std::uint64_t input = 0; input < 4; ++input)
    EXPECT_EQ(run(*proposed.provisional, input), run(fixture.graph, input));
  const auto folded_table =
      ir::SsaHandle{proposed.provisional->arena(), fixture.table.slot, fixture.table.generation};
  ASSERT_TRUE(proposed.provisional->Update(
      folded_table, [](auto& block) { block.constant_loads[0].table_bytes[2][0] = 99; }));
  EXPECT_EQ(ir::ValidateSsa(*proposed.provisional, budget), ir::SsaDecline::none);
  EXPECT_NE(run(*proposed.provisional, 2), run(fixture.graph, 2));
  const auto folded_guard =
      ir::SsaHandle{proposed.provisional->arena(), fixture.branch.slot, fixture.branch.generation};
  ASSERT_TRUE(proposed.provisional->Update(folded_guard,
                                           [](auto& block) { block.nodes[2].op = ir::Op::equal; }));
  EXPECT_EQ(ir::ValidateSsa(*proposed.provisional, budget), ir::SsaDecline::invalid_graph);

  auto changed_limit = recovery::ProposeBoundedTableLoads(fixture.graph, *address.fact,
                                                          fixture.sources, image, access, budget);
  ASSERT_TRUE(changed_limit.provisional);
  const auto limit_guard = ir::SsaHandle{changed_limit.provisional->arena(), fixture.branch.slot,
                                         fixture.branch.generation};
  const auto limit_table = ir::SsaHandle{changed_limit.provisional->arena(), fixture.table.slot,
                                         fixture.table.generation};
  // A guard edited to let a fifth index through leaves the fold's four rows
  // short of the bound it now derives, so the graph is refused.
  ASSERT_TRUE(changed_limit.provisional->Update(limit_guard,
                                                [](auto& block) { block.nodes[1].immediate = 5; }));
  EXPECT_EQ(ir::ValidateSsa(*changed_limit.provisional, budget), ir::SsaDecline::invalid_graph);

  // With a fifth declared row the fold is consistent again. Nothing binds the
  // guard to its decoded source any more, so it is the comparison with the
  // original that refutes the edit, on the index the edit let through.
  ASSERT_TRUE(changed_limit.provisional->Update(limit_table, [](auto& block) {
    block.constant_loads[0].table_bytes.push_back({55, 0, 0, 0, 0, 0, 0, 0});
  }));
  EXPECT_EQ(ir::ValidateSsa(*changed_limit.provisional, budget), ir::SsaDecline::none);
  EXPECT_NE(run(*changed_limit.provisional, 4), run(fixture.graph, 4));

  auto bypass = recovery::ProposeBoundedTableLoads(fixture.graph, *address.fact, fixture.sources,
                                                   image, access, budget);
  ASSERT_TRUE(bypass.provisional);
  const auto bypass_table =
      ir::SsaHandle{bypass.provisional->arena(), fixture.table.slot, fixture.table.generation};
  ASSERT_TRUE(bypass.provisional->Update(bypass_table, [](auto& block) {
    for (auto& phi : block.phis) phi.external_entry = true;
  }));
  bypass.provisional->SetEntries({bypass.provisional->entries()[0], bypass_table});
  EXPECT_EQ(ir::ValidateSsa(*bypass.provisional, budget), ir::SsaDecline::invalid_graph);
}

TEST(SsaGuard, BoundedTableFoldRetiresOnlyAddressInputs) {
  auto fixture = GuardedTableGraph();
  auto budget = Plenty();
  const auto bound =
      ProveSsaDirectIndexBound(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                               fixture.branch, 0, fixture.table, budget);
  ASSERT_TRUE(bound.fact);
  const auto address =
      ProveSsaBoundedTableAddress(fixture.graph, fixture.sources, *bound.fact, 5, budget);
  ASSERT_TRUE(address.fact);
  constexpr std::array<std::uint8_t, 32> bytes{11, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0,
                                               33, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0};
  const ir::ConstantImageRange range{0x2000, bytes};
  const ir::ImageFacts image{std::span(&range, 1), {}, true, {}};
  auto folded = recovery::ProposeBoundedTableLoads(fixture.graph, *address.fact, fixture.sources,
                                                   image, {true, true, true}, budget);
  ASSERT_TRUE(folded.provisional);
  auto liveness_budget = Plenty();
  const auto live = ProveSsaNodeLiveness(*folded.provisional, liveness_budget);
  ASSERT_TRUE(live.facts);
  const auto liveness_used = liveness_budget.used();
  ASSERT_LT(liveness_used.work, 100000U);
  for (std::uint64_t cut = 0; cut < liveness_used.work; ++cut) {
    Budget limited({cut, liveness_used.bytes});
    const auto refused = ProveSsaNodeLiveness(*folded.provisional, limited);
    EXPECT_FALSE(refused.facts) << cut;
    EXPECT_EQ(refused.reason, SsaLivenessRefusal::resource_limit) << cut;
  }

  const auto& table_live = live.facts->live_nodes[fixture.table.slot];
  EXPECT_EQ(table_live[0], 1U);
  for (const auto id : {1U, 2U, 3U, 4U}) EXPECT_EQ(table_live[id], 0U);
  EXPECT_EQ(table_live[5], 1U);
  auto retired =
      recovery::ProposeDeadPureNodes(*folded.provisional, *live.facts, fixture.sources, budget);
  ASSERT_TRUE(retired.provisional) << static_cast<int>(retired.reason);
  EXPECT_EQ(retired.journal.size(), 4U);
  EXPECT_EQ(ir::ValidateSsaWithSources(*retired.provisional, fixture.sources, budget),
            ir::SsaDecline::none);
  const auto table =
      ir::SsaHandle{retired.provisional->arena(), fixture.table.slot, fixture.table.generation};
  EXPECT_EQ(retired.provisional->Get(table)->dead_pure_nodes.size(), 4U);
  auto forged = *live.facts;
  forged.live_nodes[fixture.table.slot][0] = 0;
  const auto refused =
      recovery::ProposeDeadPureNodes(*folded.provisional, forged, fixture.sources, budget);
  EXPECT_FALSE(refused.provisional);
  EXPECT_EQ(refused.reason, recovery::SsaPureDceRefusal::stale_proof);
  auto dce_budget = Plenty();
  ASSERT_TRUE(
      recovery::ProposeDeadPureNodes(*folded.provisional, *live.facts, fixture.sources, dce_budget)
          .provisional);
  const auto dce_used = dce_budget.used();
  ASSERT_LT(dce_used.work, 100000U);
  for (std::uint64_t cut = 0; cut < dce_used.work; ++cut) {
    Budget limited({cut, dce_used.bytes});
    const auto limited_result =
        recovery::ProposeDeadPureNodes(*folded.provisional, *live.facts, fixture.sources, limited);
    EXPECT_FALSE(limited_result.provisional) << cut;
    EXPECT_TRUE(limited_result.journal.empty()) << cut;
    EXPECT_EQ(limited_result.reason, recovery::SsaPureDceRefusal::resource_limit) << cut;
  }
}

TEST(SsaGuard, BoundedTableFoldBudgetRefusesBeforePublishing) {
  auto fixture = GuardedTableGraph();
  auto budget = Plenty();
  const auto bound =
      ProveSsaDirectIndexBound(fixture.graph, fixture.sources, ir::SsaEntryScope::closed_population,
                               fixture.branch, 0, fixture.table, budget);
  ASSERT_TRUE(bound.fact);
  const auto address =
      ProveSsaBoundedTableAddress(fixture.graph, fixture.sources, *bound.fact, 5, budget);
  ASSERT_TRUE(address.fact);
  constexpr std::array<std::uint8_t, 32> bytes{11, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0,
                                               33, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0};
  const ir::ConstantImageRange range{0x2000, bytes};
  const ir::ImageFacts image{std::span(&range, 1), {}, true, {}};
  const ir::ImageAccessContract access{true, true, true};
  auto full = Plenty();
  ASSERT_TRUE(recovery::ProposeBoundedTableLoads(fixture.graph, *address.fact, fixture.sources,
                                                 image, access, full)
                  .provisional);
  const auto used = full.used();
  ASSERT_LT(used.work, 100000U);
  for (std::uint64_t cut = 0; cut < used.work; ++cut) {
    Budget limited({cut, used.bytes});
    const auto result = recovery::ProposeBoundedTableLoads(fixture.graph, *address.fact,
                                                           fixture.sources, image, access, limited);
    EXPECT_FALSE(result.provisional) << cut;
    EXPECT_TRUE(result.journal.empty()) << cut;
    EXPECT_EQ(result.reason, recovery::ConstantLoadRefusal::resource_limit) << cut;
  }

  for (const auto cut : {std::uint64_t{0}, used.bytes - 1}) {
    Budget limited({used.work, cut});
    const auto result = recovery::ProposeBoundedTableLoads(fixture.graph, *address.fact,
                                                           fixture.sources, image, access, limited);
    EXPECT_FALSE(result.provisional) << cut;
    EXPECT_TRUE(result.journal.empty()) << cut;
    EXPECT_EQ(result.reason, recovery::ConstantLoadRefusal::resource_limit) << cut;
  }
}
}  // namespace
}  // namespace nyx::analysis
