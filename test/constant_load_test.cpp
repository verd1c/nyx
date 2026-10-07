#include "nyx/recovery/constant_load.hpp"

#include <algorithm>
#include <array>

#include <gtest/gtest.h>

#include "nyx/analysis/ssa/image_addresses.hpp"
#include "nyx/analysis/ssa/liveness.hpp"
#include "nyx/recovery/ssa/pure_dce.hpp"

namespace nyx::recovery {
namespace {
Budget Plenty() { return Budget({1000000, 10000000}); }

ir::SsaGraph Graph(bool write = false) {
  ir::SsaGraph graph;
  ir::SsaBlock block{};
  block.original_block = 0;
  block.address = 0x100;
  block.source_groups = {0x100};
  block.original_sources = {0};
  block.nodes = {{ir::Op::image_address, 64, {}, 0x2000}, {ir::Op::load, 64, {0}}};
  if (write) {
    block.nodes.push_back({ir::Op::constant, 64, {}, 42});
    block.nodes.push_back({ir::Op::store, 64, {0, 2}});
  }

  block.boundaries = {{0, static_cast<std::uint32_t>(block.nodes.size()), {{0, 1}}, {}}};
  block.phis = {{0, 64, true, {}}};
  block.clobbers = {0};
  const auto handle = graph.Add(std::move(block));
  graph.Update(handle, [&](auto& inserted) {
    inserted.exits = {{0, {ir::SsaValueKind::node, handle, 1}}};
    inserted.edges = {{ir::SsaEdgeKind::return_, ir::SsaTargetKind::unknown, 0, {}, {}, {}, {}}};
  });
  graph.SetEntries({handle});
  return graph;
}

ir::SsaGraph SelectedGraph() {
  auto graph = Graph();
  const auto handle = graph.entries()[0];
  graph.Update(handle, [&](auto& block) {
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
    block.boundaries[0].node_count = 11;
    block.boundaries[0].writes = {{0, 10}};
    block.reads = {{0, 0, {ir::SsaValueKind::phi, handle, 0}}};
    block.exits[0].value = {ir::SsaValueKind::node, handle, 10};
  });
  return graph;
}

TEST(ConstantLoad, FoldsMappedDeclaredBytesAndRejectsMutation) {
  auto graph = Graph();
  auto budget = Plenty();
  constexpr std::array<std::uint8_t, 8> bytes{0x78, 0x56, 0x34, 0x12, 0, 0, 0, 0};
  const ir::ConstantImageRange range{0x2000, bytes};
  const ir::ImageFacts facts{std::span(&range, 1), {}, true, {}};
  const ir::ImageAccessContract mapped{true, true, true};
  auto result = ProposeConstantImageLoads(graph, facts, mapped, budget);
  ASSERT_TRUE(result.provisional) << static_cast<int>(result.reason);
  ASSERT_EQ(result.journal.size(), 1);
  const auto* changed = result.provisional->Get(result.provisional->entries()[0]);
  ASSERT_EQ(changed->constant_loads.size(), 1);
  EXPECT_EQ(changed->constant_loads[0].kind, ir::SsaConstantKind::literal);
  EXPECT_EQ(changed->constant_loads[0].value, 0x12345678U);
  EXPECT_TRUE(changed->constant_loads[0].skip_access);
  EXPECT_EQ(changed->disabled_effects, (std::vector<ir::ValueId>{1}));
  EXPECT_TRUE(graph.Get(graph.entries()[0])->disabled_effects.empty());
  EXPECT_EQ(result.journal[0].fact, ConstantLoadFact::constant_range);
  EXPECT_EQ(result.journal[0].fact_address, 0x2000U);
  EXPECT_EQ(result.journal[0].fact_bytes, 8U);
  EXPECT_TRUE(result.access.mapped_readable_lifetime && result.access.no_runtime_unmapping &&
              result.access.ordinary_reads_unobservable);

  auto full = Plenty();
  ASSERT_TRUE(ProposeConstantImageLoads(graph, facts, mapped, full).provisional);
  const auto used = full.used();
  ASSERT_LT(used.work, 10000U);
  for (std::uint64_t cut = 0; cut < used.work; ++cut) {
    Budget limited({cut, used.bytes});
    const auto refused = ProposeConstantImageLoads(graph, facts, mapped, limited);
    EXPECT_EQ(refused.reason, ConstantLoadRefusal::resource_limit) << cut;
    EXPECT_FALSE(refused.provisional) << cut;
    EXPECT_TRUE(refused.journal.empty()) << cut;
  }

  ASSERT_TRUE(result.provisional->Update(result.provisional->entries()[0], [](auto& wrong) {
    wrong.constant_loads[0].source_address++;
  }));
  EXPECT_EQ(ir::ValidateSsa(*result.provisional, budget), ir::SsaDecline::invalid_graph);

  auto mismatched = ProposeConstantImageLoads(graph, facts, mapped, budget);
  ASSERT_TRUE(mismatched.provisional);
  ASSERT_TRUE(mismatched.provisional->Update(mismatched.provisional->entries()[0],
                                             [](auto& wrong) { ++wrong.constant_loads[0].value; }));
  EXPECT_EQ(ir::ValidateSsa(*mismatched.provisional, budget), ir::SsaDecline::invalid_graph);

  auto contradicted = ProposeConstantImageLoads(graph, facts, mapped, budget);
  ASSERT_TRUE(contradicted.provisional);
  ASSERT_TRUE(
      contradicted.provisional->Update(contradicted.provisional->entries()[0], [](auto& block) {
        block.nodes.push_back({ir::Op::constant, 64, {}, 9});
        block.nodes.push_back({ir::Op::store, 64, {0, 2}});
        block.boundaries[0].node_count += 2;
      }));
  EXPECT_EQ(ir::ValidateSsa(*contradicted.provisional, budget), ir::SsaDecline::invalid_graph);

  auto forged = Graph();
  ASSERT_TRUE(forged.Update(forged.entries()[0], [](auto& block) {
    block.disabled_effects = {1};
    block.constant_loads = {{1, ir::SsaConstantKind::literal, 7, 0x2000}};
    block.constant_loads[0].skip_access = true;
  }));
  EXPECT_EQ(ir::ValidateSsa(forged, budget), ir::SsaDecline::invalid_graph);
}

TEST(ConstantLoad, NamesTheEnclosingDeclaredRange) {
  auto graph = Graph();
  auto budget = Plenty();
  std::array<std::uint8_t, 24> bytes{};
  bytes[8] = 7;
  const std::array<ir::ConstantImageRange, 2> ranges{{{0x1000, bytes}, {0x1ff8, bytes}}};
  const ir::ImageFacts facts{ranges, {}, true, {}};
  auto result = ProposeConstantImageLoads(graph, facts, {true, true, true}, budget);
  ASSERT_EQ(result.journal.size(), 1);
  EXPECT_EQ(result.journal[0].fold.value, 7U);
  EXPECT_EQ(result.journal[0].fact_address, 0x1ff8U);
  EXPECT_EQ(result.journal[0].fact_bytes, 24U);
}

TEST(ConstantLoad, KeepsRelocationSymbolicAndRefusesFaultOrWrite) {
  auto graph = Graph();
  auto budget = Plenty();
  const ir::RelocatedPointer pointer{0x2000, 0x3000, true};
  const ir::ImageFacts facts{{}, std::span(&pointer, 1), true, {}};
  const ir::ImageAccessContract mapped{true, true, true};
  auto symbolic = ProposeConstantImageLoads(graph, facts, mapped, budget);
  ASSERT_TRUE(symbolic.provisional);
  EXPECT_EQ(symbolic.journal[0].fold.kind, ir::SsaConstantKind::image_location);
  EXPECT_EQ(symbolic.journal[0].fold.value, 0x3000U);
  EXPECT_EQ(symbolic.journal[0].fact, ConstantLoadFact::relocated_slot);
  EXPECT_EQ(symbolic.journal[0].fact_address, 0x2000U);

  auto big_endian = Graph();
  ASSERT_TRUE(big_endian.Update(big_endian.entries()[0], [](auto& block) {
    block.nodes[1].access.byte_order = ir::ByteOrder::big;
  }));
  auto endian = ProposeConstantImageLoads(big_endian, facts, mapped, budget);
  EXPECT_FALSE(endian.provisional);
  ASSERT_EQ(endian.refused.size(), 1);
  EXPECT_EQ(endian.refused[0].reason, ConstantLoadRefusal::unsupported_access);

  auto faulting = ProposeConstantImageLoads(graph, facts, {}, budget);
  EXPECT_FALSE(faulting.provisional);
  ASSERT_EQ(faulting.refused.size(), 1);
  EXPECT_EQ(faulting.refused[0].reason, ConstantLoadRefusal::not_nonfaulting);

  auto unknown = ProposeConstantImageLoads(graph, {}, mapped, budget);
  EXPECT_FALSE(unknown.provisional);
  ASSERT_EQ(unknown.refused.size(), 1);
  EXPECT_EQ(unknown.refused[0].reason, ConstantLoadRefusal::no_invariant);

  auto writing = Graph(true);
  auto conflict = ProposeConstantImageLoads(writing, facts, mapped, budget);
  EXPECT_FALSE(conflict.provisional);
  ASSERT_EQ(conflict.refused.size(), 1);
  EXPECT_EQ(conflict.refused[0].reason, ConstantLoadRefusal::conflicting_store);

  auto misaligned = Graph();
  ASSERT_TRUE(misaligned.Update(misaligned.entries()[0], [](auto& block) {
    block.nodes[0].immediate = 0x2004;
    block.nodes[1].access.alignment = 8;
    block.nodes[1].access.decline_on_unaligned = false;
  }));
  constexpr std::array<std::uint8_t, 16> bytes{};
  const ir::ConstantImageRange range{0x2000, bytes};
  const ir::ImageFacts aligned_facts{std::span(&range, 1), {}, true, {}};
  auto alignment = ProposeConstantImageLoads(misaligned, aligned_facts, mapped, budget);
  EXPECT_FALSE(alignment.provisional);
  ASSERT_EQ(alignment.refused.size(), 1);
  EXPECT_EQ(alignment.refused[0].reason, ConstantLoadRefusal::alignment);
}

TEST(ConstantLoad, SelectedTableLoadRetiresAddressButKeepsCondition) {
  auto graph = SelectedGraph();
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  const auto address = analysis::ProveSsaSelectedImageAddresses(graph, budget);
  ASSERT_TRUE(address.facts);
  ASSERT_EQ(address.facts->selected.size(), 1);
  constexpr std::array<std::uint8_t, 8> first{11}, second{22};
  const std::array<ir::ConstantImageRange, 2> ranges{{{0x2000, first}, {0x2008, second}}};
  const ir::ImageFacts facts{ranges, {}, true, {}};
  auto folded = ProposeSelectedImageLoads(graph, *address.facts, facts, {true, true, true}, budget);
  ASSERT_TRUE(folded.provisional) << static_cast<int>(folded.reason);
  ASSERT_EQ(folded.journal.size(), 1);
  const auto* block = folded.provisional->Get(folded.provisional->entries()[0]);
  ASSERT_EQ(block->constant_loads.size(), 1);
  EXPECT_EQ(block->constant_loads[0].condition, 2U);
  EXPECT_EQ(block->constant_loads[0].value, 22U);
  EXPECT_EQ(block->constant_loads[0].alternative_value, 11U);
  EXPECT_EQ(block->constant_loads[0].source_address, 0x2008U);
  EXPECT_EQ(block->constant_loads[0].alternative_source_address, 0x2000U);
  EXPECT_TRUE(block->constant_loads[0].skip_access);
  const auto live = analysis::ProveSsaNodeLiveness(*folded.provisional, budget);
  ASSERT_TRUE(live.facts);
  EXPECT_EQ(live.facts->live_nodes[0][2], 1);
  EXPECT_EQ(live.facts->live_nodes[0][9], 0);
  EXPECT_EQ(live.facts->live_nodes[0][10], 1);
  auto retired = ProposeDeadPureNodes(*folded.provisional, *live.facts, budget);
  ASSERT_TRUE(retired.provisional);
  EXPECT_EQ(ir::ValidateSsa(*retired.provisional, budget), ir::SsaDecline::none);
}

TEST(ConstantLoad, SelectedLoadRefusesMissingArmAndForgedAddress) {
  auto graph = SelectedGraph();
  auto budget = Plenty();
  auto addresses = analysis::ProveSsaSelectedImageAddresses(graph, budget);
  ASSERT_TRUE(addresses.facts);
  constexpr std::array<std::uint8_t, 8> first{11}, second{22};
  const std::array<ir::ConstantImageRange, 2> ranges{{{0x2000, first}, {0x2008, second}}};
  const ir::ImageFacts facts{ranges, {}, true, {}};
  auto missing = ProposeSelectedImageLoads(
      graph, *addresses.facts, {std::span(ranges.data(), 1), {}, true}, {true, true, true}, budget);
  EXPECT_FALSE(missing.provisional);
  ASSERT_EQ(missing.refused.size(), 1);
  EXPECT_EQ(missing.refused[0].reason, ConstantLoadRefusal::no_invariant);
  auto faulting = ProposeSelectedImageLoads(graph, *addresses.facts, facts, {}, budget);
  EXPECT_FALSE(faulting.provisional);
  ASSERT_EQ(faulting.refused.size(), 1);
  EXPECT_EQ(faulting.refused[0].reason, ConstantLoadRefusal::not_nonfaulting);
  ++addresses.facts->selected[0].when_true;
  auto forged =
      ProposeSelectedImageLoads(graph, *addresses.facts, facts, {true, true, true}, budget);
  EXPECT_FALSE(forged.provisional);
  EXPECT_EQ(forged.reason, ConstantLoadRefusal::invalid_graph);
}

TEST(ConstantLoad, ImageDerivedStoreContradictsBothFoldKinds) {
  auto budget = Plenty();
  constexpr std::array<std::uint8_t, 8> first{11}, second{22};
  const std::array<ir::ConstantImageRange, 2> ranges{{{0x2000, first}, {0x2008, second}}};
  const ir::ImageFacts facts{ranges, {}, true, {}};
  const ir::ImageAccessContract access{true, true, true};

  auto direct = Graph();
  ASSERT_TRUE(direct.Update(direct.entries()[0], [](auto& block) {
    block.nodes.push_back({ir::Op::image_address, 64, {}, 0x1ff8});
    block.nodes.push_back({ir::Op::constant, 64, {}, 8});
    block.nodes.push_back({ir::Op::add, 64, {2, 3}});
    block.nodes.push_back({ir::Op::constant, 64, {}, 42});
    block.nodes.push_back({ir::Op::store, 64, {4, 5}});
    block.boundaries[0].node_count = block.nodes.size();
  }));
  auto direct_fold = ProposeConstantImageLoads(direct, facts, access, budget);
  EXPECT_FALSE(direct_fold.provisional);
  ASSERT_EQ(direct_fold.refused.size(), 1);
  EXPECT_EQ(direct_fold.refused[0].reason, ConstantLoadRefusal::conflicting_store);

  auto separate = Graph();
  ASSERT_TRUE(separate.Update(separate.entries()[0], [](auto& block) {
    block.nodes.push_back({ir::Op::image_address, 64, {}, 0x3000});
    block.nodes.push_back({ir::Op::constant, 64, {}, 8});
    block.nodes.push_back({ir::Op::add, 64, {2, 3}});
    block.nodes.push_back({ir::Op::constant, 64, {}, 42});
    block.nodes.push_back({ir::Op::store, 64, {4, 5}});
    block.boundaries[0].node_count = block.nodes.size();
  }));
  EXPECT_TRUE(ProposeConstantImageLoads(separate, facts, access, budget).provisional);

  auto selected = SelectedGraph();
  auto proof = analysis::ProveSsaSelectedImageAddresses(selected, budget);
  ASSERT_TRUE(proof.facts);
  auto folded = ProposeSelectedImageLoads(selected, *proof.facts, facts, access, budget);
  ASSERT_TRUE(folded.provisional);
  const auto append_store = [](auto& block) {
    block.nodes.push_back({ir::Op::constant, 64, {}, 8});
    block.nodes.push_back({ir::Op::add, 64, {8, 11}});
    block.nodes.push_back({ir::Op::constant, 64, {}, 42});
    block.nodes.push_back({ir::Op::store, 64, {12, 13}});
    block.boundaries[0].node_count = block.nodes.size();
  };

  ASSERT_TRUE(selected.Update(selected.entries()[0], append_store));
  auto selected_proof = analysis::ProveSsaSelectedImageAddresses(selected, budget);
  ASSERT_TRUE(selected_proof.facts);
  auto conflict = ProposeSelectedImageLoads(selected, *selected_proof.facts, facts, access, budget);
  EXPECT_FALSE(conflict.provisional);
  ASSERT_EQ(conflict.refused.size(), 1);
  EXPECT_EQ(conflict.refused[0].reason, ConstantLoadRefusal::conflicting_store);
  ASSERT_TRUE(folded.provisional->Update(folded.provisional->entries()[0], append_store));
  EXPECT_EQ(ir::ValidateSsa(*folded.provisional, budget), ir::SsaDecline::invalid_graph);
}

// One block that runs `nodes` and returns the last of them.
ir::SsaGraph Straight(std::vector<ir::Node> nodes, std::optional<std::uint64_t> bias) {
  ir::SsaGraph graph;
  ir::SsaBlock block{};
  block.original_block = 0;
  block.address = 0x100;
  block.source_groups = {0x100};
  block.original_sources = {0};
  const auto last = static_cast<ir::ValueId>(nodes.size() - 1);
  block.nodes = std::move(nodes);
  block.boundaries = {{0, static_cast<std::uint32_t>(block.nodes.size()), {{0, last}}, {}}};
  block.phis = {{0, 64, true, {}}};
  block.clobbers = {0};
  const auto handle = graph.Add(std::move(block));
  graph.Update(handle, [&](auto& inserted) {
    inserted.exits = {{0, {ir::SsaValueKind::node, handle, last}}};
    inserted.edges = {{ir::SsaEdgeKind::return_, ir::SsaTargetKind::unknown, 0, {}, {}, {}, {}}};
  });
  graph.Update(handle, [&](auto& inserted) {
    for (ir::ValueId id = 0; id < inserted.nodes.size(); ++id)
      if (inserted.nodes[id].op == ir::Op::read) {
        inserted.nodes[id].storage = 0;
        inserted.reads.push_back({id, 0, {ir::SsaValueKind::phi, handle, 0}});
      }
  });
  graph.SetEntries({handle});
  graph.SetLoadBias(bias);
  return graph;
}

// Stores 42 at the address the last of `address` computes, then reads @0x3000.
ir::SsaGraph StoreThenRead(std::vector<ir::Node> address, std::optional<std::uint64_t> bias) {
  const auto at = static_cast<ir::ValueId>(address.size() - 1);
  address.push_back({ir::Op::constant, 64, {}, 42});
  address.push_back({ir::Op::store, 64, {at, at + 1}});
  address.push_back({ir::Op::image_address, 64, {}, 0x3000});
  address.push_back({ir::Op::load, 64, {at + 3}});
  return Straight(std::move(address), bias);
}

void AppendStoreThrough(ir::SsaBlock& block, ir::ValueId address) {
  block.nodes.push_back({ir::Op::constant, 64, {}, 42});
  block.nodes.push_back(
      {ir::Op::store, 64, {address, static_cast<ir::ValueId>(block.nodes.size() - 1)}});
  block.boundaries[0].node_count = block.nodes.size();
}

TEST(ConstantLoad, AStoreToTheNumberALocationHasUnderTheBiasContradictsIt) {
  constexpr std::uint64_t kBias = 0x10000;
  constexpr std::array<std::uint8_t, 8> bytes{7};
  const ir::ConstantImageRange range{0x3000, bytes};
  const ir::ImageFacts facts{std::span(&range, 1), {}, true, kBias};
  const ir::ImageAccessContract access{true, true, true};
  auto budget = Plenty();
  const auto refused = [&](const ir::SsaGraph& graph) {
    auto result = ProposeConstantImageLoads(graph, facts, access, budget);
    return !result.provisional && result.refused.size() == 1 &&
           result.refused[0].reason == ConstantLoadRefusal::conflicting_store;
  };

  // @0x3000 is the number 0x13000 once the image sits at the bias, however
  // the address that reaches it is spelled.
  EXPECT_TRUE(refused(StoreThenRead({{ir::Op::constant, 64, {}, kBias + 0x3000}}, kBias)));
  EXPECT_TRUE(refused(StoreThenRead({{ir::Op::constant, 64, {}, kBias + 0x2ff8},
                                     {ir::Op::constant, 64, {}, 8},
                                     {ir::Op::add, 64, {0, 1}}},
                                    kBias)));
  // Below the bias a number is no place in the image, and without a bias it
  // names none at all.
  EXPECT_TRUE(ProposeConstantImageLoads(StoreThenRead({{ir::Op::constant, 64, {}, 0x3000}}, kBias),
                                        facts, access, budget)
                  .provisional);
  const ir::ImageFacts unplaced{std::span(&range, 1), {}, true, {}};
  EXPECT_TRUE(
      ProposeConstantImageLoads(StoreThenRead({{ir::Op::constant, 64, {}, kBias + 0x3000}}, {}),
                                unplaced, access, budget)
          .provisional);

  // The checker reads the store the same way.
  auto folded = ProposeConstantImageLoads(
      Straight({{ir::Op::image_address, 64, {}, 0x3000}, {ir::Op::load, 64, {0}}}, kBias), facts,
      access, budget);
  ASSERT_TRUE(folded.provisional);
  ASSERT_EQ(ir::ValidateSsa(*folded.provisional, budget), ir::SsaDecline::none);
  ASSERT_TRUE(folded.provisional->Update(folded.provisional->entries()[0], [](auto& block) {
    block.nodes.push_back({ir::Op::constant, 64, {}, kBias + 0x3000});
    AppendStoreThrough(block, static_cast<ir::ValueId>(block.nodes.size() - 1));
  }));
  EXPECT_EQ(ir::ValidateSsa(*folded.provisional, budget), ir::SsaDecline::invalid_graph);
}

TEST(ConstantLoad, AStoreThroughADeclaredPointerContradictsWhereItPoints) {
  constexpr std::array<std::uint8_t, 8> target{7};
  const ir::ImageAccessContract access{true, true, true};
  auto budget = Plenty();
  const auto conflicts_at = [](const ConstantLoadResult& result, ir::ValueId node) {
    return std::any_of(result.refused.begin(), result.refused.end(), [&](const auto& item) {
      return item.node == node && item.reason == ConstantLoadRefusal::conflicting_store;
    });
  };

  // p = *@0x2000; *p = 42; then read @0x3000.
  const auto through_slot = [](std::optional<std::uint64_t> bias) {
    return StoreThenRead({{ir::Op::image_address, 64, {}, 0x2000}, {ir::Op::load, 64, {0}}}, bias);
  };

  // A relocated slot holds the location its relocation names.
  const ir::ConstantImageRange range{0x3000, target};
  const ir::RelocatedPointer pointer{0x2000, 0x3000, true};
  const ir::ImageFacts relocated{std::span(&range, 1), std::span(&pointer, 1), true, {}};
  auto result = ProposeConstantImageLoads(through_slot({}), relocated, access, budget);
  EXPECT_TRUE(conflicts_at(result, 5));
  EXPECT_EQ(result.journal.size(), 1U);
  if (!result.journal.empty()) {
    EXPECT_EQ(result.journal[0].fold.node, 1U);
  }

  // A declared constant slot holds a number, which the bias makes a place.
  constexpr std::array<std::uint8_t, 8> slot{0x00, 0x30};
  const std::array<ir::ConstantImageRange, 2> ranges{{{0x2000, slot}, {0x3000, target}}};
  const ir::ImageFacts numbered{ranges, {}, true, 0};
  result = ProposeConstantImageLoads(through_slot(0), numbered, access, budget);
  EXPECT_TRUE(conflicts_at(result, 5));

  // What an undeclared slot holds is the whole-image writer scan's to account for.
  const ir::ImageFacts undeclared{std::span(&range, 1), {}, true, {}};
  result = ProposeConstantImageLoads(through_slot({}), undeclared, access, budget);
  ASSERT_EQ(result.journal.size(), 1U);
  EXPECT_EQ(result.journal[0].fold.node, 5U);

  // A slot the graph itself overwrites no longer holds what was declared:
  // here it is made to point at @0x3000 instead of @0x5000.
  const ir::RelocatedPointer elsewhere{0x2000, 0x5000, true};
  const ir::ImageFacts misled{std::span(&range, 1), std::span(&elsewhere, 1), true, {}};
  result = ProposeConstantImageLoads(StoreThenRead({{ir::Op::image_address, 64, {}, 0x2000},
                                                    {ir::Op::image_address, 64, {}, 0x3000},
                                                    {ir::Op::store, 64, {0, 1}},
                                                    {ir::Op::load, 64, {0}}},
                                                   {}),
                                     misled, access, budget);
  EXPECT_TRUE(conflicts_at(result, 7));

  // The checker has no facts, only the fold of the slot the graph carries.
  auto folded = ProposeConstantImageLoads(Straight({{ir::Op::image_address, 64, {}, 0x2000},
                                                    {ir::Op::load, 64, {0}},
                                                    {ir::Op::image_address, 64, {}, 0x3000},
                                                    {ir::Op::load, 64, {2}}},
                                                   {}),
                                          relocated, access, budget);
  ASSERT_TRUE(folded.provisional);
  ASSERT_EQ(folded.journal.size(), 2U);
  ASSERT_TRUE(folded.provisional->Update(folded.provisional->entries()[0],
                                         [](auto& block) { AppendStoreThrough(block, 1); }));
  EXPECT_EQ(ir::ValidateSsa(*folded.provisional, budget), ir::SsaDecline::invalid_graph);
}

TEST(ConstantLoad, AStoreAtAnUnknownImageOffsetLeavesReadOnlyBytesAlone) {
  constexpr std::array<std::uint8_t, 8> bytes{7};
  const ir::ImageAccessContract access{true, true, true};
  auto budget = Plenty();

  // @0x5000 plus whatever @0x6000 holds: somewhere in the image, nobody says where.
  const std::vector<ir::Node> unplaced{{ir::Op::image_address, 64, {}, 0x5000},
                                       {ir::Op::image_address, 64, {}, 0x6000},
                                       {ir::Op::load, 64, {1}},
                                       {ir::Op::add, 64, {0, 2}}};
  const auto refused = [&](const ir::SsaGraph& graph, bool read_only) {
    const ir::ConstantImageRange range{0x3000, bytes, read_only};
    auto result =
        ProposeConstantImageLoads(graph, {std::span(&range, 1), {}, true, {}}, access, budget);
    return std::any_of(result.refused.begin(), result.refused.end(), [&](const auto& item) {
      return item.node == graph.Get(graph.entries()[0])->nodes.size() - 1 &&
             item.reason == ConstantLoadRefusal::conflicting_store;
    });
  };

  EXPECT_TRUE(refused(StoreThenRead(unplaced, {}), false));

  // A store there faults instead of changing it.
  EXPECT_FALSE(refused(StoreThenRead(unplaced, {}), true));

  // One placed there is the graph contradicting the declaration: under a bias
  // of zero, a field update through a null pointer is exactly this.
  EXPECT_TRUE(refused(StoreThenRead({{ir::Op::image_address, 64, {}, 0x3000}}, {}), true));

  // The checker takes the fold's own declaration.
  const ir::ConstantImageRange range{0x3000, bytes, true};
  auto folded = ProposeConstantImageLoads(
      Straight({{ir::Op::image_address, 64, {}, 0x3000}, {ir::Op::load, 64, {0}}}, {}),
      {std::span(&range, 1), {}, true, {}}, access, budget);
  ASSERT_TRUE(folded.provisional);
  const auto handle = folded.provisional->entries()[0];
  ASSERT_TRUE(folded.provisional->Get(handle)->constant_loads[0].read_only);
  ASSERT_TRUE(folded.provisional->Update(handle, [&](auto& block) {
    for (auto node : unplaced) {
      for (auto& input : node.inputs) input += 2;
      block.nodes.push_back(node);
    }

    AppendStoreThrough(block, static_cast<ir::ValueId>(block.nodes.size() - 1));
  }));
  EXPECT_EQ(ir::ValidateSsa(*folded.provisional, budget), ir::SsaDecline::none);
  ASSERT_TRUE(folded.provisional->Update(
      handle, [](auto& block) { block.constant_loads[0].read_only = false; }));
  EXPECT_EQ(ir::ValidateSsa(*folded.provisional, budget), ir::SsaDecline::invalid_graph);
}

TEST(ConstantLoad, AFoldThatContradictsAnOlderOneIsRefusedAloneAndSaysSo) {
  constexpr std::array<std::uint8_t, 16> bytes{7};
  const ir::ConstantImageRange range{0x3000, bytes};
  const ir::RelocatedPointer pointer{0x2000, 0x3000, true};
  const ir::ImageFacts facts{std::span(&range, 1), std::span(&pointer, 1), true, {}};
  const ir::ImageAccessContract access{true, true, true};
  auto budget = Plenty();

  // p = *x8; *p = 42; then read @0x3000. Nobody knows where x8 points, so the
  // read folds.
  auto folded = ProposeConstantImageLoads(
      StoreThenRead({{ir::Op::read, 64, {}, 0, 8}, {ir::Op::load, 64, {0}}}, {}), facts, access,
      budget);
  ASSERT_EQ(folded.journal.size(), 1U);
  ASSERT_EQ(folded.journal[0].fold.node, 5U);

  // A later edit shows x8 was the relocated slot, and adds another read.
  ASSERT_TRUE(folded.provisional->Update(folded.provisional->entries()[0], [](auto& block) {
    block.nodes[0] = {ir::Op::image_address, 64, {}, 0x2000};
    block.reads.clear();
    block.nodes.push_back({ir::Op::image_address, 64, {}, 0x3008});
    block.nodes.push_back({ir::Op::load, 64, {6}});
    block.boundaries[0].node_count = block.nodes.size();
  }));

  // The checker cannot place the store without a record of what the slot holds.
  ASSERT_EQ(ir::ValidateSsa(*folded.provisional, budget), ir::SsaDecline::none);

  // Folding the slot would give it one, contradicting the older fold. That
  // fold alone is refused, and named for what it found; the other goes ahead.
  auto again = ProposeConstantImageLoads(*folded.provisional, facts, access, budget);
  EXPECT_EQ(again.reason, ConstantLoadRefusal::none);
  ASSERT_TRUE(again.provisional);
  ASSERT_EQ(again.journal.size(), 1U);
  EXPECT_EQ(again.journal[0].fold.node, 7U);
  EXPECT_TRUE(std::any_of(again.refused.begin(), again.refused.end(), [](const auto& item) {
    return item.node == 1 && item.reason == ConstantLoadRefusal::contradicts_existing_fold;
  }));
  EXPECT_EQ(ir::ValidateSsa(*again.provisional, budget), ir::SsaDecline::none);
}

TEST(ConstantLoad, AStoreWithdrawsOnlyTheDeclaredBytesItWrites) {
  std::array<std::uint8_t, 32> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(i);
  const ir::ConstantImageRange range{0x2000, bytes, true};
  const ir::ImageFacts facts{std::span(&range, 1), {}, true, {}};
  auto budget = Plenty();

  // Four bytes at 0x2008, written by operation 9.
  const std::array<ir::ImageWrite, 1> one{{{0x2008, 32, 9}}};
  const auto spans = ir::RefuteConstantBytes(facts, one, budget);
  ASSERT_TRUE(spans);
  ASSERT_EQ(spans->size(), 1U);
  EXPECT_EQ((*spans)[0].address, 0x2008U);
  EXPECT_EQ((*spans)[0].bytes, 4U);
  EXPECT_EQ((*spans)[0].write.operation, 9U);
  EXPECT_FALSE((*spans)[0].whole);
  const auto retained = ir::RetainImageFacts(facts, *spans, {}, budget);
  ASSERT_TRUE(retained);
  ASSERT_EQ(retained->constants.size(), 2U);
  EXPECT_EQ(retained->constants[0].address, 0x2000U);
  EXPECT_EQ(retained->constants[0].bytes.size(), 8U);
  EXPECT_EQ(retained->constants[1].address, 0x200cU);
  EXPECT_EQ(retained->constants[1].bytes.size(), 20U);
  EXPECT_EQ(retained->constants[1].bytes[0], 12U);
  EXPECT_TRUE(retained->constants[1].read_only);
  EXPECT_EQ(retained->origins, (std::vector<std::size_t>{0, 0}));

  // What is left still folds; what the store wrote does not.
  const ir::ImageAccessContract access{true, true, true};
  const auto folds_at = [&](std::uint64_t location) {
    auto result = ProposeConstantImageLoads(
        Straight({{ir::Op::image_address, 64, {}, location}, {ir::Op::load, 64, {0}}}, {}),
        retained->view(), access, budget);
    return result.journal.size() == 1;
  };

  EXPECT_TRUE(folds_at(0x2000));
  EXPECT_TRUE(folds_at(0x2010));
  EXPECT_FALSE(folds_at(0x2004));
  EXPECT_FALSE(folds_at(0x2008));

  // A write across the range's first byte withdraws only what lies inside it.
  const std::array<ir::ImageWrite, 1> across{{{0x1ffc, 64, 3}}};
  const auto edge = ir::RefuteConstantBytes(facts, across, budget);
  ASSERT_TRUE(edge);
  ASSERT_EQ(edge->size(), 1U);
  EXPECT_EQ((*edge)[0].address, 0x2000U);
  EXPECT_EQ((*edge)[0].bytes, 4U);
  const auto trimmed = ir::RetainImageFacts(facts, *edge, {}, budget);
  ASSERT_TRUE(trimmed);
  ASSERT_EQ(trimmed->constants.size(), 1U);
  EXPECT_EQ(trimmed->constants[0].address, 0x2004U);
  EXPECT_EQ(trimmed->constants[0].bytes.size(), 28U);

  // Up to the bound each write cuts its own bytes; past it the range goes whole.
  for (const std::size_t count : {ir::kMaxRefutedSpans, ir::kMaxRefutedSpans + 1}) {
    std::vector<ir::ImageWrite> many;
    for (std::size_t i = 0; i < count; ++i)
      many.push_back({0x2000 + i % 32, 8, static_cast<ir::ValueId>(i)});
    const auto cut = ir::RefuteConstantBytes(facts, many, budget);
    ASSERT_TRUE(cut);
    const bool whole = count > ir::kMaxRefutedSpans;
    ASSERT_EQ(cut->size(), whole ? 1U : count);
    EXPECT_EQ((*cut)[0].whole, whole);
    if (whole) {
      EXPECT_EQ((*cut)[0].bytes, 32U);
      EXPECT_EQ((*cut)[0].write.operation, 0U);
    }

    const auto left = ir::RetainImageFacts(facts, *cut, {}, budget);
    ASSERT_TRUE(left);
    EXPECT_TRUE(left->constants.empty());
  }

  // Spans out of order, or outside their range, are refused rather than read.
  auto shuffled = *ir::RefuteConstantBytes(
      facts, std::array<ir::ImageWrite, 2>{{{0x2000, 8, 0}, {0x2010, 8, 1}}}, budget);
  std::swap(shuffled[0], shuffled[1]);
  EXPECT_FALSE(ir::RetainImageFacts(facts, shuffled, {}, budget));
  shuffled[0].address = 0x2040;
  EXPECT_FALSE(ir::RetainImageFacts(facts, std::span(shuffled).first(1), {}, budget));
}

TEST(ConstantLoad, AStoreNobodyPlacesDoesNotUnsettleADeclaredPointer) {
  // p = *@0x2000 (relocated to @0x3000); *p = 42; a store at @0x5000 plus
  // whatever @0x6000 holds; then read read-only @0x4000.
  constexpr std::array<std::uint8_t, 8> bytes{7};
  const ir::ConstantImageRange range{0x4000, bytes, true};
  const ir::RelocatedPointer pointer{0x2000, 0x3000, true};
  const ir::ImageFacts facts{std::span(&range, 1), std::span(&pointer, 1), true, {}};
  auto budget = Plenty();
  auto graph = Straight({{ir::Op::image_address, 64, {}, 0x2000},
                         {ir::Op::load, 64, {0}},
                         {ir::Op::constant, 64, {}, 42},
                         {ir::Op::store, 64, {1, 2}},
                         {ir::Op::image_address, 64, {}, 0x5000},
                         {ir::Op::image_address, 64, {}, 0x6000},
                         {ir::Op::load, 64, {5}},
                         {ir::Op::add, 64, {4, 6}},
                         {ir::Op::store, 64, {7, 2}},
                         {ir::Op::image_address, 64, {}, 0x4000},
                         {ir::Op::load, 64, {9}}},
                        {});
  // The slot the store was placed through could be what the unplaced store
  // hits, but that is the writer scan's to say, as it is for the slot itself.
  auto result = ProposeConstantImageLoads(graph, facts, {true, true, true}, budget);
  EXPECT_TRUE(std::any_of(result.journal.begin(), result.journal.end(),
                          [](const auto& edit) { return edit.fold.node == 10; }));
}
}  // namespace
}  // namespace nyx::recovery
