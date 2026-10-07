#include "nyx/analysis/frame_slots.hpp"

#include <array>

#include <gtest/gtest.h>

namespace nyx::analysis {
namespace {
Budget Plenty() { return Budget({1000000, 10000000}); }

PrivateFrameContract Contract() { return {31, -32, 0, true, true, true, 16, false, false}; }

ir::SsaGraph Graph(bool unknown_address = false, bool escape = false, bool frame_relation = false) {
  ir::SsaGraph graph;
  ir::SsaBlock block{};
  block.original_block = 0;
  block.address = 0x100;
  block.source_groups = {0x100};
  block.original_sources = {0};
  if (frame_relation) {
    block.entry_relations = {{31, 9, 16}};
    block.relation_declared_abi = true;
    block.relation_return_leaves = true;
  }

  block.nodes = {{ir::Op::read, 64, {}, 0, 31},
                 {ir::Op::constant, 64, {}, 16},
                 {ir::Op::sub, 64, {0, 1}},
                 {ir::Op::constant, 64, {}, 42},
                 {ir::Op::read, 64, {}, 0, 9},
                 {ir::Op::store, 64, {unknown_address ? 4U : 2U, escape ? 2U : 3U}},
                 {ir::Op::load, 64, {2}}};
  block.nodes[5].access.alignment = 8;
  block.nodes[5].access.decline_on_unaligned = true;
  block.nodes[6].access.alignment = 8;
  block.nodes[6].access.decline_on_unaligned = true;
  if (frame_relation) {
    block.nodes[5].inputs[0] = 4;
    block.nodes[6].inputs[0] = 4;
  }

  block.boundaries = {{0, static_cast<std::uint32_t>(block.nodes.size()), {}, {}}};
  block.phis = {{9, 64, true, {}}, {31, 64, true, {}}};
  block.clobbers = {0, 0};
  block.reads = {{0, 1, {}}, {4, 0, {}}};
  const auto handle = graph.Add(std::move(block));
  graph.Update(handle, [&](auto& inserted) {
    inserted.reads[0].value = {ir::SsaValueKind::phi, handle, 1};
    inserted.reads[1].value = {ir::SsaValueKind::phi, handle, 0};
    inserted.exits = {{9, {ir::SsaValueKind::phi, handle, 0}},
                      {31, {ir::SsaValueKind::phi, handle, 1}}};
    inserted.edges = {{ir::SsaEdgeKind::return_, ir::SsaTargetKind::unknown, 0, {}, {}, {}, {}}};
  });
  graph.SetEntries({handle});
  return graph;
}

TEST(PrivateFrame, ProvesMappedAlignedSlotAndKeepsOutsideAccessesOut) {
  auto graph = Graph();
  auto budget = Plenty();
  const auto result = ProvePrivateFrameSlots(graph, Contract(), budget);
  ASSERT_TRUE(result.facts) << static_cast<int>(result.reason);
  ASSERT_EQ(result.facts->slots.size(), 1);
  EXPECT_EQ(result.facts->slots[0].offset, -16);
  EXPECT_EQ(result.facts->slots[0].size, 8U);
  EXPECT_EQ(result.facts->slots[0].accesses.size(), 2);
  EXPECT_EQ(ValidatePrivateFrameFacts(*result.facts, graph, budget),
            ir::PrivateFrameFactDecline::none);
  ASSERT_TRUE(graph.Update(graph.entries()[0], [](auto&) {}));
  EXPECT_EQ(ValidatePrivateFrameFacts(*result.facts, graph, budget),
            ir::PrivateFrameFactDecline::invalid);

  // A frame address kept in a private slot stays inside the function; the
  // escapes are refused in the multi-block tests below.
  auto kept = Graph(false, true);
  auto budget2 = Plenty();
  const auto inside = ProvePrivateFrameSlots(kept, Contract(), budget2);
  ASSERT_TRUE(inside.facts) << static_cast<int>(inside.reason);
  EXPECT_EQ(inside.facts->slots.size(), 1U);

  // Under the contract no pointer from outside reaches the region, so a store
  // through an entry register leaves the slot to its load.
  auto unknown = Graph(true);
  auto budget3 = Plenty();
  const auto outside = ProvePrivateFrameSlots(unknown, Contract(), budget3);
  ASSERT_TRUE(outside.facts) << static_cast<int>(outside.reason);
  ASSERT_EQ(outside.facts->slots.size(), 1U);
  EXPECT_EQ(outside.facts->slots[0].accesses.size(), 1U);
  EXPECT_FALSE(outside.facts->slots[0].accesses[0].store);
}

TEST(PrivateFrame, EntryRelationDoesNotSurviveARegisterWrite) {
  ir::SsaGraph graph;
  ir::SsaBlock block{};
  block.original_block = 0;
  block.address = 0x100;
  block.source_groups = {0x100, 0x104};
  block.original_sources = {0, 1};
  block.entry_relations = {{40, 31, 16}};
  block.nodes = {
      {ir::Op::constant, 64, {}, 0x3000}, {ir::Op::read, 64, {}, 0, 40}, {ir::Op::load, 64, {1}}};
  block.nodes[2].access.alignment = 8;
  block.boundaries = {{0, 1, {{40, 0}}, {}}, {1, 2, {}, {}}};
  block.phis = {{31, 64, true, {}}, {40, 64, true, {}}};
  block.clobbers = {0, 0};
  const auto handle = graph.Add(std::move(block));
  ASSERT_TRUE(graph.Update(handle, [&](auto& inserted) {
    inserted.reads = {{1, 1, {ir::SsaValueKind::node, handle, 0}}};
    inserted.exits = {{31, {ir::SsaValueKind::phi, handle, 0}},
                      {40, {ir::SsaValueKind::node, handle, 0}}};
  }));
  graph.SetEntries({handle});
  auto budget = Plenty();
  EXPECT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  auto contract = Contract();
  contract.begin = 0;
  contract.end = 64;

  // The load's address is the literal the write put there, not SP plus 16.
  const auto result = ProvePrivateFrameSlots(graph, contract, budget);
  ASSERT_TRUE(result.facts) << static_cast<int>(result.reason);
  EXPECT_TRUE(result.facts->slots.empty());
}

TEST(PrivateFrame, RefusesMissingContractCallsAndUnknownContinuation) {
  auto graph = Graph();
  auto contract = Contract();
  contract.no_external_aliases = false;
  auto budget = Plenty();
  EXPECT_EQ(ProvePrivateFrameSlots(graph, contract, budget).reason,
            PrivateFrameDecline::missing_contract);

  graph.Update(graph.entries()[0], [](auto& block) {
    block.edges.push_back({ir::SsaEdgeKind::callee, ir::SsaTargetKind::unknown, 0, {}, {}, {}, {}});
  });
  auto budget2 = Plenty();
  EXPECT_EQ(ProvePrivateFrameSlots(graph, Contract(), budget2).reason,
            PrivateFrameDecline::unknown_call);
  auto declared = Contract();
  declared.callees_cannot_touch = true;
  declared.callees_preserve_sp = true;
  auto budget3 = Plenty();
  EXPECT_TRUE(ProvePrivateFrameSlots(graph, declared, budget3).facts);

  graph.Update(graph.entries()[0], [](auto& block) {
    block.edges.push_back(
        {ir::SsaEdgeKind::opaque_unknown, ir::SsaTargetKind::unknown, 0, {}, {}, {}, {}});
  });
  auto budget4 = Plenty();
  EXPECT_EQ(ProvePrivateFrameSlots(graph, declared, budget4).reason,
            PrivateFrameDecline::unknown_continuation);
}

TEST(PrivateFrame, AnEntryRelationIsNotAFrameAddress) {
  // X9 = SP + 16 at entry by a relation the graph does not replay: the scan
  // follows only values it derives from SP, so the accesses through X9 are
  // outside the region, which the contract keeps unreachable from outside.
  auto graph = Graph(false, false, true);
  auto budget = Plenty();
  const auto result = ProvePrivateFrameSlots(graph, Contract(), budget);
  ASSERT_TRUE(result.facts) << static_cast<int>(result.reason);
  EXPECT_TRUE(result.facts->slots.empty());
  const auto revision = graph.revision();
  EXPECT_FALSE(
      graph.Update(graph.entries()[0], [](auto& block) { block.relation_return_leaves = false; }));
  EXPECT_EQ(graph.revision(), revision);
}

TEST(PrivateFrame, RefusesCallTransferWithoutCalleeContract) {
  auto graph = Graph();
  ASSERT_TRUE(graph.Update(graph.entries()[0], [](auto& block) {
    block.boundaries[0].transfer = ir::Transfer{ir::TransferKind::call, 3, {}, {}, 3};
  }));
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  EXPECT_EQ(ProvePrivateFrameSlots(graph, Contract(), budget).reason,
            PrivateFrameDecline::unknown_call);
}

TEST(PrivateFrame, LeavesOutAMisalignedSlotEvenWithoutDeclineFlag) {
  auto graph = Graph();
  ASSERT_TRUE(graph.Update(graph.entries()[0], [](auto& block) {
    block.nodes[5].access.alignment = 16;
    block.nodes[5].access.decline_on_unaligned = false;
    block.nodes[5].inputs[0] = 2;
    block.nodes[1].immediate = 8;
    block.nodes[6].access.alignment = 16;
    block.nodes[6].access.decline_on_unaligned = false;
  }));
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  const auto result = ProvePrivateFrameSlots(graph, Contract(), budget);
  ASSERT_TRUE(result.facts) << static_cast<int>(result.reason);
  EXPECT_TRUE(result.facts->slots.empty());
}

enum class Shape { plain, escape, dynamic, overlap, unrestored };

// A prologue block moves SP down 32 and keeps SP+8 in X19; the next block
// stores and reloads through X19, restores SP and returns. Under the declared
// ABI X19 is preserved, so holding a frame address there is no escape.
ir::SsaGraph Prologue(Shape shape) {
  constexpr std::array<ir::StorageId, 4> kStorage{0, 19, 30, 31};
  ir::SsaGraph graph;
  const auto add = [&](std::vector<ir::Node> nodes, bool entry) {
    ir::SsaBlock block{};
    block.address = entry ? 0x100 : 0x110;
    block.source_groups = {block.address};
    block.original_sources = {0};
    for (const auto storage : kStorage) {
      block.phis.push_back({storage, 64, entry, {}});
      block.clobbers.push_back(0);
    }

    block.nodes = std::move(nodes);
    return graph.Add(std::move(block));
  };

  const auto a = add({{ir::Op::read, 64, {}, 0, 31},
                      {ir::Op::constant, 64, {}, 32},
                      {ir::Op::sub, 64, {0, 1}},
                      {ir::Op::constant, 64, {}, 8},
                      {ir::Op::add, 64, {2, 3}}},
                     true);
  std::vector<ir::Node> body{
      {ir::Op::read, 64, {}, 0, 19}, {ir::Op::constant, 64, {}, 42}, {ir::Op::store, 64, {0, 1}},
      {ir::Op::load, 64, {0}},       {ir::Op::read, 64, {}, 0, 31},  {ir::Op::constant, 64, {}, 32},
      {ir::Op::add, 64, {4, 5}},     {ir::Op::read, 64, {}, 0, 30},  {ir::Op::read, 64, {}, 0, 0}};
  for (const auto id : {2U, 3U}) body[id].access.alignment = 8;
  if (shape == Shape::dynamic) {
    body.push_back({ir::Op::add, 64, {0, 8}});     // 9: X19 + X0
    body.push_back({ir::Op::extract, 8, {1}, 0});  // 10
    body.push_back({ir::Op::store, 8, {9, 10}});   // 11
  } else if (shape == Shape::overlap) {
    body.push_back({ir::Op::constant, 64, {}, 4});  // 9
    body.push_back({ir::Op::add, 64, {0, 9}});      // 10: X19 + 4
    body.push_back({ir::Op::extract, 32, {1}, 0});  // 11
    body.push_back({ir::Op::store, 32, {10, 11}});  // 12
    body.back().access.alignment = 4;
  }

  const auto b = add(std::move(body), false);
  graph.Update(a, [&](ir::SsaBlock& block) {
    block.boundaries = {
        {0, 5, {{31, 2}, {19, 4}}, ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}}}};
    block.boundaries[0].transfer.reset();
    block.reads = {{0, 3, {ir::SsaValueKind::phi, a, 3}}};
    block.exits = {{0, {ir::SsaValueKind::phi, a, 0}},
                   {19, {ir::SsaValueKind::node, a, 4}},
                   {30, {ir::SsaValueKind::phi, a, 2}},
                   {31, {ir::SsaValueKind::node, a, 2}}};
    block.edges = {
        {ir::SsaEdgeKind::fallthrough, ir::SsaTargetKind::image_location, 0x110, b, {}, {}, {}}};
  });
  graph.Update(b, [&](ir::SsaBlock& block) {
    // Returning with SP still 32 below entry hands the region to the caller.
    std::vector<ir::Write> writes{{31, shape == Shape::unrestored ? 4U : 6U}};

    // Returning a frame address in X0 lets the caller reach the frame.
    if (shape == Shape::escape) writes.push_back({0, 0});
    block.boundaries = {{0, static_cast<std::uint32_t>(block.nodes.size()), writes,
                         ir::Transfer{ir::TransferKind::return_, 7, {}, {}, {}}}};
    block.reads = {{0, 1, {ir::SsaValueKind::phi, b, 1}},
                   {4, 3, {ir::SsaValueKind::phi, b, 3}},
                   {7, 2, {ir::SsaValueKind::phi, b, 2}},
                   {8, 0, {ir::SsaValueKind::phi, b, 0}}};
    block.exits = {{0, shape == Shape::escape ? ir::SsaValue{ir::SsaValueKind::node, b, 0}
                                              : ir::SsaValue{ir::SsaValueKind::phi, b, 0}},
                   {19, {ir::SsaValueKind::phi, b, 1}},
                   {30, {ir::SsaValueKind::phi, b, 2}},
                   {31, {ir::SsaValueKind::node, b, shape == Shape::unrestored ? 4U : 6U}}};
    for (std::uint32_t phi = 0; phi < 4; ++phi) {
      const auto& exit = graph.Get(a)->exits[phi];
      block.phis[phi].incoming = {{a, exit.value}};
    }

    block.edges = {{ir::SsaEdgeKind::return_,
                    ir::SsaTargetKind::unknown,
                    0,
                    {},
                    {},
                    {},
                    {.return_leaves = true}}};
  });
  graph.SetEntries({a});
  graph.SetObservability(
      {true,
       false,
       {0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
       {0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31},
       {19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31}});
  return graph;
}

TEST(PrivateFrame, FollowsAMovedStackPointerAndAFrameBaseAcrossBlocks) {
  auto graph = Prologue(Shape::plain);
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  const auto result = ProvePrivateFrameSlots(graph, Contract(), budget);
  ASSERT_TRUE(result.facts) << static_cast<int>(result.reason);
  ASSERT_EQ(result.facts->slots.size(), 1U);
  EXPECT_EQ(result.facts->slots[0].offset, -24);
  EXPECT_EQ(result.facts->slots[0].size, 8U);
  EXPECT_EQ(result.facts->slots[0].accesses.size(), 2U);
  EXPECT_EQ(ValidatePrivateFrameFacts(*result.facts, graph, budget),
            ir::PrivateFrameFactDecline::none);
  // A claimed slot the scan does not prove is refused.
  auto forged = *result.facts;
  forged.slots[0].offset = -16;
  for (auto& access : forged.slots[0].accesses) access.offset = -16;
  EXPECT_EQ(ValidatePrivateFrameFacts(forged, graph, budget), ir::PrivateFrameFactDecline::invalid);
}

TEST(PrivateFrame, RefusesAFrameAddressTheCallerCanSee) {
  auto graph = Prologue(Shape::escape);
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  EXPECT_EQ(ProvePrivateFrameSlots(graph, Contract(), budget).reason,
            PrivateFrameDecline::escaped_address);
}

TEST(PrivateFrame, RefusesAReturnThatLeavesTheRegionAsStack) {
  auto graph = Prologue(Shape::unrestored);
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  EXPECT_EQ(ProvePrivateFrameSlots(graph, Contract(), budget).reason,
            PrivateFrameDecline::escaped_address);
}

TEST(PrivateFrame, AnAccessAtAnUnknownFrameOffsetProvesNoSlot) {
  auto graph = Prologue(Shape::dynamic);
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  EXPECT_EQ(ProvePrivateFrameSlots(graph, Contract(), budget).reason,
            PrivateFrameDecline::unknown_offset);
}

TEST(PrivateFrame, LeavesOutSlotsThatOverlapAnotherShape) {
  auto graph = Prologue(Shape::overlap);
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  const auto result = ProvePrivateFrameSlots(graph, Contract(), budget);
  ASSERT_TRUE(result.facts) << static_cast<int>(result.reason);
  EXPECT_TRUE(result.facts->slots.empty());
}
}  // namespace
}  // namespace nyx::analysis
