#include <gtest/gtest.h>

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::ir {
namespace {
Budget Plenty() { return Budget({1000000, 10000000}); }

constexpr SsaHandle kFirst{1, 1, 0};
constexpr SsaHandle kSecond{1, 2, 0};

std::vector<std::uint8_t> Word() { return {0, 0, 0, 0}; }

bool Complete(const SsaBlock& block) {
  auto budget = Plenty();
  return ValidateSsaDirectSuccessors(block, budget) == SsaDecline::none;
}

SsaEdge Branch(std::uint64_t address, SsaHandle target, std::optional<ValueId> condition = {},
               std::optional<bool> when = {}) {
  SsaEdge edge{
      SsaEdgeKind::branch, SsaTargetKind::image_location, address, target, condition, when, {}};
  edge.assumptions.constant_image = true;
  return edge;
}

// A recovered path through three instructions: it computes a dispatch state
// of 5 or 7 from x0, jumps into the dispatcher, passes its bound check (state
// above 9 would leave), and the dispatch rewrite sends it to 0x500 or 0x600.
SsaBlock Transition() {
  SsaBlock block{};
  block.address = 0x100;
  block.transition = 0;
  block.source_groups = {0x100, 0x200, 0x204};
  block.source_bytes = {Word(), Word(), Word()};
  block.original_sources = {0, 1, 2};
  block.nodes = {
      {Op::read, 64, {}, 0, 0},
      {Op::constant, 64, {}, 0},
      {Op::equal, 1, {0, 1}},
      {Op::constant, 32, {}, 5},
      {Op::constant, 32, {}, 7},
      {Op::select, 32, {2, 3, 4}},
      {Op::image_address, 64, {}, 0x200},
      {Op::constant, 32, {}, 9},
      {Op::unsigned_less, 1, {7, 5}},
      {Op::image_address, 64, {}, 0x300},
      {Op::image_address, 64, {}, 0x204},
      {Op::read, 64, {}, 0, 1},
  };

  block.boundaries = {
      {0, 7, {}, Transfer{TransferKind::jump, 6, {}, {}, {}}},
      {7, 4, {}, Transfer{TransferKind::conditional, 9, 8, 10, {}}},
      {11, 1, {}, Transfer{TransferKind::jump, 11, {}, {}, {}}},
  };

  block.destination_nodes = {{Op::image_address, 64, {}, 0x500},
                             {Op::image_address, 64, {}, 0x600}};
  ConditionalRewrite rewrite{};
  rewrite.rule = RewriteRule::dispatch_branch;
  rewrite.boundary = 2;
  rewrite.original = *block.boundaries[2].transfer;
  rewrite.replacement = Transfer{TransferKind::conditional, 12, 2, 13, {}};
  rewrite.condition = 2;
  rewrite.when_true = 0x500;
  rewrite.when_false = 0x600;
  block.control_rewrites = {rewrite};
  block.edges = {Branch(0x500, kFirst, 2, true), Branch(0x600, kSecond, 2, false)};
  return block;
}

// The two refolds that prove a dispatcher's successors write a fixed bit over
// the condition and over everything SsaImpliedBits reaches back through, so a
// literal in any of those changes neither. What is computed from them does
// differ, and a pass that writes a value there leaves the two disagreeing.
SsaBlock DependencyBlock() {
  SsaBlock block{};
  block.nodes = {
      {Op::read, 64, {}, 0, 0},     // 0
      {Op::constant, 64, {}, 0},    // 1
      {Op::equal, 1, {0, 1}},       // 2  fixed: the chain below the condition
      {Op::bit_not, 1, {2}},        // 3  fixed: the condition itself
      {Op::constant, 32, {}, 5},    // 4
      {Op::constant, 32, {}, 7},    // 5
      {Op::select, 32, {2, 4, 5}},  // 6  from a fixed node
      {Op::select, 32, {3, 4, 5}},  // 7  from the condition
      {Op::add, 32, {6, 4}},        // 8  from 6
      {Op::constant, 32, {}, 1},    // 9  independent
      {Op::read, 32, {}, 0, 1},     // 10 reads storage 6 was written to
  };

  ConditionalRewrite rewrite{};
  rewrite.rule = RewriteRule::dispatch_branch;
  rewrite.condition = 3;
  block.control_rewrites = {rewrite};
  block.reads = {{10, 0, {SsaValueKind::node, kFirst, 6}}};
  return block;
}

TEST(SsaSuccessors, DispatchDependenceSkipsWhatARefoldFixes) {
  auto budget = Plenty();
  const auto block = DependencyBlock();
  const auto dependent = SsaDispatchDependent(block, kFirst, budget);
  ASSERT_TRUE(dependent);
  ASSERT_EQ(dependent->size(), block.nodes.size());
  const std::vector<std::uint8_t> expected{0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1};
  EXPECT_EQ(*dependent, expected);
}

TEST(SsaSuccessors, DispatchDependenceIsEmptyWithoutADispatch) {
  auto budget = Plenty();
  auto block = DependencyBlock();
  block.control_rewrites.front().rule = RewriteRule::folded_condition;
  const auto dependent = SsaDispatchDependent(block, kFirst, budget);
  ASSERT_TRUE(dependent);
  EXPECT_TRUE(dependent->empty());
}

// The same transition once its dispatch condition has become a literal: it
// goes to one destination and keeps the other as the provenance of that.
SsaBlock DecidedTransition() {
  auto block = Transition();
  block.nodes[2] = {Op::constant, 1, {}, 0};
  auto& rewrite = block.control_rewrites.front();
  rewrite.rule = RewriteRule::decided_dispatch;
  rewrite.condition_value = false;
  rewrite.replacement = Transfer{TransferKind::jump, 13, {}, {}, {}};
  block.edges = {Branch(0x600, kSecond)};
  return block;
}

TEST(SsaSuccessors, ADecidedDispatchHasTheOneSuccessorItsConditionPicks) {
  EXPECT_TRUE(Complete(DecidedTransition()));
}

TEST(SsaSuccessors, ADecidedDispatchRefusesEveryOtherShape) {
  auto both = DecidedTransition();
  both.edges = {Branch(0x500, kFirst, 2, true), Branch(0x600, kSecond, 2, false)};
  EXPECT_FALSE(Complete(both));
  auto unknown = DecidedTransition();
  unknown.nodes[2] = {Op::equal, 1, {0, 1}};
  EXPECT_FALSE(Complete(unknown));
  auto disagrees = DecidedTransition();
  disagrees.control_rewrites.front().condition_value = true;
  EXPECT_FALSE(Complete(disagrees));
  auto wrong_arm = DecidedTransition();
  wrong_arm.control_rewrites.front().replacement.target = 12;
  EXPECT_FALSE(Complete(wrong_arm));
  auto astray = DecidedTransition();
  astray.edges = {Branch(0x500, kFirst)};
  EXPECT_FALSE(Complete(astray));
  auto undeclared = DecidedTransition();
  undeclared.edges[0].assumptions.constant_image = false;
  EXPECT_FALSE(Complete(undeclared));
  auto conditional = DecidedTransition();
  conditional.edges[0].condition = 2;
  conditional.edges[0].when = false;
  EXPECT_FALSE(Complete(conditional));

  // The arm it did not take stays: without it the record no longer says
  // what the choice was between.
  auto forgotten = DecidedTransition();
  forgotten.destination_nodes = {{Op::image_address, 64, {}, 0x600}};
  forgotten.control_rewrites.front().replacement.target = 12;
  EXPECT_FALSE(Complete(forgotten));
}

TEST(SsaSuccessors, TransitionIsCompleteUnderEachDispatchCase) {
  EXPECT_TRUE(Complete(Transition()));
}

TEST(SsaSuccessors, TransitionRefusesAnyUnprovedInternalOrFinalTransfer) {
  auto bound = Transition();

  // 6 < 7: the bound check leaves the path when the dispatch picks state 7.
  bound.nodes[7].immediate = 6;
  EXPECT_FALSE(Complete(bound));
  auto swapped = Transition();
  std::swap(swapped.edges[0].address, swapped.edges[1].address);
  EXPECT_FALSE(Complete(swapped));
  auto undeclared = Transition();
  undeclared.edges[1].assumptions.constant_image = false;
  EXPECT_FALSE(Complete(undeclared));
  auto astray = Transition();
  astray.nodes[6].immediate = 0x208;
  EXPECT_FALSE(Complete(astray));
  auto unknown = Transition();
  unknown.nodes[3] = {Op::read, 32, {}, 0, 2};
  EXPECT_FALSE(Complete(unknown));
  auto destination = Transition();
  destination.destination_nodes[1].immediate = 0x700;
  EXPECT_FALSE(Complete(destination));
  auto extra = Transition();
  extra.edges.push_back(Branch(0x700, kFirst));
  EXPECT_FALSE(Complete(extra));
  auto misplaced = Transition();
  misplaced.control_rewrites[0].boundary = 1;
  EXPECT_FALSE(Complete(misplaced));
  auto gap = Transition();
  gap.source_groups[2] = 0x208;
  EXPECT_FALSE(Complete(gap));
}

TEST(SsaSuccessors, TransitionWithoutDispatchNeedsOneLiteralFinalTarget) {
  auto single = Transition();
  single.control_rewrites.clear();
  single.destination_nodes.clear();

  // The bound check now depends on x0 through the select, so fix the state.
  single.nodes[5] = {Op::constant, 32, {}, 5};
  single.nodes[11] = {Op::image_address, 64, {}, 0x500};
  single.edges = {Branch(0x500, kFirst)};
  EXPECT_TRUE(Complete(single));
  single.edges[0].address = 0x504;
  EXPECT_FALSE(Complete(single));
  single.edges[0].address = 0x500;
  single.nodes[11] = {Op::read, 64, {}, 0, 1};
  EXPECT_FALSE(Complete(single));
}

// The dispatcher jump reads its target from a relocated slot at 0x800.
SsaBlock SlotJump(bool declared, bool placed_address) {
  auto block = Transition();
  block.control_rewrites.clear();
  block.destination_nodes.clear();
  block.nodes[5] = {Op::constant, 32, {}, 5};
  if (placed_address) {
    // ADRP-style: a PC in the page, masked to it, plus the page offset.
    block.nodes[11] = {Op::image_address, 64, {}, 0x8f0};
    block.nodes.push_back({Op::constant, 64, {}, ~std::uint64_t{0xfff}});
    block.nodes.push_back({Op::bit_and, 64, {11, 12}});
    block.nodes.push_back({Op::constant, 64, {}, 0x800});
    block.nodes.push_back({Op::add, 64, {13, 14}});
    block.nodes.push_back({Op::load, 64, {15}});
  } else {
    block.nodes[11] = {Op::image_address, 64, {}, 0x800};
    block.nodes.push_back({Op::load, 64, {11}});
  }

  const auto load = static_cast<ValueId>(block.nodes.size() - 1);
  block.boundaries[2] = {11,
                         static_cast<std::uint32_t>(load - 10),
                         {},
                         Transfer{TransferKind::jump, load, {}, {}, {}}};
  if (declared) block.path_reads = {{load, 0x800, true, 0x500, placed_address}};
  block.edges = {Branch(0x500, kFirst)};
  return block;
}

TEST(SsaSuccessors, DeclaredPathReadsResolveLoadedTargetsUnlessContradicted) {
  EXPECT_TRUE(Complete(SlotJump(true, false)));
  EXPECT_FALSE(Complete(SlotJump(false, false)));
  EXPECT_TRUE(Complete(SlotJump(true, true)));
  auto unplaced = SlotJump(true, true);
  unplaced.path_reads[0].page_aligned_placement = false;
  EXPECT_FALSE(Complete(unplaced));
  auto elsewhere = SlotJump(true, false);
  elsewhere.path_reads[0].address = 0x808;
  EXPECT_FALSE(Complete(elsewhere));

  // A write the path provably makes to the slot contradicts the declaration.
  auto written = SlotJump(true, false);
  const auto load = written.path_reads[0].node;
  written.nodes.insert(written.nodes.begin() + load, {Op::store, 64, {11, 0}});
  written.nodes[load + 1].inputs[0] = 11;
  written.path_reads[0].node = load + 1;
  written.boundaries[2].node_count += 1;
  written.boundaries[2].transfer->target = load + 1;
  EXPECT_FALSE(Complete(written));

  // So does a wider write that covers it, from below or through a wide value.
  auto wide = written;
  wide.nodes[load] = {Op::store, 128, {11, 0}};
  EXPECT_FALSE(Complete(wide));
  auto below = written;
  below.nodes[11].immediate = 0x7fc;
  below.nodes[load + 1] = {Op::load, 64, {11}};
  below.nodes.insert(below.nodes.begin() + load + 1, {Op::image_address, 64, {}, 0x800});
  below.nodes[load + 2].inputs[0] = load + 1;
  below.path_reads[0].node = load + 2;
  below.boundaries[2].node_count += 1;
  below.boundaries[2].transfer->target = load + 2;
  EXPECT_FALSE(Complete(below));  // an 8-byte write at 0x7fc reaches 0x800..0x803
  below.nodes[11].immediate = 0x7f8;
  EXPECT_TRUE(Complete(below));  // one ending at 0x800 does not
  // A slot read decides the jump, so the edge must carry that declaration.
  auto bare = SlotJump(true, false);
  bare.edges[0].assumptions.constant_image = false;
  EXPECT_FALSE(Complete(bare));
}

SsaBlock Call(bool indirect) {
  SsaBlock block{};
  block.address = 0x100;
  block.source_groups = {0x100};
  block.source_bytes = {Word()};
  block.original_sources = {0};
  block.nodes = {indirect ? Node{Op::read, 64, {}, 0, 8} : Node{Op::image_address, 64, {}, 0x1000},
                 {Op::image_address, 64, {}, 0x104}};
  block.boundaries = {{0, 2, {}, Transfer{TransferKind::call, 0, {}, {}, 1}}};
  SsaEdge callee{SsaEdgeKind::callee,
                 indirect ? SsaTargetKind::unknown : SsaTargetKind::image_location,
                 indirect ? 0U : 0x1000U,
                 {},
                 {},
                 {},
                 {}};
  SsaEdge back{
      SsaEdgeKind::potential_return, SsaTargetKind::image_location, 0x104, kFirst, {}, {}, {}};
  back.assumptions.callee_returns_to_continuation = true;
  block.edges = {callee, back};
  return block;
}

TEST(SsaSuccessors, ExternalCallsReachOnlyTheirDeclaredContinuation) {
  EXPECT_TRUE(Complete(Call(false)));
  EXPECT_TRUE(Complete(Call(true)));
  auto undeclared = Call(false);
  undeclared.edges[1].assumptions.callee_returns_to_continuation = false;
  EXPECT_FALSE(Complete(undeclared));
  auto internal = Call(false);
  internal.edges[0].target_block = kSecond;
  EXPECT_FALSE(Complete(internal));
  auto elsewhere = Call(false);
  elsewhere.edges[0].address = 0x2000;
  EXPECT_FALSE(Complete(elsewhere));

  // An unknown callee covers a known target too: under closed entries any
  // callee re-enters only at a listed entry.
  auto hidden = Call(false);
  hidden.edges[0].target_kind = SsaTargetKind::unknown;
  EXPECT_TRUE(Complete(hidden));
  auto claimed = Call(true);
  claimed.edges[0].target_kind = SsaTargetKind::image_location;
  EXPECT_FALSE(Complete(claimed));
  auto late = Call(false);
  late.edges[1].address = 0x108;
  EXPECT_FALSE(Complete(late));
  auto noreturn = Call(false);
  noreturn.edges.pop_back();
  EXPECT_FALSE(Complete(noreturn));
  noreturn.edges[0].assumptions.declared_noreturn = true;
  EXPECT_TRUE(Complete(noreturn));
  auto both = Call(false);
  both.edges[0].assumptions.declared_noreturn = true;
  EXPECT_FALSE(Complete(both));
  auto got = Call(false);
  got.nodes[0] = {Op::image_address, 64, {}, 0x800};
  got.nodes.insert(got.nodes.begin() + 1, {Op::load, 64, {0}});
  got.nodes[2] = {Op::image_address, 64, {}, 0x104};
  got.boundaries = {{0, 3, {}, Transfer{TransferKind::call, 1, {}, {}, 2}}};
  EXPECT_FALSE(Complete(got));
  got.path_reads = {{1, 0x800, true, 0x1000, false}};
  EXPECT_FALSE(Complete(got));
  got.edges[0].assumptions.constant_image = true;
  EXPECT_TRUE(Complete(got));
  auto unbound = Call(false);
  unbound.source_bytes.clear();
  EXPECT_FALSE(Complete(unbound));
}

TEST(SsaSuccessors, ADeclaredFallthroughCompletesAnOpaqueBlock) {
  // Not knowing what an instruction computes is not knowing where it goes.
  // A block left incomplete refuses the reachability proof for the whole
  // graph, so a division the decoder cannot lift would cost nine passes.
  SsaBlock block{};
  block.address = 0x100;
  block.opaque = true;
  block.source_groups = {0x100};
  block.source_bytes = {Word()};
  block.original_sources = {0};
  SsaEdge next{
      SsaEdgeKind::fallthrough, SsaTargetKind::image_location, 0x104, SsaHandle{1, 1}, {}, {}, {}};
  next.assumptions.declared_opaque_control = true;
  block.edges = {next};
  EXPECT_TRUE(Complete(block));

  // The successor is the instruction after this one and no other.
  block.edges[0].address = 0x108;
  EXPECT_FALSE(Complete(block));
  block.edges[0].address = 0x104;
  block.edges[0].target_block = std::nullopt;
  EXPECT_FALSE(Complete(block));
  block.edges[0].target_block = SsaHandle{1, 1};
  block.edges[0].assumptions.declared_opaque_control = false;
  EXPECT_FALSE(Complete(block));
}

TEST(SsaSuccessors, OnlyADeclaredTrapOrFallthroughCompletesAnOpaqueBlock) {
  SsaBlock block{};
  block.address = 0x100;
  block.opaque = true;
  block.source_groups = {0x100};
  block.source_bytes = {Word()};
  block.original_sources = {0};
  SsaEdge trap{SsaEdgeKind::trap, SsaTargetKind::unknown, 0, {}, {}, {}, {}};
  trap.assumptions.declared_opaque_control = true;
  block.edges = {trap};
  EXPECT_TRUE(Complete(block));
  block.edges[0].assumptions.declared_opaque_control = false;
  EXPECT_FALSE(Complete(block));
  block.edges[0].assumptions.declared_opaque_control = true;
  block.edges[0].assumptions.unresolved_target = true;
  EXPECT_FALSE(Complete(block));
  block.edges[0].assumptions.unresolved_target = false;
  block.edges[0].when = true;
  EXPECT_FALSE(Complete(block));
  block.edges[0] = {SsaEdgeKind::opaque_unknown, SsaTargetKind::unknown, 0, {}, {}, {}, {}};
  block.edges[0].assumptions.declared_opaque_control = true;
  EXPECT_FALSE(Complete(block));
}

TEST(SsaSuccessors, OnlyATransitionBindsAComputedSourceTarget) {
  std::vector<Group> sources;
  sources.emplace_back(0x100, Word(), std::vector<Node>{{Op::read, 64, {}, 0, 1}},
                       std::vector<Write>{}, MemoryModel::unspecified,
                       Transfer{TransferKind::jump, 0, {}, {}, {}});
  SsaBlock block{};
  block.address = 0x100;
  block.source_groups = {0x100};
  block.source_bytes = {Word()};
  block.original_sources = {0};
  block.nodes = {{Op::image_address, 64, {}, 0x500}};
  block.boundaries = {{0, 1, {}, Transfer{TransferKind::jump, 0, {}, {}, {}}}};
  auto budget = Plenty();
  EXPECT_EQ(ValidateSsaSourceBinding(block, sources, budget), SsaDecline::invalid_graph);
  block.transition = 0;
  EXPECT_EQ(ValidateSsaSourceBinding(block, sources, budget), SsaDecline::none);

  // A literal source target still binds exactly.
  sources[0] = Group(0x100, Word(), std::vector<Node>{{Op::image_address, 64, {}, 0x400}},
                     std::vector<Write>{}, MemoryModel::unspecified,
                     Transfer{TransferKind::jump, 0, {}, {}, {}});
  EXPECT_EQ(ValidateSsaSourceBinding(block, sources, budget), SsaDecline::invalid_graph);

  // A call binds its literal callee and its continuation.
  sources[0] = Group(
      0x100, Word(),
      std::vector<Node>{{Op::image_address, 64, {}, 0x1000}, {Op::image_address, 64, {}, 0x104}},
      std::vector<Write>{}, MemoryModel::unspecified, Transfer{TransferKind::call, 0, {}, {}, 1});
  auto call = Call(false);
  EXPECT_EQ(ValidateSsaSourceBinding(call, sources, budget), SsaDecline::none);
  call.nodes[0].immediate = 0x2000;
  EXPECT_EQ(ValidateSsaSourceBinding(call, sources, budget), SsaDecline::invalid_graph);
  call = Call(false);
  call.nodes[1].immediate = 0x108;
  EXPECT_EQ(ValidateSsaSourceBinding(call, sources, budget), SsaDecline::invalid_graph);
}

// Outside a transition a `br x1` may bind a target that only folds to an image
// location, but completeness takes that fold only from a declared path read:
// anything else could name a mapped block the source never jumps to.
TEST(SsaSuccessors, AFoldedJumpTargetNeedsADeclaredRead) {
  std::vector<Group> sources;
  sources.emplace_back(0x100, Word(), std::vector<Node>{{Op::read, 64, {}, 0, 1}},
                       std::vector<Write>{}, MemoryModel::unspecified,
                       Transfer{TransferKind::jump, 0, {}, {}, {}});
  SsaBlock block{};
  block.address = 0x100;
  block.source_groups = {0x100};
  block.source_bytes = {Word()};
  block.original_sources = {0};

  // select(true, 0x500, x1).
  block.nodes = {{Op::constant, 1, {}, 1},
                 {Op::image_address, 64, {}, 0x500},
                 {Op::read, 64, {}, 0, 1},
                 {Op::select, 64, {0, 1, 2}}};
  block.boundaries = {{0, 4, {}, Transfer{TransferKind::jump, 3, {}, {}, {}}}};
  block.edges = {{SsaEdgeKind::branch, SsaTargetKind::image_location, 0x500, kFirst, {}, {}, {}}};
  auto budget = Plenty();
  EXPECT_FALSE(Complete(block));
  EXPECT_NE(ValidateSsaClosedMember(block, sources, budget), SsaDecline::none);
  block.edges[0].assumptions.constant_image = true;
  EXPECT_FALSE(Complete(block));

  // add(0x500, xor(5, 5)).
  block.nodes = {{Op::image_address, 64, {}, 0x500},
                 {Op::constant, 64, {}, 5},
                 {Op::bit_xor, 64, {1, 1}},
                 {Op::add, 64, {0, 2}}};
  block.edges[0].assumptions.constant_image = false;
  EXPECT_FALSE(Complete(block));
  EXPECT_NE(ValidateSsaClosedMember(block, sources, budget), SsaDecline::none);
}

}  // namespace
}  // namespace nyx::ir
