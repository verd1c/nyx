#include <algorithm>
#include <array>
#include <random>

#include <gtest/gtest.h>

#include "nyx/ir/fold.hpp"
#include "nyx/ir/ssa/graph.hpp"
#include "nyx/recovery/ssa/dead_loads.hpp"
#include "nyx/recovery/ssa/simplify.hpp"

namespace nyx::recovery {
namespace {
using ir::Node;
using ir::Op;
using ir::ValueId;

Budget Plenty() { return Budget({100000000, 1000000000}); }

constexpr std::array<ir::StorageId, 5> kStorage{0, 1, 2, 3, 30};
constexpr unsigned kInputs = 3;
constexpr ValueId kLink = kInputs;  // the read of the return address
constexpr std::array<unsigned, 5> kWidths{1, 8, 16, 32, 64};

// One returning block: three 64-bit inputs, a pure DAG, and its last value
// written to storage 3.
ir::SsaGraph Build(std::vector<Node> nodes, ValueId output) {
  ir::SsaGraph graph;
  ir::SsaBlock block{};
  block.address = 0x100;
  block.source_groups = {0x100};
  block.source_bytes = {{0, 0, 0, 0}};
  block.original_sources = {0};
  for (const auto storage : kStorage) {
    block.phis.push_back({storage, 64, true, {}});
    block.clobbers.push_back(0);
  }

  block.nodes = std::move(nodes);
  block.boundaries = {{0,
                       static_cast<std::uint32_t>(block.nodes.size()),
                       {{3, output}},
                       ir::Transfer{ir::TransferKind::return_, kLink, {}, {}, {}}}};
  block.edges = {{ir::SsaEdgeKind::return_,
                  ir::SsaTargetKind::unknown,
                  0,
                  {},
                  {},
                  {},
                  {.return_leaves = true}}};
  const auto handle = graph.Add(std::move(block));
  graph.Update(handle, [&](ir::SsaBlock& changed) {
    for (unsigned i = 0; i <= kInputs; ++i)
      changed.reads.push_back(
          {i, i == kLink ? 4U : i, {ir::SsaValueKind::phi, handle, i == kLink ? 4U : i}});
    for (std::uint32_t i = 0; i < kStorage.size(); ++i)
      changed.exits.push_back(
          {kStorage[i], i == 3 ? ir::SsaValue{ir::SsaValueKind::node, handle, output}
                               : ir::SsaValue{ir::SsaValueKind::phi, handle, i}});
  });
  graph.SetEntries({handle});
  return graph;
}

// One block holding a selected image load whose address reads the fold's
// condition, with that condition computable: 1 - 0 is not 0, so the choice
// collapses and the address stops depending on it.
ir::SsaGraph SelectedFold() {
  ir::SsaGraph graph;
  ir::SsaBlock block{};
  block.address = 0x100;
  block.source_groups = {0x100};
  block.source_bytes = {{0, 0, 0, 0}};
  block.original_sources = {0};
  block.phis = {{0, 64, true, {}}, {1, 64, true, {}}};
  block.clobbers = {0, 0};
  block.nodes = {
      {Op::constant, 32, {}, 1},    {Op::constant, 32, {}, 0},
      {Op::sub, 32, {0, 1}},        {Op::constant, 32, {}, 0},
      {Op::equal, 1, {2, 3}},       {Op::constant, 32, {}, 0x15},
      {Op::constant, 32, {}, 0x16}, {Op::select, 32, {4, 5, 6}},
      {Op::zext, 64, {7}},          {Op::constant, 64, {}, 2},
      {Op::mul, 64, {8, 9}},        {Op::image_address, 64, {}, 0x2000},
      {Op::add, 64, {11, 10}},      {Op::load, 16, {12}},
      {Op::zext, 64, {13}},         {Op::constant, 64, {}, 0},
      {Op::add, 64, {14, 15}},
  };

  block.disabled_effects = {13};
  block.boundaries = {{0, static_cast<std::uint32_t>(block.nodes.size()), {{1, 16}}, {}}};
  block.edges = {
      {ir::SsaEdgeKind::fallthrough, ir::SsaTargetKind::image_location, 0x104, {}, {}, {}, {}}};
  ir::SsaConstantLoad fold{};
  fold.node = 13;
  fold.kind = ir::SsaConstantKind::literal;
  fold.value = 0xbbaa;
  fold.source_address = 0x202a;
  fold.declared_bytes = {0xaa, 0xbb};
  fold.access = {true, true, true};
  fold.value_stable = true;
  fold.skip_access = true;
  fold.condition = 4;
  fold.alternative_value = 0xddcc;
  fold.alternative_source_address = 0x202c;
  fold.alternative_declared_bytes = {0xcc, 0xdd};
  block.constant_loads = {fold};
  const auto handle = graph.Add(std::move(block));
  graph.Update(handle, [&](ir::SsaBlock& changed) {
    changed.exits.push_back({0, {ir::SsaValueKind::phi, handle, 0}});
    changed.exits.push_back({1, {ir::SsaValueKind::node, handle, 16}});
  });
  graph.SetEntries({handle});
  return graph;
}

// A selected fold whose address never names its condition. The address reads
// `a & 0`; what fixes that value under each pinning is the implied-bit chain
// `SsaImpliedBits` walks down from the condition -- negation, same-width copy,
// then the masked value. The copy in the middle is in neither the address cone
// nor the condition's own mark, and `known_zero` proves it constant, which
// breaks the chain below it and leaves the address underivable.
ir::SsaGraph ImpliedChainFold() {
  ir::SsaGraph graph;
  ir::SsaBlock block{};
  block.address = 0x100;
  block.source_groups = {0x100};
  block.source_bytes = {{0, 0, 0, 0}};
  block.original_sources = {0};
  block.phis = {{0, 64, true, {}}, {1, 64, true, {}}};
  block.clobbers = {0, 0};
  block.nodes = {
      {Op::read, 64, {}, 0, 0},             // 0  opaque to the re-derivation
      {Op::extract, 1, {0}, 0},             // 1  a
      {Op::constant, 1, {}, 0},             // 2
      {Op::bit_and, 1, {1, 2}},             // 3  what the address reads
      {Op::zext, 1, {3}},                   // 4  the chain link
      {Op::bit_not, 1, {4}},                // 5  the condition
      {Op::zext, 64, {3}},                  // 6
      {Op::constant, 64, {}, 2},            // 7
      {Op::mul, 64, {6, 7}},                // 8
      {Op::image_address, 64, {}, 0x2000},  // 9
      {Op::add, 64, {9, 8}},                // 10
      {Op::load, 16, {10}},                 // 11
      {Op::zext, 64, {11}},                 // 12
      {Op::constant, 64, {}, 0},            // 13
      {Op::add, 64, {12, 13}},              // 14
  };

  block.disabled_effects = {11};
  block.boundaries = {{0, static_cast<std::uint32_t>(block.nodes.size()), {{1, 14}}, {}}};
  block.edges = {
      {ir::SsaEdgeKind::fallthrough, ir::SsaTargetKind::image_location, 0x104, {}, {}, {}, {}}};
  ir::SsaConstantLoad fold{};
  fold.node = 11;
  fold.kind = ir::SsaConstantKind::literal;
  fold.value = 0xbbaa;
  fold.source_address = 0x2000;
  fold.declared_bytes = {0xaa, 0xbb};
  fold.access = {true, true, true};
  fold.value_stable = true;
  fold.skip_access = true;

  // The condition negates, so the true pinning is the lower location.
  fold.condition = 5;
  fold.alternative_value = 0xddcc;
  fold.alternative_source_address = 0x2002;
  fold.alternative_declared_bytes = {0xcc, 0xdd};
  block.constant_loads = {fold};
  const auto handle = graph.Add(std::move(block));
  graph.Update(handle, [&](ir::SsaBlock& changed) {
    changed.reads.push_back({0, 0, {ir::SsaValueKind::phi, handle, 0}});
    changed.exits.push_back({0, {ir::SsaValueKind::phi, handle, 0}});
    changed.exits.push_back({1, {ir::SsaValueKind::node, handle, 14}});
  });
  graph.SetEntries({handle});
  return graph;
}

// The same fold with a literal offset between the location and the index, run
// at a declared placement. A location is then a number, so the offset folds
// into it and the address arrives as a number the recheck will not accept.
ir::SsaGraph OffsetFoldAtDeclaredBias() {
  ir::SsaGraph graph;
  ir::SsaBlock block{};
  block.address = 0x100;
  block.source_groups = {0x100};
  block.source_bytes = {{0, 0, 0, 0}};
  block.original_sources = {0};
  block.phis = {{0, 64, true, {}}, {1, 64, true, {}}};
  block.clobbers = {0, 0};
  block.nodes = {
      {Op::constant, 32, {}, 1},
      {Op::constant, 32, {}, 0},
      {Op::sub, 32, {0, 1}},
      {Op::constant, 32, {}, 0},
      {Op::equal, 1, {2, 3}},
      {Op::constant, 32, {}, 0x15},
      {Op::constant, 32, {}, 0x16},
      {Op::select, 32, {4, 5, 6}},
      {Op::zext, 64, {7}},
      {Op::constant, 64, {}, 2},
      {Op::mul, 64, {8, 9}},
      {Op::image_address, 64, {}, 0x2000},
      {Op::constant, 64, {}, 0x1000},
      {Op::add, 64, {11, 12}},
      {Op::add, 64, {13, 10}},
      {Op::load, 16, {14}},
      {Op::zext, 64, {15}},
      {Op::constant, 64, {}, 0},
      {Op::add, 64, {16, 17}},
  };

  block.disabled_effects = {15};
  block.boundaries = {{0, static_cast<std::uint32_t>(block.nodes.size()), {{1, 18}}, {}}};
  block.edges = {
      {ir::SsaEdgeKind::fallthrough, ir::SsaTargetKind::image_location, 0x104, {}, {}, {}, {}}};
  ir::SsaConstantLoad fold{};
  fold.node = 15;
  fold.kind = ir::SsaConstantKind::literal;
  fold.value = 0xbbaa;
  fold.source_address = 0x302a;
  fold.declared_bytes = {0xaa, 0xbb};
  fold.access = {true, true, true};
  fold.value_stable = true;
  fold.skip_access = true;
  fold.condition = 4;
  fold.alternative_value = 0xddcc;
  fold.alternative_source_address = 0x302c;
  fold.alternative_declared_bytes = {0xcc, 0xdd};
  block.constant_loads = {fold};
  const auto handle = graph.Add(std::move(block));
  graph.Update(handle, [&](ir::SsaBlock& changed) {
    changed.exits.push_back({0, {ir::SsaValueKind::phi, handle, 0}});
    changed.exits.push_back({1, {ir::SsaValueKind::node, handle, 18}});
  });
  graph.SetEntries({handle});
  graph.SetLoadBias(0x7f0000000000);
  return graph;
}

TEST(SsaSimplify, KeepsTheImpliedBitChainASelectedFoldIsRecheckedThrough) {
  // Marking the condition alone is not enough: the recheck pins every link
  // the chain walks through, and a freed link folds to a constant.
  auto budget = Plenty();
  auto graph = ImpliedChainFold();
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  const auto result = ProposeSsaSimplify(graph, {}, false, budget);
  ASSERT_EQ(result.reason, SsaSimplifyRefusal::none);
  ASSERT_TRUE(result.provisional);
  EXPECT_EQ(ir::ValidateSsa(*result.provisional, budget), ir::SsaDecline::none);
}

TEST(SsaSimplify, KeepsASelectedFoldsAddressWhereAPlacementWouldFoldItToANumber) {
  // The only case that needs the address cone rather than the condition: a
  // declared placement lets the offset collapse into the location, and the
  // recheck refuses an address that has become a bare number.
  auto budget = Plenty();
  auto graph = OffsetFoldAtDeclaredBias();
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  const auto result = ProposeSsaSimplify(graph, {}, false, budget);
  EXPECT_EQ(result.reason, SsaSimplifyRefusal::none);
}

TEST(SsaSimplify, KeepsASelectedFoldsAddressDependentOnItsCondition) {
  // The fold is rechecked by pinning its condition to each value and
  // re-deriving both locations, which needs the address to still read the
  // condition. Simplify can prove the condition constant by another route,
  // and if it then folds the choice away the record no longer holds and the
  // whole batch is refused, losing every other edit in the graph with it.
  auto budget = Plenty();
  auto graph = SelectedFold();
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  const auto result = ProposeSsaSimplify(graph, {}, false, budget);
  ASSERT_EQ(result.reason, SsaSimplifyRefusal::none);

  // A refusal is not the only failure: proposing nothing would pass a test
  // that only looked at the reason, and the batch this guards is one where
  // the pass does rewrite and the record has to survive it.
  ASSERT_FALSE(result.journal.empty());
  ASSERT_TRUE(result.provisional);
  EXPECT_EQ(ir::ValidateSsa(*result.provisional, budget), ir::SsaDecline::none);

  // The fold is still a selected one, reading a condition that still exists.
  const auto& block = *result.provisional->Get(result.provisional->entries()[0]);
  ASSERT_EQ(block.constant_loads.size(), 1U);
  EXPECT_TRUE(block.constant_loads[0].condition);
}

// The node the block writes to storage 3, wherever splices moved it.
ValueId Output(const ir::SsaGraph& graph) {
  const auto& block = *graph.Get(graph.entries()[0]);
  return block.exits[3].value.index;
}

std::uint64_t Evaluate(const std::vector<Node>& nodes, ValueId output,
                       const std::array<std::uint64_t, kInputs>& inputs) {
  std::vector<std::uint64_t> values(nodes.size());
  for (ValueId id = 0; id < nodes.size(); ++id) {
    const auto& node = nodes[id];
    if (node.op == Op::read) {
      values[id] = node.storage < kInputs ? inputs[node.storage] : 0x1234;
    } else if (node.op == Op::constant) {
      values[id] = node.immediate & ir::LowMask(node.width);
    } else {
      const auto* descriptor = ir::Descriptor(node.op);
      std::array<std::uint64_t, 3> operands{};
      for (unsigned i = 0; i < descriptor->arity; ++i) operands[i] = values[node.inputs[i]];
      const auto folded = ir::FoldPure(node, std::span(operands).first(descriptor->arity),
                                       nodes[node.inputs[node.op == Op::select ? 1 : 0]].width);
      EXPECT_TRUE(folded) << Descriptor(node.op)->name;
      values[id] = folded.value_or(0);
    }
  }

  return values[output];
}

// Random expressions biased toward the shapes the rules target: masks, casts
// between widths, literal chains, and the MBA and sign-extension idioms.
class Generator {
 public:
  explicit Generator(std::uint64_t seed) : random_(seed) {
    for (unsigned i = 0; i < kInputs; ++i) Add({Op::read, 64, {}, 0, i});
    Add({Op::read, 64, {}, 0, 30});
  }

  ValueId Next() {
    const auto w = kWidths[Pick(kWidths.size())];
    switch (Pick(9)) {
      case 0:
        return Literal(w);
      case 1: {
        const auto a = Of(w);
        return Add({Op::bit_and, w, {a, Literal(w)}});
      }
      case 2: {  // (a | b) & ~(a & b) with a mask standing in for the not
        const auto a = Of(w), b = Of(w);
        const auto both = Add({Op::bit_and, w, {a, b}});
        const auto not_both = Add({Op::bit_xor, w, {both, Constant(w, ir::LowMask(w))}});
        return Add({Op::bit_and, w, {Add({Op::bit_or, w, {a, b}}), not_both}});
      }
      case 3: {  // sel(bit n-1, high ones, 0) | (x masked to n bits)
        if (w < 16) break;
        const unsigned n = w == 16 ? 8 : 16;
        const auto low = Add({Op::bit_and, w, {Of(w), Constant(w, ir::LowMask(n))}});
        const auto bit = Add({Op::extract, 1, {low}, n - 1});
        const auto high = Add(
            {Op::select, w, {bit, Constant(w, ir::LowMask(w) & ~ir::LowMask(n)), Constant(w, 0)}});
        return Add({Op::bit_or, w, {high, low}});
      }
      case 4: {  // literal chains
        const auto a = Of(w);
        return Add(
            {Pick(2) ? Op::add : Op::sub, w, {Add({Op::add, w, {a, Literal(w)}}), Literal(w)}});
      }
      case 5: {  // rotation of a narrow value inside a wider register
        if (w < 16) break;
        const auto narrow = Add({Op::zext, w, {Add({Op::extract, 8, {Of(64)}, 0})}});
        const auto left = Add({Op::shl, w, {narrow, Constant(w, 3)}});
        const auto right = Add({Op::lshr, w, {narrow, Constant(w, w - 3)}});
        return Add({Op::bit_and, w, {Add({Op::bit_or, w, {left, right}}), Constant(w, 0x7f8)}});
      }
      case 6: {  // casts
        const auto from = kWidths[Pick(kWidths.size())];
        const auto source = Of(from);
        if (from == w) return Add({Op::zext, w, {source}});
        if (from < w) return Add({Op::zext, w, {source}});
        return Add({Op::extract, w, {source}, Pick(2) ? 0 : Pick(from - w + 1)});
      }
      case 7: {  // comparisons and selects
        const auto a = Of(w), b = Pick(2) ? Of(w) : Literal(w);
        const std::array ops{Op::equal, Op::unsigned_less, Op::signed_less};
        const auto test = Add({ops[Pick(3)], 1, {a, b}});
        return Add({Op::select, w, {test, Of(w), Pick(2) ? Of(w) : Literal(w)}});
      }
      default:
        break;
    }

    const std::array ops{Op::add,     Op::sub,     Op::mul, Op::bit_and, Op::bit_or,
                         Op::bit_xor, Op::bit_not, Op::shl, Op::lshr,    Op::ashr};
    const auto op = ops[Pick(ops.size())];
    if (op == Op::bit_not) return Add({op, w, {Of(w)}});
    const bool shift = op == Op::shl || op == Op::lshr || op == Op::ashr;
    return Add({op, w, {Of(w), shift ? Constant(w, Pick(w + 1)) : Pick(3) ? Of(w) : Literal(w)}});
  }

  std::vector<Node> nodes;

  ValueId Of(unsigned w) {
    std::vector<ValueId> fitting;
    for (ValueId id = 0; id < nodes.size(); ++id)
      if (nodes[id].width == w) fitting.push_back(id);
    if (!fitting.empty() && Pick(5)) return fitting[Pick(fitting.size())];
    if (w == 64) return static_cast<ValueId>(Pick(kInputs));
    return Add({Op::extract, w, {static_cast<ValueId>(Pick(kInputs))}, 0});
  }

 private:
  std::uint64_t Pick(std::uint64_t n) { return n ? random_() % n : 0; }

  ValueId Add(Node node) {
    nodes.push_back(node);
    return static_cast<ValueId>(nodes.size() - 1);
  }

  ValueId Constant(unsigned w, std::uint64_t value) {
    return Add({Op::constant, w, {}, value & ir::LowMask(w)});
  }

  ValueId Literal(unsigned w) {
    const std::array<std::uint64_t, 12> pool{0,
                                             1,
                                             2,
                                             3,
                                             5,
                                             0xff,
                                             0xffff,
                                             0xfff8,
                                             0x7f8,
                                             0xffff0000,
                                             ~std::uint64_t{0},
                                             0x8000000000000000};
    return Constant(w, Pick(4) ? pool[Pick(pool.size())] : random_());
  }

  std::mt19937_64 random_;
};

TEST(SsaSimplify, EveryRewriteKeepsTheValueOnRandomExpressions) {
  std::mt19937_64 inputs(7);
  std::size_t rewrites = 0, inserted = 0;
  for (std::uint64_t seed = 1; seed <= 400; ++seed) {
    Generator generator(seed);
    ValueId output = 0;
    for (unsigned count = 0; count < 24; ++count) output = generator.Next();
    auto nodes = generator.nodes;
    if (nodes[output].width != 64) {
      nodes.push_back({Op::zext, 64, {output}});
      output = static_cast<ValueId>(nodes.size() - 1);
    }

    auto budget = Plenty();
    const auto graph = Build(nodes, output);
    ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none) << seed;
    const auto result = ProposeSsaSimplify(graph, {}, false, budget);
    ASSERT_NE(result.reason, SsaSimplifyRefusal::invalid_graph) << seed;
    if (!result.provisional) continue;
    rewrites += result.journal.size();
    for (const auto& edit : result.journal) inserted += edit.inserted;
    const auto& simplified = *result.provisional->Get(result.provisional->entries()[0]);
    const auto exit = simplified.exits[3].value;
    ASSERT_EQ(exit.kind, ir::SsaValueKind::node);
    for (unsigned trial = 0; trial < 64; ++trial) {
      std::array<std::uint64_t, kInputs> values{};
      for (auto& value : values) {
        const std::array<std::uint64_t, 6> edges{0,    1,      ~std::uint64_t{0},
                                                 0x80, 0x8000, 0x8000000000000000};
        value = trial < 6 ? edges[trial] : inputs();
      }

      ASSERT_EQ(Evaluate(simplified.nodes, exit.index, values), Evaluate(nodes, output, values))
          << "seed " << seed << " trial " << trial;
    }
  }

  // The generator must exercise the rules, not only pass through them.
  EXPECT_GT(rewrites, 2000U);
  EXPECT_GT(inserted, 50U);
}

TEST(SsaSimplify, KnownIdiomsBecomeTheirPlainForms) {
  // (a | b) & ((a & b) ^ 0xff) over bytes is a ^ b.
  std::vector<Node> nodes{{Op::read, 64, {}, 0, 0},
                          {Op::read, 64, {}, 0, 1},
                          {Op::read, 64, {}, 0, 2},
                          {Op::read, 64, {}, 0, 30}};
  nodes.push_back({Op::extract, 8, {0}, 0});     // 4 a
  nodes.push_back({Op::extract, 8, {1}, 0});     // 5 b
  nodes.push_back({Op::bit_or, 8, {4, 5}});      // 6
  nodes.push_back({Op::bit_and, 8, {4, 5}});     // 7
  nodes.push_back({Op::constant, 8, {}, 0xff});  // 8
  nodes.push_back({Op::bit_xor, 8, {7, 8}});     // 9
  nodes.push_back({Op::bit_and, 8, {6, 9}});     // 10
  nodes.push_back({Op::zext, 64, {10}});         // 11
  auto budget = Plenty();
  const auto result = ProposeSsaSimplify(Build(nodes, 11), {}, false, budget);
  ASSERT_TRUE(result.provisional);
  const auto& block = *result.provisional->Get(result.provisional->entries()[0]);
  EXPECT_EQ(block.nodes[10].op, Op::bit_xor);
  EXPECT_EQ(std::minmax(block.nodes[10].inputs[0], block.nodes[10].inputs[1]), std::pair(4U, 5U));
}

TEST(SsaSimplify, TwoOperandMbaIdentitiesBecomePlainArithmetic) {
  const std::vector<Node> inputs{{Op::read, 64, {}, 0, 0},
                                 {Op::read, 64, {}, 0, 1},
                                 {Op::read, 64, {}, 0, 2},
                                 {Op::read, 64, {}, 0, 30}};

  struct Case {
    std::vector<Node> nodes;
    Op plain;
  };

  // Operands are nodes 0 and 1; 4 is the literal 1, 5 is a | b, 6 a ^ b,
  // 7 a & b, then the idiom.
  const auto with = [&](std::vector<Node> tail) {
    auto nodes = inputs;
    nodes.push_back({Op::constant, 64, {}, 1});
    nodes.push_back({Op::bit_or, 64, {0, 1}});
    nodes.push_back({Op::bit_xor, 64, {1, 0}});
    nodes.push_back({Op::bit_and, 64, {0, 1}});
    nodes.insert(nodes.end(), tail.begin(), tail.end());
    return nodes;
  };

  const std::vector<Case> cases{
      {with({{Op::shl, 64, {5, 4}}, {Op::sub, 64, {8, 6}}}), Op::add},
      {with({{Op::shl, 64, {7, 4}}, {Op::add, 64, {8, 6}}}), Op::add},
      {with({{Op::add, 64, {5, 7}}}), Op::add},
      {with({{Op::sub, 64, {5, 7}}}), Op::bit_xor},
      {with({{Op::add, 64, {0, 1}}, {Op::add, 64, {7, 7}}, {Op::sub, 64, {8, 9}}}), Op::bit_xor}};
  for (const auto& item : cases) {
    auto budget = Plenty();
    const auto output = static_cast<ValueId>(item.nodes.size() - 1);
    const auto result = ProposeSsaSimplify(Build(item.nodes, output), {}, false, budget);
    ASSERT_TRUE(result.provisional);
    const auto& block = *result.provisional->Get(result.provisional->entries()[0]);
    const auto& plain = block.nodes[Output(*result.provisional)];
    EXPECT_EQ(plain.op, item.plain);
    EXPECT_EQ(std::minmax(plain.inputs[0], plain.inputs[1]), std::pair(0U, 1U));
  }
}

// (@A ^ c) & @A: a bitwise function of one image location whose load bias
// cancels, as in an obfuscated address that is really a small number.
std::vector<Node> BiasFree(bool store) {
  std::vector<Node> nodes{{Op::read, 64, {}, 0, 0},
                          {Op::read, 64, {}, 0, 1},
                          {Op::read, 64, {}, 0, 2},
                          {Op::read, 64, {}, 0, 30}};
  nodes.push_back({Op::image_address, 64, {}, 0x1d4a4cc});         // 4
  nodes.push_back({Op::constant, 64, {}, ~std::uint64_t{0x137}});  // 5
  nodes.push_back({Op::bit_xor, 64, {4, 5}});                      // 6
  nodes.push_back({Op::bit_and, 64, {6, 4}});                      // 7
  if (store) {
    nodes.push_back({Op::extract, 32, {0}, 0});  // 8
    nodes.push_back({Op::store, 32, {7, 8}});    // 9
  }

  return nodes;
}

TEST(SsaSimplify, AnImageLocationsInPageBitsFoldOnlyUnderPlacement) {
  auto budget = Plenty();
  const auto placed = ProposeSsaSimplify(Build(BiasFree(false), 7), {}, true, budget);
  ASSERT_TRUE(placed.provisional);
  const auto& block = *placed.provisional->Get(placed.provisional->entries()[0]);
  const auto& folded = block.nodes[Output(*placed.provisional)];
  EXPECT_EQ(folded.op, Op::constant);
  EXPECT_EQ(folded.immediate, 0x1d4a4ccU & 0x137U);
  EXPECT_TRUE(std::any_of(placed.journal.begin(), placed.journal.end(),
                          [](const SsaSimplifyEdit& edit) { return edit.placement; }));
  const auto unplaced = ProposeSsaSimplify(Build(BiasFree(false), 7), {}, false, budget);
  ASSERT_TRUE(unplaced.provisional);
  const auto& kept = unplaced.provisional->Get(unplaced.provisional->entries()[0])->nodes;
  const auto& masked = kept[Output(*unplaced.provisional)];
  ASSERT_EQ(masked.op, Op::bit_and);
  EXPECT_EQ(kept[masked.inputs[0]].op, Op::image_address);
  EXPECT_EQ(kept[masked.inputs[1]].immediate, 0x137U);
}

TEST(SsaImageStores, AStoreWhoseAddressTheBiasCannotChangeIsNotAnImageStore) {
  auto budget = Plenty();
  const auto graph = Build(BiasFree(true), 2);

  // The address is 0x1d4a4cc & 0x137, a number, under placement; without it,
  // an image-derived address the checker cannot place conflicts with all.
  EXPECT_EQ(ir::SsaConflictingImageStore(graph, 0x1d4a4cc, 4, true, budget), false);
  EXPECT_EQ(ir::SsaConflictingImageStore(graph, 0x1d4a4cc, 4, false, budget), true);
  auto biased = BiasFree(true);
  biased[9].inputs[0] = 6;  // @A ^ c still moves with the bias
  EXPECT_EQ(ir::SsaConflictingImageStore(Build(biased, 2), 0x1d4a4cc, 4, true, budget), true);
}

// A store, then a load of the same bytes whose value nothing uses.
ir::SsaGraph Reload(bool same_address, bool used) {
  std::vector<Node> nodes{{Op::read, 64, {}, 0, 0},
                          {Op::read, 64, {}, 0, 1},
                          {Op::read, 64, {}, 0, 2},
                          {Op::read, 64, {}, 0, 30}};
  nodes.push_back({Op::constant, 64, {}, 16});                        // 4
  nodes.push_back({Op::add, 64, {0, 4}});                             // 5
  nodes.push_back({Op::store, 64, {5, 1}});                           // 6
  nodes.push_back({Op::constant, 64, {}, same_address ? 16U : 24U});  // 7
  nodes.push_back({Op::add, 64, {0, 7}});                             // 8
  nodes.push_back({Op::load, 64, {8}});                               // 9
  nodes.push_back({Op::add, 64, {2, 4}});                             // 10
  return Build(nodes, used ? 9 : 10);
}

TEST(SsaDeadLoads, RetiresAReloadOnlyWhenAStoreProvesItsBytesAndNothingUsesIt) {
  auto budget = Plenty();
  auto result = ProposeSsaDeadLoads(Reload(true, false), {}, {}, {}, budget);
  ASSERT_TRUE(result.provisional);
  ASSERT_EQ(result.journal.size(), 1U);
  EXPECT_EQ(result.journal[0].retired.node, 9U);
  EXPECT_EQ(result.journal[0].retired.basis, ir::SsaRetiredLoadBasis::written_before);
  EXPECT_EQ(result.journal[0].retired.store, 6U);
  EXPECT_FALSE(ProposeSsaDeadLoads(Reload(false, false), {}, {}, {}, budget).provisional);
  EXPECT_FALSE(ProposeSsaDeadLoads(Reload(true, true), {}, {}, {}, budget).provisional);

  // The checker refuses a retired load whose value is used.
  auto used = Reload(true, true);
  const auto handle = used.entries()[0];
  used.Update(handle, [](ir::SsaBlock& block) {
    block.disabled_effects = {9};
    block.retired_loads = {{9, ir::SsaRetiredLoadBasis::written_before, 6}};
  });
  EXPECT_EQ(ir::ValidateSsa(used, budget), ir::SsaDecline::invalid_graph);
}

TEST(SsaDeadLoads, RetiresAnUnusedReadOfARelocatedSlotUnderTheAccessContract) {
  std::vector<Node> nodes{{Op::read, 64, {}, 0, 0},
                          {Op::read, 64, {}, 0, 1},
                          {Op::read, 64, {}, 0, 2},
                          {Op::read, 64, {}, 0, 30}};
  nodes.push_back({Op::image_address, 64, {}, 0x2000});  // 4
  nodes.push_back({Op::load, 64, {4}});                  // 5
  const std::array<ir::RelocatedPointer, 1> slots{{{0x2000, 0x1234, true}}};
  const ir::ImageFacts facts{{}, slots, false, {}};
  auto budget = Plenty();
  const auto retired = ProposeSsaDeadLoads(Build(nodes, 2), {}, facts, {true, true, true}, budget);
  ASSERT_TRUE(retired.provisional);
  ASSERT_EQ(retired.journal.size(), 1U);
  EXPECT_EQ(retired.journal[0].retired.basis, ir::SsaRetiredLoadBasis::declared_image);
  EXPECT_EQ(retired.journal[0].retired.range_address, 0x2000U);
  EXPECT_EQ(retired.journal[0].retired.range_bytes, 8U);

  // Without the access contract the read may fault, so it stays.
  EXPECT_FALSE(ProposeSsaDeadLoads(Build(nodes, 2), {}, facts, {}, budget).provisional);

  // Nor does a slot the read overruns vouch for it.
  const std::array<ir::RelocatedPointer, 1> early{{{0x1ffc, 0x1234, true}}};
  EXPECT_FALSE(
      ProposeSsaDeadLoads(Build(nodes, 2), {}, {{}, early, false, {}}, {true, true, true}, budget)
          .provisional);
}

}  // namespace
}  // namespace nyx::recovery
