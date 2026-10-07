#include "nyx/recovery/mba.hpp"

#include <algorithm>
#include <limits>
#include <random>

#include <gtest/gtest.h>

#include "nyx/support/bit_vector.hpp"

namespace nyx::recovery {
namespace {
using ir::Node;
using ir::Op;

Budget Plenty() { return Budget({100000000, 100000000}); }

ir::Block Normalize(std::vector<Node> nodes, ir::MemoryModel model = ir::MemoryModel::unspecified) {
  auto budget = Plenty();
  const auto last = static_cast<ir::ValueId>(nodes.size() - 1);
  const std::vector<ir::Group> groups{
      ir::Group(0x1000, {1, 2, 3, 4}, std::move(nodes), {{300, last}}, model)};
  auto result = ir::Normalize(groups, budget);
  EXPECT_TRUE(result.block.has_value());
  return std::move(*result.block);
}

ir::Block Identity(unsigned width, bool commute = false) {
  return Normalize({{Op::read, width, {}, 0, 100},
                    {Op::read, width, {}, 0, 200},
                    {Op::bit_or, width, {0, 1}},
                    {Op::bit_xor, width, {commute ? 1U : 0U, commute ? 0U : 1U}},
                    {Op::constant, width, {}, 1},
                    {Op::shl, width, {2, 4}},
                    {Op::sub, width, {5, 3}}});
}

BitVector Evaluate(std::span<const Node> nodes, std::span<const std::uint64_t> a,
                   std::span<const std::uint64_t> b) {
  auto budget = Plenty();
  std::vector<BitVector> values;
  for (const auto& node : nodes) {
    auto value = [&]() -> BitVectorResult {
      switch (node.op) {
        case Op::read:
          return BitVector::from_words(node.width, node.storage == 100 ? a : b, 4096, budget);
        case Op::constant:
          return BitVector::from_u64(node.width, node.immediate, 4096, budget);
        case Op::add:
          return values[node.inputs[0]].add(values[node.inputs[1]], budget);
        case Op::sub:
          return values[node.inputs[0]].sub(values[node.inputs[1]], budget);
        case Op::mul:
          return values[node.inputs[0]].mul(values[node.inputs[1]], budget);
        case Op::bit_and:
          return values[node.inputs[0]].bit_and(values[node.inputs[1]], budget);
        case Op::bit_or:
          return values[node.inputs[0]].bit_or(values[node.inputs[1]], budget);
        case Op::bit_xor:
          return values[node.inputs[0]].bit_xor(values[node.inputs[1]], budget);
        case Op::bit_not:
          return values[node.inputs[0]].bit_not(budget);
        case Op::shl:
          return values[node.inputs[0]].shl(values[node.inputs[1]].word(0), budget);
        default:
          ADD_FAILURE() << "unexpected reference operation";
          return BitVectorDecline::invalid_width;
      }
    }();
    EXPECT_TRUE(static_cast<bool>(value));
    values.push_back(std::move(*value));
  }

  return std::move(values.back());
}

TEST(Mba, AppliesCommutedIdentityAtSameNodeAndPreservesSource) {
  for (bool commute : {false, true}) {
    const auto block = Identity(64, commute);
    auto budget = Plenty();
    auto result = SimplifyMba(block, budget);
    ASSERT_TRUE(result.block);
    ASSERT_EQ(result.journal.size(), 1);
    const auto& edit = result.journal[0];
    EXPECT_EQ(edit.rule, MbaRule::or_xor_sum);
    EXPECT_EQ(edit.node, 6);
    EXPECT_EQ(edit.original.op, Op::sub);
    EXPECT_EQ(edit.replacement.op, Op::add);
    EXPECT_EQ(edit.width, 64);
    EXPECT_EQ(edit.from_revision, 0);
    EXPECT_EQ(edit.to_revision, 1);
    EXPECT_EQ(result.block->revision(), 1);
    EXPECT_EQ(result.block->nodes()[6].op, Op::add);
    EXPECT_EQ(result.block->nodes().size(), block.nodes().size());
    EXPECT_EQ(block.nodes()[6].op, Op::sub);
    EXPECT_EQ(result.block->sources()[0].nodes()[6].op, Op::sub);
    EXPECT_EQ(result.block->sources()[0].bytes()[2], 3);
    EXPECT_EQ(result.block->boundaries()[0].writes[0].value, 6);
    EXPECT_EQ(result.block->origins()[6].operation, block.origins()[6].operation);
  }
}

TEST(Mba, IdentityMatchesIndependentIntegersAtSmallAndDeepWidths) {
  std::mt19937_64 random(0x194582);
  for (unsigned width : {1U, 2U, 3U, 7U, 31U, 64U, 65U, 127U, 256U, 1023U, 4096U}) {
    const auto block = Identity(width);
    auto budget = Plenty();
    auto result = SimplifyMba(block, budget);
    ASSERT_TRUE(result.block);
    ASSERT_EQ(result.block->nodes().back().op, Op::add);
    const unsigned cases = width <= 3 ? 1U << (2 * width) : 40;
    for (unsigned test = 0; test < cases; ++test) {
      std::vector<std::uint64_t> a((width + 63) / 64), b(a.size()), expected(a.size());
      if (width <= 3) {
        a[0] = test >> width;
        b[0] = test & ((1U << width) - 1);
      } else {
        for (std::size_t i = 0; i < a.size(); ++i) {
          a[i] = random();
          b[i] = random();
        }
      }

      // A bit-at-a-time full adder is independent of the limb arithmetic used
      // by the evaluator and covers widths beyond the constant representation.
      unsigned carry = 0;
      for (unsigned bit = 0; bit < width; ++bit) {
        const auto total =
            ((a[bit / 64] >> (bit % 64)) & 1) + ((b[bit / 64] >> (bit % 64)) & 1) + carry;
        expected[bit / 64] |= (total & 1) << (bit % 64);
        carry = static_cast<unsigned>(total >> 1);
      }

      const auto before = Evaluate(block.nodes(), a, b);
      const auto after = Evaluate(result.block->nodes(), a, b);
      EXPECT_EQ(before, after);
      for (std::size_t i = 0; i < expected.size(); ++i) EXPECT_EQ(after.word(i), expected[i]);
    }
  }
}

TEST(Mba, CollapsesNegatedUnknownLoadAndComposedConstantWithoutRemovingEffects) {
  const auto block = Normalize({{Op::constant, 64, {}, 0x4000},
                                {Op::load, 64, {0}},
                                {Op::constant, 64, {}, 0},
                                {Op::sub, 64, {2, 1}},
                                {Op::constant, 64, {}, 0x1234},
                                {Op::constant, 64, {}, 16},
                                {Op::shl, 64, {4, 5}},
                                {Op::constant, 64, {}, 0xabcd},
                                {Op::bit_or, 64, {6, 7}},
                                {Op::bit_or, 64, {3, 8}},
                                {Op::bit_xor, 64, {8, 3}},
                                {Op::constant, 64, {}, 1},
                                {Op::shl, 64, {9, 11}},
                                {Op::sub, 64, {12, 10}}},
                               ir::MemoryModel::atomic_scalar_reference);
  auto budget = Plenty();
  auto result = SimplifyMba(block, budget);
  ASSERT_TRUE(result.block);
  EXPECT_EQ(result.block->nodes()[1].op, Op::load);
  EXPECT_EQ(result.block->nodes()[8].op, Op::constant);
  EXPECT_EQ(result.block->nodes()[8].immediate, 0x1234abcd);
  EXPECT_EQ(result.block->nodes()[13].op, Op::sub);
  EXPECT_EQ(result.block->nodes()[13].inputs[0], 8);
  EXPECT_EQ(result.block->nodes()[13].inputs[1], 1);
  EXPECT_EQ(result.journal.size(), 4);
}

TEST(Mba, DistinctRepeatedLoadsAreNotEqualDefinitions) {
  const auto block = Normalize({{Op::constant, 64, {}, 0x4000},
                                {Op::load, 64, {0}},
                                {Op::load, 64, {0}},
                                {Op::read, 64, {}, 0, 200},
                                {Op::bit_or, 64, {1, 3}},
                                {Op::bit_xor, 64, {2, 3}},
                                {Op::constant, 64, {}, 1},
                                {Op::shl, 64, {4, 6}},
                                {Op::sub, 64, {7, 5}}},
                               ir::MemoryModel::atomic_scalar_reference);
  auto budget = Plenty();
  auto result = SimplifyMba(block, budget);
  ASSERT_TRUE(result.block);
  EXPECT_TRUE(result.journal.empty());
  EXPECT_EQ(result.block->revision(), block.revision());
  EXPECT_EQ(result.block->nodes()[8].op, Op::sub);
}

TEST(Mba, SeparateEqualLiteralsMatchButDifferentValuesDoNot) {
  for (unsigned width : {7U, 64U, 65U, 4096U}) {
    for (bool equal : {false, true}) {
      const auto second = equal ? (width < 64 ? 0x185ULL : 5ULL) : 6ULL;
      const auto block = Normalize({{Op::read, width, {}, 0, 100},
                                    {Op::constant, width, {}, 0},
                                    {Op::sub, width, {1, 0}},
                                    {Op::constant, width, {}, 5},
                                    {Op::constant, width, {}, second},
                                    {Op::bit_or, width, {2, 3}},
                                    {Op::bit_xor, width, {4, 2}},
                                    {Op::constant, width, {}, 1},
                                    {Op::shl, width, {5, 7}},
                                    {Op::sub, width, {8, 6}}});
      auto budget = Plenty();
      const auto result = SimplifyMba(block, budget);
      ASSERT_TRUE(result.block);
      if (equal) {
        ASSERT_EQ(result.journal.size(), 2);
        EXPECT_EQ(result.journal[0].rule, MbaRule::or_xor_sum);
        EXPECT_EQ(result.journal[1].rule, MbaRule::negated_add);
        EXPECT_EQ(result.block->nodes().back().op, Op::sub);
        EXPECT_EQ(result.block->nodes().back().inputs[0], 3);
        EXPECT_EQ(result.block->nodes().back().inputs[1], 0);
      } else {
        EXPECT_TRUE(result.journal.empty());
        EXPECT_EQ(result.block->nodes().back().inputs[0], 8);
      }
    }
  }
}

TEST(Mba, RewritesAcrossSourcesWithoutDroppingIntermediateWritesOrFlags) {
  const std::vector<ir::Group> groups{
      ir::Group(0x1000, {1, 2, 3, 4},
                {{Op::read, 64, {}, 0, 100},
                 {Op::read, 64, {}, 0, 200},
                 {Op::bit_or, 64, {0, 1}},
                 {Op::bit_xor, 64, {0, 1}},
                 {Op::constant, 64, {}, 1},
                 {Op::shl, 64, {2, 4}},
                 {Op::equal, 1, {0, 1}}},
                {{110, 5}, {120, 3}, {130, 6}}),
      ir::Group(0x1004, {5, 6, 7, 8},
                {{Op::read, 64, {}, 0, 110}, {Op::read, 64, {}, 0, 120}, {Op::sub, 64, {0, 1}}},
                {{300, 2}})};
  auto budget = Plenty();
  const auto block = ir::Normalize(groups, budget);
  ASSERT_TRUE(block.block);
  auto result = SimplifyMba(*block.block, budget);
  ASSERT_TRUE(result.block);
  ASSERT_EQ(result.journal.size(), 1);
  EXPECT_EQ(result.block->nodes().back().op, Op::add);
  EXPECT_EQ(result.block->nodes()[6].op, Op::equal);
  ASSERT_EQ(result.block->boundaries().size(), 2);
  EXPECT_EQ(result.block->boundaries()[0].writes.size(), 3);
  EXPECT_EQ(result.block->boundaries()[0].writes[0].value, 5);
  EXPECT_EQ(result.block->boundaries()[0].writes[2].value, 6);
  EXPECT_EQ(result.block->origins().back().boundary, 1);
  EXPECT_EQ(result.block->sources()[1].nodes().back().op, Op::sub);
}

TEST(Mba, InvalidWidthsAndRevisionOverflowDeclineWithoutJournal) {
  const auto original = Identity(64);
  for (bool revision : {false, true}) {
    std::vector<Node> nodes(original.nodes().begin(), original.nodes().end());
    if (!revision) nodes.back().width = 32;
    const ir::Block invalid(
        {original.sources().begin(), original.sources().end()}, std::move(nodes),
        {original.origins().begin(), original.origins().end()},
        {original.boundaries().begin(), original.boundaries().end()}, revision ? UINT64_MAX : 0);
    auto budget = Plenty();
    const auto result = SimplifyMba(invalid, budget);
    EXPECT_FALSE(result.block);
    EXPECT_TRUE(result.journal.empty());
    EXPECT_EQ(result.reason, revision ? MbaDecline::revision_overflow : MbaDecline::invalid_ir);
  }
}

TEST(Mba, EqualLiteralsAtDifferentWidthsAreNotAcceptedAsMbaOperands) {
  const auto original = Normalize({{Op::read, 64, {}, 0, 100},
                                   {Op::constant, 64, {}, 5},
                                   {Op::constant, 64, {}, 5},
                                   {Op::bit_or, 64, {0, 1}},
                                   {Op::bit_xor, 64, {0, 2}},
                                   {Op::constant, 64, {}, 1},
                                   {Op::shl, 64, {3, 5}},
                                   {Op::sub, 64, {6, 4}}});
  std::vector<Node> nodes(original.nodes().begin(), original.nodes().end());
  nodes[2].width = 32;
  const ir::Block invalid({original.sources().begin(), original.sources().end()}, std::move(nodes),
                          {original.origins().begin(), original.origins().end()},
                          {original.boundaries().begin(), original.boundaries().end()});
  auto budget = Plenty();
  const auto result = SimplifyMba(invalid, budget);
  EXPECT_FALSE(result.block);
  EXPECT_TRUE(result.journal.empty());
  EXPECT_EQ(result.reason, MbaDecline::invalid_ir);
}

TEST(Mba, WrongShiftAndDifferentSsaOperandsDoNotMatch) {
  for (bool shift : {false, true}) {
    auto block = Identity(64);
    std::vector<Node> nodes(block.nodes().begin(), block.nodes().end());
    if (shift)
      nodes[4].immediate = 2;
    else
      nodes[3].inputs[1] = 0;
    auto altered = Normalize(std::move(nodes));
    auto budget = Plenty();
    auto result = SimplifyMba(altered, budget);
    ASSERT_TRUE(result.block);
    EXPECT_TRUE(result.journal.empty());
  }
}

// The Boolean identities the obfuscator emits around the exposed variable-shift
// regions. Each fixture is the shape recovery must recognize; `commute` swaps the
// operands that the identity is allowed to see in either order.
enum class BooleanIdentity { and_xor_union, or_and_sum, xor_and_sum, xor_ones_not };

std::vector<Node> BooleanIdentityNodes(BooleanIdentity identity, unsigned width, bool commute) {
  const std::vector<Node> pair{{Op::read, width, {}, 0, 100}, {Op::read, width, {}, 0, 200}};
  const auto order = [commute](ir::ValueId first, ir::ValueId second) {
    return std::array<ir::ValueId, 3>{commute ? second : first, commute ? first : second};
  };

  std::vector<Node> nodes = pair;
  switch (identity) {
    case BooleanIdentity::and_xor_union:
      nodes.push_back({Op::bit_and, width, {0, 1}});
      nodes.push_back({Op::bit_xor, width, order(0, 1)});
      nodes.push_back({Op::bit_or, width, order(2, 3)});
      return nodes;
    case BooleanIdentity::or_and_sum:
      nodes.push_back({Op::bit_or, width, {0, 1}});
      nodes.push_back({Op::bit_and, width, order(0, 1)});
      nodes.push_back({Op::add, width, order(2, 3)});
      return nodes;
    case BooleanIdentity::xor_and_sum:
      nodes.push_back({Op::bit_xor, width, {0, 1}});
      nodes.push_back({Op::bit_and, width, order(0, 1)});
      nodes.push_back({Op::constant, width, {}, 1});
      nodes.push_back({Op::shl, width, {3, 4}});
      nodes.push_back({Op::add, width, order(2, 5)});
      return nodes;
    case BooleanIdentity::xor_ones_not:
      nodes.resize(1);
      nodes.push_back(
          {Op::constant, width, {}, width == 64 ? UINT64_MAX : (UINT64_C(1) << width) - 1});
      nodes.push_back({Op::bit_xor, width, order(0, 1)});
      return nodes;
  }

  return nodes;
}

constexpr std::array<BooleanIdentity, 4> kBooleanIdentities = {
    BooleanIdentity::and_xor_union, BooleanIdentity::or_and_sum, BooleanIdentity::xor_and_sum,
    BooleanIdentity::xor_ones_not};

MbaRule RuleOf(BooleanIdentity identity) {
  switch (identity) {
    case BooleanIdentity::and_xor_union:
      return MbaRule::and_xor_union;
    case BooleanIdentity::or_and_sum:
      return MbaRule::or_and_sum;
    case BooleanIdentity::xor_and_sum:
      return MbaRule::xor_and_sum;
    case BooleanIdentity::xor_ones_not:
      return MbaRule::xor_ones_not;
  }

  return MbaRule::constant_fold;
}

Op ReplacementOf(BooleanIdentity identity) {
  switch (identity) {
    case BooleanIdentity::and_xor_union:
      return Op::bit_or;
    case BooleanIdentity::or_and_sum:
    case BooleanIdentity::xor_and_sum:
      return Op::add;
    case BooleanIdentity::xor_ones_not:
      return Op::bit_not;
  }

  return Op::constant;
}

TEST(Mba, AppliesEachBooleanIdentityInEitherOperandOrderAndKeepsItsSource) {
  for (const auto identity : kBooleanIdentities) {
    for (const bool commute : {false, true}) {
      const auto nodes = BooleanIdentityNodes(identity, 64, commute);
      const auto block = Normalize(nodes);
      auto budget = Plenty();
      const auto result = SimplifyMba(block, budget);
      ASSERT_TRUE(result.block);
      ASSERT_EQ(result.journal.size(), 1);
      const auto& edit = result.journal[0];
      const auto last = static_cast<ir::ValueId>(nodes.size() - 1);
      EXPECT_EQ(edit.rule, RuleOf(identity));
      EXPECT_EQ(edit.node, last);
      EXPECT_EQ(edit.original.op, nodes.back().op);
      EXPECT_EQ(edit.replacement.op, ReplacementOf(identity));
      EXPECT_EQ(edit.width, 64);
      EXPECT_EQ(edit.from_revision, 0);
      EXPECT_EQ(edit.to_revision, 1);

      // Every identity resolves to the values the obfuscated form consumed; the
      // matched subexpression supplies their order, which the result may commute.
      const auto& replaced = result.block->nodes()[last];
      EXPECT_EQ(std::min(replaced.inputs[0], replaced.inputs[1]), 0U);
      if (identity != BooleanIdentity::xor_ones_not) {
        EXPECT_EQ(std::max(replaced.inputs[0], replaced.inputs[1]), 1U);
      }

      EXPECT_EQ(result.block->nodes().size(), block.nodes().size());
      EXPECT_EQ(result.block->revision(), 1);
      EXPECT_EQ(block.nodes()[last].op, nodes.back().op);
      EXPECT_EQ(result.block->sources()[0].nodes()[last].op, nodes.back().op);
      EXPECT_EQ(result.block->boundaries()[0].writes[0].value, last);
      EXPECT_EQ(result.block->origins()[last].operation, block.origins()[last].operation);
    }
  }
}

TEST(Mba, BooleanIdentitiesMatchIndependentBitwiseIntegersAtEveryWidth) {
  std::mt19937_64 random(0x0b001ea4);
  for (const auto identity : kBooleanIdentities) {
    // Above 64 bits no constant can denote all-ones, so that identity declines.
    const std::vector<unsigned> widths =
        identity == BooleanIdentity::xor_ones_not
            ? std::vector<unsigned>{1, 2, 3, 7, 31, 32, 64}
            : std::vector<unsigned>{1, 2, 3, 7, 31, 64, 65, 127, 256, 1023, 4096};
    for (const unsigned width : widths) {
      const auto block = Normalize(BooleanIdentityNodes(identity, width, false));
      auto budget = Plenty();
      const auto result = SimplifyMba(block, budget);
      ASSERT_TRUE(result.block);
      ASSERT_EQ(result.block->nodes().back().op, ReplacementOf(identity));
      const unsigned cases = width <= 3 ? 1U << (2 * width) : 40;
      for (unsigned test = 0; test < cases; ++test) {
        std::vector<std::uint64_t> a((width + 63) / 64), b(a.size()), expected(a.size());
        if (width <= 3) {
          a[0] = test >> width;
          b[0] = test & ((1U << width) - 1);
        } else {
          for (std::size_t i = 0; i < a.size(); ++i) {
            a[i] = random();
            b[i] = random();
          }
        }

        // A bit-at-a-time model is independent of the limb arithmetic the
        // evaluator uses and covers widths beyond the constant representation.
        unsigned carry = 0;
        for (unsigned bit = 0; bit < width; ++bit) {
          const auto left = (a[bit / 64] >> (bit % 64)) & 1,
                     right = (b[bit / 64] >> (bit % 64)) & 1;
          std::uint64_t value = 0;
          switch (identity) {
            case BooleanIdentity::and_xor_union:
              value = left | right;
              break;
            case BooleanIdentity::xor_ones_not:
              value = left ^ 1;
              break;
            case BooleanIdentity::or_and_sum:
            case BooleanIdentity::xor_and_sum: {
              const auto total = left + right + carry;
              value = total & 1;
              carry = static_cast<unsigned>(total >> 1);
              break;
            }
          }

          expected[bit / 64] |= value << (bit % 64);
        }

        const auto before = Evaluate(block.nodes(), a, b);
        const auto after = Evaluate(result.block->nodes(), a, b);
        EXPECT_EQ(before, after);
        for (std::size_t i = 0; i < expected.size(); ++i) EXPECT_EQ(after.word(i), expected[i]);
      }
    }
  }
}

TEST(Mba, BooleanIdentitiesDeclineNearMissesThatAreNotTheSameFunction) {
  // Each mutation keeps the shape and breaks exactly one requirement; a rule
  // that fired anyway would silently replace a different function.
  const std::vector<std::pair<BooleanIdentity, std::vector<Node>>> altered{
      {BooleanIdentity::and_xor_union,
       {{Op::read, 64, {}, 0, 100},
        {Op::read, 64, {}, 0, 200},
        {Op::bit_and, 64, {0, 1}},
        {Op::bit_xor, 64, {0, 0}},
        {Op::bit_or, 64, {2, 3}}}},
      {BooleanIdentity::and_xor_union,
       {{Op::read, 64, {}, 0, 100},
        {Op::read, 64, {}, 0, 200},
        {Op::bit_and, 64, {0, 1}},
        {Op::bit_or, 64, {0, 1}},
        {Op::bit_or, 64, {2, 3}}}},
      {BooleanIdentity::or_and_sum,
       {{Op::read, 64, {}, 0, 100},
        {Op::read, 64, {}, 0, 200},
        {Op::bit_or, 64, {0, 1}},
        {Op::bit_and, 64, {1, 1}},
        {Op::add, 64, {2, 3}}}},
      {BooleanIdentity::or_and_sum,
       {{Op::read, 64, {}, 0, 100},
        {Op::read, 64, {}, 0, 200},
        {Op::bit_or, 64, {0, 1}},
        {Op::bit_and, 64, {0, 1}},
        {Op::sub, 64, {2, 3}}}},
      {BooleanIdentity::xor_and_sum,
       {{Op::read, 64, {}, 0, 100},
        {Op::read, 64, {}, 0, 200},
        {Op::bit_xor, 64, {0, 1}},
        {Op::bit_and, 64, {0, 1}},
        {Op::constant, 64, {}, 2},
        {Op::shl, 64, {3, 4}},
        {Op::add, 64, {2, 5}}}},
      {BooleanIdentity::xor_and_sum,
       {{Op::read, 64, {}, 0, 100},
        {Op::read, 64, {}, 0, 200},
        {Op::bit_xor, 64, {0, 1}},
        {Op::bit_or, 64, {0, 1}},
        {Op::constant, 64, {}, 1},
        {Op::shl, 64, {3, 4}},
        {Op::add, 64, {2, 5}}}},
      {BooleanIdentity::xor_ones_not,
       {{Op::read, 64, {}, 0, 100},
        {Op::constant, 64, {}, UINT64_MAX - 1},
        {Op::bit_xor, 64, {0, 1}}}},
      // A 128-bit constant zero-extends its immediate, so this is not all-ones.
      {BooleanIdentity::xor_ones_not,
       {{Op::read, 128, {}, 0, 100},
        {Op::constant, 128, {}, UINT64_MAX},
        {Op::bit_xor, 128, {0, 1}}}},
      // Above 64 bits no literal can be all-ones and no width mask is meaningful,
      // so these must decline on width alone, before any literal is compared.
      {BooleanIdentity::xor_ones_not,
       {{Op::read, 128, {}, 0, 100}, {Op::constant, 128, {}, 0}, {Op::bit_xor, 128, {0, 1}}}},
      {BooleanIdentity::xor_ones_not,
       {{Op::read, 65, {}, 0, 100}, {Op::constant, 65, {}, 1}, {Op::bit_xor, 65, {0, 1}}}},
      // The union must name its own operations. A disjunction in place of the
      // conjunction, and an exclusive-or combining node, both happen to compute
      // the same value here, so only the committed rule can tell them apart.
      {BooleanIdentity::and_xor_union,
       {{Op::read, 64, {}, 0, 100},
        {Op::read, 64, {}, 0, 200},
        {Op::bit_or, 64, {0, 1}},
        {Op::bit_xor, 64, {0, 1}},
        {Op::bit_or, 64, {2, 3}}}},
      {BooleanIdentity::and_xor_union,
       {{Op::read, 64, {}, 0, 100},
        {Op::read, 64, {}, 0, 200},
        {Op::bit_and, 64, {0, 1}},
        {Op::bit_xor, 64, {0, 1}},
        {Op::bit_xor, 64, {2, 3}}}},
      // The carry sum must agree on both operands, not only on the conjunction.
      {BooleanIdentity::xor_and_sum,
       {{Op::read, 64, {}, 0, 100},
        {Op::read, 64, {}, 0, 200},
        {Op::read, 64, {}, 0, 400},
        {Op::bit_xor, 64, {0, 1}},
        {Op::bit_and, 64, {0, 2}},
        {Op::constant, 64, {}, 1},
        {Op::shl, 64, {4, 5}},
        {Op::add, 64, {3, 6}}}},
      // `(a^b) + (a&b)` is not `a+b`, and neither is `(a|b) + (a|b)`; each swaps
      // one operation class of the union-plus-intersection sum.
      {BooleanIdentity::or_and_sum,
       {{Op::read, 64, {}, 0, 100},
        {Op::read, 64, {}, 0, 200},
        {Op::bit_xor, 64, {0, 1}},
        {Op::bit_and, 64, {0, 1}},
        {Op::add, 64, {2, 3}}}},
      {BooleanIdentity::or_and_sum,
       {{Op::read, 64, {}, 0, 100},
        {Op::read, 64, {}, 0, 200},
        {Op::bit_or, 64, {0, 1}},
        {Op::bit_or, 64, {0, 1}},
        {Op::add, 64, {2, 3}}}},
  };

  for (const auto& [identity, nodes] : altered) {
    const auto block = Normalize(nodes);
    auto budget = Plenty();
    const auto result = SimplifyMba(block, budget);
    ASSERT_TRUE(result.block);
    for (const auto& edit : result.journal) EXPECT_NE(edit.rule, RuleOf(identity));
    EXPECT_EQ(result.block->nodes().back().op, nodes.back().op);
  }
}

TEST(Mba, BooleanIdentitiesNeedTheSameDefinitionsNotRepeatedLoads) {
  for (const auto identity : {BooleanIdentity::and_xor_union, BooleanIdentity::or_and_sum}) {
    const auto first = identity == BooleanIdentity::and_xor_union ? Op::bit_and : Op::bit_or;
    const auto second = identity == BooleanIdentity::and_xor_union ? Op::bit_xor : Op::bit_and;
    const auto combine = identity == BooleanIdentity::and_xor_union ? Op::bit_or : Op::add;
    for (const bool shared : {false, true}) {
      // Two loads at one address are distinct definitions; only the shared
      // definition is the same value in both halves of the identity.
      const auto block = Normalize({{Op::constant, 64, {}, 0x4000},
                                    {Op::load, 64, {0}},
                                    {Op::load, 64, {0}},
                                    {Op::read, 64, {}, 0, 200},
                                    {first, 64, {1, 3}},
                                    {second, 64, {shared ? 1U : 2U, 3}},
                                    {combine, 64, {4, 5}}},
                                   ir::MemoryModel::atomic_scalar_reference);
      auto budget = Plenty();
      const auto result = SimplifyMba(block, budget);
      ASSERT_TRUE(result.block);
      EXPECT_EQ(result.block->nodes()[1].op, Op::load);
      EXPECT_EQ(result.block->nodes()[2].op, Op::load);
      if (shared) {
        ASSERT_EQ(result.journal.size(), 1);
        EXPECT_EQ(result.journal[0].rule, RuleOf(identity));
        EXPECT_EQ(result.block->nodes().back().inputs[0], 1);
        EXPECT_EQ(result.block->nodes().back().inputs[1], 3);
      } else {
        EXPECT_TRUE(result.journal.empty());
        EXPECT_EQ(result.block->nodes().back().op, combine);
      }
    }
  }
}

// The value-numbering key must distinguish every component that changes what a
// pure node denotes. One near-miss per component; all of them must decline.
TEST(Mba, EqualValueNumberingSeparatesNodesThatDifferInAnyKeyComponent) {
  struct Case {
    const char* what;
    std::vector<Node> nodes;
    ir::MemoryModel model = ir::MemoryModel::unspecified;
  };

  const std::vector<Case> cases = {
      // operation: one operand pair under two operations.
      {"operation",
       {{Op::read, 64, {}, 0, 100},
        {Op::read, 64, {}, 0, 101},
        {Op::read, 64, {}, 0, 102},
        {Op::bit_and, 64, {0, 1}},
        {Op::bit_or, 64, {0, 1}},
        {Op::bit_and, 64, {3, 2}},
        {Op::bit_xor, 64, {4, 2}},
        {Op::bit_or, 64, {5, 6}}}},
      // width: two slices of one source at different widths, widened back.
      {"width",
       {{Op::read, 64, {}, 0, 100},
        {Op::read, 64, {}, 0, 101},
        {Op::extract, 32, {0}, 0},
        {Op::extract, 16, {0}, 0},
        {Op::zext, 32, {2}},
        {Op::zext, 32, {3}},
        {Op::extract, 32, {1}, 0},
        {Op::bit_and, 32, {4, 6}},
        {Op::bit_xor, 32, {5, 6}},
        {Op::bit_or, 32, {7, 8}}}},
      // immediate: two slices of one source at different offsets.
      {"immediate",
       {{Op::read, 64, {}, 0, 100},
        {Op::read, 64, {}, 0, 101},
        {Op::extract, 32, {0}, 0},
        {Op::extract, 32, {0}, 32},
        {Op::extract, 32, {1}, 0},
        {Op::bit_and, 32, {2, 4}},
        {Op::bit_xor, 32, {3, 4}},
        {Op::bit_or, 32, {5, 6}}}},
      // operands: two sums over different definitions.
      {"operands",
       {{Op::read, 64, {}, 0, 100},
        {Op::read, 64, {}, 0, 101},
        {Op::read, 64, {}, 0, 102},
        {Op::read, 64, {}, 0, 103},
        {Op::add, 64, {0, 1}},
        {Op::add, 64, {2, 3}},
        {Op::bit_and, 64, {4, 0}},
        {Op::bit_xor, 64, {5, 0}},
        {Op::bit_or, 64, {6, 7}}}},
      // effect: two loads at one address are two definitions.
      {"repeated load",
       {{Op::read, 64, {}, 0, 100},
        {Op::load, 64, {0}, 0, 0, {ir::ByteOrder::little, 8}},
        {Op::load, 64, {0}, 0, 0, {ir::ByteOrder::little, 8}},
        {Op::read, 64, {}, 0, 101},
        {Op::bit_and, 64, {1, 3}},
        {Op::bit_xor, 64, {2, 3}},
        {Op::bit_or, 64, {4, 5}}},
       ir::MemoryModel::atomic_scalar_reference},
  };

  for (const auto& test : cases) {
    const auto block = Normalize(test.nodes, test.model);
    auto budget = Plenty();
    const auto result = SimplifyMba(block, budget);
    ASSERT_TRUE(result.block) << test.what;
    EXPECT_TRUE(result.journal.empty()) << test.what;
  }
}

// The extend-narrow pair is the identity only when the extract starts at bit
// zero and the extended source is exactly as wide as the extract. A chain that
// narrows in the middle, or a slice taken above bit zero, is a different value.
TEST(Mba, SeeingThroughTheNormalizationPairStopsAtEveryLossyStep) {
  const std::vector<std::vector<Node>> cases = {
      // extract(zext(extract_16(zext(v)))) is v truncated to 16 bits.
      {{Op::read, 64, {}, 0, 100},
       {Op::extract, 32, {0}, 0},
       {Op::read, 64, {}, 0, 101},
       {Op::extract, 32, {2}, 0},
       {Op::zext, 32, {1}},
       {Op::extract, 16, {4}, 0},
       {Op::zext, 64, {5}},
       {Op::extract, 32, {6}, 0},
       {Op::bit_and, 32, {1, 3}},
       {Op::bit_xor, 32, {7, 3}},
       {Op::bit_or, 32, {8, 9}}},
      // extract(zext(v), 8) is v shifted, not v.
      {{Op::read, 64, {}, 0, 100},
       {Op::extract, 32, {0}, 0},
       {Op::read, 64, {}, 0, 101},
       {Op::extract, 32, {2}, 0},
       {Op::zext, 40, {1}},
       {Op::extract, 32, {4}, 8},
       {Op::bit_and, 32, {1, 3}},
       {Op::bit_xor, 32, {5, 3}},
       {Op::bit_or, 32, {6, 7}}},
  };

  for (const auto& nodes : cases) {
    const auto block = Normalize(nodes);
    auto budget = Plenty();
    const auto result = SimplifyMba(block, budget);
    ASSERT_TRUE(result.block);
    EXPECT_TRUE(result.journal.empty());
  }
}

// A register read is relative to its own instruction's entry state. Normalize
// forwards the second read away, so the block is built directly: the region
// contract admits it and the rules must still keep the two reads apart.
TEST(Mba, RegisterReadsAcrossAWriteAreNotOneValue) {
  const std::vector<Node> first{{Op::read, 64, {}, 0, 100},
                                {Op::read, 64, {}, 0, 101},
                                {Op::bit_xor, 64, {0, 1}},
                                {Op::write, 64, {2}, 0, 100}};
  const std::vector<Node> second{{Op::read, 64, {}, 0, 100},
                                 {Op::read, 64, {}, 0, 102},
                                 {Op::bit_and, 64, {0, 1}},
                                 {Op::bit_xor, 64, {0, 1}},
                                 {Op::bit_or, 64, {2, 3}}};
  const std::vector<ir::Group> sources{ir::Group(0x1000, {1, 2, 3, 4}, first, {}),
                                       ir::Group(0x1004, {5, 6, 7, 8}, second, {})};
  const std::vector<Node> nodes{first[0],
                                first[1],
                                first[2],
                                first[3],
                                {Op::read, 64, {}, 0, 100},
                                {Op::read, 64, {}, 0, 102},
                                {Op::bit_and, 64, {4, 5}},
                                {Op::bit_xor, 64, {0, 5}},
                                {Op::bit_or, 64, {6, 7}}};
  const std::vector<ir::Origin> origins{{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 0},
                                        {1, 1}, {1, 2}, {1, 3}, {1, 4}};
  std::vector<ir::Boundary> boundaries{{0, 4, {}, {}}, {4, 5, {}, {}}};
  const ir::Block block(sources, nodes, origins, std::move(boundaries), 0);
  auto budget = Plenty();
  ASSERT_EQ(ir::Validate(block, budget), ir::BlockDecline::none);
  auto second_budget = Plenty();
  const auto result = SimplifyMba(block, second_budget);
  ASSERT_TRUE(result.block);
  EXPECT_TRUE(result.journal.empty());
}

// The capability the change exists for: the carry sum whose operands each reach
// the identity through their own extend-narrow pair, and an arity-1 pair whose
// trailing inputs differ because normalization never rewrites them.
TEST(Mba, EqualValuesUnderSeveralIdentifiersStillMatchTheIdentity) {
  const std::vector<Node> carry{
      {Op::read, 64, {}, 0, 100}, {Op::read, 64, {}, 0, 101}, {Op::extract, 32, {0}, 0},
      {Op::extract, 32, {1}, 0},  {Op::zext, 64, {2}},        {Op::extract, 32, {4}, 0},
      {Op::zext, 64, {3}},        {Op::extract, 32, {6}, 0},  {Op::bit_xor, 32, {2, 3}},
      {Op::bit_and, 32, {5, 7}},  {Op::constant, 32, {}, 1},  {Op::shl, 32, {9, 10}},
      {Op::zext, 64, {8}},        {Op::extract, 32, {12}, 0}, {Op::zext, 64, {11}},
      {Op::extract, 32, {14}, 0}, {Op::add, 32, {13, 15}}};
  const auto block = Normalize(carry);
  auto budget = Plenty();
  const auto result = SimplifyMba(block, budget);
  ASSERT_TRUE(result.block);
  ASSERT_EQ(result.journal.size(), 1);
  EXPECT_EQ(result.journal[0].rule, MbaRule::xor_and_sum);
  EXPECT_EQ(result.block->nodes()[16].op, Op::add);

  const std::vector<Node> stale{{Op::read, 64, {}, 0, 100}, {Op::read, 64, {}, 0, 101},
                                {Op::bit_not, 64, {0, 0}},  {Op::bit_not, 64, {0, 1}},
                                {Op::bit_and, 64, {2, 1}},  {Op::bit_xor, 64, {3, 1}},
                                {Op::bit_or, 64, {4, 5}}};
  const auto stale_block = Normalize(stale);
  auto stale_budget = Plenty();
  const auto stale_result = SimplifyMba(stale_block, stale_budget);
  ASSERT_TRUE(stale_result.block);
  ASSERT_EQ(stale_result.journal.size(), 1);
  EXPECT_EQ(stale_result.journal[0].rule, MbaRule::and_xor_union);
}

TEST(Mba, OneIdentityRewritesTheOperandAnotherIdentityStillMatchesOn) {
  // `(a & ~b) | (a ^ ~b)` with `~b` written as `b ^ ~0`. Rewriting the inner
  // complement first must not hide the union, whose match is on operand identity.
  const auto block = Normalize({{Op::read, 64, {}, 0, 100},
                                {Op::read, 64, {}, 0, 200},
                                {Op::constant, 64, {}, UINT64_MAX},
                                {Op::bit_xor, 64, {1, 2}},
                                {Op::bit_and, 64, {0, 3}},
                                {Op::bit_xor, 64, {3, 0}},
                                {Op::bit_or, 64, {5, 4}}});
  auto budget = Plenty();
  const auto result = SimplifyMba(block, budget);
  ASSERT_TRUE(result.block);
  ASSERT_EQ(result.journal.size(), 2);
  EXPECT_EQ(result.journal[0].rule, MbaRule::xor_ones_not);
  EXPECT_EQ(result.journal[0].node, 3);
  EXPECT_EQ(result.journal[1].rule, MbaRule::and_xor_union);
  EXPECT_EQ(result.journal[1].node, 6);
  EXPECT_EQ(result.block->nodes()[3].op, Op::bit_not);
  EXPECT_EQ(result.block->nodes()[6].op, Op::bit_or);
  EXPECT_EQ(result.block->nodes()[6].inputs[0], 0);
  EXPECT_EQ(result.block->nodes()[6].inputs[1], 3);

  // A batch of edits is one committed revision, not one per rewrite.
  EXPECT_EQ(result.block->revision(), 1);
  for (const auto& edit : result.journal) {
    EXPECT_EQ(edit.from_revision, 0);
    EXPECT_EQ(edit.to_revision, 1);
  }
}

TEST(Mba, SumIdentitiesExposeTheNegatedOperandTheNextRuleRemoves) {
  // Both sum identities publish an addition whose operand is `0 - a`, which the
  // existing negation rule then commits as a subtraction at the same node. This
  // composition is what turns the development range's hot chains into `b - a`.
  for (const auto identity : {BooleanIdentity::or_and_sum, BooleanIdentity::xor_and_sum}) {
    std::vector<Node> nodes{{Op::read, 64, {}, 0, 100},
                            {Op::constant, 64, {}, 0},
                            {Op::sub, 64, {1, 0}},
                            {Op::read, 64, {}, 0, 200}};
    if (identity == BooleanIdentity::or_and_sum) {
      nodes.push_back({Op::bit_or, 64, {2, 3}});
      nodes.push_back({Op::bit_and, 64, {2, 3}});
      nodes.push_back({Op::add, 64, {4, 5}});
    } else {
      nodes.push_back({Op::bit_xor, 64, {2, 3}});
      nodes.push_back({Op::bit_and, 64, {2, 3}});
      nodes.push_back({Op::constant, 64, {}, 1});
      nodes.push_back({Op::shl, 64, {5, 6}});
      nodes.push_back({Op::add, 64, {4, 7}});
    }

    const auto last = static_cast<ir::ValueId>(nodes.size() - 1);
    const auto block = Normalize(nodes);
    auto budget = Plenty();
    const auto result = SimplifyMba(block, budget);
    ASSERT_TRUE(result.block);
    ASSERT_EQ(result.journal.size(), 2);
    EXPECT_EQ(result.journal[0].rule, RuleOf(identity));
    EXPECT_EQ(result.journal[0].node, last);
    EXPECT_EQ(result.journal[1].rule, MbaRule::negated_add);
    EXPECT_EQ(result.journal[1].node, last);
    EXPECT_EQ(result.block->nodes()[last].op, Op::sub);
    EXPECT_EQ(result.block->nodes()[last].inputs[0], 3);
    EXPECT_EQ(result.block->nodes()[last].inputs[1], 0);
    EXPECT_EQ(result.block->revision(), 1);
    std::mt19937_64 random(0x5ec0d + static_cast<unsigned>(identity));
    for (unsigned test = 0; test < 32; ++test) {
      const std::vector<std::uint64_t> a{random()}, b{random()};
      EXPECT_EQ(Evaluate(block.nodes(), a, b), Evaluate(result.block->nodes(), a, b));
      EXPECT_EQ(Evaluate(result.block->nodes(), a, b).word(0), b[0] - a[0]);
    }
  }
}

TEST(Mba, DeferredIdentitiesDoNotCostTheOuterIdentityItsMatch) {
  // Both fixtures nest an identity inside the doubled-union form. The inner
  // rewrite changes its node's operation, which is the shape the outer identity
  // matches on, so applying it first would silently lose the outer edit.
  const std::vector<std::pair<MbaRule, std::vector<Node>>> nested{
      // ((p|q)<<1) - (p^q), with p = u&v and q = u^v, so a union nests inside.
      {MbaRule::and_xor_union,
       {{Op::read, 64, {}, 0, 100},
        {Op::read, 64, {}, 0, 200},
        {Op::bit_and, 64, {0, 1}},
        {Op::bit_xor, 64, {0, 1}},
        {Op::bit_or, 64, {2, 3}},
        {Op::constant, 64, {}, 1},
        {Op::shl, 64, {4, 5}},
        {Op::bit_xor, 64, {2, 3}},
        {Op::sub, 64, {6, 7}}}},
      // ((a|K)<<1) - (a^K), with K the materialized all-ones literal.
      {MbaRule::xor_ones_not,
       {{Op::read, 64, {}, 0, 100},
        {Op::constant, 64, {}, UINT64_MAX},
        {Op::bit_or, 64, {0, 1}},
        {Op::constant, 64, {}, 1},
        {Op::shl, 64, {2, 3}},
        {Op::bit_xor, 64, {0, 1}},
        {Op::sub, 64, {4, 5}}}},
  };

  for (const auto& [deferred, nodes] : nested) {
    const auto block = Normalize(nodes);
    auto budget = Plenty();
    const auto result = SimplifyMba(block, budget);
    ASSERT_TRUE(result.block);
    std::vector<MbaRule> committed;
    for (const auto& edit : result.journal) committed.push_back(edit.rule);
    EXPECT_NE(std::find(committed.begin(), committed.end(), MbaRule::or_xor_sum), committed.end());
    EXPECT_NE(std::find(committed.begin(), committed.end(), deferred), committed.end());
    EXPECT_EQ(result.block->nodes().back().op, Op::add);
    EXPECT_EQ(result.block->revision(), 1);
    std::mt19937_64 random(0xdefe44ed);
    for (unsigned test = 0; test < 32; ++test) {
      const std::vector<std::uint64_t> a{random()}, b{random()};
      EXPECT_EQ(Evaluate(block.nodes(), a, b), Evaluate(result.block->nodes(), a, b));
    }
  }
}

TEST(Mba, LiteralOperandsAreFoldedBeforeAnIdentityCanSeeTheirShape) {
  // Folding runs on the operands first, so the union never matches here. The
  // published value must still be the one the obfuscated form computed.
  const auto block = Normalize({{Op::constant, 32, {}, 0xf0f0f0f0},
                                {Op::constant, 32, {}, 0x00ffff00},
                                {Op::bit_and, 32, {0, 1}},
                                {Op::bit_xor, 32, {0, 1}},
                                {Op::bit_or, 32, {2, 3}}});
  auto budget = Plenty();
  const auto result = SimplifyMba(block, budget);
  ASSERT_TRUE(result.block);
  ASSERT_EQ(result.journal.size(), 3);
  for (const auto& edit : result.journal) EXPECT_EQ(edit.rule, MbaRule::constant_fold);
  EXPECT_EQ(result.block->nodes().back().op, Op::constant);
  EXPECT_EQ(result.block->nodes().back().immediate, 0xf0f0f0f0 | 0x00ffff00);
  EXPECT_EQ(result.block->revision(), 1);
}

TEST(Mba, ConstantFoldingRespectsWidthsSignednessAndLargeCounts) {
  for (unsigned width : {1U, 7U, 32U, 64U}) {
    const auto mask = width == 64 ? UINT64_MAX : (UINT64_C(1) << width) - 1;
    for (auto op : {Op::shl, Op::lshr, Op::ashr}) {
      for (std::uint64_t count : {UINT64_C(0), UINT64_C(1), UINT64_C(63), UINT64_MAX}) {
        const auto block = Normalize(
            {{Op::constant, width, {}, mask}, {Op::constant, 64, {}, count}, {op, width, {0, 1}}});
        auto budget = Plenty();
        auto result = SimplifyMba(block, budget);
        ASSERT_TRUE(result.block);
        const auto& folded = result.block->nodes().back();
        ASSERT_EQ(folded.op, Op::constant);
        const auto expected = op == Op::ashr   ? mask
                              : count >= width ? 0
                              : op == Op::shl  ? (mask << count) & mask
                                               : mask >> count;
        EXPECT_EQ(folded.immediate, expected);
      }
    }
  }
}

TEST(Mba, ConstantArithmeticAndComparisonsMatchExhaustiveSmallIntegers) {
  for (unsigned width = 1; width <= 4; ++width) {
    const unsigned modulus = 1U << width;
    for (unsigned a = 0; a < modulus; ++a) {
      for (unsigned b = 0; b < modulus; ++b) {
        for (const auto op : {Op::add, Op::sub, Op::mul, Op::bit_and, Op::bit_or, Op::bit_xor,
                              Op::equal, Op::unsigned_less, Op::signed_less}) {
          const bool compare = op == Op::equal || op == Op::unsigned_less || op == Op::signed_less;
          const auto block = Normalize({{Op::constant, width, {}, a},
                                        {Op::constant, width, {}, b},
                                        {op, compare ? 1U : width, {0, 1}}});
          auto budget = Plenty();
          const auto result = SimplifyMba(block, budget);
          ASSERT_TRUE(result.block);
          ASSERT_EQ(result.block->nodes().back().op, Op::constant);
          const auto signed_a =
              static_cast<int>(a) - (a >= modulus / 2 ? static_cast<int>(modulus) : 0);
          const auto signed_b =
              static_cast<int>(b) - (b >= modulus / 2 ? static_cast<int>(modulus) : 0);
          unsigned expected = 0;
          switch (op) {
            case Op::add:
              expected = a + b;
              break;
            case Op::sub:
              expected = a + modulus - b;
              break;
            case Op::mul:
              expected = a * b;
              break;
            case Op::bit_and:
              expected = a & b;
              break;
            case Op::bit_or:
              expected = a | b;
              break;
            case Op::bit_xor:
              expected = a ^ b;
              break;
            case Op::equal:
              expected = a == b;
              break;
            case Op::unsigned_less:
              expected = a < b;
              break;
            case Op::signed_less:
              expected = signed_a < signed_b;
              break;
            default:
              FAIL();
          }

          EXPECT_EQ(result.block->nodes().back().immediate, expected % modulus);
        }
      }
    }
  }
}

TEST(Mba, ConstantNotDiscardsBitsAboveItsWidth) {
  for (unsigned width : {1U, 7U, 32U, 64U}) {
    const auto mask = width == 64 ? UINT64_MAX : (UINT64_C(1) << width) - 1;
    for (auto literal : {UINT64_C(0), UINT64_MAX, UINT64_C(0xfedcba9876543210)}) {
      const auto block = Normalize({{Op::constant, width, {}, literal}, {Op::bit_not, width, {0}}});
      auto budget = Plenty();
      const auto result = SimplifyMba(block, budget);
      ASSERT_TRUE(result.block);
      ASSERT_EQ(result.journal.size(), 1);
      EXPECT_EQ(result.journal[0].rule, MbaRule::constant_fold);
      EXPECT_EQ(result.block->nodes().back().op, Op::constant);
      EXPECT_EQ(result.block->nodes().back().immediate, mask ^ (literal & mask));
    }
  }
}

TEST(Mba, ConstantExtractSelectsOnlyTheRequestedBits) {
  constexpr auto literal = UINT64_C(0xfedcba9876543210);
  for (unsigned input_width : {1U, 7U, 32U, 64U}) {
    for (unsigned low = 0; low < input_width; ++low) {
      for (unsigned width : {1U, input_width - low}) {
        const auto block =
            Normalize({{Op::constant, input_width, {}, literal}, {Op::extract, width, {0}, low}});
        auto budget = Plenty();
        const auto result = SimplifyMba(block, budget);
        ASSERT_TRUE(result.block);
        ASSERT_EQ(result.block->nodes().back().op, Op::constant);
        std::uint64_t expected = 0;
        for (unsigned bit = 0; bit < width; ++bit) {
          if ((literal >> (low + bit)) & 1) expected |= UINT64_C(1) << bit;
        }

        EXPECT_EQ(result.block->nodes().back().immediate, expected);
        EXPECT_EQ(result.block->nodes().back().width, width);
      }
    }
  }
}

TEST(Mba, ConstantZeroExtensionMasksTheSourceBeforeExtending) {
  for (unsigned input_width : {1U, 7U, 32U, 64U}) {
    for (unsigned output_width : {input_width, 64U}) {
      const auto block =
          Normalize({{Op::constant, input_width, {}, UINT64_MAX}, {Op::zext, output_width, {0}}});
      auto budget = Plenty();
      const auto result = SimplifyMba(block, budget);
      ASSERT_TRUE(result.block);
      ASSERT_EQ(result.block->nodes().back().op, Op::constant);
      EXPECT_EQ(result.block->nodes().back().width, output_width);
      EXPECT_EQ(result.block->nodes().back().immediate,
                input_width == 64 ? UINT64_MAX : (UINT64_C(1) << input_width) - 1);
    }
  }
}

TEST(Mba, ConstantSelectUsesOnlyTheOneBitConditionAndTruncatesTheSelectedValue) {
  constexpr auto when_true = UINT64_C(0xfedcba9876543210);
  constexpr auto when_false = UINT64_C(0x0123456789abcdef);
  for (unsigned width : {1U, 7U, 32U, 64U}) {
    const auto mask = width == 64 ? UINT64_MAX : (UINT64_C(1) << width) - 1;
    for (auto condition : {UINT64_C(0), UINT64_C(1), UINT64_MAX - 1, UINT64_MAX}) {
      const auto block = Normalize({{Op::constant, 1, {}, condition},
                                    {Op::constant, width, {}, when_true},
                                    {Op::constant, width, {}, when_false},
                                    {Op::select, width, {0, 1, 2}}});
      auto budget = Plenty();
      const auto result = SimplifyMba(block, budget);
      ASSERT_TRUE(result.block);
      ASSERT_EQ(result.block->nodes().back().op, Op::constant);
      EXPECT_EQ(result.block->nodes().back().immediate,
                ((condition & 1) ? when_true : when_false) & mask);
    }
  }
}

TEST(Mba, EveryBudgetCutDiscardsAllEditsWithoutChangingInput) {
  const auto block = Identity(64);
  auto full = Plenty();
  ASSERT_TRUE(SimplifyMba(block, full).block);
  for (bool bytes : {false, true}) {
    const auto bound = bytes ? full.used().bytes : full.used().work;
    for (std::uint64_t cut = 0; cut < bound; ++cut) {
      Budget budget({bytes ? full.used().work : cut, bytes ? cut : full.used().bytes});
      const auto result = SimplifyMba(block, budget);
      ASSERT_FALSE(result.block) << cut;
      EXPECT_TRUE(result.journal.empty());
      EXPECT_EQ(result.reason, MbaDecline::resource_limit);
      EXPECT_EQ(block.nodes()[6].op, Op::sub);
      EXPECT_EQ(block.revision(), 0);
    }
  }

  auto budget = Plenty();
  MbaLimits limits;
  limits.max_edits = 0;
  const auto result = SimplifyMba(block, budget, limits);
  EXPECT_FALSE(result.block);
  EXPECT_EQ(result.reason, MbaDecline::resource_limit);
}

// The value the linear identity rule gave the final equality, if it gave one.
std::optional<std::uint64_t> Decided(std::vector<Node> nodes) {
  const auto block = Normalize(std::move(nodes));
  auto budget = Plenty();
  const auto result = SimplifyMba(block, budget);
  EXPECT_TRUE(result.block);
  const auto last = static_cast<ir::ValueId>(block.nodes().size() - 1);
  for (const auto& edit : result.journal) {
    if (edit.rule == MbaRule::linear_identity && edit.node == last)
      return edit.replacement.immediate;
  }

  return {};
}

TEST(Mba, LinearIdentityDecidesTheOpaquePredicatesOfTheDevelopmentRange) {
  // 0x15a7fd8: x * 806 against (x & s) * 1612 + ((x ^ s) - s) * 806.
  EXPECT_EQ(Decided({{Op::read, 64, {}, 0, 100},
                     {Op::read, 64, {}, 0, 200},
                     {Op::constant, 64, {}, 806},
                     {Op::mul, 64, {0, 2}},
                     {Op::bit_and, 64, {0, 1}},
                     {Op::constant, 64, {}, 1612},
                     {Op::mul, 64, {4, 5}},
                     {Op::bit_xor, 64, {0, 1}},
                     {Op::sub, 64, {7, 1}},
                     {Op::mul, 64, {8, 2}},
                     {Op::add, 64, {6, 9}},
                     {Op::equal, 1, {3, 10}}}),
            1);
  // 0x15a8228: (~s & b) * -676 against (s - (s | b)) * 676.
  EXPECT_EQ(Decided({{Op::read, 64, {}, 0, 100},
                     {Op::read, 64, {}, 0, 200},
                     {Op::bit_not, 64, {0}},
                     {Op::bit_and, 64, {2, 1}},
                     {Op::constant, 64, {}, 0 - UINT64_C(676)},
                     {Op::mul, 64, {3, 4}},
                     {Op::bit_or, 64, {0, 1}},
                     {Op::sub, 64, {0, 6}},
                     {Op::constant, 64, {}, 676},
                     {Op::mul, 64, {7, 8}},
                     {Op::equal, 1, {5, 9}}}),
            1);
}

TEST(Mba, LinearIdentityTreatsASignExtensionAsOneValue) {
  // sxtw as the decoder writes it: the high word selected by bit 31, the low
  // word masked. The mask is no uniform function of its bits, so the whole
  // extension is one free value, which is all the identity needs.
  const std::vector<Node> extension{{Op::read, 64, {}, 0, 100},
                                    {Op::read, 64, {}, 0, 200},
                                    {Op::extract, 1, {1}, 31},
                                    {Op::constant, 64, {}, 0xffffffff00000000},
                                    {Op::constant, 64, {}, 0},
                                    {Op::select, 64, {2, 3, 4}},
                                    {Op::constant, 64, {}, 0xffffffff},
                                    {Op::bit_and, 64, {1, 6}},
                                    {Op::bit_or, 64, {5, 7}}};
  auto nodes = extension;
  nodes.insert(nodes.end(), {{Op::bit_xor, 64, {0, 8}},
                             {Op::bit_and, 64, {0, 8}},
                             {Op::constant, 64, {}, 1},
                             {Op::shl, 64, {10, 11}},
                             {Op::add, 64, {9, 12}},
                             {Op::add, 64, {0, 8}},
                             {Op::equal, 1, {13, 14}}});
  EXPECT_EQ(Decided(nodes), 1);
}

TEST(Mba, LinearIdentityDecidesAConstantDifferenceFalse) {
  EXPECT_EQ(Decided({{Op::read, 32, {}, 0, 100},
                     {Op::constant, 32, {}, 5},
                     {Op::add, 32, {0, 1}},
                     {Op::equal, 1, {2, 0}}}),
            0);
}

TEST(Mba, LinearIdentityLeavesEverythingElseAlone) {
  // A coefficient off by one.
  EXPECT_FALSE(Decided({{Op::read, 64, {}, 0, 100},
                        {Op::constant, 64, {}, 806},
                        {Op::mul, 64, {0, 1}},
                        {Op::constant, 64, {}, 807},
                        {Op::mul, 64, {0, 3}},
                        {Op::equal, 1, {2, 4}}}));
  // A product of two values is not linear; it is one free value, never zero.
  EXPECT_FALSE(Decided({{Op::read, 64, {}, 0, 100},
                        {Op::read, 64, {}, 0, 200},
                        {Op::mul, 64, {0, 1}},
                        {Op::constant, 64, {}, 0},
                        {Op::equal, 1, {2, 3}}}));
  // Masks that differ bit to bit: taking either for all-ones would call them equal.
  EXPECT_FALSE(Decided({{Op::read, 64, {}, 0, 100},
                        {Op::constant, 64, {}, 0xff},
                        {Op::bit_and, 64, {0, 1}},
                        {Op::constant, 64, {}, 0xff00},
                        {Op::bit_and, 64, {0, 3}},
                        {Op::equal, 1, {2, 4}}}));
  // A two-bit shift amount holding 5 shifts by 1, not 5: x << 1 is not x * 32.
  EXPECT_FALSE(Decided({{Op::read, 8, {}, 0, 100},
                        {Op::constant, 2, {}, 5},
                        {Op::shl, 8, {0, 1}},
                        {Op::constant, 8, {}, 32},
                        {Op::mul, 8, {0, 3}},
                        {Op::equal, 1, {2, 4}}}));
  EXPECT_EQ(Decided({{Op::read, 8, {}, 0, 100},
                     {Op::constant, 2, {}, 5},
                     {Op::shl, 8, {0, 1}},
                     {Op::constant, 8, {}, 2},
                     {Op::mul, 8, {0, 3}},
                     {Op::equal, 1, {2, 4}}}),
            1);
  // Seven free values, one more than the rule enumerates.
  std::vector<Node> many;
  for (unsigned i = 0; i < 7; ++i) many.push_back({Op::read, 64, {}, 0, 100 + i});
  auto sum = static_cast<ir::ValueId>(0);
  for (unsigned i = 1; i < 7; ++i) {
    many.push_back({Op::add, 64, {sum, i}});
    sum = static_cast<ir::ValueId>(many.size() - 1);
  }

  many.push_back({Op::equal, 1, {sum, sum}});
  EXPECT_FALSE(Decided(many));
}

// Random sums over three values: whatever the rule decides must hold at random
// 64-bit inputs, and every sum built to cancel must be decided true.
TEST(Mba, LinearIdentityDecisionsHoldOnRandomInputs) {
  std::mt19937_64 random(0x5eed);
  const auto evaluate = [](std::span<const Node> nodes,
                           const std::array<std::uint64_t, 3>& inputs) {
    std::vector<std::uint64_t> values;
    for (const auto& node : nodes) {
      const auto a = node.inputs[0] < values.size() ? values[node.inputs[0]] : 0;
      const auto b = node.inputs[1] < values.size() ? values[node.inputs[1]] : 0;
      switch (node.op) {
        case Op::read:
          values.push_back(inputs[node.storage - 100]);
          break;
        case Op::constant:
          values.push_back(node.immediate);
          break;
        case Op::add:
          values.push_back(a + b);
          break;
        case Op::sub:
          values.push_back(a - b);
          break;
        case Op::mul:
          values.push_back(a * b);
          break;
        case Op::bit_and:
          values.push_back(a & b);
          break;
        case Op::bit_or:
          values.push_back(a | b);
          break;
        case Op::bit_xor:
          values.push_back(a ^ b);
          break;
        case Op::bit_not:
          values.push_back(~a);
          break;
        case Op::shl:
          values.push_back(b < 64 ? a << b : 0);
          break;
        case Op::equal:
          values.push_back(a == b);
          break;
        default:
          ADD_FAILURE();
          values.push_back(0);
      }
    }

    return values.back();
  };

  unsigned decided = 0;
  for (unsigned trial = 0; trial < 400; ++trial) {
    std::vector<Node> nodes{
        {Op::read, 64, {}, 0, 100}, {Op::read, 64, {}, 0, 101}, {Op::read, 64, {}, 0, 102}};
    const auto pick = [&] { return static_cast<ir::ValueId>(random() % nodes.size()); };
    for (unsigned i = 0; i < 6; ++i) {
      // Products and shifts of two values are not linear and must stay free values.
      const std::array ops{Op::bit_and, Op::bit_or, Op::bit_xor, Op::bit_not,
                           Op::add,     Op::sub,    Op::mul,     Op::shl};
      const auto op = ops[random() % ops.size()];
      nodes.push_back({op, 64, {pick(), pick()}});
    }

    const auto left = static_cast<ir::ValueId>(nodes.size() - 1);
    nodes.push_back({Op::constant, 64, {}, random() % 5});
    nodes.push_back({Op::mul, 64, {left, static_cast<ir::ValueId>(nodes.size() - 1)}});

    // Half the trials compare against the same sum rebuilt, which must cancel.
    const auto right = trial % 2 ? static_cast<ir::ValueId>(nodes.size() - 1) : pick();
    nodes.push_back({Op::equal, 1, {static_cast<ir::ValueId>(nodes.size() - 1), right}});
    const auto verdict = Decided(nodes);
    if (trial % 2 && nodes.size() <= 13) {
      EXPECT_EQ(verdict, 1) << trial;
    }

    if (!verdict) continue;
    ++decided;
    for (unsigned sample = 0; sample < 64; ++sample) {
      ASSERT_EQ(evaluate(nodes, {random(), random(), random()}), *verdict) << trial;
    }
  }

  EXPECT_GT(decided, 200U);
}

TEST(Mba, LinearSignatureRewritesBooleanAndArithmeticXor) {
  std::mt19937_64 random(0x6770);
  const std::array<std::vector<Node>, 2> examples{{{{Op::read, 64, {}, 0, 100},
                                                    {Op::read, 64, {}, 0, 200},
                                                    {Op::bit_not, 64, {0}},
                                                    {Op::bit_and, 64, {2, 1}},
                                                    {Op::bit_not, 64, {1}},
                                                    {Op::bit_and, 64, {0, 4}},
                                                    {Op::bit_or, 64, {3, 5}}},
                                                   {{Op::read, 64, {}, 0, 100},
                                                    {Op::read, 64, {}, 0, 200},
                                                    {Op::add, 64, {0, 1}},
                                                    {Op::bit_and, 64, {0, 1}},
                                                    {Op::constant, 64, {}, 2},
                                                    {Op::mul, 64, {3, 4}},
                                                    {Op::sub, 64, {2, 5}}}}};
  for (const auto& nodes : examples) {
    const auto block = Normalize(nodes);
    auto budget = Plenty();
    const auto result = SimplifyLinearMba(block, budget);
    ASSERT_TRUE(result.block);
    ASSERT_FALSE(result.journal.empty());
    EXPECT_EQ(result.journal.back().rule, MbaRule::linear_direct);
    EXPECT_EQ(result.block->nodes().back().op, Op::bit_xor);
    for (unsigned sample = 0; sample < 1000; ++sample) {
      const std::array a{random()}, b{random()};
      EXPECT_EQ(Evaluate(block.nodes(), a, b), Evaluate(result.block->nodes(), a, b));
    }

    std::vector<Node> wrong(result.block->nodes().begin(), result.block->nodes().end());
    wrong.back().op = Op::bit_or;
    const std::array one{UINT64_C(1)};
    EXPECT_NE(Evaluate(block.nodes(), one, one), Evaluate(wrong, one, one));
  }
}

TEST(Mba, LinearSignatureUsesExistingBitwiseSubexpression) {
  std::mt19937_64 random(0x6b20);
  const auto block = Normalize({{Op::read, 64, {}, 0, 100},
                                {Op::read, 64, {}, 0, 200},
                                {Op::read, 64, {}, 0, 300},
                                {Op::bit_xor, 64, {0, 1}},
                                {Op::bit_not, 64, {2}},
                                {Op::bit_and, 64, {3, 4}},
                                {Op::bit_not, 64, {3}},
                                {Op::bit_and, 64, {6, 2}},
                                {Op::bit_or, 64, {5, 7}}});
  auto budget = Plenty();
  const auto result = SimplifyLinearMba(block, budget);
  ASSERT_TRUE(result.block);
  ASSERT_EQ(result.journal.size(), 1);
  EXPECT_EQ(result.journal[0].node, 8);
  const auto& simplified = result.block->nodes().back();
  EXPECT_EQ(simplified.op, Op::bit_xor);
  EXPECT_EQ(simplified.inputs[0], 2);
  EXPECT_EQ(simplified.inputs[1], 3);
  for (unsigned sample = 0; sample < 1000; ++sample) {
    const std::array a{random()}, b{random()}, c{random()};
    auto original = block.nodes();
    auto changed = result.block->nodes();

    // The third source is independent of the first two in this fixture.
    const auto evaluate = [&](std::span<const Node> nodes) {
      std::vector<Node> copy(nodes.begin(), nodes.end());
      copy[2] = {Op::constant, 64, {}, c[0]};
      return Evaluate(copy, a, b);
    };

    EXPECT_EQ(evaluate(original), evaluate(changed));
  }

  std::vector<Node> wrong(result.block->nodes().begin(), result.block->nodes().end());
  wrong[2] = {Op::constant, 64, {}, 1};
  wrong.back().op = Op::bit_or;
  const std::array one{UINT64_C(1)}, zero{UINT64_C(0)};
  std::vector<Node> original(block.nodes().begin(), block.nodes().end());
  original[2] = wrong[2];
  EXPECT_NE(Evaluate(original, one, zero), Evaluate(wrong, one, zero));
  Budget short_search({30000, 100000000});
  const auto declined = SimplifyLinearMba(block, short_search);
  EXPECT_FALSE(declined.block);
  EXPECT_TRUE(declined.journal.empty());
  EXPECT_EQ(declined.reason, MbaDecline::resource_limit);
}

TEST(Mba, LinearSignatureLeavesAlreadyMinimalThreeInputFormAlone) {
  const auto block = Normalize({{Op::read, 64, {}, 0, 100},
                                {Op::read, 64, {}, 0, 200},
                                {Op::read, 64, {}, 0, 300},
                                {Op::bit_xor, 64, {0, 1}},
                                {Op::bit_xor, 64, {3, 2}}});
  auto budget = Plenty();
  const auto result = SimplifyLinearMba(block, budget);
  ASSERT_TRUE(result.block);
  EXPECT_TRUE(result.journal.empty());
  EXPECT_EQ(result.block->revision(), block.revision());
  EXPECT_EQ(result.block->nodes().back().inputs, block.nodes().back().inputs);
}

TEST(Mba, LinearSignatureThreeInputIdentityAtNarrowWidths) {
  std::mt19937_64 random(0x6b21);
  for (unsigned width : {1U, 8U, 32U}) {
    const auto block = Normalize({{Op::read, width, {}, 0, 100},
                                  {Op::read, width, {}, 0, 200},
                                  {Op::read, width, {}, 0, 300},
                                  {Op::bit_xor, width, {0, 1}},
                                  {Op::bit_not, width, {2}},
                                  {Op::bit_and, width, {3, 4}},
                                  {Op::bit_not, width, {3}},
                                  {Op::bit_and, width, {6, 2}},
                                  {Op::bit_or, width, {5, 7}}});
    auto budget = Plenty();
    const auto result = SimplifyLinearMba(block, budget);
    ASSERT_TRUE(result.block);
    EXPECT_EQ(result.block->nodes().back().op, Op::bit_xor);
    for (unsigned sample = 0; sample < 100; ++sample) {
      const std::array a{random()}, b{random()};
      const auto c = random();
      std::vector<Node> original(block.nodes().begin(), block.nodes().end());
      std::vector<Node> changed(result.block->nodes().begin(), result.block->nodes().end());
      original[2] = {Op::constant, width, {}, c};
      changed[2] = original[2];
      EXPECT_EQ(Evaluate(original, a, b), Evaluate(changed, a, b));
    }
  }
}

TEST(Mba, LinearSignatureBudgetCutDoesNotPublishPartialEdit) {
  const auto block = Normalize({{Op::read, 64, {}, 0, 100},
                                {Op::read, 64, {}, 0, 200},
                                {Op::bit_not, 64, {0}},
                                {Op::bit_and, 64, {2, 1}},
                                {Op::bit_not, 64, {1}},
                                {Op::bit_and, 64, {0, 4}},
                                {Op::bit_or, 64, {3, 5}}});
  auto full = Plenty();
  const auto accepted = SimplifyLinearMba(block, full);
  ASSERT_TRUE(accepted.block);
  ASSERT_FALSE(accepted.journal.empty());
  const auto used = full.used();
  for (const auto limit :
       {Resources{used.work - 1, used.bytes}, Resources{used.work, used.bytes - 1}}) {
    Budget cut(limit);
    const auto refused = SimplifyLinearMba(block, cut);
    EXPECT_FALSE(refused.block);
    EXPECT_TRUE(refused.journal.empty());
    EXPECT_EQ(refused.reason, MbaDecline::resource_limit);
    EXPECT_EQ(block.revision(), 0);
  }
}

TEST(Mba, LinearSignatureRefusesNonuniformMasksAndNonlinearTerms) {
  const std::array<std::vector<Node>, 2> examples{{{{Op::read, 64, {}, 0, 100},
                                                    {Op::constant, 64, {}, 0xff},
                                                    {Op::bit_and, 64, {0, 1}},
                                                    {Op::bit_or, 64, {2, 0}}},
                                                   {{Op::read, 64, {}, 0, 100},
                                                    {Op::read, 64, {}, 0, 200},
                                                    {Op::mul, 64, {0, 1}},
                                                    {Op::bit_or, 64, {2, 0}}}}};
  for (const auto& nodes : examples) {
    const auto block = Normalize(nodes);
    auto budget = Plenty();
    const auto result = SimplifyLinearMba(block, budget);
    ASSERT_TRUE(result.block);
    EXPECT_TRUE(result.journal.empty());
    EXPECT_EQ(result.block->nodes().back().op, nodes.back().op);
  }
}

TEST(Mba, LinearSignatureProposesRevisionBoundSsaEdit) {
  ir::SsaGraph graph;
  ir::SsaBlock block{};
  block.original_block = 0;
  block.address = 0x1000;
  block.source_groups = {0x1000};
  block.original_sources = {0};
  block.nodes = {{Op::read, 64, {}, 0, 100}, {Op::read, 64, {}, 0, 200}, {Op::bit_not, 64, {0}},
                 {Op::bit_and, 64, {2, 1}},  {Op::bit_not, 64, {1}},     {Op::bit_and, 64, {0, 4}},
                 {Op::bit_or, 64, {3, 5}}};
  block.boundaries = {{0, 7, {{300, 6}}, {}}};
  block.phis = {{100, 64, true, {}}, {200, 64, true, {}}, {300, 64, true, {}}};
  block.clobbers = {0, 0, 0};
  block.reads = {{0, 0, {}}, {1, 1, {}}};
  const auto handle = graph.Add(std::move(block));
  ASSERT_TRUE(graph.Update(handle, [&](auto& changed) {
    changed.reads[0].value = {ir::SsaValueKind::phi, handle, 0};
    changed.reads[1].value = {ir::SsaValueKind::phi, handle, 1};
    changed.exits = {{100, {ir::SsaValueKind::phi, handle, 0}},
                     {200, {ir::SsaValueKind::phi, handle, 1}},
                     {300, {ir::SsaValueKind::node, handle, 6}}};
  }));
  graph.SetEntries({handle});
  auto budget = Plenty();
  const auto before = graph.revision();
  auto result = ProposeLinearMba(graph, budget);
  ASSERT_TRUE(result.provisional) << static_cast<int>(result.reason);
  ASSERT_EQ(result.journal.size(), 1);
  EXPECT_EQ(result.journal[0].edit.from_revision, before);
  EXPECT_EQ(result.journal[0].edit.to_revision, result.provisional->revision());
  EXPECT_EQ(result.provisional->Get(result.provisional->entries()[0])->nodes[6].op, Op::bit_xor);
  EXPECT_EQ(graph.Get(handle)->nodes[6].op, Op::bit_or);
  const auto used = budget.used();
  for (const auto limit :
       {Resources{used.work - 1, used.bytes}, Resources{used.work, used.bytes - 1}}) {
    Budget cut(limit);
    auto refused = ProposeLinearMba(graph, cut);
    EXPECT_FALSE(refused.provisional);
    EXPECT_TRUE(refused.journal.empty());
    EXPECT_EQ(refused.reason, MbaDecline::resource_limit);
    EXPECT_EQ(graph.revision(), before);
  }

  auto twice = graph.Clone(budget);
  ASSERT_TRUE(twice);
  const auto entry = twice->entries()[0];
  ASSERT_TRUE(twice->Update(entry, [&](auto& changed) {
    changed.nodes.push_back(changed.nodes[6]);
    changed.boundaries[0].node_count = 8;
    changed.boundaries[0].writes[0].value = 7;
    changed.exits[2].value = {ir::SsaValueKind::node, entry, 7};
  }));
  MbaLimits one_edit;
  one_edit.max_edits = 1;
  auto limited = ProposeLinearMba(*twice, budget, one_edit);
  EXPECT_EQ(limited.reason, MbaDecline::resource_limit);
  EXPECT_FALSE(limited.provisional);
  EXPECT_TRUE(limited.journal.empty());
}

TEST(Mba, CanonicalLinearSynthesisBuildsMissingSsaNodes) {
  ir::SsaGraph graph;
  ir::SsaBlock block{};
  block.original_block = 0;
  block.address = 0x1000;
  block.source_groups = {0x1000};
  block.original_sources = {0};
  block.nodes = {{Op::read, 64, {}, 0, 100}, {Op::read, 64, {}, 0, 200}, {Op::read, 64, {}, 0, 250},
                 {Op::bit_not, 64, {0}},     {Op::bit_not, 64, {1}},     {Op::bit_not, 64, {2}},
                 {Op::bit_and, 64, {3, 4}},  {Op::bit_and, 64, {6, 2}},  {Op::bit_and, 64, {3, 1}},
                 {Op::bit_and, 64, {8, 5}},  {Op::bit_and, 64, {0, 4}},  {Op::bit_and, 64, {10, 5}},
                 {Op::bit_and, 64, {0, 1}},  {Op::bit_and, 64, {12, 2}}, {Op::bit_or, 64, {7, 9}},
                 {Op::bit_or, 64, {11, 13}}, {Op::bit_or, 64, {14, 15}}};
  block.boundaries = {{0, 17, {{300, 16}}, {}}};
  block.phis = {{100, 64, true, {}}, {200, 64, true, {}}, {250, 64, true, {}}, {300, 64, true, {}}};
  block.clobbers = {0, 0, 0, 0};
  block.reads = {{0, 0, {}}, {1, 1, {}}, {2, 2, {}}};
  const auto handle = graph.Add(std::move(block));
  ASSERT_TRUE(graph.Update(handle, [&](auto& changed) {
    for (unsigned i = 0; i < 3; ++i) changed.reads[i].value = {ir::SsaValueKind::phi, handle, i};
    changed.exits = {{100, {ir::SsaValueKind::phi, handle, 0}},
                     {200, {ir::SsaValueKind::phi, handle, 1}},
                     {250, {ir::SsaValueKind::phi, handle, 2}},
                     {300, {ir::SsaValueKind::node, handle, 16}}};
  }));
  ir::SsaBlock successor{};
  successor.original_block = 1;
  successor.address = 0x1004;
  successor.source_groups = {0x1004};
  successor.original_sources = {1};
  successor.nodes = {{Op::read, 64, {}, 0, 300}};
  successor.boundaries = {{0, 1, {}, {}}};
  successor.phis = {
      {100, 64, false, {}}, {200, 64, false, {}}, {250, 64, false, {}}, {300, 64, false, {}}};
  successor.clobbers = {0, 0, 0, 0};
  successor.reads = {{0, 3, {}}};
  const auto successor_handle = graph.Add(std::move(successor));
  ASSERT_TRUE(graph.Update(handle, [&](auto& changed) {
    changed.edges.push_back({ir::SsaEdgeKind::fallthrough,
                             ir::SsaTargetKind::image_location,
                             0x1004,
                             successor_handle,
                             {},
                             {},
                             {}});
  }));
  ASSERT_TRUE(graph.Update(successor_handle, [&](auto& changed) {
    changed.reads[0].value = {ir::SsaValueKind::phi, successor_handle, 3};
    changed.exits = {{100, {ir::SsaValueKind::phi, successor_handle, 0}},
                     {200, {ir::SsaValueKind::phi, successor_handle, 1}},
                     {250, {ir::SsaValueKind::phi, successor_handle, 2}},
                     {300, {ir::SsaValueKind::phi, successor_handle, 3}}};
    for (unsigned i = 0; i < 4; ++i) {
      const auto kind = i == 3 ? ir::SsaValueKind::node : ir::SsaValueKind::phi;
      changed.phis[i].incoming.push_back({handle, {kind, handle, i == 3 ? 16U : i}});
    }
  }));
  graph.SetEntries({handle});
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  const auto before = graph.revision();
  auto proposal_budget = Plenty();
  auto result = ProposeCanonicalLinearMba(graph, proposal_budget);
  const auto used = proposal_budget.used();
  ASSERT_TRUE(result.provisional) << static_cast<int>(result.reason);
  ASSERT_EQ(result.journal.size(), 1U);
  const auto& edit = result.journal[0];
  EXPECT_EQ(edit.from_revision, before);
  EXPECT_EQ(edit.to_revision, result.provisional->revision());
  ASSERT_EQ(edit.inserted.size(), 1U);
  EXPECT_EQ(edit.inserted[0].op, Op::bit_xor);
  EXPECT_EQ(edit.replacement.op, Op::bit_xor);
  const auto& changed = *result.provisional->Get(result.provisional->entries()[0]);
  EXPECT_EQ(changed.nodes.size(), 18U);
  EXPECT_EQ(changed.boundaries[0].writes[0].value, 17U);
  EXPECT_EQ(changed.exits[3].value.index, 17U);
  const auto* changed_successor =
      result.provisional->Get(*result.provisional->Handle(successor_handle.slot));
  ASSERT_NE(changed_successor, nullptr);
  EXPECT_EQ(changed_successor->phis[3].incoming[0].value.index, 17U);
  ASSERT_EQ(ir::ValidateSsa(*result.provisional, budget), ir::SsaDecline::none);
  std::mt19937_64 random(0x4d324d4241);
  for (unsigned sample = 0; sample < 1000; ++sample) {
    const std::array a{random()}, b{random()};
    const auto c = random();
    std::vector<Node> original(graph.Get(handle)->nodes.begin(), graph.Get(handle)->nodes.end());
    std::vector<Node> rewritten(changed.nodes.begin(), changed.nodes.end());
    original[2] = {Op::constant, 64, {}, c};
    rewritten[2] = original[2];
    EXPECT_EQ(Evaluate(original, a, b), Evaluate(rewritten, a, b));
  }

  std::vector<Node> wrong(changed.nodes.begin(), changed.nodes.end());
  wrong[2] = {Op::constant, 64, {}, 1};
  wrong.back().op = Op::bit_or;
  const std::array one{UINT64_C(1)}, zero{UINT64_C(0)};
  std::vector<Node> original(graph.Get(handle)->nodes.begin(), graph.Get(handle)->nodes.end());
  original[2] = wrong[2];
  EXPECT_NE(Evaluate(original, one, zero), Evaluate(wrong, one, zero));
  for (const auto limit :
       {Resources{used.work - 1, used.bytes}, Resources{used.work, used.bytes - 1}}) {
    Budget cut(limit);
    const auto refused = ProposeCanonicalLinearMba(graph, cut);
    EXPECT_FALSE(refused.provisional);
    EXPECT_TRUE(refused.journal.empty());
    EXPECT_EQ(refused.reason, MbaDecline::resource_limit);
    EXPECT_EQ(graph.revision(), before);
  }

  MbaLimits no_growth;
  no_growth.block.max_nodes = 17;
  auto no_growth_budget = Plenty();
  const auto no_growth_result = ProposeCanonicalLinearMba(graph, no_growth_budget, no_growth);
  EXPECT_EQ(no_growth_result.reason, MbaDecline::resource_limit);
  EXPECT_FALSE(no_growth_result.provisional);
  EXPECT_TRUE(no_growth_result.journal.empty());
}

TEST(Mba, CanonicalLinearSynthesisRefusesNonuniformAndNonlinearTerms) {
  const std::array<std::vector<Node>, 2> examples{{{{Op::read, 64, {}, 0, 100},
                                                    {Op::read, 64, {}, 0, 200},
                                                    {Op::constant, 64, {}, 0xff},
                                                    {Op::bit_and, 64, {0, 2}},
                                                    {Op::bit_or, 64, {3, 1}}},
                                                   {{Op::read, 64, {}, 0, 100},
                                                    {Op::read, 64, {}, 0, 200},
                                                    {Op::mul, 64, {0, 1}},
                                                    {Op::bit_or, 64, {2, 0}}}}};
  for (const auto& nodes : examples) {
    ir::SsaGraph graph;
    ir::SsaBlock block{};
    block.original_block = 0;
    block.address = 0x1000;
    block.source_groups = {0x1000};
    block.original_sources = {0};
    block.nodes = nodes;
    const auto root = static_cast<ir::ValueId>(nodes.size() - 1);
    block.boundaries = {{0, static_cast<std::uint32_t>(nodes.size()), {{300, root}}, {}}};
    block.phis = {{100, 64, true, {}}, {200, 64, true, {}}, {300, 64, true, {}}};
    block.clobbers = {0, 0, 0};
    block.reads = {{0, 0, {}}, {1, 1, {}}};
    const auto handle = graph.Add(std::move(block));
    ASSERT_TRUE(graph.Update(handle, [&](auto& changed) {
      changed.reads[0].value = {ir::SsaValueKind::phi, handle, 0};
      changed.reads[1].value = {ir::SsaValueKind::phi, handle, 1};
      changed.exits = {{100, {ir::SsaValueKind::phi, handle, 0}},
                       {200, {ir::SsaValueKind::phi, handle, 1}},
                       {300, {ir::SsaValueKind::node, handle, root}}};
    }));
    graph.SetEntries({handle});
    auto budget = Plenty();
    ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
    const auto result = ProposeCanonicalLinearMba(graph, budget);
    EXPECT_FALSE(result.provisional);
    EXPECT_TRUE(result.journal.empty());
    EXPECT_EQ(result.reason, MbaDecline::none);
    EXPECT_TRUE(std::any_of(
        result.refused.begin(), result.refused.end(), [&](const SsaMbaSynthesisRefusal& item) {
          return item.node == root &&
                 item.reason == SsaMbaSynthesisRefusalReason::outside_uniform_linear_scope;
        }));
    EXPECT_EQ(graph.Get(handle)->nodes[root].op, nodes[root].op);
  }
}

TEST(Mba, CanonicalLinearSynthesisReducesThreeValueArithmeticSignature) {
  ir::SsaGraph graph;
  ir::SsaBlock block{};
  block.original_block = 0;
  block.address = 0x1000;
  block.source_groups = {0x1000};
  block.original_sources = {0};
  block.nodes = {{Op::read, 64, {}, 0, 100}, {Op::read, 64, {}, 0, 200}, {Op::read, 64, {}, 0, 250},
                 {Op::bit_xor, 64, {0, 1}},  {Op::bit_xor, 64, {1, 2}},  {Op::bit_and, 64, {0, 1}},
                 {Op::bit_and, 64, {1, 2}},  {Op::add, 64, {5, 5}},      {Op::add, 64, {6, 6}},
                 {Op::add, 64, {3, 4}},      {Op::add, 64, {9, 7}},      {Op::add, 64, {10, 8}},
                 {Op::sub, 64, {11, 1}}};
  block.boundaries = {{0, 13, {{300, 12}}, {}}};
  block.phis = {{100, 64, true, {}}, {200, 64, true, {}}, {250, 64, true, {}}, {300, 64, true, {}}};
  block.clobbers = {0, 0, 0, 0};
  block.reads = {{0, 0, {}}, {1, 1, {}}, {2, 2, {}}};
  const auto handle = graph.Add(std::move(block));
  ASSERT_TRUE(graph.Update(handle, [&](auto& changed) {
    for (unsigned i = 0; i < 3; ++i) changed.reads[i].value = {ir::SsaValueKind::phi, handle, i};
    changed.exits = {{100, {ir::SsaValueKind::phi, handle, 0}},
                     {200, {ir::SsaValueKind::phi, handle, 1}},
                     {250, {ir::SsaValueKind::phi, handle, 2}},
                     {300, {ir::SsaValueKind::node, handle, 12}}};
  }));
  graph.SetEntries({handle});
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateSsa(graph, budget), ir::SsaDecline::none);
  auto direct = ProposeLinearMba(graph, budget);
  EXPECT_FALSE(direct.provisional);
  auto proposed = ProposeCanonicalLinearMba(graph, budget);
  ASSERT_TRUE(proposed.provisional) << static_cast<int>(proposed.reason);
  ASSERT_EQ(proposed.journal.size(), 1U);
  const auto& edit = proposed.journal[0];
  ASSERT_EQ(edit.inserted.size(), 1U);
  EXPECT_EQ(edit.inserted[0].op, Op::add);
  EXPECT_EQ(edit.replacement.op, Op::add);
  const auto& changed = *proposed.provisional->Get(proposed.provisional->entries()[0]);
  EXPECT_EQ(changed.nodes.size(), 14U);
  EXPECT_EQ(changed.exits[3].value.index, 13U);
  ASSERT_EQ(ir::ValidateSsa(*proposed.provisional, budget), ir::SsaDecline::none);
  std::mt19937_64 random(0x4d324146);
  for (unsigned sample = 0; sample < 1000; ++sample) {
    const std::array a{random()}, b{random()};
    const auto c = random();
    std::vector<Node> original(graph.Get(handle)->nodes.begin(), graph.Get(handle)->nodes.end());
    std::vector<Node> rewritten(changed.nodes.begin(), changed.nodes.end());
    original[2] = {Op::constant, 64, {}, c};
    rewritten[2] = original[2];
    EXPECT_EQ(Evaluate(original, a, b), Evaluate(rewritten, a, b));
  }

  auto negative = graph.Clone(budget);
  ASSERT_TRUE(negative);
  const auto negative_entry = negative->entries()[0];
  ASSERT_TRUE(negative->Update(negative_entry, [](auto& changed) {
    changed.nodes.push_back({Op::add, 64, {2, 2}});
    changed.nodes.push_back({Op::sub, 64, {11, 13}});
    changed.boundaries[0].node_count = 15;
    changed.boundaries[0].writes[0].value = 14;
    changed.exits[3].value.index = 14;
  }));
  ASSERT_EQ(ir::ValidateSsa(*negative, budget), ir::SsaDecline::none);
  auto signed_result = ProposeCanonicalLinearMba(*negative, budget);
  ASSERT_TRUE(signed_result.provisional) << static_cast<int>(signed_result.reason);
  ASSERT_EQ(signed_result.journal.size(), 1U);
  EXPECT_EQ(signed_result.journal[0].replacement.op, Op::sub);
  const auto& signed_block =
      *signed_result.provisional->Get(signed_result.provisional->entries()[0]);
  for (unsigned sample = 0; sample < 100; ++sample) {
    const std::array a{random()}, b{random()};
    const auto c = random();
    std::vector<Node> original(negative->Get(negative_entry)->nodes.begin(),
                               negative->Get(negative_entry)->nodes.end());
    std::vector<Node> rewritten(signed_block.nodes.begin(), signed_block.nodes.end());
    original[2] = {Op::constant, 64, {}, c};
    rewritten[2] = original[2];
    EXPECT_EQ(Evaluate(original, a, b), Evaluate(rewritten, a, b));
  }

  auto nonaffine = graph.Clone(budget);
  ASSERT_TRUE(nonaffine);
  const auto nonaffine_entry = nonaffine->entries()[0];
  ASSERT_TRUE(nonaffine->Update(nonaffine_entry,
                                [](auto& changed) { changed.nodes[12] = {Op::add, 64, {9, 5}}; }));
  auto refused = ProposeCanonicalLinearMba(*nonaffine, budget);
  EXPECT_TRUE(std::any_of(
      refused.refused.begin(), refused.refused.end(), [](const SsaMbaSynthesisRefusal& item) {
        return item.node == 12 &&
               item.reason == SsaMbaSynthesisRefusalReason::outside_uniform_linear_scope;
      }));
  for (unsigned width : {1U, 2U, 3U, 7U}) {
    auto narrow = graph.Clone(budget);
    ASSERT_TRUE(narrow);
    const auto entry = narrow->entries()[0];
    ASSERT_TRUE(narrow->Update(entry, [&](auto& changed) {
      for (auto& node : changed.nodes) node.width = width;
      for (auto& phi : changed.phis) phi.width = width;
    }));
    ASSERT_EQ(ir::ValidateSsa(*narrow, budget), ir::SsaDecline::none);
    auto result = ProposeCanonicalLinearMba(*narrow, budget);
    if (width != 1) {
      ASSERT_TRUE(result.provisional) << width;
    }

    if (!result.provisional) continue;
    const auto& rewritten = *result.provisional->Get(result.provisional->entries()[0]);
    for (unsigned sample = 0; sample < 100; ++sample) {
      const std::array a{random()}, b{random()};
      const auto c = random();
      std::vector<Node> original(narrow->Get(entry)->nodes.begin(),
                                 narrow->Get(entry)->nodes.end());
      std::vector<Node> changed(rewritten.nodes.begin(), rewritten.nodes.end());
      original[2] = {Op::constant, width, {}, c};
      changed[2] = original[2];
      EXPECT_EQ(Evaluate(original, a, b), Evaluate(changed, a, b));
    }
  }

  auto full_budget = Plenty();
  auto full_result = ProposeCanonicalLinearMba(graph, full_budget);
  ASSERT_TRUE(full_result.provisional);
  const auto used = full_budget.used();
  for (const auto limit :
       {Resources{used.work - 1, used.bytes}, Resources{used.work, used.bytes - 1}}) {
    Budget cut(limit);
    const auto declined = ProposeCanonicalLinearMba(graph, cut);
    EXPECT_FALSE(declined.provisional);
    EXPECT_TRUE(declined.journal.empty());
    EXPECT_EQ(declined.reason, MbaDecline::resource_limit);
  }
}

}  // namespace
}  // namespace nyx::recovery
