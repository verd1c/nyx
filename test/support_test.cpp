#include <array>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "nyx/support/bit_vector.hpp"
#include "nyx/support/budget.hpp"

namespace nyx {
namespace {

Budget unlimited() {
  return Budget(
      {std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max()});
}

TEST(Budget, ChargesAreTransactionalAndCannotOverflow) {
  Budget budget({10, 20});
  EXPECT_EQ(budget.try_consume({3, 5}), BudgetDecline::none);
  EXPECT_EQ(budget.try_consume({8, 0}), BudgetDecline::work_limit);
  EXPECT_EQ(budget.try_consume({7, 16}), BudgetDecline::byte_limit);
  EXPECT_EQ(budget.used().work, 3);
  EXPECT_EQ(budget.used().bytes, 5);
  EXPECT_EQ(budget.try_consume({7, 15}), BudgetDecline::none);
  EXPECT_EQ(budget.try_consume({1, 0}), BudgetDecline::work_limit);
  auto huge = unlimited();
  const auto max = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(huge.try_consume({max, max}), BudgetDecline::none);
  EXPECT_EQ(huge.try_consume({1, 0}), BudgetDecline::work_limit);
  EXPECT_EQ(huge.try_consume({0, 1}), BudgetDecline::byte_limit);
}

TEST(BitVector, ConstructionAndLimits) {
  auto budget = unlimited();
  EXPECT_EQ(BitVector::from_u64(0, 1, 4096, budget).decline(), BitVectorDecline::invalid_width);
  EXPECT_EQ(BitVector::from_u64(4097, 1, 4096, budget).decline(), BitVectorDecline::width_limit);
  std::array<std::uint64_t, 2> words{~std::uint64_t{0}, ~std::uint64_t{0}};
  EXPECT_EQ(BitVector::from_words(64, words, 4096, budget).decline(),
            BitVectorDecline::excess_words);
  auto value = BitVector::from_words(65, words, 4096, budget);
  ASSERT_TRUE(value);
  EXPECT_EQ(value->word(1), 1);
  EXPECT_EQ(value->word(2), 0);
  EXPECT_FALSE(value->bit(65));
  Budget tiny({100, 7});
  EXPECT_EQ(BitVector::from_u64(64, 1, 4096, tiny).decline(), BitVectorDecline::byte_limit);
  EXPECT_EQ(tiny.used().bytes, 0);
  auto other = BitVector::from_u64(64, 1, 4096, budget);
  EXPECT_EQ(value->add(*other, budget).decline(), BitVectorDecline::width_mismatch);
  EXPECT_FALSE(value->sub(*other, budget));
  EXPECT_FALSE(value->mul(*other, budget));
  EXPECT_FALSE(value->bit_and(*other, budget));
  EXPECT_FALSE(value->bit_or(*other, budget));
  EXPECT_FALSE(value->bit_xor(*other, budget));
  EXPECT_EQ(value->shl(1, tiny).decline(), BitVectorDecline::byte_limit);
  Budget no_work({0, 1000});
  EXPECT_EQ(value->shl(1, no_work).decline(), BitVectorDecline::work_limit);
}

TEST(BitVector, ExhaustiveSmallWidths) {
  auto budget = unlimited();
  for (unsigned width = 1; width <= 7; ++width) {
    const std::uint64_t mask = (std::uint64_t{1} << width) - 1;
    for (std::uint64_t a = 0; a <= mask; ++a) {
      const auto lhs = BitVector::from_u64(width, a, 4096, budget);
      EXPECT_EQ(lhs->bit_not(budget)->word(0), (~a) & mask);
      for (std::uint64_t b = 0; b <= mask; ++b) {
        const auto rhs = BitVector::from_u64(width, b, 4096, budget);
        EXPECT_EQ(lhs->add(*rhs, budget)->word(0), (a + b) & mask);
        EXPECT_EQ(lhs->sub(*rhs, budget)->word(0), (a - b) & mask);
        EXPECT_EQ(lhs->mul(*rhs, budget)->word(0), (a * b) & mask);
        EXPECT_EQ(lhs->bit_and(*rhs, budget)->word(0), a & b);
        EXPECT_EQ(lhs->bit_or(*rhs, budget)->word(0), a | b);
        EXPECT_EQ(lhs->bit_xor(*rhs, budget)->word(0), a ^ b);
      }

      for (std::uint64_t count = 0; count <= width + 2; ++count) {
        EXPECT_EQ(lhs->shl(count, budget)->word(0), (a << count) & mask);
        EXPECT_EQ(lhs->lshr(count, budget)->word(0), a >> count);
        const auto signed_value = (a & (std::uint64_t{1} << (width - 1)))
                                      ? static_cast<std::int64_t>(a) - (std::int64_t{1} << width)
                                      : static_cast<std::int64_t>(a);
        EXPECT_EQ(lhs->ashr(count, budget)->word(0),
                  static_cast<std::uint64_t>(signed_value >> count) & mask);
        const auto shift = count % width;
        EXPECT_EQ(lhs->rotl(count, budget)->word(0),
                  ((a << shift) | (a >> (width - shift))) & mask);
        EXPECT_EQ(lhs->rotr(count, budget)->word(0),
                  ((a >> shift) | (a << (width - shift))) & mask);
      }
    }
  }
}

TEST(BitVector, MultiLimbAgainstIndependentByteArithmetic) {
  auto budget = unlimited();
  std::mt19937_64 random(0x7ae395);
  for (unsigned width : {63U, 64U, 65U, 127U, 128U, 129U, 255U, 256U, 257U, 1024U}) {
    for (unsigned trial = 0; trial < 16; ++trial) {
      std::vector<std::uint64_t> a(width / 64 + (width % 64 != 0));
      auto b = a;
      for (auto& word : a) word = random();
      for (auto& word : b) word = random();
      const auto lhs = BitVector::from_words(width, a, 4096, budget);
      const auto rhs = BitVector::from_words(width, b, 4096, budget);
      const auto sum = lhs->add(*rhs, budget);
      const auto difference = lhs->sub(*rhs, budget);
      const auto product = lhs->mul(*rhs, budget);
      const unsigned count = (width + 7) / 8;
      std::vector<unsigned> bytes_a(count), bytes_b(count), product_bytes(count);
      unsigned carry = 0;
      int borrow = 0;
      for (unsigned i = 0; i < count; ++i) {
        bytes_a[i] = (lhs->word(i / 8) >> ((i % 8) * 8)) & 255;
        bytes_b[i] = (rhs->word(i / 8) >> ((i % 8) * 8)) & 255;
        const auto total = bytes_a[i] + bytes_b[i] + carry;
        carry = total / 256;
        const auto delta = int(bytes_a[i]) - int(bytes_b[i]) - borrow;
        borrow = delta < 0;
        const unsigned byte_mask =
            (i + 1 == count && width % 8 != 0) ? (1U << (width % 8)) - 1 : 255;
        EXPECT_EQ((sum->word(i / 8) >> ((i % 8) * 8)) & 255, total & byte_mask);
        EXPECT_EQ((difference->word(i / 8) >> ((i % 8) * 8)) & 255,
                  static_cast<unsigned>(delta) & byte_mask);
      }

      for (unsigned i = 0; i < count; ++i) {
        carry = 0;
        for (unsigned j = 0; j < count - i; ++j) {
          const auto total = product_bytes[i + j] + bytes_a[i] * bytes_b[j] + carry;
          product_bytes[i + j] = total % 256;
          carry = total / 256;
        }
      }

      for (unsigned i = 0; i < count; ++i) {
        const unsigned mask = (i + 1 == count && width % 8 != 0) ? (1U << (width % 8)) - 1 : 255;
        EXPECT_EQ((product->word(i / 8) >> ((i % 8) * 8)) & 255, product_bytes[i] & mask);
      }

      for (auto shift : {std::uint64_t{0}, std::uint64_t{63}, std::uint64_t{64}, std::uint64_t{65},
                         std::uint64_t{width}, std::numeric_limits<std::uint64_t>::max()}) {
        const auto left = lhs->shl(shift, budget);
        const auto right = lhs->lshr(shift, budget);
        const auto arithmetic = lhs->ashr(shift, budget);
        const auto rotate = lhs->rotl(shift, budget);
        for (unsigned i = 0; i < width; ++i) {
          EXPECT_EQ(left->bit(i), shift <= i && lhs->bit(i - static_cast<unsigned>(shift)));
          EXPECT_EQ(right->bit(i), shift < width && i < width - shift && lhs->bit(i + shift));
          EXPECT_EQ(arithmetic->bit(i), shift >= width || i >= width - shift ? lhs->bit(width - 1)
                                                                             : lhs->bit(i + shift));
          EXPECT_EQ(rotate->bit(i), lhs->bit((std::uint64_t{i} + width - shift % width) % width));
        }
      }
    }
  }
}

}  // namespace
}  // namespace nyx
