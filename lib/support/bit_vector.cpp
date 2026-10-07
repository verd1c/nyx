#include "nyx/support/bit_vector.hpp"

namespace nyx {

BitVectorResult BitVector::from_u64(unsigned width, std::uint64_t value, unsigned max_bits,
                                    Budget& budget) noexcept {
  return from_words(width, std::span<const std::uint64_t>(&value, 1), max_bits, budget);
}

BitVectorResult BitVector::from_words(unsigned width, std::span<const std::uint64_t> words,
                                      unsigned max_bits, Budget& budget) noexcept {
  if (width == 0) return BitVectorDecline::invalid_width;
  if (width > max_bits) return BitVectorDecline::width_limit;
  if (words.size() > width / 64 + (width % 64 != 0)) return BitVectorDecline::excess_words;
  if (const auto decline = charge(width, width / 64 + (width % 64 != 0), budget);
      decline != BitVectorDecline::none)
    return decline;
  BitVector result(width);
  for (std::size_t i = 0; i < words.size(); ++i) result.words_[i] = words[i];
  result.normalize();
  return result;
}

BitVectorDecline BitVector::charge(unsigned width, std::uint64_t work, Budget& budget) noexcept {
  const std::uint64_t limbs = width / 64 + (width % 64 != 0);
  switch (budget.try_consume({work, limbs * sizeof(std::uint64_t)})) {
    case BudgetDecline::none:
      return BitVectorDecline::none;
    case BudgetDecline::work_limit:
      return BitVectorDecline::work_limit;
    case BudgetDecline::byte_limit:
      return BitVectorDecline::byte_limit;
  }

  return BitVectorDecline::work_limit;
}

void BitVector::normalize() noexcept {
  const unsigned tail = width_ % 64;
  if (tail != 0) words_[word_count() - 1] &= (std::uint64_t{1} << tail) - 1;
}

std::uint64_t BitVector::word(unsigned index) const noexcept {
  return index < word_count() ? words_[index] : 0;
}

bool BitVector::bit(unsigned index) const noexcept {
  return index < width_ && ((words_[index / 64] >> (index % 64)) & 1) != 0;
}

void BitVector::set_bit(unsigned index) noexcept {
  words_[index / 64] |= std::uint64_t{1} << (index % 64);
}

BitVectorResult BitVector::add(const BitVector& rhs, Budget& budget) const noexcept {
  if (width_ != rhs.width_) return BitVectorDecline::width_mismatch;
  if (const auto decline = charge(width_, width_, budget); decline != BitVectorDecline::none)
    return decline;
  BitVector result(width_);
  std::uint64_t carry = 0;
  for (unsigned i = 0; i < word_count(); ++i) {
    const auto sum = words_[i] + rhs.words_[i];
    result.words_[i] = sum + carry;
    carry = (sum < words_[i]) || (result.words_[i] < sum);
  }

  result.normalize();
  return result;
}

BitVectorResult BitVector::sub(const BitVector& rhs, Budget& budget) const noexcept {
  if (width_ != rhs.width_) return BitVectorDecline::width_mismatch;
  if (const auto decline = charge(width_, width_, budget); decline != BitVectorDecline::none)
    return decline;
  BitVector result(width_);
  std::uint64_t borrow = 0;
  for (unsigned i = 0; i < word_count(); ++i) {
    const auto difference = words_[i] - rhs.words_[i];
    result.words_[i] = difference - borrow;
    borrow = (words_[i] < rhs.words_[i]) || (difference < borrow);
  }

  result.normalize();
  return result;
}

BitVectorResult BitVector::mul(const BitVector& rhs, Budget& budget) const noexcept {
  if (width_ != rhs.width_) return BitVectorDecline::width_mismatch;
  if (const auto decline = charge(width_, std::uint64_t{width_} * width_, budget);
      decline != BitVectorDecline::none)
    return decline;
  BitVector result(width_);

  // Schoolbook binary multiplication avoids host extended-integer dependencies.
  for (unsigned i = 0; i < width_; ++i) {
    if (!rhs.bit(i)) continue;
    bool carry = false;
    for (unsigned j = 0; j < width_ - i; ++j) {
      const unsigned sum = unsigned(result.bit(i + j)) + unsigned(bit(j)) + unsigned(carry);
      const auto mask = std::uint64_t{1} << ((i + j) % 64);
      auto& limb = result.words_[(i + j) / 64];
      limb = (limb & ~mask) | ((sum & 1) ? mask : 0);
      carry = sum >= 2;
    }
  }

  return result;
}

BitVectorResult BitVector::bit_and(const BitVector& rhs, Budget& budget) const noexcept {
  if (width_ != rhs.width_) return BitVectorDecline::width_mismatch;
  if (const auto decline = charge(width_, width_, budget); decline != BitVectorDecline::none)
    return decline;
  BitVector result(width_);
  for (unsigned i = 0; i < word_count(); ++i) result.words_[i] = words_[i] & rhs.words_[i];
  return result;
}

BitVectorResult BitVector::bit_or(const BitVector& rhs, Budget& budget) const noexcept {
  if (width_ != rhs.width_) return BitVectorDecline::width_mismatch;
  if (const auto decline = charge(width_, width_, budget); decline != BitVectorDecline::none)
    return decline;
  BitVector result(width_);
  for (unsigned i = 0; i < word_count(); ++i) result.words_[i] = words_[i] | rhs.words_[i];
  return result;
}

BitVectorResult BitVector::bit_xor(const BitVector& rhs, Budget& budget) const noexcept {
  if (width_ != rhs.width_) return BitVectorDecline::width_mismatch;
  if (const auto decline = charge(width_, width_, budget); decline != BitVectorDecline::none)
    return decline;
  BitVector result(width_);
  for (unsigned i = 0; i < word_count(); ++i) result.words_[i] = words_[i] ^ rhs.words_[i];
  return result;
}

BitVectorResult BitVector::bit_not(Budget& budget) const noexcept {
  if (const auto decline = charge(width_, width_, budget); decline != BitVectorDecline::none)
    return decline;
  BitVector result(width_);
  for (unsigned i = 0; i < word_count(); ++i) result.words_[i] = ~words_[i];
  result.normalize();
  return result;
}

BitVectorResult BitVector::shl(std::uint64_t count, Budget& budget) const noexcept {
  if (const auto decline = charge(width_, width_, budget); decline != BitVectorDecline::none)
    return decline;
  BitVector result(width_);
  if (count >= width_) return result;
  for (unsigned i = static_cast<unsigned>(count); i < width_; ++i) {
    if (bit(i - static_cast<unsigned>(count))) result.set_bit(i);
  }

  return result;
}

BitVectorResult BitVector::lshr(std::uint64_t count, Budget& budget) const noexcept {
  if (const auto decline = charge(width_, width_, budget); decline != BitVectorDecline::none)
    return decline;
  BitVector result(width_);
  if (count >= width_) return result;
  for (unsigned i = 0; i < width_ - count; ++i) {
    if (bit(i + static_cast<unsigned>(count))) result.set_bit(i);
  }

  return result;
}

BitVectorResult BitVector::ashr(std::uint64_t count, Budget& budget) const noexcept {
  auto result = lshr(count, budget);
  if (!result) return result.decline();
  if (bit(width_ - 1)) {
    const unsigned first = count >= width_ ? 0 : width_ - static_cast<unsigned>(count);
    for (unsigned i = first; i < width_; ++i) result->set_bit(i);
  }

  return result;
}

BitVectorResult BitVector::rotl(std::uint64_t count, Budget& budget) const noexcept {
  if (const auto decline = charge(width_, width_, budget); decline != BitVectorDecline::none)
    return decline;
  BitVector result(width_);
  const auto shift = count % width_;
  for (unsigned i = 0; i < width_; ++i) {
    if (bit(i)) result.set_bit(static_cast<unsigned>((std::uint64_t{i} + shift) % width_));
  }

  return result;
}

BitVectorResult BitVector::rotr(std::uint64_t count, Budget& budget) const noexcept {
  return rotl(width_ - count % width_, budget);
}

}  // namespace nyx
