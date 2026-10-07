#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "nyx/support/budget.hpp"

namespace nyx {

enum class BitVectorDecline {
  none,
  invalid_width,
  width_limit,
  excess_words,
  width_mismatch,
  work_limit,
  byte_limit
};

class BitVectorResult;

// Width and resource limits are supplied by the caller; no architectural width
// ceiling is built into the value representation.
class BitVector {
 public:
  BitVector(BitVector&&) noexcept = default;
  BitVector& operator=(BitVector&&) noexcept = default;
  BitVector(const BitVector&) = delete;
  BitVector& operator=(const BitVector&) = delete;

  [[nodiscard]] static BitVectorResult from_u64(unsigned width, std::uint64_t value,
                                                unsigned max_bits, Budget& budget) noexcept;
  // Words are least-significant first. Missing words are zero; excess words
  // are rejected. Bits above width in the final word are reduced modulo 2^width.
  [[nodiscard]] static BitVectorResult from_words(unsigned width,
                                                  std::span<const std::uint64_t> words,
                                                  unsigned max_bits, Budget& budget) noexcept;

  [[nodiscard]] unsigned width() const noexcept { return width_; }

  [[nodiscard]] std::span<const std::uint64_t> words() const noexcept { return words_; }

  [[nodiscard]] unsigned word_count() const noexcept { return width_ / 64 + (width_ % 64 != 0); }

  [[nodiscard]] std::uint64_t word(unsigned index) const noexcept;
  [[nodiscard]] bool bit(unsigned index) const noexcept;

  // Binary operations decline different-width operands, rather than implicitly
  // choosing a signedness or an extension rule for the caller.
  [[nodiscard]] BitVectorResult add(const BitVector& rhs, Budget& budget) const noexcept;
  [[nodiscard]] BitVectorResult sub(const BitVector& rhs, Budget& budget) const noexcept;
  [[nodiscard]] BitVectorResult mul(const BitVector& rhs, Budget& budget) const noexcept;
  [[nodiscard]] BitVectorResult bit_and(const BitVector& rhs, Budget& budget) const noexcept;
  [[nodiscard]] BitVectorResult bit_or(const BitVector& rhs, Budget& budget) const noexcept;
  [[nodiscard]] BitVectorResult bit_xor(const BitVector& rhs, Budget& budget) const noexcept;
  [[nodiscard]] BitVectorResult bit_not(Budget& budget) const noexcept;

  // These are mathematical bitvector shifts, not target-masked shifts. Counts
  // >= width yield zero (or the sign fill for ashr). Target lifting masks counts
  // explicitly when required by the architecture.
  [[nodiscard]] BitVectorResult shl(std::uint64_t count, Budget& budget) const noexcept;
  [[nodiscard]] BitVectorResult lshr(std::uint64_t count, Budget& budget) const noexcept;
  [[nodiscard]] BitVectorResult ashr(std::uint64_t count, Budget& budget) const noexcept;
  [[nodiscard]] BitVectorResult rotl(std::uint64_t count, Budget& budget) const noexcept;
  [[nodiscard]] BitVectorResult rotr(std::uint64_t count, Budget& budget) const noexcept;

  bool operator==(const BitVector&) const noexcept = default;

 private:
  explicit BitVector(unsigned width) noexcept
      : width_(width), words_(width / 64 + (width % 64 != 0), 0) {}

  [[nodiscard]] static BitVectorDecline charge(unsigned width, std::uint64_t work,
                                               Budget& budget) noexcept;
  void normalize() noexcept;
  void set_bit(unsigned index) noexcept;

  unsigned width_;
  std::vector<std::uint64_t> words_;
};

class BitVectorResult {
 public:
  BitVectorResult(BitVector value) noexcept : value_(std::move(value)) {}

  BitVectorResult(BitVectorDecline reason) noexcept : reason_(reason) {}

  [[nodiscard]] explicit operator bool() const noexcept { return value_.has_value(); }

  [[nodiscard]] BitVectorDecline decline() const noexcept { return reason_; }

  [[nodiscard]] const BitVector* operator->() const noexcept { return &*value_; }

  [[nodiscard]] BitVector* operator->() noexcept { return &*value_; }

  [[nodiscard]] const BitVector& operator*() const noexcept { return *value_; }

  [[nodiscard]] BitVector& operator*() noexcept { return *value_; }

 private:
  std::optional<BitVector> value_;
  BitVectorDecline reason_ = BitVectorDecline::none;
};

}  // namespace nyx
