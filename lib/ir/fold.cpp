#include "nyx/ir/fold.hpp"

#include <bit>

namespace nyx::ir {
namespace {

// The upper `width` bits of the 2*width-bit product of two width-bit values.
// Built from 32-bit halves because -Wpedantic rejects the 128-bit integer
// extension.
std::uint64_t UnsignedHigh(std::uint64_t a, std::uint64_t b, unsigned width) {
  const auto al = a & UINT32_MAX, ah = a >> 32, bl = b & UINT32_MAX, bh = b >> 32;
  const auto low = al * bl, cross = ah * bl + (low >> 32);
  const auto middle = al * bh + (cross & UINT32_MAX);
  const auto high = ah * bh + (cross >> 32) + (middle >> 32);
  const auto bottom = (middle << 32) | (low & UINT32_MAX);
  return width == 64 ? high : (high << (64 - width)) | (bottom >> width);
}

}  // namespace

std::optional<std::uint64_t> FoldPure(const Node& node, std::span<const std::uint64_t> operands,
                                      unsigned operand_width) {
  const auto* descriptor = Descriptor(node.op);
  if (descriptor == nullptr || descriptor->effect != Effect::pure || !descriptor->produces_value ||
      descriptor->arity == 0 || operands.size() != descriptor->arity || node.width > 64 ||
      operand_width == 0 || operand_width > 64)
    return std::nullopt;
  const auto a = operands.size() > 0 ? operands[0] : 0, b = operands.size() > 1 ? operands[1] : 0;
  const auto width = operand_width;
  std::uint64_t result = 0;
  switch (node.op) {
    case Op::add:
      result = a + b;
      break;
    case Op::sub:
      result = a - b;
      break;
    case Op::mul:
      result = a * b;
      break;
    case Op::bit_and:
      result = a & b;
      break;
    case Op::bit_or:
      result = a | b;
      break;
    case Op::bit_xor:
      result = a ^ b;
      break;
    case Op::bit_not:
      result = ~a;
      break;
    case Op::shl:
      result = b >= width ? 0 : a << b;
      break;
    case Op::lshr:
      result = b >= width ? 0 : a >> b;
      break;
    case Op::ashr: {
      const bool sign = (a >> (width - 1)) != 0;
      if (b >= width)
        result = sign ? LowMask(width) : 0;
      else {
        result = a >> b;
        if (sign && b != 0) result |= LowMask(width) ^ LowMask(width - static_cast<unsigned>(b));
      }

      break;
    }
    case Op::extract:
      result = a >> node.immediate;
      break;
    case Op::zext:
      result = a;
      break;
    case Op::select:
      result = a ? b : operands[2];
      break;
    case Op::equal:
      result = a == b;
      break;
    case Op::unsigned_less:
      result = a < b;
      break;
    case Op::signed_less: {
      const auto sign = UINT64_C(1) << (width - 1);
      result = (a ^ sign) < (b ^ sign);
      break;
    }

    // The division and high-product cases read their operands masked: a
    // stray bit above the width would otherwise divide by zero or reach the
    // high half, where the other operations only lose it to the final mask.
    case Op::udiv: {
      const auto x = a & LowMask(width), y = b & LowMask(width);
      result = y ? x / y : 0;
      break;
    }
    case Op::sdiv: {
      const auto sign = UINT64_C(1) << (width - 1);
      const bool negative_a = a & sign, negative_b = b & sign;

      // Divide magnitudes: the most negative value over -1 then wraps back to
      // itself instead of overflowing a signed division.
      const auto magnitude_a = (negative_a ? 0 - a : a) & LowMask(width);
      const auto magnitude_b = (negative_b ? 0 - b : b) & LowMask(width);
      const auto quotient = magnitude_b ? magnitude_a / magnitude_b : 0;
      result = negative_a != negative_b ? 0 - quotient : quotient;
      break;
    }
    case Op::umulh:
      result = UnsignedHigh(a & LowMask(width), b & LowMask(width), width);
      break;
    case Op::smulh: {
      // Reading a negative operand as unsigned adds 2^width times the other
      // operand to the product, so the high half comes back by subtraction.
      const auto sign = UINT64_C(1) << (width - 1);
      const auto x = a & LowMask(width), y = b & LowMask(width);
      result = UnsignedHigh(x, y, width) - (x & sign ? y : 0) - (y & sign ? x : 0);
      break;
    }
    case Op::clz:
      result = std::countl_zero(a & LowMask(width)) - (64 - width);
      break;
    case Op::rbit: {
      constexpr std::uint64_t kLanes[] = {0x5555555555555555, 0x3333333333333333,
                                          0x0f0f0f0f0f0f0f0f, 0x00ff00ff00ff00ff,
                                          0x0000ffff0000ffff, 0x00000000ffffffff};
      auto reversed = a & LowMask(width);
      for (unsigned step = 0; step < 6; ++step)
        reversed = (reversed >> (1U << step) & kLanes[step]) | (reversed & kLanes[step])
                                                                   << (1U << step);
      result = reversed >> (64 - width);
      break;
    }
    default:
      return std::nullopt;
  }

  return result & LowMask(node.width);
}

}  // namespace nyx::ir
