#include "nyx/target/toy/decode.hpp"

#include <limits>

namespace nyx::toy {
namespace {

using ir::Op;
using ir::ValueId;

DecodeDecline Convert(BudgetDecline reason) {
  if (reason == BudgetDecline::work_limit) return DecodeDecline::work_limit;
  if (reason == BudgetDecline::byte_limit) return DecodeDecline::byte_limit;
  return DecodeDecline::none;
}

// Opcode to length; zero for an unknown opcode.
unsigned Length(std::uint8_t opcode) {
  switch (opcode) {
    case 0x00:
    case 0x01:
    case 0x02:
      return 1;
    case 0x16:
    case 0x17:
    case 0x40:
    case 0x55:
    case 0x57:
      return 2;
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x18:
    case 0x32:
    case 0x53:
    case 0x54:
      return 3;
    case 0x21:
    case 0x22:
    case 0x23:
    case 0x30:
    case 0x31:
    case 0x33:
    case 0x50:
    case 0x51:
      return 4;
    case 0x52:
    case 0x56:
      return 5;
    case 0x20:
      return 10;
    default:
      return 0;
  }
}

class Lift {
 public:
  explicit Lift(Budget& budget) : budget_(budget) {
    reason = Convert(budget_.try_consume({1, 16 * sizeof(ir::Node) + 2 * sizeof(ir::Write)}));
    if (reason == DecodeDecline::none) {
      nodes.reserve(16);
      writes.reserve(2);
    }
  }

  ValueId Add(Op op, unsigned width, ValueId a = 0, ValueId b = 0, ValueId c = 0,
              std::uint64_t immediate = 0, ir::StorageId storage = 0) {
    if (reason != DecodeDecline::none) return 0;
    reason = Convert(budget_.try_consume({1, 0}));
    if (reason != DecodeDecline::none) return 0;
    if (nodes.size() == nodes.capacity()) {
      reason = DecodeDecline::byte_limit;
      return 0;
    }

    nodes.push_back({op, width, {a, b, c}, immediate, storage});
    return static_cast<ValueId>(nodes.size() - 1);
  }

  ValueId Constant(std::uint64_t value) { return Add(Op::constant, 64, 0, 0, 0, value); }

  ValueId Read(unsigned reg) { return Add(Op::read, 64, 0, 0, 0, 0, reg); }

  // Every toy memory access is a big-endian, unaligned-tolerant 64-bit one.
  ValueId Memory(Op op, ValueId address, ValueId value = 0) {
    const auto id = Add(op, 64, address, value);
    if (reason == DecodeDecline::none) nodes.back().access.byte_order = ir::ByteOrder::big;
    return id;
  }

  void Write(unsigned reg, ValueId value) {
    if (reason != DecodeDecline::none) return;
    if (writes.size() == writes.capacity()) {
      reason = DecodeDecline::byte_limit;
      return;
    }

    writes.push_back({reg, value});
  }

  DecodeDecline reason = DecodeDecline::none;
  std::vector<ir::Node> nodes;
  std::vector<ir::Write> writes;

 private:
  Budget& budget_;
};

}  // namespace

bool OpaqueTrap(std::span<const std::uint8_t> bytes) { return !bytes.empty() && bytes[0] == 0x00; }

DecodeResult Decode(std::uint64_t address, std::span<const std::uint8_t> bytes, Budget& budget) {
  if (bytes.empty()) return {std::nullopt, 0, DecodeDecline::truncated};
  const auto opcode = bytes[0];
  const auto length = Length(opcode);
  if (!length) return {std::nullopt, 0, DecodeDecline::invalid_encoding};

  // A truncated instruction occupies no bytes a sweep may step over; one that
  // would wrap the address space has no location at all.
  if (bytes.size() < length) return {std::nullopt, 0, DecodeDecline::truncated};
  if (address > std::numeric_limits<std::uint64_t>::max() - (length - 1))
    return {std::nullopt, 0, DecodeDecline::invalid_location};
  if (opcode == 0x00) return {std::nullopt, length, DecodeDecline::unsupported};
  const auto word = [&](unsigned offset, unsigned count) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < count; ++i) value = value << 8 | bytes[offset + i];
    return value;
  };

  const auto signed16 = [&](unsigned offset) {
    return static_cast<std::uint64_t>(static_cast<std::int64_t>(
        static_cast<std::int16_t>(static_cast<std::uint16_t>(word(offset, 2)))));
  };

  const unsigned d = length > 1 ? bytes[1] >> 4 : 0;
  const unsigned s = length > 1 ? bytes[1] & 15 : 0;
  Lift lift(budget);
  if (lift.reason != DecodeDecline::none) return {std::nullopt, length, lift.reason};
  auto model = ir::MemoryModel::unspecified;
  std::optional<ir::Transfer> transfer;
  const auto relative = [&](std::uint64_t offset) {
    return lift.Add(Op::add, 64, lift.Add(Op::image_address, 64, 0, 0, 0, address),
                    lift.Constant(offset));
  };

  // A register field the encoding does not use must be zero.
  const auto unused = [&](bool zero) {
    return zero ? DecodeDecline::none : DecodeDecline::invalid_encoding;
  };

  DecodeDecline shape = DecodeDecline::none;
  switch (opcode) {
    case 0x01:
      transfer = ir::Transfer{ir::TransferKind::return_, lift.Read(kLink), {}, {}, {}};
      break;
    case 0x02:
      break;
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13:
    case 0x14:
    case 0x15: {
      shape = unused((bytes[2] & 15) == 0);
      constexpr Op ops[] = {Op::add, Op::sub, Op::bit_and, Op::bit_or, Op::bit_xor, Op::mul};
      lift.Write(d, lift.Add(ops[opcode - 0x10], 64, lift.Read(s), lift.Read(bytes[2] >> 4)));
      break;
    }
    case 0x16:
      lift.Write(d, lift.Add(Op::bit_not, 64, lift.Read(s)));
      break;
    case 0x17:
      lift.Write(d, lift.Read(s));
      break;
    case 0x18:
      shape = unused(bytes[2] < 64);
      lift.Write(d, lift.Add(Op::shl, 64, lift.Read(s), lift.Constant(bytes[2])));
      break;
    case 0x20:
      shape = unused(s == 0);
      lift.Write(d, lift.Constant(word(2, 8)));
      break;
    case 0x21:
      shape = unused(s == 0);
      lift.Write(d, lift.Constant(word(2, 2)));
      break;
    case 0x22:
      lift.Write(d, lift.Add(Op::add, 64, lift.Read(s), lift.Constant(signed16(2))));
      break;
    case 0x23:
      shape = unused(s == 0);
      lift.Write(d, relative(signed16(2)));
      break;
    case 0x30:
    case 0x31:
    case 0x32: {
      model = ir::MemoryModel::atomic_scalar_reference;
      ValueId target;
      if (opcode == 0x32) {
        shape = unused((bytes[2] & 15) == 0);
        target = lift.Add(Op::add, 64, lift.Read(s),
                          lift.Add(Op::shl, 64, lift.Read(bytes[2] >> 4), lift.Constant(3)));
      } else {
        target = lift.Add(Op::add, 64, lift.Read(s), lift.Constant(signed16(2)));
      }

      if (opcode == 0x31)
        lift.Memory(Op::store, target, lift.Read(d));
      else
        lift.Write(d, lift.Memory(Op::load, target));
      break;
    }
    case 0x33:
      shape = unused(s == 0);
      model = ir::MemoryModel::atomic_scalar_reference;
      lift.Write(d, lift.Memory(Op::load,
                                lift.Add(Op::image_address, 64, 0, 0, 0, address + signed16(2))));
      break;
    case 0x40:
      lift.Write(d, lift.Add(Op::select, 64, lift.Add(Op::equal, 1, lift.Read(s), lift.Constant(0)),
                             lift.Constant(1), lift.Constant(0)));
      break;
    case 0x50:
    case 0x51:
    case 0x52:
    case 0x56: {
      // The source register sits in the low nibble, so d names nothing here.
      shape = unused(d == 0);
      ValueId condition;
      std::uint64_t displacement;
      if (opcode == 0x52) {
        condition = lift.Add(Op::bit_not, 1,
                             lift.Add(Op::unsigned_less, 1, lift.Read(s), lift.Constant(bytes[2])));
        displacement = signed16(3);
      } else if (opcode == 0x56) {
        // x > N as "not below and not equal", the one shape the generic
        // table-bound rule recognizes; x >= N (bgeu) does not bound a dispatch.
        const auto value = lift.Read(s);
        const auto limit = lift.Constant(bytes[2]);
        const auto below = lift.Add(Op::unsigned_less, 1, value, limit);
        const auto equal =
            lift.Add(Op::equal, 1, lift.Add(Op::sub, 64, value, limit), lift.Constant(0));
        condition = lift.Add(Op::bit_and, 1, lift.Add(Op::bit_not, 1, below),
                             lift.Add(Op::bit_not, 1, equal));
        displacement = signed16(3);
      } else {
        condition = lift.Add(Op::equal, 1, lift.Read(s), lift.Constant(0));
        if (opcode == 0x51) condition = lift.Add(Op::bit_not, 1, condition);
        displacement = signed16(2);
      }

      transfer = ir::Transfer{
          ir::TransferKind::conditional, relative(displacement), condition, relative(length), {}};
      break;
    }
    case 0x53:
      transfer = ir::Transfer{ir::TransferKind::jump, relative(signed16(1)), {}, {}, {}};
      break;
    case 0x54: {
      const auto continuation = relative(length);
      lift.Write(kLink, continuation);
      transfer = ir::Transfer{ir::TransferKind::call, relative(signed16(1)), {}, {}, continuation};
      break;
    }
    case 0x55:
      shape = unused(d == 0);
      transfer = ir::Transfer{ir::TransferKind::jump, lift.Read(s), {}, {}, {}};
      break;
    case 0x57: {
      // callr s: link the next instruction and call through register s. The
      // target is read before the link is written, so s may be the link.
      shape = unused(d == 0);
      const auto target = lift.Read(s);
      const auto continuation = relative(length);
      lift.Write(kLink, continuation);
      transfer = ir::Transfer{ir::TransferKind::call, target, {}, {}, continuation};
      break;
    }
  }

  if (shape != DecodeDecline::none) return {std::nullopt, length, shape};
  if (lift.reason != DecodeDecline::none) return {std::nullopt, length, lift.reason};
  return {ir::Group(address, std::vector<std::uint8_t>(bytes.begin(), bytes.begin() + length),
                    std::move(lift.nodes), std::move(lift.writes), model, transfer),
          length, DecodeDecline::none};
}

}  // namespace nyx::toy
