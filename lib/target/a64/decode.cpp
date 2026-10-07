#include "nyx/target/a64/decode.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <utility>

namespace nyx::a64 {
namespace {

// BTI is a landing-pad marker in the HINT space. It changes nothing a run can
// observe unless branch target identification is enforcing, which the run
// already declares it is not, so it models as the no-op it then is. The other
// hints are left unsupported: several of them do have effects.
constexpr bool IsBranchTargetHint(std::uint32_t word) { return (word & 0xffffff3f) == 0xd503241f; }

// The eight PACIA/PACIB/AUTIA/AUTIB forms that sign or authenticate the link
// register against SP or zero. Signing leaves the address bits alone and puts
// a code in the bits above them; authenticating takes it off again. Where
// pointer authentication is not enforcing neither is observable, which the run
// declares alongside BTI and GCS, and they model as the no-ops they then are.
// The forms taking a general register are still unsupported.
constexpr bool IsPointerAuthHint(std::uint32_t word) { return (word & 0xffffff1f) == 0xd503231f; }

using ir::Op;
using ir::ValueId;

std::uint32_t Word(std::array<std::uint8_t, 4> bytes) {
  return std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8) |
         (std::uint32_t(bytes[2]) << 16) | (std::uint32_t(bytes[3]) << 24);
}

DecodeDecline Convert(BudgetDecline reason) {
  if (reason == BudgetDecline::work_limit) return DecodeDecline::work_limit;
  if (reason == BudgetDecline::byte_limit) return DecodeDecline::byte_limit;
  return DecodeDecline::none;
}

std::uint64_t SignExtend(std::uint64_t value, unsigned bits) {
  const auto sign = std::uint64_t{1} << (bits - 1);
  return (value ^ sign) - sign;
}

class Lift {
 public:
  explicit Lift(Budget& budget, std::size_t node_capacity = 64) : budget_(budget) {
    reason = Convert(
        budget_.try_consume({1, node_capacity * sizeof(ir::Node) + 5 * sizeof(ir::Write) + 4}));
    if (reason == DecodeDecline::none) {
      nodes.reserve(node_capacity);
      writes.reserve(5);
    }
  }

  ValueId Add(Op op, unsigned width, ValueId a = 0, ValueId b = 0, ValueId c = 0,
              std::uint64_t immediate = 0, ir::StorageId storage = 0) {
    if (reason != DecodeDecline::none) return 0;
    reason = Convert(budget_.try_consume({1, 0}));
    if (reason != DecodeDecline::none) return 0;

    // A target instruction has a finite expansion; never silently grow the charged capacity.
    if (nodes.size() == nodes.capacity()) {
      reason = DecodeDecline::byte_limit;
      return 0;
    }

    nodes.push_back({op, width, {a, b, c}, immediate, storage});
    return static_cast<ValueId>(nodes.size() - 1);
  }

  ValueId Constant(unsigned width, std::uint64_t value) {
    return Add(Op::constant, width, 0, 0, 0, value);
  }

  ValueId Read(unsigned reg, unsigned width, bool sp = false) {
    if (reg == 31 && !sp) return Constant(width, 0);
    const auto value = Add(Op::read, 64, 0, 0, 0, 0, reg);
    return width == 64 ? value : Add(Op::extract, width, value);
  }

  ValueId ReadQ(unsigned reg) { return Add(Op::read, 128, 0, 0, 0, 0, kQ0 + reg); }

  void StoreAligned(unsigned width, ValueId address, ValueId value) {
    Add(Op::store, width, address, value);
    if (reason == DecodeDecline::none) {
      nodes.back().access.alignment = width / 8;
      nodes.back().access.decline_on_unaligned = true;
    }
  }

  void Write(unsigned reg, ValueId value, unsigned width, bool sp = false) {
    if (reg == 31 && !sp) return;
    if (width == 32) value = Add(Op::zext, 64, value);
    if (reason != DecodeDecline::none) return;
    if (writes.size() == writes.capacity()) {
      reason = DecodeDecline::byte_limit;
      return;
    }

    writes.push_back({reg, value});
  }

  ValueId Shift(ValueId value, unsigned width, unsigned kind, unsigned amount) {
    if (amount == 0) return value;
    const auto count = Constant(width, amount);
    if (kind < 3)
      return Add(kind == 0 ? Op::shl : kind == 1 ? Op::lshr : Op::ashr, width, value, count);
    const auto low = Add(Op::lshr, width, value, count);
    const auto high = Add(Op::shl, width, value, Constant(width, width - amount));
    return Add(Op::bit_or, width, low, high);
  }

  void Flags(ValueId lhs, ValueId rhs, ValueId result, unsigned width, bool subtract,
             bool logical = false) {
    const auto [n, z, c, v] = FlagValues(lhs, rhs, result, width, subtract, logical);
    Write(kN, n, 1);
    Write(kZ, z, 1);
    Write(kC, c, 1);
    Write(kV, v, 1);
  }

  // ADC and SBC add the carry flag in, so the carry-out is not the carry of one
  // addition: it is the carry of lhs + rhs, or of that sum plus the carry bit.
  // SBC inverts its second operand first, which makes both forms the same
  // addition and yields ARM's SBCS flag definition without a separate case.
  void CarryFlags(ValueId lhs, ValueId rhs, ValueId sum, ValueId result, unsigned width) {
    const auto n = Add(Op::extract, 1, result, 0, 0, width - 1);
    const auto z = Add(Op::equal, 1, result, Constant(width, 0));
    const auto c = Add(Op::bit_or, 1, Add(Op::unsigned_less, 1, sum, lhs),
                       Add(Op::unsigned_less, 1, result, sum));
    const auto signs = Add(Op::bit_not, width, Add(Op::bit_xor, width, lhs, rhs));
    const auto changed = Add(Op::bit_xor, width, lhs, result);
    const auto v = Add(Op::extract, 1, Add(Op::bit_and, width, signs, changed), 0, 0, width - 1);
    Write(kN, n, 1);
    Write(kZ, z, 1);
    Write(kC, c, 1);
    Write(kV, v, 1);
  }

  std::array<ValueId, 4> FlagValues(ValueId lhs, ValueId rhs, ValueId result, unsigned width,
                                    bool subtract, bool logical = false) {
    const auto n = Add(Op::extract, 1, result, 0, 0, width - 1);
    const auto z = Add(Op::equal, 1, result, Constant(width, 0));
    ValueId c, v;
    if (logical) {
      c = Constant(1, 0);
      v = c;
    } else {
      c = subtract ? Add(Op::bit_not, 1, Add(Op::unsigned_less, 1, lhs, rhs))
                   : Add(Op::unsigned_less, 1, result, lhs);
      auto signs = Add(Op::bit_xor, width, lhs, rhs);
      if (!subtract) signs = Add(Op::bit_not, width, signs);
      const auto changed = Add(Op::bit_xor, width, lhs, result);
      v = Add(Op::extract, 1, Add(Op::bit_and, width, signs, changed), 0, 0, width - 1);
    }

    return {n, z, c, v};
  }

  // A SIMD&FP load narrower than Q zeroes the rest of the register. A Q access
  // is two ordered 64-bit accesses, the boundary the selected QEMU backend
  // faults at: a store's first half stays written when its second half faults.
  ValueId LoadVector(unsigned width, ValueId address) {
    if (width < 128) return Add(Op::zext, 128, Add(Op::load, width, address));
    const auto upper_address = Add(Op::add, 64, address, Constant(64, 8));
    const auto low = Add(Op::load, 64, address);
    const auto high = Add(Op::load, 64, upper_address);
    return Add(Op::bit_or, 128, Add(Op::zext, 128, low),
               Add(Op::shl, 128, Add(Op::zext, 128, high), Constant(128, 64)));
  }

  void StoreVector(unsigned width, ValueId address, ValueId value) {
    if (width < 128) {
      Add(Op::store, width, address, Add(Op::extract, width, value));
      return;
    }

    const auto upper_address = Add(Op::add, 64, address, Constant(64, 8));
    Add(Op::store, 64, address, Add(Op::extract, 64, value));
    Add(Op::store, 64, upper_address, Add(Op::extract, 64, value, 0, 0, 64));
  }

  ValueId Condition(unsigned cond) {
    auto flag = [&](unsigned reg) { return Add(Op::read, 1, 0, 0, 0, 0, reg); };
    ValueId result;
    switch (cond >> 1) {
      case 0:
        result = flag(kZ);
        break;
      case 1:
        result = flag(kC);
        break;
      case 2:
        result = flag(kN);
        break;
      case 3:
        result = flag(kV);
        break;
      case 4:
        result = Add(Op::bit_and, 1, flag(kC), Add(Op::bit_not, 1, flag(kZ)));
        break;
      case 5:
        result = Add(Op::equal, 1, flag(kN), flag(kV));
        break;
      case 6:
        result = Add(Op::bit_and, 1, Add(Op::equal, 1, flag(kN), flag(kV)),
                     Add(Op::bit_not, 1, flag(kZ)));
        break;
      default:
        result = Constant(1, 1);
        break;
    }

    if ((cond & 1) != 0 && cond != 15) result = Add(Op::bit_not, 1, result);
    return result;
  }

  DecodeDecline reason = DecodeDecline::none;
  std::vector<ir::Node> nodes;
  std::vector<ir::Write> writes;

 private:
  Budget& budget_;
};

}  // namespace

bool OpaqueNormalFallthrough(std::array<std::uint8_t, 4> bytes) {
  const auto word = Word(bytes);

  // Each entry names a family this decoder declines to give semantics for and
  // which reaches the next instruction whatever it computes. Membership is
  // decided by encoding, not by the group an encoding sits in: `unsupported`
  // is the decoder's catch-all and holds unallocated words too, and those
  // trap rather than fall through.
  return (word & 0xbffffc00) == 0x885ffc00 ||  // exclusive and acquiring loads
         (word & 0xbfe07c00) == 0x88007c00;    // exclusive and releasing stores
}

bool OpaqueTrap(std::array<std::uint8_t, 4> bytes) {
  return (Word(bytes) & 0xffe0001f) == 0xd4200000;
}

DecodeResult Decode(std::uint64_t address, std::array<std::uint8_t, 4> bytes, Budget& budget,
                    Options options) {
  if (address % 4 != 0 || address > std::numeric_limits<std::uint64_t>::max() - 3) {
    return {std::nullopt, DecodeDecline::invalid_location};
  }

  const std::uint32_t word = Word(bytes);
  const bool cmeq_vector = (word & 0xbf20fc00) == 0x2e208c00;

  // A sixteen-lane comparison expands into more nodes than scalar instructions.
  // Charge its bounded capacity before allocating, including on a later decline.
  Lift lift(budget, cmeq_vector ? 128 : 64);
  if (lift.reason != DecodeDecline::none) return {std::nullopt, lift.reason};
  const unsigned width = (word >> 31) != 0 ? 64 : 32;
  const unsigned rd = word & 31;
  const unsigned rn = (word >> 5) & 31;
  const unsigned rm = (word >> 16) & 31;
  const bool subtract = ((word >> 30) & 1) != 0;
  const bool flags = ((word >> 29) & 1) != 0;
  const bool scalar_memory = options.memory_profile != MemoryProfile::none;
  auto memory_model = ir::MemoryModel::unspecified;
  std::optional<ir::Transfer> transfer;
  const auto relative = [&](ValueId pc, std::uint64_t offset) {
    return lift.Add(Op::add, 64, pc, lift.Constant(64, offset));
  };

  if ((word & 0x7c000000) == 0x14000000) {
    const auto pc = lift.Add(Op::image_address, 64, 0, 0, 0, address);
    const auto destination = relative(pc, SignExtend(word & 0x3ffffff, 26) << 2);
    if ((word & (1U << 31)) != 0) {
      const auto continuation = relative(pc, 4);
      lift.Write(30, continuation, 64);
      transfer = ir::Transfer{ir::TransferKind::call, destination, {}, {}, continuation};
    } else {
      transfer = ir::Transfer{ir::TransferKind::jump, destination, {}, {}, {}};
    }
  } else if ((word & 0xff000010) == 0x54000000 || (word & 0x7e000000) == 0x34000000 ||
             (word & 0x7e000000) == 0x36000000) {
    const auto pc = lift.Add(Op::image_address, 64, 0, 0, 0, address);
    ValueId condition;
    std::uint64_t displacement;
    if ((word & 0xff000010) == 0x54000000) {
      condition = lift.Condition(word & 15);
      displacement = SignExtend((word >> 5) & 0x7ffff, 19) << 2;
    } else if ((word & 0x7e000000) == 0x34000000) {
      condition = lift.Add(Op::equal, 1, lift.Read(rd, width), lift.Constant(width, 0));
      if ((word & (1U << 24)) != 0) condition = lift.Add(Op::bit_not, 1, condition);
      displacement = SignExtend((word >> 5) & 0x7ffff, 19) << 2;
    } else {
      const unsigned bit = ((word >> 26) & 32) | ((word >> 19) & 31);
      condition = lift.Add(Op::extract, 1, lift.Read(rd, width), 0, 0, bit);
      if ((word & (1U << 24)) == 0) condition = lift.Add(Op::bit_not, 1, condition);
      displacement = SignExtend((word >> 5) & 0x3fff, 14) << 2;
    }

    const auto destination = relative(pc, displacement);
    const auto fallthrough = relative(pc, 4);
    transfer = ir::Transfer{ir::TransferKind::conditional, destination, condition, fallthrough, {}};
  } else if ((word & 0xfffffc1f) == 0xd61f0000 || (word & 0xfffffc1f) == 0xd63f0000 ||
             (word & 0xfffffc1f) == 0xd65f0000) {
    // A BLR X30 target reads entry X30 before the explicit final link write.
    const auto destination = lift.Read(rn, 64);
    if ((word & 0xfffffc1f) == 0xd63f0000) {
      const auto pc = lift.Add(Op::image_address, 64, 0, 0, 0, address);
      const auto continuation = relative(pc, 4);
      lift.Write(30, continuation, 64);
      transfer = ir::Transfer{ir::TransferKind::call, destination, {}, {}, continuation};
    } else {
      const auto kind =
          (word & 0xfffffc1f) == 0xd65f0000 ? ir::TransferKind::return_ : ir::TransferKind::jump;
      transfer = ir::Transfer{kind, destination, {}, {}, {}};
    }
  } else if ((word & 0xff000000) == 0x58000000) {
    if (!scalar_memory) return {std::nullopt, DecodeDecline::unsupported};
    memory_model = ir::MemoryModel::atomic_scalar_reference;
    const auto displacement = SignExtend((word >> 5) & 0x7ffff, 19) << 2;
    const auto target = lift.Add(Op::image_address, 64, 0, 0, 0, address + displacement);
    lift.Write(rd, lift.Add(Op::load, 64, target), 64);
  } else if ((word & 0x1f000000) == 0x10000000) {
    const bool page = (word & (1U << 31)) != 0;
    const auto displacement = SignExtend((((word >> 5) & 0x7ffffU) << 2) | ((word >> 29) & 3), 21);
    auto pc = lift.Add(Op::image_address, 64, 0, 0, 0, address);
    if (page) pc = lift.Add(Op::bit_and, 64, pc, lift.Constant(64, ~std::uint64_t{0xfff}));
    const auto offset = lift.Constant(64, page ? displacement << 12 : displacement);
    lift.Write(rd, lift.Add(Op::add, 64, pc, offset), 64);
  } else if ((word & 0xbffffc00) == 0x885ffc00 || (word & 0xbfe07c00) == 0x88007c00) {
    if (options.memory_profile != MemoryProfile::concrete_exclusive_scalar || rn == 31) {
      return {std::nullopt, DecodeDecline::unsupported};
    }

    memory_model = ir::MemoryModel::qemu_exclusive_scalar_reference;
    const unsigned access_width = (word & (1U << 30)) != 0 ? 64 : 32;
    const auto base = lift.Read(rn, 64);
    if ((word & 0xbffffc00) == 0x885ffc00) {
      const auto loaded = lift.Add(Op::exclusive_load, access_width, base);
      if (lift.reason == DecodeDecline::none) {
        lift.nodes.back().access = {ir::ByteOrder::little, access_width / 8, true};
      }

      lift.Write(rd, loaded, access_width);
    } else {
      const auto status = lift.Add(Op::exclusive_store, 32, base, lift.Read(rd, access_width));
      if (lift.reason == DecodeDecline::none) {
        lift.nodes.back().access = {ir::ByteOrder::little, access_width / 8, true};
      }

      lift.Write(rm, status, 32);
    }
  } else if ((word & 0xfffff0ff) == 0xd503305f) {
    if (options.memory_profile != MemoryProfile::concrete_exclusive_scalar) {
      return {std::nullopt, DecodeDecline::unsupported};
    }

    memory_model = ir::MemoryModel::qemu_exclusive_scalar_reference;
    lift.Add(Op::exclusive_clear, 1);
  } else if ((word & 0x3fbffc00) == 0x089ffc00) {
    if (!scalar_memory) {
      return {std::nullopt, DecodeDecline::unsupported};
    }

    memory_model = ir::MemoryModel::atomic_scalar_reference;
    const unsigned access_width = 8U << (word >> 30);
    const auto base = lift.Read(rn, 64, true);
    if ((word & (1U << 22)) != 0) {
      auto loaded = lift.Add(Op::load, access_width, base);
      if (lift.reason == DecodeDecline::none) {
        lift.nodes.back().access = {ir::ByteOrder::little, access_width / 8, true};
      }

      if (access_width < 64) loaded = lift.Add(Op::zext, 64, loaded);
      lift.Write(rd, loaded, 64);
    } else {
      lift.StoreAligned(access_width, base, lift.Read(rd, access_width));
    }
  } else if ((word & 0x3a000000) == 0x28000000) {
    if (!scalar_memory) {
      return {std::nullopt, DecodeDecline::unsupported};
    }

    const bool vector = (word & (1U << 26)) != 0;
    const unsigned mode = (word >> 23) & 3;
    const unsigned opc = word >> 30;
    const bool load = (word & (1U << 22)) != 0;
    const unsigned rt2 = (word >> 10) & 31;
    if (mode == 0 || (!vector && opc == 1 && !load)) {
      return {std::nullopt, DecodeDecline::unsupported};
    }

    if (opc == 3) return {std::nullopt, DecodeDecline::invalid_encoding};
    const bool writeback = mode != 2;

    // SIMD&FP pair registers are never the general base register.
    if ((load && rd == rt2) || (!vector && writeback && rn != 31 && (rn == rd || rn == rt2))) {
      return {std::nullopt, DecodeDecline::invalid_encoding};
    }

    memory_model = ir::MemoryModel::atomic_scalar_reference;
    const unsigned element_width = vector ? 32U << opc : opc == 2 ? 64 : 32;
    const unsigned scale = std::countr_zero(element_width / 8);
    const auto offset = SignExtend((word >> 15) & 0x7f, 7) << scale;
    const auto base = lift.Read(rn, 64, true);
    const auto updated = lift.Add(Op::add, 64, base, lift.Constant(64, offset));
    const auto first_address = mode == 1 ? base : updated;
    if (vector) {
      // QEMU 10.2.1 does not merge SIMD&FP pairs: each element is its own
      // access, so a second-element store fault leaves the first written.
      const auto second_address =
          lift.Add(Op::add, 64, first_address, lift.Constant(64, element_width / 8));
      if (load) {
        const auto first = lift.LoadVector(element_width, first_address);
        const auto second = lift.LoadVector(element_width, second_address);
        lift.Write(kQ0 + rd, first, 128);
        lift.Write(kQ0 + rt2, second, 128);
      } else {
        lift.StoreVector(element_width, first_address, lift.ReadQ(rd));
        lift.StoreVector(element_width, second_address, lift.ReadQ(rt2));
      }
    } else if (load) {
      // QEMU 10.2.1 trans_LDP uses one 64-bit access for W pairs and LDPSW.
      // Keep that concrete fault boundary; it is not architectural fault authority.
      ValueId first, second;
      if (element_width == 32) {
        const auto pair = lift.Add(Op::load, 64, first_address);
        first = lift.Add(Op::extract, 32, pair);
        second = lift.Add(Op::extract, 32, pair, 0, 0, 32);
      } else {
        const auto second_address = lift.Add(Op::add, 64, first_address, lift.Constant(64, 8));
        first = lift.Add(Op::load, 64, first_address);
        second = lift.Add(Op::load, 64, second_address);
      }

      if (opc == 1) {
        const auto count = lift.Constant(64, 32);
        first = lift.Add(Op::ashr, 64, lift.Add(Op::shl, 64, lift.Add(Op::zext, 64, first), count),
                         count);
        second = lift.Add(Op::ashr, 64,
                          lift.Add(Op::shl, 64, lift.Add(Op::zext, 64, second), count), count);
      }

      lift.Write(rd, first, opc == 1 ? 64 : element_width);
      lift.Write(rt2, second, opc == 1 ? 64 : element_width);
    } else {
      const auto first = lift.Read(rd, element_width);
      const auto second = lift.Read(rt2, element_width);
      if (element_width == 32) {
        // Match QEMU's concatenated 64-bit W-pair store, including cross-page
        // failure. Two 32-bit stores incorrectly publish a prefix on that fault.
        const auto low = lift.Add(Op::zext, 64, first);
        const auto high =
            lift.Add(Op::shl, 64, lift.Add(Op::zext, 64, second), lift.Constant(64, 32));
        lift.Add(Op::store, 64, first_address, lift.Add(Op::bit_or, 64, low, high));
      } else {
        const auto second_address = lift.Add(Op::add, 64, first_address, lift.Constant(64, 8));

        // The selected backend exposes the first X-pair element on a second
        // element fault. Keep its two scalar effects in program order.
        lift.Add(Op::store, 64, first_address, first);
        lift.Add(Op::store, 64, second_address, second);
      }
    }

    if (writeback) lift.Write(rn, updated, 64, true);
  } else if ((word & 0x3b000000) == 0x39000000 || (word & 0x3b200000) == 0x38000000 ||
             (word & 0x3b200c00) == 0x38200800) {
    if (!scalar_memory) {
      return {std::nullopt, DecodeDecline::unsupported};
    }

    const bool vector = (word & (1U << 26)) != 0;
    const bool unsigned_offset = (word & 0x3b000000) == 0x39000000;
    const bool register_offset = (word & 0x3b200c00) == 0x38200800;
    const unsigned mode = (word >> 10) & 3;
    if (!unsigned_offset && !register_offset && mode == 2)
      return {std::nullopt, DecodeDecline::unsupported};
    const unsigned extend = (word >> 13) & 7;
    if (register_offset && (extend & 2) == 0)
      return {std::nullopt, DecodeDecline::invalid_encoding};
    const unsigned size = (word >> 30) & 3;
    const unsigned opc = (word >> 22) & 3;
    if (vector) {
      if (opc >= 2 && size != 0) return {std::nullopt, DecodeDecline::invalid_encoding};
    } else {
      if (opc == 2 && size == 3) return {std::nullopt, DecodeDecline::unsupported};
      if (opc == 3 && size >= 2) return {std::nullopt, DecodeDecline::invalid_encoding};
    }

    const bool writeback = !unsigned_offset && !register_offset && (mode == 1 || mode == 3);
    if (!vector && writeback && rn != 31 && rn == rd) {
      return {std::nullopt, DecodeDecline::invalid_encoding};
    }

    memory_model = ir::MemoryModel::atomic_scalar_reference;

    // A SIMD&FP transfer takes opc<1> as a fifth size bit, which selects Q.
    const unsigned scale = vector && opc >= 2 ? 4 : size;
    const unsigned memory_width = 8U << scale;
    const unsigned register_width = size == 3 || opc == 2 ? 64 : 32;
    const auto base = lift.Read(rn, 64, true);
    ValueId offset;
    if (register_offset) {
      const unsigned index_width = (extend & 1) != 0 ? 64 : 32;
      offset = lift.Read(rm, index_width);
      if (index_width == 32) {
        offset = lift.Add(Op::zext, 64, offset);
        if ((extend & 4) != 0) {
          const auto count = lift.Constant(64, 32);
          offset = lift.Add(Op::ashr, 64, lift.Add(Op::shl, 64, offset, count), count);
        }
      }

      if ((word & (1U << 12)) != 0) offset = lift.Shift(offset, 64, 0, scale);
    } else {
      const auto immediate = unsigned_offset ? std::uint64_t((word >> 10) & 0xfff) << scale
                                             : SignExtend((word >> 12) & 0x1ff, 9);
      offset = lift.Constant(64, immediate);
    }

    const auto updated = lift.Add(Op::add, 64, base, offset);
    const auto access_address = !unsigned_offset && !register_offset && mode == 1 ? base : updated;
    if (vector) {
      if ((opc & 1) != 0) {
        lift.Write(kQ0 + rd, lift.LoadVector(memory_width, access_address), 128);
      } else {
        lift.StoreVector(memory_width, access_address, lift.ReadQ(rd));
      }
    } else if (opc == 0) {
      auto value = lift.Read(rd, register_width);
      if (memory_width < register_width) value = lift.Add(Op::extract, memory_width, value);
      lift.Add(Op::store, memory_width, access_address, value);
    } else {
      auto value = lift.Add(Op::load, memory_width, access_address);
      if (memory_width < register_width) {
        value = lift.Add(Op::zext, register_width, value);
        if (opc >= 2) {
          const auto count = lift.Constant(register_width, register_width - memory_width);
          value = lift.Add(Op::ashr, register_width,
                           lift.Add(Op::shl, register_width, value, count), count);
        }
      }

      lift.Write(rd, value, register_width);
    }

    if (writeback) lift.Write(rn, updated, 64, true);
  } else if ((word & 0x1f800000) == 0x11000000 || (word & 0x1f200000) == 0x0b000000) {
    const bool immediate = (word & 0x1f800000) == 0x11000000;
    const auto lhs = lift.Read(rn, width, immediate);
    ValueId rhs;
    if (immediate) {
      rhs = lift.Constant(width, std::uint64_t((word >> 10) & 0xfff) << (((word >> 22) & 1) * 12));
    } else {
      const unsigned shift = (word >> 22) & 3;
      const unsigned amount = (word >> 10) & 63;
      if (shift == 3 || amount >= width) return {std::nullopt, DecodeDecline::invalid_encoding};
      rhs = lift.Shift(lift.Read(rm, width), width, shift, amount);
    }

    const auto result = lift.Add(subtract ? Op::sub : Op::add, width, lhs, rhs);
    lift.Write(rd, result, width, immediate && !flags);
    if (flags) lift.Flags(lhs, rhs, result, width, subtract);
  } else if ((word & 0x1fe0fc00) == 0x1a000000) {
    // Add/subtract with carry. The subtract form is the same addition with its
    // second operand inverted, so one path covers ADC, ADCS, SBC and SBCS.
    const auto lhs = lift.Read(rn, width);
    auto rhs = lift.Read(rm, width);
    if (subtract) rhs = lift.Add(Op::bit_not, width, rhs);
    const auto carry = lift.Add(Op::zext, width, lift.Add(Op::read, 1, 0, 0, 0, 0, kC));
    const auto sum = lift.Add(Op::add, width, lhs, rhs);
    const auto result = lift.Add(Op::add, width, sum, carry);
    lift.Write(rd, result, width);
    if (flags) lift.CarryFlags(lhs, rhs, sum, result, width);
  } else if ((word & 0x1fe00000) == 0x0b200000) {
    // Extended register: Rm narrowed to a byte, halfword, word or doubleword,
    // zero- or sign-extended to the operation width, then shifted left by at
    // most four. Unlike the shifted form, Rn and a non-flag Rd may be SP.
    const unsigned option = (word >> 13) & 7;
    const unsigned amount = (word >> 10) & 7;
    if (amount > 4) return {std::nullopt, DecodeDecline::invalid_encoding};
    const unsigned from = 8U << (option & 3);
    const unsigned read = width == 64 && from == 64 ? 64 : 32;
    const unsigned kept = std::min(from, width);
    auto rhs = lift.Read(rm, read);
    if (kept < read) rhs = lift.Add(Op::extract, kept, rhs);
    if (kept < width) {
      rhs = lift.Add(Op::zext, width, rhs);
      if (option >= 4) {
        const auto count = lift.Constant(width, width - kept);
        rhs = lift.Add(Op::ashr, width, lift.Add(Op::shl, width, rhs, count), count);
      }
    }

    rhs = lift.Shift(rhs, width, 0, amount);
    const auto lhs = lift.Read(rn, width, true);
    const auto result = lift.Add(subtract ? Op::sub : Op::add, width, lhs, rhs);
    lift.Write(rd, result, width, !flags);
    if (flags) lift.Flags(lhs, rhs, result, width, subtract);
  } else if ((word & 0x1f000000) == 0x0a000000) {
    const unsigned amount = (word >> 10) & 63;
    if (amount >= width) return {std::nullopt, DecodeDecline::invalid_encoding};
    auto rhs = lift.Shift(lift.Read(rm, width), width, (word >> 22) & 3, amount);
    if ((word & (1U << 21)) != 0) rhs = lift.Add(Op::bit_not, width, rhs);
    const auto lhs = lift.Read(rn, width);
    const unsigned opc = (word >> 29) & 3;
    const auto result = lift.Add(opc == 1   ? Op::bit_or
                                 : opc == 2 ? Op::bit_xor
                                            : Op::bit_and,
                                 width, lhs, rhs);
    lift.Write(rd, result, width);
    if (opc == 3) lift.Flags(lhs, rhs, result, width, false, true);
  } else if ((word & 0x1f800000) == 0x12000000) {
    const unsigned n = (word >> 22) & 1;
    const unsigned imms = (word >> 10) & 63;
    const unsigned immr = (word >> 16) & 63;
    const unsigned encoded_size = (n << 6) | (~imms & 63);
    if ((width == 32 && n) || encoded_size < 2) {
      return {std::nullopt, DecodeDecline::invalid_encoding};
    }

    const unsigned element = 1U << (std::bit_width(encoded_size) - 1);
    const unsigned ones = (imms & (element - 1)) + 1;
    if (ones == element) return {std::nullopt, DecodeDecline::invalid_encoding};
    const auto charged = Convert(budget.try_consume({width, 0}));
    if (charged != DecodeDecline::none) return {std::nullopt, charged};

    // Build repeated rotated runs directly, avoiding a host shift by 64 for
    // unrotated full-width elements. High immr bits are ignored per element.
    std::uint64_t mask = 0;
    for (unsigned bit = 0; bit < width; ++bit) {
      if (((bit + immr) & (element - 1)) < ones) mask |= UINT64_C(1) << bit;
    }

    const unsigned opc = (word >> 29) & 3;
    const auto lhs = lift.Read(rn, width);
    const auto rhs = lift.Constant(width, mask);
    const auto result = lift.Add(opc == 1   ? Op::bit_or
                                 : opc == 2 ? Op::bit_xor
                                            : Op::bit_and,
                                 width, lhs, rhs);
    // Nonflag logical immediates allow SP as destination, but Rn=31 is ZR.
    lift.Write(rd, result, width, opc != 3);
    if (opc == 3) lift.Flags(lhs, rhs, result, width, false, true);
  } else if ((word & 0x1f800000) == 0x12800000) {
    const unsigned opc = (word >> 29) & 3;
    const unsigned shift = ((word >> 21) & 3) * 16;
    if (opc == 1 || shift >= width) return {std::nullopt, DecodeDecline::invalid_encoding};
    const auto immediate = std::uint64_t((word >> 5) & 0xffff) << shift;
    auto result = lift.Constant(width, opc == 0 ? ~immediate : immediate);
    if (opc == 3) {
      const auto preserved = lift.Add(Op::bit_and, width, lift.Read(rd, width),
                                      lift.Constant(width, ~(std::uint64_t(0xffff) << shift)));
      result = lift.Add(Op::bit_or, width, preserved, result);
    }

    lift.Write(rd, result, width);
  } else if ((word & 0x1f800000) == 0x13000000) {
    const unsigned opc = (word >> 29) & 3;
    const unsigned immr = (word >> 16) & 63;
    const unsigned imms = (word >> 10) & 63;
    if (((word >> 22) & 1) != (width == 64 ? 1U : 0U) || immr >= width || imms >= width ||
        opc == 3) {
      return {std::nullopt, DecodeDecline::invalid_encoding};
    }

    ValueId result;
    const auto value = lift.Read(rn, width);
    if (opc == 2 && imms + 1 == immr) {
      result = lift.Shift(value, width, 0, width - immr);
    } else if ((opc == 0 || opc == 2) && imms == width - 1) {
      result = lift.Shift(value, width, opc == 0 ? 2 : 1, immr);
    } else {
      const auto charged = Convert(budget.try_consume({width, 0}));
      if (charged != DecodeDecline::none) return {std::nullopt, charged};
      std::uint64_t wmask = 0, tmask = 0;
      const unsigned top = (imms - immr) & (width - 1);

      // Bitfield DecodeBitMasks uses a full-register element and permits all
      // ones, unlike logical immediates. Bit construction avoids shift-by-64.
      for (unsigned bit = 0; bit < width; ++bit) {
        if (((bit + immr) & (width - 1)) <= imms) wmask |= UINT64_C(1) << bit;
        if (bit <= top) tmask |= UINT64_C(1) << bit;
      }

      const auto mask = wmask & tmask;

      // UBFX: the bits the rotation brings round land at or above width - immr,
      // above the mask's top bit imms - immr, so a plain shift reads the same.
      const auto aligned = opc == 2 && imms >= immr ? lift.Shift(value, width, 1, immr)
                                                    : lift.Shift(value, width, 3, immr);
      result = lift.Add(Op::bit_and, width, aligned, lift.Constant(width, mask));
      if (opc == 1) {
        const auto preserved =
            lift.Add(Op::bit_and, width, lift.Read(rd, width), lift.Constant(width, ~mask));
        result = lift.Add(Op::bit_or, width, preserved, result);
      } else if (opc == 0) {
        const auto sign = lift.Add(Op::extract, 1, value, 0, 0, imms);
        const auto extension = lift.Add(Op::select, width, sign, lift.Constant(width, ~tmask),
                                        lift.Constant(width, 0));
        result = lift.Add(Op::bit_or, width, extension, result);
      }
    }

    lift.Write(rd, result, width);
  } else if ((word & 0x7fa00000) == 0x13800000) {
    const unsigned amount = (word >> 10) & 63;
    if (((word >> 22) & 1) != (width == 64 ? 1U : 0U) || amount >= width) {
      return {std::nullopt, DecodeDecline::invalid_encoding};
    }

    auto result = lift.Read(rm, width);
    if (amount != 0) {
      const auto low = lift.Shift(result, width, 1, amount);
      const auto high = lift.Shift(lift.Read(rn, width), width, 0, width - amount);
      result = lift.Add(Op::bit_or, width, low, high);
    }

    lift.Write(rd, result, width);
  } else if ((word & 0x7fe0f800) == 0x1ac00800) {
    // UDIV and SDIV. A zero divisor yields zero rather than trapping.
    const auto op = (word & (1U << 10)) != 0 ? Op::sdiv : Op::udiv;
    lift.Write(rd, lift.Add(op, width, lift.Read(rn, width), lift.Read(rm, width)), width);
  } else if ((word & 0xff60fc00) == 0x9b407c00) {
    // UMULH and SMULH. Ra is fixed at 31; other values are left unsupported.
    const auto op = (word & (1U << 23)) != 0 ? Op::umulh : Op::smulh;
    lift.Write(rd, lift.Add(op, 64, lift.Read(rn, 64), lift.Read(rm, 64)), 64);
  } else if ((word & 0x7fffe000) == 0x5ac00000 && ((word >> 10) & 7) < 6 &&
             (width == 64 || ((word >> 10) & 7) != 3)) {
    // RBIT, REV16, REV32, REV, CLZ and CLS. The byte reversals swap ever wider
    // neighbours up to their container: halfwords for REV16, words for REV32
    // and the 32-bit REV, the whole register for the 64-bit one.
    const unsigned opc = (word >> 10) & 7;
    const auto value = lift.Read(rn, width);
    ValueId result;
    if (opc == 0) {
      result = lift.Add(Op::rbit, width, value);
    } else if (opc < 4) {
      const unsigned container = opc == 1 ? 16 : opc == 2 ? 32 : 64;
      constexpr std::uint64_t kLanes[] = {0x00ff00ff00ff00ff, 0x0000ffff0000ffff,
                                          0x00000000ffffffff};
      result = value;
      for (unsigned step = 0; (8U << step) < container; ++step) {
        const auto count = lift.Constant(width, 8U << step);
        const auto lanes = lift.Constant(width, kLanes[step] & (UINT64_MAX >> (64 - width)));
        const auto down =
            lift.Add(Op::bit_and, width, lift.Add(Op::lshr, width, result, count), lanes);
        const auto up =
            lift.Add(Op::shl, width, lift.Add(Op::bit_and, width, result, lanes), count);
        result = lift.Add(Op::bit_or, width, down, up);
      }
    } else if (opc == 4) {
      result = lift.Add(Op::clz, width, value);
    } else {
      // The sign bits below the top one are the leading zeros of each bit
      // exclusive-ored with its upper neighbour, less the top bit itself.
      const auto neighbours = lift.Add(Op::ashr, width, value, lift.Constant(width, 1));
      const auto differences = lift.Add(Op::bit_xor, width, value, neighbours);
      result =
          lift.Add(Op::sub, width, lift.Add(Op::clz, width, differences), lift.Constant(width, 1));
    }

    lift.Write(rd, result, width);
  } else if ((word & 0x7fe0f000) == 0x1ac02000) {
    const unsigned kind = (word >> 10) & 3;
    const auto value = lift.Read(rn, width);
    const auto count =
        lift.Add(Op::bit_and, width, lift.Read(rm, width), lift.Constant(width, width - 1));
    ValueId result;
    if (kind < 3) {
      result = lift.Add(kind == 0 ? Op::shl : kind == 1 ? Op::lshr : Op::ashr, width, value, count);
    } else {
      const auto complement = lift.Add(Op::sub, width, lift.Constant(width, width), count);

      // Generic shifts saturate at the operand width: when count is zero the
      // left term vanishes, preserving the unrotated value without a special case.
      const auto left = lift.Add(Op::shl, width, value, complement);
      const auto right = lift.Add(Op::lshr, width, value, count);
      result = lift.Add(Op::bit_or, width, left, right);
    }

    lift.Write(rd, result, width);
  } else if ((word & 0x3fe00800) == 0x1a800000) {
    const auto lhs = lift.Read(rn, width);
    auto rhs = lift.Read(rm, width);
    if ((word & (1U << 30)) != 0) rhs = lift.Add(Op::bit_not, width, rhs);
    if ((word & (1U << 10)) != 0) rhs = lift.Add(Op::add, width, rhs, lift.Constant(width, 1));
    const auto condition = lift.Condition((word >> 12) & 15);
    lift.Write(rd, lift.Add(Op::select, width, condition, lhs, rhs), width);
  } else if ((word & 0x3fe00410) == 0x3a400000) {
    // CCMP and CCMN: the comparison's flags when the condition holds, the
    // #nzcv immediate otherwise. The condition reads the entry flags.
    const auto lhs = lift.Read(rn, width);
    const auto rhs = (word & (1U << 11)) != 0 ? lift.Constant(width, rm) : lift.Read(rm, width);
    const auto result = lift.Add(subtract ? Op::sub : Op::add, width, lhs, rhs);
    const auto computed = lift.FlagValues(lhs, rhs, result, width, subtract);
    const auto condition = lift.Condition((word >> 12) & 15);
    constexpr std::array<ir::StorageId, 4> kFlags = {kN, kZ, kC, kV};
    for (unsigned i = 0; i < 4; ++i) {
      const auto otherwise = lift.Constant(1, (word >> (3 - i)) & 1);
      lift.Write(kFlags[i], lift.Add(Op::select, 1, condition, computed[i], otherwise), 1);
    }
  } else if (cmeq_vector) {
    const bool q = (word & (1U << 30)) != 0;
    const unsigned size = (word >> 22) & 3;
    if (!q && size == 3) return {std::nullopt, DecodeDecline::invalid_encoding};
    const unsigned element_width = 8U << size;
    const auto lhs = lift.ReadQ(rn);
    const auto rhs = lift.ReadQ(rm);
    const auto zero = lift.Constant(128, 0);
    const auto ones = lift.Constant(128, UINT64_MAX >> (64 - element_width));
    auto result = zero;
    for (unsigned at = 0; at < (q ? 128U : 64U); at += element_width) {
      const auto left = lift.Add(Op::extract, element_width, lhs, 0, 0, at);
      const auto right = lift.Add(Op::extract, element_width, rhs, 0, 0, at);
      const auto equal = lift.Add(Op::equal, 1, left, right);
      auto lane = lift.Add(Op::select, 128, equal, ones, zero);
      if (at != 0) lane = lift.Shift(lane, 128, 0, at);
      result = lift.Add(Op::bit_or, 128, result, lane);
    }

    // Building from zero also clears Q[127:64] for the 64-bit arrangement.
    lift.Write(kQ0 + rd, result, 128);
  } else if ((word & 0xbfe0fc00) == 0x0e003c00) {
    const unsigned imm5 = (word >> 16) & 31;
    const bool q = (word & (1U << 30)) != 0;
    if ((imm5 & 15) == 0) return {std::nullopt, DecodeDecline::invalid_encoding};
    const unsigned size = std::countr_zero(imm5);
    if (q != (size == 3)) return {std::nullopt, DecodeDecline::invalid_encoding};
    const unsigned element_width = 8U << size;
    const unsigned result_width = q ? 64 : 32;
    const unsigned index = imm5 >> (size + 1);
    auto value = lift.Add(Op::extract, element_width, lift.ReadQ(rn), 0, 0, index * element_width);
    if (element_width < result_width) value = lift.Add(Op::zext, result_width, value);
    lift.Write(rd, value, result_width);
  } else if ((word & 0x9ff80400) == 0x0f000400) {
    // MOVI and MVNI (vector, immediate): a constant, replicated per element,
    // with a 64-bit result's upper half zero. ORR and BIC read the destination
    // and FMOV expands a floating-point immediate; those stay unsupported.
    const bool q = (word & (1U << 30)) != 0;
    const bool op = (word & (1U << 29)) != 0;
    const unsigned cmode = (word >> 12) & 15;
    const std::uint64_t imm8 = ((word >> 11) & 0xe0) | ((word >> 5) & 31);
    if ((word & (1U << 11)) != 0 || cmode == 15 || ((cmode & 1) != 0 && (cmode & 12) != 12)) {
      return {std::nullopt, DecodeDecline::unsupported};
    }

    std::uint64_t element;
    unsigned element_width;
    if (cmode < 8) {
      element = imm8 << (8 * (cmode >> 1));
      element_width = 32;
    } else if (cmode < 12) {
      element = imm8 << (8 * ((cmode >> 1) & 1));
      element_width = 16;
    } else if (cmode < 14) {
      // MSL shifts ones in, not zeros.
      element = (imm8 << (8 * (cmode - 11))) | ((std::uint64_t{1} << (8 * (cmode - 11))) - 1);
      element_width = 32;
    } else if (!op) {
      element = imm8;
      element_width = 8;
    } else {
      element = 0;
      for (unsigned bit = 0; bit < 8; ++bit) {
        if ((imm8 >> bit) & 1) element |= std::uint64_t{0xff} << (8 * bit);
      }

      element_width = 64;
    }

    std::uint64_t replicated = 0;
    for (unsigned at = 0; at < 64; at += element_width) replicated |= element << at;
    if (op && cmode != 14) replicated = ~replicated;
    auto value = lift.Constant(128, replicated);
    if (q && replicated != 0) {
      value =
          lift.Add(Op::bit_or, 128, value, lift.Add(Op::shl, 128, value, lift.Constant(128, 64)));
    }

    lift.Write(kQ0 + rd, value, 128);
  } else if ((word & 0xfffefc00) == 0x1e260000 || (word & 0xfffefc00) == 0x9e660000 ||
             (word & 0xfffefc00) == 0x9eae0000) {
    // FMOV between a general and a SIMD&FP register: the bits unchanged. A
    // write of S or D zeroes the rest of Q; a write of V.D[1] keeps D[0].
    const bool to_vector = (word & (1U << 16)) != 0;
    const bool upper = (word & 0xfffefc00) == 0x9eae0000;
    if (to_vector) {
      auto value = lift.Add(Op::zext, 128, lift.Read(rn, width));
      if (upper) {
        const auto low = lift.Add(Op::zext, 128, lift.Add(Op::extract, 64, lift.ReadQ(rd)));
        value =
            lift.Add(Op::bit_or, 128, low, lift.Add(Op::shl, 128, value, lift.Constant(128, 64)));
      }

      lift.Write(kQ0 + rd, value, 128);
    } else {
      lift.Write(rd, lift.Add(Op::extract, width, lift.ReadQ(rn), 0, 0, upper ? 64 : 0), width);
    }
  } else if ((word & 0xffbffc00) == 0x1e204000) {
    // FMOV (register) of S or D: a copy of the low bits that zeroes the rest.
    const unsigned copied = (word & (1U << 22)) != 0 ? 64 : 32;
    lift.Write(kQ0 + rd, lift.Add(Op::zext, 128, lift.Add(Op::extract, copied, lift.ReadQ(rn))),
               128);
  } else if ((word & 0xff600000) == 0x9b200000) {
    // SMADDL, SMSUBL, UMADDL, UMSUBL: two W operands widened to 64 bits,
    // signed unless bit 23 is set, then multiplied into an X accumulator.
    const bool is_unsigned = (word & (1U << 23)) != 0;
    const auto widen = [&](unsigned reg) {
      auto value = lift.Add(Op::zext, 64, lift.Read(reg, 32));
      if (is_unsigned) return value;
      const auto count = lift.Constant(64, 32);
      return lift.Add(Op::ashr, 64, lift.Add(Op::shl, 64, value, count), count);
    };

    const auto product = lift.Add(Op::mul, 64, widen(rn), widen(rm));
    const auto accumulator = lift.Read((word >> 10) & 31, 64);
    lift.Write(rd, lift.Add((word & (1U << 15)) != 0 ? Op::sub : Op::add, 64, accumulator, product),
               64);
  } else if ((word & 0x7fe00000) == 0x1b000000) {
    const auto product = lift.Add(Op::mul, width, lift.Read(rn, width), lift.Read(rm, width));
    const auto accumulator = lift.Read((word >> 10) & 31, width);
    const auto result =
        lift.Add((word & (1U << 15)) != 0 ? Op::sub : Op::add, width, accumulator, product);
    lift.Write(rd, result, width);
  } else if ((word & 0xffffffe0) == 0xd53b4200) {
    auto value = lift.Constant(64, 0);
    for (unsigned index = 0; index < 4; ++index) {
      const auto flag = lift.Add(Op::read, 1, 0, 0, 0, 0, kN + index);
      const auto placed = lift.Shift(lift.Add(Op::zext, 64, flag), 64, 0, 31 - index);
      value = lift.Add(Op::bit_or, 64, value, placed);
    }

    lift.Write(rd, value, 64);
  } else if ((word & 0xffffffe0) == 0xd51b4200) {
    // Only the four architectural flag bits are writable; other source bits
    // neither become state nor select a different system register.
    const auto value = lift.Read(rd, 64);
    for (unsigned index = 0; index < 4; ++index) {
      lift.Write(kN + index, lift.Add(Op::extract, 1, value, 0, 0, 31 - index), 1);
    }
  } else if ((word & 0xffffffe0) == 0xd53bd040) {
    // MRS Xt, TPIDR_EL0: a copy of the thread pointer, as a stack protector
    // reads it. Only the read is modeled; writing it stays unsupported.
    lift.Write(rd, lift.Add(Op::read, 64, 0, 0, 0, 0, kTpidrEl0), 64);
  } else if (word != 0xd503201f && !IsBranchTargetHint(word) && !IsPointerAuthHint(word)) {
    return {std::nullopt, DecodeDecline::unsupported};
  }

  if (lift.reason != DecodeDecline::none) return {std::nullopt, lift.reason};
  return {ir::Group(address, std::vector<std::uint8_t>(bytes.begin(), bytes.end()),
                    std::move(lift.nodes), std::move(lift.writes), memory_model, transfer),
          DecodeDecline::none};
}

}  // namespace nyx::a64
