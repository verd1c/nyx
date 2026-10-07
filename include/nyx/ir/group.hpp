#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace nyx::ir {

using ValueId = std::uint32_t;
using StorageId = std::uint32_t;

enum class Op {
  constant,
  read,
  add,
  sub,
  mul,
  bit_and,
  bit_or,
  bit_xor,
  bit_not,
  shl,
  lshr,
  ashr,
  extract,
  zext,
  select,
  equal,
  unsigned_less,
  signed_less,
  udiv,
  sdiv,
  umulh,
  smulh,
  clz,
  rbit,
  load,
  store,
  write,
  image_address,
  exclusive_load,
  exclusive_store,
  exclusive_clear
};

enum class Effect {
  pure,
  storage_read,
  memory_read,
  memory_write,
  storage_write,
  exclusive_read,
  exclusive_write,
  exclusive_monitor
};

enum class ByteOrder { little, big };
enum class MemoryModel { unspecified, atomic_scalar_reference, qemu_exclusive_scalar_reference };

struct Access {
  ByteOrder byte_order = ByteOrder::little;
  unsigned alignment = 1;
  bool decline_on_unaligned = false;
};

struct OperationDescriptor {
  std::string_view name;
  unsigned arity;
  Effect effect = Effect::pure;
  bool produces_value = true;
};

// udiv through rbit follow AArch64 rather than C: a zero divisor gives zero,
// sdiv of the most negative value by -1 gives that value back, umulh and smulh
// are the upper half of the double-width product, and clz of zero is the
// width. They are defined for widths up to 64; a wider one is an invalid group.
inline constexpr std::array<OperationDescriptor, 31> kOperationDescriptors = {
    {{"constant", 0},
     {"read", 0, Effect::storage_read},
     {"add", 2},
     {"sub", 2},
     {"mul", 2},
     {"bit_and", 2},
     {"bit_or", 2},
     {"bit_xor", 2},
     {"bit_not", 1},
     {"shl", 2},
     {"lshr", 2},
     {"ashr", 2},
     {"extract", 1},
     {"zext", 1},
     {"select", 3},
     {"equal", 2},
     {"unsigned_less", 2},
     {"signed_less", 2},
     {"udiv", 2},
     {"sdiv", 2},
     {"umulh", 2},
     {"smulh", 2},
     {"clz", 1},
     {"rbit", 1},
     {"load", 1, Effect::memory_read},
     {"store", 2, Effect::memory_write, false},
     {"write", 1, Effect::storage_write, false},
     {"image_address", 0},
     {"exclusive_load", 1, Effect::exclusive_read},
     {"exclusive_store", 2, Effect::exclusive_write},
     {"exclusive_clear", 0, Effect::exclusive_monitor, false}}};

constexpr const OperationDescriptor* Descriptor(Op op) {
  const auto index = static_cast<std::size_t>(op);
  return index < kOperationDescriptors.size() ? &kOperationDescriptors[index] : nullptr;
}

// The operations defined only up to 64 bits, which validation holds them to.
constexpr bool NarrowOnly(Op op) {
  return op == Op::udiv || op == Op::sdiv || op == Op::umulh || op == Op::smulh || op == Op::clz ||
         op == Op::rbit;
}

constexpr bool HasMemoryOrMonitorEffect(Op op) {
  const auto* descriptor = Descriptor(op);
  return descriptor &&
         (descriptor->effect == Effect::memory_read || descriptor->effect == Effect::memory_write ||
          descriptor->effect == Effect::exclusive_read ||
          descriptor->effect == Effect::exclusive_write ||
          descriptor->effect == Effect::exclusive_monitor);
}

constexpr bool MayWriteMemory(Op op) {
  const auto* descriptor = Descriptor(op);
  return descriptor && (descriptor->effect == Effect::memory_write ||
                        descriptor->effect == Effect::exclusive_write);
}

struct Node {
  Op op;
  unsigned width;
  std::array<ValueId, 3> inputs{};

  // Constants zero-extend this literal to width; extract uses it as a low bit index.
  std::uint64_t immediate = 0;
  StorageId storage = 0;
  Access access{};
};

struct Write {
  StorageId storage;
  ValueId value;
};

enum class TransferKind { jump, conditional, call, return_ };

// A terminal instruction effect. It does not list all successors or summarize
// the callee; a call continuation records only its link address.
struct Transfer {
  TransferKind kind;
  ValueId target;
  std::optional<ValueId> condition;
  std::optional<ValueId> alternative;
  std::optional<ValueId> continuation;
};

// `width` answers for every value the transfer may name and yields zero for an
// id that names none. A recovered transfer can name a destination that is a node
// of no instruction, so the lookup is not always an index into a node sequence.
template <class Width>
constexpr bool ValidTransferWidths(const Transfer& transfer, Width width) {
  if (width(transfer.target) != 64) return false;
  switch (transfer.kind) {
    case TransferKind::jump:
    case TransferKind::return_:
      return !transfer.condition && !transfer.alternative && !transfer.continuation;
    case TransferKind::conditional:
      return transfer.condition && transfer.alternative && !transfer.continuation &&
             width(*transfer.condition) == 1 && width(*transfer.alternative) == 64;
    case TransferKind::call:
      return !transfer.condition && !transfer.alternative && transfer.continuation &&
             width(*transfer.continuation) == 64;
  }

  return false;
}

inline bool ValidTransfer(const Transfer& transfer, std::span<const Node> nodes) {
  return ValidTransferWidths(transfer, [&](ValueId id) -> unsigned {
    if (id >= nodes.size()) return 0;
    const auto* descriptor = Descriptor(nodes[id].op);
    return descriptor != nullptr && descriptor->produces_value ? nodes[id].width : 0;
  });
}

class Group {
 public:
  Group(std::uint64_t source_address, std::vector<std::uint8_t> bytes, std::vector<Node> nodes,
        std::vector<Write> writes, MemoryModel memory_model = MemoryModel::unspecified,
        std::optional<Transfer> transfer = {})
      : source_address_(source_address),
        bytes_(std::move(bytes)),
        nodes_(std::move(nodes)),
        writes_(std::move(writes)),
        memory_model_(memory_model),
        transfer_(transfer) {}

  std::uint64_t source_address() const { return source_address_; }

  std::span<const std::uint8_t> bytes() const { return bytes_; }

  std::span<const Node> nodes() const { return nodes_; }

  std::span<const Write> writes() const { return writes_; }

  MemoryModel memory_model() const { return memory_model_; }

  const std::optional<Transfer>& transfer() const { return transfer_; }

 private:
  std::uint64_t source_address_;
  std::vector<std::uint8_t> bytes_;
  std::vector<Node> nodes_;
  std::vector<Write> writes_;
  MemoryModel memory_model_;
  std::optional<Transfer> transfer_;
};

}  // namespace nyx::ir
