#include "nyx/ir/block.hpp"

#include <algorithm>
#include <bit>
#include <limits>

#include "nyx/ir/path.hpp"

namespace nyx::ir {
namespace {

enum class RegionMode { block, path };

bool Value(ValueId id, std::span<const Node> prior, unsigned width = 0) {
  if (id >= prior.size()) return false;
  const auto* descriptor = Descriptor(prior[id].op);
  return descriptor && descriptor->produces_value && (width == 0 || prior[id].width == width);
}

bool Shape(const Node& node, std::span<const Node> prior, unsigned max_width, MemoryModel model) {
  const auto* descriptor = Descriptor(node.op);
  if (!descriptor || !node.width || node.width > max_width) return false;
  for (unsigned i = 0; i < descriptor->arity; ++i) {
    if (!Value(node.inputs[i], prior)) return false;
  }

  if (node.op == Op::image_address) return node.width == 64;
  if (node.op == Op::exclusive_clear) {
    return model == MemoryModel::qemu_exclusive_scalar_reference && node.width == 1;
  }

  if (!descriptor->arity) return true;
  if (NarrowOnly(node.op) && node.width > 64) return false;
  const auto a = prior[node.inputs[0]].width;
  const auto b = descriptor->arity > 1 ? prior[node.inputs[1]].width : 0;
  switch (node.op) {
    case Op::load:
    case Op::store:
      return model == MemoryModel::atomic_scalar_reference && a == 64 && node.width % 8 == 0 &&
             node.width <= 128 && (node.op != Op::store || b == node.width) &&
             node.access.alignment && !(node.access.alignment & (node.access.alignment - 1)) &&
             (node.access.byte_order == ByteOrder::little ||
              node.access.byte_order == ByteOrder::big);
    case Op::exclusive_load:
    case Op::exclusive_store: {
      const auto size = node.op == Op::exclusive_load ? node.width : b;
      return model == MemoryModel::qemu_exclusive_scalar_reference && a == 64 &&
             (size == 32 || size == 64) && (node.op != Op::exclusive_store || node.width == 32) &&
             node.access.alignment == size / 8 && node.access.byte_order == ByteOrder::little &&
             node.access.decline_on_unaligned;
    }
    case Op::extract:
      return node.immediate < a && node.width <= a - node.immediate;
    case Op::zext:
      return node.width >= a;
    case Op::select:
      return a == 1 && b == node.width && prior[node.inputs[2]].width == node.width;
    case Op::equal:
    case Op::unsigned_less:
    case Op::signed_less:
      return a == b && node.width == 1;
    case Op::shl:
    case Op::lshr:
    case Op::ashr:
    case Op::write:
      return node.width == a;
    default:
      return node.width == a && (descriptor->arity != 2 || b == node.width);
  }
}

bool Writes(std::span<const Write> writes, std::span<const Node> nodes) {
  for (std::size_t i = 0; i < writes.size(); ++i) {
    if (!Value(writes[i].value, nodes)) return false;
    for (std::size_t j = 0; j < i; ++j)
      if (writes[i].storage == writes[j].storage) return false;
  }

  return true;
}

BlockDecline ConsistentBytes(std::span<const Group> groups, Budget& budget) {
  if (budget.try_consume({groups.size() * (std::bit_width(groups.size()) + 1),
                          groups.size() * sizeof(std::size_t)}) != BudgetDecline::none) {
    return BlockDecline::resource_limit;
  }

  std::vector<std::size_t> order(groups.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](auto a, auto b) {
    return groups[a].source_address() < groups[b].source_address();
  });
  auto covering = order.front();
  for (std::size_t i = 1; i < order.size(); ++i) {
    const auto& previous = groups[covering];
    const auto& current = groups[order[i]];
    const auto offset = current.source_address() - previous.source_address();
    if (offset < previous.bytes().size()) {
      const auto count =
          std::min<std::size_t>(previous.bytes().size() - offset, current.bytes().size());
      if (budget.try_consume({count, 0}) != BudgetDecline::none)
        return BlockDecline::resource_limit;
      if (!std::equal(current.bytes().begin(), current.bytes().begin() + count,
                      previous.bytes().begin() + offset))
        return BlockDecline::invalid_source;
    }

    if (current.source_address() + current.bytes().size() - 1 >
        previous.source_address() + previous.bytes().size() - 1)
      covering = order[i];
  }

  return BlockDecline::none;
}

// A direct call transfers to its target whether or not the callee ever returns,
// so an itinerary that continues at that target records the instruction the
// machine executed next. That holds only when the group itself settles the
// target; a computed target names no address here and is declined.
// Obfuscators emit `bl` as an unconditional jump to break call-graph
// reconstruction, so itineraries through such code take this shape.
std::optional<std::uint64_t> SettledAddress(std::span<const Node> nodes, ValueId id,
                                            unsigned depth = 0) {
  constexpr unsigned kMaxDepth = 8;
  if (depth > kMaxDepth || id >= nodes.size()) return {};
  const auto& node = nodes[id];
  if (node.width != 64) return {};
  switch (node.op) {
    case Op::constant:
    case Op::image_address:
      return node.immediate;
    case Op::add: {
      const auto left = SettledAddress(nodes, node.inputs[0], depth + 1);
      if (!left) return {};
      const auto right = SettledAddress(nodes, node.inputs[1], depth + 1);
      if (!right) return {};
      return static_cast<std::uint64_t>(*left + *right);
    }
    default:
      return {};
  }
}

BlockDecline Sources(std::span<const Group> groups, Budget& budget, BlockLimits limits,
                     std::uint64_t& node_count, RegionMode mode) {
  if (groups.empty()) return BlockDecline::invalid_source;
  if (groups.size() > limits.max_groups) return BlockDecline::resource_limit;
  node_count = 0;
  for (std::size_t i = 0; i < groups.size(); ++i) {
    const auto& group = groups[i];
    if (group.bytes().empty() ||
        group.bytes().size() - 1 >
            std::numeric_limits<std::uint64_t>::max() - group.source_address())
      return BlockDecline::invalid_source;
    if (i != 0 && (mode == RegionMode::block || !groups[i - 1].transfer())) {
      const auto& previous = groups[i - 1];
      if ((mode == RegionMode::block &&
           previous.bytes().size() >
               std::numeric_limits<std::uint64_t>::max() - previous.source_address()) ||
          previous.source_address() + previous.bytes().size() != group.source_address())
        return BlockDecline::invalid_source;
    }

    if (group.transfer() && i + 1 != groups.size()) {
      const auto kind = group.transfer()->kind;

      // A jump or conditional names only addresses the machine goes to next.
      const bool linear = kind == TransferKind::jump || kind == TransferKind::conditional;

      // A call names two successors of different character. Continuing at the
      // settled target elides nothing; continuing at the link address would
      // silently splice out an unmodelled callee body and is never admitted.
      const bool followed_target =
          mode == RegionMode::path && kind == TransferKind::call &&
          SettledAddress(group.nodes(), group.transfer()->target) == groups[i + 1].source_address();
      if (mode == RegionMode::block || !(linear || followed_target)) {
        return BlockDecline::unsupported_control;
      }
    }

    if (group.memory_model() != MemoryModel::unspecified &&
        group.memory_model() != MemoryModel::atomic_scalar_reference &&
        group.memory_model() != MemoryModel::qemu_exclusive_scalar_reference)
      return BlockDecline::invalid_ir;
    if (group.nodes().size() > limits.max_nodes - node_count ||
        group.writes().size() > limits.max_storage)
      return BlockDecline::resource_limit;
    node_count += group.nodes().size();
    if (budget.try_consume({group.nodes().size() * 8 + 1, 0}) != BudgetDecline::none ||
        budget.try_consume({group.writes().size() * (group.writes().size() + 1), 0}) !=
            BudgetDecline::none)
      return BlockDecline::resource_limit;
    for (std::size_t n = 0; n < group.nodes().size(); ++n) {
      if (!Shape(group.nodes()[n], group.nodes().first(n), limits.max_width, group.memory_model()))
        return BlockDecline::invalid_ir;
    }

    if (!Writes(group.writes(), group.nodes()) ||
        (group.transfer() && !ValidTransfer(*group.transfer(), group.nodes()))) {
      return BlockDecline::invalid_ir;
    }
  }

  return mode == RegionMode::path ? ConsistentBytes(groups, budget) : BlockDecline::none;
}

struct StorageType {
  StorageId storage;
  unsigned width;
};

BlockDecline Register(std::vector<StorageType>& slots, StorageId storage, unsigned width,
                      Budget& budget, std::uint32_t maximum) {
  if (budget.try_consume({static_cast<std::uint64_t>(std::bit_width(slots.size()) + 1), 0}) !=
      BudgetDecline::none)
    return BlockDecline::resource_limit;
  const auto position =
      std::lower_bound(slots.begin(), slots.end(), storage,
                       [](const StorageType& slot, StorageId id) { return slot.storage < id; });
  if (position != slots.end() && position->storage == storage) {
    return position->width == width ? BlockDecline::none : BlockDecline::invalid_ir;
  }

  if (slots.size() == maximum ||
      budget.try_consume({static_cast<std::uint64_t>(slots.end() - position) + 1, 0}) !=
          BudgetDecline::none)
    return BlockDecline::resource_limit;
  slots.insert(position, {storage, width});
  return BlockDecline::none;
}

std::size_t StorageIndex(std::span<const StorageType> slots, StorageId storage) {
  return std::lower_bound(slots.begin(), slots.end(), storage,
                          [](const StorageType& slot, StorageId id) { return slot.storage < id; }) -
         slots.begin();
}

Transfer Remap(Transfer transfer, std::span<const ValueId> values) {
  transfer.target = values[transfer.target];
  if (transfer.condition) transfer.condition = values[*transfer.condition];
  if (transfer.alternative) transfer.alternative = values[*transfer.alternative];
  if (transfer.continuation) transfer.continuation = values[*transfer.continuation];
  return transfer;
}

template <class Region>
BlockDecline ValidateImpl(const Region&, Budget&, BlockLimits, RegionMode);

template <class Region>
std::pair<std::optional<Region>, BlockDecline> NormalizeImpl(std::span<const Group> sources,
                                                             Budget& budget, BlockLimits limits,
                                                             RegionMode mode) {
  std::uint64_t count = 0;
  const auto reason = Sources(sources, budget, limits, count, mode);
  if (reason != BlockDecline::none) return {{}, reason};
  std::uint64_t occurrences = count;
  for (const auto& source : sources) occurrences += source.writes().size();
  const auto slot_capacity = std::min<std::uint64_t>(limits.max_storage, occurrences);
  if (budget.try_consume({count * 8, count * (sizeof(Node) + sizeof(Origin) + sizeof(ValueId)) +
                                         sources.size() * (sizeof(Group) + sizeof(Boundary)) +
                                         slot_capacity * sizeof(StorageType)}) !=
      BudgetDecline::none) {
    return {{}, BlockDecline::resource_limit};
  }

  // Sparse storage IDs are compressed into sorted slots. Search and insertion
  // work depends on cells actually present, never on their numeric IDs or the
  // configured capacity. Entry/current values remain distinct snapshots.
  std::vector<StorageType> slots;
  slots.reserve(slot_capacity);
  for (const auto& source : sources) {
    for (const auto& node : source.nodes()) {
      if (node.op == Op::read || node.op == Op::write) {
        const auto status = Register(slots, node.storage, node.width, budget, limits.max_storage);
        if (status != BlockDecline::none) return {{}, status};
      }
    }

    for (const auto& write : source.writes()) {
      const auto status = Register(slots, write.storage, source.nodes()[write.value].width, budget,
                                   limits.max_storage);
      if (status != BlockDecline::none) return {{}, status};
    }
  }

  if (budget.try_consume({2 * slots.size(), 2 * slots.size() * sizeof(std::optional<ValueId>)}) !=
      BudgetDecline::none)
    return {{}, BlockDecline::resource_limit};
  // Charge the exact deep copies before any source or boundary is published.
  for (const auto& source : sources) {
    if (budget.try_consume({source.bytes().size() + source.nodes().size() + source.writes().size(),
                            source.bytes().size() + source.nodes().size() * sizeof(Node) +
                                2 * source.writes().size() * sizeof(Write)}) !=
        BudgetDecline::none) {
      return {{}, BlockDecline::resource_limit};
    }
  }

  std::vector<Group> originals(sources.begin(), sources.end());
  std::vector<Node> nodes;
  std::vector<Origin> origins;
  std::vector<Boundary> boundaries;
  std::vector<ValueId> values;
  std::vector<std::optional<ValueId>> definitions(slots.size()), entry(slots.size());
  nodes.reserve(count);
  origins.reserve(count);
  values.reserve(count);
  boundaries.reserve(sources.size());
  const std::uint64_t lookup_work = std::bit_width(slots.size()) + 1;
  for (std::size_t index = 0; index < sources.size(); ++index) {
    const auto& source = sources[index];
    const auto first = static_cast<ValueId>(nodes.size());
    if (budget.try_consume({slots.size(), 0}) != BudgetDecline::none)
      return {{}, BlockDecline::resource_limit};
    entry.assign(definitions.begin(), definitions.end());
    values.clear();
    for (std::size_t local = 0; local < source.nodes().size(); ++local) {
      auto node = source.nodes()[local];
      std::size_t storage = 0;
      if (node.op == Op::read || node.op == Op::write) {
        if (budget.try_consume({lookup_work, 0}) != BudgetDecline::none)
          return {{}, BlockDecline::resource_limit};
        storage = StorageIndex(slots, node.storage);
      }

      if (node.op == Op::read && entry[storage]) {
        values.push_back(*entry[storage]);
        continue;
      }

      for (unsigned operand = 0; operand < Descriptor(node.op)->arity; ++operand)
        node.inputs[operand] = values[node.inputs[operand]];
      const auto value = static_cast<ValueId>(nodes.size());
      nodes.push_back(node);
      origins.push_back({static_cast<std::uint32_t>(index), static_cast<ValueId>(local)});
      values.push_back(value);
      if (node.op == Op::read) {
        entry[storage] = value;
        if (!definitions[storage]) definitions[storage] = value;
      } else if (node.op == Op::write) {
        definitions[storage] = node.inputs[0];
      }
    }

    Boundary boundary{first, static_cast<std::uint32_t>(nodes.size() - first), {}, {}};
    boundary.writes.reserve(source.writes().size());
    for (const auto& write : source.writes()) {
      if (budget.try_consume({lookup_work, 0}) != BudgetDecline::none)
        return {{}, BlockDecline::resource_limit};
      const auto value = values[write.value];
      definitions[StorageIndex(slots, write.storage)] = value;
      boundary.writes.push_back({write.storage, value});
    }

    if (source.transfer()) boundary.transfer = Remap(*source.transfer(), values);
    boundaries.push_back(std::move(boundary));
  }

  Region result(std::move(originals), std::move(nodes), std::move(origins), std::move(boundaries));
  const auto valid = ValidateImpl(result, budget, limits, mode);
  if (valid != BlockDecline::none) return {{}, valid};
  return {std::move(result), BlockDecline::none};
}

template <class Region>
BlockDecline ValidateImpl(const Region& block, Budget& budget, BlockLimits limits,
                          RegionMode mode) {
  std::uint64_t source_nodes = 0;
  const auto reason = Sources(block.sources(), budget, limits, source_nodes, mode);
  if (reason != BlockDecline::none) return reason;
  if (block.boundaries().size() != block.sources().size() ||
      block.origins().size() != block.nodes().size())
    return BlockDecline::invalid_ir;
  if (block.nodes().size() > limits.max_nodes) return BlockDecline::resource_limit;
  if (budget.try_consume({block.nodes().size() * 10 + source_nodes + block.boundaries().size(),
                          0}) != BudgetDecline::none)
    return BlockDecline::resource_limit;
  std::size_t next = 0;
  for (std::size_t i = 0; i < block.boundaries().size(); ++i) {
    const auto& boundary = block.boundaries()[i];
    const auto& source = block.sources()[i];
    if (boundary.first_node != next || boundary.node_count > block.nodes().size() - next ||
        boundary.writes.size() != source.writes().size())
      return BlockDecline::invalid_ir;
    const auto end = next + boundary.node_count;
    const auto prefix = block.nodes().first(end);
    std::size_t previous_operation = 0;
    std::size_t effect = 0;
    for (std::size_t n = next; n < end; ++n) {
      const auto& node = block.nodes()[n];
      const auto& origin = block.origins()[n];
      if (origin.boundary != i || origin.operation >= source.nodes().size() ||
          (n != next && origin.operation <= previous_operation) ||
          !Shape(node, block.nodes().first(n), limits.max_width, source.memory_model()))
        return BlockDecline::invalid_ir;
      previous_operation = origin.operation;
      const auto& original = source.nodes()[origin.operation];
      if (node.width != original.width) return BlockDecline::invalid_ir;
      const auto original_effect = Descriptor(original.op)->effect;
      const auto node_effect = Descriptor(node.op)->effect;
      if (original_effect != node_effect) return BlockDecline::invalid_ir;
      if (node_effect == Effect::memory_read || node_effect == Effect::memory_write ||
          node_effect == Effect::storage_write || node_effect == Effect::exclusive_read ||
          node_effect == Effect::exclusive_write || node_effect == Effect::exclusive_monitor) {
        while (effect < source.nodes().size() &&
               (Descriptor(source.nodes()[effect].op)->effect == Effect::pure ||
                Descriptor(source.nodes()[effect].op)->effect == Effect::storage_read))
          ++effect;
        if (effect != origin.operation || node.op != original.op ||
            node.storage != original.storage ||
            node.access.alignment != original.access.alignment ||
            node.access.byte_order != original.access.byte_order ||
            node.access.decline_on_unaligned != original.access.decline_on_unaligned)
          return BlockDecline::invalid_ir;
        ++effect;
      } else if (node_effect == Effect::storage_read && node.storage != original.storage)
        return BlockDecline::invalid_ir;
    }
    while (effect < source.nodes().size() &&
           (Descriptor(source.nodes()[effect].op)->effect == Effect::pure ||
            Descriptor(source.nodes()[effect].op)->effect == Effect::storage_read))
      ++effect;
    if (effect != source.nodes().size() || !Writes(boundary.writes, prefix))
      return BlockDecline::invalid_ir;
    for (std::size_t w = 0; w < boundary.writes.size(); ++w) {
      if (boundary.writes[w].storage != source.writes()[w].storage ||
          prefix[boundary.writes[w].value].width != source.nodes()[source.writes()[w].value].width)
        return BlockDecline::invalid_ir;
    }

    if (boundary.transfer.has_value() != source.transfer().has_value() ||
        (boundary.transfer && (boundary.transfer->kind != source.transfer()->kind ||
                               !ValidTransfer(*boundary.transfer, prefix))))
      return BlockDecline::invalid_ir;
    next = end;
  }

  return next == block.nodes().size() ? BlockDecline::none : BlockDecline::invalid_ir;
}

}  // namespace

BlockResult Normalize(std::span<const Group> sources, Budget& budget, BlockLimits limits) {
  auto result = NormalizeImpl<Block>(sources, budget, limits, RegionMode::block);
  return {std::move(result.first), result.second};
}

BlockDecline Validate(const Block& block, Budget& budget, BlockLimits limits) {
  return ValidateImpl(block, budget, limits, RegionMode::block);
}

PathResult NormalizePath(std::span<const Group> sources, Budget& budget, BlockLimits limits) {
  auto result = NormalizeImpl<Path>(sources, budget, limits, RegionMode::path);
  return {std::move(result.first), result.second};
}

BlockDecline ValidatePath(const Path& path, Budget& budget, BlockLimits limits) {
  return ValidateImpl(path, budget, limits, RegionMode::path);
}

}  // namespace nyx::ir
