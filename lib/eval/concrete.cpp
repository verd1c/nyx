#include "nyx/eval/concrete.hpp"

#include <algorithm>
#include <bit>
#include <limits>

#include "nyx/eval/block.hpp"
#include "nyx/eval/path.hpp"
#include "nyx/eval/ssa.hpp"
#include "nyx/ir/fold.hpp"

namespace nyx::eval {
namespace {

const Cell* Find(const State& state, ir::StorageId id) {
  for (const auto& cell : state.cells) {
    if (cell.id == id) return &cell;
  }

  return nullptr;
}

BitVectorResult Copy(const BitVector& value, Budget& budget, unsigned max_width) {
  return BitVector::from_words(value.width(), value.words(), max_width, budget);
}

BitVectorResult Slice(const BitVector& value, unsigned low, unsigned width, Budget& budget,
                      unsigned max_width) {
  const std::size_t count = width / 64 + (width % 64 != 0);
  const auto charged = budget.try_consume({width, count * sizeof(std::uint64_t)});
  if (charged != BudgetDecline::none) {
    return charged == BudgetDecline::work_limit ? BitVectorDecline::work_limit
                                                : BitVectorDecline::byte_limit;
  }

  std::vector<std::uint64_t> words(count);
  for (unsigned i = 0; i < width; ++i) {
    if (i < value.width() - low && value.bit(low + i)) {
      words[i / 64] |= std::uint64_t{1} << (i % 64);
    }
  }

  return BitVector::from_words(width, words, max_width, budget);
}

bool Less(const BitVector& lhs, const BitVector& rhs, bool is_signed) {
  if (is_signed && lhs.bit(lhs.width() - 1) != rhs.bit(rhs.width() - 1)) {
    return lhs.bit(lhs.width() - 1);
  }

  for (unsigned i = lhs.word_count(); i != 0; --i) {
    if (lhs.word(i - 1) != rhs.word(i - 1)) return lhs.word(i - 1) < rhs.word(i - 1);
  }

  return false;
}

std::uint64_t ShiftCount(const BitVector& count) {
  for (unsigned i = 1; i < count.word_count(); ++i) {
    if (count.word(i) != 0) return std::numeric_limits<std::uint64_t>::max();
  }

  return count.word(0);
}

struct InstructionView {
  std::span<const ir::Node> nodes;
  std::size_t first_node;
  std::size_t node_count;
  std::span<const ir::Write> writes;
  const std::optional<ir::Transfer>& transfer;
  ir::MemoryModel memory_model;

  // Destinations a recovered transfer names, with ids from `nodes.size()`
  // upward. They belong to no instruction, so nothing but the transfer reads
  // one and their evaluation is not part of any boundary.
  std::span<const std::optional<BitVector>> destinations = {};
  const ir::RecoveredPath* recovered = nullptr;
  const ir::SsaBlock* ssa_block = nullptr;
  std::vector<std::optional<BitVector>>* frame_values = nullptr;
  std::optional<ir::SsaHandle> predecessor = std::nullopt;
  const std::vector<std::optional<BitVector>>* predecessor_values = nullptr;
  std::optional<std::size_t> ssa_boundary = std::nullopt;
};

Outcome OmittedAccessPrecondition(const Memory& memory, std::uint64_t address, unsigned size,
                                  unsigned alignment, bool write, Budget& budget) {
  if (address >> 56 != 0) return Outcome::unsupported;
  if (address % alignment) return Outcome::unsupported;
  const auto checked = memory.CheckAccess(address, size, write, budget);
  if (checked.status == MemoryStatus::resource_limit) return Outcome::resource_limit;
  return checked.status == MemoryStatus::ok ? Outcome::completed : Outcome::unsupported;
}

ExecutionResult ExecuteImpl(const InstructionView& instruction, State& state, Memory* memory,
                            Budget& budget, Limits limits, ExecutionContext context,
                            std::vector<std::optional<BitVector>>& values) {
  const auto nodes = instruction.nodes;
  const auto writes = instruction.writes;
  const auto end_node = instruction.first_node + instruction.node_count;
  if (end_node > nodes.size() || values.size() != instruction.first_node) {
    return Outcome::invalid_group;
  }

  if (instruction.memory_model != ir::MemoryModel::unspecified &&
      instruction.memory_model != ir::MemoryModel::atomic_scalar_reference &&
      instruction.memory_model != ir::MemoryModel::qemu_exclusive_scalar_reference)
    return Outcome::invalid_group;
  if (instruction.node_count > limits.max_nodes || state.cells.size() > limits.max_cells ||
      writes.size() > limits.max_cells)
    return Outcome::resource_limit;
  const std::uint64_t cells_count = state.cells.size();
  const std::uint64_t nodes_count = instruction.node_count;
  const std::uint64_t writes_count = writes.size();
  if (budget.try_consume({cells_count * cells_count, 0}) != BudgetDecline::none) {
    return Outcome::resource_limit;
  }

  for (std::size_t i = 0; i < state.cells.size(); ++i) {
    if (state.cells[i].value.width() == 0 || state.cells[i].value.width() > limits.max_width ||
        state.cells[i].value.words().size() != state.cells[i].value.word_count()) {
      return Outcome::invalid_state;
    }

    for (std::size_t j = 0; j < i; ++j) {
      if (state.cells[i].id == state.cells[j].id) return Outcome::invalid_state;
    }
  }

  if (state.exclusive &&
      ((state.exclusive->size != 4 && state.exclusive->size != 8) ||
       state.exclusive->address % state.exclusive->size != 0 ||
       state.exclusive->address >> 56 != 0 || !state.exclusive->memory_identity ||
       (memory && state.exclusive->memory_identity != memory->identity()))) {
    return Outcome::invalid_state;
  }

  // Each factor is bounded by an unsigned limit. Charge products separately so
  // summing several individually representable products cannot wrap the budget.
  if (budget.try_consume(
          {nodes_count * (cells_count + 1),
           nodes_count * sizeof(AccessEvent) + (nodes_count + writes_count) * sizeof(Cell)}) !=
          BudgetDecline::none ||
      budget.try_consume({writes_count * cells_count, 0}) != BudgetDecline::none ||
      budget.try_consume({writes_count * writes_count, 0}) != BudgetDecline::none)
    return Outcome::resource_limit;

  // Validate every edge before evaluating; unused malformed nodes cannot hide in
  // a successfully executed group, and only backward references form a DAG.
  for (std::size_t i = instruction.first_node; i < end_node; ++i) {
    const auto& node = nodes[i];
    const auto* descriptor = ir::Descriptor(node.op);
    const unsigned arity = descriptor == nullptr ? 4 : descriptor->arity;
    if (node.width == 0 || node.width > limits.max_width || arity > 3) {
      return Outcome::invalid_group;
    }

    for (unsigned j = 0; j < arity; ++j) {
      if (node.inputs[j] >= i || ir::Descriptor(nodes[node.inputs[j]].op) == nullptr ||
          !ir::Descriptor(nodes[node.inputs[j]].op)->produces_value) {
        return Outcome::invalid_group;
      }
    }

    using ir::Op;
    if (node.op == Op::image_address) {
      if (node.width != 64) return Outcome::invalid_group;
      if (!context.load_bias) return Outcome::unsupported;
    } else if (node.op == Op::exclusive_clear) {
      if (instruction.memory_model != ir::MemoryModel::qemu_exclusive_scalar_reference ||
          node.width != 1)
        return Outcome::invalid_group;
    } else if (node.op == Op::read) {
      const auto* cell = Find(state, node.storage);
      if (cell == nullptr || cell->value.width() != node.width) return Outcome::invalid_state;
    } else if (arity != 0) {
      const auto width0 = nodes[node.inputs[0]].width;
      const auto width1 = arity > 1 ? nodes[node.inputs[1]].width : 0;
      switch (node.op) {
        case Op::load:
        case Op::store:
          if (width0 != 64 || node.width % 8 != 0 || node.width > 128 ||
              (node.op == Op::store && width1 != node.width) || node.access.alignment == 0 ||
              (node.access.alignment & (node.access.alignment - 1)) != 0 ||
              (node.access.byte_order != ir::ByteOrder::little &&
               node.access.byte_order != ir::ByteOrder::big))
            return Outcome::invalid_group;
          if (memory == nullptr) return Outcome::unsupported;
          if (instruction.memory_model != ir::MemoryModel::atomic_scalar_reference)
            return Outcome::unsupported;
          break;
        case Op::exclusive_load:
        case Op::exclusive_store: {
          const auto access_width = node.op == Op::exclusive_load ? node.width : width1;
          if (width0 != 64 || (access_width != 32 && access_width != 64) ||
              (node.op == Op::exclusive_store && node.width != 32) ||
              node.access.alignment != access_width / 8 ||
              node.access.byte_order != ir::ByteOrder::little || !node.access.decline_on_unaligned)
            return Outcome::invalid_group;
          if (memory == nullptr) return Outcome::unsupported;
          if (instruction.memory_model != ir::MemoryModel::qemu_exclusive_scalar_reference)
            return Outcome::unsupported;
          break;
        }
        case Op::write: {
          const auto* cell = Find(state, node.storage);
          if (node.width != width0) return Outcome::invalid_group;
          if (cell == nullptr || cell->value.width() != node.width) return Outcome::invalid_state;
          break;
        }
        case Op::extract:
          if (node.immediate >= width0 || node.width > width0 - node.immediate) {
            return Outcome::invalid_group;
          }

          break;
        case Op::zext:
          if (node.width < width0) return Outcome::invalid_group;
          break;
        case Op::select:
          if (width0 != 1 || width1 != node.width || nodes[node.inputs[2]].width != node.width)
            return Outcome::invalid_group;
          break;
        case Op::equal:
        case Op::unsigned_less:
        case Op::signed_less:
          if (width0 != width1 || node.width != 1) return Outcome::invalid_group;
          break;
        case Op::shl:
        case Op::lshr:
        case Op::ashr:
          if (node.width != width0) return Outcome::invalid_group;
          break;
        default:
          if (node.width != width0 || (arity == 2 && width0 != width1) ||
              (ir::NarrowOnly(node.op) && node.width > 64)) {
            return Outcome::invalid_group;
          }

          break;
      }
    }
  }

  for (std::size_t i = 0; i < writes.size(); ++i) {
    const auto& write = writes[i];
    if (write.value >= end_node || ir::Descriptor(nodes[write.value].op) == nullptr ||
        !ir::Descriptor(nodes[write.value].op)->produces_value) {
      return Outcome::invalid_group;
    }

    const auto* cell = Find(state, write.storage);
    if (cell == nullptr || cell->value.width() != nodes[write.value].width) {
      return Outcome::invalid_state;
    }

    for (std::size_t j = 0; j < i; ++j) {
      if (write.storage == writes[j].storage) return Outcome::invalid_group;
    }
  }

  // A replacement may name a destination above the nodes, so the transfer is
  // checked against what this instruction can read.
  if (instruction.transfer &&
      !ir::ValidTransferWidths(*instruction.transfer, [&](ir::ValueId id) -> unsigned {
        if (id < end_node) {
          const auto* descriptor = ir::Descriptor(nodes[id].op);
          return descriptor != nullptr && descriptor->produces_value ? nodes[id].width : 0;
        }

        const auto named = static_cast<std::size_t>(id) - nodes.size();
        return id >= nodes.size() && named < instruction.destinations.size() &&
                       instruction.destinations[named]
                   ? instruction.destinations[named]->width()
                   : 0;
      }))
    return Outcome::invalid_group;

  std::vector<Cell> pending;
  pending.reserve(instruction.node_count + writes.size());
  std::vector<AccessEvent> events;
  events.reserve(instruction.node_count);
  auto pending_exclusive = state.exclusive;
  std::optional<Memory::Transaction> transaction;
  if (memory != nullptr) transaction.emplace(memory->Begin(budget));
  const auto publish = [&]() {
    if (transaction) transaction->Commit();
    state.exclusive = pending_exclusive;
    for (auto& cell : pending) {
      for (auto& destination : state.cells) {
        if (destination.id == cell.id) {
          destination.value = std::move(cell.value);
          break;
        }
      }
    }
  };

  const auto access_failure = [&](MemoryStatus status, std::uint64_t address, ir::ValueId index,
                                  bool write) -> ExecutionResult {
    FaultKind kind;
    switch (status) {
      case MemoryStatus::unmapped:
        kind = FaultKind::unmapped;
        break;
      case MemoryStatus::permission:
        kind = FaultKind::permission;
        break;
      case MemoryStatus::address_overflow:
        kind = FaultKind::address_overflow;
        break;
      case MemoryStatus::resource_limit:
        return Outcome::resource_limit;
      default:
        return Outcome::unsupported;
    }

    publish();
    return {Outcome::fault, Fault{kind, address, index, write}, std::move(events)};
  };

  for (std::size_t index = instruction.first_node; index < end_node; ++index) {
    const auto& node = nodes[index];
    if (budget.try_consume({limits.max_width, 0}) != BudgetDecline::none) {
      return Outcome::resource_limit;
    }

    if (instruction.ssa_block) {
      const auto& block = *instruction.ssa_block;
      const auto id = static_cast<ir::ValueId>(index);
      if (std::binary_search(block.dead_pure_nodes.begin(), block.dead_pure_nodes.end(), id)) {
        values.emplace_back();
        continue;
      }

      const auto disabled =
          std::binary_search(block.disabled_effects.begin(), block.disabled_effects.end(), id);
      if (disabled) {
        const auto fold = std::lower_bound(
            block.constant_loads.begin(), block.constant_loads.end(), id,
            [](const ir::SsaConstantLoad& item, ir::ValueId value) { return item.node < value; });
        const bool folded = fold != block.constant_loads.end() && fold->node == id;
        if (!folded || !fold->skip_access) {
          const auto status = memory ? OmittedAccessPrecondition(
                                           *memory, values[node.inputs[0]]->word(0), node.width / 8,
                                           node.access.alignment, node.op == ir::Op::store, budget)
                                     : Outcome::unsupported;
          if (status != Outcome::completed) return status;
        }

        const auto frame =
            std::lower_bound(block.frame_accesses.begin(), block.frame_accesses.end(), id,
                             [](const ir::SsaFrameAccess& access, ir::ValueId value) {
                               return access.node < value;
                             });
        if (frame != block.frame_accesses.end() && frame->node == id) {
          if (!instruction.frame_values || frame->phi >= instruction.frame_values->size())
            return Outcome::invalid_group;
          auto& value = (*instruction.frame_values)[frame->phi];
          if (node.op == ir::Op::store) {
            auto copied = Copy(*values[node.inputs[1]], budget, limits.max_width);
            if (!copied) return Outcome::resource_limit;
            value = std::move(*copied);
            values.emplace_back();
          } else {
            if (!value) {
              if (!transaction) return Outcome::unsupported;
              const auto loaded =
                  transaction->Read(values[node.inputs[0]]->word(0), node.width / 8);
              if (loaded.status != MemoryStatus::ok) return Outcome::unsupported;
              std::array<std::uint64_t, 2> words{};
              for (unsigned byte = 0; byte < node.width / 8; ++byte) {
                const auto position = node.access.byte_order == ir::ByteOrder::little
                                          ? byte
                                          : node.width / 8 - byte - 1;
                words[byte / 8] |= std::uint64_t{loaded.bytes[position]} << (byte % 8 * 8);
              }

              auto initial = BitVector::from_words(node.width,
                                                   std::span(words).first((node.width / 8 + 7) / 8),
                                                   limits.max_width, budget);
              if (!initial) return Outcome::resource_limit;
              value = std::move(*initial);
            }

            auto copied = Copy(*value, budget, limits.max_width);
            if (!copied) return Outcome::resource_limit;
            values.emplace_back(std::move(*copied));
          }

          continue;
        }

        if (node.op == ir::Op::store) {
          values.emplace_back();
          continue;
        }

        if (folded && fold->kind == ir::SsaConstantKind::bounded_table) {
          if (fold->table_index >= values.size() || !values[fold->table_index])
            return Outcome::invalid_group;
          const auto row = values[fold->table_index]->word(0);
          if (row >= fold->table_bytes.size()) return Outcome::unsupported;
          if (budget.try_consume({1 + node.width / 8, 0}) != BudgetDecline::none)
            return Outcome::resource_limit;
          std::uint64_t bits = 0;
          for (unsigned byte = 0; byte < node.width / 8; ++byte) {
            const auto shift =
                node.access.byte_order == ir::ByteOrder::little ? byte : node.width / 8 - 1 - byte;
            bits |= std::uint64_t(fold->table_bytes[row][byte]) << (shift * 8);
          }

          auto replacement = BitVector::from_u64(node.width, bits, limits.max_width, budget);
          if (!replacement) return Outcome::resource_limit;
          values.emplace_back(std::move(*replacement));
          continue;
        }

        const auto selected = folded && fold->condition && values[*fold->condition]->word(0) == 0;
        const auto literal = folded ? (selected ? fold->alternative_value
                                       : fold->kind == ir::SsaConstantKind::literal
                                           ? fold->value
                                           : *context.load_bias + fold->value)
                                    : 0;
        auto replacement = BitVector::from_u64(node.width, literal, limits.max_width, budget);
        if (!replacement) return Outcome::resource_limit;
        values.emplace_back(std::move(*replacement));
        continue;
      }

      const auto omitted_store =
          std::any_of(block.store_omissions.begin(), block.store_omissions.end(),
                      [&](const ir::StoreOmission& omission) { return omission.store == id; });
      if (omitted_store) {
        const auto status =
            memory ? OmittedAccessPrecondition(*memory, values[node.inputs[0]]->word(0),
                                               node.width / 8, node.access.alignment, true, budget)
                   : Outcome::unsupported;
        if (status != Outcome::completed) return status;
        values.emplace_back();
        continue;
      }

      for (const auto& pair : block.paired_load_omissions) {
        if (pair.loads[0] != id && pair.loads[1] != id) continue;
        if (pair.loads[0] == id) {
          for (const auto load : pair.loads) {
            const auto& access = nodes[load];
            const auto status =
                memory ? OmittedAccessPrecondition(*memory, values[access.inputs[0]]->word(0),
                                                   access.width / 8, access.access.alignment, false,
                                                   budget)
                       : Outcome::unsupported;
            if (status != Outcome::completed) return status;
          }
        }

        values.emplace_back();
        break;
      }

      if (values.size() == index + 1) continue;
    }

    if (instruction.recovered &&
        instruction.recovered->omits_store(static_cast<ir::ValueId>(index))) {
      const auto status =
          memory ? OmittedAccessPrecondition(*memory, values[node.inputs[0]]->word(0),
                                             node.width / 8, node.access.alignment, true, budget)
                 : Outcome::unsupported;
      if (status != Outcome::completed) return status;
      values.emplace_back();
      continue;
    }

    if (instruction.recovered) {
      if (const auto* pair =
              instruction.recovered->first_omitted_load(static_cast<ir::ValueId>(index))) {
        for (const auto load : pair->loads) {
          const auto& access = nodes[load];
          const auto status = memory ? OmittedAccessPrecondition(
                                           *memory, values[access.inputs[0]]->word(0),
                                           access.width / 8, access.access.alignment, false, budget)
                                     : Outcome::unsupported;
          if (status != Outcome::completed) return status;
        }

        values.emplace_back();
        continue;
      }

      if (instruction.recovered->omits_second_load(static_cast<ir::ValueId>(index))) {
        values.emplace_back();
        continue;
      }
    }

    if (node.op == ir::Op::write) {
      auto value = Copy(*values[node.inputs[0]], budget, limits.max_width);
      if (!value) return Outcome::resource_limit;
      pending.push_back({node.storage, std::move(*value)});
      values.emplace_back();
      continue;
    }

    if (node.op == ir::Op::exclusive_clear) {
      pending_exclusive.reset();
      values.emplace_back();
      continue;
    }

    if (node.op == ir::Op::exclusive_load || node.op == ir::Op::exclusive_store) {
      const auto address = values[node.inputs[0]]->word(0);

      // This profile has no TBI/MTE configuration with which to clean a tagged VA.
      if (address >> 56 != 0) return Outcome::unsupported;
      const bool store = node.op == ir::Op::exclusive_store;
      const unsigned size = (store ? nodes[node.inputs[1]].width : node.width) / 8;
      AccessEvent event{static_cast<ir::ValueId>(index), address, size, store, false};
      event.conditional = store;
      if (store && (!pending_exclusive || pending_exclusive->address != address)) {
        // QEMU takes the monitor-miss branch before any memory access.
        auto status = BitVector::from_u64(32, 1, limits.max_width, budget);
        if (!status) return Outcome::resource_limit;
        pending_exclusive.reset();
        values.emplace_back(std::move(*status));
        continue;
      }

      if (store && pending_exclusive->size != size) return Outcome::unsupported;
      if (address % size != 0) {
        if (node.access.decline_on_unaligned) return Outcome::unsupported;
        event.performed = false;
        events.push_back(event);
        publish();
        return {Outcome::fault, Fault{FaultKind::alignment, address, event.operation, store},
                std::move(events)};
      }

      if (store) {
        const auto checked = memory->CheckAccess(address, size, true, budget);
        if (checked.status != MemoryStatus::ok) {
          event.performed = false;
          events.push_back(event);
          return access_failure(checked.status, checked.fault_address, event.operation, true);
        }
      }

      const auto loaded = transaction->Read(address, size);
      if (loaded.status != MemoryStatus::ok) {
        event.performed = false;
        events.push_back(event);
        return access_failure(loaded.status, loaded.fault_address, event.operation, store);
      }

      if (!store) {
        event.completed = true;
        event.bytes = loaded.bytes;
        events.push_back(event);
        std::array<std::uint64_t, 2> words{};
        for (unsigned i = 0; i < size; ++i)
          words[i / 8] |= std::uint64_t{loaded.bytes[i]} << ((i % 8) * 8);
        auto value = BitVector::from_words(node.width, std::span(words).first((size + 7) / 8),
                                           limits.max_width, budget);
        if (!value) return Outcome::resource_limit;
        ExclusiveReservation reservation{address, size, {}, memory->identity()};
        std::copy_n(loaded.bytes.begin(), size, reservation.bytes.begin());
        pending_exclusive = reservation;
        values.emplace_back(std::move(*value));
      } else {
        const auto& value = *values[node.inputs[1]];
        for (unsigned i = 0; i < size; ++i) {
          event.bytes[i] = static_cast<std::uint8_t>(value.word(i / 8) >> ((i % 8) * 8));
        }

        const bool same = std::equal(loaded.bytes.begin(), loaded.bytes.begin() + size,
                                     pending_exclusive->bytes.begin());
        if (same) {
          const auto written = transaction->Write(address, std::span(event.bytes).first(size));
          if (written.status != MemoryStatus::ok) {
            event.performed = false;
            events.push_back(event);
            return access_failure(written.status, written.fault_address, event.operation, true);
          }
        }

        auto status = BitVector::from_u64(32, same ? 0 : 1, limits.max_width, budget);
        if (!status) return Outcome::resource_limit;
        event.completed = true;
        event.performed = same;
        events.push_back(event);
        pending_exclusive.reset();
        values.emplace_back(std::move(*status));
      }

      continue;
    }

    if (node.op == ir::Op::load || node.op == ir::Op::store) {
      const auto address = values[node.inputs[0]]->word(0);
      if (address >> 56 != 0) return Outcome::unsupported;
      const bool write = node.op == ir::Op::store;
      const unsigned size = node.width / 8;
      AccessEvent event{static_cast<ir::ValueId>(index), address, size, write, false};
      if (address % node.access.alignment != 0) {
        if (node.access.decline_on_unaligned) return Outcome::unsupported;
        events.push_back(event);
        publish();
        return {Outcome::fault, Fault{FaultKind::alignment, address, event.operation, write},
                std::move(events)};
      }

      if (write) {
        const auto& value = *values[node.inputs[1]];
        for (unsigned i = 0; i < size; ++i) {
          const unsigned position =
              node.access.byte_order == ir::ByteOrder::little ? i : size - i - 1;
          event.bytes[position] = static_cast<std::uint8_t>(value.word(i / 8) >> ((i % 8) * 8));
        }

        const auto result = transaction->Write(address, std::span(event.bytes).first(size));
        event.completed = result.status == MemoryStatus::ok;
        events.push_back(event);
        if (!event.completed)
          return access_failure(result.status, result.fault_address, event.operation, true);
        values.emplace_back();
      } else {
        const auto result = transaction->Read(address, size);
        event.completed = result.status == MemoryStatus::ok;
        event.bytes = result.bytes;
        events.push_back(event);
        if (!event.completed)
          return access_failure(result.status, result.fault_address, event.operation, false);
        std::array<std::uint64_t, 2> words{};
        for (unsigned i = 0; i < size; ++i) {
          const unsigned position =
              node.access.byte_order == ir::ByteOrder::little ? i : size - i - 1;
          words[i / 8] |= std::uint64_t{result.bytes[position]} << ((i % 8) * 8);
        }

        auto value = BitVector::from_words(node.width, std::span(words).first((size + 7) / 8),
                                           limits.max_width, budget);
        if (!value) return Outcome::resource_limit;
        values.emplace_back(std::move(*value));
      }

      continue;
    }

    if (node.op == ir::Op::read && instruction.ssa_block) {
      const auto& reads = instruction.ssa_block->reads;
      if (budget.try_consume({reads.size(), 0}) != BudgetDecline::none)
        return Outcome::resource_limit;
      const auto found = std::find_if(reads.begin(), reads.end(),
                                      [&](const ir::SsaRead& read) { return read.node == index; });
      if (found != reads.end() && found->predecessor_copy) {
        const auto source = *found->predecessor_copy;
        if (!instruction.predecessor || source.block != *instruction.predecessor ||
            !instruction.predecessor_values ||
            source.index >= instruction.predecessor_values->size() ||
            !(*instruction.predecessor_values)[source.index])
          return Outcome::invalid_group;
        auto copied =
            Copy(*(*instruction.predecessor_values)[source.index], budget, limits.max_width);
        if (!copied) return Outcome::resource_limit;
        values.emplace_back(std::move(*copied));
        continue;
      }
    }

    auto evaluate = [&]() -> BitVectorResult {
      using ir::Op;
      if (node.op == Op::constant) {
        return BitVector::from_u64(node.width, node.immediate, limits.max_width, budget);
      }

      if (node.op == Op::image_address) {
        return BitVector::from_u64(64, *context.load_bias + node.immediate, limits.max_width,
                                   budget);
      }

      if (node.op == Op::read)
        return Copy(Find(state, node.storage)->value, budget, limits.max_width);
      // Validation holds these to 64 bits, so one word is the whole operand.
      const auto narrow = [&](std::array<std::uint64_t, 2> operands) -> BitVectorResult {
        const auto folded =
            ir::FoldPure(node, std::span(operands).first(ir::Descriptor(node.op)->arity),
                         values[node.inputs[0]]->width());
        if (!folded) return BitVectorDecline::invalid_width;
        return BitVector::from_u64(node.width, *folded, limits.max_width, budget);
      };

      const auto& a = *values[node.inputs[0]];
      if (node.op == Op::bit_not) return a.bit_not(budget);
      if (node.op == Op::clz || node.op == Op::rbit) return narrow({a.word(0), 0});
      if (node.op == Op::extract)
        return Slice(a, node.immediate, node.width, budget, limits.max_width);
      if (node.op == Op::zext) return Slice(a, 0, node.width, budget, limits.max_width);
      const auto& b = *values[node.inputs[1]];
      switch (node.op) {
        case Op::add:
          return a.add(b, budget);
        case Op::sub:
          return a.sub(b, budget);
        case Op::mul:
          return a.mul(b, budget);
        case Op::bit_and:
          return a.bit_and(b, budget);
        case Op::bit_or:
          return a.bit_or(b, budget);
        case Op::bit_xor:
          return a.bit_xor(b, budget);
        case Op::shl:
          return a.shl(ShiftCount(b), budget);
        case Op::lshr:
          return a.lshr(ShiftCount(b), budget);
        case Op::ashr:
          return a.ashr(ShiftCount(b), budget);
        case Op::select:
          return Copy(a.bit(0) ? b : *values[node.inputs[2]], budget, limits.max_width);
        case Op::equal:
          return BitVector::from_u64(1, a == b, limits.max_width, budget);
        case Op::unsigned_less:
        case Op::signed_less:
          return BitVector::from_u64(1, Less(a, b, node.op == Op::signed_less), limits.max_width,
                                     budget);
        case Op::udiv:
        case Op::sdiv:
        case Op::umulh:
        case Op::smulh:
          return narrow({a.word(0), b.word(0)});
        default:
          return BitVectorDecline::invalid_width;
      }
    };

    auto result = evaluate();
    if (!result) return Outcome::resource_limit;
    values.emplace_back(std::move(*result));
  }

  for (std::size_t index = 0; index < writes.size(); ++index) {
    const auto& write = writes[index];
    if (instruction.ssa_block && instruction.ssa_boundary) {
      const auto& retired = instruction.ssa_block->dead_storage_writes;
      if (budget.try_consume({std::uint64_t{1} + std::bit_width(retired.size()), 0}) !=
          BudgetDecline::none)
        return Outcome::resource_limit;
      const auto found = std::lower_bound(
          retired.begin(), retired.end(), std::pair{*instruction.ssa_boundary, index},
          [](const ir::SsaDeadStorageWrite& mark, const auto& key) {
            return mark.boundary < key.first ||
                   (mark.boundary == key.first && mark.index < key.second);
          });
      if (found != retired.end() && found->boundary == *instruction.ssa_boundary &&
          found->index == index)
        continue;
    }

    auto value = Copy(*values[write.value], budget, limits.max_width);
    if (!value) return Outcome::resource_limit;
    pending.push_back({write.storage, std::move(*value)});
  }

  std::optional<ResolvedTransfer> transfer;
  if (instruction.transfer) {
    const auto& control = *instruction.transfer;
    const auto resolved = [&](ir::ValueId id) -> const BitVector* {
      if (id < values.size()) return values[id] ? &*values[id] : nullptr;
      const auto named = static_cast<std::size_t>(id) - nodes.size();
      return id >= nodes.size() && named < instruction.destinations.size() &&
                     instruction.destinations[named]
                 ? &*instruction.destinations[named]
                 : nullptr;
    };

    const auto* target = resolved(control.target);
    if (target == nullptr) return Outcome::invalid_group;
    transfer = ResolvedTransfer{control.kind, target->word(0), {}, {}};
    if (control.condition) {
      const auto* condition = resolved(*control.condition);
      const auto* alternative = control.alternative ? resolved(*control.alternative) : nullptr;
      if (condition == nullptr || alternative == nullptr) return Outcome::invalid_group;
      transfer->condition = condition->bit(0);
      if (!*transfer->condition) transfer->target = alternative->word(0);
    }

    if (control.continuation) {
      const auto* continuation = resolved(*control.continuation);
      if (continuation == nullptr) return Outcome::invalid_group;
      transfer->continuation = continuation->word(0);
    }
  }

  publish();
  return {Outcome::completed, {}, std::move(events), transfer};
}

ExecutionResult ExecuteGroup(const ir::Group& group, State& state, Memory* memory, Budget& budget,
                             Limits limits, ExecutionContext context) {
  if (group.nodes().size() > limits.max_nodes ||
      budget.try_consume({0, group.nodes().size() * sizeof(std::optional<BitVector>)}) !=
          BudgetDecline::none)
    return Outcome::resource_limit;
  std::vector<std::optional<BitVector>> values;
  values.reserve(group.nodes().size());
  return ExecuteImpl({group.nodes(), 0, group.nodes().size(), group.writes(), group.transfer(),
                      group.memory_model()},
                     state, memory, budget, limits, context, values);
}

}  // namespace

Outcome Execute(const ir::Group& group, State& state, Budget& budget, Limits limits,
                ExecutionContext context) {
  if (group.transfer()) return Outcome::unsupported;
  return ExecuteGroup(group, state, nullptr, budget, limits, context).outcome;
}

ExecutionResult ExecuteDetailed(const ir::Group& group, State& state, Budget& budget, Limits limits,
                                ExecutionContext context) {
  return ExecuteGroup(group, state, nullptr, budget, limits, context);
}

ExecutionResult Execute(const ir::Group& group, State& state, Memory& memory, Budget& budget,
                        Limits limits, ExecutionContext context) {
  return ExecuteGroup(group, state, &memory, budget, limits, context);
}

BlockExecutionResult ExecuteBlock(const ir::Block& block, State& state, Memory& memory,
                                  Budget& budget, Limits limits, ExecutionContext context) {
  BlockExecutionResult result;
  if (!block.sources().empty()) result.source_address = block.sources().front().source_address();
  ir::BlockLimits block_limits;
  block_limits.max_storage = limits.max_cells;
  block_limits.max_width = limits.max_width;
  const auto valid = ir::Validate(block, budget, block_limits);
  if (valid != ir::BlockDecline::none) {
    result.outcome = valid == ir::BlockDecline::resource_limit        ? Outcome::resource_limit
                     : valid == ir::BlockDecline::unsupported_control ? Outcome::unsupported
                                                                      : Outcome::invalid_group;
    return result;
  }

  // Reserve all persistent slots and trace entries before publishing any state.
  // Later instruction failure may retain the prefix, never an unrecorded step.
  if (budget.try_consume({0, block.nodes().size() * sizeof(std::optional<BitVector>) +
                                 block.boundaries().size() * sizeof(ExecutionResult)}) !=
      BudgetDecline::none) {
    result.outcome = Outcome::resource_limit;
    return result;
  }

  std::vector<std::optional<BitVector>> values;
  values.reserve(block.nodes().size());
  result.trace.reserve(block.boundaries().size());
  for (std::size_t index = 0; index < block.boundaries().size(); ++index) {
    const auto& boundary = block.boundaries()[index];
    const auto& source = block.sources()[index];
    result.source_address = source.source_address();
    auto step = ExecuteImpl({block.nodes(), boundary.first_node, boundary.node_count,
                             boundary.writes, boundary.transfer, source.memory_model()},
                            state, &memory, budget, limits, context, values);
    for (auto& event : step.events) event.operation = block.origins()[event.operation].operation;
    if (step.fault) step.fault->operation = block.origins()[step.fault->operation].operation;
    result.outcome = step.outcome;
    result.trace.push_back(std::move(step));
    if (result.outcome != Outcome::completed) {
      values.resize(boundary.first_node);
      return result;
    }

    ++result.completed_boundaries;
  }

  result.outcome = Outcome::completed;
  return result;
}

namespace {
PathExecutionResult ExecutePathImpl(const ir::Path& path, const ir::RecoveredPath* recovered,
                                    State& state, Memory& memory, Budget& budget, Limits limits,
                                    ExecutionContext context, ir::ImageFacts facts) {
  PathExecutionResult result;
  if (!path.sources().empty()) result.source_address = path.sources().front().source_address();
  ir::BlockLimits path_limits;
  path_limits.max_storage = limits.max_cells;
  path_limits.max_width = limits.max_width;
  const auto valid = recovered ? ir::ValidateRecoveredPath(*recovered, budget, path_limits, facts)
                               : ir::ValidatePath(path, budget, path_limits);
  if (valid != ir::BlockDecline::none) {
    result.outcome = valid == ir::BlockDecline::resource_limit        ? Outcome::resource_limit
                     : valid == ir::BlockDecline::unsupported_control ? Outcome::unsupported
                                                                      : Outcome::invalid_group;
    return result;
  }

  if (!context.load_bias) {
    result.outcome = Outcome::unsupported;
    return result;
  }

  const auto lookup_work = recovered ? std::bit_width(recovered->rewrites().size()) + 1 : 0;
  const auto omission_work = recovered
                                 ? std::bit_width(recovered->omissions().size()) +
                                       std::bit_width(recovered->paired_load_omissions().size()) + 2
                                 : 0;
  if (budget.try_consume(
          {path.boundaries().size() * (1 + lookup_work) + path.nodes().size() * omission_work,
           path.nodes().size() * sizeof(std::optional<BitVector>) +
               path.boundaries().size() * sizeof(ExecutionResult)}) != BudgetDecline::none) {
    result.outcome = Outcome::resource_limit;
    return result;
  }

  // A recovered transfer's destinations belong to no instruction, so they are
  // evaluated once, here, and read only where a replacement names them.
  const auto named = recovered ? recovered->destinations() : std::span<const ir::Node>{};
  std::vector<std::optional<BitVector>> destinations;
  if (budget.try_consume({named.size(), named.size() * sizeof(std::optional<BitVector>)}) !=
      BudgetDecline::none) {
    result.outcome = Outcome::resource_limit;
    return result;
  }

  destinations.reserve(named.size());
  for (const auto& node : named) {
    if (node.op != ir::Op::image_address || node.width != 64) {
      result.outcome = Outcome::invalid_group;
      return result;
    }

    auto value =
        BitVector::from_u64(64, *context.load_bias + node.immediate, limits.max_width, budget);
    if (!value) {
      result.outcome = Outcome::resource_limit;
      return result;
    }

    destinations.emplace_back(std::move(*value));
  }

  std::vector<std::optional<BitVector>> values;
  values.reserve(path.nodes().size());
  result.trace.reserve(path.boundaries().size());
  for (std::size_t index = 0; index < path.boundaries().size(); ++index) {
    const auto& boundary = path.boundaries()[index];
    const auto& source = path.sources()[index];
    result.source_address = source.source_address();
    result.runtime_next.reset();
    const auto transfer = recovered ? recovered->effective_transfer(index) : boundary.transfer;
    auto step =
        ExecuteImpl({path.nodes(), boundary.first_node, boundary.node_count, boundary.writes,
                     transfer, source.memory_model(), destinations, recovered},
                    state, &memory, budget, limits, context, values);
    for (auto& event : step.events) event.operation = path.origins()[event.operation].operation;
    if (step.fault) step.fault->operation = path.origins()[step.fault->operation].operation;
    result.outcome = step.outcome;
    result.trace.push_back(std::move(step));
    if (result.outcome != Outcome::completed) {
      values.resize(boundary.first_node);
      result.stop = result.outcome == Outcome::fault ? PathStop::fault : PathStop::declined;
      return result;
    }

    ++result.completed_boundaries;
    const auto& executed = result.trace.back();
    result.runtime_next =
        executed.transfer ? executed.transfer->target
                          : *context.load_bias + source.source_address() + source.bytes().size();
    const auto expected = path.expected_successor(index);
    if (expected && *result.runtime_next != *context.load_bias + *expected) {
      result.stop = PathStop::diverged;
      return result;
    }
  }

  result.stop = PathStop::completed;
  return result;
}
}  // namespace

PathExecutionResult ExecutePath(const ir::Path& path, State& state, Memory& memory, Budget& budget,
                                Limits limits, ExecutionContext context) {
  return ExecutePathImpl(path, nullptr, state, memory, budget, limits, context, {});
}

PathExecutionResult ExecuteRecoveredPath(const ir::RecoveredPath& path, State& state,
                                         Memory& memory, Budget& budget, Limits limits,
                                         ExecutionContext context, ir::ImageFacts facts) {
  return ExecutePathImpl(path.basis(), &path, state, memory, budget, limits, context, facts);
}

SsaExecutionResult ExecuteSsa(const ir::SsaGraph& graph, std::span<const ir::Group> sources,
                              ir::SsaHandle entry, State& state, Memory& memory, Budget& budget,
                              ExecutionContext context, SsaCallee callee, std::size_t max_blocks,
                              Limits limits) {
  SsaExecutionResult result;
  if (!context.load_bias || !graph.Get(entry) ||
      std::find(graph.entries().begin(), graph.entries().end(), entry) == graph.entries().end()) {
    result.outcome = Outcome::invalid_group;
    return result;
  }

  const auto valid = ir::ValidateSsaWithSources(graph, sources, budget);
  if (valid != ir::SsaDecline::none) {
    result.outcome =
        valid == ir::SsaDecline::resource_limit ? Outcome::resource_limit : Outcome::invalid_group;
    return result;
  }

  const auto* first = graph.Get(entry);
  if (budget.try_consume({first->frame_phis.size(),
                          first->frame_phis.size() * sizeof(std::optional<BitVector>)}) !=
      BudgetDecline::none) {
    result.outcome = Outcome::resource_limit;
    return result;
  }

  std::vector<std::optional<BitVector>> frame_values(first->frame_phis.size());
  auto current = entry;
  std::optional<ir::SsaHandle> predecessor;
  std::vector<std::optional<BitVector>> predecessor_values;
  for (std::size_t step = 0; step < max_blocks; ++step) {
    const auto* block = graph.Get(current);
    if (block && block->opaque) {
      // A target-classified trap stops here as a trap edge would; the edge into
      // it already set the runtime PC. Other opaque control is not modeled.
      const bool trap = block->edges.size() == 1 && block->edges[0].kind == ir::SsaEdgeKind::trap &&
                        !block->edges[0].target_block &&
                        block->edges[0].assumptions.declared_opaque_control;
      result.outcome = trap ? Outcome::completed : Outcome::unsupported;
      if (trap) result.stop = SsaStop::trap;
      return result;
    }

    if (!block || block->original_sources.size() != block->boundaries.size() ||
        block->source_bytes.size() != block->boundaries.size()) {
      result.outcome = Outcome::invalid_group;
      return result;
    }

    const bool grows = result.visited_blocks.size() == result.visited_blocks.capacity();
    const auto allocation = grows ? (result.visited_blocks.size() + 1) * sizeof(ir::SsaHandle) : 0;
    if (budget.try_consume({1, allocation}) != BudgetDecline::none) {
      result.outcome = Outcome::resource_limit;
      return result;
    }

    if (grows) result.visited_blocks.reserve(result.visited_blocks.size() + 1);
    result.visited_blocks.push_back(current);
    if (budget.try_consume({block->nodes.size() + block->boundaries.size(),
                            block->nodes.size() * sizeof(std::optional<BitVector>) +
                                block->destination_nodes.size() *
                                    sizeof(std::optional<BitVector>)}) != BudgetDecline::none) {
      result.outcome = Outcome::resource_limit;
      return result;
    }

    std::vector<std::optional<BitVector>> values;
    values.reserve(block->nodes.size());
    std::vector<std::optional<BitVector>> destinations;
    destinations.reserve(block->destination_nodes.size());
    for (const auto& node : block->destination_nodes) {
      auto value = BitVector::from_u64(
          64,
          node.op == ir::Op::image_address ? *context.load_bias + node.immediate : node.immediate,
          limits.max_width, budget);
      if (!value) {
        result.outcome = Outcome::resource_limit;
        return result;
      }

      destinations.emplace_back(std::move(*value));
    }

    for (std::size_t index = 0; index < block->boundaries.size(); ++index) {
      const auto source = block->original_sources[index];
      if (source >= sources.size() ||
          sources[source].source_address() != block->source_groups[index] ||
          sources[source].bytes().size() != block->source_bytes[index].size()) {
        result.outcome = Outcome::invalid_group;
        return result;
      }

      if (budget.try_consume({block->source_bytes[index].size(), 0}) != BudgetDecline::none) {
        result.outcome = Outcome::resource_limit;
        return result;
      }

      if (!std::equal(block->source_bytes[index].begin(), block->source_bytes[index].end(),
                      sources[source].bytes().begin())) {
        result.outcome = Outcome::invalid_group;
        return result;
      }

      const auto& boundary = block->boundaries[index];
      const auto rewrite =
          std::lower_bound(block->control_rewrites.begin(), block->control_rewrites.end(), index,
                           [](const ir::ConditionalRewrite& item, std::size_t value) {
                             return item.boundary < value;
                           });
      const std::optional<ir::Transfer> transfer =
          rewrite != block->control_rewrites.end() && rewrite->boundary == index
              ? std::optional(rewrite->replacement)
              : boundary.transfer;
      auto executed =
          ExecuteImpl({block->nodes, boundary.first_node, boundary.node_count, boundary.writes,
                       transfer, sources[source].memory_model(), destinations, nullptr, block,
                       &frame_values, predecessor, &predecessor_values, index},
                      state, &memory, budget, limits, context, values);
      if (budget.try_consume({1, sizeof(ExecutionResult)}) != BudgetDecline::none) {
        result.outcome = Outcome::resource_limit;
        return result;
      }

      result.outcome = executed.outcome;
      result.fault = executed.fault;
      result.runtime_pc = *context.load_bias + block->source_groups[index];
      result.trace.push_back(std::move(executed));
      if (result.outcome != Outcome::completed) {
        result.stop = result.outcome == Outcome::fault ? SsaStop::fault : SsaStop::declined;
        return result;
      }

      if (index + 1 < block->boundaries.size()) {
        const auto& completed = result.trace.back();
        auto next = completed.transfer ? completed.transfer->target
                                       : *context.load_bias + block->source_groups[index] +
                                             block->source_bytes[index].size();
        if (completed.transfer && completed.transfer->kind == ir::TransferKind::call) {
          if (!callee || !completed.transfer->continuation) {
            result.outcome = Outcome::unsupported;
            result.stop = SsaStop::unresolved;
            return result;
          }

          const auto returned = callee(completed.transfer->target, state, memory, budget);
          if (returned.outcome != Outcome::completed) {
            result.outcome = returned.outcome;
            return result;
          }

          if (!returned.return_pc || *returned.return_pc != *completed.transfer->continuation) {
            result.outcome = Outcome::unsupported;
            result.stop = SsaStop::unresolved;
            return result;
          }

          next = *returned.return_pc;
        }

        if (next != *context.load_bias + block->source_groups[index + 1]) {
          result.outcome = Outcome::unsupported;
          result.stop = SsaStop::unresolved;
          result.runtime_pc = next;
          return result;
        }
      }
    }

    ++result.completed_blocks;
    if (budget.try_consume({block->edges.size(), 0}) != BudgetDecline::none) {
      result.outcome = Outcome::resource_limit;
      return result;
    }

    const ir::SsaEdge* selected = nullptr;
    for (const auto& edge : block->edges) {
      if (edge.kind == ir::SsaEdgeKind::potential_return) continue;
      if (edge.condition) {
        if (*edge.condition >= values.size() || !values[*edge.condition]) {
          result.outcome = Outcome::invalid_group;
          return result;
        }

        if (values[*edge.condition]->bit(0) != *edge.when) continue;
      }

      if (selected) {
        result.outcome = Outcome::unsupported;
        result.stop = SsaStop::unresolved;
        return result;
      }

      selected = &edge;
    }

    if (!selected) {
      result.outcome = Outcome::unsupported;
      result.stop = SsaStop::unresolved;
      return result;
    }

    const auto runtime_target = [&](const ir::SsaEdge& edge) -> std::optional<std::uint64_t> {
      if (edge.target_kind == ir::SsaTargetKind::image_location)
        return *context.load_bias + edge.address;
      if (edge.target_kind == ir::SsaTargetKind::absolute_runtime) return edge.address;
      if (result.trace.empty() || !result.trace.back().transfer) return {};
      const auto& terminal = *result.trace.back().transfer;
      return edge.kind == ir::SsaEdgeKind::potential_return ? terminal.continuation
                                                            : std::optional(terminal.target);
    };

    if (selected->kind != ir::SsaEdgeKind::callee && selected->kind != ir::SsaEdgeKind::trap) {
      const auto expected = result.trace.back().transfer
                                ? result.trace.back().transfer->target
                                : *context.load_bias + block->source_groups.back() +
                                      block->source_bytes.back().size();
      if (runtime_target(*selected) != expected) {
        result.outcome = Outcome::unsupported;
        result.stop = SsaStop::unresolved;
        return result;
      }
    }

    if (selected->kind == ir::SsaEdgeKind::callee) {
      const auto target = runtime_target(*selected);
      const auto terminal = result.trace.back().transfer;
      if (!target || !callee || !terminal || terminal->kind != ir::TransferKind::call ||
          terminal->target != *target) {
        result.outcome = Outcome::unsupported;
        result.stop = SsaStop::unresolved;
        return result;
      }

      // The graph assumes callees preserve what it keeps across this call. If
      // a callee broke that, the run no longer matches the graph, so stop.
      const auto words = [](const BitVector& value) {
        std::vector<std::uint64_t> out((value.width() + 63) / 64);
        for (unsigned i = 0; i < out.size(); ++i) out[i] = value.word(i);
        return std::pair{value.width(), std::move(out)};
      };

      std::vector<std::pair<ir::StorageId, std::pair<unsigned, std::vector<std::uint64_t>>>> kept;
      if (budget.try_consume({block->phis.size() * (1 + state.cells.size()),
                              block->phis.size() * 64}) != BudgetDecline::none) {
        result.outcome = Outcome::resource_limit;
        return result;
      }

      for (std::size_t index = 0; index < block->phis.size() && index < block->clobbers.size();
           ++index) {
        if (block->clobbers[index]) continue;
        for (const auto& cell : state.cells)
          if (cell.id == block->phis[index].storage) kept.push_back({cell.id, words(cell.value)});
      }

      const auto returned = callee(*target, state, memory, budget);
      result.outcome = returned.outcome;
      if (result.outcome != Outcome::completed) return result;
      for (const auto& [id, before] : kept) {
        const auto after = std::find_if(state.cells.begin(), state.cells.end(),
                                        [&](const Cell& cell) { return cell.id == id; });
        if (after == state.cells.end() || words(after->value) != before) {
          result.outcome = Outcome::unsupported;
          result.stop = SsaStop::declined;
          return result;
        }
      }

      if (budget.try_consume({block->edges.size(), 0}) != BudgetDecline::none) {
        result.outcome = Outcome::resource_limit;
        return result;
      }

      selected = nullptr;
      for (const auto& edge : block->edges) {
        if (edge.kind != ir::SsaEdgeKind::potential_return) continue;
        if (selected) {
          result.outcome = Outcome::invalid_group;
          return result;
        }

        selected = &edge;
      }

      if (!selected) {
        result.outcome = Outcome::unsupported;
        result.stop = SsaStop::unresolved;
        return result;
      }

      if (!returned.return_pc || runtime_target(*selected) != returned.return_pc) {
        result.outcome = Outcome::unsupported;
        result.stop = SsaStop::unresolved;
        return result;
      }
    }

    const auto target = runtime_target(*selected);
    if (target) result.runtime_pc = *target;
    if (selected->kind == ir::SsaEdgeKind::return_) {
      const auto terminal = result.trace.back().transfer;
      if (!target || !terminal || terminal->kind != ir::TransferKind::return_)
        result.outcome = Outcome::unsupported;
      // A return into this population has not left the function. Until a
      // source-bound edge represents it, stop at the actual PC as unresolved.
      if (target) {
        if (budget.try_consume({sources.size(), 0}) != BudgetDecline::none) {
          result.outcome = Outcome::resource_limit;
          return result;
        }

        for (const auto& source : sources)
          if (*target - *context.load_bias >= source.source_address() &&
              *target - *context.load_bias - source.source_address() < source.bytes().size())
            result.outcome = Outcome::unsupported;
      }

      result.stop = result.outcome == Outcome::completed ? SsaStop::returned : SsaStop::unresolved;
      return result;
    }

    if (selected->kind == ir::SsaEdgeKind::trap) {
      result.stop = SsaStop::trap;
      return result;
    }

    if (selected->target_block) {
      const auto* successor = graph.Get(*selected->target_block);
      if (!successor || successor->frame_phis.size() != frame_values.size()) {
        result.outcome = Outcome::invalid_group;
        return result;
      }

      for (std::size_t phi = 0; phi < successor->frame_phis.size(); ++phi) {
        const auto& incoming = successor->frame_phis[phi].incoming;
        if (budget.try_consume({incoming.size(), 0}) != BudgetDecline::none) {
          result.outcome = Outcome::resource_limit;
          return result;
        }

        const auto found = std::find_if(incoming.begin(), incoming.end(), [&](const auto& item) {
          return item.predecessor == current;
        });
        if (found == incoming.end()) {
          result.outcome = Outcome::invalid_group;
          return result;
        }

        const std::optional<BitVector>* expected = nullptr;
        if (found->value.kind == ir::SsaValueKind::node && found->value.index < values.size())
          expected = &values[found->value.index];
        else if (found->value.kind == ir::SsaValueKind::frame_phi &&
                 found->value.index < frame_values.size())
          expected = &frame_values[found->value.index];
        if (!expected || *expected != frame_values[phi]) {
          result.outcome = Outcome::invalid_group;
          return result;
        }
      }

      predecessor = current;
      predecessor_values = std::move(values);
      current = *selected->target_block;
      continue;
    }

    result.outcome = Outcome::unsupported;
    result.stop = SsaStop::unresolved;
    return result;
  }

  result.outcome = Outcome::resource_limit;
  result.stop = SsaStop::step_limit;
  return result;
}

}  // namespace nyx::eval
