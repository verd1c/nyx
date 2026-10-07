#include "nyx/ir/print.hpp"

#include <bit>
#include <charconv>
#include <cstring>
#include <limits>
#include <string_view>
#include <utility>

namespace nyx::ir {
namespace {

PrintDecline Convert(BudgetDecline reason) {
  if (reason == BudgetDecline::work_limit) return PrintDecline::work_limit;
  if (reason == BudgetDecline::byte_limit) return PrintDecline::byte_limit;
  return PrintDecline::none;
}

class Counter {
 public:
  explicit Counter(Budget& budget) : budget_(budget) {}

  bool Text(std::string_view text) {
    // Both traversals are charged before allocation. No output prefix is returned
    // if either deterministic budget is exhausted.
    if (text.size() > (std::numeric_limits<std::size_t>::max() - size) ||
        text.size() > std::numeric_limits<std::uint64_t>::max() / 2) {
      reason = PrintDecline::byte_limit;
      return false;
    }

    reason = Convert(budget_.try_consume({2 * text.size(), 0}));
    if (reason != PrintDecline::none) return false;
    size += text.size();
    if (size >= budget_.remaining().bytes) {
      reason = PrintDecline::byte_limit;
      return false;
    }

    return true;
  }

  std::size_t size = 0;
  PrintDecline reason = PrintDecline::none;

 private:
  Budget& budget_;
};

class Writer {
 public:
  explicit Writer(char* output) : output_(output) {}

  bool Text(std::string_view text) {
    std::memcpy(output_, text.data(), text.size());
    output_ += text.size();
    return true;
  }

 private:
  char* output_;
};

template <class Sink>
bool Number(Sink& sink, std::uint64_t value) {
  char buffer[20];
  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
  return sink.Text(std::string_view(buffer, result.ptr));
}

template <class Sink>
bool EmitNode(std::span<const Node> nodes, std::size_t index, MemoryModel model,
              const Origin* origin, Sink& sink, PrintDecline& reason) {
  const auto& node = nodes[index];
  const auto* operation = Descriptor(node.op);
  if (operation == nullptr || node.width == 0 ||
      (node.op == Op::image_address && node.width != 64) ||
      index > std::numeric_limits<ValueId>::max()) {
    reason = PrintDecline::invalid_group;
    return false;
  }

  if (!sink.Text("{\"id\":") || !Number(sink, index) || !sink.Text(",\"op\":\"") ||
      !sink.Text(operation->name) || !sink.Text("\",\"width\":") || !Number(sink, node.width) ||
      !sink.Text(",\"produces_value\":") ||
      !sink.Text(operation->produces_value ? "true" : "false") || !sink.Text(",\"inputs\":["))
    return false;
  for (unsigned input = 0; input < operation->arity; ++input) {
    if (node.inputs[input] >= index || !Descriptor(nodes[node.inputs[input]].op)->produces_value) {
      reason = PrintDecline::invalid_group;
      return false;
    }

    if ((input != 0 && !sink.Text(",")) || !Number(sink, node.inputs[input])) return false;
  }

  if (!sink.Text("]")) return false;
  if (node.op == Op::constant || node.op == Op::extract || node.op == Op::image_address) {
    if (!sink.Text(",\"immediate\":") || !Number(sink, node.immediate)) return false;
  }

  if (node.op == Op::load || node.op == Op::store || node.op == Op::exclusive_load ||
      node.op == Op::exclusive_store) {
    const bool exclusive = node.op == Op::exclusive_load || node.op == Op::exclusive_store;
    if (model != (exclusive ? MemoryModel::qemu_exclusive_scalar_reference
                            : MemoryModel::atomic_scalar_reference) ||
        nodes[node.inputs[0]].width != 64 || node.width % 8 != 0 || node.width > 128 ||
        (node.op == Op::store && nodes[node.inputs[1]].width != node.width) ||
        (exclusive && ((node.op == Op::exclusive_load && node.width != 32 && node.width != 64) ||
                       (node.op == Op::exclusive_store &&
                        (node.width != 32 || (nodes[node.inputs[1]].width != 32 &&
                                              nodes[node.inputs[1]].width != 64))))) ||
        node.access.alignment == 0 || (node.access.alignment & (node.access.alignment - 1)) != 0 ||
        (exclusive &&
         (node.access.alignment !=
              (node.op == Op::exclusive_load ? node.width : nodes[node.inputs[1]].width) / 8 ||
          !node.access.decline_on_unaligned || node.access.byte_order != ByteOrder::little)) ||
        (node.access.byte_order != ByteOrder::little && node.access.byte_order != ByteOrder::big)) {
      reason = PrintDecline::invalid_group;
      return false;
    }

    if (!sink.Text(",\"access\":{\"byte_order\":\"") ||
        !sink.Text(node.access.byte_order == ByteOrder::little ? "little" : "big") ||
        !sink.Text("\",\"alignment\":") || !Number(sink, node.access.alignment) ||
        !sink.Text(",\"decline_on_unaligned\":") ||
        !sink.Text(node.access.decline_on_unaligned ? "true" : "false") || !sink.Text("}"))
      return false;
  }

  if (node.op == Op::exclusive_clear &&
      (model != MemoryModel::qemu_exclusive_scalar_reference || node.width != 1)) {
    reason = PrintDecline::invalid_group;
    return false;
  }

  if (node.op == Op::write && node.width != nodes[node.inputs[0]].width) {
    reason = PrintDecline::invalid_group;
    return false;
  }

  if (node.op == Op::read || node.op == Op::write) {
    if (!sink.Text(",\"storage\":") || !Number(sink, node.storage)) return false;
  }

  if (origin &&
      (!sink.Text(",\"origin\":{\"boundary\":") || !Number(sink, origin->boundary) ||
       !sink.Text(",\"operation\":") || !Number(sink, origin->operation) || !sink.Text("}")))
    return false;
  return sink.Text("}");
}

template <class Sink>
bool EmitWrites(std::span<const Write> writes, std::span<const Node> nodes, Sink& sink,
                PrintDecline& reason) {
  if (!sink.Text("[")) return false;
  bool first = true;
  for (const auto& write : writes) {
    if (write.value >= nodes.size() || !Descriptor(nodes[write.value].op)->produces_value) {
      reason = PrintDecline::invalid_group;
      return false;
    }

    if ((!first && !sink.Text(",")) || !sink.Text("{\"storage\":") ||
        !Number(sink, write.storage) || !sink.Text(",\"value\":") || !Number(sink, write.value) ||
        !sink.Text("}"))
      return false;
    first = false;
  }

  return sink.Text("]");
}

template <class Sink>
bool EmitTransfer(const std::optional<Transfer>& control, Sink& sink, PrintDecline& reason) {
  if (!control) return sink.Text("null");
  const auto& transfer = *control;
  std::string_view kind;
  switch (transfer.kind) {
    case TransferKind::jump:
      kind = "jump";
      break;
    case TransferKind::conditional:
      kind = "conditional";
      break;
    case TransferKind::call:
      kind = "call";
      break;
    case TransferKind::return_:
      kind = "return";
      break;
    default:
      reason = PrintDecline::invalid_group;
      return false;
  }

  if (!sink.Text("{\"kind\":\"") || !sink.Text(kind) || !sink.Text("\",\"target\":") ||
      !Number(sink, transfer.target))
    return false;
  if (transfer.condition && (!sink.Text(",\"condition\":") || !Number(sink, *transfer.condition))) {
    return false;
  }

  if (transfer.alternative &&
      (!sink.Text(",\"alternative\":") || !Number(sink, *transfer.alternative))) {
    return false;
  }

  if (transfer.continuation &&
      (!sink.Text(",\"continuation\":") || !Number(sink, *transfer.continuation))) {
    return false;
  }

  return sink.Text("}");
}

template <class Sink>
bool Emit(const Group& group, Sink& sink, PrintDecline& reason) {
  if (group.memory_model() != MemoryModel::unspecified &&
      group.memory_model() != MemoryModel::atomic_scalar_reference &&
      group.memory_model() != MemoryModel::qemu_exclusive_scalar_reference) {
    reason = PrintDecline::invalid_group;
    return false;
  }

  if (group.transfer() && !ValidTransfer(*group.transfer(), group.nodes())) {
    reason = PrintDecline::invalid_group;
    return false;
  }

  if (!sink.Text("{\"schema\":1,\"kind\":\"nyx.ir.group\",\"source_address\":") ||
      !Number(sink, group.source_address()) || !sink.Text(",\"bytes\":\""))
    return false;
  constexpr char hex[] = "0123456789abcdef";
  for (const auto byte : group.bytes()) {
    const char pair[] = {hex[byte >> 4], hex[byte & 15]};
    if (!sink.Text(std::string_view(pair, sizeof(pair)))) return false;
  }

  if (!sink.Text("\",\"memory_model\":\"") ||
      !sink.Text(group.memory_model() == MemoryModel::unspecified ? "unspecified"
                 : group.memory_model() == MemoryModel::atomic_scalar_reference
                     ? "atomic_scalar_reference"
                     : "qemu_exclusive_scalar_reference") ||
      !sink.Text("\",\"nodes\":["))
    return false;
  for (std::size_t index = 0; index < group.nodes().size(); ++index) {
    if ((index != 0 && !sink.Text(",")) ||
        !EmitNode(group.nodes(), index, group.memory_model(), nullptr, sink, reason))
      return false;
  }

  return sink.Text("],\"writes\":") && EmitWrites(group.writes(), group.nodes(), sink, reason) &&
         sink.Text(",\"transfer\":") && EmitTransfer(group.transfer(), sink, reason) &&
         sink.Text("}");
}

template <class Region, class Sink>
bool EmitRegion(const Region& block, bool itinerary, Sink& sink, PrintDecline& reason,
                const RecoveredPath* recovered = nullptr) {
  if (!sink.Text(recovered   ? "{\"schema\":1,\"kind\":\"nyx.ir.recovered_path\",\"revision\":"
                 : itinerary ? "{\"schema\":1,\"kind\":\"nyx.ir.path\",\"revision\":"
                             : "{\"schema\":1,\"kind\":\"nyx.ir.block\",\"revision\":") ||
      !Number(sink, recovered ? recovered->revision() : block.revision()) ||
      !sink.Text(itinerary ? ",\"entry_scope\":\"selected_itinerary\",\"sources\":["
                           : ",\"entry_scope\":\"single_entry_straight_line\",\"sources\":["))
    return false;
  for (std::size_t index = 0; index < block.sources().size(); ++index) {
    if ((index != 0 && !sink.Text(",")) || !Emit(block.sources()[index], sink, reason))
      return false;
  }

  if (!sink.Text("],\"nodes\":[")) return false;
  for (std::size_t index = 0; index < block.nodes().size(); ++index) {
    const auto& origin = block.origins()[index];
    if ((index != 0 && !sink.Text(",")) ||
        !EmitNode(block.nodes(), index, block.sources()[origin.boundary].memory_model(), &origin,
                  sink, reason))
      return false;
  }

  if (!sink.Text("],\"boundaries\":[")) return false;
  for (std::size_t index = 0; index < block.boundaries().size(); ++index) {
    const auto& boundary = block.boundaries()[index];
    if ((index != 0 && !sink.Text(",")) || !sink.Text("{\"first_node\":") ||
        !Number(sink, boundary.first_node) || !sink.Text(",\"node_count\":") ||
        !Number(sink, boundary.node_count) || !sink.Text(",\"writes\":") ||
        !EmitWrites(boundary.writes, block.nodes(), sink, reason) || !sink.Text(",\"transfer\":") ||
        !EmitTransfer(recovered ? recovered->effective_transfer(index) : boundary.transfer, sink,
                      reason))
      return false;
    if (itinerary) {
      if (!sink.Text(",\"expected_successor_image\":")) return false;
      if (index + 1 < block.sources().size()) {
        if (!Number(sink, block.sources()[index + 1].source_address())) return false;
      } else if (!sink.Text("null"))
        return false;
    }

    if (!sink.Text("}")) return false;
  }

  if (!sink.Text("]")) return false;
  if (recovered) {
    if (!sink.Text(",\"basis_revision\":") || !Number(sink, block.revision()) ||
        !sink.Text(",\"control_rewrites\":["))
      return false;
    bool first = true;
    for (const auto& edit : recovered->rewrites()) {
      const bool dispatch = edit.rule == RewriteRule::dispatch_branch;
      if ((!first && !sink.Text(",")) ||
          !sink.Text(dispatch ? "{\"rule\":\"dispatch_branch\",\"boundary\":"
                              : "{\"rule\":\"folded_condition\",\"boundary\":") ||
          !Number(sink, edit.boundary) || !sink.Text(",\"condition\":") ||
          !Number(sink, edit.condition) || !sink.Text(",\"condition_value\":") ||
          !sink.Text(edit.condition_value ? "true" : "false") ||
          !sink.Text(",\"from_revision\":") || !Number(sink, edit.from_revision) ||
          !sink.Text(",\"to_revision\":") || !Number(sink, edit.to_revision) ||
          !sink.Text(",\"original\":") || !EmitTransfer(edit.original, sink, reason) ||
          !sink.Text(",\"replacement\":") || !EmitTransfer(edit.replacement, sink, reason))
        return false;
      if (dispatch) {
        if (!sink.Text(",\"when_true\":") || !Number(sink, edit.when_true) ||
            !sink.Text(",\"when_false\":") || !Number(sink, edit.when_false) ||
            !sink.Text(",\"witness\":["))
          return false;
        for (std::size_t read = 0; read < edit.witness.size(); ++read) {
          const auto& entry = edit.witness[read];
          if ((read != 0 && !sink.Text(",")) || !sink.Text("{\"node\":") ||
              !Number(sink, entry.node) || !sink.Text(",\"when\":") ||
              !sink.Text(entry.when ? "true" : "false") || !sink.Text(",\"address\":") ||
              !Number(sink, entry.address) || !sink.Text(",\"width\":") ||
              !Number(sink, entry.width) || !sink.Text(",\"value\":") ||
              !Number(sink, entry.value) || !sink.Text(",\"relocated\":") ||
              !sink.Text(entry.relocated ? "true" : "false") || !sink.Text("}"))
            return false;
        }

        if (!sink.Text("]")) return false;
      }

      if (!sink.Text("}")) return false;
      first = false;
    }

    // The values a recovered transfer names beyond the basis, so a reader can
    // resolve every id the boundaries above refer to.
    if (!sink.Text("],\"destinations\":[")) return false;
    for (std::size_t index = 0; index < recovered->destinations().size(); ++index) {
      if ((index != 0 && !sink.Text(",")) || !sink.Text("{\"value\":") ||
          !Number(sink, recovered->first_destination() + index) ||
          !sink.Text(",\"image_address\":") ||
          !Number(sink, recovered->destinations()[index].immediate) || !sink.Text("}"))
        return false;
    }

    if (!sink.Text("],\"store_omissions\":[")) return false;
    for (std::size_t index = 0; index < recovered->omissions().size(); ++index) {
      const auto& omission = recovered->omissions()[index];
      if ((index != 0 && !sink.Text(",")) ||
          !sink.Text("{\"rule\":\"overwritten_store\",\"store\":") ||
          !Number(sink, omission.store) || !sink.Text(",\"overwriter\":") ||
          !Number(sink, omission.overwriter) || !sink.Text(",\"base_storage\":") ||
          !Number(sink, omission.base_storage) || !sink.Text(",\"offset_bytes\":") ||
          !Number(sink, omission.offset_bytes) || !sink.Text(",\"from_revision\":") ||
          !Number(sink, omission.from_revision) || !sink.Text(",\"to_revision\":") ||
          !Number(sink, omission.to_revision) || !sink.Text("}"))
        return false;
    }

    if (!sink.Text("],\"paired_load_omissions\":[")) return false;
    for (std::size_t index = 0; index < recovered->paired_load_omissions().size(); ++index) {
      const auto& omission = recovered->paired_load_omissions()[index];
      if ((index != 0 && !sink.Text(",")) ||
          !sink.Text("{\"rule\":\"forwarded_pair_load\",\"stores\":[") ||
          !Number(sink, omission.stores[0]) || !sink.Text(",") ||
          !Number(sink, omission.stores[1]) || !sink.Text("],\"loads\":[") ||
          !Number(sink, omission.loads[0]) || !sink.Text(",") || !Number(sink, omission.loads[1]) ||
          !sink.Text("],\"base_storage\":") || !Number(sink, omission.base_storage) ||
          !sink.Text(",\"offset_bytes\":[") || !Number(sink, omission.offset_bytes[0]) ||
          !sink.Text(",") || !Number(sink, omission.offset_bytes[1]) ||
          !sink.Text("],\"from_revision\":") || !Number(sink, omission.from_revision) ||
          !sink.Text(",\"to_revision\":") || !Number(sink, omission.to_revision) || !sink.Text("}"))
        return false;
    }

    if (!sink.Text("]")) return false;
    if (!recovered->omissions().empty() &&
        !sink.Text(
            ",\"store_omission_scope\":\"single-threaded atomic-scalar execution without"
            " resource-limit outcomes, with a mapped writable normal-memory interval at the omitted"
            " access and no asynchronous observer\""))
      return false;
    if (!recovered->paired_load_omissions().empty() &&
        !sink.Text(
            ",\"paired_load_omission_scope\":\"single-threaded atomic-scalar execution without"
            " resource-limit outcomes, with both omitted reads mapped in readable normal memory"
            " at their access and no concurrent writer or observer\""))
      return false;
  }

  return sink.Text("}");
}

template <class Sink>
bool Emit(const RecoveredPath& path, Sink& sink, PrintDecline& reason) {
  return EmitRegion(path.basis(), true, sink, reason, &path);
}

template <class Sink>
bool Emit(const Block& block, Sink& sink, PrintDecline& reason) {
  return EmitRegion(block, false, sink, reason);
}

template <class Sink>
bool Emit(const Path& path, Sink& sink, PrintDecline& reason) {
  return EmitRegion(path, true, sink, reason);
}

template <class Input>
PrintResult Print(const Input& input, Budget& budget) {
  Counter counter(budget);
  auto reason = PrintDecline::none;
  if (!Emit(input, counter, reason)) {
    return {std::nullopt, reason == PrintDecline::none ? counter.reason : reason};
  }

  if (counter.size > std::string().max_size()) return {std::nullopt, PrintDecline::byte_limit};

  // Reserving the exact counted extent avoids uncharged geometric string growth.
  reason = Convert(budget.try_consume({0, counter.size + 1}));
  if (reason != PrintDecline::none) return {std::nullopt, reason};
  std::string result(counter.size, '\0');
  Writer writer(result.data());
  if (!Emit(input, writer, reason)) return {std::nullopt, reason};
  return {std::move(result), PrintDecline::none};
}

}  // namespace

PrintResult PrintJson(const Group& group, Budget& budget) { return Print(group, budget); }

PrintResult PrintJson(const Block& block, Budget& budget) {
  const auto valid = Validate(block, budget);
  if (valid != BlockDecline::none) {
    return {std::nullopt, valid == BlockDecline::resource_limit ? PrintDecline::work_limit
                                                                : PrintDecline::invalid_group};
  }

  return Print(block, budget);
}

PrintResult PrintJson(const Path& path, Budget& budget) {
  const auto valid = ValidatePath(path, budget);
  if (valid != BlockDecline::none) {
    return {std::nullopt, valid == BlockDecline::resource_limit ? PrintDecline::resource_limit
                                                                : PrintDecline::invalid_group};
  }

  return Print(path, budget);
}

PrintResult PrintJson(const RecoveredPath& path, Budget& budget, ImageFacts facts) {
  const auto valid = ValidateRecoveredPath(path, budget, {}, facts);
  if (valid != BlockDecline::none) {
    return {std::nullopt, valid == BlockDecline::resource_limit ? PrintDecline::resource_limit
                                                                : PrintDecline::invalid_group};
  }

  if (budget.try_consume(
          {2 * path.basis().boundaries().size() * (std::bit_width(path.rewrites().size()) + 1),
           0}) != BudgetDecline::none)
    return {std::nullopt, PrintDecline::resource_limit};
  return Print(path, budget);
}

}  // namespace nyx::ir
