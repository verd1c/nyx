#include "nyx/ir/recovered_path.hpp"

#include <algorithm>
#include <atomic>

namespace nyx::ir {
std::uint64_t NextRecoveredPathIdentity() {
  static std::atomic<std::uint64_t> next{1};
  return next.fetch_add(1, std::memory_order_relaxed);
}

std::optional<StorageAddress> ResolveStorageAddress(std::span<const Node> nodes, ValueId id,
                                                    StorageId storage) {
  std::uint64_t offset = 0;
  for (unsigned steps = 0; steps < 16; ++steps) {
    if (id >= nodes.size()) return {};
    const auto& node = nodes[id];
    if (node.width != 64) return {};
    if (node.op == Op::read) {
      if (node.storage != storage) return {};
      return StorageAddress{id, offset};
    }

    if (node.op == Op::add) {
      const auto lhs = node.inputs[0], rhs = node.inputs[1];
      if (lhs >= nodes.size() || rhs >= nodes.size()) return {};
      if (nodes[rhs].op == Op::constant && nodes[rhs].width == 64) {
        offset += nodes[rhs].immediate;
        id = lhs;
        continue;
      }

      if (nodes[lhs].op == Op::constant && nodes[lhs].width == 64) {
        offset += nodes[lhs].immediate;
        id = rhs;
        continue;
      }
    }

    if (node.op == Op::sub && node.inputs[1] < nodes.size() &&
        nodes[node.inputs[1]].op == Op::constant && nodes[node.inputs[1]].width == 64) {
      offset -= nodes[node.inputs[1]].immediate;
      id = node.inputs[0];
      continue;
    }

    return {};
  }

  return {};
}

namespace {
bool SameTransfer(const Transfer& a, const Transfer& b) {
  return a.kind == b.kind && a.target == b.target && a.condition == b.condition &&
         a.alternative == b.alternative && a.continuation == b.continuation;
}

bool SameRead(const ImageRead& a, const ImageRead& b) {
  return a.node == b.node && a.when == b.when && a.address == b.address && a.width == b.width &&
         a.value == b.value && a.relocated == b.relocated;
}

// A destination the rewrite introduced: a literal place, reachable only from a
// replacement transfer, so admitting it adds no operation to the path.
bool Destination(const Node& node, std::uint64_t address) {
  return node.op == Op::image_address && node.width == 64 && node.immediate == address &&
         node.storage == 0;
}

bool ValidOmission(const Path& basis, const StoreOmission& omission) {
  const auto nodes = basis.nodes();
  if (omission.store >= nodes.size() || omission.overwriter >= nodes.size() ||
      omission.store >= omission.overwriter || nodes[omission.store].op != Op::store ||
      nodes[omission.overwriter].op != Op::store)
    return false;
  const auto first = basis.origins()[omission.store].boundary;
  const auto second = basis.origins()[omission.overwriter].boundary;
  if (second <= first || second - first > kMaxOverwriteBoundaryDistance) return false;
  const auto& first_boundary = basis.boundaries()[first];
  const auto& second_boundary = basis.boundaries()[second];
  for (std::uint32_t boundary = first; boundary < second; ++boundary) {
    if (basis.boundaries()[boundary].transfer) return false;
  }

  if (std::uint64_t(omission.overwriter) + 1 !=
          std::uint64_t(second_boundary.first_node) + second_boundary.node_count ||
      !second_boundary.writes.empty() || second_boundary.transfer)
    return false;
  for (std::uint64_t id = first_boundary.first_node;
       id < std::uint64_t(first_boundary.first_node) + first_boundary.node_count; ++id) {
    if (id != omission.store && MayWriteMemory(nodes[id].op)) return false;
  }

  const auto& store = nodes[omission.store];
  const auto& overwriter = nodes[omission.overwriter];
  if (store.width != overwriter.width || store.access.byte_order != overwriter.access.byte_order ||
      store.access.alignment != overwriter.access.alignment ||
      store.access.decline_on_unaligned != overwriter.access.decline_on_unaligned)
    return false;
  const auto a = ResolveStorageAddress(nodes, store.inputs[0], omission.base_storage);
  const auto b = ResolveStorageAddress(nodes, overwriter.inputs[0], omission.base_storage);
  if (!a || !b || a->root != b->root || a->offset != b->offset ||
      a->offset != omission.offset_bytes)
    return false;
  for (ValueId id = omission.store + 1; id < omission.overwriter; ++id) {
    if (HasMemoryOrMonitorEffect(nodes[id].op)) return false;
  }

  return true;
}

bool Uses(std::span<const Node> nodes, ValueId id) {
  for (const auto& node : nodes) {
    const auto* descriptor = Descriptor(node.op);
    if (!descriptor) return true;
    for (unsigned operand = 0; operand < descriptor->arity; ++operand) {
      if (node.inputs[operand] == id) return true;
    }
  }

  return false;
}

bool DisjointBytes(std::uint64_t first, std::uint64_t first_size, std::uint64_t second,
                   std::uint64_t second_size) {
  const auto delta = second - first;
  return delta >= first_size && std::uint64_t(0 - delta) >= second_size;
}
}  // namespace

bool ValidPairedLoadOmission(const Path& basis, const PairedLoadOmission& omission) {
  const auto nodes = basis.nodes();
  const auto boundaries = basis.boundaries();
  if (basis.origins().size() != nodes.size() || omission.stores[0] >= nodes.size() ||
      omission.stores[1] >= nodes.size() || omission.loads[0] >= nodes.size() ||
      omission.loads[1] >= nodes.size() ||
      std::uint64_t(omission.stores[0]) + 1 != omission.stores[1] ||
      std::uint64_t(omission.loads[0]) + 1 != omission.loads[1] ||
      omission.stores[1] >= omission.loads[0])
    return false;
  const auto store_boundary = basis.origins()[omission.stores[0]].boundary;
  const auto load_boundary = basis.origins()[omission.loads[0]].boundary;
  if (store_boundary >= boundaries.size() || load_boundary >= boundaries.size() ||
      basis.origins()[omission.stores[1]].boundary != store_boundary ||
      basis.origins()[omission.loads[1]].boundary != load_boundary ||
      store_boundary >= load_boundary ||
      std::uint64_t(omission.loads[1]) + 1 != std::uint64_t(boundaries[load_boundary].first_node) +
                                                  boundaries[load_boundary].node_count)
    return false;
  for (unsigned i = 0; i < 2; ++i) {
    const auto& store = nodes[omission.stores[i]];
    const auto& load = nodes[omission.loads[i]];
    if (store.op != Op::store || load.op != Op::load || store.width != 64 || load.width != 64 ||
        store.access.byte_order != load.access.byte_order || store.inputs[0] >= nodes.size() ||
        load.inputs[0] >= omission.loads[0] || Uses(nodes, omission.loads[i]))
      return false;
    const auto a = ResolveStorageAddress(nodes, store.inputs[0], omission.base_storage);
    const auto b = ResolveStorageAddress(nodes, load.inputs[0], omission.base_storage);
    if (!a || !b || a->root != b->root || a->offset != b->offset ||
        a->offset != omission.offset_bytes[i])
      return false;
  }

  const auto first =
      ResolveStorageAddress(nodes, nodes[omission.loads[0]].inputs[0], omission.base_storage);
  const auto second =
      ResolveStorageAddress(nodes, nodes[omission.loads[1]].inputs[0], omission.base_storage);
  if (!first || !second || first->root != second->root || second->offset - first->offset != 8)
    return false;
  for (ValueId id = omission.stores[1] + 1; id < omission.loads[0]; ++id) {
    // An ordinary load still executes, so its fault remains and it cannot
    // change the bytes forwarded to the later pair.
    if (nodes[id].op == Op::load) continue;
    if (HasMemoryOrMonitorEffect(nodes[id].op) && nodes[id].op != Op::store) return false;
    if (nodes[id].op != Op::store) continue;
    const auto& store = nodes[id];
    if (store.width == 0 || store.width % 8 || store.inputs[0] >= nodes.size()) return false;
    const auto address = ResolveStorageAddress(nodes, store.inputs[0], omission.base_storage);
    if (!address || address->root != first->root ||
        !DisjointBytes(first->offset, 8, address->offset, store.width / 8) ||
        !DisjointBytes(second->offset, 8, address->offset, store.width / 8))
      return false;
  }

  for (std::size_t boundary = store_boundary; boundary < load_boundary; ++boundary) {
    const auto& transfer = boundaries[boundary].transfer;
    if (transfer && transfer->kind == TransferKind::call) return false;
  }

  for (const auto& boundary : boundaries) {
    for (const auto& write : boundary.writes) {
      if (write.value == omission.loads[0] || write.value == omission.loads[1]) return false;
    }

    if (!boundary.transfer) continue;
    const auto& transfer = *boundary.transfer;
    for (const auto id :
         {transfer.target, transfer.condition.value_or(UINT32_MAX),
          transfer.alternative.value_or(UINT32_MAX), transfer.continuation.value_or(UINT32_MAX)}) {
      if (id == omission.loads[0] || id == omission.loads[1]) return false;
    }
  }

  return true;
}

std::optional<Transfer> RecoveredPath::effective_transfer(std::size_t boundary) const {
  if (boundary >= basis_.boundaries().size()) return {};
  const auto rewrite = std::lower_bound(
      rewrites_.begin(), rewrites_.end(), boundary,
      [](const ConditionalRewrite& value, std::size_t index) { return value.boundary < index; });
  if (rewrite != rewrites_.end() && rewrite->boundary == boundary) return rewrite->replacement;
  return basis_.boundaries()[boundary].transfer;
}

bool RecoveredPath::omits_store(ValueId id) const {
  const auto found = std::lower_bound(
      omissions_.begin(), omissions_.end(), id,
      [](const StoreOmission& value, ValueId target) { return value.store < target; });
  return found != omissions_.end() && found->store == id;
}

const PairedLoadOmission* RecoveredPath::first_omitted_load(ValueId id) const {
  const auto found = std::lower_bound(
      paired_load_omissions_.begin(), paired_load_omissions_.end(), id,
      [](const PairedLoadOmission& value, ValueId target) { return value.loads[0] < target; });
  return found != paired_load_omissions_.end() && found->loads[0] == id ? &*found : nullptr;
}

bool RecoveredPath::omits_second_load(ValueId id) const {
  const auto found = std::lower_bound(
      paired_load_omissions_.begin(), paired_load_omissions_.end(), id,
      [](const PairedLoadOmission& value, ValueId target) { return value.loads[1] < target; });
  return found != paired_load_omissions_.end() && found->loads[1] == id;
}

BlockDecline ValidateRecoveredPath(const RecoveredPath& path, Budget& budget, BlockLimits limits,
                                   ImageFacts facts) {
  const auto& basis = path.basis();
  const auto valid = ValidatePath(basis, budget, limits);
  if (valid != BlockDecline::none) return valid;
  if (path.rewrites().size() > basis.boundaries().size()) return BlockDecline::invalid_ir;
  if (path.omissions().size() > basis.nodes().size()) return BlockDecline::invalid_ir;
  if (path.paired_load_omissions().size() > basis.boundaries().size())
    return BlockDecline::invalid_ir;
  if (budget.try_consume({path.rewrites().size() * 16 +
                              path.omissions().size() * (2 * basis.nodes().size() + 32) +
                              path.paired_load_omissions().size() *
                                  (4 * basis.nodes().size() + 2 * basis.boundaries().size() + 32) +
                              1,
                          0}) != BudgetDecline::none)
    return BlockDecline::resource_limit;
  if (path.rewrites().empty() && path.omissions().empty() && path.paired_load_omissions().empty())
    return path.destinations().empty() && path.revision() == basis.revision()
               ? BlockDecline::none
               : BlockDecline::invalid_ir;
  const std::uint64_t stages =
      !path.rewrites().empty() + !path.paired_load_omissions().empty() + !path.omissions().empty();
  if (basis.revision() > UINT64_MAX - stages || path.revision() != basis.revision() + stages)
    return BlockDecline::invalid_ir;
  const auto control_revision = basis.revision() + !path.rewrites().empty();
  const auto load_revision = control_revision + !path.paired_load_omissions().empty();
  std::optional<ValueId> previous_load;
  for (const auto& omission : path.paired_load_omissions()) {
    if ((previous_load && *previous_load >= omission.loads[0]) ||
        omission.from_revision != control_revision || omission.to_revision != load_revision ||
        !ValidPairedLoadOmission(basis, omission))
      return BlockDecline::invalid_ir;
    previous_load = omission.loads[1];
  }

  std::optional<ValueId> previous_store;
  for (const auto& omission : path.omissions()) {
    if ((previous_store && *previous_store >= omission.store) ||
        omission.from_revision != load_revision || omission.to_revision != path.revision() ||
        !ValidOmission(basis, omission))
      return BlockDecline::invalid_ir;
    previous_store = omission.store;
  }

  for (const auto& omission : path.omissions()) {
    if (path.omits_store(omission.overwriter)) return BlockDecline::invalid_ir;
  }

  for (const auto& omission : path.paired_load_omissions()) {
    if (path.omits_store(omission.stores[0]) || path.omits_store(omission.stores[1]))
      return BlockDecline::invalid_ir;
    for (const auto& rewrite : path.rewrites()) {
      const auto names_load = [&](ValueId id) {
        return id == omission.loads[0] || id == omission.loads[1];
      };

      if (names_load(rewrite.condition) || names_load(rewrite.replacement.target) ||
          (rewrite.replacement.condition && names_load(*rewrite.replacement.condition)) ||
          (rewrite.replacement.alternative && names_load(*rewrite.replacement.alternative)) ||
          (rewrite.replacement.continuation && names_load(*rewrite.replacement.continuation)))
        return BlockDecline::invalid_ir;
    }
  }

  // Two destinations per dispatch branch at most, and their ids have to stay
  // addressable above the basis.
  if (path.destinations().size() > 2 * path.rewrites().size() ||
      basis.nodes().size() > UINT32_MAX - path.destinations().size())
    return BlockDecline::invalid_ir;
  std::optional<std::uint32_t> previous;
  std::size_t named = 0;
  for (const auto& rewrite : path.rewrites()) {
    if ((previous && *previous >= rewrite.boundary) ||
        rewrite.boundary >= basis.boundaries().size() ||
        rewrite.from_revision != basis.revision() || rewrite.to_revision != control_revision)
      return BlockDecline::invalid_ir;
    previous = rewrite.boundary;
    const auto& original = basis.boundaries()[rewrite.boundary].transfer;
    if (!original || !SameTransfer(rewrite.original, *original) ||
        rewrite.condition >= basis.nodes().size())
      return BlockDecline::invalid_ir;
    if (rewrite.rule == RewriteRule::folded_condition) {
      if (!rewrite.witness.empty() || rewrite.when_true != 0 || rewrite.when_false != 0 ||
          original->kind != TransferKind::conditional || rewrite.condition != original->condition)
        return BlockDecline::invalid_ir;
      const auto& condition = basis.nodes()[rewrite.condition];
      if (condition.op != Op::constant || condition.width != 1 ||
          rewrite.condition_value != ((condition.immediate & 1) != 0))
        return BlockDecline::invalid_ir;
      const auto selected = rewrite.condition_value ? original->target : *original->alternative;
      const Transfer expected{TransferKind::jump, selected, {}, {}, {}};
      if (!SameTransfer(rewrite.replacement, expected)) return BlockDecline::invalid_ir;
      continue;
    }

    // Only recovery decides a restored conditional or retires a loop or a
    // call, long after a path is built, so a path carrying any is malformed.
    if (rewrite.rule == RewriteRule::decided_dispatch ||
        rewrite.rule == RewriteRule::bounded_exit || rewrite.rule == RewriteRule::retired_call ||
        rewrite.rule == RewriteRule::resolved_call)
      return BlockDecline::invalid_ir;
    // A dispatch branch is re-resolved here rather than read off the record:
    // nothing about the replacement follows structurally from the original, so
    // the destinations and the reads behind them are derived again.
    if (rewrite.condition_value || original->kind != TransferKind::jump ||
        named + 2 > path.destinations().size())
      return BlockDecline::invalid_ir;
    const auto resolved = ResolveDispatchBranch(basis.nodes(), original->target, facts, budget);
    if (resolved.reason == DispatchDecline::resource_limit) return BlockDecline::resource_limit;
    const auto& branch = resolved.branch;
    if (!branch || branch->condition != rewrite.condition ||
        branch->when_true != rewrite.when_true || branch->when_false != rewrite.when_false ||
        branch->witness.size() != rewrite.witness.size())
      return BlockDecline::invalid_ir;
    if (budget.try_consume({branch->witness.size() + 1, 0}) != BudgetDecline::none)
      return BlockDecline::resource_limit;
    if (!std::equal(branch->witness.begin(), branch->witness.end(), rewrite.witness.begin(),
                    SameRead))
      return BlockDecline::invalid_ir;
    const auto first = static_cast<ValueId>(path.first_destination() + named);
    const Transfer expected{TransferKind::conditional, first, rewrite.condition, first + 1, {}};
    if (!SameTransfer(rewrite.replacement, expected) ||
        !Destination(path.destinations()[named], rewrite.when_true) ||
        !Destination(path.destinations()[named + 1], rewrite.when_false))
      return BlockDecline::invalid_ir;
    named += 2;
  }

  return named == path.destinations().size() ? BlockDecline::none : BlockDecline::invalid_ir;
}

}  // namespace nyx::ir
