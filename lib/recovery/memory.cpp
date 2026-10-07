#include "nyx/recovery/memory.hpp"

#include <algorithm>

namespace nyx::recovery {
namespace {
using ir::Op;

bool SameBase(const AffineAddress& a, const AffineAddress& b) {
  return a.base == b.base && (a.base != AddressBase::ssa_value || a.value == b.value);
}

bool SameAddress(const AffineAddress& a, const AffineAddress& b) {
  return SameBase(a, b) && a.offset == b.offset;
}

bool MayOverlap(const AffineAddress& a, unsigned a_size, const AffineAddress& b, unsigned b_size) {
  if (!SameBase(a, b)) return true;
  return b.offset - a.offset < a_size || a.offset - b.offset < b_size;
}

// An access's own address, and the same address placed through the supplied
// entry relations. The two differ only past a read of related storage, and the
// second is consulted only where the first cannot decide.
struct Placed {
  AffineAddress address;
  AffineAddress entry;
};

AffineAddress Address(ir::ValueId id, std::span<const ir::Node> nodes,
                      std::span<const Placed> placed, AffineAddress Placed::* field) {
  const auto& node = nodes[id];
  if (node.width != 64) return {AddressBase::ssa_value, id, 0};
  if (node.op == Op::constant) return {AddressBase::absolute, 0, node.immediate};
  if (node.op == Op::image_address) return {AddressBase::load_bias, 0, node.immediate};
  if (node.op == Op::add || node.op == Op::sub) {
    const auto a = placed[node.inputs[0]].*field, b = placed[node.inputs[1]].*field;
    if (b.base == AddressBase::absolute) {
      return {a.base, a.value, node.op == Op::add ? a.offset + b.offset : a.offset - b.offset};
    }

    if (node.op == Op::add && a.base == AddressBase::absolute) {
      return {b.base, b.value, a.offset + b.offset};
    }
  }

  return {AddressBase::ssa_value, id, 0};
}

// Whether two accesses may overlap, and whether ruling it out took a relation.
struct Overlap {
  bool may;
  bool related;
};

Overlap Compare(const Placed& a, unsigned a_size, const Placed& b, unsigned b_size) {
  if (SameBase(a.address, b.address))
    return {MayOverlap(a.address, a_size, b.address, b_size), false};
  return {MayOverlap(a.entry, a_size, b.entry, b_size), true};
}

bool ValidRelations(std::span<const ir::StorageRelation> relations) {
  const auto related = [&](ir::StorageId storage) {
    const auto found = std::lower_bound(relations.begin(), relations.end(), storage,
                                        [](const ir::StorageRelation& relation, ir::StorageId key) {
                                          return relation.storage < key;
                                        });
    return found != relations.end() && found->storage == storage;
  };

  for (std::size_t i = 1; i < relations.size(); ++i) {
    if (relations[i - 1].storage >= relations[i].storage) return false;
  }

  return std::none_of(relations.begin(), relations.end(), [&](const ir::StorageRelation& relation) {
    return relation.root >= relation.storage || related(relation.root);
  });
}

struct Store {
  ir::ValueId operation;
  ir::ValueId value;
  Placed address;
  unsigned width;
  ir::ByteOrder byte_order;
  bool entry_relation;
};

struct Class {
  ir::StorageId root;
  ir::ValueId base;
  std::uint64_t offset;
};

struct Replacement {
  ir::ValueId value;
  std::uint32_t fact;
};

template <class Result>
Result Decline(MemoryRecoveryDecline reason) {
  return {{}, {}, {}, reason};
}

template <class Region, class Result>
Result Forward(const Region& input, Budget& budget, MemoryRecoveryLimits limits,
               ir::BlockDecline (*validate)(const Region&, Budget&, ir::BlockLimits),
               MemoryProofScope scope, std::span<const ir::StorageRelation> entry) {
  const auto valid = validate(input, budget, limits.block);
  if (valid != ir::BlockDecline::none) {
    return Decline<Result>(valid == ir::BlockDecline::resource_limit
                               ? MemoryRecoveryDecline::resource_limit
                           : valid == ir::BlockDecline::unsupported_control
                               ? MemoryRecoveryDecline::unsupported_control
                               : MemoryRecoveryDecline::invalid_ir);
  }

  if (!ValidRelations(entry)) return Decline<Result>(MemoryRecoveryDecline::invalid_ir);
  if (input.revision() == UINT64_MAX)
    return Decline<Result>(MemoryRecoveryDecline::revision_overflow);
  const auto count = input.nodes().size();
  const auto store_capacity = std::min<std::uint64_t>(limits.max_stores, count);
  auto edit_capacity =
      std::min<std::uint64_t>(limits.max_edits, 3ULL * count + 4ULL * input.boundaries().size());
  for (const auto& boundary : input.boundaries()) {
    edit_capacity +=
        std::min<std::uint64_t>(limits.max_edits - edit_capacity, boundary.writes.size());
  }

  const auto charge = [&](std::uint64_t amount, std::uint64_t size) {
    return amount <= UINT64_MAX / size &&
           budget.try_consume({amount, amount * size}) == BudgetDecline::none;
  };

  if (!charge(input.sources().size(), sizeof(ir::Group)) || !charge(count, sizeof(ir::Node)) ||
      !charge(count, sizeof(ir::Origin)) ||
      !charge(input.boundaries().size(), sizeof(ir::Boundary)) || !charge(count, sizeof(Placed)) ||
      !charge(entry.size(), sizeof(Class)) || !charge(count, sizeof(std::optional<Replacement>)) ||
      !charge(count, sizeof(ForwardingFact)) || !charge(store_capacity, sizeof(Store)) ||
      !charge(edit_capacity, sizeof(MemoryEdit)))
    return Decline<Result>(MemoryRecoveryDecline::resource_limit);
  for (const auto& source : input.sources()) {
    if (!charge(source.bytes().size(), 1) || !charge(source.nodes().size(), sizeof(ir::Node)) ||
        !charge(source.writes().size(), sizeof(ir::Write)))
      return Decline<Result>(MemoryRecoveryDecline::resource_limit);
  }

  for (const auto& boundary : input.boundaries()) {
    if (!charge(boundary.writes.size(), sizeof(ir::Write)))
      return Decline<Result>(MemoryRecoveryDecline::resource_limit);
  }

  std::vector<ir::Group> sources(input.sources().begin(), input.sources().end());
  std::vector<ir::Node> nodes(input.nodes().begin(), input.nodes().end());
  std::vector<ir::Origin> origins(input.origins().begin(), input.origins().end());
  std::vector<ir::Boundary> boundaries(input.boundaries().begin(), input.boundaries().end());
  std::vector<Placed> addresses;
  addresses.reserve(count);

  // The first read of each related class is the base every other read of that
  // class is placed against. A normalized path reads each storage once, at entry.
  std::vector<Class> classes;
  classes.reserve(entry.size());
  const auto place_read = [&](ir::ValueId id) -> std::optional<AffineAddress> {
    const auto& node = nodes[id];
    if (node.op != Op::read || node.width != 64) return {};
    auto root = node.storage;
    std::uint64_t offset = 0;
    bool member = false;
    for (const auto& relation : entry) {
      if (relation.storage == node.storage) {
        root = relation.root;
        offset = relation.offset;
      }

      member |= relation.storage == node.storage || relation.root == node.storage;
    }

    if (!member) return {};
    for (const auto& known : classes) {
      if (known.root == root) {
        return AffineAddress{AddressBase::ssa_value, known.base, offset - known.offset};
      }
    }

    classes.push_back({root, id, offset});
    return {};
  };

  std::vector<std::optional<Replacement>> replacements(count);
  std::vector<Store> stores;
  stores.reserve(store_capacity);
  std::vector<ForwardingFact> facts;
  facts.reserve(count);
  std::vector<MemoryEdit> journal;
  journal.reserve(edit_capacity);
  const auto redirect = [&](ir::ValueId& value, MemoryUse use, std::uint32_t owner,
                            std::uint32_t slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return false;
    if (!replacements[value]) return true;
    if (journal.size() == edit_capacity) return false;
    const auto replacement = *replacements[value];
    journal.push_back({replacement.fact, use, owner, slot, value, replacement.value,
                       input.revision(), input.revision() + 1});
    value = replacement.value;
    return true;
  };

  for (std::size_t boundary_id = 0; boundary_id < boundaries.size(); ++boundary_id) {
    auto& boundary = boundaries[boundary_id];
    const auto end = std::uint64_t(boundary.first_node) + boundary.node_count;
    for (std::uint64_t index = boundary.first_node; index < end; ++index) {
      const auto id = static_cast<ir::ValueId>(index);
      auto& node = nodes[id];
      if (budget.try_consume({8, 0}) != BudgetDecline::none)
        return Decline<Result>(MemoryRecoveryDecline::resource_limit);
      const auto arity = ir::Descriptor(node.op)->arity;
      for (unsigned operand = 0; operand < arity; ++operand) {
        if (!redirect(node.inputs[operand], MemoryUse::operand, id, operand))
          return Decline<Result>(MemoryRecoveryDecline::resource_limit);
      }

      if (budget.try_consume({entry.size() + 1, 0}) != BudgetDecline::none)
        return Decline<Result>(MemoryRecoveryDecline::resource_limit);
      const auto own = Address(id, nodes, addresses, &Placed::address);
      addresses.push_back(
          {own, place_read(id).value_or(Address(id, nodes, addresses, &Placed::entry))});
      if (node.op == Op::exclusive_store) {
        stores.clear();
        continue;
      }

      if (node.op != Op::load && node.op != Op::store) continue;
      const auto& address = addresses[node.inputs[0]];
      if (budget.try_consume({stores.size() + 1, 0}) != BudgetDecline::none)
        return Decline<Result>(MemoryRecoveryDecline::resource_limit);
      if (node.op == Op::store) {
        std::erase_if(stores, [&](Store& store) {
          const auto overlap = Compare(address, node.width / 8, store.address, store.width / 8);
          store.entry_relation |= overlap.related;
          return overlap.may;
        });
        if (stores.size() == store_capacity)
          return Decline<Result>(MemoryRecoveryDecline::resource_limit);
        stores.push_back({id, node.inputs[1], address, node.width, node.access.byte_order, false});
      } else {
        for (const auto& store : stores) {
          if (store.width != node.width || store.byte_order != node.access.byte_order) continue;
          const bool own = SameAddress(store.address.address, address.address);
          if (!own && !SameAddress(store.address.entry, address.entry)) continue;
          replacements[id] = Replacement{store.value, static_cast<std::uint32_t>(facts.size())};
          facts.push_back({store.operation, id, store.value, address.address, node.width,
                           node.access.byte_order, input.revision(), input.revision() + 1, scope,
                           store.entry_relation || !own});
          break;
        }
      }
    }

    for (std::size_t i = 0; i < boundary.writes.size(); ++i) {
      if (!redirect(boundary.writes[i].value, MemoryUse::final_write, boundary_id, i))
        return Decline<Result>(MemoryRecoveryDecline::resource_limit);
    }

    if (boundary.transfer) {
      auto& transfer = *boundary.transfer;
      if (!redirect(transfer.target, MemoryUse::target, boundary_id, 0))
        return Decline<Result>(MemoryRecoveryDecline::resource_limit);
      for (const auto& [field, use] :
           {std::pair{&transfer.condition, MemoryUse::condition},
            std::pair{&transfer.alternative, MemoryUse::alternative},
            std::pair{&transfer.continuation, MemoryUse::continuation}}) {
        if (*field && !redirect(**field, use, boundary_id, 0))
          return Decline<Result>(MemoryRecoveryDecline::resource_limit);
      }
    }
  }

  const auto revision = input.revision() + (journal.empty() ? 0 : 1);
  for (auto& fact : facts) fact.to_revision = revision;
  Region output(std::move(sources), std::move(nodes), std::move(origins), std::move(boundaries),
                revision);
  const auto checked = validate(output, budget, limits.block);
  if (checked != ir::BlockDecline::none) {
    return Decline<Result>(checked == ir::BlockDecline::resource_limit
                               ? MemoryRecoveryDecline::resource_limit
                               : MemoryRecoveryDecline::invalid_ir);
  }

  return {std::move(output), std::move(facts), std::move(journal), MemoryRecoveryDecline::none};
}

}  // namespace

MemoryRecoveryResult ForwardMemoryValues(const ir::Block& input, Budget& budget,
                                         MemoryRecoveryLimits limits) {
  return Forward<ir::Block, MemoryRecoveryResult>(input, budget, limits, ir::Validate,
                                                  MemoryProofScope::straightline_entry, {});
}

PathMemoryRecoveryResult ForwardMemoryValues(const ir::Path& input, Budget& budget,
                                             MemoryRecoveryLimits limits,
                                             std::span<const ir::StorageRelation> entry) {
  return Forward<ir::Path, PathMemoryRecoveryResult>(input, budget, limits, ir::ValidatePath,
                                                     MemoryProofScope::successful_itinerary_prefix,
                                                     entry);
}

}  // namespace nyx::recovery
