#include "nyx/recovery/constant_load.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <variant>

namespace nyx::recovery {
namespace {
std::optional<bool> Omitted(const ir::SsaBlock& block, ir::ValueId id, Budget& budget) {
  for (const auto& omission : block.paired_load_omissions) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return {};
    if (omission.loads[0] == id || omission.loads[1] == id) return true;
  }

  return false;
}

bool ChargeInsert(std::size_t size, std::size_t element, Budget& budget) {
  const auto search = static_cast<std::uint64_t>(std::bit_width(size)) + 1;
  if (size == SIZE_MAX || size > UINT64_MAX - search || size + 1 > UINT64_MAX / element)
    return false;
  return budget.try_consume({size + search, (size + 1) * element}) == BudgetDecline::none;
}

// The folds `keep` selects written into a copy of `original`, not yet checked.
template <class Item>
std::variant<ir::SsaGraph, ConstantLoadRefusal> WriteFolds(const ir::SsaGraph& original,
                                                           const std::vector<Item>& items,
                                                           const std::vector<std::uint8_t>& keep,
                                                           Budget& budget) {
  auto candidate = original.Clone(budget);
  if (!candidate) return ConstantLoadRefusal::resource_limit;
  for (std::size_t index = 0; index < items.size(); ++index) {
    if (!keep[index]) continue;
    const auto& item = items[index];
    const ir::SsaHandle handle{candidate->arena(), item.block.slot, item.block.generation};
    if (!candidate->Get(handle)) return ConstantLoadRefusal::invalid_graph;
    auto copied = candidate->CopyBlock(handle, budget);
    if (!copied) return ConstantLoadRefusal::resource_limit;
    auto block = std::move(*copied);
    if (!ChargeInsert(block.disabled_effects.size(), sizeof(ir::ValueId), budget) ||
        !ChargeInsert(block.constant_loads.size(), sizeof(ir::SsaConstantLoad), budget))
      return ConstantLoadRefusal::resource_limit;
    block.disabled_effects.insert(std::lower_bound(block.disabled_effects.begin(),
                                                   block.disabled_effects.end(), item.fold.node),
                                  item.fold.node);
    block.constant_loads.insert(
        std::lower_bound(
            block.constant_loads.begin(), block.constant_loads.end(), item.fold.node,
            [](const ir::SsaConstantLoad& fold, ir::ValueId id) { return fold.node < id; }),
        item.fold);
    if (!candidate->Replace(handle, std::move(block))) return ConstantLoadRefusal::invalid_graph;
  }

  return std::move(*candidate);
}

// Whether the checker, reading `candidate` as it stands, places a store on
// what `fold` reads.
std::optional<bool> Contradicted(const ir::SsaGraph& candidate, const ir::SsaBlock& block,
                                 const ir::SsaConstantLoad& fold,
                                 std::array<std::optional<ir::SsaImageStores>, 2>& stores,
                                 Budget& budget) {
  auto& placed = stores[fold.page_aligned_placement];
  if (!placed &&
      !(placed = ir::SsaImageStores::Collect(candidate, fold.page_aligned_placement, budget)))
    return std::nullopt;
  if (fold.node >= block.nodes.size()) return true;
  const auto size = block.nodes[fold.node].width / 8;
  if (fold.kind == ir::SsaConstantKind::bounded_table) {
    if (budget.try_consume({fold.table_bytes.size(), 0}) != BudgetDecline::none)
      return std::nullopt;
    for (std::uint64_t row = 0; row < fold.table_bytes.size(); ++row)
      if (placed->Conflicts(fold.source_address + row * fold.table_stride, size, fold.read_only))
        return true;
    return false;
  }

  return placed->Conflicts(fold.source_address, size, fold.read_only) ||
         (fold.condition &&
          placed->Conflicts(fold.alternative_source_address, size, fold.read_only));
}

// Why the checker rejects `candidate`, which adds only `item` to `original`:
// the new fold's own recheck, or one the graph already carried.
template <class Item>
std::optional<ConstantLoadRefusal> WhyRefused(const ir::SsaGraph& original,
                                              const ir::SsaGraph& candidate, const Item& item,
                                              Budget& budget) {
  std::array<std::optional<ir::SsaImageStores>, 2> stores;
  const ir::SsaHandle handle{candidate.arena(), item.block.slot, item.block.generation};
  const auto* changed = candidate.Get(handle);
  if (!changed) return ConstantLoadRefusal::invalid_graph;
  const auto own = Contradicted(candidate, *changed, item.fold, stores, budget);
  if (!own) return std::nullopt;
  if (*own) return ConstantLoadRefusal::conflicting_store;
  for (std::size_t slot = 0; slot < original.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return std::nullopt;
    const auto before = original.Handle(slot);
    if (!before) continue;
    const auto* block = candidate.Get({candidate.arena(), before->slot, before->generation});
    if (!block) continue;
    for (const auto& fold : original.Get(*before)->constant_loads) {
      if (!fold.skip_access) continue;
      const auto old = Contradicted(candidate, *block, fold, stores, budget);
      if (!old) return std::nullopt;
      if (*old) return ConstantLoadRefusal::contradicts_existing_fold;
    }
  }

  return ConstantLoadRefusal::invalid_graph;
}

// The candidate carrying as many of `items` as the checker accepts. Each was
// proved against the run's facts, which see every store the checker sees and
// more, but a fold also records a value the checker then reads: a pointer
// slot's target places a store the checker could not place before, on a fold
// the graph already carries or on another new one. Together first; failing
// that, each alone, dropping any the checker rejects; then the rest together,
// or failing that one at a time. Every attempt is a full check, charged.
template <class Item>
std::variant<std::pair<ir::SsaGraph, std::vector<std::uint8_t>>, ConstantLoadRefusal> Admit(
    const ir::SsaGraph& original, const std::vector<Item>& items,
    std::vector<RefusedConstantLoad>& refused, Budget& budget) {
  using Admitted =
      std::variant<std::pair<ir::SsaGraph, std::vector<std::uint8_t>>, ConstantLoadRefusal>;
  // A checked candidate, or nothing when the checker rejects it.
  const auto attempt = [&](const std::vector<std::uint8_t>& keep)
      -> std::variant<std::optional<ir::SsaGraph>, ConstantLoadRefusal> {
    auto written = WriteFolds(original, items, keep, budget);
    if (auto* reason = std::get_if<ConstantLoadRefusal>(&written)) return *reason;
    auto& candidate = std::get<ir::SsaGraph>(written);
    const auto checked = ir::ValidateSsa(candidate, budget);
    if (checked == ir::SsaDecline::resource_limit) return ConstantLoadRefusal::resource_limit;
    if (checked != ir::SsaDecline::none) return std::optional<ir::SsaGraph>{};
    return std::optional(std::move(candidate));
  };

  const auto refuse = [&](std::size_t index, ConstantLoadRefusal reason) {
    if (budget.try_consume({1, sizeof(RefusedConstantLoad)}) != BudgetDecline::none) return false;
    refused.push_back({items[index].block, items[index].fold.node, reason});
    return true;
  };

  std::vector<std::uint8_t> keep(items.size(), 1);
  auto together = attempt(keep);
  if (auto* reason = std::get_if<ConstantLoadRefusal>(&together)) return *reason;
  if (auto& graph = std::get<std::optional<ir::SsaGraph>>(together))
    return Admitted(std::pair{std::move(*graph), std::move(keep)});
  std::vector<std::uint8_t> alone(items.size(), 0);
  for (std::size_t index = 0; index < items.size(); ++index) {
    alone[index] = 1;
    auto written = WriteFolds(original, items, alone, budget);
    alone[index] = 0;
    if (auto* reason = std::get_if<ConstantLoadRefusal>(&written)) return *reason;
    const auto& candidate = std::get<ir::SsaGraph>(written);
    const auto checked = ir::ValidateSsa(candidate, budget);
    if (checked == ir::SsaDecline::resource_limit) return ConstantLoadRefusal::resource_limit;
    if (checked == ir::SsaDecline::none) continue;
    const auto why = WhyRefused(original, candidate, items[index], budget);
    if (!why || !refuse(index, *why)) return ConstantLoadRefusal::resource_limit;
    keep[index] = 0;
  }

  if (std::none_of(keep.begin(), keep.end(), [](std::uint8_t kept) { return kept; }))
    return ConstantLoadRefusal::none;
  auto rest = attempt(keep);
  if (auto* reason = std::get_if<ConstantLoadRefusal>(&rest)) return *reason;
  if (auto& graph = std::get<std::optional<ir::SsaGraph>>(rest))
    return Admitted(std::pair{std::move(*graph), std::move(keep)});
  std::vector<std::uint8_t> growing(items.size(), 0);
  std::optional<ir::SsaGraph> last;
  for (std::size_t index = 0; index < items.size(); ++index) {
    if (!keep[index]) continue;
    growing[index] = 1;
    auto grown = attempt(growing);
    if (auto* reason = std::get_if<ConstantLoadRefusal>(&grown)) return *reason;
    if (auto& graph = std::get<std::optional<ir::SsaGraph>>(grown)) {
      last = std::move(*graph);
      continue;
    }

    growing[index] = 0;
    if (!refuse(index, ConstantLoadRefusal::conflicting_store))
      return ConstantLoadRefusal::resource_limit;
  }

  if (!last) return ConstantLoadRefusal::none;
  return Admitted(std::pair{std::move(*last), std::move(growing)});
}
}  // namespace

ConstantLoadResult ProposeConstantImageLoads(const ir::SsaGraph& original, ir::ImageFacts facts,
                                             ir::ImageAccessContract access, Budget& budget) {
  ConstantLoadResult result;
  const auto decline = [](ConstantLoadRefusal reason) {
    ConstantLoadResult refused;
    refused.reason = reason;
    return refused;
  };

  const auto valid = ir::ValidateSsa(original, budget);
  if (valid != ir::SsaDecline::none) {
    return decline(valid == ir::SsaDecline::resource_limit ? ConstantLoadRefusal::resource_limit
                                                           : ConstantLoadRefusal::invalid_graph);
  }

  struct Proposed {
    ir::SsaHandle block;
    ir::SsaConstantLoad fold;
    ConstantLoadFact fact;
    std::uint64_t fact_address;
    std::uint64_t fact_bytes;
    bool placement;
  };

  std::vector<Proposed> proposed;
  std::optional<ir::SsaImageStores> stores;
  for (std::size_t slot = 0; slot < original.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return decline(ConstantLoadRefusal::resource_limit);
    const auto handle = original.Handle(slot);
    if (!handle) continue;
    const auto& block = *original.Get(*handle);
    for (std::uint32_t id = 0; id < block.nodes.size(); ++id) {
      if (budget.try_consume(
              {1 + static_cast<std::uint64_t>(std::bit_width(block.disabled_effects.size())), 0}) !=
          BudgetDecline::none)
        return decline(ConstantLoadRefusal::resource_limit);
      const auto& node = block.nodes[id];
      if (node.op != ir::Op::load && node.op != ir::Op::exclusive_load) continue;

      // Where the load reads: an image location its address evaluates to,
      // possibly through a PC page that only the placement declaration fixes.
      std::optional<std::pair<std::uint64_t, bool>> located;
      if (node.op == ir::Op::load)
        located = ir::SsaLoadLocation(block.nodes, id, original.load_bias());
      else if (node.inputs[0] < block.nodes.size() &&
               block.nodes[node.inputs[0]].op == ir::Op::image_address)
        located = std::pair{block.nodes[node.inputs[0]].immediate, false};
      if (!located || (located->second && !facts.page_aligned_placement)) continue;
      if (std::binary_search(block.disabled_effects.begin(), block.disabled_effects.end(), id))
        continue;
      if (budget.try_consume({1, sizeof(RefusedConstantLoad)}) != BudgetDecline::none)
        return decline(ConstantLoadRefusal::resource_limit);
      const auto location = located->first;
      const auto refuse = [&](ConstantLoadRefusal reason) {
        result.refused.push_back({*handle, id, reason});
      };

      if (node.op != ir::Op::load || !node.width || node.width > 64 || node.width % 8) {
        refuse(ConstantLoadRefusal::unsupported_access);
        continue;
      }

      const auto omitted = Omitted(block, id, budget);
      if (!omitted) return decline(ConstantLoadRefusal::resource_limit);
      if (*omitted) {
        refuse(ConstantLoadRefusal::existing_omission);
        continue;
      }

      if (budget.try_consume({facts.constants.size(), 0}) != BudgetDecline::none ||
          budget.try_consume({facts.constants.size(), 0}) != BudgetDecline::none ||
          budget.try_consume({facts.pointers.size(), 0}) != BudgetDecline::none ||
          budget.try_consume({16, 0}) != BudgetDecline::none)
        return decline(ConstantLoadRefusal::resource_limit);
      Proposed proposal{*handle,
                        {id, ir::SsaConstantKind::literal, 0, location},
                        ConstantLoadFact::relocated_slot,
                        location,
                        8,
                        false};
      proposal.fold.access = access;
      proposal.fold.page_aligned_placement = facts.page_aligned_placement;
      proposal.fold.value_stable = true;
      proposal.fold.skip_access = true;
      const auto image = ir::ReadRelocated(facts, location, node.width);
      if (node.access.byte_order == ir::ByteOrder::little && image) {
        proposal.fold.kind = ir::SsaConstantKind::image_location;
        proposal.fold.value = *image;
        proposal.fold.declared_target = *image;
      } else if (const auto literal =
                     ir::ReadConstant(facts, location, node.width, node.access.byte_order)) {
        proposal.fold.value = *literal;
        const auto range = std::find_if(
            facts.constants.begin(), facts.constants.end(),
            [&](const ir::ConstantImageRange& item) {
              return location >= item.address && location - item.address < item.bytes.size() &&
                     node.width / 8 <= item.bytes.size() - (location - item.address);
            });
        if (range == facts.constants.end()) {
          refuse(ConstantLoadRefusal::no_invariant);
          continue;
        }

        proposal.fact = ConstantLoadFact::constant_range;
        proposal.fold.read_only = range->read_only;
        proposal.fact_address = range->address;
        proposal.fact_bytes = range->bytes.size();
        const auto offset = location - range->address;
        std::copy_n(range->bytes.begin() + offset, node.width / 8,
                    proposal.fold.declared_bytes.begin());
      } else {
        refuse(node.access.byte_order != ir::ByteOrder::little && image
                   ? ConstantLoadRefusal::unsupported_access
                   : ConstantLoadRefusal::no_invariant);
        continue;
      }

      if (!access.mapped_readable_lifetime || !access.no_runtime_unmapping ||
          !access.ordinary_reads_unobservable) {
        refuse(ConstantLoadRefusal::not_nonfaulting);
        continue;
      }

      if (node.access.alignment > 1 &&
          (!facts.page_aligned_placement || node.access.alignment > 4096 ||
           location % node.access.alignment)) {
        refuse(ConstantLoadRefusal::alignment);
        continue;
      }

      proposal.placement = node.access.alignment > 1 || located->second;
      if (!stores && !(stores = ir::SsaImageStores::Collect(original, facts.page_aligned_placement,
                                                            budget, &facts)))
        return decline(ConstantLoadRefusal::resource_limit);
      if (stores->Conflicts(location, node.width / 8, proposal.fold.read_only)) {
        refuse(ConstantLoadRefusal::conflicting_store);
        continue;
      }

      if (budget.try_consume({1, sizeof(Proposed) + sizeof(ConstantLoadEdit)}) !=
          BudgetDecline::none) {
        return decline(ConstantLoadRefusal::resource_limit);
      }

      proposed.push_back(proposal);
    }
  }

  if (proposed.empty()) return result;
  auto admitted = Admit(original, proposed, result.refused, budget);
  if (const auto* reason = std::get_if<ConstantLoadRefusal>(&admitted)) {
    if (*reason == ConstantLoadRefusal::none) return result;
    return decline(*reason);
  }

  auto& [candidate, kept] = std::get<0>(admitted);
  if (budget.try_consume({proposed.size(), proposed.size() * sizeof(ConstantLoadEdit)}) !=
      BudgetDecline::none)
    return decline(ConstantLoadRefusal::resource_limit);
  for (std::size_t index = 0; index < proposed.size(); ++index) {
    if (!kept[index]) continue;
    const auto& proposal = proposed[index];
    result.journal.push_back({proposal.block,
                              {candidate.arena(), proposal.block.slot, proposal.block.generation},
                              proposal.fold,
                              proposal.fact,
                              proposal.fact_address,
                              proposal.fact_bytes,
                              proposal.placement,
                              original.revision(),
                              candidate.revision()});
  }

  result.provisional = std::move(candidate);
  result.access = access;
  return result;
}

ConstantLoadResult ProposeSelectedImageLoads(const ir::SsaGraph& original,
                                             const ir::SsaImageAddressFacts& addresses,
                                             ir::ImageFacts facts, ir::ImageAccessContract access,
                                             Budget& budget) {
  const auto decline = [](ConstantLoadRefusal reason) {
    ConstantLoadResult refused;
    refused.reason = reason;
    return refused;
  };

  const auto checked = ir::ValidateSsaImageAddressFacts(original, addresses, budget);
  if (checked != ir::SsaDecline::none)
    return decline(checked == ir::SsaDecline::resource_limit ? ConstantLoadRefusal::resource_limit
                                                             : ConstantLoadRefusal::invalid_graph);
  const auto scope = ir::ValidateSsa(original, budget);
  if (scope != ir::SsaDecline::none)
    return decline(scope == ir::SsaDecline::resource_limit ? ConstantLoadRefusal::resource_limit
                                                           : ConstantLoadRefusal::invalid_graph);

  struct Proposed {
    ir::SsaHandle block;
    ir::SsaConstantLoad fold;
    std::uint64_t true_range;
    std::uint64_t true_bytes;
    std::uint64_t false_range;
    std::uint64_t false_bytes;
    bool placement;
  };

  ConstantLoadResult result;
  std::vector<Proposed> proposed;
  std::optional<ir::SsaImageStores> stores;
  for (const auto& address : addresses.selected) {
    if (budget.try_consume({1, sizeof(RefusedConstantLoad)}) != BudgetDecline::none)
      return decline(ConstantLoadRefusal::resource_limit);
    const auto& block = *original.Get(address.block);
    const auto& load = block.nodes[address.load];
    const auto refuse = [&](ConstantLoadRefusal reason) {
      result.refused.push_back({address.block, address.load, reason});
    };

    if (!load.width || load.width > 64 || load.width % 8) {
      refuse(ConstantLoadRefusal::unsupported_access);
      continue;
    }

    const auto omitted = Omitted(block, address.load, budget);
    if (!omitted) return decline(ConstantLoadRefusal::resource_limit);
    if (*omitted) {
      refuse(ConstantLoadRefusal::existing_omission);
      continue;
    }

    for (unsigned search = 0; search < 4; ++search)
      if (budget.try_consume({facts.constants.size(), 0}) != BudgetDecline::none)
        return decline(ConstantLoadRefusal::resource_limit);
    for (unsigned search = 0; search < 2; ++search)
      if (budget.try_consume({facts.pointers.size(), 0}) != BudgetDecline::none)
        return decline(ConstantLoadRefusal::resource_limit);
    if (budget.try_consume({32, 0}) != BudgetDecline::none)
      return decline(ConstantLoadRefusal::resource_limit);
    const auto truth =
        ir::ReadConstant(facts, address.when_true, load.width, load.access.byte_order);
    const auto falsity =
        ir::ReadConstant(facts, address.when_false, load.width, load.access.byte_order);
    const auto range = [&](std::uint64_t location) {
      return std::find_if(
          facts.constants.begin(), facts.constants.end(), [&](const ir::ConstantImageRange& item) {
            return location >= item.address && location - item.address < item.bytes.size() &&
                   load.width / 8 <= item.bytes.size() - (location - item.address);
          });
    };

    const auto true_range = range(address.when_true);
    const auto false_range = range(address.when_false);
    if (!truth || !falsity || true_range == facts.constants.end() ||
        false_range == facts.constants.end()) {
      refuse(ConstantLoadRefusal::no_invariant);
      continue;
    }

    if (!access.mapped_readable_lifetime || !access.no_runtime_unmapping ||
        !access.ordinary_reads_unobservable) {
      refuse(ConstantLoadRefusal::not_nonfaulting);
      continue;
    }

    if (load.access.alignment > 1 &&
        (!facts.page_aligned_placement || load.access.alignment > 4096 ||
         address.when_true % load.access.alignment || address.when_false % load.access.alignment)) {
      refuse(ConstantLoadRefusal::alignment);
      continue;
    }

    if (!stores && !(stores = ir::SsaImageStores::Collect(original, facts.page_aligned_placement,
                                                          budget, &facts)))
      return decline(ConstantLoadRefusal::resource_limit);
    const bool read_only = true_range->read_only && false_range->read_only;
    if (stores->Conflicts(address.when_true, load.width / 8, read_only) ||
        stores->Conflicts(address.when_false, load.width / 8, read_only)) {
      refuse(ConstantLoadRefusal::conflicting_store);
      continue;
    }

    ir::SsaConstantLoad fold{address.load, ir::SsaConstantKind::literal, *truth, address.when_true};
    fold.access = access;
    fold.page_aligned_placement = facts.page_aligned_placement;
    fold.value_stable = true;
    fold.skip_access = true;
    fold.read_only = read_only;
    fold.condition = address.condition;
    fold.alternative_value = *falsity;
    fold.alternative_source_address = address.when_false;
    std::copy_n(true_range->bytes.begin() + (address.when_true - true_range->address),
                load.width / 8, fold.declared_bytes.begin());
    std::copy_n(false_range->bytes.begin() + (address.when_false - false_range->address),
                load.width / 8, fold.alternative_declared_bytes.begin());
    if (budget.try_consume({1, sizeof(Proposed) + sizeof(ConstantLoadEdit)}) != BudgetDecline::none)
      return decline(ConstantLoadRefusal::resource_limit);
    proposed.push_back({address.block, fold, true_range->address, true_range->bytes.size(),
                        false_range->address, false_range->bytes.size(),
                        load.access.alignment > 1});
  }

  if (proposed.empty()) return result;
  auto admitted = Admit(original, proposed, result.refused, budget);
  if (const auto* reason = std::get_if<ConstantLoadRefusal>(&admitted)) {
    if (*reason == ConstantLoadRefusal::none) return result;
    return decline(*reason);
  }

  auto& [candidate, kept] = std::get<0>(admitted);
  if (budget.try_consume({proposed.size(), proposed.size() * sizeof(ConstantLoadEdit)}) !=
      BudgetDecline::none)
    return decline(ConstantLoadRefusal::resource_limit);
  for (std::size_t index = 0; index < proposed.size(); ++index) {
    if (!kept[index]) continue;
    const auto& proposal = proposed[index];
    result.journal.push_back({proposal.block,
                              {candidate.arena(), proposal.block.slot, proposal.block.generation},
                              proposal.fold,
                              ConstantLoadFact::constant_range,
                              proposal.true_range,
                              proposal.true_bytes,
                              proposal.placement,
                              original.revision(),
                              candidate.revision(),
                              proposal.false_range,
                              proposal.false_bytes});
  }

  result.provisional = std::move(candidate);
  result.access = access;
  return result;
}

namespace {
// One table fold, ready to write into a candidate.
struct TableFold {
  ir::SsaHandle block;
  ir::SsaConstantLoad fold;
  std::uint64_t fact_address;
  std::uint64_t fact_bytes;
  bool placement;
  ir::SsaBoundedTableAddressFact address;
  std::uint64_t work;
  std::uint64_t bytes;
};

// Everything one fold rests on that can be checked before the others join
// it: the fact, the declared rows, the access contract and the load itself.
std::variant<TableFold, ConstantLoadRefusal> PrepareTableFold(
    const ir::SsaGraph& original, const ir::SsaBoundedTableAddressFact& address,
    std::span<const ir::Group> sources, ir::ImageFacts facts, ir::ImageAccessContract access,
    Budget& budget) {
  const auto refuse = [](ConstantLoadRefusal reason) {
    return std::variant<TableFold, ConstantLoadRefusal>(reason);
  };

  const auto checked = ir::ValidateSsaBoundedTableAddressFact(original, address, sources, budget);
  if (checked != ir::SsaDecline::none)
    return refuse(checked == ir::SsaDecline::resource_limit ? ConstantLoadRefusal::resource_limit
                                                            : ConstantLoadRefusal::invalid_graph);
  const auto handle = address.index_bound.guard.guarded_block;
  const auto& block = *original.Get(handle);
  const auto& load = block.nodes[address.load];
  const auto count = address.index_bound.exclusive_upper;
  if (count > 64) return refuse(ConstantLoadRefusal::unsupported_access);
  const auto size = load.width / 8;
  const auto last_offset = (count - 1) * address.stride;
  if (last_offset > UINT64_MAX - size) return refuse(ConstantLoadRefusal::unsupported_access);
  const auto span = last_offset + size;
  if (budget.try_consume({facts.constants.size(), 0}) != BudgetDecline::none)
    return refuse(ConstantLoadRefusal::resource_limit);
  const auto range = std::find_if(
      facts.constants.begin(), facts.constants.end(), [&](const ir::ConstantImageRange& item) {
        return address.base >= item.address && address.base - item.address <= item.bytes.size() &&
               span <= item.bytes.size() - (address.base - item.address);
      });
  if (range == facts.constants.end()) return refuse(ConstantLoadRefusal::no_invariant);
  if (!access.mapped_readable_lifetime || !access.no_runtime_unmapping ||
      !access.ordinary_reads_unobservable)
    return refuse(ConstantLoadRefusal::not_nonfaulting);
  // A base taken from a PC page names the table only where the image is
  // placed at a page-aligned address.
  if (address.placed && !facts.page_aligned_placement)
    return refuse(ConstantLoadRefusal::no_invariant);
  if (load.access.alignment > 1 &&
      (!facts.page_aligned_placement || load.access.alignment > 4096 ||
       address.base % load.access.alignment || address.stride % load.access.alignment))
    return refuse(ConstantLoadRefusal::alignment);
  const auto omitted = Omitted(block, address.load, budget);
  if (!omitted) return refuse(ConstantLoadRefusal::resource_limit);
  if (*omitted || std::binary_search(block.disabled_effects.begin(), block.disabled_effects.end(),
                                     address.load))
    return refuse(ConstantLoadRefusal::existing_omission);
  if (budget.try_consume({count * (size + 2), count * sizeof(std::array<std::uint8_t, 8>)}) !=
      BudgetDecline::none)
    return refuse(ConstantLoadRefusal::resource_limit);
  ir::SsaConstantLoad fold{address.load, ir::SsaConstantKind::bounded_table, 0, address.base};
  fold.access = access;
  fold.page_aligned_placement = facts.page_aligned_placement;
  fold.value_stable = true;
  fold.skip_access = true;
  fold.read_only = range->read_only;
  if (budget.try_consume({64, 0}) != BudgetDecline::none)
    return refuse(ConstantLoadRefusal::resource_limit);
  const auto shape = ir::SsaTableLoadAddress(block.nodes, address.load, original.load_bias());
  if (!shape) return refuse(ConstantLoadRefusal::invalid_graph);
  fold.table_index = shape->index;
  fold.table_stride = address.stride;
  fold.table_guard = address.index_bound.guard.branch;
  fold.table_guard_edge = address.index_bound.guard.edge_index;
  fold.table_storage = address.index_bound.storage;
  fold.table_closed_entries =
      address.index_bound.guard.entry_scope == ir::SsaEntryScope::closed_population;
  const auto* guard = original.Get(address.index_bound.guard.branch);
  if (!guard) return refuse(ConstantLoadRefusal::invalid_graph);
  fold.table_bytes.reserve(count);
  for (std::uint64_t row = 0; row < count; ++row) {
    const auto location = address.base + row * address.stride;
    if (budget.try_consume({facts.constants.size() + facts.pointers.size(), 0}) !=
        BudgetDecline::none)
      return refuse(ConstantLoadRefusal::resource_limit);
    const auto declared = ir::ReadConstant(facts, location, load.width, load.access.byte_order);
    if (!declared) return refuse(ConstantLoadRefusal::no_invariant);
    const auto conflict = ir::SsaConflictingImageStore(
        original, location, size, facts.page_aligned_placement, budget, range->read_only, &facts);
    if (!conflict) return refuse(ConstantLoadRefusal::resource_limit);
    if (*conflict) return refuse(ConstantLoadRefusal::conflicting_store);
    std::array<std::uint8_t, 8> value{};
    const auto offset = location - range->address;
    std::copy_n(range->bytes.begin() + offset, size, value.begin());
    std::uint64_t copied = 0;
    if (load.access.byte_order == ir::ByteOrder::little) {
      for (std::size_t byte = 0; byte < size; ++byte)
        copied |= std::uint64_t{value[byte]} << (byte * 8);
    } else {
      for (std::size_t byte = 0; byte < size; ++byte) copied = (copied << 8) | value[byte];
    }

    if (copied != *declared) return refuse(ConstantLoadRefusal::no_invariant);
    fold.table_bytes.push_back(value);
  }

  return TableFold{handle,
                   std::move(fold),
                   range->address,
                   range->bytes.size(),
                   load.access.alignment > 1 || address.placed,
                   address,
                   count,
                   count * sizeof(std::array<std::uint8_t, 8>)};
}
}  // namespace

ConstantLoadResult ProposeBoundedTableLoads(
    const ir::SsaGraph& original, std::span<const ir::SsaBoundedTableAddressFact> addresses,
    std::span<const ir::Group> sources, ir::ImageFacts facts, ir::ImageAccessContract access,
    Budget& budget) {
  const auto decline = [](ConstantLoadRefusal reason) {
    ConstantLoadResult refused;
    refused.reason = reason;
    return refused;
  };

  ConstantLoadResult result;
  std::vector<TableFold> folds;
  if (addresses.size() > SIZE_MAX / sizeof(TableFold) ||
      budget.try_consume({addresses.size(), addresses.size() * sizeof(TableFold)}) !=
          BudgetDecline::none)
    return decline(ConstantLoadRefusal::resource_limit);
  folds.reserve(addresses.size());
  for (const auto& address : addresses) {
    auto prepared = PrepareTableFold(original, address, sources, facts, access, budget);
    const auto handle = address.index_bound.guard.guarded_block;
    if (const auto* reason = std::get_if<ConstantLoadRefusal>(&prepared)) {
      if (*reason == ConstantLoadRefusal::resource_limit) return decline(*reason);
      if (budget.try_consume({1, sizeof(RefusedConstantLoad)}) != BudgetDecline::none)
        return decline(ConstantLoadRefusal::resource_limit);
      result.refused.push_back({handle, address.load, *reason});
      continue;
    }

    auto& fold = std::get<TableFold>(prepared);
    if (budget.try_consume({folds.size(), 0}) != BudgetDecline::none)
      return decline(ConstantLoadRefusal::resource_limit);
    // The same load twice would fold it twice.
    if (std::any_of(folds.begin(), folds.end(), [&](const TableFold& other) {
          return other.block == fold.block && other.fold.node == fold.fold.node;
        })) {
      if (budget.try_consume({1, sizeof(RefusedConstantLoad)}) != BudgetDecline::none)
        return decline(ConstantLoadRefusal::resource_limit);
      result.refused.push_back({handle, address.load, ConstantLoadRefusal::existing_omission});
      continue;
    }

    folds.push_back(std::move(fold));
  }

  if (folds.empty()) {
    result.reason =
        result.refused.empty() ? ConstantLoadRefusal::invalid_graph : result.refused.front().reason;
    return result;
  }

  // A fold stands only where every block that can run is complete, and a
  // dispatch not yet folded is not; so every dispatch the graph can fold is
  // written into one candidate and checked once; one failure rejects all.
  auto candidate = original.Clone(budget);
  if (!candidate) return decline(ConstantLoadRefusal::resource_limit);
  for (auto& table : folds) {
    const ir::SsaHandle result_handle{candidate->arena(), table.block.slot, table.block.generation};
    table.fold.table_guard->arena = candidate->arena();
    auto copied = candidate->CopyBlock(result_handle, budget);
    if (!copied) return decline(ConstantLoadRefusal::resource_limit);
    auto changed = std::move(*copied);
    if (!ChargeInsert(changed.disabled_effects.size(), sizeof(ir::ValueId), budget) ||
        !ChargeInsert(changed.constant_loads.size(), sizeof(ir::SsaConstantLoad), budget) ||
        budget.try_consume({table.work, table.bytes}) != BudgetDecline::none)
      return decline(ConstantLoadRefusal::resource_limit);
    const auto node = table.fold.node;
    changed.disabled_effects.insert(
        std::lower_bound(changed.disabled_effects.begin(), changed.disabled_effects.end(), node),
        node);
    changed.constant_loads.insert(
        std::lower_bound(
            changed.constant_loads.begin(), changed.constant_loads.end(), node,
            [](const ir::SsaConstantLoad& item, ir::ValueId id) { return item.node < id; }),
        table.fold);
    if (!candidate->Replace(result_handle, std::move(changed)))
      return decline(ConstantLoadRefusal::invalid_graph);
  }

  const auto valid = ir::ValidateSsaWithSources(*candidate, sources, budget);
  if (valid != ir::SsaDecline::none)
    return decline(valid == ir::SsaDecline::resource_limit ? ConstantLoadRefusal::resource_limit
                                                           : ConstantLoadRefusal::invalid_graph);
  result.access = access;
  if (folds.size() > SIZE_MAX / sizeof(ConstantLoadEdit) ||
      budget.try_consume({folds.size(), folds.size() * sizeof(ConstantLoadEdit)}) !=
          BudgetDecline::none)
    return decline(ConstantLoadRefusal::resource_limit);
  for (const auto& table : folds) {
    if (budget.try_consume({table.work, table.bytes}) != BudgetDecline::none)
      return decline(ConstantLoadRefusal::resource_limit);
    result.journal.push_back({table.block,
                              {candidate->arena(), table.block.slot, table.block.generation},
                              table.fold,
                              ConstantLoadFact::constant_range,
                              table.fact_address,
                              table.fact_bytes,
                              table.placement,
                              original.revision(),
                              candidate->revision(),
                              0,
                              0,
                              table.address});
  }

  result.provisional = std::move(*candidate);
  return result;
}

ConstantLoadResult ProposeBoundedTableLoads(const ir::SsaGraph& original,
                                            const ir::SsaBoundedTableAddressFact& address,
                                            std::span<const ir::Group> sources,
                                            ir::ImageFacts facts, ir::ImageAccessContract access,
                                            Budget& budget) {
  auto result =
      ProposeBoundedTableLoads(original, std::span(&address, 1), sources, facts, access, budget);
  result.refused.clear();
  return result;
}

}  // namespace nyx::recovery
