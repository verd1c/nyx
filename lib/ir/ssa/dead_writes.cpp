#include <algorithm>
#include <bit>
#include <limits>

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::ir {
namespace {

enum : std::uint8_t { kUndecided, kLive, kDead };

bool Listed(std::span<const StorageId> sorted, StorageId storage) {
  return std::binary_search(sorted.begin(), sorted.end(), storage);
}

}  // namespace

std::optional<SsaStorageLiveness> SsaStorageLiveness::Compute(const SsaGraph& graph,
                                                              Budget& budget) {
  SsaStorageLiveness result(graph);
  const auto& observed = graph.observability();
  auto& storages = result.storages_;
  const auto add = [&](StorageId storage) { storages.push_back(storage); };
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto& block = *graph.Get(*handle);
    if (budget.try_consume({1 + block.phis.size() + block.nodes.size() + block.boundaries.size(),
                            0}) != BudgetDecline::none)
      return {};
    for (const auto& phi : block.phis) add(phi.storage);
    for (const auto& node : block.nodes)
      if (node.op == Op::read || node.op == Op::write) add(node.storage);
    for (const auto& boundary : block.boundaries) {
      if (budget.try_consume({boundary.writes.size(), 0}) != BudgetDecline::none) return {};
      for (const auto& write : boundary.writes) add(write.storage);
    }
  }

  // A callee body may write storage the population never names; skipping the
  // call leaves that storage as it was, so its liveness must be decided too.
  for (const auto& body : graph.callee_bodies())
    for (const auto& group : body.groups) {
      if (budget.try_consume({1 + group.nodes().size() + group.writes().size(), 0}) !=
          BudgetDecline::none)
        return {};
      for (const auto& node : group.nodes())
        if (node.op == Op::read || node.op == Op::write) add(node.storage);
      for (const auto& write : group.writes()) add(write.storage);
    }
  if (budget.try_consume({storages.size() * (1 + std::bit_width(storages.size())),
                          storages.size() * sizeof(StorageId)}) != BudgetDecline::none)
    return {};
  std::sort(storages.begin(), storages.end());
  storages.erase(std::unique(storages.begin(), storages.end()), storages.end());
  const auto count = storages.size();
  const auto slots = graph.slots();
  if (slots > std::numeric_limits<std::size_t>::max() / (2 * count + 2) ||
      budget.try_consume(
          {slots, slots * (2 * count + 1 + 2 * sizeof(std::vector<std::uint8_t>))}) !=
          BudgetDecline::none)
    return {};
  result.complete_.assign(slots, 0);
  result.leaf_reads_.assign(slots, std::nullopt);
  result.live_in_.assign(slots, std::vector<std::uint8_t>(count, observed.declared ? 0 : 1));

  // What each block decides before any successor: live, dead, or undecided.
  std::vector<std::vector<std::uint8_t>> entry(slots);
  for (std::size_t slot = 0; slot < slots; ++slot) {
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto& block = *graph.Get(*handle);
    const auto direct = ValidateSsaDirectSuccessors(block, budget);
    if (direct == SsaDecline::resource_limit) return {};
    result.complete_[slot] = direct == SsaDecline::none;
    result.leaf_reads_[slot] = SsaLeafCalleeReads(graph, block, budget);
    entry[slot].assign(count, kUndecided);
    if (!result.Scan(block, 0, entry[slot], budget)) return {};
  }

  // Monotone from either end, so each round changes a bit or stops.
  for (bool changed = true; changed;) {
    changed = false;
    for (std::size_t slot = slots; slot-- > 0;) {
      const auto handle = graph.Handle(slot);
      if (!handle) continue;
      const auto& block = *graph.Get(*handle);
      if (budget.try_consume({1 + count * (1 + block.edges.size()), 0}) != BudgetDecline::none)
        return {};
      for (std::size_t index = 0; index < count; ++index) {
        const auto state = entry[slot][index];
        const std::uint8_t live =
            state == kLive || (state == kUndecided && result.Leaves(block, slot, index));
        if (live != result.live_in_[slot][index]) {
          result.live_in_[slot][index] = live;
          changed = true;
        }
      }
    }
  }

  return result;
}

// Advances each undecided storage through the block from `first`: a read, or
// a possible fault the contract observes, makes it live; a write, dead.
bool SsaStorageLiveness::Scan(const SsaBlock& block, std::size_t first,
                              std::vector<std::uint8_t>& states, Budget& budget) const {
  const bool faults_observed = !graph_->observability().faults_terminal;
  const auto set = [&](StorageId storage, std::uint8_t state) {
    const auto at = std::lower_bound(storages_.begin(), storages_.end(), storage);
    if (at != storages_.end() && *at == storage && states[at - storages_.begin()] == kUndecided)
      states[at - storages_.begin()] = state;
  };

  // Only an opaque block the declared-trap rule completes decides nothing
  // here; any other leaves every storage live through its incomplete edges.
  if (block.opaque) return true;
  for (std::size_t index = first; index < block.boundaries.size(); ++index) {
    const auto& boundary = block.boundaries[index];
    if (budget.try_consume({std::uint64_t{1} + 2 * boundary.node_count + boundary.writes.size(),
                            0}) != BudgetDecline::none)
      return false;
    bool faults = false;

    // An instruction reads its inputs, and may fault, before it writes.
    for (std::size_t node = boundary.first_node; node < boundary.first_node + boundary.node_count;
         ++node) {
      faults |= HasMemoryOrMonitorEffect(block.nodes[node].op);
      if (block.nodes[node].op == Op::read) set(block.nodes[node].storage, kLive);
    }

    if (faults && faults_observed) {
      std::replace(states.begin(), states.end(), std::uint8_t{kUndecided}, std::uint8_t{kLive});
      return true;
    }

    for (std::size_t node = boundary.first_node; node < boundary.first_node + boundary.node_count;
         ++node)
      if (block.nodes[node].op == Op::write) set(block.nodes[node].storage, kDead);
    for (const auto& write : boundary.writes) set(write.storage, kDead);
  }

  return true;
}

bool SsaStorageLiveness::Leaves(const SsaBlock& block, std::size_t slot, std::size_t index) const {
  const auto& observed = graph_->observability();
  if (!complete_[slot] || block.edges.empty()) return true;
  const auto storage = storages_[index];
  for (const auto& edge : block.edges) {
    bool live = true;
    if (edge.target_block) {
      live = !graph_->Get(*edge.target_block) || live_in_[edge.target_block->slot][index];
    } else if (edge.kind == SsaEdgeKind::callee && leaf_reads_[slot]) {
      // A leaf's body is what it does: it reads these and cannot fault.
      live = Listed(*leaf_reads_[slot], storage);
    } else if (edge.kind == SsaEdgeKind::callee && observed.declared) {
      // An unknown callee may fault before it returns, holding every storage.
      live = !observed.faults_terminal || Listed(observed.at_call, storage);
    } else if (edge.kind == SsaEdgeKind::return_ && observed.declared) {
      live = Listed(observed.at_return, storage);
    } else if (edge.kind == SsaEdgeKind::trap) {
      live = !observed.faults_terminal;
    }

    if (live) return true;
  }

  return false;
}

bool SsaStorageLiveness::LiveIn(std::size_t slot, StorageId storage) const {
  const auto at = std::lower_bound(storages_.begin(), storages_.end(), storage);
  if (at == storages_.end() || *at != storage || slot >= live_in_.size()) return false;
  return live_in_[slot][static_cast<std::size_t>(at - storages_.begin())] != 0;
}

namespace {

// One block's use of its frame slots' entry values, given which slots are
// live where it leaves. A value is live when it reaches something execution
// needs: an access that still runs or still checks its address, a register
// write that was kept, a transfer or edge condition, a value another block
// reads, or a frame exit into a slot that is live after the block. A load of
// a promoted slot passes that need on to whatever the slot held.
bool UsedEntries(const SsaGraph& graph, SsaHandle handle, std::span<const std::size_t> key_of,
                 std::span<const std::uint8_t> live_out, std::span<const std::uint8_t> foreign,
                 std::vector<std::uint8_t>& used, Budget& budget) {
  const auto& block = *graph.Get(handle);
  const auto& nodes = block.nodes;
  std::uint64_t writes = 0;
  for (const auto& boundary : block.boundaries) writes += boundary.writes.size();
  if (budget.try_consume({1 + 2 * nodes.size() + block.boundaries.size() + block.reads.size() +
                              block.exits.size() * (1 + writes + block.dead_storage_writes.size()) +
                              block.frame_exits.size() + block.edges.size() +
                              block.frame_accesses.size(),
                          2 * nodes.size()}) != BudgetDecline::none)
    return false;
  const auto listed = [](const std::vector<ValueId>& sorted, ValueId id) {
    return std::binary_search(sorted.begin(), sorted.end(), id);
  };

  std::vector<std::uint8_t> live(nodes.size());
  std::vector<ValueId> stack;
  const auto root = [&](ValueId id) {
    if (id < nodes.size() && !live[id]) {
      live[id] = 1;
      stack.push_back(id);
    }
  };

  const auto value = [&](const SsaValue& named) {
    if (named.block != handle) return;
    if (named.kind == SsaValueKind::node)
      root(named.index);
    else if (named.kind == SsaValueKind::frame_phi && named.index < key_of.size())
      used[key_of[named.index]] = 1;
  };

  for (ValueId id = 0; id < nodes.size(); ++id) {
    if (foreign[id]) root(id);
    const auto& node = nodes[id];
    if (!HasMemoryOrMonitorEffect(node.op)) continue;
    const auto fold =
        std::lower_bound(block.constant_loads.begin(), block.constant_loads.end(), id,
                         [](const SsaConstantLoad& load, ValueId key) { return load.node < key; });
    const bool skipped =
        fold != block.constant_loads.end() && fold->node == id && fold->skip_access;
    const auto* descriptor = Descriptor(node.op);
    if (descriptor && descriptor->arity && !skipped) root(node.inputs[0]);
    if (MayWriteMemory(node.op) && !listed(block.disabled_effects, id) && descriptor &&
        descriptor->arity > 1)
      root(node.inputs[1]);
  }

  for (std::size_t index = 0; index < block.boundaries.size(); ++index) {
    const auto& boundary = block.boundaries[index];
    for (auto id = boundary.first_node; id < boundary.first_node + boundary.node_count; ++id)
      if (id < nodes.size() && nodes[id].op == Op::write) root(nodes[id].inputs[0]);
    for (std::uint32_t at = 0; at < boundary.writes.size(); ++at)
      if (!std::any_of(block.dead_storage_writes.begin(), block.dead_storage_writes.end(),
                       [&](const SsaDeadStorageWrite& item) {
                         return item.boundary == index && item.index == at;
                       }))
        root(boundary.writes[at].value);
    const auto rewrite =
        std::find_if(block.control_rewrites.begin(), block.control_rewrites.end(),
                     [&](const ConditionalRewrite& item) { return item.boundary == index; });
    const auto* transfer = rewrite != block.control_rewrites.end() ? &rewrite->replacement
                           : boundary.transfer                     ? &*boundary.transfer
                                                                   : nullptr;
    if (!transfer) continue;
    root(transfer->target);
    if (transfer->condition) root(*transfer->condition);
    if (transfer->alternative) root(*transfer->alternative);
    if (transfer->continuation) root(*transfer->continuation);
  }

  for (const auto& edge : block.edges)
    if (edge.condition) root(*edge.condition);
  for (const auto& fold : block.constant_loads) {
    if (fold.condition) root(*fold.condition);
    if (fold.kind == SsaConstantKind::bounded_table) root(fold.table_index);
  }

  // An exit whose storage's last write here was retired as dead publishes a
  // value nothing observes.
  for (const auto& exit : block.exits) {
    std::optional<std::pair<std::size_t, std::uint32_t>> last;
    for (std::size_t index = 0; index < block.boundaries.size(); ++index) {
      const auto& writes = block.boundaries[index].writes;
      for (std::uint32_t at = 0; at < writes.size(); ++at)
        if (writes[at].storage == exit.storage) last = std::pair{index, at};
    }

    if (last && std::any_of(block.dead_storage_writes.begin(), block.dead_storage_writes.end(),
                            [&](const SsaDeadStorageWrite& item) {
                              return item.boundary == last->first && item.index == last->second;
                            }))
      continue;
    value(exit.value);
  }

  for (std::size_t index = 0; index < block.frame_exits.size() && index < key_of.size(); ++index) {
    const auto& exit = block.frame_exits[index];
    const bool passes =
        exit.kind == SsaValueKind::frame_phi && exit.block == handle && exit.index == index;
    if (!passes && live_out[key_of[index]]) value(exit);
  }
  while (!stack.empty()) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return false;
    const auto id = stack.back();
    stack.pop_back();
    const auto& node = nodes[id];
    if (node.op == Op::read) {
      for (const auto& read : block.reads)
        if (read.node == id) value(read.value);
      continue;
    }

    if (node.op == Op::load) {
      const auto frame = std::lower_bound(
          block.frame_accesses.begin(), block.frame_accesses.end(), id,
          [](const SsaFrameAccess& access, ValueId key) { return access.node < key; });
      if (frame != block.frame_accesses.end() && frame->node == id && frame->replacement)
        value(*frame->replacement);
      continue;
    }

    if (HasMemoryOrMonitorEffect(node.op)) continue;
    const auto* descriptor = Descriptor(node.op);
    for (unsigned input = 0; descriptor && input < descriptor->arity; ++input)
      root(node.inputs[input]);
  }

  return true;
}

}  // namespace

std::optional<SsaFrameLiveness> SsaFrameLiveness::Compute(const SsaGraph& graph, Budget& budget) {
  SsaFrameLiveness result;
  auto& keys = result.slots_;
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto& block = *graph.Get(*handle);
    if (budget.try_consume({1 + block.frame_phis.size(),
                            block.frame_phis.size() * sizeof(keys[0])}) != BudgetDecline::none)
      return {};
    for (const auto& phi : block.frame_phis) keys.push_back({phi.offset, phi.size});
  }

  if (budget.try_consume({keys.size() * (1 + std::bit_width(keys.size())), 0}) !=
      BudgetDecline::none)
    return {};
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  const auto count = keys.size();
  const auto slots = graph.slots();
  if (slots > std::numeric_limits<std::size_t>::max() / (4 * count + 4) ||
      budget.try_consume({slots, slots * (4 * count + 4 * sizeof(std::vector<std::uint8_t>))}) !=
          BudgetDecline::none)
    return {};
  result.live_in_.assign(slots, std::vector<std::uint8_t>(count, 0));
  std::vector<std::vector<std::size_t>> key_of(slots);
  std::vector<std::vector<std::uint8_t>> passes(slots), foreign(slots);
  std::vector<std::uint8_t> open(slots);
  for (std::size_t slot = 0; slot < slots; ++slot) {
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto& block = *graph.Get(*handle);
    if (budget.try_consume(
            {block.frame_phis.size() * (1 + std::bit_width(count)) + block.nodes.size(),
             block.nodes.size()}) != BudgetDecline::none)
      return {};
    for (const auto& phi : block.frame_phis)
      key_of[slot].push_back(static_cast<std::size_t>(
          std::lower_bound(keys.begin(), keys.end(), std::pair{phi.offset, phi.size}) -
          keys.begin()));
    passes[slot].assign(count, 0);
    foreign[slot].assign(block.nodes.size(), 0);
    const auto direct = ValidateSsaDirectSuccessors(block, budget);
    if (direct == SsaDecline::resource_limit) return {};
    open[slot] = block.opaque || direct != SsaDecline::none;
    for (std::size_t index = 0; index < block.frame_exits.size() && index < block.frame_phis.size();
         ++index) {
      const auto& exit = block.frame_exits[index];
      if (exit.kind == SsaValueKind::frame_phi && exit.block == *handle && exit.index == index)
        passes[slot][key_of[slot][index]] = 1;
    }
  }

  // A value one block reads out of another, not through the phis that stand
  // for storage at its entry, is live wherever it comes from.
  for (std::size_t slot = 0; slot < slots; ++slot) {
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto& block = *graph.Get(*handle);
    if (budget.try_consume({block.reads.size(), 0}) != BudgetDecline::none) return {};
    for (const auto& read : block.reads)
      for (const auto* named :
           {&read.value, read.predecessor_copy ? &*read.predecessor_copy : nullptr})
        if (named && named->kind == SsaValueKind::node && named->block != *handle &&
            graph.Get(named->block) && named->block.slot < slots &&
            named->index < foreign[named->block.slot].size())
          foreign[named->block.slot][named->index] = 1;
  }

  // Least solution: what is live only grows, so this stops within blocks
  // times slots rounds.
  std::vector<std::uint8_t> live_out(count), used(count);
  for (bool changed = true; changed;) {
    changed = false;
    for (std::size_t slot = slots; slot-- > 0;) {
      const auto handle = graph.Handle(slot);
      if (!handle) continue;
      const auto& block = *graph.Get(*handle);
      if (budget.try_consume({1 + count * (2 + block.edges.size()), 0}) != BudgetDecline::none)
        return {};
      std::fill(live_out.begin(), live_out.end(), std::uint8_t{open[slot]});
      for (const auto& edge : block.edges) {
        if (edge.target_block && graph.Get(*edge.target_block)) {
          for (std::size_t key = 0; key < count; ++key)
            live_out[key] |= result.live_in_[edge.target_block->slot][key];
        } else if (edge.kind != SsaEdgeKind::return_ && edge.kind != SsaEdgeKind::callee &&
                   edge.kind != SsaEdgeKind::potential_return && edge.kind != SsaEdgeKind::trap) {
          std::fill(live_out.begin(), live_out.end(), std::uint8_t{1});
        }
      }

      std::fill(used.begin(), used.end(), std::uint8_t{0});
      if (open[slot])
        std::fill(used.begin(), used.end(), std::uint8_t{1});
      else if (!UsedEntries(graph, *handle, key_of[slot], live_out, foreign[slot], used, budget))
        return {};
      for (std::size_t key = 0; key < count; ++key) {
        const bool live = used[key] || (passes[slot][key] && live_out[key]);
        if (live && !result.live_in_[slot][key]) {
          result.live_in_[slot][key] = 1;
          changed = true;
        }
      }
    }
  }

  return result;
}

bool SsaFrameLiveness::LiveIn(std::size_t slot, std::int64_t offset, std::uint32_t size) const {
  const auto at = std::lower_bound(slots_.begin(), slots_.end(), std::pair{offset, size});
  if (at == slots_.end() || *at != std::pair{offset, size} || slot >= live_in_.size()) return false;
  return live_in_[slot][static_cast<std::size_t>(at - slots_.begin())] != 0;
}

SsaDecline SsaStorageLiveness::CheckDeadWrite(SsaDeadWriteFact fact, Budget& budget) const {
  const auto* source = graph_->Get(fact.block);
  if (!source || fact.boundary >= source->boundaries.size() ||
      fact.index >= source->boundaries[fact.boundary].writes.size() ||
      fact.block.slot >= live_in_.size())
    return SsaDecline::invalid_graph;
  const auto& writes = source->boundaries[fact.boundary].writes;
  const auto storage = writes[fact.index].storage;
  if (budget.try_consume({writes.size() + storages_.size(), storages_.size()}) !=
      BudgetDecline::none)
    return SsaDecline::resource_limit;
  for (std::size_t i = 0; i < writes.size(); ++i)
    if (i != fact.index && writes[i].storage == storage) return SsaDecline::invalid_graph;
  const auto at = std::lower_bound(storages_.begin(), storages_.end(), storage);
  if (at == storages_.end() || *at != storage) return SsaDecline::invalid_graph;
  const auto index = static_cast<std::size_t>(at - storages_.begin());
  std::vector<std::uint8_t> states(storages_.size(), kUndecided);
  if (!Scan(*source, std::size_t{fact.boundary} + 1, states, budget))
    return SsaDecline::resource_limit;
  const bool live = states[index] == kLive ||
                    (states[index] == kUndecided && Leaves(*source, fact.block.slot, index));
  return live ? SsaDecline::invalid_graph : SsaDecline::none;
}

SsaDecline CheckSsaDeadStorageWrite(const SsaGraph& graph, SsaDeadWriteFact fact, Budget& budget) {
  const auto liveness = SsaStorageLiveness::Compute(graph, budget);
  return liveness ? liveness->CheckDeadWrite(fact, budget) : SsaDecline::resource_limit;
}

SsaDecline ValidateSsaDeadWriteFacts(const SsaGraph& graph, const SsaReachabilityFacts& reachable,
                                     const SsaDeadWriteFacts& facts, std::span<const Group> sources,
                                     Budget& budget) {
  const auto valid = ValidateSsaWithSources(graph, sources, budget);
  if (valid != SsaDecline::none) return valid;
  const auto scope = ValidateSsaReachabilityFacts(graph, reachable, sources, budget);
  if (scope != SsaDecline::none) return scope;
  if (reachable.entry_scope != SsaEntryScope::closed_population) return SsaDecline::invalid_graph;
  if (facts.graph_arena != graph.arena() || facts.graph_revision != graph.revision())
    return SsaDecline::invalid_graph;
  const auto liveness = SsaStorageLiveness::Compute(graph, budget);
  if (!liveness) return SsaDecline::resource_limit;
  SsaDeadWriteFact previous{};
  bool first = true;
  for (const auto& fact : facts.writes) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return SsaDecline::resource_limit;
    if (!graph.Get(fact.block) || !reachable.reachable[fact.block.slot] ||
        (!first && (fact.block.slot < previous.block.slot ||
                    (fact.block.slot == previous.block.slot &&
                     (fact.boundary < previous.boundary ||
                      (fact.boundary == previous.boundary && fact.index <= previous.index))))))
      return SsaDecline::invalid_graph;
    const auto checked = liveness->CheckDeadWrite(fact, budget);
    if (checked != SsaDecline::none) return checked;
    previous = fact;
    first = false;
  }

  return SsaDecline::none;
}

}  // namespace nyx::ir
