#include "nyx/ir/frame.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <tuple>

namespace nyx::ir {
namespace {
using Kind = FrameAddress::Kind;

std::optional<std::int64_t> Add(std::int64_t a, std::int64_t b) {
  std::int64_t result;
  if (__builtin_add_overflow(a, b, &result)) return {};
  return result;
}

std::optional<std::int64_t> Sub(std::int64_t a, std::int64_t b) {
  std::int64_t result;
  if (__builtin_sub_overflow(a, b, &result)) return {};
  return result;
}

constexpr FrameAddress kOther{Kind::other};
constexpr FrameAddress kAny{Kind::any};

FrameAddress Join(FrameAddress a, FrameAddress b) {
  if (a.kind == Kind::none) return b;
  if (b.kind == Kind::none || a == b) return a;
  if (a.kind == Kind::other && b.kind == Kind::other) return kOther;
  return kAny;
}

bool Frame(FrameAddress value) { return value.kind == Kind::exact || value.kind == Kind::any; }

bool Listed(std::span<const StorageId> sorted, StorageId storage) {
  return std::binary_search(sorted.begin(), sorted.end(), storage);
}

// The least solution over every SSA value of the graph. Each value only
// climbs none < other < any or none < exact < any, and a literal only loses
// its value, so the rounds stop.
class Frames {
 public:
  Frames(const SsaGraph& graph, const PrivateFrameContract& contract, Budget& budget)
      : graph_(graph), contract_(contract), budget_(budget) {}

  bool Solve() {
    const auto count = graph_.slots();
    if (!Charge(count, count * 5 * sizeof(std::vector<FrameAddress>))) return false;
    nodes_.resize(count);
    phis_.resize(count);
    frame_phis_.resize(count);
    reads_.resize(count);
    ends_.resize(count);
    for (std::size_t slot = 0; slot < count; ++slot) {
      const auto handle = graph_.Handle(slot);
      if (!handle) continue;
      const auto& block = *graph_.Get(*handle);
      if (!Charge(block.nodes.size() * 2 + block.phis.size() + block.frame_phis.size(),
                  block.nodes.size() * (sizeof(FrameAddress) + sizeof(std::uint32_t)) +
                      block.phis.size() * (sizeof(FrameAddress) + sizeof(ValueId)) +
                      block.frame_phis.size() * sizeof(FrameAddress)))
        return false;
      nodes_[slot].assign(block.nodes.size(), {});
      phis_[slot].assign(block.phis.size(), {});
      frame_phis_[slot].assign(block.frame_phis.size(), {});
      reads_[slot].assign(block.nodes.size(), kNoRead);
      for (std::uint32_t index = 0; index < block.reads.size(); ++index)
        if (block.reads[index].node < block.nodes.size())
          reads_[slot][block.reads[index].node] = index;
      // The value each storage holds at the block's end, before a call clobbers it.
      ends_[slot].assign(block.phis.size(), kNoWrite);
      const auto last = [&](StorageId storage, ValueId value) {
        for (std::size_t phi = 0; phi < block.phis.size(); ++phi)
          if (block.phis[phi].storage == storage) ends_[slot][phi] = value;
      };

      for (const auto& boundary : block.boundaries) {
        if (!Charge((boundary.node_count + boundary.writes.size()) * (1 + block.phis.size()), 0))
          return false;
        for (auto id = boundary.first_node; id < boundary.first_node + boundary.node_count; ++id)
          if (id < block.nodes.size() && block.nodes[id].op == Op::write)
            last(block.nodes[id].storage, block.nodes[id].inputs[0]);
        for (const auto& write : boundary.writes) last(write.storage, write.value);
      }
    }

    for (bool changed = true; changed;) {
      changed = false;
      for (std::size_t slot = 0; slot < count; ++slot) {
        const auto handle = graph_.Handle(slot);
        if (!handle) continue;
        const auto& block = *graph_.Get(*handle);
        if (!Charge(1 + block.nodes.size() + block.phis.size() + block.frame_phis.size(), 0))
          return false;
        const auto update = [&](FrameAddress& value, FrameAddress next) {
          if (value == next) return;
          value = next;
          changed = true;
        };

        for (std::size_t index = 0; index < block.phis.size(); ++index) {
          const auto& phi = block.phis[index];
          if (!Charge(phi.incoming.size(), 0)) return false;
          FrameAddress value;
          if (phi.external_entry)
            value = phi.storage == contract_.sp_storage ? FrameAddress{Kind::exact, 0} : kOther;
          for (const auto& input : phi.incoming) value = Join(value, Resolve(input.value));
          update(phis_[slot][index], value);
        }

        for (std::size_t index = 0; index < block.frame_phis.size(); ++index) {
          const auto& phi = block.frame_phis[index];
          if (!Charge(phi.incoming.size(), 0)) return false;

          // A fresh region holds nothing that points into it.
          FrameAddress value = phi.external_entry ? kOther : FrameAddress{};
          for (const auto& input : phi.incoming) value = Join(value, Resolve(input.value));
          update(frame_phis_[slot][index], value);
        }

        for (ValueId id = 0; id < block.nodes.size(); ++id) {
          update(nodes_[slot][id], Evaluate(block, slot, id));

          // A frame address kept in a private slot is what later loads of
          // that slot can read back.
          const auto& node = block.nodes[id];
          if (node.op != Op::store || node.inputs[1] >= id || node.inputs[0] >= id) continue;
          const auto value = nodes_[slot][node.inputs[1]];
          const auto address = nodes_[slot][node.inputs[0]];
          if (!Frame(value) || address.kind != Kind::exact || node.width % 8) continue;
          if (!Charge(1 + std::bit_width(contents_.size()), sizeof(Content))) return false;
          const Content key{address.offset, node.width / 8, {}};
          auto at = std::lower_bound(contents_.begin(), contents_.end(), key, Before);
          if (at == contents_.end() || at->offset != key.offset || at->size != key.size)
            at = contents_.insert(at, key);
          update(at->value, Join(at->value, value));
        }
      }
    }

    return true;
  }

  FrameAddress Resolve(const SsaValue& value) const {
    const auto* block = graph_.Get(value.block);
    if (!block) return kAny;
    const auto slot = value.block.slot;
    switch (value.kind) {
      case SsaValueKind::node:
        return value.index < nodes_[slot].size() ? nodes_[slot][value.index] : kAny;
      case SsaValueKind::phi:
        return value.index < phis_[slot].size() ? phis_[slot][value.index] : kAny;
      case SsaValueKind::frame_phi:
        return value.index < frame_phis_[slot].size() ? frame_phis_[slot][value.index] : kAny;
      case SsaValueKind::clobber: {
        if (value.index >= block->phis.size() || block->opaque) return kAny;
        const bool call =
            std::any_of(block->edges.begin(), block->edges.end(),
                        [](const SsaEdge& edge) { return edge.kind == SsaEdgeKind::callee; });
        // A retired call no longer runs its callee, so the storage it would
        // have left fresh still holds what it held before the call.
        if (!call && std::any_of(block->control_rewrites.begin(), block->control_rewrites.end(),
                                 [](const ConditionalRewrite& rewrite) {
                                   return rewrite.rule == RewriteRule::retired_call;
                                 }))
          return AtEnd(slot, value.index);
        if (!call) return kAny;

        // A callee leaves what it preserves as it found it, and may leave
        // anything else so too; under the contract it hands back no address
        // into the frame.
        const auto before = AtEnd(slot, value.index);
        return Preserved(block->phis[value.index].storage) ? before : Join(kOther, before);
      }
      case SsaValueKind::input:
      case SsaValueKind::frame_input:
        return kOther;
    }

    return kAny;
  }

  // Whether an exact store of `size` bytes at `offset` keeps a frame address
  // where only the analysis's own loads can read it back.
  bool Private(std::int64_t offset, std::uint32_t size) const {
    const auto end = Add(offset, size);
    return offset >= contract_.begin && end && *end <= contract_.end;
  }

  FrameAddress Node(std::size_t slot, ValueId id) const {
    return id < nodes_[slot].size() ? nodes_[slot][id] : kAny;
  }

  FrameAddress AtEnd(std::size_t slot, std::size_t phi) const {
    const auto write = ends_[slot][phi];
    return write == kNoWrite ? phis_[slot][phi] : Node(slot, write);
  }

  bool Preserved(StorageId storage) const {
    const auto& observed = graph_.observability();
    if (observed.declared) return Listed(observed.preserved, storage);
    return storage == contract_.sp_storage && contract_.callees_preserve_sp;
  }

  bool Charge(std::uint64_t work, std::uint64_t bytes) {
    return budget_.try_consume({work, bytes}) == BudgetDecline::none;
  }

 private:
  static constexpr std::uint32_t kNoRead = std::numeric_limits<std::uint32_t>::max();
  static constexpr ValueId kNoWrite = std::numeric_limits<ValueId>::max();

  FrameAddress Evaluate(const SsaBlock& block, std::size_t slot, ValueId id) const {
    const auto& node = block.nodes[id];
    const auto in = [&](unsigned input) {
      return node.inputs[input] < id ? nodes_[slot][node.inputs[input]] : kAny;
    };

    switch (node.op) {
      case Op::read:
        return reads_[slot][id] == kNoRead ? kAny : Resolve(block.reads[reads_[slot][id]].value);
      case Op::constant:
        if (node.width == 64)
          return {Kind::other, std::bit_cast<std::int64_t>(node.immediate), true};
        return kOther;
      case Op::image_address:
        return kOther;
      case Op::load:
      case Op::exclusive_load: {
        // A promoted slot's load yields its value. Any other load reads what
        // was stored where it reads, or what the fresh region held, which is
        // no frame address; only private slots ever hold one.
        const auto frame = std::lower_bound(
            block.frame_accesses.begin(), block.frame_accesses.end(), id,
            [](const SsaFrameAccess& access, ValueId value) { return access.node < value; });
        if (frame != block.frame_accesses.end() && frame->node == id && frame->replacement)
          return Resolve(*frame->replacement);
        if (contents_.empty()) return kOther;
        const auto address = in(0);
        if (address.kind == Kind::none) return {};
        if (address.kind == Kind::other) return kOther;
        if (address.kind == Kind::any || node.width % 8) return kAny;
        FrameAddress value = kOther;
        const auto end = Add(address.offset, node.width / 8);
        if (!end) return kAny;
        for (const auto& content : contents_) {
          if (content.offset >= *end) break;

          // Contents are stored exactly, so their ends do not overflow.
          if (content.offset + static_cast<std::int64_t>(content.size) <= address.offset) continue;
          value = content.offset == address.offset && content.size == node.width / 8
                      ? Join(value, content.value)
                      : kAny;
        }

        return value;
      }

      // A comparison yields a flag, never an address.
      case Op::equal:
      case Op::unsigned_less:
      case Op::signed_less:
        return kOther;
      default:
        break;
    }

    const auto* descriptor = Descriptor(node.op);
    if (!descriptor) return kAny;
    if (!descriptor->produces_value) return kOther;
    std::array<FrameAddress, 3> inputs{};
    for (unsigned input = 0; input < descriptor->arity; ++input) {
      inputs[input] = in(input);
      if (inputs[input].kind == Kind::none) return {};
    }

    const auto& a = inputs[0];
    const auto& b = inputs[1];
    if (node.width == 64 && (node.op == Op::add || node.op == Op::sub)) {
      const bool add = node.op == Op::add;
      if (a.kind == Kind::exact && b.literal) {
        const auto offset = add ? Add(a.offset, b.offset) : Sub(a.offset, b.offset);
        return offset ? FrameAddress{Kind::exact, *offset} : kAny;
      }

      if (add && b.kind == Kind::exact && a.literal) {
        const auto offset = Add(b.offset, a.offset);
        return offset ? FrameAddress{Kind::exact, *offset} : kAny;
      }

      // The distance between two frame addresses is a number.
      if (!add && a.kind == Kind::exact && b.kind == Kind::exact) return kOther;
      if (a.kind == Kind::other && b.kind == Kind::other) {
        if (!a.literal || !b.literal) return kOther;
        return {
            Kind::other,
            static_cast<std::int64_t>(
                add ? static_cast<std::uint64_t>(a.offset) + static_cast<std::uint64_t>(b.offset)
                    : static_cast<std::uint64_t>(a.offset) - static_cast<std::uint64_t>(b.offset)),
            true};
      }

      return kAny;
    }

    if (node.width == 64 && descriptor->arity == 1 &&
        (node.op == Op::zext || (node.op == Op::extract && !node.immediate)) &&
        node.inputs[0] < id && block.nodes[node.inputs[0]].width == 64)
      return a;
    for (unsigned input = 0; input < descriptor->arity; ++input)
      if (Frame(inputs[input])) return kAny;
    return kOther;
  }

  // What exact stores left in one shape of private slot, sorted by place.
  struct Content {
    std::int64_t offset;
    std::uint32_t size;
    FrameAddress value;
  };

  static bool Before(const Content& a, const Content& b) {
    return a.offset < b.offset || (a.offset == b.offset && a.size < b.size);
  }

  const SsaGraph& graph_;
  const PrivateFrameContract& contract_;
  Budget& budget_;
  std::vector<Content> contents_;
  std::vector<std::vector<FrameAddress>> nodes_, phis_, frame_phis_;
  std::vector<std::vector<std::uint32_t>> reads_;
  std::vector<std::vector<ValueId>> ends_;
};

bool ContractHolds(const PrivateFrameContract& contract) {
  return contract.begin < contract.end && contract.fresh_mapped_writable &&
         contract.no_external_aliases && contract.no_async_observers && contract.sp_alignment &&
         !(contract.sp_alignment & (contract.sp_alignment - 1));
}

auto Key(const PrivateFrameAccess& access) {
  return std::tuple{access.offset,           access.size, access.block.arena, access.block.slot,
                    access.block.generation, access.node, access.store};
}

}  // namespace

FrameScan ScanPrivateFrame(const SsaGraph& graph, const PrivateFrameContract& contract,
                           Budget& budget) {
  FrameScan scan;
  const auto decline = [&](FrameScanDecline reason) {
    FrameScan refused;
    refused.reason = reason;
    return refused;
  };

  if (!ContractHolds(contract)) return decline(FrameScanDecline::missing_contract);
  Frames frames(graph, contract, budget);
  if (!frames.Solve()) return decline(FrameScanDecline::resource_limit);
  const auto& observed = graph.observability();

  // Storage a callee or the caller may use as a pointer. Preserved storage
  // only comes back; SP is the stack a callee may not touch the frame through.
  const auto exposed = [&](StorageId storage, std::span<const StorageId> leaving) {
    if (storage == contract.sp_storage) return false;
    if (!observed.declared) return true;
    return Listed(leaving, storage) && !Listed(observed.preserved, storage);
  };

  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto& block = *graph.Get(*handle);

    // An opaque instruction the decoder classified as always trapping has no
    // effect; any other may do anything.
    if (block.opaque && (!block.nodes.empty() || !block.boundaries.empty()))
      return decline(FrameScanDecline::opaque_effect);
    if (!frames.Charge(1 + block.nodes.size() + block.edges.size() * (1 + block.phis.size()), 0))
      return decline(FrameScanDecline::resource_limit);
    for (const auto& edge : block.edges) {
      if (edge.kind == SsaEdgeKind::callee || edge.kind == SsaEdgeKind::potential_return) {
        if (!contract.callees_cannot_touch || !contract.callees_preserve_sp)
          return decline(FrameScanDecline::unknown_call);
      } else if (!edge.target_block && edge.kind != SsaEdgeKind::return_ &&
                 edge.kind != SsaEdgeKind::trap) {
        return decline(FrameScanDecline::unknown_continuation);
      }

      if (edge.target_block) continue;
      const bool call = edge.kind == SsaEdgeKind::callee;
      if (!call && edge.kind != SsaEdgeKind::return_) continue;

      // Returning with SP still inside the region would hand it to the
      // caller as live stack.
      if (!call) {
        for (std::size_t phi = 0; phi < block.phis.size(); ++phi) {
          if (block.phis[phi].storage != contract.sp_storage) continue;
          const auto sp = frames.AtEnd(slot, phi);
          if (sp.kind != Kind::exact || sp.offset < contract.end)
            return decline(FrameScanDecline::escaped_address);
        }
      }

      for (std::size_t phi = 0; phi < block.phis.size(); ++phi)
        if (exposed(block.phis[phi].storage, call ? observed.at_call : observed.at_return) &&
            Frame(frames.AtEnd(slot, phi)))
          return decline(FrameScanDecline::escaped_address);
    }

    for (ValueId id = 0; id < block.nodes.size(); ++id) {
      const auto& node = block.nodes[id];
      if (!HasMemoryOrMonitorEffect(node.op)) continue;
      const bool store = node.op == Op::store || node.op == Op::exclusive_store;
      if (store && Frame(frames.Node(slot, node.inputs[1]))) {
        const auto address = frames.Node(slot, node.inputs[0]);
        if (node.op != Op::store || address.kind != Kind::exact || node.width % 8 ||
            !frames.Private(address.offset, node.width / 8))
          return decline(FrameScanDecline::escaped_address);
      }

      const bool load = node.op == Op::load || node.op == Op::exclusive_load;
      if (!load && !store) {
        if (Descriptor(node.op) && Descriptor(node.op)->arity &&
            Frame(frames.Node(slot, node.inputs[0])))
          scan.dynamic = true;
        continue;
      }

      const auto address = frames.Node(slot, node.inputs[0]);
      if (address.kind == Kind::any || address.kind == Kind::none) {
        scan.dynamic |= address.kind == Kind::any;
        continue;
      }

      if (address.kind != Kind::exact) continue;
      const unsigned width =
          node.op == Op::exclusive_store ? block.nodes[node.inputs[1]].width : node.width;
      const auto end = Add(address.offset, width / 8);
      if (!width || width % 8 || !end) {
        scan.dynamic = true;
        continue;
      }

      if (*end <= contract.begin || address.offset >= contract.end) continue;
      if (!frames.Charge(1, sizeof(PrivateFrameAccess)))
        return decline(FrameScanDecline::resource_limit);
      if (address.offset < contract.begin || *end > contract.end) {
        scan.blocked.push_back({address.offset, *end});
        continue;
      }

      scan.exact.push_back({*handle, id, address.offset, width / 8, store});
    }

    std::size_t rewrite_index = 0;
    for (std::size_t index = 0; index < block.boundaries.size(); ++index) {
      const auto& boundary = block.boundaries[index];
      const Transfer* transfer = boundary.transfer ? &*boundary.transfer : nullptr;
      if (rewrite_index < block.control_rewrites.size() &&
          block.control_rewrites[rewrite_index].boundary == index)
        transfer = &block.control_rewrites[rewrite_index++].replacement;
      if (!transfer) continue;
      if (transfer->kind == TransferKind::call &&
          (!contract.callees_cannot_touch || !contract.callees_preserve_sp))
        return decline(FrameScanDecline::unknown_call);
      // A recovered transfer may name destination nodes past the block's own,
      // which build its targets from the block's values.
      const auto frame_operand = [&](auto&& self, ValueId id, unsigned depth) -> bool {
        if (id < block.nodes.size()) return Frame(frames.Node(slot, id));
        const auto index = static_cast<std::size_t>(id) - block.nodes.size();
        if (depth > 16 || index >= block.destination_nodes.size()) return true;
        const auto& node = block.destination_nodes[index];
        const auto* descriptor = Descriptor(node.op);
        if (!descriptor) return true;
        for (unsigned input = 0; input < descriptor->arity; ++input)
          if (self(self, node.inputs[input], depth + 1)) return true;
        return false;
      };

      if (!frames.Charge(1 + block.destination_nodes.size(), 0))
        return decline(FrameScanDecline::resource_limit);
      if (frame_operand(frame_operand, transfer->target, 0) ||
          (transfer->condition && frame_operand(frame_operand, *transfer->condition, 0)) ||
          (transfer->alternative && frame_operand(frame_operand, *transfer->alternative, 0)))
        return decline(FrameScanDecline::escaped_address);
    }
  }

  return scan;
}

std::optional<std::vector<PrivateFrameSlot>> PrivateFrameSlots(const FrameScan& scan,
                                                               const SsaGraph& graph,
                                                               const PrivateFrameContract& contract,
                                                               Budget& budget) {
  std::vector<PrivateFrameSlot> slots;
  if (scan.reason != FrameScanDecline::none || scan.dynamic) return slots;
  auto accesses = scan.exact;
  const auto depth = std::bit_width(accesses.size());
  if (budget.try_consume({accesses.size() * (depth + 2) + scan.blocked.size(), 0}) !=
      BudgetDecline::none)
    return std::nullopt;
  std::sort(accesses.begin(), accesses.end(),
            [](const auto& a, const auto& b) { return Key(a) < Key(b); });
  // Group by shape; a group any other group or blocked range overlaps is out.
  std::vector<std::pair<std::size_t, std::size_t>> groups;
  for (std::size_t at = 0; at < accesses.size();) {
    auto next = at;
    while (next < accesses.size() && accesses[next].offset == accesses[at].offset &&
           accesses[next].size == accesses[at].size)
      ++next;
    groups.push_back({at, next});
    at = next;
  }

  const auto overlaps = [](std::int64_t a, std::int64_t a_end, std::int64_t b, std::int64_t b_end) {
    return a < b_end && b < a_end;
  };

  std::int64_t reach = std::numeric_limits<std::int64_t>::min();
  for (std::size_t group = 0; group < groups.size(); ++group) {
    const auto& first = accesses[groups[group].first];
    const auto end = first.offset + static_cast<std::int64_t>(first.size);
    bool owned = first.offset >= reach;
    if (group + 1 < groups.size() && accesses[groups[group + 1].first].offset < end) owned = false;
    reach = std::max(reach, end);
    for (const auto& [begin, stop] : scan.blocked)
      owned &= !overlaps(first.offset, end, begin, stop);
    for (auto at = groups[group].first; owned && at < groups[group].second; ++at) {
      const auto& access = accesses[at];
      const auto* block = graph.Get(access.block);
      const auto& node = block->nodes[access.node];
      owned = node.op == (access.store ? Op::store : Op::load) && node.access.alignment &&
              node.access.alignment <= contract.sp_alignment &&
              access.offset % static_cast<std::int64_t>(node.access.alignment) == 0;
    }

    if (!owned) continue;
    if (budget.try_consume(
            {groups[group].second - groups[group].first,
             sizeof(PrivateFrameSlot) + (groups[group].second - groups[group].first) *
                                            sizeof(PrivateFrameAccess)}) != BudgetDecline::none)
      return std::nullopt;
    slots.push_back({first.offset,
                     first.size,
                     {accesses.begin() + static_cast<std::ptrdiff_t>(groups[group].first),
                      accesses.begin() + static_cast<std::ptrdiff_t>(groups[group].second)}});
  }

  return slots;
}

PrivateFrameFactDecline ValidatePrivateFrameFacts(const PrivateFrameFacts& facts,
                                                  const SsaGraph& graph, Budget& budget) {
  const auto invalid = PrivateFrameFactDecline::invalid;
  const auto limited = PrivateFrameFactDecline::resource_limit;
  if (facts.arena != graph.arena() || facts.revision != graph.revision() ||
      !ContractHolds(facts.contract) || facts.entry_relations || facts.relation_declared_abi ||
      facts.relation_return_leaves || facts.relation_constant_image ||
      facts.relation_declared_opaque_control)
    return invalid;
  const auto validated = ValidateSsa(graph, budget);
  if (validated != SsaDecline::none)
    return validated == SsaDecline::resource_limit ? limited : invalid;
  const auto scan = ScanPrivateFrame(graph, facts.contract, budget);
  if (scan.reason == FrameScanDecline::resource_limit) return limited;
  if (scan.reason != FrameScanDecline::none) return invalid;
  const auto proved = PrivateFrameSlots(scan, graph, facts.contract, budget);
  if (!proved) return limited;

  // Each claimed slot must be one the scan proves, with every access it has.
  std::size_t at = 0;
  for (const auto& slot : facts.slots) {
    if (budget.try_consume({1 + slot.accesses.size(), 0}) != BudgetDecline::none) return limited;
    while (at < proved->size() && (*proved)[at].offset < slot.offset) ++at;
    if (at == proved->size() || (*proved)[at].offset != slot.offset ||
        (*proved)[at].size != slot.size || (*proved)[at].accesses.size() != slot.accesses.size())
      return invalid;
    for (std::size_t i = 0; i < slot.accesses.size(); ++i)
      if (Key(slot.accesses[i]) != Key((*proved)[at].accesses[i])) return invalid;
    ++at;
  }

  return PrivateFrameFactDecline::none;
}

}  // namespace nyx::ir
