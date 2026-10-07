#include "nyx/ir/ssa/graph.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdio>
#include <functional>
#include <limits>
#include <utility>

#include "nyx/ir/fold.hpp"

namespace nyx::ir {

bool SsaEffectFree(const SsaBlock& block, ValueId id) {
  if (id >= block.nodes.size()) return false;
  const auto* descriptor = Descriptor(block.nodes[id].op);
  if (!descriptor) return false;
  if (descriptor->effect == Effect::pure) return true;
  if (block.nodes[id].op != Op::load) return false;
  const auto fold = std::lower_bound(
      block.constant_loads.begin(), block.constant_loads.end(), id,
      [](const SsaConstantLoad& item, ValueId value) { return item.node < value; });
  return fold != block.constant_loads.end() && fold->node == id && fold->skip_access;
}

bool SsaNeedsInputs(const SsaBlock& block, ValueId id) {
  if (id >= block.nodes.size()) return false;
  if (block.nodes[id].op != Op::load || !SsaEffectFree(block, id)) return true;
  return false;
}

std::optional<ValueId> SsaFoldCondition(const SsaBlock& block, ValueId id) {
  if (id >= block.nodes.size() || block.nodes[id].op != Op::load) return {};
  const auto fold = std::lower_bound(
      block.constant_loads.begin(), block.constant_loads.end(), id,
      [](const SsaConstantLoad& item, ValueId value) { return item.node < value; });
  return fold != block.constant_loads.end() && fold->node == id && fold->skip_access
             ? fold->condition
             : std::nullopt;
}

std::optional<ValueId> SsaFoldIndex(const SsaBlock& block, ValueId id) {
  if (id >= block.nodes.size() || block.nodes[id].op != Op::load) return {};
  const auto fold = std::lower_bound(
      block.constant_loads.begin(), block.constant_loads.end(), id,
      [](const SsaConstantLoad& item, ValueId value) { return item.node < value; });
  return fold != block.constant_loads.end() && fold->node == id &&
                 fold->kind == SsaConstantKind::bounded_table
             ? std::optional<ValueId>{fold->table_index}
             : std::nullopt;
}

namespace {
std::atomic<std::uint64_t> next_arena{1};

bool ValidNode(const SsaBlock& block, ValueId id) {
  return id < block.nodes.size() && Descriptor(block.nodes[id].op) &&
         Descriptor(block.nodes[id].op)->produces_value;
}

bool Shape(const SsaBlock& block, ValueId id) {
  const auto& node = block.nodes[id];
  const auto* descriptor = Descriptor(node.op);
  if (!descriptor || !node.width || node.width > 4096) return false;
  for (unsigned i = 0; i < descriptor->arity; ++i) {
    if (node.inputs[i] >= id || !ValidNode(block, node.inputs[i])) return false;
  }

  if (node.op == Op::image_address) return node.width == 64;
  if (node.op == Op::exclusive_clear) return node.width == 1;
  if (!descriptor->arity) return true;
  if (NarrowOnly(node.op) && node.width > 64) return false;
  const auto width = [&](unsigned input) { return block.nodes[node.inputs[input]].width; };
  const auto a = width(0);
  const auto b = descriptor->arity > 1 ? width(1) : 0;
  switch (node.op) {
    case Op::load:
    case Op::store:
      return a == 64 && node.width % 8 == 0 && node.width <= 128 &&
             (node.op != Op::store || b == node.width) && node.access.alignment &&
             !(node.access.alignment & (node.access.alignment - 1));
    case Op::exclusive_load:
    case Op::exclusive_store:
      return a == 64 &&
             (node.op == Op::exclusive_load ? (node.width == 32 || node.width == 64)
                                            : (node.width == 32 && (b == 32 || b == 64))) &&
             node.access.alignment == (node.op == Op::exclusive_load ? node.width : b) / 8;
    case Op::extract:
      return node.immediate < a && node.width <= a - node.immediate;
    case Op::zext:
      return node.width >= a;
    case Op::select:
      return a == 1 && b == node.width && width(2) == node.width;
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

bool ValidValue(const SsaGraph& graph, SsaValue value, unsigned width) {
  const auto* block = graph.Get(value.block);
  if (!block) return false;
  switch (value.kind) {
    case SsaValueKind::input:
      return value.index < block->phis.size() && block->phis[value.index].external_entry &&
             block->phis[value.index].width == width;
    case SsaValueKind::phi:
      return value.index < block->phis.size() && block->phis[value.index].width == width;
    case SsaValueKind::node:
      return ValidNode(*block, value.index) && block->nodes[value.index].width == width;
    case SsaValueKind::clobber:
      return value.index < block->phis.size() && value.index < block->clobbers.size() &&
             block->clobbers[value.index] && block->phis[value.index].width == width;
    case SsaValueKind::frame_phi:
      return value.index < block->frame_phis.size() &&
             block->frame_phis[value.index].size * 8 == width;
    case SsaValueKind::frame_input:
      return value.index < block->frame_phis.size() &&
             block->frame_phis[value.index].external_entry &&
             block->frame_phis[value.index].size * 8 == width;
  }

  return false;
}

std::optional<SsaValue> ExpectedPredecessorCopy(const SsaGraph& graph, SsaHandle handle,
                                                const SsaRead& read) {
  const auto* block = graph.Get(handle);
  if (!block || read.phi >= block->phis.size() || read.value.kind != SsaValueKind::phi ||
      read.value.block != handle || read.value.index != read.phi)
    return {};
  const auto& phi = block->phis[read.phi];
  if (phi.external_entry || phi.incoming.size() != 1) return {};
  const auto& input = phi.incoming[0];
  const auto* predecessor = graph.Get(input.predecessor);
  const auto* source_descriptor = predecessor && input.value.index < predecessor->nodes.size()
                                      ? Descriptor(predecessor->nodes[input.value.index].op)
                                      : nullptr;
  if (!predecessor || predecessor->opaque || input.predecessor == handle ||
      input.value.kind != SsaValueKind::node || input.value.block != input.predecessor ||
      !source_descriptor || !source_descriptor->produces_value ||
      std::binary_search(predecessor->dead_pure_nodes.begin(), predecessor->dead_pure_nodes.end(),
                         input.value.index))
    return {};
  return input.value;
}

bool ChargeCount(std::uint64_t& total, std::size_t count) {
  if (count > std::numeric_limits<std::uint64_t>::max() - total) return false;
  total += count;
  return true;
}

bool ChargeBytes(std::uint64_t& total, std::size_t count, std::size_t size) {
  if (count > (std::numeric_limits<std::uint64_t>::max() - total) / size) return false;
  total += count * size;
  return true;
}

bool BlockCopyCost(const SsaBlock& block, std::uint64_t& work, std::uint64_t& bytes) {
  if (!ChargeBytes(bytes, block.nodes.size(), sizeof(Node)) ||
      !ChargeBytes(bytes, block.boundaries.size(), sizeof(Boundary)) ||
      !ChargeBytes(bytes, block.source_groups.size(), sizeof(std::uint64_t)) ||
      !ChargeBytes(bytes, block.source_bytes.size(), sizeof(std::vector<std::uint8_t>)) ||
      !ChargeBytes(bytes, block.original_sources.size(), sizeof(std::uint32_t)) ||
      !ChargeBytes(bytes, block.entry_relations.size(), sizeof(StorageRelation)) ||
      !ChargeBytes(bytes, block.control_rewrites.size(), sizeof(ConditionalRewrite)) ||
      !ChargeBytes(bytes, block.store_omissions.size(), sizeof(StoreOmission)) ||
      !ChargeBytes(bytes, block.paired_load_omissions.size(), sizeof(PairedLoadOmission)) ||
      !ChargeBytes(bytes, block.destination_nodes.size(), sizeof(Node)) ||
      !ChargeBytes(bytes, block.phis.size(), sizeof(SsaPhi)) ||
      !ChargeBytes(bytes, block.clobbers.size(), sizeof(std::uint8_t)) ||
      !ChargeBytes(bytes, block.reads.size(), sizeof(SsaRead)) ||
      !ChargeBytes(bytes, block.dead_storage_writes.size(), sizeof(SsaDeadStorageWrite)) ||
      !ChargeBytes(bytes, block.exits.size(), sizeof(SsaExitValue)) ||
      !ChargeBytes(bytes, block.edges.size(), sizeof(SsaEdge)) ||
      !ChargeBytes(bytes, block.frame_phis.size(), sizeof(SsaFramePhi)) ||
      !ChargeBytes(bytes, block.frame_accesses.size(), sizeof(SsaFrameAccess)) ||
      !ChargeBytes(bytes, block.frame_exits.size(), sizeof(SsaValue)) ||
      !ChargeBytes(bytes, block.constant_loads.size(), sizeof(SsaConstantLoad)) ||
      !ChargeBytes(bytes, block.path_reads.size(), sizeof(SsaPathRead)) ||
      !ChargeBytes(bytes, block.retired_loads.size(), sizeof(SsaRetiredLoad)) ||
      !ChargeBytes(bytes, block.disabled_effects.size(), sizeof(ValueId)) ||
      !ChargeBytes(bytes, block.dead_pure_nodes.size(), sizeof(ValueId)) ||
      !ChargeCount(work, block.entry_relations.size()) || !ChargeCount(work, block.nodes.size()) ||
      !ChargeCount(work, block.phis.size()) || !ChargeCount(work, block.frame_phis.size()) ||
      !ChargeCount(work, block.edges.size()))
    return false;
  for (const auto& boundary : block.boundaries)
    if (!ChargeBytes(bytes, boundary.writes.size(), sizeof(Write))) return false;
  for (const auto& source : block.source_bytes)
    if (!ChargeBytes(bytes, source.size(), sizeof(std::uint8_t))) return false;
  for (const auto& rewrite : block.control_rewrites)
    if (!ChargeBytes(bytes, rewrite.witness.size(), sizeof(ImageRead))) return false;
  for (const auto& fold : block.constant_loads) {
    if (!ChargeBytes(bytes, fold.table_bytes.size(), sizeof(std::array<std::uint8_t, 8>)) ||
        !ChargeCount(work, fold.table_bytes.size()))
      return false;
  }

  for (const auto& phi : block.phis)
    if (!ChargeBytes(bytes, phi.incoming.size(), sizeof(SsaPhiInput)) ||
        !ChargeCount(work, phi.incoming.size()))
      return false;
  for (const auto& phi : block.frame_phis)
    if (!ChargeBytes(bytes, phi.incoming.size(), sizeof(SsaPhiInput)) ||
        !ChargeCount(work, phi.incoming.size()))
      return false;
  return true;
}

SsaDecline RootsAndInputsLive(const SsaBlock& block, std::span<const std::uint8_t> live,
                              Budget& budget) {
  if (live.size() != block.nodes.size()) return SsaDecline::invalid_graph;
  const auto node_work = 2 + 4 * std::bit_width(block.constant_loads.size());
  if (block.nodes.size() > std::numeric_limits<std::size_t>::max() / node_work ||
      budget.try_consume({block.nodes.size() * node_work, 0}) != BudgetDecline::none)
    return SsaDecline::resource_limit;
  const auto present = [&](ValueId id) { return id >= block.nodes.size() || live[id] == 1; };
  for (std::size_t id = 0; id < block.nodes.size(); ++id) {
    if (live[id] > 1) return SsaDecline::invalid_graph;
    const auto* descriptor = Descriptor(block.nodes[id].op);
    if (!descriptor || (!SsaEffectFree(block, static_cast<ValueId>(id)) && !live[id]))
      return SsaDecline::invalid_graph;
    if (!live[id]) continue;
    if (const auto condition = SsaFoldCondition(block, static_cast<ValueId>(id));
        condition && !present(*condition))
      return SsaDecline::invalid_graph;
    if (const auto index = SsaFoldIndex(block, static_cast<ValueId>(id)); index && !present(*index))
      return SsaDecline::invalid_graph;
    if (!SsaNeedsInputs(block, static_cast<ValueId>(id))) continue;
    for (unsigned input = 0; input < descriptor->arity; ++input)
      if (!present(block.nodes[id].inputs[input])) return SsaDecline::invalid_graph;
  }

  if (budget.try_consume({block.control_rewrites.size(), 0}) != BudgetDecline::none ||
      budget.try_consume({block.edges.size(), 0}) != BudgetDecline::none)
    return SsaDecline::resource_limit;
  std::size_t rewrite_index = 0;
  for (std::size_t index = 0; index < block.boundaries.size(); ++index) {
    const auto& boundary = block.boundaries[index];
    if (budget.try_consume({1 + boundary.writes.size(), 0}) != BudgetDecline::none)
      return SsaDecline::resource_limit;
    for (std::size_t write_index = 0; write_index < boundary.writes.size(); ++write_index) {
      if (budget.try_consume({std::uint64_t{0} + std::bit_width(block.dead_storage_writes.size()),
                              0}) != BudgetDecline::none)
        return SsaDecline::resource_limit;
      const auto retired = std::lower_bound(
          block.dead_storage_writes.begin(), block.dead_storage_writes.end(),
          std::pair{index, write_index}, [](const SsaDeadStorageWrite& item, const auto& key) {
            return item.boundary < key.first ||
                   (item.boundary == key.first && item.index < key.second);
          });
      if ((retired == block.dead_storage_writes.end() || retired->boundary != index ||
           retired->index != write_index) &&
          !present(boundary.writes[write_index].value))
        return SsaDecline::invalid_graph;
    }

    const Transfer* transfer = boundary.transfer ? &*boundary.transfer : nullptr;
    if (rewrite_index < block.control_rewrites.size() &&
        block.control_rewrites[rewrite_index].boundary == index)
      transfer = &block.control_rewrites[rewrite_index++].replacement;
    if (!transfer) continue;
    if (!present(transfer->target) || (transfer->condition && !present(*transfer->condition)) ||
        (transfer->alternative && !present(*transfer->alternative)) ||
        (transfer->continuation && !present(*transfer->continuation)))
      return SsaDecline::invalid_graph;
  }

  for (const auto& edge : block.edges)
    if (edge.condition && !present(*edge.condition)) return SsaDecline::invalid_graph;
  return SsaDecline::none;
}

bool ImageOverlap(std::uint64_t a, unsigned asize, std::uint64_t b, unsigned bsize) {
  if (a > UINT64_MAX - asize || b > UINT64_MAX - bsize) return true;
  return a < b + bsize && b < a + asize;
}

}  // namespace

std::optional<SsaImageStores> SsaImageStores::Collect(const SsaGraph& graph,
                                                      bool page_aligned_placement, Budget& budget,
                                                      const ImageFacts* facts) {
  SsaImageStores result;
  const auto unknown = [&] {
    result.unknown_ = true;
    result.stores_.clear();
    result.placed_.clear();
    return result;
  };

  // `lanes` is a bitwise function of one image location `bits`, kept as its
  // value with the location's unknown bits all clear (`low`) and all set
  // (`high`). Placed on a page, a location's in-page bits are exact and each
  // bit of a bitwise function depends only on the same bit of its inputs, so
  // equal lanes prove a number that no load bias changes.
  enum class Kind { other, number, image, lanes, derived };

  struct Address {
    Kind kind = Kind::other;
    std::uint64_t bits = 0, low = 0, high = 0;

    // The declared read this value was computed from, one past its index in
    // `relied`; kMany for more than one.
    std::uint32_t rests = 0;
  };

  const auto image = [](Address value) {
    return value.kind == Kind::image || value.kind == Kind::lanes || value.kind == Kind::derived;
  };

  const auto bias = graph.load_bias();

  // Under a bias a location and the number it sits at are one value, so an
  // address means the same whichever way it was spelled. Below the bias a
  // number is no place in the image.
  const auto location = [&](Address value) -> std::optional<std::uint64_t> {
    if (value.kind == Kind::image) return value.bits;
    if (value.kind == Kind::number && bias && value.bits >= *bias) return value.bits - *bias;
    return std::nullopt;
  };

  const auto runtime = [&](Address value) -> std::optional<std::uint64_t> {
    if (value.kind == Kind::number) return value.bits;
    if (value.kind == Kind::image && bias) return value.bits + *bias;
    return std::nullopt;
  };

  constexpr std::uint64_t kInPage = 0xfff;
  const auto lanes = [&](Address value) -> std::optional<Address> {
    if (value.kind == Kind::number)
      return Address{Kind::number, value.bits, value.bits, value.bits};
    if (value.kind == Kind::lanes) return value;
    if (value.kind == Kind::image && page_aligned_placement)
      return Address{Kind::lanes, value.bits, value.bits & kInPage,
                     ~kInPage | (value.bits & kInPage)};
    return std::nullopt;
  };

  const auto apply = [](Op op, std::uint64_t a, std::uint64_t b) {
    switch (op) {
      case Op::add:
        return a + b;
      case Op::sub:
        return a - b;
      case Op::bit_and:
        return a & b;
      case Op::bit_or:
        return a | b;
      case Op::bit_xor:
        return a ^ b;
      default:
        return ~a;
    }
  };

  const auto bitwise = [&](const Node& node,
                           std::span<const Address> in) -> std::optional<Address> {
    std::array<Address, 2> operand{};
    std::optional<std::uint64_t> leaf;
    for (std::size_t i = 0; i < in.size(); ++i) {
      const auto value = lanes(in[i]);
      if (!value) return std::nullopt;
      if (value->kind == Kind::lanes) {
        if (leaf && *leaf != value->bits) return std::nullopt;
        leaf = value->bits;
      }

      operand[i] = *value;
    }

    if (!leaf) return std::nullopt;
    const auto low = apply(node.op, operand[0].low, operand[1].low);
    const auto high = apply(node.op, operand[0].high, operand[1].high);
    if (low == high) return Address{Kind::number, low, low, low};
    return Address{Kind::lanes, *leaf, low, high};
  };

  // The declared image bytes a known load value rests on. A store the graph
  // makes to any of them leaves the value unknown, which matters only for
  // those a store's placed address was computed from (`placing`).
  struct Relied {
    std::uint64_t address;
    unsigned size;
    bool placing = false;
  };

  constexpr std::uint32_t kMany = UINT32_MAX;
  std::vector<Relied> relied;
  bool placed_by_many = false;
  const auto rely = [&](Kind kind, std::uint64_t value,
                        std::uint64_t address) -> std::optional<Address> {
    if (relied.size() >= kMany - 1 ||
        budget.try_consume({1, sizeof(Relied)}) != BudgetDecline::none)
      return std::nullopt;
    relied.push_back({address, 8});
    return Address{kind, value, 0, 0, static_cast<std::uint32_t>(relied.size())};
  };

  // What a 64-bit load yields when what it reads is declared. A fold that
  // yields one of several values leaves a store through it unplaced where any
  // of them is a location, and a runtime value otherwise.
  const auto loaded = [&](const SsaBlock& block, ValueId id,
                          Address address) -> std::optional<Address> {
    const auto& node = block.nodes[id];
    if (node.width != 64) return Address{};
    if (budget.try_consume(
            {2 + static_cast<std::uint64_t>(std::bit_width(block.constant_loads.size()) +
                                            std::bit_width(block.path_reads.size())),
             0}) != BudgetDecline::none)
      return std::nullopt;
    const auto fold =
        std::lower_bound(block.constant_loads.begin(), block.constant_loads.end(), id,
                         [](const SsaConstantLoad& item, ValueId key) { return item.node < key; });
    if (fold != block.constant_loads.end() && fold->node == id) {
      if (fold->kind == SsaConstantKind::image_location) {
        return rely(Kind::image, fold->value, fold->source_address);
      }

      if (fold->kind == SsaConstantKind::literal && !fold->condition) {
        return rely(Kind::number, fold->value, fold->source_address);
      }

      const auto placed = [&](std::uint64_t value) { return bias && value >= *bias; };
      bool any = fold->condition && (placed(fold->value) || placed(fold->alternative_value));
      if (budget.try_consume({fold->table_bytes.size(), 0}) != BudgetDecline::none)
        return std::nullopt;
      for (const auto& row : fold->table_bytes) {
        std::uint64_t value = 0;
        for (unsigned byte = 0; byte < 8; ++byte)
          value |= std::uint64_t{row[byte]}
                   << (8 * (node.access.byte_order == ByteOrder::little ? byte : 7 - byte));
        any |= placed(value);
      }

      return Address{any ? Kind::derived : Kind::other};
    }

    const auto read =
        std::lower_bound(block.path_reads.begin(), block.path_reads.end(), id,
                         [](const SsaPathRead& item, ValueId key) { return item.node < key; });
    if (read != block.path_reads.end() && read->node == id) {
      return rely(read->relocated ? Kind::image : Kind::number, read->value, read->address);
    }

    const auto at = location(address);
    if (!facts || !at) return Address{};
    if (budget.try_consume({2 * facts->constants.size() + facts->pointers.size() + 8, 0}) !=
        BudgetDecline::none)
      return std::nullopt;
    if (const auto target = ReadRelocated(*facts, *at, 64);
        target && node.access.byte_order == ByteOrder::little) {
      return rely(Kind::image, *target, *at);
    }

    const auto value = ReadConstant(*facts, *at, 64, node.access.byte_order);
    if (!value) return Address{};
    return rely(Kind::number, *value, *at);
  };

  // Gathered on the first register that needs it.
  std::optional<SsaImageReach> reach;
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return {};
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto& block = *graph.Get(*handle);
    if (block.nodes.size() > SIZE_MAX / sizeof(Address) ||
        budget.try_consume({block.nodes.size(), block.nodes.size() * sizeof(Address)}) !=
            BudgetDecline::none)
      return {};
    std::vector<Address> addresses(block.nodes.size());
    for (std::size_t id = 0; id < block.nodes.size(); ++id) {
      if (budget.try_consume({1, 0}) != BudgetDecline::none) return {};
      const auto& node = block.nodes[id];
      const auto* descriptor = Descriptor(node.op);
      if (!descriptor) return unknown();
      if (node.op == Op::image_address) {
        addresses[id] = {Kind::image, node.immediate};
      } else if (node.op == Op::read && node.width == 64) {
        // A register that carries a location is one, even before the read
        // fold spells it as one. Only with every way in listed is the one
        // every listed way computes the only one it can hold; otherwise, or
        // where they disagree or a loop moves it on, it is some location the
        // graph does not name.
        // Only a register some path fills with a location can be one, so the
        // whole-graph reach set, built once, gates the per-register settle
        // walk: a read nothing image-based reaches is `other` at no cost.
        if (!reach && !(reach = SsaImageReaching(graph, budget, facts))) return {};
        if (reach->nodes[slot][id]) {
          const SsaValue value{SsaValueKind::node, *handle, static_cast<std::uint32_t>(id)};
          const auto settled =
              graph.entries_closed() ? SsaSettledLocation(graph, value, budget) : std::nullopt;
          addresses[id] = settled ? Address{Kind::image, *settled} : Address{Kind::derived, 0};
        }
      } else if (node.op == Op::constant && node.width == 64) {
        addresses[id] = {Kind::number, node.immediate};
      } else if (node.op == Op::load || node.op == Op::exclusive_load) {
        // What a load yields is memory, not arithmetic on the place it reads:
        // unless a declaration says what is there, the same pointer reaching
        // this block in a register is `other`, and a store through either is
        // the whole-image writer scan's to account for.
        if (node.inputs[0] >= id) return unknown();
        const auto value = loaded(block, static_cast<ValueId>(id), addresses[node.inputs[0]]);
        if (!value) return {};
        addresses[id] = *value;
      } else {
        bool derived = false;
        std::uint32_t rests = 0;
        for (unsigned input = 0; input < descriptor->arity; ++input) {
          if (node.inputs[input] >= id) return unknown();
          derived |= image(addresses[node.inputs[input]]);
          const auto source = addresses[node.inputs[input]].rests;
          if (source) rests = rests && rests != source ? kMany : source;
        }

        std::array<Address, 2> in{};
        for (unsigned input = 0; input < descriptor->arity && input < 2; ++input)
          in[input] = addresses[node.inputs[input]];
        const bool bitwise_op = node.op == Op::bit_and || node.op == Op::bit_or ||
                                node.op == Op::bit_xor || node.op == Op::bit_not;
        if (node.width == 64 && (bitwise_op || (descriptor->arity == 2 &&
                                                (node.op == Op::add || node.op == Op::sub)))) {
          const auto a = in[0], b = in[1];
          const auto ra = runtime(a);
          const auto rb = descriptor->arity == 2 ? runtime(b) : std::optional<std::uint64_t>(0);
          if (!bitwise_op && a.kind == Kind::image && b.kind == Kind::number)
            addresses[id] = {Kind::image, apply(node.op, a.bits, b.bits)};
          else if (node.op == Op::add && a.kind == Kind::number && b.kind == Kind::image)
            addresses[id] = {Kind::image, a.bits + b.bits};
          else if (ra && rb)
            addresses[id] = {Kind::number, apply(node.op, *ra, *rb)};
          else if (const auto value = bitwise_op
                                          ? bitwise(node, std::span(in).first(descriptor->arity))
                                          : std::nullopt)
            addresses[id] = *value;
          else if (derived)
            addresses[id] = {Kind::derived, 0};
        } else if (node.width == 64 && block.nodes[node.inputs[0]].width == 64 &&
                   (node.op == Op::zext || (node.op == Op::extract && !node.immediate))) {
          addresses[id] = addresses[node.inputs[0]];
        } else if (derived)
          addresses[id] = {Kind::derived, 0};
        addresses[id].rests = rests;
      }

      if (!MayWriteMemory(node.op)) continue;
      if (node.inputs[0] >= id) return unknown();
      const auto base = addresses[node.inputs[0]];
      const auto at = location(base);
      if (!at && base.kind != Kind::derived && base.kind != Kind::lanes) continue;
      if (node.op == Op::exclusive_store && node.inputs[1] >= id) return unknown();
      const auto width =
          node.op == Op::exclusive_store ? block.nodes[node.inputs[1]].width : node.width;
      if (width % 8) return unknown();
      if (!at) {
        result.unplaced_ = true;
        continue;
      }

      if (budget.try_consume({1, sizeof(result.stores_.front()) + sizeof(Placed)}) !=
          BudgetDecline::none)
        return {};
      if (base.rests == kMany)
        placed_by_many = true;
      else if (base.rests)
        relied[base.rests - 1].placing = true;
      result.stores_.push_back({*at, width / 8});
      result.placed_.push_back({*at, width / 8, block.address, static_cast<ValueId>(id)});
      result.widest_ = std::max(result.widest_, width / 8);
    }
  }

  if (budget.try_consume({result.stores_.size() * (1 + std::bit_width(result.stores_.size())),
                          0}) != BudgetDecline::none)
    return {};
  std::sort(result.stores_.begin(), result.stores_.end());
  if (budget.try_consume({relied.size() * (1 + std::bit_width(result.stores_.size())), 0}) !=
      BudgetDecline::none)
    return {};
  // Only a store it places can contradict the declaration a read rests on:
  // one nobody can place is the writer scan's, as for the declared value itself.
  for (const auto& read : relied)
    if ((placed_by_many || read.placing) && result.ConflictsPlaced(read.address, read.size))
      return unknown();
  return result;
}

bool SsaImageStores::Conflicts(std::uint64_t address, unsigned size, bool read_only) const {
  if (ConflictsPlaced(address, size)) return true;

  // A store placed on read-only bytes is the graph contradicting the
  // declaration, whatever the loader would do; one nobody can place might
  // equally be aimed at writable bytes, and would fault on these.
  return unplaced_ && !read_only;
}

bool SsaImageStores::ConflictsPlaced(std::uint64_t address, unsigned size) const {
  if (unknown_) return true;

  // Stores are at most `widest_` bytes, so only those starting from that far
  // below the read can reach it; a wrapped start is checked from zero.
  const auto from = address >= widest_ ? address - widest_ : 0;
  for (auto at = std::lower_bound(stores_.begin(), stores_.end(), std::pair{from, 0U});
       at != stores_.end() && (at->first <= address || at->first - address < size); ++at)
    if (ImageOverlap(address, size, at->first, at->second)) return true;
  return false;
}

std::optional<std::vector<SsaDroppedPathRead>> SsaDropWrittenPathReads(SsaGraph& graph,
                                                                       Budget& budget,
                                                                       const ImageFacts* facts) {
  std::array<std::optional<SsaImageStores>, 2> stores;
  std::vector<SsaDroppedPathRead> dropped;
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return std::nullopt;
    const auto handle = graph.Handle(slot);
    if (!handle || graph.Get(*handle)->path_reads.empty()) continue;
    const auto& block = *graph.Get(*handle);
    std::vector<SsaPathRead> kept;
    if (budget.try_consume({block.path_reads.size(),
                            block.path_reads.size() * sizeof(SsaPathRead)}) != BudgetDecline::none)
      return std::nullopt;
    for (const auto& read : block.path_reads) {
      auto& placed = stores[read.page_aligned_placement];
      if (!placed &&
          !(placed = SsaImageStores::Collect(graph, read.page_aligned_placement, budget, facts)))
        return std::nullopt;
      if (read.node >= block.nodes.size() ||
          !placed->ConflictsPlaced(read.address, block.nodes[read.node].width / 8)) {
        kept.push_back(read);
      } else {
        if (budget.try_consume({1, sizeof(SsaDroppedPathRead)}) != BudgetDecline::none)
          return std::nullopt;
        dropped.push_back({*handle, read.node, read.address});
      }
    }

    if (kept.size() == block.path_reads.size()) continue;
    auto copied = graph.CopyBlock(*handle, budget);
    if (!copied) return std::nullopt;
    copied->path_reads = std::move(kept);
    if (!graph.Replace(*handle, std::move(*copied))) return std::nullopt;
  }

  return dropped;
}

std::optional<bool> SsaConflictingImageStore(const SsaGraph& graph, std::uint64_t address,
                                             unsigned size, bool page_aligned_placement,
                                             Budget& budget, bool read_only,
                                             const ImageFacts* facts) {
  const auto stores = SsaImageStores::Collect(graph, page_aligned_placement, budget, facts);
  if (!stores) return std::nullopt;
  return stores->Conflicts(address, size, read_only);
}

SsaGraph::SsaGraph() : arena_(next_arena.fetch_add(1, std::memory_order_relaxed)) {}

SsaGraph::SsaGraph(SsaGraph&& other) noexcept
    : arena_(std::exchange(other.arena_, next_arena.fetch_add(1, std::memory_order_relaxed))),
      revision_(std::exchange(other.revision_, 0)),
      slots_(std::move(other.slots_)),
      entries_(std::move(other.entries_)),
      observability_(std::move(other.observability_)),
      call_results_(std::move(other.call_results_)),
      callee_bodies_(std::move(other.callee_bodies_)),
      load_bias_(std::exchange(other.load_bias_, std::nullopt)),
      entries_closed_(std::exchange(other.entries_closed_, false)) {
  other.slots_.clear();
  other.entries_.clear();
  other.observability_ = {};
  other.call_results_.clear();
  other.callee_bodies_.clear();
}

SsaGraph& SsaGraph::operator=(SsaGraph&& other) noexcept {
  if (this == &other) return *this;
  arena_ = std::exchange(other.arena_, next_arena.fetch_add(1, std::memory_order_relaxed));
  revision_ = std::exchange(other.revision_, 0);
  slots_ = std::move(other.slots_);
  entries_ = std::move(other.entries_);
  observability_ = std::move(other.observability_);
  call_results_ = std::move(other.call_results_);
  callee_bodies_ = std::move(other.callee_bodies_);
  load_bias_ = std::exchange(other.load_bias_, std::nullopt);
  entries_closed_ = std::exchange(other.entries_closed_, false);
  other.slots_.clear();
  other.entries_.clear();
  other.observability_ = {};
  other.call_results_.clear();
  other.callee_bodies_.clear();
  return *this;
}

SsaHandle SsaGraph::Add(SsaBlock block) {
  const auto slot = static_cast<std::uint32_t>(slots_.size());
  slots_.push_back({std::move(block), 0});
  ++revision_;
  return {arena_, slot, 0};
}

const SsaBlock* SsaGraph::Get(SsaHandle handle) const {
  if (handle.arena != arena_ || handle.slot >= slots_.size()) return nullptr;
  const auto& slot = slots_[handle.slot];
  return slot.generation == handle.generation && slot.block ? &*slot.block : nullptr;
}

std::optional<SsaBlock> SsaGraph::CopyBlock(SsaHandle handle, Budget& budget) const {
  const auto* block = Get(handle);
  if (!block) return {};
  std::uint64_t work = 0, bytes = sizeof(SsaBlock);
  if (!BlockCopyCost(*block, work, bytes) ||
      budget.try_consume({work, bytes}) != BudgetDecline::none)
    return {};
  return *block;
}

bool SsaGraph::Replace(SsaHandle handle, SsaBlock block) {
  const auto* original = Get(handle);
  if (!original || revision_ == std::numeric_limits<std::uint64_t>::max() ||
      block.entry_relations != original->entry_relations ||
      block.relation_declared_abi != original->relation_declared_abi ||
      block.relation_return_leaves != original->relation_return_leaves ||
      block.relation_constant_image != original->relation_constant_image ||
      block.relation_declared_opaque_control != original->relation_declared_opaque_control)
    return false;
  slots_[handle.slot].block = std::move(block);
  ++revision_;
  return true;
}

std::optional<SsaHandle> SsaGraph::Handle(std::size_t slot) const {
  if (slot >= slots_.size() || !slots_[slot].block) return {};
  return SsaHandle{arena_, static_cast<std::uint32_t>(slot), slots_[slot].generation};
}

std::optional<SsaGraph> SsaGraph::Clone(Budget& budget) const {
  std::uint64_t bytes = 0;
  std::uint64_t work = 0;
  if (!ChargeBytes(bytes, slots_.size(), sizeof(Slot)) ||
      !ChargeBytes(bytes, entries_.size(), sizeof(SsaHandle)) ||
      !ChargeBytes(bytes,
                   observability_.at_call.size() + observability_.at_return.size() +
                       observability_.preserved.size(),
                   sizeof(StorageId)) ||
      !ChargeBytes(bytes, call_results_.size(), sizeof(SsaCallResult)) ||
      !ChargeCount(work, slots_.size()))
    return {};
  for (const auto& body : callee_bodies_) {
    if (!ChargeBytes(bytes, body.groups.size(), sizeof(Group)) ||
        !ChargeCount(work, body.groups.size()))
      return {};
    for (const auto& group : body.groups)
      if (!ChargeBytes(bytes, group.nodes().size(), sizeof(Node)) ||
          !ChargeBytes(bytes, group.writes().size(), sizeof(Write)) ||
          !ChargeBytes(bytes, group.bytes().size(), 1))
        return {};
  }

  for (const auto& slot : slots_) {
    if (!slot.block) continue;
    const auto& block = *slot.block;
    if (!BlockCopyCost(block, work, bytes)) return {};
  }

  if (budget.try_consume({work, bytes}) != BudgetDecline::none) return {};
  SsaGraph copy;
  copy.slots_ = slots_;
  copy.revision_ = revision_;
  copy.entries_ = entries_;
  copy.observability_ = observability_;
  copy.call_results_ = call_results_;
  copy.callee_bodies_ = callee_bodies_;
  copy.load_bias_ = load_bias_;
  copy.entries_closed_ = entries_closed_;
  const auto remap = [&](SsaHandle& handle) {
    if (handle.arena == arena_) handle.arena = copy.arena_;
  };

  for (auto& entry : copy.entries_) remap(entry);
  for (auto& slot : copy.slots_) {
    if (!slot.block) continue;
    auto& block = *slot.block;
    for (auto& phi : block.phis) {
      for (auto& input : phi.incoming) {
        remap(input.predecessor);
        remap(input.value.block);
      }
    }

    for (auto& phi : block.frame_phis) {
      for (auto& input : phi.incoming) {
        remap(input.predecessor);
        remap(input.value.block);
      }
    }

    for (auto& exit : block.exits) remap(exit.value.block);
    for (auto& read : block.reads) {
      remap(read.value.block);
      if (read.predecessor_copy) remap(read.predecessor_copy->block);
    }

    for (auto& exit : block.frame_exits) remap(exit.block);
    for (auto& access : block.frame_accesses)
      if (access.replacement) remap(access.replacement->block);
    for (auto& edge : block.edges)
      if (edge.target_block) remap(*edge.target_block);
    for (auto& fold : block.constant_loads)
      if (fold.table_guard) remap(*fold.table_guard);
  }

  return copy;
}

bool SsaGraph::Erase(SsaHandle handle) {
  if (!std::as_const(*this).Get(handle) ||
      std::find(entries_.begin(), entries_.end(), handle) != entries_.end() ||
      slots_[handle.slot].generation == std::numeric_limits<std::uint32_t>::max())
    return false;
  for (std::size_t i = 0; i < slots_.size(); ++i) {
    const auto other = Handle(i);
    if (!other || *other == handle) continue;
    const auto& block = *std::as_const(*this).Get(*other);
    for (const auto& edge : block.edges)
      if (edge.target_block == handle) return false;
    for (const auto& phi : block.phis) {
      for (const auto& input : phi.incoming) {
        if (input.predecessor == handle || input.value.block == handle) return false;
      }
    }

    for (const auto& phi : block.frame_phis) {
      for (const auto& input : phi.incoming) {
        if (input.predecessor == handle || input.value.block == handle) return false;
      }
    }

    for (const auto& read : block.reads)
      if (read.value.block == handle ||
          (read.predecessor_copy && read.predecessor_copy->block == handle))
        return false;
  }

  slots_[handle.slot].block.reset();
  ++slots_[handle.slot].generation;
  ++revision_;
  return true;
}

namespace {
SsaDecline TableGuardStillBounds(const SsaGraph& graph, SsaHandle table_handle,
                                 const SsaConstantLoad& fold, Budget& budget) {
  if (budget.try_consume({graph.entries().size(), 0}) != BudgetDecline::none)
    return SsaDecline::resource_limit;
  if (!fold.table_guard || !fold.table_closed_entries || graph.entries().empty() ||
      std::find(graph.entries().begin(), graph.entries().end(), table_handle) !=
          graph.entries().end())
    return SsaDecline::invalid_graph;
  const auto* guard = graph.Get(*fold.table_guard);
  const auto* table = graph.Get(table_handle);

  // Everything the bound rests on is read off the graph as it stands: the
  // guard's condition and literal (SsaGuardEdgeBound), the register that
  // carries the compared number out (SsaExitHolding, and ValidateSsa's check
  // of exits against the guard's writes), the table block's one phi for it
  // and the index read of that phi with no write between (ValidateSsa's read
  // check), the load's address (SsaTableLoadAddress, in ValidateSsa's fold
  // check), and that the guard's edge is the only way in. So later passes may
  // rewrite either block: an edit that keeps the bound derivable keeps the
  // fold, and one that does not invalidates the graph. Dead, disabled, folded and
  // promoted nodes in the guard change nothing here, since ValidateSsa keeps
  // every node the condition and the exits use live and gives the compare and
  // the exit the same value. A recovered transfer, though, replaces the
  // conditional the guard's edges describe.
  if (!guard || !table || *fold.table_guard == table_handle || guard->opaque || guard->transition ||
      !guard->control_rewrites.empty() || !guard->store_omissions.empty() ||
      !guard->paired_load_omissions.empty() || !guard->destination_nodes.empty() ||
      guard->boundaries.empty() || fold.table_guard_edge >= guard->edges.size())
    return SsaDecline::invalid_graph;
  if (budget.try_consume({guard->edges.size(), 0}) != BudgetDecline::none)
    return SsaDecline::resource_limit;
  const auto& edge = guard->edges[fold.table_guard_edge];
  if (edge.target_block != table_handle || !edge.condition || !edge.when ||
      !guard->boundaries.back().transfer ||
      guard->boundaries.back().transfer->condition != edge.condition)
    return SsaDecline::invalid_graph;
  for (std::size_t i = 0; i < guard->edges.size(); ++i)
    if (i != fold.table_guard_edge && guard->edges[i].target_block == table_handle)
      return SsaDecline::invalid_graph;
  if (budget.try_consume({guard->reads.size() + guard->exits.size() + table->phis.size() +
                              table->reads.size() + 4 * kSsaCoreSteps,
                          0}) != BudgetDecline::none)
    return SsaDecline::resource_limit;
  const auto bound = SsaGuardEdgeBound(guard->nodes, *edge.condition, *edge.when);
  if (!bound || bound->exclusive_upper != fold.table_bytes.size()) return SsaDecline::invalid_graph;
  const auto* exit = SsaExitHolding(*guard, *fold.table_guard, fold.table_storage, bound->value);
  const auto phi = std::find_if(table->phis.begin(), table->phis.end(), [&](const SsaPhi& item) {
    return item.storage == fold.table_storage;
  });
  const auto table_read =
      std::find_if(table->reads.begin(), table->reads.end(),
                   [&](const SsaRead& item) { return item.node == fold.table_index; });
  if (!exit || phi == table->phis.end() || table_read == table->reads.end() ||
      phi->external_entry || phi->incoming.size() != 1 ||
      phi->incoming[0].predecessor != *fold.table_guard || phi->incoming[0].value != exit->value ||
      table_read->value != SsaValue{SsaValueKind::phi, table_handle,
                                    static_cast<std::uint32_t>(phi - table->phis.begin())})
    return SsaDecline::invalid_graph;
  if (graph.slots() > SIZE_MAX / (sizeof(SsaHandle) + 1) ||
      budget.try_consume({graph.slots(), graph.slots() * (sizeof(SsaHandle) + 1)}) !=
          BudgetDecline::none)
    return SsaDecline::resource_limit;
  // Every block that can run is complete here, with no allowance for a table
  // dispatch whose fold is still to come (ValidateSsaDispatchSuccessors): a
  // fold's value, and the completeness of the dispatch it proves, are used by
  // passes that never ask whether the rest of the graph is complete, so the
  // guard must hold on every run of the graph as it stands. That a dispatch
  // this fold completes is itself visited is not circular. Take the first
  // step of a run where a folded dispatch jumps outside its edges: before it
  // every block kept to its edges, so each guard was the only way into its
  // table block and bounded its index, so the dispatch read one of its rows
  // and went to one of its edges.
  std::vector<std::uint8_t> visited(graph.slots());
  std::vector<SsaHandle> queue;
  queue.reserve(graph.slots());
  for (const auto entry : graph.entries()) {
    if (visited[entry.slot]) continue;
    visited[entry.slot] = 1;
    queue.push_back(entry);
  }

  for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
    const auto* block = graph.Get(queue[cursor]);
    if (!block || budget.try_consume({1 + block->edges.size(), 0}) != BudgetDecline::none)
      return SsaDecline::resource_limit;
    const auto complete = ValidateSsaDirectSuccessors(*block, budget);
    if (complete != SsaDecline::none) return complete;
    for (const auto& outgoing : block->edges)
      if (outgoing.target_block && !visited[outgoing.target_block->slot]) {
        visited[outgoing.target_block->slot] = 1;
        queue.push_back(*outgoing.target_block);
      }
  }

  return visited[table_handle.slot] ? SsaDecline::none : SsaDecline::invalid_graph;
}
}  // namespace

namespace {
std::optional<std::pair<std::uint64_t, bool>> LoadLocation(std::span<const Node> nodes,
                                                           ValueId load,
                                                           std::optional<std::uint64_t> bias);
}  // namespace

namespace {

// Per node, whether it comes out the same on every run of a loop block: built
// only from literals, image locations, and registers the block reads as they
// arrived and hands on unchanged. Inputs precede their users, so one pass.
std::vector<std::uint8_t> SteadyNodes(const SsaBlock& block, SsaHandle handle) {
  std::vector<std::uint8_t> steady(block.nodes.size());
  for (ValueId id = 0; id < block.nodes.size(); ++id) {
    const auto& node = block.nodes[id];
    if (node.op == Op::constant || node.op == Op::image_address) {
      steady[id] = 1;
    } else if (node.op == Op::read) {
      const auto read = std::find_if(block.reads.begin(), block.reads.end(),
                                     [&](const SsaRead& item) { return item.node == id; });
      steady[id] = read != block.reads.end() && read->value.kind == SsaValueKind::phi &&
                   read->value.block == handle && read->value.index < block.exits.size() &&
                   block.exits[read->value.index].value == read->value;
    } else if (const auto* descriptor = Descriptor(node.op);
               descriptor && descriptor->effect == Effect::pure && descriptor->produces_value) {
      bool all = true;
      for (unsigned input = 0; all && input < descriptor->arity; ++input)
        all = node.inputs[input] < id && steady[node.inputs[input]];
      steady[id] = all;
    }
  }

  return steady;
}

bool TargetFoldsTo(const SsaBlock& block, ValueId target, std::uint64_t expected, Budget& budget,
                   bool& exhausted);

// What a callee body does, when it is a leaf that does nothing but compute:
// contiguous groups from its address, no access to memory or monitor, control
// only at the end, and that a return through a register the body never
// writes. `written` gets every storage it writes; `link` the one it returns
// through.
bool LeafBody(const SsaCalleeBody& body, std::vector<StorageId>& written, StorageId& link,
              Budget& budget) {
  if (body.groups.empty() || body.groups.size() > kMaxCalleeBodyGroups) return false;
  written.clear();
  std::uint64_t next = body.address;
  for (std::size_t index = 0; index < body.groups.size(); ++index) {
    const auto& group = body.groups[index];
    if (budget.try_consume({1 + group.nodes().size() + group.writes().size(), 0}) !=
        BudgetDecline::none)
      return false;
    if (group.source_address() != next || group.bytes().empty() ||
        next > UINT64_MAX - group.bytes().size())
      return false;
    next += group.bytes().size();
    for (const auto& node : group.nodes()) {
      const auto* descriptor = Descriptor(node.op);
      if (!descriptor) return false;
      if (node.op == Op::write)
        written.push_back(node.storage);
      else if (descriptor->effect != Effect::pure && descriptor->effect != Effect::storage_read)
        return false;
    }

    for (const auto& write : group.writes()) written.push_back(write.storage);
    const auto& transfer = group.transfer();
    const bool last = index + 1 == body.groups.size();
    if (transfer.has_value() != last) return false;
    if (last) {
      if (transfer->kind != TransferKind::return_ || transfer->target >= group.nodes().size() ||
          group.nodes()[transfer->target].op != Op::read)
        return false;
      link = group.nodes()[transfer->target].storage;
    }
  }

  std::sort(written.begin(), written.end());
  written.erase(std::unique(written.begin(), written.end()), written.end());
  return !std::binary_search(written.begin(), written.end(), link);
}

const SsaCalleeBody* BodyAt(const SsaGraph& graph, std::uint64_t address) {
  const auto bodies = graph.callee_bodies();
  const auto at = std::lower_bound(
      bodies.begin(), bodies.end(), address,
      [](const SsaCalleeBody& body, std::uint64_t key) { return body.address < key; });
  return at != bodies.end() && at->address == address ? &*at : nullptr;
}

// What makes jumping past a call the same as making it, as far as the graph
// says at the revision the call is retired: the call reaches the body's
// address, the body is a leaf, and the register the call leaves its
// continuation in is the one the body returns through, so it does return
// there. That what the body writes is dead where the call returns is
// ValidateSsaRetiredWork's.
SsaDecline RetiredCallHolds(const SsaGraph& graph, SsaHandle handle,
                            const ConditionalRewrite& rewrite, Budget& budget) {
  const auto& block = *graph.Get(handle);
  const auto* body = BodyAt(graph, rewrite.when_true);
  std::vector<StorageId> written;
  StorageId link = 0;
  if (!body || !LeafBody(*body, written, link, budget) || !rewrite.original.continuation)
    return SsaDecline::invalid_graph;
  // The target resolves as a known callee edge's does -- folded through the
  // block's own declared image reads -- or settles through the graph.
  bool exhausted = false;
  const bool local =
      TargetFoldsTo(block, rewrite.original.target, rewrite.when_true, budget, exhausted);
  if (exhausted) return SsaDecline::resource_limit;
  if (!local) {
    const auto target = SsaSettledTarget(graph, block, rewrite.original.target, budget);
    if (!target || *target != rewrite.when_true) return SsaDecline::invalid_graph;
  }

  const auto& boundary = block.boundaries[rewrite.boundary];
  bool linked = false;
  for (auto id = boundary.first_node; id < boundary.first_node + boundary.node_count; ++id)
    if (block.nodes[id].op == Op::write)
      linked = block.nodes[id].storage == link
                   ? SsaCopySource(block.nodes, block.nodes[id].inputs[0]) ==
                         SsaCopySource(block.nodes, *rewrite.original.continuation)
                   : linked;
  for (const auto& write : boundary.writes)
    if (write.storage == link)
      linked = SsaCopySource(block.nodes, write.value) ==
               SsaCopySource(block.nodes, *rewrite.original.continuation);
  return linked ? SsaDecline::none : SsaDecline::invalid_graph;
}

// What makes one run of a retired loop stand for all of them, as far as the
// block alone says: it leaves only by its exit, and does nothing but compute
// and write storage -- no access to memory executes, and any that still
// checks its address computes it alike on every run. That everything it
// changes is dead where it leaves is ValidateSsaBoundedExits'; that it did
// leave, after the iterations it records, the bounded-loop fact's.
SsaDecline BoundedExitRunsOnce(const SsaGraph& graph, SsaHandle handle, Budget& budget) {
  const auto& block = *graph.Get(handle);
  if (block.opaque || block.edges.size() != 1 || !block.edges.front().target_block ||
      block.edges.front().target_block->slot == handle.slot)
    return SsaDecline::invalid_graph;
  if (budget.try_consume({block.nodes.size(), 0}) != BudgetDecline::none)
    return SsaDecline::resource_limit;
  const auto listed = [](const auto& sorted, ValueId id) {
    return std::binary_search(sorted.begin(), sorted.end(), id);
  };

  if (budget.try_consume({block.nodes.size() * (1 + block.reads.size()), block.nodes.size()}) !=
      BudgetDecline::none)
    return SsaDecline::resource_limit;
  const auto steady = SteadyNodes(block, handle);
  for (ValueId id = 0; id < block.nodes.size(); ++id) {
    const auto op = block.nodes[id].op;
    const auto* descriptor = Descriptor(op);
    if (!descriptor) return SsaDecline::invalid_graph;
    if (descriptor->effect == Effect::pure || descriptor->effect == Effect::storage_read ||
        op == Op::write || listed(block.dead_pure_nodes, id))
      continue;
    // Any other effect runs once where it used to run many times.
    if (!listed(block.disabled_effects, id)) return SsaDecline::invalid_graph;

    // A disabled access no longer reads or writes, but unless a fold skips it
    // entirely it still checks its address where it would run. One whose
    // address every run computes alike was checked by the first.
    const auto fold = std::lower_bound(
        block.constant_loads.begin(), block.constant_loads.end(), id,
        [](const SsaConstantLoad& load, ValueId value) { return load.node < value; });
    if (fold != block.constant_loads.end() && fold->node == id && fold->skip_access) continue;
    if (!steady[block.nodes[id].inputs[0]]) return SsaDecline::invalid_graph;
  }

  return SsaDecline::none;
}

}  // namespace

SsaDecline ValidateSsa(const SsaGraph& graph, Budget& budget) {
  const auto invalid = SsaDecline::invalid_graph;
  if (graph.slots() > std::numeric_limits<std::uint32_t>::max()) return invalid;
  if (budget.try_consume({graph.slots(), graph.slots() * (sizeof(std::vector<SsaHandle>) +
                                                          sizeof(std::uint8_t))}) !=
      BudgetDecline::none)
    return SsaDecline::resource_limit;
  std::vector<std::vector<SsaHandle>> predecessors(graph.slots());
  std::vector<std::uint8_t> entry(graph.slots());
  const SsaBlock* frame_shape = nullptr;

  // Gathered on the first image fold that needs them, then shared by all:
  // one view without the placement declaration, one with it.
  std::array<std::optional<SsaImageStores>, 2> placed_stores;
  const auto& observed = graph.observability();
  if (budget.try_consume(
          {observed.at_call.size() + observed.at_return.size() + observed.preserved.size(), 0}) !=
      BudgetDecline::none)
    return SsaDecline::resource_limit;
  const auto strictly_sorted = [](const std::vector<StorageId>& ids) {
    return std::adjacent_find(ids.begin(), ids.end(), std::greater_equal<>()) == ids.end();
  };

  const auto& results = graph.call_results();
  if (budget.try_consume({results.size(), 0}) != BudgetDecline::none)
    return SsaDecline::resource_limit;
  for (std::size_t i = 1; i < results.size(); ++i)
    if (std::pair{results[i - 1].target, results[i - 1].storage} >=
        std::pair{results[i].target, results[i].storage})
      return invalid;
  const auto bodies = graph.callee_bodies();
  if (budget.try_consume({bodies.size(), 0}) != BudgetDecline::none)
    return SsaDecline::resource_limit;
  for (std::size_t i = 1; i < bodies.size(); ++i)
    if (bodies[i - 1].address >= bodies[i].address) return invalid;
  if (!strictly_sorted(observed.at_call) || !strictly_sorted(observed.at_return) ||
      !strictly_sorted(observed.preserved) ||
      (!observed.declared &&
       (!observed.at_call.empty() || !observed.at_return.empty() || !observed.preserved.empty())))
    return invalid;
  for (const auto handle : graph.entries()) {
    if (!graph.Get(handle) || entry[handle.slot]) return invalid;
    entry[handle.slot] = 1;
  }

  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto& block = *graph.Get(*handle);
    if (!frame_shape) frame_shape = &block;
    if (block.frame_phis.size() != frame_shape->frame_phis.size()) return invalid;
    for (std::size_t i = 0; i < block.frame_phis.size(); ++i) {
      if (block.frame_phis[i].offset != frame_shape->frame_phis[i].offset ||
          block.frame_phis[i].size != frame_shape->frame_phis[i].size)
        return invalid;
    }

    if (budget.try_consume({1 + block.nodes.size() + block.phis.size() + block.edges.size() +
                                block.boundaries.size(),
                            0}) != BudgetDecline::none)
      return SsaDecline::resource_limit;
    if (block.original_sources.size() != block.source_groups.size() ||
        (!block.source_bytes.empty() &&
         (block.source_bytes.size() != block.source_groups.size() ||
          std::any_of(block.source_bytes.begin(), block.source_bytes.end(),
                      [](const auto& bytes) { return bytes.empty(); }))) ||
        (!block.opaque && block.boundaries.size() != block.source_groups.size()) ||
        (block.opaque && (!block.boundaries.empty() || !block.nodes.empty())))
      return invalid;
    // Outside a recovered transition, only a call resolved to one location
    // names a destination: the one it calls.
    const bool resolved_call =
        std::any_of(block.control_rewrites.begin(), block.control_rewrites.end(),
                    [](const ConditionalRewrite& rewrite) {
                      return rewrite.rule == RewriteRule::resolved_call;
                    });
    if (!block.transition &&
        (!block.store_omissions.empty() || !block.paired_load_omissions.empty() ||
         block.destination_nodes.size() > (resolved_call ? 1U : 0U)))
      return invalid;
    const bool follows_call =
        !block.boundaries.empty() && block.boundaries.back().transfer &&
        block.boundaries.back().transfer->kind == TransferKind::call &&
        std::any_of(block.edges.begin(), block.edges.end(),
                    [](const SsaEdge& edge) { return edge.kind == SsaEdgeKind::branch; });
    if (!block.transition && (!block.control_rewrites.empty() || follows_call)) {
      const auto direct = ValidateSsaDirectSuccessors(block, budget);
      if (direct != SsaDecline::none) return direct;
    }

    for (std::size_t i = 0; i < block.disabled_effects.size(); ++i) {
      const auto id = block.disabled_effects[i];
      if (id >= block.nodes.size() ||
          (block.nodes[id].op != Op::load && block.nodes[id].op != Op::store) ||
          (i && block.disabled_effects[i - 1] >= id))
        return invalid;
    }

    for (std::size_t i = 0; i < block.path_reads.size(); ++i) {
      const auto& read = block.path_reads[i];
      if (budget.try_consume({65, 0}) != BudgetDecline::none) return SsaDecline::resource_limit;
      const auto location = LoadLocation(block.nodes, read.node, graph.load_bias());
      if ((i && block.path_reads[i - 1].node >= read.node) || !location ||
          location->first != read.address || (location->second && !read.page_aligned_placement) ||
          read.value > LowMask(block.nodes[read.node].width) ||
          (read.relocated && (block.nodes[read.node].width != 64 ||
                              block.nodes[read.node].access.byte_order != ByteOrder::little)))
        return invalid;
      // A declared value a store the graph places writes is not what the
      // load reads.
      auto& stores = placed_stores[read.page_aligned_placement];
      if (!stores &&
          !(stores = SsaImageStores::Collect(graph, read.page_aligned_placement, budget)))
        return SsaDecline::resource_limit;
      if (stores->ConflictsPlaced(read.address, block.nodes[read.node].width / 8)) return invalid;
    }

    for (std::size_t i = 0; i < block.retired_loads.size(); ++i) {
      const auto& retired = block.retired_loads[i];
      if (budget.try_consume({64 + block.store_omissions.size() + block.constant_loads.size() +
                                  block.frame_accesses.size(),
                              0}) != BudgetDecline::none)
        return SsaDecline::resource_limit;
      if ((i && block.retired_loads[i - 1].node >= retired.node) ||
          retired.node >= block.nodes.size() || block.nodes[retired.node].op != Op::load ||
          !std::binary_search(block.disabled_effects.begin(), block.disabled_effects.end(),
                              retired.node) ||
          std::any_of(block.constant_loads.begin(), block.constant_loads.end(),
                      [&](const SsaConstantLoad& fold) { return fold.node == retired.node; }) ||
          std::any_of(block.frame_accesses.begin(), block.frame_accesses.end(),
                      [&](const SsaFrameAccess& access) { return access.node == retired.node; }))
        return invalid;
      const auto& load = block.nodes[retired.node];
      if (retired.basis == SsaRetiredLoadBasis::written_before) {
        if (retired.store >= retired.node) return invalid;
        const auto& store = block.nodes[retired.store];
        if (store.op != Op::store || store.inputs[1] >= block.nodes.size() ||
            std::binary_search(block.disabled_effects.begin(), block.disabled_effects.end(),
                               retired.store) ||
            std::any_of(
                block.store_omissions.begin(), block.store_omissions.end(),
                [&](const StoreOmission& omission) { return omission.store == retired.store; }) ||
            !SsaSameValue(block, store.inputs[0], load.inputs[0]) ||
            block.nodes[store.inputs[1]].width < load.width ||
            load.access.alignment > store.access.alignment)
          return invalid;
      } else {
        const auto span = SsaLoadImageSpan(block, retired.node, graph.load_bias());
        if (!span || !retired.access.mapped_readable_lifetime ||
            !retired.access.no_runtime_unmapping || !retired.access.ordinary_reads_unobservable ||
            retired.range_bytes > UINT64_MAX - retired.range_address ||
            span->first < retired.range_address ||
            span->second > retired.range_address + retired.range_bytes)
          return invalid;
      }
    }

    if (!block.retired_loads.empty()) {
      const auto live = SsaLiveValues(graph, *handle, budget);
      if (!live) return SsaDecline::resource_limit;
      for (const auto& retired : block.retired_loads)
        if ((*live)[retired.node]) return invalid;
    }

    for (std::size_t i = 0; i < block.constant_loads.size(); ++i) {
      if (budget.try_consume({17, 0}) != BudgetDecline::none ||
          budget.try_consume(
              {static_cast<std::uint64_t>(std::bit_width(block.disabled_effects.size())), 0}) !=
              BudgetDecline::none ||
          budget.try_consume({block.frame_accesses.size(), 0}) != BudgetDecline::none)
        return SsaDecline::resource_limit;
      const auto& fold = block.constant_loads[i];
      if ((fold.kind != SsaConstantKind::literal && fold.kind != SsaConstantKind::image_location &&
           fold.kind != SsaConstantKind::bounded_table) ||
          fold.node >= block.nodes.size() || block.nodes[fold.node].op != Op::load ||
          !std::binary_search(block.disabled_effects.begin(), block.disabled_effects.end(),
                              fold.node) ||
          (i && block.constant_loads[i - 1].node >= fold.node))
        return invalid;
      const auto& load = block.nodes[fold.node];
      if (load.inputs[0] >= block.nodes.size() ||
          (fold.kind == SsaConstantKind::image_location && load.width != 64))
        return invalid;
      if (std::any_of(block.frame_accesses.begin(), block.frame_accesses.end(),
                      [&](const SsaFrameAccess& access) { return access.node == fold.node; }))
        return invalid;
      if (fold.kind == SsaConstantKind::bounded_table) {
        if (!fold.skip_access || !fold.value_stable || fold.condition || fold.value ||
            !fold.table_guard || !fold.table_closed_entries ||
            !fold.access.mapped_readable_lifetime || !fold.access.no_runtime_unmapping ||
            !fold.access.ordinary_reads_unobservable || !load.width || load.width > 64 ||
            load.width % 8 || !fold.table_stride || fold.table_bytes.empty() ||
            fold.table_bytes.size() > 64 || fold.table_index >= fold.node ||
            fold.table_index >= block.nodes.size() ||
            block.nodes[fold.table_index].op != Op::read ||
            block.nodes[fold.table_index].width != 64)
          return invalid;
        if (budget.try_consume({64, 0}) != BudgetDecline::none) return SsaDecline::resource_limit;
        const auto shape = SsaTableLoadAddress(block.nodes, fold.node, graph.load_bias());
        if (!shape || shape->base != fold.source_address || shape->index != fold.table_index ||
            shape->stride != fold.table_stride || (shape->placed && !fold.page_aligned_placement) ||
            (load.access.alignment > 1 &&
             (!fold.page_aligned_placement || load.access.alignment > 4096 ||
              fold.source_address % load.access.alignment ||
              fold.table_stride % load.access.alignment)))
          return invalid;
        const auto maximum = fold.table_bytes.size() - 1;
        if (maximum > UINT64_MAX / fold.table_stride ||
            fold.source_address > UINT64_MAX - maximum * fold.table_stride ||
            fold.source_address + maximum * fold.table_stride > UINT64_MAX - (load.width / 8 - 1))
          return invalid;
        for (std::size_t row = 0; row < fold.table_bytes.size(); ++row) {
          if (budget.try_consume({1 + load.width / 8, 0}) != BudgetDecline::none)
            return SsaDecline::resource_limit;
          const auto location = fold.source_address + row * fold.table_stride;
          auto& stores = placed_stores[fold.page_aligned_placement];
          if (!stores &&
              !(stores = SsaImageStores::Collect(graph, fold.page_aligned_placement, budget)))
            return SsaDecline::resource_limit;
          if (stores->Conflicts(location, load.width / 8, fold.read_only)) return invalid;
          for (unsigned byte = load.width / 8; byte < 8; ++byte)
            if (fold.table_bytes[row][byte]) return invalid;
        }

        continue;
      }

      if (!fold.table_bytes.empty() || fold.table_stride || fold.table_index || fold.table_guard ||
          fold.table_guard_edge || fold.table_storage || fold.table_closed_entries)
        return invalid;
      if (fold.condition) {
        if (!fold.skip_access || fold.kind != SsaConstantKind::literal) return invalid;
        const auto checked =
            CheckSsaSelectedImageAddress(block, fold.node, *fold.condition, fold.source_address,
                                         fold.alternative_source_address, budget);
        if (checked != SsaDecline::none) return checked;
      } else {
        // The address is evaluated, so a PC page as well as a literal names
        // the location; a page names it only under the placement the fold carries.
        const auto location = LoadLocation(block.nodes, fold.node, graph.load_bias());
        if (!location || location->first != fold.source_address ||
            (location->second && !fold.page_aligned_placement))
          return invalid;
      }

      if (fold.skip_access) {
        if (!fold.value_stable || !fold.access.mapped_readable_lifetime ||
            !fold.access.no_runtime_unmapping || !fold.access.ordinary_reads_unobservable ||
            (load.access.alignment > 1 &&
             (!fold.page_aligned_placement || load.access.alignment > 4096 ||
              fold.source_address % load.access.alignment ||
              (fold.condition && fold.alternative_source_address % load.access.alignment))))
          return invalid;
        if (fold.kind == SsaConstantKind::image_location) {
          if (load.access.byte_order != ByteOrder::little || fold.declared_target != fold.value)
            return invalid;
        } else {
          if (!load.width || load.width > 64 || load.width % 8) return invalid;
          std::uint64_t bits = 0;
          const auto size = load.width / 8;
          for (unsigned byte = 0; byte < size; ++byte) {
            const auto shift = load.access.byte_order == ByteOrder::little ? byte : size - 1 - byte;
            bits |= std::uint64_t(fold.declared_bytes[byte]) << (shift * 8);
          }

          if (bits != fold.value) return invalid;
          if (fold.condition) {
            bits = 0;
            for (unsigned byte = 0; byte < size; ++byte) {
              const auto shift =
                  load.access.byte_order == ByteOrder::little ? byte : size - 1 - byte;
              bits |= std::uint64_t(fold.alternative_declared_bytes[byte]) << (shift * 8);
            }

            if (bits != fold.alternative_value) return invalid;
          }
        }

        auto& stores = placed_stores[fold.page_aligned_placement];
        if (!stores &&
            !(stores = SsaImageStores::Collect(graph, fold.page_aligned_placement, budget)))
          return SsaDecline::resource_limit;
        if (stores->Conflicts(fold.source_address, load.width / 8, fold.read_only) ||
            (fold.condition &&
             stores->Conflicts(fold.alternative_source_address, load.width / 8, fold.read_only)))
          return invalid;
      }
    }

    for (std::size_t i = 0; i < block.entry_relations.size(); ++i) {
      const auto& relation = block.entry_relations[i];
      if (relation.root >= relation.storage ||
          (i && block.entry_relations[i - 1].storage >= relation.storage))
        return invalid;
    }

    for (const auto& node : block.destination_nodes) {
      if (node.width != 64 || (node.op != Op::constant && node.op != Op::image_address))
        return invalid;
    }

    for (std::size_t rewrite_index = 0; rewrite_index < block.control_rewrites.size();
         ++rewrite_index) {
      const auto& rewrite = block.control_rewrites[rewrite_index];
      if (rewrite.boundary >= block.boundaries.size() ||
          !block.boundaries[rewrite.boundary].transfer ||
          (rewrite_index && block.control_rewrites[rewrite_index - 1].boundary >= rewrite.boundary))
        return invalid;
      const auto& original = *block.boundaries[rewrite.boundary].transfer;
      if (rewrite.original.kind != original.kind || rewrite.original.target != original.target ||
          rewrite.original.condition != original.condition ||
          rewrite.original.alternative != original.alternative ||
          rewrite.original.continuation != original.continuation)
        return invalid;
      const auto end = block.boundaries[rewrite.boundary].first_node +
                       block.boundaries[rewrite.boundary].node_count;
      if (!ValidTransferWidths(rewrite.replacement, [&](ValueId id) -> unsigned {
            if (id < end) return ValidNode(block, id) ? block.nodes[id].width : 0;
            if (id < block.nodes.size()) return 0;
            const auto named = static_cast<std::size_t>(id) - block.nodes.size();
            return named < block.destination_nodes.size() ? 64 : 0;
          }))
        return invalid;
    }

    for (const auto& omission : block.store_omissions) {
      if (omission.store >= block.nodes.size() || omission.overwriter >= block.nodes.size() ||
          block.nodes[omission.store].op != Op::store ||
          block.nodes[omission.overwriter].op != Op::store)
        return invalid;
    }

    for (const auto& omission : block.paired_load_omissions) {
      for (const auto id : omission.loads) {
        if (id >= block.nodes.size() || block.nodes[id].op != Op::load) return invalid;
      }

      for (const auto id : omission.stores) {
        if (id >= block.nodes.size() || block.nodes[id].op != Op::store) return invalid;
      }
    }

    if (block.clobbers.size() != block.phis.size()) return invalid;

    // Control that leaves through a call or an opaque instruction returns
    // with every register fresh. The one exception is the declared contract's
    // claim that a callee hands back what it preserves.
    if (budget.try_consume({block.edges.size() + block.phis.size(), 0}) != BudgetDecline::none)
      return SsaDecline::resource_limit;
    if (block.opaque ||
        std::any_of(block.edges.begin(), block.edges.end(), [](const SsaEdge& edge) {
          return edge.kind == SsaEdgeKind::potential_return;
        })) {
      const auto& contract = graph.observability();
      for (std::size_t i = 0; i < block.phis.size(); ++i)
        if (!block.clobbers[i] &&
            (block.opaque || !contract.declared ||
             !std::binary_search(contract.preserved.begin(), contract.preserved.end(),
                                 block.phis[i].storage)))
          return invalid;
    }

    std::uint64_t next = 0;
    for (const auto& boundary : block.boundaries) {
      if (boundary.first_node != next || boundary.node_count > block.nodes.size() - next)
        return invalid;
      next += boundary.node_count;
      for (const auto& write : boundary.writes)
        if (write.value >= boundary.first_node + boundary.node_count ||
            !ValidNode(block, write.value))
          return invalid;
      if (boundary.transfer && !ValidTransfer(*boundary.transfer, block.nodes)) return invalid;
    }

    if (next != block.nodes.size()) return invalid;
    if (budget.try_consume({block.dead_storage_writes.size(), 0}) != BudgetDecline::none)
      return SsaDecline::resource_limit;
    for (std::size_t i = 0; i < block.dead_storage_writes.size(); ++i) {
      const auto& mark = block.dead_storage_writes[i];
      if (!mark.closed_entries || mark.boundary >= block.boundaries.size() ||
          mark.index >= block.boundaries[mark.boundary].writes.size() ||
          (i && (block.dead_storage_writes[i - 1].boundary > mark.boundary ||
                 (block.dead_storage_writes[i - 1].boundary == mark.boundary &&
                  block.dead_storage_writes[i - 1].index >= mark.index))))
        return invalid;
    }

    for (std::size_t id = 0; id < block.nodes.size(); ++id) {
      if (!Shape(block, static_cast<ValueId>(id))) return invalid;
    }

    if (!block.dead_pure_nodes.empty()) {
      const auto dead_work = 1 + std::bit_width(block.constant_loads.size());
      if (budget.try_consume({block.nodes.size(), block.nodes.size()}) != BudgetDecline::none ||
          block.dead_pure_nodes.size() > std::numeric_limits<std::size_t>::max() / dead_work ||
          budget.try_consume({block.dead_pure_nodes.size() * dead_work, 0}) != BudgetDecline::none)
        return SsaDecline::resource_limit;
      std::vector<std::uint8_t> live(block.nodes.size(), 1);
      for (std::size_t i = 0; i < block.dead_pure_nodes.size(); ++i) {
        const auto id = block.dead_pure_nodes[i];
        if (id >= block.nodes.size() || !SsaEffectFree(block, id) ||
            (i && block.dead_pure_nodes[i - 1] >= id))
          return invalid;
        live[id] = 0;
      }

      const auto checked = RootsAndInputsLive(block, live, budget);
      if (checked != SsaDecline::none) return checked;
    }

    for (std::size_t i = 0; i < block.phis.size(); ++i) {
      if (!block.phis[i].width) return invalid;
      if (i && block.phis[i - 1].storage >= block.phis[i].storage) return invalid;
      if (block.phis[i].external_entry != bool(entry[slot])) return invalid;
    }

    if (block.frame_exits.size() != block.frame_phis.size()) return invalid;
    for (const auto& phi : block.frame_phis) {
      if (!phi.size || phi.size > 16 || phi.external_entry != bool(entry[slot])) return invalid;
    }

    if (budget.try_consume({block.nodes.size(), block.nodes.size() * sizeof(std::size_t)}) !=
        BudgetDecline::none)
      return SsaDecline::resource_limit;
    std::vector<std::size_t> read_at(block.nodes.size(), std::numeric_limits<std::size_t>::max());
    for (std::size_t i = 0; i < block.reads.size(); ++i) {
      const auto& read = block.reads[i];
      if (read.node >= block.nodes.size() || block.nodes[read.node].op != Op::read ||
          read.phi >= block.phis.size() ||
          read_at[read.node] != std::numeric_limits<std::size_t>::max() ||
          block.nodes[read.node].storage != block.phis[read.phi].storage ||
          block.nodes[read.node].width != block.phis[read.phi].width ||
          read.value.block != *handle || !ValidValue(graph, read.value, block.phis[read.phi].width))
        return invalid;
      if (read.predecessor_copy) {
        const auto* predecessor = graph.Get(read.predecessor_copy->block);
        if (budget.try_consume(
                {std::uint64_t{1} +
                     (predecessor ? std::bit_width(predecessor->dead_pure_nodes.size()) : 0),
                 0}) != BudgetDecline::none)
          return SsaDecline::resource_limit;
        if (!read.copy_closed_entries ||
            ExpectedPredecessorCopy(graph, *handle, read) != read.predecessor_copy)
          return invalid;
      } else if (read.copy_closed_entries)
        return invalid;
      read_at[read.node] = i;
    }

    for (std::size_t id = 0; id < block.nodes.size(); ++id) {
      if (block.nodes[id].op == Op::read && read_at[id] == std::numeric_limits<std::size_t>::max())
        return invalid;
    }

    if (budget.try_consume({block.frame_phis.size() + block.frame_accesses.size(),
                            block.frame_phis.size() * sizeof(SsaValue)}) != BudgetDecline::none)
      return SsaDecline::resource_limit;
    std::vector<SsaValue> frame_expected;
    for (std::size_t i = 0; i < block.frame_phis.size(); ++i)
      frame_expected.push_back({SsaValueKind::frame_phi, *handle, static_cast<std::uint32_t>(i)});
    for (std::size_t i = 0; i < block.frame_accesses.size(); ++i) {
      const auto& access = block.frame_accesses[i];
      if (access.node >= block.nodes.size() || access.phi >= block.frame_phis.size() ||
          (i && block.frame_accesses[i - 1].node >= access.node) ||
          !std::binary_search(block.disabled_effects.begin(), block.disabled_effects.end(),
                              access.node))
        return invalid;
      const auto& node = block.nodes[access.node];
      if (node.op == Op::load) {
        if (!access.replacement || *access.replacement != frame_expected[access.phi] ||
            node.width != block.frame_phis[access.phi].size * 8)
          return invalid;
      } else if (node.op == Op::store) {
        if (access.replacement || node.width != block.frame_phis[access.phi].size * 8)
          return invalid;
        frame_expected[access.phi] = {SsaValueKind::node, *handle, node.inputs[1]};
      } else
        return invalid;
    }

    if (block.frame_exits != frame_expected) return invalid;
    if (budget.try_consume({block.phis.size(), block.phis.size() * sizeof(SsaValue)}) !=
        BudgetDecline::none)
      return SsaDecline::resource_limit;
    std::vector<SsaValue> expected;
    std::vector<SsaValue> current;
    expected.reserve(block.phis.size());
    current.reserve(block.phis.size());
    for (std::size_t i = 0; i < block.phis.size(); ++i) {
      expected.push_back({block.clobbers[i] ? SsaValueKind::clobber : SsaValueKind::phi, *handle,
                          static_cast<std::uint32_t>(i)});
      current.push_back({SsaValueKind::phi, *handle, static_cast<std::uint32_t>(i)});
    }

    for (const auto& boundary : block.boundaries) {
      for (std::uint32_t id = boundary.first_node; id < boundary.first_node + boundary.node_count;
           ++id) {
        if (block.nodes[id].op != Op::read) continue;
        const auto& read = block.reads[read_at[id]];
        if (read.value != current[read.phi]) return invalid;
      }

      for (std::uint32_t id = boundary.first_node; id < boundary.first_node + boundary.node_count;
           ++id) {
        const auto& node = block.nodes[id];
        if (node.op != Op::write) continue;
        const auto found = std::lower_bound(
            block.phis.begin(), block.phis.end(), node.storage,
            [](const SsaPhi& phi, StorageId storage) { return phi.storage < storage; });
        if (found == block.phis.end() || found->storage != node.storage ||
            found->width != node.width)
          return invalid;
        const auto index = static_cast<std::size_t>(found - block.phis.begin());
        current[index] = {SsaValueKind::node, *handle, node.inputs[0]};
        if (!block.clobbers[index]) expected[index] = current[index];
      }

      for (const auto& write : boundary.writes) {
        if (budget.try_consume({1, 0}) != BudgetDecline::none) return SsaDecline::resource_limit;
        const auto found = std::lower_bound(
            block.phis.begin(), block.phis.end(), write.storage,
            [](const SsaPhi& phi, StorageId storage) { return phi.storage < storage; });
        if (found == block.phis.end() || found->storage != write.storage) return invalid;
        const auto index = static_cast<std::size_t>(found - block.phis.begin());
        current[index] = {SsaValueKind::node, *handle, write.value};
        if (!block.clobbers[index]) expected[index] = {SsaValueKind::node, *handle, write.value};
      }
    }

    if (block.exits.size() != block.phis.size()) return invalid;
    for (std::size_t i = 0; i < block.exits.size(); ++i) {
      const auto& exit = block.exits[i];
      const auto& phi = block.phis[i];
      if (exit.storage != phi.storage || !ValidValue(graph, exit.value, phi.width) ||
          exit.value.block != *handle)
        return invalid;
      if (exit.value != expected[i]) return invalid;
    }

    for (const auto& edge : block.edges) {
      if (edge.condition &&
          (!ValidNode(block, *edge.condition) || block.nodes[*edge.condition].width != 1))
        return invalid;
      if (edge.when.has_value() != edge.condition.has_value()) return invalid;
      if (edge.target_block) {
        if (!graph.Get(*edge.target_block) ||
            graph.Get(*edge.target_block)->address != edge.address ||
            edge.target_kind != SsaTargetKind::image_location)
          return invalid;
        auto& incoming = predecessors[edge.target_block->slot];
        if (std::find(incoming.begin(), incoming.end(), *handle) == incoming.end())
          incoming.push_back(*handle);
      }
    }
  }

  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto& block = *graph.Get(*handle);
    for (std::size_t i = 0; i < block.frame_phis.size(); ++i) {
      const auto& phi = block.frame_phis[i];
      if (phi.incoming.size() != predecessors[slot].size()) return invalid;
      std::vector<SsaHandle> seen;
      for (const auto& input : phi.incoming) {
        if (std::find(predecessors[slot].begin(), predecessors[slot].end(), input.predecessor) ==
                predecessors[slot].end() ||
            std::find(seen.begin(), seen.end(), input.predecessor) != seen.end() ||
            input.value.block != input.predecessor || !ValidValue(graph, input.value, phi.size * 8))
          return invalid;
        const auto* predecessor = graph.Get(input.predecessor);
        if (predecessor->frame_exits[i] != input.value) return invalid;
        seen.push_back(input.predecessor);
      }
    }

    for (std::size_t i = 0; i < block.phis.size(); ++i) {
      const auto& phi = block.phis[i];
      if (phi.incoming.size() != predecessors[slot].size()) return invalid;
      std::vector<SsaHandle> seen;
      for (const auto& input : phi.incoming) {
        if (std::find(predecessors[slot].begin(), predecessors[slot].end(), input.predecessor) ==
                predecessors[slot].end() ||
            std::find(seen.begin(), seen.end(), input.predecessor) != seen.end() ||
            input.value.block != input.predecessor || !ValidValue(graph, input.value, phi.width))
          return invalid;
        const auto* predecessor = graph.Get(input.predecessor);
        const auto exit = std::lower_bound(
            predecessor->exits.begin(), predecessor->exits.end(), phi.storage,
            [](const SsaExitValue& value, StorageId storage) { return value.storage < storage; });
        if (exit == predecessor->exits.end() || exit->storage != phi.storage ||
            exit->value != input.value)
          return invalid;
        seen.push_back(input.predecessor);
      }
    }
  }

  std::optional<SsaStorageLiveness> liveness;
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    for (const auto& mark : graph.Get(*handle)->dead_storage_writes) {
      if (!liveness && !(liveness = SsaStorageLiveness::Compute(graph, budget)))
        return SsaDecline::resource_limit;
      const auto checked = liveness->CheckDeadWrite({*handle, mark.boundary, mark.index}, budget);
      if (checked != SsaDecline::none) return checked;
    }

    const auto& rewrites = graph.Get(*handle)->control_rewrites;
    if (std::any_of(rewrites.begin(), rewrites.end(), [](const ConditionalRewrite& rewrite) {
          return rewrite.rule == RewriteRule::bounded_exit;
        })) {
      const auto checked = BoundedExitRunsOnce(graph, *handle, budget);
      if (checked != SsaDecline::none) return checked;
    }

    // Which callee a retired call reached, and that it was a leaf returning
    // to the continuation, is evidence about the call as it was made; the
    // graph as it stands need only still carry the body it names.
    for (const auto& rewrite : rewrites)
      if (rewrite.rule == RewriteRule::retired_call && !BodyAt(graph, rewrite.when_true))
        return invalid;
    for (const auto& fold : graph.Get(*handle)->constant_loads) {
      if (fold.kind != SsaConstantKind::bounded_table) continue;
      const auto checked = TableGuardStillBounds(graph, *handle, fold, budget);
      if (checked != SsaDecline::none) return checked;
    }
  }

  return SsaDecline::none;
}

SsaDecline ValidateSsaRetiredCalls(const SsaGraph& graph, std::span<const std::uint32_t> slots,
                                   Budget& budget) {
  const auto valid = ValidateSsa(graph, budget);
  if (valid != SsaDecline::none) return valid;
  for (const auto slot : slots) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return SsaDecline::resource_limit;
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    for (const auto& rewrite : graph.Get(*handle)->control_rewrites) {
      if (rewrite.rule == RewriteRule::resolved_call) {
        bool exhausted = false;
        const auto& block = *graph.Get(*handle);
        if (TargetFoldsTo(block, rewrite.original.target, rewrite.when_true, budget, exhausted))
          continue;
        if (exhausted) return SsaDecline::resource_limit;
        const auto target = SsaSettledTarget(graph, block, rewrite.original.target, budget);
        if (!target || *target != rewrite.when_true) return SsaDecline::invalid_graph;
        continue;
      }

      if (rewrite.rule != RewriteRule::retired_call) continue;
      const auto checked = RetiredCallHolds(graph, *handle, rewrite, budget);
      if (checked != SsaDecline::none) return checked;
    }
  }

  return SsaDecline::none;
}

SsaDecline ValidateSsaRetiredWork(const SsaGraph& graph, Budget& budget) {
  const auto valid = ValidateSsa(graph, budget);
  if (valid != SsaDecline::none) return valid;
  std::optional<SsaStorageLiveness> liveness;
  std::optional<SsaFrameLiveness> frames;
  std::vector<StorageId> written;
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return SsaDecline::resource_limit;
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto& block = *graph.Get(*handle);
    const auto retired = std::find_if(block.control_rewrites.begin(), block.control_rewrites.end(),
                                      [](const ConditionalRewrite& rewrite) {
                                        return rewrite.rule == RewriteRule::bounded_exit ||
                                               rewrite.rule == RewriteRule::retired_call;
                                      });
    if (retired == block.control_rewrites.end()) continue;
    if (!liveness && !(liveness = SsaStorageLiveness::Compute(graph, budget)))
      return SsaDecline::resource_limit;
    // ValidateSsa has already required the one edge, to where control goes.
    const auto next = block.edges.front().target_block->slot;
    if (retired->rule == RewriteRule::retired_call) {
      StorageId link = 0;
      if (!LeafBody(*BodyAt(graph, retired->when_true), written, link, budget))
        return SsaDecline::invalid_graph;
      for (const auto storage : written)
        if (liveness->LiveIn(next, storage)) return SsaDecline::invalid_graph;
      continue;
    }

    if (!frames && !(frames = SsaFrameLiveness::Compute(graph, budget)))
      return SsaDecline::resource_limit;
    if (budget.try_consume({block.phis.size() + block.frame_phis.size(), 0}) != BudgetDecline::none)
      return SsaDecline::resource_limit;
    for (std::uint32_t index = 0; index < block.phis.size() && index < block.exits.size();
         ++index) {
      const SsaValue unchanged{SsaValueKind::phi, *handle, index};
      if (block.exits[index].value != unchanged &&
          liveness->LiveIn(next, block.phis[index].storage))
        return SsaDecline::invalid_graph;
    }

    for (std::uint32_t index = 0;
         index < block.frame_phis.size() && index < block.frame_exits.size(); ++index) {
      const SsaValue unchanged{SsaValueKind::frame_phi, *handle, index};
      if (block.frame_exits[index] != unchanged &&
          frames->LiveIn(next, block.frame_phis[index].offset, block.frame_phis[index].size))
        return SsaDecline::invalid_graph;
    }
  }

  return SsaDecline::none;
}

namespace {

struct AddressExpr {
  unsigned image_bases;
  std::uint64_t offset;

  // The value masked an image location to its page, which is its image value
  // only when the image is placed at a page-aligned address.
  bool placed = false;
};

constexpr std::uint64_t kPageMask = ~std::uint64_t{0xfff};

std::optional<AddressExpr> Address(std::span<const Node> nodes, ValueId id, unsigned& visits,
                                   bool allow_page = false) {
  if (id >= nodes.size() || ++visits > 64) return {};
  const auto& node = nodes[id];
  if (node.width != 64) return {};
  if (node.op == Op::image_address) return AddressExpr{1, node.immediate};
  if (node.op == Op::constant) return AddressExpr{0, node.immediate};

  // A same-width zext or low extract is a copy of its operand.
  if ((node.op == Op::zext || (node.op == Op::extract && !node.immediate)) &&
      node.inputs[0] < nodes.size() && nodes[node.inputs[0]].width == 64)
    return Address(nodes, node.inputs[0], visits, allow_page);
  if (allow_page && node.op == Op::bit_and) {
    auto left = Address(nodes, node.inputs[0], visits, allow_page);
    auto right = Address(nodes, node.inputs[1], visits, allow_page);
    if (!left || !right) return {};
    if (right->image_bases) std::swap(left, right);
    if (left->image_bases != 1 || right->image_bases || right->offset != kPageMask) return {};
    return AddressExpr{1, left->offset & kPageMask, true};
  }

  if (node.op != Op::add && node.op != Op::sub) return {};
  auto left = Address(nodes, node.inputs[0], visits, allow_page);
  auto right = Address(nodes, node.inputs[1], visits, allow_page);
  if (!left || !right || left->image_bases + right->image_bases > 1 ||
      (node.op == Op::sub && right->image_bases))
    return {};
  return AddressExpr{
      left->image_bases + right->image_bases,
      node.op == Op::add ? left->offset + right->offset : left->offset - right->offset,
      left->placed || right->placed};
}

bool ImageTarget(std::span<const Node> nodes, ValueId id, std::uint64_t& address, Budget& budget,
                 bool& exhausted) {
  if (budget.try_consume({64, 0}) != BudgetDecline::none) {
    exhausted = true;
    return false;
  }

  unsigned visits = 0;
  const auto value = Address(nodes, id, visits);
  if (!value || value->image_bases != 1) return false;
  address = value->offset;
  return true;
}

// The image location a load reads, when the whole address folds to one, and
// whether that needed page-aligned placement.
std::optional<std::pair<std::uint64_t, bool>> LoadLocation(std::span<const Node> nodes,
                                                           ValueId load,
                                                           std::optional<std::uint64_t> bias) {
  if (load >= nodes.size() || nodes[load].op != Op::load || !nodes[load].width ||
      nodes[load].width > 64 || nodes[load].width % 8)
    return std::nullopt;
  unsigned visits = 0;
  const auto address = Address(nodes, nodes[load].inputs[0], visits, true);
  if (!address) return std::nullopt;

  // With the placement declared, an address that is only a number names the
  // location standing at it; without one, a number is not a place.
  if (address->image_bases == 0)
    return bias && address->offset >= *bias
               ? std::optional(std::pair{address->offset - *bias, false})
               : std::nullopt;
  if (address->image_bases != 1) return std::nullopt;
  return std::pair{address->offset, address->placed};
}

bool SameTransferShape(const Transfer& a, const Transfer& b) {
  return a.kind == b.kind && a.target == b.target && a.condition == b.condition &&
         a.alternative == b.alternative && a.continuation == b.continuation;
}

// A pure value of a recovered path under one case of its dispatch condition:
// a literal, an image-relative address, or unknown.
struct Folded {
  enum class Kind : std::uint8_t { unknown, number, image } kind = Kind::unknown;
  std::uint64_t bits = 0;

  // An image value that rests on page-aligned placement; only a declared read
  // that records that declaration may use it.
  bool placed = false;

  // The value depends on a declared image read, so whatever it decides rests
  // on the constant-image declaration too.
  bool declared = false;
};

// `declared`, when given, records whether the value rests on a declared read.
bool FoldedImage(const std::vector<Folded>& values, ValueId id, std::uint64_t& address,
                 bool* declared = nullptr) {
  if (id >= values.size() || values[id].kind != Folded::Kind::image || values[id].placed)
    return false;
  address = values[id].bits;
  if (declared) *declared |= values[id].declared;
  return true;
}

// Folds every pure node from literals, image addresses and `fixed`; reads,
// loads and every other effect stay unknown. Each operation is evaluated at its
// own width, so a folded value is the node's value on every execution that
// reaches it with `fixed` holding, and with each `assigned` node holding its
// value. Destination nodes follow the path's nodes.
bool FoldPath(const SsaBlock& block, std::optional<std::pair<ValueId, bool>> fixed,
              std::vector<Folded>& values, Budget& budget, bool& exhausted,
              std::span<const std::pair<ValueId, Folded>> assigned = {}) {
  using Kind = Folded::Kind;
  const auto count = block.nodes.size() + block.destination_nodes.size();
  if (count > std::numeric_limits<std::uint32_t>::max() || assigned.size() > 64 ||
      budget.try_consume(
          {count * (1 + assigned.size()),
           count * (sizeof(Folded) + sizeof(std::pair<std::uint64_t, std::uint64_t>))}) !=
          BudgetDecline::none) {
    exhausted = true;
    return false;
  }

  values.assign(count, {});
  std::vector<std::pair<ValueId, bool>> implied;
  if (fixed) {
    if (fixed->first >= block.nodes.size() || block.nodes[fixed->first].width != 1) return false;
    implied = SsaImpliedBits(block.nodes, fixed->first, fixed->second);
  }

  // Image-located writes seen so far, as (location, bytes).
  std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
  writes.reserve(count);
  for (std::size_t id = 0; id < count; ++id) {
    const auto& node = id < block.nodes.size() ? block.nodes[id]
                                               : block.destination_nodes[id - block.nodes.size()];
    const auto given = std::find_if(assigned.begin(), assigned.end(),
                                    [&](const auto& item) { return item.first == id; });
    if (given != assigned.end()) {
      values[id] = given->second;
      continue;
    }

    const auto decided = std::find_if(implied.begin(), implied.end(),
                                      [&](const auto& bit) { return bit.first == id; });
    if (decided != implied.end()) {
      values[id] = {Kind::number, decided->second ? 1U : 0U};
      continue;
    }

    if (MayWriteMemory(node.op)) {
      // A write that provably reaches a declared read location contradicts the
      // declaration, so later reads of it stay unknown. A write to an unknown
      // address is excluded from declared locations by the run's own printed
      // declaration, exactly as when the path was recovered.
      if (node.inputs[0] >= id || node.inputs[1] >= id || node.inputs[1] >= block.nodes.size())
        return false;
      const auto base = values[node.inputs[0]];
      const auto bits = std::max(node.width, block.nodes[node.inputs[1]].width);
      if (base.kind == Kind::image) writes.push_back({base.bits, (bits + 7U) / 8U});
      continue;
    }

    if (!node.width || node.width > 64) continue;
    if (node.op == Op::constant) {
      values[id] = {Kind::number, node.immediate & LowMask(node.width)};
      continue;
    }

    if (node.op == Op::image_address) {
      if (node.width == 64) values[id] = {Kind::image, node.immediate};
      continue;
    }

    if (node.op == Op::load) {
      const auto read = std::lower_bound(
          block.path_reads.begin(), block.path_reads.end(), id,
          [](const SsaPathRead& item, std::size_t node) { return item.node < node; });
      const auto& location = values[node.inputs[0] < id ? node.inputs[0] : id];
      if (read == block.path_reads.end() || read->node != id || node.inputs[0] >= id ||
          location.kind != Kind::image || location.bits != read->address ||
          (location.placed && !read->page_aligned_placement) ||
          (read->relocated && (node.width != 64 || node.access.byte_order != ByteOrder::little)) ||
          read->value > LowMask(node.width))
        continue;
      const auto bytes = node.width / 8U;
      if (budget.try_consume(
              {1 + writes.size() +
                   static_cast<std::uint64_t>(std::bit_width(block.path_reads.size())),
               0}) != BudgetDecline::none) {
        exhausted = true;
        return false;
      }

      if (std::any_of(writes.begin(), writes.end(), [&](const auto& write) {
            // Compared as offsets from the read so a range near 2^64 cannot wrap.
            return write.second &&
                   (write.first <= read->address ? read->address - write.first < write.second
                                                 : write.first - read->address < bytes);
          }))
        continue;
      values[id] = {read->relocated ? Kind::image : Kind::number, read->value, false, true};
      continue;
    }

    const auto* descriptor = Descriptor(node.op);
    if (!descriptor || descriptor->effect != Effect::pure || !descriptor->produces_value ||
        !descriptor->arity)
      continue;
    std::array<Folded, 3> in{};
    std::array<std::uint64_t, 3> operands{};
    bool known = true;
    bool declared = false;
    for (unsigned input = 0; input < descriptor->arity; ++input) {
      // A path is in SSA order; anything else is not a path this rule reads.
      if (node.inputs[input] >= id || node.inputs[input] >= block.nodes.size()) return false;
      in[input] = values[node.inputs[input]];
      operands[input] = in[input].bits;
      known &= in[input].kind == Kind::number;
      declared |= in[input].declared;
    }

    // A decided select takes its arm whatever it holds, and a same-width copy
    // keeps an image location, so value-keeping rewrites still fold.
    if (node.op == Op::select && in[0].kind == Kind::number) {
      values[id] = in[in[0].bits ? 1 : 2];
      values[id].declared |= in[0].declared;
      continue;
    }

    if (node.width == 64 && in[0].kind == Kind::image && block.nodes[node.inputs[0]].width == 64 &&
        (node.op == Op::zext || (node.op == Op::extract && !node.immediate))) {
      values[id] = in[0];
      continue;
    }

    if (node.width == 64 && (node.op == Op::add || node.op == Op::sub)) {
      if (node.op == Op::add && in[0].kind == Kind::image && in[1].kind == Kind::number)
        values[id] = {Kind::image, in[0].bits + in[1].bits, in[0].placed, declared};
      else if (node.op == Op::add && in[0].kind == Kind::number && in[1].kind == Kind::image)
        values[id] = {Kind::image, in[0].bits + in[1].bits, in[1].placed, declared};
      else if (node.op == Op::sub && in[0].kind == Kind::image && in[1].kind == Kind::number)
        values[id] = {Kind::image, in[0].bits - in[1].bits, in[0].placed, declared};
      if (values[id].kind == Kind::image) continue;
    }

    if (node.width == 64 && node.op == Op::bit_and) {
      const auto& image = in[0].kind == Kind::image ? in[0] : in[1];
      const auto& mask = in[0].kind == Kind::image ? in[1] : in[0];
      if (image.kind == Kind::image && mask.kind == Kind::number && mask.bits == kPageMask) {
        values[id] = {Kind::image, image.bits & kPageMask, true, declared};
        continue;
      }
    }

    if (!known) continue;
    const auto& first = block.nodes[node.inputs[node.op == Op::select ? 1 : 0]];
    if (const auto folded =
            FoldPure(node, std::span(operands.data(), descriptor->arity), first.width))
      values[id] = {Kind::number, *folded, false, declared};
  }

  return true;
}

bool SingleSuccessor(const SsaBlock& block, std::uint64_t address) {
  if (block.edges.size() != 1) return false;
  const auto& edge = block.edges[0];
  return (edge.kind == SsaEdgeKind::branch || edge.kind == SsaEdgeKind::fallthrough) &&
         edge.target_kind == SsaTargetKind::image_location && edge.target_block &&
         edge.address == address && !edge.condition && !edge.when &&
         !edge.assumptions.unresolved_target;
}

bool TwoSuccessors(const SsaBlock& block, ValueId condition, std::uint64_t when_true,
                   std::uint64_t when_false) {
  if (block.edges.size() != 2) return false;
  bool seen_true = false, seen_false = false;
  for (const auto& edge : block.edges) {
    if ((edge.kind != SsaEdgeKind::branch && edge.kind != SsaEdgeKind::fallthrough) ||
        edge.target_kind != SsaTargetKind::image_location || !edge.target_block ||
        edge.condition != condition || !edge.when || edge.assumptions.unresolved_target)
      return false;
    auto& seen = *edge.when ? seen_true : seen_false;
    if (seen || edge.address != (*edge.when ? when_true : when_false)) return false;
    seen = true;
  }

  return seen_true && seen_false;
}

// A recovered transition is one path through several original instructions.
// Its successors are complete when, under each value of its dispatch condition
// (or once, without one), every internal transfer provably reaches the next
// source of the path and the final transfer reaches exactly the listed edges.
// The rule reads only the block's own nodes and recorded rewrites; which
// destinations a dispatch selects rests on the declared image values its
// edges carry, and execution rechecks the computed target against the edge.
bool CompleteTransition(const SsaBlock& block, Budget& budget, bool& exhausted) {
  const auto& boundaries = block.boundaries;
  if (block.source_bytes.size() != boundaries.size() ||
      block.source_groups.size() != boundaries.size())
    return false;
  const auto last = boundaries.size() - 1;
  if (budget.try_consume({boundaries.size() + block.control_rewrites.size(),
                          boundaries.size() * sizeof(const ConditionalRewrite*)}) !=
      BudgetDecline::none) {
    exhausted = true;
    return false;
  }

  std::vector<const ConditionalRewrite*> rewrites(boundaries.size());
  const ConditionalRewrite* dispatch = nullptr;

  // Whether that dispatch's condition has since become a literal, so the
  // block goes to one of its two destinations and keeps the other as the
  // provenance of the choice.
  bool decided = false;
  for (const auto& rewrite : block.control_rewrites) {
    if (rewrite.boundary >= boundaries.size() || rewrites[rewrite.boundary] ||
        !boundaries[rewrite.boundary].transfer ||
        !SameTransferShape(rewrite.original, *boundaries[rewrite.boundary].transfer) ||
        rewrite.condition >= block.nodes.size() || block.nodes[rewrite.condition].width != 1)
      return false;
    rewrites[rewrite.boundary] = &rewrite;
    const bool exit = rewrite.rule == RewriteRule::bounded_exit;
    if (exit && (rewrite.boundary != last || !rewrite.iterations ||
                 rewrite.iterations > kMaxBoundedLoopIterations))
      return false;
    if (rewrite.rule == RewriteRule::dispatch_branch ||
        rewrite.rule == RewriteRule::decided_dispatch ||
        (exit && rewrite.original.kind == TransferKind::jump)) {
      if (rewrite.boundary != last || dispatch) return false;
      dispatch = &rewrite;
      decided = rewrite.rule != RewriteRule::dispatch_branch;
    } else if (exit) {
      if (rewrite.original.kind != TransferKind::conditional ||
          rewrite.condition != rewrite.original.condition ||
          rewrite.replacement.kind != TransferKind::jump ||
          rewrite.replacement.target !=
              (rewrite.condition_value ? rewrite.original.target : *rewrite.original.alternative))
        return false;
    } else if (rewrite.rule != RewriteRule::folded_condition ||
               rewrite.original.kind != TransferKind::conditional ||
               rewrite.condition != rewrite.original.condition ||
               rewrite.replacement.kind != TransferKind::jump ||
               rewrite.replacement.target != (rewrite.condition_value
                                                  ? rewrite.original.target
                                                  : *rewrite.original.alternative)) {
      return false;
    }
  }

  std::optional<std::pair<std::uint64_t, std::uint64_t>> destinations;
  if (dispatch) {
    const auto& replacement = dispatch->replacement;
    const auto base = block.nodes.size();

    // A dispatch restores a conditional the table hid behind an indirect
    // jump. Over anything else the destinations it names are unrelated to
    // what the boundary does, and `folded_condition` is the rule for a
    // decoded conditional.
    if (dispatch->original.kind != TransferKind::jump) return false;
    if (decided) {
      // The condition is a number now, so the block has one successor. Both
      // destinations stay, and the refold below still runs under each value
      // of the condition, so what the record claims is still rechecked.
      const auto& literal = block.nodes[dispatch->condition];
      const bool exit = dispatch->rule == RewriteRule::bounded_exit;
      if ((!exit && (literal.op != Op::constant || literal.width != 1 ||
                     dispatch->condition_value != bool(literal.immediate & 1))) ||
          replacement.kind != TransferKind::jump || replacement.condition ||
          replacement.alternative || replacement.continuation || replacement.target < base ||
          replacement.target - base >= block.destination_nodes.size())
        return false;
      const auto taken = dispatch->condition_value ? dispatch->when_true : dispatch->when_false;
      const auto spurned = dispatch->condition_value ? dispatch->when_false : dispatch->when_true;

      // A loop's exit spurns the destination that is the loop itself.
      if (exit && spurned != block.address) return false;
      const auto& target = block.destination_nodes[replacement.target - base];
      if (target.op != Op::image_address || target.width != 64 || target.immediate != taken)
        return false;
      if (budget.try_consume({block.destination_nodes.size(), 0}) != BudgetDecline::none) {
        exhausted = true;
        return false;
      }

      if (std::none_of(block.destination_nodes.begin(), block.destination_nodes.end(),
                       [&](const Node& node) {
                         return node.op == Op::image_address && node.width == 64 &&
                                node.immediate == spurned;
                       }))
        return false;
      for (const auto& edge : block.edges)
        if (!edge.assumptions.constant_image) return false;
      if (!SingleSuccessor(block, taken)) return false;
      destinations = {dispatch->when_true, dispatch->when_false};
    } else if (replacement.kind != TransferKind::conditional || !replacement.condition ||
               *replacement.condition != dispatch->condition || !replacement.alternative ||
               replacement.target < base || *replacement.alternative < base ||
               replacement.target - base >= block.destination_nodes.size() ||
               *replacement.alternative - base >= block.destination_nodes.size()) {
      return false;
    } else {
      const auto& on_true = block.destination_nodes[replacement.target - base];
      const auto& on_false = block.destination_nodes[*replacement.alternative - base];
      if (on_true.op != Op::image_address || on_true.width != 64 ||
          on_false.op != Op::image_address || on_false.width != 64 ||
          on_true.immediate != dispatch->when_true || on_false.immediate != dispatch->when_false)
        return false;
      // The two destinations were read from declared image bytes; each edge
      // must carry that declaration forward.
      for (const auto& edge : block.edges)
        if (!edge.assumptions.constant_image) return false;
      if (!TwoSuccessors(block, dispatch->condition, on_true.immediate, on_false.immediate))
        return false;
      destinations = {on_true.immediate, on_false.immediate};
    }
  }

  std::vector<Folded> values;

  // Whether a declared image read decided any transfer; the edges then rest
  // on that declaration and must carry it.
  bool rests = false;
  std::optional<std::uint64_t> single;
  std::optional<std::pair<std::uint64_t, std::uint64_t>> undecided;
  for (unsigned value = 0; value < (dispatch ? 2U : 1U); ++value) {
    std::optional<std::pair<ValueId, bool>> fixed;
    if (dispatch) fixed = std::pair{dispatch->condition, value != 0};
    if (!FoldPath(block, fixed, values, budget, exhausted)) return false;
    for (std::size_t index = 0; index < boundaries.size(); ++index) {
      if (budget.try_consume({1, 0}) != BudgetDecline::none) {
        exhausted = true;
        return false;
      }

      if (block.source_bytes[index].empty() ||
          block.source_groups[index] > UINT64_MAX - block.source_bytes[index].size())
        return false;
      const auto end = block.source_groups[index] + block.source_bytes[index].size();
      const auto& transfer = boundaries[index].transfer;
      if (index == last) {
        // A dispatch states its destinations rather than recomputing them
        // here: the target is an indirect jump through a table the path only
        // resolves with the run's declared image values, which is what the
        // record carries and what each edge has to declare. Deciding one
        // changes which of the two it goes to, not how either was derived.
        if (dispatch) break;
        std::uint64_t target = 0, alternative = 0;
        if (!transfer) {
          single = end;
        } else if (transfer->kind == TransferKind::jump) {
          if (!FoldedImage(values, transfer->target, target, &rests)) return false;
          single = target;
        } else if (transfer->kind == TransferKind::conditional && rewrites[index]) {
          if (!FoldedImage(values, rewrites[index]->replacement.target, target, &rests))
            return false;
          if (rewrites[index]->rule == RewriteRule::bounded_exit) {
            // Which way the condition goes varies with the iteration; only
            // the way back into the block has to be the block.
            const auto& original = rewrites[index]->original;
            const auto back =
                rewrites[index]->condition_value ? *original.alternative : original.target;
            if (!FoldedImage(values, back, alternative, &rests) || alternative != block.address)
              return false;
            single = target;
            break;
          }

          const auto& condition = values[rewrites[index]->condition];
          rests |= condition.declared;
          if (condition.kind != Folded::Kind::number ||
              (condition.bits != 0) != rewrites[index]->condition_value)
            return false;
          single = target;
        } else if (transfer->kind == TransferKind::conditional) {
          if (!transfer->condition || !transfer->alternative ||
              !FoldedImage(values, transfer->target, target, &rests) ||
              !FoldedImage(values, *transfer->alternative, alternative, &rests))
            return false;
          const auto& condition = values[*transfer->condition];
          rests |= condition.declared;
          if (condition.kind == Folded::Kind::number)
            single = condition.bits ? target : alternative;
          else
            undecided = {target, alternative};
        } else {
          // Calls and returns end paths through their own rules, not here.
          return false;
        }

        break;
      }

      const auto next = block.source_groups[index + 1];
      if (!transfer) {
        if (end != next) return false;
        continue;
      }

      std::uint64_t target = 0;
      if (transfer->kind == TransferKind::jump) {
        if (!FoldedImage(values, transfer->target, target, &rests) || target != next) return false;
      } else if (transfer->kind == TransferKind::conditional) {
        if (!transfer->condition || !transfer->alternative) return false;
        const auto& condition =
            values[rewrites[index] ? rewrites[index]->condition : *transfer->condition];
        rests |= condition.declared;

        // The branch has to be decided in this case, and decided along the path.
        if (condition.kind != Folded::Kind::number ||
            (rewrites[index] && (condition.bits != 0) != rewrites[index]->condition_value) ||
            !FoldedImage(values, condition.bits ? transfer->target : *transfer->alternative, target,
                         &rests) ||
            target != next)
          return false;
      } else {
        return false;
      }
    }
  }

  if (rests && std::any_of(block.edges.begin(), block.edges.end(),
                           [](const SsaEdge& edge) { return !edge.assumptions.constant_image; }))
    return false;
  if (dispatch) return destinations.has_value();
  if (undecided) {
    const auto& transfer = *boundaries[last].transfer;
    return TwoSuccessors(block, *transfer.condition, undecided->first, undecided->second);
  }

  return single && SingleSuccessor(block, *single);
}

// Whether a call target folds, through the block's own path and declared
// image reads, to `expected`: the same evidence a known callee edge rests on.
bool TargetFoldsTo(const SsaBlock& block, ValueId target, std::uint64_t expected, Budget& budget,
                   bool& exhausted) {
  std::vector<Folded> values;
  bool declared = false;
  std::uint64_t folded = 0;
  return FoldPath(block, std::nullopt, values, budget, exhausted) &&
         FoldedImage(values, target, folded, &declared) && folded == expected;
}

// A call to a callee outside the graph: the continuation is its only successor
// in this population when the callee returns only there, and it has none when
// the caller declared every possible callee non-returning. An interior body is
// followed directly instead, with no potential-return edge on the call.
bool CompleteCall(const SsaBlock& block, const Transfer& transfer, Budget& budget,
                  bool& exhausted) {
  // A direct call whose body is in the graph transfers to that body, carrying
  // the link write just like every other decoded effect. No callee is skipped
  // and no return-to-continuation or ABI assumption is needed for this edge.
  if (block.control_rewrites.empty() && block.edges.size() == 1 &&
      block.edges.front().kind == SsaEdgeKind::branch) {
    std::uint64_t target = 0;
    const auto& edge = block.edges.front();
    return ImageTarget(block.nodes, transfer.target, target, budget, exhausted) &&
           edge.target_kind == SsaTargetKind::image_location && edge.target_block &&
           edge.address == target && !edge.condition && !edge.when &&
           !edge.assumptions.unresolved_target &&
           !edge.assumptions.callee_returns_to_continuation && !edge.assumptions.declared_noreturn;
  }

  std::uint64_t continuation = 0;

  // A callee is often read from a relocated slot; its declared value travels
  // with the block as a path read.
  std::vector<Folded> values;

  // A retired call goes straight to where it would have returned; that the
  // callee it names is a leaf whose writes nobody reads is the graph's to say.
  const ConditionalRewrite* retired = nullptr;
  const ConditionalRewrite* resolved = nullptr;
  if (block.control_rewrites.size() == 1 &&
      block.control_rewrites.front().rule == RewriteRule::retired_call)
    retired = &block.control_rewrites.front();
  if (block.control_rewrites.size() == 1 &&
      block.control_rewrites.front().rule == RewriteRule::resolved_call)
    resolved = &block.control_rewrites.front();
  if ((!block.control_rewrites.empty() && !retired && !resolved) || !transfer.continuation ||
      block.source_groups.size() != block.boundaries.size() ||
      block.source_bytes.size() != block.boundaries.size() || block.source_bytes.back().empty() ||
      !FoldPath(block, std::nullopt, values, budget, exhausted) ||
      !ImageTarget(block.nodes, *transfer.continuation, continuation, budget, exhausted) ||
      block.source_groups.back() > UINT64_MAX - block.source_bytes.back().size() ||
      continuation != block.source_groups.back() + block.source_bytes.back().size())
    return false;
  if (retired) {
    if (retired->boundary != block.boundaries.size() - 1 ||
        !SameTransferShape(retired->original, transfer) ||
        retired->original.target != transfer.target ||
        retired->original.continuation != transfer.continuation ||
        retired->condition != transfer.target || !retired->witness.empty() ||
        retired->replacement.kind != TransferKind::jump ||
        retired->replacement.target != *transfer.continuation || retired->replacement.condition ||
        retired->replacement.alternative || retired->replacement.continuation ||
        !retired->to_revision || retired->to_revision != block.path_revision ||
        retired->from_revision != retired->to_revision - 1 || block.edges.size() != 1)
      return false;
    const auto& edge = block.edges.front();
    return (edge.kind == SsaEdgeKind::branch || edge.kind == SsaEdgeKind::fallthrough) &&
           edge.target_kind == SsaTargetKind::image_location && edge.target_block &&
           edge.address == continuation && !edge.condition && !edge.when &&
           !edge.assumptions.unresolved_target;
  }

  std::uint64_t callee = 0;
  bool declared = false;
  if (resolved) {
    // The call goes to the destination the rewrite names, with the same
    // continuation; that the original target settles there is the graph's.
    const auto base = block.nodes.size();
    const auto& replacement = resolved->replacement;
    if (resolved->boundary != block.boundaries.size() - 1 ||
        !SameTransferShape(resolved->original, transfer) ||
        resolved->original.target != transfer.target ||
        resolved->original.continuation != transfer.continuation ||
        resolved->condition != transfer.target || !resolved->witness.empty() ||
        replacement.kind != TransferKind::call || replacement.condition ||
        replacement.alternative || replacement.continuation != transfer.continuation ||
        replacement.target < base || replacement.target - base >= block.destination_nodes.size() ||
        !resolved->to_revision || resolved->to_revision != block.path_revision ||
        resolved->from_revision != resolved->to_revision - 1)
      return false;
    const auto& destination = block.destination_nodes[replacement.target - base];
    if (destination.op != Op::image_address || destination.width != 64 ||
        destination.immediate != resolved->when_true)
      return false;
    callee = resolved->when_true;
  }

  const bool resolved_target = resolved || FoldedImage(values, transfer.target, callee, &declared);
  const SsaEdge* call = nullptr;
  const SsaEdge* back = nullptr;
  for (const auto& edge : block.edges) {
    if (edge.condition || edge.when || edge.assumptions.unresolved_target) return false;
    if (edge.kind == SsaEdgeKind::callee && !call)
      call = &edge;
    else if (edge.kind == SsaEdgeKind::potential_return && !back)
      back = &edge;
    else
      return false;
  }

  // Under the closed-entry declaration a callee, known or not, re-enters the
  // population only at a listed entry, which is already a root. So an unknown
  // callee covers any target, resolved or not; a named one must be the one the
  // target resolves to.
  if (!call || call->target_block ||
      (resolved && call->target_kind != SsaTargetKind::image_location) ||
      (call->target_kind != SsaTargetKind::unknown &&
       (!resolved_target || call->target_kind != SsaTargetKind::image_location ||
        call->address != callee || (declared && !call->assumptions.constant_image))))
    return false;
  if (!back) return call->assumptions.declared_noreturn;
  return !call->assumptions.declared_noreturn && back->target_block &&
         back->target_kind == SsaTargetKind::image_location && back->address == continuation &&
         back->assumptions.callee_returns_to_continuation;
}

// The edges the CFG builder gives a jump it proved goes one of several ways:
// plain branches to distinct blocks, none conditional or unresolved.
bool DispatchEdges(const SsaBlock& block, Budget& budget, bool& exhausted) {
  if (!block.control_rewrites.empty() || block.edges.size() < 2) return false;
  const std::uint64_t size = block.edges.size();
  if (size > UINT64_MAX / size || budget.try_consume({size * size, 0}) != BudgetDecline::none) {
    exhausted = true;
    return false;
  }

  for (std::size_t index = 0; index < block.edges.size(); ++index) {
    const auto& edge = block.edges[index];
    if (edge.kind != SsaEdgeKind::branch || edge.target_kind != SsaTargetKind::image_location ||
        !edge.target_block || edge.condition || edge.when || edge.assumptions.unresolved_target)
      return false;
    for (std::size_t other = 0; other < index; ++other)
      if (block.edges[other].address == edge.address) return false;
  }

  return true;
}

// A jump through a bounded table goes where the rows send it. The fold says
// the index is one of the rows and the load reads that row's declared bytes,
// which ValidateSsa rechecks against the guard that bounds it
// (TableGuardStillBounds). So the path folded under each row, index and load
// together, gives every target the jump can take, and the edges must be
// exactly those targets. The index and the load are set as a pair: either
// alone folds a path no run takes.
bool CompleteTableJump(const SsaBlock& block, const Transfer& transfer, Budget& budget,
                       bool& exhausted) {
  if (!DispatchEdges(block, budget, exhausted)) return false;
  std::vector<std::uint8_t> produced(block.edges.size());
  std::vector<Folded> values;
  for (const auto& fold : block.constant_loads) {
    if (budget.try_consume({1 + block.edges.size(), 0}) != BudgetDecline::none) {
      exhausted = true;
      return false;
    }

    if (fold.kind != SsaConstantKind::bounded_table || fold.node >= block.nodes.size() ||
        fold.table_index >= block.nodes.size() || fold.table_bytes.empty() ||
        fold.table_bytes.size() > 64)
      continue;
    const auto& load = block.nodes[fold.node];
    const auto& index = block.nodes[fold.table_index];
    if (load.op != Op::load || !load.width || load.width > 64 || load.width % 8 ||
        index.op != Op::read || index.width != 64)
      continue;
    std::fill(produced.begin(), produced.end(), 0);
    bool covers = true;
    for (std::size_t row = 0; covers && row < fold.table_bytes.size(); ++row) {
      if (budget.try_consume({1 + block.edges.size(), 0}) != BudgetDecline::none) {
        exhausted = true;
        return false;
      }

      std::uint64_t bits = 0;
      const auto size = load.width / 8;
      for (unsigned byte = 0; byte < size; ++byte) {
        const auto shift = load.access.byte_order == ByteOrder::little ? byte : size - 1 - byte;
        bits |= std::uint64_t(fold.table_bytes[row][byte]) << (shift * 8);
      }

      const std::array<std::pair<ValueId, Folded>, 2> assigned{
          {{fold.table_index, {Folded::Kind::number, row}},
           {fold.node, {Folded::Kind::number, bits, false, true}}}};
      std::uint64_t target = 0;
      bool declared = false;
      if (!FoldPath(block, std::nullopt, values, budget, exhausted, assigned)) return false;
      if (!FoldedImage(values, transfer.target, target, &declared)) {
        covers = false;
        break;
      }

      const auto edge = std::find_if(block.edges.begin(), block.edges.end(),
                                     [&](const SsaEdge& item) { return item.address == target; });
      if (edge == block.edges.end() || (declared && !edge->assumptions.constant_image)) {
        covers = false;
        break;
      }

      produced[edge - block.edges.begin()] = 1;
    }

    if (covers &&
        std::all_of(produced.begin(), produced.end(), [](std::uint8_t seen) { return seen != 0; }))
      return true;
  }

  return false;
}

// Direct control, recovered transitions and calls to callees outside the graph
// have edge-completeness rules; computed targets and opaque control do not.
bool CompleteSuccessors(const SsaBlock& block, Budget& budget, bool& exhausted) {
  if (block.boundaries.size() > std::numeric_limits<std::size_t>::max() - block.edges.size() ||
      budget.try_consume({block.boundaries.size() + block.edges.size(), 0}) !=
          BudgetDecline::none) {
    exhausted = true;
    return false;
  }

  if (block.opaque) {
    if (!block.boundaries.empty() || !block.nodes.empty() || block.edges.size() != 1) return false;
    const auto& edge = block.edges[0];
    if (edge.condition || edge.when || edge.assumptions.unresolved_target ||
        !edge.assumptions.declared_opaque_control)
      return false;
    // An opaque instruction the decoder classified as always trapping, such as
    // BRK, never completes; its handler can re-enter only at a listed entry.
    if (edge.kind == SsaEdgeKind::trap)
      return !edge.target_block && edge.target_kind == SsaTargetKind::unknown;
    // One it classified as reaching the next instruction has exactly that
    // successor: an unknown computation still has a known successor, and a
    // block left incomplete here fails the reachability proof for the whole
    // graph.
    return edge.kind == SsaEdgeKind::fallthrough &&
           edge.target_kind == SsaTargetKind::image_location && edge.target_block &&
           !block.source_groups.empty() &&
           block.source_bytes.size() == block.source_groups.size() &&
           !block.source_bytes.back().empty() &&
           block.source_groups.back() <= UINT64_MAX - block.source_bytes.back().size() &&
           edge.address == block.source_groups.back() + block.source_bytes.back().size();
  }

  if (block.boundaries.empty()) return false;
  if (block.transition) return CompleteTransition(block, budget, exhausted);
  for (std::size_t index = 0; index + 1 < block.boundaries.size(); ++index) {
    if (block.boundaries[index].transfer || block.source_bytes.size() != block.boundaries.size() ||
        block.source_bytes[index].empty() ||
        block.source_groups[index] > UINT64_MAX - block.source_bytes[index].size() ||
        block.source_groups[index] + block.source_bytes[index].size() !=
            block.source_groups[index + 1])
      return false;
  }

  const auto& boundary = block.boundaries.back();
  if (!boundary.transfer) {
    if (!block.control_rewrites.empty()) return false;
    if (block.source_bytes.size() != block.boundaries.size() || block.source_bytes.back().empty() ||
        block.edges.size() != 1 ||
        block.source_groups.back() > UINT64_MAX - block.source_bytes.back().size())
      return false;
    const auto& edge = block.edges[0];
    return edge.kind == SsaEdgeKind::fallthrough &&
           edge.target_kind == SsaTargetKind::image_location && edge.target_block &&
           edge.address == block.source_groups.back() + block.source_bytes.back().size() &&
           !edge.condition && !edge.when && !edge.assumptions.unresolved_target;
  }

  const auto& transfer = *boundary.transfer;
  switch (transfer.kind) {
    case TransferKind::jump: {
      if (block.control_rewrites.empty() && block.edges.size() > 1)
        return CompleteTableJump(block, transfer, budget, exhausted);
      if (!block.control_rewrites.empty() || block.edges.size() != 1) return false;
      std::uint64_t target = 0;

      // A target read from a relocated or declared-constant slot travels with
      // the block as a path read, as a callee's does; the edge then rests on
      // the same declaration the read does.
      bool declared = false;
      if (!ImageTarget(block.nodes, transfer.target, target, budget, exhausted)) {
        if (exhausted) return false;
        std::vector<Folded> values;
        if (!FoldPath(block, std::nullopt, values, budget, exhausted) ||
            !FoldedImage(values, transfer.target, target, &declared) || !declared)
          return false;
      }

      const auto& edge = block.edges[0];
      return (edge.kind == SsaEdgeKind::branch || edge.kind == SsaEdgeKind::fallthrough) &&
             edge.target_kind == SsaTargetKind::image_location && edge.target_block &&
             edge.address == target && !edge.condition && !edge.when &&
             !edge.assumptions.unresolved_target && (!declared || edge.assumptions.constant_image);
    }
    case TransferKind::conditional: {
      std::uint64_t target = 0, alternative = 0;
      if (!transfer.condition || !transfer.alternative ||
          !ImageTarget(block.nodes, transfer.target, target, budget, exhausted) ||
          !ImageTarget(block.nodes, *transfer.alternative, alternative, budget, exhausted))
        return false;
      if (!block.control_rewrites.empty()) {
        if (block.control_rewrites.size() != 1 || block.edges.size() != 1) return false;
        const auto& rewrite = block.control_rewrites.front();
        const bool exit = rewrite.rule == RewriteRule::bounded_exit;
        if (rewrite.boundary != block.boundaries.size() - 1 ||
            (rewrite.rule != RewriteRule::folded_condition && !exit) || !rewrite.to_revision ||
            rewrite.to_revision != block.path_revision ||
            rewrite.from_revision != rewrite.to_revision - 1 ||
            rewrite.original.kind != transfer.kind || rewrite.original.target != transfer.target ||
            rewrite.original.condition != transfer.condition ||
            rewrite.original.alternative != transfer.alternative ||
            rewrite.original.continuation != transfer.continuation ||
            rewrite.condition != *transfer.condition || rewrite.condition >= block.nodes.size() ||
            !rewrite.witness.empty() || rewrite.when_true || rewrite.when_false ||
            rewrite.replacement.kind != TransferKind::jump || rewrite.replacement.condition ||
            rewrite.replacement.alternative || rewrite.replacement.continuation)
          return false;
        if (rewrite.replacement.target !=
            (rewrite.condition_value ? transfer.target : *transfer.alternative))
          return false;
        if (exit) {
          // The other way leads back into this block; that it stops doing so
          // is the loop's record, rechecked with the graph (BoundedExitHolds).
          if ((rewrite.condition_value ? alternative : target) != block.address ||
              !rewrite.iterations || rewrite.iterations > kMaxBoundedLoopIterations)
            return false;
        } else {
          const auto& condition = block.nodes[rewrite.condition];
          if (condition.op != Op::constant || condition.width != 1 ||
              rewrite.condition_value != bool(condition.immediate & 1))
            return false;
        }

        const auto& edge = block.edges.front();
        return (edge.kind == SsaEdgeKind::branch || edge.kind == SsaEdgeKind::fallthrough) &&
               edge.target_kind == SsaTargetKind::image_location && edge.target_block &&
               edge.address == (rewrite.condition_value ? target : alternative) &&
               !edge.condition && !edge.when && !edge.assumptions.unresolved_target;
      }

      if (block.edges.size() != 2) return false;
      bool seen_true = false, seen_false = false;
      for (const auto& edge : block.edges) {
        if ((edge.kind != SsaEdgeKind::branch && edge.kind != SsaEdgeKind::fallthrough) ||
            edge.target_kind != SsaTargetKind::image_location || !edge.target_block ||
            edge.condition != transfer.condition || !edge.when ||
            edge.assumptions.unresolved_target)
          return false;
        if (*edge.when) {
          if (seen_true || edge.address != target) return false;
          seen_true = true;
        } else {
          if (seen_false || edge.address != alternative) return false;
          seen_false = true;
        }
      }

      return seen_true && seen_false;
    }
    case TransferKind::return_:
      if (!block.control_rewrites.empty()) return false;
      return block.edges.size() == 1 && block.edges[0].kind == SsaEdgeKind::return_ &&
             !block.edges[0].target_block && block.edges[0].assumptions.return_leaves &&
             !block.edges[0].assumptions.unresolved_target;
    case TransferKind::call:
      return CompleteCall(block, transfer, budget, exhausted);
  }

  return false;
}
}  // namespace

SsaDecline ValidateSsaDirectSuccessors(const SsaBlock& block, Budget& budget) {
  bool exhausted = false;
  if (CompleteSuccessors(block, budget, exhausted)) return SsaDecline::none;
  return exhausted ? SsaDecline::resource_limit : SsaDecline::invalid_graph;
}

SsaDecline ValidateSsaDispatchSuccessors(const SsaBlock& block, Budget& budget) {
  bool exhausted = false;
  if (CompleteSuccessors(block, budget, exhausted)) return SsaDecline::none;
  if (exhausted) return SsaDecline::resource_limit;
  if (block.opaque || block.transition || block.boundaries.empty() ||
      !block.boundaries.back().transfer ||
      block.boundaries.back().transfer->kind != TransferKind::jump)
    return SsaDecline::invalid_graph;
  if (budget.try_consume({block.boundaries.size(), 0}) != BudgetDecline::none)
    return SsaDecline::resource_limit;
  for (std::size_t index = 0; index + 1 < block.boundaries.size(); ++index)
    if (block.boundaries[index].transfer) return SsaDecline::invalid_graph;
  if (DispatchEdges(block, budget, exhausted)) return SsaDecline::none;
  return exhausted ? SsaDecline::resource_limit : SsaDecline::invalid_graph;
}

std::vector<std::pair<ValueId, bool>> SsaImpliedBits(std::span<const Node> nodes, ValueId condition,
                                                     bool value) {
  std::vector<std::pair<ValueId, bool>> implied;
  for (auto at = condition; at < nodes.size() && nodes[at].width == 1 && implied.size() < 64;) {
    implied.push_back({at, value});
    const auto& node = nodes[at];
    const bool copy = node.op == Op::zext || (node.op == Op::extract && !node.immediate);
    if ((node.op != Op::bit_not && !copy) || node.inputs[0] >= at) break;
    if (node.op == Op::bit_not) value = !value;
    at = node.inputs[0];
  }

  return implied;
}

namespace {

// `cycled` records that the answer rested on an entry still being computed,
// so it holds only while that entry's frame is open and cannot be reused once
// it closes. Without the distinction an intermediate value on a loop answers
// "unknown" and is indistinguishable from one that really is unknown.
struct SettledEntry {
  std::uint64_t key;
  std::optional<std::uint64_t> value;
  bool pending;
  bool cycled;
};

struct SettledAnswer {
  std::optional<std::uint64_t> value;
  bool cycled = false;
};

const SsaRead* ReadOf(const SsaBlock& block, ValueId id) {
  const auto at =
      std::lower_bound(block.reads.begin(), block.reads.end(), id,
                       [](const SsaRead& read, ValueId key) { return read.node < key; });
  return at != block.reads.end() && at->node == id ? &*at : nullptr;
}

const SsaConstantLoad* FoldOf(const SsaBlock& block, ValueId id) {
  const auto at =
      std::lower_bound(block.constant_loads.begin(), block.constant_loads.end(), id,
                       [](const SsaConstantLoad& fold, ValueId key) { return fold.node < key; });
  return at != block.constant_loads.end() && at->node == id ? &*at : nullptr;
}

const SsaFrameAccess* FrameOf(const SsaBlock& block, ValueId id) {
  const auto at =
      std::lower_bound(block.frame_accesses.begin(), block.frame_accesses.end(), id,
                       [](const SsaFrameAccess& access, ValueId key) { return access.node < key; });
  return at != block.frame_accesses.end() && at->node == id ? &*at : nullptr;
}

SettledAnswer Settled(const SsaGraph& graph, const SsaValue& value,
                      std::vector<SettledEntry>& settled, std::uint64_t& visits, Budget& budget,
                      unsigned depth) {
  // A value can be formed far from where it is used, so the walk is long and
  // rejoins itself: each value is answered once and the answer reused, or
  // two paths meeting at a merge would look like a contradiction. A phi
  // reached while it is still being answered carries the value round a loop
  // unchanged, so it constrains nothing.
  if (depth == 0) {
    settled.clear();
    visits = 0;
  }

  if (depth > 512 || settled.size() > 8192) return {};
  if (budget.try_consume({1 + settled.size(), 0}) != BudgetDecline::none) return {};
  const auto* block = graph.Get(value.block);
  if (!block) return {};
  const auto key = (static_cast<std::uint64_t>(value.block.slot) << 35) |
                   (static_cast<std::uint64_t>(value.kind) << 32) | value.index;
  auto index = settled.size();
  for (std::size_t i = 0; i < settled.size(); ++i) {
    if (settled[i].key != key) continue;

    // Re-entering an open frame is the value coming back round the loop.
    if (settled[i].pending) return {std::nullopt, true};
    if (!settled[i].cycled) return {settled[i].value, false};
    index = i;
    settled[i].pending = true;
    break;
  }

  // Recomputing is the cost this bounds, so an answer served from the memo
  // is not one of them. Giving up says unknown, which only loses names.
  if (++visits > (1U << 16)) return {};
  if (index == settled.size()) settled.push_back({key, std::nullopt, true, false});
  bool cycled = false;
  const auto answer = [&]() -> std::optional<std::uint64_t> {
    switch (value.kind) {
      case SsaValueKind::node: {
        if (value.index >= block->nodes.size()) return std::nullopt;
        const auto id = SsaCopySource(block->nodes, value.index);
        const auto& node = block->nodes[id];
        if (node.op == Op::image_address) return node.immediate;

        // A number names a place only once the run says where the image is.
        // This is the same reading the transfer target gets, so a value that
        // settles through a slot resolves where one held directly does.
        if (node.op == Op::constant && graph.load_bias() && node.width == 64 &&
            node.immediate >= *graph.load_bias())
          return node.immediate - *graph.load_bias();
        const auto through = [&](const SsaValue& source) {
          const auto one = Settled(graph, source, settled, visits, budget, depth + 1);
          cycled = cycled || one.cycled;
          return one.value;
        };

        // A place plus a literal offset is another place in the image.
        if (node.op == Op::add && node.width == 64) {
          for (unsigned side = 0; side < 2; ++side) {
            const auto offset = SsaCopySource(block->nodes, node.inputs[side]);
            if (offset >= block->nodes.size() || block->nodes[offset].op != Op::constant) continue;

            // A base still being resolved round a loop is a value the add moves
            // on, not this value coming back, so an unknown base makes the sum
            // unknown rather than merely cycled.
            const auto base = through({SsaValueKind::node, value.block, node.inputs[1 - side]});
            if (!base) {
              cycled = false;
              return std::nullopt;
            }

            return *base + block->nodes[offset].immediate;
          }

          return std::nullopt;
        }

        if (node.op == Op::read) {
          const auto* read = ReadOf(*block, id);
          return read ? through(read->value) : std::nullopt;
        }

        if (const auto* frame = FrameOf(*block, id); frame && frame->replacement)
          return through(*frame->replacement);
        // A folded load reads as the location its record names.
        // Only a fold that disabled the access has its value checked against
        // the declared bytes; without that the record's value is tied to
        // nothing and reading it would believe whatever it says.
        if (const auto* fold = FoldOf(*block, id); fold && fold->skip_access && !fold->condition &&
                                                   fold->kind == SsaConstantKind::image_location)
          return fold->value;
        return std::nullopt;
      }
      case SsaValueKind::phi:
      case SsaValueKind::frame_phi: {
        // A block the graph is entered at also receives whatever the caller
        // held, and that path contributes no incoming entry, so the ones
        // listed are not all of them.
        const bool external =
            value.kind == SsaValueKind::phi
                ? value.index < block->phis.size() && block->phis[value.index].external_entry
                : value.index < block->frame_phis.size() &&
                      block->frame_phis[value.index].external_entry;
        if (external) return std::nullopt;
        const auto incoming = [&]() -> std::span<const SsaPhiInput> {
          if (value.kind == SsaValueKind::phi)
            return value.index < block->phis.size() ? block->phis[value.index].incoming
                                                    : std::span<const SsaPhiInput>{};
          return value.index < block->frame_phis.size() ? block->frame_phis[value.index].incoming
                                                        : std::span<const SsaPhiInput>{};
        }();
        if (incoming.empty()) return std::nullopt;
        std::optional<std::uint64_t> agreed;
        bool any = false;
        for (const auto& input : incoming) {
          const auto one = Settled(graph, input.value, settled, visits, budget, depth + 1);
          cycled = cycled || one.cycled;

          // The loop's own value coming back, however many blocks it passed
          // through on the way. It equals whatever the other inputs agree on,
          // so it rules nothing out; an input that changed it on the way is
          // not this: a value formed by an operation is resolved on its own
          // and disagrees.
          if (!one.value) {
            if (one.cycled) continue;

            // This input is unknown independently of any open frame, so the
            // answer is unknown however the loop resolves.
            // Leaving the flag set would let a caller skip this phi as the
            // loop's own value and agree with the inputs that are left.
            cycled = false;
            return std::nullopt;
          }

          if (any && *agreed != *one.value) {
            cycled = false;
            return std::nullopt;
          }

          agreed = one.value;
          any = true;
        }

        return any ? agreed : std::nullopt;
      }
      default:
        return std::nullopt;
    }
  }();
  settled[index].value = answer;
  settled[index].pending = false;
  settled[index].cycled = cycled;
  return {answer, cycled};
}

}  // namespace

std::optional<std::uint64_t> SsaSettledLocation(const SsaGraph& graph, const SsaValue& value,
                                                Budget& budget) {
  std::vector<SettledEntry> settled;
  std::uint64_t visits = 0;
  return Settled(graph, value, settled, visits, budget, 0).value;
}

// What a block's transfer target settles to: the location a computed call
// reaches, when every way of arriving computes the same one.
std::optional<std::uint64_t> SsaSettledTarget(const SsaGraph& graph, const SsaBlock& block,
                                              ValueId target, Budget& budget) {
  if (target >= block.nodes.size()) return std::nullopt;
  const auto id = SsaCopySource(block.nodes, target);
  const auto& node = block.nodes[id];
  if (node.op == Op::image_address) return node.immediate;
  if (node.op == Op::constant && graph.load_bias() && node.width == 64 &&
      node.immediate >= *graph.load_bias())
    return node.immediate - *graph.load_bias();
  // A place plus a literal offset is another place in the image.
  if (node.op == Op::add && node.width == 64) {
    for (unsigned side = 0; side < 2; ++side) {
      const auto offset = SsaCopySource(block.nodes, node.inputs[side]);
      if (offset >= block.nodes.size() || block.nodes[offset].op != Op::constant ||
          node.inputs[1 - side] >= id)
        continue;
      const auto base = SsaSettledTarget(graph, block, node.inputs[1 - side], budget);
      return base ? std::optional(*base + block.nodes[offset].immediate) : std::nullopt;
    }

    return std::nullopt;
  }

  std::vector<SettledEntry> settled;
  std::uint64_t visits = 0;
  if (node.op == Op::read) {
    const auto* read = ReadOf(block, id);
    return read ? Settled(graph, read->value, settled, visits, budget, 0).value : std::nullopt;
  }

  if (const auto* frame = FrameOf(block, id); frame && frame->replacement)
    return Settled(graph, *frame->replacement, settled, visits, budget, 0).value;
  return std::nullopt;
}

std::optional<std::uint64_t> SsaCallTarget(const SsaGraph& graph, const SsaBlock& block,
                                           Budget& budget) {
  // Only a block whose single callee edge names one known location has a
  // callee to attribute anything to.
  const SsaEdge* call = nullptr;
  for (const auto& edge : block.edges) {
    if (edge.kind != SsaEdgeKind::callee) continue;
    if (call) return std::nullopt;
    call = &edge;
  }

  if (!call || call->target_block) return std::nullopt;
  if (call->target_kind == SsaTargetKind::image_location) return call->address;

  // A call through a register names no target on its edge. The value it
  // transfers to may still be settled, and that is the callee reached.
  if (call->target_kind != SsaTargetKind::unknown || block.boundaries.empty() ||
      !block.boundaries.back().transfer)
    return std::nullopt;
  return SsaSettledTarget(graph, block, block.boundaries.back().transfer->target, budget);
}

std::optional<std::uint64_t> SsaClobberResult(const SsaGraph& graph, const SsaValue& clobber) {
  if (clobber.kind != SsaValueKind::clobber || graph.call_results().empty()) return std::nullopt;
  const auto* block = graph.Get(clobber.block);
  if (!block || clobber.index >= block->phis.size()) return std::nullopt;
  Budget walk({1 << 20, 1 << 20});
  const auto reached = SsaCallTarget(graph, *block, walk);
  if (!reached) return std::nullopt;
  const auto target = *reached;
  const auto storage = block->phis[clobber.index].storage;
  const auto results = graph.call_results();
  const auto at = std::lower_bound(results.begin(), results.end(), std::pair{target, storage},
                                   [](const SsaCallResult& result, const auto& key) {
                                     return std::pair{result.target, result.storage} < key;
                                   });
  if (at == results.end() || at->target != target || at->storage != storage) return std::nullopt;
  return at->value;
}

std::optional<std::vector<std::uint8_t>> SsaDispatchDependent(const SsaBlock& block,
                                                              SsaHandle handle, Budget& budget) {
  const ConditionalRewrite* dispatch = nullptr;
  for (const auto& rewrite : block.control_rewrites)
    if (rewrite.rule == RewriteRule::dispatch_branch ||
        rewrite.rule == RewriteRule::decided_dispatch ||
        (rewrite.rule == RewriteRule::bounded_exit && rewrite.original.kind == TransferKind::jump))
      dispatch = &rewrite;
  if (!dispatch || dispatch->condition >= block.nodes.size()) return std::vector<std::uint8_t>{};
  if (budget.try_consume({2 * block.nodes.size(), 2 * block.nodes.size()}) != BudgetDecline::none)
    return std::nullopt;
  // Each refold writes its fixed bit over the condition and over everything
  // SsaImpliedBits reaches backwards from it, so a literal in any of those
  // changes neither record. Which nodes those are does not depend on the
  // value, only on their operations, so one call names them all.
  std::vector<std::uint8_t> fixed(block.nodes.size());
  for (const auto& [id, value] : SsaImpliedBits(block.nodes, dispatch->condition, true))
    if (id < fixed.size()) fixed[id] = 1;
  auto dependent = fixed;

  // What is computed from a fixed node does differ between the refolds. It
  // can reach a use through an input, through a read of storage this block
  // wrote, or through a promoted frame slot, and only the first of those
  // follows node order -- so this repeats until nothing new is marked.
  for (bool changed = true; changed;) {
    changed = false;
    if (budget.try_consume({block.nodes.size() + block.reads.size() + block.frame_accesses.size(),
                            0}) != BudgetDecline::none)
      return std::nullopt;
    const auto mark = [&](ValueId id) {
      if (id < dependent.size() && !dependent[id]) {
        dependent[id] = 1;
        changed = true;
      }
    };

    // Only a node of this block: other kinds are entry values, which no
    // refold changes, and a record naming another block's node carries an
    // index that means nothing here.
    const auto names = [&](const SsaValue& value) {
      return value.kind == SsaValueKind::node && value.block == handle &&
             value.index < dependent.size() && dependent[value.index];
    };

    for (ValueId id = 0; id < block.nodes.size(); ++id) {
      if (dependent[id]) continue;
      const auto* descriptor = Descriptor(block.nodes[id].op);
      if (!descriptor) continue;
      for (unsigned input = 0; input < descriptor->arity; ++input) {
        const auto source = block.nodes[id].inputs[input];
        if (source < dependent.size() && dependent[source]) mark(id);
      }
    }

    for (const auto& read : block.reads)
      if (names(read.value)) mark(read.node);
    for (const auto& access : block.frame_accesses)
      if (access.replacement && names(*access.replacement)) mark(access.node);
  }

  for (ValueId id = 0; id < dependent.size(); ++id)
    if (fixed[id]) dependent[id] = 0;
  return dependent;
}

ValueId SsaCopySource(std::span<const Node> nodes, ValueId id) {
  for (unsigned step = 0; step < 64 && id < nodes.size(); ++step) {
    const auto& node = nodes[id];
    if ((node.op != Op::zext && !(node.op == Op::extract && !node.immediate)) ||
        node.inputs[0] >= nodes.size() || nodes[node.inputs[0]].width != node.width)
      break;
    id = node.inputs[0];
  }

  return id;
}

std::optional<std::vector<StorageId>> SsaLeafCalleeReads(const SsaGraph& graph,
                                                         const SsaBlock& block, Budget& budget) {
  if (std::none_of(block.edges.begin(), block.edges.end(),
                   [](const SsaEdge& edge) { return edge.kind == SsaEdgeKind::callee; }))
    return std::nullopt;
  const auto target = SsaCallTarget(graph, block, budget);
  const auto* body = target ? BodyAt(graph, *target) : nullptr;
  std::vector<StorageId> written;
  StorageId link = 0;
  if (!body || !LeafBody(*body, written, link, budget) || block.boundaries.empty() ||
      !block.boundaries.back().transfer || !block.boundaries.back().transfer->continuation)
    return std::nullopt;
  // Only a body that returns through the register the call links returns to
  // the call's continuation; any other is not the leaf this claims it is.
  const auto continuation =
      SsaCopySource(block.nodes, *block.boundaries.back().transfer->continuation);
  if (std::none_of(block.boundaries.back().writes.begin(), block.boundaries.back().writes.end(),
                   [&](const Write& write) {
                     return write.storage == link &&
                            SsaCopySource(block.nodes, write.value) == continuation;
                   }))
    return std::nullopt;
  std::vector<StorageId> reads, seen;
  for (const auto& group : body->groups) {
    for (const auto& node : group.nodes())
      if (node.op == Op::read && std::find(seen.begin(), seen.end(), node.storage) == seen.end())
        reads.push_back(node.storage);
    for (const auto& node : group.nodes())
      if (node.op == Op::write) seen.push_back(node.storage);
    for (const auto& write : group.writes()) seen.push_back(write.storage);
  }

  std::sort(reads.begin(), reads.end());
  reads.erase(std::unique(reads.begin(), reads.end()), reads.end());
  return reads;
}

std::optional<std::vector<std::uint8_t>> SsaLiveValues(const SsaGraph& graph, SsaHandle handle,
                                                       Budget& budget) {
  const auto* block = graph.Get(handle);
  if (!block) return std::nullopt;
  const auto& nodes = block->nodes;
  if (budget.try_consume({1 + nodes.size() + block->boundaries.size() + block->reads.size() +
                              block->exits.size() + block->edges.size(),
                          nodes.size() + 8}) != BudgetDecline::none)
    return std::nullopt;
  std::vector<std::uint8_t> live(nodes.size());
  std::vector<ValueId> stack;
  const auto root = [&](ValueId id) {
    if (id < nodes.size() && !live[id]) {
      live[id] = 1;
      stack.push_back(id);
    }
  };

  const auto here = [&](const SsaValue& value) {
    if (value.kind == SsaValueKind::node && value.block == handle) root(value.index);
  };

  const auto listed = [&](const std::vector<ValueId>& sorted, ValueId id) {
    return std::binary_search(sorted.begin(), sorted.end(), id);
  };

  for (ValueId id = 0; id < nodes.size(); ++id) {
    const auto& node = nodes[id];
    if (!HasMemoryOrMonitorEffect(node.op)) continue;
    const auto* descriptor = Descriptor(node.op);
    if (descriptor && descriptor->arity) root(node.inputs[0]);
    if (MayWriteMemory(node.op) && !listed(block->disabled_effects, id) && descriptor->arity > 1)
      root(node.inputs[1]);
  }

  for (std::size_t index = 0; index < block->boundaries.size(); ++index) {
    const auto& boundary = block->boundaries[index];

    // A write node always commits; only boundary writes can be retired.
    for (auto id = boundary.first_node; id < boundary.first_node + boundary.node_count; ++id)
      if (id < nodes.size() && nodes[id].op == Op::write) root(nodes[id].inputs[0]);
    for (std::uint32_t at = 0; at < boundary.writes.size(); ++at)
      if (!std::any_of(block->dead_storage_writes.begin(), block->dead_storage_writes.end(),
                       [&](const SsaDeadStorageWrite& item) {
                         return item.boundary == index && item.index == at;
                       }))
        root(boundary.writes[at].value);
    const auto rewrite =
        std::find_if(block->control_rewrites.begin(), block->control_rewrites.end(),
                     [&](const ConditionalRewrite& item) { return item.boundary == index; });
    const auto* transfer = rewrite != block->control_rewrites.end() ? &rewrite->replacement
                           : boundary.transfer                      ? &*boundary.transfer
                                                                    : nullptr;
    if (!transfer) continue;
    root(transfer->target);
    if (transfer->condition) root(*transfer->condition);
    if (transfer->alternative) root(*transfer->alternative);
    if (transfer->continuation) root(*transfer->continuation);
  }

  for (const auto& read : block->reads) {
    if (!listed(block->dead_pure_nodes, read.node)) here(read.value);
    if (read.predecessor_copy) here(*read.predecessor_copy);
  }

  // An exit whose storage's last write here was retired as dead publishes a
  // value nothing observes.
  for (const auto& exit : block->exits) {
    std::optional<SsaDeadStorageWrite> last;
    for (std::uint32_t index = 0; index < block->boundaries.size(); ++index) {
      const auto& writes = block->boundaries[index].writes;
      for (std::uint32_t at = 0; at < writes.size(); ++at)
        if (writes[at].storage == exit.storage) last = SsaDeadStorageWrite{index, at};
    }

    const bool retired =
        last && std::any_of(block->dead_storage_writes.begin(), block->dead_storage_writes.end(),
                            [&](const SsaDeadStorageWrite& item) {
                              return item.boundary == last->boundary && item.index == last->index;
                            });
    if (!retired) here(exit.value);
  }

  for (const auto& exit : block->frame_exits) here(exit);
  for (const auto& access : block->frame_accesses)
    if (access.replacement) here(*access.replacement);
  for (const auto& fold : block->constant_loads) {
    if (fold.condition) root(*fold.condition);
    if (fold.kind == SsaConstantKind::bounded_table) root(fold.table_index);
  }

  for (const auto& edge : block->edges)
    if (edge.condition) root(*edge.condition);
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    const auto other = graph.Handle(slot);
    if (!other) continue;
    const auto& successor = *graph.Get(*other);
    if (budget.try_consume({1 + successor.frame_phis.size() + successor.reads.size(), 0}) !=
        BudgetDecline::none)
      return std::nullopt;
    // A register phi only names what the storage holds; execution reads the
    // storage, which a kept write already roots. A promoted frame slot has no
    // storage behind it, so its phis carry the value.
    for (const auto& phi : successor.frame_phis)
      for (const auto& input : phi.incoming) here(input.value);
    if (*other == handle) continue;
    for (const auto& read : successor.reads) {
      here(read.value);
      if (read.predecessor_copy) here(*read.predecessor_copy);
    }
  }
  while (!stack.empty()) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return std::nullopt;
    const auto id = stack.back();
    stack.pop_back();
    const auto& node = nodes[id];
    if (node.op == Op::read) {
      const auto read = std::find_if(block->reads.begin(), block->reads.end(),
                                     [&](const SsaRead& item) { return item.node == id; });
      if (read != block->reads.end()) here(read->value);
      continue;
    }

    // A load's value does not need its address beyond the access root above,
    // and an effect's inputs are roots already.
    if (HasMemoryOrMonitorEffect(node.op)) continue;
    const auto* descriptor = Descriptor(node.op);
    for (unsigned input = 0; descriptor && input < descriptor->arity; ++input)
      root(node.inputs[input]);
  }

  return live;
}

bool SsaSameValue(const SsaBlock& block, ValueId a, ValueId b) {
  const auto& nodes = block.nodes;
  const auto same = [&](auto&& self, ValueId x, ValueId y, unsigned depth) -> bool {
    if (x >= nodes.size() || y >= nodes.size()) return false;
    x = SsaCopySource(nodes, x);
    y = SsaCopySource(nodes, y);
    if (x == y) return true;
    if (depth == 16) return false;
    const auto& left = nodes[x];
    const auto& right = nodes[y];
    if (left.op != right.op || left.width != right.width) return false;
    if (left.op == Op::constant || left.op == Op::image_address)
      return (left.immediate & LowMask(left.width)) == (right.immediate & LowMask(right.width));
    if (left.op == Op::read) {
      const auto find = [&](ValueId id) {
        return std::find_if(block.reads.begin(), block.reads.end(),
                            [&](const SsaRead& item) { return item.node == id; });
      };

      const auto first = find(x), second = find(y);
      return first != block.reads.end() && second != block.reads.end() &&
             first->value == second->value;
    }

    const auto* descriptor = Descriptor(left.op);
    if (!descriptor || descriptor->effect != Effect::pure || left.immediate != right.immediate)
      return false;
    for (unsigned input = 0; input < descriptor->arity; ++input)
      if (!self(self, left.inputs[input], right.inputs[input], depth + 1)) return false;
    return true;
  };

  return same(same, a, b, 0);
}

std::optional<std::pair<std::uint64_t, std::uint64_t>> SsaLoadImageSpan(
    const SsaBlock& block, ValueId load, std::optional<std::uint64_t> bias) {
  const auto& nodes = block.nodes;
  if (load >= nodes.size() || nodes[load].op != Op::load || !nodes[load].width ||
      nodes[load].width > 64 || nodes[load].width % 8)
    return std::nullopt;

  // Image arithmetic over literals and one 0/1 choice, evaluated per choice.
  struct Value {
    bool image = false;
    std::uint64_t bits = 0;
  };

  std::optional<ValueId> choice;
  const auto evaluate = [&](auto&& self, ValueId id, std::uint64_t pick,
                            unsigned depth) -> std::optional<Value> {
    if (depth > 32 || id >= nodes.size()) return std::nullopt;
    id = SsaCopySource(nodes, id);
    const auto& node = nodes[id];
    if (node.op == Op::image_address && node.width == 64) return Value{true, node.immediate};
    if (node.op == Op::constant && node.width <= 64)
      return Value{false, node.immediate & LowMask(node.width)};
    const auto* descriptor = Descriptor(node.op);
    if (node.width == 1 || !descriptor || descriptor->effect != Effect::pure ||
        !descriptor->produces_value || !descriptor->arity || node.op == Op::read ||
        node.width > 64) {
      if (node.width != 1 || (choice && *choice != id)) return std::nullopt;
      choice = id;
      return Value{false, pick};
    }

    std::array<Value, 3> in{};
    for (unsigned input = 0; input < descriptor->arity; ++input) {
      const auto value = self(self, node.inputs[input], pick, depth + 1);
      if (!value) return std::nullopt;
      in[input] = *value;
    }

    if (node.width == 64 && node.op == Op::add && in[0].image != in[1].image)
      return Value{true, in[0].bits + in[1].bits};
    if (node.width == 64 && node.op == Op::sub && in[0].image && !in[1].image)
      return Value{true, in[0].bits - in[1].bits};
    if (in[0].image || in[1].image || in[2].image) return std::nullopt;
    std::array<std::uint64_t, 3> operands{in[0].bits, in[1].bits, in[2].bits};
    const auto folded = FoldPure(node, std::span(operands).first(descriptor->arity),
                                 nodes[node.inputs[node.op == Op::select ? 1 : 0]].width);
    if (!folded) return std::nullopt;
    return Value{false, *folded};
  };

  const auto bytes = nodes[load].width / 8U;
  std::optional<std::pair<std::uint64_t, std::uint64_t>> span;
  for (std::uint64_t pick = 0; pick < 2; ++pick) {
    auto address = evaluate(evaluate, nodes[load].inputs[0], pick, 0);

    // Under a declared bias a number at or past it is the location it names,
    // as a load address folded from a page base carried in a register is.
    if (address && !address->image && bias && address->bits >= *bias)
      address = Value{true, address->bits - *bias};
    if (!address || !address->image || address->bits > UINT64_MAX - bytes) return std::nullopt;
    span = span ? std::pair{std::min(span->first, address->bits),
                            std::max(span->second, address->bits + bytes)}
                : std::pair{address->bits, address->bits + bytes};
    if (!choice) break;
  }

  return span;
}

std::optional<std::pair<std::uint64_t, bool>> SsaLoadLocation(std::span<const Node> nodes,
                                                              ValueId load,
                                                              std::optional<std::uint64_t> bias) {
  return LoadLocation(nodes, load, bias);
}

namespace {
std::optional<bool> ImageBased(const SsaGraph& graph, const SsaValue& value,
                               std::vector<SsaValue>& open, unsigned& visits, Budget& budget) {
  if (++visits > 256) return false;
  if (budget.try_consume({65, 0}) != BudgetDecline::none) return std::nullopt;
  const auto* block = graph.Get(value.block);
  if (!block) return false;
  if (value.kind == SsaValueKind::node) {
    if (value.index >= block->nodes.size()) return false;
    if (block->nodes[value.index].op != Op::read)
      return SsaImageLocation(block->nodes, value.index, std::nullopt).has_value();
    if (budget.try_consume({block->reads.size(), 0}) != BudgetDecline::none) return std::nullopt;
    const auto read = std::find_if(block->reads.begin(), block->reads.end(),
                                   [&](const SsaRead& item) { return item.node == value.index; });
    if (read == block->reads.end()) return false;
    return ImageBased(graph, read->value, open, visits, budget);
  }

  if (value.kind != SsaValueKind::phi || value.index >= block->phis.size()) return false;
  if (budget.try_consume({open.size(), sizeof(SsaValue)}) != BudgetDecline::none)
    return std::nullopt;
  if (std::find(open.begin(), open.end(), value) != open.end()) return true;
  const auto& phi = block->phis[value.index];
  if (phi.external_entry || phi.incoming.empty()) return false;
  open.push_back(value);
  for (const auto& input : phi.incoming) {
    const auto based = ImageBased(graph, input.value, open, visits, budget);
    if (!based || !*based) return based;
  }

  return true;
}
}  // namespace

std::optional<bool> SsaImageBasedValue(const SsaGraph& graph, const SsaValue& value,
                                       Budget& budget) {
  std::vector<SsaValue> open;
  unsigned visits = 0;
  return ImageBased(graph, value, open, visits, budget);
}

std::optional<SsaImageReach> SsaImageReaching(const SsaGraph& graph, Budget& budget,
                                              const ImageFacts* facts) {
  SsaImageReach reach;
  if (budget.try_consume({graph.slots(), graph.slots() * 3 * sizeof(std::vector<std::uint8_t>)}) !=
      BudgetDecline::none)
    return std::nullopt;
  reach.phis.resize(graph.slots());
  reach.frames.resize(graph.slots());
  reach.nodes.resize(graph.slots());
  const auto lookup =
      1 + static_cast<std::uint64_t>(std::bit_width(facts ? facts->pointers.size() : 0));
  // What holds a location whatever else does: image arithmetic, and a load
  // that yields one -- the graph's own record of it, or a declared relocated
  // slot. It does not change as marks spread, so it is found once.
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto& block = *graph.Get(*handle);
    const auto count = block.nodes.size() + block.phis.size() + block.frame_phis.size();
    if (budget.try_consume({count, count}) != BudgetDecline::none) return std::nullopt;
    reach.phis[slot].assign(block.phis.size(), 0);
    reach.frames[slot].assign(block.frame_phis.size(), 0);
    reach.nodes[slot].assign(block.nodes.size(), 0);
    for (ValueId id = 0; id < block.nodes.size(); ++id) {
      const auto& node = block.nodes[id];
      if (budget.try_consume({65, 0}) != BudgetDecline::none) return std::nullopt;
      bool on = false;
      if (node.op == Op::load || node.op == Op::exclusive_load) {
        const auto* fold = FoldOf(block, id);
        const auto read =
            std::lower_bound(block.path_reads.begin(), block.path_reads.end(), id,
                             [](const SsaPathRead& item, ValueId key) { return item.node < key; });
        on = (fold && fold->skip_access && !fold->condition &&
              fold->kind == SsaConstantKind::image_location) ||
             (read != block.path_reads.end() && read->node == id && read->relocated);
        if (!on && facts && node.op == Op::load && node.access.byte_order == ByteOrder::little) {
          if (budget.try_consume({64 + lookup, 0}) != BudgetDecline::none) return std::nullopt;
          const auto location = LoadLocation(block.nodes, id, graph.load_bias());
          on = location && (!location->second || facts->page_aligned_placement) &&
               ReadRelocated(*facts, location->first, node.width).has_value();
        }
      } else if (node.op != Op::read) {
        on = SsaImageLocation(block.nodes, id, std::nullopt).has_value();
      }

      reach.nodes[slot][id] = on;
    }
  }

  const auto marked = [&](const SsaValue& value) -> bool {
    const auto slot = value.block.slot;
    if (slot >= graph.slots() || !graph.Get(value.block)) return false;
    const auto* bits = value.kind == SsaValueKind::phi         ? &reach.phis[slot]
                       : value.kind == SsaValueKind::frame_phi ? &reach.frames[slot]
                       : value.kind == SsaValueKind::node      ? &reach.nodes[slot]
                                                               : nullptr;
    return bits && value.index < bits->size() && (*bits)[value.index];
  };

  // A mark only ever turns on, so sweeping until a sweep turns none on ends,
  // and a loop's step is carried round by the sweep after the one that met it.
  for (bool changed = true; changed;) {
    changed = false;
    for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
      const auto handle = graph.Handle(slot);
      if (!handle) continue;
      const auto& block = *graph.Get(*handle);
      if (budget.try_consume({1 + 4 * block.nodes.size() + block.phis.size() +
                                  block.frame_phis.size() + block.reads.size(),
                              0}) != BudgetDecline::none)
        return std::nullopt;
      const auto join = [&](const auto& phis, std::vector<std::uint8_t>& bits) {
        for (std::size_t index = 0; index < phis.size(); ++index) {
          if (bits[index]) continue;
          if (budget.try_consume({phis[index].incoming.size(), 0}) != BudgetDecline::none)
            return false;
          if (std::any_of(phis[index].incoming.begin(), phis[index].incoming.end(),
                          [&](const SsaPhiInput& input) { return marked(input.value); }))
            bits[index] = changed = true;
        }

        return true;
      };

      if (!join(block.phis, reach.phis[slot]) || !join(block.frame_phis, reach.frames[slot]))
        return std::nullopt;
      for (std::size_t id = 0; id < block.nodes.size(); ++id) {
        if (reach.nodes[slot][id]) continue;
        const auto& node = block.nodes[id];
        bool on = false;
        if (node.op == Op::read) {
          const auto read = ReadOf(block, static_cast<ValueId>(id));
          on = read && marked(read->value);
        } else if (node.op == Op::load || node.op == Op::exclusive_load) {
          // A promoted slot's load yields the value last put there.
          const auto* frame = FrameOf(block, static_cast<ValueId>(id));
          on = frame && frame->replacement && marked(*frame->replacement);
        } else if (const auto* descriptor = Descriptor(node.op);
                   descriptor && descriptor->effect == Effect::pure && descriptor->produces_value) {
          // Arithmetic moves a location on without making it anything else.
          for (unsigned input = 0; input < descriptor->arity; ++input)
            on = on || (node.inputs[input] < id && reach.nodes[slot][node.inputs[input]]);
        }

        if (on) reach.nodes[slot][id] = changed = true;
      }
    }
  }

  return reach;
}

std::optional<std::pair<std::uint64_t, bool>> SsaImageLocation(std::span<const Node> nodes,
                                                               ValueId id,
                                                               std::optional<std::uint64_t> bias) {
  unsigned visits = 0;
  const auto address = Address(nodes, id, visits, true);
  if (!address) return std::nullopt;
  if (address->image_bases == 0)
    return bias && address->offset >= *bias
               ? std::optional(std::pair{address->offset - *bias, false})
               : std::nullopt;
  if (address->image_bases != 1) return std::nullopt;
  return std::pair{address->offset, address->placed};
}

std::optional<SsaPathRead> ResolveSsaPathRead(std::span<const Node> nodes, ValueId load,
                                              ImageFacts facts) {
  const auto location = LoadLocation(nodes, load, facts.load_bias);
  if (!location || (location->second && !facts.page_aligned_placement)) return std::nullopt;
  const auto& node = nodes[load];
  if (node.access.byte_order == ByteOrder::little)
    if (const auto target = ReadRelocated(facts, location->first, node.width))
      return SsaPathRead{load, location->first, true, *target, location->second};
  if (const auto literal = ReadConstant(facts, location->first, node.width, node.access.byte_order))
    return SsaPathRead{load, location->first, false, *literal, location->second};
  return std::nullopt;
}

SsaDecline ValidateSsaSourceBinding(const SsaBlock& block, std::span<const Group> sources,
                                    Budget& budget) {
  if (block.opaque) {
    // No semantics to compare: the bytes are the binding.
    if (block.original_sources.size() != block.source_groups.size() ||
        block.source_bytes.size() != block.source_groups.size())
      return SsaDecline::invalid_graph;
    for (std::size_t index = 0; index < block.original_sources.size(); ++index) {
      const auto source = block.original_sources[index];
      if (source >= sources.size()) return SsaDecline::invalid_graph;
      const auto& group = sources[source];
      const auto& bytes = block.source_bytes[index];
      if (budget.try_consume({1 + bytes.size(), 0}) != BudgetDecline::none)
        return SsaDecline::resource_limit;
      if (group.source_address() != block.source_groups[index] || !group.nodes().empty() ||
          group.transfer() || group.bytes().size() != bytes.size() ||
          !std::equal(bytes.begin(), bytes.end(), group.bytes().begin()))
        return SsaDecline::invalid_graph;
    }

    return SsaDecline::none;
  }

  if (block.original_sources.size() != block.boundaries.size() ||
      block.source_groups.size() != block.boundaries.size() ||
      block.source_bytes.size() != block.boundaries.size())
    return SsaDecline::invalid_graph;
  for (std::size_t index = 0; index < block.boundaries.size(); ++index) {
    const auto source = block.original_sources[index];
    if (source >= sources.size()) return SsaDecline::invalid_graph;
    const auto& group = sources[source];
    const auto& bytes = block.source_bytes[index];
    if (budget.try_consume({1 + bytes.size(), 0}) != BudgetDecline::none)
      return SsaDecline::resource_limit;
    if (group.source_address() != block.source_groups[index] ||
        group.bytes().size() != bytes.size() ||
        !std::equal(bytes.begin(), bytes.end(), group.bytes().begin()) ||
        group.transfer().has_value() != block.boundaries[index].transfer.has_value() ||
        (group.transfer() && group.transfer()->kind != block.boundaries[index].transfer->kind))
      return SsaDecline::invalid_graph;
    if (!group.transfer()) continue;
    const auto& source_transfer = *group.transfer();
    const auto& mapped_transfer = *block.boundaries[index].transfer;
    if (source_transfer.kind == TransferKind::call) {
      // A literal callee and the continuation bind exactly; a computed callee
      // is left to the block's own fold, which needs a declared read for it.
      bool exhausted = false;
      std::uint64_t source_address = 0, mapped_address = 0;
      const auto literal = [&](std::span<const Node> nodes, std::optional<ValueId> id,
                               std::uint64_t& address) {
        return id && ImageTarget(nodes, *id, address, budget, exhausted);
      };

      const bool bound = (!literal(group.nodes(), source_transfer.target, source_address) ||
                          (literal(block.nodes, mapped_transfer.target, mapped_address) &&
                           source_address == mapped_address)) &&
                         literal(group.nodes(), source_transfer.continuation, source_address) &&
                         literal(block.nodes, mapped_transfer.continuation, mapped_address) &&
                         source_address == mapped_address;
      if (!bound) return exhausted ? SsaDecline::resource_limit : SsaDecline::invalid_graph;
      continue;
    }

    if (source_transfer.kind != TransferKind::jump &&
        source_transfer.kind != TransferKind::conditional)
      continue;
    bool exhausted = false;

    // A recovered transition may resolve a computed source target, such as a
    // dispatcher's register jump; completeness then rests on the path's own
    // value. A plain jump may resolve one only through a slot its path reads
    // under a declaration, so a literal standing in for the computed target
    // would bind a destination nothing declared. A literal source target must
    // still be the one the path uses.
    const auto same_target = [&](ValueId source_node, ValueId mapped_node) {
      std::uint64_t source_address = 0, mapped_address = 0;
      if (!ImageTarget(group.nodes(), source_node, source_address, budget, exhausted)) {
        if (exhausted) return false;
        if (block.transition) return true;
        if (source_transfer.kind != TransferKind::jump) return false;
        return !ImageTarget(block.nodes, mapped_node, mapped_address, budget, exhausted) &&
               !exhausted;
      }

      return ImageTarget(block.nodes, mapped_node, mapped_address, budget, exhausted) &&
             source_address == mapped_address;
    };

    if (!same_target(source_transfer.target, mapped_transfer.target) ||
        (source_transfer.kind == TransferKind::conditional &&
         (!source_transfer.alternative || !mapped_transfer.alternative ||
          !same_target(*source_transfer.alternative, *mapped_transfer.alternative))))
      return exhausted ? SsaDecline::resource_limit : SsaDecline::invalid_graph;
  }

  return SsaDecline::none;
}

SsaDecline ValidateSsaClosedMember(const SsaBlock& block, std::span<const Group> sources,
                                   Budget& budget, bool through_dispatches) {
  const auto bound = ValidateSsaSourceBinding(block, sources, budget);
  if (bound != SsaDecline::none) return bound;
  const auto complete = through_dispatches ? ValidateSsaDispatchSuccessors(block, budget)
                                           : ValidateSsaDirectSuccessors(block, budget);
  if (complete != SsaDecline::none) return complete;
  if (budget.try_consume({1 + block.edges.size(), 0}) != BudgetDecline::none)
    return SsaDecline::resource_limit;
  for (const auto& edge : block.edges) {
    if (edge.assumptions.unresolved_target) return SsaDecline::invalid_graph;
    switch (edge.kind) {
      case SsaEdgeKind::branch:
      case SsaEdgeKind::fallthrough:
        if (!edge.target_block) return SsaDecline::invalid_graph;
        break;
      case SsaEdgeKind::return_:
        if (edge.target_block || !edge.assumptions.return_leaves) return SsaDecline::invalid_graph;
        break;
      case SsaEdgeKind::trap:
        if (edge.target_block) return SsaDecline::invalid_graph;
        break;
      // The call rule admitted these: the callee leaves the population and
      // the continuation is reached only through a declared return to it.
      case SsaEdgeKind::callee:
        if (edge.target_block) return SsaDecline::invalid_graph;
        break;
      case SsaEdgeKind::potential_return:
        if (!edge.target_block || !edge.assumptions.callee_returns_to_continuation)
          return SsaDecline::invalid_graph;
        break;
      case SsaEdgeKind::opaque_unknown:
        return SsaDecline::invalid_graph;
    }
  }

  return SsaDecline::none;
}

SsaDecline ValidateSsaReachabilityFacts(const SsaGraph& graph, const SsaReachabilityFacts& facts,
                                        std::span<const Group> sources, Budget& budget,
                                        bool through_dispatches) {
  if (facts.graph_arena != graph.arena() || facts.graph_revision != graph.revision() ||
      facts.entry_scope != SsaEntryScope::closed_population ||
      facts.reachable.size() != graph.slots() || graph.entries().empty())
    return SsaDecline::invalid_graph;
  if (budget.try_consume({graph.slots(), 0}) != BudgetDecline::none)
    return SsaDecline::resource_limit;
  for (const auto entry : graph.entries())
    if (!facts.reachable[entry.slot]) return SsaDecline::invalid_graph;
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    if (facts.reachable[slot] > 1) return SsaDecline::invalid_graph;
    const auto handle = graph.Handle(slot);
    if (!handle) {
      if (facts.reachable[slot]) return SsaDecline::invalid_graph;
      continue;
    }

    if (!facts.reachable[slot]) continue;
    const auto* block = graph.Get(*handle);
    const auto member = ValidateSsaClosedMember(*block, sources, budget, through_dispatches);
    if (member != SsaDecline::none) return member;
    if (budget.try_consume({block->edges.size(), 0}) != BudgetDecline::none)
      return SsaDecline::resource_limit;
    for (const auto& edge : block->edges)
      if (edge.target_block && !facts.reachable[edge.target_block->slot])
        return SsaDecline::invalid_graph;
  }

  return SsaDecline::none;
}

SsaDecline ValidateSsaLivenessFacts(const SsaGraph& graph, const SsaLivenessFacts& facts,
                                    Budget& budget) {
  if (facts.graph_arena != graph.arena() || facts.graph_revision != graph.revision() ||
      facts.live_nodes.size() != graph.slots())
    return SsaDecline::invalid_graph;
  if (budget.try_consume({graph.slots(), 0}) != BudgetDecline::none)
    return SsaDecline::resource_limit;
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    const auto handle = graph.Handle(slot);
    if (!handle) {
      if (!facts.live_nodes[slot].empty()) return SsaDecline::invalid_graph;
      continue;
    }

    const auto checked = RootsAndInputsLive(*graph.Get(*handle), facts.live_nodes[slot], budget);
    if (checked != SsaDecline::none) return checked;
  }

  return SsaDecline::none;
}

}  // namespace nyx::ir
