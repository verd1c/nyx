#include "nyx/recovery/image.hpp"

#include <algorithm>
#include <array>

#include "nyx/ir/fold.hpp"

namespace nyx::recovery {
namespace {
using ir::Node;
using ir::Op;

// What a node is known to compute, and which declared restrictions that rests on.
struct Known {
  enum class Kind { none, literal, image } kind = Kind::none;
  std::uint64_t bits = 0;
  bool constant_bytes = false;
  bool relocated_slot = false;
  bool page_placement = false;
};

// Nodes are visited in SSA order after their operands have been rewritten, so a
// load's address is already an image location whenever one can be known.
Known Classify(const Node& node, std::span<const Node> nodes, std::span<const Known> known,
               const ir::ImageFacts& facts) {
  if (node.width > 64) return {};
  if (node.op == Op::constant)
    return {Known::Kind::literal, node.immediate & ir::LowMask(node.width)};
  if (node.op == Op::image_address) return {Known::Kind::image, node.immediate};
  if (node.op != Op::load) return {};
  const auto& address = nodes[node.inputs[0]];
  if (address.op != Op::image_address) return {};

  // The value rests on whatever located the address, as well as on the read.
  const auto& basis = known[node.inputs[0]];
  if (const auto target = ir::ReadRelocated(facts, address.immediate, node.width)) {
    return {Known::Kind::image, *target, basis.constant_bytes, true, basis.page_placement};
  }

  if (const auto bits =
          ir::ReadConstant(facts, address.immediate, node.width, node.access.byte_order)) {
    return {Known::Kind::literal, *bits, true, basis.relocated_slot, basis.page_placement};
  }

  return {};
}

bool PageMask(std::uint64_t mask) {
  // ~(2^k - 1) for a page of at most 4 KiB, the granule a loader aligns to.
  const auto low = ~mask;
  return low != 0 && low < 4096 && (low & (low + 1)) == 0;
}

std::optional<std::pair<ImageRule, Node>> Rewrite(const Node& node, std::span<const Node> nodes,
                                                  std::span<const Known> known,
                                                  const ir::ImageFacts& facts) {
  if (node.op == Op::constant || node.op == Op::image_address || node.width > 64) return {};
  const auto* descriptor = ir::Descriptor(node.op);
  if (descriptor == nullptr || descriptor->effect != ir::Effect::pure ||
      !descriptor->produces_value || descriptor->arity == 0)
    return {};
  std::array<std::uint64_t, 3> values{};
  unsigned images = 0;
  for (unsigned i = 0; i < descriptor->arity; ++i) {
    const auto& operand = known[node.inputs[i]];
    if (operand.kind == Known::Kind::none) return {};
    images += operand.kind == Known::Kind::image;
    values[i] = operand.bits;
  }

  if (images == 0) {
    const auto result =
        ir::FoldPure(node, std::span(values).first(descriptor->arity), nodes[node.inputs[0]].width);
    if (!result) return {};
    return std::pair{ImageRule::constant_fold, Node{Op::constant, node.width, {}, *result}};
  }

  if (node.width != 64 || descriptor->arity != 2) return {};
  const bool first_image = known[node.inputs[0]].kind == Known::Kind::image;
  const auto image = [&](std::uint64_t address) {
    return std::pair{ImageRule::image_offset, Node{Op::image_address, 64, {}, address}};
  };

  if (node.op == Op::add && images == 1) return image(values[0] + values[1]);
  if (node.op == Op::sub && images == 1 && first_image) return image(values[0] - values[1]);

  // Two image locations share the bias, so their difference is a plain number.
  if (node.op == Op::sub && images == 2) {
    return std::pair{ImageRule::image_offset, Node{Op::constant, 64, {}, values[0] - values[1]}};
  }

  if (node.op == Op::bit_and && images == 1 && facts.page_aligned_placement) {
    const auto mask = first_image ? values[1] : values[0];
    const auto address = first_image ? values[0] : values[1];
    if (PageMask(mask))
      return std::pair{ImageRule::page_base, Node{Op::image_address, 64, {}, address & mask}};
  }

  return {};
}

}  // namespace

ImagePathResult FoldImageValues(const ir::Path& input, const ir::ImageFacts& facts, Budget& budget,
                                ImageLimits limits) {
  const auto decline = [](ImageDecline reason) {
    return ImagePathResult{{}, {}, {}, {}, {}, {}, reason};
  };

  const auto valid = ir::ValidatePath(input, budget, limits.block);
  if (valid != ir::BlockDecline::none) {
    return decline(valid == ir::BlockDecline::resource_limit ? ImageDecline::resource_limit
                                                             : ImageDecline::invalid_ir);
  }

  if (input.revision() == UINT64_MAX) return decline(ImageDecline::revision_overflow);
  const auto capacity = std::min<std::uint64_t>(limits.max_edits, input.nodes().size());
  const auto charge = [&](std::uint64_t count, std::uint64_t size) {
    return count <= UINT64_MAX / size &&
           budget.try_consume({count, count * size}) == BudgetDecline::none;
  };

  if (!charge(input.sources().size(), sizeof(ir::Group)) ||
      !charge(input.nodes().size(), sizeof(Node)) || !charge(input.nodes().size(), sizeof(Known)) ||
      !charge(input.origins().size(), sizeof(ir::Origin)) ||
      !charge(input.boundaries().size(), sizeof(ir::Boundary)) ||
      !charge(capacity, sizeof(ImageEdit))) {
    return decline(ImageDecline::resource_limit);
  }

  for (const auto& source : input.sources()) {
    if (!charge(source.bytes().size(), 1) || !charge(source.nodes().size(), sizeof(Node)) ||
        !charge(source.writes().size(), sizeof(ir::Write)))
      return decline(ImageDecline::resource_limit);
  }

  for (const auto& boundary : input.boundaries()) {
    if (!charge(boundary.writes.size(), sizeof(ir::Write)))
      return decline(ImageDecline::resource_limit);
  }

  std::vector<Node> nodes;
  std::vector<Known> known;
  std::vector<ImageEdit> journal;
  journal.reserve(static_cast<std::size_t>(capacity));

  // One pass of the rewrite. It runs again, on the original nodes and a smaller
  // declaration, when the path turns out to refute one of the ranges it read.
  const auto evaluate = [&](const ir::ImageFacts& declared) {
    nodes.assign(input.nodes().begin(), input.nodes().end());
    known.assign(nodes.size(), Known{});
    journal.clear();
    for (std::size_t id = 0; id < nodes.size(); ++id) {
      if (budget.try_consume({8, 0}) != BudgetDecline::none) return ImageDecline::resource_limit;
      if (const auto edit = Rewrite(nodes[id], nodes, known, declared)) {
        if (journal.size() == capacity) return ImageDecline::resource_limit;
        const auto* descriptor = ir::Descriptor(nodes[id].op);
        bool constant_bytes = false, relocated_slot = false;
        bool page_placement = edit->first == ImageRule::page_base;
        for (unsigned i = 0; i < descriptor->arity; ++i) {
          constant_bytes = constant_bytes || known[nodes[id].inputs[i]].constant_bytes;
          relocated_slot = relocated_slot || known[nodes[id].inputs[i]].relocated_slot;
          page_placement = page_placement || known[nodes[id].inputs[i]].page_placement;
        }

        journal.push_back({edit->first, static_cast<ir::ValueId>(id), nodes[id], edit->second,
                           constant_bytes, relocated_slot, page_placement, input.revision(),
                           input.revision() + 1});
        nodes[id] = edit->second;
        known[id] = Classify(nodes[id], nodes, known, declared);
        known[id].constant_bytes = constant_bytes;
        known[id].relocated_slot = relocated_slot;
        known[id].page_placement = page_placement;
        continue;
      }

      known[id] = Classify(nodes[id], nodes, known, declared);
    }

    return ImageDecline::none;
  };

  auto status = evaluate(facts);
  if (status != ImageDecline::none) return decline(status);

  // A store this path places inside a declared range refutes that range. It is
  // dropped and the pass runs again without it, so no published edit can rest on
  // a declaration the same path contradicts.
  std::vector<ContradictedRange> contradicted;
  std::vector<ContradictedPointer> contradicted_pointers;
  std::vector<ir::ImageWrite> writes;
  for (std::size_t id = 0; id < nodes.size(); ++id) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return decline(ImageDecline::resource_limit);
    if (!ir::MayWriteMemory(nodes[id].op) || nodes[id].width % 8 != 0 || nodes[id].width == 0)
      continue;
    const auto& address = known[nodes[id].inputs[0]];
    if (address.kind != Known::Kind::image) continue;
    if (!charge(1, sizeof(ir::ImageWrite))) return decline(ImageDecline::resource_limit);
    const auto width =
        nodes[id].op == Op::exclusive_store ? nodes[nodes[id].inputs[1]].width : nodes[id].width;
    writes.push_back({address.bits, width, static_cast<ir::ValueId>(id)});
  }

  const auto refutations = ir::RefuteImageFacts(facts, writes, budget);
  if (!refutations) return decline(ImageDecline::resource_limit);
  if (!charge(facts.constants.size(), sizeof(std::uint8_t) + sizeof(ir::ConstantImageRange)) ||
      !charge(facts.pointers.size(), sizeof(std::uint8_t))) {
    return decline(ImageDecline::resource_limit);
  }

  std::vector<std::uint8_t> refuted(facts.constants.size());
  std::vector<std::uint8_t> refuted_pointers(facts.pointers.size());
  for (const auto& conflict : refutations->constants) {
    refuted[conflict.index] = 1;
    const auto& write = conflict.write;
    if (!charge(1, sizeof(ContradictedRange))) return decline(ImageDecline::resource_limit);
    contradicted.push_back(
        {facts.constants[conflict.index].address, write.operation, write.address, write.width});
  }

  for (const auto& conflict : refutations->pointers) {
    refuted_pointers[conflict.index] = 1;
    const auto& write = conflict.write;
    if (!charge(1, sizeof(ContradictedPointer))) return decline(ImageDecline::resource_limit);
    contradicted_pointers.push_back(
        {facts.pointers[conflict.index].address, write.operation, write.address, write.width});
  }

  std::vector<ir::ConstantImageRange> kept;
  std::vector<ir::RelocatedPointer> kept_pointers;
  kept.reserve(facts.constants.size());
  for (std::size_t range = 0; range < facts.constants.size(); ++range) {
    if (!refuted[range]) kept.push_back(facts.constants[range]);
  }

  if (!contradicted_pointers.empty()) {
    if (!charge(facts.pointers.size(), sizeof(ir::RelocatedPointer))) {
      return decline(ImageDecline::resource_limit);
    }

    kept_pointers.reserve(facts.pointers.size());
    for (std::size_t slot = 0; slot < facts.pointers.size(); ++slot) {
      kept_pointers.push_back(facts.pointers[slot]);
      if (refuted_pointers[slot]) kept_pointers.back().value_stable = false;
    }
  }

  if (!contradicted.empty() || !contradicted_pointers.empty()) {
    const ir::ImageFacts reduced{kept,
                                 contradicted_pointers.empty()
                                     ? facts.pointers
                                     : std::span<const ir::RelocatedPointer>(kept_pointers),
                                 facts.page_aligned_placement, facts.load_bias};
    status = evaluate(reduced);
    if (status != ImageDecline::none) return decline(status);
  }

  ir::Path output(std::vector<ir::Group>(input.sources().begin(), input.sources().end()),
                  std::move(nodes),
                  std::vector<ir::Origin>(input.origins().begin(), input.origins().end()),
                  std::vector<ir::Boundary>(input.boundaries().begin(), input.boundaries().end()),
                  input.revision() + (journal.empty() ? 0 : 1));
  const auto checked = ir::ValidatePath(output, budget, limits.block);
  if (checked != ir::BlockDecline::none) {
    return decline(checked == ir::BlockDecline::resource_limit ? ImageDecline::resource_limit
                                                               : ImageDecline::invalid_ir);
  }

  return {std::move(output),       std::move(journal),
          std::move(contradicted), std::move(contradicted_pointers),
          std::move(kept),         std::move(kept_pointers),
          ImageDecline::none};
}

}  // namespace nyx::recovery
