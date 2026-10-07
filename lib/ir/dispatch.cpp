#include "nyx/ir/dispatch.hpp"

#include <algorithm>
#include <array>
#include <bit>

#include "nyx/ir/fold.hpp"

namespace nyx::ir {
namespace {

// What a node is known to compute. An image location is a place, so it stays
// symbolic in the load bias; a literal is a plain number that does not.
struct Known {
  enum class Kind { unknown, literal, image } kind = Kind::unknown;
  std::uint64_t bits = 0;
};

bool Same(const Known& a, const Known& b) { return a.kind == b.kind && a.bits == b.bits; }

// ADRP masks the program counter down to its page. On an image location that is
// only meaningful if every admitted placement preserves the bits being cleared,
// so accept no mask coarser than the granule a loader aligns to.
bool PageMask(std::uint64_t mask) {
  const auto low = ~mask;
  return low != 0 && low <= 0xfff && (low & (low + 1)) == 0;
}

// Nodes are evaluated in ascending id, so every operand is already known. A
// resolved load appends to `witness` when one is supplied; the load itself is
// untouched and still executes wherever this expression lives.
Known Evaluate(ValueId id, std::span<const Node> nodes, std::span<const Known> known,
               const ImageFacts& facts, std::vector<ImageRead>* witness, bool when) {
  const auto& node = nodes[id];
  if (node.width > 64) return {};
  if (node.op == Op::constant) return {Known::Kind::literal, node.immediate & LowMask(node.width)};
  if (node.op == Op::image_address) return {Known::Kind::image, node.immediate};
  if (node.op == Op::load) {
    const auto& address = known[node.inputs[0]];
    if (address.kind != Known::Kind::image) return {};

    // A relocated slot holds a place in the image; a table entry is a number.
    if (const auto target = ReadRelocated(facts, address.bits, node.width)) {
      if (witness != nullptr)
        witness->push_back({id, when, address.bits, node.width, *target, true});
      return {Known::Kind::image, *target};
    }

    const auto bits = ReadConstant(facts, address.bits, node.width, node.access.byte_order);
    if (!bits) return {};
    if (witness != nullptr) witness->push_back({id, when, address.bits, node.width, *bits, false});
    return {Known::Kind::literal, *bits};
  }

  const auto* descriptor = Descriptor(node.op);
  if (descriptor == nullptr || descriptor->effect != Effect::pure || !descriptor->produces_value ||
      descriptor->arity == 0)
    return {};
  if (node.op == Op::select) {
    const auto& condition = known[node.inputs[0]];
    const auto& taken = known[node.inputs[1]];
    const auto& other = known[node.inputs[2]];
    if (condition.kind == Known::Kind::literal) return (condition.bits & 1) != 0 ? taken : other;

    // A choice between two of the same value is no choice about this expression.
    return taken.kind != Known::Kind::unknown && Same(taken, other) ? taken : Known{};
  }

  std::array<std::uint64_t, 3> operands{};
  unsigned images = 0;
  for (unsigned i = 0; i < descriptor->arity; ++i) {
    const auto& operand = known[node.inputs[i]];
    if (operand.kind == Known::Kind::unknown) return {};
    images += operand.kind == Known::Kind::image;
    operands[i] = operand.bits;
  }

  if (images == 0) {
    const auto folded =
        FoldPure(node, std::span(operands).first(descriptor->arity), nodes[node.inputs[0]].width);
    if (!folded) return {};
    return {Known::Kind::literal, *folded};
  }

  if (node.width != 64) return {};
  const bool first_image = known[node.inputs[0]].kind == Known::Kind::image;

  // Only arithmetic that holds at every admitted load bias may touch a place.
  if (descriptor->arity == 1 && nodes[node.inputs[0]].width == 64 &&
      (node.op == Op::zext || (node.op == Op::extract && node.immediate == 0))) {
    return known[node.inputs[0]];
  }

  if (descriptor->arity != 2) return {};
  if (node.op == Op::add && images == 1) return {Known::Kind::image, operands[0] + operands[1]};
  if (node.op == Op::sub && images == 1 && first_image) {
    return {Known::Kind::image, operands[0] - operands[1]};
  }

  // Two locations share the bias, so their difference is a plain number.
  if (node.op == Op::sub && images == 2) {
    return {Known::Kind::literal, operands[0] - operands[1]};
  }

  if (node.op == Op::bit_and && images == 1 && facts.page_aligned_placement) {
    const auto mask = first_image ? operands[1] : operands[0];
    const auto address = first_image ? operands[0] : operands[1];
    if (PageMask(mask)) return {Known::Kind::image, address & mask};
  }

  return {};
}

}  // namespace

DispatchResolution ResolveDispatchBranch(std::span<const Node> nodes, ValueId target,
                                         const ImageFacts& facts, Budget& budget) {
  const auto none = DispatchResolution{{}, DispatchDecline::no_branch};
  const auto starved = DispatchResolution{{}, DispatchDecline::resource_limit};
  if (target >= nodes.size()) return none;
  const auto count = static_cast<std::uint64_t>(target) + 1;
  if (budget.try_consume({count * 6, count * (2 * sizeof(Known) + sizeof(ValueId) + 1)}) !=
      BudgetDecline::none)
    return starved;
  std::vector<Known> known(count);
  for (ValueId id = 0; id < count; ++id)
    known[id] = Evaluate(id, nodes, known, facts, nullptr, false);
  if (known[target].kind != Known::Kind::unknown) return none;

  // The cone of the destination, stopping at a choice that did not settle: the
  // arms of such a choice are what the two resolutions substitute, so they stay
  // outside it and keep the values they already have.
  std::vector<std::uint8_t> inside(count);
  std::vector<ValueId> cone, stack{target};
  std::optional<ValueId> choice;
  while (!stack.empty()) {
    if (budget.try_consume({4, 0}) != BudgetDecline::none) return starved;
    const auto id = stack.back();
    stack.pop_back();
    if (inside[id]) continue;
    inside[id] = 1;
    cone.push_back(id);
    const auto& node = nodes[id];
    if (node.op == Op::select && known[id].kind == Known::Kind::unknown &&
        known[node.inputs[0]].kind != Known::Kind::literal) {
      // Only one unresolved select is enumerated; a product of arms is not supported.
      if (choice && *choice != id) return none;
      choice = id;
      continue;
    }

    const auto* descriptor = Descriptor(node.op);
    if (descriptor == nullptr || !descriptor->produces_value) continue;
    for (unsigned i = 0; i < descriptor->arity; ++i) stack.push_back(node.inputs[i]);
  }

  if (!choice) return none;
  if (budget.try_consume({cone.size() * (std::bit_width(cone.size()) + 3), 0}) !=
      BudgetDecline::none) {
    return starved;
  }

  std::sort(cone.begin(), cone.end());
  DispatchBranch branch{nodes[*choice].inputs[0], 0, 0, {}};
  const auto& select = nodes[*choice];
  std::vector<Known> pinned;
  for (unsigned arm = 0; arm < 2; ++arm) {
    const bool when = arm == 0;
    pinned.assign(known.begin(), known.end());
    pinned[*choice] = known[select.inputs[when ? 1 : 2]];
    for (const auto id : cone) {
      if (id == *choice) continue;
      pinned[id] = Evaluate(id, nodes, pinned, facts, &branch.witness, when);
      if (budget.try_consume({1, sizeof(ImageRead)}) != BudgetDecline::none) return starved;
    }

    if (pinned[target].kind != Known::Kind::image) return none;
    (when ? branch.when_true : branch.when_false) = pinned[target].bits;
  }

  if (branch.when_true == branch.when_false) return none;
  return {std::move(branch), DispatchDecline::none};
}

}  // namespace nyx::ir
