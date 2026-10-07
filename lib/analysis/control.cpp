#include "nyx/analysis/control.hpp"

#include <algorithm>
#include <bit>
#include <limits>

#include "nyx/analysis/path_control.hpp"
#include "nyx/ir/fold.hpp"
#include "nyx/ir/value_numbering.hpp"

namespace nyx::analysis {
namespace {
using ir::Op;

struct Value {
  TargetKind kind = TargetKind::unknown;
  std::uint64_t bits = 0;

  // True once a constant image read has contributed to this value, so the
  // dependency travels with it to whatever destination it ends up forming.
  bool constant_image = false;

  // True once an image location has been computed into this value, whether or
  // not it still resolves to one. A store through such a value may reach a
  // declared location even when the abstraction cannot say which.
  bool image_derived = false;
};

std::uint64_t Mask(unsigned width) { return width == 64 ? UINT64_MAX : (UINT64_C(1) << width) - 1; }

// The image location a 64-bit address names. Under a declared load bias a
// number at or above it is the location standing there, as SSA's LoadLocation
// reads it, so an address means the same however it was spelled.
std::optional<std::uint64_t> Location(const Value& address, unsigned width,
                                      const ImageFacts& facts) {
  if (address.kind == TargetKind::image_location) return address.bits;
  if (address.kind == TargetKind::absolute_runtime && width == 64 && facts.load_bias &&
      address.bits >= *facts.load_bias)
    return address.bits - *facts.load_bias;
  return {};
}

// A relocated slot read whole is the image location the loader put there.
std::optional<Value> RelocatedLoad(const ir::Node& node, std::optional<std::uint64_t> address,
                                   const ImageFacts& facts) {
  if (!address) return {};
  const auto target = ir::ReadRelocated(facts, *address, node.width);
  if (!target) return {};
  return Value{TargetKind::image_location, *target, true};
}

// A load whose address is a known image location inside bytes the caller declared
// constant has a known value. The load still executes; reading it here only
// resolves the destination it feeds.
std::optional<std::uint64_t> ConstantLoad(const ir::Node& node,
                                          std::optional<std::uint64_t> address,
                                          const ImageFacts& facts) {
  if (!address) return {};
  return ir::ReadConstant(facts, *address, node.width, node.access.byte_order);
}

Value AbstractValue(const ir::Node& node, std::span<const ir::Node> nodes,
                    std::span<const Value> values, const ImageFacts& facts_in) {
  if (node.op == Op::image_address) return {TargetKind::image_location, node.immediate, false};
  if (node.width > 64) {
    const auto* descriptor = ir::Descriptor(node.op);
    bool dependency = false;
    if (descriptor && descriptor->produces_value)
      for (unsigned i = 0; i < descriptor->arity; ++i)
        dependency |= values[node.inputs[i]].constant_image;
    return {TargetKind::unknown, 0, dependency};
  }

  if (node.op == Op::constant) {
    return {TargetKind::absolute_runtime, node.immediate & Mask(node.width), false};
  }

  if (node.op == Op::load) {
    // A relocated slot holds a place in the image; a table entry is a number.
    const auto at = Location(values[node.inputs[0]], nodes[node.inputs[0]].width, facts_in);
    if (const auto pointer = RelocatedLoad(node, at, facts_in)) {
      return *pointer;
    }

    const auto loaded = ConstantLoad(node, at, facts_in);
    if (!loaded) return {TargetKind::unknown, 0, values[node.inputs[0]].constant_image};
    return {TargetKind::absolute_runtime, *loaded, true};
  }

  const auto* descriptor = ir::Descriptor(node.op);
  if (descriptor->effect != ir::Effect::pure || descriptor->arity == 0) return {};
  const auto a = values[node.inputs[0]];
  const auto b = descriptor->arity > 1 ? values[node.inputs[1]] : Value{};
  const auto carry = [&](Value result, bool dependency) {
    result.constant_image = result.constant_image || dependency;
    return result;
  };

  const auto dependency = a.constant_image || b.constant_image ||
                          (descriptor->arity > 2 && values[node.inputs[2]].constant_image);
  if (node.op == Op::select) {
    if (a.kind == TargetKind::absolute_runtime) {
      return carry(values[node.inputs[a.bits ? 1 : 2]], a.constant_image);
    }

    const auto c = values[node.inputs[2]];
    if (b.kind == c.kind && b.bits == c.bits) return carry(b, dependency);
    return {TargetKind::unknown, 0, dependency};
  }

  if (node.width == 64) {
    if (node.op == Op::add) {
      if (a.kind == TargetKind::image_location && b.kind == TargetKind::absolute_runtime) {
        return {TargetKind::image_location, a.bits + b.bits, dependency};
      }

      if (b.kind == TargetKind::image_location && a.kind == TargetKind::absolute_runtime) {
        return {TargetKind::image_location, a.bits + b.bits, dependency};
      }
    } else if (node.op == Op::sub) {
      if (a.kind == TargetKind::image_location && b.kind == TargetKind::absolute_runtime) {
        return {TargetKind::image_location, a.bits - b.bits, dependency};
      }

      if (a.kind == TargetKind::image_location && b.kind == TargetKind::image_location) {
        return {TargetKind::absolute_runtime, a.bits - b.bits, dependency};
      }
    } else if (node.op == Op::bit_and && a.kind == TargetKind::image_location &&
               b.kind == TargetKind::absolute_runtime) {
      // ADRP masks the program counter down to its page. On an image location
      // that is only meaningful if every admitted placement preserves the bits
      // being cleared, so accept a mask that clears no more than a page. Load
      // biases are page-aligned, which is what makes ADRP position-independent
      // in the first place; a finer mask would depend on the actual placement.
      const auto low = ~b.bits;
      if (facts_in.page_aligned_placement && low != 0 && low <= 0xfff && (low & (low + 1)) == 0) {
        return {TargetKind::image_location, a.bits & b.bits, dependency};
      }

      return {TargetKind::unknown, 0, dependency};
    } else if ((node.op == Op::zext || (node.op == Op::extract && node.immediate == 0)) &&
               nodes[node.inputs[0]].width == 64) {
      return carry(a, dependency);
    }
  }

  for (unsigned i = 0; i < descriptor->arity; ++i) {
    if (values[node.inputs[i]].kind != TargetKind::absolute_runtime ||
        nodes[node.inputs[i]].width > 64)
      return {TargetKind::unknown, 0, dependency};
  }

  std::uint64_t bits;
  const auto width = nodes[node.inputs[0]].width;
  switch (node.op) {
    case Op::add:
      bits = a.bits + b.bits;
      break;
    case Op::sub:
      bits = a.bits - b.bits;
      break;
    case Op::mul:
      bits = a.bits * b.bits;
      break;
    case Op::bit_and:
      bits = a.bits & b.bits;
      break;
    case Op::bit_or:
      bits = a.bits | b.bits;
      break;
    case Op::bit_xor:
      bits = a.bits ^ b.bits;
      break;
    case Op::bit_not:
      bits = ~a.bits;
      break;
    case Op::shl:
      bits = b.bits >= width ? 0 : a.bits << b.bits;
      break;
    case Op::lshr:
      bits = b.bits >= width ? 0 : a.bits >> b.bits;
      break;
    case Op::ashr: {
      const bool sign = (a.bits >> (width - 1)) != 0;
      if (b.bits >= width)
        bits = sign ? Mask(width) : 0;
      else {
        bits = a.bits >> b.bits;
        if (sign && b.bits != 0) bits |= Mask(width) ^ Mask(width - static_cast<unsigned>(b.bits));
      }

      break;
    }
    case Op::extract:
      bits = a.bits >> node.immediate;
      break;
    case Op::zext:
      bits = a.bits;
      break;
    case Op::equal:
      bits = a.bits == b.bits;
      break;
    case Op::unsigned_less:
      bits = a.bits < b.bits;
      break;
    case Op::signed_less: {
      const auto sign = UINT64_C(1) << (width - 1);
      bits = (a.bits ^ sign) < (b.bits ^ sign);
      break;
    }
    case Op::udiv:
    case Op::sdiv:
    case Op::umulh:
    case Op::smulh:
    case Op::clz:
    case Op::rbit: {
      const std::array operands{a.bits, b.bits};
      const auto folded = ir::FoldPure(node, std::span(operands).first(descriptor->arity), width);
      if (!folded) return {TargetKind::unknown, 0, dependency};
      bits = *folded;
      break;
    }
    default:
      return {TargetKind::unknown, 0, dependency};
  }

  return {TargetKind::absolute_runtime, bits & Mask(node.width), dependency};
}

// Image provenance travels with a computed value even when its location is
// lost. A load is where provenance stops: what an image address points at is
// data, and a relocated slot already abstracts to the location it holds.
Value Abstract(const ir::Node& node, std::span<const ir::Node> nodes, std::span<const Value> values,
               const ImageFacts& facts_in) {
  auto result = AbstractValue(node, nodes, values, facts_in);
  result.image_derived |= result.kind == TargetKind::image_location;
  const auto* descriptor = ir::Descriptor(node.op);
  if (descriptor && descriptor->produces_value && node.op != Op::load &&
      node.op != Op::exclusive_load)
    for (unsigned input = 0; input < descriptor->arity; ++input)
      result.image_derived |= values[node.inputs[input]].image_derived;
  return result;
}

// The lifted form of an unsigned "greater than a literal": carry set and zero
// clear after subtracting it. Taking the other arm of such a branch is what
// bounds the value, and the bound is what makes a table's successor set
// complete rather than merely observed.
// x >u N, as a target lifts it: N <u x, not x <u N+1, or the AArch64 flags
// form not-carry and not-zero of x - N. Each is true exactly when x exceeds N,
// so the false arm bounds x by N inclusive.
std::optional<std::pair<ir::ValueId, std::uint64_t>> UnsignedAbove(
    ir::ValueId condition, std::span<const ir::Node> nodes) {
  const auto& root = nodes[condition];
  if (root.op == Op::unsigned_less && root.width == 1 && nodes[root.inputs[0]].op == Op::constant)
    return std::pair{ir::Unwrap(root.inputs[1], nodes), nodes[root.inputs[0]].immediate};
  if (root.op == Op::bit_not && root.width == 1) {
    const auto& less = nodes[root.inputs[0]];
    if (less.op == Op::unsigned_less && less.width == 1 &&
        nodes[less.inputs[1]].op == Op::constant && nodes[less.inputs[1]].immediate != 0)
      return std::pair{ir::Unwrap(less.inputs[0], nodes), nodes[less.inputs[1]].immediate - 1};
    return {};
  }

  if (root.op != Op::bit_and) return {};
  const auto& carry = nodes[root.inputs[0]];
  const auto& zero = nodes[root.inputs[1]];
  if (carry.op != Op::bit_not || zero.op != Op::bit_not) return {};
  const auto& below = nodes[carry.inputs[0]];
  const auto& equal = nodes[zero.inputs[0]];
  if (below.op != Op::unsigned_less || equal.op != Op::equal) return {};
  const auto& difference = nodes[equal.inputs[0]];
  const auto& against = nodes[below.inputs[1]];
  if (difference.op != Op::sub || against.op != Op::constant) return {};
  if (nodes[equal.inputs[1]].op != Op::constant || nodes[equal.inputs[1]].immediate != 0) return {};
  const auto& limit = nodes[difference.inputs[1]];
  if (limit.op != Op::constant || limit.immediate != against.immediate) return {};
  if (ir::Unwrap(below.inputs[0], nodes) != ir::Unwrap(difference.inputs[0], nodes)) return {};
  return std::pair{ir::Unwrap(below.inputs[0], nodes), against.immediate};
}

// Scratch reused across boundaries so splitting a destination costs no
// per-boundary allocation.
struct SplitScratch {
  std::vector<ir::ValueId> cone;
  std::vector<ir::ValueId> stack;
  std::vector<std::uint32_t> seen;
  std::vector<Value> local;
  std::uint32_t stamp = 0;
};

struct Split {
  ir::ValueId condition;
  Value when_true;
  Value when_false;
};

// Flattening replaces an original conditional branch with a table index chosen
// by that same condition, so a destination, or a guard on it, rests on one
// unresolved choice rather than on none. Evaluating the root once per arm
// recovers what each arm makes of it; what counts as settled is the caller's
// question. A second unresolved choice is left alone, because enumerating a
// product of arms is a different claim from reading one branch.
std::optional<Split> SplitOnChoice(ir::ValueId target, std::span<const ir::Node> nodes,
                                   std::span<const Value> values, const ImageFacts& facts,
                                   std::span<const std::uint8_t> image_dependent,
                                   SplitScratch& scratch, Budget& budget) {
  if (scratch.seen.size() != nodes.size()) scratch.seen.assign(nodes.size(), 0);
  ++scratch.stamp;
  scratch.cone.clear();
  scratch.stack.assign(1, target);
  std::optional<ir::ValueId> choice;
  while (!scratch.stack.empty()) {
    if (budget.try_consume({4, 0}) != BudgetDecline::none) return {};
    const auto id = scratch.stack.back();
    scratch.stack.pop_back();
    if (scratch.seen[id] == scratch.stamp) continue;
    scratch.seen[id] = scratch.stamp;
    scratch.cone.push_back(id);
    const auto& node = nodes[id];

    // The arms are already valued; descending through them would only find
    // choices whose own arms are unresolved, which cannot resolve anyway.
    if (node.op == Op::select && values[id].kind == TargetKind::unknown &&
        values[node.inputs[0]].kind != TargetKind::absolute_runtime) {
      if (choice && *choice != id) return {};
      choice = id;
      continue;
    }

    // Descend through a load as well: the table read is the whole point, and
    // the choice that selects its index sits underneath it.
    const auto* descriptor = ir::Descriptor(node.op);
    if (descriptor == nullptr || !descriptor->produces_value) continue;
    for (unsigned i = 0; i < descriptor->arity; ++i) scratch.stack.push_back(node.inputs[i]);
  }

  if (!choice) return {};
  if (budget.try_consume({scratch.cone.size() * 2 + values.size() * 2,
                          values.size() * sizeof(Value) * 2}) != BudgetDecline::none) {
    return {};
  }

  std::sort(scratch.cone.begin(), scratch.cone.end());
  const auto& select = nodes[*choice];
  Split split{select.inputs[0], {}, {}};
  for (unsigned arm = 0; arm < 2; ++arm) {
    scratch.local.assign(values.begin(), values.end());
    scratch.local[*choice] = values[select.inputs[arm == 0 ? 1 : 2]];
    for (const auto id : scratch.cone) {
      if (id == *choice) continue;
      scratch.local[id] = Abstract(nodes[id], nodes, scratch.local, facts);
      if (id < image_dependent.size() && image_dependent[id])
        scratch.local[id].constant_image = true;
    }

    (arm == 0 ? split.when_true : split.when_false) = scratch.local[target];
  }

  return split;
}

// Two known destinations are the conditional the transformation hid, not an
// unknown successor.
std::optional<Split> SplitTarget(ir::ValueId target, std::span<const ir::Node> nodes,
                                 std::span<const Value> values, const ImageFacts& facts,
                                 std::span<const std::uint8_t> image_dependent,
                                 SplitScratch& scratch, Budget& budget) {
  auto split = SplitOnChoice(target, nodes, values, facts, image_dependent, scratch, budget);
  if (!split || split->when_true.kind != TargetKind::image_location ||
      split->when_false.kind != TargetKind::image_location)
    return {};
  return split;
}

// A guard that takes the same side under both arms of its only unresolved
// choice takes that side: the choice cannot change which edge is followed. A
// dispatcher's range check over a state chosen from two in-range values is the
// case that needs it.
std::optional<Value> AgreedCondition(ir::ValueId condition, std::span<const ir::Node> nodes,
                                     std::span<const Value> values, const ImageFacts& facts,
                                     std::span<const std::uint8_t> image_dependent,
                                     SplitScratch& scratch, Budget& budget) {
  const auto split =
      SplitOnChoice(condition, nodes, values, facts, image_dependent, scratch, budget);
  if (!split || split->when_true.kind != TargetKind::absolute_runtime ||
      split->when_false.kind != TargetKind::absolute_runtime ||
      (split->when_true.bits != 0) != (split->when_false.bits != 0))
    return {};
  return Value{TargetKind::absolute_runtime, split->when_true.bits != 0 ? 1U : 0U,
               split->when_true.constant_image || split->when_false.constant_image};
}

// Every destination a bounded table dispatch can reach. The guard proves the
// index lies in [0, bound]; evaluating the target once per value enumerates the
// set, and any value that does not settle on an image location abandons the
// whole set rather than reporting a partial one.
std::optional<BoundedTargets> EnumerateDispatch(
    ir::ValueId target, ir::ValueId index, std::uint64_t bound, std::span<const ir::Node> nodes,
    std::span<const Value> values, const ImageFacts& facts,
    std::span<const std::uint8_t> image_dependent, SplitScratch& scratch, Budget& budget) {
  if (bound >= 4096) return {};
  if (scratch.seen.size() != nodes.size()) scratch.seen.assign(nodes.size(), 0);
  ++scratch.stamp;
  scratch.cone.clear();
  scratch.stack.assign(1, target);
  while (!scratch.stack.empty()) {
    if (budget.try_consume({4, 0}) != BudgetDecline::none) return {};
    const auto id = scratch.stack.back();
    scratch.stack.pop_back();
    if (scratch.seen[id] == scratch.stamp) continue;
    scratch.seen[id] = scratch.stamp;
    scratch.cone.push_back(id);
    if (id == index) continue;
    const auto* descriptor = ir::Descriptor(nodes[id].op);
    if (descriptor == nullptr || !descriptor->produces_value) continue;
    for (unsigned i = 0; i < descriptor->arity; ++i) scratch.stack.push_back(nodes[id].inputs[i]);
  }

  if (scratch.seen[index] != scratch.stamp) return {};
  std::sort(scratch.cone.begin(), scratch.cone.end());
  if (budget.try_consume({values.size() + (bound + 1) * scratch.cone.size(),
                          values.size() * sizeof(Value) + (bound + 1) * sizeof(std::uint64_t)}) !=
      BudgetDecline::none) {
    return {};
  }

  BoundedTargets result;
  result.destinations.reserve(static_cast<std::size_t>(bound) + 1);
  scratch.local.assign(values.begin(), values.end());
  for (std::uint64_t value = 0; value <= bound; ++value) {
    scratch.local[index] = {TargetKind::absolute_runtime, value, false};
    for (const auto id : scratch.cone) {
      if (id == index) continue;
      scratch.local[id] = Abstract(nodes[id], nodes, scratch.local, facts);
      if (id < image_dependent.size() && image_dependent[id])
        scratch.local[id].constant_image = true;
    }

    const auto& resolved = scratch.local[target];
    if (resolved.kind != TargetKind::image_location) return {};
    result.destinations.push_back(resolved.bits);
    result.constant_image_dependency |= resolved.constant_image;
  }

  return result;
}

}  // namespace

std::optional<std::pair<ir::ValueId, std::uint64_t>> BoundFromGuard(std::span<const ir::Node> nodes,
                                                                    ir::ValueId condition) {
  return UnsignedAbove(condition, nodes);
}

BoundedTargets EnumerateBoundedTarget(std::span<const ir::Node> nodes, ir::ValueId target,
                                      ir::ValueId index, std::uint64_t bound,
                                      const ImageFacts& facts, Budget& budget) {
  if (budget.try_consume({nodes.size(), nodes.size() * sizeof(Value)}) != BudgetDecline::none) {
    return {};
  }

  std::vector<Value> values;
  values.reserve(nodes.size());
  for (const auto& node : nodes) values.push_back(Abstract(node, nodes, values, facts));
  SplitScratch scratch;
  auto destinations =
      EnumerateDispatch(target, index, bound, nodes, values, facts, {}, scratch, budget);
  return destinations ? std::move(*destinations) : BoundedTargets{};
}

std::optional<ImageWriteScan> KnownImageWrites(const ir::Block& block, const ImageFacts& facts,
                                               Budget& budget) {
  const auto nodes = block.nodes();
  if (budget.try_consume({nodes.size() * 8ULL, nodes.size() * sizeof(Value)}) !=
      BudgetDecline::none) {
    return {};
  }

  std::vector<Value> values;
  ImageWriteScan scan;
  values.reserve(nodes.size());
  for (std::size_t id = 0; id < nodes.size(); ++id) {
    const auto& node = nodes[id];
    if (ir::MayWriteMemory(node.op) && node.width != 0 && node.width % 8 == 0) {
      const auto& address = values[node.inputs[0]];
      const auto width = node.op == Op::exclusive_store ? nodes[node.inputs[1]].width : node.width;
      if (const auto at = Location(address, nodes[node.inputs[0]].width, facts)) {
        if (budget.try_consume({1, sizeof(ir::ImageWrite)}) != BudgetDecline::none) return {};
        scan.writes.push_back({*at, width, static_cast<ir::ValueId>(id)});
      } else if (address.image_derived) {
        if (scan.unresolved == std::numeric_limits<std::uint64_t>::max()) return {};
        ++scan.unresolved;
      }
    }

    values.push_back(Abstract(node, nodes, values, facts));
  }

  return scan;
}

ControlResult AnalyzeControl(const ir::Block& block, Budget& budget, ir::BlockLimits limits,
                             ImageFacts facts_in) {
  const auto valid = ir::Validate(block, budget, limits);
  if (valid != ir::BlockDecline::none) {
    return {{},
            valid == ir::BlockDecline::resource_limit ? ControlDecline::resource_limit
                                                      : ControlDecline::invalid_ir};
  }

  const auto boundaries = block.boundaries();
  const auto& transfer = boundaries.back().transfer;
  if (!transfer) return {{}, ControlDecline::no_transfer};
  if (budget.try_consume({block.nodes().size() * 16ULL, block.nodes().size() * sizeof(Value)}) !=
      BudgetDecline::none) {
    return {{}, ControlDecline::resource_limit};
  }

  std::vector<Value> values;
  values.reserve(block.nodes().size());
  for (const auto& node : block.nodes()) {
    values.push_back(Abstract(node, block.nodes(), values, facts_in));
  }

  ControlFacts facts{block.revision(), block.sources().back().source_address(), transfer->kind};
  const auto target = [&](ir::ValueId id) {
    return SymbolicTarget{values[id].kind, values[id].bits, id, values[id].constant_image};
  };

  const auto append = [&](EdgeRole role, ir::ValueId id, std::optional<ir::ValueId> condition = {},
                          std::optional<bool> when = {}) {
    facts.edges[facts.edge_count++] = {role, target(id), condition, when};
  };

  if (transfer->kind == ir::TransferKind::conditional) {
    append(EdgeRole::branch, transfer->target, transfer->condition, true);
    append(EdgeRole::branch, *transfer->alternative, transfer->condition, false);
  } else {
    const auto role = transfer->kind == ir::TransferKind::call      ? EdgeRole::callee
                      : transfer->kind == ir::TransferKind::return_ ? EdgeRole::return_
                                                                    : EdgeRole::branch;
    const auto& node = block.nodes()[transfer->target];
    if (node.op == Op::select && values[node.inputs[0]].kind == TargetKind::unknown) {
      append(role, node.inputs[1], node.inputs[0], true);
      append(role, node.inputs[2], node.inputs[0], false);
    } else if (values[transfer->target].kind == TargetKind::unknown) {
      // The destination is opaque but may rest on a single choice whose arms
      // both settle. Both edges name the same target expression: one transfer
      // with two guarded outcomes, which is the branch flattening concealed.
      SplitScratch scratch;
      const auto split =
          SplitTarget(transfer->target, block.nodes(), values, facts_in, {}, scratch, budget);
      if (split) {
        facts.edges[facts.edge_count++] = {role,
                                           {split->when_true.kind, split->when_true.bits,
                                            transfer->target, split->when_true.constant_image},
                                           split->condition,
                                           true};
        facts.edges[facts.edge_count++] = {role,
                                           {split->when_false.kind, split->when_false.bits,
                                            transfer->target, split->when_false.constant_image},
                                           split->condition,
                                           false};
      } else
        append(role, transfer->target);
    } else {
      append(role, transfer->target);
    }

    if (transfer->kind == ir::TransferKind::call) {
      append(EdgeRole::potential_return, *transfer->continuation);
      facts.callee_return_unknown = true;
    }
  }

  return {facts, ControlDecline::none};
}

namespace {
PathControlResult AnalyzePathControlImpl(const ir::Path& path, const ir::RecoveredPath* recovered,
                                         const ImageFacts& facts_in,
                                         PathImageDependencies image_dependencies, Budget& budget,
                                         ir::BlockLimits limits) {
  const auto valid = recovered ? ir::ValidateRecoveredPath(*recovered, budget, limits, facts_in)
                               : ir::ValidatePath(path, budget, limits);
  if (valid != ir::BlockDecline::none) {
    return {{},
            valid == ir::BlockDecline::resource_limit ? ControlDecline::resource_limit
                                                      : ControlDecline::invalid_ir};
  }

  const auto resource = [] { return PathControlResult{{}, ControlDecline::resource_limit}; };
  if (image_dependencies.nodes.size() > path.nodes().size() ||
      budget.try_consume({image_dependencies.nodes.size(), 0}) != BudgetDecline::none)
    return resource();
  if (image_dependencies.basis_revision != path.revision()) {
    return {{}, ControlDecline::invalid_ir};
  }

  std::optional<ir::ValueId> previous;
  for (const auto id : image_dependencies.nodes) {
    if ((previous && id <= *previous) || id >= path.nodes().size() ||
        (path.nodes()[id].op != Op::constant && path.nodes()[id].op != Op::image_address))
      return {{}, ControlDecline::invalid_ir};
    previous = id;
  }

  if (!image_dependencies.nodes.empty() &&
      budget.try_consume({path.nodes().size(), path.nodes().size() * sizeof(std::uint8_t)}) !=
          BudgetDecline::none)
    return resource();
  std::vector<std::uint8_t> image_dependent(image_dependencies.nodes.empty() ? 0
                                                                             : path.nodes().size());
  for (const auto id : image_dependencies.nodes) image_dependent[id] = 1;

  // A recovered dispatch names destinations that are no node of any
  // instruction; they sit above the basis, so the sequence a transfer's ids
  // index is the basis followed by them.
  const auto named = recovered ? recovered->destinations() : std::span<const ir::Node>{};
  std::vector<ir::Node> appended;
  auto nodes = path.nodes();
  if (!named.empty()) {
    if (budget.try_consume(
            {nodes.size() + named.size(), (nodes.size() + named.size()) * sizeof(ir::Node)}) !=
        BudgetDecline::none)
      return resource();
    appended.assign(nodes.begin(), nodes.end());
    appended.insert(appended.end(), named.begin(), named.end());
    nodes = appended;
  }

  if (budget.try_consume({nodes.size(), nodes.size() * (sizeof(Value) + sizeof(std::uint32_t) +
                                                        sizeof(ir::ValueId) * 2)}) !=
          BudgetDecline::none ||
      budget.try_consume(
          {path.boundaries().size(), path.boundaries().size() * sizeof(BoundaryControl)}) !=
          BudgetDecline::none)
    return resource();
  // Every slot exists from the start so a destination above the basis can be
  // valued before the boundary whose transfer names it; a slot no boundary has
  // reached yet stays unknown, which is what it was.
  std::vector<Value> values(nodes.size());
  for (std::size_t i = path.nodes().size(); i < nodes.size(); ++i)
    values[i] = Abstract(nodes[i], nodes, values, facts_in);
  // A destination is a bare place in the image by the time it is a node, but
  // declared bytes are what made it one, so the dependency travels with it.
  if (!named.empty()) {
    std::size_t at = path.nodes().size();
    for (const auto& rewrite : recovered->rewrites()) {
      if (rewrite.rule != ir::RewriteRule::dispatch_branch) continue;
      for (unsigned i = 0; i < 2 && at < nodes.size(); ++i, ++at)
        values[at].constant_image = !rewrite.witness.empty();
    }
  }

  std::vector<std::uint32_t> visited(nodes.size(), 0);
  std::vector<ir::ValueId> pending, loads;
  SplitScratch scratch;
  pending.reserve(nodes.size());
  loads.reserve(nodes.size());
  PathControlFacts facts{recovered ? recovered->revision() : path.revision(),
                         static_cast<std::uint32_t>(path.boundaries().size()),
                         {},
                         {}};
  facts.path_identity = recovered ? recovered->identity() : 0;
  facts.boundaries.reserve(path.boundaries().size());
  for (std::uint32_t index = 0; index < path.boundaries().size(); ++index) {
    const auto& boundary = path.boundaries()[index];
    if (recovered && budget.try_consume({std::bit_width(recovered->rewrites().size()) + 1ULL, 0}) !=
                         BudgetDecline::none)
      return resource();
    const auto effective = recovered ? recovered->effective_transfer(index) : boundary.transfer;
    if (budget.try_consume({static_cast<std::uint64_t>(boundary.node_count) * 16 + 16, 0}) !=
        BudgetDecline::none)
      return resource();
    for (std::size_t i = boundary.first_node;
         i < static_cast<std::size_t>(boundary.first_node) + boundary.node_count; ++i) {
      values[i] = Abstract(nodes[i], nodes, values, facts_in);
      if (!image_dependent.empty() && image_dependent[i]) values[i].constant_image = true;
    }

    const ir::ConditionalRewrite* rewrite = nullptr;
    if (recovered) {
      const auto rewrites = recovered->rewrites();
      const auto found =
          std::lower_bound(rewrites.begin(), rewrites.end(), index,
                           [](const ir::ConditionalRewrite& item, std::uint32_t boundary) {
                             return item.boundary < boundary;
                           });
      if (found != rewrites.end() && found->boundary == index) rewrite = &*found;
    }

    const bool folded_condition_image = rewrite &&
                                        rewrite->rule == ir::RewriteRule::folded_condition &&
                                        values[rewrite->condition].constant_image;
    if (rewrite && rewrite->rule == ir::RewriteRule::dispatch_branch) {
      const bool original_image = values[rewrite->original.target].constant_image ||
                                  values[rewrite->condition].constant_image;
      values[rewrite->replacement.target].constant_image |= original_image;
      values[*rewrite->replacement.alternative].constant_image |= original_image;
    }

    BoundaryControl result{};
    result.boundary = index;
    result.source_address = path.sources()[index].source_address();
    result.expected_image_successor = path.expected_successor(index);
    const auto append_resolved = [&](PathEdgeRole role, ir::ValueId value, const Value& resolved,
                                     std::optional<ir::ValueId> condition,
                                     std::optional<bool> when) {
      std::optional<bool> known;
      std::optional<Value> decided;
      if (condition && values[*condition].kind == TargetKind::absolute_runtime)
        decided = values[*condition];
      else if (condition)
        decided =
            AgreedCondition(*condition, nodes, values, facts_in, image_dependent, scratch, budget);
      if (decided) known = decided->bits != 0;

      // A guard folded from a constant read decides the edge just as a folded
      // destination does, so the dependency has to travel with either one.
      const bool rests_on_image = resolved.constant_image ||
                                  (condition && values[*condition].constant_image) ||
                                  (decided && decided->constant_image) || folded_condition_image;
      result.edges[result.edge_count++] = {role,      resolved.kind, resolved.bits, value,
                                           condition, when,          known,         rests_on_image};
    };

    const auto append = [&](PathEdgeRole role, ir::ValueId value,
                            std::optional<ir::ValueId> condition = {},
                            std::optional<bool> when = {}) {
      append_resolved(role, value, values[value], condition, when);
    };

    if (!effective) {
      result.edges[result.edge_count++] = {
          PathEdgeRole::fallthrough,
          TargetKind::image_location,
          result.source_address + path.sources()[index].bytes().size(),
          {},
          {},
          {},
          {},
          false};
    } else {
      const auto& transfer = *effective;
      result.transfer_kind = transfer.kind;
      result.transfer_target = transfer.target;
      if (transfer.kind == ir::TransferKind::conditional) {
        append(PathEdgeRole::branch, transfer.target, transfer.condition, true);
        append(PathEdgeRole::branch, *transfer.alternative, transfer.condition, false);
      } else {
        const auto role = transfer.kind == ir::TransferKind::call      ? PathEdgeRole::callee
                          : transfer.kind == ir::TransferKind::return_ ? PathEdgeRole::return_
                                                                       : PathEdgeRole::branch;
        const auto& target = nodes[transfer.target];
        if (target.op == Op::select) {
          append(role, target.inputs[1], target.inputs[0], true);
          append(role, target.inputs[2], target.inputs[0], false);
        } else if (values[transfer.target].kind == TargetKind::unknown) {
          // The destination itself is opaque, but it may rest on a single
          // choice whose arms both settle. Both edges name the same target
          // expression: it is one transfer with two guarded outcomes.
          const auto split = SplitTarget(transfer.target, nodes, values, facts_in, image_dependent,
                                         scratch, budget);
          if (split) {
            append_resolved(role, transfer.target, split->when_true, split->condition, true);
            append_resolved(role, transfer.target, split->when_false, split->condition, false);
          } else {
            append(role, transfer.target);

            // No single destination, and no single choice behind it. The index
            // may still be bounded by a guard this path already passed, and a
            // bound turns the table into a complete successor set.
            for (const auto& earlier : facts.boundaries) {
              if (earlier.transfer_kind != ir::TransferKind::conditional) continue;
              if (!earlier.expected_image_successor || earlier.edge_count < 2) continue;
              const auto& alternative = earlier.edges[1];
              if (alternative.when != false || !alternative.condition) continue;
              if (alternative.target_kind != TargetKind::image_location ||
                  alternative.target_address != *earlier.expected_image_successor)
                continue;
              const auto bound = UnsignedAbove(*alternative.condition, nodes);
              if (!bound) continue;
              auto destinations =
                  EnumerateDispatch(transfer.target, bound->first, bound->second, nodes, values,
                                    facts_in, image_dependent, scratch, budget);
              if (!destinations) continue;
              result.dispatch = BoundedDispatch{
                  bound->first, bound->second, earlier.boundary,
                  std::move(destinations->destinations),
                  destinations->constant_image_dependency || alternative.constant_image_dependency};
              break;
            }
          }
        } else
          append(role, transfer.target);
        if (transfer.kind == ir::TransferKind::call) {
          append(PathEdgeRole::potential_return, *transfer.continuation);
          result.callee_return_unknown = true;
        }
      }
    }

    if (result.expected_image_successor) {
      bool all_match = true, all_mismatch = true;
      for (unsigned i = 0; i < result.edge_count; ++i) {
        const auto& edge = result.edges[i];
        if (edge.role == PathEdgeRole::potential_return ||
            (edge.known_condition && edge.known_condition != edge.when))
          continue;
        if (edge.target_kind != TargetKind::image_location) {
          all_match = all_mismatch = false;
        } else if (edge.target_address == *result.expected_image_successor) {
          all_mismatch = false;
        } else
          all_match = false;
      }

      result.expected_match = all_match      ? ExpectedMatch::always
                              : all_mismatch ? ExpectedMatch::never
                                             : ExpectedMatch::unknown;
    }

    // Traverse the effective target and guard DAG, including both select arms.
    // This is deliberately conservative structural dependency provenance.
    const auto enqueue = [&](ir::ValueId value) {
      if (visited[value] != index + 1) {
        visited[value] = index + 1;
        pending.push_back(value);
      }
    };

    loads.clear();
    if (effective) {
      enqueue(effective->target);
      if (effective->condition) enqueue(*effective->condition);
      if (effective->alternative) enqueue(*effective->alternative);
      if (effective->continuation) enqueue(*effective->continuation);
    }

    // A replacement rests on the expression it replaced, which still executes,
    // so the original destination's loads remain dependencies of this boundary.
    if (boundary.transfer) {
      enqueue(boundary.transfer->target);
      if (boundary.transfer->condition) enqueue(*boundary.transfer->condition);
      if (boundary.transfer->alternative) enqueue(*boundary.transfer->alternative);
      if (boundary.transfer->continuation) enqueue(*boundary.transfer->continuation);
    }
    while (!pending.empty()) {
      if (budget.try_consume({4, 0}) != BudgetDecline::none) return resource();
      const auto id = pending.back();
      pending.pop_back();
      const auto& node = nodes[id];
      if (node.op == Op::load || node.op == Op::exclusive_load) loads.push_back(id);
      const auto* descriptor = ir::Descriptor(node.op);
      for (unsigned i = 0; i < descriptor->arity; ++i) enqueue(node.inputs[i]);
    }

    if (budget.try_consume({loads.size() * (std::bit_width(loads.size()) + 1),
                            loads.size() * sizeof(ir::ValueId)}) != BudgetDecline::none)
      return resource();
    std::sort(loads.begin(), loads.end());
    result.load_dependencies.assign(loads.begin(), loads.end());
    const bool diverges = result.expected_match == ExpectedMatch::never;
    facts.boundaries.push_back(std::move(result));
    if (diverges) {
      facts.proved_divergence = index;
      break;
    }
  }

  return {std::move(facts), ControlDecline::none};
}

}  // namespace

PathControlResult AnalyzePathControl(const ir::Path& path, Budget& budget, ir::BlockLimits limits,
                                     ImageFacts facts_in) {
  return AnalyzePathControlImpl(path, nullptr, facts_in, {path.revision(), {}}, budget, limits);
}

PathControlResult AnalyzePathControl(const ir::Path& path, PathImageDependencies image_dependencies,
                                     Budget& budget, ir::BlockLimits limits, ImageFacts facts_in) {
  return AnalyzePathControlImpl(path, nullptr, facts_in, image_dependencies, budget, limits);
}

PathControlResult AnalyzePathControl(const ir::RecoveredPath& path,
                                     PathImageDependencies image_dependencies, Budget& budget,
                                     ir::BlockLimits limits, ImageFacts facts_in) {
  return AnalyzePathControlImpl(path.basis(), &path, facts_in, image_dependencies, budget, limits);
}

}  // namespace nyx::analysis
