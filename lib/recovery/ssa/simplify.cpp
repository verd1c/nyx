#include "nyx/recovery/ssa/simplify.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <tuple>

#include "nyx/ir/fold.hpp"
#include "nyx/recovery/ssa/splice.hpp"

namespace nyx::recovery {
namespace {
using ir::Node;
using ir::Op;
using ir::ValueId;

constexpr std::uint64_t kPageMask = ~std::uint64_t{0xfff};
constexpr unsigned kMaxSweeps = 8;
constexpr unsigned kMaxInsertions = 16;
constexpr unsigned kMaxBooleanDepth = 4;
constexpr unsigned kMaxCaseCone = 24;
constexpr unsigned kMaxMaskDepth = 12;
constexpr unsigned kMaxBitwiseCone = 16;
constexpr unsigned kMaxBitwiseLeaves = 3;

SsaSimplifyResult Decline(SsaSimplifyRefusal reason) {
  SsaSimplifyResult result;
  result.reason = reason;
  return result;
}

struct Rule {
  Node node;
  std::string_view name;
  bool placement = false;

  // Nodes spliced in before the rewritten one, named by the ids they get:
  // the k-th becomes `id + k`, and the rewritten node moves past them.
  std::vector<Node> inserted = {};
};

// A rebuilt identity names its operands in whatever order its leaves were
// found, so a commutative operation with them swapped is no rewrite at all:
// counting it as one flips the node on every sweep and never settles.
bool Same(const Node& a, const Node& b) {
  if (a.op != b.op || a.width != b.width || a.immediate != b.immediate) return false;
  if (a.inputs == b.inputs) return true;
  return (a.op == Op::bit_xor || a.op == Op::bit_or || a.op == Op::bit_and) &&
         a.inputs[0] == b.inputs[1] && a.inputs[1] == b.inputs[0] && a.inputs[2] == b.inputs[2];
}

// An identity may rebuild its value from an operand dead_effect_free_nodes
// already retired. The retirement holds only while nothing live reads that
// value, so the rewrite would leave a live node reading one the record says
// nothing computes. Ids at or past `first_new` name nodes spliced in with it.
bool ReadsRetired(const ir::SsaBlock& block, const Node& node, ValueId first_new) {
  const auto* descriptor = ir::Descriptor(node.op);
  if (!descriptor) return true;
  for (unsigned input = 0; input < descriptor->arity; ++input)
    if (node.inputs[input] < first_new &&
        std::binary_search(block.dead_pure_nodes.begin(), block.dead_pure_nodes.end(),
                           node.inputs[input]))
      return true;
  return false;
}

class Rules {
 public:
  // `masks` caches each node's possibly-nonzero bits for one block. Every
  // rewrite keeps its node's value, so a cached mask stays an upper bound.
  Rules(std::span<const Node> nodes, bool placement,
        std::vector<std::optional<std::uint64_t>>& masks,
        std::span<const std::optional<std::uint64_t>> folded, std::optional<std::uint64_t> bias)
      : nodes_(nodes), placement_(placement), masks_(masks), folded_(folded), bias_(bias) {}

  // Whether the last rule may rest on a folded load's declared value.
  bool UsedFolded() const { return used_folded_; }

  std::optional<Rule> Apply(ValueId id) const {
    used_folded_ = false;
    const auto& node = nodes_[id];
    const auto* descriptor = ir::Descriptor(node.op);
    if (!descriptor || descriptor->effect != ir::Effect::pure || !descriptor->produces_value ||
        !node.width || node.width > 64 || node.op == Op::constant || node.op == Op::image_address ||
        node.op == Op::read)
      return std::nullopt;
    for (unsigned input = 0; input < descriptor->arity; ++input)
      if (node.inputs[input] >= id) return std::nullopt;
    if (auto folded = Fold(node, descriptor->arity)) return folded;
    if (!Mask(id)) return Constant(node.width, 0, "known_zero");
    const auto a = node.inputs[0], b = node.inputs[1], c = node.inputs[2];
    const auto w = node.width;
    switch (node.op) {
      case Op::add:
        if (Is(b, 0)) return Copy(a, w, "add_zero");
        if (Is(a, 0)) return Copy(b, w, "add_zero");
        if (w == 64 && Image(a) && Literal(b))
          return Rule{Image(nodes_[a].immediate + *Literal(b)), "image_offset"};
        if (w == 64 && Image(b) && Literal(a))
          return Rule{Image(nodes_[b].immediate + *Literal(a)), "image_offset"};
        break;
      case Op::sub:
        if (Is(b, 0)) return Copy(a, w, "sub_zero");
        if (a == b) return Constant(w, 0, "sub_self");
        if (w == 64 && Image(a) && Literal(b))
          return Rule{Image(nodes_[a].immediate - *Literal(b)), "image_offset"};
        break;
      case Op::mul:
        if (Is(b, 1)) return Copy(a, w, "mul_one");
        if (Is(a, 1)) return Copy(b, w, "mul_one");
        if (Is(a, 0) || Is(b, 0)) return Constant(w, 0, "mul_zero");
        break;
      case Op::bit_and:
        for (const auto [x, y] : {std::pair{a, b}, std::pair{b, a}}) {
          // A mask keeping every bit x can set changes nothing.
          if (Literal(y) && !(Mask(x) & ~*Literal(y) & ir::LowMask(w)))
            return Copy(x, w, "and_covers");
          if (Is(y, ir::LowMask(w))) return Copy(x, w, "and_ones");
          if (Is(y, 0)) return Constant(w, 0, "and_zero");

          // A 0/1 value keeps itself under a mask that keeps bit 0.
          if (Boolean(x, 0) && Literal(y) && (*Literal(y) & 1)) return Copy(x, w, "and_boolean");
          if (nodes_[x].op == Op::zext && Literal(y) &&
              (*Literal(y) & ir::LowMask(nodes_[nodes_[x].inputs[0]].width)) ==
                  ir::LowMask(nodes_[nodes_[x].inputs[0]].width))
            return Copy(x, w, "and_extended");
          if (placement_ && w == 64 && Image(x) && Is(y, kPageMask))
            return Rule{Image(nodes_[x].immediate & kPageMask), "image_page", true};
          // Placed on a page boundary, a location's in-page bits are its own.
          if (placement_ && w == 64 && Image(x) && Literal(y) && !(*Literal(y) & kPageMask))
            return Rule{Node{Op::constant, w, {}, nodes_[x].immediate & *Literal(y)}, "image_low",
                        true};
        }

        if (a == b) return Copy(a, w, "and_self");
        break;
      case Op::bit_or:
        if (!Mask(b)) return Copy(a, w, "or_zero");
        if (!Mask(a)) return Copy(b, w, "or_zero");
        if (Is(b, 0)) return Copy(a, w, "or_zero");
        if (Is(a, 0)) return Copy(b, w, "or_zero");
        if (a == b) return Copy(a, w, "or_self");
        break;
      case Op::bit_xor:
        if (!Mask(b)) return Copy(a, w, "xor_zero");
        if (!Mask(a)) return Copy(b, w, "xor_zero");
        if (Is(b, 0)) return Copy(a, w, "xor_zero");
        if (Is(a, 0)) return Copy(b, w, "xor_zero");
        if (a == b) return Constant(w, 0, "xor_self");
        if (Is(b, ir::LowMask(w))) return Rule{Node{Op::bit_not, w, {a}}, "xor_ones"};
        if (Is(a, ir::LowMask(w))) return Rule{Node{Op::bit_not, w, {b}}, "xor_ones"};
        break;
      case Op::bit_not:
        if (nodes_[a].op == Op::bit_not) return Copy(nodes_[a].inputs[0], w, "not_not");
        break;
      case Op::shl:
      case Op::lshr:
      case Op::ashr:
        if (Is(b, 0)) return Copy(a, w, "shift_zero");

        // A left shift discards the bits a mask above them would clear.
        if (node.op == Op::shl && Literal(b) && *Literal(b) < w && nodes_[a].op == Op::bit_and) {
          const auto kept = ir::LowMask(w - static_cast<unsigned>(*Literal(b)));
          for (const auto [x, y] : {std::pair{nodes_[a].inputs[0], nodes_[a].inputs[1]},
                                    std::pair{nodes_[a].inputs[1], nodes_[a].inputs[0]}})
            if (Literal(y) && (*Literal(y) & kept) == kept)
              return Rule{Node{Op::shl, w, {x, b}}, "shift_mask"};
        }

        break;
      case Op::extract: {
        if (node.immediate) break;
        const auto& source = nodes_[a];
        if (source.width == w) return Copy(a, w, "extract_whole");
        if (source.op == Op::zext) {
          const auto inner = source.inputs[0];
          const auto width = nodes_[inner].width;
          if (width == w) return Copy(inner, w, "extract_extended");
          if (width > w) return Rule{Node{Op::extract, w, {inner}, 0}, "extract_extended"};
          return Rule{Node{Op::zext, w, {inner}}, "extract_extended"};
        }

        if (source.op == Op::extract && !source.immediate)
          return Rule{Node{Op::extract, w, {source.inputs[0]}, 0}, "extract_extract"};
        // A mask that keeps every truncated bit changes nothing below them.
        if (source.op == Op::bit_and)
          for (const auto [x, y] : {std::pair{source.inputs[0], source.inputs[1]},
                                    std::pair{source.inputs[1], source.inputs[0]}})
            if (Literal(y) && (*Literal(y) & ir::LowMask(w)) == ir::LowMask(w))
              return Rule{Node{Op::extract, w, {x}, 0}, "extract_mask"};
        if (w == 1 && source.op == Op::select && Is(source.inputs[1], 1) && Is(source.inputs[2], 0))
          return Copy(source.inputs[0], 1, "extract_boolean");
        if (w == 1 && source.op == Op::select && Is(source.inputs[1], 0) && Is(source.inputs[2], 1))
          return Rule{Node{Op::bit_not, 1, {source.inputs[0]}}, "extract_boolean"};
        break;
      }
      case Op::zext:
        if (nodes_[a].op == Op::zext)
          return Rule{Node{Op::zext, w, {nodes_[a].inputs[0]}}, "zext_zext"};
        break;
      case Op::select:
        if (const auto condition = Literal(a)) return Copy(*condition ? b : c, w, "select_known");
        if (b == c) return Copy(b, w, "select_same");
        if (Is(b, 1) && Is(c, 0))
          return w == 1 ? Copy(a, 1, "select_boolean")
                        : Rule{Node{Op::zext, w, {a}}, "select_boolean"};
        if (w == 1 && Is(b, 0) && Is(c, 1))
          return Rule{Node{Op::bit_not, 1, {a}}, "select_boolean"};
        if (nodes_[a].op == Op::bit_not)
          return Rule{Node{Op::select, w, {nodes_[a].inputs[0], c, b}}, "select_not"};
        break;
      case Op::equal:
        if (a == b) return Constant(1, 1, "equal_self");
        for (const auto [x, y] : {std::pair{a, b}, std::pair{b, a}}) {
          const auto value = Literal(y);
          if (!value || Literal(x)) continue;
          const auto& operand = nodes_[Canonical(x)];
          if (operand.width == 1)
            return *value ? Copy(x, 1, "equal_boolean")
                          : Rule{Node{Op::bit_not, 1, {x}}, "equal_boolean"};
          // The low bit alone: (x & 1) == 1 is bit 0 of x.
          if (operand.op == Op::bit_and && *value == 1)
            for (const auto [bits, mask] : {std::pair{operand.inputs[0], operand.inputs[1]},
                                            std::pair{operand.inputs[1], operand.inputs[0]}})
              if (Is(mask, 1)) return Rule{Node{Op::extract, 1, {bits}, 0}, "equal_low_bit"};
          if (operand.op == Op::sub && *value == 0)
            return Rule{Node{Op::equal, 1, {operand.inputs[0], operand.inputs[1]}},
                        "equal_difference"};
          if (operand.op == Op::zext && nodes_[operand.inputs[0]].width == 1) {
            if (*value == 0)
              return Rule{Node{Op::bit_not, 1, {operand.inputs[0]}}, "equal_boolean"};
            if (*value == 1) return Copy(operand.inputs[0], 1, "equal_boolean");
            return Constant(1, 0, "equal_boolean");
          }

          if (operand.op == Op::select && Literal(operand.inputs[1]) &&
              Literal(operand.inputs[2])) {
            const bool on_true = *Literal(operand.inputs[1]) == *value;
            const bool on_false = *Literal(operand.inputs[2]) == *value;
            if (on_true == on_false) return Constant(1, on_true, "equal_select");
            return on_true ? Copy(operand.inputs[0], 1, "equal_select")
                           : Rule{Node{Op::bit_not, 1, {operand.inputs[0]}}, "equal_select"};
          }
        }

        break;
      case Op::unsigned_less:
        if (a == b || Is(b, 0)) return Constant(1, 0, "less_never");
        break;
      case Op::signed_less:
        if (a == b) return Constant(1, 0, "less_never");
        break;
      default:
        break;
    }

    if (auto rule = Mba(id)) return rule;
    if (auto cases = Cases(id)) return cases;
    if (auto rule = Bitwise(id)) return rule;

    // A splicing rewrite whose operands all exist already applies in place.
    if (auto rule = Insert(id); rule && rule->inserted.empty()) return rule;
    return std::nullopt;
  }

 private:
  // The two-operand MBA identities, exact modulo 2^w: a + b as
  // 2(a | b) - (a ^ b), (a ^ b) + 2(a & b) and (a | b) + (a & b), and a ^ b as
  // (a | b) - (a & b) and (a + b) - 2(a & b). Doubling may be a shift or a
  // multiply.
  std::optional<Rule> Mba(ValueId id) const {
    const auto& node = nodes_[id];
    if (node.op != Op::add && node.op != Op::sub) return std::nullopt;
    const auto w = node.width;

    // The operands of a two-input `op` at this width, in either order.
    const auto pair = [&](ValueId value, Op op) -> std::optional<std::pair<ValueId, ValueId>> {
      const auto& operand = nodes_[Canonical(value)];
      if (operand.op != op || operand.width != w) return std::nullopt;
      return std::pair{Canonical(operand.inputs[0]), Canonical(operand.inputs[1])};
    };

    const auto doubled = [&](ValueId value) -> std::optional<ValueId> {
      const auto& operand = nodes_[Canonical(value)];
      if (operand.width != w) return std::nullopt;
      if (operand.op == Op::shl && Is(operand.inputs[1], 1)) return operand.inputs[0];
      if (operand.op == Op::mul && Is(operand.inputs[1], 2)) return operand.inputs[0];
      if (operand.op == Op::mul && Is(operand.inputs[0], 2)) return operand.inputs[1];
      if (operand.op == Op::add && Canonical(operand.inputs[0]) == Canonical(operand.inputs[1]))
        return operand.inputs[0];
      return std::nullopt;
    };

    const auto same = [](std::pair<ValueId, ValueId> x, std::pair<ValueId, ValueId> y) {
      return x == y || (x.first == y.second && x.second == y.first);
    };

    const auto result = [&](Op op, std::pair<ValueId, ValueId> operands) {
      return Rule{Node{op, w, {operands.first, operands.second}}, "mba_pair"};
    };

    const auto a = node.inputs[0], b = node.inputs[1];
    if (node.op == Op::sub) {
      // 2(a | b) - (a ^ b) = a + b
      if (const auto inner = doubled(a)) {
        const auto either = pair(*inner, Op::bit_or), differ = pair(b, Op::bit_xor);
        if (either && differ && same(*either, *differ)) return result(Op::add, *either);
      }

      // (a | b) - (a & b) = a ^ b
      if (const auto either = pair(a, Op::bit_or), both = pair(b, Op::bit_and);
          either && both && same(*either, *both))
        return result(Op::bit_xor, *either);
      // (a + b) - 2(a & b) = a ^ b
      if (const auto inner = doubled(b)) {
        const auto sum = pair(a, Op::add), both = pair(*inner, Op::bit_and);
        if (sum && both && same(*sum, *both)) return result(Op::bit_xor, *sum);
      }

      return std::nullopt;
    }

    for (const auto [x, y] : {std::pair{a, b}, std::pair{b, a}}) {
      // (a ^ b) + 2(a & b) = a + b
      if (const auto inner = doubled(y)) {
        const auto differ = pair(x, Op::bit_xor), both = pair(*inner, Op::bit_and);
        if (differ && both && same(*differ, *both)) return result(Op::add, *differ);
      }

      // (a | b) + (a & b) = a + b
      if (const auto either = pair(x, Op::bit_or), both = pair(y, Op::bit_and);
          either && both && same(*either, *both))
        return result(Op::add, *either);
    }

    return std::nullopt;
  }

  // The bits a value can have set: an upper bound, never a claim they are set.
  std::uint64_t Mask(ValueId id, unsigned depth = 0) const {
    if (masks_[id]) return *masks_[id];
    const auto& node = nodes_[id];
    const auto full = node.width && node.width <= 64 ? ir::LowMask(node.width) : ~std::uint64_t{0};

    // This abstraction holds at most 64 bits. A wide intermediate is not
    // truncated to those bits: a later extract may observe its upper half.
    if (depth == kMaxMaskDepth || node.width > 64) return full;
    const auto in = [&](unsigned input) { return Mask(node.inputs[input], depth + 1); };
    const auto shift = [&]() { return Literal(node.inputs[1]); };
    std::uint64_t mask = full;
    switch (node.op) {
      case Op::constant:
        mask = *Literal(id);
        break;
      case Op::zext:
        mask = in(0);
        break;
      case Op::extract:
        if (nodes_[node.inputs[0]].width <= 64)
          mask = (node.immediate < 64 ? in(0) >> node.immediate : 0) & full;
        break;
      case Op::bit_and:
        mask = in(0) & in(1);
        break;
      case Op::bit_or:
      case Op::bit_xor:
        mask = in(0) | in(1);
        break;
      case Op::select:
        mask = in(1) | in(2);
        break;
      case Op::equal:
      case Op::unsigned_less:
      case Op::signed_less:
        mask = 1;
        break;
      case Op::shl:
        if (const auto k = shift(); k && *k < 64)
          mask = (in(0) << *k) & full;
        else if (k)
          mask = 0;
        break;
      case Op::lshr:
        if (const auto k = shift(); k && *k < 64)
          mask = in(0) >> *k;
        else if (k)
          mask = 0;
        break;
      case Op::ashr:
        // Without its sign bit an arithmetic shift is a logical one.
        if (const auto k = shift(); k && *k < 64 && !((in(0) >> (node.width - 1)) & 1))
          mask = in(0) >> *k;
        break;
      case Op::add: {
        const auto either = in(0) | in(1);
        mask = either ? ir::LowMask(std::min<unsigned>(node.width, std::bit_width(either) + 1)) : 0;
        break;
      }

      // A quotient is no larger than its dividend, and a count no larger than the width.
      case Op::udiv:
        mask = in(0) ? ir::LowMask(std::bit_width(in(0))) : 0;
        break;
      case Op::clz:
        mask = ir::LowMask(std::bit_width(node.width));
        break;
      default:
        break;
    }

    mask &= full;
    masks_[id] = mask;
    return mask;
  }

  // A bitwise function of at most three values, evaluated on every bit at once:
  // each leaf is all zeros or all ones per assignment, restricted to the bits
  // it can set, and a candidate matches where every feasible bit agrees.
  // Shallow cuts first: an inner bitwise value can be a leaf when that is
  // what exposes the identity.
  std::optional<Rule> Bitwise(ValueId id) const {
    const auto op = nodes_[id].op;
    if (op != Op::bit_and && op != Op::bit_or && op != Op::bit_xor && op != Op::bit_not)
      return std::nullopt;
    for (unsigned levels = 1; levels <= 4; ++levels)
      if (auto rule =
              Bitwise(id, levels, id, nodes_[id].width, false, ir::LowMask(nodes_[id].width)))
        return rule;
    return std::nullopt;
  }

  // The function of the cone rooted at `id`, seen at `w` bits, as a rewrite
  // of `replaced`. With `insert`, an operand of the wrong width or a missing
  // literal may be spliced in rather than refuse the candidate.
  // Only the `demanded` bits of the result must agree.
  std::optional<Rule> Bitwise(ValueId id, unsigned levels, ValueId replaced, unsigned w,
                              bool insert, std::uint64_t demanded) const {
    const auto bitwise = [&](Op op) {
      return op == Op::bit_and || op == Op::bit_or || op == Op::bit_xor || op == Op::bit_not;
    };

    const auto cast = [&](const Node& node) {
      return node.op == Op::zext || (node.op == Op::extract && !node.immediate);
    };

    // First pass, breadth first: each bitwise node's shallowest level, and how
    // many cone nodes use it. Casts carry a value without adding a level.
    constexpr std::size_t kSeen = kMaxBitwiseCone * 4;
    std::array<ValueId, kSeen> seen{};
    std::array<unsigned, kSeen> level{}, uses{};
    std::size_t known = 0;
    const auto find = [&](ValueId at) {
      return static_cast<std::size_t>(std::find(seen.begin(), seen.begin() + known, at) -
                                      seen.begin());
    };

    seen[known++] = id;
    for (std::size_t head = 0; head < known; ++head) {
      const auto& node = nodes_[seen[head]];
      if (node.op == Op::constant || node.width > 64 ||
          (seen[head] != id && !bitwise(node.op) && !cast(node)) ||
          (bitwise(node.op) && level[head] > levels))
        continue;
      const auto* descriptor = ir::Descriptor(node.op);
      for (unsigned input = 0; input < descriptor->arity; ++input) {
        const auto child = Canonical(node.inputs[input]);
        const auto at = find(child);
        if (at == known) {
          if (known == seen.size()) return std::nullopt;
          seen[known] = child;
          level[known] = level[head] + (bitwise(node.op) ? 1U : 0U);
          ++known;
        }

        ++uses[at];
      }
    }

    // Second pass: a leaf is anything not bitwise, anything past the cut, and
    // any bitwise value the cone uses twice, so a shared subexpression can be
    // one input of the identity.
    std::array<ValueId, kMaxBitwiseCone> cone{};
    std::array<ValueId, kMaxBitwiseLeaves> leaves{};
    std::size_t size = 0, count = 0;
    std::array<ValueId, kMaxBitwiseCone * 2> stack{};
    std::size_t depth = 0;
    stack[depth++] = id;
    while (depth) {
      const auto at = Canonical(stack[--depth]);
      if (std::find(cone.begin(), cone.begin() + size, at) != cone.begin() + size ||
          std::find(leaves.begin(), leaves.begin() + count, at) != leaves.begin() + count)
        continue;
      const auto& node = nodes_[at];
      const auto index = find(at);
      const bool inner =
          at == id || node.op == Op::constant || cast(node) ||
          (bitwise(node.op) && index < known && level[index] <= levels && uses[index] < 2);
      if (!inner || node.width > 64) {
        if (count == leaves.size()) return std::nullopt;
        leaves[count++] = at;
        continue;
      }

      if (size == cone.size()) return std::nullopt;
      cone[size++] = at;
      const auto* descriptor = ir::Descriptor(node.op);
      for (unsigned input = 0; node.op != Op::constant && input < descriptor->arity; ++input) {
        if (depth == stack.size()) return std::nullopt;
        stack[depth++] = node.inputs[input];
      }
    }

    if (!count) return std::nullopt;
    const auto assignments = 1U << count;
    std::array<std::uint64_t, 1U << kMaxBitwiseLeaves> value{}, feasible{};
    for (unsigned j = 0; j < assignments; ++j) {
      feasible[j] = ir::LowMask(w) & demanded;
      for (unsigned k = 0; k < count; ++k)
        if ((j >> k) & 1) feasible[j] &= Mask(leaves[k]);
      std::array<std::optional<std::uint64_t>, kMaxBitwiseCone> memo{};
      const auto result = Lanes(id, j, leaves, count, cone, size, memo);
      if (!result) return std::nullopt;
      value[j] = *result;
    }

    const auto leaf = [&](unsigned k, unsigned j) {
      return (j >> k) & 1 ? ir::LowMask(nodes_[leaves[k]].width) : std::uint64_t{0};
    };

    const auto matches = [&](auto&& candidate) {
      for (unsigned j = 0; j < assignments; ++j)
        if ((candidate(j) ^ value[j]) & feasible[j]) return false;
      return true;
    };

    if (matches([](unsigned) { return std::uint64_t{0}; })) return Constant(w, 0, "bitwise_zero");

    // A leaf at `w` bits: an existing value, or with `insert` a new cast.
    const auto operand = [&](unsigned k, std::vector<Node>& inserted) -> std::optional<ValueId> {
      if (const auto existing = Widened(leaves[k], w, replaced)) return existing;
      if (!insert) return std::nullopt;
      const auto width = nodes_[leaves[k]].width;
      inserted.push_back(width < w ? Node{Op::zext, w, {leaves[k]}}
                                   : Node{Op::extract, w, {leaves[k]}, 0});
      return replaced + static_cast<ValueId>(inserted.size() - 1);
    };

    const auto literal_node = [&](std::uint64_t value,
                                  std::vector<Node>& inserted) -> std::optional<ValueId> {
      if (const auto existing = Existing(value, w, replaced)) return existing;
      if (!insert) return std::nullopt;
      inserted.push_back(Node{Op::constant, w, {}, value & ir::LowMask(w)});
      return replaced + static_cast<ValueId>(inserted.size() - 1);
    };

    for (unsigned k = 0; k < count; ++k) {
      const auto width = nodes_[leaves[k]].width;
      if (matches([&](unsigned j) { return leaf(k, j) & ir::LowMask(w); })) {
        if (width == w) return Copy(leaves[k], w, "bitwise_leaf");
        return Rule{
            width < w ? Node{Op::zext, w, {leaves[k]}} : Node{Op::extract, w, {leaves[k]}, 0},
            "bitwise_leaf"};
      }

      if (matches([&](unsigned j) { return ~leaf(k, j) & ir::LowMask(w); })) {
        std::vector<Node> inserted;
        if (const auto value = operand(k, inserted))
          return Rule{Node{Op::bit_not, w, {*value}}, "bitwise_not", false, std::move(inserted)};
      }
    }

    // A leaf with a literal: the literal is what the function gives with the
    // leaf clear (for xor and or) or set (for and).
    for (unsigned k = 0; k < count && count == 1; ++k) {
      for (const auto op : {Op::bit_xor, Op::bit_or, Op::bit_and}) {
        const auto literal = (op == Op::bit_and ? value[1] : value[0]) & ir::LowMask(w);
        const auto candidate = [&](unsigned j) {
          const auto a = leaf(k, j);
          return op == Op::bit_xor ? a ^ literal : op == Op::bit_or ? a | literal : a & literal;
        };

        if (!matches(candidate)) continue;
        std::vector<Node> inserted;
        const auto value_id = operand(k, inserted);
        const auto constant = value_id ? literal_node(literal, inserted) : std::nullopt;
        if (value_id && constant)
          return Rule{Node{op, w, {*value_id, *constant}}, "bitwise_literal", false,
                      std::move(inserted)};
      }
    }

    for (unsigned x = 0; x < count; ++x) {
      for (unsigned y = x + 1; y < count; ++y) {
        for (const auto op : {Op::bit_xor, Op::bit_or, Op::bit_and}) {
          const auto candidate = [&](unsigned j) {
            const auto a = leaf(x, j), b = leaf(y, j);
            return op == Op::bit_xor ? a ^ b : op == Op::bit_or ? a | b : a & b;
          };

          if (!matches(candidate)) continue;
          std::vector<Node> inserted;
          const auto left = operand(x, inserted);
          const auto right = left ? operand(y, inserted) : std::nullopt;
          if (left && right)
            return Rule{Node{op, w, {*left, *right}}, "bitwise_pair", false, std::move(inserted)};
        }
      }
    }

    return std::nullopt;
  }

 public:
  // Rewrites that need new nodes; see Rule::inserted.
  std::optional<Rule> Insert(ValueId id) const {
    const auto& node = nodes_[id];
    const auto* descriptor = ir::Descriptor(node.op);
    if (!descriptor || descriptor->effect != ir::Effect::pure || !node.width || node.width > 64)
      return std::nullopt;
    for (unsigned input = 0; input < descriptor->arity; ++input)
      if (node.inputs[input] >= id) return std::nullopt;
    if (auto rule = SignExtension(id)) return rule;
    if (auto rule = Reassociate(id)) return rule;

    // A bitwise function of flags widened to words is that function of the
    // flags, widened: the obfuscator's 0/1 arithmetic becomes logic.
    if ((node.op == Op::bit_and || node.op == Op::bit_or || node.op == Op::bit_xor) &&
        node.width > 1) {
      std::array<ValueId, 2> flags{};
      bool widened = true;
      for (unsigned input = 0; input < 2; ++input) {
        const auto& operand = nodes_[Canonical(node.inputs[input])];
        widened &= operand.op == Op::zext && nodes_[operand.inputs[0]].width == 1;
        if (widened) flags[input] = operand.inputs[0];
      }

      if (widened)
        return Rule{Node{Op::zext, node.width, {id}},
                    "flag_bitwise",
                    false,
                    {Node{node.op, 1, {flags[0], flags[1]}}}};
    }

    // A nested bitwise function whose single-operation form needs a literal
    // the block does not have yet, such as (x ^ c) & x as x & ~c.
    if ((node.op == Op::bit_and || node.op == Op::bit_or || node.op == Op::bit_xor) &&
        std::any_of(node.inputs.begin(), node.inputs.begin() + 2, [&](ValueId input) {
          const auto op = nodes_[Canonical(input)].op;
          return op == Op::bit_and || op == Op::bit_or || op == Op::bit_xor || op == Op::bit_not;
        }))
      for (unsigned levels = 1; levels <= 4; ++levels)
        if (auto rule = Bitwise(id, levels, id, node.width, true, ir::LowMask(node.width));
            rule && !rule->inserted.empty())
          return rule;
    // A truncated bitwise function, rebuilt at the narrow width.
    if (node.op == Op::extract && !node.immediate) {
      const auto input = Canonical(node.inputs[0]);
      const auto op = nodes_[input].op;
      if (op == Op::bit_and || op == Op::bit_or || op == Op::bit_xor || op == Op::bit_not)
        for (unsigned levels = 1; levels <= 4; ++levels)
          if (auto rule = Bitwise(input, levels, id, node.width, true, ir::LowMask(node.width)))
            return rule;
    }

    // A left shift discards its operand's high bits, so a bitwise operand need
    // only agree on the low ones; the rebuilt operand is spliced in under it.
    if (node.op == Op::shl) {
      const auto amount = Literal(node.inputs[1]);
      const auto input = Canonical(node.inputs[0]);
      const auto op = nodes_[input].op;
      if (amount && *amount && *amount < node.width && nodes_[input].width == node.width &&
          (op == Op::bit_and || op == Op::bit_or || op == Op::bit_xor))
        for (unsigned levels = 1; levels <= 4; ++levels) {
          auto rule =
              Bitwise(input, levels, id, node.width, true, ir::LowMask(node.width - *amount));
          if (!rule || rule->node.op == Op::zext) continue;
          auto inserted = std::move(rule->inserted);
          inserted.push_back(rule->node);
          const auto operand = id + static_cast<ValueId>(inserted.size() - 1);
          if (Same(rule->node, nodes_[input])) break;
          return Rule{Node{Op::shl, node.width, {operand, node.inputs[1]}}, "demanded_shift", false,
                      std::move(inserted)};
        }
    }

    return std::nullopt;
  }

 private:
  // sel(bit n-1 of v, ones above n, 0) | v, with v unable to set bits at or
  // above n, is v sign-extended from n bits: (v << (w - n)) >>s (w - n).
  std::optional<Rule> SignExtension(ValueId id) const {
    const auto& node = nodes_[id];
    if (node.op != Op::bit_or) return std::nullopt;
    const auto w = node.width;
    for (const auto [select, value] :
         {std::pair{node.inputs[0], node.inputs[1]}, std::pair{node.inputs[1], node.inputs[0]}}) {
      const auto& choice = nodes_[Canonical(select)];
      if (choice.op != Op::select || !Is(choice.inputs[2], 0)) continue;
      const auto high = Literal(choice.inputs[1]);
      if (!high || !*high) continue;
      const auto n = static_cast<unsigned>(std::countr_zero(*high));
      if (!n || n >= w || *high != (ir::LowMask(w) & ~ir::LowMask(n)) ||
          (Mask(value) & ~ir::LowMask(n)) || !Bit(choice.inputs[0], value, n - 1))
        continue;
      std::vector<Node> inserted;
      auto amount = Existing(w - n, w, id);
      if (!amount) {
        inserted.push_back(Node{Op::constant, w, {}, w - n});
        amount = id;
      }

      inserted.push_back(Node{Op::shl, w, {value, *amount}});
      const auto shifted = id + static_cast<ValueId>(inserted.size() - 1);
      return Rule{Node{Op::ashr, w, {shifted, *amount}}, "sign_extension", false,
                  std::move(inserted)};
    }

    return std::nullopt;
  }

  // Whether `condition` is bit `bit` of `value`, or of what `value` masks
  // while keeping that bit.
  bool Bit(ValueId condition, ValueId value, unsigned bit) const {
    const auto& masked = nodes_[Canonical(value)];
    if (masked.op == Op::bit_and && bit < 64)
      for (const auto [x, y] : {std::pair{masked.inputs[0], masked.inputs[1]},
                                std::pair{masked.inputs[1], masked.inputs[0]}})
        if (Literal(y) && ((*Literal(y) >> bit) & 1) && Bit(condition, x, bit)) return true;
    const auto& node = nodes_[Canonical(condition)];
    if (node.op != Op::extract || node.width != 1) return false;
    if (node.immediate == bit) return Canonical(node.inputs[0]) == Canonical(value);
    if (node.immediate) return false;
    const auto& shift = nodes_[Canonical(node.inputs[0])];
    return shift.op == Op::lshr && Is(shift.inputs[1], bit) &&
           Canonical(shift.inputs[0]) == Canonical(value);
  }

  // A chain of additions and subtractions of literals is one addition.
  std::optional<Rule> Reassociate(ValueId id) const {
    const auto w = nodes_[id].width;
    if (nodes_[id].op != Op::add && nodes_[id].op != Op::sub) return std::nullopt;
    std::uint64_t total = 0;
    unsigned folded = 0;
    auto at = id;
    for (unsigned step = 0; step < 8; ++step) {
      const auto& node = nodes_[at];
      if (node.width != w) break;
      if (node.op == Op::add && Literal(node.inputs[1])) {
        total += *Literal(node.inputs[1]);
        at = node.inputs[0];
      } else if (node.op == Op::add && Literal(node.inputs[0])) {
        total += *Literal(node.inputs[0]);
        at = node.inputs[1];
      } else if (node.op == Op::sub && Literal(node.inputs[1])) {
        total -= *Literal(node.inputs[1]);
        at = node.inputs[0];
      } else
        break;
      at = Canonical(at);
      ++folded;
    }

    if (folded < 2 || nodes_[at].width != w) return std::nullopt;
    total &= ir::LowMask(w);
    if (!total) return Copy(at, w, "reassociate");
    if (auto constant = Existing(total, w, id))
      return Rule{Node{Op::add, w, {at, *constant}}, "reassociate"};
    return Rule{
        Node{Op::add, w, {at, id}}, "reassociate", false, {Node{Op::constant, w, {}, total}}};
  }

  // An existing value of `width` equal to the leaf at that width: itself, its
  // zero extension, or its truncation, computed before `before`. A rewrite can
  // only name what exists.
  std::optional<ValueId> Widened(ValueId leaf, unsigned width, ValueId before) const {
    if (nodes_[leaf].width == width) return leaf;

    // A wider leaf seen at `width` bits: an existing truncation of it.
    if (nodes_[leaf].width > width) {
      for (ValueId id = 0; id < before; ++id)
        if (nodes_[id].op == Op::extract && !nodes_[id].immediate && nodes_[id].width == width &&
            Canonical(nodes_[id].inputs[0]) == leaf)
          return id;
      return std::nullopt;
    }

    // Zero extensions, and low extracts no narrower than the leaf, keep its value.
    const auto keeps = [&](ValueId id) {
      for (unsigned step = 0; step < kMaxCaseCone; ++step) {
        const auto& node = nodes_[id];
        if (id == leaf) return true;
        if ((node.op != Op::zext && !(node.op == Op::extract && !node.immediate)) ||
            node.width < nodes_[leaf].width)
          return false;
        id = node.inputs[0];
      }

      return false;
    };

    for (ValueId id = 0; id < before; ++id)
      if (nodes_[id].width == width && id != leaf && keeps(id)) return id;
    return std::nullopt;
  }

  std::optional<ValueId> Existing(std::uint64_t value, unsigned width, ValueId before) const {
    for (ValueId id = 0; id < before; ++id)
      if (nodes_[id].op == Op::constant && nodes_[id].width == width && Is(id, value)) return id;
    return std::nullopt;
  }

  std::optional<std::uint64_t> Lanes(
      ValueId id, unsigned assignment, const std::array<ValueId, kMaxBitwiseLeaves>& leaves,
      std::size_t count, const std::array<ValueId, kMaxBitwiseCone>& cone, std::size_t size,
      std::array<std::optional<std::uint64_t>, kMaxBitwiseCone>& memo) const {
    id = Canonical(id);
    for (unsigned k = 0; k < count; ++k)
      if (leaves[k] == id) return (assignment >> k) & 1 ? ir::LowMask(nodes_[id].width) : 0;
    const auto at =
        static_cast<std::size_t>(std::find(cone.begin(), cone.begin() + size, id) - cone.begin());
    if (at == size) return std::nullopt;
    if (memo[at]) return memo[at];
    const auto& node = nodes_[id];
    if (node.op == Op::constant) return memo[at] = *Literal(id);
    const auto* descriptor = ir::Descriptor(node.op);
    std::array<std::uint64_t, 3> operands{};
    for (unsigned input = 0; input < descriptor->arity; ++input) {
      const auto operand = Lanes(node.inputs[input], assignment, leaves, count, cone, size, memo);
      if (!operand) return std::nullopt;
      operands[input] = *operand;
    }

    memo[at] = ir::FoldPure(node, std::span(operands).first(descriptor->arity),
                            nodes_[node.inputs[0]].width);
    return memo[at];
  }

  // Looks through same-width copies, so two copies of one value are one leaf.
  ValueId Canonical(ValueId id) const {
    for (unsigned step = 0; step < kMaxCaseCone; ++step) {
      const auto& node = nodes_[id];
      if ((node.op != Op::zext && !(node.op == Op::extract && !node.immediate)) ||
          nodes_[node.inputs[0]].width != node.width)
        break;
      id = node.inputs[0];
    }

    return id;
  }

  // A value that depends on a single 0/1 value takes one of two results. When
  // both are known it is a literal, that value, its negation or its widening.
  std::optional<Rule> Cases(ValueId id) const {
    std::array<ValueId, kMaxCaseCone> cone{};
    std::size_t size = 0;
    std::optional<ValueId> leaf;
    std::array<ValueId, kMaxCaseCone> stack{};
    std::size_t depth = 0;
    stack[depth++] = id;
    while (depth) {
      const auto popped = stack[--depth];
      const auto at = popped == id ? id : Canonical(popped);
      if (std::find(cone.begin(), cone.begin() + size, at) != cone.begin() + size || at == leaf)
        continue;
      const auto& node = nodes_[at];
      const auto* descriptor = ir::Descriptor(node.op);
      const bool foldable =
          node.op == Op::constant ||
          (descriptor && descriptor->effect == ir::Effect::pure && descriptor->produces_value &&
           descriptor->arity && node.op != Op::read && node.op != Op::image_address && node.width &&
           node.width <= 64);
      // A 1-bit value below the root is the choice itself; its own cone is
      // irrelevant to which of the two results the root takes.
      if (!foldable || (at != id && node.width == 1 && node.op != Op::constant)) {
        if (leaf || at == id || !Boolean(at, 0)) return std::nullopt;
        leaf = at;
        continue;
      }

      if (size == cone.size()) return std::nullopt;
      cone[size++] = at;
      for (unsigned input = 0; descriptor && input < descriptor->arity && node.op != Op::constant;
           ++input) {
        if (depth == stack.size()) return std::nullopt;
        stack[depth++] = node.inputs[input];
      }
    }

    if (!leaf) return std::nullopt;
    std::array<std::uint64_t, 2> results{};
    for (std::uint64_t value = 0; value < 2; ++value) {
      std::array<std::optional<std::uint64_t>, kMaxCaseCone> memo{};
      const auto folded = Evaluate(id, *leaf, value, cone, size, memo);
      if (!folded) return std::nullopt;
      results[value] = *folded;
    }

    const auto w = nodes_[id].width;
    const auto leaf_width = nodes_[*leaf].width;
    if (results[0] == results[1]) return Constant(w, results[0], "single_boolean");
    if (results[0] == 0 && results[1] == 1) {
      if (leaf_width == w) return Copy(*leaf, w, "single_boolean");
      if (leaf_width < w) return Rule{Node{Op::zext, w, {*leaf}}, "single_boolean"};
      return Rule{Node{Op::extract, w, {*leaf}, 0}, "single_boolean"};
    }

    if (w == 1 && leaf_width == 1 && results[0] == 1 && results[1] == 0)
      return Rule{Node{Op::bit_not, 1, {*leaf}}, "single_boolean"};
    return std::nullopt;
  }

  // Each cone node folds once per case, so shared inputs cost nothing extra.
  std::optional<std::uint64_t> Evaluate(
      ValueId id, ValueId leaf, std::uint64_t value, const std::array<ValueId, kMaxCaseCone>& cone,
      std::size_t size, std::array<std::optional<std::uint64_t>, kMaxCaseCone>& memo) const {
    if (Canonical(id) == leaf) return value;
    if (Canonical(id) != id) return Evaluate(Canonical(id), leaf, value, cone, size, memo);
    const auto& node = nodes_[id];
    if (node.op == Op::constant) return node.immediate & ir::LowMask(node.width);
    const auto at =
        static_cast<std::size_t>(std::find(cone.begin(), cone.begin() + size, id) - cone.begin());
    if (at == size) return std::nullopt;
    if (memo[at]) return memo[at];
    const auto* descriptor = ir::Descriptor(node.op);
    std::array<std::uint64_t, 3> operands{};
    for (unsigned input = 0; input < descriptor->arity; ++input) {
      const auto operand = Evaluate(node.inputs[input], leaf, value, cone, size, memo);
      if (!operand) return std::nullopt;
      operands[input] = *operand;
    }

    memo[at] = ir::FoldPure(node, std::span(operands).first(descriptor->arity),
                            nodes_[node.inputs[node.op == Op::select ? 1 : 0]].width);
    return memo[at];
  }

  std::optional<std::uint64_t> Literal(ValueId id) const {
    const auto& node = nodes_[id];
    if (!node.width || node.width > 64) return std::nullopt;
    if (node.op == Op::constant) return node.immediate & ir::LowMask(node.width);

    // Once the run says where the image sits, a location is a number, and
    // arithmetic that needs its runtime address folds like any other.
    if (bias_ && node.op == Op::image_address && node.width == 64) {
      used_folded_ = true;
      return *bias_ + node.immediate;
    }

    if (id < folded_.size() && folded_[id]) {
      used_folded_ = true;
      return *folded_[id] & ir::LowMask(node.width);
    }

    return std::nullopt;
  }

  bool Is(ValueId id, std::uint64_t value) const {
    const auto literal = Literal(id);
    return literal && *literal == (value & ir::LowMask(nodes_[id].width));
  }

  bool Image(ValueId id) const {
    return nodes_[id].op == Op::image_address && nodes_[id].width == 64;
  }

  static Node Image(std::uint64_t address) { return Node{Op::image_address, 64, {}, address}; }

  // Known to be 0 or 1 at its own width.
  bool Boolean(ValueId id, unsigned depth) const {
    const auto& node = nodes_[id];
    if (node.width == 1) return true;
    if (depth == kMaxBooleanDepth) return false;
    switch (node.op) {
      case Op::constant:
        return *Literal(id) <= 1;
      case Op::zext:
        return Boolean(node.inputs[0], depth + 1);
      case Op::select:
        return Boolean(node.inputs[1], depth + 1) && Boolean(node.inputs[2], depth + 1);
      case Op::bit_and:
        return Boolean(node.inputs[0], depth + 1) || Boolean(node.inputs[1], depth + 1);
      case Op::bit_or:
      case Op::bit_xor:
        return Boolean(node.inputs[0], depth + 1) && Boolean(node.inputs[1], depth + 1);
      default:
        return false;
    }
  }

  std::optional<Rule> Fold(const Node& node, unsigned arity) const {
    std::array<std::uint64_t, 3> values{};
    for (unsigned input = 0; input < arity; ++input) {
      const auto literal = Literal(node.inputs[input]);
      if (!literal) return std::nullopt;
      values[input] = *literal;
    }

    const auto width = nodes_[node.inputs[node.op == Op::select ? 1 : 0]].width;
    const auto folded = ir::FoldPure(node, std::span(values).first(arity), width);
    if (!folded) return std::nullopt;
    return Constant(node.width, *folded, "fold_literals");
  }

  // A value equal to an existing one of the same width becomes its copy.
  std::optional<Rule> Copy(ValueId id, unsigned width, std::string_view name) const {
    if (nodes_[id].width != width) return std::nullopt;
    return Rule{Node{Op::zext, width, {id}}, name};
  }

  static std::optional<Rule> Constant(unsigned width, std::uint64_t value, std::string_view name) {
    return Rule{Node{Op::constant, width, {}, value & ir::LowMask(width)}, name};
  }

  std::span<const Node> nodes_;
  bool placement_;
  std::vector<std::optional<std::uint64_t>>& masks_;
  std::span<const std::optional<std::uint64_t>> folded_;
  std::optional<std::uint64_t> bias_;
  mutable bool used_folded_ = false;
};

// A loop exit over a restored dispatch is a dispatch rewrite like any other:
// it supersedes the jump, and its condition is only evaluated for its value.
bool Dispatched(const ir::ConditionalRewrite& rewrite) {
  return rewrite.rule == ir::RewriteRule::bounded_exit &&
         rewrite.original.kind == ir::TransferKind::jump;
}

// Nodes whose exact shape another record of the block rereads, with every
// input they depend on.
bool Protected(const ir::SsaBlock& block, std::vector<std::uint8_t>& marked, bool path_reads,
               Budget& budget) {
  marked.assign(block.nodes.size(), 0);
  std::vector<ValueId> stack;
  const auto root = [&](ValueId id) {
    if (id < block.nodes.size() && !marked[id]) {
      marked[id] = 1;
      stack.push_back(id);
    }
  };

  // A dispatch rewrite supersedes its boundary's jump: nothing rereads that
  // jump's target, and its condition is only ever evaluated for its value.
  const auto dispatched = [&](std::size_t index) {
    return std::any_of(block.control_rewrites.begin(), block.control_rewrites.end(),
                       [&](const ir::ConditionalRewrite& rewrite) {
                         return (rewrite.rule == ir::RewriteRule::dispatch_branch ||
                                 rewrite.rule == ir::RewriteRule::decided_dispatch ||
                                 Dispatched(rewrite)) &&
                                rewrite.boundary == index;
                       });
  };

  // A recovered transition's transfers and folded conditions are rechecked by
  // folding its path under each value of its condition, so any rewrite that
  // keeps their values keeps them valid.
  const bool folded_path = block.transition.has_value();
  for (std::size_t index = 0; index < block.boundaries.size() && !folded_path; ++index) {
    const auto& transfer = block.boundaries[index].transfer;
    if (!transfer || dispatched(index)) continue;

    // A call's successor rule accepts an unknown callee whatever its target
    // computes, and names a known one only from a literal the edge records.
    if (transfer->kind == ir::TransferKind::call &&
        std::any_of(block.edges.begin(), block.edges.end(), [](const ir::SsaEdge& edge) {
          return edge.kind == ir::SsaEdgeKind::callee &&
                 edge.target_kind == ir::SsaTargetKind::unknown;
        })) {
      if (transfer->continuation) root(*transfer->continuation);
      continue;
    }

    root(transfer->target);
    if (transfer->alternative) root(*transfer->alternative);
    if (transfer->continuation) root(*transfer->continuation);
  }

  for (const auto& rewrite : block.control_rewrites) {
    if (rewrite.rule == ir::RewriteRule::dispatch_branch ||
        rewrite.rule == ir::RewriteRule::decided_dispatch || Dispatched(rewrite) || folded_path)
      continue;
    root(rewrite.condition);
    for (const auto* transfer : {&rewrite.original, &rewrite.replacement}) {
      root(transfer->target);
      if (transfer->condition) root(*transfer->condition);
      if (transfer->alternative) root(*transfer->alternative);
      if (transfer->continuation) root(*transfer->continuation);
    }
  }

  if (path_reads)
    for (const auto& read : block.path_reads) root(read.node);
  for (const auto& fold : block.constant_loads) {
    root(fold.node);

    // A selected fold rechecks its address by pinning its condition to each
    // value and re-deriving the two locations, which needs the address to
    // still read that condition. The condition always sits inside the address
    // cone rooted just above, so this assignment is not what protects it: it
    // marks without queueing, which stops that traversal there and leaves
    // whatever computes the condition free. Written as `root` it would freeze
    // every value behind the condition as well, and in flattened code that is
    // the whole loop the condition tests.
    //
    // What the recheck pins beyond the condition itself is the implied-bit
    // chain, which `SsaImpliedBits` walks down through negations and same-width
    // copies. Those have to stand, or the chain breaks and the recheck cannot
    // fix the value the address reads. Which nodes the walk visits depends on
    // their operations and never on the value, so one call covers both
    // pinnings. Marking rather than rooting truncates here for the same reason
    // it does at the condition: the recheck fixes these nodes, so nothing
    // rereads what feeds them.
    if (fold.condition) {
      if (budget.try_consume({64, 64 * sizeof(std::pair<ValueId, bool>)}) != BudgetDecline::none)
        return false;
      marked[*fold.condition] = 1;
      for (const auto& [id, value] : ir::SsaImpliedBits(block.nodes, *fold.condition, true))
        if (id < block.nodes.size()) marked[id] = 1;
    }

    if (fold.kind == ir::SsaConstantKind::bounded_table) root(fold.table_index);
  }

  // Access records reread an access's address; a stored value is only named.
  const auto access = [&](ValueId id) {
    if (id >= block.nodes.size() || marked[id]) return;
    marked[id] = 1;
    root(block.nodes[id].inputs[0]);
  };

  for (const auto& frame : block.frame_accesses) access(frame.node);
  for (const auto& retired : block.retired_loads) {
    access(retired.node);
    if (retired.basis == ir::SsaRetiredLoadBasis::written_before) access(retired.store);
  }

  for (const auto& omission : block.store_omissions) {
    access(omission.store);
    access(omission.overwriter);
  }

  for (const auto& omission : block.paired_load_omissions) {
    for (const auto id : omission.stores) access(id);
    for (const auto id : omission.loads) access(id);
  }

  // A value a record rereads depends on a load it reaches only through the
  // loaded value, never through the shape of that load's address; records
  // that reread an address root their load themselves.
  while (!stack.empty()) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return false;
    const auto id = stack.back();
    stack.pop_back();
    const auto* descriptor = ir::Descriptor(block.nodes[id].op);
    for (unsigned input = 0; descriptor && input < descriptor->arity; ++input) {
      const auto operand = block.nodes[id].inputs[input];
      if (operand < block.nodes.size() && ir::HasMemoryOrMonitorEffect(block.nodes[operand].op))
        marked[operand] = 1;
      else
        root(operand);
    }
  }

  return true;
}

}  // namespace

SsaSimplifyResult ProposeSsaSimplify(const ir::SsaGraph& original,
                                     std::span<const ir::Group> sources,
                                     bool page_aligned_placement, Budget& budget) {
  const auto valid = sources.empty() ? ir::ValidateSsa(original, budget)
                                     : ir::ValidateSsaWithSources(original, sources, budget);
  if (valid != ir::SsaDecline::none)
    return Decline(valid == ir::SsaDecline::resource_limit ? SsaSimplifyRefusal::resource_limit
                                                           : SsaSimplifyRefusal::invalid_graph);
  if (original.revision() == std::numeric_limits<std::uint64_t>::max())
    return Decline(SsaSimplifyRefusal::invalid_graph);
  auto candidate = original.Clone(budget);
  if (!candidate) return Decline(SsaSimplifyRefusal::resource_limit);
  SsaSimplifyResult result;
  std::vector<std::uint8_t> guarded;
  for (std::size_t slot = 0; slot < candidate->slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return Decline(SsaSimplifyRefusal::resource_limit);
    const auto handle = candidate->Handle(slot);
    if (!handle) continue;
    const auto first = result.journal.size();

    // In-place sweeps settle first; then one splicing rewrite at a time, after
    // which the block is taken again, since node ids have moved.
    for (unsigned round = 0; round <= kMaxInsertions; ++round) {
      auto block = candidate->CopyBlock(*handle, budget);
      if (!block) return Decline(SsaSimplifyRefusal::resource_limit);
      if (block->opaque) break;
      if (!Protected(*block, guarded, true, budget))
        return Decline(SsaSimplifyRefusal::resource_limit);
      if (budget.try_consume({block->nodes.size(), block->nodes.size() * sizeof(std::uint64_t) *
                                                       2}) != BudgetDecline::none)
        return Decline(SsaSimplifyRefusal::resource_limit);
      std::vector<std::optional<std::uint64_t>> masks(block->nodes.size());

      // Loads folded to one literal whatever their condition, read as it.
      std::vector<std::optional<std::uint64_t>> folded(block->nodes.size());
      for (const auto& fold : block->constant_loads) {
        if (fold.condition || !fold.value_stable || fold.node >= folded.size()) continue;
        if (fold.kind == ir::SsaConstantKind::literal)
          folded[fold.node] = fold.value;
        else if (fold.kind == ir::SsaConstantKind::image_location && original.load_bias())
          folded[fold.node] = *original.load_bias() + fold.value;
      }

      const auto skip = [&](ValueId id) {
        return guarded[id] ||
               std::binary_search(block->dead_pure_nodes.begin(), block->dead_pure_nodes.end(),
                                  id) ||
               std::binary_search(block->disabled_effects.begin(), block->disabled_effects.end(),
                                  id);
      };

      const auto before = result.journal.size();

      // A path read's address becomes the image location the record already
      // names, which is what its check re-derives, unless another record
      // rereads the address's shape.
      if (round == 0 && !block->path_reads.empty()) {
        std::vector<std::uint8_t> others;
        if (!Protected(*block, others, false, budget))
          return Decline(SsaSimplifyRefusal::resource_limit);
        for (const auto& read : block->path_reads) {
          const auto id = block->nodes[read.node].inputs[0];
          const ir::Node location{Op::image_address, 64, {}, read.address};
          if (id >= read.node || others[id] || block->nodes[id].width != 64 ||
              Same(block->nodes[id], location))
            continue;
          if (budget.try_consume({1, sizeof(SsaSimplifyEdit)}) != BudgetDecline::none)
            return Decline(SsaSimplifyRefusal::resource_limit);
          result.journal.push_back({*original.Handle(slot), *handle, id, block->nodes[id], location,
                                    "path_read_location", read.page_aligned_placement, false, 0,
                                    original.revision(), 0});
          block->nodes[id] = location;
        }
      }

      // Each sweep can expose another rewrite, as a copy makes its user's
      // operand literal or boolean; a bounded number of sweeps settles it.
      for (unsigned sweep = 0; sweep < kMaxSweeps; ++sweep) {
        bool changed = false;
        for (ValueId id = 0; id < block->nodes.size(); ++id) {
          if (budget.try_consume({272, 0}) != BudgetDecline::none)
            return Decline(SsaSimplifyRefusal::resource_limit);
          if (skip(id)) continue;
          const Rules rules(block->nodes, page_aligned_placement, masks, folded,
                            original.load_bias());
          const auto rule = rules.Apply(id);
          if (!rule || Same(rule->node, block->nodes[id]) || ReadsRetired(*block, rule->node, id))
            continue;
          if (budget.try_consume({1, sizeof(SsaSimplifyEdit)}) != BudgetDecline::none)
            return Decline(SsaSimplifyRefusal::resource_limit);
          result.journal.push_back({*original.Handle(slot), *handle, id, block->nodes[id],
                                    rule->node, rule->name, rule->placement, rules.UsedFolded(), 0,
                                    original.revision(), 0});
          block->nodes[id] = rule->node;
          changed = true;
        }

        if (!changed) break;
      }

      // Every splicing rewrite of this round, applied from the highest node
      // down: a splice moves only ids at or above it, and each rewrite names
      // only values below its own node.
      std::vector<std::tuple<ValueId, Rule, bool>> splices;
      for (ValueId id = 0; id < block->nodes.size() && round < kMaxInsertions; ++id) {
        if (budget.try_consume({256, 0}) != BudgetDecline::none)
          return Decline(SsaSimplifyRefusal::resource_limit);
        if (skip(id)) continue;
        const Rules rules(block->nodes, page_aligned_placement, masks, folded,
                          original.load_bias());
        auto rule = rules.Insert(id);
        if (!rule || rule->inserted.empty() || ReadsRetired(*block, rule->node, id) ||
            std::any_of(rule->inserted.begin(), rule->inserted.end(),
                        [&](const Node& node) { return ReadsRetired(*block, node, id); }))
          continue;
        splices.emplace_back(id, std::move(*rule), rules.UsedFolded());
      }

      if (result.journal.size() != before && !candidate->Replace(*handle, std::move(*block)))
        return Decline(SsaSimplifyRefusal::invalid_graph);
      if (splices.empty()) break;
      for (auto at = splices.rbegin(); at != splices.rend(); ++at) {
        const auto& [id, rule, used_folded] = *at;
        const auto count = static_cast<ValueId>(rule.inserted.size());
        if (!SpliceBefore(*candidate, *handle, id, rule.inserted, ir::BlockLimits{}.max_nodes,
                          budget))
          return Decline(SsaSimplifyRefusal::resource_limit);
        // Earlier edits of this block name ids that just moved.
        for (auto edit = result.journal.begin() + static_cast<std::ptrdiff_t>(first);
             edit != result.journal.end(); ++edit)
          if (edit->node >= id) edit->node += count;
        auto moved = candidate->CopyBlock(*handle, budget);
        if (!moved || budget.try_consume({1, sizeof(SsaSimplifyEdit)}) != BudgetDecline::none)
          return Decline(SsaSimplifyRefusal::resource_limit);
        result.journal.push_back({*original.Handle(slot), *handle, id + count,
                                  moved->nodes[id + count], rule.node, rule.name, rule.placement,
                                  used_folded, count, original.revision(), 0});
        moved->nodes[id + count] = rule.node;
        if (!candidate->Replace(*handle, std::move(*moved)))
          return Decline(SsaSimplifyRefusal::invalid_graph);
      }
    }
  }

  if (result.journal.empty()) return {};

  // A rewrite that makes an address image-based lets the graph place a store
  // that writes a declared value a path read carries; the record is dropped,
  // as the SCCP fold does, so the candidate is not refused over it.
  auto dropped = ir::SsaDropWrittenPathReads(*candidate, budget);
  if (!dropped) return Decline(SsaSimplifyRefusal::resource_limit);
  result.dropped_path_reads = std::move(*dropped);
  const auto checked = sources.empty() ? ir::ValidateSsa(*candidate, budget)
                                       : ir::ValidateSsaWithSources(*candidate, sources, budget);
  if (checked != ir::SsaDecline::none)
    return Decline(checked == ir::SsaDecline::resource_limit ? SsaSimplifyRefusal::resource_limit
                                                             : SsaSimplifyRefusal::invalid_graph);
  for (auto& edit : result.journal) edit.to_revision = candidate->revision();
  result.provisional = std::move(*candidate);
  return result;
}

}  // namespace nyx::recovery
