#include "nyx/recovery/mba.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <map>
#include <span>

#include "nyx/ir/fold.hpp"
#include "nyx/ir/value_numbering.hpp"
#include "nyx/recovery/ssa/splice.hpp"

namespace nyx::recovery {
namespace {
using ir::Node;
using ir::Op;

std::uint64_t Mask(unsigned width) { return width == 64 ? UINT64_MAX : (UINT64_C(1) << width) - 1; }

bool Literal(const Node& node, std::uint64_t value) {
  if (node.op != Op::constant) return false;
  return (node.width < 64 ? node.immediate & Mask(node.width) : node.immediate) == value;
}

// The graph plus, for each node, the identifiers that denote the same value.
struct Graph {
  std::span<const Node> nodes;
  std::span<const ir::ValueId> same_value;
  std::span<const ir::ValueId> unwrapped;

  const Node& operand(ir::ValueId id) const { return nodes[unwrapped[id]]; }
};

bool SameOperand(ir::ValueId a, ir::ValueId b, const Graph& graph) {
  a = graph.unwrapped[a];
  b = graph.unwrapped[b];
  if (a == b || graph.same_value[a] == graph.same_value[b]) return true;
  const auto& left = graph.nodes[a];
  const auto& right = graph.nodes[b];
  if (left.op != Op::constant || right.op != Op::constant || left.width != right.width)
    return false;
  const auto mask = left.width < 64 ? Mask(left.width) : UINT64_MAX;
  return (left.immediate & mask) == (right.immediate & mask);
}

// Both identity operands must denote the same value in either order. Distinct
// load definitions never match, even at an identical address.
bool SameOperands(const Node& left, const Node& right, const Graph& graph) {
  return (SameOperand(left.inputs[0], right.inputs[0], graph) &&
          SameOperand(left.inputs[1], right.inputs[1], graph)) ||
         (SameOperand(left.inputs[0], right.inputs[1], graph) &&
          SameOperand(left.inputs[1], right.inputs[0], graph));
}

// Operands are rewritten before the consumers that match on their shape, so a
// rule that changes a node's operation can cost an outer identity its match.
// `and_xor_union` and `xor_ones_not` are the only two that do; they are deferred
// to a second sweep, after every rule that leaves the matched shapes intact has
// had its chance. Both sweeps run in SSA order, so a rewrite is still visible to
// the consumers of the node it replaced.
inline constexpr std::array<MbaRule, 6> kShapePreservingRules = {
    MbaRule::or_xor_sum,  MbaRule::or_and_sum,      MbaRule::xor_and_sum,
    MbaRule::negated_add, MbaRule::linear_identity, MbaRule::constant_fold};
inline constexpr std::array<MbaRule, 8> kAllRules = {
    MbaRule::or_xor_sum,      MbaRule::and_xor_union, MbaRule::or_and_sum,
    MbaRule::xor_and_sum,     MbaRule::xor_ones_not,  MbaRule::negated_add,
    MbaRule::linear_identity, MbaRule::constant_fold};

// A sum of constant multiples of bitwise functions of some values is zero at
// every width exactly when it is zero with each value one bit wide: bit j of
// every bitwise term depends only on bit j of its values, so the sum is
// sum_j 2^j S(bits_j), where S is the one-bit sum. A constant c is -c times the
// all-ones term. Anything that is not linear here becomes one more free value,
// which proves a more general identity and so stays sound.
inline constexpr unsigned kMaxLinearAtoms = 6;
inline constexpr unsigned kMaxLinearVisits = 64;
inline constexpr std::uint64_t kLinearWork = kMaxLinearVisits << kMaxLinearAtoms;
inline constexpr unsigned kMaxCanonicalTerms = 12;
inline constexpr std::uint64_t kCanonicalWork =
    kMaxCanonicalTerms * kMaxCanonicalTerms * (1U << kMaxLinearAtoms) * 8;

class Linear {
 public:
  Linear(const Graph& graph, unsigned width, bool strict = false)
      : graph_(graph), width_(width), strict_(strict) {}

  // Records the free values of an arithmetic operand; false if it is not linear.
  bool Collect(ir::ValueId id) { return Visit(id, false); }

  std::size_t atoms() const { return atoms_.size(); }

  std::span<const ir::ValueId> atom_ids() const { return atom_ids_; }

  // The one-bit sum with free value i taking bit i of `assignment`.
  std::uint64_t Sum(ir::ValueId id, unsigned assignment) const {
    const auto& node = graph_.operand(id);
    switch (node.op) {
      case Op::constant:
        return (0 - node.immediate) & Mask(width_);
      case Op::add:
        return (Sum(node.inputs[0], assignment) + Sum(node.inputs[1], assignment)) & Mask(width_);
      case Op::sub:
        return (Sum(node.inputs[0], assignment) - Sum(node.inputs[1], assignment)) & Mask(width_);
      case Op::mul:
      case Op::shl: {
        const auto scaled = Scaled(node);
        if (!scaled) return Bit(id, assignment);
        const auto& [operand, factor] = *scaled;
        return factor * Sum(operand, assignment) & Mask(width_);
      }
      default:
        return Bit(id, assignment);
    }
  }

 private:
  // A product by a literal or a shift by one within the width, as the operand
  // scaled and the factor; anything else is a free value. Collecting and
  // summing must agree on this, or the sum evaluates a different expression.
  std::optional<std::pair<ir::ValueId, std::uint64_t>> Scaled(const Node& node) const {
    const auto literal = [&](unsigned operand) {
      return graph_.operand(node.inputs[operand]).op == Op::constant;
    };

    if (node.op == Op::mul && (literal(0) || literal(1))) {
      const unsigned constant = literal(0) ? 0 : 1;
      return std::pair{node.inputs[1 - constant], graph_.operand(node.inputs[constant]).immediate};
    }

    // A literal is its immediate within its own width, which for a shift
    // amount need not be the width shifted.
    if (node.op == Op::shl && literal(1)) {
      const auto& amount = graph_.operand(node.inputs[1]);
      if (amount.width > 64) return {};
      const auto count = amount.immediate & Mask(amount.width);
      if (count < width_) return std::pair{node.inputs[0], UINT64_C(1) << count};
    }

    return {};
  }

  static bool Bitwise(Op op) {
    return op == Op::bit_and || op == Op::bit_or || op == Op::bit_xor || op == Op::bit_not;
  }

  // A bitwise operation on 0, all-ones or other values is the same function at
  // every bit. One on any other literal, a sign extension's mask for one, is a
  // value of its own.
  bool Uniform(ir::ValueId id) const {
    const auto& node = graph_.operand(id);
    const auto same = [&](ir::ValueId input) {
      const auto& operand = graph_.operand(input);
      const auto value = operand.immediate & Mask(width_);
      return operand.op != Op::constant || value == 0 || value == Mask(width_);
    };

    if (node.op == Op::constant) return same(id);
    return same(node.inputs[0]) && (node.op == Op::bit_not || same(node.inputs[1]));
  }

  bool Bit(ir::ValueId id, unsigned assignment) const {
    const auto& node = graph_.operand(id);
    if (node.op == Op::constant) return node.immediate & 1;
    if (Bitwise(node.op) && Uniform(id)) {
      const auto a = Bit(node.inputs[0], assignment);
      if (node.op == Op::bit_not) return !a;
      const auto b = Bit(node.inputs[1], assignment);
      return node.op == Op::bit_and ? a && b : node.op == Op::bit_or ? a || b : a != b;
    }

    const auto atom = std::find(atoms_.begin(), atoms_.end(), Key(id));
    return assignment >> (atom - atoms_.begin()) & 1;
  }

  ir::ValueId Key(ir::ValueId id) const { return graph_.same_value[graph_.unwrapped[id]]; }

  bool Visit(ir::ValueId id, bool bitwise) {
    if (++visits_ > kMaxLinearVisits) return false;
    const auto& node = graph_.operand(id);
    if (node.width != width_) return false;
    if (node.op == Op::constant) return !bitwise || Uniform(id);
    if (Bitwise(node.op) && Uniform(id)) {
      return Visit(node.inputs[0], true) && (node.op == Op::bit_not || Visit(node.inputs[1], true));
    }

    if (strict_ && Bitwise(node.op)) return false;
    if (strict_ && bitwise &&
        (node.op == Op::add || node.op == Op::sub || node.op == Op::mul || node.op == Op::shl))
      return false;
    if (!bitwise && (node.op == Op::add || node.op == Op::sub)) {
      return Visit(node.inputs[0], false) && Visit(node.inputs[1], false);
    }

    if (const auto scaled = Scaled(node); !bitwise && scaled) return Visit(scaled->first, false);
    if (strict_ && !bitwise && (node.op == Op::mul || node.op == Op::shl)) return false;
    if (std::find(atoms_.begin(), atoms_.end(), Key(id)) != atoms_.end()) return true;
    if (atoms_.size() == kMaxLinearAtoms) return false;
    atoms_.push_back(Key(id));
    atom_ids_.push_back(graph_.unwrapped[id]);
    return true;
  }

  const Graph& graph_;
  unsigned width_;
  unsigned visits_ = 0;
  std::vector<ir::ValueId> atoms_;
  std::vector<ir::ValueId> atom_ids_;
  bool strict_ = false;
};

bool DirectShape(const Node& node, const Graph& graph) {
  if (node.op != Op::bit_or && node.op != Op::bit_and && node.op != Op::bit_xor &&
      node.op != Op::add && node.op != Op::sub)
    return false;
  const auto a = graph.operand(node.inputs[0]).op;
  const auto b = graph.operand(node.inputs[1]).op;
  const auto mixed = [](Op op) {
    return op == Op::bit_and || op == Op::bit_or || op == Op::bit_xor || op == Op::bit_not ||
           op == Op::mul || op == Op::shl;
  };

  return mixed(a) || mixed(b);
}

std::optional<Node> LinearDirect(const Node& node, const Graph& graph, Budget& budget,
                                 bool& exhausted) {
  if (!DirectShape(node, graph) || node.width > 64) return {};
  const auto root = static_cast<ir::ValueId>(&node - graph.nodes.data());
  Linear linear(graph, node.width, true);
  if (!linear.Collect(root) || linear.atoms() == 0) return {};
  const auto atoms = linear.atom_ids();
  const auto assignments = 1U << atoms.size();
  std::array<std::uint64_t, 1U << kMaxLinearAtoms> signature{};
  for (unsigned assignment = 0; assignment < assignments; ++assignment)
    signature[assignment] = linear.Sum(root, assignment);
  const auto matches = [&](auto value) {
    for (unsigned assignment = 0; assignment < assignments; ++assignment) {
      if ((value(assignment) & Mask(node.width)) != signature[assignment]) return false;
    }

    return true;
  };

  const auto different = [&](Node candidate) -> std::optional<Node> {
    if (candidate.op == node.op && candidate.inputs == node.inputs &&
        candidate.immediate == node.immediate)
      return {};
    return candidate;
  };

  if (matches([&](unsigned) { return signature[0]; }))
    return different(Node{Op::constant, node.width, {}, (0 - signature[0]) & Mask(node.width)});
  for (unsigned i = 0; i < atoms.size(); ++i) {
    if (atoms[i] >= root) continue;
    if (matches([&](unsigned assignment) { return 1U - (assignment >> i & 1U); }))
      return different(Node{Op::bit_not, node.width, {atoms[i]}});
  }

  constexpr std::array operations{Op::bit_xor, Op::bit_and, Op::bit_or, Op::add, Op::sub};
  for (const auto op : operations) {
    for (unsigned i = 0; i < atoms.size(); ++i) {
      for (unsigned j = i + 1; j < atoms.size(); ++j) {
        if (atoms[i] >= root || atoms[j] >= root) continue;
        const auto value = [&](unsigned assignment) {
          const auto a = assignment >> i & 1U;
          const auto b = assignment >> j & 1U;
          switch (op) {
            case Op::bit_xor:
              return std::uint64_t{a ^ b};
            case Op::bit_and:
              return std::uint64_t{a & b};
            case Op::bit_or:
              return std::uint64_t{a | b};
            case Op::add:
              return std::uint64_t{a + b};
            case Op::sub:
              return std::uint64_t(a) - std::uint64_t(b);
            default:
              return std::uint64_t{0};
          }
        };

        if (matches(value)) return different(Node{op, node.width, {atoms[i], atoms[j]}});
        if (op == Op::sub && matches([&](unsigned assignment) { return 0 - value(assignment); }))
          return different(Node{op, node.width, {atoms[j], atoms[i]}});
      }
    }
  }

  // Existing bitwise subexpressions can stand for more than one free value.
  // This makes a three-value identity reducible without inserting nodes or
  // changing source boundaries. Only descendants of the root are considered.
  std::array<ir::ValueId, kMaxLinearVisits * 3> pending{};
  std::array<ir::ValueId, kMaxLinearVisits * 2> seen{};
  std::array<ir::ValueId, kMaxCanonicalTerms> terms{};
  unsigned pending_count = 1, seen_count = 0, term_count = 0;
  pending[0] = root;
  for (const auto atom : atoms) terms[term_count++] = atom;
  const auto is_atom = [&](ir::ValueId id) {
    return std::find(atoms.begin(), atoms.end(), id) != atoms.end();
  };
  while (pending_count) {
    const auto id = graph.unwrapped[pending[--pending_count]];
    if (std::find(seen.begin(), seen.begin() + seen_count, id) != seen.begin() + seen_count)
      continue;
    if (seen_count == seen.size()) return {};
    seen[seen_count++] = id;
    const auto& part = graph.nodes[id];
    if (id < root && !is_atom(id) &&
        (part.op == Op::bit_and || part.op == Op::bit_or || part.op == Op::bit_xor ||
         part.op == Op::bit_not) &&
        term_count < terms.size())
      terms[term_count++] = id;
    if (is_atom(id) || part.op == Op::constant) continue;
    const auto* descriptor = ir::Descriptor(part.op);
    if (!descriptor || pending_count + descriptor->arity > pending.size()) return {};
    for (unsigned input = 0; input < descriptor->arity; ++input)
      pending[pending_count++] = part.inputs[input];
  }

  if (term_count == atoms.size()) return {};
  if (budget.try_consume({kCanonicalWork, 0}) != BudgetDecline::none) {
    exhausted = true;
    return {};
  }

  std::sort(terms.begin() + atoms.size(), terms.begin() + term_count);
  const auto height = [&](auto&& self, ir::ValueId id) -> unsigned {
    id = graph.unwrapped[id];
    if (is_atom(id) || graph.nodes[id].op == Op::constant) return 0;
    const auto* descriptor = ir::Descriptor(graph.nodes[id].op);
    unsigned result = 0;
    for (unsigned input = 0; input < descriptor->arity; ++input)
      result = std::max(result, self(self, graph.nodes[id].inputs[input]) + 1);
    return result;
  };

  const auto original_height = height(height, root);
  std::array<unsigned, kMaxCanonicalTerms> heights{};
  std::array<std::array<std::uint64_t, 1U << kMaxLinearAtoms>, kMaxCanonicalTerms> values{};
  for (unsigned i = 0; i < term_count; ++i) {
    heights[i] = height(height, terms[i]);
    for (unsigned assignment = 0; assignment < assignments; ++assignment)
      values[i][assignment] = linear.Sum(terms[i], assignment);
  }

  for (const auto op : operations) {
    for (unsigned i = 0; i < term_count; ++i) {
      for (unsigned j = i + 1; j < term_count; ++j) {
        if (1 + std::max(heights[i], heights[j]) >= original_height) continue;
        const auto value = [&](unsigned assignment, bool reverse) {
          const auto a = values[reverse ? j : i][assignment];
          const auto b = values[reverse ? i : j][assignment];
          switch (op) {
            case Op::bit_xor:
              return a ^ b;
            case Op::bit_and:
              return a & b;
            case Op::bit_or:
              return a | b;
            case Op::add:
              return a + b;
            case Op::sub:
              return a - b;
            default:
              return std::uint64_t{0};
          }
        };

        if (matches([&](unsigned assignment) { return value(assignment, false); }))
          return different(Node{op, node.width, {terms[i], terms[j]}});
        if (op == Op::sub && matches([&](unsigned assignment) { return value(assignment, true); }))
          return different(Node{op, node.width, {terms[j], terms[i]}});
      }
    }
  }

  return {};
}

struct BooleanRecipe {
  Node replacement;
  std::vector<Node> inserted;
};

struct BooleanSearch {
  std::optional<BooleanRecipe> recipe;
  bool exhausted = false;
  bool bounded = false;
  bool outside_scope = false;
};

// Search by operation count, so the first signature found has a minimum-size
// tree in this basis. The 1-bit theorem above lifts a Boolean signature to
// every bit of a linear MBA, including one expressed with additions/carries.
BooleanSearch CanonicalBoolean(ir::ValueId root, const Graph& graph, Budget& budget) {
  BooleanSearch result;
  const auto& node = graph.nodes[root];
  if (!DirectShape(node, graph) || !node.width || node.width > 64) return result;
  Linear linear(graph, node.width, true);
  if (!linear.Collect(root)) {
    result.outside_scope = true;
    return result;
  }

  const auto atoms = linear.atom_ids();
  if (atoms.size() < 3) return result;
  const unsigned assignments = 1U << atoms.size();
  const auto truth_mask = assignments == 64 ? UINT64_MAX : (UINT64_C(1) << assignments) - 1;
  std::uint64_t signature = 0;
  for (unsigned assignment = 0; assignment < assignments; ++assignment) {
    if (budget.try_consume({kLinearWork, 0}) != BudgetDecline::none) {
      result.exhausted = true;
      return result;
    }

    const auto value = linear.Sum(root, assignment);
    if (value > 1) return result;
    signature |= value << assignment;
  }

  std::array<ir::ValueId, kMaxLinearVisits * 3> pending{};
  std::array<ir::ValueId, kMaxLinearVisits * 2> seen{};
  unsigned pending_count = 1, seen_count = 0, source_cost = 0;
  pending[0] = root;
  while (pending_count) {
    if (budget.try_consume({seen.size() + atoms.size(), 0}) != BudgetDecline::none) {
      result.exhausted = true;
      return result;
    }

    const auto id = graph.unwrapped[pending[--pending_count]];
    if (std::find(seen.begin(), seen.begin() + seen_count, id) != seen.begin() + seen_count)
      continue;
    if (seen_count == seen.size()) {
      result.bounded = true;
      return result;
    }

    seen[seen_count++] = id;
    if (std::find(atoms.begin(), atoms.end(), id) != atoms.end() ||
        graph.nodes[id].op == Op::constant)
      continue;
    ++source_cost;
    const auto* descriptor = ir::Descriptor(graph.nodes[id].op);
    if (!descriptor || pending_count + descriptor->arity > pending.size()) {
      result.bounded = true;
      return result;
    }

    for (unsigned input = 0; input < descriptor->arity; ++input)
      pending[pending_count++] = graph.nodes[id].inputs[input];
  }

  if (source_cost <= 2) return result;

  struct Form {
    std::uint64_t bits;
    Op op;
    std::uint16_t left;
    std::uint16_t right;
    std::uint8_t atom;
  };

  constexpr unsigned kMaxForms = 4096;
  constexpr unsigned kMaxGates = 7;
  constexpr unsigned kMaxSearchAttempts = 131072;
  const unsigned maximum = std::min(source_cost - 1, kMaxGates);
  std::vector<Form> forms;
  std::array<std::vector<std::uint16_t>, kMaxGates + 1> by_cost;
  std::map<std::uint64_t, std::uint16_t> known;
  unsigned attempts = 0;
  const auto add = [&](std::uint64_t bits, Op op, unsigned left, unsigned right, unsigned atom,
                       unsigned cost) -> std::optional<std::uint16_t> {
    if (++attempts > kMaxSearchAttempts) {
      result.bounded = true;
      return {};
    }

    if (budget.try_consume({std::bit_width(known.size() + 1U) * 4U + 3U, 0}) !=
        BudgetDecline::none) {
      result.exhausted = true;
      return {};
    }

    bits &= truth_mask;
    if (const auto found = known.find(bits); found != known.end()) return found->second;
    if (forms.size() == kMaxForms) {
      result.bounded = true;
      return {};
    }

    if (budget.try_consume(
            {0, sizeof(Form) + sizeof(std::pair<const std::uint64_t, std::uint16_t>) + 64}) !=
        BudgetDecline::none) {
      result.exhausted = true;
      return {};
    }

    const auto id = static_cast<std::uint16_t>(forms.size());
    forms.push_back({bits, op, static_cast<std::uint16_t>(left), static_cast<std::uint16_t>(right),
                     static_cast<std::uint8_t>(atom)});
    by_cost[cost].push_back(id);
    known.emplace(bits, id);
    return id;
  };

  for (unsigned atom = 0; atom < atoms.size(); ++atom) {
    std::uint64_t bits = 0;
    for (unsigned assignment = 0; assignment < assignments; ++assignment)
      bits |= std::uint64_t((assignment >> atom) & 1U) << assignment;
    if (!add(bits, Op::read, 0, 0, atom, 0)) return result;
  }

  std::optional<std::uint16_t> chosen;
  for (unsigned cost = 1; cost <= maximum && !chosen && !result.bounded && !result.exhausted;
       ++cost) {
    for (const auto child : by_cost[cost - 1]) {
      const auto id = add(~forms[child].bits, Op::bit_not, child, 0, 0, cost);
      if (!id) break;
      if (forms[*id].bits == signature) {
        chosen = *id;
        break;
      }
    }

    if (chosen || result.bounded || result.exhausted) break;
    for (const auto op : {Op::bit_xor, Op::bit_and, Op::bit_or}) {
      for (unsigned left_cost = 0;
           left_cost < cost && !chosen && !result.bounded && !result.exhausted; ++left_cost) {
        const auto right_cost = cost - 1 - left_cost;
        for (const auto left : by_cost[left_cost]) {
          for (const auto right : by_cost[right_cost]) {
            if (forms[left].bits > forms[right].bits) {
              if (++attempts > kMaxSearchAttempts)
                result.bounded = true;
              else if (budget.try_consume({1, 0}) != BudgetDecline::none)
                result.exhausted = true;
              if (result.bounded || result.exhausted) break;
              continue;
            }

            const auto bits = op == Op::bit_xor   ? forms[left].bits ^ forms[right].bits
                              : op == Op::bit_and ? forms[left].bits & forms[right].bits
                                                  : forms[left].bits | forms[right].bits;
            const auto id = add(bits, op, left, right, 0, cost);
            if (!id) break;
            if (forms[*id].bits == signature) {
              chosen = *id;
              break;
            }
          }

          if (chosen || result.bounded || result.exhausted) break;
        }
      }

      if (chosen || result.bounded || result.exhausted) break;
    }
  }

  if (!chosen || forms[*chosen].op == Op::read) return result;
  if (budget.try_consume({forms.size(), forms.size() * sizeof(std::optional<ir::ValueId>) +
                                            maximum * sizeof(Node)}) != BudgetDecline::none) {
    result.exhausted = true;
    return result;
  }

  std::vector<std::optional<ir::ValueId>> materialized(forms.size());
  BooleanRecipe recipe;
  recipe.inserted.reserve(maximum);
  const auto materialize = [&](auto&& self, std::uint16_t id) -> ir::ValueId {
    const auto& form = forms[id];
    if (form.op == Op::read) return atoms[form.atom];
    if (materialized[id]) return *materialized[id];
    const auto a = self(self, form.left);
    const auto b = form.op == Op::bit_not ? 0 : self(self, form.right);
    const auto value = static_cast<ir::ValueId>(root + recipe.inserted.size());
    recipe.inserted.push_back(Node{form.op, node.width, {a, b}});
    materialized[id] = value;
    return value;
  };

  const auto& top = forms[*chosen];
  const auto a = materialize(materialize, top.left);
  const auto b = top.op == Op::bit_not ? 0 : materialize(materialize, top.right);
  recipe.replacement = Node{top.op, node.width, {a, b}};
  if (recipe.inserted.empty()) return result;
  result.recipe = std::move(recipe);
  return result;
}

// An affine one-bit sum has the same integer coefficients at every bit of a
// uniform linear MBA. Rechecking every assignment prevents carry-dependent
// or masked terms from being mistaken for a sum of the free values.
BooleanSearch CanonicalAffine(ir::ValueId root, const Graph& graph, Budget& budget) {
  BooleanSearch result;
  const auto& node = graph.nodes[root];
  if ((node.op != Op::add && node.op != Op::sub) || !node.width || node.width > 64) return result;
  Linear linear(graph, node.width, true);
  if (!linear.Collect(root)) {
    result.outside_scope = true;
    return result;
  }

  const auto atoms = linear.atom_ids();
  if (atoms.size() < 3) return result;
  const auto mask = Mask(node.width);
  const unsigned assignments = 1U << atoms.size();
  std::array<std::uint64_t, 1U << kMaxLinearAtoms> signature{};
  for (unsigned assignment = 0; assignment < assignments; ++assignment) {
    if (budget.try_consume({kLinearWork, 0}) != BudgetDecline::none) {
      result.exhausted = true;
      return result;
    }

    signature[assignment] = linear.Sum(root, assignment);
  }

  if (signature[0] != 0) {
    result.outside_scope = true;
    return result;
  }

  std::array<int, kMaxLinearAtoms> coefficients{};
  unsigned positives = 0, negatives = 0;
  for (unsigned atom = 0; atom < atoms.size(); ++atom) {
    const auto value = signature[1U << atom];
    const auto coefficient = value == 1                              ? 1
                             : node.width > 1 && value == 2          ? 2
                             : value == mask                         ? -1
                             : node.width > 2 && value == (mask - 1) ? -2
                                                                     : 0;
    if (!coefficient) {
      result.bounded = true;
      return result;
    }

    coefficients[atom] = coefficient;
    if (coefficient > 0)
      positives += coefficient;
    else
      negatives -= coefficient;
  }

  const unsigned terms = positives + negatives;
  if (!positives || terms > kMaxLinearAtoms) {
    result.bounded = true;
    return result;
  }

  if (terms < 3) return result;
  for (unsigned assignment = 0; assignment < assignments; ++assignment) {
    std::uint64_t expected = 0;
    for (unsigned atom = 0; atom < atoms.size(); ++atom)
      expected += std::uint64_t(coefficients[atom]) * ((assignment >> atom) & 1U);
    if ((expected & mask) != signature[assignment]) {
      result.outside_scope = true;
      return result;
    }
  }

  std::array<ir::ValueId, kMaxLinearVisits * 3> pending{};
  std::array<ir::ValueId, kMaxLinearVisits * 2> seen{};
  unsigned pending_count = 1, seen_count = 0, source_cost = 0;
  pending[0] = root;
  while (pending_count) {
    if (budget.try_consume({seen.size() + atoms.size(), 0}) != BudgetDecline::none) {
      result.exhausted = true;
      return result;
    }

    const auto id = graph.unwrapped[pending[--pending_count]];
    if (std::find(seen.begin(), seen.begin() + seen_count, id) != seen.begin() + seen_count)
      continue;
    if (seen_count == seen.size()) {
      result.bounded = true;
      return result;
    }

    seen[seen_count++] = id;
    if (std::find(atoms.begin(), atoms.end(), id) != atoms.end() ||
        graph.nodes[id].op == Op::constant)
      continue;
    ++source_cost;
    const auto* descriptor = ir::Descriptor(graph.nodes[id].op);
    if (!descriptor || pending_count + descriptor->arity > pending.size()) {
      result.bounded = true;
      return result;
    }

    for (unsigned input = 0; input < descriptor->arity; ++input)
      pending[pending_count++] = graph.nodes[id].inputs[input];
  }

  if (terms - 1 >= source_cost) return result;
  if (budget.try_consume({terms, (terms - 2) * sizeof(Node)}) != BudgetDecline::none) {
    result.exhausted = true;
    return result;
  }

  BooleanRecipe recipe;
  recipe.inserted.reserve(terms - 2);
  ir::ValueId current = 0;
  unsigned emitted = 0;
  const auto append = [&](ir::ValueId atom, Op op) {
    if (emitted++ == 0) {
      current = atom;
      return;
    }

    const Node next{op, node.width, {current, atom}};
    if (emitted == terms)
      recipe.replacement = next;
    else {
      current = static_cast<ir::ValueId>(root + recipe.inserted.size());
      recipe.inserted.push_back(next);
    }
  };

  for (unsigned atom = 0; atom < atoms.size(); ++atom)
    for (int copy = 0; copy < std::max(coefficients[atom], 0); ++copy) append(atoms[atom], Op::add);
  for (unsigned atom = 0; atom < atoms.size(); ++atom)
    for (int copy = 0; copy < std::max(-coefficients[atom], 0); ++copy)
      append(atoms[atom], Op::sub);
  result.recipe = std::move(recipe);
  return result;
}

std::optional<Node> Rewrite(MbaRule rule, const Node& node, const Graph& graph) {
  const auto nodes = graph.nodes;
  if (rule == MbaRule::or_xor_sum) {
    if (node.op != Op::sub) return {};
    const auto& shift = graph.operand(node.inputs[0]);
    const auto& exclusive = graph.operand(node.inputs[1]);
    if (shift.op != Op::shl || exclusive.op != Op::bit_xor || shift.width != node.width ||
        exclusive.width != node.width || !Literal(nodes[shift.inputs[1]], 1))
      return {};
    const auto& inclusive = graph.operand(shift.inputs[0]);
    if (inclusive.op != Op::bit_or || inclusive.width != node.width) return {};
    if (!SameOperands(exclusive, inclusive, graph)) return {};
    return Node{Op::add, node.width, {inclusive.inputs[0], inclusive.inputs[1]}};
  }

  // (a & b) | (a ^ b) == a | b. The intersection contributes no bit the union
  // does not already hold.
  if (rule == MbaRule::and_xor_union) {
    if (node.op != Op::bit_or) return {};
    for (unsigned i = 0; i < 2; ++i) {
      const auto& conjunction = graph.operand(node.inputs[i]);
      const auto& exclusive = graph.operand(node.inputs[1 - i]);
      if (conjunction.op != Op::bit_and || exclusive.op != Op::bit_xor ||
          conjunction.width != node.width || exclusive.width != node.width ||
          !SameOperands(conjunction, exclusive, graph))
        continue;
      return Node{Op::bit_or, node.width, {conjunction.inputs[0], conjunction.inputs[1]}};
    }

    return {};
  }

  // (a | b) + (a & b) == a + b. Shared bits are counted once by the union and
  // once by the intersection, which is what the carrying sum does.
  if (rule == MbaRule::or_and_sum) {
    if (node.op != Op::add) return {};
    for (unsigned i = 0; i < 2; ++i) {
      const auto& inclusive = graph.operand(node.inputs[i]);
      const auto& conjunction = graph.operand(node.inputs[1 - i]);
      if (inclusive.op != Op::bit_or || conjunction.op != Op::bit_and ||
          inclusive.width != node.width || conjunction.width != node.width ||
          !SameOperands(inclusive, conjunction, graph))
        continue;
      return Node{Op::add, node.width, {inclusive.inputs[0], inclusive.inputs[1]}};
    }

    return {};
  }

  // (a ^ b) + ((a & b) << 1) == a + b: the carry-free sum plus the carries.
  // Truncating the shifted carries to the width is the same reduction the sum
  // itself performs, so the identity survives modular arithmetic.
  if (rule == MbaRule::xor_and_sum) {
    if (node.op != Op::add) return {};
    for (unsigned i = 0; i < 2; ++i) {
      const auto& exclusive = graph.operand(node.inputs[i]);
      const auto& carries = graph.operand(node.inputs[1 - i]);
      if (exclusive.op != Op::bit_xor || carries.op != Op::shl || exclusive.width != node.width ||
          carries.width != node.width || !Literal(nodes[carries.inputs[1]], 1))
        continue;
      const auto& conjunction = graph.operand(carries.inputs[0]);
      if (conjunction.op != Op::bit_and || conjunction.width != node.width ||
          !SameOperands(exclusive, conjunction, graph))
        continue;
      return Node{Op::add, node.width, {conjunction.inputs[0], conjunction.inputs[1]}};
    }

    return {};
  }

  // a ^ ~0 == ~a, where the obfuscator materializes the all-ones literal in a
  // register. Above 64 bits a constant cannot represent all-ones, because its
  // immediate zero-extends to the node width.
  if (rule == MbaRule::xor_ones_not) {
    if (node.op != Op::bit_xor || node.width > 64) return {};
    for (unsigned i = 0; i < 2; ++i) {
      if (nodes[node.inputs[i]].width != node.width ||
          !Literal(nodes[node.inputs[i]], Mask(node.width)))
        continue;
      return Node{Op::bit_not, node.width, {node.inputs[1 - i]}};
    }

    return {};
  }

  if (rule == MbaRule::negated_add) {
    if (node.op != Op::add) return {};
    for (unsigned i = 0; i < 2; ++i) {
      const auto& negative = nodes[node.inputs[i]];
      if (negative.op == Op::sub && negative.width == node.width &&
          Literal(nodes[negative.inputs[0]], 0)) {
        return Node{Op::sub, node.width, {node.inputs[1 - i], negative.inputs[1]}};
      }
    }

    return {};
  }

  // An equality decided by its operands' difference: identically zero makes it
  // true, a nonzero constant makes it false, and anything else is left alone.
  if (rule == MbaRule::linear_identity) {
    if (node.op != Op::equal) return {};
    const auto width = nodes[node.inputs[0]].width;
    if (width > 64) return {};
    Linear linear(graph, width);
    if (!linear.Collect(node.inputs[0]) || !linear.Collect(node.inputs[1]) || linear.atoms() == 0)
      return {};
    std::optional<std::uint64_t> difference;
    for (unsigned assignment = 0; assignment < 1U << linear.atoms(); ++assignment) {
      const auto value =
          (linear.Sum(node.inputs[0], assignment) - linear.Sum(node.inputs[1], assignment)) &
          Mask(width);
      if (difference && *difference != value) return {};
      difference = value;
    }

    return Node{Op::constant, 1, {}, *difference == 0 ? 1U : 0U};
  }

  if (rule != MbaRule::constant_fold || node.op == Op::constant || node.width > 64) return {};
  const auto* descriptor = ir::Descriptor(node.op);
  if (descriptor == nullptr || descriptor->effect != ir::Effect::pure || descriptor->arity == 0)
    return {};
  std::array<std::uint64_t, 3> values{};
  for (unsigned i = 0; i < descriptor->arity; ++i) {
    const auto& operand = nodes[node.inputs[i]];
    if (operand.op != Op::constant || operand.width > 64) return {};
    values[i] = operand.immediate & Mask(operand.width);
  }

  const auto result =
      ir::FoldPure(node, std::span(values).first(descriptor->arity), nodes[node.inputs[0]].width);
  if (!result) return {};
  return Node{Op::constant, node.width, {}, *result};
}

ir::BlockDecline ValidateRegion(const ir::Block& region, Budget& budget, ir::BlockLimits limits) {
  return ir::Validate(region, budget, limits);
}

ir::BlockDecline ValidateRegion(const ir::Path& region, Budget& budget, ir::BlockLimits limits) {
  return ir::ValidatePath(region, budget, limits);
}

template <class Region, class Result>
Result Simplify(const Region& input, Budget& budget, MbaLimits limits) {
  const auto Decline = [](MbaDecline reason) -> Result { return {{}, {}, reason}; };
  const auto valid = ValidateRegion(input, budget, limits.block);
  if (valid != ir::BlockDecline::none) {
    return Decline(valid == ir::BlockDecline::resource_limit ? MbaDecline::resource_limit
                                                             : MbaDecline::invalid_ir);
  }

  if (input.revision() == UINT64_MAX) return Decline(MbaDecline::revision_overflow);
  const auto attempts = kShapePreservingRules.size() + kAllRules.size();
  const auto capacity = std::min<std::uint64_t>(limits.max_edits, attempts * input.nodes().size());
  const auto charge = [&](std::uint64_t count, std::uint64_t size) {
    return count <= UINT64_MAX / size &&
           budget.try_consume({count, count * size}) == BudgetDecline::none;
  };

  if (!charge(input.sources().size(), sizeof(ir::Group)) ||
      !charge(input.nodes().size(), sizeof(Node)) ||
      !charge(input.origins().size(), sizeof(ir::Origin)) ||
      !charge(input.boundaries().size(), sizeof(ir::Boundary)) ||
      !charge(capacity, sizeof(MbaEdit))) {
    return Decline(MbaDecline::resource_limit);
  }

  for (const auto& source : input.sources()) {
    if (!charge(source.bytes().size(), 1) || !charge(source.nodes().size(), sizeof(Node)) ||
        !charge(source.writes().size(), sizeof(ir::Write)))
      return Decline(MbaDecline::resource_limit);
  }

  for (const auto& boundary : input.boundaries()) {
    if (!charge(boundary.writes.size(), sizeof(ir::Write)))
      return Decline(MbaDecline::resource_limit);
  }

  std::vector<ir::Group> sources(input.sources().begin(), input.sources().end());
  std::vector<Node> nodes(input.nodes().begin(), input.nodes().end());
  std::vector<ir::Origin> origins(input.origins().begin(), input.origins().end());
  std::vector<ir::Boundary> boundaries(input.boundaries().begin(), input.boundaries().end());
  std::vector<MbaEdit> journal;
  journal.reserve(static_cast<std::size_t>(capacity));
  std::vector<ir::ValueId> same_value;
  std::vector<ir::ValueId> unwrapped;
  if (!ir::NumberValues(nodes, budget, same_value, unwrapped))
    return Decline(MbaDecline::resource_limit);
  const Graph graph{nodes, same_value, unwrapped};
  const auto sweep = [&](std::span<const MbaRule> rules) {
    for (std::size_t id = 0; id < nodes.size(); ++id) {
      for (const auto rule : rules) {
        const bool identity = rule == MbaRule::linear_identity && nodes[id].op == Op::equal;
        if (budget.try_consume({identity ? kLinearWork : 32, 0}) != BudgetDecline::none) {
          return MbaDecline::resource_limit;
        }

        const auto replacement = Rewrite(rule, nodes[id], graph);
        if (!replacement) continue;
        if (journal.size() == capacity) return MbaDecline::resource_limit;
        journal.push_back({rule, static_cast<ir::ValueId>(id), nodes[id], *replacement,
                           nodes[id].width, input.revision(), input.revision() + 1});
        nodes[id] = *replacement;
      }
    }

    return MbaDecline::none;
  };

  for (const auto rules :
       {std::span<const MbaRule>(kShapePreservingRules), std::span<const MbaRule>(kAllRules)}) {
    if (const auto declined = sweep(rules); declined != MbaDecline::none) return Decline(declined);
  }

  Region output(std::move(sources), std::move(nodes), std::move(origins), std::move(boundaries),
                input.revision() + (journal.empty() ? 0 : 1));
  const auto checked = ValidateRegion(output, budget, limits.block);
  if (checked != ir::BlockDecline::none) {
    return Decline(checked == ir::BlockDecline::resource_limit ? MbaDecline::resource_limit
                                                               : MbaDecline::invalid_ir);
  }

  return {std::move(output), std::move(journal), MbaDecline::none};
}

template <class Region, class Result>
Result SimplifyDirect(const Region& input, Budget& budget, MbaLimits limits) {
  const auto decline = [](MbaDecline reason) -> Result { return {{}, {}, reason}; };
  const auto valid = ValidateRegion(input, budget, limits.block);
  if (valid != ir::BlockDecline::none)
    return decline(valid == ir::BlockDecline::resource_limit ? MbaDecline::resource_limit
                                                             : MbaDecline::invalid_ir);
  if (input.revision() == UINT64_MAX) return decline(MbaDecline::revision_overflow);
  const auto charge = [&](std::uint64_t count, std::uint64_t size) {
    return count <= UINT64_MAX / size &&
           budget.try_consume({count, count * size}) == BudgetDecline::none;
  };

  if (!charge(input.sources().size(), sizeof(ir::Group)) ||
      !charge(input.nodes().size(), sizeof(Node)) ||
      !charge(input.origins().size(), sizeof(ir::Origin)) ||
      !charge(input.boundaries().size(), sizeof(ir::Boundary)) ||
      !charge(std::min<std::size_t>(input.nodes().size(), limits.max_edits), sizeof(MbaEdit)))
    return decline(MbaDecline::resource_limit);
  for (const auto& source : input.sources()) {
    if (!charge(source.bytes().size(), 1) || !charge(source.nodes().size(), sizeof(Node)) ||
        !charge(source.writes().size(), sizeof(ir::Write)))
      return decline(MbaDecline::resource_limit);
  }

  for (const auto& boundary : input.boundaries()) {
    if (!charge(boundary.writes.size(), sizeof(ir::Write)))
      return decline(MbaDecline::resource_limit);
  }

  std::vector<ir::Group> sources(input.sources().begin(), input.sources().end());
  std::vector<Node> nodes(input.nodes().begin(), input.nodes().end());
  std::vector<ir::Origin> origins(input.origins().begin(), input.origins().end());
  std::vector<ir::Boundary> boundaries(input.boundaries().begin(), input.boundaries().end());
  std::vector<MbaEdit> journal;
  std::vector<ir::ValueId> same_value, unwrapped;
  if (!ir::NumberValues(nodes, budget, same_value, unwrapped))
    return decline(MbaDecline::resource_limit);
  const Graph graph{nodes, same_value, unwrapped};
  for (std::size_t id = 0; id < nodes.size(); ++id) {
    if (!DirectShape(nodes[id], graph)) continue;
    if (budget.try_consume({2 * kLinearWork, 0}) != BudgetDecline::none)
      return decline(MbaDecline::resource_limit);
    bool exhausted = false;
    const auto replacement = LinearDirect(nodes[id], graph, budget, exhausted);
    if (exhausted) return decline(MbaDecline::resource_limit);
    if (!replacement) continue;
    if (journal.size() == limits.max_edits) return decline(MbaDecline::resource_limit);
    journal.push_back({MbaRule::linear_direct, static_cast<ir::ValueId>(id), nodes[id],
                       *replacement, nodes[id].width, input.revision(), input.revision() + 1});
    nodes[id] = *replacement;
  }

  Region output(std::move(sources), std::move(nodes), std::move(origins), std::move(boundaries),
                input.revision() + (journal.empty() ? 0 : 1));
  const auto checked = ValidateRegion(output, budget, limits.block);
  if (checked != ir::BlockDecline::none)
    return decline(checked == ir::BlockDecline::resource_limit ? MbaDecline::resource_limit
                                                               : MbaDecline::invalid_ir);
  return {std::move(output), std::move(journal), MbaDecline::none};
}

}  // namespace

MbaResult SimplifyMba(const ir::Block& input, Budget& budget, MbaLimits limits) {
  return Simplify<ir::Block, MbaResult>(input, budget, limits);
}

MbaPathResult SimplifyMba(const ir::Path& input, Budget& budget, MbaLimits limits) {
  return Simplify<ir::Path, MbaPathResult>(input, budget, limits);
}

MbaResult SimplifyLinearMba(const ir::Block& input, Budget& budget, MbaLimits limits) {
  return SimplifyDirect<ir::Block, MbaResult>(input, budget, limits);
}

MbaPathResult SimplifyLinearMba(const ir::Path& input, Budget& budget, MbaLimits limits) {
  return SimplifyDirect<ir::Path, MbaPathResult>(input, budget, limits);
}

SsaMbaResult ProposeLinearMba(const ir::SsaGraph& original, Budget& budget, MbaLimits limits) {
  SsaMbaResult result;
  const auto decline = [](MbaDecline reason) {
    SsaMbaResult refused;
    refused.reason = reason;
    return refused;
  };

  const auto valid = ir::ValidateSsa(original, budget);
  if (valid != ir::SsaDecline::none) {
    return decline(valid == ir::SsaDecline::resource_limit ? MbaDecline::resource_limit
                                                           : MbaDecline::invalid_ir);
  }

  auto candidate = original.Clone(budget);
  if (!candidate) return decline(MbaDecline::resource_limit);
  for (std::size_t slot = 0; slot < candidate->slots(); ++slot) {
    const auto result_handle = candidate->Handle(slot);
    if (!result_handle) continue;
    const ir::SsaHandle original_handle{original.arena(), result_handle->slot,
                                        result_handle->generation};
    auto copied = candidate->CopyBlock(*result_handle, budget);
    if (!copied) return decline(MbaDecline::resource_limit);
    auto block = std::move(*copied);
    const auto edits_before = result.journal.size();
    std::vector<ir::ValueId> same_value, unwrapped;
    if (!ir::NumberValues(block.nodes, budget, same_value, unwrapped))
      return decline(MbaDecline::resource_limit);
    const Graph graph{block.nodes, same_value, unwrapped};
    for (std::size_t id = 0; id < block.nodes.size(); ++id) {
      if (!DirectShape(block.nodes[id], graph)) continue;
      if (budget.try_consume({2 * kLinearWork, 0}) != BudgetDecline::none)
        return decline(MbaDecline::resource_limit);
      bool exhausted = false;
      const auto replacement = LinearDirect(block.nodes[id], graph, budget, exhausted);
      if (exhausted) return decline(MbaDecline::resource_limit);
      if (!replacement) continue;
      if (result.journal.size() == limits.max_edits ||
          budget.try_consume({1, sizeof(SsaMbaEdit)}) != BudgetDecline::none)
        return decline(MbaDecline::resource_limit);
      result.journal.push_back(
          {original_handle,
           *result_handle,
           {MbaRule::linear_direct, static_cast<ir::ValueId>(id), block.nodes[id], *replacement,
            block.nodes[id].width, original.revision(), 0}});
      block.nodes[id] = *replacement;
    }

    if (result.journal.size() != edits_before &&
        !candidate->Replace(*result_handle, std::move(block)))
      return decline(MbaDecline::invalid_ir);
  }

  if (result.journal.empty()) return result;
  const auto checked = ir::ValidateSsa(*candidate, budget);
  if (checked != ir::SsaDecline::none) {
    return decline(checked == ir::SsaDecline::resource_limit ? MbaDecline::resource_limit
                                                             : MbaDecline::invalid_ir);
  }

  for (auto& edit : result.journal) edit.edit.to_revision = candidate->revision();
  result.provisional = std::move(*candidate);
  return result;
}

SsaMbaSynthesisResult ProposeCanonicalLinearMba(const ir::SsaGraph& original, Budget& budget,
                                                MbaLimits limits) {
  const auto decline = [](MbaDecline reason) {
    SsaMbaSynthesisResult refused;
    refused.reason = reason;
    return refused;
  };

  const auto valid = ir::ValidateSsa(original, budget);
  if (valid != ir::SsaDecline::none)
    return decline(valid == ir::SsaDecline::resource_limit ? MbaDecline::resource_limit
                                                           : MbaDecline::invalid_ir);
  auto candidate = original.Clone(budget);
  if (!candidate) return decline(MbaDecline::resource_limit);
  SsaMbaSynthesisResult result;
  for (std::size_t slot = 0; slot < candidate->slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return decline(MbaDecline::resource_limit);
    const auto handle = candidate->Handle(slot);
    if (!handle) continue;
    auto copied = candidate->CopyBlock(*handle, budget);
    if (!copied) return decline(MbaDecline::resource_limit);
    const auto& block = *copied;
    std::vector<ir::ValueId> same_value, unwrapped;
    if (!ir::NumberValues(block.nodes, budget, same_value, unwrapped))
      return decline(MbaDecline::resource_limit);
    const Graph graph{block.nodes, same_value, unwrapped};
    for (std::size_t id = block.nodes.size(); id-- > 0;) {
      if ((!DirectShape(block.nodes[id], graph) && block.nodes[id].op != Op::add &&
           block.nodes[id].op != Op::sub) ||
          std::binary_search(block.dead_pure_nodes.begin(), block.dead_pure_nodes.end(), id))
        continue;
      if (budget.try_consume({2 * kLinearWork, 0}) != BudgetDecline::none)
        return decline(MbaDecline::resource_limit);
      auto search = CanonicalBoolean(static_cast<ir::ValueId>(id), graph, budget);
      if (!search.recipe && !search.exhausted &&
          (block.nodes[id].op == Op::add || block.nodes[id].op == Op::sub)) {
        auto affine = CanonicalAffine(static_cast<ir::ValueId>(id), graph, budget);
        if (affine.recipe || affine.exhausted || affine.bounded || affine.outside_scope)
          search = std::move(affine);
      }

      if (search.exhausted) return decline(MbaDecline::resource_limit);
      if (search.outside_scope || search.bounded) {
        // A long list of what the pass could not rewrite is a report, not
        // work the pass needs: past its bound it is counted, not recorded.
        if (result.refused.size() == 4096) {
          ++result.refused_unlisted;
          continue;
        }

        if (budget.try_consume({1, sizeof(SsaMbaSynthesisRefusal)}) != BudgetDecline::none)
          return decline(MbaDecline::resource_limit);
        result.refused.push_back(
            {{original.arena(), static_cast<std::uint32_t>(slot), handle->generation},
             static_cast<ir::ValueId>(id),
             search.outside_scope ? SsaMbaSynthesisRefusalReason::outside_uniform_linear_scope
                                  : SsaMbaSynthesisRefusalReason::search_bound});
        continue;
      }

      if (!search.recipe) continue;
      if (result.journal.size() == limits.max_edits ||
          budget.try_consume(
              {1, sizeof(SsaMbaSynthesisEdit) + search.recipe->inserted.size() * sizeof(Node)}) !=
              BudgetDecline::none)
        return decline(MbaDecline::resource_limit);
      const auto old = block.nodes[id];
      const auto added = search.recipe->inserted.size();
      if (!SpliceBefore(*candidate, *handle, static_cast<ir::ValueId>(id), search.recipe->inserted,
                        limits.block.max_nodes, budget))
        return decline(MbaDecline::resource_limit);
      auto replacement_block = candidate->CopyBlock(*handle, budget);
      if (!replacement_block) return decline(MbaDecline::resource_limit);
      const auto result_node = static_cast<ir::ValueId>(id + added);
      replacement_block->nodes[result_node] = search.recipe->replacement;
      if (!candidate->Replace(*handle, std::move(*replacement_block)))
        return decline(MbaDecline::invalid_ir);
      result.journal.push_back(
          {{original.arena(), static_cast<std::uint32_t>(slot), handle->generation},
           *handle,
           static_cast<ir::ValueId>(id),
           result_node,
           old,
           search.recipe->replacement,
           std::move(search.recipe->inserted),
           original.revision(),
           0});
      break;
    }
  }

  if (result.journal.empty()) return result;
  const auto checked = ir::ValidateSsa(*candidate, budget);
  if (checked != ir::SsaDecline::none)
    return decline(checked == ir::SsaDecline::resource_limit ? MbaDecline::resource_limit
                                                             : MbaDecline::invalid_ir);
  for (auto& edit : result.journal) edit.to_revision = candidate->revision();
  result.provisional = std::move(*candidate);
  return result;
}

}  // namespace nyx::recovery
