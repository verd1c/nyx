#include <algorithm>

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::ir {

namespace {
ValueId Core(std::span<const Node> nodes, ValueId id, unsigned& steps) {
  while (id < nodes.size() && ++steps <= kSsaCoreSteps) {
    const auto& node = nodes[id];
    if (node.inputs[0] >= id) break;

    // A widening keeps the number; so does taking the low bits of one that
    // already fits in them.
    if (node.op == Op::zext && nodes[node.inputs[0]].width <= node.width) {
      id = node.inputs[0];
    } else if (node.op == Op::extract && !node.immediate) {
      const auto inner = Core(nodes, node.inputs[0], steps);
      if (nodes[inner].width > node.width) break;
      id = inner;
    } else {
      break;
    }
  }

  return id;
}
}  // namespace

ValueId SsaUnsignedCore(std::span<const Node> nodes, ValueId id) {
  unsigned steps = 0;
  return Core(nodes, id, steps);
}

std::optional<SsaEdgeBound> SsaGuardEdgeBound(std::span<const Node> nodes, ValueId condition,
                                              bool when) {
  const auto at = [&](ValueId id) { return id < nodes.size() ? &nodes[id] : nullptr; };
  const auto literal = [&](ValueId id, unsigned width) -> std::optional<std::uint64_t> {
    const auto* node = at(id);
    if (!node || node->op != Op::constant || node->width != width ||
        (width < 64 && node->immediate >> width))
      return {};
    return node->immediate;
  };

  const auto below = [&](const Node* less) -> std::optional<SsaEdgeBound> {
    if (!less || less->op != Op::unsigned_less || less->width != 1 || !at(less->inputs[0]))
      return {};
    const auto limit = literal(less->inputs[1], at(less->inputs[0])->width);
    if (!limit) return {};
    return SsaEdgeBound{SsaUnsignedCore(nodes, less->inputs[0]), *limit};
  };

  const auto* root = at(condition);
  if (!root || root->width != 1) return {};
  if (when) return below(root);
  if (root->op == Op::bit_not) return below(at(root->inputs[0]));
  if (root->op != Op::bit_and) return {};
  const auto* carry = at(root->inputs[0]);
  const auto* zero = at(root->inputs[1]);
  if (!carry || !zero || carry->op != Op::bit_not || zero->op != Op::bit_not) return {};
  auto bound = below(at(carry->inputs[0]));
  const auto* equal = at(zero->inputs[0]);
  if (!bound || !equal || equal->op != Op::equal || equal->width != 1 || !at(equal->inputs[0]))
    return {};
  // x == N, or as the flags compute it, x - N == 0; the same at x's width.
  const auto width = at(equal->inputs[0])->width;
  const auto* difference = at(equal->inputs[0]);
  ValueId compared = equal->inputs[0];
  std::optional<std::uint64_t> limit = literal(equal->inputs[1], width);
  if (difference->op == Op::sub && limit && *limit == 0) {
    compared = difference->inputs[0];
    limit = literal(difference->inputs[1], width);
  }

  if (!limit || SsaUnsignedCore(nodes, compared) != bound->value ||
      at(compared)->width != at(at(carry->inputs[0])->inputs[0])->width)
    return {};
  // Not below N and not N, so not taken means at most N.
  if (*limit != bound->exclusive_upper || *limit == UINT64_MAX) return {};
  ++bound->exclusive_upper;
  return bound;
}

const SsaExitValue* SsaExitHolding(const SsaBlock& block, SsaHandle handle, StorageId storage,
                                   ValueId id) {
  const auto exit = std::find_if(block.exits.begin(), block.exits.end(),
                                 [&](const SsaExitValue& item) { return item.storage == storage; });
  if (exit == block.exits.end()) return nullptr;
  if (exit->value.kind == SsaValueKind::node && exit->value.block == handle &&
      exit->value.index < block.nodes.size() &&
      SsaUnsignedCore(block.nodes, exit->value.index) == id)
    return &*exit;
  const auto read = std::find_if(block.reads.begin(), block.reads.end(),
                                 [&](const SsaRead& item) { return item.node == id; });
  return read != block.reads.end() && exit->value == read->value ? &*exit : nullptr;
}

SsaDecline ValidateSsaIndexBoundFact(const SsaGraph& graph, const SsaIndexBoundFact& fact,
                                     std::span<const Group> sources, Budget& budget) {
  const auto guard = ValidateSsaGuardEdgeFact(graph, fact.guard, sources, budget);
  if (guard != SsaDecline::none) return guard;
  const auto predicate = ValidateSsaBranchPredicateFact(graph, fact.predicate, sources, budget);
  if (predicate != SsaDecline::none) return predicate;
  if (fact.guard.branch != fact.predicate.branch ||
      fact.guard.edge_index != fact.predicate.edge_index || !fact.exclusive_upper)
    return SsaDecline::invalid_graph;
  const auto* branch = graph.Get(fact.guard.branch);
  const auto* target = graph.Get(fact.guard.guarded_block);
  if (!branch || !target || fact.guard.edge_index >= branch->edges.size())
    return SsaDecline::invalid_graph;
  const auto& edge = branch->edges[fact.guard.edge_index];
  if (edge.target_block != fact.guard.guarded_block || edge.when != fact.predicate.when ||
      fact.predicate.condition >= branch->nodes.size())
    return SsaDecline::invalid_graph;
  if (budget.try_consume({4 * kSsaCoreSteps, 0}) != BudgetDecline::none)
    return SsaDecline::resource_limit;
  const auto bound =
      SsaGuardEdgeBound(branch->nodes, fact.predicate.condition, fact.predicate.when);
  if (!bound || bound->exclusive_upper != fact.exclusive_upper) return SsaDecline::invalid_graph;
  if (budget.try_consume({branch->reads.size() + branch->exits.size() + target->phis.size(), 0}) !=
      BudgetDecline::none)
    return SsaDecline::resource_limit;
  const auto* exit = SsaExitHolding(*branch, fact.guard.branch, fact.storage, bound->value);
  const auto phi = std::find_if(target->phis.begin(), target->phis.end(),
                                [&](const SsaPhi& item) { return item.storage == fact.storage; });
  if (!exit || phi == target->phis.end() || phi->external_entry || phi->incoming.size() != 1 ||
      phi->incoming[0].predecessor != fact.guard.branch || phi->incoming[0].value != exit->value)
    return SsaDecline::invalid_graph;
  return SsaDecline::none;
}

}  // namespace nyx::ir
