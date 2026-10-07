#include "nyx/analysis/entry_relations.hpp"

#include <algorithm>

namespace nyx::analysis {
namespace {

// A storage's value as an unknown symbol plus a literal. In a canonical state a
// symbol is the index of the lowest storage holding it, so two states with the
// same relations compare equal whatever symbols produced them.
struct Term {
  std::uint32_t symbol;
  std::uint64_t offset;
  bool operator==(const Term&) const = default;
};

using State = std::vector<Term>;

State Canonical(const State& state) {
  State result(state.size());
  for (std::uint32_t i = 0; i < state.size(); ++i) {
    std::uint32_t first = 0;
    while (state[first].symbol != state[i].symbol) ++first;
    result[i] = {first, state[i].offset - state[first].offset};
  }

  return result;
}

// What both states agree on: two storages stay related only if each state
// relates them by the same displacement.
State Join(const State& a, const State& b) {
  State result(a.size());
  for (std::uint32_t i = 0; i < a.size(); ++i) {
    std::uint32_t first = 0;
    while (a[first].symbol != a[i].symbol || b[first].symbol != b[i].symbol ||
           a[first].offset - b[first].offset != a[i].offset - b[i].offset)
      ++first;
    result[i] = {first, a[i].offset - a[first].offset};
  }

  return result;
}

State Transfer(const CfgBlock& block, const State& in, std::span<const ir::StorageId> tracked) {
  auto fresh = static_cast<std::uint32_t>(tracked.size());
  State out = in;
  if (!block.ssa) {
    for (auto& term : out) term = {fresh++, 0};
    return Canonical(out);
  }

  const auto index = [&](ir::StorageId storage) -> std::optional<std::size_t> {
    const auto found = std::lower_bound(tracked.begin(), tracked.end(), storage);
    if (found == tracked.end() || *found != storage) return {};
    return found - tracked.begin();
  };

  const auto nodes = block.ssa->nodes();
  std::vector<std::optional<Term>> terms(nodes.size());
  const auto assign = [&](ir::StorageId storage, ir::ValueId value) {
    if (const auto slot = index(storage)) out[*slot] = terms[value].value_or(Term{fresh++, 0});
  };

  const auto literal = [&](ir::ValueId id) -> std::optional<std::uint64_t> {
    if (nodes[id].op != ir::Op::constant) return {};
    return nodes[id].immediate;
  };

  for (const auto& boundary : block.ssa->boundaries()) {
    const auto end = boundary.first_node + boundary.node_count;
    for (ir::ValueId id = boundary.first_node; id < end; ++id) {
      const auto& node = nodes[id];
      if (node.op == ir::Op::write) {
        assign(node.storage, node.inputs[0]);
        continue;
      }

      if (node.width != 64) continue;
      if (node.op == ir::Op::read) {
        if (const auto slot = index(node.storage)) terms[id] = in[*slot];
      } else if (node.op == ir::Op::add || node.op == ir::Op::sub) {
        const auto& a = terms[node.inputs[0]];
        if (const auto b = literal(node.inputs[1]); a && b) {
          terms[id] = Term{a->symbol, node.op == ir::Op::add ? a->offset + *b : a->offset - *b};
        } else if (const auto c = literal(node.inputs[0]);
                   c && node.op == ir::Op::add && terms[node.inputs[1]]) {
          terms[id] = Term{terms[node.inputs[1]]->symbol, terms[node.inputs[1]]->offset + *c};
        }
      }
    }

    for (const auto& write : boundary.writes) assign(write.storage, write.value);
  }

  return Canonical(out);
}

}  // namespace

EntryRelationsResult ProveEntryRelations(const Cfg& graph, Budget& budget,
                                         RelationAssumptions assumptions) {
  const auto decline = [](RelationDecline reason) { return EntryRelationsResult{{}, reason}; };
  const auto blocks = graph.blocks();
  const auto sources = graph.sources();
  const auto entries = graph.entries();
  const auto preserved = assumptions.preserved;
  if (!std::is_sorted(entries.begin(), entries.end()) ||
      !std::is_sorted(preserved.begin(), preserved.end())) {
    return decline(RelationDecline::invalid_input);
  }

  // Storage the graph uses at exactly 64 bits; anything narrower or mixed is left
  // out rather than related through a width it does not have.
  std::vector<std::pair<ir::StorageId, unsigned>> uses;
  for (const auto& block : blocks) {
    if (!block.ssa) continue;
    const auto nodes = block.ssa->nodes();
    if (budget.try_consume({nodes.size(), nodes.size() * sizeof(uses[0])}) != BudgetDecline::none) {
      return decline(RelationDecline::resource_limit);
    }

    for (const auto& node : nodes) {
      if (node.op == ir::Op::read)
        uses.emplace_back(node.storage, node.width);
      else if (node.op == ir::Op::write)
        uses.emplace_back(node.storage, nodes[node.inputs[0]].width);
    }

    for (const auto& boundary : block.ssa->boundaries()) {
      if (budget.try_consume({boundary.writes.size(), boundary.writes.size() * sizeof(uses[0])}) !=
          BudgetDecline::none)
        return decline(RelationDecline::resource_limit);
      for (const auto& write : boundary.writes)
        uses.emplace_back(write.storage, nodes[write.value].width);
    }
  }

  std::sort(uses.begin(), uses.end());
  uses.erase(std::unique(uses.begin(), uses.end()), uses.end());
  std::vector<ir::StorageId> tracked;
  for (std::size_t i = 0; i < uses.size(); ++i) {
    const bool mixed = (i > 0 && uses[i - 1].first == uses[i].first) ||
                       (i + 1 < uses.size() && uses[i + 1].first == uses[i].first);
    if (uses[i].second == 64 && !mixed) tracked.push_back(uses[i].first);
  }

  const auto width = tracked.size();
  if (budget.try_consume(
          {blocks.size(), blocks.size() * (sizeof(std::optional<State>) + width * sizeof(Term) +
                                           sizeof(std::vector<ir::StorageRelation>) + 1)}) !=
      BudgetDecline::none) {
    return decline(RelationDecline::resource_limit);
  }

  std::vector<std::optional<State>> states(blocks.size());
  std::vector<std::uint8_t> queued(blocks.size());
  std::vector<std::uint32_t> work;
  const auto merge = [&](std::uint32_t target, const State& state) {
    auto& current = states[target];
    auto joined = current ? Join(*current, state) : state;
    const bool changed = !current || joined != *current;
    current = std::move(joined);
    if (changed && !queued[target]) {
      queued[target] = 1;
      work.push_back(target);
    }
  };

  State distinct(width);
  for (std::uint32_t i = 0; i < width; ++i) distinct[i] = {i, 0};
  for (std::uint32_t id = 0; id < blocks.size(); ++id) {
    if (std::binary_search(entries.begin(), entries.end(),
                           sources[blocks[id].first_source].address)) {
      merge(id, distinct);
    }
  }

  EntryRelations result;
  result.source_identity = graph.identity();
  result.blocks.resize(blocks.size());
  if (budget.try_consume({preserved.size(), preserved.size() * sizeof(ir::StorageId)}) !=
      BudgetDecline::none)
    return decline(RelationDecline::resource_limit);
  result.preserved.assign(preserved.begin(), preserved.end());
  const auto refuse = [&] {
    EntryRelations none;
    none.source_identity = graph.identity();
    none.blocks.resize(blocks.size());
    none.preserved = std::move(result.preserved);
    return EntryRelationsResult{std::move(none), RelationDecline::none};
  };
  while (!work.empty()) {
    const auto id = work.back();
    work.pop_back();
    queued[id] = 0;
    const auto& block = blocks[id];
    const auto nodes = block.ssa ? block.ssa->nodes().size() : 0;
    if (budget.try_consume({width * width * (block.edges.size() + 2) + nodes + 1,
                            (block.edges.size() + 2) * width * sizeof(Term) +
                                nodes * sizeof(Term)}) != BudgetDecline::none)
      return decline(RelationDecline::resource_limit);
    result.declared_opaque_control |=
        !block.ssa && sources[block.first_source].opaque_control != OpaqueControl::unknown;
    const auto out = Transfer(block, *states[id], tracked);
    for (const auto& edge : block.edges) {
      if (edge.target_block && *edge.target_block >= blocks.size())
        return decline(RelationDecline::invalid_input);
      const bool returns = edge.kind == CfgEdgeKind::return_ && !edge.target_block;
      const bool leaves = edge.kind == CfgEdgeKind::callee || returns;
      if (leaves && !assumptions.calling_convention) return refuse();
      if (returns && !assumptions.return_leaves) return refuse();
      result.calling_convention |= leaves;
      result.return_leaves |= returns;
      if (!edge.target_block) {
        if (!leaves && edge.kind != CfgEdgeKind::trap) return refuse();
        continue;
      }

      result.constant_image |= edge.constant_image_dependency;
      if (edge.kind != CfgEdgeKind::potential_return) {
        merge(*edge.target_block, out);
        continue;
      }

      auto returned = out;
      auto fresh = static_cast<std::uint32_t>(width);
      for (std::size_t i = 0; i < width; ++i) {
        if (!std::binary_search(preserved.begin(), preserved.end(), tracked[i]))
          returned[i] = {fresh++, 0};
      }

      merge(*edge.target_block, Canonical(returned));
    }
  }

  for (std::uint32_t id = 0; id < blocks.size(); ++id) {
    if (!states[id]) continue;
    for (std::uint32_t i = 0; i < width; ++i) {
      const auto& term = (*states[id])[i];
      if (term.symbol != i)
        result.blocks[id].push_back({tracked[i], tracked[term.symbol], term.offset});
    }
  }

  return {std::move(result), RelationDecline::none};
}

}  // namespace nyx::analysis
