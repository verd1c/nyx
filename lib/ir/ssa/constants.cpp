#include <algorithm>
#include <array>
#include <bit>
#include <limits>

#include "nyx/ir/fold.hpp"
#include "nyx/ir/ssa/graph.hpp"

namespace nyx::ir {
namespace {
using Values = std::vector<std::vector<std::optional<std::uint64_t>>>;
using ReadAt = std::vector<std::vector<std::size_t>>;

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

std::optional<std::uint64_t> Claimed(const SsaGraph& graph, const SsaValue& value,
                                     const Values& phis, const Values& nodes,
                                     const Values& frames) {
  switch (value.kind) {
    case SsaValueKind::phi:
      return phis[value.block.slot][value.index];
    case SsaValueKind::node:
      return nodes[value.block.slot][value.index];
    case SsaValueKind::frame_phi:
      return frames[value.block.slot][value.index];
    // The run may declare what a callee leaves; that is the caller's claim,
    // rechecked here against the same declaration the graph carries.
    case SsaValueKind::clobber:
      return SsaClobberResult(graph, value);
    default:
      return {};
  }
}

// Registers, then private slots, then nodes: an order both the prover and
// this check agree on, so a fact set has one spelling.
unsigned Rank(SsaValueKind kind) {
  return kind == SsaValueKind::phi ? 0 : kind == SsaValueKind::frame_phi ? 1 : 2;
}

bool Before(const SsaConstantValue& left, const SsaConstantValue& right) {
  if (left.block.slot != right.block.slot) return left.block.slot < right.block.slot;
  if (left.kind != right.kind) return Rank(left.kind) < Rank(right.kind);
  return left.index < right.index;
}

// What a bounded loop establishes for one of its successor's phis. The
// incoming behind it is the loop's own accumulator, so the ordinary rule --
// every reachable incoming agrees -- cannot see it.
std::optional<std::uint64_t> FromLoop(const SsaBoundedLoopFacts* loops,
                                      const SsaConstantValue& fact) {
  if (!loops) return std::nullopt;
  for (const auto& loop : loops->loops) {
    if (loop.successor != fact.block) continue;
    for (const auto& exit : loop.exits)
      if (exit.kind == fact.kind && exit.index == fact.index) return exit.value;
  }

  return std::nullopt;
}

SsaDecline ValidateConstants(const SsaGraph& graph, std::span<const std::uint8_t> reachable,
                             const std::vector<std::vector<std::uint8_t>>* edges,
                             const SsaConstantFacts& facts, Budget& budget,
                             const SsaBoundedLoopFacts* loops) {
  if (facts.graph_arena != graph.arena() || facts.graph_revision != graph.revision())
    return SsaDecline::invalid_graph;
  if (graph.slots() > std::numeric_limits<std::size_t>::max() /
                          (2 * sizeof(std::vector<std::optional<std::uint64_t>>) +
                           sizeof(std::vector<std::size_t>)) ||
      budget.try_consume(
          {graph.slots(), graph.slots() * (2 * sizeof(std::vector<std::optional<std::uint64_t>>) +
                                           sizeof(std::vector<std::size_t>))}) !=
          BudgetDecline::none)
    return SsaDecline::resource_limit;
  Values phis(graph.slots()), nodes(graph.slots()), frames(graph.slots());
  ReadAt read_at(graph.slots());
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return SsaDecline::resource_limit;
    const auto handle = graph.Handle(slot);
    if (!handle || !reachable[slot]) continue;
    const auto& block = *graph.Get(*handle);
    if (block.phis.size() >
            std::numeric_limits<std::size_t>::max() / sizeof(std::optional<std::uint64_t>) ||
        block.nodes.size() >
            std::numeric_limits<std::size_t>::max() / sizeof(std::optional<std::uint64_t>) ||
        block.nodes.size() > std::numeric_limits<std::size_t>::max() /
                                 (sizeof(std::optional<std::uint64_t>) + sizeof(std::size_t)) ||
        budget.try_consume({block.phis.size() + block.nodes.size(),
                            block.phis.size() * sizeof(std::optional<std::uint64_t>) +
                                block.nodes.size() * (sizeof(std::optional<std::uint64_t>) +
                                                      sizeof(std::size_t))}) != BudgetDecline::none)
      return SsaDecline::resource_limit;
    if (block.frame_phis.size() >
            std::numeric_limits<std::size_t>::max() / sizeof(std::optional<std::uint64_t>) ||
        budget.try_consume({block.frame_phis.size(),
                            block.frame_phis.size() * sizeof(std::optional<std::uint64_t>)}) !=
            BudgetDecline::none)
      return SsaDecline::resource_limit;
    phis[slot].resize(block.phis.size());
    nodes[slot].resize(block.nodes.size());
    frames[slot].resize(block.frame_phis.size());
    read_at[slot].resize(block.nodes.size(), SIZE_MAX);
    if (budget.try_consume({block.reads.size(), 0}) != BudgetDecline::none)
      return SsaDecline::resource_limit;
    for (std::size_t index = 0; index < block.reads.size(); ++index)
      read_at[slot][block.reads[index].node] = index;
  }

  std::optional<SsaConstantValue> previous;
  for (const auto& fact : facts.values) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return SsaDecline::resource_limit;
    const auto* block = graph.Get(fact.block);
    if (!block || !reachable[fact.block.slot] || (previous && !Before(*previous, fact)))
      return SsaDecline::invalid_graph;
    unsigned width = 0;
    if (fact.kind == SsaValueKind::phi && fact.index < block->phis.size()) {
      width = block->phis[fact.index].width;
      phis[fact.block.slot][fact.index] = fact.value;
    } else if (fact.kind == SsaValueKind::frame_phi && fact.index < block->frame_phis.size()) {
      const auto bytes = block->frame_phis[fact.index].size;
      width = bytes > 8 ? 0 : bytes * 8;
      frames[fact.block.slot][fact.index] = fact.value;
    } else if (fact.kind == SsaValueKind::node && fact.index < block->nodes.size()) {
      width = block->nodes[fact.index].width;
      nodes[fact.block.slot][fact.index] = fact.value;
    } else
      return SsaDecline::invalid_graph;
    if (!width || width > 64 || (fact.value & ~LowMask(width))) return SsaDecline::invalid_graph;
    previous = fact;
  }

  for (const auto& fact : facts.values) {
    const auto& block = *graph.Get(fact.block);
    if (fact.kind == SsaValueKind::phi || fact.kind == SsaValueKind::frame_phi) {
      const bool frame = fact.kind == SsaValueKind::frame_phi;
      const bool external = frame ? block.frame_phis[fact.index].external_entry
                                  : block.phis[fact.index].external_entry;
      const auto incoming_values =
          frame ? std::span<const SsaPhiInput>(block.frame_phis[fact.index].incoming)
                : std::span<const SsaPhiInput>(block.phis[fact.index].incoming);
      if (budget.try_consume({1 + incoming_values.size(), 0}) != BudgetDecline::none)
        return SsaDecline::resource_limit;
      if (external || incoming_values.empty()) return SsaDecline::invalid_graph;

      // A run licenses the value the join cannot reach, but where the join
      // does reach one the two have to agree.
      if (const auto run = FromLoop(loops, fact)) {
        if (*run != fact.value) return SsaDecline::invalid_graph;
        for (const auto& input : incoming_values) {
          if (!reachable[input.predecessor.slot]) continue;
          if (budget.try_consume({1, 0}) != BudgetDecline::none) return SsaDecline::resource_limit;
          const auto claimed = Claimed(graph, input.value, phis, nodes, frames);
          if (claimed && *claimed != fact.value) return SsaDecline::invalid_graph;
        }

        continue;
      }

      bool incoming = false;
      for (const auto& input : incoming_values) {
        if (!reachable[input.predecessor.slot]) continue;
        if (edges) {
          bool selected = false;
          const auto& predecessor = *graph.Get(input.predecessor);
          if (budget.try_consume({predecessor.edges.size(), 0}) != BudgetDecline::none)
            return SsaDecline::resource_limit;
          for (std::size_t index = 0; index < predecessor.edges.size(); ++index)
            if ((*edges)[input.predecessor.slot][index] &&
                predecessor.edges[index].target_block == fact.block)
              selected = true;
          if (!selected) continue;
        }

        incoming = true;
        if (Claimed(graph, input.value, phis, nodes, frames) != fact.value)
          return SsaDecline::invalid_graph;
      }

      if (!incoming) return SsaDecline::invalid_graph;
      continue;
    }

    const auto& node = block.nodes[fact.index];
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return SsaDecline::resource_limit;
    if (node.op == Op::constant) {
      if ((node.immediate & LowMask(node.width)) != fact.value) return SsaDecline::invalid_graph;
      continue;
    }

    if (node.op == Op::image_address) {
      if (!graph.load_bias() ||
          ((*graph.load_bias() + node.immediate) & LowMask(node.width)) != fact.value)
        return SsaDecline::invalid_graph;
      continue;
    }

    if (node.op == Op::read) {
      const auto index = read_at[fact.block.slot][fact.index];
      if (index == SIZE_MAX ||
          Claimed(graph, block.reads[index].value, phis, nodes, frames) != fact.value)
        return SsaDecline::invalid_graph;
      continue;
    }

    // Promotion recorded which value this load reads, so the load's value is
    // that one. The load itself is kept; only its value is recorded.
    if (const auto* frame = FrameOf(block, fact.index); frame && frame->replacement) {
      if (Claimed(graph, *frame->replacement, phis, nodes, frames) != fact.value)
        return SsaDecline::invalid_graph;
      continue;
    }

    // The fold record is already checked against the declared bytes; this
    // only says what the load it records yields, and a location is a number
    // only where the run declared the placement that makes it one.
    if (const auto* fold = FoldOf(block, fact.index);
        fold && fold->skip_access && !fold->condition) {
      std::optional<std::uint64_t> expected;
      if (fold->kind == SsaConstantKind::literal)
        expected = fold->value & LowMask(node.width);
      else if (fold->kind == SsaConstantKind::image_location && graph.load_bias())
        expected = (*graph.load_bias() + fold->value) & LowMask(node.width);
      if (expected != fact.value) return SsaDecline::invalid_graph;
      continue;
    }

    const auto* descriptor = Descriptor(node.op);
    if (!descriptor || descriptor->effect != Effect::pure || !descriptor->produces_value ||
        !descriptor->arity || budget.try_consume({4 * descriptor->arity, 0}) != BudgetDecline::none)
      return descriptor && descriptor->effect == Effect::pure && descriptor->produces_value &&
                     descriptor->arity
                 ? SsaDecline::resource_limit
                 : SsaDecline::invalid_graph;
    std::array<std::optional<std::uint64_t>, 3> inputs{};
    for (unsigned i = 0; i < descriptor->arity; ++i)
      inputs[i] = nodes[fact.block.slot][node.inputs[i]];
    std::optional<std::uint64_t> expected;
    if (node.op == Op::select && inputs[0])
      expected = inputs[*inputs[0] ? 1 : 2];
    else if (node.op == Op::select && inputs[1] && inputs[2] && *inputs[1] == *inputs[2])
      expected = inputs[1];
    else if (std::all_of(inputs.begin(), inputs.begin() + descriptor->arity,
                         [](const auto& value) { return value.has_value(); })) {
      std::array<std::uint64_t, 3> values{};
      for (unsigned i = 0; i < descriptor->arity; ++i) values[i] = *inputs[i];
      expected = FoldPure(node, std::span(values.data(), descriptor->arity),
                          block.nodes[node.inputs[0]].width);
    }

    if (expected != fact.value) return SsaDecline::invalid_graph;
  }

  return SsaDecline::none;
}
}  // namespace

SsaDecline ValidateSsaConstantFacts(const SsaGraph& graph, const SsaReachabilityFacts& reachable,
                                    const SsaConstantFacts& facts, std::span<const Group> sources,
                                    Budget& budget, const SsaBoundedLoopFacts* loops) {
  const auto valid = ValidateSsa(graph, budget);
  if (valid != SsaDecline::none) return valid;
  const auto scope = ValidateSsaReachabilityFacts(graph, reachable, sources, budget);
  if (scope != SsaDecline::none) return scope;
  if (loops) {
    const auto runs = ValidateSsaBoundedLoopFacts(graph, *loops, sources, budget);
    if (runs != SsaDecline::none) return runs;
  }

  return ValidateConstants(graph, reachable.reachable, nullptr, facts, budget, loops);
}

SsaDecline ValidateSsaSccpFacts(const SsaGraph& graph, const SsaSccpFacts& facts,
                                std::span<const Group> sources, Budget& budget,
                                const SsaBoundedLoopFacts* loops) {
  const auto valid = ValidateSsa(graph, budget);
  if (valid != SsaDecline::none) return valid;
  if (facts.graph_arena != graph.arena() || facts.graph_revision != graph.revision() ||
      facts.entry_scope != SsaEntryScope::closed_population || graph.entries().empty() ||
      facts.executable.size() != graph.slots() || facts.edges.size() != graph.slots())
    return SsaDecline::invalid_graph;
  std::optional<SsaConstantValue> previous;
  for (const auto& value : facts.constants.values) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return SsaDecline::resource_limit;
    if (previous && !Before(*previous, value)) return SsaDecline::invalid_graph;
    previous = value;
  }

  for (const auto entry : graph.entries())
    if (!facts.executable[entry.slot]) return SsaDecline::invalid_graph;
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return SsaDecline::resource_limit;
    const auto handle = graph.Handle(slot);
    if (facts.executable[slot] > 1 || (!handle && facts.executable[slot]))
      return SsaDecline::invalid_graph;
    if (!handle) {
      if (!facts.edges[slot].empty()) return SsaDecline::invalid_graph;
      continue;
    }

    const auto& block = *graph.Get(*handle);
    if (facts.executable[slot]) {
      const auto member = ValidateSsaClosedMember(block, sources, budget);
      if (member != SsaDecline::none) return member;
    }

    if (facts.edges[slot].size() != block.edges.size()) return SsaDecline::invalid_graph;
    if (budget.try_consume({block.edges.size(), 0}) != BudgetDecline::none)
      return SsaDecline::resource_limit;
    for (std::size_t index = 0; index < block.edges.size(); ++index) {
      const auto& edge = block.edges[index];
      const auto selected = facts.edges[slot][index];
      if (selected > 1 || (!facts.executable[slot] && selected)) return SsaDecline::invalid_graph;
      if (!facts.executable[slot]) continue;
      if (edge.condition) {
        const auto& claim = facts.constants.values;
        const SsaConstantValue key{*handle, SsaValueKind::node, *edge.condition, 0};
        if (budget.try_consume({static_cast<std::uint64_t>(std::bit_width(claim.size())), 0}) !=
            BudgetDecline::none)
          return SsaDecline::resource_limit;
        const auto found = std::lower_bound(claim.begin(), claim.end(), key, Before);
        if (found == claim.end() || found->block != key.block || found->kind != key.kind ||
            found->index != key.index || ((found->value & 1) == *edge.when)) {
          if (!selected) return SsaDecline::invalid_graph;
        }
      } else if (!selected)
        return SsaDecline::invalid_graph;
      if (selected && edge.target_block && !facts.executable[edge.target_block->slot])
        return SsaDecline::invalid_graph;
    }
  }

  if (loops) {
    const auto runs = ValidateSsaBoundedLoopFacts(graph, *loops, sources, budget);
    if (runs != SsaDecline::none) return runs;
  }

  return ValidateConstants(graph, facts.executable, &facts.edges, facts.constants, budget, loops);
}

}  // namespace nyx::ir
