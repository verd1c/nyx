#include <algorithm>
#include <array>
#include <limits>
#include <optional>

#include "nyx/ir/fold.hpp"
#include "nyx/ir/ssa/graph.hpp"

namespace nyx::ir {
namespace {

using Slot = std::optional<std::uint64_t>;

const SsaConstantLoad* FoldAt(const SsaBlock& block, ValueId id) {
  const auto at =
      std::lower_bound(block.constant_loads.begin(), block.constant_loads.end(), id,
                       [](const SsaConstantLoad& fold, ValueId key) { return fold.node < key; });
  return at != block.constant_loads.end() && at->node == id ? &*at : nullptr;
}

const SsaFrameAccess* FrameAt(const SsaBlock& block, ValueId id) {
  const auto at =
      std::lower_bound(block.frame_accesses.begin(), block.frame_accesses.end(), id,
                       [](const SsaFrameAccess& access, ValueId key) { return access.node < key; });
  return at != block.frame_accesses.end() && at->node == id ? &*at : nullptr;
}

bool Listed(std::span<const ValueId> sorted, ValueId id) {
  return std::binary_search(sorted.begin(), sorted.end(), id);
}

// A value of the loop block, read against one iteration's state. Only the
// block's own phis, frame phis and nodes can appear; anything reaching outside
// the iteration is not something a run of this block establishes.
Slot Here(const SsaValue& value, SsaHandle loop, const std::vector<Slot>& phis,
          const std::vector<Slot>& frames, const std::vector<Slot>& nodes) {
  if (value.block != loop) return std::nullopt;
  switch (value.kind) {
    case SsaValueKind::phi:
      return value.index < phis.size() ? phis[value.index] : std::nullopt;
    case SsaValueKind::frame_phi:
      return value.index < frames.size() ? frames[value.index] : std::nullopt;
    case SsaValueKind::node:
      return value.index < nodes.size() ? nodes[value.index] : std::nullopt;
    default:
      return std::nullopt;
  }
}

// Whether the block's effects are all ones a run of it accounts for. A store
// the frame did not absorb, or a load that is neither folded, promoted nor
// retired, can carry state between iterations that the state vector does not
// model, and then every value below it is a guess.
bool EffectsAccounted(const SsaBlock& block, Budget& budget, bool& exhausted) {
  for (ValueId id = 0; id < block.nodes.size(); ++id) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) {
      exhausted = true;
      return false;
    }

    const auto op = block.nodes[id].op;
    const auto* descriptor = Descriptor(op);
    if (!descriptor || descriptor->effect == Effect::pure ||
        descriptor->effect == Effect::storage_read)
      continue;
    if (Listed(block.disabled_effects, id) || Listed(block.dead_pure_nodes, id)) continue;
    if (op == Op::load) {
      const auto* frame = FrameAt(block, id);
      if (frame && frame->replacement) continue;
      if (FoldAt(block, id) != nullptr) continue;
      const auto retired =
          std::lower_bound(block.retired_loads.begin(), block.retired_loads.end(), id,
                           [](const SsaRetiredLoad& load, ValueId key) { return load.node < key; });
      if (retired != block.retired_loads.end() && retired->node == id) continue;
      return false;
    }

    if (op == Op::store) {
      const auto* frame = FrameAt(block, id);
      if (frame && !frame->replacement) continue;
      return false;
    }

    return false;
  }

  return true;
}

// One pass over the block's nodes against the current state. Mirrors the
// constant lattice's transfer function on a concrete state rather than a
// lattice one, so a value is a number or it is not established.
bool Step(const SsaGraph& graph, const SsaBlock& block, SsaHandle loop,
          const std::vector<Slot>& phis, const std::vector<Slot>& frames, std::vector<Slot>& nodes,
          const std::vector<std::uint32_t>& read_at, Budget& budget, bool& exhausted) {
  for (ValueId id = 0; id < block.nodes.size(); ++id) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) {
      exhausted = true;
      return false;
    }

    const auto& node = block.nodes[id];
    nodes[id] = std::nullopt;
    if (node.width > 64) continue;
    if (node.op == Op::constant) {
      nodes[id] = node.immediate & LowMask(node.width);
      continue;
    }

    if (node.op == Op::read) {
      if (read_at[id] == std::numeric_limits<std::uint32_t>::max()) continue;
      nodes[id] = Here(block.reads[read_at[id]].value, loop, phis, frames, nodes);
      continue;
    }

    if (const auto* frame = FrameAt(block, id); frame && frame->replacement) {
      nodes[id] = Here(*frame->replacement, loop, phis, frames, nodes);
      continue;
    }

    if (const auto* fold = FoldAt(block, id); fold && fold->skip_access) {
      // A selected fold names the value under each of its condition's two
      // outcomes. Concretely the condition is a number, so the fold is one
      // value, where the lattice must decline it.
      auto value = fold->value;
      if (fold->condition) {
        if (*fold->condition >= id || !nodes[*fold->condition]) continue;
        if (*nodes[*fold->condition] == 0) value = fold->alternative_value;
      }

      if (fold->kind == SsaConstantKind::literal) {
        nodes[id] = value & LowMask(node.width);
        continue;
      }

      if (fold->kind == SsaConstantKind::image_location && graph.load_bias()) {
        nodes[id] = (*graph.load_bias() + value) & LowMask(node.width);
        continue;
      }

      continue;
    }

    const auto* descriptor = Descriptor(node.op);
    if (!descriptor || descriptor->effect != Effect::pure || !descriptor->produces_value ||
        !descriptor->arity)
      continue;
    std::array<std::uint64_t, 3> inputs{};
    bool known = true;
    for (unsigned i = 0; i < descriptor->arity; ++i) {
      if (node.inputs[i] >= id || !nodes[node.inputs[i]])
        known = false;
      else
        inputs[i] = *nodes[node.inputs[i]];
    }

    if (node.op == Op::select && node.inputs[0] < id && nodes[node.inputs[0]]) {
      const auto taken = node.inputs[*nodes[node.inputs[0]] ? 1 : 2];
      nodes[id] = taken < id ? nodes[taken] : std::nullopt;
      continue;
    }

    if (!known) continue;
    nodes[id] = FoldPure(node, std::span(inputs.data(), descriptor->arity),
                         block.nodes[node.inputs[0]].width);
  }

  return true;
}

// Which edge the state takes, or none when the block's conditions do not
// select exactly one.
std::optional<std::uint32_t> Taken(const SsaBlock& block, const std::vector<Slot>& nodes) {
  std::optional<std::uint32_t> taken;
  for (std::uint32_t index = 0; index < block.edges.size(); ++index) {
    const auto& edge = block.edges[index];
    if (!edge.condition || !edge.when) return std::nullopt;
    if (*edge.condition >= nodes.size() || !nodes[*edge.condition]) return std::nullopt;
    if ((*nodes[*edge.condition] != 0) != *edge.when) continue;
    if (taken) return std::nullopt;
    taken = index;
  }

  return taken;
}

struct Shape {
  SsaHandle predecessor{};
  std::uint32_t self_edge = 0;
  std::uint32_t exit_edge = 0;
  bool valid = false;
};

// A block this analysis can run: two conditional edges, one back to itself and
// one to a block it is the only predecessor of, and exactly one other
// predecessor to take its entry values from. An architectural entry is not
// one of those -- its storage arrives from outside the graph, so the values
// the predecessor publishes are not what the block runs from.
Shape LoopShape(const SsaGraph& graph, SsaHandle loop,
                const std::vector<std::vector<SsaHandle>>& predecessors) {
  Shape shape;
  const auto* block = graph.Get(loop);
  if (!block || block->edges.size() != 2 || block->opaque) return shape;
  for (const auto& phi : block->phis)
    if (phi.external_entry) return shape;
  for (const auto& phi : block->frame_phis)
    if (phi.external_entry) return shape;
  std::optional<std::uint32_t> self, exit;
  for (std::uint32_t index = 0; index < 2; ++index) {
    const auto& edge = block->edges[index];
    if (edge.kind != SsaEdgeKind::branch || !edge.target_block || !edge.condition || !edge.when ||
        edge.assumptions.unresolved_target)
      return shape;
    if (*edge.target_block == loop)
      self = index;
    else
      exit = index;
  }

  if (!self || !exit) return shape;
  const auto successor = *block->edges[*exit].target_block;
  if (successor.slot >= predecessors.size() || loop.slot >= predecessors.size()) return shape;
  const auto& after = predecessors[successor.slot];
  if (after.size() != 1 || after[0] != loop) return shape;
  const auto& incoming = predecessors[loop.slot];
  if (incoming.size() != 2) return shape;
  const auto other = incoming[0] == loop ? incoming[1] : incoming[0];
  if (other == loop) return shape;
  shape.predecessor = other;
  shape.self_edge = *self;
  shape.exit_edge = *exit;
  shape.valid = true;
  return shape;
}

// The value a predecessor publishes for one of the loop's phis, which is a
// number only where the predecessor already folded it to a constant.
Slot Published(const SsaGraph& graph, const SsaValue& value) {
  if (value.kind != SsaValueKind::node) return std::nullopt;
  const auto* block = graph.Get(value.block);
  if (!block || value.index >= block->nodes.size()) return std::nullopt;
  const auto& node = block->nodes[value.index];
  if (node.op != Op::constant || node.width > 64) return std::nullopt;
  return node.immediate & LowMask(node.width);
}

struct Outcome {
  std::uint32_t iterations = 0;
  std::vector<Slot> phis, frames, nodes;
  bool ran = false;
};

// Runs the block from its entry values until it leaves by the exit edge.
Outcome Run(const SsaGraph& graph, SsaHandle loop, const Shape& shape, Budget& budget,
            bool& exhausted) {
  Outcome outcome;
  const auto& block = *graph.Get(loop);
  if (!EffectsAccounted(block, budget, exhausted)) return outcome;
  if (block.nodes.size() > std::numeric_limits<std::size_t>::max() / sizeof(Slot) ||
      budget.try_consume({block.nodes.size() + block.phis.size() + block.frame_phis.size(),
                          (block.nodes.size() + block.phis.size() + block.frame_phis.size()) *
                              sizeof(Slot)}) != BudgetDecline::none) {
    exhausted = true;
    return outcome;
  }

  std::vector<std::uint32_t> read_at(block.nodes.size(), std::numeric_limits<std::uint32_t>::max());
  for (std::size_t index = 0; index < block.reads.size(); ++index) {
    if (block.reads[index].node >= block.nodes.size()) return outcome;
    read_at[block.reads[index].node] = static_cast<std::uint32_t>(index);
  }

  std::vector<Slot> phis(block.phis.size()), frames(block.frame_phis.size());
  outcome.nodes.assign(block.nodes.size(), std::nullopt);
  const auto incoming = [&](const std::vector<SsaPhiInput>& inputs,
                            SsaHandle from) -> const SsaValue* {
    for (const auto& input : inputs)
      if (input.predecessor == from) return &input.value;
    return nullptr;
  };

  for (std::size_t index = 0; index < block.phis.size(); ++index)
    if (const auto* value = incoming(block.phis[index].incoming, shape.predecessor))
      phis[index] = Published(graph, *value);
  for (std::size_t index = 0; index < block.frame_phis.size(); ++index)
    if (const auto* value = incoming(block.frame_phis[index].incoming, shape.predecessor))
      frames[index] = Published(graph, *value);
  for (unsigned iteration = 0; iteration < kMaxBoundedLoopIterations; ++iteration) {
    if (!Step(graph, block, loop, phis, frames, outcome.nodes, read_at, budget, exhausted))
      return outcome;
    const auto taken = Taken(block, outcome.nodes);
    if (!taken) return outcome;
    if (*taken == shape.exit_edge) {
      outcome.iterations = iteration + 1;
      outcome.phis = std::move(phis);
      outcome.frames = std::move(frames);
      outcome.ran = true;
      return outcome;
    }

    if (*taken != shape.self_edge) return outcome;
    std::vector<Slot> next_phis(block.phis.size()), next_frames(block.frame_phis.size());
    for (std::size_t index = 0; index < block.phis.size(); ++index)
      if (const auto* value = incoming(block.phis[index].incoming, loop))
        next_phis[index] = Here(*value, loop, phis, frames, outcome.nodes);
    for (std::size_t index = 0; index < block.frame_phis.size(); ++index)
      if (const auto* value = incoming(block.frame_phis[index].incoming, loop))
        next_frames[index] = Here(*value, loop, phis, frames, outcome.nodes);
    phis = std::move(next_phis);
    frames = std::move(next_frames);
  }

  return outcome;
}

}  // namespace

SsaDecline ValidateSsaBoundedLoopFacts(const SsaGraph& graph, const SsaBoundedLoopFacts& facts,
                                       std::span<const Group> sources, Budget& budget) {
  const auto valid = ValidateSsaWithSources(graph, sources, budget);
  if (valid != SsaDecline::none) return valid;
  if (facts.graph_arena != graph.arena() || facts.graph_revision != graph.revision())
    return SsaDecline::invalid_graph;
  // One pass for the predecessors: the shape test needs them per block, and
  // rescanning the graph for each would cost the square of its size.
  if (budget.try_consume({graph.slots(), graph.slots() * sizeof(std::vector<SsaHandle>)}) !=
      BudgetDecline::none)
    return SsaDecline::resource_limit;
  std::vector<std::vector<SsaHandle>> predecessors(graph.slots());
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto& edges = graph.Get(*handle)->edges;
    if (budget.try_consume({1 + edges.size(), edges.size() * sizeof(SsaHandle)}) !=
        BudgetDecline::none)
      return SsaDecline::resource_limit;
    for (const auto& edge : edges) {
      if (!edge.target_block || edge.target_block->slot >= predecessors.size()) continue;
      auto& into = predecessors[edge.target_block->slot];
      if (std::find(into.begin(), into.end(), *handle) == into.end()) into.push_back(*handle);
    }
  }

  std::size_t claimed = 0;
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return SsaDecline::resource_limit;
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto shape = LoopShape(graph, *handle, predecessors);
    if (!shape.valid) continue;
    bool exhausted = false;
    const auto outcome = Run(graph, *handle, shape, budget, exhausted);
    if (exhausted) return SsaDecline::resource_limit;
    if (!outcome.ran) continue;
    const auto& block = *graph.Get(*handle);
    const auto successor = *block.edges[shape.exit_edge].target_block;
    const auto& after = *graph.Get(successor);

    // What the successor's phis hold is what the last iteration published,
    // read through the incoming the successor names for this block.
    std::vector<SsaBoundedLoopExit> exits;

    // A slot wider than a number is not one a value can be stated about, and
    // every consumer of these exits rejects such a fact.
    const auto record = [&](SsaValueKind kind, std::size_t index, unsigned width,
                            const std::vector<SsaPhiInput>& inputs) {
      if (!width || width > 64) return;
      for (const auto& input : inputs) {
        if (input.predecessor != *handle) continue;
        const auto value = Here(input.value, *handle, outcome.phis, outcome.frames, outcome.nodes);
        if (value && (*value & ~LowMask(width)) == 0)
          exits.push_back({kind, static_cast<std::uint32_t>(index), *value});
      }
    };

    if (budget.try_consume({after.phis.size() + after.frame_phis.size(),
                            (after.phis.size() + after.frame_phis.size()) *
                                sizeof(SsaBoundedLoopExit)}) != BudgetDecline::none)
      return SsaDecline::resource_limit;
    for (std::size_t index = 0; index < after.phis.size(); ++index)
      record(SsaValueKind::phi, index, after.phis[index].width, after.phis[index].incoming);
    for (std::size_t index = 0; index < after.frame_phis.size(); ++index)
      record(SsaValueKind::frame_phi, index,
             after.frame_phis[index].size > 8 ? 0 : after.frame_phis[index].size * 8,
             after.frame_phis[index].incoming);
    if (claimed >= facts.loops.size()) return SsaDecline::invalid_graph;
    const auto& fact = facts.loops[claimed++];
    if (fact.loop != *handle || fact.successor != successor || fact.self_edge != shape.self_edge ||
        fact.exit_edge != shape.exit_edge || fact.iterations != outcome.iterations ||
        fact.exits != exits)
      return SsaDecline::invalid_graph;
  }

  return claimed == facts.loops.size() ? SsaDecline::none : SsaDecline::invalid_graph;
}

}  // namespace nyx::ir
