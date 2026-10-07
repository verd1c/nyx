#include "nyx/analysis/ssa/bounded_loop.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <map>

#include "nyx/ir/fold.hpp"

namespace nyx::analysis {
namespace {

using Number = std::optional<std::uint64_t>;

SsaBoundedLoopResult Decline(SsaBoundedLoopRefusal reason) { return {{}, reason}; }

// Per-iteration state: one entry per phi, frame phi and node of the block.
struct State {
  std::vector<Number> phis, frames, nodes;
};

Number Read(const ir::SsaValue& value, ir::SsaHandle loop, const State& state) {
  if (value.block != loop) return std::nullopt;
  const std::vector<Number>* from = nullptr;
  switch (value.kind) {
    case ir::SsaValueKind::phi:
      from = &state.phis;
      break;
    case ir::SsaValueKind::frame_phi:
      from = &state.frames;
      break;
    case ir::SsaValueKind::node:
      from = &state.nodes;
      break;
    default:
      return std::nullopt;
  }

  return value.index < from->size() ? (*from)[value.index] : std::nullopt;
}

const ir::SsaValue* From(const std::vector<ir::SsaPhiInput>& incoming, ir::SsaHandle predecessor) {
  const auto at = std::find_if(incoming.begin(), incoming.end(), [&](const ir::SsaPhiInput& input) {
    return input.predecessor == predecessor;
  });
  return at == incoming.end() ? nullptr : &at->value;
}

struct Records {
  // Node to its index in each sorted record list, or none.
  std::vector<std::uint32_t> read_of, fold_of, frame_of, retired_of;
  bool charged = false;  // the budget allowed the tables
  bool usable = false;   // and every record names a node of this block
};

constexpr auto kNone = std::numeric_limits<std::uint32_t>::max();

Records Index(const ir::SsaBlock& block, Budget& budget) {
  Records records;
  if (budget.try_consume({block.nodes.size() * 4,
                          block.nodes.size() * 4 * sizeof(std::uint32_t)}) != BudgetDecline::none)
    return records;
  records.charged = true;
  records.read_of.assign(block.nodes.size(), kNone);
  records.fold_of.assign(block.nodes.size(), kNone);
  records.frame_of.assign(block.nodes.size(), kNone);
  records.retired_of.assign(block.nodes.size(), kNone);
  const auto place = [&](std::vector<std::uint32_t>& into, ir::ValueId node, std::size_t index) {
    if (node >= into.size()) return false;
    into[node] = static_cast<std::uint32_t>(index);
    return true;
  };

  for (std::size_t i = 0; i < block.reads.size(); ++i)
    if (!place(records.read_of, block.reads[i].node, i)) return records;
  for (std::size_t i = 0; i < block.constant_loads.size(); ++i)
    if (!place(records.fold_of, block.constant_loads[i].node, i)) return records;
  for (std::size_t i = 0; i < block.frame_accesses.size(); ++i)
    if (!place(records.frame_of, block.frame_accesses[i].node, i)) return records;
  for (std::size_t i = 0; i < block.retired_loads.size(); ++i)
    if (!place(records.retired_of, block.retired_loads[i].node, i)) return records;
  records.usable = true;
  return records;
}

// Whether every effect the block still performs is one the state models. A
// store the frame did not absorb, or a load nothing resolved, moves state
// between iterations outside the vectors, so the run below would be reading
// its own omission back as a value.
bool Modeled(const ir::SsaBlock& block, const Records& records) {
  for (ir::ValueId id = 0; id < block.nodes.size(); ++id) {
    const auto op = block.nodes[id].op;
    const auto* descriptor = ir::Descriptor(op);
    if (!descriptor || descriptor->effect == ir::Effect::pure ||
        descriptor->effect == ir::Effect::storage_read)
      continue;
    if (std::binary_search(block.disabled_effects.begin(), block.disabled_effects.end(), id) ||
        std::binary_search(block.dead_pure_nodes.begin(), block.dead_pure_nodes.end(), id))
      continue;
    const auto frame = records.frame_of[id];
    if (op == ir::Op::load) {
      if (frame != kNone && block.frame_accesses[frame].replacement) continue;
      if (records.fold_of[id] != kNone || records.retired_of[id] != kNone) continue;
      return false;
    }

    if (op == ir::Op::store && frame != kNone && !block.frame_accesses[frame].replacement) continue;
    return false;
  }

  return true;
}

void Iterate(const ir::SsaGraph& graph, const ir::SsaBlock& block, ir::SsaHandle loop,
             const Records& records, State& state) {
  for (ir::ValueId id = 0; id < block.nodes.size(); ++id) {
    const auto& node = block.nodes[id];
    auto& here = state.nodes[id];
    here = std::nullopt;
    if (node.width > 64) continue;
    if (node.op == ir::Op::constant) {
      here = node.immediate & ir::LowMask(node.width);
      continue;
    }

    if (node.op == ir::Op::read) {
      if (records.read_of[id] != kNone)
        here = Read(block.reads[records.read_of[id]].value, loop, state);
      continue;
    }

    if (const auto frame = records.frame_of[id]; frame != kNone) {
      if (const auto& replacement = block.frame_accesses[frame].replacement)
        here = Read(*replacement, loop, state);
      continue;
    }

    if (const auto index = records.fold_of[id]; index != kNone) {
      const auto& fold = block.constant_loads[index];
      if (!fold.skip_access) continue;
      auto value = fold.value;
      if (fold.condition) {
        // Concretely the condition is one value, so the selected fold reads
        // one of its two declarations and not the other.
        if (*fold.condition >= id || !state.nodes[*fold.condition]) continue;
        if (*state.nodes[*fold.condition] == 0) value = fold.alternative_value;
      }

      if (fold.kind == ir::SsaConstantKind::literal)
        here = value & ir::LowMask(node.width);
      else if (fold.kind == ir::SsaConstantKind::image_location && graph.load_bias())
        here = (*graph.load_bias() + value) & ir::LowMask(node.width);
      continue;
    }

    const auto* descriptor = ir::Descriptor(node.op);
    if (!descriptor || descriptor->effect != ir::Effect::pure || !descriptor->produces_value ||
        !descriptor->arity)
      continue;
    if (node.op == ir::Op::select && node.inputs[0] < id && state.nodes[node.inputs[0]]) {
      const auto taken = node.inputs[*state.nodes[node.inputs[0]] ? 1 : 2];
      if (taken < id) here = state.nodes[taken];
      continue;
    }

    std::array<std::uint64_t, 3> operands{};
    bool known = true;
    for (unsigned i = 0; i < descriptor->arity; ++i) {
      if (node.inputs[i] >= id || !state.nodes[node.inputs[i]])
        known = false;
      else
        operands[i] = *state.nodes[node.inputs[i]];
    }

    if (!known) continue;
    here = ir::FoldPure(node, std::span(operands.data(), descriptor->arity),
                        block.nodes[node.inputs[0]].width);
  }
}

std::optional<std::uint32_t> Edge(const ir::SsaBlock& block, const State& state) {
  std::optional<std::uint32_t> taken;
  for (std::uint32_t index = 0; index < block.edges.size(); ++index) {
    const auto& edge = block.edges[index];
    if (!edge.condition || !edge.when || *edge.condition >= state.nodes.size() ||
        !state.nodes[*edge.condition])
      return std::nullopt;
    if ((*state.nodes[*edge.condition] != 0) != *edge.when) continue;
    if (taken) return std::nullopt;
    taken = index;
  }

  return taken;
}

Number Constant(const ir::SsaGraph& graph, const ir::SsaValue& value) {
  if (value.kind != ir::SsaValueKind::node) return std::nullopt;
  const auto* block = graph.Get(value.block);
  if (!block || value.index >= block->nodes.size()) return std::nullopt;
  const auto& node = block->nodes[value.index];
  if (node.op != ir::Op::constant || node.width > 64) return std::nullopt;
  return node.immediate & ir::LowMask(node.width);
}

}  // namespace

SsaBoundedLoopResult ProveSsaBoundedLoops(const ir::SsaGraph& graph,
                                          std::span<const ir::Group> sources, Budget& budget) {
  const auto valid = ir::ValidateSsaWithSources(graph, sources, budget);
  if (valid != ir::SsaDecline::none)
    return Decline(valid == ir::SsaDecline::resource_limit ? SsaBoundedLoopRefusal::resource_limit
                                                           : SsaBoundedLoopRefusal::invalid_graph);
  // One pass for the predecessor map: a loop needs exactly one other
  // predecessor, and its successor needs no predecessor but the loop.
  std::map<std::uint32_t, std::vector<ir::SsaHandle>> predecessors;
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return Decline(SsaBoundedLoopRefusal::resource_limit);
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    for (const auto& edge : graph.Get(*handle)->edges) {
      if (!edge.target_block) continue;
      if (budget.try_consume({1, sizeof(ir::SsaHandle)}) != BudgetDecline::none)
        return Decline(SsaBoundedLoopRefusal::resource_limit);
      auto& into = predecessors[edge.target_block->slot];
      if (std::find(into.begin(), into.end(), *handle) == into.end()) into.push_back(*handle);
    }
  }

  ir::SsaBoundedLoopFacts facts{graph.arena(), graph.revision(), {}};
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return Decline(SsaBoundedLoopRefusal::resource_limit);
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto& block = *graph.Get(*handle);
    if (block.opaque || block.edges.size() != 2) continue;

    // An architectural entry is entered with storage from outside the graph,
    // so what its one other predecessor publishes is not what it runs from.
    if (budget.try_consume({block.phis.size() + block.frame_phis.size(), 0}) != BudgetDecline::none)
      return Decline(SsaBoundedLoopRefusal::resource_limit);
    if (std::any_of(block.phis.begin(), block.phis.end(),
                    [](const ir::SsaPhi& phi) { return phi.external_entry; }) ||
        std::any_of(block.frame_phis.begin(), block.frame_phis.end(),
                    [](const ir::SsaFramePhi& phi) { return phi.external_entry; }))
      continue;
    std::optional<std::uint32_t> self, exit;
    for (std::uint32_t index = 0; index < 2; ++index) {
      const auto& edge = block.edges[index];
      if (edge.kind != ir::SsaEdgeKind::branch || !edge.target_block || !edge.condition ||
          !edge.when || edge.assumptions.unresolved_target) {
        self.reset();
        break;
      }

      (*edge.target_block == *handle ? self : exit) = index;
    }

    if (!self || !exit) continue;
    const auto successor = *block.edges[*exit].target_block;
    const auto& incoming = predecessors[slot];
    if (incoming.size() != 2) continue;
    const auto other = incoming[0] == *handle ? incoming[1] : incoming[0];
    if (other == *handle) continue;
    const auto& after_predecessors = predecessors[successor.slot];
    if (after_predecessors.size() != 1 || after_predecessors[0] != *handle) continue;
    const auto records = Index(block, budget);
    if (!records.usable)
      return Decline(records.charged ? SsaBoundedLoopRefusal::invalid_graph
                                     : SsaBoundedLoopRefusal::resource_limit);
    if (!Modeled(block, records)) continue;
    if (budget.try_consume({block.nodes.size() + block.phis.size() + block.frame_phis.size(),
                            (block.nodes.size() + block.phis.size() + block.frame_phis.size()) *
                                sizeof(Number)}) != BudgetDecline::none)
      return Decline(SsaBoundedLoopRefusal::resource_limit);
    State state;
    state.phis.assign(block.phis.size(), std::nullopt);
    state.frames.assign(block.frame_phis.size(), std::nullopt);
    state.nodes.assign(block.nodes.size(), std::nullopt);
    for (std::size_t index = 0; index < block.phis.size(); ++index)
      if (const auto* value = From(block.phis[index].incoming, other))
        state.phis[index] = Constant(graph, *value);
    for (std::size_t index = 0; index < block.frame_phis.size(); ++index)
      if (const auto* value = From(block.frame_phis[index].incoming, other))
        state.frames[index] = Constant(graph, *value);
    std::uint32_t iterations = 0;
    for (unsigned round = 0; round < ir::kMaxBoundedLoopIterations && iterations == 0; ++round) {
      if (budget.try_consume({block.nodes.size(), 0}) != BudgetDecline::none)
        return Decline(SsaBoundedLoopRefusal::resource_limit);
      Iterate(graph, block, *handle, records, state);
      const auto taken = Edge(block, state);
      if (!taken) break;
      if (*taken == *exit) {
        iterations = round + 1;
        break;
      }

      if (*taken != *self) break;
      State next;
      next.phis.assign(block.phis.size(), std::nullopt);
      next.frames.assign(block.frame_phis.size(), std::nullopt);
      for (std::size_t index = 0; index < block.phis.size(); ++index)
        if (const auto* value = From(block.phis[index].incoming, *handle))
          next.phis[index] = Read(*value, *handle, state);
      for (std::size_t index = 0; index < block.frame_phis.size(); ++index)
        if (const auto* value = From(block.frame_phis[index].incoming, *handle))
          next.frames[index] = Read(*value, *handle, state);
      // Only now: what the next iteration carries is read out of this one.
      next.nodes = std::move(state.nodes);
      state = std::move(next);
    }

    if (iterations == 0) continue;
    ir::SsaBoundedLoopFact fact{*handle, successor, *self, *exit, iterations, {}};
    const auto& after = *graph.Get(successor);
    if (budget.try_consume({after.phis.size() + after.frame_phis.size(),
                            (after.phis.size() + after.frame_phis.size()) *
                                sizeof(ir::SsaBoundedLoopExit)}) != BudgetDecline::none)
      return Decline(SsaBoundedLoopRefusal::resource_limit);
    // A slot wider than a number cannot carry one, and stating a value about
    // it would be a fact every consumer rejects.
    const auto gather = [&](ir::SsaValueKind kind, std::size_t index, unsigned width,
                            const std::vector<ir::SsaPhiInput>& inputs) {
      if (!width || width > 64) return;
      if (const auto* value = From(inputs, *handle))
        if (const auto number = Read(*value, *handle, state))
          if ((*number & ~ir::LowMask(width)) == 0)
            fact.exits.push_back({kind, static_cast<std::uint32_t>(index), *number});
    };

    for (std::size_t index = 0; index < after.phis.size(); ++index)
      gather(ir::SsaValueKind::phi, index, after.phis[index].width, after.phis[index].incoming);
    for (std::size_t index = 0; index < after.frame_phis.size(); ++index)
      gather(ir::SsaValueKind::frame_phi, index,
             after.frame_phis[index].size > 8 ? 0 : after.frame_phis[index].size * 8,
             after.frame_phis[index].incoming);
    if (budget.try_consume({1, sizeof(ir::SsaBoundedLoopFact)}) != BudgetDecline::none)
      return Decline(SsaBoundedLoopRefusal::resource_limit);
    facts.loops.push_back(std::move(fact));
  }

  const auto checked = ir::ValidateSsaBoundedLoopFacts(graph, facts, sources, budget);
  if (checked != ir::SsaDecline::none)
    return Decline(checked == ir::SsaDecline::resource_limit
                       ? SsaBoundedLoopRefusal::resource_limit
                       : SsaBoundedLoopRefusal::invalid_graph);
  return {std::move(facts), SsaBoundedLoopRefusal::none};
}

}  // namespace nyx::analysis
