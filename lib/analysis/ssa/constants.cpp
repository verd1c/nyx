#include "nyx/analysis/ssa/constants.hpp"

#include <algorithm>
#include <array>
#include <limits>

#include "nyx/analysis/ssa/sccp.hpp"
#include "nyx/ir/fold.hpp"

namespace nyx::analysis {
namespace {
enum class Kind { unknown, constant, overdefined };

struct Value {
  Kind kind = Kind::unknown;
  std::uint64_t number = 0;
};

using Values = std::vector<std::vector<Value>>;
using ReadAt = std::vector<std::vector<std::size_t>>;

Value Merge(Value left, Value right) {
  if (left.kind == Kind::overdefined || right.kind == Kind::unknown) return left;
  if (right.kind == Kind::overdefined || left.kind == Kind::unknown) return right;
  return left.number == right.number ? left : Value{Kind::overdefined, 0};
}

bool Advance(Value& value, Value next) {
  const auto joined = Merge(value, next);
  if (joined.kind == value.kind && joined.number == value.number) return false;
  value = joined;
  return true;
}

const ir::SsaConstantLoad* FoldOf(const ir::SsaBlock& block, ir::ValueId id) {
  const auto at = std::lower_bound(
      block.constant_loads.begin(), block.constant_loads.end(), id,
      [](const ir::SsaConstantLoad& fold, ir::ValueId key) { return fold.node < key; });
  return at != block.constant_loads.end() && at->node == id ? &*at : nullptr;
}

const ir::SsaFrameAccess* FrameOf(const ir::SsaBlock& block, ir::ValueId id) {
  const auto at = std::lower_bound(
      block.frame_accesses.begin(), block.frame_accesses.end(), id,
      [](const ir::SsaFrameAccess& access, ir::ValueId key) { return access.node < key; });
  return at != block.frame_accesses.end() && at->node == id ? &*at : nullptr;
}

Value At(const ir::SsaGraph& graph, const ir::SsaValue& value, const Values& phis,
         const Values& nodes, const Values& frames) {
  switch (value.kind) {
    case ir::SsaValueKind::phi:
      return phis[value.block.slot][value.index];
    case ir::SsaValueKind::node:
      return nodes[value.block.slot][value.index];
    // A promoted private slot holds a value like any other, and a constant
    // put in one reaches its use only through here.
    case ir::SsaValueKind::frame_phi:
      return frames[value.block.slot][value.index];
    // A callee's effect is fresh state, but the run may declare what it left.
    case ir::SsaValueKind::clobber:
      if (const auto result = ir::SsaClobberResult(graph, value)) return {Kind::constant, *result};
      return {Kind::overdefined, 0};
    default:
      return {Kind::overdefined, 0};
  }
}

Value Evaluate(const ir::SsaGraph& graph, const ir::SsaBlock& block, ir::ValueId id,
               const Values& phis, const Values& nodes, const Values& frames, const ReadAt& read_at,
               std::size_t slot) {
  const auto& node = block.nodes[id];
  if (node.width > 64) return {Kind::overdefined, 0};
  if (node.op == ir::Op::constant)
    return {Kind::constant, node.immediate & ir::LowMask(node.width)};
  // A location is a number once the run declares where the image sits, so a
  // page base carried round a loop settles like any other constant.
  if (node.op == ir::Op::image_address && graph.load_bias())
    return {Kind::constant, (*graph.load_bias() + node.immediate) & ir::LowMask(node.width)};
  if (node.op == ir::Op::read) {
    return At(graph, block.reads[read_at[slot][id]].value, phis, nodes, frames);
  }

  // Promotion already decided what this load reads; that is its value here,
  // and the load itself stays because taking it out is another pass's to
  // justify.
  if (const auto* frame = FrameOf(block, id); frame && frame->replacement)
    return At(graph, *frame->replacement, phis, nodes, frames);
  // A load the graph already folded yields what its record says. A location
  // is a place, not a number, until the run declares where the image sits;
  // then the two are the same and arithmetic on one folds.
  if (const auto* fold = FoldOf(block, id); fold && fold->skip_access && !fold->condition) {
    if (fold->kind == ir::SsaConstantKind::literal)
      return {Kind::constant, fold->value & ir::LowMask(node.width)};
    if (fold->kind == ir::SsaConstantKind::image_location && graph.load_bias())
      return {Kind::constant, (*graph.load_bias() + fold->value) & ir::LowMask(node.width)};
  }

  const auto* descriptor = ir::Descriptor(node.op);
  if (!descriptor || descriptor->effect != ir::Effect::pure || !descriptor->produces_value ||
      !descriptor->arity)
    return {Kind::overdefined, 0};
  std::array<Value, 3> inputs{};
  for (unsigned i = 0; i < descriptor->arity; ++i) inputs[i] = nodes[slot][node.inputs[i]];
  if (node.op == ir::Op::select) {
    if (inputs[0].kind == Kind::constant) return inputs[inputs[0].number ? 1 : 2];
    if (inputs[1].kind == Kind::constant && inputs[2].kind == Kind::constant &&
        inputs[1].number == inputs[2].number)
      return inputs[1];
  }

  if (std::any_of(inputs.begin(), inputs.begin() + descriptor->arity,
                  [](Value value) { return value.kind == Kind::overdefined; }))
    return {Kind::overdefined, 0};
  if (std::any_of(inputs.begin(), inputs.begin() + descriptor->arity,
                  [](Value value) { return value.kind == Kind::unknown; }))
    return {};
  std::array<std::uint64_t, 3> numbers{};
  for (unsigned i = 0; i < descriptor->arity; ++i) numbers[i] = inputs[i].number;
  const auto folded = ir::FoldPure(node, std::span(numbers.data(), descriptor->arity),
                                   block.nodes[node.inputs[0]].width);
  return folded ? Value{Kind::constant, *folded} : Value{Kind::overdefined, 0};
}

SsaConstantResult Decline(SsaConstantRefusal reason) { return {{}, reason}; }

// Values bounded runs of self-looping predecessors established, by block. The
// lattice cannot reach them: joining the first iteration with the second
// sends every loop-carried value overdefined, so these replace the join
// rather than feeding it. Indexed once, because the fixpoint below visits
// every phi of every block many times.
struct LoopValues {
  std::vector<std::vector<std::optional<std::uint64_t>>> phis, frames;

  std::optional<std::uint64_t> At(std::size_t slot, ir::SsaValueKind kind,
                                  std::size_t index) const {
    if (phis.empty()) return std::nullopt;
    const auto& from = kind == ir::SsaValueKind::phi ? phis[slot] : frames[slot];
    return index < from.size() ? from[index] : std::nullopt;
  }
};

bool IndexLoops(const ir::SsaGraph& graph, const ir::SsaBoundedLoopFacts* loops, LoopValues& into,
                Budget& budget) {
  if (!loops || loops->loops.empty()) return true;
  if (budget.try_consume({2 * graph.slots(),
                          2 * graph.slots() * sizeof(std::vector<std::optional<std::uint64_t>>)}) !=
      BudgetDecline::none)
    return false;
  into.phis.resize(graph.slots());
  into.frames.resize(graph.slots());
  for (const auto& loop : loops->loops) {
    const auto* block = graph.Get(loop.successor);
    if (!block || loop.successor.slot >= graph.slots()) return false;
    if (budget.try_consume({block->phis.size() + block->frame_phis.size() + loop.exits.size(),
                            (block->phis.size() + block->frame_phis.size()) *
                                sizeof(std::optional<std::uint64_t>)}) != BudgetDecline::none)
      return false;
    into.phis[loop.successor.slot].resize(block->phis.size());
    into.frames[loop.successor.slot].resize(block->frame_phis.size());
    for (const auto& exit : loop.exits) {
      auto& row = exit.kind == ir::SsaValueKind::phi ? into.phis[loop.successor.slot]
                                                     : into.frames[loop.successor.slot];
      if (exit.index >= row.size()) return false;
      row[exit.index] = exit.value;
    }
  }

  return true;
}
}  // namespace

SsaConstantResult ProveSsaConstants(const ir::SsaGraph& graph,
                                    const ir::SsaReachabilityFacts& reachable,
                                    std::span<const ir::Group> sources, Budget& budget,
                                    const ir::SsaBoundedLoopFacts* loops) {
  const auto valid = ir::ValidateSsa(graph, budget);
  if (valid != ir::SsaDecline::none)
    return Decline(valid == ir::SsaDecline::resource_limit ? SsaConstantRefusal::resource_limit
                                                           : SsaConstantRefusal::invalid_graph);
  const auto scope = ir::ValidateSsaReachabilityFacts(graph, reachable, sources, budget);
  if (scope != ir::SsaDecline::none)
    return Decline(scope == ir::SsaDecline::resource_limit
                       ? SsaConstantRefusal::resource_limit
                       : SsaConstantRefusal::stale_reachability);
  LoopValues from_loop;
  if (!IndexLoops(graph, loops, from_loop, budget))
    return Decline(SsaConstantRefusal::resource_limit);
  if (graph.slots() > std::numeric_limits<std::size_t>::max() /
                          (2 * sizeof(std::vector<Value>) + sizeof(std::vector<std::size_t>)) ||
      budget.try_consume({graph.slots(), graph.slots() * (2 * sizeof(std::vector<Value>) +
                                                          sizeof(std::vector<std::size_t>))}) !=
          BudgetDecline::none)
    return Decline(SsaConstantRefusal::resource_limit);
  Values phis(graph.slots()), nodes(graph.slots()), frames(graph.slots());
  ReadAt read_at(graph.slots());
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return Decline(SsaConstantRefusal::resource_limit);
    const auto handle = graph.Handle(slot);
    if (!handle || !reachable.reachable[slot]) continue;
    const auto& block = *graph.Get(*handle);
    if (block.phis.size() > std::numeric_limits<std::size_t>::max() / sizeof(Value) ||
        block.nodes.size() > std::numeric_limits<std::size_t>::max() / sizeof(Value) ||
        block.nodes.size() >
            std::numeric_limits<std::size_t>::max() / (sizeof(Value) + sizeof(std::size_t)) ||
        budget.try_consume({block.phis.size() + block.nodes.size() + block.reads.size(),
                            block.phis.size() * sizeof(Value) +
                                block.nodes.size() * (sizeof(Value) + sizeof(std::size_t))}) !=
            BudgetDecline::none)
      return Decline(SsaConstantRefusal::resource_limit);
    if (block.frame_phis.size() > std::numeric_limits<std::size_t>::max() / sizeof(Value) ||
        budget.try_consume({block.frame_phis.size(), block.frame_phis.size() * sizeof(Value)}) !=
            BudgetDecline::none)
      return Decline(SsaConstantRefusal::resource_limit);
    phis[slot].resize(block.phis.size());
    nodes[slot].resize(block.nodes.size());
    frames[slot].resize(block.frame_phis.size());
    read_at[slot].resize(block.nodes.size(), SIZE_MAX);
    for (std::size_t index = 0; index < block.reads.size(); ++index)
      read_at[slot][block.reads[index].node] = index;
  }

  for (unsigned phase = 0; phase < 2; ++phase) {
    bool changed = true;
    while (changed) {
      changed = false;
      for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
        if (budget.try_consume({1, 0}) != BudgetDecline::none)
          return Decline(SsaConstantRefusal::resource_limit);
        const auto handle = graph.Handle(slot);
        if (!handle || !reachable.reachable[slot]) continue;
        const auto& block = *graph.Get(*handle);
        for (std::size_t index = 0; index < block.phis.size(); ++index) {
          const auto& phi = block.phis[index];
          if (budget.try_consume({1 + phi.incoming.size(), 0}) != BudgetDecline::none)
            return Decline(SsaConstantRefusal::resource_limit);
          Value value = phi.external_entry ? Value{Kind::overdefined, 0} : Value{};

          // A run supplies what the join cannot reach. Whether the two ever
          // contradict each other is checked once, after this settles, by
          // ValidateSsaConstantFacts, since a value part-way through a fixpoint
          // is not final.
          if (const auto run = phi.external_entry
                                   ? std::nullopt
                                   : from_loop.At(slot, ir::SsaValueKind::phi, index))
            value = {Kind::constant, *run};
          else
            for (const auto& input : phi.incoming)
              if (reachable.reachable[input.predecessor.slot])
                value = Merge(value, At(graph, input.value, phis, nodes, frames));
          changed |= Advance(phis[slot][index], value);
        }

        for (std::size_t index = 0; index < block.frame_phis.size(); ++index) {
          const auto& phi = block.frame_phis[index];
          if (budget.try_consume({1 + phi.incoming.size(), 0}) != BudgetDecline::none)
            return Decline(SsaConstantRefusal::resource_limit);
          Value value = phi.external_entry ? Value{Kind::overdefined, 0} : Value{};
          if (const auto run = phi.external_entry
                                   ? std::nullopt
                                   : from_loop.At(slot, ir::SsaValueKind::frame_phi, index))
            value = {Kind::constant, *run};
          else
            for (const auto& input : phi.incoming)
              if (reachable.reachable[input.predecessor.slot])
                value = Merge(value, At(graph, input.value, phis, nodes, frames));
          changed |= Advance(frames[slot][index], value);
        }

        for (ir::ValueId id = 0; id < block.nodes.size(); ++id) {
          const auto& node = block.nodes[id];
          const auto work = 1 + 4 * ir::Descriptor(node.op)->arity;
          if (budget.try_consume({work, 0}) != BudgetDecline::none)
            return Decline(SsaConstantRefusal::resource_limit);
          changed |= Advance(nodes[slot][id],
                             Evaluate(graph, block, id, phis, nodes, frames, read_at, slot));
        }
      }
    }

    if (phase == 0)
      for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
        if (budget.try_consume({1, 0}) != BudgetDecline::none)
          return Decline(SsaConstantRefusal::resource_limit);
        for (auto& value : phis[slot])
          if (value.kind == Kind::unknown) value.kind = Kind::overdefined;
        for (auto& value : frames[slot])
          if (value.kind == Kind::unknown) value.kind = Kind::overdefined;
        for (auto& value : nodes[slot])
          if (value.kind == Kind::unknown) value.kind = Kind::overdefined;
      }
  }

  ir::SsaConstantFacts facts{graph.arena(), graph.revision(), {}};
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return Decline(SsaConstantRefusal::resource_limit);
    const auto handle = graph.Handle(slot);
    if (!handle || !reachable.reachable[slot]) continue;
    for (std::size_t index = 0; index < phis[slot].size(); ++index) {
      if (budget.try_consume({1, 0}) != BudgetDecline::none)
        return Decline(SsaConstantRefusal::resource_limit);
      if (phis[slot][index].kind != Kind::constant) continue;
      if (budget.try_consume({1, sizeof(ir::SsaConstantValue)}) != BudgetDecline::none)
        return Decline(SsaConstantRefusal::resource_limit);
      facts.values.push_back({*handle, ir::SsaValueKind::phi, static_cast<std::uint32_t>(index),
                              phis[slot][index].number});
    }

    for (std::size_t index = 0; index < frames[slot].size(); ++index) {
      if (budget.try_consume({1, 0}) != BudgetDecline::none)
        return Decline(SsaConstantRefusal::resource_limit);
      if (frames[slot][index].kind != Kind::constant) continue;
      if (budget.try_consume({1, sizeof(ir::SsaConstantValue)}) != BudgetDecline::none)
        return Decline(SsaConstantRefusal::resource_limit);
      facts.values.push_back({*handle, ir::SsaValueKind::frame_phi,
                              static_cast<std::uint32_t>(index), frames[slot][index].number});
    }

    for (std::size_t index = 0; index < nodes[slot].size(); ++index) {
      if (budget.try_consume({1, 0}) != BudgetDecline::none)
        return Decline(SsaConstantRefusal::resource_limit);
      if (nodes[slot][index].kind != Kind::constant) continue;
      if (budget.try_consume({1, sizeof(ir::SsaConstantValue)}) != BudgetDecline::none)
        return Decline(SsaConstantRefusal::resource_limit);
      facts.values.push_back({*handle, ir::SsaValueKind::node, static_cast<std::uint32_t>(index),
                              nodes[slot][index].number});
    }
  }

  const auto checked =
      ir::ValidateSsaConstantFacts(graph, reachable, facts, sources, budget, loops);
  if (checked != ir::SsaDecline::none)
    return Decline(checked == ir::SsaDecline::resource_limit ? SsaConstantRefusal::resource_limit
                                                             : SsaConstantRefusal::invalid_graph);
  return {std::move(facts), SsaConstantRefusal::none};
}

SsaSccpResult ProveSsaSccp(const ir::SsaGraph& graph, ir::SsaEntryScope scope,
                           std::span<const ir::Group> sources, Budget& budget,
                           const ir::SsaBoundedLoopFacts* loops) {
  const auto decline = [](SsaSccpRefusal reason) { return SsaSccpResult{{}, reason}; };
  const auto valid = ir::ValidateSsa(graph, budget);
  if (valid != ir::SsaDecline::none)
    return decline(valid == ir::SsaDecline::resource_limit ? SsaSccpRefusal::resource_limit
                                                           : SsaSccpRefusal::invalid_graph);
  if (scope != ir::SsaEntryScope::closed_population) return decline(SsaSccpRefusal::open_entries);
  if (graph.entries().empty()) return decline(SsaSccpRefusal::invalid_graph);
  const auto count = graph.slots();
  if (count > SIZE_MAX / (3 * sizeof(std::vector<Value>) + sizeof(std::vector<std::size_t>) + 1) ||
      budget.try_consume({count, count * (3 * sizeof(std::vector<Value>) +
                                          sizeof(std::vector<std::size_t>) + 1)}) !=
          BudgetDecline::none)
    return decline(SsaSccpRefusal::resource_limit);
  Values phis(count), nodes(count), frames(count);
  ReadAt read_at(count);
  LoopValues from_loop;
  if (!IndexLoops(graph, loops, from_loop, budget)) return decline(SsaSccpRefusal::resource_limit);
  ir::SsaSccpFacts facts{graph.arena(),
                         graph.revision(),
                         scope,
                         std::vector<std::uint8_t>(count),
                         std::vector<std::vector<std::uint8_t>>(count),
                         {graph.arena(), graph.revision(), {}}};
  for (const auto entry : graph.entries()) facts.executable[entry.slot] = 1;
  for (std::size_t slot = 0; slot < count; ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return decline(SsaSccpRefusal::resource_limit);
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto& block = *graph.Get(*handle);
    if (block.phis.size() > SIZE_MAX - block.nodes.size() ||
        block.phis.size() + block.nodes.size() > SIZE_MAX - block.reads.size() ||
        block.phis.size() + block.nodes.size() + block.reads.size() > SIZE_MAX - block.edges.size())
      return decline(SsaSccpRefusal::resource_limit);
    if (block.phis.size() > SIZE_MAX / sizeof(Value) ||
        block.nodes.size() > SIZE_MAX / (sizeof(Value) + sizeof(std::size_t)) ||
        block.phis.size() * sizeof(Value) >
            SIZE_MAX - block.nodes.size() * (sizeof(Value) + sizeof(std::size_t)) ||
        block.phis.size() * sizeof(Value) +
                block.nodes.size() * (sizeof(Value) + sizeof(std::size_t)) >
            SIZE_MAX - block.edges.size() ||
        budget.try_consume(
            {block.phis.size() + block.nodes.size() + block.reads.size() + block.edges.size(),
             block.phis.size() * sizeof(Value) +
                 block.nodes.size() * (sizeof(Value) + sizeof(std::size_t)) +
                 block.edges.size()}) != BudgetDecline::none)
      return decline(SsaSccpRefusal::resource_limit);
    if (block.frame_phis.size() > SIZE_MAX / sizeof(Value) ||
        budget.try_consume({block.frame_phis.size(), block.frame_phis.size() * sizeof(Value)}) !=
            BudgetDecline::none)
      return decline(SsaSccpRefusal::resource_limit);
    phis[slot].resize(block.phis.size());
    nodes[slot].resize(block.nodes.size());
    frames[slot].resize(block.frame_phis.size());
    read_at[slot].resize(block.nodes.size(), SIZE_MAX);
    facts.edges[slot].resize(block.edges.size());
    for (std::size_t index = 0; index < block.reads.size(); ++index)
      read_at[slot][block.reads[index].node] = index;
  }

  for (;;) {
    bool changed = true;
    while (changed) {
      changed = false;
      for (std::size_t slot = 0; slot < count; ++slot) {
        if (budget.try_consume({1, 0}) != BudgetDecline::none)
          return decline(SsaSccpRefusal::resource_limit);
        const auto handle = graph.Handle(slot);
        if (!handle || !facts.executable[slot]) continue;
        const auto& block = *graph.Get(*handle);
        for (std::size_t index = 0; index < block.phis.size(); ++index) {
          const auto& phi = block.phis[index];
          if (budget.try_consume({1 + phi.incoming.size(), 0}) != BudgetDecline::none)
            return decline(SsaSccpRefusal::resource_limit);
          Value value = phi.external_entry ? Value{Kind::overdefined, 0} : Value{};
          for (const auto& input : phi.incoming) {
            if (!facts.executable[input.predecessor.slot]) continue;
            const auto& predecessor = *graph.Get(input.predecessor);
            if (budget.try_consume({predecessor.edges.size(), 0}) != BudgetDecline::none)
              return decline(SsaSccpRefusal::resource_limit);
            bool selected = false;
            for (std::size_t edge = 0; edge < predecessor.edges.size(); ++edge)
              if (facts.edges[input.predecessor.slot][edge] &&
                  predecessor.edges[edge].target_block == *handle)
                selected = true;
            if (selected) value = Merge(value, At(graph, input.value, phis, nodes, frames));
          }

          if (const auto run = phi.external_entry
                                   ? std::nullopt
                                   : from_loop.At(slot, ir::SsaValueKind::phi, index))
            value = {Kind::constant, *run};
          changed |= Advance(phis[slot][index], value);
        }

        // A promoted private slot merges over the same selected edges: which
        // arm of a branch is live decides what reaches a frame slot exactly
        // as it decides what reaches a register.
        for (std::size_t index = 0; index < block.frame_phis.size(); ++index) {
          const auto& phi = block.frame_phis[index];
          if (budget.try_consume({1 + phi.incoming.size(), 0}) != BudgetDecline::none)
            return decline(SsaSccpRefusal::resource_limit);
          Value value = phi.external_entry ? Value{Kind::overdefined, 0} : Value{};
          for (const auto& input : phi.incoming) {
            if (!facts.executable[input.predecessor.slot]) continue;
            const auto& predecessor = *graph.Get(input.predecessor);
            if (budget.try_consume({predecessor.edges.size(), 0}) != BudgetDecline::none)
              return decline(SsaSccpRefusal::resource_limit);
            bool selected = false;
            for (std::size_t edge = 0; edge < predecessor.edges.size(); ++edge)
              if (facts.edges[input.predecessor.slot][edge] &&
                  predecessor.edges[edge].target_block == *handle)
                selected = true;
            if (selected) value = Merge(value, At(graph, input.value, phis, nodes, frames));
          }

          if (const auto run = phi.external_entry
                                   ? std::nullopt
                                   : from_loop.At(slot, ir::SsaValueKind::frame_phi, index))
            value = {Kind::constant, *run};
          changed |= Advance(frames[slot][index], value);
        }

        for (ir::ValueId id = 0; id < block.nodes.size(); ++id) {
          const auto* descriptor = ir::Descriptor(block.nodes[id].op);
          if (budget.try_consume({1 + 4 * descriptor->arity, 0}) != BudgetDecline::none)
            return decline(SsaSccpRefusal::resource_limit);
          changed |= Advance(nodes[slot][id],
                             Evaluate(graph, block, id, phis, nodes, frames, read_at, slot));
        }

        for (std::size_t edge = 0; edge < block.edges.size(); ++edge) {
          if (budget.try_consume({1, 0}) != BudgetDecline::none)
            return decline(SsaSccpRefusal::resource_limit);
          const auto& successor = block.edges[edge];
          if (successor.condition) {
            const auto condition = nodes[slot][*successor.condition];
            if (condition.kind == Kind::unknown ||
                (condition.kind == Kind::constant &&
                 static_cast<bool>(condition.number & 1) != *successor.when))
              continue;
          }

          if (!facts.edges[slot][edge]) {
            facts.edges[slot][edge] = 1;
            changed = true;
          }

          if (successor.target_block && !facts.executable[successor.target_block->slot]) {
            facts.executable[successor.target_block->slot] = 1;
            changed = true;
          }
        }
      }
    }

    bool forced = false;
    for (std::size_t slot = 0; slot < count; ++slot) {
      if (budget.try_consume({1, 0}) != BudgetDecline::none)
        return decline(SsaSccpRefusal::resource_limit);
      if (!facts.executable[slot]) continue;
      if (budget.try_consume({phis[slot].size() + nodes[slot].size() + frames[slot].size(), 0}) !=
          BudgetDecline::none)
        return decline(SsaSccpRefusal::resource_limit);
      for (auto& value : phis[slot])
        if (value.kind == Kind::unknown) {
          value.kind = Kind::overdefined;
          forced = true;
        }
      for (auto& value : frames[slot])
        if (value.kind == Kind::unknown) {
          value.kind = Kind::overdefined;
          forced = true;
        }
      for (auto& value : nodes[slot])
        if (value.kind == Kind::unknown) {
          value.kind = Kind::overdefined;
          forced = true;
        }
    }

    if (!forced) break;
  }

  for (std::size_t slot = 0; slot < count; ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return decline(SsaSccpRefusal::resource_limit);
    const auto handle = graph.Handle(slot);
    if (!handle || !facts.executable[slot]) continue;
    const auto append = [&](ir::SsaValueKind kind, std::size_t index, Value value) {
      if (value.kind != Kind::constant) return true;
      if (budget.try_consume({1, sizeof(ir::SsaConstantValue)}) != BudgetDecline::none)
        return false;
      facts.constants.values.push_back(
          {*handle, kind, static_cast<std::uint32_t>(index), value.number});
      return true;
    };

    for (std::size_t index = 0; index < phis[slot].size(); ++index)
      if (!append(ir::SsaValueKind::phi, index, phis[slot][index]))
        return decline(SsaSccpRefusal::resource_limit);
    for (std::size_t index = 0; index < frames[slot].size(); ++index)
      if (!append(ir::SsaValueKind::frame_phi, index, frames[slot][index]))
        return decline(SsaSccpRefusal::resource_limit);
    for (std::size_t index = 0; index < nodes[slot].size(); ++index)
      if (!append(ir::SsaValueKind::node, index, nodes[slot][index]))
        return decline(SsaSccpRefusal::resource_limit);
  }

  // Completeness is asked only of what runs, and only once the lattice has
  // settled: a block whose guard folds away never executes, and what its
  // unresolved jump might reach cannot enter the population.
  for (std::size_t slot = 0; slot < count; ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return decline(SsaSccpRefusal::resource_limit);
    const auto handle = graph.Handle(slot);
    if (!handle || !facts.executable[slot]) continue;
    const auto& block = *graph.Get(*handle);
    const auto complete = ir::ValidateSsaDirectSuccessors(block, budget);
    if (complete != ir::SsaDecline::none)
      return decline(complete == ir::SsaDecline::resource_limit
                         ? SsaSccpRefusal::resource_limit
                         : SsaSccpRefusal::incomplete_successors);
    const auto bound = ir::ValidateSsaSourceBinding(block, sources, budget);
    if (bound != ir::SsaDecline::none)
      return decline(bound == ir::SsaDecline::resource_limit ? SsaSccpRefusal::resource_limit
                                                             : SsaSccpRefusal::invalid_graph);
    // Complete successors can still carry an edge a closed population does
    // not admit, such as an unresolved one beside them; that is the same
    // refusal, not a malformed graph.
    const auto member = ir::ValidateSsaClosedMember(block, sources, budget);
    if (member != ir::SsaDecline::none)
      return decline(member == ir::SsaDecline::resource_limit
                         ? SsaSccpRefusal::resource_limit
                         : SsaSccpRefusal::incomplete_successors);
  }

  const auto checked = ir::ValidateSsaSccpFacts(graph, facts, sources, budget, loops);
  if (checked != ir::SsaDecline::none)
    return decline(checked == ir::SsaDecline::resource_limit ? SsaSccpRefusal::resource_limit
                                                             : SsaSccpRefusal::invalid_graph);
  return {std::move(facts), SsaSccpRefusal::none};
}

}  // namespace nyx::analysis
