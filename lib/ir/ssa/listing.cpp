#include "nyx/ir/ssa/listing.hpp"

#include <algorithm>
#include <charconv>

#include "nyx/ir/fold.hpp"

namespace nyx::ir {
namespace {

enum class Role : std::uint8_t {
  hidden,  // dead, retired or omitted: nothing to show
  value,   // a value without an effect of its own
  effect,  // an access that still executes
};

struct BlockState {
  std::vector<std::uint8_t> live_out;  // per phi index
  std::vector<std::uint8_t> live_in;
  std::vector<std::uint8_t> frame_live_out;  // per frame phi index
  std::vector<std::uint8_t> frame_live_in;
  std::vector<std::uint8_t> frame_reads;  // slots a shown value reads at entry
  std::vector<std::uint8_t> marked;       // per node
  std::vector<std::uint32_t> uses;        // per node, among marked consumers
  std::vector<std::uint32_t> weight;      // per node, operators its inlined text spells
  std::vector<std::uint8_t> split;        // per node, named because inlining it reads badly
};

class Lister {
 public:
  Lister(const SsaGraph& graph, const SsaListingTarget& target, Budget& budget)
      : graph_(graph), target_(target), budget_(budget) {}

  SsaListingResult Run() {
    const auto valid = ValidateSsa(graph_, budget_);
    if (valid != SsaDecline::none) return {{}, valid};
    if (!Order() || !Liveness() || !Print()) return {{}, SsaDecline::resource_limit};
    return {std::move(out_), SsaDecline::none};
  }

 private:
  bool Charge(std::uint64_t work, std::uint64_t bytes = 0) const {
    if (exhausted_ || budget_.try_consume({work, bytes}) != BudgetDecline::none) {
      exhausted_ = true;
      return false;
    }

    return true;
  }

  const SsaBlock& At(std::size_t slot) const { return *graph_.Get(*graph_.Handle(slot)); }

  // Reverse postorder from the entries; unreachable blocks follow by slot.
  // An edge whose condition is a literal that never selects it.
  static bool Dead(const SsaBlock& block, const SsaEdge& edge) {
    return edge.condition && edge.when && *edge.condition < block.nodes.size() &&
           block.nodes[*edge.condition].op == Op::constant &&
           (block.nodes[*edge.condition].immediate != 0) != *edge.when;
  }

  bool Order() {
    const auto count = graph_.slots();
    if (!Charge(count, count * (sizeof(BlockState) + 2))) return false;
    states_.resize(count);
    std::vector<std::uint8_t> seen(count);
    std::vector<std::pair<std::size_t, std::size_t>> stack;
    std::vector<std::size_t> post;
    for (const auto entry : graph_.entries()) {
      if (seen[entry.slot]) continue;
      seen[entry.slot] = 1;
      stack.push_back({entry.slot, 0});
      while (!stack.empty()) {
        if (!Charge(1)) return false;
        auto& [slot, next] = stack.back();
        const auto& edges = At(slot).edges;
        if (next < edges.size()) {
          const auto& edge = edges[next++];
          if (!Dead(At(slot), edge) && edge.target_block && graph_.Get(*edge.target_block) &&
              !seen[edge.target_block->slot]) {
            seen[edge.target_block->slot] = 1;
            stack.push_back({edge.target_block->slot, 0});
          }

          continue;
        }

        post.push_back(slot);
        stack.pop_back();
      }
    }

    order_.assign(post.rbegin(), post.rend());

    // What no live edge reaches from an entry never runs, so it is left out.
    for (std::size_t slot = 0; slot < count; ++slot) omitted_ += !seen[slot] && graph_.Handle(slot);
    for (const auto slot : order_) {
      const auto& block = At(slot);
      auto& state = states_[slot];
      if (!Charge(block.phis.size() + block.nodes.size(),
                  2 * block.phis.size() + block.nodes.size() * (1 + sizeof(std::uint32_t))))
        return false;
      state.live_out.assign(block.phis.size(), 0);
      state.live_in.assign(block.phis.size(), 0);
      state.frame_live_out.assign(block.frame_phis.size(), 0);
      state.frame_live_in.assign(block.frame_phis.size(), 0);
      state.frame_reads.assign(block.frame_phis.size(), 0);
      state.marked.assign(block.nodes.size(), 0);
      state.uses.assign(block.nodes.size(), 0);
      state.weight.assign(block.nodes.size(), 0);
      state.split.assign(block.nodes.size(), 0);
    }

    return true;
  }

  static bool Contains(std::span<const StorageId> set, StorageId storage) {
    return std::binary_search(set.begin(), set.end(), storage);
  }

  Role RoleOf(const SsaBlock& block, ValueId id) const {
    const auto& node = block.nodes[id];
    if (std::binary_search(block.dead_pure_nodes.begin(), block.dead_pure_nodes.end(), id) ||
        node.op == Op::write)
      return Role::hidden;
    if (std::any_of(block.store_omissions.begin(), block.store_omissions.end(),
                    [&](const StoreOmission& omission) { return omission.store == id; }) ||
        std::any_of(block.paired_load_omissions.begin(), block.paired_load_omissions.end(),
                    [&](const PairedLoadOmission& pair) {
                      return pair.loads[0] == id || pair.loads[1] == id;
                    }))
      return Role::hidden;
    if (std::binary_search(block.disabled_effects.begin(), block.disabled_effects.end(), id))
      return node.op == Op::store ? Role::hidden : Role::value;
    return HasMemoryOrMonitorEffect(node.op) ? Role::effect : Role::value;
  }

  const SsaRead* ReadOf(const SsaBlock& block, ValueId id) const {
    const auto found = std::find_if(block.reads.begin(), block.reads.end(),
                                    [&](const SsaRead& read) { return read.node == id; });
    return found == block.reads.end() ? nullptr : &*found;
  }

  const SsaConstantLoad* FoldOf(const SsaBlock& block, ValueId id) const {
    const auto found = std::lower_bound(
        block.constant_loads.begin(), block.constant_loads.end(), id,
        [](const SsaConstantLoad& fold, ValueId value) { return fold.node < value; });
    return found != block.constant_loads.end() && found->node == id ? &*found : nullptr;
  }

  const SsaFrameAccess* FrameOf(const SsaBlock& block, ValueId id) const {
    const auto found = std::lower_bound(
        block.frame_accesses.begin(), block.frame_accesses.end(), id,
        [](const SsaFrameAccess& access, ValueId value) { return access.node < value; });
    return found != block.frame_accesses.end() && found->node == id ? &*found : nullptr;
  }

  // The node ids a value consumes, as the listing renders it.
  void Inputs(const SsaBlock& block, ValueId id, std::vector<ValueId>& inputs) const {
    inputs.clear();
    const auto& node = block.nodes[id];
    const auto here = [&](const SsaValue& value) {
      if (value.kind == SsaValueKind::node && value.index < block.nodes.size() &&
          graph_.Get(value.block) == &block)
        inputs.push_back(value.index);
    };

    if (node.op == Op::read) {
      if (const auto* read = ReadOf(block, id)) here(read->value);
      return;
    }

    if (std::binary_search(block.disabled_effects.begin(), block.disabled_effects.end(), id)) {
      if (const auto* frame = FrameOf(block, id); frame && frame->replacement)
        here(*frame->replacement);
      if (const auto* fold = FoldOf(block, id)) {
        if (fold->condition) inputs.push_back(*fold->condition);
        if (fold->kind == SsaConstantKind::bounded_table) inputs.push_back(fold->table_index);
      }

      return;
    }

    const auto* descriptor = Descriptor(node.op);
    for (unsigned input = 0; descriptor && input < descriptor->arity; ++input)
      if (node.inputs[input] < block.nodes.size()) inputs.push_back(node.inputs[input]);
  }

  // A call the graph resolved to one location, which it now makes directly.
  static const ConditionalRewrite* Resolved(const SsaBlock& block) {
    for (const auto& rewrite : block.control_rewrites)
      if (rewrite.rule == RewriteRule::resolved_call) return &rewrite;
    return nullptr;
  }

  static bool Calls(const SsaBlock& block) {
    return std::any_of(block.edges.begin(), block.edges.end(),
                       [](const SsaEdge& edge) { return edge.kind == SsaEdgeKind::callee; });
  }

  // The register a block's call reads its target from, which the call line
  // already names. Only that register: another holding the same value is
  // still something the callee is handed. The decoded instruction says which
  // it is even once folding has replaced the read with what it read.
  std::optional<StorageId> CallRegister(const SsaBlock& block) const {
    if (block.boundaries.empty() || !block.boundaries.back().transfer ||
        block.source_groups.empty())
      return std::nullopt;
    const auto address = block.source_groups.back();
    if (!Charge(target_.sources.size())) return std::nullopt;
    for (const auto& group : target_.sources) {
      if (group.source_address() != address) continue;
      const auto& transfer = group.transfer();
      if (!transfer || transfer->target >= group.nodes().size()) return std::nullopt;
      const auto& read = group.nodes()[transfer->target];
      return read.op == Op::read ? std::optional(read.storage) : std::nullopt;
    }

    const auto target = block.boundaries.back().transfer->target;
    if (target >= block.nodes.size() || block.nodes[target].op != Op::read) return std::nullopt;
    return block.nodes[target].storage;
  }

  // What each register holds when a block's call is made, per phi index, and
  // which the call instruction writes itself (its link). The exits cannot say:
  // for a register the call leaves fresh they name what comes back.
  void BeforeCall(std::size_t slot, std::vector<SsaValue>& held,
                  std::vector<std::uint8_t>& linked) const {
    const auto& block = At(slot);
    const auto self = *graph_.Handle(slot);
    held.clear();
    linked.assign(block.phis.size(), 0);
    for (std::uint32_t i = 0; i < block.phis.size(); ++i)
      held.push_back({SsaValueKind::phi, self, i});
    const auto place = [&](StorageId storage) -> std::optional<std::size_t> {
      const auto found =
          std::lower_bound(block.phis.begin(), block.phis.end(), storage,
                           [](const SsaPhi& phi, StorageId key) { return phi.storage < key; });
      if (found == block.phis.end() || found->storage != storage) return std::nullopt;
      return static_cast<std::size_t>(found - block.phis.begin());
    };

    for (std::size_t at = 0; at < block.boundaries.size(); ++at) {
      const auto& boundary = block.boundaries[at];
      const bool last = at + 1 == block.boundaries.size();
      for (auto id = boundary.first_node; id < boundary.first_node + boundary.node_count; ++id) {
        const auto& node = block.nodes[id];
        if (node.op != Op::write) continue;
        if (const auto index = place(node.storage)) {
          held[*index] = {SsaValueKind::node, self, node.inputs[0]};
          linked[*index] = last;
        }
      }

      for (const auto& write : boundary.writes)
        if (const auto index = place(write.storage)) {
          held[*index] = {SsaValueKind::node, self, write.value};
          linked[*index] = last;
        }
    }
  }

  bool Exported(const SsaBlock& block, std::size_t index) const {
    const auto& exit = block.exits[index].value;
    return exit.kind == SsaValueKind::node && graph_.Get(exit.block) == &block &&
           exit.index < block.nodes.size();
  }

  // Marks what a block shows given what is live when it ends, and reports
  // which storage it reads from its entry.
  bool Mark(std::size_t slot, std::vector<std::uint8_t>& reads_entry) {
    const auto& block = At(slot);
    auto& state = states_[slot];
    std::fill(state.marked.begin(), state.marked.end(), 0);
    std::fill(state.uses.begin(), state.uses.end(), 0);
    reads_entry.assign(block.phis.size(), 0);
    std::vector<ValueId> stack, inputs;
    const auto root = [&](ValueId id) {
      if (id >= block.nodes.size()) return;

      // A copy reads as its source, so each of its uses is one of the source.
      if (const auto source = SsaCopySource(block.nodes, id); source != id) {
        state.marked[id] = 1;
        id = source;
      }

      ++state.uses[id];
      if (!state.marked[id]) {
        state.marked[id] = 1;
        stack.push_back(id);
      }
    };

    for (ValueId id = 0; id < block.nodes.size(); ++id)
      if (RoleOf(block, id) == Role::effect) root(id);
    for (const auto& edge : block.edges) {
      if (edge.condition &&
          !(&edge != &block.edges.front() && block.edges.front().condition == edge.condition))
        root(*edge.condition);
      const bool computed_call =
          edge.kind == SsaEdgeKind::callee && edge.target_kind == SsaTargetKind::unknown;
      if (computed_call && !block.boundaries.empty() && block.boundaries.back().transfer &&
          !Resolved(block))
        root(block.boundaries.back().transfer->target);
    }

    for (std::size_t index = 0; index < block.exits.size() && index < block.phis.size(); ++index)
      if (state.live_out[index] && Exported(block, index)) root(block.exits[index].value.index);
    // A call observes what the convention lets it at the moment it is made:
    // a value this block wrote is shown, and one it did not is live at entry.
    if (Calls(block)) {
      std::vector<SsaValue> held;
      std::vector<std::uint8_t> linked;
      BeforeCall(slot, held, linked);
      const auto target = CallRegister(block);
      for (std::size_t index = 0; index < held.size(); ++index) {
        const auto storage = block.phis[index].storage;
        if (!Contains(target_.observable_at_call, storage) || linked[index] || storage == target)
          continue;
        if (held[index].kind != SsaValueKind::node) reads_entry[index] = 1;

        // A value kept across the call is already rooted as its exit.
        else if (!(state.live_out[index] && block.exits[index].value == held[index]))
          root(held[index].index);
      }
    }

    for (std::size_t index = 0;
         index < block.frame_exits.size() && index < state.frame_live_out.size(); ++index) {
      const auto& exit = block.frame_exits[index];
      if (state.frame_live_out[index] && exit.kind == SsaValueKind::node &&
          graph_.Get(exit.block) == &block)
        root(exit.index);
    }

    std::fill(state.frame_reads.begin(), state.frame_reads.end(), 0);
    while (!stack.empty()) {
      if (!Charge(1)) return false;
      const auto id = stack.back();
      stack.pop_back();
      if (const auto* frame = FrameOf(block, id);
          frame && frame->replacement && frame->replacement->kind == SsaValueKind::frame_phi &&
          graph_.Get(frame->replacement->block) == &block &&
          frame->replacement->index < state.frame_reads.size())
        state.frame_reads[frame->replacement->index] = 1;
      if (block.nodes[id].op == Op::read)
        if (const auto* read = ReadOf(block, id);
            read && read->phi < reads_entry.size() &&
            !(read->value.kind == SsaValueKind::node && graph_.Get(read->value.block) == &block))
          reads_entry[read->phi] = 1;
      Inputs(block, id, inputs);
      for (const auto input : inputs) root(input);
    }

    return true;
  }

  // Inlining every single-use value turns a chain of rounds into one nested
  // expression. When a value's inlined text would spell more than this many
  // operators, its heaviest inlined input gets a name instead, so a chain
  // breaks along its spine rather than mid-step. Inputs precede their users,
  // so one forward pass sees every input's final weight. Named and Atom read
  // slot_, so this runs only while printing.
  static constexpr std::uint32_t kInlineOperators = 4;

  bool Weigh(const SsaBlock& block, BlockState& state) {
    std::fill(state.weight.begin(), state.weight.end(), 0);
    std::fill(state.split.begin(), state.split.end(), 0);
    std::vector<ValueId> inputs;
    for (ValueId id = 0; id < block.nodes.size(); ++id) {
      if (!state.marked[id]) continue;
      if (!Charge(1)) return false;
      if (Atom(block, id) || SsaCopySource(block.nodes, id) != id) continue;
      std::uint32_t weight = 1;
      ValueId heaviest = id;
      Inputs(block, id, inputs);
      for (auto input : inputs) {
        input = SsaCopySource(block.nodes, input);
        if (input >= block.nodes.size() || Named(block, input)) continue;
        weight += state.weight[input];
        if (heaviest == id || state.weight[input] > state.weight[heaviest]) heaviest = input;
      }

      if (weight > kInlineOperators && heaviest != id && state.weight[heaviest] > 1) {
        state.split[heaviest] = 1;
        weight -= state.weight[heaviest];
      }

      state.weight[id] = weight;
    }

    return true;
  }

  // Storage is live at a block's end when a successor reads it, passes it on,
  // or control leaves where the declared convention lets it be observed.
  bool Liveness() {
    std::vector<std::uint8_t> reads_entry;

    // Monotone: each round only sets bits, so it stops within blocks x storage.
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto at = order_.rbegin(); at != order_.rend(); ++at) {
        const auto slot = *at;
        const auto& block = At(slot);
        auto& state = states_[slot];
        if (!Charge(1 + block.edges.size() * block.phis.size())) return false;
        for (const auto& edge : block.edges) {
          if (Dead(block, edge)) continue;
          for (std::size_t index = 0; index < block.phis.size(); ++index) {
            const auto storage = block.phis[index].storage;
            bool live = false;
            if (edge.target_block && graph_.Get(*edge.target_block)) {
              const auto& successor = states_[edge.target_block->slot];
              live = index < successor.live_in.size() && successor.live_in[index];
            } else if (edge.kind == SsaEdgeKind::return_) {
              live = Contains(target_.observable_at_return, storage);
            } else if (edge.kind == SsaEdgeKind::callee) {
              live = Contains(target_.observable_at_call, storage);
            } else {
              live = true;
            }

            if (live && !state.live_out[index]) {
              state.live_out[index] = 1;
              changed = true;
            }
          }

          // A private slot is dead once the function leaves: neither a callee
          // nor the caller can reach it.
          const bool known =
              (edge.target_block && graph_.Get(*edge.target_block)) ||
              edge.kind == SsaEdgeKind::return_ || edge.kind == SsaEdgeKind::callee ||
              edge.kind == SsaEdgeKind::potential_return || edge.kind == SsaEdgeKind::trap;
          for (std::size_t index = 0; index < state.frame_live_out.size(); ++index) {
            bool live = !known;
            if (edge.target_block && graph_.Get(*edge.target_block)) {
              const auto& successor = states_[edge.target_block->slot];
              live = index < successor.frame_live_in.size() && successor.frame_live_in[index];
            }

            if (live && !state.frame_live_out[index]) {
              state.frame_live_out[index] = 1;
              changed = true;
            }
          }
        }

        if (block.opaque) {
          for (auto& live : state.live_in) live = 1;
          for (auto& live : state.frame_live_in) live = 1;
          continue;
        }

        if (!Mark(slot, reads_entry)) return false;
        for (std::size_t index = 0; index < block.phis.size(); ++index) {
          const auto& exit = index < block.exits.size() ? block.exits[index].value : SsaValue{};
          const bool passes = exit.kind == SsaValueKind::phi && graph_.Get(exit.block) == &block &&
                              exit.index == index;
          const bool live = reads_entry[index] || (passes && state.live_out[index]);
          if (live && !state.live_in[index]) {
            state.live_in[index] = 1;
            changed = true;
          }
        }

        for (std::size_t index = 0; index < state.frame_live_in.size(); ++index) {
          const auto& exit =
              index < block.frame_exits.size() ? block.frame_exits[index] : SsaValue{};
          const bool passes = exit.kind == SsaValueKind::frame_phi &&
                              graph_.Get(exit.block) == &block && exit.index == index;
          const bool live = state.frame_reads[index] || (passes && state.frame_live_out[index]);
          if (live && !state.frame_live_in[index]) {
            state.frame_live_in[index] = 1;
            changed = true;
          }
        }
      }
    }

    return true;
  }

  std::string Name(StorageId storage) const {
    const auto found = std::lower_bound(
        target_.names.begin(), target_.names.end(), storage,
        [](const SsaStorageName& item, StorageId value) { return item.storage < value; });
    if (found != target_.names.end() && found->storage == storage) return std::string(found->name);
    return "s" + std::to_string(storage);
  }

  static std::string Hex(std::uint64_t value) {
    char digits[24];
    const auto end = std::to_chars(digits, digits + sizeof(digits), value, 16).ptr;
    return "0x" + std::string(digits, end);
  }

  static std::string Literal(std::uint64_t value, unsigned width) {
    if (width == 1) return value ? "true" : "false";
    if (value < 10) return std::to_string(value);
    return Hex(value);
  }

  std::string Label(std::size_t slot) const { return "L_" + Hex(At(slot).address).substr(2); }

  std::string ValueText(const SsaBlock& block, const SsaValue& value) const {
    switch (value.kind) {
      case SsaValueKind::phi:
        if (graph_.Get(value.block) && value.index < graph_.Get(value.block)->phis.size())
          return Name(graph_.Get(value.block)->phis[value.index].storage);
        break;
      case SsaValueKind::node:
        if (graph_.Get(value.block) == &block && value.index < block.nodes.size())
          return Operand(block, value.index);
        break;
      case SsaValueKind::frame_phi:
        if (graph_.Get(value.block) && value.index < graph_.Get(value.block)->frame_phis.size())
          return "frame[" +
                 std::to_string(graph_.Get(value.block)->frame_phis[value.index].offset) + "]";
        break;
      default:
        break;
    }

    return "?";
  }

  // Written without parentheses: names, literals, bracketed loads, calls, and
  // casts, folds or zero extensions of those.
  bool Atom(const SsaBlock& block, ValueId id) const {
    const auto& node = block.nodes[id];
    if (Named(block, id)) return true;
    if (std::binary_search(block.disabled_effects.begin(), block.disabled_effects.end(), id)) {
      const auto* fold = FoldOf(block, id);
      return !fold || !fold->condition;
    }

    switch (node.op) {
      case Op::constant:
      case Op::image_address:
      case Op::read:
      case Op::load:
      case Op::exclusive_load:
      case Op::umulh:
      case Op::smulh:
      case Op::clz:
      case Op::rbit:
        return true;
      case Op::zext:
        return Atom(block, node.inputs[0]);
      case Op::extract:
        return !node.immediate && Atom(block, node.inputs[0]);
      default:
        return false;
    }
  }

  // A marked value used more than once, or one with its own effect, gets a
  // name; anything else is written where it is used.
  bool Named(const SsaBlock& block, ValueId id) const {
    const auto op = block.nodes[id].op;

    // A zero extension keeps its operand's value, so it reads as the operand.
    if (op == Op::constant || op == Op::image_address || op == Op::read || op == Op::zext)
      return false;
    const auto& state = states_[slot_];
    return RoleOf(block, id) == Role::effect || state.uses[id] > 1 ||
           (id < state.split.size() && state.split[id]);
  }

  std::string Operand(const SsaBlock& block, ValueId id) const {
    // A copy reads exactly as its source.
    if (const auto source = SsaCopySource(block.nodes, id); source != id)
      return Operand(block, source);
    if (Named(block, id)) return "v" + std::to_string(id);
    const auto text = Expression(block, id);
    return Atom(block, id) ? text : "(" + text + ")";
  }

  // An operand the call's own parentheses already delimit.
  std::string Argument(const SsaBlock& block, ValueId id) const {
    if (const auto source = SsaCopySource(block.nodes, id); source != id)
      return Argument(block, source);
    return Named(block, id) ? "v" + std::to_string(id) : Expression(block, id);
  }

  std::string Expression(const SsaBlock& block, ValueId id) const {
    const auto& node = block.nodes[id];
    const auto in = [&](unsigned input) { return Operand(block, node.inputs[input]); };
    const auto width = std::to_string(node.width);
    if (std::binary_search(block.disabled_effects.begin(), block.disabled_effects.end(), id)) {
      if (const auto* frame = FrameOf(block, id); frame && frame->replacement)
        return ValueText(block, *frame->replacement);
      if (const auto* fold = FoldOf(block, id)) {
        if (fold->kind == SsaConstantKind::bounded_table)
          return "table" + width + "@" + Hex(fold->source_address) + "[" +
                 Operand(block, fold->table_index) + "]";
        const auto shown = [&](std::uint64_t value) {
          return fold->kind == SsaConstantKind::image_location ? "@" + Hex(value)
                                                               : Literal(value, node.width);
        };

        if (fold->condition)
          return Operand(block, *fold->condition) + " ? " + shown(fold->value) + " : " +
                 shown(fold->alternative_value);
        return shown(fold->value);
      }
    }

    switch (node.op) {
      case Op::constant: {
        const auto value = node.immediate & LowMask(node.width);
        return Literal(value, node.width);
      }
      case Op::image_address:
        return "@" + Hex(node.immediate);
      case Op::read: {
        const auto* read = ReadOf(block, id);
        if (read && read->value.kind == SsaValueKind::node &&
            graph_.Get(read->value.block) == &block)
          return ValueText(block, read->value);
        return read && read->phi < block.phis.size() ? Name(block.phis[read->phi].storage)
                                                     : Name(node.storage);
      }
      case Op::add: {
        const auto& right = block.nodes[node.inputs[1]];

        // Two's-complement literals read better as subtraction.
        if (right.op == Op::constant && node.width > 1) {
          const auto value = right.immediate & LowMask(node.width);

          // A record can pin an address's shape, zero offset and all; the
          // value is still just the base.
          if (!value) return in(0);
          if (value >> (node.width - 1))
            return in(0) + " - " + Literal(LowMask(node.width) - value + 1, node.width);
        }

        return in(0) + " + " + in(1);
      }
      case Op::sub:
        return in(0) + " - " + in(1);
      case Op::mul:
        return in(0) + " * " + in(1);
      case Op::bit_and:
        return in(0) + " & " + in(1);
      case Op::bit_or:
        return in(0) + " | " + in(1);
      case Op::bit_xor:
        return in(0) + " ^ " + in(1);
      case Op::bit_not:
        return (node.width == 1 ? "!" : "~") + in(0);
      case Op::shl:
        return in(0) + " << " + in(1);
      case Op::lshr:
        return in(0) + " >> " + in(1);
      case Op::ashr: {
        // (x << k) >>s k is x sign-extended from its low w - k bits.
        const auto& shifted = block.nodes[node.inputs[0]];
        const auto& amount = block.nodes[node.inputs[1]];
        if (shifted.op == Op::shl && !Named(block, node.inputs[0]) && amount.op == Op::constant &&
            shifted.inputs[1] == node.inputs[1] && amount.immediate > 0 &&
            amount.immediate < node.width) {
          const auto bits = node.width - amount.immediate;
          auto source = shifted.inputs[0];

          // A truncation to at least those bits, then a widening, is invisible here.
          for (unsigned step = 0; step < 4; ++step) {
            const auto& cast = block.nodes[source];
            if (Named(block, source) ||
                !(cast.op == Op::zext ||
                  (cast.op == Op::extract && !cast.immediate && cast.width >= bits)))
              break;
            source = cast.inputs[0];
          }

          return "(s" + std::to_string(bits) + ")" + Operand(block, source);
        }

        return in(0) + " >>s " + in(1);
      }
      case Op::extract:
        return "(u" + width + ")" +
               (node.immediate ? "(" + in(0) + " >> " + std::to_string(node.immediate) + ")"
                               : in(0));
      case Op::zext:
        return Operand(block, node.inputs[0]);  // widening keeps the value
      case Op::select:
        return in(0) + " ? " + in(1) + " : " + in(2);
      case Op::equal:
        return in(0) + " == " + in(1);
      case Op::unsigned_less:
        return in(0) + " < " + in(1);
      case Op::signed_less:
        return in(0) + " <s " + in(1);
      case Op::udiv:
        return in(0) + " / " + in(1);
      case Op::sdiv:
        return in(0) + " /s " + in(1);
      case Op::umulh:
      case Op::smulh:
      case Op::clz:
      case Op::rbit: {
        std::string text(Descriptor(node.op)->name);
        text += "(" + Argument(block, node.inputs[0]);
        if (Descriptor(node.op)->arity == 2) text += ", " + Argument(block, node.inputs[1]);
        return text + ")";
      }
      case Op::load:
        return "load" + width + (node.access.byte_order == ByteOrder::big ? "be" : "") + "[" +
               in(0) + "]";
      case Op::exclusive_load:
        return "exclusive_load" + width + "[" + in(0) + "]";
      default:
        break;
    }

    return std::string(Descriptor(node.op) ? Descriptor(node.op)->name : "?");
  }

  bool Line(const std::string& text) {
    if (!Charge(1 + text.size(), text.size() + 1)) return false;
    *sink_ += text;
    *sink_ += '\n';
    return true;
  }

  // The value a register or frame slot must hold at this block, when every
  // way of reaching it computes the same image location. A phi's inputs are
  // followed through predecessors, which is what carries a location formed in
  // one block to the call that uses it in another.
  // The graph answers this, so a target the reading view names is the one a
  // pass would attribute the call to.
  std::optional<std::uint64_t> SettledTarget(const SsaBlock& block, ValueId target) const {
    return SsaSettledTarget(graph_, block, target, budget_);
  }

  std::string Target(const SsaEdge& edge) const {
    if (edge.target_block && graph_.Get(*edge.target_block))
      return Label(forward_.empty() ? edge.target_block->slot : forward_[edge.target_block->slot]);
    if (edge.target_kind == SsaTargetKind::image_location) return "@" + Hex(edge.address);
    if (edge.target_kind == SsaTargetKind::absolute_runtime) return Hex(edge.address);
    return "?";
  }

  bool Control(const SsaBlock& block) {
    const SsaEdge* callee = nullptr;
    const SsaEdge* resume = nullptr;
    const SsaEdge* taken = nullptr;
    const SsaEdge* other = nullptr;
    for (const auto& edge : block.edges) {
      if (edge.kind == SsaEdgeKind::callee)
        callee = &edge;
      else if (edge.kind == SsaEdgeKind::potential_return)
        resume = &edge;
    }

    if (callee) {
      auto text = std::string("    call ");
      if (const auto* resolved = Resolved(block))
        text += "@" + Hex(resolved->when_true);
      else if (callee->target_kind == SsaTargetKind::unknown && !block.boundaries.empty() &&
               block.boundaries.back().transfer) {
        const auto target = block.boundaries.back().transfer->target;

        // The edge stays unresolved, because completeness is decided per block
        // and this value is formed in another. Naming it is still what the
        // call does, and this view exists to say so.
        const auto settled = SettledTarget(block, target);
        const auto operand = Operand(block, target);
        if (!settled)
          text += "[" + operand + "]";
        else if (operand == Hex(*settled))
          text += "@" + Hex(*settled);
        else
          text += "@" + Hex(*settled) + "  ; through " + operand;
      } else
        text += Target(*callee);
      if (!Line(text)) return false;

      // A value this call leaves came from running the callee once, and every
      // constant folded from it rests on that run. The reading view says which
      // ones, or a reader takes an observation for something the image states.
      if (!graph_.call_results().empty()) {
        if (const auto reached = SsaCallTarget(graph_, block, budget_)) {
          std::string observed;
          for (const auto& result : graph_.call_results()) {
            if (result.target != *reached) continue;
            observed +=
                (observed.empty() ? "" : ", ") + Name(result.storage) + "=" + Hex(result.value);
          }

          if (!observed.empty() && !Line("    ; one run left " + observed)) return false;
        }
      }

      return Line(resume                                  ? "    goto " + Target(*resume)
                  : callee->assumptions.declared_noreturn ? "    ; declared not to return"
                                                          : "    ; no continuation");
    }

    for (const auto& edge : block.edges) {
      if (edge.when && *edge.when)
        taken = &edge;
      else if (edge.when)
        other = &edge;
    }

    if (taken && other && block.edges.size() == 2) {
      // A negated condition reads better as the plain one with the arms swapped.
      const auto& condition = block.nodes[*taken->condition];
      if (condition.op == Op::constant)
        return Line("    goto " + Target(condition.immediate ? *taken : *other));
      // Both arms reaching one block decide nothing.
      if (Target(*taken) == Target(*other)) return Line("    goto " + Target(*taken));

      // Negations, and flags compared with zero, read better peeled off with
      // the arms swapped for each.
      auto shown = *taken->condition;
      bool swapped = false;
      for (unsigned step = 0; step < 8 && !Named(block, shown); ++step) {
        const auto& node = block.nodes[shown];
        std::optional<ValueId> inner;
        if (node.op == Op::bit_not) inner = node.inputs[0];
        for (unsigned side = 0; node.op == Op::equal && side < 2 && !inner; ++side) {
          const auto flag = node.inputs[side], zero = node.inputs[1 - side];
          if (block.nodes[zero].op == Op::constant && !block.nodes[zero].immediate &&
              block.nodes[SsaCopySource(block.nodes, flag)].width == 1)
            inner = flag;
        }

        if (!inner) break;
        shown = *inner;
        swapped = !swapped;
      }

      return Line("    if (" + Operand(block, shown) + ") goto " +
                  Target(swapped ? *other : *taken) + " else goto " +
                  Target(swapped ? *taken : *other));
    }

    for (const auto& edge : block.edges) {
      if (Dead(block, edge)) continue;
      std::string text;
      switch (edge.kind) {
        case SsaEdgeKind::return_:
          text = "    return";
          break;
        case SsaEdgeKind::trap:
          text = "    trap";
          break;
        case SsaEdgeKind::opaque_unknown:
          text = "    unknown successor";
          break;
        default:
          text = "    " +
                 (edge.condition ? "if (" + std::string(*edge.when ? "" : "!") +
                                       Operand(block, *edge.condition) + ") "
                                 : std::string()) +
                 "goto " + Target(edge);
      }

      if (!Line(text)) return false;
    }

    return true;
  }

  bool Print() {
    const auto names = [&](std::span<const StorageId> set) {
      std::string text;
      for (const auto storage : set) text += (text.empty() ? "" : " ") + Name(storage);
      return text;
    };

    std::string entries;
    for (const auto entry : graph_.entries()) entries += " " + Label(entry.slot);
    if (!Line("; SSA listing of revision " + std::to_string(graph_.revision()) +
              ": a reading view; the SSA text is the audit record") ||
        !Line("; entries:" + entries) ||
        !Line("; shows effects that still execute and register writes a later block, call or"
              " caller observes") ||
        !Line("; observed at calls: " + names(target_.observable_at_call)) ||
        !Line("; observed at returns: " + names(target_.observable_at_return)) ||
        !Line("; hides dead values and retired or omitted accesses, whose address execution"
              " still checks") ||
        !Line("; comparisons are unsigned unless marked s; @ names an image location") ||
        (omitted_ &&
         !Line("; omits " + std::to_string(omitted_) + (omitted_ == 1 ? " block" : " blocks") +
               " no live edge reaches from an entry")))
      return false;
    // A block that only jumps on reads as its target: every goto to it is
    // retargeted and it is left out. Found by rendering each block once.
    // Rounds repeat while one exposes another, as when both arms of a branch
    // turn out to reach one block.
    std::string body;
    std::vector<std::uint8_t> entry(graph_.slots());
    for (const auto handle : graph_.entries()) entry[handle.slot] = 1;
    forward_.resize(graph_.slots());
    for (std::size_t slot = 0; slot < forward_.size(); ++slot) forward_[slot] = slot;
    for (bool changed = true; changed;) {
      changed = false;
      for (const auto slot : order_) {
        if (forward_[slot] != slot || entry[slot]) continue;
        body.clear();
        sink_ = &body;
        if (!Block(slot)) return false;
        const auto line = body.substr(body.find('\n') + 1);
        if (line.rfind("    goto ", 0) != 0 || line.find('\n') + 1 != line.size()) continue;
        const auto& block = At(slot);
        for (const auto& edge : block.edges) {
          if (Dead(block, edge) || !edge.target_block || !graph_.Get(*edge.target_block)) continue;
          const auto target = forward_[edge.target_block->slot];
          if (target == slot || line != "    goto " + Label(target) + "\n") continue;

          // Every goto that reached this block now reaches its target.
          for (auto& other : forward_)
            if (other == slot) other = target;
          changed = true;
          break;
        }
      }
    }

    sink_ = &out_;
    for (const auto slot : order_) {
      if (forward_[slot] != slot) continue;
      if (!Line("") || !Block(slot)) return false;
    }

    return true;
  }

  bool Block(std::size_t slot) {
    std::vector<std::uint8_t> reads_entry;
    {
      slot_ = slot;
      const auto& block = At(slot);
      std::string head = Label(slot) + ":";
      if (block.transition) head += "  ; recovered transition";
      if (block.opaque) {
        if (!Line(head + "  ; opaque instruction")) return false;
        return Control(block);
      }

      auto& state = states_[slot];
      if (!Line(head) || !Mark(slot, reads_entry) || !Weigh(block, state)) return false;
      for (ValueId id = 0; id < block.nodes.size(); ++id) {
        if (!state.marked[id] || !Named(block, id)) continue;
        const auto& node = block.nodes[id];
        std::string text;
        if (node.op == Op::store || node.op == Op::exclusive_store) {
          text = std::string("    ") + (node.op == Op::store ? "store" : "exclusive_store") +
                 std::to_string(block.nodes[node.inputs[1]].width) +
                 (node.access.byte_order == ByteOrder::big ? "be" : "") + "[" +
                 Operand(block, node.inputs[0]) + "] = " + Operand(block, node.inputs[1]);
          // An exclusive store also yields whether it failed, which control reads.
          if (node.op == Op::exclusive_store && state.uses[id] > 1)
            text += ", status v" + std::to_string(id);
        } else if (!Descriptor(node.op) || !Descriptor(node.op)->produces_value) {
          text = "    " + std::string(Descriptor(node.op) ? Descriptor(node.op)->name : "?");
        } else if (state.uses[id] == 1 && RoleOf(block, id) == Role::effect) {
          text = "    " + Expression(block, id);
        } else {
          text = "    v" + std::to_string(id) + " = " + Expression(block, id);
        }

        if (!Line(text)) return false;
      }

      // Register writes happen together at the block's end, so a write that
      // reads a register another write replaces reads its entry value.
      std::vector<SsaValue> held;
      std::vector<std::uint8_t> linked;
      if (Calls(block)) BeforeCall(slot, held, linked);
      for (std::size_t index = 0; index < block.exits.size() && index < block.phis.size();
           ++index) {
        // The link register written by a call does not count as a register write.
        if (!state.live_out[index] || (index < linked.size() && linked[index])) continue;
        const auto& exit = block.exits[index].value;
        const bool unchanged = exit.kind == SsaValueKind::phi && graph_.Get(exit.block) == &block &&
                               exit.index == index;
        if (unchanged || exit.kind == SsaValueKind::clobber) continue;
        if (!Line("    " + Name(block.phis[index].storage) + " = " + ValueText(block, exit)))
          return false;
      }

      // What the call is handed, less what the call line already says: the
      // link the call instruction writes and the register holding its target.
      if (Calls(block)) {
        const auto target = CallRegister(block);
        for (std::size_t index = 0; index < held.size(); ++index) {
          const auto& value = held[index];
          if (value.kind != SsaValueKind::node || linked[index] ||
              !Contains(target_.observable_at_call, block.phis[index].storage) ||
              block.phis[index].storage == target ||
              (state.live_out[index] && block.exits[index].value == value))
            continue;
          if (!Line("    " + Name(block.phis[index].storage) + " = " + ValueText(block, value)))
            return false;
        }
      }

      for (std::size_t index = 0;
           index < block.frame_exits.size() && index < block.frame_phis.size(); ++index) {
        if (!state.frame_live_out[index]) continue;
        const auto& exit = block.frame_exits[index];
        if (exit.kind == SsaValueKind::frame_phi && graph_.Get(exit.block) == &block &&
            exit.index == index)
          continue;
        if (!Line("    frame[" + std::to_string(block.frame_phis[index].offset) +
                  "] = " + ValueText(block, exit)))
          return false;
      }

      if (!Control(block)) return false;
    }

    return true;
  }

  const SsaGraph& graph_;
  const SsaListingTarget& target_;
  Budget& budget_;
  mutable bool exhausted_ = false;
  std::vector<BlockState> states_;
  std::vector<std::size_t> order_;
  std::size_t omitted_ = 0;
  std::size_t slot_ = 0;

  // Per slot, the block a goto to it reads as; empty while finding them.
  std::vector<std::size_t> forward_;
  std::string out_;
  std::string* sink_ = &out_;
};

}  // namespace

SsaListingResult ListSsa(const SsaGraph& graph, const SsaListingTarget& target, Budget& budget) {
  return Lister(graph, target, budget).Run();
}

}  // namespace nyx::ir
