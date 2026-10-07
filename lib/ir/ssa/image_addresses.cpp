#include <algorithm>
#include <bit>

#include "nyx/ir/fold.hpp"
#include "nyx/ir/ssa/graph.hpp"

namespace nyx::ir {
namespace {
enum class Kind { unknown, number, image };

struct Value {
  Kind kind = Kind::unknown;
  std::uint64_t bits = 0;
};

Value Apply(const Node& node, Value a, Value b, unsigned operand_width) {
  if (node.width > 64 || a.kind == Kind::unknown || b.kind == Kind::unknown) return {};

  // A same-width copy keeps an image location as it is.
  if (a.kind == Kind::image && node.width == 64 && operand_width == 64 &&
      (node.op == Op::zext || (node.op == Op::extract && !node.immediate)))
    return a;
  if (node.width == 64 && node.op == Op::add) {
    if (a.kind == Kind::image && b.kind == Kind::number) return {Kind::image, a.bits + b.bits};
    if (a.kind == Kind::number && b.kind == Kind::image) return {Kind::image, a.bits + b.bits};
  }

  if (node.width == 64 && node.op == Op::sub && a.kind == Kind::image && b.kind == Kind::number)
    return {Kind::image, a.bits - b.bits};
  if (a.kind != Kind::number || b.kind != Kind::number) return {};
  const auto* descriptor = Descriptor(node.op);
  if (!descriptor) return {};
  const std::array operands{a.bits, b.bits};
  const auto folded = FoldPure(node, std::span(operands.data(), descriptor->arity), operand_width);
  return folded ? Value{Kind::number, *folded} : Value{};
}

std::optional<std::uint64_t> ArmAddress(const SsaBlock& block, ValueId address, ValueId condition,
                                        bool when, Budget& budget, bool& exhausted) {
  if (address >= block.nodes.size() || address == UINT32_MAX ||
      static_cast<std::uint64_t>(address) + 1 > SIZE_MAX / sizeof(Value) ||
      budget.try_consume({static_cast<std::uint64_t>(address) + 1,
                          (static_cast<std::size_t>(address) + 1) * sizeof(Value)}) !=
          BudgetDecline::none) {
    exhausted = true;
    return {};
  }

  // The address is evaluated with the condition fixed, so any computation of
  // it that keeps the value, not only one shape, rechecks the same way.
  std::vector<Value> values(static_cast<std::size_t>(address) + 1);
  const auto implied = SsaImpliedBits(block.nodes, condition, when);
  for (ValueId id = 0; id <= address; ++id) {
    const auto& node = block.nodes[id];
    const auto decided = std::find_if(implied.begin(), implied.end(),
                                      [&](const auto& bit) { return bit.first == id; });
    if (decided != implied.end()) {
      values[id] = {Kind::number, decided->second ? 1U : 0U};
    } else if (node.op == Op::constant && node.width <= 64) {
      values[id] = {Kind::number, node.immediate & LowMask(node.width)};
    } else if (node.op == Op::image_address) {
      values[id] = {Kind::image, node.immediate};
    } else if (node.op == Op::select) {
      const auto selector = node.inputs[0];
      if (selector == condition) {
        values[id] = values[node.inputs[when ? 1 : 2]];
      } else if (values[selector].kind == Kind::number) {
        values[id] = values[node.inputs[values[selector].bits ? 1 : 2]];
      }
    } else if (const auto* descriptor = Descriptor(node.op);
               descriptor && descriptor->effect == Effect::pure && descriptor->produces_value &&
               descriptor->arity > 0 && descriptor->arity <= 2) {
      const auto a = values[node.inputs[0]];
      const auto b = descriptor->arity == 2 ? values[node.inputs[1]] : Value{Kind::number, 0};
      values[id] = Apply(node, a, b, block.nodes[node.inputs[0]].width);
    }
  }

  return values[address].kind == Kind::image ? std::optional(values[address].bits) : std::nullopt;
}
}  // namespace

SsaDecline CheckSsaSelectedImageAddress(const SsaBlock& block, ValueId load, ValueId condition,
                                        std::uint64_t when_true, std::uint64_t when_false,
                                        Budget& budget) {
  if (load >= block.nodes.size() || block.nodes[load].op != Op::load || condition >= load ||
      block.nodes[condition].width != 1 || when_true == when_false)
    return SsaDecline::invalid_graph;
  bool exhausted = false;
  const auto address = block.nodes[load].inputs[0];
  const auto truth = ArmAddress(block, address, condition, true, budget, exhausted);
  const auto falsity = ArmAddress(block, address, condition, false, budget, exhausted);
  if (exhausted) return SsaDecline::resource_limit;
  return truth && falsity && *truth == when_true && *falsity == when_false
             ? SsaDecline::none
             : SsaDecline::invalid_graph;
}

SsaDecline ValidateSsaImageAddressFacts(const SsaGraph& graph, const SsaImageAddressFacts& facts,
                                        Budget& budget) {
  const auto valid = ValidateSsa(graph, budget);
  if (valid != SsaDecline::none) return valid;
  if (facts.graph_arena != graph.arena() || facts.graph_revision != graph.revision())
    return SsaDecline::invalid_graph;
  SsaHandle prior{};
  ValueId prior_load = 0;
  bool seen = false;
  for (const auto& fact : facts.selected) {
    if (budget.try_consume(
            {2 + static_cast<std::uint64_t>(std::bit_width(
                     graph.Get(fact.block) ? graph.Get(fact.block)->disabled_effects.size() : 0)),
             0}) != BudgetDecline::none)
      return SsaDecline::resource_limit;
    if (seen && (fact.block.slot < prior.slot ||
                 (fact.block.slot == prior.slot && fact.load <= prior_load)))
      return SsaDecline::invalid_graph;
    seen = true;
    prior = fact.block;
    prior_load = fact.load;
    const auto* block = graph.Get(fact.block);
    if (!block || fact.load >= block->nodes.size() ||
        std::binary_search(block->disabled_effects.begin(), block->disabled_effects.end(),
                           fact.load))
      return SsaDecline::invalid_graph;
    const auto checked = CheckSsaSelectedImageAddress(*block, fact.load, fact.condition,
                                                      fact.when_true, fact.when_false, budget);
    if (checked != SsaDecline::none) return checked;
  }

  return SsaDecline::none;
}

}  // namespace nyx::ir
