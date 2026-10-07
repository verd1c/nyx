#include "nyx/analysis/ssa/image_addresses.hpp"

#include <algorithm>
#include <array>
#include <bit>

#include "nyx/ir/fold.hpp"

namespace nyx::analysis {
namespace {
enum class Kind { unknown, number, image };

struct Atom {
  Kind kind = Kind::unknown;
  std::uint64_t bits = 0;
};

struct Choice {
  std::optional<ir::ValueId> condition;
  Atom truth;
  Atom falsity;
};

Atom Apply(const ir::Node& node, Atom a, Atom b, unsigned operand_width) {
  if (node.width > 64 || a.kind == Kind::unknown || b.kind == Kind::unknown) return {};
  if (node.width == 64 && node.op == ir::Op::add) {
    if (a.kind == Kind::image && b.kind == Kind::number) return {Kind::image, a.bits + b.bits};
    if (a.kind == Kind::number && b.kind == Kind::image) return {Kind::image, a.bits + b.bits};
  }

  if (node.width == 64 && node.op == ir::Op::sub && a.kind == Kind::image && b.kind == Kind::number)
    return {Kind::image, a.bits - b.bits};
  if (a.kind != Kind::number || b.kind != Kind::number) return {};
  const auto* descriptor = ir::Descriptor(node.op);
  if (!descriptor) return {};
  const std::array operands{a.bits, b.bits};
  const auto folded =
      ir::FoldPure(node, std::span(operands.data(), descriptor->arity), operand_width);
  return folded ? Atom{Kind::number, *folded} : Atom{};
}

Choice Compute(const ir::SsaBlock& block, ir::ValueId id, std::span<const Choice> values) {
  const auto& node = block.nodes[id];
  if (node.op == ir::Op::constant && node.width <= 64) {
    const Atom value{Kind::number, node.immediate & ir::LowMask(node.width)};
    return {{}, value, value};
  }

  if (node.op == ir::Op::image_address) {
    const Atom value{Kind::image, node.immediate};
    return {{}, value, value};
  }

  if (node.op == ir::Op::select) {
    const auto& selector = values[node.inputs[0]];
    const auto& truth = values[node.inputs[1]];
    const auto& falsity = values[node.inputs[2]];
    if (!selector.condition && selector.truth.kind == Kind::number)
      return selector.truth.bits ? truth : falsity;
    if (truth.condition || falsity.condition || truth.truth.kind == Kind::unknown ||
        falsity.truth.kind == Kind::unknown)
      return {};
    return {node.inputs[0], truth.truth, falsity.truth};
  }

  const auto* descriptor = ir::Descriptor(node.op);
  if (!descriptor || descriptor->effect != ir::Effect::pure || !descriptor->produces_value ||
      descriptor->arity == 0 || descriptor->arity > 2)
    return {};
  const auto& left = values[node.inputs[0]];
  const Choice empty{{}, {Kind::number, 0}, {Kind::number, 0}};
  const auto& right = descriptor->arity == 2 ? values[node.inputs[1]] : empty;
  if (left.condition && right.condition && left.condition != right.condition) return {};
  const auto condition = left.condition ? left.condition : right.condition;
  return {condition, Apply(node, left.truth, right.truth, block.nodes[node.inputs[0]].width),
          Apply(node, left.falsity, right.falsity, block.nodes[node.inputs[0]].width)};
}
}  // namespace

SsaImageAddressResult ProveSsaSelectedImageAddresses(const ir::SsaGraph& graph, Budget& budget) {
  SsaImageAddressResult result;
  const auto valid = ir::ValidateSsa(graph, budget);
  if (valid != ir::SsaDecline::none) {
    result.reason = valid == ir::SsaDecline::resource_limit ? SsaImageAddressRefusal::resource_limit
                                                            : SsaImageAddressRefusal::invalid_graph;
    return result;
  }

  ir::SsaImageAddressFacts facts{graph.arena(), graph.revision(), {}};
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) {
      result.reason = SsaImageAddressRefusal::resource_limit;
      return result;
    }

    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto& block = *graph.Get(*handle);
    if (block.nodes.size() > SIZE_MAX / sizeof(Choice) ||
        budget.try_consume({block.nodes.size(), block.nodes.size() * sizeof(Choice)}) !=
            BudgetDecline::none) {
      result.reason = SsaImageAddressRefusal::resource_limit;
      return result;
    }

    std::vector<Choice> values(block.nodes.size());
    for (std::size_t id = 0; id < block.nodes.size(); ++id) {
      values[id] = Compute(block, static_cast<ir::ValueId>(id), values);
      if (block.nodes[id].op != ir::Op::load) continue;
      if (budget.try_consume(
              {static_cast<std::uint64_t>(std::bit_width(block.disabled_effects.size())), 0}) !=
          BudgetDecline::none) {
        result.reason = SsaImageAddressRefusal::resource_limit;
        return result;
      }

      if (std::binary_search(block.disabled_effects.begin(), block.disabled_effects.end(),
                             static_cast<ir::ValueId>(id)))
        continue;
      if (budget.try_consume(
              {1 + static_cast<std::uint64_t>(std::bit_width(block.disabled_effects.size())),
               std::max(sizeof(ir::SsaSelectedImageAddress), sizeof(RefusedSsaImageAddress))}) !=
          BudgetDecline::none) {
        result.reason = SsaImageAddressRefusal::resource_limit;
        return result;
      }

      const auto& address = values[block.nodes[id].inputs[0]];
      if (address.condition && address.truth.kind == Kind::image &&
          address.falsity.kind == Kind::image && address.truth.bits != address.falsity.bits) {
        facts.selected.push_back({*handle, static_cast<ir::ValueId>(id), *address.condition,
                                  address.truth.bits, address.falsity.bits});
      } else {
        result.refused.push_back(
            {*handle, static_cast<ir::ValueId>(id), SsaImageAddressRefusal::not_selected_image});
      }
    }
  }

  const auto checked = ir::ValidateSsaImageAddressFacts(graph, facts, budget);
  if (checked != ir::SsaDecline::none) {
    result.reason = checked == ir::SsaDecline::resource_limit
                        ? SsaImageAddressRefusal::resource_limit
                        : SsaImageAddressRefusal::invalid_graph;
    result.refused.clear();
    return result;
  }

  result.facts = std::move(facts);
  return result;
}

}  // namespace nyx::analysis
