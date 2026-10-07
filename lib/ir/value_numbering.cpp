#include "nyx/ir/value_numbering.hpp"

#include <array>
#include <bit>
#include <map>

namespace nyx::ir {
namespace {

struct NodeKey {
  Op op;
  unsigned width;
  std::uint64_t immediate;
  StorageId storage;
  ByteOrder byte_order;
  unsigned alignment;
  bool decline_on_unaligned;
  std::array<ValueId, 3> inputs;
  auto operator<=>(const NodeKey&) const = default;
};

}  // namespace

ValueId Unwrap(ValueId id, std::span<const Node> nodes) {
  for (unsigned step = 0; step < 8; ++step) {
    const auto& outer = nodes[id];
    if (outer.op != Op::extract || outer.immediate != 0) break;
    const auto& middle = nodes[outer.inputs[0]];
    if (middle.op != Op::zext || middle.width < outer.width) break;
    if (nodes[middle.inputs[0]].width != outer.width) break;
    id = middle.inputs[0];
  }

  return id;
}

bool NumberValues(std::span<const Node> nodes, Budget& budget, std::vector<ValueId>& same_value,
                  std::vector<ValueId>& unwrapped) {
  std::map<NodeKey, ValueId> seen;
  same_value.assign(nodes.size(), 0);
  unwrapped.assign(nodes.size(), 0);
  for (std::size_t id = 0; id < nodes.size(); ++id) {
    same_value[id] = static_cast<ValueId>(id);
    unwrapped[id] = Unwrap(static_cast<ValueId>(id), nodes);
    const auto& node = nodes[id];
    const auto* descriptor = Descriptor(node.op);
    if (descriptor == nullptr || descriptor->effect != Effect::pure || !descriptor->produces_value)
      continue;
    NodeKey key{node.op,
                node.width,
                node.immediate,
                node.storage,
                node.access.byte_order,
                node.access.alignment,
                node.access.decline_on_unaligned,
                {}};
    for (unsigned i = 0; i < descriptor->arity; ++i) key.inputs[i] = same_value[node.inputs[i]];
    if (budget.try_consume({std::bit_width(seen.size()) + 1ULL,
                            sizeof(NodeKey) + sizeof(ValueId)}) != BudgetDecline::none) {
      return false;
    }

    const auto position = seen.emplace(key, static_cast<ValueId>(id));
    if (!position.second) same_value[id] = position.first->second;
  }

  return true;
}

}  // namespace nyx::ir
