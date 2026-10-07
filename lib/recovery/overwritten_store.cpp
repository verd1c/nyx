#include "nyx/recovery/overwritten_store.hpp"

#include <algorithm>

namespace nyx::recovery {
namespace {

std::optional<ir::ValueId> LastAccess(std::span<const ir::Node> nodes,
                                      const ir::Boundary& boundary) {
  std::optional<ir::ValueId> found;
  unsigned stores = 0;
  for (std::uint64_t id = boundary.first_node;
       id < std::uint64_t(boundary.first_node) + boundary.node_count; ++id) {
    if (ir::HasMemoryOrMonitorEffect(nodes[id].op)) found = static_cast<ir::ValueId>(id);
    stores += ir::MayWriteMemory(nodes[id].op);
  }

  // An earlier store can change whether the first instruction commits under
  // a bounded transaction, so it cannot be skipped independently.
  return stores == 1 ? found : std::nullopt;
}

std::optional<ir::ValueId> FirstAccess(std::span<const ir::Node> nodes,
                                       const ir::Boundary& boundary) {
  for (std::uint64_t id = boundary.first_node;
       id < std::uint64_t(boundary.first_node) + boundary.node_count; ++id) {
    if (ir::HasMemoryOrMonitorEffect(nodes[id].op)) return static_cast<ir::ValueId>(id);
  }

  return {};
}
}  // namespace

StoreCleanupResult OmitOverwrittenStores(ir::RecoveredPath&& input, ir::StorageId base_storage,
                                         Budget& budget, ir::ImageFacts facts,
                                         std::uint32_t max_omissions) {
  const auto valid = ir::ValidateRecoveredPath(input, budget, {}, facts);
  if (valid != ir::BlockDecline::none || !input.omissions().empty())
    return {{},
            valid == ir::BlockDecline::resource_limit ? StoreCleanupDecline::resource_limit
                                                      : StoreCleanupDecline::invalid_ir};
  const auto& basis = input.basis();
  const auto nodes = basis.nodes();
  const auto boundaries = basis.boundaries();
  if (budget.try_consume({2 * nodes.size() + ir::kMaxOverwriteBoundaryDistance * boundaries.size(),
                          std::min<std::size_t>(max_omissions, boundaries.size()) *
                              sizeof(ir::StoreOmission)}) != BudgetDecline::none)
    return {{}, StoreCleanupDecline::resource_limit};
  std::vector<ir::StoreOmission> omissions;
  omissions.reserve(std::min<std::size_t>(max_omissions, boundaries.size()));
  for (std::size_t boundary = 0; boundary + 1 < boundaries.size(); ++boundary) {
    if (boundaries[boundary].transfer) continue;
    const auto first = LastAccess(nodes, boundaries[boundary]);
    if (!first || nodes[*first].op != ir::Op::store ||
        (!omissions.empty() && omissions.back().overwriter == *first))
      continue;
    const auto& store = nodes[*first];
    if (budget.try_consume({16, 0}) != BudgetDecline::none)
      return {{}, StoreCleanupDecline::resource_limit};
    const auto a = ir::ResolveStorageAddress(nodes, store.inputs[0], base_storage);
    if (!a) continue;
    for (std::size_t next = boundary + 1;
         next < boundaries.size() && next - boundary <= ir::kMaxOverwriteBoundaryDistance; ++next) {
      const auto& successor = boundaries[next];
      if (successor.transfer) break;
      if (budget.try_consume({successor.node_count, 0}) != BudgetDecline::none)
        return {{}, StoreCleanupDecline::resource_limit};
      const auto second = FirstAccess(nodes, successor);
      if (!second) continue;
      if (nodes[*second].op == ir::Op::store && successor.writes.empty() &&
          std::uint64_t(*second) + 1 ==
              std::uint64_t(successor.first_node) + successor.node_count) {
        const auto& overwrite = nodes[*second];
        if (store.width == overwrite.width &&
            store.access.byte_order == overwrite.access.byte_order &&
            store.access.alignment == overwrite.access.alignment &&
            store.access.decline_on_unaligned == overwrite.access.decline_on_unaligned) {
          if (budget.try_consume({32, 0}) != BudgetDecline::none)
            return {{}, StoreCleanupDecline::resource_limit};
          const auto b = ir::ResolveStorageAddress(nodes, overwrite.inputs[0], base_storage);
          if (b && a->root == b->root && a->offset == b->offset) {
            if (omissions.size() == max_omissions) return {{}, StoreCleanupDecline::resource_limit};
            if (input.revision() == UINT64_MAX) return {{}, StoreCleanupDecline::revision_overflow};
            omissions.push_back(
                {*first, *second, base_storage, a->offset, input.revision(), input.revision() + 1});
          }
        }
      }

      // Any memory access before a later overwriter could observe or change the slot.
      break;
    }
  }

  if (omissions.empty()) return {std::move(input), StoreCleanupDecline::none};
  const auto revision = input.revision() + 1;
  auto result = std::move(input).with_omissions(std::move(omissions), revision);
  const auto checked = ir::ValidateRecoveredPath(result, budget, {}, facts);
  if (checked != ir::BlockDecline::none)
    return {{},
            checked == ir::BlockDecline::resource_limit ? StoreCleanupDecline::resource_limit
                                                        : StoreCleanupDecline::invalid_ir};
  return {std::move(result), StoreCleanupDecline::none};
}

}  // namespace nyx::recovery
