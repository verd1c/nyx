#include "nyx/recovery/paired_load.hpp"

#include <algorithm>

namespace nyx::recovery {

PairedLoadResult OmitForwardedPairLoads(ir::RecoveredPath&& input,
                                        std::span<const ForwardingFact> forwarding,
                                        ir::StorageId base_storage, Budget& budget,
                                        ir::ImageFacts facts, std::uint32_t max_pairs) {
  const auto valid = ir::ValidateRecoveredPath(input, budget, {}, facts);
  if (valid != ir::BlockDecline::none || !input.paired_load_omissions().empty() ||
      !input.omissions().empty()) {
    return {{},
            valid == ir::BlockDecline::resource_limit ? PairedLoadDecline::resource_limit
                                                      : PairedLoadDecline::invalid_ir};
  }

  const auto& basis = input.basis();
  const auto nodes = basis.nodes();
  const auto boundaries = basis.boundaries();
  if (forwarding.size() > nodes.size()) return {{}, PairedLoadDecline::invalid_ir};
  if (forwarding.empty()) return {std::move(input), PairedLoadDecline::none};
  const auto capacity = std::min<std::size_t>(max_pairs, boundaries.size());
  if (budget.try_consume({nodes.size() + forwarding.size() + boundaries.size(),
                          nodes.size() * sizeof(const ForwardingFact*) +
                              nodes.size() * sizeof(std::uint8_t) +
                              capacity * sizeof(ir::PairedLoadOmission)}) != BudgetDecline::none)
    return {{}, PairedLoadDecline::resource_limit};
  std::vector<const ForwardingFact*> by_load(nodes.size());
  std::vector<std::uint8_t> ambiguous(nodes.size());
  for (const auto& fact : forwarding) {
    if (fact.scope == MemoryProofScope::successful_itinerary_prefix && fact.load < nodes.size()) {
      if (by_load[fact.load]) {
        by_load[fact.load] = nullptr;
        ambiguous[fact.load] = 1;
      } else if (!ambiguous[fact.load])
        by_load[fact.load] = &fact;
    }
  }

  std::vector<ir::PairedLoadOmission> omissions;
  omissions.reserve(capacity);
  for (const auto& boundary : boundaries) {
    if (boundary.node_count < 2) continue;
    const auto second =
        static_cast<ir::ValueId>(std::uint64_t(boundary.first_node) + boundary.node_count - 1);
    const auto first = second - 1;
    const auto* a = by_load[first];
    const auto* b = by_load[second];
    if (!a || !b || a->store >= nodes.size() || b->store >= nodes.size()) continue;
    const ir::PairedLoadOmission witness{
        {a->store, b->store}, {first, second},
        base_storage,         {a->address.offset, b->address.offset},
        input.revision(),     input.revision() + 1};
    if (budget.try_consume({4 * nodes.size() + 2 * boundaries.size() + 32, 0}) !=
        BudgetDecline::none)
      return {{}, PairedLoadDecline::resource_limit};
    if (!ir::ValidPairedLoadOmission(basis, witness)) continue;
    if (omissions.size() == max_pairs) return {{}, PairedLoadDecline::resource_limit};
    if (input.revision() == UINT64_MAX) return {{}, PairedLoadDecline::revision_overflow};
    omissions.push_back(witness);
  }

  if (omissions.empty()) return {std::move(input), PairedLoadDecline::none};
  const auto revision = input.revision() + 1;
  auto result = std::move(input).with_paired_load_omissions(std::move(omissions), revision);
  const auto checked = ir::ValidateRecoveredPath(result, budget, {}, facts);
  if (checked != ir::BlockDecline::none) {
    return {{},
            checked == ir::BlockDecline::resource_limit ? PairedLoadDecline::resource_limit
                                                        : PairedLoadDecline::invalid_ir};
  }

  return {std::move(result), PairedLoadDecline::none};
}

}  // namespace nyx::recovery
