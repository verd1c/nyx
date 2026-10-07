#include "nyx/recovery/control.hpp"

namespace nyx::recovery {
namespace {

// What one boundary earned, held until the batch is known: a batch advances the
// revision exactly once, so no record can be written before the last is found.
struct Pending {
  std::uint32_t boundary;
  ir::RewriteRule rule;
  bool condition_value = false;
  ir::DispatchBranch branch;
};

}  // namespace

ControlRecoveryResult RecoverControl(ir::Path&& input, const ir::ImageFacts& facts, Budget& budget,
                                     ControlRecoveryLimits limits) {
  const auto decline = [](ControlRecoveryDecline reason) {
    return ControlRecoveryResult{{}, reason};
  };

  const auto valid = ir::ValidatePath(input, budget, limits.path);
  if (valid != ir::BlockDecline::none) {
    return decline(valid == ir::BlockDecline::resource_limit
                       ? ControlRecoveryDecline::resource_limit
                       : ControlRecoveryDecline::invalid_ir);
  }

  if (budget.try_consume({input.boundaries().size() * 2ULL, 0}) != BudgetDecline::none)
    return decline(ControlRecoveryDecline::resource_limit);
  const auto decided = [&](const ir::Boundary& boundary) {
    if (!boundary.transfer || boundary.transfer->kind != ir::TransferKind::conditional)
      return false;
    const auto& predicate = input.nodes()[*boundary.transfer->condition];
    return predicate.op == ir::Op::constant && predicate.width == 1;
  };

  // Only a jump whose destination is still an expression can hide a branch. One
  // the image fold already settled names its place directly, and one that names
  // a place cannot rest on a choice.
  const auto computed = [&](const ir::Boundary& boundary) {
    return boundary.transfer && boundary.transfer->kind == ir::TransferKind::jump &&
           input.nodes()[boundary.transfer->target].op != ir::Op::image_address;
  };

  std::vector<Pending> pending;
  std::size_t destination_count = 0;
  for (std::size_t index = 0; index < input.boundaries().size(); ++index) {
    const auto& boundary = input.boundaries()[index];
    const auto admit = [&](Pending entry) {
      if (pending.size() == limits.max_edits ||
          budget.try_consume(
              {1, sizeof(Pending) + entry.branch.witness.size() * sizeof(ir::ImageRead)}) !=
              BudgetDecline::none)
        return false;
      pending.push_back(std::move(entry));
      return true;
    };

    if (decided(boundary)) {
      const auto& predicate = input.nodes()[*boundary.transfer->condition];
      if (!admit({static_cast<std::uint32_t>(index),
                  ir::RewriteRule::folded_condition,
                  (predicate.immediate & 1) != 0,
                  {}}))
        return decline(ControlRecoveryDecline::resource_limit);
      continue;
    }

    if (!computed(boundary)) continue;
    auto resolved =
        ir::ResolveDispatchBranch(input.nodes(), boundary.transfer->target, facts, budget);
    // An exhausted budget means the answer is unknown, so decline instead of
    // publishing the smaller batch as if the transfer hid no branch.
    if (resolved.reason == ir::DispatchDecline::resource_limit)
      return decline(ControlRecoveryDecline::resource_limit);
    if (!resolved.branch) continue;
    if (!admit({static_cast<std::uint32_t>(index), ir::RewriteRule::dispatch_branch, false,
                std::move(*resolved.branch)}))
      return decline(ControlRecoveryDecline::resource_limit);
    destination_count += 2;
  }

  if (!pending.empty() && input.revision() == UINT64_MAX)
    return decline(ControlRecoveryDecline::revision_overflow);
  // Each destination becomes a value above the basis, so the basis has to leave
  // room for one.
  if (input.nodes().size() > UINT32_MAX - destination_count)
    return decline(ControlRecoveryDecline::resource_limit);
  if (budget.try_consume({pending.size() + destination_count,
                          pending.size() * sizeof(ir::ConditionalRewrite) +
                              destination_count * sizeof(ir::Node)}) != BudgetDecline::none)
    return decline(ControlRecoveryDecline::resource_limit);
  const auto revision = input.revision() + (pending.empty() ? 0 : 1);
  const auto first_destination = static_cast<ir::ValueId>(input.nodes().size());
  std::vector<ir::ConditionalRewrite> rewrites;
  std::vector<ir::Node> destinations;
  rewrites.reserve(pending.size());
  destinations.reserve(destination_count);
  for (auto& entry : pending) {
    const auto& original = *input.boundaries()[entry.boundary].transfer;
    ir::ConditionalRewrite rewrite{entry.rule, entry.boundary,   original, {}, 0, false, 0, 0,
                                   {},         input.revision(), revision};
    if (entry.rule == ir::RewriteRule::folded_condition) {
      rewrite.condition = *original.condition;
      rewrite.condition_value = entry.condition_value;
      rewrite.replacement = {ir::TransferKind::jump,
                             entry.condition_value ? original.target : *original.alternative,
                             {},
                             {},
                             {}};
    } else {
      const auto named = static_cast<ir::ValueId>(first_destination + destinations.size());
      rewrite.condition = entry.branch.condition;
      rewrite.when_true = entry.branch.when_true;
      rewrite.when_false = entry.branch.when_false;
      rewrite.witness = std::move(entry.branch.witness);
      rewrite.replacement = {
          ir::TransferKind::conditional, named, rewrite.condition, named + 1, {}};
      destinations.push_back({ir::Op::image_address, 64, {}, rewrite.when_true});
      destinations.push_back({ir::Op::image_address, 64, {}, rewrite.when_false});
    }

    rewrites.push_back(std::move(rewrite));
  }

  ir::RecoveredPath output(std::move(input), std::move(rewrites), revision,
                           std::move(destinations));
  // The consumer-facing validator independently rechecks the complete batch.
  const auto checked = ir::ValidateRecoveredPath(output, budget, limits.path, facts);
  if (checked != ir::BlockDecline::none) {
    return decline(checked == ir::BlockDecline::resource_limit
                       ? ControlRecoveryDecline::resource_limit
                       : ControlRecoveryDecline::invalid_ir);
  }

  return {std::move(output), ControlRecoveryDecline::none};
}

}  // namespace nyx::recovery
