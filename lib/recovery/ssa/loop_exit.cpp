#include "nyx/recovery/ssa/loop_exit.hpp"

#include <algorithm>

namespace nyx::recovery {
namespace {

SsaLoopExitResult Decline(SsaLoopExitRefusal reason) {
  SsaLoopExitResult result;
  result.reason = reason;
  return result;
}

SsaLoopExitRefusal From(ir::SsaDecline decline, SsaLoopExitRefusal otherwise) {
  return decline == ir::SsaDecline::resource_limit ? SsaLoopExitRefusal::resource_limit : otherwise;
}

// Which of the block's values every run computes alike: literals, image
// locations, registers read as they arrived and handed on unchanged, and pure
// operations over those. One forward pass, since inputs precede their users.
std::vector<std::uint8_t> SteadyValues(const ir::SsaBlock& block, ir::SsaHandle self) {
  std::vector<std::uint8_t> steady(block.nodes.size());
  for (ir::ValueId id = 0; id < block.nodes.size(); ++id) {
    const auto& node = block.nodes[id];
    if (node.op == ir::Op::constant || node.op == ir::Op::image_address) {
      steady[id] = 1;
      continue;
    }

    if (node.op == ir::Op::read) {
      for (const auto& read : block.reads) {
        if (read.node != id) continue;
        if (read.value.kind != ir::SsaValueKind::phi || read.value.block != self) break;
        const auto storage = block.phis[read.value.index].storage;
        for (const auto& exit : block.exits)
          if (exit.storage == storage) steady[id] = exit.value == read.value;
        break;
      }

      continue;
    }

    const auto* descriptor = ir::Descriptor(node.op);
    if (!descriptor || descriptor->effect != ir::Effect::pure || !descriptor->produces_value)
      continue;
    bool all = true;
    for (unsigned input = 0; input < descriptor->arity; ++input)
      all = all && node.inputs[input] < id && steady[node.inputs[input]];
    steady[id] = all;
  }

  return steady;
}

// Whether running the block once instead of many times drops no effect: it
// computes and writes registers, and its memory accesses are all disabled --
// absorbed, omitted, retired or dead -- so each at most rechecks an address
// that every run computes alike and the first run already checked.
bool OnlyStorage(const ir::SsaBlock& block, ir::SsaHandle self) {
  const auto steady = SteadyValues(block, self);
  for (ir::ValueId id = 0; id < block.nodes.size(); ++id) {
    const auto& node = block.nodes[id];
    const auto* descriptor = ir::Descriptor(node.op);
    if (!descriptor) return false;
    if (descriptor->effect == ir::Effect::pure || descriptor->effect == ir::Effect::storage_read ||
        node.op == ir::Op::write ||
        std::binary_search(block.dead_pure_nodes.begin(), block.dead_pure_nodes.end(), id))
      continue;
    if (!std::binary_search(block.disabled_effects.begin(), block.disabled_effects.end(), id))
      return false;
    const auto fold =
        std::find_if(block.constant_loads.begin(), block.constant_loads.end(),
                     [&](const ir::SsaConstantLoad& load) { return load.node == id; });
    if (fold != block.constant_loads.end() && fold->skip_access) continue;
    if (!steady[node.inputs[0]]) return false;
  }

  return true;
}

}  // namespace

SsaLoopExitResult ProposeSsaLoopExits(const ir::SsaGraph& original,
                                      const ir::SsaBoundedLoopFacts& loops,
                                      std::span<const ir::Group> sources, Budget& budget) {
  const auto valid = ir::ValidateSsaWithSources(original, sources, budget);
  if (valid != ir::SsaDecline::none) return Decline(From(valid, SsaLoopExitRefusal::invalid_graph));
  const auto checked = ir::ValidateSsaBoundedLoopFacts(original, loops, sources, budget);
  if (checked != ir::SsaDecline::none)
    return Decline(From(checked, SsaLoopExitRefusal::stale_proof));
  if (loops.loops.empty()) return {};
  const auto liveness = ir::SsaStorageLiveness::Compute(original, budget);
  const auto frames = liveness ? ir::SsaFrameLiveness::Compute(original, budget) : std::nullopt;
  if (!liveness || !frames) return Decline(SsaLoopExitRefusal::resource_limit);

  SsaLoopExitResult result;

  // Per edit: the block, which way the condition leaves, and whether its
  // conditional is a restored dispatch rather than a decoded branch.
  struct Planned {
    ir::SsaHandle block;
    bool leaves_when;
    bool dispatch;
    std::uint32_t iterations;
    ir::SsaHandle exit;
  };

  std::vector<Planned> planned;
  for (const auto& fact : loops.loops) {
    const auto* block = original.Get(fact.loop);
    if (!block) return Decline(SsaLoopExitRefusal::stale_proof);
    if (budget.try_consume({block->nodes.size() * (2 + block->reads.size() + block->exits.size() +
                                                   block->constant_loads.size()) +
                                block->phis.size() + block->frame_phis.size(),
                            block->nodes.size() + sizeof(Planned) + sizeof(SsaLoopExitRefused)}) !=
        BudgetDecline::none)
      return Decline(SsaLoopExitRefusal::resource_limit);
    const auto refuse = [&](SsaLoopExitDecline reason) {
      result.refused.push_back({fact.loop, reason});
    };

    if (fact.exit_edge >= block->edges.size() || block->boundaries.empty() ||
        !block->boundaries.back().transfer)
      return Decline(SsaLoopExitRefusal::stale_proof);
    const auto& leaving = block->edges[fact.exit_edge];
    const auto& transfer = *block->boundaries.back().transfer;
    const auto last = static_cast<std::uint32_t>(block->boundaries.size() - 1);
    bool dispatch = false;
    if (block->control_rewrites.size() == 1 &&
        block->control_rewrites.front().rule == ir::RewriteRule::dispatch_branch &&
        block->control_rewrites.front().boundary == last &&
        transfer.kind == ir::TransferKind::jump) {
      dispatch = true;
    } else if (!block->control_rewrites.empty() || transfer.kind != ir::TransferKind::conditional ||
               !transfer.condition || leaving.condition != transfer.condition) {
      refuse(SsaLoopExitDecline::unsupported_control);
      continue;
    }

    if (!leaving.when) return Decline(SsaLoopExitRefusal::stale_proof);
    if (!OnlyStorage(*block, fact.loop)) {
      refuse(SsaLoopExitDecline::unsupported_effect);
      continue;
    }

    bool dead = true;
    for (std::uint32_t index = 0; dead && index < block->phis.size() && index < block->exits.size();
         ++index) {
      const ir::SsaValue unchanged{ir::SsaValueKind::phi, fact.loop, index};
      dead = block->exits[index].value == unchanged ||
             !liveness->LiveIn(fact.successor.slot, block->phis[index].storage);
    }

    for (std::uint32_t index = 0;
         dead && index < block->frame_phis.size() && index < block->frame_exits.size(); ++index) {
      const ir::SsaValue unchanged{ir::SsaValueKind::frame_phi, fact.loop, index};
      dead = block->frame_exits[index] == unchanged ||
             !frames->LiveIn(fact.successor.slot, block->frame_phis[index].offset,
                             block->frame_phis[index].size);
    }

    if (!dead) {
      refuse(SsaLoopExitDecline::live_state);
      continue;
    }

    planned.push_back({fact.loop, *leaving.when, dispatch, fact.iterations, fact.successor});
  }

  if (planned.empty()) return result;

  auto candidate = original.Clone(budget);
  if (!candidate) return Decline(SsaLoopExitRefusal::resource_limit);
  for (const auto& plan : planned) {
    // The clone is a new arena: its handles are the same slots, renamed.
    const auto here = candidate->Handle(plan.block.slot);
    const auto there = candidate->Handle(plan.exit.slot);
    if (!here || !there) return Decline(SsaLoopExitRefusal::invalid_graph);
    auto copied = candidate->CopyBlock(*here, budget);
    if (!copied) return Decline(SsaLoopExitRefusal::resource_limit);
    auto block = std::move(*copied);
    if (budget.try_consume({1 + block.phis.size() + block.frame_phis.size(),
                            sizeof(ir::ConditionalRewrite) + sizeof(SsaLoopExitEdit)}) !=
        BudgetDecline::none)
      return Decline(SsaLoopExitRefusal::resource_limit);
    const auto last = static_cast<std::uint32_t>(block.boundaries.size() - 1);
    const auto from = block.path_revision;
    if (plan.dispatch) {
      // The dispatch keeps both destinations, as a decided one does; which of
      // the two the loop leaves by is the rewrite's replacement now.
      auto& rewrite = block.control_rewrites.front();
      if (!rewrite.replacement.alternative) return Decline(SsaLoopExitRefusal::invalid_graph);
      const auto selected =
          plan.leaves_when ? rewrite.replacement.target : *rewrite.replacement.alternative;
      rewrite.rule = ir::RewriteRule::bounded_exit;
      rewrite.replacement = {ir::TransferKind::jump, selected, {}, {}, {}};
      rewrite.condition_value = plan.leaves_when;
      rewrite.from_revision = from;
      rewrite.to_revision = from + 1;
      rewrite.iterations = plan.iterations;
    } else {
      const auto& transfer = *block.boundaries.back().transfer;
      const auto selected = plan.leaves_when ? transfer.target : *transfer.alternative;
      block.control_rewrites.push_back({ir::RewriteRule::bounded_exit,
                                        last,
                                        transfer,
                                        {ir::TransferKind::jump, selected, {}, {}, {}},
                                        *transfer.condition,
                                        plan.leaves_when,
                                        0,
                                        0,
                                        {},
                                        from,
                                        from + 1,
                                        plan.iterations});
    }

    ++block.path_revision;
    const auto kept = std::find_if(
        block.edges.begin(), block.edges.end(),
        [&](const ir::SsaEdge& edge) { return edge.target_block && *edge.target_block == *there; });
    if (kept == block.edges.end()) return Decline(SsaLoopExitRefusal::invalid_graph);
    auto edge = *kept;
    edge.condition.reset();
    edge.when.reset();
    block.edges = {edge};

    // The block no longer reaches itself, so nothing arrives from it.
    const auto prune = [&](auto& phis) {
      for (auto& phi : phis)
        std::erase_if(phi.incoming,
                      [&](const ir::SsaPhiInput& input) { return input.predecessor == *here; });
    };

    prune(block.phis);
    prune(block.frame_phis);
    if (!candidate->Replace(*here, std::move(block)))
      return Decline(SsaLoopExitRefusal::invalid_graph);
    result.journal.push_back({plan.block, plan.exit, plan.iterations, original.revision(), 0});
  }

  const auto candidate_valid = ir::ValidateSsaWithSources(*candidate, sources, budget);
  if (candidate_valid != ir::SsaDecline::none)
    return Decline(From(candidate_valid, SsaLoopExitRefusal::invalid_graph));
  const auto exits_dead = ir::ValidateSsaRetiredWork(*candidate, budget);
  if (exits_dead != ir::SsaDecline::none)
    return Decline(From(exits_dead, SsaLoopExitRefusal::invalid_graph));
  for (const auto& edit : result.journal) {
    const auto complete = ir::ValidateSsaDirectSuccessors(
        *candidate->Get(*candidate->Handle(edit.block.slot)), budget);
    if (complete != ir::SsaDecline::none)
      return Decline(From(complete, SsaLoopExitRefusal::invalid_graph));
  }

  for (auto& edit : result.journal) edit.to_revision = candidate->revision();
  result.provisional = std::move(*candidate);
  return result;
}

}  // namespace nyx::recovery
