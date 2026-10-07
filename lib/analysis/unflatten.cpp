#include "nyx/analysis/unflatten.hpp"

#include <algorithm>

namespace nyx::analysis {
namespace {

// The block a proved destination names, if it is one of the dispatch's edges.
// Those edges are the jump's complete successor set in the graph, so a
// path-level destination outside them means the two facts disagree. The graph
// keeps a dispatch's edges strictly increasing by address. An opaque source is
// a block like any other; its own outgoing edges describe where control goes next.
std::optional<std::uint32_t> DispatchEdge(const CfgBlock& dispatch, std::uint64_t address) {
  const auto edge = std::lower_bound(
      dispatch.edges.begin(), dispatch.edges.end(), address,
      [](const CfgEdge& value, std::uint64_t key) { return value.target.address < key; });
  if (edge == dispatch.edges.end() || edge->target.address != address ||
      (edge->resolution != TargetResolution::block_entry &&
       edge->resolution != TargetResolution::opaque_entry) ||
      !edge->target_block)
    return {};
  return edge->target_block;
}

}  // namespace

UnflattenResult Unflatten(const Regions& regions,
                          std::span<const std::optional<PathControlFacts>> control, Budget& budget,
                          UnflattenAssumptions assumptions, EntryRelationUse relations) {
  const auto decline = [](UnflattenDecline reason) { return UnflattenResult{{}, reason}; };
  const auto candidates = regions.candidates();
  const auto blocks = regions.graph().blocks();
  if (control.size() != candidates.size() ||
      (!relations.dependent.empty() && relations.dependent.size() != candidates.size()) ||
      !std::is_sorted(assumptions.noreturn_callees.begin(), assumptions.noreturn_callees.end())) {
    return decline(UnflattenDecline::invalid_input);
  }

  // A call has no continuation only when every possible callee is declared
  // non-returning. A guarded call may have both returning and non-returning arms.
  const auto noreturn = [&](const CfgBlock& block) {
    bool has_callee = false;
    for (const auto& edge : block.edges) {
      if (edge.kind != CfgEdgeKind::callee) continue;
      has_callee = true;
      if (edge.target.kind != TargetKind::image_location ||
          !std::binary_search(assumptions.noreturn_callees.begin(),
                              assumptions.noreturn_callees.end(), edge.target.address))
        return false;
    }

    return has_callee;
  };

  // Eight per-block tables, two reachability passes, and at most one
  // transition or refusal per candidate.
  if (budget.try_consume(
          {candidates.size() + blocks.size(),
           blocks.size() *
                   (sizeof(std::optional<std::size_t>) + sizeof(std::optional<std::uint32_t>) + 6 +
                    sizeof(std::uint32_t) + sizeof(std::vector<std::uint32_t>)) +
               candidates.size() * (sizeof(Transition) + 2 * sizeof(std::uint32_t))}) !=
      BudgetDecline::none) {
    return decline(UnflattenDecline::resource_limit);
  }

  Unflattening result;
  result.source_identity = regions.graph().identity();

  // A later transition from the same entry replaces an earlier one only if the
  // two agree; candidates from one entry that disagree cannot both be the route.
  std::vector<std::optional<std::size_t>> by_entry(blocks.size());
  std::vector<std::uint8_t> conflicted(blocks.size());
  for (std::uint32_t index = 0; index < candidates.size(); ++index) {
    const auto& region = candidates[index];
    if (region.stop != RegionStop::dispatch || region.block_ids.empty() || !control[index])
      continue;
    const auto dispatch_block = region.block_ids.back();
    const auto& facts = *control[index];
    const auto refuse = [&](TransitionRefusal reason) {
      result.refused.push_back({index, reason});
    };

    if (dispatch_block >= blocks.size() || !blocks[dispatch_block].dispatch) {
      return decline(UnflattenDecline::invalid_input);
    }

    if (facts.proved_divergence || facts.boundaries.size() != facts.total_boundaries ||
        facts.boundaries.empty()) {
      refuse(TransitionRefusal::unproved_route);
      continue;
    }

    if (budget.try_consume({facts.boundaries.size(), 0}) != BudgetDecline::none) {
      return decline(UnflattenDecline::resource_limit);
    }

    const auto& terminal = facts.boundaries.back();
    if (!std::all_of(facts.boundaries.begin(), facts.boundaries.end() - 1,
                     [](const BoundaryControl& boundary) {
                       return boundary.expected_match == ExpectedMatch::always;
                     })) {
      refuse(TransitionRefusal::unproved_route);
      continue;
    }

    // The dispatch is a jump the case body computes. Recovery may already have
    // replaced it by the conditional it hid, in which case the two destinations
    // arrive as the arms of that conditional instead of as one split expression;
    // either way they are still checked against the dispatch's own edge set.
    const bool transfers = terminal.transfer_kind == ir::TransferKind::jump ||
                           terminal.transfer_kind == ir::TransferKind::conditional;
    if (!transfers || terminal.edge_count == 0 || terminal.edge_count > 2) {
      refuse(TransitionRefusal::unresolved_target);
      continue;
    }

    Transition transition{index, region.entry_block, dispatch_block, {}, {}};
    transition.path_revision = facts.path_revision;
    transition.path_identity = facts.path_identity;
    transition.entry_relation_dependency =
        !relations.dependent.empty() && relations.dependent[index] != 0;
    transition.constant_image_dependency =
        transition.entry_relation_dependency && relations.constant_image;
    bool resolved = true, inside = true;
    for (unsigned i = 0; i < terminal.edge_count; ++i) {
      const auto& edge = terminal.edges[i];
      if (edge.target_kind != TargetKind::image_location) {
        resolved = false;
        break;
      }

      const auto block = DispatchEdge(blocks[dispatch_block], edge.target_address);
      if (!block)
        inside = false;
      else
        transition.destinations.push_back(*block);
    }

    if (!resolved) {
      refuse(TransitionRefusal::unresolved_target);
      continue;
    }

    if (!inside) {
      refuse(TransitionRefusal::outside_dispatch_set);
      continue;
    }

    if (terminal.edge_count == 2) {
      const auto& first = terminal.edges[0];
      const auto& second = terminal.edges[1];
      if (!first.condition || first.condition != second.condition || !first.when || !second.when ||
          *first.when == *second.when) {
        refuse(TransitionRefusal::unresolved_target);
        continue;
      }

      transition.condition = first.condition;
      if (!*first.when) std::swap(transition.destinations[0], transition.destinations[1]);
    }

    for (const auto& boundary : facts.boundaries) {
      for (unsigned i = 0; i < boundary.edge_count; ++i) {
        transition.constant_image_dependency |= boundary.edges[i].constant_image_dependency;
      }

      if (boundary.dispatch) {
        transition.constant_image_dependency |= boundary.dispatch->constant_image_dependency;
      }
    }

    auto& seen = by_entry[transition.entry_block];
    if (seen) {
      const auto& previous = result.transitions[*seen];
      const auto& prior_route = candidates[previous.candidate];

      // Value ids belong to a candidate's path. They can be compared only
      // when both candidates describe the same route through the same blocks.
      if (prior_route.block_ids != region.block_ids ||
          prior_route.transition_edges != region.transition_edges ||
          previous.destinations != transition.destinations ||
          previous.condition != transition.condition ||
          previous.constant_image_dependency != transition.constant_image_dependency ||
          previous.entry_relation_dependency != transition.entry_relation_dependency) {
        conflicted[transition.entry_block] = 1;
      }

      continue;
    }

    seen = result.transitions.size();
    result.transitions.push_back(std::move(transition));
  }

  // An entry whose candidates disagree has no single route to publish.
  std::erase_if(result.transitions, [&](const Transition& transition) {
    return conflicted[transition.entry_block] != 0;
  });

  // The recovered graph's known edges: a transition's entry block takes the
  // transition's successors in place of its own route.
  std::vector<std::vector<std::uint32_t>> replaced(blocks.size());
  std::vector<std::uint8_t> has_transition(blocks.size());

  // An entry keeps at most one transition, so this names which one it is.
  std::vector<std::optional<std::uint32_t>> transition_of(blocks.size());
  for (std::uint32_t index = 0; index < result.transitions.size(); ++index) {
    const auto& transition = result.transitions[index];
    has_transition[transition.entry_block] = 1;
    transition_of[transition.entry_block] = index;
    auto& successors = replaced[transition.entry_block];
    successors.insert(successors.end(), transition.destinations.begin(),
                      transition.destinations.end());
  }

  std::vector<std::uint8_t> unresolved(blocks.size());
  const auto listed = [&](std::uint32_t id) {
    const auto entries = regions.graph().entries();
    return id < blocks.size() &&
           std::binary_search(entries.begin(), entries.end(),
                              regions.graph().sources()[blocks[id].first_source].address);
  };

  const auto reach = [&](bool recovered, std::uint32_t* callees, std::uint32_t* returns) {
    std::vector<std::uint8_t> seen(blocks.size());
    std::vector<std::uint32_t> stack;
    const auto sources = regions.graph().sources();
    const auto entries = regions.graph().entries();
    for (std::uint32_t id = 0; id < blocks.size(); ++id) {
      if (std::binary_search(entries.begin(), entries.end(),
                             sources[blocks[id].first_source].address)) {
        seen[id] = 1;
        stack.push_back(id);
      }
    }
    while (!stack.empty()) {
      if (budget.try_consume({1, 0}) != BudgetDecline::none) return std::vector<std::uint8_t>{};
      const auto id = stack.back();
      stack.pop_back();
      const auto visit = [&](std::uint32_t next) {
        if (!seen[next]) {
          seen[next] = 1;
          stack.push_back(next);
        }
      };

      if (recovered && has_transition[id]) {
        for (const auto next : replaced[id]) visit(next);
        continue;
      }

      const bool ends = noreturn(blocks[id]);
      for (const auto& edge : blocks[id].edges) {
        if (ends && edge.kind == CfgEdgeKind::potential_return) continue;
        if (edge.target_block) visit(*edge.target_block);

        // A return leaves only under the caller's explicit restriction: its
        // target is a value and may otherwise re-enter this population. A
        // trap never completes, so it has no successor. A
        // call leaves too and comes back through its continuation, which is an
        // edge of its own. Every other edge the graph did not resolve to one of
        // its blocks (unknown, runtime-absolute, outside the population or
        // inside an instruction) could lead back into any block.
        // A call to a listed entry, recursion say, leaves the same way.
        if (callees && edge.kind == CfgEdgeKind::callee && edge.target_block &&
            listed(*edge.target_block))
          ++*callees;
        if (!callees || edge.target_block || edge.kind == CfgEdgeKind::trap) continue;
        if (edge.kind == CfgEdgeKind::return_) {
          ++*returns;
          if (!assumptions.return_leaves) unresolved[id] = 1;
        } else if (edge.kind == CfgEdgeKind::callee)
          ++*callees;
        else
          unresolved[id] = 1;
      }
    }

    return seen;
  };

  const auto original = reach(false, nullptr, nullptr);
  const auto recovered = reach(true, &result.reachable_unknown_callees, &result.reachable_returns);
  if (original.size() != blocks.size() || recovered.size() != blocks.size()) {
    return decline(UnflattenDecline::resource_limit);
  }

  for (std::uint32_t id = 0; id < blocks.size(); ++id) {
    if (unresolved[id]) result.unresolved_successor_blocks.push_back(id);
  }

  if (result.unresolved_successor_blocks.empty()) {
    for (std::uint32_t id = 0; id < blocks.size(); ++id) {
      if (original[id] && !recovered[id]) result.retired_blocks.push_back(id);
    }
  }

  // The same walk, written down: every block it reached, with the successors it
  // followed there and what each of those rests on.
  if (budget.try_consume({blocks.size(), blocks.size() * sizeof(RecoveredBlock)}) !=
      BudgetDecline::none) {
    return decline(UnflattenDecline::resource_limit);
  }

  const auto sources = regions.graph().sources();
  const auto entries = regions.graph().entries();
  for (std::uint32_t id = 0; id < blocks.size(); ++id) {
    if (!recovered[id]) continue;
    const auto address = sources[blocks[id].first_source].address;
    RecoveredBlock stitched{id, address, transition_of[id], {}};
    if (std::binary_search(entries.begin(), entries.end(), address))
      result.graph.entries.push_back(id);
    if (stitched.transition) {
      const auto& transition = result.transitions[*stitched.transition];
      if (budget.try_consume({transition.destinations.size(),
                              transition.destinations.size() * sizeof(RecoveredEdge)}) !=
          BudgetDecline::none) {
        return decline(UnflattenDecline::resource_limit);
      }

      // The route through the dispatcher, replaced by where it goes. The
      // condition is a value of the transition's path, and the destination it
      // selects when it holds is listed first.
      for (std::size_t i = 0; i < transition.destinations.size(); ++i) {
        const auto target = transition.destinations[i];
        RecoveredEdge edge{
            CfgEdgeKind::branch,
            {TargetKind::image_location, sources[blocks[target].first_source].address, {}},
            target,
            transition.condition,
            {},
            {}};
        edge.condition_owner = ConditionOwner::recovered_path;
        if (transition.condition) edge.when = i == 0;
        edge.assumptions.constant_image = transition.constant_image_dependency;
        edge.assumptions.entry_relations = transition.entry_relation_dependency;
        edge.assumptions.declared_opaque_control =
            transition.entry_relation_dependency && relations.declared_opaque_control;
        edge.assumptions.declared_abi =
            transition.entry_relation_dependency && relations.declared_abi;
        edge.assumptions.declared_return_leaves =
            transition.entry_relation_dependency && relations.declared_return_leaves;
        stitched.edges.push_back(std::move(edge));
      }
    } else {
      const bool ends = noreturn(blocks[id]);

      // A callee re-enters the population only at a listed entry, so one that
      // is itself a listed entry returns through the continuation as any does.
      const bool opaque_callee =
          std::any_of(blocks[id].edges.begin(), blocks[id].edges.end(), [&](const CfgEdge& edge) {
            return edge.kind == CfgEdgeKind::callee &&
                   (!edge.target_block || listed(*edge.target_block));
          });
      if (budget.try_consume(
              {blocks[id].edges.size(), blocks[id].edges.size() * sizeof(RecoveredEdge)}) !=
          BudgetDecline::none) {
        return decline(UnflattenDecline::resource_limit);
      }

      for (const auto& edge : blocks[id].edges) {
        if (ends && edge.kind == CfgEdgeKind::potential_return) continue;
        RecoveredEdge kept{edge.kind,      edge.target, edge.target_block,
                           edge.condition, edge.when,   {}};
        kept.assumptions.constant_image = edge.constant_image_dependency;
        kept.assumptions.declared_opaque_control =
            !blocks[id].ssa &&
            sources[blocks[id].first_source].opaque_control != OpaqueControl::unknown;
        kept.assumptions.callee_returns_to_continuation =
            edge.kind == CfgEdgeKind::potential_return && opaque_callee;
        kept.assumptions.declared_noreturn = ends && edge.kind == CfgEdgeKind::callee;
        kept.assumptions.return_leaves =
            !edge.target_block && edge.kind == CfgEdgeKind::return_ && assumptions.return_leaves;
        kept.assumptions.unresolved_target =
            !edge.target_block &&
            (edge.kind != CfgEdgeKind::return_ || !kept.assumptions.return_leaves) &&
            edge.kind != CfgEdgeKind::callee && edge.kind != CfgEdgeKind::trap;
        stitched.edges.push_back(std::move(kept));
      }
    }

    result.graph.blocks.push_back(std::move(stitched));
  }

  return {std::move(result), UnflattenDecline::none};
}

}  // namespace nyx::analysis
