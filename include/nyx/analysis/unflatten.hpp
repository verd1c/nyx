#pragma once

#include "nyx/analysis/path_control.hpp"
#include "nyx/analysis/regions.hpp"

namespace nyx::analysis {

// A case body's route through a dispatcher, replaced by where it provably goes.
// The candidate's recovered path is the new body: it still executes every
// original instruction of the route, dispatcher included, and ends at
// `destinations` instead of the dispatch's whole table. One destination is an
// unconditional successor; two are selected by `condition`, a value of that
// path, with the destination taken when it is true listed first.
struct Transition {
  std::uint32_t candidate;
  std::uint32_t entry_block;
  std::uint32_t dispatch_block;
  std::vector<std::uint32_t> destinations;
  std::optional<ir::ValueId> condition;

  // Some route or destination fact rests on declared image values, whether
  // file bytes or a relocated pointer slot.
  bool constant_image_dependency = false;

  // The candidate's path rests on relations between storage values at its
  // entry block, which hold only on arrival along the graph's known entries,
  // and on whatever their proof rested on.
  bool entry_relation_dependency = false;

  // The recovered path revision whose control facts proved this transition.
  std::uint64_t path_revision = 0;
  std::uint64_t path_identity = 0;
};

// Why a candidate that reached a dispatch was not made a transition.
enum class TransitionRefusal {
  unproved_route,       // an internal transfer is not proved to continue along the path
  unresolved_target,    // the dispatch's destination is not proved for this path
  outside_dispatch_set  // a proved destination is not among the dispatch's own edges
};

struct RefusedTransition {
  std::uint32_t candidate;
  TransitionRefusal reason;
};

// What an edge of the recovered graph rests on beyond the original graph. All
// false is an edge the graph had already resolved on its own.
struct EdgeAssumptions {
  // The transition that produced this edge resolved something through declared
  // image values, either file bytes or a loader-written pointer slot.
  bool constant_image = false;

  // The edge is the continuation of a call to a callee this graph does not know.
  // It is a successor only under the restriction that such a callee comes back
  // nowhere else.
  bool callee_returns_to_continuation = false;

  // A return the graph did not resolve, taken to leave for the caller.
  bool return_leaves = false;

  // The edge leaves through a target that was never resolved to a block, so it
  // can arrive anywhere, including at a block this graph does not list.
  bool unresolved_target = false;

  // The target decoder declared the control of an otherwise opaque source.
  // Its instruction bytes and control behavior need independent checking.
  bool declared_opaque_control = false;

  // The transition that produced this edge rests on relations between storage
  // values at its entry block, proved over this graph from its known entries,
  // and through calls or returns only under a declared calling convention.
  bool entry_relations = false;
  bool declared_abi = false;
  bool declared_return_leaves = false;

  // A callee edge whose call has no continuation because the caller declared
  // every possible callee non-returning. Without it a missing continuation
  // would be indistinguishable from a dropped one.
  bool declared_noreturn = false;
};

enum class ConditionOwner { original_block, recovered_path };

struct RecoveredEdge {
  CfgEdgeKind kind;
  CfgTarget target;
  std::optional<std::uint32_t> target_block;

  // A transition's condition belongs to the candidate named by the containing
  // block's `transition`; a preserved condition belongs to this original block.
  std::optional<ir::ValueId> condition;
  std::optional<bool> when;
  EdgeAssumptions assumptions;
  ConditionOwner condition_owner = ConditionOwner::original_block;
};

// A block the recovered graph reaches, with the successors it has there. A block
// a transition leaves from takes that transition's destinations in place of its
// whole route through the dispatcher; every other block keeps its own edges,
// less the continuation of a call declared never to return.
struct RecoveredBlock {
  std::uint32_t block;
  std::uint64_t address;

  // Index into `transitions` when this block's successors came from one.
  std::optional<std::uint32_t> transition;
  std::vector<RecoveredEdge> edges;
};

// One graph, stitched: transition sources joined straight to their destinations
// and nothing the result no longer reaches. Closure is only within the supplied
// population and known entries: neither excludes undiscovered interior entries.
// While an unresolved successor is reachable, a block absent from this list is
// merely unreached, not proved gone, which is the retirement condition.
struct RecoveredGraph {
  std::vector<RecoveredBlock> blocks;  // ascending block id, which is ascending address
  std::vector<std::uint32_t> entries;
};

struct Unflattening {
  std::uint64_t source_identity = 0;
  std::vector<Transition> transitions;
  std::vector<RefusedTransition> refused;

  // The blocks above, connected: what the function looks like without its
  // dispatcher, under the assumptions each edge names.
  RecoveredGraph graph;

  // Blocks the recovered graph no longer reaches from its entries once each
  // transition's entry block takes its transition's successors. Their
  // instructions still run, but only inside transitions' paths, never as blocks
  // of their own; on a flattened function that is the dispatcher, together with
  // any case code a route passes through after its entry block. An unresolved
  // successor (any edge the graph did not resolve to one of its blocks, such as
  // an unmodeled instruction, an unknown or runtime-absolute target, a target
  // outside the population or inside an instruction) could lead anywhere, so
  // while one is reachable nothing is retired and the blocks it leaves from are
  // listed instead. A declared trap is none: it never completes, so it has no
  // successor. An unknown callee is counted apart: it leaves through its
  // call and comes back through the continuation edge, so retirement survives it
  // only under the restriction that callees return nowhere else. A return whose
  // target the graph did not resolve is counted apart too, under the
  // restriction that it leaves for the caller.
  std::vector<std::uint32_t> retired_blocks;
  std::vector<std::uint32_t> unresolved_successor_blocks;
  std::uint32_t reachable_unknown_callees = 0;
  std::uint32_t reachable_returns = 0;
};

// Restrictions a caller declares for the recovered graph. None is inferred:
// an opaque call is never treated as non-returning unless it is named here, and
// the caller must publish what it named.
struct UnflattenAssumptions {
  // Sorted image locations of call targets that never return, so the
  // continuation after a call to one is no successor of it.
  std::span<const std::uint64_t> noreturn_callees;

  // Untargeted returns leave the selected function population.
  bool return_leaves = false;
};

enum class UnflattenDecline { none, invalid_input, resource_limit };

struct UnflattenResult {
  std::optional<Unflattening> unflattening;
  UnflattenDecline reason = UnflattenDecline::none;
};

// Which candidates' recovered paths rest on relations between storage values
// at their entry blocks, and what the proof of those relations rested on beyond
// the graph. A dependent transition takes on all of it.
struct EntryRelationUse {
  std::span<const std::uint8_t> dependent;  // empty, or one per candidate
  bool constant_image = false;
  bool declared_opaque_control = false;
  bool declared_abi = false;
  bool declared_return_leaves = false;
};

// `control` holds, per candidate, the control facts of that candidate's
// recovered path, or nothing where it has none. Candidates that do not end at a
// dispatch are ignored; the graph itself is not modified.
[[nodiscard]] UnflattenResult Unflatten(const Regions&,
                                        std::span<const std::optional<PathControlFacts>> control,
                                        Budget&, UnflattenAssumptions = {}, EntryRelationUse = {});

}  // namespace nyx::analysis
