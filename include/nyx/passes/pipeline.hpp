#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "nyx/ir/frame.hpp"
#include "nyx/ir/group.hpp"
#include "nyx/ir/image_facts.hpp"
#include "nyx/ir/ssa/graph.hpp"
#include "nyx/support/budget.hpp"

namespace nyx::passes {

// What the caller declared for the SSA proposals. Each is a restriction the
// caller takes on; nothing here is inferred, and all of it is printed.
struct SsaDeclarations {
  std::optional<ir::PrivateFrameContract> frame;
  bool image_access = false;
  bool closed_entries = false;
  bool return_leaves = false;

  // Every callee returns only through its call's continuation; the ABI
  // declaration states it. Needed only when the graph calls out.
  bool call_returns = false;

  // A trap never resumes inside the population except at a listed entry.
  // Needed only when the graph has a declared trap.
  bool trap_stops = false;
};

// Declarations a pass needs before it may run. `return_leaves` is required
// only when the graph has a return edge.
enum SsaRequirement : unsigned {
  kNeedsClosedEntries = 1U << 0,
  kNeedsImageAccess = 1U << 1,
  kNeedsReturnLeaves = 1U << 2,
  kNeedsPrivateFrame = 1U << 3,
  // Only when the graph calls out, and only when it has a declared trap.
  kNeedsCallReturns = 1U << 4,
  kNeedsTrapStops = 1U << 5,
};

// The state one pipeline run hands each pass: the current provisional graph,
// the declarations, the proof cache and the stage record being written.
class SsaPassContext;

struct SsaPassInfo {
  std::string_view name;
  unsigned requirements;

  // The stage record's members after the revisions when the pass does not run,
  // so every record of one pass has the same members.
  std::string_view empty_record;

  // Writes the pass's stage record and replaces the current graph when it
  // proposes. False means the budget ran out.
  bool (*run)(SsaPassContext&);
};

// The one production registry. Its order is the default pipeline; a named
// pipeline selects registered passes in the caller's order. Scheduling and the
// published stage list both come from here.
[[nodiscard]] std::span<const SsaPassInfo> SsaPassRegistry();
[[nodiscard]] std::optional<std::size_t> FindSsaPass(std::string_view name);

enum class SsaStageOutcome { proposed, unchanged, declined, not_run };

struct SsaStageSummary {
  std::size_t pass;  // registry index
  SsaStageOutcome outcome;
  std::string_view reason;
  std::uint64_t from_revision;
  std::optional<std::uint64_t> to_revision;

  // Journal entries of a proposed stage that change executed code: every
  // entry except a removed block, and one for a proposal with no entries. An
  // independent check must witness each of them.
  std::size_t executable_edits = 0;
};

struct SsaPipelineResult {
  // JSON members that follow the SSA summary: revision, base text and declarations.
  std::string head;

  // The proposal members: the pipeline that ran and each stage's record.
  std::string body;

  // The final provisional text as a JSON string, or null when nothing changed.
  std::string text;
  std::vector<SsaStageSummary> stages;

  // Every declared read a fold in this batch made the graph drop because a
  // store it placed writes it: locations the CFG may have resolved from.
  std::vector<ir::SsaDroppedPathRead> dropped_path_reads;

  // The exact graph described by the final provisional text, if any pass edited it.
  std::optional<ir::SsaGraph> provisional;
};

// `incomplete_record` is an internal defect: a pass returned without a record.
enum class SsaPipelineDecline {
  none,
  resource_limit,
  invalid_selection,
  incomplete_record,
  retired_work_observed
};

struct SsaPipelineRun {
  std::optional<SsaPipelineResult> result;
  SsaPipelineDecline reason = SsaPipelineDecline::none;
};

// Runs `selection` (registry indices; empty means the default pipeline), each
// pass over the previous stage's provisional graph. Every edit stays
// provisional until a caller's acceptance policy says otherwise. A budget
// decline returns no partial record; a selection naming an index outside the
// registry or one pass twice is refused before anything runs.
//
// `rounds` repeats the whole sequence. One pass can only use what the passes
// before it have already established, so a value that needs a later pass to
// become knowable is out of reach in a single sweep: folding a slot can settle
// a call, which settles an address, which folds another slot. Rounds stop as
// soon as one changes nothing, so a second round costs a sweep and no edits
// where there is nothing left to find.
[[nodiscard]] SsaPipelineRun RunSsaPipeline(const ir::SsaGraph&, const SsaDeclarations&,
                                            std::span<const ir::Group> decoded_sources,
                                            ir::ImageFacts, Budget&,
                                            std::span<const std::size_t> selection = {},
                                            unsigned rounds = 1);

}  // namespace nyx::passes
