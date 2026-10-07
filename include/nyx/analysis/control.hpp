#pragma once

#include <span>
#include <utility>
#include <vector>

#include "nyx/ir/block.hpp"
#include "nyx/ir/image_facts.hpp"

namespace nyx::analysis {

// Declared image facts are shared vocabulary: recovery folds the same bytes
// this analysis reads, so they live in the IR layer both depend on.
using ir::ConstantImageRange;
using ir::ImageFacts;
using ir::RelocatedPointer;

enum class TargetKind { unknown, image_location, absolute_runtime };

struct SymbolicTarget {
  TargetKind kind = TargetKind::unknown;

  // image_location means load_bias + address modulo 2^64. An absolute_runtime
  // value is never implicitly an image coordinate, including at zero load bias.
  std::uint64_t address = 0;
  ir::ValueId value = 0;

  // See PathControlEdge: the read stays an executed load.
  bool constant_image_dependency = false;
};

enum class EdgeRole { branch, callee, return_, potential_return };

struct ControlEdge {
  EdgeRole role = EdgeRole::branch;
  SymbolicTarget target;
  std::optional<ir::ValueId> condition;
  std::optional<bool> when;
};

enum class ControlProof { all_inputs_successful_terminal };

struct ControlFacts {
  std::uint64_t block_revision;
  std::uint64_t terminal_source;
  ir::TransferKind kind;
  ControlProof proof = ControlProof::all_inputs_successful_terminal;
  std::array<ControlEdge, 3> edges{};
  unsigned edge_count = 0;

  // A potential_return is link metadata, never evidence of a returning callee.
  bool callee_return_unknown = false;
};

enum class ControlDecline { none, no_transfer, invalid_ir, resource_limit };

struct ControlResult {
  std::optional<ControlFacts> facts;
  ControlDecline reason = ControlDecline::none;
};

// What a normalized block writes to the image, as far as this abstraction can
// tell. `writes` is complete for addresses it resolves. `unresolved` counts
// stores whose address is derived from an image location but did not resolve
// to one: each could write any declared location, so a declaration holds over
// the block only where those stores do not reach it. Stores to addresses with
// no image provenance, such as a stack slot, are neither listed nor counted.
struct ImageWriteScan {
  std::vector<ir::ImageWrite> writes;
  std::uint64_t unresolved = 0;
};

// An empty optional means the budget declined, not that there were no stores.
[[nodiscard]] std::optional<ImageWriteScan> KnownImageWrites(const ir::Block&, const ImageFacts&,
                                                             Budget&);

// The value a conditional's false arm bounds, and the inclusive bound, when the
// condition is the lifted form of an unsigned comparison against a literal.
// Taking that arm is what proves the bound; taking the other proves nothing.
[[nodiscard]] std::optional<std::pair<ir::ValueId, std::uint64_t>> BoundFromGuard(
    std::span<const ir::Node>, ir::ValueId condition);

// Every destination `target` can reach when `index` is known to lie in
// [0, bound]. Empty unless every admissible value settles on an image location,
// so a partial set is never returned in place of a complete one.
struct BoundedTargets {
  std::vector<std::uint64_t> destinations;
  bool constant_image_dependency = false;
};

[[nodiscard]] BoundedTargets EnumerateBoundedTarget(std::span<const ir::Node>, ir::ValueId target,
                                                    ir::ValueId index, std::uint64_t bound,
                                                    const ImageFacts&, Budget&);

// Facts refer only to this block revision and successful execution of its last
// instruction. They prove neither graph reachability nor fault-free execution.
// A top-level target select may supply two guarded edges; nested unresolved
// expressions remain explicit unknown targets rather than guessed successors.
// Supplying constant image ranges lets a table-derived target resolve. The read
// stays an executed load and remains listed as a dependency; resolution is a
// fact about the destination, never permission to delete the access.
[[nodiscard]] ControlResult AnalyzeControl(const ir::Block&, Budget&, ir::BlockLimits = {},
                                           ImageFacts = {});

}  // namespace nyx::analysis
