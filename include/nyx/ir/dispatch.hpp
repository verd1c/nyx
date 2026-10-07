#pragma once

#include "nyx/ir/image_facts.hpp"
#include "nyx/support/budget.hpp"

namespace nyx::ir {

// A read of declared image bytes a resolved destination rests on. The load stays
// in the path and still executes; this records only what the resolution read, so
// a consumer can publish the dependency and a checker can read the same bytes
// for itself instead of trusting the value it is handed.
struct ImageRead {
  ValueId node;

  // The arm of the choice under which this read resolved. One load resolves
  // once per arm, because the arm is what supplies its address.
  bool when;
  std::uint64_t address;
  unsigned width;
  std::uint64_t value;

  // The loader, not the file, supplied the bytes, so the value is an image
  // location rather than a number.
  bool relocated = false;
};

// A computed transfer whose destination rests on one unresolved choice, with
// both arms settling on a place in the image: the conditional branch that
// flattening replaced by a table index. Addresses are image locations, so they
// hold at any load bias.
struct DispatchBranch {
  ValueId condition;
  std::uint64_t when_true;
  std::uint64_t when_false;

  // Every read the two resolutions rest on, in arm then node order.
  std::vector<ImageRead> witness;
};

// An exhausted budget is kept apart from a transfer that simply hides no
// branch, so a caller cannot publish a smaller answer than it would have found.
enum class DispatchDecline { none, no_branch, resource_limit };

struct DispatchResolution {
  std::optional<DispatchBranch> branch;
  DispatchDecline reason = DispatchDecline::none;
};

// Pins the single unresolved choice `target` rests on to each arm in turn and
// evaluates the rest of its expression under the declared facts. Nothing unless
// exactly one choice is unresolved, both arms settle on an image location, and
// the two differ: equal arms are no branch, and a second unresolved choice would
// make this a claim about a product of arms rather than about one condition.
// Nothing is read but the declared bytes, and nothing is rewritten.
[[nodiscard]] DispatchResolution ResolveDispatchBranch(std::span<const Node> nodes, ValueId target,
                                                       const ImageFacts&, Budget&);

}  // namespace nyx::ir
