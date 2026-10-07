#pragma once

#include "nyx/ir/image_facts.hpp"
#include "nyx/ir/path.hpp"

namespace nyx::recovery {

// constant_fold evaluates literal operands; image_offset moves an image location
// by a literal, or takes the literal distance between two of them, both of which
// hold at any load bias; page_base masks an image location to its page, which
// holds only under a declared page-aligned placement.
enum class ImageRule { constant_fold, image_offset, page_base };

struct ImageEdit {
  ImageRule rule;
  ir::ValueId node;
  ir::Node original;
  ir::Node replacement;

  // Declared restrictions the edit rests on, directly or through earlier edits
  // whose results it consumes: constant bytes, a relocated slot, or page-aligned
  // placement for anything downstream of a page base.
  bool constant_bytes = false;
  bool relocated_slot = false;
  bool page_placement = false;
  std::uint64_t from_revision = 0;
  std::uint64_t to_revision = 0;
};

// A declared constant range this path itself writes. The declaration says the
// bytes keep their file value for the run; a store the path resolves into them
// says otherwise. The two cannot both hold, so nothing here rests on that range
// and the contradiction is published rather than silently preferred either way.
// A store whose address does not resolve is not one of these: the declaration is
// exactly what covers the stores this analysis cannot place.
struct ContradictedRange {
  std::uint64_t range;    // the declared range's first address
  ir::ValueId store;      // the node that writes into it
  std::uint64_t address;  // the image location that store writes
  unsigned width;
};

struct ContradictedPointer {
  std::uint64_t slot;  // the relocated pointer slot's first address
  ir::ValueId store;
  std::uint64_t address;  // the image location that store writes
  unsigned width;
};

struct ImageLimits {
  ir::BlockLimits block{};
  std::uint32_t max_edits = 196608;
};

enum class ImageDecline { none, invalid_ir, resource_limit, revision_overflow };

struct ImagePathResult {
  std::optional<ir::Path> path;
  std::vector<ImageEdit> journal;

  // Declared values this path contradicts, in declaration order. They were
  // retracted before the published journal was produced.
  std::vector<ContradictedRange> contradicted;
  std::vector<ContradictedPointer> contradicted_pointers;

  // The facts with those values retracted. Every later stage must use these,
  // since a contradiction invalidates more than this pass's own journal.
  std::vector<ir::ConstantImageRange> constants;

  // Populated only when a pointer value was contradicted. Its slot remains to
  // block file-byte reads; the value is marked unstable. Otherwise the caller's
  // original pointer span is still valid and need not be copied.
  std::vector<ir::RelocatedPointer> pointers;
  ImageDecline reason = ImageDecline::none;

  std::span<const ir::RelocatedPointer> RetainedPointers(const ir::ImageFacts& original) const {
    return contradicted_pointers.empty() ? original.pointers
                                         : std::span<const ir::RelocatedPointer>(pointers);
  }
};

// Replaces, in place, each pure node whose operands are literals, image
// locations, or loads the supplied facts give a value, by the constant or image
// location it computes. Every load stays in the path and still executes; only
// its consumers change, never the access. A declared range or relocated slot
// this path visibly writes has its value retracted and reported, so no edit
// rests on a value the same path refutes. Declines publish no partial result.
[[nodiscard]] ImagePathResult FoldImageValues(const ir::Path&, const ir::ImageFacts&, Budget&,
                                              ImageLimits = {});

}  // namespace nyx::recovery
