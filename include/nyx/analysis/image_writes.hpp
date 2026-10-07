#pragma once

#include <optional>
#include <span>
#include <vector>

#include "nyx/analysis/cfg.hpp"
#include "nyx/analysis/control.hpp"

namespace nyx::analysis {

// What the supplied code writes into the image. The population is whatever the
// caller hands over: scanning one function says only that this function does
// not contradict a declaration, while scanning every executable byte of an
// image is what can turn that into a statement about the image.
//
// Only refutation rests on this. A write it places removes a fact; a write it
// cannot place is counted, never silently dropped, because a declaration holds
// over the population only where those stores do not reach it.
struct ImageWriterReport {
  std::vector<ir::ImageWrite> writes;

  // The instruction behind each write, by the same index, so a report can say
  // where the write comes from rather than only what it lands on.
  std::vector<std::uint64_t> write_instructions;
  std::uint64_t unresolved_writes = 0;
  std::uint64_t scanned_groups = 0;

  // Groups with no modeled semantics. Their effects were not examined.
  std::uint64_t unmodeled_groups = 0;
};

// Sources are read in address order as straight-line runs, split at a transfer
// and wherever they stop being contiguous. A run is not a proved basic block:
// control may enter one partway, so an address this forms need not be one the
// program ever computes. That direction is safe, because the result is only
// ever used to withdraw a declaration, never to establish one.
[[nodiscard]] std::optional<ImageWriterReport> ScanImageWrites(std::span<const SourceRecord>,
                                                               const ImageFacts&, Budget&,
                                                               ir::BlockLimits = {});

}  // namespace nyx::analysis
