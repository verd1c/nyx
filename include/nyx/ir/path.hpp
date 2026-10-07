#pragma once

#include "nyx/ir/block.hpp"

namespace nyx::ir {

// Sources enumerate a selected itinerary that may not be feasible. Every internal
// transfer remains executable; continuing to the next source requires its actual
// successor to match. Repeated source locations retain consistent original bytes.
class Path {
 public:
  Path(std::vector<Group> sources, std::vector<Node> nodes, std::vector<Origin> origins,
       std::vector<Boundary> boundaries, std::uint64_t revision = 0)
      : sources_(std::move(sources)),
        nodes_(std::move(nodes)),
        origins_(std::move(origins)),
        boundaries_(std::move(boundaries)),
        revision_(revision) {}

  std::span<const Group> sources() const { return sources_; }

  std::span<const Node> nodes() const { return nodes_; }

  std::span<const Origin> origins() const { return origins_; }

  std::span<const Boundary> boundaries() const { return boundaries_; }

  std::uint64_t revision() const { return revision_; }

  std::optional<std::uint64_t> expected_successor(std::size_t boundary) const {
    if (boundary >= sources_.size() || boundary == sources_.size() - 1) return {};
    return sources_[boundary + 1].source_address();
  }

 private:
  std::vector<Group> sources_;
  std::vector<Node> nodes_;
  std::vector<Origin> origins_;
  std::vector<Boundary> boundaries_;
  std::uint64_t revision_;
};

struct PathResult {
  std::optional<Path> path;
  BlockDecline reason = BlockDecline::none;
};

PathResult NormalizePath(std::span<const Group>, Budget&, BlockLimits = {});
BlockDecline ValidatePath(const Path&, Budget&, BlockLimits = {});

}  // namespace nyx::ir
