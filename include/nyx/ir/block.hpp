#pragma once

#include "nyx/ir/group.hpp"
#include "nyx/support/budget.hpp"

namespace nyx::ir {

struct Origin {
  std::uint32_t boundary;
  ValueId operation;
};

struct Boundary {
  ValueId first_node;
  std::uint32_t node_count;
  std::vector<Write> writes;
  std::optional<Transfer> transfer;
};

// A single-entry straight-line SSA region, distinct from the immutable machine
// groups. Boundaries retain register publication and faults at each instruction.
class Block {
 public:
  Block(std::vector<Group> sources, std::vector<Node> nodes, std::vector<Origin> origins,
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

 private:
  std::vector<Group> sources_;
  std::vector<Node> nodes_;
  std::vector<Origin> origins_;
  std::vector<Boundary> boundaries_;
  std::uint64_t revision_;
};

struct BlockLimits {
  std::uint32_t max_groups = 4096;
  std::uint32_t max_nodes = 65536;
  std::uint32_t max_storage = 256;
  unsigned max_width = 4096;
};

enum class BlockDecline { none, invalid_source, invalid_ir, unsupported_control, resource_limit };

struct BlockResult {
  std::optional<Block> block;
  BlockDecline reason = BlockDecline::none;
};

// No entry into an interior source boundary is represented. Loads keep distinct
// definitions even at identical addresses; register reads use each group's entry.
BlockResult Normalize(std::span<const Group> sources, Budget&, BlockLimits = {});
BlockDecline Validate(const Block&, Budget&, BlockLimits = {});

}  // namespace nyx::ir
