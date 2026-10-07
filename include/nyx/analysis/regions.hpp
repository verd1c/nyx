#pragma once

#include "nyx/analysis/cfg.hpp"
#include "nyx/ir/path.hpp"

namespace nyx::analysis {

enum class RegionStop {
  unresolved,
  opaque,
  call,
  return_,
  cycle,
  second_fork,
  block_limit,
  source_limit,
  dispatch
};

struct Region {
  std::uint32_t entry_block;
  std::vector<std::uint32_t> block_ids;
  std::vector<std::uint32_t> source_ids;

  // Entry i selects an edge of block_ids[i] to block_ids[i + 1]. These
  // references belong to the owned CFG, never to a normalized Path's SSA.
  std::vector<std::uint32_t> transition_edges;
  RegionStop stop;
  std::optional<std::uint32_t> stopped_edge;
};

// Independent entry-scoped selections. No incoming edge is redirected and no
// reachability or exclusion of unknown interior entries is proved. Unselected
// exits, including conditional alternatives, remain in the owned original CFG.
class Regions {
 public:
  Regions(Cfg graph, std::vector<Region> candidates)
      : graph_(std::move(graph)), candidates_(std::move(candidates)) {}

  const Cfg& graph() const { return graph_; }

  std::span<const Region> candidates() const { return candidates_; }

 private:
  Cfg graph_;
  std::vector<Region> candidates_;
};

struct RegionLimits {
  std::uint32_t max_graph_blocks = 16384;
  std::uint32_t max_graph_sources = 65536;
  std::uint32_t max_graph_block_edges = 4096;
  std::uint32_t max_candidates = 32768;
  std::uint32_t max_blocks_per_region = 8;
  std::uint32_t max_sources_per_region = 512;
  std::uint64_t max_total_source_occurrences = 1024 * 1024;
};

enum class RegionDecline { none, invalid_graph, resource_limit };

struct RegionsResult {
  std::optional<Regions> regions;
  RegionDecline reason = RegionDecline::none;
};

// Validates structural source/edge identities. Supplied CFG facts are selection
// hints, not renewed semantic proofs; materialized original transfers remain
// authoritative. A source_limit at an oversized entry retains an empty selection
// and its entry_block rather than taking an incomplete machine block.
[[nodiscard]] RegionsResult BuildRegions(Cfg&&, Budget&, RegionLimits = {});
[[nodiscard]] ir::PathResult NormalizeRegion(const Regions&, std::size_t candidate, Budget&,
                                             ir::BlockLimits = {});

}  // namespace nyx::analysis
