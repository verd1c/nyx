#pragma once

#include <span>

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::recovery {

enum class SsaDeadLoadRefusal { none, invalid_graph, resource_limit };

struct SsaDeadLoadEdit {
  ir::SsaHandle original_block;
  ir::SsaHandle result_block;
  ir::SsaRetiredLoad retired;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
};

struct SsaDeadLoadResult {
  std::optional<ir::SsaGraph> provisional;
  std::vector<SsaDeadLoadEdit> journal;
  SsaDeadLoadRefusal reason = SsaDeadLoadRefusal::none;
};

// Retires loads whose values nothing uses when they cannot fault: an earlier
// store of the same block wrote the same bytes, or every address they can
// read lies in one declared read-only image range or relocated slot and
// `access` declares those bytes mapped with unobservable reads. Loads that are already folded,
// promoted or omitted are left alone.
[[nodiscard]] SsaDeadLoadResult ProposeSsaDeadLoads(const ir::SsaGraph&,
                                                    std::span<const ir::Group> decoded_sources,
                                                    ir::ImageFacts, ir::ImageAccessContract,
                                                    Budget&);

}  // namespace nyx::recovery
