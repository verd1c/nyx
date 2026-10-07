#pragma once

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::recovery {

enum class SsaPureDceRefusal { none, stale_proof, invalid_graph, resource_limit };

struct SsaPureDceEdit {
  ir::SsaHandle original_block;
  ir::SsaHandle result_block;
  ir::ValueId node;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
};

struct SsaPureDceResult {
  std::optional<ir::SsaGraph> provisional;
  std::vector<SsaPureDceEdit> journal;
  SsaPureDceRefusal reason = SsaPureDceRefusal::none;
};

[[nodiscard]] SsaPureDceResult ProposeDeadPureNodes(const ir::SsaGraph&,
                                                    const ir::SsaLivenessFacts&, Budget&);
[[nodiscard]] SsaPureDceResult ProposeDeadPureNodes(const ir::SsaGraph&,
                                                    const ir::SsaLivenessFacts&,
                                                    std::span<const ir::Group> decoded_sources,
                                                    Budget&);

}  // namespace nyx::recovery
