#pragma once

#include "nyx/ir/image_facts.hpp"
#include "nyx/ir/ssa/graph.hpp"

namespace nyx::recovery {

// `contradicts_existing_fold`: the fold is sound by the run's facts, but the
// value it records lets the checker place a store on a fold the graph already
// carries, one the checker accepted only because it could not place that
// store before. The older fold is what is wrong, and it stays: removing it is
// not this pass's to do.
enum class ConstantLoadRefusal {
  none,
  invalid_graph,
  no_invariant,
  not_nonfaulting,
  alignment,
  conflicting_store,
  existing_omission,
  unsupported_access,
  resource_limit,
  contradicts_existing_fold
};

struct RefusedConstantLoad {
  ir::SsaHandle block;
  ir::ValueId node;
  ConstantLoadRefusal reason;
};

// The declared fact a fold read: a constant range, or a relocated slot whose
// loader-written value was declared stable. A recheck reads the same fact.
enum class ConstantLoadFact { constant_range, relocated_slot };

struct ConstantLoadEdit {
  ir::SsaHandle original_block;
  ir::SsaHandle result_block;
  ir::SsaConstantLoad fold;
  ConstantLoadFact fact;
  std::uint64_t fact_address;
  std::uint64_t fact_bytes;

  // Only an alignment-checked load relies on page-aligned placement.
  bool placement;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
  std::uint64_t alternative_fact_address = 0;
  std::uint64_t alternative_fact_bytes = 0;
  std::optional<ir::SsaBoundedTableAddressFact> table_address = std::nullopt;
};

struct ConstantLoadResult {
  // A candidate until an independent whole-function comparison accepts it.
  std::optional<ir::SsaGraph> provisional;

  // Every fold also rests on the declared access contract it was given.
  ir::ImageAccessContract access;
  std::vector<ConstantLoadEdit> journal;
  std::vector<RefusedConstantLoad> refused;
  ConstantLoadRefusal reason = ConstantLoadRefusal::none;
};

[[nodiscard]] ConstantLoadResult ProposeConstantImageLoads(const ir::SsaGraph&, ir::ImageFacts,
                                                           ir::ImageAccessContract, Budget&);

[[nodiscard]] ConstantLoadResult ProposeSelectedImageLoads(const ir::SsaGraph&,
                                                           const ir::SsaImageAddressFacts&,
                                                           ir::ImageFacts, ir::ImageAccessContract,
                                                           Budget&);

[[nodiscard]] ConstantLoadResult ProposeBoundedTableLoads(
    const ir::SsaGraph&, const ir::SsaBoundedTableAddressFact&,
    std::span<const ir::Group> decoded_sources, ir::ImageFacts, ir::ImageAccessContract, Budget&);
// Every fold in one candidate, validated once. A table fold stands only where
// every block that can run is complete, and a dispatch not yet folded is not,
// so a graph with two switches folds both or neither. A fact refused on its
// own is listed in `refused` and left out; the rest are proposed together,
// and if the candidate is refused, nothing is proposed.
[[nodiscard]] ConstantLoadResult ProposeBoundedTableLoads(
    const ir::SsaGraph&, std::span<const ir::SsaBoundedTableAddressFact>,
    std::span<const ir::Group> decoded_sources, ir::ImageFacts, ir::ImageAccessContract, Budget&);

}  // namespace nyx::recovery
