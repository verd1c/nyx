#pragma once

#include "nyx/analysis/cfg.hpp"
#include "nyx/ir/storage_relation.hpp"

namespace nyx::analysis {

// Restrictions a caller declares for the proof. None is inferred from the code.
struct RelationAssumptions {
  // A declared calling convention: every callee returns only through its call's
  // continuation with `preserved` as it found them. An untargeted return's
  // destination requires a separate population-leaving declaration.
  bool calling_convention = false;
  std::span<const ir::StorageId> preserved;  // sorted
  bool return_leaves = false;
};

// Indexed by block. A block the known entries do not reach has no relations.
// Every relation rests on everything the proof followed, wherever it was, since
// a wrong edge anywhere admits arrivals the proof never saw.
struct EntryRelations {
  std::uint64_t source_identity = 0;
  std::vector<std::vector<ir::StorageRelation>> blocks;

  // The call-preservation set used by the proof, retained for source-bound replay.
  std::vector<ir::StorageId> preserved;
  bool calling_convention = false;       // a call or return was reachable
  bool constant_image = false;           // an edge rests on declared image values
  bool declared_opaque_control = false;  // an opaque source's declared control
  bool return_leaves = false;            // an untargeted return was reachable
};

enum class RelationDecline { none, invalid_input, resource_limit };

struct EntryRelationsResult {
  std::optional<EntryRelations> relations;
  RelationDecline reason = RelationDecline::none;
};

// Relations between 64-bit storage values that hold on entry to each block on
// every path the graph admits from its known entries, a value being related only
// through a copy or a literal displacement. The scope is the graph's own: an
// undiscovered interior entry is not excluded. An unresolved transfer could
// arrive anywhere with anything, so while one is reachable no block has any.
[[nodiscard]] EntryRelationsResult ProveEntryRelations(const Cfg&, Budget&,
                                                       RelationAssumptions = {});

}  // namespace nyx::analysis
