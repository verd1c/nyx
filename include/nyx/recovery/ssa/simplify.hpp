#pragma once

#include <span>
#include <string_view>

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::recovery {

enum class SsaSimplifyRefusal { none, invalid_graph, resource_limit };

struct SsaSimplifyEdit {
  ir::SsaHandle original_block;
  ir::SsaHandle result_block;
  ir::ValueId node;
  ir::Node original;
  ir::Node replacement;
  std::string_view rule;

  // The rewrite reads an image page as its image value, which holds only
  // under the run's page-aligned placement declaration.
  bool placement = false;

  // The rewrite read a folded image load as its declared literal, so it rests
  // on that fold's declarations as the fold does.
  bool folded_value = false;

  // Pure nodes spliced in just before `node`, which the replacement uses.
  std::uint32_t inserted = 0;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
};

struct SsaSimplifyResult {
  std::optional<ir::SsaGraph> provisional;
  std::vector<SsaSimplifyEdit> journal;

  // A rewrite can make a store's address image-based, so the graph places a
  // store that writes a declared value a path read carries. That read's
  // record is dropped, named here, rather than the candidate refused.
  std::vector<ir::SsaDroppedPathRead> dropped_path_reads;
  SsaSimplifyRefusal reason = SsaSimplifyRefusal::none;
};

// Rewrites pure nodes in place by identities that hold for every input at
// every width: neutral and absorbing operands, x op x, cast round trips,
// boolean selects and comparisons, literal folding and image-address
// arithmetic, reading a load folded to a literal as that literal. A node
// equal to an existing value becomes a same-width zext of
// it, the IR's copy. A few rewrites splice in up to two new pure nodes: sign
// extensions, literal chains and truncated bitwise functions rebuilt at their
// narrow width. Folding an image page needs `page_aligned_placement`.
// Values whose shape another record rereads (transfer targets, the address
// cones of declared or folded loads, frame accesses, recovered rewrites) are
// left as they are, so those records keep binding.
[[nodiscard]] SsaSimplifyResult ProposeSsaSimplify(const ir::SsaGraph&,
                                                   std::span<const ir::Group> decoded_sources,
                                                   bool page_aligned_placement, Budget&);

}  // namespace nyx::recovery
