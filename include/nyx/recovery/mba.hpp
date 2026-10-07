#pragma once

#include "nyx/ir/block.hpp"
#include "nyx/ir/path.hpp"
#include "nyx/ir/ssa/graph.hpp"

namespace nyx::recovery {

enum class MbaRule {
  or_xor_sum,
  and_xor_union,
  or_and_sum,
  xor_and_sum,
  xor_ones_not,
  negated_add,
  constant_fold,
  linear_identity,
  linear_direct
};

enum class MbaDecline { none, invalid_ir, resource_limit, revision_overflow };

struct MbaLimits {
  ir::BlockLimits block{};
  std::uint32_t max_edits = 196608;
};

struct MbaEdit {
  MbaRule rule;
  ir::ValueId node;
  ir::Node original;
  ir::Node replacement;
  unsigned width;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
};

struct MbaResult {
  std::optional<ir::Block> block;
  std::vector<MbaEdit> journal;
  MbaDecline reason = MbaDecline::none;
};

// Edits pure RHS nodes in place by identity, retaining every source boundary,
// memory operation and architectural write. Declines publish no partial result.
[[nodiscard]] MbaResult SimplifyMba(const ir::Block&, Budget&, MbaLimits = {});

struct MbaPathResult {
  std::optional<ir::Path> path;
  std::vector<MbaEdit> journal;
  MbaDecline reason = MbaDecline::none;
};

[[nodiscard]] MbaPathResult SimplifyMba(const ir::Path&, Budget&, MbaLimits = {});

// A bounded signature search for a one-operation result over free values or
// earlier bitwise subexpressions. It leaves expressions without that form unchanged.
[[nodiscard]] MbaResult SimplifyLinearMba(const ir::Block&, Budget&, MbaLimits = {});
[[nodiscard]] MbaPathResult SimplifyLinearMba(const ir::Path&, Budget&, MbaLimits = {});

struct SsaMbaEdit {
  ir::SsaHandle original_block;
  ir::SsaHandle result_block;
  MbaEdit edit;
};

struct SsaMbaResult {
  std::optional<ir::SsaGraph> provisional;
  std::vector<SsaMbaEdit> journal;
  MbaDecline reason = MbaDecline::none;
};

[[nodiscard]] SsaMbaResult ProposeLinearMba(const ir::SsaGraph&, Budget&, MbaLimits = {});

enum class SsaMbaSynthesisRefusalReason { outside_uniform_linear_scope, search_bound };

struct SsaMbaSynthesisRefusal {
  ir::SsaHandle block;
  ir::ValueId node;
  SsaMbaSynthesisRefusalReason reason;
};

struct SsaMbaSynthesisEdit {
  ir::SsaHandle original_block;
  ir::SsaHandle result_block;
  ir::ValueId original_node;
  ir::ValueId result_node;
  ir::Node original;
  ir::Node replacement;
  std::vector<ir::Node> inserted;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
};

struct SsaMbaSynthesisResult {
  std::optional<ir::SsaGraph> provisional;
  std::vector<SsaMbaSynthesisEdit> journal;
  std::vector<SsaMbaSynthesisRefusal> refused;

  // Refusals past the listed ones, counted but not listed.
  std::uint64_t refused_unlisted = 0;
  MbaDecline reason = MbaDecline::none;
};

[[nodiscard]] SsaMbaSynthesisResult ProposeCanonicalLinearMba(const ir::SsaGraph&, Budget&,
                                                              MbaLimits = {});

}  // namespace nyx::recovery
