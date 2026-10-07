#include "nyx/passes/pipeline.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <limits>
#include <set>
#include <string_view>

#include "nyx/analysis/frame_slots.hpp"
#include "nyx/analysis/ssa/cache.hpp"
#include "nyx/analysis/ssa/constants.hpp"
#include "nyx/analysis/ssa/copy.hpp"
#include "nyx/analysis/ssa/dead_writes.hpp"
#include "nyx/analysis/ssa/image_addresses.hpp"
#include "nyx/analysis/ssa/index_bound.hpp"
#include "nyx/analysis/ssa/liveness.hpp"
#include "nyx/analysis/ssa/phi_constants.hpp"
#include "nyx/analysis/ssa/reachability.hpp"
#include "nyx/analysis/ssa/sccp.hpp"
#include "nyx/analysis/ssa/table_address.hpp"
#include "nyx/ir/ssa/print.hpp"
#include "nyx/recovery/constant_load.hpp"
#include "nyx/recovery/dead_state.hpp"
#include "nyx/recovery/frame_promotion.hpp"
#include "nyx/recovery/mba.hpp"
#include "nyx/recovery/ssa/branch_retirement.hpp"
#include "nyx/recovery/ssa/constants.hpp"
#include "nyx/recovery/ssa/copy.hpp"
#include "nyx/recovery/ssa/dce.hpp"
#include "nyx/recovery/ssa/dead_loads.hpp"
#include "nyx/recovery/ssa/dead_writes.hpp"
#include "nyx/recovery/ssa/leaf_calls.hpp"
#include "nyx/recovery/ssa/loop_exit.hpp"
#include "nyx/recovery/ssa/phi_constants.hpp"
#include "nyx/recovery/ssa/pure_dce.hpp"
#include "nyx/recovery/ssa/resolved_calls.hpp"
#include "nyx/recovery/ssa/simplify.hpp"

namespace nyx::passes {
namespace {
class Json {
 public:
  explicit Json(Budget& budget) : budget_(budget) {}

  Json& raw(std::string_view text) {
    if (Charge(1, text.size())) out_ += text;
    return *this;
  }

  Json& number(std::uint64_t value) {
    char digits[20];
    const auto end = std::to_chars(digits, digits + sizeof(digits), value);
    return raw({digits, end.ptr});
  }

  Json& signed_number(std::int64_t value) {
    char digits[21];
    const auto end = std::to_chars(digits, digits + sizeof(digits), value);
    return raw({digits, end.ptr});
  }

  Json& boolean(bool value) { return raw(value ? "true" : "false"); }

  Json& string(std::string_view text) {
    if (failed_) return *this;
    std::size_t bytes = 2;
    for (const unsigned char ch : text) {
      const auto count = ch == '"' || ch == '\\' ? 2U : ch < 0x20 ? 6U : 1U;
      if (bytes > std::numeric_limits<std::size_t>::max() - count) {
        failed_ = true;
        return *this;
      }

      bytes += count;
    }

    if (!Charge(text.size(), bytes)) return *this;
    out_.reserve(out_.size() + bytes);
    out_ += '"';
    for (const char ch : text) {
      if (ch == '"' || ch == '\\') {
        out_ += '\\';
        out_ += ch;
      } else if (ch == '\n')
        out_ += "\\n";
      else if (static_cast<unsigned char>(ch) < 0x20) {
        constexpr char hex[] = "0123456789abcdef";
        out_ += "\\u00";
        out_ += hex[(ch >> 4) & 15];
        out_ += hex[ch & 15];
      } else
        out_ += ch;
    }

    out_ += '"';
    return *this;
  }

  Json& handle(ir::SsaHandle handle) {
    return raw("\"b").number(handle.slot).raw(".").number(handle.generation).raw("\"");
  }

  std::string& text() { return out_; }

  bool failed() const { return failed_; }

 private:
  bool Charge(std::size_t work, std::size_t bytes) {
    if (failed_) return false;
    if (bytes > std::numeric_limits<std::size_t>::max() - out_.size() ||
        budget_.try_consume({work, bytes}) != BudgetDecline::none) {
      failed_ = true;
      return false;
    }

    return true;
  }

  Budget& budget_;
  bool failed_ = false;
  std::string out_;
};

// A list writer that emits its separators, so every stage prints the same way.
class List {
 public:
  // `count`, when given, tallies the entries written, as journals do.
  explicit List(Json& json, std::size_t* count = nullptr) : json_(json), count_(count) {}

  Json& next() {
    if (!first_) json_.raw(",");
    first_ = false;
    if (count_) ++*count_;
    return json_;
  }

 private:
  Json& json_;
  std::size_t* count_;
  bool first_ = true;
};

template <class T>
bool AppendCharged(std::vector<T>& values, T value, Budget& budget) {
  if (values.size() == values.capacity()) {
    if (values.size() == SIZE_MAX || values.size() + 1 > UINT64_MAX / sizeof(T) ||
        budget.try_consume({values.size() + 1, (values.size() + 1) * sizeof(T)}) !=
            BudgetDecline::none)
      return false;
    values.reserve(values.size() + 1);
  } else if (budget.try_consume({1, 0}) != BudgetDecline::none) {
    return false;
  }

  values.push_back(std::move(value));
  return true;
}

const char* Name(recovery::ConstantLoadRefusal reason) {
  using R = recovery::ConstantLoadRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::no_invariant:
      return "no_invariant";
    case R::not_nonfaulting:
      return "not_nonfaulting";
    case R::alignment:
      return "alignment";
    case R::conflicting_store:
      return "conflicting_store";
    case R::existing_omission:
      return "existing_omission";
    case R::unsupported_access:
      return "unsupported_access";
    case R::resource_limit:
      return "resource_limit";
    case R::contradicts_existing_fold:
      return "contradicts_existing_fold";
  }

  return "invalid";
}

const char* Name(analysis::SsaImageAddressRefusal reason) {
  using R = analysis::SsaImageAddressRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::resource_limit:
      return "resource_limit";
    case R::not_selected_image:
      return "not_selected_image";
  }

  return "invalid_graph";
}

const char* Name(analysis::SsaIndexBoundRefusal reason) {
  using R = analysis::SsaIndexBoundRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::guard_unproved:
      return "guard_unproved";
    case R::predicate_unbound:
      return "predicate_unbound";
    case R::unsupported_condition:
      return "unsupported_condition";
    case R::changed_index:
      return "changed_index";
    case R::invalid_graph:
      return "invalid_graph";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid_graph";
}

const char* Name(analysis::SsaTableAddressRefusal reason) {
  using R = analysis::SsaTableAddressRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::index_unbounded:
      return "index_unbounded";
    case R::source_unbound:
      return "source_unbound";
    case R::unsupported_address:
      return "unsupported_address";
    case R::index_changed:
      return "index_changed";
    case R::address_overflow:
      return "address_overflow";
    case R::invalid_graph:
      return "invalid_graph";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid_graph";
}

const char* Name(recovery::MbaDecline reason) {
  using R = recovery::MbaDecline;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_ir:
      return "invalid_graph";
    case R::resource_limit:
      return "resource_limit";
    case R::revision_overflow:
      return "revision_overflow";
  }

  return "invalid";
}

const char* Name(recovery::SsaMbaSynthesisRefusalReason reason) {
  using R = recovery::SsaMbaSynthesisRefusalReason;
  switch (reason) {
    case R::outside_uniform_linear_scope:
      return "outside_uniform_linear_scope";
    case R::search_bound:
      return "search_bound";
  }

  return "invalid";
}

const char* Name(analysis::PrivateFrameDecline reason) {
  using R = analysis::PrivateFrameDecline;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::missing_contract:
      return "missing_contract";
    case R::unknown_call:
      return "unknown_call";
    case R::opaque_effect:
      return "opaque_effect";
    case R::unknown_continuation:
      return "unknown_continuation";
    case R::escaped_address:
      return "escaped_address";
    case R::unknown_offset:
      return "unknown_offset";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid";
}

const char* Name(analysis::SsaBoundedLoopRefusal reason) {
  using R = analysis::SsaBoundedLoopRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid";
}

const char* Name(recovery::PromotionRefusal reason) {
  using R = recovery::PromotionRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::stale_proof:
      return "stale_proof";
    case R::invalid_graph:
      return "invalid_graph";
    case R::existing_omission:
      return "existing_omission";
    case R::already_promoted:
      return "already_promoted";
    case R::unsupported_access:
      return "unsupported_access";
    case R::incompatible_access:
      return "incompatible_access";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid";
}

const char* Name(recovery::DeadStateRefusal reason) {
  using R = recovery::DeadStateRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::stale_proof:
      return "stale_proof";
    case R::invalid_graph:
      return "invalid_graph";
    case R::slot_used:
      return "slot_used";
    case R::unsupported_effect:
      return "unsupported_effect";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid";
}

const char* Name(analysis::SsaLivenessRefusal reason) {
  using R = analysis::SsaLivenessRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid";
}

const char* Name(recovery::SsaPureDceRefusal reason) {
  using R = recovery::SsaPureDceRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::stale_proof:
      return "stale_proof";
    case R::invalid_graph:
      return "invalid_graph";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid";
}

const char* Name(analysis::SsaReachabilityRefusal reason) {
  using R = analysis::SsaReachabilityRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::open_entries:
      return "open_entries";
    case R::incomplete_successors:
      return "incomplete_successors";
    case R::invalid_graph:
      return "invalid_graph";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid_graph";
}

const char* Name(recovery::SsaDceRefusal reason) {
  using R = recovery::SsaDceRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::stale_proof:
      return "stale_proof";
    case R::invalid_graph:
      return "invalid_graph";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid_graph";
}

const char* Name(analysis::SsaPhiConstantRefusal reason) {
  using R = analysis::SsaPhiConstantRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::stale_reachability:
      return "stale_reachability";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid_graph";
}

const char* Name(recovery::SsaPhiFoldRefusal reason) {
  using R = recovery::SsaPhiFoldRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::stale_proof:
      return "stale_proof";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid_graph";
}

const char* Name(analysis::SsaConstantRefusal reason) {
  using R = analysis::SsaConstantRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::stale_reachability:
      return "stale_reachability";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid_graph";
}

const char* Name(analysis::SsaSccpRefusal reason) {
  using R = analysis::SsaSccpRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::open_entries:
      return "open_entries";
    case R::incomplete_successors:
      return "incomplete_successors";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid_graph";
}

const char* Name(analysis::SsaCopyRefusal reason) {
  using R = analysis::SsaCopyRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::stale_reachability:
      return "stale_reachability";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid_graph";
}

const char* Name(recovery::SsaCopyRefusal reason) {
  using R = recovery::SsaCopyRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::stale_proof:
      return "stale_proof";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid_graph";
}

const char* Name(analysis::SsaDeadWriteRefusal reason) {
  using R = analysis::SsaDeadWriteRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::stale_reachability:
      return "stale_reachability";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid_graph";
}

const char* Name(recovery::SsaDeadWriteRefusal reason) {
  using R = recovery::SsaDeadWriteRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::stale_proof:
      return "stale_proof";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid_graph";
}

const char* Name(recovery::SsaConstantFoldRefusal reason) {
  using R = recovery::SsaConstantFoldRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::stale_proof:
      return "stale_proof";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid_graph";
}

const char* Name(recovery::SsaResolvedCallRefusal reason) {
  using R = recovery::SsaResolvedCallRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid_graph";
}

const char* Name(recovery::SsaLeafCallRefusal reason) {
  using R = recovery::SsaLeafCallRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid_graph";
}

const char* Name(recovery::SsaLeafCallDecline reason) {
  using R = recovery::SsaLeafCallDecline;
  switch (reason) {
    case R::no_body:
      return "no_body";
    case R::not_leaf:
      return "not_leaf";
    case R::live_result:
      return "live_result";
  }

  return "not_leaf";
}

const char* Name(recovery::SsaLoopExitRefusal reason) {
  using R = recovery::SsaLoopExitRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::stale_proof:
      return "stale_proof";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid_graph";
}

const char* Name(recovery::SsaLoopExitDecline reason) {
  using R = recovery::SsaLoopExitDecline;
  switch (reason) {
    case R::unsupported_control:
      return "unsupported_control";
    case R::unsupported_effect:
      return "unsupported_effect";
    case R::live_state:
      return "live_state";
  }

  return "unsupported_control";
}

const char* Name(recovery::SsaBranchRetirementRefusal reason) {
  using R = recovery::SsaBranchRetirementRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::stale_proof:
      return "stale_proof";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid_graph";
}

const char* Name(recovery::SsaSimplifyRefusal reason) {
  using R = recovery::SsaSimplifyRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid";
}

const char* Name(recovery::SsaDeadLoadRefusal reason) {
  using R = recovery::SsaDeadLoadRefusal;
  switch (reason) {
    case R::none:
      return "none";
    case R::invalid_graph:
      return "invalid_graph";
    case R::resource_limit:
      return "resource_limit";
  }

  return "invalid";
}

const char* Name(recovery::SsaDceEditKind kind) {
  using K = recovery::SsaDceEditKind;
  switch (kind) {
    case K::pruned_storage_phi:
      return "pruned_storage_phi";
    case K::pruned_frame_phi:
      return "pruned_frame_phi";
    case K::removed_block:
      return "removed_block";
  }

  return "invalid";
}

// Opens a stage record. `outcome` is proposed, unchanged, declined or not_run.
void Open(Json& json, std::string_view pass, const char* outcome, const char* reason,
          std::uint64_t from, std::optional<std::uint64_t> to) {
  json.raw("{\"pass\":")
      .string(pass)
      .raw(",\"outcome\":")
      .string(outcome)
      .raw(",\"reason\":")
      .string(reason)
      .raw(",\"from_revision\":")
      .number(from)
      .raw(",\"to_revision\":");
  if (to)
    json.number(*to);
  else
    json.raw("null");
}

void FrameAssumptions(List& list, const ir::PrivateFrameBasis& basis) {
  const auto& contract = basis.contract;
  list.next().string("entry SP" + std::string(contract.begin < 0 ? "" : "+") +
                     std::to_string(contract.begin) + " to entry SP" +
                     (contract.end < 0 ? "" : "+") + std::to_string(contract.end) +
                     " is fresh, mapped, writable memory that no external alias or"
                     " asynchronous observer reaches, addressed only by arithmetic on SP");
  list.next().string("SP is " + std::to_string(contract.sp_alignment) + "-byte aligned on entry");
  if (contract.callees_cannot_touch)
    list.next().string(
        "no callee reads or writes that frame range or hands back an address into it");
  if (contract.callees_preserve_sp)
    list.next().string("every callee returns with SP as it found it");
  if (basis.entry_relations)
    list.next().string("storage relations proved from the known entries hold at each block entry");
  if (basis.relation_declared_abi)
    list.next().string("calls and returns keep the declared ABI's discipline");
  if (basis.relation_return_leaves)
    list.next().string("caller declares untargeted returns leave the selected population");
  if (basis.relation_constant_image)
    list.next().string("declared image values keep their stated values");
  if (basis.relation_declared_opaque_control)
    list.next().string("caller-declared opaque control behavior matches the source instruction");
}

void FrameContract(Json& json, const ir::PrivateFrameContract& contract) {
  json.raw("{\"begin\":")
      .signed_number(contract.begin)
      .raw(",\"end\":")
      .signed_number(contract.end)
      .raw(",\"sp_alignment\":")
      .number(contract.sp_alignment)
      .raw(",\"fresh_mapped_writable\":")
      .boolean(contract.fresh_mapped_writable)
      .raw(",\"no_external_aliases\":")
      .boolean(contract.no_external_aliases)
      .raw(",\"no_async_observers\":")
      .boolean(contract.no_async_observers)
      .raw(",\"callees_cannot_touch\":")
      .boolean(contract.callees_cannot_touch)
      .raw(",\"callees_preserve_sp\":")
      .boolean(contract.callees_preserve_sp)
      .raw("}");
}

}  // namespace

class SsaPassContext {
 public:
  SsaPassContext(const ir::SsaGraph& base, const SsaDeclarations& declared,
                 std::span<const ir::Group> sources, ir::ImageFacts facts, Budget& budget)
      : base(base),
        declared(declared),
        sources(sources),
        facts(facts),
        budget(budget),
        json(budget),
        stages(json),
        proof_cache(sources, ir::SsaEntryScope::closed_population) {}

  const ir::SsaGraph& current() const { return owned ? *owned : base; }

  // Declared reads the batch's folds dropped because a store they let the
  // graph place writes them: the driver reruns only when a withdrawn byte is
  // one the graph read.
  std::vector<ir::SsaDroppedPathRead> dropped_path_reads;

  // The declarations every closed-population proof over this graph uses.
  void ControlAssumptions(List& assumptions) {
    assumptions.next().string(
        "every reachable successor is proved complete from its block:"
        " direct and recovered transfers, calls and declared traps, or a"
        " declared leaving return");
    if (has_call)
      assumptions.next().string(
          "every callee returns only through its call's continuation,"
          " and re-enters the population only at a listed entry");
    if (has_noreturn)
      assumptions.next().string("caller-declared non-returning callees never return");
    if (has_trap)
      assumptions.next().string(
          "a trap never resumes inside the population except at a listed entry");
    if (has_declared_reads)
      assumptions.next().string(
          "declared image values keep their stated values, including"
          " loader-written relocated slots");
    if (has_placed_reads)
      assumptions.next().string("the image is placed at a page-aligned address");
  }

  // Opens the running pass's stage record under its registered name.
  void Begin(const char* outcome, const char* reason, std::uint64_t from,
             std::optional<std::uint64_t> to) {
    Open(stages.next(), SsaPassRegistry()[pass].name, outcome, reason, from, to);
    const std::string_view name = outcome;
    summary = {pass,
               name == "proposed"    ? SsaStageOutcome::proposed
               : name == "unchanged" ? SsaStageOutcome::unchanged
               : name == "declined"  ? SsaStageOutcome::declined
                                     : SsaStageOutcome::not_run,
               reason, from, to};
    opened = true;
  }

  bool UnreachableBlocks() {
    const auto& proof = proof_cache.Reachability(current(), budget);
    if (proof.reason == analysis::SsaReachabilityRefusal::resource_limit) return false;
    if (!proof.facts) {
      Begin("declined", Name(proof.reason), current().revision(), std::nullopt);
      json.raw(",\"assumptions\":[],\"journal\":[],\"refused\":[]}");
    } else {
      auto result = recovery::ProposeUnreachableBlocks(current(), *proof.facts, sources, budget);
      if (result.reason == recovery::SsaDceRefusal::resource_limit) return false;
      const char* outcome = result.provisional                               ? "proposed"
                            : result.reason != recovery::SsaDceRefusal::none ? "declined"
                                                                             : "unchanged";
      Begin(outcome, Name(result.reason), current().revision(),
            result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
      json.raw(",\"assumptions\":[");
      List assumptions(json);
      assumptions.next().string("the listed entries exhaust incoming function entries");
      ControlAssumptions(assumptions);
      if (has_return)
        assumptions.next().string(
            "caller declares untargeted returns leave the selected population");
      json.raw("],\"journal\":[");
      List journal(json, &journaled);
      for (const auto& edit : result.journal) {
        journal.next()
            .raw("{\"kind\":")
            .string(Name(edit.kind))
            .raw(",\"block\":")
            .handle(edit.original_block)
            .raw(",\"result_block\":");
        if (edit.result_block)
          json.handle(*edit.result_block);
        else
          json.raw("null");
        json.raw(",\"index\":").number(edit.index).raw(",\"predecessor\":");
        if (edit.kind == recovery::SsaDceEditKind::removed_block) {
          json.raw("null");
          ++removed_blocks;
        } else
          json.handle(edit.predecessor);
        json.raw(",\"from_revision\":")
            .number(edit.from_revision)
            .raw(",\"to_revision\":")
            .number(edit.to_revision)
            .raw("}");
      }

      json.raw("],\"refused\":[]}");
      if (result.provisional) owned = std::move(result.provisional);
    }

    return true;
  }

  bool PhiConstantFold() {
    const auto& reachable = proof_cache.Reachability(current(), budget);
    if (reachable.reason == analysis::SsaReachabilityRefusal::resource_limit) return false;
    if (!reachable.facts) {
      Begin("declined", Name(reachable.reason), current().revision(), std::nullopt);
      json.raw(",\"assumptions\":[],\"proved_phis\":0,\"journal\":[],\"refused\":[]}");
    } else {
      auto proved = analysis::ProveSsaPhiConstants(current(), *reachable.facts, sources, budget);
      if (proved.reason == analysis::SsaPhiConstantRefusal::resource_limit) return false;
      if (!proved.facts) {
        Begin("declined", Name(proved.reason), current().revision(), std::nullopt);
        json.raw(",\"assumptions\":[],\"proved_phis\":0,\"journal\":[],\"refused\":[]}");
      } else {
        auto result = recovery::ProposeSsaPhiConstantFold(current(), *reachable.facts,
                                                          *proved.facts, sources, budget);
        if (result.reason == recovery::SsaPhiFoldRefusal::resource_limit) return false;
        const char* outcome = result.provisional                                   ? "proposed"
                              : result.reason != recovery::SsaPhiFoldRefusal::none ? "declined"
                                                                                   : "unchanged";
        Begin(outcome, Name(result.reason), current().revision(),
              result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
        json.raw(",\"assumptions\":[");
        List assumptions(json);
        assumptions.next().string("the listed entries exhaust incoming function entries");
        ControlAssumptions(assumptions);
        assumptions.next().string(
            "the current SSA graph comes from decoded groups and"
            " preceding journaled stages in this provisional batch");
        if (has_return)
          assumptions.next().string(
              "caller declares untargeted returns leave the selected population");
        json.raw("],\"proved_phis\":").number(proved.facts->constants.size()).raw(",\"journal\":[");
        List journal(json, &journaled);
        for (const auto& edit : result.journal)
          journal.next()
              .raw("{\"block\":")
              .handle(edit.original_block)
              .raw(",\"result_block\":")
              .handle(edit.result_block)
              .raw(",\"node\":")
              .number(edit.node)
              .raw(",\"original_op\":")
              .string(ir::Descriptor(edit.original.op)->name)
              .raw(",\"value\":")
              .number(edit.value)
              .raw(",\"from_revision\":")
              .number(edit.from_revision)
              .raw(",\"to_revision\":")
              .number(edit.to_revision)
              .raw("}");
        json.raw("],\"refused\":[]}");
        if (result.provisional) owned = std::move(result.provisional);
      }
    }

    return true;
  }

  bool ConstantPropagation() {
    const auto& reachable = proof_cache.Reachability(current(), budget);
    if (reachable.reason == analysis::SsaReachabilityRefusal::resource_limit) return false;
    if (!reachable.facts) {
      Begin("declined", Name(reachable.reason), current().revision(), std::nullopt);
      json.raw(
          ",\"assumptions\":[],\"bounded_loops\":[],\"bounded_loop_refusal\":\"none\",\"proved_"
          "phis\":0,"
          "\"proved_nodes\":0,\"journal\":[],\"refused\":[]}");
    } else {
      auto proved = analysis::ProveSsaConstants(current(), *reachable.facts, sources, budget,
                                                proof_cache.Loops(current(), budget));
      if (proved.reason == analysis::SsaConstantRefusal::resource_limit) return false;
      if (!proved.facts) {
        Begin("declined", Name(proved.reason), current().revision(), std::nullopt);
        json.raw(
            ",\"assumptions\":[],\"bounded_loops\":[],\"bounded_loop_refusal\":\"none\",\"proved_"
            "phis\":0,"
            "\"proved_nodes\":0,\"journal\":[],\"refused\":[]}");
      } else {
        auto result =
            recovery::ProposeSsaConstantFold(current(), *reachable.facts, *proved.facts, sources,
                                             budget, proof_cache.Loops(current(), budget));
        if (result.reason == recovery::SsaConstantFoldRefusal::resource_limit) return false;
        const char* outcome = result.provisional ? "proposed"
                              : result.reason != recovery::SsaConstantFoldRefusal::none
                                  ? "declined"
                                  : "unchanged";
        Begin(outcome, Name(result.reason), current().revision(),
              result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
        // Declared reads the folds let the graph place a store on, whose
        // records the candidate no longer carries.
        json.raw(",\"dropped_path_reads\":[");
        {
          List dropped(json);
          for (const auto& read : result.dropped_path_reads)
            dropped.next()
                .raw("{\"block\":")
                .handle(read.block)
                .raw(",\"node\":")
                .number(read.node)
                .raw(",\"address\":")
                .number(read.address)
                .raw("}");
        }

        json.raw("]");
        dropped_path_reads.insert(dropped_path_reads.end(), result.dropped_path_reads.begin(),
                                  result.dropped_path_reads.end());
        json.raw(",\"assumptions\":[");
        List assumptions(json);
        assumptions.next().string("the listed entries exhaust incoming function entries");
        ControlAssumptions(assumptions);
        assumptions.next().string(
            "the current SSA graph comes from decoded groups and"
            " preceding journaled stages in this provisional batch");
        if (has_return)
          assumptions.next().string(
              "caller declares untargeted returns leave the selected population");
        if (budget.try_consume({proved.facts->values.size(), 0}) != BudgetDecline::none)
          return false;
        const auto kind_count = [&](ir::SsaValueKind kind) {
          return std::count_if(
              proved.facts->values.begin(), proved.facts->values.end(),
              [&](const ir::SsaConstantValue& value) { return value.kind == kind; });
        };

        const auto phis = kind_count(ir::SsaValueKind::phi);
        const auto slots = kind_count(ir::SsaValueKind::frame_phi);

        // What running each self-looping block established. The lattice above
        // reads these as values its join cannot reach, so a reader checking a
        // constant that has no incoming to agree with finds it here.
        json.raw("],\"bounded_loops\":[");
        {
          List runs(json);
          if (const auto* loops = proof_cache.Loops(current(), budget)) {
            if (budget.try_consume({loops->loops.size(), 0}) != BudgetDecline::none) return false;
            for (const auto& loop : loops->loops)
              runs.next()
                  .raw("{\"block\":")
                  .handle(loop.loop)
                  .raw(",\"successor\":")
                  .handle(loop.successor)
                  .raw(",\"iterations\":")
                  .number(loop.iterations)
                  .raw(",\"values\":")
                  .number(loop.exits.size())
                  .raw("}");
          }
        }

        json.raw("],\"bounded_loop_refusal\":")
            .string(Name(proof_cache.loops_reason()))
            .raw(",\"proved_phis\":")
            .number(phis)
            .raw(",\"proved_frame_slots\":")
            .number(slots)
            .raw(",\"proved_nodes\":")
            .number(proved.facts->values.size() - phis - slots)
            .raw(",\"journal\":[");
        List journal(json, &journaled);
        for (const auto& edit : result.journal)
          journal.next()
              .raw("{\"block\":")
              .handle(edit.original_block)
              .raw(",\"result_block\":")
              .handle(edit.result_block)
              .raw(",\"node\":")
              .number(edit.node)
              .raw(",\"original_op\":")
              .string(ir::Descriptor(edit.original.op)->name)
              .raw(",\"value\":")
              .number(edit.value)
              .raw(",\"from_revision\":")
              .number(edit.from_revision)
              .raw(",\"to_revision\":")
              .number(edit.to_revision)
              .raw("}");
        json.raw("],\"refused\":[]}");
        if (result.provisional) owned = std::move(result.provisional);
      }
    }

    return true;
  }

  bool SccpConstantFold() {
    const auto& proved = proof_cache.Sccp(current(), budget);
    if (proved.reason == analysis::SsaSccpRefusal::resource_limit) return false;
    if (!proved.facts) {
      Begin("declined", Name(proved.reason), current().revision(), std::nullopt);
      json.raw(
          ",\"assumptions\":[],\"executable_blocks\":0,\"selected_edges\":0,"
          "\"proved_phis\":0,\"proved_nodes\":0,\"excluded_edges\":[],"
          "\"journal\":[],\"refused\":[]}");
    } else {
      auto result = recovery::ProposeSsaSccpFold(current(), *proved.facts, sources, budget,
                                                 proof_cache.Loops(current(), budget));
      if (result.reason == recovery::SsaConstantFoldRefusal::resource_limit) return false;
      const char* outcome = result.provisional                                        ? "proposed"
                            : result.reason != recovery::SsaConstantFoldRefusal::none ? "declined"
                                                                                      : "unchanged";
      Begin(outcome, Name(result.reason), current().revision(),
            result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
      // Declared reads the folds let the graph place a store on, whose
      // records the candidate no longer carries.
      json.raw(",\"dropped_path_reads\":[");
      {
        List dropped(json);
        for (const auto& read : result.dropped_path_reads)
          dropped.next()
              .raw("{\"block\":")
              .handle(read.block)
              .raw(",\"node\":")
              .number(read.node)
              .raw(",\"address\":")
              .number(read.address)
              .raw("}");
      }

      json.raw("]");
      dropped_path_reads.insert(dropped_path_reads.end(), result.dropped_path_reads.begin(),
                                result.dropped_path_reads.end());
      json.raw(",\"assumptions\":[");
      List assumptions(json);
      assumptions.next().string("the listed entries exhaust incoming function entries");
      ControlAssumptions(assumptions);
      assumptions.next().string(
          "the current SSA graph comes from decoded groups and"
          " preceding journaled stages in this provisional batch");
      if (has_return)
        assumptions.next().string(
            "caller declares untargeted returns leave the selected population");
      if (budget.try_consume({proved.facts->executable.size(), 0}) != BudgetDecline::none)
        return false;
      const auto executable =
          std::count(proved.facts->executable.begin(), proved.facts->executable.end(), 1);
      std::size_t selected = 0;
      for (const auto& edges : proved.facts->edges) {
        if (edges.size() == std::numeric_limits<std::size_t>::max() ||
            selected > std::numeric_limits<std::size_t>::max() - edges.size() ||
            budget.try_consume({edges.size() + 1, 0}) != BudgetDecline::none)
          return false;
        selected += std::count(edges.begin(), edges.end(), 1);
      }

      if (budget.try_consume({proved.facts->constants.values.size(), 0}) != BudgetDecline::none)
        return false;
      const auto kind_count = [&](ir::SsaValueKind kind) {
        return std::count_if(proved.facts->constants.values.begin(),
                             proved.facts->constants.values.end(),
                             [&](const ir::SsaConstantValue& value) { return value.kind == kind; });
      };

      const auto phis = kind_count(ir::SsaValueKind::phi);
      const auto slots = kind_count(ir::SsaValueKind::frame_phi);
      json.raw("],\"executable_blocks\":")
          .number(executable)
          .raw(",\"selected_edges\":")
          .number(selected)
          .raw(",\"proved_phis\":")
          .number(phis)
          .raw(",\"proved_frame_slots\":")
          .number(slots)
          .raw(",\"proved_nodes\":")
          .number(proved.facts->constants.values.size() - phis - slots)
          .raw(",\"excluded_edges\":[");
      List excluded(json);
      for (std::size_t slot = 0; slot < current().slots(); ++slot) {
        if (budget.try_consume({1, 0}) != BudgetDecline::none) return false;
        if (!proved.facts->executable[slot]) continue;
        const auto handle = current().Handle(slot);
        if (!handle) return false;
        const auto& block = *current().Get(*handle);
        for (std::size_t index = 0; index < block.edges.size(); ++index) {
          if (budget.try_consume({1, 0}) != BudgetDecline::none) return false;
          if (proved.facts->edges[slot][index]) continue;
          const auto& edge = block.edges[index];
          if (!edge.condition || !edge.when || !edge.target_block ||
              budget.try_consume({proved.facts->constants.values.size(), 0}) != BudgetDecline::none)
            return false;
          const auto constant = std::find_if(
              proved.facts->constants.values.begin(), proved.facts->constants.values.end(),
              [&](const ir::SsaConstantValue& value) {
                return value.block == *handle && value.kind == ir::SsaValueKind::node &&
                       value.index == *edge.condition;
              });
          if (constant == proved.facts->constants.values.end()) return false;
          excluded.next()
              .raw("{\"block\":")
              .handle(*handle)
              .raw(",\"edge\":")
              .number(index)
              .raw(",\"target_block\":")
              .handle(*edge.target_block)
              .raw(",\"condition\":")
              .number(*edge.condition)
              .raw(",\"condition_value\":")
              .number(constant->value)
              .raw(",\"when\":")
              .boolean(*edge.when)
              .raw(",\"graph_revision\":")
              .number(current().revision())
              .raw("}");
        }
      }

      json.raw("],\"journal\":[");
      List journal(json, &journaled);
      for (const auto& edit : result.journal)
        journal.next()
            .raw("{\"block\":")
            .handle(edit.original_block)
            .raw(",\"result_block\":")
            .handle(edit.result_block)
            .raw(",\"node\":")
            .number(edit.node)
            .raw(",\"original_op\":")
            .string(ir::Descriptor(edit.original.op)->name)
            .raw(",\"value\":")
            .number(edit.value)
            .raw(",\"from_revision\":")
            .number(edit.from_revision)
            .raw(",\"to_revision\":")
            .number(edit.to_revision)
            .raw("}");
      json.raw("],\"refused\":[]}");
      if (result.provisional) owned = std::move(result.provisional);
    }

    return true;
  }

  bool ConstantImageLoads() {
    const ir::ImageAccessContract access{declared.image_access, declared.image_access,
                                         declared.image_access};
    auto result = recovery::ProposeConstantImageLoads(current(), facts, access, budget);
    if (result.reason == recovery::ConstantLoadRefusal::resource_limit) return false;
    const char* outcome = result.provisional                                     ? "proposed"
                          : result.reason != recovery::ConstantLoadRefusal::none ? "declined"
                                                                                 : "unchanged";
    Begin(outcome, Name(result.reason), current().revision(),
          result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
    json.raw(",\"assumptions\":[");
    List assumptions(json);
    if (result.provisional) {
      assumptions.next().string(
          "every folded image byte stays mapped readable for the run,"
          " and reading it has no observable effect");
      if (std::any_of(result.journal.begin(), result.journal.end(),
                      [](const auto& edit) { return edit.placement; }))
        assumptions.next().string("the image is placed at a page-aligned address");
      constexpr std::size_t kSetNodeBytes =
          sizeof(std::pair<std::uint64_t, std::uint64_t>) + 4 * sizeof(void*) + 16;
      if (result.journal.size() > std::numeric_limits<std::size_t>::max() / kSetNodeBytes ||
          budget.try_consume({result.journal.size(), result.journal.size() * kSetNodeBytes}) !=
              BudgetDecline::none)
        return false;
      std::set<std::pair<std::uint64_t, std::uint64_t>> ranges, slots;
      for (const auto& edit : result.journal)
        (edit.fact == recovery::ConstantLoadFact::constant_range ? ranges : slots)
            .insert({edit.fact_address, edit.fact_bytes});
      for (const auto& [address, bytes] : ranges)
        assumptions.next().string("the image range " + std::to_string(address) + "+" +
                                  std::to_string(bytes) + " keeps its file-initialized value");
      for (const auto& slot : slots)
        assumptions.next().string("the relocated slot " + std::to_string(slot.first) +
                                  " keeps the value the loader wrote");
    }

    json.raw("],\"journal\":[");
    List journal(json, &journaled);
    for (const auto& edit : result.journal) {
      // The pass only folds nodes of the graph it was handed.
      const auto& node = current().Get(edit.original_block)->nodes[edit.fold.node];
      journal.next()
          .raw("{\"block\":")
          .handle(edit.original_block)
          .raw(",\"result_block\":")
          .handle(edit.result_block)
          .raw(",\"node\":")
          .number(edit.fold.node)
          .raw(",\"width\":")
          .number(node.width)
          .raw(",\"byte_order\":")
          .string(node.access.byte_order == ir::ByteOrder::little ? "little" : "big")
          .raw(",\"kind\":")
          .string(edit.fold.kind == ir::SsaConstantKind::literal ? "literal" : "image_location")
          .raw(",\"value\":")
          .number(edit.fold.value)
          .raw(",\"retired_access\":")
          .boolean(edit.fold.skip_access)
          .raw(",\"source_address\":")
          .number(edit.fold.source_address)
          .raw(",\"fact\":")
          .string(edit.fact == recovery::ConstantLoadFact::constant_range ? "constant_range"
                                                                          : "relocated_slot")
          .raw(",\"fact_address\":")
          .number(edit.fact_address)
          .raw(",\"fact_bytes\":")
          .number(edit.fact_bytes)
          .raw(",\"read_only\":")
          .boolean(edit.fold.read_only)
          .raw(",\"page_aligned_placement\":")
          .boolean(edit.placement)
          .raw("}");
    }

    json.raw("],\"refused\":[");
    List refused(json);
    for (const auto& item : result.refused)
      refused.next()
          .raw("{\"block\":")
          .handle(item.block)
          .raw(",\"node\":")
          .number(item.node)
          .raw(",\"reason\":")
          .string(Name(item.reason))
          .raw("}");
    json.raw("]}");
    if (result.provisional) owned = std::move(result.provisional);
    return true;
  }

  bool SelectedImageLoads() {
    auto proved = analysis::ProveSsaSelectedImageAddresses(current(), budget);
    if (proved.reason == analysis::SsaImageAddressRefusal::resource_limit) return false;
    if (!proved.facts) {
      Begin("declined", Name(proved.reason), current().revision(), std::nullopt);
      json.raw(",\"assumptions\":[],\"journal\":[],\"refused\":[]}");
    } else {
      const ir::ImageAccessContract access{declared.image_access, declared.image_access,
                                           declared.image_access};
      auto result =
          recovery::ProposeSelectedImageLoads(current(), *proved.facts, facts, access, budget);
      if (result.reason == recovery::ConstantLoadRefusal::resource_limit) return false;
      const char* outcome = result.provisional                                     ? "proposed"
                            : result.reason != recovery::ConstantLoadRefusal::none ? "declined"
                                                                                   : "unchanged";
      Begin(outcome, Name(result.reason), current().revision(),
            result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
      json.raw(",\"assumptions\":[");
      List assumptions(json);
      if (result.provisional) {
        assumptions.next().string(
            "every selected image byte stays mapped readable for the run,"
            " and reading it has no observable effect");
        for (const auto& edit : result.journal) {
          assumptions.next().string("the image range " + std::to_string(edit.fact_address) + "+" +
                                    std::to_string(edit.fact_bytes) +
                                    " keeps its file-initialized value");
          assumptions.next().string(
              "the image range " + std::to_string(edit.alternative_fact_address) + "+" +
              std::to_string(edit.alternative_fact_bytes) + " keeps its file-initialized value");
          if (edit.placement)
            assumptions.next().string("the image is placed at a page-aligned address");
        }
      }

      json.raw("],\"journal\":[");
      List journal(json, &journaled);
      for (const auto& edit : result.journal)
        journal.next()
            .raw("{\"block\":")
            .handle(edit.original_block)
            .raw(",\"result_block\":")
            .handle(edit.result_block)
            .raw(",\"node\":")
            .number(edit.fold.node)
            .raw(",\"condition\":")
            .number(*edit.fold.condition)
            .raw(",\"width\":")
            .number(current().Get(edit.original_block)->nodes[edit.fold.node].width)
            .raw(",\"retired_access\":true,\"read_only\":")
            .boolean(edit.fold.read_only)
            .raw(",\"when_true\":{\"address\":")
            .number(edit.fold.source_address)
            .raw(",\"value\":")
            .number(edit.fold.value)
            .raw(",\"fact_address\":")
            .number(edit.fact_address)
            .raw(",\"fact_bytes\":")
            .number(edit.fact_bytes)
            .raw("},\"when_false\":{\"address\":")
            .number(edit.fold.alternative_source_address)
            .raw(",\"value\":")
            .number(edit.fold.alternative_value)
            .raw(",\"fact_address\":")
            .number(edit.alternative_fact_address)
            .raw(",\"fact_bytes\":")
            .number(edit.alternative_fact_bytes)
            .raw("}}");
      json.raw("],\"refused\":[");
      List refused(json);
      for (const auto& item : proved.refused)
        refused.next()
            .raw("{\"block\":")
            .handle(item.block)
            .raw(",\"node\":")
            .number(item.load)
            .raw(",\"reason\":")
            .string(Name(item.reason))
            .raw("}");
      for (const auto& item : result.refused)
        refused.next()
            .raw("{\"block\":")
            .handle(item.block)
            .raw(",\"node\":")
            .number(item.node)
            .raw(",\"reason\":")
            .string(Name(item.reason))
            .raw("}");
      json.raw("]}");
      if (result.provisional) owned = std::move(result.provisional);
    }

    return true;
  }

  bool LinearMba() {
    auto result = recovery::ProposeLinearMba(current(), budget);
    if (result.reason == recovery::MbaDecline::resource_limit) return false;
    const char* outcome = result.provisional                            ? "proposed"
                          : result.reason != recovery::MbaDecline::none ? "declined"
                                                                        : "unchanged";
    Begin(outcome, Name(result.reason), current().revision(),
          result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
    // A linear identity holds at every width and input; it adds no assumption.
    json.raw(",\"assumptions\":[],\"journal\":[");
    List journal(json, &journaled);
    for (const auto& item : result.journal) {
      const auto& edit = item.edit;
      journal.next()
          .raw("{\"block\":")
          .handle(item.original_block)
          .raw(",\"result_block\":")
          .handle(item.result_block)
          .raw(",\"node\":")
          .number(edit.node)
          .raw(",\"rule\":\"linear_direct\",\"width\":")
          .number(edit.width)
          .raw(",\"original\":{\"op\":")
          .string(ir::Descriptor(edit.original.op)->name)
          .raw("},\"replacement\":{\"op\":")
          .string(ir::Descriptor(edit.replacement.op)->name)
          .raw(",\"inputs\":[");
      const auto arity = ir::Descriptor(edit.replacement.op)->arity;
      for (unsigned i = 0; i < arity; ++i) {
        if (i) json.raw(",");
        json.number(edit.replacement.inputs[i]);
      }

      json.raw("],\"immediate\":").number(edit.replacement.immediate).raw("}}");
    }

    json.raw("],\"refused\":[]}");
    if (result.provisional) owned = std::move(result.provisional);
    return true;
  }

  bool CanonicalLinearMba() {
    auto result = recovery::ProposeCanonicalLinearMba(current(), budget);
    if (result.reason == recovery::MbaDecline::resource_limit) return false;
    const char* outcome = result.provisional                            ? "proposed"
                          : result.reason != recovery::MbaDecline::none ? "declined"
                                                                        : "unchanged";
    Begin(outcome, Name(result.reason), current().revision(),
          result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
    json.raw(",\"assumptions\":[],\"journal\":[");
    List journal(json, &journaled);
    for (const auto& edit : result.journal) {
      journal.next()
          .raw("{\"block\":")
          .handle(edit.original_block)
          .raw(",\"result_block\":")
          .handle(edit.result_block)
          .raw(",\"node\":")
          .number(edit.original_node)
          .raw(",\"result_node\":")
          .number(edit.result_node)
          .raw(",\"original_op\":")
          .string(ir::Descriptor(edit.original.op)->name)
          .raw(",\"replacement_op\":")
          .string(ir::Descriptor(edit.replacement.op)->name)
          .raw(",\"replacement_inputs\":[");
      const auto arity = ir::Descriptor(edit.replacement.op)->arity;
      for (unsigned input = 0; input < arity; ++input) {
        if (input) json.raw(",");
        json.number(edit.replacement.inputs[input]);
      }

      json.raw("],\"inserted\":[");
      List inserted(json);
      for (const auto& node : edit.inserted) {
        inserted.next().raw("{\"op\":").string(ir::Descriptor(node.op)->name).raw(",\"inputs\":[");
        const auto inputs = ir::Descriptor(node.op)->arity;
        for (unsigned input = 0; input < inputs; ++input) {
          if (input) json.raw(",");
          json.number(node.inputs[input]);
        }

        json.raw("]}");
      }

      json.raw("],\"from_revision\":")
          .number(edit.from_revision)
          .raw(",\"to_revision\":")
          .number(edit.to_revision)
          .raw("}");
    }

    json.raw("],\"refused\":[");
    List refused(json);
    for (const auto& item : result.refused)
      refused.next()
          .raw("{\"block\":")
          .handle(item.block)
          .raw(",\"node\":")
          .number(item.node)
          .raw(",\"reason\":")
          .string(Name(item.reason))
          .raw("}");
    json.raw("],\"refused_unlisted\":").number(result.refused_unlisted).raw("}");
    if (result.provisional) owned = std::move(result.provisional);
    return true;
  }

  // The frame stages each prove the slots of the graph they read, so the
  // second stage proves again over the first's output.
  bool FrameStage(auto propose) {
    auto proof = analysis::ProvePrivateFrameSlots(current(), *declared.frame, budget);
    if (proof.reason == analysis::PrivateFrameDecline::resource_limit) return false;
    if (!proof.facts) {
      Begin("declined", Name(proof.reason), current().revision(), std::nullopt);
      json.raw(",\"assumptions\":[],\"journal\":[],\"refused\":[]}");
      return true;
    }

    return propose(*proof.facts);
  }

  void FrameRefusals(const auto& refused) {
    json.raw("],\"refused\":[");
    List list(json);
    for (const auto& item : refused)
      list.next()
          .raw("{\"offset\":")
          .signed_number(item.offset)
          .raw(",\"size\":")
          .number(item.size)
          .raw(",\"reason\":")
          .string(Name(item.reason))
          .raw("}");
    json.raw("]}");
  }

  bool DeadPrivateState() {
    return FrameStage([&](const ir::PrivateFrameFacts& proved) {
      auto result = recovery::ProposeDeadPrivateState(current(), proved, budget);
      if (result.reason == recovery::DeadStateRefusal::resource_limit) return false;
      const char* outcome = result.provisional                                  ? "proposed"
                            : result.reason != recovery::DeadStateRefusal::none ? "declined"
                                                                                : "unchanged";
      Begin(outcome, Name(result.reason), current().revision(),
            result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
      json.raw(",\"assumptions\":[");
      List assumptions(json);
      if (result.basis) FrameAssumptions(assumptions, *result.basis);
      json.raw("],\"journal\":[");
      List journal(json, &journaled);
      for (const auto& edit : result.journal) {
        journal.next()
            .raw("{\"block\":")
            .handle(edit.original_block)
            .raw(",\"result_block\":")
            .handle(edit.result_block)
            .raw(",\"node\":")
            .number(edit.node)
            .raw(",\"op\":")
            .string(ir::Descriptor(edit.operation)->name)
            .raw(",\"offset\":")
            .signed_number(edit.slot_offset)
            .raw(",\"size\":")
            .number(edit.slot_size)
            .raw("}");
      }

      FrameRefusals(result.refused);
      if (result.provisional) owned = std::move(result.provisional);
      return true;
    });
  }

  bool FramePromotion() {
    return FrameStage([&](const ir::PrivateFrameFacts& proved) {
      auto result = recovery::ProposeFramePromotion(current(), proved, budget);
      if (result.reason == recovery::PromotionRefusal::resource_limit) return false;
      const char* outcome = result.provisional                                  ? "proposed"
                            : result.reason != recovery::PromotionRefusal::none ? "declined"
                                                                                : "unchanged";
      Begin(outcome, Name(result.reason), current().revision(),
            result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
      json.raw(",\"assumptions\":[");
      List assumptions(json);
      if (result.basis) FrameAssumptions(assumptions, *result.basis);
      json.raw("],\"journal\":[");
      List journal(json, &journaled);
      for (const auto& edit : result.journal) {
        journal.next()
            .raw("{\"block\":")
            .handle(edit.original_block)
            .raw(",\"result_block\":")
            .handle(edit.result_block)
            .raw(",\"node\":")
            .number(edit.node)
            .raw(",\"offset\":")
            .signed_number(edit.slot_offset)
            .raw(",\"size\":")
            .number(edit.slot_size)
            .raw(",\"access\":")
            .string(edit.replacement ? "load" : "store")
            .raw("}");
      }

      FrameRefusals(result.refused);
      if (result.provisional) owned = std::move(result.provisional);
      return true;
    });
  }

  bool BoundedTableLoads() {
    struct Edit {
      ir::SsaHandle guard;
      ir::SsaHandle block;
      ir::SsaHandle result_block;
      ir::ValueId node;
      std::uint64_t base;
      std::uint64_t stride;
      std::uint64_t rows;
      std::uint64_t fact_address;
      std::uint64_t fact_bytes;
      std::uint64_t from_revision;
      std::uint64_t to_revision;
      bool placement;
    };

    struct Refused {
      ir::SsaHandle guard;
      ir::SsaHandle block;
      ir::ValueId node;
      const char* phase;
      const char* reason;
    };

    const auto from_revision = current().revision();
    std::vector<Edit> edits;
    std::vector<Refused> refused;
    {
      const auto refuse = [&](ir::SsaHandle guard, ir::SsaHandle block, ir::ValueId node,
                              const char* phase, const char* reason) {
        return AppendCharged(refused, Refused{guard, block, node, phase, reason}, budget);
      };

      struct Candidate {
        ir::SsaHandle guard;
        ir::SsaBoundedTableAddressFact address;
      };

      std::vector<Candidate> candidates;
      for (std::size_t slot = 0; slot < current().slots(); ++slot) {
        if (budget.try_consume({1, 0}) != BudgetDecline::none) return false;
        const auto guard = current().Handle(slot);
        if (!guard) continue;
        const auto* guard_block = current().Get(*guard);
        for (std::uint32_t edge_index = 0; edge_index < guard_block->edges.size(); ++edge_index) {
          if (budget.try_consume({1, 0}) != BudgetDecline::none) return false;
          const auto& edge = guard_block->edges[edge_index];
          if (!edge.target_block || !edge.condition || !edge.when) continue;
          const auto table = *edge.target_block;
          const auto* table_block = current().Get(table);
          if (!table_block) continue;

          // The bound belongs to the guard edge, not to a load, so one proof
          // serves every load of the table block.
          std::optional<analysis::SsaIndexBoundResult> bound;
          for (ir::ValueId node = 0; node < table_block->nodes.size(); ++node) {
            if (budget.try_consume({1 + table_block->disabled_effects.size(), 0}) !=
                BudgetDecline::none)
              return false;
            if (table_block->nodes[node].op != ir::Op::load ||
                std::find(table_block->disabled_effects.begin(),
                          table_block->disabled_effects.end(),
                          node) != table_block->disabled_effects.end())
              continue;
            if (!bound)
              bound = analysis::ProveSsaDirectIndexBound(current(), sources,
                                                         ir::SsaEntryScope::closed_population,
                                                         *guard, edge_index, table, budget);
            if (!bound->fact) {
              if (bound->reason == analysis::SsaIndexBoundRefusal::resource_limit ||
                  !refuse(*guard, table, node, "index_bound", Name(bound->reason)))
                return false;
              continue;
            }

            const auto address = analysis::ProveSsaBoundedTableAddress(current(), sources,
                                                                       *bound->fact, node, budget);
            if (!address.fact) {
              if (address.reason == analysis::SsaTableAddressRefusal::resource_limit ||
                  !refuse(*guard, table, node, "table_address", Name(address.reason)))
                return false;
              continue;
            }

            if (!AppendCharged(candidates, Candidate{*guard, *address.fact}, budget)) return false;
          }
        }
      }

      // Every table is folded in one candidate: a fold is kept only where no
      // dispatch is left unfolded. When the candidate is refused, the fold
      // that spoils it may be one alone, so each is left out in turn, a
      // bounded number of times, before the whole set is refused.
      const ir::ImageAccessContract access{true, true, true};
      const auto propose =
          [&](std::optional<std::size_t> omit) -> std::optional<recovery::ConstantLoadResult> {
        std::vector<ir::SsaBoundedTableAddressFact> addresses;
        if (candidates.size() > SIZE_MAX / sizeof(ir::SsaBoundedTableAddressFact) ||
            budget.try_consume(
                {candidates.size(), candidates.size() * sizeof(ir::SsaBoundedTableAddressFact)}) !=
                BudgetDecline::none)
          return {};
        for (std::size_t index = 0; index < candidates.size(); ++index)
          if (index != omit) addresses.push_back(candidates[index].address);
        auto folded = recovery::ProposeBoundedTableLoads(current(), addresses, sources, facts,
                                                         access, budget);
        if (folded.reason == recovery::ConstantLoadRefusal::resource_limit) return {};
        return folded;
      };

      const auto guard_of = [&](ir::SsaHandle block, ir::ValueId node) {
        const auto found =
            std::find_if(candidates.begin(), candidates.end(), [&](const Candidate& item) {
              return item.address.index_bound.guard.guarded_block == block &&
                     item.address.load == node;
            });
        return found != candidates.end() ? found->address.index_bound.guard.branch : block;
      };

      if (!candidates.empty()) {
        auto folded = propose(std::nullopt);
        if (!folded) return false;
        std::optional<std::size_t> omitted;
        constexpr std::size_t kRetries = 16;
        if (!folded->provisional && candidates.size() > 1) {
          std::size_t tries = 0;
          for (std::size_t index = 0; index < candidates.size() && tries < kRetries; ++index) {
            if (budget.try_consume({1 + folded->refused.size(), 0}) != BudgetDecline::none)
              return false;
            // Leaving out a fold the set already refused changes nothing.
            const auto& address = candidates[index].address;
            if (std::any_of(folded->refused.begin(), folded->refused.end(),
                            [&](const recovery::RefusedConstantLoad& item) {
                              return item.block == address.index_bound.guard.guarded_block &&
                                     item.node == address.load;
                            }))
              continue;
            ++tries;
            auto retry = propose(index);
            if (!retry) return false;
            if (retry->provisional) {
              folded = std::move(retry);
              omitted = index;
              break;
            }
          }
        }

        for (const auto& item : folded->refused)
          if (!refuse(guard_of(item.block, item.node), item.block, item.node, "constant_load",
                      Name(item.reason)))
            return false;
        if (omitted) {
          const auto& left = candidates[*omitted].address;
          if (!refuse(candidates[*omitted].guard, left.index_bound.guard.guarded_block, left.load,
                      "constant_load", "invalid_graph"))
            return false;
        }

        if (!folded->provisional) {
          // All-or-nothing: every fold the set held is refused with it.
          for (const auto& candidate : candidates) {
            const auto& address = candidate.address;
            if (std::any_of(folded->refused.begin(), folded->refused.end(),
                            [&](const recovery::RefusedConstantLoad& item) {
                              return item.block == address.index_bound.guard.guarded_block &&
                                     item.node == address.load;
                            }))
              continue;
            if (!refuse(candidate.guard, address.index_bound.guard.guarded_block, address.load,
                        "constant_load", Name(folded->reason)))
              return false;
          }
        } else {
          for (const auto& journal : folded->journal)
            if (!AppendCharged(
                    edits,
                    Edit{guard_of(journal.original_block, journal.fold.node),
                         journal.original_block, journal.result_block, journal.fold.node,
                         journal.fold.source_address, journal.fold.table_stride,
                         journal.fold.table_bytes.size(), journal.fact_address, journal.fact_bytes,
                         journal.from_revision, journal.to_revision, journal.placement},
                    budget))
              return false;
          owned = std::move(folded->provisional);
        }
      }
    }

    const char* outcome = !edits.empty() ? "proposed" : !refused.empty() ? "declined" : "unchanged";
    const char* reason = edits.empty() && !refused.empty() ? "candidate_refused" : "none";
    Begin(outcome, reason, from_revision,
          edits.empty() ? std::nullopt : std::optional(current().revision()));
    json.raw(",\"assumptions\":[");
    List assumptions(json);
    if (!edits.empty()) {
      assumptions.next().string("the listed entries exhaust incoming function entries");
      ControlAssumptions(assumptions);
      if (has_return)
        assumptions.next().string(
            "caller declares untargeted returns leave the selected population");
      assumptions.next().string(
          "every folded table row stays mapped readable for the run,"
          " and reading it has no observable effect");
      for (const auto& edit : edits) {
        assumptions.next().string("the image range " + std::to_string(edit.fact_address) + "+" +
                                  std::to_string(edit.fact_bytes) +
                                  " keeps its file-initialized value");
        if (edit.placement)
          assumptions.next().string("the image is placed at a page-aligned address");
      }
    }

    json.raw("],\"journal\":[");
    List journal(json, &journaled);
    for (const auto& edit : edits)
      journal.next()
          .raw("{\"guard\":")
          .handle(edit.guard)
          .raw(",\"block\":")
          .handle(edit.block)
          .raw(",\"result_block\":")
          .handle(edit.result_block)
          .raw(",\"node\":")
          .number(edit.node)
          .raw(",\"base\":")
          .number(edit.base)
          .raw(",\"stride\":")
          .number(edit.stride)
          .raw(",\"rows\":")
          .number(edit.rows)
          .raw(",\"retired_access\":true,\"fact_address\":")
          .number(edit.fact_address)
          .raw(",\"fact_bytes\":")
          .number(edit.fact_bytes)
          .raw(",\"from_revision\":")
          .number(edit.from_revision)
          .raw(",\"to_revision\":")
          .number(edit.to_revision)
          .raw("}");
    json.raw("],\"refused\":[");
    List refusals(json);
    for (const auto& item : refused)
      refusals.next()
          .raw("{\"guard\":")
          .handle(item.guard)
          .raw(",\"block\":")
          .handle(item.block)
          .raw(",\"node\":")
          .number(item.node)
          .raw(",\"phase\":")
          .string(item.phase)
          .raw(",\"reason\":")
          .string(item.reason)
          .raw("}");
    json.raw("]}");
    return true;
  }

  bool SccpStorageReads() {
    const auto& proved = proof_cache.Sccp(current(), budget);
    if (proved.reason == analysis::SsaSccpRefusal::resource_limit) return false;
    if (!proved.facts) {
      Begin("declined", Name(proved.reason), current().revision(), std::nullopt);
      json.raw(
          ",\"assumptions\":[],\"selected_edges\":0,\"excluded_edges\":[],"
          "\"journal\":[],\"refused\":[]}");
    } else {
      auto result = recovery::ProposeSsaSccpReadFold(current(), *proved.facts, sources, budget,
                                                     proof_cache.Loops(current(), budget));
      if (result.reason == recovery::SsaConstantFoldRefusal::resource_limit) return false;
      const char* outcome = result.provisional                                        ? "proposed"
                            : result.reason != recovery::SsaConstantFoldRefusal::none ? "declined"
                                                                                      : "unchanged";
      Begin(outcome, Name(result.reason), current().revision(),
            result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
      // Declared reads the folds let the graph place a store on, whose
      // records the candidate no longer carries.
      json.raw(",\"dropped_path_reads\":[");
      {
        List dropped(json);
        for (const auto& read : result.dropped_path_reads)
          dropped.next()
              .raw("{\"block\":")
              .handle(read.block)
              .raw(",\"node\":")
              .number(read.node)
              .raw(",\"address\":")
              .number(read.address)
              .raw("}");
      }

      json.raw("]");
      dropped_path_reads.insert(dropped_path_reads.end(), result.dropped_path_reads.begin(),
                                result.dropped_path_reads.end());
      json.raw(",\"assumptions\":[");
      List assumptions(json);
      assumptions.next().string("the listed entries exhaust incoming function entries");
      ControlAssumptions(assumptions);
      assumptions.next().string(
          "the current SSA graph comes from decoded groups and"
          " preceding journaled stages in this provisional batch");
      if (result.provisional)
        assumptions.next().string(
            "admitted entry states contain architectural register"
            " cells with target-defined widths");
      if (has_return)
        assumptions.next().string(
            "caller declares untargeted returns leave the selected population");
      std::size_t selected = 0;
      for (const auto& edges : proved.facts->edges) {
        if (edges.size() == std::numeric_limits<std::size_t>::max() ||
            selected > std::numeric_limits<std::size_t>::max() - edges.size() ||
            budget.try_consume({edges.size() + 1, 0}) != BudgetDecline::none)
          return false;
        selected += std::count(edges.begin(), edges.end(), 1);
      }

      json.raw("],\"selected_edges\":").number(selected).raw(",\"excluded_edges\":[");
      List excluded(json);
      for (std::size_t slot = 0; slot < current().slots(); ++slot) {
        if (budget.try_consume({1, 0}) != BudgetDecline::none) return false;
        if (!proved.facts->executable[slot]) continue;
        const auto handle = current().Handle(slot);
        if (!handle) return false;
        const auto& block = *current().Get(*handle);
        for (std::size_t index = 0; index < block.edges.size(); ++index) {
          if (budget.try_consume({1, 0}) != BudgetDecline::none) return false;
          if (proved.facts->edges[slot][index]) continue;
          const auto& edge = block.edges[index];
          if (!edge.condition || !edge.when || !edge.target_block ||
              budget.try_consume({proved.facts->constants.values.size(), 0}) != BudgetDecline::none)
            return false;
          const auto constant = std::find_if(
              proved.facts->constants.values.begin(), proved.facts->constants.values.end(),
              [&](const ir::SsaConstantValue& value) {
                return value.block == *handle && value.kind == ir::SsaValueKind::node &&
                       value.index == *edge.condition;
              });
          if (constant == proved.facts->constants.values.end()) return false;
          excluded.next()
              .raw("{\"block\":")
              .handle(*handle)
              .raw(",\"edge\":")
              .number(index)
              .raw(",\"condition\":")
              .number(*edge.condition)
              .raw(",\"condition_value\":")
              .number(constant->value)
              .raw(",\"graph_revision\":")
              .number(current().revision())
              .raw("}");
        }
      }

      json.raw("],\"journal\":[");
      List journal(json, &journaled);
      for (const auto& edit : result.journal) {
        if (!edit.removed_read) return false;
        journal.next()
            .raw("{\"block\":")
            .handle(edit.original_block)
            .raw(",\"result_block\":")
            .handle(edit.result_block)
            .raw(",\"node\":")
            .number(edit.node)
            .raw(",\"original_op\":")
            .string(ir::Descriptor(edit.original.op)->name)
            .raw(",\"value\":")
            .number(edit.value)
            .raw(",\"read_phi\":")
            .number(edit.removed_read->phi)
            .raw(",\"read_value_kind\":")
            .number(static_cast<unsigned>(edit.removed_read->value.kind))
            .raw(",\"read_value_block\":")
            .handle(edit.removed_read->value.block)
            .raw(",\"read_value_index\":")
            .number(edit.removed_read->value.index)
            .raw(",\"from_revision\":")
            .number(edit.from_revision)
            .raw(",\"to_revision\":")
            .number(edit.to_revision)
            .raw("}");
      }

      json.raw("],\"refused\":[]}");
      if (result.provisional) owned = std::move(result.provisional);
    }

    return true;
  }

  bool BranchRetirement() {
    const auto& proved = proof_cache.Sccp(current(), budget);
    if (proved.reason == analysis::SsaSccpRefusal::resource_limit) return false;
    if (!proved.facts) {
      Begin("declined", Name(proved.reason), current().revision(), std::nullopt);
      json.raw(",\"assumptions\":[],\"journal\":[],\"phi_journal\":[],\"refused\":[]}");
    } else {
      auto result = recovery::ProposeSsaBranchRetirement(current(), *proved.facts, sources, budget,
                                                         proof_cache.Loops(current(), budget));
      if (result.reason == recovery::SsaBranchRetirementRefusal::resource_limit) return false;
      const char* outcome = result.provisional ? "proposed"
                            : result.reason != recovery::SsaBranchRetirementRefusal::none
                                ? "declined"
                                : "unchanged";
      Begin(outcome, Name(result.reason), current().revision(),
            result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
      json.raw(",\"assumptions\":[");
      List assumptions(json);
      assumptions.next().string("the listed entries exhaust incoming function entries");
      ControlAssumptions(assumptions);
      assumptions.next().string(
          "the current SSA graph comes from decoded groups and"
          " preceding journaled stages in this provisional batch");
      if (has_return)
        assumptions.next().string(
            "caller declares untargeted returns leave the selected population");
      json.raw("],\"journal\":[");
      List journal(json, &journaled);
      for (const auto& edit : result.journal)
        journal.next()
            .raw("{\"block\":")
            .handle(edit.original_block)
            .raw(",\"result_block\":")
            .handle(edit.result_block)
            .raw(",\"removed_edge\":")
            .number(edit.removed_edge)
            .raw(",\"removed_target\":")
            .handle(*edit.removed.target_block)
            .raw(",\"retained_target\":")
            .handle(*edit.retained.target_block)
            .raw(",\"condition\":")
            .number(*edit.removed.condition)
            .raw(",\"when\":")
            .boolean(*edit.removed.when)
            .raw(",\"from_revision\":")
            .number(edit.from_revision)
            .raw(",\"to_revision\":")
            .number(edit.to_revision)
            .raw("}");
      json.raw("],\"phi_journal\":[");
      List phis(json, &journaled);
      for (const auto& edit : result.phi_journal)
        phis.next()
            .raw("{\"block\":")
            .handle(edit.original_block)
            .raw(",\"result_block\":")
            .handle(edit.result_block)
            .raw(",\"predecessor\":")
            .handle(edit.predecessor)
            .raw(",\"phi\":")
            .number(edit.phi)
            .raw(",\"frame\":")
            .boolean(edit.frame)
            .raw(",\"from_revision\":")
            .number(edit.from_revision)
            .raw(",\"to_revision\":")
            .number(edit.to_revision)
            .raw("}");
      json.raw("],\"refused\":[]}");
      if (result.provisional) owned = std::move(result.provisional);
    }

    return true;
  }

  bool LoopExits() {
    const auto* loops = proof_cache.Loops(current(), budget);
    if (proof_cache.loops_reason() == analysis::SsaBoundedLoopRefusal::resource_limit) return false;
    if (!loops) {
      // No loops means nothing to retire; only a failed proof is a decline.
      const auto reason = proof_cache.loops_reason();
      Begin(reason == analysis::SsaBoundedLoopRefusal::none ? "unchanged" : "declined",
            Name(reason), current().revision(), std::nullopt);
      json.raw(",\"assumptions\":[],\"journal\":[],\"refused\":[]}");
      return true;
    }

    auto result = recovery::ProposeSsaLoopExits(current(), *loops, sources, budget);
    if (result.reason == recovery::SsaLoopExitRefusal::resource_limit) return false;
    const char* outcome = result.provisional                                    ? "proposed"
                          : result.reason != recovery::SsaLoopExitRefusal::none ? "declined"
                                                                                : "unchanged";
    Begin(outcome, Name(result.reason), current().revision(),
          result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
    json.raw(",\"assumptions\":[");
    List assumptions(json);
    assumptions.next().string("the listed entries exhaust incoming function entries");
    ControlAssumptions(assumptions);
    assumptions.next().string(
        "the current SSA graph comes from decoded groups and"
        " preceding journaled stages in this provisional batch");
    if (declared.frame)
      assumptions.next().string(
          "a private frame slot is unreachable once control leaves the function");
    if (has_return)
      assumptions.next().string("caller declares untargeted returns leave the selected population");
    json.raw("],\"journal\":[");
    List journal(json, &journaled);
    for (const auto& edit : result.journal)
      journal.next()
          .raw("{\"block\":")
          .handle(edit.block)
          .raw(",\"exit\":")
          .handle(edit.exit)
          .raw(",\"iterations\":")
          .number(edit.iterations)
          .raw(",\"from_revision\":")
          .number(edit.from_revision)
          .raw(",\"to_revision\":")
          .number(edit.to_revision)
          .raw("}");
    json.raw("],\"refused\":[");
    List refused(json);
    for (const auto& item : result.refused)
      refused.next()
          .raw("{\"block\":")
          .handle(item.block)
          .raw(",\"reason\":")
          .string(Name(item.reason))
          .raw("}");
    json.raw("]}");
    if (result.provisional) owned = std::move(result.provisional);
    return true;
  }

  bool LeafCalls() {
    auto result = recovery::ProposeSsaLeafCalls(current(), sources, budget);
    if (result.reason == recovery::SsaLeafCallRefusal::resource_limit) return false;
    const char* outcome = result.provisional                                    ? "proposed"
                          : result.reason != recovery::SsaLeafCallRefusal::none ? "declined"
                                                                                : "unchanged";
    Begin(outcome, Name(result.reason), current().revision(),
          result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
    json.raw(",\"assumptions\":[");
    List assumptions(json);
    assumptions.next().string("the listed entries exhaust incoming function entries");
    ControlAssumptions(assumptions);
    assumptions.next().string(
        "the current SSA graph comes from decoded groups and"
        " preceding journaled stages in this provisional batch");
    assumptions.next().string(
        "a callee runs the instructions its body records, which were"
        " decoded from the image at its address");
    if (has_return)
      assumptions.next().string("caller declares untargeted returns leave the selected population");
    json.raw("],\"journal\":[");
    List journal(json, &journaled);
    for (const auto& edit : result.journal)
      journal.next()
          .raw("{\"block\":")
          .handle(edit.block)
          .raw(",\"callee\":")
          .number(edit.callee)
          .raw(",\"from_revision\":")
          .number(edit.from_revision)
          .raw(",\"to_revision\":")
          .number(edit.to_revision)
          .raw("}");
    json.raw("],\"refused\":[");
    List refused(json);
    for (const auto& item : result.refused)
      refused.next()
          .raw("{\"block\":")
          .handle(item.block)
          .raw(",\"callee\":")
          .number(item.callee)
          .raw(",\"reason\":")
          .string(Name(item.reason))
          .raw("}");
    json.raw("]}");
    if (result.provisional) owned = std::move(result.provisional);
    return true;
  }

  bool ResolvedCalls() {
    auto result = recovery::ProposeSsaResolvedCalls(current(), sources, budget);
    if (result.reason == recovery::SsaResolvedCallRefusal::resource_limit) return false;
    const char* outcome = result.provisional                                        ? "proposed"
                          : result.reason != recovery::SsaResolvedCallRefusal::none ? "declined"
                                                                                    : "unchanged";
    Begin(outcome, Name(result.reason), current().revision(),
          result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
    json.raw(",\"assumptions\":[");
    List assumptions(json);
    assumptions.next().string("the listed entries exhaust incoming function entries");
    ControlAssumptions(assumptions);
    assumptions.next().string(
        "the current SSA graph comes from decoded groups and"
        " preceding journaled stages in this provisional batch");
    if (has_return)
      assumptions.next().string("caller declares untargeted returns leave the selected population");
    json.raw("],\"journal\":[");
    List journal(json, &journaled);
    for (const auto& edit : result.journal)
      journal.next()
          .raw("{\"block\":")
          .handle(edit.block)
          .raw(",\"callee\":")
          .number(edit.callee)
          .raw(",\"from_revision\":")
          .number(edit.from_revision)
          .raw(",\"to_revision\":")
          .number(edit.to_revision)
          .raw("}");
    json.raw("],\"refused\":[]}");
    if (result.provisional) owned = std::move(result.provisional);
    return true;
  }

  bool UnreachableAfterSccp() {
    const auto& proof = proof_cache.Reachability(current(), budget);
    if (proof.reason == analysis::SsaReachabilityRefusal::resource_limit) return false;
    if (!proof.facts) {
      Begin("declined", Name(proof.reason), current().revision(), std::nullopt);
      json.raw(",\"assumptions\":[],\"journal\":[],\"refused\":[]}");
    } else {
      auto result = recovery::ProposeUnreachableBlocks(current(), *proof.facts, sources, budget);
      if (result.reason == recovery::SsaDceRefusal::resource_limit) return false;
      const char* outcome = result.provisional                               ? "proposed"
                            : result.reason != recovery::SsaDceRefusal::none ? "declined"
                                                                             : "unchanged";
      Begin(outcome, Name(result.reason), current().revision(),
            result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
      json.raw(",\"assumptions\":[");
      List assumptions(json);
      assumptions.next().string("the listed entries exhaust incoming function entries");
      ControlAssumptions(assumptions);
      assumptions.next().string(
          "the current SSA graph comes from decoded groups and"
          " preceding journaled stages and declarations in this provisional batch");
      if (has_return)
        assumptions.next().string(
            "caller declares untargeted returns leave the selected population");
      json.raw("],\"journal\":[");
      List journal(json, &journaled);
      for (const auto& edit : result.journal) {
        journal.next()
            .raw("{\"kind\":")
            .string(Name(edit.kind))
            .raw(",\"block\":")
            .handle(edit.original_block)
            .raw(",\"result_block\":");
        if (edit.result_block)
          json.handle(*edit.result_block);
        else
          json.raw("null");
        json.raw(",\"index\":").number(edit.index).raw(",\"predecessor\":");
        if (edit.kind == recovery::SsaDceEditKind::removed_block) {
          json.raw("null");
          ++removed_blocks;
        } else
          json.handle(edit.predecessor);
        json.raw(",\"from_revision\":")
            .number(edit.from_revision)
            .raw(",\"to_revision\":")
            .number(edit.to_revision)
            .raw("}");
      }

      json.raw("],\"refused\":[]}");
      if (result.provisional) owned = std::move(result.provisional);
    }

    return true;
  }

  bool PredecessorCopies() {
    const auto& reachable = proof_cache.Reachability(current(), budget);
    if (reachable.reason == analysis::SsaReachabilityRefusal::resource_limit) return false;
    if (!reachable.facts) {
      Begin("declined", Name(reachable.reason), current().revision(), std::nullopt);
      json.raw(",\"assumptions\":[],\"journal\":[],\"refused\":[]}");
    } else {
      auto proved =
          analysis::ProveSsaPredecessorCopies(current(), *reachable.facts, sources, budget);
      if (proved.reason == analysis::SsaCopyRefusal::resource_limit) return false;
      if (!proved.facts) {
        Begin("declined", Name(proved.reason), current().revision(), std::nullopt);
        json.raw(",\"assumptions\":[],\"journal\":[],\"refused\":[]}");
      } else {
        auto result = recovery::ProposeSsaPredecessorCopies(current(), *reachable.facts,
                                                            *proved.facts, sources, budget);
        if (result.reason == recovery::SsaCopyRefusal::resource_limit) return false;
        const char* outcome = result.provisional                                ? "proposed"
                              : result.reason != recovery::SsaCopyRefusal::none ? "declined"
                                                                                : "unchanged";
        Begin(outcome, Name(result.reason), current().revision(),
              result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
        json.raw(",\"assumptions\":[");
        List assumptions(json);
        assumptions.next().string("the listed entries exhaust incoming function entries");
        ControlAssumptions(assumptions);
        assumptions.next().string(
            "the current SSA graph comes from decoded groups and"
            " preceding journaled stages in this provisional batch");
        if (has_return)
          assumptions.next().string(
              "caller declares untargeted returns leave the selected population");
        json.raw("],\"journal\":[");
        List journal(json, &journaled);
        for (const auto& edit : result.journal)
          journal.next()
              .raw("{\"block\":")
              .handle(edit.original_block)
              .raw(",\"result_block\":")
              .handle(edit.result_block)
              .raw(",\"read\":")
              .number(edit.read)
              .raw(",\"source_block\":")
              .handle(edit.source.block)
              .raw(",\"source_node\":")
              .number(edit.source.index)
              .raw(",\"from_revision\":")
              .number(edit.from_revision)
              .raw(",\"to_revision\":")
              .number(edit.to_revision)
              .raw("}");
        json.raw("],\"refused\":[]}");
        if (result.provisional) owned = std::move(result.provisional);
      }
    }

    return true;
  }

  bool DeadStorageWrites() {
    const auto& reachable = proof_cache.Reachability(current(), budget);
    if (reachable.reason == analysis::SsaReachabilityRefusal::resource_limit) return false;
    if (!reachable.facts) {
      Begin("declined", Name(reachable.reason), current().revision(), std::nullopt);
      json.raw(",\"assumptions\":[],\"journal\":[],\"refused\":[]}");
    } else {
      auto proved = analysis::ProveSsaDeadWrites(current(), *reachable.facts, sources, budget);
      if (proved.reason == analysis::SsaDeadWriteRefusal::resource_limit) return false;
      if (!proved.facts) {
        Begin("declined", Name(proved.reason), current().revision(), std::nullopt);
        json.raw(",\"assumptions\":[],\"journal\":[],\"refused\":[]}");
      } else {
        auto result = recovery::ProposeSsaDeadWrites(current(), *reachable.facts, *proved.facts,
                                                     sources, budget);
        if (result.reason == recovery::SsaDeadWriteRefusal::resource_limit) return false;
        const char* outcome = result.provisional                                     ? "proposed"
                              : result.reason != recovery::SsaDeadWriteRefusal::none ? "declined"
                                                                                     : "unchanged";
        Begin(outcome, Name(result.reason), current().revision(),
              result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
        json.raw(",\"assumptions\":[");
        List assumptions(json);
        assumptions.next().string("the listed entries exhaust incoming function entries");
        ControlAssumptions(assumptions);
        assumptions.next().string(
            "each retired register write is overwritten on every"
            " path before any read or observable exit");
        const auto& observed = current().observability();
        if (observed.declared)
          assumptions.next().string(
              "a call observes only the declared ABI's arguments and"
              " preserved registers, a return only its results and"
              " preserved registers, and a run that never leaves"
              " observes none");
        if (observed.faults_terminal)
          assumptions.next().string(
              "a fault or trap ends the run without its registers"
              " being observed");
        assumptions.next().string(
            "intermediate register state is not observed asynchronously"
            " between modeled instruction boundaries");
        if (has_return)
          assumptions.next().string(
              "caller declares untargeted returns leave the selected population");
        json.raw("],\"journal\":[");
        List journal(json, &journaled);
        for (const auto& edit : result.journal)
          journal.next()
              .raw("{\"block\":")
              .handle(edit.original_block)
              .raw(",\"result_block\":")
              .handle(edit.result_block)
              .raw(",\"boundary\":")
              .number(edit.boundary)
              .raw(",\"index\":")
              .number(edit.index)
              .raw(",\"storage\":")
              .number(edit.original.storage)
              .raw(",\"from_revision\":")
              .number(edit.from_revision)
              .raw(",\"to_revision\":")
              .number(edit.to_revision)
              .raw("}");
        json.raw("],\"refused\":[");
        List refused(json);
        for (const auto& rejection : result.refused)
          refused.next()
              .raw("{\"block\":")
              .handle(rejection.block)
              .raw(",\"boundary\":")
              .number(rejection.boundary)
              .raw(",\"index\":")
              .number(rejection.index)
              .raw("}");
        json.raw("]}");
        if (result.provisional) owned = std::move(result.provisional);
      }
    }

    return true;
  }

  bool AlgebraicSimplify() {
    auto result =
        recovery::ProposeSsaSimplify(current(), sources, facts.page_aligned_placement, budget);
    if (result.reason == recovery::SsaSimplifyRefusal::resource_limit) return false;
    const char* outcome = result.provisional                                    ? "proposed"
                          : result.reason != recovery::SsaSimplifyRefusal::none ? "declined"
                                                                                : "unchanged";
    Begin(outcome, Name(result.reason), current().revision(),
          result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
    // Every other rule is an identity for all inputs at every width.
    json.raw(",\"assumptions\":[");
    List assumptions(json);
    if (std::any_of(result.journal.begin(), result.journal.end(),
                    [](const auto& edit) { return edit.placement; }))
      assumptions.next().string("the image is placed at a page-aligned address");
    if (std::any_of(result.journal.begin(), result.journal.end(),
                    [](const auto& edit) { return edit.folded_value; }))
      assumptions.next().string("a folded image load keeps the declared value its fold rests on");
    json.raw("],\"journal\":[");
    List journal(json, &journaled);
    for (const auto& edit : result.journal)
      journal.next()
          .raw("{\"block\":")
          .handle(edit.original_block)
          .raw(",\"result_block\":")
          .handle(edit.result_block)
          .raw(",\"node\":")
          .number(edit.node)
          .raw(",\"rule\":")
          .string(edit.rule)
          .raw(",\"folded_value\":")
          .boolean(edit.folded_value)
          .raw(",\"original_op\":")
          .string(ir::Descriptor(edit.original.op)->name)
          .raw(",\"replacement_op\":")
          .string(ir::Descriptor(edit.replacement.op)->name)
          .raw(",\"from_revision\":")
          .number(edit.from_revision)
          .raw(",\"to_revision\":")
          .number(edit.to_revision)
          .raw("}");
    json.raw("],\"dropped_path_reads\":[");
    {
      List dropped(json);
      for (const auto& read : result.dropped_path_reads)
        dropped.next()
            .raw("{\"block\":")
            .handle(read.block)
            .raw(",\"node\":")
            .number(read.node)
            .raw(",\"address\":")
            .number(read.address)
            .raw("}");
    }

    dropped_path_reads.insert(dropped_path_reads.end(), result.dropped_path_reads.begin(),
                              result.dropped_path_reads.end());
    json.raw("],\"refused\":[]}");
    if (result.provisional) owned = std::move(result.provisional);
    return true;
  }

  bool DeadLoads() {
    const ir::ImageAccessContract access{declared.image_access, declared.image_access,
                                         declared.image_access};
    auto result = recovery::ProposeSsaDeadLoads(current(), sources, facts, access, budget);
    if (result.reason == recovery::SsaDeadLoadRefusal::resource_limit) return false;
    const char* outcome = result.provisional                                    ? "proposed"
                          : result.reason != recovery::SsaDeadLoadRefusal::none ? "declined"
                                                                                : "unchanged";
    Begin(outcome, Name(result.reason), current().revision(),
          result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
    json.raw(",\"assumptions\":[");
    List assumptions(json);
    const auto uses = [&](ir::SsaRetiredLoadBasis basis) {
      return std::any_of(result.journal.begin(), result.journal.end(),
                         [&](const auto& edit) { return edit.retired.basis == basis; });
    };

    if (uses(ir::SsaRetiredLoadBasis::written_before))
      assumptions.next().string(
          "memory a store wrote stays mapped until a later load of the same"
          " bytes in the same block");
    if (uses(ir::SsaRetiredLoadBasis::declared_image))
      assumptions.next().string(
          "every retired image read's bytes stay mapped readable for the run,"
          " and reading them has no observable effect");
    json.raw("],\"journal\":[");
    List journal(json, &journaled);
    for (const auto& edit : result.journal) {
      journal.next()
          .raw("{\"block\":")
          .handle(edit.original_block)
          .raw(",\"result_block\":")
          .handle(edit.result_block)
          .raw(",\"node\":")
          .number(edit.retired.node)
          .raw(",\"basis\":")
          .string(edit.retired.basis == ir::SsaRetiredLoadBasis::written_before ? "written_before"
                                                                                : "declared_image");
      if (edit.retired.basis == ir::SsaRetiredLoadBasis::written_before)
        json.raw(",\"store\":").number(edit.retired.store);
      else
        json.raw(",\"range_address\":")
            .number(edit.retired.range_address)
            .raw(",\"range_bytes\":")
            .number(edit.retired.range_bytes);
      json.raw(",\"from_revision\":")
          .number(edit.from_revision)
          .raw(",\"to_revision\":")
          .number(edit.to_revision)
          .raw("}");
    }

    json.raw("],\"refused\":[]}");
    if (result.provisional) owned = std::move(result.provisional);
    return true;
  }

  bool DeadEffectFreeNodes() {
    auto proof = analysis::ProveSsaNodeLiveness(current(), budget);
    if (proof.reason == analysis::SsaLivenessRefusal::resource_limit) return false;
    if (!proof.facts) {
      Begin("declined", Name(proof.reason), current().revision(), std::nullopt);
      json.raw(",\"assumptions\":[],\"journal\":[],\"refused\":[]}");
    } else {
      auto result = recovery::ProposeDeadPureNodes(current(), *proof.facts, sources, budget);
      if (result.reason == recovery::SsaPureDceRefusal::resource_limit) return false;
      const char* outcome = result.provisional                                   ? "proposed"
                            : result.reason != recovery::SsaPureDceRefusal::none ? "declined"
                                                                                 : "unchanged";
      Begin(outcome, Name(result.reason), current().revision(),
            result.provisional ? std::optional(result.provisional->revision()) : std::nullopt);
      json.raw(",\"assumptions\":[");
      List assumptions(json);
      bool constant_image_load = false;
      bool selected_image_load = false;
      bool bounded_table_load = false;
      for (const auto& edit : result.journal) {
        const auto* block = current().Get(edit.original_block);
        if (block->nodes[edit.node].op != ir::Op::load) continue;
        if (budget.try_consume({block->constant_loads.size(), 0}) != BudgetDecline::none)
          return false;
        const auto fold =
            std::find_if(block->constant_loads.begin(), block->constant_loads.end(),
                         [&](const ir::SsaConstantLoad& item) { return item.node == edit.node; });
        if (fold == block->constant_loads.end()) return false;
        if (fold->kind == ir::SsaConstantKind::bounded_table)
          bounded_table_load = true;
        else if (fold->condition)
          selected_image_load = true;
        else
          constant_image_load = true;
      }

      if (constant_image_load)
        assumptions.next().string(
            "retired image loads inherit the constant-image stage's"
            " mapped, invariant and unobservable-read declarations");
      if (selected_image_load)
        assumptions.next().string(
            "retired selected image loads inherit the selected-image stage's"
            " mapped, invariant and unobservable-read declarations");
      if (bounded_table_load)
        assumptions.next().string(
            "retired table loads inherit the bounded-table stage's"
            " mapped, invariant and unobservable-read declarations");
      json.raw("],\"journal\":[");
      List journal(json, &journaled);
      for (const auto& edit : result.journal)
        journal.next()
            .raw("{\"block\":")
            .handle(edit.original_block)
            .raw(",\"result_block\":")
            .handle(edit.result_block)
            .raw(",\"node\":")
            .number(edit.node)
            .raw(",\"op\":")
            .string(ir::Descriptor(current().Get(edit.original_block)->nodes[edit.node].op)->name)
            .raw("}");
      json.raw("],\"refused\":[]}");
      if (result.provisional) owned = std::move(result.provisional);
    }

    return true;
  }

  const ir::SsaGraph& base;
  const SsaDeclarations& declared;
  std::span<const ir::Group> sources;
  ir::ImageFacts facts;
  Budget& budget;
  Json json;
  List stages;

  // Each stage replaces `owned` only when it proposes; the base stays owned
  // by the caller, so the first replacement is the first owned graph.
  std::optional<ir::SsaGraph> owned;
  analysis::SsaProofCache proof_cache;
  bool has_return = false;

  // What the successor proof of this graph rests on besides the entry and
  // return declarations: calls, declared traps, and declared image reads.
  bool has_call = false;
  bool has_noreturn = false;
  bool has_trap = false;
  bool has_declared_reads = false;
  bool has_placed_reads = false;
  std::size_t pass = 0;
  SsaStageSummary summary{};

  // Whether the running pass opened its record, and the journal entries it
  // wrote, of which `removed_blocks` delete a block rather than change code.
  bool opened = false;
  std::size_t journaled = 0;
  std::size_t removed_blocks = 0;
};

namespace {

constexpr std::string_view kEmpty = ",\"assumptions\":[],\"journal\":[],\"refused\":[]}";
constexpr unsigned kClosedPopulation =
    kNeedsClosedEntries | kNeedsReturnLeaves | kNeedsCallReturns | kNeedsTrapStops;

template <bool (SsaPassContext::*Run)()>
bool Call(SsaPassContext& context) {
  return (context.*Run)();
}

// The default pipeline is this order. Reachability-based passes need the
// closed entry declaration and, with a return edge, the return-leaves one.
constexpr std::array<SsaPassInfo, 22> kRegistry = {{
    {"unreachable_blocks", kClosedPopulation, kEmpty, Call<&SsaPassContext::UnreachableBlocks>},
    {"phi_constant_fold", kClosedPopulation,
     ",\"assumptions\":[],\"proved_phis\":0,\"journal\":[],\"refused\":[]}",
     Call<&SsaPassContext::PhiConstantFold>},
    {"constant_propagation", kClosedPopulation,
     ",\"assumptions\":[],\"bounded_loops\":[],\"bounded_loop_refusal\":\"none\","
     "\"proved_phis\":0,\"proved_nodes\":0,\"journal\":[],\"refused\":[]}",
     Call<&SsaPassContext::ConstantPropagation>},
    {"sccp_constant_fold", kClosedPopulation,
     ",\"assumptions\":[],\"executable_blocks\":0,\"selected_edges\":0,\"proved_phis\":0,"
     "\"proved_nodes\":0,\"excluded_edges\":[],\"journal\":[],\"refused\":[]}",
     Call<&SsaPassContext::SccpConstantFold>},
    {"constant_image_loads", 0, kEmpty, Call<&SsaPassContext::ConstantImageLoads>},
    {"selected_image_loads", 0, kEmpty, Call<&SsaPassContext::SelectedImageLoads>},
    {"linear_mba", 0, kEmpty, Call<&SsaPassContext::LinearMba>},
    {"canonical_linear_mba", 0, kEmpty, Call<&SsaPassContext::CanonicalLinearMba>},
    {"dead_private_state", kNeedsPrivateFrame, kEmpty, Call<&SsaPassContext::DeadPrivateState>},
    {"frame_promotion", kNeedsPrivateFrame, kEmpty, Call<&SsaPassContext::FramePromotion>},
    {"bounded_table_loads", kClosedPopulation | kNeedsImageAccess, kEmpty,
     Call<&SsaPassContext::BoundedTableLoads>},
    {"sccp_storage_reads", kClosedPopulation,
     ",\"assumptions\":[],\"selected_edges\":0,\"excluded_edges\":[],\"journal\":[],\"refused\":[]"
     "}",
     Call<&SsaPassContext::SccpStorageReads>},
    {"branch_retirement", kClosedPopulation,
     ",\"assumptions\":[],\"journal\":[],\"phi_journal\":[],\"refused\":[]}",
     Call<&SsaPassContext::BranchRetirement>},
    {"loop_exits", kClosedPopulation, kEmpty, Call<&SsaPassContext::LoopExits>},
    {"leaf_calls", kClosedPopulation, kEmpty, Call<&SsaPassContext::LeafCalls>},
    {"resolved_calls", kClosedPopulation, kEmpty, Call<&SsaPassContext::ResolvedCalls>},
    {"unreachable_after_sccp", kClosedPopulation, kEmpty,
     Call<&SsaPassContext::UnreachableAfterSccp>},
    {"predecessor_copies", kClosedPopulation, kEmpty, Call<&SsaPassContext::PredecessorCopies>},
    {"dead_storage_writes", kClosedPopulation, kEmpty, Call<&SsaPassContext::DeadStorageWrites>},
    {"algebraic_simplify", 0, kEmpty, Call<&SsaPassContext::AlgebraicSimplify>},
    {"dead_loads", 0, kEmpty, Call<&SsaPassContext::DeadLoads>},
    {"dead_effect_free_nodes", 0, kEmpty, Call<&SsaPassContext::DeadEffectFreeNodes>},
}};

// The first declaration a pass lacks, in a fixed order, or null.
const char* Missing(unsigned requirements, const SsaDeclarations& declared,
                    const SsaPassContext& graph) {
  const bool has_return = graph.has_return;
  if (requirements & kNeedsClosedEntries && !declared.closed_entries)
    return "no_closed_entry_declaration";
  if (requirements & kNeedsImageAccess && !declared.image_access)
    return "no_image_access_declaration";
  if (requirements & kNeedsReturnLeaves && has_return && !declared.return_leaves)
    return "no_return_leaves_declaration";
  if (requirements & kNeedsCallReturns && graph.has_call && !declared.call_returns)
    return "no_call_return_declaration";
  if (requirements & kNeedsTrapStops && graph.has_trap && !declared.trap_stops)
    return "no_trap_stop_declaration";
  if (requirements & kNeedsPrivateFrame && !declared.frame) return "no_private_frame_declaration";
  return nullptr;
}

}  // namespace

std::span<const SsaPassInfo> SsaPassRegistry() { return kRegistry; }

std::optional<std::size_t> FindSsaPass(std::string_view name) {
  for (std::size_t index = 0; index < kRegistry.size(); ++index)
    if (kRegistry[index].name == name) return index;
  return std::nullopt;
}

SsaPipelineRun RunSsaPipeline(const ir::SsaGraph& base, const SsaDeclarations& declared,
                              std::span<const ir::Group> sources, ir::ImageFacts facts,
                              Budget& budget, std::span<const std::size_t> selection,
                              unsigned rounds) {
  const auto decline = [](SsaPipelineDecline reason) {
    SsaPipelineRun refused;
    refused.reason = reason;
    return refused;
  };

  std::array<std::size_t, kRegistry.size()> order{};
  std::size_t count = 0;
  if (selection.empty()) {
    for (; count < kRegistry.size(); ++count) order[count] = count;
  } else {
    std::array<bool, kRegistry.size()> seen{};
    if (selection.size() > kRegistry.size()) return decline(SsaPipelineDecline::invalid_selection);
    for (const auto index : selection) {
      if (index >= kRegistry.size() || seen[index])
        return decline(SsaPipelineDecline::invalid_selection);
      seen[index] = true;
      order[count++] = index;
    }
  }

  SsaPassContext context(base, declared, sources, facts, budget);
  const auto base_text = ir::PrintSsa(base, budget);
  if (!base_text.text) return decline(SsaPipelineDecline::resource_limit);
  for (std::size_t slot = 0; slot < base.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none)
      return decline(SsaPipelineDecline::resource_limit);
    const auto handle = base.Handle(slot);
    if (!handle) continue;
    const auto& edges = base.Get(*handle)->edges;
    if (budget.try_consume({edges.size(), 0}) != BudgetDecline::none)
      return decline(SsaPipelineDecline::resource_limit);
    context.has_return |= std::any_of(edges.begin(), edges.end(), [](const ir::SsaEdge& edge) {
      return edge.kind == ir::SsaEdgeKind::return_;
    });
    const auto& block = *base.Get(*handle);
    if (budget.try_consume({block.path_reads.size(), 0}) != BudgetDecline::none)
      return decline(SsaPipelineDecline::resource_limit);
    for (const auto& edge : edges) {
      context.has_call |= edge.kind == ir::SsaEdgeKind::callee;
      context.has_noreturn |= edge.assumptions.declared_noreturn;
      context.has_trap |= edge.kind == ir::SsaEdgeKind::trap;

      // A dispatch's destinations come from declared table bytes, recorded on
      // its edges rather than as path reads.
      context.has_declared_reads |= edge.assumptions.constant_image;
    }

    context.has_declared_reads |= !block.path_reads.empty();
    for (const auto& read : block.path_reads)
      context.has_placed_reads |= read.page_aligned_placement;
  }

  SsaPipelineResult result;
  Json head(budget);
  head.raw(",\"revision\":")
      .number(base.revision())
      .raw(",\"text\":")
      .string(*base_text.text)
      .raw(",\"declarations\":{\"private_frame\":");
  if (declared.frame)
    FrameContract(head, *declared.frame);
  else
    head.raw("null");
  head.raw(",\"image_access\":")
      .boolean(declared.image_access)
      .raw(",\"closed_entries\":")
      .boolean(declared.closed_entries)
      .raw(",\"return_leaves\":")
      .boolean(declared.return_leaves)
      .raw(",\"call_returns\":")
      .boolean(declared.call_returns)
      .raw(",\"trap_stops\":")
      .boolean(declared.trap_stops)
      .raw("}");
  auto& json = context.json;

  // How many times the sequence may repeat. A reader counting stages cannot
  // tell four rounds of a sequence that stopped early from one long one, and
  // the count changes the candidate, so the record says it.
  json.raw("\"pipeline\":{\"name\":")
      .string(selection.empty() ? "default" : "named")
      .raw(",\"rounds\":")
      .number(std::max(1U, rounds))
      .raw(",\"passes\":[");
  for (std::size_t i = 0; i < count; ++i) {
    if (i) json.raw(",");
    json.string(kRegistry[order[i]].name);
  }

  json.raw("]},\"stages\":[");
  if (budget.try_consume({count, count * sizeof(SsaStageSummary)}) != BudgetDecline::none)
    return decline(SsaPipelineDecline::resource_limit);
  result.stages.reserve(count);
  for (unsigned round = 0; round < std::max(1U, rounds); ++round) {
    bool advanced = false;
    for (std::size_t i = 0; i < count; ++i) {
      const auto& pass = kRegistry[order[i]];
      context.pass = order[i];
      context.summary = {};
      context.opened = false;
      context.journaled = context.removed_blocks = 0;
      if (const auto* missing = Missing(pass.requirements, declared, context)) {
        context.Begin("not_run", missing, context.current().revision(), std::nullopt);
        json.raw(pass.empty_record);
      } else if (!pass.run(context)) {
        return decline(SsaPipelineDecline::resource_limit);
      }

      // A pass that wrote no record would publish another stage's summary.
      if (!context.opened) return decline(SsaPipelineDecline::incomplete_record);
      auto summary = context.summary;
      if (summary.outcome == SsaStageOutcome::proposed) {
        summary.executable_edits =
            context.journaled == 0 ? 1 : context.journaled - context.removed_blocks;
        advanced = true;
      }

      result.stages.push_back(summary);
    }

    // A round that established nothing leaves the next one nothing to use.
    if (!advanced) break;
  }

  json.raw("]");

  // What a retired loop or call no longer does was dead when it was retired,
  // and no pass adds a read; the graph published is checked for it once.
  if (context.owned) {
    const auto retired = ir::ValidateSsaRetiredWork(*context.owned, budget);
    if (retired == ir::SsaDecline::resource_limit)
      return decline(SsaPipelineDecline::resource_limit);
    if (retired != ir::SsaDecline::none) return decline(SsaPipelineDecline::retired_work_observed);
  }

  Json text(budget);
  if (context.owned) {
    const auto printed = ir::PrintSsa(*context.owned, sources, budget);
    if (!printed.text) return decline(SsaPipelineDecline::resource_limit);
    text.string(*printed.text);
  } else {
    text.raw("null");
  }

  if (head.failed() || json.failed() || text.failed())
    return decline(SsaPipelineDecline::resource_limit);
  result.head = std::move(head.text());
  result.body = std::move(json.text());
  result.text = std::move(text.text());
  result.dropped_path_reads = std::move(context.dropped_path_reads);
  result.provisional = std::move(context.owned);
  SsaPipelineRun run;
  run.result = std::move(result);
  return run;
}

}  // namespace nyx::passes
