#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace nyx::verify {

enum class CheckStatus { matched, refuted, inconclusive, not_checked };
enum class CallTrace { matched, mismatch, not_checked };

// An independent check's result, as the separate verifier wrote it (NYXCHK01).
// The record names the exact candidate it examined. It is not authenticated.
struct CheckRecord {
  std::string candidate;
  std::uint64_t states = 0;
  std::uint64_t matched = 0;
  std::uint64_t refuted = 0;
  std::uint64_t inconclusive = 0;
  std::uint64_t edits_executable = 0;
  std::uint64_t edits_covered = 0;
  std::uint64_t stubs_declared = 0;
  std::uint64_t stubs_called = 0;
  CallTrace call_trace = CallTrace::not_checked;
  CheckStatus status = CheckStatus::not_checked;
};

// Strict: the exact field order, one field per LF-terminated line, decimal
// counts without leading zeros, and counts that agree with the status. Scope
// must be whole_function, the only scope a record can currently claim.
[[nodiscard]] std::optional<CheckRecord> ParseCheckRecord(std::string_view text);

enum class Acceptance { provisional, accepted, rejected };

struct AcceptanceDecision {
  Acceptance status = Acceptance::provisional;

  // Why the batch has this status: none when accepted.
  const char* reason = "whole_function_check_not_run";
};

// What the candidate itself says an independent check must cover. The
// record's own denominators are a producer's claim; these are the CLI's.
struct CandidateCoverage {
  std::uint64_t proposed_stages = 0;
  std::uint64_t executable_edits = 0;
  std::uint64_t declared_stubs = 0;
};

// A candidate batch is accepted only with a whole-function record for exactly
// this candidate that matched every examined state, covered every executable
// edit and declared stub the candidate itself counts, and compared the ordered
// call-event trace. The candidate identity covers the stub declaration, so a
// record made with other stubs names another candidate. A record for this
// candidate that refutes it rejects the batch. Anything else leaves it
// provisional.
[[nodiscard]] AcceptanceDecision Decide(const CheckRecord* record, std::string_view candidate,
                                        const CandidateCoverage& coverage);

}  // namespace nyx::verify
