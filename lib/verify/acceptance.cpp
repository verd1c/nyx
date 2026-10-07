#include "nyx/verify/acceptance.hpp"

#include <array>
#include <charconv>

namespace nyx::verify {
namespace {

// Splits off the next LF-terminated line; false at the end or on a missing LF.
bool Line(std::string_view& text, std::string_view& line) {
  const auto end = text.find('\n');
  if (end == std::string_view::npos) return false;
  line = text.substr(0, end);
  text.remove_prefix(end + 1);
  return true;
}

bool Field(std::string_view& text, std::string_view key, std::string_view& value) {
  std::string_view line;
  if (!Line(text, line) || line.size() <= key.size() || line.substr(0, key.size()) != key ||
      line[key.size()] != ' ')
    return false;
  value = line.substr(key.size() + 1);
  return true;
}

bool Count(std::string_view& text, std::string_view key, std::uint64_t& value) {
  std::string_view digits;
  if (!Field(text, key, digits) || digits.empty() || digits.size() > 20 ||
      (digits.size() > 1 && digits[0] == '0'))
    return false;
  const auto result = std::from_chars(digits.data(), digits.data() + digits.size(), value, 10);
  return result.ec == std::errc() && result.ptr == digits.data() + digits.size();
}

bool Candidate(std::string_view value) {
  if (value == "none") return true;
  constexpr std::string_view prefix = "sha256:";
  if (value.size() != prefix.size() + 64 || value.substr(0, prefix.size()) != prefix) return false;
  for (const char ch : value.substr(prefix.size()))
    if (!(ch >= '0' && ch <= '9') && !(ch >= 'a' && ch <= 'f')) return false;
  return true;
}

}  // namespace

std::optional<CheckRecord> ParseCheckRecord(std::string_view text) {
  if (text.size() > 4096) return std::nullopt;
  std::string_view line, value;
  CheckRecord record;
  if (!Line(text, line) || line != "NYXCHK01" || !Field(text, "candidate", value) ||
      !Candidate(value))
    return std::nullopt;
  record.candidate = value;
  if (!Field(text, "scope", value) || value != "whole_function") return std::nullopt;
  if (!Count(text, "states", record.states) || !Count(text, "matched", record.matched) ||
      !Count(text, "refuted", record.refuted) ||
      !Count(text, "inconclusive", record.inconclusive) ||
      !Count(text, "edits_executable", record.edits_executable) ||
      !Count(text, "edits_covered", record.edits_covered) ||
      !Count(text, "stubs_declared", record.stubs_declared) ||
      !Count(text, "stubs_called", record.stubs_called) || !Field(text, "call_trace", value))
    return std::nullopt;
  if (value == "matched")
    record.call_trace = CallTrace::matched;
  else if (value == "mismatch")
    record.call_trace = CallTrace::mismatch;
  else if (value == "not_checked")
    record.call_trace = CallTrace::not_checked;
  else
    return std::nullopt;
  if (!Field(text, "status", value) || !text.empty()) return std::nullopt;
  if (value == "matched")
    record.status = CheckStatus::matched;
  else if (value == "refuted")
    record.status = CheckStatus::refuted;
  else if (value == "inconclusive")
    record.status = CheckStatus::inconclusive;
  else if (value == "not_checked")
    record.status = CheckStatus::not_checked;
  else
    return std::nullopt;
  // The counts must add up and name the status the verifier derived from them.
  if (record.matched > record.states || record.refuted > record.states - record.matched ||
      record.inconclusive != record.states - record.matched - record.refuted ||
      record.edits_covered > record.edits_executable || record.stubs_called > record.stubs_declared)
    return std::nullopt;
  const bool refuted = record.refuted || record.call_trace == CallTrace::mismatch;
  const bool examined = record.matched || record.refuted;
  const bool complete = !record.inconclusive && record.edits_covered == record.edits_executable &&
                        record.stubs_called == record.stubs_declared &&
                        record.call_trace == CallTrace::matched;
  const auto derived = refuted     ? CheckStatus::refuted
                       : !examined ? CheckStatus::not_checked
                       : complete  ? CheckStatus::matched
                                   : CheckStatus::inconclusive;
  if (record.status != derived) return std::nullopt;
  return record;
}

AcceptanceDecision Decide(const CheckRecord* record, std::string_view candidate,
                          const CandidateCoverage& coverage) {
  if (!record) return {};
  if (record->candidate != candidate) return {Acceptance::provisional, "check_for_other_candidate"};
  switch (record->status) {
    case CheckStatus::refuted:
      return {Acceptance::rejected, "whole_function_refuted"};
    case CheckStatus::not_checked:
      return {};
    case CheckStatus::inconclusive:
      return {Acceptance::provisional, "whole_function_check_inconclusive"};
    case CheckStatus::matched:
      break;
  }

  if (!coverage.proposed_stages) return {Acceptance::provisional, "no_proposed_edits"};

  // A complete record over the wrong denominators proves nothing about the rest.
  if (record->edits_executable != coverage.executable_edits ||
      record->stubs_declared != coverage.declared_stubs)
    return {Acceptance::provisional, "check_coverage_differs"};
  return {Acceptance::accepted, "none"};
}

}  // namespace nyx::verify
