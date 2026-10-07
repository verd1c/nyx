#include "nyx/verify/acceptance.hpp"

#include <string>

#include <gtest/gtest.h>

#include "nyx/support/sha256.hpp"

namespace nyx::verify {
namespace {

// Matches Record(): two executable edits and one declared stub.
constexpr CandidateCoverage kCoverage{1, 2, 1};

constexpr std::string_view kCandidate =
    "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

std::string Record(std::string_view candidate = kCandidate,
                   std::string_view counts = "states 4\nmatched 4\nrefuted 0\ninconclusive 0\n",
                   std::string_view trace = "matched", std::string_view status = "matched") {
  return "NYXCHK01\ncandidate " + std::string(candidate) + "\nscope whole_function\n" +
         std::string(counts) +
         "edits_executable 2\nedits_covered 2\nstubs_declared 1\nstubs_called 1\n"
         "call_trace " +
         std::string(trace) + "\nstatus " + std::string(status) + "\n";
}

TEST(Acceptance, Sha256MatchesPublishedVectors) {
  const auto hex = [](std::string_view text) {
    Sha256 hash;
    hash.Update(text);
    return hash.FinishHex();
  };

  EXPECT_EQ(hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  // Split updates across the block boundary give the same digest.
  Sha256 split;
  const std::string text(130, 'a');
  split.Update(std::string_view(text).substr(0, 63));
  split.Update(std::string_view(text).substr(63));
  EXPECT_EQ(split.FinishHex(), hex(text));
}

TEST(Acceptance, ParsesOnlyAConsistentWholeFunctionRecord) {
  const auto record = ParseCheckRecord(Record());
  ASSERT_TRUE(record);
  EXPECT_EQ(record->candidate, kCandidate);
  EXPECT_EQ(record->states, 4U);
  EXPECT_EQ(record->status, CheckStatus::matched);
  EXPECT_EQ(record->call_trace, CallTrace::matched);
  EXPECT_TRUE(ParseCheckRecord(Record("none")));

  const std::string valid = Record();
  EXPECT_FALSE(ParseCheckRecord(valid.substr(0, valid.size() - 1)));  // missing final LF
  EXPECT_FALSE(ParseCheckRecord(valid + "extra 1\n"));
  EXPECT_FALSE(ParseCheckRecord("NYXCHK02" + valid.substr(8)));
  EXPECT_FALSE(ParseCheckRecord(Record("sha256:0123")));
  EXPECT_FALSE(ParseCheckRecord(
      Record("sha256:0123456789ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef")));
  auto scoped = valid;
  scoped.replace(scoped.find("whole_function"), 14, "region_only___");
  EXPECT_FALSE(ParseCheckRecord(scoped));
  EXPECT_FALSE(
      ParseCheckRecord(Record(kCandidate, "states 04\nmatched 4\nrefuted 0\ninconclusive 0\n")));
  EXPECT_FALSE(
      ParseCheckRecord(Record(kCandidate, "states 4\nmatched 3\nrefuted 0\ninconclusive 0\n")));
  EXPECT_FALSE(
      ParseCheckRecord(Record(kCandidate, "states 4\nmatched 5\nrefuted 0\ninconclusive 0\n")));
  EXPECT_FALSE(
      ParseCheckRecord(Record(kCandidate, "states 4\nmatched 3\nrefuted 0\ninconclusive 1\n")));
  EXPECT_TRUE(ParseCheckRecord(Record(
      kCandidate, "states 4\nmatched 3\nrefuted 0\ninconclusive 1\n", "matched", "inconclusive")));
  EXPECT_FALSE(
      ParseCheckRecord(Record(kCandidate, "states 4\nmatched 3\nrefuted 1\ninconclusive 0\n")));
  EXPECT_TRUE(ParseCheckRecord(Record(
      kCandidate, "states 4\nmatched 3\nrefuted 1\ninconclusive 0\n", "matched", "refuted")));
  EXPECT_FALSE(
      ParseCheckRecord(Record(kCandidate, "states 0\nmatched 0\nrefuted 0\ninconclusive 0\n")));
  EXPECT_TRUE(ParseCheckRecord(Record(
      kCandidate, "states 0\nmatched 0\nrefuted 0\ninconclusive 0\n", "matched", "not_checked")));
  // A matched status needs a compared call trace; a mismatched trace is a refutation.
  EXPECT_FALSE(ParseCheckRecord(
      Record(kCandidate, "states 4\nmatched 4\nrefuted 0\ninconclusive 0\n", "not_checked")));
  EXPECT_TRUE(
      ParseCheckRecord(Record(kCandidate, "states 4\nmatched 4\nrefuted 0\ninconclusive 0\n",
                              "not_checked", "inconclusive")));
  EXPECT_FALSE(ParseCheckRecord(
      Record(kCandidate, "states 4\nmatched 4\nrefuted 0\ninconclusive 0\n", "mismatch")));
  EXPECT_FALSE(ParseCheckRecord(Record(
      kCandidate, "states 4\nmatched 4\nrefuted 0\ninconclusive 0\n", "matched", "accepted")));
  auto uncovered = valid;
  uncovered.replace(uncovered.find("edits_covered 2"), 15, "edits_covered 1");
  EXPECT_FALSE(ParseCheckRecord(uncovered));
  auto overcovered = valid;
  overcovered.replace(overcovered.find("edits_covered 2"), 15, "edits_covered 3");
  EXPECT_FALSE(ParseCheckRecord(overcovered));
  auto uncalled = valid;
  uncalled.replace(uncalled.find("stubs_called 1"), 14, "stubs_called 0");
  EXPECT_FALSE(ParseCheckRecord(uncalled));
  EXPECT_FALSE(ParseCheckRecord(Record() + std::string(4096, ' ')));
}

TEST(Acceptance, DecidesOnlyForTheExactCandidate) {
  EXPECT_EQ(Decide(nullptr, kCandidate, kCoverage).status, Acceptance::provisional);
  EXPECT_STREQ(Decide(nullptr, kCandidate, kCoverage).reason, "whole_function_check_not_run");

  const auto matched = *ParseCheckRecord(Record());
  const auto accepted = Decide(&matched, kCandidate, kCoverage);
  EXPECT_EQ(accepted.status, Acceptance::accepted);
  EXPECT_STREQ(accepted.reason, "none");
  const auto other = Decide(&matched, "sha256:" + std::string(64, 'f'), kCoverage);
  EXPECT_EQ(other.status, Acceptance::provisional);
  EXPECT_STREQ(other.reason, "check_for_other_candidate");
  EXPECT_EQ(Decide(&matched, kCandidate, CandidateCoverage{0, 2, 1}).status,
            Acceptance::provisional);
  // The record's denominators must be the candidate's own.
  for (const auto coverage :
       {CandidateCoverage{1, 3, 1}, CandidateCoverage{1, 2, 0}, CandidateCoverage{1, 0, 1}}) {
    const auto differs = Decide(&matched, kCandidate, coverage);
    EXPECT_EQ(differs.status, Acceptance::provisional);
    EXPECT_STREQ(differs.reason, "check_coverage_differs");
  }

  const auto refuted = *ParseCheckRecord(
      Record(kCandidate, "states 4\nmatched 3\nrefuted 1\ninconclusive 0\n", "matched", "refuted"));
  EXPECT_EQ(Decide(&refuted, kCandidate, kCoverage).status, Acceptance::rejected);

  // A refutation of another candidate says nothing about this one.
  EXPECT_EQ(Decide(&refuted, "sha256:" + std::string(64, 'f'), kCoverage).status,
            Acceptance::provisional);

  const auto partial = *ParseCheckRecord(Record(
      kCandidate, "states 4\nmatched 3\nrefuted 0\ninconclusive 1\n", "matched", "inconclusive"));
  EXPECT_EQ(Decide(&partial, kCandidate, kCoverage).status, Acceptance::provisional);
  EXPECT_STREQ(Decide(&partial, kCandidate, kCoverage).reason, "whole_function_check_inconclusive");
  const auto untraced =
      *ParseCheckRecord(Record(kCandidate, "states 4\nmatched 4\nrefuted 0\ninconclusive 0\n",
                               "not_checked", "inconclusive"));
  EXPECT_EQ(Decide(&untraced, kCandidate, kCoverage).status, Acceptance::provisional);
}

}  // namespace
}  // namespace nyx::verify
