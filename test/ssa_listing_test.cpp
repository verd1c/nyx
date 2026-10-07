#include <array>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "nyx/analysis/cfg.hpp"
#include "nyx/analysis/regions.hpp"
#include "nyx/analysis/ssa/build.hpp"
#include "nyx/analysis/unflatten.hpp"
#include "nyx/ir/ssa/listing.hpp"
#include "nyx/recovery/ssa/simplify.hpp"
#include "nyx/target/a64/decode.hpp"

namespace nyx::ir {
namespace {

Budget Plenty() { return Budget({10000000, 10000000}); }

constexpr std::array<SsaStorageName, 3> kNames{SsaStorageName{8, "x8"}, SsaStorageName{30, "x30"},
                                               SsaStorageName{31, "sp"}};
constexpr std::array<StorageId, 2> kObserved{8, 30};

// adr puts an image location in x8, a branch separates the blocks, and the
// call goes through x8: the target is formed in one block and used in another,
// which is the case the reading view has to follow to name it.
constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 5> kWords{{{0x100, 0x10000088},
                                                                         {0x104, 0xd503201f},
                                                                         {0x108, 0x14000001},
                                                                         {0x10c, 0xd63f0100},
                                                                         {0x110, 0xd65f03c0}}};

// The same location, but the call sits inside a loop three blocks round, so
// the value reaching it comes back to itself through two other blocks. Each
// of those has one predecessor and contributes nothing of its own.
//   0x100 adr x8,0x118 ; 0x104 b 0x108
//   0x108 cbz x9,0x114 ; 0x10c b 0x110 ; 0x110 b 0x108
//   0x114 blr x8       ; 0x118 ret
constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 7> kLoopWords{{{0x100, 0x100000c8},
                                                                             {0x104, 0x14000001},
                                                                             {0x108, 0xb4000069},
                                                                             {0x10c, 0x14000001},
                                                                             {0x110, 0x17fffffe},
                                                                             {0x114, 0xd63f0100},
                                                                             {0x118, 0xd65f03c0}}};

// The same loop, but one block inside it writes x8 from a register the
// caller left, so arriving round the loop is not arriving with the location.
//   0x100 adr x8,0x138 ; 0x104 b 0x108
//   0x108 L: cbz x10,0x130 ; 0x10c b 0x110 ; 0x110 cbz x11,0x11c
//   0x114 b 0x118 ; 0x118 b 0x108
//   0x11c mov x8,x12 ; 0x120 b 0x118
//   0x130 blr x8 ; 0x134 ret ; 0x138 ret
constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 16> kOverwrittenWords{
    {{0x100, 0x100001c8},
     {0x104, 0x14000001},
     {0x108, 0xb400014a},
     {0x10c, 0x14000001},
     {0x110, 0xb400006b},
     {0x114, 0x14000001},
     {0x118, 0x17fffffc},
     {0x11c, 0xaa0c03e8},
     {0x120, 0x17fffffe},
     {0x124, 0xd503201f},
     {0x128, 0xd503201f},
     {0x12c, 0xd503201f},
     {0x130, 0xd63f0100},
     {0x134, 0xd65f03c0},
     {0x138, 0xd65f03c0},
     {0x13c, 0xd65f03c0}}};

std::optional<SsaGraph> Built(std::span<const std::uint64_t> entries, Budget& budget,
                              std::span<const std::pair<std::uint64_t, std::uint32_t>> words,
                              SsaObservability observed = {},
                              std::vector<std::uint64_t> noreturn = {}) {
  std::vector<analysis::SourceRecord> records;
  for (const auto [address, word] : words) {
    const std::array<std::uint8_t, 4> bytes{
        static_cast<std::uint8_t>(word), static_cast<std::uint8_t>(word >> 8),
        static_cast<std::uint8_t>(word >> 16), static_cast<std::uint8_t>(word >> 24)};
    auto decoded =
        a64::Decode(address, bytes, budget, {a64::MemoryProfile::concrete_atomic_scalar});
    if (!decoded.group) return {};
    records.push_back({address,
                       {bytes.begin(), bytes.end()},
                       std::move(*decoded.group),
                       analysis::OpaqueReason::none});
  }

  auto cfg = analysis::BuildCfg(records, entries, budget);
  if (!cfg.cfg) return {};
  analysis::Regions regions(std::move(*cfg.cfg), {});
  auto unflattened = analysis::Unflatten(regions, {}, budget, {noreturn, true});
  if (!unflattened.unflattening) return {};
  const std::vector<std::optional<RecoveredPath>> paths(regions.candidates().size());
  auto built = analysis::BuildSsa(regions, *unflattened.unflattening, paths, budget, nullptr, {},
                                  std::move(observed));
  if (!built.graph) return {};

  // The pipeline folds image arithmetic before anything reads the graph, and
  // a target only names a location once it is one.
  auto simplified = recovery::ProposeSsaSimplify(*built.graph, {}, false, budget);
  if (simplified.provisional) return std::move(*simplified.provisional);
  return std::move(*built.graph);
}

std::string Listing(std::span<const std::uint64_t> entries, std::vector<SsaCallResult> results = {},
                    std::span<const std::pair<std::uint64_t, std::uint32_t>> words = kWords) {
  auto budget = Plenty();
  auto graph = Built(entries, budget, words);
  EXPECT_TRUE(graph);
  if (!graph) return "";
  graph->SetCallResults(std::move(results));
  const auto result = ListSsa(*graph, {kNames, kObserved, kObserved}, budget);
  EXPECT_EQ(result.reason, SsaDecline::none);
  return result.text.value_or("");
}

TEST(SsaListing, NamesACallTargetFormedInAnotherBlock) {
  constexpr std::array<std::uint64_t, 1> entries{0x100};
  const auto text = Listing(entries);
  EXPECT_NE(text.find("call @0x110 "), std::string::npos) << text;
}

TEST(SsaListing, WillNotNameATargetAnEntryPathCouldOverride) {
  // The calling block is a graph entry too, so on that path x8 holds whatever
  // the caller left, not the location the earlier block computes. The entry
  // path contributes no phi input, so agreement among the inputs that are
  // listed is not agreement among the ways of arriving.
  constexpr std::array<std::uint64_t, 2> entries{0x100, 0x10c};
  const auto text = Listing(entries);
  EXPECT_EQ(text.find("call @"), std::string::npos) << text;
  EXPECT_NE(text.find("call [x8]"), std::string::npos) << text;
}

TEST(SsaListing, SaysWhichValuesAtACallCameFromRunningIt) {
  // A constant folded from a declared result rests on one execution. A reader
  // who cannot see which values those were reads it as something the image
  // states, so the call carries them.
  constexpr std::array<std::uint64_t, 1> entries{0x100};
  const auto text = Listing(entries, {{0x110, 8, 0x1234}});
  EXPECT_NE(text.find("; one run left x8=0x1234"), std::string::npos) << text;
}

TEST(SsaListing, SaysNothingAboutACallNoRunReached) {
  // The result belongs to another callee, so this call carries no note.
  constexpr std::array<std::uint64_t, 1> entries{0x100};
  const auto text = Listing(entries, {{0x200, 8, 0x1234}});
  EXPECT_EQ(text.find("; one run left"), std::string::npos) << text;
}

TEST(SsaListing, NamesATargetCarriedRoundALoopSeveralBlocksLong) {
  // Nothing in the loop touches x8, so every way of arriving at the call
  // computes the same location. A value coming back to itself constrains
  // nothing, however many blocks it passes through on the way.
  constexpr std::array<std::uint64_t, 1> entries{0x100};
  const auto text = Listing(entries, {}, kLoopWords);
  EXPECT_NE(text.find("call @0x118"), std::string::npos) << text;
}

TEST(SsaListing, WillNotNameATargetOneWayRoundTheLoopOverwrites) {
  // A value that came back changed is not the loop carrying it unchanged.
  // Reaching the call through 0x11c leaves whatever the caller put in x12,
  // so the ways of arriving do not agree and nothing may be named.
  constexpr std::array<std::uint64_t, 1> entries{0x100};
  const auto text = Listing(entries, {}, kOverwrittenWords);
  EXPECT_EQ(text.find("call @"), std::string::npos) << text;
}

// What a call is handed is shown where it is set, even though the call leaves
// the register fresh and nothing after it reads what was there.
//   0x100 mov x0,x19 ; 0x104 b 0x108 ; 0x108 bl 0x200 ; 0x10c ret
TEST(SsaListing, ShowsWhatACallIsHanded) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 4> words{
      {{0x100, 0xaa1303e0}, {0x104, 0x14000001}, {0x108, 0x9400003e}, {0x10c, 0xd65f03c0}}};
  constexpr std::array<SsaStorageName, 4> names{SsaStorageName{0, "x0"}, SsaStorageName{19, "x19"},
                                                SsaStorageName{30, "x30"},
                                                SsaStorageName{31, "sp"}};
  constexpr std::array<StorageId, 2> observed{0, 30};
  constexpr std::array<std::uint64_t, 1> entries{0x100};
  auto budget = Plenty();
  auto graph = Built(entries, budget, words);
  ASSERT_TRUE(graph);
  const auto result = ListSsa(*graph, {names, observed, observed}, budget);
  ASSERT_TRUE(result.text);
  const auto& text = *result.text;
  const auto handed = text.find("    x0 = x19\n");
  ASSERT_NE(handed, std::string::npos) << text;
  EXPECT_LT(handed, text.find("call @0x200")) << text;

  // The link the call writes itself is the call's, not something it is handed.
  EXPECT_EQ(text.find("x30 = "), std::string::npos) << text;
}

// A register the declared contract says callees preserve still holds, after
// the call, what was put in it before; without the contract it comes back
// fresh and the write before the call is dead.
//   0x100 mov x19,x0 ; 0x104 bl 0x200 ; 0x108 mov x0,x19 ; 0x10c ret
TEST(SsaListing, AValueKeptAcrossACallIsShownWhereItIsSet) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 4> words{
      {{0x100, 0xaa0003f3}, {0x104, 0x9400003f}, {0x108, 0xaa1303e0}, {0x10c, 0xd65f03c0}}};
  constexpr std::array<SsaStorageName, 3> names{SsaStorageName{0, "x0"}, SsaStorageName{19, "x19"},
                                                SsaStorageName{30, "x30"}};
  constexpr std::array<StorageId, 1> at_return{0};
  constexpr std::array<std::uint64_t, 1> entries{0x100};
  for (const bool declared : {false, true}) {
    SsaObservability contract;
    contract.declared = declared;
    if (declared) contract.preserved = {19, 31};
    auto budget = Plenty();
    auto graph = Built(entries, budget, words, contract);
    ASSERT_TRUE(graph);
    const auto result = ListSsa(*graph, {names, {}, at_return}, budget);
    ASSERT_TRUE(result.text);
    EXPECT_EQ(result.text->find("    x19 = x0\n") != std::string::npos, declared) << *result.text;
  }
}

// Only the register the call reads its target from may be left out of what it
// is handed; another register holding the same value is still an argument.
//   0x100 ldr x8,[x1] ; 0x104 mov x0,x8 ; 0x108 blr x8 ; 0x10c ret
TEST(SsaListing, ShowsAnArgumentThatHappensToHoldTheTarget) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 4> words{
      {{0x100, 0xf9400028}, {0x104, 0xaa0803e0}, {0x108, 0xd63f0100}, {0x10c, 0xd65f03c0}}};
  constexpr std::array<SsaStorageName, 4> names{SsaStorageName{0, "x0"}, SsaStorageName{1, "x1"},
                                                SsaStorageName{8, "x8"}, SsaStorageName{30, "x30"}};
  constexpr std::array<StorageId, 3> observed{0, 8, 30};
  constexpr std::array<std::uint64_t, 1> entries{0x100};
  auto budget = Plenty();
  auto graph = Built(entries, budget, words);
  ASSERT_TRUE(graph);
  const auto result = ListSsa(*graph, {names, observed, observed}, budget);
  ASSERT_TRUE(result.text);
  EXPECT_NE(result.text->find("    x0 = "), std::string::npos) << *result.text;
}

// A call that never returns still writes its link; the listing attributes that
// write to the call and does not show it on the block.
//   0x100 bl 0x200 (declared not to return)
TEST(SsaListing, LeavesOutTheLinkOfACallWithNoContinuation) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 1> words{{{0x100, 0x94000040}}};
  constexpr std::array<SsaStorageName, 1> names{SsaStorageName{30, "x30"}};
  constexpr std::array<StorageId, 1> observed{30};
  constexpr std::array<std::uint64_t, 1> entries{0x100};
  auto budget = Plenty();
  auto graph = Built(entries, budget, words, {}, {0x200});
  ASSERT_TRUE(graph);
  const auto result = ListSsa(*graph, {names, observed, observed}, budget);
  ASSERT_TRUE(result.text);
  EXPECT_NE(result.text->find("call @0x200"), std::string::npos) << *result.text;
  EXPECT_EQ(result.text->find("x30 = "), std::string::npos) << *result.text;
}

// A value kept across a call is used once, as what the register holds there,
// so it reads inline rather than as a name.
//   0x100 add x19,x0,#1 ; 0x104 bl 0x200 ; 0x108 mov x0,x19 ; 0x10c ret
TEST(SsaListing, AValueKeptAcrossACallReadsInline) {
  constexpr std::array<std::pair<std::uint64_t, std::uint32_t>, 4> words{
      {{0x100, 0x91000413}, {0x104, 0x9400003f}, {0x108, 0xaa1303e0}, {0x10c, 0xd65f03c0}}};
  constexpr std::array<SsaStorageName, 3> names{SsaStorageName{0, "x0"}, SsaStorageName{19, "x19"},
                                                SsaStorageName{30, "x30"}};
  constexpr std::array<StorageId, 2> at_call{0, 19};
  constexpr std::array<StorageId, 1> at_return{0};
  constexpr std::array<std::uint64_t, 1> entries{0x100};
  SsaObservability contract;
  contract.declared = true;
  contract.preserved = {19, 31};
  auto budget = Plenty();
  auto graph = Built(entries, budget, words, contract);
  ASSERT_TRUE(graph);
  const auto result = ListSsa(*graph, {names, at_call, at_return}, budget);
  ASSERT_TRUE(result.text);
  EXPECT_NE(result.text->find("    x19 = (x0 + 1)\n"), std::string::npos) << *result.text;
}

}  // namespace
}  // namespace nyx::ir
