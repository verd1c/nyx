#include "nyx/analysis/entry_relations.hpp"

#include <algorithm>
#include <array>

#include <gtest/gtest.h>

#include "nyx/eval/path.hpp"
#include "nyx/recovery/memory.hpp"
#include "nyx/target/a64/decode.hpp"

namespace nyx {
namespace {

// A flattened function that keeps its state at [x19+8] and case data through
// x27 = x19+16. Case 0 stores the next state, then writes [x27+8]; case 1 calls
// out before storing. The dispatch reads a halfword table after the code.
constexpr std::uint64_t kBase = 0x1000;
constexpr std::array<std::uint32_t, 27> kFunction{
    0xa9bf7bfd, 0x910003fd, 0xd10103ff, 0x910003f3, 0x9100427b, 0xb9000a7f, 0x1400000d,
    0x52800028, 0xb9000a68, 0xf900077f, 0x14000009, 0x940003f5, 0x52800048, 0xb9000a68,
    0xf900037f, 0x14000004, 0x910003bf, 0xa8c17bfd, 0xd65f03c0, 0xb9400a68, 0x71000909,
    0x54ffff68, 0x100000a9, 0x10fffe0a, 0x7868792b, 0x8b0b094a, 0xd61f0140};
constexpr std::array<std::uint8_t, 8> kTable{0, 0, 4, 0, 9, 0, 0, 0};
constexpr std::uint64_t kCase0 = 0x101c, kCall = 0x102c, kContinuation = 0x1030, kDispatch = 0x104c;

std::array<std::uint8_t, 4> Bytes(std::uint32_t word) {
  return {static_cast<std::uint8_t>(word), static_cast<std::uint8_t>(word >> 8),
          static_cast<std::uint8_t>(word >> 16), static_cast<std::uint8_t>(word >> 24)};
}

ir::Group Decode(std::uint64_t address, std::uint32_t word, Budget& budget) {
  const auto bytes = Bytes(word);
  auto decoded = a64::Decode(address, bytes, budget, {a64::MemoryProfile::concrete_atomic_scalar});
  EXPECT_TRUE(decoded.group) << std::hex << address;
  return std::move(*decoded.group);
}

analysis::Cfg Graph(std::span<const std::uint32_t> words, Budget& budget) {
  std::vector<analysis::SourceRecord> sources;
  for (std::size_t index = 0; index < words.size(); ++index) {
    const auto address = kBase + index * 4;
    const auto bytes = Bytes(words[index]);
    auto decoded =
        a64::Decode(address, bytes, budget, {a64::MemoryProfile::concrete_atomic_scalar});
    if (decoded.group) {
      sources.push_back({address, {bytes.begin(), bytes.end()}, std::move(*decoded.group)});
      continue;
    }

    EXPECT_TRUE(a64::OpaqueTrap(bytes)) << std::hex << address;
    sources.push_back({address,
                       {bytes.begin(), bytes.end()},
                       {},
                       analysis::OpaqueReason::unsupported,
                       analysis::OpaqueControl::trap});
  }

  const std::array<ir::ConstantImageRange, 1> table{{{kBase + words.size() * 4, kTable}}};
  const std::array<std::uint64_t, 1> entries{kBase};
  auto built = analysis::BuildCfg(sources, entries, budget, {}, ir::ImageFacts{table, {}, false});
  EXPECT_TRUE(built.cfg);
  return std::move(*built.cfg);
}

using Relations = std::vector<ir::StorageRelation>;
const Relations kFrame{{27, 19, 16}, {29, 19, 64}, {a64::kSp, 19, 0}};

const Relations& At(const analysis::Cfg& graph, const analysis::EntryRelations& proved,
                    std::uint64_t address) {
  for (std::size_t id = 0; id < graph.blocks().size(); ++id) {
    if (graph.sources()[graph.blocks()[id].first_source].address == address)
      return proved.blocks[id];
  }

  ADD_FAILURE() << std::hex << address;
  return proved.blocks.front();
}

analysis::EntryRelations Prove(std::span<const std::uint32_t> words, Budget& budget, bool abi,
                               std::span<const ir::StorageId> preserved = a64::kCalleeSaved,
                               bool return_leaves = true) {
  auto proved =
      analysis::ProveEntryRelations(Graph(words, budget), budget, {abi, preserved, return_leaves});
  EXPECT_TRUE(proved.relations);
  return std::move(*proved.relations);
}

bool Empty(const analysis::EntryRelations& proved) {
  return std::all_of(proved.blocks.begin(), proved.blocks.end(),
                     [](const Relations& block) { return block.empty(); });
}

TEST(EntryRelations, DeclaredAbiCarriesTheFrameAcrossTheCall) {
  Budget budget({100000000, 100000000});
  const auto graph = Graph(kFunction, budget);
  const auto proved = analysis::ProveEntryRelations(graph, budget, {true, a64::kCalleeSaved, true});
  ASSERT_TRUE(proved.relations);
  EXPECT_TRUE(At(graph, *proved.relations, kBase).empty());
  for (const auto address : {kCase0, kCall, kContinuation, kDispatch}) {
    EXPECT_EQ(At(graph, *proved.relations, address), kFrame) << std::hex << address;
  }

  EXPECT_TRUE(proved.relations->calling_convention);
  EXPECT_TRUE(proved.relations->return_leaves);
  EXPECT_TRUE(proved.relations->constant_image);
  EXPECT_FALSE(proved.relations->declared_opaque_control);
}

TEST(EntryRelations, UndeclaredCallsReturnsAndUnresolvedExitsRefuseEveryRelation) {
  Budget budget({100000000, 100000000});
  EXPECT_TRUE(Empty(Prove(kFunction, budget, false)));

  // The block at 0x1014 is reached without passing the call, but nothing says
  // an undeclared callee could not come back there instead.
  const std::array<std::uint32_t, 6> branch{0x910003f3, 0x9100427b, 0xb4000060,
                                            0x94000400, 0xd65f03c0, 0xd65f03c0};
  EXPECT_TRUE(Empty(Prove(branch, budget, false)));
  const Relations frame{{27, 19, 16}, {a64::kSp, 19, 0}};
  EXPECT_EQ(At(Graph(branch, budget), Prove(branch, budget, true), 0x1014), frame);

  // RET x5 may jump back to 0x1014 with x27 moved; only a declared return
  // discipline rules that out. The trap keeps any other return out of the way.
  const std::array<std::uint32_t, 6> jump{0x9100427b, 0xaa0103e5, 0xb4000060,
                                          0x9100237b, 0xd65f00a0, 0xd4200020};
  EXPECT_TRUE(Empty(Prove(jump, budget, false)));
  EXPECT_TRUE(Empty(Prove(jump, budget, true, a64::kCalleeSaved, false)));
  const Relations displaced{{27, 19, 16}};
  EXPECT_EQ(At(Graph(jump, budget), Prove(jump, budget, true), 0x1014), displaced);
  auto words = kFunction;
  words[11] = 0xd61f0100;  // br x8 in place of the call
  EXPECT_TRUE(Empty(Prove(words, budget, true)));
  const std::array<ir::StorageId, 2> unsorted{29, 19};
  EXPECT_EQ(
      analysis::ProveEntryRelations(Graph(kFunction, budget), budget, {true, unsorted}).reason,
      analysis::RelationDecline::invalid_input);
}

TEST(EntryRelations, TrapNeedsNoDeclarationButIsPublishedAsOne) {
  Budget budget({100000000, 100000000});
  const std::array<std::uint32_t, 4> trap{0x9100427b, 0xb4000040, 0xaa0203e1, 0xd4200020};
  const auto graph = Graph(trap, budget);
  const auto proved = Prove(trap, budget, false);
  const Relations displaced{{27, 19, 16}};
  EXPECT_EQ(At(graph, proved, 0x100c), displaced);
  EXPECT_FALSE(proved.calling_convention);
  EXPECT_TRUE(proved.declared_opaque_control);
}

TEST(EntryRelations, OnlyPreservedStorageSurvivesTheCall) {
  Budget budget({100000000, 100000000});
  const auto graph = Graph(kFunction, budget);
  const std::array<ir::StorageId, 3> preserved{19, 29, a64::kSp};
  const auto proved = Prove(kFunction, budget, true, preserved);
  const Relations kept{{29, 19, 64}, {a64::kSp, 19, 0}};
  EXPECT_EQ(At(graph, proved, kContinuation), kept);
  EXPECT_EQ(At(graph, proved, kDispatch), kept);
}

TEST(EntryRelations, JoinKeepsOnlyWhatEveryArrivalAgreesOn) {
  Budget budget({100000000, 100000000});
  auto words = kFunction;
  words[9] = 0xaa0803fb;  // mov x27, x8
  const Relations kept{{29, 19, 64}, {a64::kSp, 19, 0}};
  EXPECT_EQ(At(Graph(words, budget), Prove(words, budget, true), kDispatch), kept);
  words[9] = 0x9100427b;  // add x27, x19, #16: the same relation, rebuilt
  EXPECT_EQ(At(Graph(words, budget), Prove(words, budget, true), kDispatch), kFrame);

  // A loop that moves x27 on every turn leaves its head with no relation.
  const std::array<std::uint32_t, 5> loop{0x9100427b, 0xb4000060, 0x9100237b, 0x17fffffe,
                                          0xd4200020};
  const auto looping = Graph(loop, budget);
  const auto turned = Prove(loop, budget, false);
  EXPECT_TRUE(At(looping, turned, 0x1004).empty());
  EXPECT_TRUE(At(looping, turned, 0x1010).empty());

  // A W write zero-extends: x27 is not x19 plus anything.
  const std::array<std::uint32_t, 2> narrow{0x1100427b, 0xd4200020};
  EXPECT_TRUE(At(Graph(narrow, budget), Prove(narrow, budget, false), 0x1004).empty());
}

// Case 0 through to the dispatch's state load: the load can take the stored
// state only if [x27+8] is known to be somewhere else.
ir::Path CaseRoute(Budget& budget) {
  std::vector<ir::Group> groups;
  for (const std::size_t index : {7, 8, 9, 10, 19}) {
    groups.push_back(Decode(kBase + index * 4, kFunction[index], budget));
  }

  auto normalized = ir::NormalizePath(groups, budget);
  EXPECT_TRUE(normalized.path);
  return std::move(*normalized.path);
}

TEST(EntryRelations, ForwardingPastAnAliasRestsOnTheRelation) {
  Budget budget({100000000, 100000000});
  const auto route = CaseRoute(budget);
  const auto related = recovery::ForwardMemoryValues(route, budget, {}, kFrame);
  ASSERT_TRUE(related.path);
  ASSERT_EQ(related.facts.size(), 1);
  EXPECT_TRUE(related.facts[0].entry_relation);
  EXPECT_EQ(route.sources()[route.origins()[related.facts[0].load].boundary].source_address(),
            kDispatch);
  EXPECT_FALSE(related.journal.empty());

  const auto unrelated = recovery::ForwardMemoryValues(route, budget);
  ASSERT_TRUE(unrelated.path);
  EXPECT_TRUE(unrelated.facts.empty());

  // x27 = x19 puts [x27+8] on the state itself.
  const Relations aliased{{27, 19, 0}};
  const auto overlapping = recovery::ForwardMemoryValues(route, budget, {}, aliased);
  ASSERT_TRUE(overlapping.path);
  EXPECT_TRUE(overlapping.facts.empty());

  // SP is related to x19, which is read first, but a store and load through SP
  // alone need no relation: the fact neither rests on one nor moves its base.
  std::vector<ir::Group> groups;
  groups.push_back(Decode(0x2000, 0xf9400268, budget));  // ldr x8, [x19]
  groups.push_back(Decode(0x2004, 0xf90007e1, budget));  // str x1, [sp, #8]
  groups.push_back(Decode(0x2008, 0xf9000be3, budget));  // str x3, [sp, #16]
  groups.push_back(Decode(0x200c, 0xf94007e2, budget));  // ldr x2, [sp, #8]
  const auto direct = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(direct.path);
  const auto own = recovery::ForwardMemoryValues(*direct.path, budget, {}, kFrame);
  ASSERT_TRUE(own.path);
  ASSERT_EQ(own.facts.size(), 1);
  EXPECT_FALSE(own.facts[0].entry_relation);
  const auto& base = direct.path->nodes()[own.facts[0].address.value];
  EXPECT_EQ(base.op, ir::Op::read);
  EXPECT_EQ(base.storage, a64::kSp);
  EXPECT_EQ(own.facts[0].address.offset, 8);

  for (const Relations& invalid : {Relations{{19, 5, 0}, {27, 19, 16}},
                                   Relations{{29, 19, 64}, {27, 19, 16}}, Relations{{19, 27, 0}}}) {
    EXPECT_EQ(recovery::ForwardMemoryValues(route, budget, {}, invalid).reason,
              recovery::MemoryRecoveryDecline::invalid_ir);
  }
}

// The forwarded route agrees with the original where the relation holds and
// differs where it fails, so the relation is a required precondition.
TEST(EntryRelations, ForwardedRouteMatchesOnlyWhereTheRelationHolds) {
  Budget budget({100000000, 100000000});
  const auto route = CaseRoute(budget);
  const auto forwarded = recovery::ForwardMemoryValues(route, budget, {}, kFrame);
  ASSERT_TRUE(forwarded.path);
  constexpr std::uint64_t scratch_base = 0x100000000;
  for (const std::uint64_t displacement : {16, 0}) {
    std::array<std::uint8_t, 256> scratch{};
    const eval::RegionInput mapped[] = {{scratch_base, scratch}};
    auto original_memory = eval::Memory::Create(mapped, budget);
    auto forwarded_memory = eval::Memory::Create(mapped, budget);
    ASSERT_TRUE(original_memory.memory);
    ASSERT_TRUE(forwarded_memory.memory);
    const auto state = [&] {
      eval::State result;
      const std::array<std::pair<ir::StorageId, std::uint64_t>, 3> cells{
          {{8, 7}, {19, scratch_base + 64}, {27, scratch_base + 64 + displacement}}};
      for (const auto& [id, value] : cells) {
        auto bits = BitVector::from_u64(64, value, 64, budget);
        EXPECT_TRUE(bits);
        result.cells.push_back({id, std::move(*bits)});
      }

      return result;
    };

    auto original_state = state(), forwarded_state = state();
    const auto original =
        eval::ExecutePath(route, original_state, *original_memory.memory, budget, {}, {0});
    const auto actual = eval::ExecutePath(*forwarded.path, forwarded_state,
                                          *forwarded_memory.memory, budget, {}, {0});
    ASSERT_EQ(original.outcome, eval::Outcome::completed);
    ASSERT_EQ(actual.outcome, eval::Outcome::completed);
    EXPECT_EQ(original_memory.memory->Regions()[0].bytes,
              forwarded_memory.memory->Regions()[0].bytes);
    EXPECT_EQ(original_state.cells[0].value.word(0), displacement ? 1 : 0);
    EXPECT_EQ(forwarded_state.cells[0].value.word(0), 1);
  }
}

}  // namespace
}  // namespace nyx
