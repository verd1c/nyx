#include "nyx/eval/path.hpp"

#include <gtest/gtest.h>

#include "nyx/ir/print.hpp"

namespace nyx {
namespace {

Budget Plenty() { return Budget({UINT64_MAX, UINT64_MAX}); }

std::vector<ir::Group> Itinerary(bool conditional = false, bool fault = false) {
  std::vector<ir::Group> groups;
  groups.emplace_back(0x100, std::vector<std::uint8_t>{1, 2},
                      std::vector<ir::Node>{{ir::Op::constant, 64, {}, 0x4000},
                                            {ir::Op::constant, 64, {}, 42},
                                            {ir::Op::store, 64, {0, 1}}},
                      std::vector<ir::Write>{{100, 1}}, ir::MemoryModel::atomic_scalar_reference);
  if (conditional) {
    groups.emplace_back(0x102, std::vector<std::uint8_t>{3, 4},
                        std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x200},
                                              {ir::Op::image_address, 64, {}, 0x900},
                                              {ir::Op::read, 1, {}, 0, 33}},
                        std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                        ir::Transfer{ir::TransferKind::conditional, 0, 2, 1, {}});
  } else {
    groups.emplace_back(0x102, std::vector<std::uint8_t>{3, 4},
                        std::vector<ir::Node>{{ir::Op::image_address, 64, {}, 0x200}},
                        std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                        ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}});
  }

  groups.emplace_back(0x200, std::vector<std::uint8_t>{5, 6},
                      std::vector<ir::Node>{{ir::Op::constant, 64, {}, fault ? 0x5000U : 0x4000U},
                                            {ir::Op::load, 64, {0}},
                                            {ir::Op::read, 64, {}, 0, 100},
                                            {ir::Op::add, 64, {1, 2}},
                                            {ir::Op::image_address, 64, {}, 0x300},
                                            {ir::Op::image_address, 64, {}, 0x400},
                                            {ir::Op::read, 1, {}, 0, 33}},
                      std::vector<ir::Write>{{200, 3}}, ir::MemoryModel::atomic_scalar_reference,
                      ir::Transfer{ir::TransferKind::conditional, 4, 6, 5, {}});
  return groups;
}

eval::State State(bool condition = true) {
  auto budget = Plenty();
  eval::State state;
  for (const auto& [id, value] : {std::pair{100U, 7U}, {200U, 11U}, {33U, unsigned(condition)}}) {
    auto bits = BitVector::from_u64(id == 33 ? 1 : 64, value, 64, budget);
    state.cells.push_back({id, std::move(*bits)});
  }

  return state;
}

eval::Memory Memory() {
  auto budget = Plenty();
  const std::array<std::uint8_t, 8> bytes{};
  const eval::RegionInput regions[] = {{0x4000, bytes}};
  return std::move(*eval::Memory::Create(regions, budget).memory);
}

TEST(Path, KeepsNoncontiguousOriginalsInternalTransfersAndCrossStepDefinitions) {
  const auto sources = Itinerary();
  auto budget = Plenty();
  EXPECT_EQ(ir::Normalize(sources, budget).reason, ir::BlockDecline::unsupported_control);
  const auto result = ir::NormalizePath(sources, budget);
  ASSERT_TRUE(result.path);
  ASSERT_EQ(result.path->sources().size(), 3);
  EXPECT_EQ(result.path->sources()[2].source_address(), 0x200);
  EXPECT_EQ(result.path->sources()[2].bytes()[0], 5);
  EXPECT_EQ(result.path->expected_successor(0), 0x102);
  EXPECT_EQ(result.path->expected_successor(1), 0x200);
  EXPECT_FALSE(result.path->expected_successor(2));
  ASSERT_TRUE(result.path->boundaries()[1].transfer);
  EXPECT_EQ(result.path->boundaries()[1].transfer->kind, ir::TransferKind::jump);
  const auto first = result.path->boundaries()[2].first_node;
  const auto& sum = result.path->nodes()[first + 2];
  EXPECT_EQ(sum.op, ir::Op::add);
  EXPECT_EQ(sum.inputs[1], 1);
}

TEST(Path, ExclusiveMonitorSurvivesNormalizationAndTracksConditionalWrite) {
  for (const bool changed : {false, true}) {
    std::vector<ir::Group> groups;
    groups.emplace_back(
        0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
        std::vector<ir::Node>{
            {ir::Op::constant, 64, {}, 0x4000},
            {ir::Op::exclusive_load, 32, {0}, 0, 0, {ir::ByteOrder::little, 4, true}}},
        std::vector<ir::Write>{{100, 1}}, ir::MemoryModel::qemu_exclusive_scalar_reference);
    groups.emplace_back(0x104, std::vector<std::uint8_t>{5, 6, 7, 8},
                        std::vector<ir::Node>{{ir::Op::constant, 64, {}, 0x4000},
                                              {ir::Op::constant, 32, {}, changed ? 77U : 42U},
                                              {ir::Op::store, 32, {0, 1}}},
                        std::vector<ir::Write>{}, ir::MemoryModel::atomic_scalar_reference);
    groups.emplace_back(
        0x108, std::vector<std::uint8_t>{9, 10, 11, 12},
        std::vector<ir::Node>{
            {ir::Op::constant, 64, {}, 0x4000},
            {ir::Op::constant, 32, {}, 55},
            {ir::Op::exclusive_store, 32, {0, 1}, 0, 0, {ir::ByteOrder::little, 4, true}}},
        std::vector<ir::Write>{{200, 2}}, ir::MemoryModel::qemu_exclusive_scalar_reference);
    auto budget = Plenty();
    auto normalized = ir::NormalizePath(groups, budget);
    ASSERT_TRUE(normalized.path);
    EXPECT_EQ(ir::ValidatePath(*normalized.path, budget), ir::BlockDecline::none);
    eval::State state;
    for (const auto id : {100U, 200U}) {
      auto zero = BitVector::from_u64(32, 0, 64, budget);
      ASSERT_TRUE(zero);
      state.cells.push_back({id, std::move(*zero)});
    }

    const std::array<std::uint8_t, 8> bytes{42};
    const eval::RegionInput region{0x4000, bytes};
    auto created = eval::Memory::Create(std::span(&region, 1), budget);
    ASSERT_TRUE(created.memory);
    auto memory = std::move(*created.memory);
    const auto result = eval::ExecutePath(*normalized.path, state, memory, budget, {}, {0});
    ASSERT_EQ(result.outcome, eval::Outcome::completed);
    ASSERT_EQ(result.trace.size(), 3);
    EXPECT_EQ(result.completed_boundaries, 3);
    EXPECT_EQ(state.cells[0].value.word(0), 42);
    EXPECT_EQ(state.cells[1].value.word(0), changed ? 1 : 0);
    EXPECT_FALSE(state.exclusive);
    EXPECT_EQ(memory.Regions()[0].bytes[0], changed ? 77 : 55);
    ASSERT_EQ(result.trace[2].events.size(), 1);
    EXPECT_TRUE(result.trace[2].events[0].conditional);
    EXPECT_EQ(result.trace[2].events[0].performed, !changed);
    EXPECT_EQ(result.trace[2].events[0].write, true);
  }
}

TEST(Path, RejectsChangedAlignmentDeclinePolicyInEffectiveStore) {
  auto sources = Itinerary();
  auto first_nodes = std::vector<ir::Node>(sources[0].nodes().begin(), sources[0].nodes().end());
  first_nodes[2].access.alignment = 8;
  sources[0] = ir::Group(0x100, {1, 2}, std::move(first_nodes), {{100, 1}},
                         ir::MemoryModel::atomic_scalar_reference);
  auto budget = Plenty();
  auto normalized = ir::NormalizePath(sources, budget);
  ASSERT_TRUE(normalized.path);
  EXPECT_EQ(ir::ValidatePath(*normalized.path, budget), ir::BlockDecline::none);
  auto nodes =
      std::vector<ir::Node>(normalized.path->nodes().begin(), normalized.path->nodes().end());
  nodes[2].access.decline_on_unaligned = true;
  ir::Path forged({normalized.path->sources().begin(), normalized.path->sources().end()},
                  std::move(nodes),
                  {normalized.path->origins().begin(), normalized.path->origins().end()},
                  {normalized.path->boundaries().begin(), normalized.path->boundaries().end()});
  EXPECT_EQ(ir::ValidatePath(forged, budget), ir::BlockDecline::invalid_ir);
}

TEST(Path, ExecutesMatchingTailJumpAndReportsTerminalConditionalTransfer) {
  auto budget = Plenty();
  const auto path = ir::NormalizePath(Itinerary(), budget);
  ASSERT_TRUE(path.path);
  auto state = State();
  auto memory = Memory();
  const auto result = eval::ExecutePath(*path.path, state, memory, budget, {}, {0x1000});
  EXPECT_EQ(result.outcome, eval::Outcome::completed);
  EXPECT_EQ(result.stop, eval::PathStop::completed);
  EXPECT_EQ(result.completed_boundaries, 3);
  EXPECT_EQ(result.source_address, 0x200);
  EXPECT_EQ(result.runtime_next, 0x1300);
  ASSERT_EQ(result.trace.size(), 3);
  EXPECT_EQ(result.trace[1].transfer->target, 0x1200);
  EXPECT_EQ(result.trace[2].transfer->condition, true);
  EXPECT_EQ(result.trace[2].events[0].operation, 1);
  EXPECT_EQ(state.cells[0].value.word(0), 42);
  EXPECT_EQ(state.cells[1].value.word(0), 84);
  EXPECT_EQ(memory.Regions()[0].bytes[0], 42);
}

TEST(Path, DivergentActualSuccessorStopsAfterCommittedBranch) {
  auto budget = Plenty();
  const auto path = ir::NormalizePath(Itinerary(true), budget);
  ASSERT_TRUE(path.path);
  auto state = State(false);
  auto memory = Memory();
  const auto result = eval::ExecutePath(*path.path, state, memory, budget, {}, {0x1000});
  EXPECT_EQ(result.outcome, eval::Outcome::completed);
  EXPECT_EQ(result.stop, eval::PathStop::diverged);
  EXPECT_EQ(result.completed_boundaries, 2);
  EXPECT_EQ(result.source_address, 0x102);
  EXPECT_EQ(result.runtime_next, 0x1900);
  ASSERT_EQ(result.trace.size(), 2);
  EXPECT_EQ(result.trace.back().transfer->condition, false);
  EXPECT_EQ(state.cells[0].value.word(0), 42);
  EXPECT_EQ(state.cells[1].value.word(0), 11);
  EXPECT_EQ(memory.Regions()[0].bytes[0], 42);
}

TEST(Path, MissingPlacementAndMalformedLateOriginsCannotPublishEffects) {
  auto budget = Plenty();
  auto normalized = ir::NormalizePath(Itinerary(), budget);
  ASSERT_TRUE(normalized.path);
  const auto& path = *normalized.path;
  auto state = State();
  auto memory = Memory();
  const auto missing = eval::ExecutePath(path, state, memory, budget);
  EXPECT_EQ(missing.outcome, eval::Outcome::unsupported);
  EXPECT_EQ(missing.stop, eval::PathStop::declined);
  EXPECT_TRUE(missing.trace.empty());
  auto origins = std::vector<ir::Origin>(path.origins().begin(), path.origins().end());
  origins.back().boundary = 99;
  const ir::Path malformed({path.sources().begin(), path.sources().end()},
                           {path.nodes().begin(), path.nodes().end()}, std::move(origins),
                           {path.boundaries().begin(), path.boundaries().end()});
  const auto result = eval::ExecutePath(malformed, state, memory, budget, {}, {0});
  EXPECT_EQ(result.outcome, eval::Outcome::invalid_group);
  EXPECT_TRUE(result.trace.empty());
  EXPECT_EQ(state.cells[0].value.word(0), 7);
  EXPECT_EQ(memory.Regions()[0].bytes[0], 0);
}

TEST(Path, ModeledFaultRetainsPredecessorEffectsAndLocalOperationIdentity) {
  auto budget = Plenty();
  const auto path = ir::NormalizePath(Itinerary(false, true), budget);
  ASSERT_TRUE(path.path);
  auto state = State();
  auto memory = Memory();
  const auto result = eval::ExecutePath(*path.path, state, memory, budget, {}, {0});
  EXPECT_EQ(result.outcome, eval::Outcome::fault);
  EXPECT_EQ(result.stop, eval::PathStop::fault);
  EXPECT_EQ(result.completed_boundaries, 2);
  EXPECT_EQ(result.source_address, 0x200);
  EXPECT_FALSE(result.runtime_next);
  ASSERT_EQ(result.trace.size(), 3);
  ASSERT_TRUE(result.trace.back().fault);
  EXPECT_EQ(result.trace.back().fault->operation, 1);
  EXPECT_EQ(result.trace.back().fault->address, 0x5000);
  EXPECT_FALSE(result.trace.back().transfer);
  EXPECT_EQ(state.cells[0].value.word(0), 42);
  EXPECT_EQ(state.cells[1].value.word(0), 11);
  EXPECT_EQ(memory.Regions()[0].bytes[0], 42);
}

TEST(Path, RepeatedAndOverlappingSourcesRequireConsistentOriginalBytes) {
  const ir::Group loop(0x100, {1, 2, 3, 4}, {{ir::Op::image_address, 64, {}, 0x100}}, {},
                       ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}});
  auto budget = Plenty();
  const std::vector<ir::Group> repeated{loop, loop};
  const auto path = ir::NormalizePath(repeated, budget);
  ASSERT_TRUE(path.path);
  auto state = State();
  auto memory = Memory();
  const auto ran = eval::ExecutePath(*path.path, state, memory, budget, {}, {0});
  EXPECT_EQ(ran.stop, eval::PathStop::completed);
  EXPECT_EQ(ran.completed_boundaries, 2);
  EXPECT_EQ(ran.runtime_next, 0x100);
  const std::vector<ir::Group> matching{loop, ir::Group(0x102, {3, 4}, {}, {})};
  EXPECT_TRUE(ir::NormalizePath(matching, budget).path);
  const std::vector<ir::Group> conflicting{loop, ir::Group(0x102, {3, 9}, {}, {})};
  EXPECT_EQ(ir::NormalizePath(conflicting, budget).reason, ir::BlockDecline::invalid_source);

  // A short nested source must not hide overlap with the longer covering source.
  const ir::Group nested(0x101, {2}, {{ir::Op::image_address, 64, {}, 0x103}}, {},
                         ir::MemoryModel::unspecified,
                         ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}});
  const std::vector<ir::Group> hidden{loop, nested, ir::Group(0x103, {9}, {}, {})};
  EXPECT_EQ(ir::NormalizePath(hidden, budget).reason, ir::BlockDecline::invalid_source);
}

TEST(Path, DeclinesImplicitJumpsAndAdmitsACallOnlyWhenFollowedToItsSettledTarget) {
  auto budget = Plenty();
  const std::vector<ir::Group> gap{{0x100, {1}, {}, {}}, {0x200, {2}, {}, {}}};
  EXPECT_EQ(ir::NormalizePath(gap, budget).reason, ir::BlockDecline::invalid_source);

  // A direct call transfers to its target whether or not the callee returns, so
  // an itinerary continuing there names the instruction executed next and elides
  // nothing. Obfuscated code reaches this shape by using `bl` as a plain jump.
  const ir::Group call(0x100, {1}, {{ir::Op::image_address, 64, {}, 0x200}}, {},
                       ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::call, 0, {}, {}, 0});
  const std::vector<ir::Group> followed{call, ir::Group(0x200, {2}, {}, {})};
  EXPECT_TRUE(ir::NormalizePath(followed, budget).path);

  // Continuing at the link address instead would splice out the callee body.
  const ir::Group linked(
      0x100, {1}, {{ir::Op::image_address, 64, {}, 0x300}, {ir::Op::image_address, 64, {}, 0x101}},
      {}, ir::MemoryModel::unspecified, ir::Transfer{ir::TransferKind::call, 0, {}, {}, 1});
  const std::vector<ir::Group> continuation{linked, ir::Group(0x101, {2}, {}, {})};
  EXPECT_EQ(ir::NormalizePath(continuation, budget).reason, ir::BlockDecline::unsupported_control);

  // A computed target names no address in the group, so it stays declined.
  const ir::Group computed(0x100, {1}, {{ir::Op::read, 64, {}, 0, 5}}, {},
                           ir::MemoryModel::unspecified,
                           ir::Transfer{ir::TransferKind::call, 0, {}, {}, 0});
  const std::vector<ir::Group> indirect{computed, ir::Group(0x200, {2}, {}, {})};
  EXPECT_EQ(ir::NormalizePath(indirect, budget).reason, ir::BlockDecline::unsupported_control);

  // An interior return still names no successor the itinerary may continue at.
  const ir::Group ret(0x100, {1}, {{ir::Op::image_address, 64, {}, 0x200}}, {},
                      ir::MemoryModel::unspecified,
                      ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}});
  const std::vector<ir::Group> returning{ret, ir::Group(0x200, {2}, {}, {})};
  EXPECT_EQ(ir::NormalizePath(returning, budget).reason, ir::BlockDecline::unsupported_control);
  const std::vector<ir::Group> terminal{call};
  EXPECT_TRUE(ir::NormalizePath(terminal, budget).path);
  const std::vector<ir::Group> wrap{{UINT64_MAX, {1}, {}, {}}, {0, {2}, {}, {}}};
  const auto path = ir::NormalizePath(wrap, budget);
  ASSERT_TRUE(path.path);
  auto state = State();
  auto memory = Memory();
  EXPECT_EQ(eval::ExecutePath(*path.path, state, memory, budget, {}, {0}).stop,
            eval::PathStop::completed);
}

TEST(Path, JsonRetainsSourceLocationsTransfersAndExplicitItineraryGuards) {
  auto budget = Plenty();
  const auto path = ir::NormalizePath(Itinerary(true), budget);
  ASSERT_TRUE(path.path);
  const auto printed = ir::PrintJson(*path.path, budget);
  ASSERT_TRUE(printed.json);
  EXPECT_TRUE(printed.json->starts_with("{\"schema\":1,\"kind\":\"nyx.ir.path\",\"revision\":0,"));
  EXPECT_NE(printed.json->find("\"entry_scope\":\"selected_itinerary\""), std::string::npos);
  EXPECT_NE(printed.json->find("\"source_address\":512,\"bytes\":\"0506\""), std::string::npos);
  EXPECT_NE(printed.json->find("\"expected_successor_image\":512"), std::string::npos);
  EXPECT_TRUE(printed.json->ends_with("\"expected_successor_image\":null}]}"));
}

TEST(Path, NormalizationAndPrintingResourceDeclinesPublishNoPartialArtifact) {
  const auto sources = Itinerary();
  auto full = Plenty();
  const auto normalized = ir::NormalizePath(sources, full);
  ASSERT_TRUE(normalized.path);
  for (bool bytes : {false, true}) {
    const auto maximum = bytes ? full.used().bytes : full.used().work;
    for (std::uint64_t cut = 0; cut < maximum; ++cut) {
      Budget budget({bytes ? full.used().work : cut, bytes ? cut : full.used().bytes});
      const auto result = ir::NormalizePath(sources, budget);
      ASSERT_FALSE(result.path) << cut;
      EXPECT_EQ(result.reason, ir::BlockDecline::resource_limit);
    }
  }

  auto printed_budget = Plenty();
  const auto printed = ir::PrintJson(*normalized.path, printed_budget);
  ASSERT_TRUE(printed.json);
  Budget exact(printed_budget.used());
  EXPECT_EQ(ir::PrintJson(*normalized.path, exact).json, printed.json);
  Budget short_bytes({printed_budget.used().work, printed_budget.used().bytes - 1});
  const auto bytes = ir::PrintJson(*normalized.path, short_bytes);
  EXPECT_FALSE(bytes.json);
  EXPECT_EQ(bytes.reason, ir::PrintDecline::byte_limit);
  Budget short_work({printed_budget.used().work - 1, printed_budget.used().bytes});
  const auto work = ir::PrintJson(*normalized.path, short_work);
  EXPECT_FALSE(work.json);
  EXPECT_EQ(work.reason, ir::PrintDecline::work_limit);
  Budget no_scratch({printed_budget.used().work, 0});
  const auto validation = ir::PrintJson(*normalized.path, no_scratch);
  EXPECT_FALSE(validation.json);
  EXPECT_EQ(validation.reason, ir::PrintDecline::resource_limit);
}

TEST(Path, EveryExecutionResourceCutRetainsExactlyTheCompletedPrefix) {
  auto setup = Plenty();
  const auto path = ir::NormalizePath(Itinerary(), setup);
  ASSERT_TRUE(path.path);
  auto full = Plenty();
  auto initial = State();
  auto original = Memory();
  eval::Limits limits;
  limits.max_width = 64;
  ASSERT_EQ(eval::ExecutePath(*path.path, initial, original, full, limits, {0}).stop,
            eval::PathStop::completed);
  bool saw_predecessor = false;
  for (bool bytes : {false, true}) {
    const auto maximum = bytes ? full.used().bytes : full.used().work;
    for (std::uint64_t cut = 0; cut < maximum; ++cut) {
      Budget budget({bytes ? full.used().work : cut, bytes ? cut : full.used().bytes});
      auto state = State();
      auto memory = Memory();
      const auto result = eval::ExecutePath(*path.path, state, memory, budget, limits, {0});
      ASSERT_EQ(result.outcome, eval::Outcome::resource_limit) << cut;
      EXPECT_EQ(result.stop, eval::PathStop::declined);
      ASSERT_LE(result.completed_boundaries, 2);
      saw_predecessor |= result.completed_boundaries != 0;
      EXPECT_EQ(state.cells[0].value.word(0), result.completed_boundaries ? 42 : 7);
      EXPECT_EQ(state.cells[1].value.word(0), 11);
      EXPECT_EQ(memory.Regions()[0].bytes[0], result.completed_boundaries ? 42 : 0);
      EXPECT_FALSE(result.runtime_next);
      if (!result.trace.empty()) {
        EXPECT_TRUE(result.trace.back().events.empty());
      }
    }
  }

  EXPECT_TRUE(saw_predecessor);
}

}  // namespace
}  // namespace nyx
