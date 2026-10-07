#include <gtest/gtest.h>

#include "nyx/eval/path.hpp"
#include "nyx/recovery/overwritten_store.hpp"

namespace nyx {
namespace {
Budget Plenty() { return Budget({UINT64_MAX, UINT64_MAX}); }

ir::Path Basis(std::uint64_t literal = 1, bool fault = false, bool equal = false,
               bool terminal = false) {
  std::vector<ir::Group> groups;
  groups.emplace_back(
      0x100, std::vector<std::uint8_t>{1, 2, 3, 4},
      std::vector<ir::Node>{{ir::Op::constant, 1, {}, literal},
                            {ir::Op::image_address, 64, {}, 0x200},
                            {ir::Op::image_address, 64, {}, equal ? 0x200U : 0x300U},
                            {ir::Op::constant, 64, {}, 42},
                            {ir::Op::constant, 64, {}, 0x4000},
                            {ir::Op::store, 64, {4, 3}},
                            {ir::Op::constant, 64, {}, fault ? 0x5000U : 0x4000U},
                            {ir::Op::load, 64, {6}}},
      std::vector<ir::Write>{{11, 3}, {33, 0}}, ir::MemoryModel::atomic_scalar_reference,
      ir::Transfer{ir::TransferKind::conditional, 1, 0, 2, {}});
  if (!terminal)
    groups.emplace_back(0x200, std::vector<std::uint8_t>{5, 6, 7, 8},
                        std::vector<ir::Node>{{ir::Op::constant, 64, {}, 99}},
                        std::vector<ir::Write>{{11, 0}});
  auto budget = Plenty();
  auto result = ir::NormalizePath(groups, budget);
  EXPECT_TRUE(result.path);
  return std::move(*result.path);
}

ir::ConditionalRewrite Rewrite(const ir::Path& basis, std::uint32_t boundary = 0) {
  const auto original = *basis.boundaries()[boundary].transfer;
  const bool value = (basis.nodes()[*original.condition].immediate & 1) != 0;
  return {ir::RewriteRule::folded_condition,
          boundary,
          original,
          {ir::TransferKind::jump, value ? original.target : *original.alternative, {}, {}, {}},
          *original.condition,
          value,
          0,
          0,
          {},
          basis.revision(),
          basis.revision() + 1};
}

ir::RecoveredPath Recovered(ir::Path path) {
  const auto rewrite = Rewrite(path);
  return ir::RecoveredPath(std::move(path), {rewrite}, rewrite.to_revision);
}

eval::State State() {
  auto budget = Plenty();
  eval::State state;
  state.cells.push_back({11, std::move(*BitVector::from_u64(64, 7, 64, budget))});
  state.cells.push_back({33, std::move(*BitVector::from_u64(1, 0, 64, budget))});
  return state;
}

eval::Memory Memory() {
  auto budget = Plenty();
  const std::array<std::uint8_t, 8> bytes{};
  const eval::RegionInput input[] = {{0x4000, bytes}};
  return std::move(*eval::Memory::Create(input, budget).memory);
}

ir::Path AdjacentStackStores(std::uint64_t second_offset = 56) {
  std::vector<ir::Group> groups;
  for (const auto [address, offset] : {std::pair{UINT64_C(0x859244), UINT64_C(56)},
                                       std::pair{UINT64_C(0x859248), second_offset}}) {
    groups.emplace_back(address,
                        offset == 56 ? std::vector<std::uint8_t>{0xe0, 0x1f, 0x00, 0xf9}
                                     : std::vector<std::uint8_t>{0xe0, 0x23, 0x00, 0xf9},
                        std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 31},
                                              {ir::Op::constant, 64, {}, offset},
                                              {ir::Op::add, 64, {0, 1}},
                                              {ir::Op::read, 64, {}, 0, 0},
                                              {ir::Op::store, 64, {2, 3}}},
                        std::vector<ir::Write>{}, ir::MemoryModel::atomic_scalar_reference);
  }

  auto budget = Plenty();
  auto normalized = ir::NormalizePath(groups, budget);
  EXPECT_TRUE(normalized.path);
  return std::move(*normalized.path);
}

TEST(RecoveredPath, ExclusiveReadBlocksOverwrittenStoreOmission) {
  std::vector<ir::Group> groups;
  for (const auto address : {UINT64_C(0x859244), UINT64_C(0x85924c)}) {
    groups.emplace_back(address, std::vector<std::uint8_t>{1, 2, 3, 4},
                        std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 31},
                                              {ir::Op::constant, 64, {}, 56},
                                              {ir::Op::add, 64, {0, 1}},
                                              {ir::Op::read, 64, {}, 0, 0},
                                              {ir::Op::store, 64, {2, 3}}},
                        std::vector<ir::Write>{}, ir::MemoryModel::atomic_scalar_reference);
  }

  groups.insert(
      groups.begin() + 1,
      ir::Group(0x859248, {5, 6, 7, 8},
                {{ir::Op::read, 64, {}, 0, 31},
                 {ir::Op::constant, 64, {}, 56},
                 {ir::Op::add, 64, {0, 1}},
                 {ir::Op::exclusive_load, 64, {2}, 0, 0, {ir::ByteOrder::little, 8, true}}},
                {}, ir::MemoryModel::qemu_exclusive_scalar_reference));
  auto budget = Plenty();
  auto normalized = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(normalized.path);
  std::vector<ir::ValueId> stores;
  for (ir::ValueId id = 0; id < normalized.path->nodes().size(); ++id) {
    if (normalized.path->nodes()[id].op == ir::Op::store) stores.push_back(id);
  }

  ASSERT_EQ(stores.size(), 2);
  ir::RecoveredPath forged(*normalized.path, {}, 1, {}, {{stores[0], stores[1], 31, 56, 0, 1}});
  EXPECT_EQ(ir::ValidateRecoveredPath(forged, budget), ir::BlockDecline::invalid_ir);
  auto result = recovery::OmitOverwrittenStores(
      ir::RecoveredPath(std::move(*normalized.path), {}, 0), 31, budget);
  ASSERT_TRUE(result.path);
  EXPECT_TRUE(result.path->omissions().empty());
}

eval::State StackState() {
  auto budget = Plenty();
  eval::State state;
  state.cells.push_back({0, std::move(*BitVector::from_u64(64, 0x1234, 64, budget))});
  state.cells.push_back({31, std::move(*BitVector::from_u64(64, 0x1000, 64, budget))});
  return state;
}

// Only recovery decides a restored conditional, and only long after a path
// is built, so a path carrying that rule is malformed however well formed
// the rest of it looks.
TEST(RecoveredPath, ADecidedDispatchIsNotAPathRule) {
  auto path = Basis();
  auto rewrite = Rewrite(path);
  rewrite.rule = ir::RewriteRule::decided_dispatch;
  ir::RecoveredPath recovered(std::move(path), {rewrite}, rewrite.to_revision);
  auto budget = Plenty();
  EXPECT_EQ(ir::ValidateRecoveredPath(recovered, budget), ir::BlockDecline::invalid_ir);
}

TEST(RecoveredPath, OverwrittenStackStoreNeedsMappedFrameAndSkipsOnlyFirstEffect) {
  auto basis = AdjacentStackStores();
  ASSERT_EQ(basis.nodes().size(), 8);
  ir::RecoveredPath recovered(basis, {}, 1, {}, {{4, 7, 31, 56, 0, 1}});
  auto budget = Plenty();
  ASSERT_EQ(ir::ValidateRecoveredPath(recovered, budget), ir::BlockDecline::none);
  const std::array<std::uint8_t, 128> bytes{};
  const eval::RegionInput region[] = {{0x1000, bytes, true, true}};
  auto original_memory = std::move(*eval::Memory::Create(region, budget).memory);
  auto recovered_memory = std::move(*eval::Memory::Create(region, budget).memory);
  auto original_state = StackState(), recovered_state = StackState();
  const auto original = eval::ExecutePath(basis, original_state, original_memory, budget, {}, {0});
  const auto actual =
      eval::ExecuteRecoveredPath(recovered, recovered_state, recovered_memory, budget, {}, {0});
  EXPECT_EQ(original.outcome, eval::Outcome::completed);
  EXPECT_EQ(actual.outcome, original.outcome);
  EXPECT_EQ(actual.runtime_next, original.runtime_next);
  ASSERT_EQ(original.trace.size(), 2);
  ASSERT_EQ(actual.trace.size(), 2);
  EXPECT_EQ(original.trace[0].events.size(), 1);
  EXPECT_TRUE(actual.trace[0].events.empty());
  EXPECT_EQ(actual.trace[1].events.size(), 1);
  const auto original_bytes = original_memory.Read(0x1038, 8, budget);
  const auto actual_bytes = recovered_memory.Read(0x1038, 8, budget);
  EXPECT_EQ(actual_bytes.status, eval::MemoryStatus::ok);
  EXPECT_EQ(actual_bytes.bytes, original_bytes.bytes);
  EXPECT_EQ(recovered_state.cells[0].value.word(0), original_state.cells[0].value.word(0));

  const std::array<std::uint8_t, 16> elsewhere{};
  const eval::RegionInput missing[] = {{0x2000, elsewhere}};
  auto unmapped = std::move(*eval::Memory::Create(missing, budget).memory);
  auto initial = StackState();
  const auto declined = eval::ExecuteRecoveredPath(recovered, initial, unmapped, budget, {}, {0});
  EXPECT_EQ(declined.outcome, eval::Outcome::unsupported);
  EXPECT_EQ(declined.completed_boundaries, 0);
  ASSERT_EQ(declined.trace.size(), 1);
  EXPECT_TRUE(declined.trace[0].events.empty());
  EXPECT_EQ(initial.cells[0].value.word(0), original_state.cells[0].value.word(0));
}

TEST(RecoveredPath, CleanupProducesValidatedOmissionAndRespectsAddressAndLimit) {
  auto budget = Plenty();
  auto proposed =
      recovery::OmitOverwrittenStores(ir::RecoveredPath(AdjacentStackStores(), {}, 0), 31, budget);
  ASSERT_TRUE(proposed.path);
  ASSERT_EQ(proposed.path->omissions().size(), 1);
  EXPECT_EQ(proposed.path->omissions()[0].store, 4);
  EXPECT_EQ(proposed.path->omissions()[0].overwriter, 7);
  EXPECT_EQ(proposed.path->omissions()[0].offset_bytes, 56);
  EXPECT_EQ(ir::ValidateRecoveredPath(*proposed.path, budget), ir::BlockDecline::none);

  auto different = recovery::OmitOverwrittenStores(
      ir::RecoveredPath(AdjacentStackStores(64), {}, 0), 31, budget);
  ASSERT_TRUE(different.path);
  EXPECT_TRUE(different.path->omissions().empty());
  EXPECT_EQ(different.path->revision(), 0);

  auto limited = recovery::OmitOverwrittenStores(ir::RecoveredPath(AdjacentStackStores(), {}, 0),
                                                 31, budget, {}, 0);
  EXPECT_FALSE(limited.path);
  EXPECT_EQ(limited.reason, recovery::StoreCleanupDecline::resource_limit);
}

TEST(RecoveredPath, CleanupComposesAfterControlRecoveryRevision) {
  const auto pair = AdjacentStackStores();
  std::vector<ir::Group> groups(pair.sources().begin(), pair.sources().end());
  groups.emplace_back(0x85924c, std::vector<std::uint8_t>{1, 2, 3, 4},
                      std::vector<ir::Node>{{ir::Op::constant, 1, {}, 1},
                                            {ir::Op::image_address, 64, {}, 0x900000},
                                            {ir::Op::image_address, 64, {}, 0x900004}},
                      std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                      ir::Transfer{ir::TransferKind::conditional, 1, 0, 2, {}});
  auto budget = Plenty();
  auto normalized = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(normalized.path);
  auto rewrite = Rewrite(*normalized.path, 2);
  auto cleaned = recovery::OmitOverwrittenStores(
      ir::RecoveredPath(std::move(*normalized.path), {rewrite}, 1), 31, budget);
  ASSERT_TRUE(cleaned.path);
  ASSERT_EQ(cleaned.path->omissions().size(), 1);
  EXPECT_EQ(cleaned.path->revision(), 2);
  EXPECT_EQ(cleaned.path->rewrites()[0].to_revision, 1);
  EXPECT_EQ(cleaned.path->omissions()[0].from_revision, 1);
  EXPECT_EQ(cleaned.path->omissions()[0].to_revision, 2);
  EXPECT_EQ(ir::ValidateRecoveredPath(*cleaned.path, budget), ir::BlockDecline::none);
}

TEST(RecoveredPath, StoreOmissionRefusesEarlierStoreInSameBoundary) {
  std::vector<ir::Group> groups;
  groups.emplace_back(0x859244, std::vector<std::uint8_t>{0xe0, 0x1f, 0x00, 0xf9},
                      std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 31},
                                            {ir::Op::constant, 64, {}, 56},
                                            {ir::Op::add, 64, {0, 1}},
                                            {ir::Op::read, 64, {}, 0, 0},
                                            {ir::Op::store, 64, {2, 3}},
                                            {ir::Op::store, 64, {2, 3}}},
                      std::vector<ir::Write>{}, ir::MemoryModel::atomic_scalar_reference);
  groups.emplace_back(0x859248, std::vector<std::uint8_t>{0xe0, 0x1f, 0x00, 0xf9},
                      std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 31},
                                            {ir::Op::constant, 64, {}, 56},
                                            {ir::Op::add, 64, {0, 1}},
                                            {ir::Op::read, 64, {}, 0, 0},
                                            {ir::Op::store, 64, {2, 3}}},
                      std::vector<ir::Write>{}, ir::MemoryModel::atomic_scalar_reference);
  auto budget = Plenty();
  auto normalized = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(normalized.path);
  std::vector<ir::ValueId> stores;
  for (ir::ValueId id = 0; id < normalized.path->nodes().size(); ++id)
    if (normalized.path->nodes()[id].op == ir::Op::store) stores.push_back(id);
  ASSERT_EQ(stores.size(), 3);
  EXPECT_EQ(ir::ValidateRecoveredPath(ir::RecoveredPath(*normalized.path, {}, 1, {},
                                                        {{stores[1], stores[2], 31, 56, 0, 1}}),
                                      budget),
            ir::BlockDecline::invalid_ir);
  auto proposed = recovery::OmitOverwrittenStores(
      ir::RecoveredPath(std::move(*normalized.path), {}, 0), 31, budget);
  ASSERT_TRUE(proposed.path);
  EXPECT_TRUE(proposed.path->omissions().empty());
}

TEST(RecoveredPath, StoreOmissionRefusesOverwriterBoundaryThatCanRollBack) {
  const auto pair = AdjacentStackStores();
  std::vector<ir::Group> groups{pair.sources()[0]};
  groups.emplace_back(0x859248, std::vector<std::uint8_t>{1, 2, 3, 4},
                      std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 31},
                                            {ir::Op::constant, 64, {}, 56},
                                            {ir::Op::add, 64, {0, 1}},
                                            {ir::Op::read, 64, {}, 0, 1},
                                            {ir::Op::store, 64, {2, 3}},
                                            {ir::Op::store, 64, {2, 3}}},
                      std::vector<ir::Write>{}, ir::MemoryModel::atomic_scalar_reference);
  auto budget = Plenty();
  auto normalized = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(normalized.path);
  std::vector<ir::ValueId> stores;
  for (ir::ValueId id = 0; id < normalized.path->nodes().size(); ++id)
    if (normalized.path->nodes()[id].op == ir::Op::store) stores.push_back(id);
  ASSERT_EQ(stores.size(), 3);
  EXPECT_EQ(ir::ValidateRecoveredPath(ir::RecoveredPath(*normalized.path, {}, 1, {},
                                                        {{stores[0], stores[1], 31, 56, 0, 1}}),
                                      budget),
            ir::BlockDecline::invalid_ir);
  auto proposed = recovery::OmitOverwrittenStores(
      ir::RecoveredPath(std::move(*normalized.path), {}, 0), 31, budget);
  ASSERT_TRUE(proposed.path);
  EXPECT_TRUE(proposed.path->omissions().empty());
}

TEST(RecoveredPath, StoreOmissionRejectsDifferentAddressAndForgedWitness) {
  auto budget = Plenty();
  const auto different = AdjacentStackStores(64);
  EXPECT_EQ(ir::ValidateRecoveredPath(
                ir::RecoveredPath(different, {}, 1, {}, {{4, 7, 31, 56, 0, 1}}), budget),
            ir::BlockDecline::invalid_ir);
  const auto same = AdjacentStackStores();
  for (const auto forged :
       {ir::StoreOmission{4, 7, 0, 56, 0, 1}, ir::StoreOmission{4, 7, 31, 64, 0, 1},
        ir::StoreOmission{7, 4, 31, 56, 0, 1}, ir::StoreOmission{4, 7, 31, 56, 1, 2}}) {
    EXPECT_EQ(ir::ValidateRecoveredPath(ir::RecoveredPath(same, {}, 1, {}, {forged}), budget),
              ir::BlockDecline::invalid_ir);
  }
}

TEST(RecoveredPath, OmittedStoreChecksAddressAfterPriorStackPointerWrite) {
  std::vector<ir::Group> groups{ir::Group(
      0x859240, {0xff, 0x03, 0x04, 0x91},
      {{ir::Op::read, 64, {}, 0, 31}, {ir::Op::constant, 64, {}, 256}, {ir::Op::add, 64, {0, 1}}},
      {{31, 2}})};
  for (const auto address : {UINT64_C(0x859244), UINT64_C(0x859248)}) {
    groups.emplace_back(address, std::vector<std::uint8_t>{0xe0, 0x1f, 0x00, 0xf9},
                        std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 31},
                                              {ir::Op::constant, 64, {}, 56},
                                              {ir::Op::add, 64, {0, 1}},
                                              {ir::Op::read, 64, {}, 0, 0},
                                              {ir::Op::store, 64, {2, 3}}},
                        std::vector<ir::Write>{}, ir::MemoryModel::atomic_scalar_reference);
  }

  auto budget = Plenty();
  auto normalized = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(normalized.path);
  std::vector<ir::ValueId> stores;
  for (ir::ValueId id = 0; id < normalized.path->nodes().size(); ++id)
    if (normalized.path->nodes()[id].op == ir::Op::store) stores.push_back(id);
  ASSERT_EQ(stores.size(), 2);
  auto proposed =
      recovery::OmitOverwrittenStores(ir::RecoveredPath(*normalized.path, {}, 0), 31, budget);
  ASSERT_TRUE(proposed.path);
  ASSERT_EQ(proposed.path->omissions().size(), 1);
  EXPECT_EQ(proposed.path->omissions()[0].store, stores[0]);
  EXPECT_EQ(proposed.path->omissions()[0].overwriter, stores[1]);
  EXPECT_EQ(proposed.path->omissions()[0].offset_bytes, 312);
  auto recovered = std::move(*proposed.path);
  ASSERT_EQ(ir::ValidateRecoveredPath(recovered, budget), ir::BlockDecline::none);
  const std::array<std::uint8_t, 256> bytes{};
  const eval::RegionInput region[] = {{0x1000, bytes}};
  auto memory = std::move(*eval::Memory::Create(region, budget).memory);
  auto state = StackState();
  const auto result = eval::ExecuteRecoveredPath(recovered, state, memory, budget, {}, {0});
  EXPECT_EQ(result.outcome, eval::Outcome::unsupported);
  EXPECT_EQ(result.completed_boundaries, 1);
  EXPECT_EQ(state.cells[1].value.word(0), 0x1100);
}

TEST(RecoveredPath, FoldsBothLowBitValuesAndPreservesFlagsEffectsAndOriginals) {
  for (const auto literal : {UINT64_C(0), UINT64_C(1), UINT64_C(2), UINT64_C(3), UINT64_MAX}) {
    auto recovered = Recovered(Basis(literal));
    auto budget = Plenty();
    EXPECT_EQ(ir::ValidateRecoveredPath(recovered, budget), ir::BlockDecline::none);
    EXPECT_EQ(ir::ValidatePath(recovered.basis(), budget), ir::BlockDecline::none);
    EXPECT_EQ(recovered.basis().sources()[0].transfer()->kind, ir::TransferKind::conditional);
    EXPECT_EQ(recovered.basis().boundaries()[0].transfer->kind, ir::TransferKind::conditional);
    ASSERT_TRUE(recovered.effective_transfer(0));
    EXPECT_EQ(recovered.effective_transfer(0)->kind, ir::TransferKind::jump);
    auto original_state = State(), recovered_state = State();
    auto original_memory = Memory(), recovered_memory = Memory();
    const auto original =
        eval::ExecutePath(recovered.basis(), original_state, original_memory, budget, {}, {0x1000});
    const auto actual = eval::ExecuteRecoveredPath(recovered, recovered_state, recovered_memory,
                                                   budget, {}, {0x1000});
    EXPECT_EQ(actual.outcome, eval::Outcome::completed);
    EXPECT_EQ(actual.stop, original.stop);
    EXPECT_EQ(actual.runtime_next, original.runtime_next);
    EXPECT_EQ(actual.completed_boundaries, original.completed_boundaries);
    EXPECT_EQ(recovered_state.cells[0].value.word(0), original_state.cells[0].value.word(0));
    EXPECT_EQ(recovered_state.cells[1].value.word(0), literal & 1);
    ASSERT_EQ(actual.trace[0].events.size(), 2);
    EXPECT_EQ(actual.trace[0].events[0].bytes, original.trace[0].events[0].bytes);
    EXPECT_EQ(actual.trace[0].events[1].bytes, original.trace[0].events[1].bytes);
    ASSERT_TRUE(actual.trace[0].transfer);
    EXPECT_EQ(actual.trace[0].transfer->kind, ir::TransferKind::jump);
    EXPECT_FALSE(actual.trace[0].transfer->condition);
    EXPECT_EQ(actual.trace[0].transfer->target, original.trace[0].transfer->target);
    EXPECT_EQ(actual.stop, (literal & 1) ? eval::PathStop::completed : eval::PathStop::diverged);
  }
}

TEST(RecoveredPath, EqualDestinationAndTerminalControlDoNotInventContinuation) {
  auto budget = Plenty();
  auto equal = Recovered(Basis(0, false, true));
  auto state = State();
  auto memory = Memory();
  EXPECT_EQ(eval::ExecuteRecoveredPath(equal, state, memory, budget, {}, {0}).stop,
            eval::PathStop::completed);
  auto terminal = Recovered(Basis(0, false, false, true));
  state = State();
  const auto result = eval::ExecuteRecoveredPath(terminal, state, memory, budget, {}, {0x1000});
  EXPECT_EQ(result.stop, eval::PathStop::completed);
  EXPECT_EQ(result.runtime_next, 0x1300);
  EXPECT_EQ(result.completed_boundaries, 1);
  EXPECT_FALSE(terminal.effective_transfer(100));
}

TEST(RecoveredPath, FaultRetainsOrderedPrefixWithoutPublishingRewrittenTransfer) {
  auto recovered = Recovered(Basis(1, true));
  auto budget = Plenty();
  auto state = State();
  auto memory = Memory();
  const auto result = eval::ExecuteRecoveredPath(recovered, state, memory, budget, {}, {0});
  EXPECT_EQ(result.outcome, eval::Outcome::fault);
  EXPECT_EQ(result.stop, eval::PathStop::fault);
  EXPECT_EQ(result.completed_boundaries, 0);
  EXPECT_FALSE(result.runtime_next);
  ASSERT_EQ(result.trace.size(), 1);
  EXPECT_FALSE(result.trace[0].transfer);
  ASSERT_TRUE(result.trace[0].fault);
  EXPECT_EQ(result.trace[0].fault->operation, 7);
  ASSERT_EQ(result.trace[0].events.size(), 2);
  EXPECT_TRUE(result.trace[0].events[0].completed);
  EXPECT_FALSE(result.trace[0].events[1].completed);
  EXPECT_EQ(state.cells[0].value.word(0), 7);
  EXPECT_EQ(state.cells[1].value.word(0), 0);
  const auto bytes = memory.Read(0x4000, 8, budget);
  ASSERT_EQ(bytes.status, eval::MemoryStatus::ok);
  EXPECT_EQ(bytes.bytes[0], 42);
}

TEST(RecoveredPath, EmptyWrapperRetainsCallsAndStrictPathContract) {
  auto budget = Plenty();
  const std::vector<ir::Group> groups{ir::Group(
      0x100, {1, 2, 3, 4},
      {{ir::Op::image_address, 64, {}, 0x300}, {ir::Op::image_address, 64, {}, 0x104}}, {},
      ir::MemoryModel::unspecified, ir::Transfer{ir::TransferKind::call, 0, {}, {}, 1})};
  auto normalized = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(normalized.path);
  ir::RecoveredPath path(std::move(*normalized.path), {}, 0);
  EXPECT_EQ(ir::ValidateRecoveredPath(path, budget), ir::BlockDecline::none);
  auto state = State();
  auto memory = Memory();
  const auto result = eval::ExecuteRecoveredPath(path, state, memory, budget, {}, {0});
  ASSERT_TRUE(result.trace[0].transfer);
  EXPECT_EQ(result.trace[0].transfer->kind, ir::TransferKind::call);
  EXPECT_EQ(result.trace[0].transfer->continuation, 0x104);
  EXPECT_EQ(result.runtime_next, 0x300);
  auto basis = Basis();
  auto boundaries = std::vector<ir::Boundary>(basis.boundaries().begin(), basis.boundaries().end());
  boundaries[0].transfer = Rewrite(basis).replacement;
  ir::Path invalid({basis.sources().begin(), basis.sources().end()},
                   {basis.nodes().begin(), basis.nodes().end()},
                   {basis.origins().begin(), basis.origins().end()}, std::move(boundaries));
  EXPECT_EQ(ir::ValidatePath(invalid, budget), ir::BlockDecline::invalid_ir);
}

TEST(RecoveredPath, ForgedOrStaleWitnessesDeclineBeforeAnyEffects) {
  const auto basis = Basis();
  for (unsigned mutation = 0; mutation < 10; ++mutation) {
    auto rewrite = Rewrite(basis);
    auto revision = rewrite.to_revision;
    std::vector<ir::ConditionalRewrite> rewrites;
    switch (mutation) {
      case 0:
        rewrite.replacement.target = *rewrite.original.alternative;
        break;
      case 1:
        rewrite.condition_value = false;
        break;
      case 2:
        rewrite.condition = 1;
        break;
      case 3:
        rewrite.original.target = *rewrite.original.alternative;
        break;
      case 4:
        rewrite.boundary = UINT32_MAX;
        break;
      case 5:
        rewrite.from_revision = 9;
        break;
      case 6:
        rewrite.to_revision = 9;
        break;
      case 7:
        revision = 9;
        break;
      case 8:
        rewrite.replacement.condition = 0;
        break;
      case 9:
        rewrites.push_back(rewrite);
        break;
    }

    rewrites.push_back(rewrite);
    ir::RecoveredPath forged(basis, std::move(rewrites), revision);
    auto budget = Plenty();
    EXPECT_EQ(ir::ValidateRecoveredPath(forged, budget), ir::BlockDecline::invalid_ir);
    auto state = State();
    auto memory = Memory();
    const auto result = eval::ExecuteRecoveredPath(forged, state, memory, budget, {}, {0});
    EXPECT_EQ(result.outcome, eval::Outcome::invalid_group);
    EXPECT_TRUE(result.trace.empty());
    EXPECT_EQ(state.cells[0].value.word(0), 7);
    const auto bytes = memory.Read(0x4000, 8, budget);
    ASSERT_EQ(bytes.status, eval::MemoryStatus::ok);
    EXPECT_EQ(bytes.bytes[0], 0);
  }
}

TEST(RecoveredPath, RevisionOverflowRequiresEmptyUnchangedWrapper) {
  auto basis = Basis();
  ir::Path max({basis.sources().begin(), basis.sources().end()},
               {basis.nodes().begin(), basis.nodes().end()},
               {basis.origins().begin(), basis.origins().end()},
               {basis.boundaries().begin(), basis.boundaries().end()}, UINT64_MAX);
  auto budget = Plenty();
  EXPECT_EQ(ir::ValidateRecoveredPath(ir::RecoveredPath(max, {}, UINT64_MAX), budget),
            ir::BlockDecline::none);
  EXPECT_EQ(ir::ValidateRecoveredPath(ir::RecoveredPath(max, {}, 0), budget),
            ir::BlockDecline::invalid_ir);
  const auto rewrite = Rewrite(max);
  EXPECT_EQ(ir::ValidateRecoveredPath(ir::RecoveredPath(std::move(max), {rewrite}, 0), budget),
            ir::BlockDecline::invalid_ir);
}

TEST(RecoveredPath, LateInvalidOrUnsortedWitnessRejectsBeforeEarlierStore) {
  const auto first = Basis();
  std::vector<ir::Group> groups{
      first.sources()[0], ir::Group(0x200, {5, 6, 7, 8},
                                    {{ir::Op::constant, 1, {}, 0},
                                     {ir::Op::image_address, 64, {}, 0x300},
                                     {ir::Op::image_address, 64, {}, 0x400}},
                                    {}, ir::MemoryModel::unspecified,
                                    ir::Transfer{ir::TransferKind::conditional, 1, 0, 2, {}})};
  auto budget = Plenty();
  auto normalized = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(normalized.path);
  const auto& basis = *normalized.path;
  for (const bool unsorted : {false, true}) {
    std::vector<ir::ConditionalRewrite> rewrites{Rewrite(basis), Rewrite(basis, 1)};
    if (unsorted)
      std::swap(rewrites[0], rewrites[1]);
    else
      rewrites[1].condition_value = true;
    ir::RecoveredPath invalid(basis, std::move(rewrites), 1);
    auto state = State();
    auto memory = Memory();
    const auto result = eval::ExecuteRecoveredPath(invalid, state, memory, budget, {}, {0});
    EXPECT_EQ(result.outcome, eval::Outcome::invalid_group);
    EXPECT_TRUE(result.trace.empty());
    EXPECT_EQ(state.cells[0].value.word(0), 7);
    const auto bytes = memory.Read(0x4000, 8, budget);
    ASSERT_EQ(bytes.status, eval::MemoryStatus::ok);
    EXPECT_EQ(bytes.bytes[0], 0);
  }
}

TEST(RecoveredPath, AllResourceCutsRetainInstructionTransactionAndExactBudgetSucceeds) {
  auto path = Recovered(Basis(1, false, false, true));
  auto budget = Plenty();
  auto state = State();
  auto memory = Memory();
  ASSERT_EQ(eval::ExecuteRecoveredPath(path, state, memory, budget, {}, {0}).outcome,
            eval::Outcome::completed);
  const auto required = budget.used();
  const auto check = [&](Resources cap) {
    Budget limited(cap);
    auto initial = State();
    auto data = Memory();
    const auto result = eval::ExecuteRecoveredPath(path, initial, data, limited, {}, {0});
    EXPECT_EQ(result.outcome, eval::Outcome::resource_limit);
    EXPECT_EQ(result.completed_boundaries, 0);
    EXPECT_FALSE(result.runtime_next);
    EXPECT_EQ(initial.cells[0].value.word(0), 7);
    auto inspect = Plenty();
    const auto bytes = data.Read(0x4000, 8, inspect);
    ASSERT_EQ(bytes.status, eval::MemoryStatus::ok);
    EXPECT_EQ(bytes.bytes[0], 0);
  };

  for (std::uint64_t work = 0; work < required.work; ++work) check({work, UINT64_MAX});
  for (std::uint64_t bytes = 0; bytes < required.bytes; ++bytes) check({UINT64_MAX, bytes});
  Budget exact(required);
  state = State();
  memory = Memory();
  EXPECT_EQ(eval::ExecuteRecoveredPath(path, state, memory, exact, {}, {0}).outcome,
            eval::Outcome::completed);
  auto no_context = Plenty();
  state = State();
  const auto declined = eval::ExecuteRecoveredPath(path, state, memory, no_context);
  EXPECT_EQ(declined.outcome, eval::Outcome::unsupported);
  EXPECT_TRUE(declined.trace.empty());
  EXPECT_EQ(state.cells[0].value.word(0), 7);
}

}  // namespace
}  // namespace nyx
