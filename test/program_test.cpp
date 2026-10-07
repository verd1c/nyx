#include "nyx/eval/program.hpp"

#include <algorithm>
#include <limits>

#include <gtest/gtest.h>

namespace nyx::eval {
namespace {

Budget Plenty() { return Budget({10000000, 10000000}); }

State Initial() {
  auto budget = Plenty();
  State state;
  for (const auto& [id, value] : {std::pair{100U, 7U}, {200U, 11U}, {999U, 0xbadU}}) {
    auto bits = BitVector::from_u64(64, value, 64, budget);
    state.cells.push_back({id, std::move(*bits)});
  }

  return state;
}

Memory DataMemory() {
  auto budget = Plenty();
  const std::array<std::uint8_t, 16> bytes{};
  const RegionInput regions[] = {{0x4000, bytes}};
  return std::move(*Memory::Create(regions, budget).memory);
}

ir::Group Call() {
  return ir::Group(
      0x10, {1, 2}, {{ir::Op::image_address, 64, {}, 0x20}, {ir::Op::image_address, 64, {}, 0x12}},
      {{999, 1}}, ir::MemoryModel::unspecified, ir::Transfer{ir::TransferKind::call, 0, {}, {}, 1});
}

std::vector<ir::Group> ReturningProgram() {
  return {ir::Group(0x23, {6}, {{ir::Op::read, 64, {}, 0, 999}}, {}, ir::MemoryModel::unspecified,
                    ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}}),
          Call(),
          ir::Group(0x12, {7, 8, 9, 10, 11},
                    {{ir::Op::constant, 64, {}, 0x4000}, {ir::Op::load, 64, {0}}}, {{200, 1}},
                    ir::MemoryModel::atomic_scalar_reference),
          ir::Group(0x20, {3, 4, 5},
                    {{ir::Op::constant, 64, {}, 0x4000},
                     {ir::Op::constant, 64, {}, 42},
                     {ir::Op::store, 64, {0, 1}}},
                    {{100, 1}}, ir::MemoryModel::atomic_scalar_reference)};
}

TEST(Program, RunsActualReturningCalleeAndObservesItsMemoryEffects) {
  const auto groups = ReturningProgram();
  auto state = Initial();
  auto memory = DataMemory();
  auto budget = Plenty();
  const std::uint64_t exits[] = {0x1017};
  const auto result = nyx::eval::Run(groups, state, memory, 0x1010, exits, budget, {}, {0x1000});
  ASSERT_EQ(result.status, ProgramStatus::exit);
  EXPECT_EQ(result.runtime_pc, 0x1017);
  EXPECT_EQ(result.committed_steps, 4);
  ASSERT_EQ(result.trace.size(), 4);
  EXPECT_EQ(result.trace[0].runtime_pc, 0x1010);
  EXPECT_EQ(result.trace[1].runtime_pc, 0x1020);
  EXPECT_EQ(result.trace[2].runtime_pc, 0x1023);
  EXPECT_EQ(result.trace[3].runtime_pc, 0x1012);
  ASSERT_TRUE(result.trace[0].execution.transfer);
  EXPECT_EQ(result.trace[0].execution.transfer->kind, ir::TransferKind::call);
  ASSERT_TRUE(result.trace[2].execution.transfer);
  EXPECT_EQ(result.trace[2].execution.transfer->kind, ir::TransferKind::return_);
  ASSERT_EQ(result.trace[1].execution.events.size(), 1);
  EXPECT_TRUE(result.trace[1].execution.events[0].write);
  ASSERT_EQ(result.trace[3].execution.events.size(), 1);
  EXPECT_FALSE(result.trace[3].execution.events[0].write);
  EXPECT_EQ(state.cells[0].value.word(0), 42);
  EXPECT_EQ(state.cells[1].value.word(0), 42);
  EXPECT_EQ(state.cells[2].value.word(0), 0x1012);
  EXPECT_EQ(memory.Regions()[0].bytes[0], 42);
}

TEST(Program, MissingCalleeSuspendsAfterCallWithoutInventingReturn) {
  const std::vector<ir::Group> groups{Call()};
  auto state = Initial();
  auto memory = DataMemory();
  auto budget = Plenty();
  const auto result = nyx::eval::Run(groups, state, memory, 0x1010, {}, budget, {}, {0x1000});
  EXPECT_EQ(result.status, ProgramStatus::unresolved);
  EXPECT_EQ(result.runtime_pc, 0x1020);
  EXPECT_EQ(result.committed_steps, 1);
  ASSERT_EQ(result.trace.size(), 1);
  EXPECT_EQ(state.cells[2].value.word(0), 0x1012);
  EXPECT_EQ(state.cells[0].value.word(0), 7);
  EXPECT_EQ(memory.Regions()[0].bytes[0], 0);
}

TEST(Program, DeclaredExitStopsBeforeDispatchAndInteriorTargetsStayUnresolved) {
  const std::vector<ir::Group> groups{Call()};
  for (const auto entry : {0x3000ULL, 0x1011ULL}) {
    auto state = Initial();
    auto memory = DataMemory();
    auto budget = Plenty();
    const std::uint64_t exits[] = {0x3000};
    const auto result = nyx::eval::Run(groups, state, memory, entry, exits, budget, {}, {0x1000});
    EXPECT_EQ(result.status, entry == 0x3000 ? ProgramStatus::exit : ProgramStatus::unresolved);
    EXPECT_EQ(result.committed_steps, 0);
    EXPECT_TRUE(result.trace.empty());
    EXPECT_EQ(state.cells[2].value.word(0), 0xbad);
  }
}

TEST(Program, LoopsExhaustDeterministicStepsAndKeepRecordedPrefix) {
  const std::vector<ir::Group> groups{
      ir::Group(0x10, {1}, {{ir::Op::image_address, 64, {}, 0x10}}, {},
                ir::MemoryModel::unspecified, ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}})};
  auto state = Initial();
  auto memory = DataMemory();
  auto budget = Plenty();
  const auto result =
      nyx::eval::Run(groups, state, memory, 0x1010, {}, budget, {4096, 3}, {0x1000});
  EXPECT_EQ(result.status, ProgramStatus::step_limit);
  EXPECT_EQ(result.runtime_pc, 0x1010);
  EXPECT_EQ(result.committed_steps, 3);
  EXPECT_EQ(result.trace.size(), 3);
}

TEST(Program, LaterFaultCommitsOnlyItsActualOrderedPrefixAndNoReturn) {
  const std::vector<ir::Group> groups{
      Call(), ir::Group(0x20, {3},
                        {{ir::Op::constant, 64, {}, 0x4000},
                         {ir::Op::constant, 64, {}, 42},
                         {ir::Op::store, 64, {0, 1}},
                         {ir::Op::constant, 64, {}, 0x5000},
                         {ir::Op::load, 64, {3}},
                         {ir::Op::read, 64, {}, 0, 999}},
                        {{100, 1}}, ir::MemoryModel::atomic_scalar_reference,
                        ir::Transfer{ir::TransferKind::return_, 5, {}, {}, {}})};
  auto state = Initial();
  auto memory = DataMemory();
  auto budget = Plenty();
  const auto result = nyx::eval::Run(groups, state, memory, 0x1010, {}, budget, {}, {0x1000});
  EXPECT_EQ(result.status, ProgramStatus::fault);
  EXPECT_EQ(result.runtime_pc, 0x1020);
  EXPECT_EQ(result.committed_steps, 2);
  ASSERT_EQ(result.trace.size(), 2);
  ASSERT_TRUE(result.trace.back().execution.fault);
  EXPECT_EQ(result.trace.back().execution.fault->address, 0x5000);
  EXPECT_FALSE(result.trace.back().execution.transfer);
  EXPECT_EQ(memory.Regions()[0].bytes[0], 42);
  EXPECT_EQ(state.cells[0].value.word(0), 7);
  EXPECT_EQ(state.cells[2].value.word(0), 0x1012);
}

TEST(Program, InvalidArenaShapesAndExitAmbiguitiesHaveNoEffects) {
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  const std::vector<std::vector<ir::Group>> invalid{
      {ir::Group(0, {}, {}, {})},
      {ir::Group(0x10, {1, 2}, {}, {}), ir::Group(0x11, {1}, {}, {})},
      {ir::Group(maximum, {1, 2}, {}, {})},
      {ir::Group(maximum - 1, {1, 2}, {}, {})}};
  for (const auto& groups : invalid) {
    auto state = Initial();
    auto memory = DataMemory();
    auto budget = Plenty();
    const auto result = nyx::eval::Run(groups, state, memory, 0, {}, budget, {}, {1});
    EXPECT_EQ(result.status, ProgramStatus::invalid_program);
    EXPECT_EQ(result.committed_steps, 0);
    EXPECT_TRUE(result.trace.empty());
  }

  const std::vector<ir::Group> groups{Call()};
  for (const auto& exits : {std::vector<std::uint64_t>{0x1010}, {0x1011}, {0x3000, 0x3000}}) {
    auto state = Initial();
    auto memory = DataMemory();
    auto budget = Plenty();
    EXPECT_EQ(nyx::eval::Run(groups, state, memory, 0x1010, exits, budget, {}, {0x1000}).status,
              ProgramStatus::invalid_program);
    EXPECT_EQ(state.cells[2].value.word(0), 0xbad);
  }
}

TEST(Program, CodeDataAliasesMustMatchAndRemainReadonly) {
  const std::vector<ir::Group> groups{ir::Group(0x10, {1, 2}, {}, {})};
  for (bool writable : {false, true}) {
    for (bool matching : {false, true}) {
      auto state = Initial();
      auto budget = Plenty();
      const std::array<std::uint8_t, 2> data{1, static_cast<std::uint8_t>(matching ? 2 : 9)};
      const RegionInput region[] = {{0x1010, data, true, writable}};
      auto memory = Memory::Create(region, budget);
      ASSERT_TRUE(memory.memory);
      const std::uint64_t exits[] = {0x1012};
      const auto result =
          nyx::eval::Run(groups, state, *memory.memory, 0x1010, exits, budget, {}, {0x1000});
      EXPECT_EQ(result.status,
                !writable && matching ? ProgramStatus::exit : ProgramStatus::invalid_program);
      EXPECT_EQ(result.committed_steps, !writable && matching ? 1 : 0);
    }
  }
}

TEST(Program, PlacementAndFallthroughUseModulo64WithoutWrappingByteRanges) {
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  const std::vector<ir::Group> groups{ir::Group(0x10, {1}, {}, {}), ir::Group(0x11, {2}, {}, {})};
  auto state = Initial();
  auto memory = DataMemory();
  auto budget = Plenty();
  const std::uint64_t exits[] = {1};
  const auto result =
      nyx::eval::Run(groups, state, memory, maximum, exits, budget, {}, {maximum - 0x10});
  EXPECT_EQ(result.status, ProgramStatus::exit);
  EXPECT_EQ(result.committed_steps, 2);
  ASSERT_EQ(result.trace.size(), 2);
  EXPECT_EQ(result.trace[0].runtime_pc, maximum);
  EXPECT_EQ(result.trace[1].runtime_pc, 0);
  EXPECT_EQ(result.runtime_pc, 1);
}

TEST(Program, MissingPlacementAndLimitsDeclineWithoutDispatch) {
  const auto groups = ReturningProgram();
  auto state = Initial();
  auto memory = DataMemory();
  auto budget = Plenty();
  EXPECT_EQ(nyx::eval::Run(groups, state, memory, 0x1010, {}, budget).status,
            ProgramStatus::unsupported);
  EXPECT_EQ(nyx::eval::Run(groups, state, memory, 0x1010, {}, budget, {1, 100}, {0x1000}).status,
            ProgramStatus::resource_limit);
  const auto zero = nyx::eval::Run(groups, state, memory, 0x1010, {}, budget, {4096, 0}, {0x1000});
  EXPECT_EQ(zero.status, ProgramStatus::step_limit);
  EXPECT_TRUE(zero.trace.empty());
  EXPECT_EQ(state.cells[2].value.word(0), 0xbad);
}

TEST(Program, LaterMalformedGroupKeepsEarlierCallAndInvalidStateIsExplicit) {
  const std::vector<ir::Group> groups{Call(),
                                      ir::Group(0x20, {3}, {{ir::Op::bit_not, 64, {0}}}, {})};
  auto state = Initial();
  auto memory = DataMemory();
  auto budget = Plenty();
  const auto result = nyx::eval::Run(groups, state, memory, 0x1010, {}, budget, {}, {0x1000});
  EXPECT_EQ(result.status, ProgramStatus::invalid_program);
  EXPECT_EQ(result.committed_steps, 1);
  ASSERT_EQ(result.trace.size(), 2);
  EXPECT_EQ(result.trace.back().execution.outcome, Outcome::invalid_group);
  EXPECT_EQ(state.cells[2].value.word(0), 0x1012);
  State missing;
  auto again = Plenty();
  const auto invalid = nyx::eval::Run(groups, missing, memory, 0x1010, {}, again, {}, {0x1000});
  EXPECT_EQ(invalid.status, ProgramStatus::invalid_state);
  EXPECT_EQ(invalid.committed_steps, 0);
}

TEST(Program, LaterInvalidStateKeepsEarlierCompletedWrite) {
  const std::vector<ir::Group> groups{
      ir::Group(0x10, {1}, {{ir::Op::constant, 64, {}, 42}}, {{100, 0}}),
      ir::Group(0x11, {2}, {{ir::Op::read, 64, {}, 0, 12345}}, {{200, 0}})};
  auto state = Initial();
  auto memory = DataMemory();
  auto budget = Plenty();
  const auto result = nyx::eval::Run(groups, state, memory, 0x1010, {}, budget, {}, {0x1000});
  EXPECT_EQ(result.status, ProgramStatus::invalid_state);
  EXPECT_EQ(result.runtime_pc, 0x1011);
  EXPECT_EQ(result.committed_steps, 1);
  ASSERT_EQ(result.trace.size(), 2);
  EXPECT_EQ(result.trace[0].execution.outcome, Outcome::completed);
  EXPECT_EQ(result.trace[1].execution.outcome, Outcome::invalid_state);
  EXPECT_TRUE(result.trace[1].execution.events.empty());
  EXPECT_FALSE(result.trace[1].execution.transfer);
  EXPECT_EQ(state.cells[0].value.word(0), 42);
  EXPECT_EQ(state.cells[1].value.word(0), 11);
  EXPECT_EQ(memory.Regions()[0].bytes[0], 0);
}

TEST(Program, LaterUnsupportedMemoryModelKeepsEarlierCompletedWrite) {
  const std::vector<ir::Group> groups{
      ir::Group(0x10, {1}, {{ir::Op::constant, 64, {}, 42}}, {{100, 0}}),
      ir::Group(0x11, {2}, {{ir::Op::constant, 64, {}, 0x4000}, {ir::Op::load, 64, {0}}},
                {{200, 1}})};
  auto state = Initial();
  auto memory = DataMemory();
  auto budget = Plenty();
  const auto result = nyx::eval::Run(groups, state, memory, 0x1010, {}, budget, {}, {0x1000});
  EXPECT_EQ(result.status, ProgramStatus::unsupported);
  EXPECT_EQ(result.runtime_pc, 0x1011);
  EXPECT_EQ(result.committed_steps, 1);
  ASSERT_EQ(result.trace.size(), 2);
  EXPECT_EQ(result.trace[0].execution.outcome, Outcome::completed);
  EXPECT_EQ(result.trace[1].execution.outcome, Outcome::unsupported);
  EXPECT_TRUE(result.trace[1].execution.events.empty());
  EXPECT_FALSE(result.trace[1].execution.transfer);
  EXPECT_EQ(state.cells[0].value.word(0), 42);
  EXPECT_EQ(state.cells[1].value.word(0), 11);
  EXPECT_EQ(memory.Regions()[0].bytes[0], 0);
}

TEST(Program, EveryResourceCutPreservesExactlyTheTracedCommittedPrefix) {
  const auto groups = ReturningProgram();
  const std::uint64_t exits[] = {0x1017};
  auto full = Plenty();
  auto complete_state = Initial();
  auto complete_memory = DataMemory();
  ASSERT_EQ(
      nyx::eval::Run(groups, complete_state, complete_memory, 0x1010, exits, full, {}, {0x1000})
          .status,
      ProgramStatus::exit);
  for (bool bytes : {false, true}) {
    const auto bound = bytes ? full.used().bytes : full.used().work;
    for (std::uint64_t cut = 0; cut < bound; ++cut) {
      auto state = Initial();
      auto memory = DataMemory();
      Budget budget({bytes ? full.used().work : cut, bytes ? cut : full.used().bytes});
      const auto result =
          nyx::eval::Run(groups, state, memory, 0x1010, exits, budget, {}, {0x1000});
      ASSERT_EQ(result.status, ProgramStatus::resource_limit) << cut;
      std::uint64_t complete = 0;
      for (const auto& step : result.trace) {
        if (step.execution.outcome == Outcome::completed)
          ++complete;
        else {
          EXPECT_EQ(step.execution.outcome, Outcome::resource_limit);
          EXPECT_TRUE(step.execution.events.empty());
          EXPECT_FALSE(step.execution.transfer);
        }
      }

      EXPECT_EQ(result.committed_steps, complete);
      EXPECT_EQ(state.cells[2].value.word(0), complete >= 1 ? 0x1012 : 0xbad);
      EXPECT_EQ(state.cells[0].value.word(0), complete >= 2 ? 42 : 7);
      EXPECT_EQ(memory.Regions()[0].bytes[0], complete >= 2 ? 42 : 0);
      EXPECT_EQ(state.cells[1].value.word(0), complete >= 4 ? 42 : 11);
    }
  }
}

}  // namespace
}  // namespace nyx::eval
