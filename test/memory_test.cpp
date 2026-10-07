#include "nyx/eval/memory.hpp"

#include <array>
#include <limits>
#include <utility>

#include <gtest/gtest.h>

namespace nyx::eval {
namespace {

Budget Unlimited() { return Budget({1000000, 1000000}); }

TEST(Memory, CopiesAndSortsMappingsWithoutMaterializingGaps) {
  auto budget = Unlimited();
  std::array<std::uint8_t, 3> first{1, 2, 3};
  const std::array<std::uint8_t, 2> second{4, 5};
  const RegionInput regions[] = {{0x2000, second}, {0x1000, first}};
  auto result = Memory::Create(regions, budget);
  ASSERT_TRUE(result.memory);
  first[0] = 9;
  ASSERT_EQ(result.memory->Regions().size(), 2);
  EXPECT_EQ(result.memory->Regions()[0].address, 0x1000);
  EXPECT_EQ(result.memory->Regions()[0].bytes.size(), 3);
  auto read = result.memory->Read(0x1000, 3, budget);
  ASSERT_EQ(read.status, MemoryStatus::ok);
  EXPECT_EQ(read.bytes[0], 1);
  EXPECT_EQ(read.bytes[2], 3);
  read = result.memory->Read(0x1002, 2, budget);
  EXPECT_EQ(read.status, MemoryStatus::unmapped);
  EXPECT_EQ(read.fault_address, 0x1003);
  EXPECT_EQ(read.size, 0);
}

TEST(Memory, SupportsAdjacentRegionsAndChecksEveryBytePermission) {
  auto budget = Unlimited();
  const std::array<std::uint8_t, 2> a{1, 2}, b{3, 4};
  const RegionInput regions[] = {{8, a}, {10, b, true, false}};
  auto result = Memory::Create(regions, budget);
  ASSERT_TRUE(result.memory);
  auto read = result.memory->Read(9, 3, budget);
  ASSERT_EQ(read.status, MemoryStatus::ok);
  EXPECT_EQ(read.bytes[0], 2);
  EXPECT_EQ(read.bytes[1], 3);
  EXPECT_EQ(read.bytes[2], 4);
  EXPECT_EQ(result.memory->CheckAccess(9, 3, false, budget).status, MemoryStatus::ok);
  EXPECT_EQ(result.memory->CheckAccess(9, 3, true, budget).status, MemoryStatus::permission);
  auto transaction = result.memory->Begin(budget);
  const std::array<std::uint8_t, 2> replacement{9, 9};
  const auto write = transaction.Write(9, replacement);
  EXPECT_EQ(write.status, MemoryStatus::permission);
  EXPECT_EQ(write.fault_address, 10);
  transaction.Commit();
  EXPECT_EQ(result.memory->Read(9, 1, budget).bytes[0], 2);
}

// A byte the model cannot know, such as a slot the loader fills from an
// import, must not read as whatever the file happened to hold there: a run
// that reads it has nothing to compute with. Writing it gives it a value.
TEST(Memory, AnUnknownByteReadsAsUnknownUntilWritten) {
  auto budget = Unlimited();
  const std::array<std::uint8_t, 8> bytes{1, 2, 3, 4, 5, 6, 7, 8};
  const std::array<std::uint64_t, 2> unknown{2, 3};
  const RegionInput regions[] = {{0x100, bytes, true, true, unknown}};
  auto result = Memory::Create(regions, budget);
  ASSERT_TRUE(result.memory);
  EXPECT_EQ(result.memory->Read(0x100, 2, budget).status, MemoryStatus::ok);
  auto read = result.memory->Read(0x101, 2, budget);
  EXPECT_EQ(read.status, MemoryStatus::unknown_value);
  EXPECT_EQ(read.fault_address, 0x102);

  // Mapping and permission still come first.
  EXPECT_EQ(result.memory->CheckAccess(0x102, 2, false, budget).status, MemoryStatus::ok);

  auto transaction = result.memory->Begin(budget);
  const std::array<std::uint8_t, 1> one{0xaa};
  ASSERT_EQ(transaction.Write(0x102, one).status, MemoryStatus::ok);

  // The pending write covers one of the two unknown bytes, not both.
  EXPECT_EQ(transaction.Read(0x102, 1).bytes[0], 0xaa);
  read = transaction.Read(0x102, 2);
  EXPECT_EQ(read.status, MemoryStatus::unknown_value);
  EXPECT_EQ(read.fault_address, 0x103);
  ASSERT_EQ(transaction.Write(0x103, one).status, MemoryStatus::ok);
  transaction.Commit();
  read = result.memory->Read(0x100, 8, budget);
  ASSERT_EQ(read.status, MemoryStatus::ok);
  EXPECT_EQ(read.bytes[2], 0xaa);
  EXPECT_EQ(read.bytes[3], 0xaa);

  const std::array<std::uint64_t, 1> outside{8};
  const RegionInput invalid[] = {{0x100, bytes, true, true, outside}};
  EXPECT_EQ(Memory::Create(invalid, budget).status, MemoryStatus::invalid_mapping);
}

TEST(Memory, AdjacentWriteCommitsAcrossBothRegions) {
  auto budget = Unlimited();
  const std::array<std::uint8_t, 2> a{1, 2}, b{3, 4};
  const RegionInput regions[] = {{8, a}, {10, b}};
  auto result = Memory::Create(regions, budget);
  ASSERT_TRUE(result.memory);
  auto transaction = result.memory->Begin(budget);
  const std::array<std::uint8_t, 3> replacement{7, 8, 9};
  ASSERT_EQ(transaction.Write(9, replacement).status, MemoryStatus::ok);
  transaction.Commit();
  const auto read = result.memory->Read(8, 4, budget);
  ASSERT_EQ(read.status, MemoryStatus::ok);
  EXPECT_EQ(read.bytes[0], 1);
  EXPECT_EQ(read.bytes[1], 7);
  EXPECT_EQ(read.bytes[2], 8);
  EXPECT_EQ(read.bytes[3], 9);
}

TEST(Memory, RejectsOverlapEmptyAndWrappingRegions) {
  auto budget = Unlimited();
  const std::array<std::uint8_t, 4> bytes{};
  const RegionInput overlapping[] = {{8, bytes}, {10, bytes}};
  EXPECT_EQ(Memory::Create(overlapping, budget).status, MemoryStatus::invalid_mapping);
  const RegionInput wrapping[] = {{std::numeric_limits<std::uint64_t>::max() - 1, bytes}};
  EXPECT_EQ(Memory::Create(wrapping, budget).status, MemoryStatus::invalid_mapping);
  const RegionInput empty[] = {{8, {}}};
  EXPECT_EQ(Memory::Create(empty, budget).status, MemoryStatus::invalid_mapping);
  EXPECT_EQ(budget.used().bytes, 0);
}

TEST(Memory, FinalAddressAndAccessBoundariesAreCheckedWithoutWrap) {
  auto budget = Unlimited();
  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  const std::array<std::uint8_t, 1> bytes{3};
  const RegionInput regions[] = {{maximum, bytes}};
  auto result = Memory::Create(regions, budget);
  ASSERT_TRUE(result.memory);
  EXPECT_EQ(result.memory->Read(maximum, 1, budget).bytes[0], 3);
  EXPECT_EQ(result.memory->Read(maximum, 2, budget).status, MemoryStatus::address_overflow);
  EXPECT_EQ(result.memory->Read(maximum, 0, budget).status, MemoryStatus::unsupported_access);
  EXPECT_EQ(result.memory->Read(0, 17, budget).status, MemoryStatus::unsupported_access);
  auto transaction = result.memory->Begin(budget);
  const std::array<std::uint8_t, 1> replacement{9};
  ASSERT_EQ(transaction.Write(maximum, replacement).status, MemoryStatus::ok);
  transaction.Commit();
  EXPECT_EQ(result.memory->Read(maximum, 1, budget).bytes[0], 9);
}

TEST(Memory, OverlayReadsObserveOrderedOverlappingWrites) {
  auto budget = Unlimited();
  const std::array<std::uint8_t, 8> bytes{};
  const RegionInput regions[] = {{0x1000, bytes}};
  auto result = Memory::Create(regions, budget);
  ASSERT_TRUE(result.memory);
  auto transaction = result.memory->Begin(budget);
  const std::array<std::uint8_t, 4> first{1, 2, 3, 4};
  const std::array<std::uint8_t, 2> second{9, 8};
  ASSERT_EQ(transaction.Write(0x1000, first).status, MemoryStatus::ok);
  ASSERT_EQ(transaction.Write(0x1001, second).status, MemoryStatus::ok);
  auto read = transaction.Read(0x1000, 4);
  ASSERT_EQ(read.status, MemoryStatus::ok);
  EXPECT_EQ(read.bytes[0], 1);
  EXPECT_EQ(read.bytes[1], 9);
  EXPECT_EQ(read.bytes[2], 8);
  EXPECT_EQ(read.bytes[3], 4);
  EXPECT_EQ(result.memory->Read(0x1000, 4, budget).bytes[0], 0);
  transaction.Commit();
  EXPECT_EQ(result.memory->Read(0x1001, 2, budget).bytes[0], 9);
  EXPECT_EQ(transaction.Read(0x1000, 1).status, MemoryStatus::invalid_transaction);
  EXPECT_EQ(transaction.Write(0x1000, second).status, MemoryStatus::invalid_transaction);
}

TEST(Memory, DestructionDiscardsAndModeledFaultCanCommitPrefix) {
  auto budget = Unlimited();
  const std::array<std::uint8_t, 8> bytes{};
  const RegionInput regions[] = {{0x1000, bytes}};
  auto result = Memory::Create(regions, budget);
  ASSERT_TRUE(result.memory);
  const std::array<std::uint8_t, 8> value{1, 2, 3, 4, 5, 6, 7, 8};
  {
    auto transaction = result.memory->Begin(budget);
    ASSERT_EQ(transaction.Write(0x1000, value).status, MemoryStatus::ok);
  }

  EXPECT_EQ(result.memory->Read(0x1000, 8, budget).bytes[0], 0);
  auto transaction = result.memory->Begin(budget);
  ASSERT_EQ(transaction.Write(0x1000, value).status, MemoryStatus::ok);
  EXPECT_EQ(transaction.Write(0x1008, value).status, MemoryStatus::unmapped);
  transaction.Commit();
  EXPECT_EQ(result.memory->Read(0x1000, 8, budget).bytes[0], 1);
  EXPECT_EQ(result.memory->Read(0x1000, 8, budget).bytes[7], 8);
}

TEST(Memory, UnreadableMemoryCannotBeReadThroughOverlay) {
  auto budget = Unlimited();
  const std::array<std::uint8_t, 1> bytes{2};
  const RegionInput regions[] = {{8, bytes, false, true}};
  auto result = Memory::Create(regions, budget);
  ASSERT_TRUE(result.memory);
  auto transaction = result.memory->Begin(budget);
  ASSERT_EQ(transaction.Write(8, bytes).status, MemoryStatus::ok);
  EXPECT_EQ(transaction.Read(8, 1).status, MemoryStatus::permission);
}

TEST(Memory, BudgetFailuresNeverAllocateOrPublishPartialMemory) {
  const std::array<std::uint8_t, 8> bytes{};
  const RegionInput regions[] = {{8, bytes}};
  Budget none({10000, 0});
  EXPECT_EQ(Memory::Create(regions, none).status, MemoryStatus::resource_limit);
  EXPECT_EQ(none.used().bytes, 0);
  auto budget = Unlimited();
  EXPECT_EQ(Memory::Create(regions, budget, {0, 1024, 10}).status, MemoryStatus::resource_limit);
  EXPECT_EQ(Memory::Create(regions, budget, {10, 7, 10}).status, MemoryStatus::resource_limit);
  auto result = Memory::Create(regions, budget, {10, 1024, 1});
  ASSERT_TRUE(result.memory);
  auto transaction = result.memory->Begin(budget);
  const std::array<std::uint8_t, 1> value{9};
  ASSERT_EQ(transaction.Write(8, value).status, MemoryStatus::ok);
  EXPECT_EQ(transaction.Write(9, value).status, MemoryStatus::resource_limit);
  transaction.Commit();
  EXPECT_EQ(result.memory->Read(8, 1, budget).bytes[0], 0);
  Budget limited({100, 0});
  auto allocation_failure = result.memory->Begin(limited);
  EXPECT_EQ(allocation_failure.Write(8, value).status, MemoryStatus::resource_limit);
  allocation_failure.Commit();
  EXPECT_EQ(result.memory->Read(8, 1, budget).bytes[0], 0);
}

TEST(Memory, OverlayReadBudgetFailureDiscardsPreviouslyQueuedWrites) {
  auto budget = Unlimited();
  const std::array<std::uint8_t, 8> bytes{};
  const RegionInput regions[] = {{8, bytes}};
  auto result = Memory::Create(regions, budget);
  ASSERT_TRUE(result.memory);
  Budget limited({4, 10000});
  auto transaction = result.memory->Begin(limited);
  const std::array<std::uint8_t, 1> value{9};
  ASSERT_EQ(transaction.Write(8, value).status, MemoryStatus::ok);
  EXPECT_EQ(transaction.Read(8, 1).status, MemoryStatus::resource_limit);
  transaction.Commit();
  EXPECT_EQ(result.memory->Read(8, 1, budget).bytes[0], 0);
}

TEST(Memory, MovingTransactionTransfersOnlyCommitAuthority) {
  auto budget = Unlimited();
  const std::array<std::uint8_t, 1> bytes{};
  const RegionInput regions[] = {{8, bytes}};
  auto result = Memory::Create(regions, budget);
  ASSERT_TRUE(result.memory);
  auto original = result.memory->Begin(budget);
  const std::array<std::uint8_t, 1> value{9};
  ASSERT_EQ(original.Write(8, value).status, MemoryStatus::ok);
  auto moved = std::move(original);
  original.Commit();
  EXPECT_EQ(result.memory->Read(8, 1, budget).bytes[0], 0);
  moved.Commit();
  EXPECT_EQ(result.memory->Read(8, 1, budget).bytes[0], 9);
}

}  // namespace
}  // namespace nyx::eval
