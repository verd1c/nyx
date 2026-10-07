#include "nyx/analysis/image_writes.hpp"

#include <array>

#include <gtest/gtest.h>

namespace nyx::analysis {
namespace {
using ir::Node;
using ir::Op;

Budget Plenty() { return Budget({100000000, 100000000}); }

// An address is formed in one instruction and used by a later one, so a scan
// that looked at instructions singly would resolve nothing.
SourceRecord Holds(std::uint64_t address, std::uint64_t location) {
  return {address,
          {1, 2, 3, 4},
          ir::Group(address, {1, 2, 3, 4}, {{Op::image_address, 64, {}, location}}, {{9, 0}},
                    ir::MemoryModel::atomic_scalar_reference, {}),
          OpaqueReason::none,
          OpaqueControl::unknown};
}

SourceRecord StoresThrough(std::uint64_t address) {
  return {address,
          {1, 2, 3, 4},
          ir::Group(address, {1, 2, 3, 4},
                    {{Op::read, 64, {}, 0, 9}, {Op::read, 64, {}, 0, 0}, {Op::store, 64, {0, 1}}},
                    {}, ir::MemoryModel::atomic_scalar_reference, {}),
          OpaqueReason::none,
          OpaqueControl::unknown};
}

SourceRecord Undecodable(std::uint64_t address) {
  return {address, {1, 2, 3, 4}, std::nullopt, OpaqueReason::unsupported, OpaqueControl::unknown};
}

const std::array<RelocatedPointer, 1> kSlot{RelocatedPointer{0x2000, 0x1234, true}};

TEST(ImageWrites, PlacesAStoreThroughAnAddressFormedByAnEarlierInstruction) {
  const std::array<SourceRecord, 2> sources{Holds(0x1000, 0x2000), StoresThrough(0x1004)};
  auto budget = Plenty();
  const auto report = ScanImageWrites(sources, ImageFacts{{}, kSlot, false, {}}, budget);
  ASSERT_TRUE(report);
  ASSERT_EQ(report->writes.size(), 1U);
  EXPECT_EQ(report->writes[0].address, 0x2000U);
  EXPECT_EQ(report->scanned_groups, 2U);
  EXPECT_EQ(report->unmodeled_groups, 0U);
}

TEST(ImageWrites, AnInstructionWithNoSemanticsEndsTheRunAndIsCounted) {
  // What the undecodable instruction did to the address register is unknown,
  // so the store after it is not attributed to the location formed before it.
  const std::array<SourceRecord, 3> sources{Holds(0x1000, 0x2000), Undecodable(0x1004),
                                            StoresThrough(0x1008)};
  auto budget = Plenty();
  const auto report = ScanImageWrites(sources, ImageFacts{{}, kSlot, false, {}}, budget);
  ASSERT_TRUE(report);
  EXPECT_TRUE(report->writes.empty());
  EXPECT_EQ(report->unmodeled_groups, 1U);
  EXPECT_EQ(report->scanned_groups, 2U);
}

TEST(ImageWrites, AGapInAddressesEndsTheRun) {
  // Two instructions that do not follow one another are not one straight line,
  // whatever their order in the population.
  const std::array<SourceRecord, 2> sources{Holds(0x1000, 0x2000), StoresThrough(0x4000)};
  auto budget = Plenty();
  const auto report = ScanImageWrites(sources, ImageFacts{{}, kSlot, false, {}}, budget);
  ASSERT_TRUE(report);
  EXPECT_TRUE(report->writes.empty());
}

TEST(ImageWrites, WithoutDeclaredFactsThereIsNothingToRefute) {
  const std::array<SourceRecord, 2> sources{Holds(0x1000, 0x2000), StoresThrough(0x1004)};
  auto budget = Plenty();
  const auto report = ScanImageWrites(sources, {}, budget);
  ASSERT_TRUE(report);
  EXPECT_TRUE(report->writes.empty());
  EXPECT_EQ(report->scanned_groups, 0U);
}

TEST(ImageWrites, EveryBudgetCutPublishesNoPartialReport) {
  const std::array<SourceRecord, 2> sources{Holds(0x1000, 0x2000), StoresThrough(0x1004)};
  for (std::uint64_t work = 0; work < 400; ++work) {
    Budget budget({work, 100000000});
    const auto report = ScanImageWrites(sources, ImageFacts{{}, kSlot, false, {}}, budget);
    if (report) {
      EXPECT_EQ(report->writes.size(), 1U);
      return;
    }
  }

  FAIL() << "no work budget completed the scan";
}

}  // namespace
}  // namespace nyx::analysis
