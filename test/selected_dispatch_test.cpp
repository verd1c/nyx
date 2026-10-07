#include <array>

#include <gtest/gtest.h>

#include "nyx/eval/path.hpp"
#include "nyx/recovery/control.hpp"
#include "nyx/recovery/image.hpp"
#include "nyx/recovery/mba.hpp"
#include "nyx/recovery/memory.hpp"
#include "nyx/target/a64/decode.hpp"

namespace nyx {
namespace {

struct SourceWord {
  std::uint64_t address;
  std::uint32_t word;
};

struct SelectedRoute {
  std::array<SourceWord, 16> sources;
  std::uint64_t table_page;
  std::uint64_t table_base;
  std::array<std::uint64_t, 2> table_addresses;
  std::array<std::uint16_t, 2> table_values;
  std::array<std::uint64_t, 2> destinations;
  ir::StorageId selector;
  ir::StorageId pointer;
  std::array<std::uint32_t, 2> inputs;
  std::array<std::uint32_t, 2> stored_indices;
  std::array<std::uint32_t, 2> expected_nzcv;
};

void CheckRoute(const SelectedRoute& route) {
  Budget budget({100000000, 100000000});
  std::vector<ir::Group> groups;
  for (const auto [address, word] : route.sources) {
    const std::array<std::uint8_t, 4> bytes{
        static_cast<std::uint8_t>(word), static_cast<std::uint8_t>(word >> 8),
        static_cast<std::uint8_t>(word >> 16), static_cast<std::uint8_t>(word >> 24)};
    auto decoded =
        a64::Decode(address, bytes, budget, {a64::MemoryProfile::concrete_atomic_scalar});
    ASSERT_TRUE(decoded.group) << std::hex << address;
    groups.push_back(std::move(*decoded.group));
  }

  auto normalized = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(normalized.path);
  auto forwarded = recovery::ForwardMemoryValues(*normalized.path, budget);
  ASSERT_TRUE(forwarded.path);
  auto simplified = recovery::SimplifyMba(*forwarded.path, budget);
  ASSERT_TRUE(simplified.path);
  std::array<std::array<std::uint8_t, 2>, 2> entries{};
  std::array<ir::ConstantImageRange, 2> ranges{};
  for (unsigned index = 0; index < 2; ++index) {
    entries[index] = {static_cast<std::uint8_t>(route.table_values[index]),
                      static_cast<std::uint8_t>(route.table_values[index] >> 8)};
    ranges[index] = {route.table_addresses[index], entries[index]};
  }

  const ir::ImageFacts facts{ranges, {}, true, {}};
  auto folded = recovery::FoldImageValues(*simplified.path, facts, budget);
  ASSERT_TRUE(folded.path);
  const ir::ImageFacts held{folded.constants, folded.RetainedPointers(facts), true, {}};
  auto recovered = recovery::RecoverControl(std::move(*folded.path), held, budget);
  ASSERT_TRUE(recovered.path);
  ASSERT_EQ(recovered.path->rewrites().size(), 1);
  const auto& rewrite = recovered.path->rewrites()[0];
  EXPECT_EQ(rewrite.rule, ir::RewriteRule::dispatch_branch);
  EXPECT_EQ(rewrite.boundary, 15);
  EXPECT_EQ(rewrite.when_true, route.destinations[0]);
  EXPECT_EQ(rewrite.when_false, route.destinations[1]);
  EXPECT_EQ(ir::ValidateRecoveredPath(*recovered.path, budget, {}, held), ir::BlockDecline::none);

  constexpr std::uint64_t scratch_base = 0x100000000;
  for (const auto bias : {0x200000000ULL, 0x600000000ULL}) {
    for (unsigned arm = 0; arm < 2; ++arm) {
      std::array<std::uint8_t, 4096> scratch{}, table{};
      for (unsigned index = 0; index < 2; ++index) {
        const auto offset = route.table_addresses[index] - route.table_page;
        table[offset] = entries[index][0];
        table[offset + 1] = entries[index][1];
      }

      const std::array<eval::RegionInput, 2> mapped{
          {{scratch_base, scratch}, {bias + route.table_page, table, true, false}}};
      auto original_memory = eval::Memory::Create(mapped, budget);
      auto recovered_memory = eval::Memory::Create(mapped, budget);
      ASSERT_TRUE(original_memory.memory);
      ASSERT_TRUE(recovered_memory.memory);
      const auto initial_state = [&] {
        eval::State state;
        for (unsigned id = 0; id < 36; ++id) {
          std::uint64_t value = id < 32 ? id + 1 : id == a64::kN || id == a64::kV;
          if (id == a64::kSp) value = scratch_base + 2048;
          if (id == route.selector) value = route.inputs[arm];
          if (id == route.pointer && route.pointer != a64::kSp) value = scratch_base + 1024;
          auto bits = BitVector::from_u64(id < 32 ? 64 : 1, value, 64, budget);
          EXPECT_TRUE(bits);
          state.cells.push_back({id, std::move(*bits)});
        }

        return state;
      };

      auto original_state = initial_state(), recovered_state = initial_state();
      const auto original = eval::ExecutePath(*normalized.path, original_state,
                                              *original_memory.memory, budget, {}, {bias});
      const auto actual = eval::ExecuteRecoveredPath(
          *recovered.path, recovered_state, *recovered_memory.memory, budget, {}, {bias}, held);
      ASSERT_EQ(original.stop, eval::PathStop::completed) << arm;
      ASSERT_EQ(actual.stop, eval::PathStop::completed) << arm;
      EXPECT_EQ(original.outcome, eval::Outcome::completed);
      EXPECT_EQ(actual.outcome, original.outcome);
      EXPECT_EQ(original.runtime_next, bias + route.destinations[arm]);
      EXPECT_EQ(actual.runtime_next, original.runtime_next);
      EXPECT_EQ(original_state.cells[8].value.word(0), route.stored_indices[arm]);
      EXPECT_EQ(original_state.cells[9].value.word(0), bias + route.table_base);
      EXPECT_EQ(original_state.cells[10].value.word(0), bias + route.destinations[arm]);
      EXPECT_EQ(original_state.cells[11].value.word(0), route.table_values[arm]);
      std::uint32_t nzcv = 0;
      for (unsigned id = 32; id < 36; ++id)
        nzcv |= original_state.cells[id].value.word(0) << (63 - id);
      EXPECT_EQ(nzcv, route.expected_nzcv[arm]);
      ASSERT_EQ(original_state.cells.size(), recovered_state.cells.size());
      for (std::size_t id = 0; id < original_state.cells.size(); ++id)
        EXPECT_EQ(original_state.cells[id].value, recovered_state.cells[id].value) << id;
      ASSERT_EQ(original_memory.memory->Regions().size(),
                recovered_memory.memory->Regions().size());
      for (std::size_t index = 0; index < original_memory.memory->Regions().size(); ++index)
        EXPECT_EQ(original_memory.memory->Regions()[index].bytes,
                  recovered_memory.memory->Regions()[index].bytes);
      const auto stored = route.pointer == a64::kSp ? 2064U : 1032U;
      const auto& memory = original_memory.memory->Regions()[0].bytes;
      EXPECT_EQ(std::uint32_t(memory[stored]) | std::uint32_t(memory[stored + 1]) << 8 |
                    std::uint32_t(memory[stored + 2]) << 16 |
                    std::uint32_t(memory[stored + 3]) << 24,
                route.stored_indices[arm]);
      ASSERT_EQ(original.trace.size(), actual.trace.size());
      for (std::size_t boundary = 0; boundary < original.trace.size(); ++boundary) {
        const auto& a = original.trace[boundary].events;
        const auto& b = actual.trace[boundary].events;
        ASSERT_EQ(a.size(), b.size()) << boundary;
        for (std::size_t event = 0; event < a.size(); ++event) {
          EXPECT_EQ(a[event].address, b[event].address);
          EXPECT_EQ(a[event].size, b[event].size);
          EXPECT_EQ(a[event].write, b[event].write);
          EXPECT_EQ(a[event].completed, b[event].completed);
          EXPECT_EQ(a[event].conditional, b[event].conditional);
          EXPECT_EQ(a[event].performed, b[event].performed);
          EXPECT_EQ(a[event].bytes, b[event].bytes);
        }
      }
    }
  }
}

TEST(SelectedDispatch, DevelopmentConditionMapsToBothOriginalDestinations) {
  CheckRoute({{{{0xbdca84, 0x528001e8},
                {0xbdca88, 0x52800209},
                {0xbdca8c, 0x7100041f},
                {0xbdca90, 0x1a89b108},
                {0xbdca94, 0xb9000a68},
                {0xbdca98, 0x17fffee7},
                {0xbdc634, 0x14000128},
                {0xbdcad4, 0xb9400a68},
                {0xbdcad8, 0x71008d09},
                {0xbdcadc, 0x54ffe1c8},
                {0xbdcae0, 0xd0ffb2c9},
                {0xbdcae4, 0x913ba129},
                {0xbdcae8, 0x10ffda8a},
                {0xbdcaec, 0x7868792b},
                {0xbdcaf0, 0x8b0b094a},
                {0xbdcaf4, 0xd61f0140}}},
              0x236000,
              0x236ee8,
              {0x236f06, 0x236f08},
              {322, 596},
              {0xbdcb40, 0xbdcf88},
              0,
              19,
              {0, 1},
              {15, 16},
              {0x80000000, 0x80000000}});
}

TEST(SelectedDispatch, SampleConditionMapsToBothOriginalDestinations) {
  CheckRoute({{{{0x144d064, 0x52800028},
                {0x144d068, 0x52800249},
                {0x144d06c, 0x7200033f},
                {0x144d070, 0x1a891108},
                {0x144d074, 0xb90013e8},
                {0x144d078, 0x1400015f},
                {0x144d5f4, 0x17fffe2e},
                {0x144ceac, 0xb94013e8},
                {0x144ceb0, 0x71004909},
                {0x144ceb4, 0x54ffe3a8},
                {0x144ceb8, 0xd0ff71c9},
                {0x144cebc, 0x911bb129},
                {0x144cec0, 0x10ffce8a},
                {0x144cec4, 0x7868792b},
                {0x144cec8, 0x8b0b094a},
                {0x144cecc, 0xd61f0140}}},
              0x286000,
              0x2866ec,
              {0x2866ee, 0x286710},
              {677, 352},
              {0x144d324, 0x144ce10},
              25,
              a64::kSp,
              {1, 0},
              {1, 18},
              {0x80000000, 0x60000000}});
}

}  // namespace
}  // namespace nyx
