#include <algorithm>
#include <array>

#include <gtest/gtest.h>

#include "nyx/analysis/cfg.hpp"
#include "nyx/analysis/path_control.hpp"
#include "nyx/eval/path.hpp"
#include "nyx/recovery/control.hpp"
#include "nyx/recovery/image.hpp"
#include "nyx/recovery/mba.hpp"
#include "nyx/recovery/memory.hpp"
#include "nyx/recovery/overwritten_store.hpp"
#include "nyx/recovery/paired_load.hpp"
#include "nyx/target/a64/decode.hpp"

namespace nyx {
namespace {

TEST(SelectedStack, OriginalStackAdjustmentPermitsProvedStoreOmission) {
  constexpr std::uint64_t base = 0x152869c;
  constexpr std::array<std::uint32_t, 5> words{0xd100c3ff, 0xa901fbfd, 0x910063fd, 0xf90003e1,
                                               0xf90003e2};
  Budget budget({1000000, 1000000});
  std::vector<ir::Group> groups;
  for (std::size_t index = 0; index < words.size(); ++index) {
    const auto word = words[index];
    const std::array<std::uint8_t, 4> bytes{
        static_cast<std::uint8_t>(word), static_cast<std::uint8_t>(word >> 8),
        static_cast<std::uint8_t>(word >> 16), static_cast<std::uint8_t>(word >> 24)};
    auto decoded =
        a64::Decode(base + index * 4, bytes, budget, {a64::MemoryProfile::concrete_atomic_scalar});
    ASSERT_TRUE(decoded.group) << index;
    groups.push_back(std::move(*decoded.group));
  }

  auto normalized = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(normalized.path);
  const auto basis = *normalized.path;
  auto proposed = recovery::OmitOverwrittenStores(
      ir::RecoveredPath(std::move(*normalized.path), {}, 0), a64::kSp, budget);
  ASSERT_TRUE(proposed.path);
  ASSERT_EQ(proposed.path->omissions().size(), 1);
  const auto& omission = proposed.path->omissions()[0];
  EXPECT_EQ(basis.sources()[basis.origins()[omission.store].boundary].source_address(), base + 12);
  EXPECT_EQ(basis.sources()[basis.origins()[omission.overwriter].boundary].source_address(),
            base + 16);
  EXPECT_EQ(omission.offset_bytes, UINT64_MAX - 47);
  EXPECT_EQ(ir::ValidateRecoveredPath(*proposed.path, budget), ir::BlockDecline::none);
  constexpr std::array<std::uint8_t, 4> different_address{0xe2, 0x07, 0x00, 0xf9};
  auto changed_store = a64::Decode(base + 16, different_address, budget,
                                   {a64::MemoryProfile::concrete_atomic_scalar});
  ASSERT_TRUE(changed_store.group);
  groups.back() = std::move(*changed_store.group);
  auto changed_path = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(changed_path.path);
  auto refused = recovery::OmitOverwrittenStores(
      ir::RecoveredPath(std::move(*changed_path.path), {}, 0), a64::kSp, budget);
  ASSERT_TRUE(refused.path);
  EXPECT_TRUE(refused.path->omissions().empty());

  constexpr std::uint64_t scratch_base = 0x100000000;
  std::array<std::uint8_t, 256> scratch{};
  const auto state = [&]() {
    eval::State result;
    for (const unsigned id : {1U, 2U, 29U, 30U, 31U}) {
      auto value = BitVector::from_u64(64, id == 31 ? scratch_base + 128 : id + 1, 64, budget);
      EXPECT_TRUE(value);
      result.cells.push_back({id, std::move(*value)});
    }

    return result;
  };

  const eval::RegionInput mapped[] = {{scratch_base, scratch}};
  auto original_memory = eval::Memory::Create(mapped, budget);
  auto recovered_memory = eval::Memory::Create(mapped, budget);
  ASSERT_TRUE(original_memory.memory);
  ASSERT_TRUE(recovered_memory.memory);
  auto original_state = state(), recovered_state = state();
  const auto original =
      eval::ExecutePath(basis, original_state, *original_memory.memory, budget, {}, {0});
  const auto recovered = eval::ExecuteRecoveredPath(*proposed.path, recovered_state,
                                                    *recovered_memory.memory, budget, {}, {0});
  ASSERT_EQ(original.outcome, eval::Outcome::completed);
  EXPECT_EQ(recovered.outcome, original.outcome);
  ASSERT_EQ(original_state.cells.size(), recovered_state.cells.size());
  for (std::size_t index = 0; index < original_state.cells.size(); ++index)
    EXPECT_EQ(original_state.cells[index].value, recovered_state.cells[index].value);
  EXPECT_EQ(original_memory.memory->Regions()[0].bytes,
            recovered_memory.memory->Regions()[0].bytes);
  ASSERT_EQ(original.trace.size(), words.size());
  ASSERT_EQ(recovered.trace.size(), words.size());
  EXPECT_EQ(original.trace[3].events.size(), 1);
  EXPECT_TRUE(recovered.trace[3].events.empty());
  EXPECT_EQ(recovered.trace[4].events.size(), 1);

  const eval::RegionInput guarded[] = {
      {scratch_base, std::span(scratch).first(80)},
      {scratch_base + 80, std::span(scratch).subspan(80, 8), true, false},
      {scratch_base + 88, std::span(scratch).subspan(88)}};
  auto guarded_memory = eval::Memory::Create(guarded, budget);
  ASSERT_TRUE(guarded_memory.memory);
  auto guarded_state = state();
  const auto declined = eval::ExecuteRecoveredPath(*proposed.path, guarded_state,
                                                   *guarded_memory.memory, budget, {}, {0});
  EXPECT_EQ(declined.outcome, eval::Outcome::unsupported);
  EXPECT_EQ(declined.completed_boundaries, 3);
  ASSERT_EQ(declined.trace.size(), 4);
  EXPECT_TRUE(declined.trace[3].events.empty());
}

TEST(SelectedStack, PureInstructionBetweenOriginalStoresPermitsOmission) {
  constexpr std::uint64_t base = 0x81a934;
  constexpr std::array<std::uint32_t, 3> words{0xf9000fe9, 0x9280050a, 0xf9000fea};
  Budget budget({1000000, 1000000});
  const auto path = [&](std::span<const std::uint32_t> source) {
    std::vector<ir::Group> groups;
    for (std::size_t index = 0; index < source.size(); ++index) {
      const auto word = source[index];
      const std::array<std::uint8_t, 4> bytes{
          static_cast<std::uint8_t>(word), static_cast<std::uint8_t>(word >> 8),
          static_cast<std::uint8_t>(word >> 16), static_cast<std::uint8_t>(word >> 24)};
      auto decoded = a64::Decode(base + index * 4, bytes, budget,
                                 {a64::MemoryProfile::concrete_atomic_scalar});
      EXPECT_TRUE(decoded.group) << index;
      groups.push_back(std::move(*decoded.group));
    }

    auto normalized = ir::NormalizePath(groups, budget);
    EXPECT_TRUE(normalized.path);
    return std::move(*normalized.path);
  };

  auto original = path(words);
  auto proposed =
      recovery::OmitOverwrittenStores(ir::RecoveredPath(original, {}, 0), a64::kSp, budget);
  ASSERT_TRUE(proposed.path);
  ASSERT_EQ(proposed.path->omissions().size(), 1);
  const auto& omission = proposed.path->omissions()[0];
  EXPECT_EQ(original.sources()[original.origins()[omission.store].boundary].source_address(), base);
  EXPECT_EQ(original.sources()[original.origins()[omission.overwriter].boundary].source_address(),
            base + 8);
  EXPECT_EQ(ir::ValidateRecoveredPath(*proposed.path, budget), ir::BlockDecline::none);

  constexpr std::uint64_t scratch_base = 0x100000000;
  std::array<std::uint8_t, 64> scratch{};
  const eval::RegionInput mapped[] = {{scratch_base, scratch}};
  auto original_memory = eval::Memory::Create(mapped, budget);
  auto recovered_memory = eval::Memory::Create(mapped, budget);
  ASSERT_TRUE(original_memory.memory);
  ASSERT_TRUE(recovered_memory.memory);
  const auto state = [&]() {
    eval::State result;
    for (const auto [id, value] : std::array<std::pair<unsigned, std::uint64_t>, 3>{
             std::pair{9U, std::uint64_t{17}}, std::pair{10U, std::uint64_t{0}},
             std::pair{a64::kSp, scratch_base}}) {
      auto bits = BitVector::from_u64(64, value, 64, budget);
      EXPECT_TRUE(bits);
      result.cells.push_back({id, std::move(*bits)});
    }

    return result;
  };

  auto original_state = state(), recovered_state = state();
  const auto before =
      eval::ExecutePath(original, original_state, *original_memory.memory, budget, {}, {0});
  const auto after = eval::ExecuteRecoveredPath(*proposed.path, recovered_state,
                                                *recovered_memory.memory, budget, {}, {0});
  ASSERT_EQ(before.outcome, eval::Outcome::completed)
      << before.completed_boundaries << " " << before.source_address << " "
      << (before.trace.empty() || !before.trace.back().fault ? 0
                                                             : before.trace.back().fault->address);
  EXPECT_EQ(after.outcome, before.outcome);
  EXPECT_EQ(original_state.cells.size(), recovered_state.cells.size());
  for (std::size_t id = 0; id < original_state.cells.size(); ++id)
    EXPECT_EQ(original_state.cells[id].value, recovered_state.cells[id].value);
  EXPECT_EQ(original_memory.memory->Regions()[0].bytes,
            recovered_memory.memory->Regions()[0].bytes);
  ASSERT_EQ(before.trace.size(), 3);
  ASSERT_EQ(after.trace.size(), 3);
  EXPECT_EQ(before.trace[0].events.size(), 1);
  EXPECT_TRUE(after.trace[0].events.empty());
  EXPECT_EQ(after.trace[2].events.size(), 1);

  constexpr std::array<std::uint32_t, 3> read_between{0xf9000fe9, 0xf9400fea, 0xf9000fea};
  auto refused = recovery::OmitOverwrittenStores(ir::RecoveredPath(path(read_between), {}, 0),
                                                 a64::kSp, budget);
  ASSERT_TRUE(refused.path);
  EXPECT_TRUE(refused.path->omissions().empty());

  std::vector<std::uint32_t> distant{words.front()};
  distant.insert(distant.end(), ir::kMaxOverwriteBoundaryDistance, 0xd503201f);
  distant.push_back(words.back());
  auto long_path = path(distant);
  auto too_far =
      recovery::OmitOverwrittenStores(ir::RecoveredPath(long_path, {}, 0), a64::kSp, budget);
  ASSERT_TRUE(too_far.path);
  EXPECT_TRUE(too_far.path->omissions().empty());
  std::vector<ir::ValueId> stores;
  for (ir::ValueId id = 0; id < long_path.nodes().size(); ++id) {
    if (long_path.nodes()[id].op == ir::Op::store) stores.push_back(id);
  }

  ASSERT_EQ(stores.size(), 2);
  ir::RecoveredPath forged(std::move(long_path), {}, 1, {},
                           {{stores[0], stores[1], a64::kSp, 24, 0, 1}});
  EXPECT_EQ(ir::ValidateRecoveredPath(forged, budget), ir::BlockDecline::invalid_ir);
}

TEST(SelectedStack, DisjointInterveningStoresPreservePairWitness) {
  constexpr std::uint64_t base = 0x14e0d08;
  constexpr std::array<std::uint32_t, 8> words{0xd10083ff, 0xa9017bfd, 0x910043fd, 0xf90007e0,
                                               0xf90003e0, 0xf94003e0, 0xf94007e0, 0xa9417bfd};
  Budget budget({1000000, 1000000});
  std::vector<ir::Group> groups;
  for (std::size_t index = 0; index < words.size(); ++index) {
    const auto word = words[index];
    const std::array<std::uint8_t, 4> bytes{
        static_cast<std::uint8_t>(word), static_cast<std::uint8_t>(word >> 8),
        static_cast<std::uint8_t>(word >> 16), static_cast<std::uint8_t>(word >> 24)};
    auto decoded =
        a64::Decode(base + index * 4, bytes, budget, {a64::MemoryProfile::concrete_atomic_scalar});
    ASSERT_TRUE(decoded.group);
    groups.push_back(std::move(*decoded.group));
  }

  auto normalized = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(normalized.path);
  auto forwarded = recovery::ForwardMemoryValues(*normalized.path, budget);
  ASSERT_TRUE(forwarded.path);
  auto original = *forwarded.path;
  auto result = recovery::OmitForwardedPairLoads(
      ir::RecoveredPath(std::move(*forwarded.path), {}, original.revision()), forwarded.facts,
      a64::kSp, budget);
  ASSERT_TRUE(result.path);
  ASSERT_EQ(result.path->paired_load_omissions().size(), 1);
  const auto& witness = result.path->paired_load_omissions()[0];
  EXPECT_TRUE(ir::ValidPairedLoadOmission(original, witness));
  auto nodes = std::vector<ir::Node>(original.nodes().begin(), original.nodes().end());
  nodes[16].inputs[0] = nodes[witness.stores[0]].inputs[0];
  ir::Path overlapping({original.sources().begin(), original.sources().end()}, std::move(nodes),
                       {original.origins().begin(), original.origins().end()},
                       {original.boundaries().begin(), original.boundaries().end()},
                       original.revision());
  EXPECT_FALSE(ir::ValidPairedLoadOmission(overlapping, witness));

  constexpr std::uint64_t scratch_base = 0x100000000;
  std::array<std::uint8_t, 4096> scratch{};
  const eval::RegionInput mapped[] = {{scratch_base, scratch}};
  auto original_memory = eval::Memory::Create(mapped, budget);
  auto omitted_memory = eval::Memory::Create(mapped, budget);
  ASSERT_TRUE(original_memory.memory);
  ASSERT_TRUE(omitted_memory.memory);
  const auto state = [&]() {
    eval::State result;
    for (unsigned id = 0; id < 36; ++id) {
      auto value = BitVector::from_u64(id < 32 ? 64 : 1, id == 31 ? scratch_base + 2048 : id + 1,
                                       64, budget);
      EXPECT_TRUE(value);
      result.cells.push_back({id, std::move(*value)});
    }

    return result;
  };

  auto original_state = state(), omitted_state = state();
  const auto before =
      eval::ExecutePath(original, original_state, *original_memory.memory, budget, {}, {0});
  const auto after = eval::ExecuteRecoveredPath(*result.path, omitted_state, *omitted_memory.memory,
                                                budget, {}, {0});
  ASSERT_EQ(before.outcome, eval::Outcome::completed);
  EXPECT_EQ(after.outcome, before.outcome);
  EXPECT_EQ(after.runtime_next, before.runtime_next);
  ASSERT_EQ(original_state.cells.size(), omitted_state.cells.size());
  for (std::size_t id = 0; id < original_state.cells.size(); ++id)
    EXPECT_EQ(original_state.cells[id].value, omitted_state.cells[id].value) << id;
  EXPECT_EQ(original_memory.memory->Regions()[0].bytes, omitted_memory.memory->Regions()[0].bytes);
  ASSERT_EQ(before.trace.size(), words.size());
  ASSERT_EQ(after.trace.size(), words.size());
  EXPECT_EQ(before.trace[7].events.size(), 2);
  EXPECT_TRUE(after.trace[7].events.empty());
}

TEST(SelectedStack, OriginalDevelopmentBytesHaveOneProvedItineraryToPairOverwrite) {
  // Sample bytes from 0x78a3c4+84. The repeated addresses below are occurrences
  // on the route through the loop, not duplicate machine instructions.
  constexpr std::uint64_t base = 0x78a3c4;
  constexpr std::array<std::uint32_t, 21> words{
      0xa9bf7bfd, 0x910003fd, 0xd283e049, 0x2a1f03eb, 0xb000a5c8, 0xf2aeb889, 0x5280038a,
      0xf2c25909, 0xf2f53029, 0x2a0b03ec, 0x5280002b, 0x34ffffcc, 0x7100059f, 0x540000c1,
      0xf947490b, 0xcb0b012b, 0xa9002faa, 0x5280004b, 0x17fffff7, 0xa8c17bfd, 0xd65f03c0};
  constexpr std::array<std::uint32_t, 29> occurrences{0,  1,  2, 3,  4,  5,  6,  7,  8,  9,
                                                      10, 11, 9, 10, 11, 12, 13, 14, 15, 16,
                                                      17, 18, 9, 10, 11, 12, 13, 19, 20};
  Budget budget({100000000, 100000000});
  std::vector<analysis::SourceRecord> sources;
  for (std::size_t index = 0; index < words.size(); ++index) {
    const auto word = words[index];
    const std::array<std::uint8_t, 4> bytes{
        static_cast<std::uint8_t>(word), static_cast<std::uint8_t>(word >> 8),
        static_cast<std::uint8_t>(word >> 16), static_cast<std::uint8_t>(word >> 24)};
    auto decoded =
        a64::Decode(base + index * 4, bytes, budget, {a64::MemoryProfile::concrete_atomic_scalar});
    ASSERT_TRUE(decoded.group) << index;
    sources.push_back({base + index * 4, {bytes.begin(), bytes.end()}, std::move(*decoded.group)});
  }

  const std::array entries{base};
  auto cfg = analysis::BuildCfg(sources, entries, budget);
  ASSERT_TRUE(cfg.cfg);
  ASSERT_EQ(cfg.cfg->blocks().size(), 5);
  for (std::size_t index = 0; index < cfg.cfg->blocks().size(); ++index) {
    const auto& block = cfg.cfg->blocks()[index];
    EXPECT_EQ(block.first_source, (std::array<unsigned, 5>{0, 9, 12, 14, 19})[index]);
    EXPECT_EQ(block.edges.size(), (std::array<unsigned, 5>{1, 2, 2, 1, 1})[index]);
    for (const auto& edge : block.edges) {
      if (edge.kind == analysis::CfgEdgeKind::return_) continue;
      EXPECT_EQ(edge.resolution, analysis::TargetResolution::block_entry);
    }
  }

  std::vector<ir::Group> groups;
  groups.reserve(occurrences.size());
  for (const auto index : occurrences) {
    groups.push_back(*sources[index].semantics);
  }

  auto normalized = ir::NormalizePath(groups, budget);
  ASSERT_TRUE(normalized.path);
  auto forwarded = recovery::ForwardMemoryValues(*normalized.path, budget);
  ASSERT_TRUE(forwarded.path);
  auto simplified = recovery::SimplifyMba(*forwarded.path, budget);
  ASSERT_TRUE(simplified.path);
  auto folded = recovery::FoldImageValues(*simplified.path, {}, budget);
  ASSERT_TRUE(folded.path);
  auto recovered = recovery::RecoverControl(std::move(*folded.path), {}, budget);
  ASSERT_TRUE(recovered.path);
  ASSERT_EQ(recovered.path->rewrites().size(), 5);
  constexpr std::array<bool, 5> route{true, false, false, false, true};
  for (std::size_t i = 0; i < route.size(); ++i) {
    const auto& rewrite = recovered.path->rewrites()[i];
    EXPECT_EQ(rewrite.rule, ir::RewriteRule::folded_condition);
    EXPECT_EQ(rewrite.condition_value, route[i]);
    EXPECT_EQ(rewrite.replacement.kind, ir::TransferKind::jump);
  }

  EXPECT_EQ(ir::ValidateRecoveredPath(*recovered.path, budget), ir::BlockDecline::none);
  auto control = analysis::AnalyzePathControl(*recovered.path,
                                              {recovered.path->basis().revision(), {}}, budget);
  ASSERT_TRUE(control.facts);
  EXPECT_FALSE(control.facts->proved_divergence);
  for (const auto& boundary : control.facts->boundaries) {
    if (boundary.expected_image_successor && boundary.transfer_kind) {
      EXPECT_EQ(boundary.expected_match, analysis::ExpectedMatch::always) << boundary.boundary;
    }
  }

  std::vector<ir::ValueId> stores, loads;
  const auto nodes = recovered.path->basis().nodes();
  for (ir::ValueId id = 0; id < nodes.size(); ++id) {
    if (nodes[id].op == ir::Op::store) stores.push_back(id);
    if (nodes[id].op == ir::Op::load) loads.push_back(id);
  }

  EXPECT_EQ(stores, (std::vector<ir::ValueId>{7, 8, 89, 90}));
  EXPECT_EQ(loads, (std::vector<ir::ValueId>{83, 132, 133}));
  EXPECT_EQ(recovered.path->basis().sources()[0].source_address(), base);
  EXPECT_EQ(recovered.path->basis().sources()[19].source_address(), 0x78a404);

  const auto original_path = recovered.path->basis();
  auto omitted = recovery::OmitForwardedPairLoads(std::move(*recovered.path), forwarded.facts,
                                                  a64::kSp, budget);
  ASSERT_TRUE(omitted.path);
  ASSERT_EQ(omitted.path->paired_load_omissions().size(), 1);
  const auto& witness = omitted.path->paired_load_omissions()[0];
  EXPECT_EQ(witness.stores, (std::array<ir::ValueId, 2>{89, 90}));
  EXPECT_EQ(witness.loads, (std::array<ir::ValueId, 2>{132, 133}));
  EXPECT_EQ(witness.offset_bytes, (std::array<std::uint64_t, 2>{UINT64_MAX - 15, UINT64_MAX - 7}));
  EXPECT_EQ(ir::ValidateRecoveredPath(*omitted.path, budget), ir::BlockDecline::none);
  for (unsigned mutation = 0; mutation < 3; ++mutation) {
    auto forged = witness;
    if (mutation == 0) forged.base_storage = 29;
    if (mutation == 1) ++forged.offset_bytes[1];
    if (mutation == 2) forged.stores[1] = 8;
    EXPECT_EQ(
        ir::ValidateRecoveredPath(
            ir::RecoveredPath(original_path, {}, forged.to_revision, {}, {}, {forged}), budget),
        ir::BlockDecline::invalid_ir)
        << mutation;
  }

  const auto altered_basis = [&](std::vector<ir::Node> altered) {
    return ir::Path({original_path.sources().begin(), original_path.sources().end()},
                    std::move(altered),
                    {original_path.origins().begin(), original_path.origins().end()},
                    {original_path.boundaries().begin(), original_path.boundaries().end()},
                    original_path.revision());
  };

  auto live_boundaries = std::vector<ir::Boundary>(original_path.boundaries().begin(),
                                                   original_path.boundaries().end());
  live_boundaries[original_path.origins()[132].boundary].writes.push_back({0, 132});
  ir::Path live_path({original_path.sources().begin(), original_path.sources().end()},
                     {nodes.begin(), nodes.end()},
                     {original_path.origins().begin(), original_path.origins().end()},
                     std::move(live_boundaries), original_path.revision());
  EXPECT_FALSE(ir::ValidPairedLoadOmission(live_path, witness));
  auto interposed_nodes = std::vector<ir::Node>(nodes.begin(), nodes.end());
  interposed_nodes[91].op = ir::Op::store;
  interposed_nodes[91].width = 64;
  interposed_nodes[91].inputs[0] = nodes[witness.stores[0]].inputs[0];
  EXPECT_FALSE(ir::ValidPairedLoadOmission(altered_basis(std::move(interposed_nodes)), witness));
  auto no_facts = recovery::OmitForwardedPairLoads(
      ir::RecoveredPath(original_path, {}, original_path.revision()), {}, a64::kSp, budget);
  ASSERT_TRUE(no_facts.path);
  EXPECT_TRUE(no_facts.path->paired_load_omissions().empty());
  auto duplicate_facts = forwarded.facts;
  const auto repeated = std::find_if(duplicate_facts.begin(), duplicate_facts.end(),
                                     [](const auto& fact) { return fact.load == 132; });
  ASSERT_NE(repeated, duplicate_facts.end());
  duplicate_facts.push_back(*repeated);
  auto ambiguous = recovery::OmitForwardedPairLoads(
      ir::RecoveredPath(original_path, {}, original_path.revision()), duplicate_facts, a64::kSp,
      budget);
  ASSERT_TRUE(ambiguous.path);
  EXPECT_TRUE(ambiguous.path->paired_load_omissions().empty());

  constexpr std::uint64_t bias = 0x200000000;
  constexpr std::uint64_t scratch_base = 0x100000000;
  std::array<std::uint8_t, 4096> scratch{}, global{};
  scratch[2032] = 0xa5;
  global[0xe90] = 123;
  const eval::RegionInput mapped[] = {{scratch_base, scratch},
                                      {bias + 0x1c43000, global, true, false}};
  auto original_memory = eval::Memory::Create(mapped, budget);
  auto omitted_memory = eval::Memory::Create(mapped, budget);
  ASSERT_TRUE(original_memory.memory);
  ASSERT_TRUE(omitted_memory.memory);
  const auto state = [&]() {
    eval::State result;
    for (unsigned id = 0; id < 36; ++id) {
      const auto initial = id == 31 ? scratch_base + 2048 : id == 28 ? scratch_base : id + 1;
      auto value = BitVector::from_u64(id < 32 ? 64 : 1, initial, 64, budget);
      EXPECT_TRUE(value);
      result.cells.push_back({id, std::move(*value)});
    }

    return result;
  };

  auto original_state = state(), omitted_state = state();
  const auto original =
      eval::ExecutePath(original_path, original_state, *original_memory.memory, budget, {}, {bias});
  const auto actual = eval::ExecuteRecoveredPath(*omitted.path, omitted_state,
                                                 *omitted_memory.memory, budget, {}, {bias});
  EXPECT_EQ(original.outcome, eval::Outcome::completed);
  EXPECT_EQ(actual.outcome, original.outcome);
  EXPECT_EQ(actual.runtime_next, original.runtime_next);
  ASSERT_EQ(original_state.cells.size(), omitted_state.cells.size());
  for (std::size_t id = 0; id < original_state.cells.size(); ++id)
    EXPECT_EQ(original_state.cells[id].value, omitted_state.cells[id].value) << id;
  EXPECT_EQ(original_memory.memory->Regions()[0].bytes, omitted_memory.memory->Regions()[0].bytes);
  ASSERT_EQ(original.trace.size(), 29);
  ASSERT_EQ(actual.trace.size(), 29);
  EXPECT_EQ(original.trace[27].events.size(), 2);
  EXPECT_TRUE(actual.trace[27].events.empty());

  const eval::RegionInput unreadable[] = {
      {scratch_base, std::span(scratch).first(2040)},
      {scratch_base + 2040, std::span(scratch).subspan(2040), false, true},
      {bias + 0x1c43000, global, true, false}};
  auto guarded = eval::Memory::Create(unreadable, budget);
  ASSERT_TRUE(guarded.memory);
  auto guarded_state = state();
  const auto declined =
      eval::ExecuteRecoveredPath(*omitted.path, guarded_state, *guarded.memory, budget, {}, {bias});
  EXPECT_EQ(declined.outcome, eval::Outcome::unsupported);
  EXPECT_EQ(declined.completed_boundaries, 27);
  ASSERT_EQ(declined.trace.size(), 28);
  EXPECT_TRUE(declined.trace[27].events.empty());

  const eval::RegionInput unreadable_first[] = {
      {scratch_base, std::span(scratch).first(2032)},
      {scratch_base + 2032, std::span(scratch).subspan(2032, 8), false, true},
      {scratch_base + 2040, std::span(scratch).subspan(2040)},
      {bias + 0x1c43000, global, true, false}};
  auto first_guarded = eval::Memory::Create(unreadable_first, budget);
  ASSERT_TRUE(first_guarded.memory);
  auto first_state = state();
  const auto first_declined = eval::ExecuteRecoveredPath(*omitted.path, first_state,
                                                         *first_guarded.memory, budget, {}, {bias});
  EXPECT_EQ(first_declined.outcome, eval::Outcome::unsupported);
  EXPECT_EQ(first_declined.completed_boundaries, 27);
  ASSERT_EQ(first_declined.trace.size(), 28);
  EXPECT_TRUE(first_declined.trace[27].events.empty());
}

}  // namespace
}  // namespace nyx
