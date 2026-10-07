#include <set>
#include <string>

#include <gtest/gtest.h>

#include "nyx/passes/pipeline.hpp"

namespace nyx::passes {
namespace {
Budget Plenty() { return Budget({1000000, 10000000}); }

std::vector<std::uint8_t> Bytes(std::uint32_t word) {
  return {static_cast<std::uint8_t>(word), static_cast<std::uint8_t>(word >> 8),
          static_cast<std::uint8_t>(word >> 16), static_cast<std::uint8_t>(word >> 24)};
}

// One returning block: every reachability pass needs both scope declarations.
std::vector<ir::Group> Sources() {
  std::vector<ir::Group> sources;
  sources.emplace_back(0x100, Bytes(0xd65f03c0),
                       std::vector<ir::Node>{{ir::Op::read, 64, {}, 0, 30}},
                       std::vector<ir::Write>{}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}});
  return sources;
}

ir::SsaGraph Graph() {
  ir::SsaGraph graph;
  ir::SsaBlock entry{};
  entry.address = 0x100;
  entry.source_groups = {0x100};
  entry.source_bytes = {Bytes(0xd65f03c0)};
  entry.original_sources = {0};
  entry.nodes = {{ir::Op::read, 64, {}, 0, 30}};
  entry.boundaries = {{0, 1, {}, ir::Transfer{ir::TransferKind::return_, 0, {}, {}, {}}}};
  entry.phis = {{30, 64, true, {}}};
  entry.clobbers = {0};
  entry.edges = {{ir::SsaEdgeKind::return_,
                  ir::SsaTargetKind::unknown,
                  0,
                  {},
                  {},
                  {},
                  {.return_leaves = true}}};
  const auto start = graph.Add(std::move(entry));
  graph.Update(start, [&](auto& block) {
    block.reads = {{0, 0, {ir::SsaValueKind::phi, start, 0}}};
    block.exits = {{30, {ir::SsaValueKind::phi, start, 0}}};
  });
  graph.SetEntries({start});
  return graph;
}

std::size_t Index(std::string_view name) {
  const auto index = FindSsaPass(name);
  EXPECT_TRUE(index) << name;
  return index.value_or(0);
}

std::string Stage(std::string_view pass, std::string_view outcome, std::string_view reason) {
  return "{\"pass\":\"" + std::string(pass) + "\",\"outcome\":\"" + std::string(outcome) +
         "\",\"reason\":\"" + std::string(reason) + "\"";
}

TEST(SsaRegistry, NamesAreUniqueFindableAndRunnable) {
  const auto registry = SsaPassRegistry();
  ASSERT_FALSE(registry.empty());
  std::set<std::string_view> names;
  for (std::size_t index = 0; index < registry.size(); ++index) {
    const auto& pass = registry[index];
    EXPECT_TRUE(names.insert(pass.name).second) << pass.name;
    EXPECT_EQ(FindSsaPass(pass.name), index);
    EXPECT_NE(pass.run, nullptr);
    EXPECT_EQ(pass.empty_record.substr(0, 16), ",\"assumptions\":[");
    EXPECT_EQ(pass.empty_record.back(), '}');
  }

  EXPECT_FALSE(FindSsaPass("no_such_pass"));
  EXPECT_FALSE(FindSsaPass(""));
}

TEST(SsaRegistry, DefaultPipelinePublishesEveryPassInRegistryOrder) {
  auto budget = Plenty();
  const auto graph = Graph();
  const auto sources = Sources();
  const auto run = RunSsaPipeline(graph, {}, sources, {}, budget);
  ASSERT_TRUE(run.result);
  const auto& result = *run.result;
  const auto registry = SsaPassRegistry();
  ASSERT_EQ(result.stages.size(), registry.size());
  EXPECT_NE(result.body.find("\"pipeline\":{\"name\":\"default\""), std::string::npos);
  std::size_t position = 0;
  for (std::size_t index = 0; index < registry.size(); ++index) {
    EXPECT_EQ(result.stages[index].pass, index);
    const auto found =
        result.body.find("{\"pass\":\"" + std::string(registry[index].name) + "\"", position);
    ASSERT_NE(found, std::string::npos) << registry[index].name;
    position = found;
  }
}

TEST(SsaRegistry, NamedPipelineRunsTheSelectionInTheCallersOrder) {
  auto budget = Plenty();
  const auto graph = Graph();
  const auto sources = Sources();
  const std::array<std::size_t, 2> selection{Index("dead_effect_free_nodes"),
                                             Index("unreachable_blocks")};
  SsaDeclarations declared;
  declared.closed_entries = declared.return_leaves = true;
  const auto run = RunSsaPipeline(graph, declared, sources, {}, budget, selection);
  ASSERT_TRUE(run.result);
  ASSERT_EQ(run.result->stages.size(), 2U);
  EXPECT_EQ(run.result->stages[0].pass, selection[0]);
  EXPECT_EQ(run.result->stages[1].pass, selection[1]);
  EXPECT_EQ(run.result->stages[1].outcome, SsaStageOutcome::unchanged);
  EXPECT_NE(run.result->body.find("\"pipeline\":{\"name\":\"named\",\"rounds\":1,\"passes\":"
                                  "[\"dead_effect_free_nodes\",\"unreachable_blocks\"]}"),
            std::string::npos);
  EXPECT_EQ(run.result->body.find("\"pass\":\"linear_mba\""), std::string::npos);
  EXPECT_LT(run.result->body.find("\"pass\":\"dead_effect_free_nodes\""),
            run.result->body.find("\"pass\":\"unreachable_blocks\""));
}

TEST(SsaRegistry, RefusesAnUnregisteredOrRepeatedSelection) {
  const auto graph = Graph();
  const auto sources = Sources();
  const std::array<std::size_t, 2> repeated{0, 0};
  auto budget = Plenty();
  auto run = RunSsaPipeline(graph, {}, sources, {}, budget, repeated);
  EXPECT_EQ(run.reason, SsaPipelineDecline::invalid_selection);
  EXPECT_FALSE(run.result);
  const std::array<std::size_t, 1> outside{SsaPassRegistry().size()};
  run = RunSsaPipeline(graph, {}, sources, {}, budget, outside);
  EXPECT_EQ(run.reason, SsaPipelineDecline::invalid_selection);
  EXPECT_FALSE(run.result);
}

TEST(SsaRegistry, RequirementsGateEachPassWithTheFirstMissingDeclaration) {
  const auto graph = Graph();
  const auto sources = Sources();
  const auto registry = SsaPassRegistry();
  auto budget = Plenty();
  const auto none = RunSsaPipeline(graph, {}, sources, {}, budget);
  ASSERT_TRUE(none.result);
  for (std::size_t index = 0; index < registry.size(); ++index) {
    const auto& stage = none.result->stages[index];
    const auto requirements = registry[index].requirements;
    const char* expected = requirements & kNeedsClosedEntries  ? "no_closed_entry_declaration"
                           : requirements & kNeedsPrivateFrame ? "no_private_frame_declaration"
                                                               : nullptr;
    if (expected) {
      EXPECT_EQ(stage.outcome, SsaStageOutcome::not_run) << registry[index].name;
      EXPECT_EQ(stage.reason, expected) << registry[index].name;
      EXPECT_NE(none.result->body.find(Stage(registry[index].name, "not_run", expected)),
                std::string::npos)
          << registry[index].name;
    } else {
      EXPECT_NE(stage.outcome, SsaStageOutcome::not_run) << registry[index].name;
    }
  }

  // The graph returns, so closed entries alone still leave the return scope open.
  SsaDeclarations closed;
  closed.closed_entries = true;
  const auto open_return = RunSsaPipeline(graph, closed, sources, {}, budget);
  ASSERT_TRUE(open_return.result);
  EXPECT_EQ(open_return.result->stages[Index("unreachable_blocks")].reason,
            "no_return_leaves_declaration");
  EXPECT_EQ(open_return.result->stages[Index("bounded_table_loads")].reason,
            "no_image_access_declaration");
  closed.return_leaves = true;
  const auto scoped = RunSsaPipeline(graph, closed, sources, {}, budget);
  ASSERT_TRUE(scoped.result);
  EXPECT_NE(scoped.result->stages[Index("unreachable_blocks")].outcome, SsaStageOutcome::not_run);
  EXPECT_EQ(scoped.result->stages[Index("bounded_table_loads")].reason,
            "no_image_access_declaration");
  closed.image_access = true;
  const auto image = RunSsaPipeline(graph, closed, sources, {}, budget);
  ASSERT_TRUE(image.result);
  EXPECT_NE(image.result->stages[Index("bounded_table_loads")].outcome, SsaStageOutcome::not_run);
  EXPECT_EQ(image.result->stages[Index("frame_promotion")].reason, "no_private_frame_declaration");
  EXPECT_NE(image.result->head.find("\"image_access\":true"), std::string::npos);
}

TEST(SsaRegistry, BudgetExhaustionPublishesNoPartialRecord) {
  const auto graph = Graph();
  const auto sources = Sources();
  std::uint64_t work = 1;
  bool completed = false;

  // Every cut before completion must decline wholesale.
  for (; work < 1000000 && !completed; work = work * 3 / 2 + 1) {
    Budget budget({work, 10000000});
    const auto run = RunSsaPipeline(graph, {}, sources, {}, budget);
    if (run.result) {
      completed = true;
      EXPECT_EQ(run.reason, SsaPipelineDecline::none);
    } else {
      EXPECT_EQ(run.reason, SsaPipelineDecline::resource_limit);
    }
  }

  EXPECT_TRUE(completed);
  EXPECT_GT(work, 2U);
}

}  // namespace
}  // namespace nyx::passes
