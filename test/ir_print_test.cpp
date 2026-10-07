#include <limits>

#include <gtest/gtest.h>

#include "nyx/ir/print.hpp"

namespace nyx::ir {
namespace {

Group Fixture() {
  return Group(4660, {0x00, 0xab, 0xff},
               {{Op::read, 128, {}, 0, 100}, {Op::constant, 128, {}, 42}, {Op::add, 128, {0, 1}}},
               {{100, 2}});
}

TEST(IrPrint, EmitsDeterministicGenericSchema) {
  Budget budget({10000, 10000});
  const auto result = PrintJson(Fixture(), budget);
  ASSERT_TRUE(result.json);
  EXPECT_EQ(result.reason, PrintDecline::none);
  EXPECT_EQ(*result.json,
            "{\"schema\":1,\"kind\":\"nyx.ir.group\",\"source_address\":4660,\"bytes\":\"00abff\","
            "\"memory_model\":\"unspecified\",\"nodes\":[{\"id\":0,\"op\":\"read\",\"width\":128,"
            "\"produces_value\":true,\"inputs\":[],\"storage\":100},"
            "{\"id\":1,\"op\":\"constant\",\"width\":128,\"produces_value\":true,\"inputs\":[],"
            "\"immediate\":42},"
            "{\"id\":2,\"op\":\"add\",\"width\":128,\"produces_value\":true,\"inputs\":[0,1]}],"
            "\"writes\":[{\"storage\":100,\"value\":2}],\"transfer\":null}");
  EXPECT_EQ(budget.used().bytes, result.json->size() + 1);
  EXPECT_EQ(budget.used().work, 2 * result.json->size());
  Budget again({10000, 10000});
  EXPECT_EQ(PrintJson(Fixture(), again).json, result.json);
}

TEST(IrPrint, ExactAllocationAndWorkBoundaries) {
  Budget measure({10000, 10000});
  const auto result = PrintJson(Fixture(), measure);
  ASSERT_TRUE(result.json);
  const auto exact = measure.used();
  Budget byte_short({exact.work, exact.bytes - 1});
  const auto bytes = PrintJson(Fixture(), byte_short);
  EXPECT_FALSE(bytes.json);
  EXPECT_EQ(bytes.reason, PrintDecline::byte_limit);
  EXPECT_EQ(byte_short.used().bytes, 0);
  Budget work_short({exact.work - 1, exact.bytes});
  const auto work = PrintJson(Fixture(), work_short);
  EXPECT_FALSE(work.json);
  EXPECT_EQ(work.reason, PrintDecline::work_limit);
  EXPECT_EQ(work_short.used().bytes, 0);
  Budget sufficient(exact);
  EXPECT_EQ(PrintJson(Fixture(), sufficient).json, result.json);
}

TEST(IrPrint, RejectsUnknownOperationsAndMalformedReferences) {
  const Group bad[] = {
      Group(0, {}, {{static_cast<Op>(255), 64}}, {}), Group(0, {}, {{Op::constant, 0}}, {}),
      Group(0, {}, {{Op::bit_not, 64, {0}}}, {}), Group(0, {}, {{Op::constant, 64}}, {{0, 1}})};
  for (const auto& group : bad) {
    Budget budget({10000, 10000});
    const auto result = PrintJson(group, budget);
    EXPECT_FALSE(result.json);
    EXPECT_EQ(result.reason, PrintDecline::invalid_group);
    EXPECT_EQ(budget.used().bytes, 0);
  }

  EXPECT_EQ(Descriptor(static_cast<Op>(-1)), nullptr);
}

TEST(IrPrint, PreservesOrderedMemoryAndStorageEffects) {
  const Group group(0, {},
                    {{Op::constant, 64, {}, 4096},
                     {Op::load, 32, {0}, 0, 0, {ByteOrder::big, 4, true}},
                     {Op::write, 32, {1}, 0, 37},
                     {Op::store, 32, {0, 1}, 0, 0, {ByteOrder::little, 1}}},
                    {{99, 1}}, MemoryModel::atomic_scalar_reference);
  Budget budget({10000, 10000});
  const auto result = PrintJson(group, budget);
  ASSERT_TRUE(result.json);
  EXPECT_NE(
      result.json->find(
          "{\"id\":1,\"op\":\"load\",\"width\":32,\"produces_value\":true,\"inputs\":[0],"
          "\"access\":{\"byte_order\":\"big\",\"alignment\":4,\"decline_on_unaligned\":true}}"),
      std::string::npos);
  EXPECT_NE(result.json->find("{\"id\":2,\"op\":\"write\",\"width\":32,\"produces_value\":false,"
                              "\"inputs\":[1],\"storage\":37}"),
            std::string::npos);
  EXPECT_NE(
      result.json->find(
          "{\"id\":3,\"op\":\"store\",\"width\":32,\"produces_value\":false,\"inputs\":[0,1],"
          "\"access\":{\"byte_order\":\"little\",\"alignment\":1,\"decline_on_unaligned\":false}}"),
      std::string::npos);
  EXPECT_NE(result.json->find("\"writes\":[{\"storage\":99,\"value\":1}]"), std::string::npos);
}

TEST(IrPrint, RejectsEffectOnlyReferencesAndInvalidMemoryDescriptors) {
  const std::vector<Node> prefix{{Op::constant, 64, {}, 4096}, {Op::constant, 32, {}, 42}};
  const Node bad_nodes[] = {{Op::load, 32, {1}},
                            {Op::load, 7, {0}},
                            {Op::load, 256, {0}},
                            {Op::load, 32, {0}, 0, 0, {ByteOrder::little, 0}},
                            {Op::load, 32, {0}, 0, 0, {ByteOrder::little, 3}},
                            {Op::load, 32, {0}, 0, 0, {static_cast<ByteOrder>(99), 4}},
                            {Op::store, 64, {0, 1}},
                            {Op::write, 64, {1}, 0, 9}};
  for (const auto& bad : bad_nodes) {
    auto nodes = prefix;
    nodes.push_back(bad);
    Budget budget({10000, 10000});
    const auto result =
        PrintJson(Group(0, {}, std::move(nodes), {}, MemoryModel::atomic_scalar_reference), budget);
    EXPECT_FALSE(result.json);
    EXPECT_EQ(result.reason, PrintDecline::invalid_group);
    EXPECT_EQ(budget.used().bytes, 0);
  }

  for (const auto effect : {Op::store, Op::write}) {
    auto nodes = prefix;
    nodes.push_back(effect == Op::store ? Node{Op::store, 32, {0, 1}}
                                        : Node{Op::write, 32, {1}, 0, 9});
    Budget final_budget({10000, 10000});
    EXPECT_EQ(
        PrintJson(Group(0, {}, nodes, {{9, 2}}, MemoryModel::atomic_scalar_reference), final_budget)
            .reason,
        PrintDecline::invalid_group);
    nodes.push_back({Op::bit_not, 32, {2}});
    Budget input_budget({10000, 10000});
    EXPECT_EQ(PrintJson(Group(0, {}, std::move(nodes), {}, MemoryModel::atomic_scalar_reference),
                        input_budget)
                  .reason,
              PrintDecline::invalid_group);
  }
}

TEST(IrPrint, ImageAddressesAndMemoryModelsAreExplicit) {
  Budget budget({10000, 10000});
  const auto address = PrintJson(Group(0, {}, {{Op::image_address, 64, {}, 8192}}, {}), budget);
  ASSERT_TRUE(address.json);
  EXPECT_NE(address.json->find("\"op\":\"image_address\""), std::string::npos);
  EXPECT_NE(address.json->find("\"immediate\":8192"), std::string::npos);
  const Group bad[] = {
      Group(0, {}, {{Op::image_address, 32, {}, 8192}}, {}),
      Group(0, {}, {}, {}, static_cast<MemoryModel>(99)),
      Group(0, {}, {{Op::constant, 64}, {Op::load, 8, {0}}}, {}),
      Group(0, {}, {{Op::constant, 64}, {Op::constant, 8}, {Op::store, 8, {0, 1}}}, {})};
  for (const auto& group : bad) {
    Budget check({10000, 10000});
    EXPECT_EQ(PrintJson(group, check).reason, PrintDecline::invalid_group);
  }

  Budget tagged_budget({10000, 10000});
  const auto tagged = PrintJson(Group(0, {}, {{Op::constant, 64}, {Op::load, 8, {0}}}, {},
                                      MemoryModel::atomic_scalar_reference),
                                tagged_budget);
  ASSERT_TRUE(tagged.json);
  EXPECT_NE(tagged.json->find("\"memory_model\":\"atomic_scalar_reference\""), std::string::npos);
}

TEST(IrPrint, SerializesEveryTransferKindWithExplicitReferences) {
  const std::vector<Node> nodes{
      {Op::constant, 64, {}, 4096}, {Op::constant, 64, {}, 8192}, {Op::constant, 1, {}, 1}};
  const Transfer transfers[] = {{TransferKind::jump, 0, {}, {}, {}},
                                {TransferKind::conditional, 0, 2, 1, {}},
                                {TransferKind::call, 0, {}, {}, 1},
                                {TransferKind::return_, 0, {}, {}, {}}};
  const char* expected[] = {
      "\"transfer\":{\"kind\":\"jump\",\"target\":0}",
      "\"transfer\":{\"kind\":\"conditional\",\"target\":0,\"condition\":2,\"alternative\":1}",
      "\"transfer\":{\"kind\":\"call\",\"target\":0,\"continuation\":1}",
      "\"transfer\":{\"kind\":\"return\",\"target\":0}"};
  for (unsigned i = 0; i < 4; ++i) {
    const Group group(0, {}, nodes, {}, MemoryModel::unspecified, transfers[i]);
    Budget budget({10000, 10000});
    const auto result = PrintJson(group, budget);
    ASSERT_TRUE(result.json);
    EXPECT_NE(result.json->find(expected[i]), std::string::npos);
    EXPECT_EQ(budget.used().work, 2 * result.json->size());
    EXPECT_EQ(budget.used().bytes, result.json->size() + 1);
    Budget exact(budget.used());
    EXPECT_EQ(PrintJson(group, exact).json, result.json);
    Budget short_bytes({budget.used().work, budget.used().bytes - 1});
    const auto failed = PrintJson(group, short_bytes);
    EXPECT_FALSE(failed.json);
    EXPECT_EQ(failed.reason, PrintDecline::byte_limit);
    EXPECT_EQ(short_bytes.used().bytes, 0);
  }
}

TEST(IrPrint, RejectsMalformedTransferShapesAndWidths) {
  const std::vector<Node> nodes{
      {Op::constant, 64, {}, 4096}, {Op::constant, 64, {}, 8192}, {Op::constant, 1, {}, 1}};
  const Transfer invalid[] = {{static_cast<TransferKind>(99), 0, {}, {}, {}},
                              {TransferKind::jump, 3, {}, {}, {}},
                              {TransferKind::jump, 2, {}, {}, {}},
                              {TransferKind::jump, 0, 2, {}, {}},
                              {TransferKind::jump, 0, {}, 1, {}},
                              {TransferKind::jump, 0, {}, {}, 1},
                              {TransferKind::return_, 0, 2, {}, {}},
                              {TransferKind::conditional, 0, {}, 1, {}},
                              {TransferKind::conditional, 0, 2, {}, {}},
                              {TransferKind::conditional, 0, 2, 1, 1},
                              {TransferKind::conditional, 0, 0, 1, {}},
                              {TransferKind::conditional, 0, 2, 2, {}},
                              {TransferKind::conditional, 0, 3, 1, {}},
                              {TransferKind::conditional, 0, 2, 3, {}},
                              {TransferKind::call, 0, {}, {}, {}},
                              {TransferKind::call, 0, {}, {}, 2},
                              {TransferKind::call, 0, {}, {}, 3},
                              {TransferKind::call, 0, 2, {}, 1},
                              {TransferKind::call, 0, {}, 1, 1}};
  for (const auto& transfer : invalid) {
    Budget budget({10000, 10000});
    const auto result =
        PrintJson(Group(0, {}, nodes, {}, MemoryModel::unspecified, transfer), budget);
    EXPECT_FALSE(result.json);
    EXPECT_EQ(result.reason, PrintDecline::invalid_group);
    EXPECT_EQ(budget.used().bytes, 0);
  }
}

TEST(IrPrint, RejectsTransferReferencesToEffectsAndUnknownOperations) {
  for (const auto op : {Op::write, static_cast<Op>(255)}) {
    const std::vector<Node> nodes{
        {Op::constant, 64, {}, 4096}, {Op::constant, 1, {}, 1}, {op, 64, {0}, 0, 42}};
    const Transfer invalid[] = {{TransferKind::jump, 2, {}, {}, {}},
                                {TransferKind::conditional, 0, 1, 2, {}},
                                {TransferKind::call, 0, {}, {}, 2}};
    for (const auto& transfer : invalid) {
      Budget budget({10000, 10000});
      EXPECT_EQ(
          PrintJson(Group(0, {}, nodes, {}, MemoryModel::unspecified, transfer), budget).reason,
          PrintDecline::invalid_group);
      EXPECT_EQ(budget.used().bytes, 0);
    }
  }

  const std::vector<Node> nodes{
      {Op::constant, 64, {}, 4096}, {Op::constant, 1, {}, 1}, {Op::write, 1, {1}, 0, 42}};
  Budget budget({10000, 10000});
  const Transfer invalid{TransferKind::conditional, 0, 2, 0, {}};
  EXPECT_EQ(PrintJson(Group(0, {}, nodes, {}, MemoryModel::unspecified, invalid), budget).reason,
            PrintDecline::invalid_group);
}

TEST(IrPrint, EmptySourceAndMaximumUnsignedLiteralAreExplicit) {
  Budget budget({10000, 10000});
  const Group group(std::numeric_limits<std::uint64_t>::max(), {},
                    {{Op::constant, 4096, {}, std::numeric_limits<std::uint64_t>::max()}}, {});
  const auto result = PrintJson(group, budget);
  ASSERT_TRUE(result.json);
  EXPECT_NE(result.json->find("\"source_address\":18446744073709551615"), std::string::npos);
  EXPECT_NE(result.json->find("\"immediate\":18446744073709551615"), std::string::npos);
  EXPECT_NE(result.json->find("\"width\":4096"), std::string::npos);
  EXPECT_NE(result.json->find("\"bytes\":\"\""), std::string::npos);
}

TEST(IrPrint, OperationDescriptorsCoverTheCurrentVocabulary) {
  for (unsigned op = 0; op <= static_cast<unsigned>(Op::image_address); ++op) {
    const auto* descriptor = Descriptor(static_cast<Op>(op));
    ASSERT_NE(descriptor, nullptr);
    EXPECT_FALSE(descriptor->name.empty());
    EXPECT_LE(descriptor->arity, 3);
    for (unsigned previous = 0; previous < op; ++previous) {
      EXPECT_NE(descriptor->name, Descriptor(static_cast<Op>(previous))->name);
    }
  }

  EXPECT_EQ(Descriptor(Op::select)->arity, 3);
  EXPECT_EQ(Descriptor(Op::read)->arity, 0);
}

Block BlockFixture() {
  const std::vector<Group> sources{
      {0x100, {0x00, 0xab}, {{Op::constant, 64, {}, 42}}, {{100, 0}}},
      {0x102,
       {0xff},
       {{Op::read, 64, {}, 0, 100}, {Op::constant, 64, {}, 9}, {Op::add, 64, {0, 1}}},
       {{200, 2}},
       MemoryModel::unspecified,
       Transfer{TransferKind::call, 0, {}, {}, 2}}};
  Budget budget({UINT64_MAX, UINT64_MAX});
  auto normalized = Normalize(sources, budget);
  return std::move(*normalized.block);
}

TEST(IrPrint, BlockPreservesImmutableSourcesAndGlobalBoundaryReferences) {
  const auto normalized = BlockFixture();
  auto nodes = std::vector<Node>(normalized.nodes().begin(), normalized.nodes().end());
  nodes[1].immediate = 11;
  const Block block({normalized.sources().begin(), normalized.sources().end()}, std::move(nodes),
                    {normalized.origins().begin(), normalized.origins().end()},
                    {normalized.boundaries().begin(), normalized.boundaries().end()}, 7);
  Budget budget({100000, 100000});
  const auto printed = PrintJson(block, budget);
  ASSERT_TRUE(printed.json);
  EXPECT_TRUE(
      printed.json->starts_with("{\"schema\":1,\"kind\":\"nyx.ir.block\",\"revision\":7,"
                                "\"entry_scope\":\"single_entry_straight_line\",\"sources\":["));
  Budget source_budget({100000, 100000});
  const auto source0 = PrintJson(block.sources()[0], source_budget);
  const auto source1 = PrintJson(block.sources()[1], source_budget);
  ASSERT_TRUE(source0.json);
  ASSERT_TRUE(source1.json);
  EXPECT_NE(printed.json->find("\"sources\":[" + *source0.json + "," + *source1.json + "]"),
            std::string::npos);
  EXPECT_NE(printed.json->find("\"source_address\":256,\"bytes\":\"00ab\""), std::string::npos);
  EXPECT_NE(printed.json->find("\"source_address\":258,\"bytes\":\"ff\""), std::string::npos);
  EXPECT_NE(printed.json->find(
                "{\"id\":1,\"op\":\"constant\",\"width\":64,\"produces_value\":true,\"inputs\":[],"
                "\"immediate\":11,\"origin\":{\"boundary\":1,\"operation\":1}}"),
            std::string::npos);
  EXPECT_NE(printed.json->find(
                "{\"id\":2,\"op\":\"add\",\"width\":64,\"produces_value\":true,\"inputs\":[0,1],"
                "\"origin\":{\"boundary\":1,\"operation\":2}}"),
            std::string::npos);
  EXPECT_TRUE(printed.json->ends_with(
      "\"boundaries\":[{\"first_node\":0,\"node_count\":1,\"writes\":[{\"storage\":100,\"value\":0}"
      "],\"transfer\":null},"
      "{\"first_node\":1,\"node_count\":2,\"writes\":[{\"storage\":200,\"value\":2}],"
      "\"transfer\":{\"kind\":\"call\",\"target\":0,\"continuation\":2}}]}"));
  EXPECT_EQ(budget.used().bytes, printed.json->size() + 1);
}

TEST(IrPrint, BlockRejectsMalformedOriginsBeforeAllocation) {
  const auto valid = BlockFixture();
  for (unsigned mutation = 0; mutation < 4; ++mutation) {
    auto origins = std::vector<Origin>(valid.origins().begin(), valid.origins().end());
    if (mutation == 0) origins[1].boundary = 99;
    if (mutation == 1) origins[1].operation = 99;
    if (mutation == 2) origins[2].operation = origins[1].operation;
    if (mutation == 3) origins.pop_back();
    const Block malformed({valid.sources().begin(), valid.sources().end()},
                          {valid.nodes().begin(), valid.nodes().end()}, std::move(origins),
                          {valid.boundaries().begin(), valid.boundaries().end()});
    Budget budget({100000, 100000});
    const auto result = PrintJson(malformed, budget);
    EXPECT_FALSE(result.json);
    EXPECT_EQ(result.reason, PrintDecline::invalid_group);
    EXPECT_EQ(budget.used().bytes, 0);
  }
}

TEST(IrPrint, BlockDeclinesEveryResourceCutWithoutPartialOutputOrAllocation) {
  const auto block = BlockFixture();
  Budget full({100000, 100000});
  const auto expected = PrintJson(block, full);
  ASSERT_TRUE(expected.json);
  for (bool bytes : {false, true}) {
    const auto maximum = bytes ? full.used().bytes : full.used().work;
    for (std::uint64_t cut = 0; cut < maximum; ++cut) {
      Budget budget({bytes ? full.used().work : cut, bytes ? cut : full.used().bytes});
      const auto result = PrintJson(block, budget);
      ASSERT_FALSE(result.json) << cut;
      EXPECT_EQ(result.reason, bytes ? PrintDecline::byte_limit : PrintDecline::work_limit);
      EXPECT_EQ(budget.used().bytes, 0);
    }
  }

  Budget exact(full.used());
  EXPECT_EQ(PrintJson(block, exact).json, expected.json);
}

TEST(IrPrint, RecoveredTransfersRetainWitnessesAndRefuseInvalidOrExhaustedPublication) {
  const Group source(0x100, {1, 2, 3, 4},
                     {{Op::constant, 1, {}, 2},
                      {Op::image_address, 64, {}, 0x200},
                      {Op::image_address, 64, {}, 0x300}},
                     {}, MemoryModel::unspecified,
                     Transfer{TransferKind::conditional, 1, 0, 2, {}});
  Budget setup({100000, 100000});
  auto normalized = NormalizePath(std::span(&source, 1), setup);
  ASSERT_TRUE(normalized.path);
  const auto original = *normalized.path->boundaries()[0].transfer;
  const ConditionalRewrite witness{RewriteRule::folded_condition,
                                   0,
                                   original,
                                   {TransferKind::jump, 2, {}, {}, {}},
                                   0,
                                   false,
                                   0,
                                   0,
                                   {},
                                   0,
                                   1};
  const RecoveredPath path(*normalized.path, {witness}, 1);
  Budget measure({100000, 100000});
  auto printed = PrintJson(path, measure);
  ASSERT_TRUE(printed.json);
  EXPECT_TRUE(
      printed.json->starts_with("{\"schema\":1,\"kind\":\"nyx.ir.recovered_path\",\"revision\":1"));
  EXPECT_NE(printed.json->find("\"transfer\":{\"kind\":\"jump\",\"target\":2}"), std::string::npos);
  EXPECT_NE(printed.json->find(
                "\"basis_revision\":0,\"control_rewrites\":[{\"rule\":\"folded_condition\","
                "\"boundary\":0,\"condition\":0,\"condition_value\":false"),
            std::string::npos);
  EXPECT_TRUE(printed.json->ends_with(
      "\"destinations\":[],\"store_omissions\":[],\"paired_load_omissions\":[]}"));
  for (bool bytes : {false, true}) {
    const auto maximum = bytes ? measure.used().bytes : measure.used().work;
    for (std::uint64_t cut = 0; cut < maximum; ++cut) {
      Budget budget({bytes ? measure.used().work : cut, bytes ? cut : measure.used().bytes});
      EXPECT_FALSE(PrintJson(path, budget).json) << cut;
    }
  }

  Budget exact(measure.used());
  EXPECT_EQ(PrintJson(path, exact).json, printed.json);
  auto wrong = witness;
  wrong.replacement.target = 1;
  Budget invalid({100000, 100000});
  auto refused = PrintJson(RecoveredPath(*normalized.path, {wrong}, 1), invalid);
  EXPECT_FALSE(refused.json);
  EXPECT_EQ(refused.reason, PrintDecline::invalid_group);
}

}  // namespace
}  // namespace nyx::ir
