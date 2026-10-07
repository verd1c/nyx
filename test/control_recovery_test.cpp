#include <gtest/gtest.h>

#include "nyx/eval/path.hpp"
#include "nyx/ir/print.hpp"
#include "nyx/recovery/control.hpp"
#include "nyx/recovery/image.hpp"

namespace nyx::recovery {
namespace {

Budget Plenty() { return Budget({UINT64_MAX, UINT64_MAX}); }

// A flattened dispatch: a choice between two table offsets, the halfword read
// that choice indexes, and a jump to `code + 4 * entry`. The condition is a
// runtime value, so nothing settles until one arm is pinned, and the read
// cannot be folded without duplicating the access.
constexpr std::uint64_t kTable = 0x3000, kCode = 0x1000;

ir::Group DispatchGroup(std::uint64_t when_true = 2, std::uint64_t when_false = 0) {
  return ir::Group(0x100, {1, 2, 3, 4},
                   {{ir::Op::read, 1, {}, 0, 7},
                    {ir::Op::constant, 64, {}, when_true},
                    {ir::Op::constant, 64, {}, when_false},
                    {ir::Op::select, 64, {0, 1, 2}},
                    {ir::Op::image_address, 64, {}, kTable},
                    {ir::Op::add, 64, {4, 3}},
                    {ir::Op::load, 16, {5}},
                    {ir::Op::zext, 64, {6}},
                    {ir::Op::constant, 64, {}, 4},
                    {ir::Op::mul, 64, {7, 8}},
                    {ir::Op::image_address, 64, {}, kCode},
                    {ir::Op::add, 64, {10, 9}}},
                   {}, ir::MemoryModel::atomic_scalar_reference,
                   ir::Transfer{ir::TransferKind::jump, 11, {}, {}, {}});
}

// Entry 0 sends the jump to kCode + 20, entry 1 to kCode + 36.
constexpr std::array<std::uint8_t, 4> kTableBytes{5, 0, 9, 0};
constexpr std::uint64_t kWhenTrue = kCode + 36, kWhenFalse = kCode + 20;

ir::ImageFacts Declared(std::span<const ir::ConstantImageRange> ranges) {
  return {ranges, {}, false, {}};
}

eval::State Condition(bool taken) {
  auto budget = Plenty();
  eval::State state;
  state.cells.push_back({7, std::move(*BitVector::from_u64(1, taken ? 1 : 0, 64, budget))});
  return state;
}

// The same bytes the declaration names, at the placement execution uses.
eval::Memory Mapped(std::uint64_t bias) {
  auto budget = Plenty();
  const eval::RegionInput input[] = {{bias + kTable, kTableBytes}};
  return std::move(*eval::Memory::Create(input, budget).memory);
}

ir::Group Branch(std::uint64_t address, std::uint64_t literal, ir::Op op = ir::Op::constant) {
  return ir::Group(address, {1, 2, 3, 4},
                   {{op, 1, {}, literal, 7},
                    {ir::Op::image_address, 64, {}, address + 0x100},
                    {ir::Op::image_address, 64, {}, address + 0x200}},
                   {{7, 0}}, ir::MemoryModel::unspecified,
                   ir::Transfer{ir::TransferKind::conditional, 1, 0, 2, {}});
}

ir::Path Path(std::vector<ir::Group> sources, std::uint64_t revision = 0) {
  auto budget = Plenty();
  auto normalized = ir::NormalizePath(sources, budget);
  EXPECT_TRUE(normalized.path);
  const auto& path = *normalized.path;
  return ir::Path({path.sources().begin(), path.sources().end()},
                  {path.nodes().begin(), path.nodes().end()},
                  {path.origins().begin(), path.origins().end()},
                  {path.boundaries().begin(), path.boundaries().end()}, revision);
}

TEST(ControlRecovery, ChoosesActualLowBitArmAndKeepsEntireBasis) {
  for (const auto literal : {UINT64_C(0), UINT64_C(1), UINT64_C(2), UINT64_C(3), UINT64_MAX}) {
    auto path = Path({Branch(0x100, literal)}, 12);
    auto budget = Plenty();
    const auto before = ir::PrintJson(path, budget);
    ASSERT_TRUE(before.json);
    auto result = RecoverControl(std::move(path), {}, budget);
    ASSERT_TRUE(result.path);
    EXPECT_EQ(result.reason, ControlRecoveryDecline::none);
    EXPECT_EQ(result.path->revision(), 13);
    EXPECT_EQ(result.path->basis().revision(), 12);
    ASSERT_EQ(result.path->rewrites().size(), 1);
    const auto& edit = result.path->rewrites()[0];
    EXPECT_EQ(edit.boundary, 0);
    EXPECT_EQ(edit.condition_value, (literal & 1) != 0);
    EXPECT_EQ(edit.from_revision, 12);
    EXPECT_EQ(edit.to_revision, 13);
    EXPECT_EQ(edit.original.kind, ir::TransferKind::conditional);
    EXPECT_EQ(edit.replacement.kind, ir::TransferKind::jump);
    EXPECT_EQ(edit.replacement.target,
              (literal & 1) ? edit.original.target : *edit.original.alternative);
    EXPECT_FALSE(edit.replacement.condition);
    EXPECT_FALSE(edit.replacement.alternative);
    EXPECT_FALSE(edit.replacement.continuation);
    const auto effective = result.path->effective_transfer(0);
    ASSERT_TRUE(effective);
    EXPECT_EQ(effective->kind, ir::TransferKind::jump);
    EXPECT_EQ(effective->target, edit.replacement.target);
    const auto after = ir::PrintJson(result.path->basis(), budget);
    ASSERT_TRUE(after.json);
    EXPECT_EQ(*before.json, *after.json);
  }
}

TEST(ControlRecovery, LiteralBatchesHaveSortedUniqueBoundariesAndOneRevision) {
  auto path = Path({Branch(0x100, 1), Branch(0x200, 0), Branch(0x300, 1)}, 90);
  auto budget = Plenty();
  auto result = RecoverControl(std::move(path), {}, budget);
  ASSERT_TRUE(result.path);
  ASSERT_EQ(result.path->rewrites().size(), 3);
  for (std::size_t i = 0; i < 3; ++i) {
    const auto& edit = result.path->rewrites()[i];
    EXPECT_EQ(edit.boundary, i);
    EXPECT_EQ(edit.from_revision, 90);
    EXPECT_EQ(edit.to_revision, 91);
  }

  EXPECT_EQ(ir::ValidateRecoveredPath(*result.path, budget), ir::BlockDecline::none);
}

TEST(ControlRecovery, UnknownReadAndNonliteralConstantExpressionAreNotInferred) {
  auto expression = ir::Group(0x200, {1, 2, 3, 4},
                              {{ir::Op::constant, 1, {}, 0},
                               {ir::Op::bit_not, 1, {0}},
                               {ir::Op::image_address, 64, {}, 0x300},
                               {ir::Op::image_address, 64, {}, 0x400}},
                              {}, ir::MemoryModel::unspecified,
                              ir::Transfer{ir::TransferKind::conditional, 2, 1, 3, {}});
  auto path = Path({Branch(0x100, 0, ir::Op::read), std::move(expression)}, UINT64_MAX);
  auto budget = Plenty();
  auto result = RecoverControl(std::move(path), {}, budget);
  ASSERT_TRUE(result.path);
  EXPECT_TRUE(result.path->rewrites().empty());
  EXPECT_EQ(result.path->revision(), UINT64_MAX);
  EXPECT_EQ(result.path->effective_transfer(0)->kind, ir::TransferKind::conditional);
  EXPECT_EQ(result.path->effective_transfer(1)->kind, ir::TransferKind::conditional);
}

TEST(ControlRecovery, RuntimeLoadPredicateRemainsConditional) {
  auto source = ir::Group(0x100, {1, 2, 3, 4},
                          {{ir::Op::constant, 64, {}, 0x800},
                           {ir::Op::load, 8, {0}},
                           {ir::Op::extract, 1, {1}, 0},
                           {ir::Op::image_address, 64, {}, 0x200},
                           {ir::Op::image_address, 64, {}, 0x300}},
                          {}, ir::MemoryModel::atomic_scalar_reference,
                          ir::Transfer{ir::TransferKind::conditional, 3, 2, 4, {}});
  auto path = Path({std::move(source)});
  auto budget = Plenty();
  auto result = RecoverControl(std::move(path), {}, budget);
  ASSERT_TRUE(result.path);
  EXPECT_TRUE(result.path->rewrites().empty());
  EXPECT_EQ(result.path->basis().nodes()[1].op, ir::Op::load);
  EXPECT_EQ(result.path->effective_transfer(0)->kind, ir::TransferKind::conditional);
}

TEST(ControlRecovery, ConstantBranchRetainsEarlierFaultingEffectsAndPhysicalWrites) {
  auto source = ir::Group(0x100, {1, 2, 3, 4},
                          {{ir::Op::constant, 64, {}, 0x800},
                           {ir::Op::load, 8, {0}},
                           {ir::Op::constant, 1, {}, 0},
                           {ir::Op::image_address, 64, {}, 0x200},
                           {ir::Op::image_address, 64, {}, 0x300},
                           {ir::Op::write, 8, {1}, 0, 19}},
                          {{20, 1}}, ir::MemoryModel::atomic_scalar_reference,
                          ir::Transfer{ir::TransferKind::conditional, 3, 2, 4, {}});
  auto path = Path({std::move(source)});
  auto budget = Plenty();
  const auto before = ir::PrintJson(path, budget);
  auto result = RecoverControl(std::move(path), {}, budget);
  ASSERT_TRUE(result.path);
  ASSERT_EQ(result.path->rewrites().size(), 1);
  EXPECT_EQ(ir::PrintJson(result.path->basis(), budget).json, before.json);
}

TEST(ControlRecovery, InvalidPredicateWidthAndRevisionOverflowPublishNothing) {
  auto path = Path({Branch(0x100, 1)});
  auto nodes = std::vector<ir::Node>(path.nodes().begin(), path.nodes().end());
  nodes[*path.boundaries()[0].transfer->condition].width = 2;
  auto invalid = ir::Path({path.sources().begin(), path.sources().end()}, std::move(nodes),
                          {path.origins().begin(), path.origins().end()},
                          {path.boundaries().begin(), path.boundaries().end()});
  auto budget = Plenty();
  auto result = RecoverControl(std::move(invalid), {}, budget);
  EXPECT_FALSE(result.path);
  EXPECT_EQ(result.reason, ControlRecoveryDecline::invalid_ir);
  auto overflow = Path({Branch(0x100, 1)}, UINT64_MAX);
  result = RecoverControl(std::move(overflow), {}, budget);
  EXPECT_FALSE(result.path);
  EXPECT_EQ(result.reason, ControlRecoveryDecline::revision_overflow);
}

TEST(ControlRecovery, EveryBudgetCutPublishesNoPartialRecoveredPath) {
  const auto path = Path({Branch(0x100, 1), Branch(0x200, 0)});
  auto budget = Plenty();
  auto copy = path;
  ASSERT_TRUE(RecoverControl(std::move(copy), {}, budget).path);
  const auto required = budget.used();
  for (std::uint64_t work = 0; work < required.work; ++work) {
    Budget limited({work, UINT64_MAX});
    copy = path;
    const auto result = RecoverControl(std::move(copy), {}, limited);
    ASSERT_FALSE(result.path);
    ASSERT_EQ(result.reason, ControlRecoveryDecline::resource_limit);
  }

  for (std::uint64_t bytes = 0; bytes < required.bytes; ++bytes) {
    Budget limited({UINT64_MAX, bytes});
    copy = path;
    const auto result = RecoverControl(std::move(copy), {}, limited);
    ASSERT_FALSE(result.path);
    ASSERT_EQ(result.reason, ControlRecoveryDecline::resource_limit);
  }

  Budget exact(required);
  copy = path;
  EXPECT_TRUE(RecoverControl(std::move(copy), {}, exact).path);
  auto limits = ControlRecoveryLimits{};
  limits.max_edits = 1;
  copy = path;
  EXPECT_EQ(RecoverControl(std::move(copy), {}, budget, limits).reason,
            ControlRecoveryDecline::resource_limit);
}

TEST(ControlRecovery, DispatchBranchNamesBothDestinationsAndEveryByteItRead) {
  const ir::ConstantImageRange ranges[] = {{kTable, kTableBytes}};
  const auto facts = Declared(ranges);
  auto path = Path({DispatchGroup()}, 7);
  auto budget = Plenty();
  const auto before = ir::PrintJson(path, budget);
  ASSERT_TRUE(before.json);
  auto result = RecoverControl(std::move(path), facts, budget);
  ASSERT_TRUE(result.path);
  EXPECT_EQ(result.reason, ControlRecoveryDecline::none);
  EXPECT_EQ(result.path->revision(), 8);
  EXPECT_EQ(result.path->basis().revision(), 7);
  ASSERT_EQ(result.path->rewrites().size(), 1);
  const auto& edit = result.path->rewrites()[0];
  EXPECT_EQ(edit.rule, ir::RewriteRule::dispatch_branch);
  EXPECT_EQ(edit.boundary, 0);
  EXPECT_FALSE(edit.condition_value);
  EXPECT_EQ(edit.from_revision, 7);
  EXPECT_EQ(edit.to_revision, 8);
  EXPECT_EQ(result.path->basis().nodes()[edit.condition].op, ir::Op::read);
  EXPECT_EQ(result.path->basis().nodes()[edit.condition].width, 1);
  EXPECT_EQ(edit.when_true, kWhenTrue);
  EXPECT_EQ(edit.when_false, kWhenFalse);
  EXPECT_EQ(edit.original.kind, ir::TransferKind::jump);
  const auto first = result.path->first_destination();
  EXPECT_EQ(first, result.path->basis().nodes().size());
  EXPECT_EQ(edit.replacement.kind, ir::TransferKind::conditional);
  EXPECT_EQ(edit.replacement.target, first);
  EXPECT_EQ(edit.replacement.condition, edit.condition);
  EXPECT_EQ(edit.replacement.alternative, first + 1);
  EXPECT_FALSE(edit.replacement.continuation);
  ASSERT_EQ(result.path->destinations().size(), 2);
  EXPECT_EQ(result.path->destinations()[0].op, ir::Op::image_address);
  EXPECT_EQ(result.path->destinations()[0].immediate, kWhenTrue);
  EXPECT_EQ(result.path->destinations()[1].immediate, kWhenFalse);

  // One read per arm, each naming the entry that arm indexes.
  ASSERT_EQ(edit.witness.size(), 2);
  EXPECT_EQ(result.path->basis().nodes()[edit.witness[0].node].op, ir::Op::load);
  EXPECT_EQ(edit.witness[0].node, edit.witness[1].node);
  EXPECT_TRUE(edit.witness[0].when);
  EXPECT_EQ(edit.witness[0].address, kTable + 2);
  EXPECT_EQ(edit.witness[0].width, 16);
  EXPECT_EQ(edit.witness[0].value, 9);
  EXPECT_FALSE(edit.witness[0].relocated);
  EXPECT_FALSE(edit.witness[1].when);
  EXPECT_EQ(edit.witness[1].address, kTable);
  EXPECT_EQ(edit.witness[1].value, 5);

  // The read stays in the path and the basis is byte-for-byte what it was.
  EXPECT_EQ(result.path->basis().nodes()[6].op, ir::Op::load);
  EXPECT_EQ(result.path->basis().boundaries()[0].transfer->kind, ir::TransferKind::jump);
  EXPECT_EQ(ir::PrintJson(result.path->basis(), budget).json, before.json);
  EXPECT_EQ(ir::ValidateRecoveredPath(*result.path, budget, {}, facts), ir::BlockDecline::none);
}

TEST(ControlRecovery, EachArmReachesItsNamedDestinationAndStillPerformsTheTableRead) {
  const ir::ConstantImageRange ranges[] = {{kTable, kTableBytes}};
  const auto facts = Declared(ranges);
  constexpr std::uint64_t bias = 0x40000;
  for (const bool taken : {false, true}) {
    auto budget = Plenty();
    auto result = RecoverControl(Path({DispatchGroup()}), facts, budget);
    ASSERT_TRUE(result.path);
    ASSERT_EQ(result.path->rewrites().size(), 1);
    auto original_state = Condition(taken), recovered_state = Condition(taken);
    auto original_memory = Mapped(bias), recovered_memory = Mapped(bias);
    const auto original = eval::ExecutePath(result.path->basis(), original_state, original_memory,
                                            budget, {}, {bias});
    const auto actual = eval::ExecuteRecoveredPath(*result.path, recovered_state, recovered_memory,
                                                   budget, {}, {bias}, facts);
    EXPECT_EQ(original.outcome, eval::Outcome::completed);
    EXPECT_EQ(actual.outcome, eval::Outcome::completed);
    ASSERT_TRUE(actual.runtime_next);
    EXPECT_EQ(actual.runtime_next, original.runtime_next);
    EXPECT_EQ(*actual.runtime_next, bias + (taken ? kWhenTrue : kWhenFalse));

    // Recovering the branch does not remove the access that resolved it.
    ASSERT_EQ(actual.trace.size(), 1);
    ASSERT_EQ(actual.trace[0].events.size(), 1);
    EXPECT_EQ(actual.trace[0].events[0].address, original.trace[0].events[0].address);
    EXPECT_EQ(actual.trace[0].events[0].bytes, original.trace[0].events[0].bytes);
    ASSERT_TRUE(actual.trace[0].transfer);
    EXPECT_EQ(actual.trace[0].transfer->kind, ir::TransferKind::conditional);
    EXPECT_EQ(actual.trace[0].transfer->condition, taken);
    EXPECT_EQ(original.trace[0].transfer->kind, ir::TransferKind::jump);
  }
}

TEST(ControlRecovery, SwappedSuccessorsAndAlteredWitnessesAreRefused) {
  const ir::ConstantImageRange ranges[] = {{kTable, kTableBytes}};
  const auto facts = Declared(ranges);
  auto budget = Plenty();
  auto result = RecoverControl(Path({DispatchGroup()}), facts, budget);
  ASSERT_TRUE(result.path);
  const auto& basis = result.path->basis();
  const auto revision = result.path->revision();
  const std::vector<ir::ConditionalRewrite> accepted(result.path->rewrites().begin(),
                                                     result.path->rewrites().end());
  const std::vector<ir::Node> named(result.path->destinations().begin(),
                                    result.path->destinations().end());
  const auto refuses = [&](std::vector<ir::ConditionalRewrite> rewrites,
                           std::vector<ir::Node> nodes) {
    const ir::RecoveredPath mutant(basis, std::move(rewrites), revision, std::move(nodes));
    return ir::ValidateRecoveredPath(mutant, budget, {}, facts);
  };

  EXPECT_EQ(refuses(accepted, named), ir::BlockDecline::none);
  {
    // The mandatory refuting mutation: the condition selects the other way.
    auto rewrites = accepted;
    auto nodes = named;
    std::swap(rewrites[0].when_true, rewrites[0].when_false);
    std::swap(nodes[0].immediate, nodes[1].immediate);
    EXPECT_EQ(refuses(std::move(rewrites), std::move(nodes)), ir::BlockDecline::invalid_ir);
  }

  {
    // The same swap made only in the transfer, leaving the record alone.
    auto rewrites = accepted;
    std::swap(rewrites[0].replacement.target, *rewrites[0].replacement.alternative);
    EXPECT_EQ(refuses(std::move(rewrites), named), ir::BlockDecline::invalid_ir);
  }

  {
    // A destination node that does not hold the address the record claims.
    auto nodes = named;
    nodes[0].immediate = kWhenFalse;
    EXPECT_EQ(refuses(accepted, std::move(nodes)), ir::BlockDecline::invalid_ir);
  }

  for (std::size_t read = 0; read < accepted[0].witness.size(); ++read) {
    auto rewrites = accepted;
    rewrites[0].witness[read].value ^= 1;
    EXPECT_EQ(refuses(rewrites, named), ir::BlockDecline::invalid_ir);
    rewrites = accepted;
    rewrites[0].witness[read].when = !rewrites[0].witness[read].when;
    EXPECT_EQ(refuses(rewrites, named), ir::BlockDecline::invalid_ir);
    rewrites = accepted;
    rewrites[0].witness.erase(rewrites[0].witness.begin() + static_cast<std::ptrdiff_t>(read));
    EXPECT_EQ(refuses(std::move(rewrites), named), ir::BlockDecline::invalid_ir);
  }

  {
    // A record claiming the other rule for the same replacement.
    auto rewrites = accepted;
    rewrites[0].rule = ir::RewriteRule::folded_condition;
    EXPECT_EQ(refuses(std::move(rewrites), named), ir::BlockDecline::invalid_ir);
  }

  // Without the declaration nothing can recheck the record, so nothing accepts it.
  EXPECT_EQ(ir::ValidateRecoveredPath(*result.path, budget), ir::BlockDecline::invalid_ir);
  EXPECT_FALSE(ir::PrintJson(*result.path, budget).json);
  auto state = Condition(true);
  auto memory = Mapped(0);
  EXPECT_EQ(eval::ExecuteRecoveredPath(*result.path, state, memory, budget, {}, {0}).outcome,
            eval::Outcome::invalid_group);
}

TEST(ControlRecovery, UndeclaredBytesEqualArmsASecondChoiceAndRelocatedSlotsRecoverNothing) {
  const ir::ConstantImageRange ranges[] = {{kTable, kTableBytes}};
  auto budget = Plenty();
  const auto none = [&](ControlRecoveryResult result) {
    ASSERT_TRUE(result.path);
    EXPECT_TRUE(result.path->rewrites().empty());
    EXPECT_TRUE(result.path->destinations().empty());
    EXPECT_EQ(result.path->revision(), result.path->basis().revision());
    EXPECT_EQ(result.path->effective_transfer(0)->kind, ir::TransferKind::jump);
  };

  // Nothing declared: the table read has no value, so the jump keeps its own.
  none(RecoverControl(Path({DispatchGroup()}), {}, budget));

  // Both arms index entries holding the same number, so this is no branch.
  constexpr std::array<std::uint8_t, 4> flat{5, 0, 5, 0};
  const ir::ConstantImageRange same[] = {{kTable, flat}};
  none(RecoverControl(Path({DispatchGroup()}), Declared(same), budget));

  // A relocated slot overlapping the entry: the loader supplies those bytes.
  const ir::RelocatedPointer slot[] = {{kTable + 2, kCode}};
  none(RecoverControl(Path({DispatchGroup()}), ir::ImageFacts{ranges, slot, false, {}}, budget));

  // A second unresolved choice: enumerating a product of arms is another claim.
  const auto hidden = DispatchGroup();
  auto nodes = std::vector<ir::Node>(hidden.nodes().begin(), hidden.nodes().end());
  nodes.resize(11);
  nodes.push_back({ir::Op::read, 1, {}, 0, 9});
  nodes.push_back({ir::Op::image_address, 64, {}, kCode + 4});
  nodes.push_back({ir::Op::select, 64, {11, 10, 12}});
  nodes.push_back({ir::Op::add, 64, {13, 9}});
  none(RecoverControl(Path({ir::Group(0x100, {1, 2, 3, 4}, std::move(nodes), {},
                                      ir::MemoryModel::atomic_scalar_reference,
                                      ir::Transfer{ir::TransferKind::jump, 14, {}, {}, {}})}),
                      Declared(ranges), budget));
}

TEST(ControlRecovery, EveryBudgetCutRefusesRatherThanRecoveringFewerBranches) {
  const ir::ConstantImageRange ranges[] = {{kTable, kTableBytes}};
  const auto facts = Declared(ranges);
  const auto path = Path({DispatchGroup()});
  auto budget = Plenty();
  auto copy = path;
  ASSERT_EQ(RecoverControl(std::move(copy), facts, budget).path->rewrites().size(), 1);
  const auto required = budget.used();
  for (std::uint64_t work = 0; work < required.work; ++work) {
    Budget limited({work, UINT64_MAX});
    copy = path;
    const auto result = RecoverControl(std::move(copy), facts, limited);
    ASSERT_FALSE(result.path) << work;
    ASSERT_EQ(result.reason, ControlRecoveryDecline::resource_limit);
  }

  for (std::uint64_t bytes = 0; bytes < required.bytes; ++bytes) {
    Budget limited({UINT64_MAX, bytes});
    copy = path;
    const auto result = RecoverControl(std::move(copy), facts, limited);
    ASSERT_FALSE(result.path) << bytes;
    ASSERT_EQ(result.reason, ControlRecoveryDecline::resource_limit);
  }

  Budget exact(required);
  copy = path;
  EXPECT_EQ(RecoverControl(std::move(copy), facts, exact).path->rewrites().size(), 1);
}

TEST(ControlRecovery, ARangeThePathRefutesResolvesNoBranchInAnyLaterStage) {
  // The same dispatcher, behind a store that overwrites the table entry the
  // false arm goes on to read. The fold refutes the range; the branch must not
  // be recovered through it afterwards.
  const auto hidden = DispatchGroup();
  auto path = Path({ir::Group(0xfc, {9, 9, 9, 9},
                              {{ir::Op::image_address, 64, {}, kTable},
                               {ir::Op::constant, 16, {}, 0x1234},
                               {ir::Op::store, 16, {0, 1}}},
                              {}, ir::MemoryModel::atomic_scalar_reference,
                              ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}}),
                    hidden});
  const ir::ConstantImageRange ranges[] = {{kTable, kTableBytes}};
  const auto facts = Declared(ranges);
  auto budget = Plenty();
  auto folded = FoldImageValues(path, facts, budget);
  ASSERT_TRUE(folded.path);
  ASSERT_EQ(folded.contradicted.size(), 1);
  EXPECT_EQ(folded.contradicted[0].range, kTable);
  EXPECT_TRUE(folded.constants.empty());

  // The control: with the refuted range still declared, the branch does resolve,
  // on bytes this path has just overwritten. That is what must not reach a stage.
  auto control = FoldImageValues(path, facts, budget);
  ASSERT_TRUE(control.path);
  EXPECT_EQ(RecoverControl(std::move(*control.path), facts, budget).path->rewrites().size(), 1);

  // With the declaration the fold actually leaves, nothing is recovered.
  const ir::ImageFacts held{folded.constants, facts.pointers, facts.page_aligned_placement};
  auto recovered = RecoverControl(std::move(*folded.path), held, budget);
  ASSERT_TRUE(recovered.path);
  EXPECT_TRUE(recovered.path->rewrites().empty());
  EXPECT_TRUE(recovered.path->destinations().empty());
  EXPECT_EQ(recovered.path->effective_transfer(1)->kind, ir::TransferKind::jump);
}

}  // namespace
}  // namespace nyx::recovery
