#include <algorithm>
#include <limits>
#include <set>
#include <tuple>

#include <gtest/gtest.h>

#include "nyx/eval/concrete.hpp"
#include "nyx/target/a64/decode.hpp"

namespace nyx::a64 {
namespace {

std::array<std::uint8_t, 4> Bytes(std::uint32_t word) {
  return {static_cast<std::uint8_t>(word), static_cast<std::uint8_t>(word >> 8),
          static_cast<std::uint8_t>(word >> 16), static_cast<std::uint8_t>(word >> 24)};
}

DecodeResult LiftWord(std::uint32_t word) {
  Budget budget({10000, 10000});
  return Decode(0xc687bc, Bytes(word), budget);
}

TEST(A64Decode, LinkRegisterSigningHintsModelAsNothingAndTheRestStayUnsupported) {
  // PACIAZ, PACIASP, PACIBZ, PACIBSP, AUTIAZ, AUTIASP, AUTIBZ, AUTIBSP: the
  // eight that sign or authenticate the link register against SP or zero.
  for (unsigned op2 = 0; op2 < 8; ++op2) {
    const auto word = 0xd503231fU | (op2 << 5);
    const auto lifted = LiftWord(word);
    ASSERT_TRUE(lifted.group) << std::hex << word;
    EXPECT_TRUE(lifted.group->writes().empty()) << std::hex << word;
  }

  // The forms taking a general register are not the same instruction and
  // do not model as nothing: PACIA, AUTIA, XPACD, and the register HINTs
  // that are neither these nor BTI.
  for (auto word : {0xdac10020U, 0xdac11020U, 0xdac147e0U, 0xd503227fU}) {
    EXPECT_FALSE(LiftWord(word).group) << std::hex << word;
  }
}

TEST(A64Decode, OpaqueControlRecognizesExclusiveScalarAccesses) {
  for (auto word : {0x885ffc0aU, 0xc85ffc0aU, 0x885ffc00U, 0xc85fffe0U}) {
    EXPECT_TRUE(OpaqueNormalFallthrough(Bytes(word))) << std::hex << word;
    EXPECT_FALSE(LiftWord(word).group) << std::hex << word;
  }

  for (auto word : {0x880bfd09U, 0xc80bfd09U, 0x880b7d09U, 0xc80b7d09U}) {
    EXPECT_TRUE(OpaqueNormalFallthrough(Bytes(word))) << std::hex << word;
    EXPECT_FALSE(LiftWord(word).group) << std::hex << word;
  }

  // Division has exact semantics now, so it is lifted rather than opaque.
  for (auto word : {0x9acc090dU, 0x9ac80c0aU, 0x1ac00800U, 0x1ac00c00U}) {
    EXPECT_FALSE(OpaqueNormalFallthrough(Bytes(word))) << std::hex << word;
    EXPECT_TRUE(LiftWord(word).group) << std::hex << word;
  }

  for (auto word : {0x885f7c0aU, 0xc85f7c0aU, 0x88dffc0aU, 0x889ffd09U, 0x882b2909U, 0x080b7d09U,
                    0x480b7d09U, 0xd4200000U, 0xffffffffU}) {
    EXPECT_FALSE(OpaqueNormalFallthrough(Bytes(word))) << std::hex << word;
  }
}

TEST(A64Decode, OpaqueTrapRecognizesOnlyBreakpoints) {
  for (auto word : {0xd4200000U, 0xd4200020U, 0xd43fffe0U}) {
    EXPECT_TRUE(OpaqueTrap(Bytes(word))) << std::hex << word;
    EXPECT_FALSE(OpaqueNormalFallthrough(Bytes(word))) << std::hex << word;
    EXPECT_FALSE(LiftWord(word).group) << std::hex << word;
  }

  // HLT, SVC, a BRK encoding with nonzero low bits, and UDF are not breakpoints.
  for (auto word : {0xd4400000U, 0xd4000001U, 0xd4200001U, 0x00000000U}) {
    EXPECT_FALSE(OpaqueTrap(Bytes(word))) << std::hex << word;
  }
}

TEST(A64Decode, LdrLiteralNamesImageAddressAndKeepsDiscardedRead) {
  Budget budget({10000, 10000});
  EXPECT_EQ(Decode(0x100, Bytes(0x5800009f), budget).reason, DecodeDecline::unsupported);
  auto decoded = Decode(0x100, Bytes(0x5800009f), budget, {MemoryProfile::concrete_atomic_scalar});
  ASSERT_TRUE(decoded.group);
  EXPECT_EQ(decoded.group->memory_model(), ir::MemoryModel::atomic_scalar_reference);
  ASSERT_EQ(decoded.group->nodes().size(), 2);
  EXPECT_EQ(decoded.group->nodes()[0].op, ir::Op::image_address);
  EXPECT_EQ(decoded.group->nodes()[0].immediate, 0x110U);
  EXPECT_EQ(decoded.group->nodes()[1].op, ir::Op::load);
  EXPECT_TRUE(decoded.group->writes().empty());
  auto reverse = Decode(0x100, Bytes(0x58ffffc0), budget, {MemoryProfile::concrete_atomic_scalar});
  ASSERT_TRUE(reverse.group);
  EXPECT_EQ(reverse.group->nodes()[0].immediate, 0xf8U);
}

TEST(A64Decode, ExclusiveProfileLiftsMonitorEffectsAndDeclinesSpBase) {
  for (const auto [word, effect, width] : {std::tuple{0x885ffd0aU, ir::Op::exclusive_load, 32U},
                                           std::tuple{0xc85ffd0aU, ir::Op::exclusive_load, 64U},
                                           std::tuple{0x880bfd09U, ir::Op::exclusive_store, 32U},
                                           std::tuple{0xc80bfd09U, ir::Op::exclusive_store, 64U},
                                           std::tuple{0x880b7d09U, ir::Op::exclusive_store, 32U},
                                           std::tuple{0xc80b7d09U, ir::Op::exclusive_store, 64U}}) {
    Budget budget({10000, 10000});
    EXPECT_EQ(Decode(0x1000, Bytes(word), budget, {MemoryProfile::concrete_atomic_scalar}).reason,
              DecodeDecline::unsupported);
    auto decoded = Decode(0x1000, Bytes(word), budget, {MemoryProfile::concrete_exclusive_scalar});
    ASSERT_TRUE(decoded.group) << std::hex << word;
    EXPECT_EQ(decoded.group->memory_model(), ir::MemoryModel::qemu_exclusive_scalar_reference);
    const auto nodes = decoded.group->nodes();
    const auto access =
        std::ranges::find_if(nodes, [effect](const auto& node) { return node.op == effect; });
    ASSERT_NE(access, nodes.end());
    EXPECT_EQ(access->access.alignment, width / 8);
    EXPECT_TRUE(access->access.decline_on_unaligned);
    EXPECT_EQ(access->width, effect == ir::Op::exclusive_load ? width : 32);
    EXPECT_EQ(
        Decode(0x1000, Bytes(word | 0x3e0U), budget, {MemoryProfile::concrete_exclusive_scalar})
            .reason,
        DecodeDecline::unsupported);
  }

  Budget budget({10000, 10000});
  auto clear =
      Decode(0x1000, Bytes(0xd503375f), budget, {MemoryProfile::concrete_exclusive_scalar});
  ASSERT_TRUE(clear.group);
  ASSERT_EQ(clear.group->nodes().size(), 1);
  EXPECT_EQ(clear.group->nodes()[0].op, ir::Op::exclusive_clear);
  EXPECT_EQ(
      Decode(0x1000, Bytes(0xd503375f), budget, {MemoryProfile::concrete_atomic_scalar}).reason,
      DecodeDecline::unsupported);
}

TEST(A64Decode, ExclusiveStoreRegisterAliasesReadEntrySnapshot) {
  for (const auto [word, data, status] :
       {std::tuple{0x8809fd09U, 55U, 9U}, std::tuple{0x8808fd09U, 55U, 8U},
        std::tuple{0x880bfd08U, 0x1000U, 11U}}) {
    Budget budget({100000, 100000});
    const std::array<std::uint8_t, 4> bytes{42, 0, 0, 0};
    const eval::RegionInput region{0x1000, bytes};
    auto memory = eval::Memory::Create(std::span(&region, 1), budget);
    ASSERT_TRUE(memory.memory);
    eval::State state;
    for (const auto [id, value] :
         {std::pair{8U, 0x1000U}, std::pair{9U, 55U}, std::pair{10U, 0U}, std::pair{11U, 1U}}) {
      auto bits = BitVector::from_u64(64, value, 64, budget);
      ASSERT_TRUE(bits);
      state.cells.push_back({id, std::move(*bits)});
    }

    auto load =
        Decode(0x2000, Bytes(0x885ffd0aU), budget, {MemoryProfile::concrete_exclusive_scalar});
    auto store = Decode(0x2004, Bytes(word), budget, {MemoryProfile::concrete_exclusive_scalar});
    ASSERT_TRUE(load.group);
    ASSERT_TRUE(store.group);
    EXPECT_EQ(eval::Execute(*load.group, state, *memory.memory, budget).outcome,
              eval::Outcome::completed);
    EXPECT_EQ(eval::Execute(*store.group, state, *memory.memory, budget).outcome,
              eval::Outcome::completed);
    EXPECT_EQ(state.cells[status == 8 ? 0 : status == 9 ? 1 : 3].value.word(0), 0);
    const auto written = memory.memory->Regions()[0].bytes;
    EXPECT_EQ(std::uint32_t(written[0]) | (std::uint32_t(written[1]) << 8), data);
  }
}

TEST(A64Decode, PreservesSourceAndSupportsHotGlobalRegisterChain) {
  // GNU assembler independently supplies these words; the load is deliberately absent.
  const std::uint32_t words[] = {0xcb0903e9, 0xd28ea46b, 0xf2a8d7eb, 0xf2deaecb, 0xf2f1d48b,
                                 0xaa0b012b, 0xd37ff96b, 0xca0c0129, 0xcb090169, 0x9b0b3529};
  for (auto word : words) {
    auto result = LiftWord(word);
    ASSERT_TRUE(result.group) << std::hex << word;
    EXPECT_EQ(result.group->source_address(), 0xc687bc);
    const auto bytes = Bytes(word);
    EXPECT_TRUE(std::ranges::equal(result.group->bytes(), bytes));
    EXPECT_EQ(result.group->writes().size(), 1);
  }
}

TEST(A64Decode, ZeroRegisterDoesNotIntroduceStorage) {
  const auto result = LiftWord(0xcb0903e9);  // sub x9, xzr, x9
  ASSERT_TRUE(result.group);
  bool reads_x9 = false;
  for (const auto& node : result.group->nodes()) {
    if (node.op == ir::Op::read) {
      EXPECT_NE(node.storage, kSp);
      reads_x9 |= node.storage == 9;
    }
  }

  EXPECT_TRUE(reads_x9);
  const auto cmp = LiftWord(0xeb02003f);  // cmp x1, x2
  ASSERT_TRUE(cmp.group);
  ASSERT_EQ(cmp.group->writes().size(), 4);
  for (const auto& write : cmp.group->writes()) EXPECT_GE(write.storage, kN);
}

TEST(A64Decode, StackPointerAndWRegisterWritesAreExplicit) {
  const auto sp = LiftWord(0x910043ff);  // add sp, sp, #16
  ASSERT_TRUE(sp.group);
  ASSERT_EQ(sp.group->writes().size(), 1);
  EXPECT_EQ(sp.group->writes()[0].storage, kSp);
  EXPECT_TRUE(std::ranges::any_of(sp.group->nodes(), [](const auto& node) {
    return node.op == ir::Op::read && node.storage == kSp && node.width == 64;
  }));
  const auto w = LiftWord(0x0b020020);  // add w0, w1, w2
  ASSERT_TRUE(w.group);
  const auto& value = w.group->nodes()[w.group->writes()[0].value];
  EXPECT_EQ(value.op, ir::Op::zext);
  EXPECT_EQ(value.width, 64);
  EXPECT_EQ(w.group->nodes()[value.inputs[0]].width, 32);
}

TEST(A64Decode, ExtendedRegisterFormsReadSpAsBaseAndWriteItOnlyWithoutFlags) {
  // GNU assembler words; the register-only differential cannot reach SP.
  const auto reads_sp = [](const auto& group) {
    return std::ranges::any_of(group.nodes(), [](const auto& node) {
      return node.op == ir::Op::read && node.storage == kSp;
    });
  };

  const auto add = LiftWord(0x8b2253ff);  // add sp, sp, w2, uxtw #4
  ASSERT_TRUE(add.group);
  EXPECT_TRUE(reads_sp(*add.group));
  ASSERT_EQ(add.group->writes().size(), 1);
  EXPECT_EQ(add.group->writes()[0].storage, kSp);
  const auto compare = LiftWord(0xeb2303ff);  // cmp sp, w3, uxtb
  ASSERT_TRUE(compare.group);
  EXPECT_TRUE(reads_sp(*compare.group));
  EXPECT_EQ(compare.group->writes().size(), 4) << "flags only; Rd 31 is the zero register";
  EXPECT_TRUE(std::ranges::none_of(compare.group->writes(),
                                   [](const auto& write) { return write.storage == kSp; }));
  // A left shift above four is reserved.
  for (auto word : {0x8b221420U, 0x8b221c20U}) {  // add x0, x1, w2, uxtb #5 / #7
    const auto result = LiftWord(word);
    EXPECT_FALSE(result.group) << std::hex << word;
    EXPECT_EQ(result.reason, DecodeDecline::invalid_encoding) << std::hex << word;
  }
}

TEST(A64Decode, ThreadPointerReadIsARuntimeValueAndItsWriteIsNotModeled) {
  const auto read = LiftWord(0xd53bd048);  // mrs x8, tpidr_el0
  ASSERT_TRUE(read.group);
  ASSERT_EQ(read.group->writes().size(), 1);
  EXPECT_EQ(read.group->writes()[0].storage, 8);
  const auto& value = read.group->nodes()[read.group->writes()[0].value];
  EXPECT_EQ(value.op, ir::Op::read);
  EXPECT_EQ(value.storage, kTpidrEl0);
  EXPECT_EQ(value.width, 64);

  // Into the zero register the read has no effect to record.
  const auto discarded = LiftWord(0xd53bd05f);  // mrs xzr, tpidr_el0
  ASSERT_TRUE(discarded.group);
  EXPECT_TRUE(discarded.group->writes().empty());

  // Neither the write nor a neighbouring system register is taken for it.
  for (auto word : {0xd51bd048U, 0xd53bd068U}) {  // msr tpidr_el0, x8 / mrs x8, tpidrro_el0
    EXPECT_FALSE(LiftWord(word).group) << std::hex << word;
  }
}

TEST(A64Decode, FlagsAndSelectionsRemainGenericOperations) {
  const std::uint32_t words[] = {0xab020020, 0x9a820020, 0xda82c420, 0xea02003f, 0x9351fc20};
  for (auto word : words) {
    auto result = LiftWord(word);
    ASSERT_TRUE(result.group) << std::hex << word;
    for (std::size_t i = 0; i < result.group->nodes().size(); ++i) {
      const auto& node = result.group->nodes()[i];
      if (node.op != ir::Op::read && node.op != ir::Op::constant) {
        EXPECT_LT(node.inputs[0], i);
      }
    }
  }

  const auto nop = LiftWord(0xd503201f);
  ASSERT_TRUE(nop.group);
  EXPECT_TRUE(nop.group->writes().empty());
  EXPECT_TRUE(nop.group->nodes().empty());
}

TEST(A64Decode, NzcvTransfersUseOnlyArchitecturalFlagBitsAndRespectZeroRegister) {
  for (unsigned reg = 0; reg < 32; ++reg) {
    const auto read = LiftWord(0xd53b4200U | reg);
    ASSERT_TRUE(read.group);
    EXPECT_EQ(read.group->writes().size(), reg == 31 ? 0U : 1U);
    if (reg != 31) {
      EXPECT_EQ(read.group->writes()[0].storage, reg);
    }

    for (unsigned flag = kN; flag <= kV; ++flag) {
      EXPECT_TRUE(std::ranges::any_of(read.group->nodes(), [&](const auto& node) {
        return node.op == ir::Op::read && node.storage == flag && node.width == 1;
      }));
    }

    const auto write = LiftWord(0xd51b4200U | reg);
    ASSERT_TRUE(write.group);
    ASSERT_EQ(write.group->writes().size(), 4);
    for (unsigned index = 0; index < 4; ++index) {
      const auto& effect = write.group->writes()[index];
      EXPECT_EQ(effect.storage, kN + index);
      const auto& value = write.group->nodes()[effect.value];
      EXPECT_EQ(value.op, ir::Op::extract);
      EXPECT_EQ(value.width, 1);
      EXPECT_EQ(value.immediate, 31 - index);
      const auto& source = write.group->nodes()[value.inputs[0]];
      EXPECT_EQ(source.op, reg == 31 ? ir::Op::constant : ir::Op::read);
      if (reg == 31) {
        EXPECT_EQ(source.immediate, 0);
      } else {
        EXPECT_EQ(source.storage, reg);
      }
    }
  }

  for (auto word : {0xd53b4220U, 0xd51b4220U, 0xd53b4400U, 0xd51b4400U}) {
    EXPECT_EQ(LiftWord(word).reason, DecodeDecline::unsupported) << std::hex << word;
  }

  for (auto word : {0xd53b4200U, 0xd53b421fU, 0xd51b4200U, 0xd51b421fU}) {
    Budget full({10000, 10000});
    ASSERT_TRUE(Decode(0x1000, Bytes(word), full).group);
    for (std::uint64_t work = 0; work < full.used().work; ++work) {
      Budget limited({work, 10000});
      const auto result = Decode(0x1000, Bytes(word), limited);
      EXPECT_FALSE(result.group);
      EXPECT_EQ(result.reason, DecodeDecline::work_limit);
    }
  }
}

TEST(A64Decode, CarryArithmeticReadsTheCarryFlagAndOnlySetsFlagsWhenAsked) {
  // adc x0, x1, x2 / adcs / sbc / sbcs, then the 32-bit adc w0, w1, w2.
  for (auto word : {0x9a020020U, 0xba020020U, 0xda020020U, 0xfa020020U, 0x1a020020U}) {
    const auto lifted = LiftWord(word);
    ASSERT_TRUE(lifted.group) << std::hex << word;

    // The carry-in must be read, or the result is wrong for half of all states.
    const auto nodes = lifted.group->nodes();
    EXPECT_TRUE(std::any_of(nodes.begin(), nodes.end(),
                            [](const ir::Node& node) {
                              return node.op == ir::Op::read && node.width == 1 &&
                                     node.storage == kC;
                            }))
        << std::hex << word;
    const bool sets_flags = (word & (1U << 29)) != 0;
    const auto writes = lifted.group->writes();
    const auto flags = std::count_if(writes.begin(), writes.end(), [](const ir::Write& write) {
      return write.storage >= kN && write.storage <= kV;
    });
    EXPECT_EQ(flags, sets_flags ? 4 : 0) << std::hex << word;
    EXPECT_EQ(writes.size(), sets_flags ? 5U : 1U) << std::hex << word;
  }

  // The reserved shift/extend fields of the carry forms are not this encoding.
  for (auto word : {0x9a020420U, 0x9a028020U}) {
    EXPECT_FALSE(LiftWord(word).group) << std::hex << word;
  }
}

TEST(A64Decode, DeclinesUnsupportedAndReservedEncodings) {
  for (auto word : {0xf9400129U, 0xd4000001U, 0xd65f0bffU}) {
    const auto result = LiftWord(word);
    EXPECT_FALSE(result.group);
    EXPECT_EQ(result.reason, DecodeDecline::unsupported);
  }

  for (auto word : {0x0b028020U, 0x0bc20020U, 0x52c00000U, 0x33400020U}) {
    const auto result = LiftWord(word);
    EXPECT_FALSE(result.group) << std::hex << word;
    EXPECT_EQ(result.reason, DecodeDecline::invalid_encoding) << std::hex << word;
  }
}

TEST(A64Decode, AdrUsesExplicitImageAddressAndSignedDisplacement) {
  for (const auto word : {0x10000040U, 0x10ffffc0U}) {
    const auto result = LiftWord(word);
    ASSERT_TRUE(result.group);
    EXPECT_EQ(result.group->memory_model(), ir::MemoryModel::unspecified);
    const auto nodes = result.group->nodes();
    ASSERT_EQ(nodes.size(), 3);
    EXPECT_EQ(nodes[0].op, ir::Op::image_address);
    EXPECT_EQ(nodes[0].immediate, 0xc687bc);
    EXPECT_EQ(nodes[1].op, ir::Op::constant);
    EXPECT_EQ(nodes[1].immediate, word == 0x10000040U ? 8 : std::uint64_t{0} - 8);
    EXPECT_EQ(nodes[2].op, ir::Op::add);
  }
}

TEST(A64Decode, AdrpMasksRuntimeAddressBeforeAddingPageDisplacement) {
  for (const auto word : {0xb0000000U, 0xf0ffffe0U}) {
    const auto result = LiftWord(word);
    ASSERT_TRUE(result.group);
    const auto nodes = result.group->nodes();
    const auto& sum = nodes[result.group->writes()[0].value];
    ASSERT_EQ(sum.op, ir::Op::add);
    const auto& page = nodes[sum.inputs[0]];
    ASSERT_EQ(page.op, ir::Op::bit_and);
    EXPECT_EQ(nodes[page.inputs[0]].op, ir::Op::image_address);
    EXPECT_EQ(nodes[page.inputs[0]].immediate, 0xc687bc);
    EXPECT_EQ(nodes[page.inputs[1]].immediate, ~std::uint64_t{0xfff});
    EXPECT_EQ(nodes[sum.inputs[1]].immediate, word == 0xb0000000U ? 4096 : std::uint64_t{0} - 4096);
  }
}

DecodeResult LiftMemory(std::uint32_t word) {
  Budget budget({10000, 10000});
  return Decode(0x1000, Bytes(word), budget, {MemoryProfile::concrete_atomic_scalar});
}

TEST(A64Decode, ScalarMemoryRequiresExplicitModelAndRetainsAccessWidths) {
  const std::uint32_t loads[] = {0x39400c20, 0x79400c20, 0xb9400c20, 0xf9400c20};
  const std::uint32_t stores[] = {0x39000c20, 0x79000c20, 0xb9000c20, 0xf9000c20};
  for (unsigned size = 0; size < 4; ++size) {
    for (const auto word : {loads[size], stores[size]}) {
      EXPECT_EQ(LiftWord(word).reason, DecodeDecline::unsupported);
      const auto result = LiftMemory(word);
      ASSERT_TRUE(result.group);
      EXPECT_EQ(result.group->memory_model(), ir::MemoryModel::atomic_scalar_reference);
      const auto expected = word == loads[size] ? ir::Op::load : ir::Op::store;
      const auto nodes = result.group->nodes();
      const auto access =
          std::ranges::find_if(nodes, [expected](const auto& node) { return node.op == expected; });
      ASSERT_NE(access, nodes.end());
      EXPECT_EQ(access->width, 8U << size);
      EXPECT_EQ(access->access.alignment, 1);
      EXPECT_EQ(access->access.byte_order, ir::ByteOrder::little);
      const auto& address = nodes[access->inputs[0]];
      EXPECT_EQ(address.op, ir::Op::add);
      EXPECT_EQ(nodes[address.inputs[1]].immediate, 3U << size);
    }
  }
}

TEST(A64Decode, StoreReleaseRequiresAlignedConcreteMemory) {
  for (const auto [word, width] :
       {std::pair{0x889ffc20U, 32U}, std::pair{0xc89ffc20U, 64U}, std::pair{0x889ffd1fU, 32U}}) {
    EXPECT_EQ(LiftWord(word).reason, DecodeDecline::unsupported);
    const auto result = LiftMemory(word);
    ASSERT_TRUE(result.group);
    EXPECT_EQ(result.group->memory_model(), ir::MemoryModel::atomic_scalar_reference);
    const auto nodes = result.group->nodes();
    const auto store =
        std::ranges::find_if(nodes, [](const auto& node) { return node.op == ir::Op::store; });
    ASSERT_NE(store, nodes.end());
    EXPECT_EQ(store->width, width);
    EXPECT_EQ(store->access.alignment, width / 8);
    EXPECT_TRUE(store->access.decline_on_unaligned);
  }

  Budget budget({100000, 100000});
  const auto group = LiftMemory(0x889ffc20U).group;
  ASSERT_TRUE(group);
  eval::State state;
  auto value = BitVector::from_u64(64, 0x12345678, 64, budget);
  auto address = BitVector::from_u64(64, 0x100000001, 64, budget);
  ASSERT_TRUE(value);
  ASSERT_TRUE(address);
  state.cells.push_back({0, std::move(*value)});
  state.cells.push_back({1, std::move(*address)});
  const std::array<std::uint8_t, 8> bytes{};
  const eval::RegionInput region[] = {{0x100000000, bytes}};
  auto memory = eval::Memory::Create(region, budget);
  ASSERT_TRUE(memory.memory);
  const auto result = eval::Execute(*group, state, *memory.memory, budget);
  EXPECT_EQ(result.outcome, eval::Outcome::unsupported);
  EXPECT_TRUE(std::ranges::equal(memory.memory->Regions()[0].bytes, bytes));
}

TEST(A64Decode, AcquireReleaseAllWidthsRetainAccessAndZeroExtension) {
  for (unsigned size = 0; size < 4; ++size) {
    const unsigned width = 8U << size;
    for (const bool load : {false, true}) {
      for (const unsigned destination : {0U, 31U}) {
        const auto word = 0x089ffc20U | (size << 30) | (load ? 1U << 22 : 0) | destination;
        EXPECT_EQ(LiftWord(word).reason, DecodeDecline::unsupported);
        const auto result = LiftMemory(word);
        ASSERT_TRUE(result.group);
        const auto nodes = result.group->nodes();
        const auto access = std::ranges::find_if(nodes, [load](const auto& node) {
          return node.op == (load ? ir::Op::load : ir::Op::store);
        });
        ASSERT_NE(access, nodes.end());
        EXPECT_EQ(access->width, width);
        EXPECT_EQ(access->access.alignment, width / 8);
        EXPECT_TRUE(access->access.decline_on_unaligned);
        EXPECT_EQ(access->access.byte_order, ir::ByteOrder::little);
        if (load && destination != 31) {
          ASSERT_EQ(result.group->writes().size(), 1);
          const auto& value = nodes[result.group->writes()[0].value];
          EXPECT_EQ(value.width, 64);
          EXPECT_EQ(value.op, width == 64 ? ir::Op::load : ir::Op::zext);
        } else {
          EXPECT_TRUE(result.group->writes().empty());
        }
      }

      if (size == 0) continue;
      const auto result = LiftMemory(0x089ffc20U | (size << 30) | (load ? 1U << 22 : 0));
      ASSERT_TRUE(result.group);
      Budget budget({100000, 100000});
      eval::State state;
      auto value = BitVector::from_u64(64, 0x12345678, 64, budget);
      auto address = BitVector::from_u64(64, 0x100000001, 64, budget);
      ASSERT_TRUE(value);
      ASSERT_TRUE(address);
      state.cells.push_back({0, std::move(*value)});
      state.cells.push_back({1, std::move(*address)});
      const std::array<std::uint8_t, 16> bytes{};
      const eval::RegionInput region[] = {{0x100000000, bytes}};
      auto memory = eval::Memory::Create(region, budget);
      ASSERT_TRUE(memory.memory);
      const auto executed = eval::Execute(*result.group, state, *memory.memory, budget);
      EXPECT_EQ(executed.outcome, eval::Outcome::unsupported);
      EXPECT_TRUE(executed.events.empty());
      EXPECT_EQ(state.cells[0].value.word(0), 0x12345678);
      EXPECT_TRUE(std::ranges::equal(memory.memory->Regions()[0].bytes, bytes));
    }
  }

  for (const auto word : {0x085ffc20U, 0x485ffc20U, 0x089f7c20U, 0x08df7c20U}) {
    EXPECT_EQ(LiftMemory(word).reason, DecodeDecline::unsupported);
  }
}

TEST(A64Decode, UnsignedOffsetQLoadUsesSeparateStorageAndOrderedReads) {
  for (const auto [word, destination, offset] :
       {std::tuple{0x3dc00120U, kQ0, 0U},       // ldr q0, [x9]
        std::tuple{0x3dc00be7U, kQ0 + 7, 32U},  // ldr q7, [sp, #32]
        std::tuple{0x3dfffd1fU, kQ0 + 31, 65520U}}) {
    EXPECT_EQ(LiftWord(word).reason, DecodeDecline::unsupported);
    const auto result = LiftMemory(word);
    ASSERT_TRUE(result.group);
    EXPECT_EQ(result.group->memory_model(), ir::MemoryModel::atomic_scalar_reference);
    ASSERT_EQ(result.group->writes().size(), 1);
    EXPECT_EQ(result.group->writes()[0].storage, destination);
    const auto nodes = result.group->nodes();
    EXPECT_EQ(nodes[result.group->writes()[0].value].width, 128);
    std::vector<ir::ValueId> loads;
    for (ir::ValueId id = 0; id < nodes.size(); ++id) {
      if (nodes[id].op == ir::Op::load) loads.push_back(id);
    }

    ASSERT_EQ(loads.size(), 2);
    EXPECT_EQ(nodes[loads[0]].width, 64);
    EXPECT_EQ(nodes[loads[1]].width, 64);
    const auto& address = nodes[nodes[loads[0]].inputs[0]];
    ASSERT_EQ(address.op, ir::Op::add);
    EXPECT_EQ(nodes[address.inputs[1]].immediate, offset);
  }

  const auto group = LiftMemory(0x3dc00120U).group;
  ASSERT_TRUE(group);
  constexpr std::uint64_t base = 0x100000000;
  std::array<std::uint8_t, 16> bytes{};
  for (unsigned i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(i);
  const auto run = [&](std::size_t readable) {
    Budget budget({1000000, 1000000});
    eval::State state;
    auto address = BitVector::from_u64(64, base, 128, budget);
    const std::array<std::uint64_t, 2> seed{0xa7a6a5a4a3a2a1a0, 0xafaeadacabaaa9a8};
    auto vector = BitVector::from_words(128, seed, 128, budget);
    EXPECT_TRUE(address);
    EXPECT_TRUE(vector);
    state.cells.push_back({9, std::move(*address)});
    state.cells.push_back({kQ0, std::move(*vector)});
    const eval::RegionInput region[] = {{base, std::span(bytes).first(readable)}};
    auto memory = eval::Memory::Create(region, budget);
    EXPECT_TRUE(memory.memory);
    auto result = eval::Execute(*group, state, *memory.memory, budget);
    return std::pair{std::move(result), std::move(state)};
  };

  const auto [completed, loaded] = run(16);
  EXPECT_EQ(completed.outcome, eval::Outcome::completed);
  ASSERT_EQ(completed.events.size(), 2);
  EXPECT_EQ(completed.events[0].address, base);
  EXPECT_EQ(completed.events[1].address, base + 8);
  EXPECT_EQ(completed.events[0].size, 8);
  EXPECT_EQ(completed.events[1].size, 8);
  EXPECT_EQ(loaded.cells[1].value.word(0), 0x0706050403020100);
  EXPECT_EQ(loaded.cells[1].value.word(1), 0x0f0e0d0c0b0a0908);
  const auto [fault, retained] = run(8);
  EXPECT_EQ(fault.outcome, eval::Outcome::fault);
  ASSERT_TRUE(fault.fault);
  EXPECT_EQ(fault.fault->address, base + 8);
  ASSERT_EQ(fault.events.size(), 2);
  EXPECT_TRUE(fault.events[0].completed);
  EXPECT_FALSE(fault.events[1].completed);
  EXPECT_EQ(retained.cells[1].value.word(0), 0xa7a6a5a4a3a2a1a0);
  EXPECT_EQ(retained.cells[1].value.word(1), 0xafaeadacabaaa9a8);
}

TEST(A64Decode, UnsignedOffsetQStoreRetainsFirstHalfOnSecondFault) {
  constexpr std::uint32_t word = 0x3d800120U;  // str q0, [x9]
  EXPECT_EQ(LiftWord(word).reason, DecodeDecline::unsupported);
  const auto group = LiftMemory(word).group;
  ASSERT_TRUE(group);
  EXPECT_EQ(group->memory_model(), ir::MemoryModel::atomic_scalar_reference);
  EXPECT_TRUE(group->writes().empty());
  const auto nodes = group->nodes();
  std::vector<ir::ValueId> stores;
  for (ir::ValueId id = 0; id < nodes.size(); ++id) {
    if (nodes[id].op == ir::Op::store) stores.push_back(id);
  }

  ASSERT_EQ(stores.size(), 2);
  EXPECT_EQ(nodes[stores[0]].width, 64);
  EXPECT_EQ(nodes[stores[1]].width, 64);
  EXPECT_EQ(nodes[nodes[stores[0]].inputs[1]].op, ir::Op::extract);
  constexpr std::uint64_t base = 0x100000000;
  std::array<std::uint8_t, 16> bytes{};
  Budget budget({1000000, 1000000});
  eval::State state;
  auto address = BitVector::from_u64(64, base, 128, budget);
  const std::array<std::uint64_t, 2> seed{0xa7a6a5a4a3a2a1a0, 0xafaeadacabaaa9a8};
  auto vector = BitVector::from_words(128, seed, 128, budget);
  ASSERT_TRUE(address);
  ASSERT_TRUE(vector);
  state.cells.push_back({9, std::move(*address)});
  state.cells.push_back({kQ0, std::move(*vector)});
  const eval::RegionInput region[] = {{base, std::span(bytes).first(8)}};
  auto memory = eval::Memory::Create(region, budget);
  ASSERT_TRUE(memory.memory);
  const auto fault = eval::Execute(*group, state, *memory.memory, budget);
  EXPECT_EQ(fault.outcome, eval::Outcome::fault);
  ASSERT_TRUE(fault.fault);
  EXPECT_EQ(fault.fault->address, base + 8);
  ASSERT_EQ(fault.events.size(), 2);
  EXPECT_TRUE(fault.events[0].completed);
  EXPECT_FALSE(fault.events[1].completed);
  const std::array<std::uint8_t, 8> expected{0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7};
  EXPECT_TRUE(std::ranges::equal(memory.memory->Regions()[0].bytes, expected));
}

TEST(A64Decode, ScalarWritebackIsDeferredAndSpDiffersFromZeroRegister) {
  const auto pre = LiftMemory(0xf85f8fe0);  // ldr x0, [sp, #-8]!
  ASSERT_TRUE(pre.group);
  ASSERT_EQ(pre.group->writes().size(), 2);
  EXPECT_EQ(pre.group->writes()[0].storage, 0);
  EXPECT_EQ(pre.group->writes()[1].storage, kSp);
  const auto pre_nodes = pre.group->nodes();
  const auto access =
      std::ranges::find_if(pre_nodes, [](const auto& node) { return node.op == ir::Op::load; });
  ASSERT_NE(access, pre_nodes.end());
  const auto& updated = pre_nodes[access->inputs[0]];
  EXPECT_EQ(updated.op, ir::Op::add);
  EXPECT_EQ(pre_nodes[updated.inputs[1]].immediate, std::uint64_t{0} - 8);
  EXPECT_EQ(pre.group->writes()[1].value, access->inputs[0]);
  const auto post = LiftMemory(0xf80087e0);  // str x0, [sp], #8
  ASSERT_TRUE(post.group);
  const auto post_nodes = post.group->nodes();
  const auto store =
      std::ranges::find_if(post_nodes, [](const auto& node) { return node.op == ir::Op::store; });
  ASSERT_NE(store, post_nodes.end());
  EXPECT_EQ(post_nodes[store->inputs[0]].op, ir::Op::read);
  EXPECT_EQ(post_nodes[store->inputs[0]].storage, kSp);
  for (const auto word : {0xf84087ffU, 0xf80087ffU}) {
    const auto zero = LiftMemory(word);
    ASSERT_TRUE(zero.group);
    ASSERT_EQ(zero.group->writes().size(), 1);
    EXPECT_EQ(zero.group->writes()[0].storage, kSp);
    EXPECT_TRUE(std::ranges::any_of(zero.group->nodes(), [](const auto& node) {
      return node.op == ir::Op::load || node.op == ir::Op::store;
    }));
  }
}

TEST(A64Decode, SignExtendingLoadsAndUnscaledOffsetsRemainExplicit) {
  for (const auto word :
       {0x39800c20U, 0x39c00c20U, 0x79800c20U, 0x79c00c20U, 0xb9800c20U, 0x38dfd020U}) {
    const auto result = LiftMemory(word);
    ASSERT_TRUE(result.group) << std::hex << word;
    EXPECT_TRUE(std::ranges::any_of(result.group->nodes(),
                                    [](const auto& node) { return node.op == ir::Op::ashr; }));
  }

  const auto unscaled = LiftMemory(0xf85f8020);  // ldur x0, [x1, #-8]
  ASSERT_TRUE(unscaled.group);
  EXPECT_TRUE(std::ranges::any_of(unscaled.group->nodes(), [](const auto& node) {
    return node.op == ir::Op::constant && node.immediate == std::uint64_t{0} - 8;
  }));
}

TEST(A64Decode, ScalarProfileDeclinesUnpredictableAndUnimplementedForms) {
  for (const auto word : {0xf8408400U, 0xf8008400U, 0xb9c00020U}) {
    EXPECT_EQ(LiftMemory(word).reason, DecodeDecline::invalid_encoding);
  }

  for (const auto word : {0xac400440U, 0x3cc00920U, 0xb8400820U, 0xf9800020U}) {
    EXPECT_EQ(LiftMemory(word).reason, DecodeDecline::unsupported);
  }
}

TEST(A64Decode, RegisterOffsetDispatcherLoadRetainsOriginalBytesAndScaledAddress) {
  constexpr std::uint32_t word = 0xb8a8792b;  // GNU: ldrsw x11, [x9, x8, lsl #2]
  EXPECT_EQ(LiftWord(word).reason, DecodeDecline::unsupported);
  Budget budget({10000, 10000});
  const auto result =
      Decode(0x752e64, Bytes(word), budget, {MemoryProfile::concrete_atomic_scalar});
  ASSERT_TRUE(result.group);
  EXPECT_EQ(result.group->source_address(), 0x752e64);
  EXPECT_TRUE(std::ranges::equal(result.group->bytes(), Bytes(word)));
  EXPECT_EQ(result.group->memory_model(), ir::MemoryModel::atomic_scalar_reference);
  const auto nodes = result.group->nodes();
  const auto load =
      std::ranges::find_if(nodes, [](const auto& node) { return node.op == ir::Op::load; });
  ASSERT_NE(load, nodes.end());
  EXPECT_EQ(load->width, 32);
  const auto& address = nodes[load->inputs[0]];
  ASSERT_EQ(address.op, ir::Op::add);
  EXPECT_EQ(nodes[address.inputs[0]].op, ir::Op::read);
  EXPECT_EQ(nodes[address.inputs[0]].storage, 9);
  const auto& offset = nodes[address.inputs[1]];
  ASSERT_EQ(offset.op, ir::Op::shl);
  EXPECT_EQ(nodes[offset.inputs[0]].op, ir::Op::read);
  EXPECT_EQ(nodes[offset.inputs[0]].storage, 8);
  EXPECT_EQ(nodes[offset.inputs[1]].immediate, 2);
  ASSERT_EQ(result.group->writes().size(), 1);
  EXPECT_EQ(result.group->writes()[0].storage, 11);
  const auto& value = nodes[result.group->writes()[0].value];
  EXPECT_EQ(value.op, ir::Op::ashr);
  EXPECT_EQ(value.width, 64);
  EXPECT_EQ(nodes[value.inputs[1]].immediate, 32);
}

TEST(A64Decode, RegisterOffsetsImplementAllFourExtensionsAndBothScaleBits) {
  const std::uint32_t words[] = {0xf8624820, 0xf8625820, 0xf8626820, 0xf8627820,
                                 0xf862c820, 0xf862d820, 0xf862e820, 0xf862f820};
  for (unsigned index = 0; index < std::size(words); ++index) {
    const auto result = LiftMemory(words[index]);
    ASSERT_TRUE(result.group);
    const auto nodes = result.group->nodes();
    const auto load =
        std::ranges::find_if(nodes, [](const auto& node) { return node.op == ir::Op::load; });
    ASSERT_NE(load, nodes.end());
    const auto& address = nodes[load->inputs[0]];
    ASSERT_EQ(address.op, ir::Op::add);
    auto offset = address.inputs[1];
    if (index % 2) {
      ASSERT_EQ(nodes[offset].op, ir::Op::shl);
      EXPECT_EQ(nodes[nodes[offset].inputs[1]].immediate, 3);
      offset = nodes[offset].inputs[0];
    }

    if (index / 2 == 2) {
      ASSERT_EQ(nodes[offset].op, ir::Op::ashr);
      EXPECT_EQ(nodes[nodes[offset].inputs[1]].immediate, 32);
      offset = nodes[nodes[offset].inputs[0]].inputs[0];
    }

    if (index / 2 == 0 || index / 2 == 2) {
      ASSERT_EQ(nodes[offset].op, ir::Op::zext);
      EXPECT_EQ(nodes[offset].width, 64);
      offset = nodes[offset].inputs[0];
      ASSERT_EQ(nodes[offset].op, ir::Op::extract);
      EXPECT_EQ(nodes[offset].width, 32);
      offset = nodes[offset].inputs[0];
    }

    EXPECT_EQ(nodes[offset].op, ir::Op::read);
    EXPECT_EQ(nodes[offset].width, 64);
    EXPECT_EQ(nodes[offset].storage, 2);
    ASSERT_EQ(result.group->writes().size(), 1);
    EXPECT_EQ(result.group->writes()[0].storage, 0);
  }
}

TEST(A64Decode, RegisterOffsetEncodingMatrixDeclinesReservedAndUnmodeledFamilies) {
  unsigned admitted = 0, invalid = 0, unsupported = 0;
  for (unsigned size = 0; size < 4; ++size) {
    for (unsigned opc = 0; opc < 4; ++opc) {
      for (unsigned extend = 0; extend < 8; ++extend) {
        for (unsigned scale = 0; scale < 2; ++scale) {
          const auto word = 0x38200820U | (size << 30) | (opc << 22) | (2U << 16) | (extend << 13) |
                            (scale << 12);
          const auto result = LiftMemory(word);
          if ((extend & 2) == 0 || (opc == 3 && size >= 2)) {
            EXPECT_FALSE(result.group);
            EXPECT_EQ(result.reason, DecodeDecline::invalid_encoding);
            ++invalid;
          } else if (size == 3 && opc == 2) {
            EXPECT_FALSE(result.group);
            EXPECT_EQ(result.reason, DecodeDecline::unsupported);
            ++unsupported;
          } else {
            ASSERT_TRUE(result.group) << std::hex << word;
            ++admitted;
            const auto nodes = result.group->nodes();
            const auto access = std::ranges::find_if(nodes, [](const auto& node) {
              return node.op == ir::Op::load || node.op == ir::Op::store;
            });
            ASSERT_NE(access, nodes.end());
            EXPECT_EQ(access->op, opc == 0 ? ir::Op::store : ir::Op::load);
            EXPECT_EQ(access->width, 8U << size);
            EXPECT_EQ(access->access.alignment, 1);
            EXPECT_EQ(result.group->writes().size(), opc == 0 ? 0 : 1);
            if (opc >= 2) {
              auto value = result.group->writes()[0].value;
              if (opc == 3) {
                ASSERT_EQ(nodes[value].op, ir::Op::zext);
                value = nodes[value].inputs[0];
              }

              EXPECT_EQ(nodes[value].op, ir::Op::ashr);
              EXPECT_EQ(nodes[value].width, opc == 3 ? 32 : 64);
            }
          }
        }
      }
    }
  }

  EXPECT_EQ(admitted, 104);
  EXPECT_EQ(invalid, 144);
  EXPECT_EQ(unsupported, 8);

  // GNU encodings for PRFM and LDADD must not become scalar accesses.
  for (const auto word : {0xf8a26820U, 0xf8220020U}) {
    EXPECT_EQ(LiftMemory(word).reason, DecodeDecline::unsupported);
  }
}

TEST(A64Decode, RegisterOffsetSpAndZeroRegistersKeepEntryAliasSemantics) {
  for (const auto word : {0x38ffcbe0U, 0x38bf6be0U, 0x383f6bffU, 0xf87f6bffU}) {
    const auto result = LiftMemory(word);
    ASSERT_TRUE(result.group);
    unsigned sp_reads = 0;
    for (const auto& node : result.group->nodes()) {
      if (node.op == ir::Op::read && node.storage == kSp) ++sp_reads;
    }

    EXPECT_EQ(sp_reads, 1);
    for (const auto& write : result.group->writes()) EXPECT_NE(write.storage, kSp);
    if ((word & 31) == 31) {
      EXPECT_TRUE(result.group->writes().empty());
    }
  }

  for (const auto word : {0xf8616821U, 0xf8626822U}) {
    const auto result = LiftMemory(word);
    ASSERT_TRUE(result.group);
    EXPECT_EQ(std::ranges::count_if(result.group->nodes(),
                                    [](const auto& node) { return node.op == ir::Op::read; }),
              2);
    EXPECT_EQ(std::ranges::count_if(result.group->nodes(),
                                    [](const auto& node) { return node.op == ir::Op::write; }),
              0);
    ASSERT_EQ(result.group->writes().size(), 1);
    EXPECT_EQ(result.group->writes()[0].storage, word & 31);
  }
}

TEST(A64Decode, RegisterOffsetBudgetDeclinesPublishNoGroup) {
  constexpr std::uint32_t word = 0x78a2d820;  // ldrsh x0, [x1, w2, sxtw #1]
  Budget full({10000, 10000});
  ASSERT_TRUE(Decode(0x1000, Bytes(word), full, {MemoryProfile::concrete_atomic_scalar}).group);
  for (bool bytes : {false, true}) {
    const auto maximum = bytes ? full.used().bytes : full.used().work;
    for (std::uint64_t cut = 0; cut < maximum; ++cut) {
      Budget budget({bytes ? full.used().work : cut, bytes ? cut : full.used().bytes});
      const auto result =
          Decode(0x1000, Bytes(word), budget, {MemoryProfile::concrete_atomic_scalar});
      EXPECT_FALSE(result.group);
      EXPECT_EQ(result.reason, bytes ? DecodeDecline::byte_limit : DecodeDecline::work_limit);
    }
  }
}

TEST(A64Decode, PairMemoryStaysOneSourceGroupWithReferenceAccessBoundaries) {
  for (const auto word : {0xa95f8440U, 0xa9a00440U, 0x28e00440U, 0x291f8440U, 0x68c10440U}) {
    EXPECT_EQ(LiftWord(word).reason, DecodeDecline::unsupported);
    const auto result = LiftMemory(word);
    ASSERT_TRUE(result.group) << std::hex << word;
    EXPECT_EQ(result.group->memory_model(), ir::MemoryModel::atomic_scalar_reference);
    EXPECT_TRUE(std::ranges::equal(result.group->bytes(), Bytes(word)));
    EXPECT_EQ(result.group->source_address(), 0x1000);
    const auto nodes = result.group->nodes();
    std::vector<std::size_t> accesses;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      if (nodes[i].op == ir::Op::load || nodes[i].op == ir::Op::store) accesses.push_back(i);
      EXPECT_NE(nodes[i].op, ir::Op::write);
    }

    const bool wide_pair = word == 0xa95f8440U || word == 0xa9a00440U;
    ASSERT_EQ(accesses.size(), wide_pair ? 2 : 1);
    const bool load = word == 0xa95f8440U || word == 0x28e00440U || word == 0x68c10440U;
    EXPECT_EQ(nodes[accesses[0]].op, load ? ir::Op::load : ir::Op::store);
    EXPECT_EQ(nodes[accesses[0]].width, 64);
    if (wide_pair) {
      EXPECT_EQ(nodes[accesses[1]].op, nodes[accesses[0]].op);
      EXPECT_EQ(nodes[accesses[1]].width, 64);
      const auto& second_address = nodes[nodes[accesses[1]].inputs[0]];
      EXPECT_EQ(second_address.op, ir::Op::add);
      EXPECT_EQ(second_address.inputs[0], nodes[accesses[0]].inputs[0]);
      EXPECT_EQ(nodes[second_address.inputs[1]].immediate, 8);
    }

    if (word == 0xa95f8440U || word == 0x291f8440U || word == 0xa9a00440U) {
      const auto& first_address = nodes[nodes[accesses[0]].inputs[0]];
      EXPECT_EQ(first_address.op, ir::Op::add);
      const auto expected = word == 0xa95f8440U   ? 504
                            : word == 0x291f8440U ? 252
                                                  : std::uint64_t{0} - 512;
      EXPECT_EQ(nodes[first_address.inputs[1]].immediate, expected);
    } else {
      EXPECT_EQ(nodes[nodes[accesses[0]].inputs[0]].op, ir::Op::read);
    }

    EXPECT_LE(nodes.size(), 64);
    EXPECT_LE(result.group->writes().size(), 3);
  }
}

TEST(A64Decode, PairZeroRegisterAndDeferredWritebackAreSeparate) {
  const auto load = LiftMemory(0xa8c103ff);  // ldp xzr, x0, [sp], #16
  ASSERT_TRUE(load.group);
  ASSERT_EQ(load.group->writes().size(), 2);
  EXPECT_EQ(load.group->writes()[0].storage, 0);
  EXPECT_EQ(load.group->writes()[1].storage, kSp);
  const auto store = LiftMemory(0xa9bf7fff);  // stp xzr, xzr, [sp, #-16]!
  ASSERT_TRUE(store.group);
  ASSERT_EQ(store.group->writes().size(), 1);
  EXPECT_EQ(store.group->writes()[0].storage, kSp);
  for (const auto& node : store.group->nodes()) {
    if (node.op == ir::Op::store) {
      const auto& value = store.group->nodes()[node.inputs[1]];
      EXPECT_EQ(value.op, ir::Op::constant);
      EXPECT_EQ(value.immediate, 0);
    }
  }

  const auto overlap_without_writeback = LiftMemory(0xa9400400);  // ldp x0, x1, [x0]
  EXPECT_TRUE(overlap_without_writeback.group);
  const auto signed_pair = LiftMemory(0x68c10440);  // ldpsw x0, x1, [x2], #8
  ASSERT_TRUE(signed_pair.group);
  ASSERT_EQ(signed_pair.group->writes().size(), 3);
  for (unsigned i = 0; i < 2; ++i) {
    const auto& value = signed_pair.group->nodes()[signed_pair.group->writes()[i].value];
    EXPECT_EQ(value.op, ir::Op::ashr);
    EXPECT_EQ(value.width, 64);
  }
}

TEST(A64Decode, PairProfileDeclinesUnsafeAndOtherMemoryFamilies) {
  for (const auto word : {0xa9400040U, 0xa8c10400U, 0xa9810420U, 0xe9400440U}) {
    EXPECT_EQ(LiftMemory(word).reason, DecodeDecline::invalid_encoding) << std::hex << word;
  }

  for (const auto word : {0xa8400440U, 0xa8000440U, 0xac400440U, 0x69000440U, 0xc85f7c20U}) {
    EXPECT_EQ(LiftMemory(word).reason, DecodeDecline::unsupported) << std::hex << word;
  }
}

TEST(A64Decode, DirectTransfersPreserveSignedOffsetsAndRuntimeImageBase) {
  for (const auto link : {false, true}) {
    for (const auto immediate : {0U, 1U, 0x1ffffffU, 0x2000000U, 0x3ffffffU}) {
      const auto word = (link ? 0x94000000U : 0x14000000U) | immediate;
      const auto result = LiftWord(word);
      ASSERT_TRUE(result.group);
      ASSERT_TRUE(result.group->transfer());
      const auto& transfer = *result.group->transfer();
      EXPECT_TRUE(ir::ValidTransfer(transfer, result.group->nodes()));
      EXPECT_EQ(transfer.kind, link ? ir::TransferKind::call : ir::TransferKind::jump);
      const auto nodes = result.group->nodes();
      const auto& target = nodes[transfer.target];
      ASSERT_EQ(target.op, ir::Op::add);
      EXPECT_EQ(nodes[target.inputs[0]].op, ir::Op::image_address);
      EXPECT_EQ(nodes[target.inputs[0]].immediate, 0xc687bc);
      const auto displacement = immediate < 0x2000000U
                                    ? std::uint64_t(immediate) * 4
                                    : std::uint64_t{0} - std::uint64_t(0x4000000U - immediate) * 4;
      EXPECT_EQ(nodes[target.inputs[1]].immediate, displacement);
      if (link) {
        ASSERT_EQ(result.group->writes().size(), 1);
        EXPECT_EQ(result.group->writes()[0].storage, 30);
        EXPECT_EQ(result.group->writes()[0].value, *transfer.continuation);
        const auto& continuation = nodes[*transfer.continuation];
        EXPECT_EQ(nodes[continuation.inputs[0]].op, ir::Op::image_address);
        EXPECT_EQ(nodes[continuation.inputs[1]].immediate, 4);
      } else {
        EXPECT_TRUE(result.group->writes().empty());
      }

      EXPECT_TRUE(std::ranges::equal(result.group->bytes(), Bytes(word)));
    }
  }
}

TEST(A64Decode, ConditionalBranchesKeepBothEdgesWithoutWritingFlags) {
  for (unsigned condition = 0; condition < 16; ++condition) {
    for (const auto immediate : {0U, 0x3ffffU, 0x40000U, 0x7ffffU}) {
      const auto result = LiftWord(0x54000000U | (immediate << 5) | condition);
      ASSERT_TRUE(result.group);
      ASSERT_TRUE(result.group->transfer());
      const auto& transfer = *result.group->transfer();
      EXPECT_TRUE(ir::ValidTransfer(transfer, result.group->nodes()));
      EXPECT_EQ(transfer.kind, ir::TransferKind::conditional);
      EXPECT_TRUE(result.group->writes().empty());
      const auto nodes = result.group->nodes();
      const auto displacement = immediate < 0x40000U
                                    ? std::uint64_t(immediate) * 4
                                    : std::uint64_t{0} - std::uint64_t(0x80000U - immediate) * 4;
      EXPECT_EQ(nodes[nodes[transfer.target].inputs[1]].immediate, displacement);
      EXPECT_EQ(nodes[nodes[*transfer.alternative].inputs[1]].immediate, 4);
      if (condition >= 14) {
        EXPECT_EQ(nodes[*transfer.condition].op, ir::Op::constant);
        EXPECT_EQ(nodes[*transfer.condition].immediate, 1);
      }
    }
  }
}

TEST(A64Decode, CompareAndTestBranchesRespectWidthsAndZeroRegister) {
  for (const auto word :
       {0x34000020U, 0xb5000020U, 0x3400003fU, 0x36f80020U, 0xb7f80020U, 0x3600003fU}) {
    const auto result = LiftWord(word);
    ASSERT_TRUE(result.group);
    ASSERT_TRUE(result.group->transfer());
    EXPECT_TRUE(ir::ValidTransfer(*result.group->transfer(), result.group->nodes()));
    EXPECT_TRUE(result.group->writes().empty());
    for (const auto& node : result.group->nodes()) {
      if (node.op == ir::Op::read) {
        EXPECT_NE(node.storage, kSp);
      }
    }

    const auto nodes = result.group->nodes();
    if (word == 0x36f80020U || word == 0xb7f80020U) {
      const auto bit = std::ranges::find_if(
          nodes, [](const auto& node) { return node.op == ir::Op::extract && node.width == 1; });
      ASSERT_NE(bit, nodes.end());
      EXPECT_EQ(bit->immediate, word == 0x36f80020U ? 31 : 63);
      EXPECT_EQ(nodes[bit->inputs[0]].width, word == 0x36f80020U ? 32 : 64);
    }
  }

  for (const auto immediate : {0x1fffU, 0x2000U, 0x3fffU}) {
    const auto result = LiftWord(0x36000000U | (immediate << 5));
    ASSERT_TRUE(result.group);
    const auto nodes = result.group->nodes();
    const auto displacement = immediate < 0x2000U
                                  ? std::uint64_t(immediate) * 4
                                  : std::uint64_t{0} - std::uint64_t(0x4000U - immediate) * 4;
    EXPECT_EQ(nodes[nodes[result.group->transfer()->target].inputs[1]].immediate, displacement);
  }
}

TEST(A64Decode, IndirectCallsAndReturnsUseExplicitEntryRegisters) {
  for (const auto word : {0xd61f0000U, 0xd63f03c0U, 0xd65f03c0U, 0xd65f0000U}) {
    const auto result = LiftWord(word);
    ASSERT_TRUE(result.group);
    ASSERT_TRUE(result.group->transfer());
    const auto& transfer = *result.group->transfer();
    EXPECT_TRUE(ir::ValidTransfer(transfer, result.group->nodes()));
    const auto& target = result.group->nodes()[transfer.target];
    EXPECT_EQ(target.op, ir::Op::read);
    EXPECT_EQ(target.storage, (word >> 5) & 31);
    if (word == 0xd63f03c0U) {
      EXPECT_EQ(transfer.kind, ir::TransferKind::call);
      ASSERT_EQ(result.group->writes().size(), 1);
      EXPECT_EQ(result.group->writes()[0].storage, 30);
      EXPECT_EQ(result.group->writes()[0].value, *transfer.continuation);
      EXPECT_NE(transfer.target, *transfer.continuation);
    } else {
      EXPECT_TRUE(result.group->writes().empty());
      EXPECT_EQ(transfer.kind,
                word == 0xd61f0000U ? ir::TransferKind::jump : ir::TransferKind::return_);
    }
  }
}

TEST(A64Decode, NewerAndReservedControlEncodingsRemainUnsupported) {
  for (const auto word : {0x54000010U, 0x55000000U, 0xd61f0001U, 0xd63f0400U, 0xd65f0bffU,
                          0xd65f0fffU, 0xd71f0800U, 0xd73f0800U, 0xd69f03e0U, 0xd6bf03e0U}) {
    EXPECT_EQ(LiftWord(word).reason, DecodeDecline::unsupported) << std::hex << word;
  }

  for (const auto word : {0x94000000U, 0x54000000U, 0xd63f03c0U}) {
    Budget full({10000, 10000});
    const auto result = Decode(0x1000, Bytes(word), full);
    ASSERT_TRUE(result.group);
    Budget short_budget({full.used().work - 1, full.used().bytes});
    const auto short_result = Decode(0x1000, Bytes(word), short_budget);
    EXPECT_FALSE(short_result.group);
    EXPECT_EQ(short_result.reason, DecodeDecline::work_limit);
  }
}

TEST(A64Decode, SourceLocationsAreAlignedAndDoNotWrap) {
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  for (const auto address :
       {std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{3}, maximum - 2, maximum - 1, maximum}) {
    Budget budget({1000, 10000});
    const auto result = Decode(address, Bytes(0xd503201f), budget);
    EXPECT_FALSE(result.group);
    EXPECT_EQ(result.reason, DecodeDecline::invalid_location);
    EXPECT_EQ(budget.used().bytes, 0);
    EXPECT_EQ(budget.used().work, 0);
  }

  for (const auto address : {std::uint64_t{0}, maximum - 3}) {
    Budget budget({1000, 10000});
    const auto result = Decode(address, Bytes(0xd503201f), budget);
    ASSERT_TRUE(result.group);
    EXPECT_EQ(result.group->source_address(), address);
  }
}

TEST(A64Decode, BudgetsCannotProducePartialAuthority) {
  Budget bytes({1000, 0});
  EXPECT_EQ(Decode(0, Bytes(0xab020020), bytes).reason, DecodeDecline::byte_limit);
  EXPECT_EQ(bytes.used().bytes, 0);
  Budget work({1, 10000});
  const auto result = Decode(0, Bytes(0xab020020), work);
  EXPECT_FALSE(result.group);
  EXPECT_EQ(result.reason, DecodeDecline::work_limit);
  EXPECT_EQ(work.used().work, 1);
}

TEST(A64Decode, LogicalImmediateEncodingSpaceHasNoZeroOrAllOnesMasks) {
  for (const unsigned width : {32U, 64U}) {
    unsigned admitted = 0;
    std::set<std::uint64_t> masks;
    for (unsigned encoded = 0; encoded < 8192; ++encoded) {
      const auto word = 0x12000020U | (width == 64 ? 0x80000000U : 0) | (encoded << 10);
      const auto result = LiftWord(word);
      if (!result.group) {
        EXPECT_EQ(result.reason, DecodeDecline::invalid_encoding);
        continue;
      }

      ++admitted;
      const auto nodes = result.group->nodes();
      const auto operation =
          std::ranges::find_if(nodes, [](const auto& node) { return node.op == ir::Op::bit_and; });
      ASSERT_NE(operation, nodes.end());
      const auto mask = nodes[operation->inputs[1]].immediate;
      EXPECT_NE(mask, 0);
      EXPECT_NE(mask, width == 64 ? UINT64_MAX : UINT64_C(0xffffffff));
      if (width == 32) {
        EXPECT_EQ(mask >> 32, 0);
      }

      masks.insert(mask);
    }

    // Rotations above the element size are aliases, not reserved encodings.
    EXPECT_EQ(admitted, width == 64 ? 7680U : 3648U);
    EXPECT_EQ(masks.size(), width == 64 ? 5334U : 1302U);
  }
}

TEST(A64Decode, LogicalImmediateBudgetDeclinesPublishNoGroup) {
  constexpr std::uint32_t word = 0xf240003f;  // tst x1, #1
  Budget full({10000, 10000});
  ASSERT_TRUE(Decode(0x1000, Bytes(word), full).group);
  for (std::uint64_t work = 0; work < full.used().work; ++work) {
    Budget short_budget({work, 10000});
    const auto result = Decode(0x1000, Bytes(word), short_budget);
    EXPECT_FALSE(result.group);
    EXPECT_EQ(result.reason, DecodeDecline::work_limit);
  }
}

TEST(A64Decode, FullBitfieldEncodingMatrixRejectsReservedFields) {
  std::uint64_t admitted = 0, invalid = 0;
  for (unsigned sf = 0; sf < 2; ++sf) {
    const unsigned width = sf ? 64 : 32;
    for (unsigned opc = 0; opc < 4; ++opc) {
      for (unsigned n = 0; n < 2; ++n) {
        for (unsigned r = 0; r < 64; ++r) {
          for (unsigned s = 0; s < 64; ++s) {
            const auto word =
                0x13000020U | (sf << 31) | (opc << 29) | (n << 22) | (r << 16) | (s << 10);
            const auto result = LiftWord(word);
            if (n == sf && r < width && s < width && opc < 3) {
              ASSERT_TRUE(result.group) << std::hex << word;
              ++admitted;
              ASSERT_EQ(result.group->writes().size(), 1);
              EXPECT_EQ(result.group->writes()[0].storage, 0);
            } else {
              ASSERT_FALSE(result.group) << std::hex << word;
              EXPECT_EQ(result.reason, DecodeDecline::invalid_encoding);
              ++invalid;
            }
          }
        }
      }
    }
  }

  EXPECT_EQ(admitted, 15360);
  EXPECT_EQ(invalid, 50176);
}

TEST(A64Decode, BitfieldAliasesExecuteSignedWrappedAndPreservedDestinationCases) {
  struct Example {
    std::uint32_t word;
    std::uint64_t source, destination, expected;
  };

  // GNU assembler independently encodes the aliases; expected values below are
  // explicit examples, not a second implementation of DecodeBitMasks.
  const Example examples[] = {
      {0x13001c20, 0x80, UINT64_MAX, 0xffffff80},                    // sxtb w0,w1
      {0x93401c20, 0x80, 0, 0xffffffffffffff80},                     // sxtb x0,w1
      {0x93407c20, 0x80000000, 0, 0xffffffff80000000},               // sxtw x0,w1
      {0xd3484c20, 0x123456789abcdef0, 0, 0xcde},                    // ubfx x0,x1,#8,#12
      {0xd3782c20, 0x123456789abcdef0, 0, 0xef000},                  // ubfiz x0,x1,#8,#12
      {0x93483c20, 0x8000, 0, 0xffffffffffffff80},                   // sbfx x0,x1,#8,#8
      {0x93781c20, 0x80, 0, 0xffffffffffff8000},                     // sbfiz x0,x1,#8,#8
      {0xb3483c20, 0xaabb, 0x1122334455667788, 0x11223344556677aa},  // bfxil x0,x1,#8,#8
      {0xb3781c20, 0xaa, 0x1122334455667788, 0x112233445566aa88},    // bfi x0,x1,#8,#8
      {0xb3781c00, 0, 0x1122334455667788, 0x1122334455668888},       // bfi x0,x0,#8,#8
      {0xb3781fe0, 0, 0x1122334455667788, 0x1122334455660088},       // bfc x0,#8,#8
      {0x934527e0, UINT64_MAX, UINT64_MAX, 0},                       // sbfx x0,xzr,#5,#5
      {0x33007c20, 0xfedcba9876543210, UINT64_MAX, 0x76543210},      // bfm w0,w1,#0,#31
      {0xb340fc20, 0xfedcba9876543210, 0, 0xfedcba9876543210},       // bfm x0,x1,#0,#63
      {0x93400020, 1, 0, UINT64_MAX},                                // sbfm x0,x1,#0,#0
      {0x93400020, 2, UINT64_MAX, 0},                                // sign from bit0, not bit63
  };

  for (const auto& example : examples) {
    auto result = LiftWord(example.word);
    ASSERT_TRUE(result.group) << std::hex << example.word;
    Budget budget({100000, 100000});
    eval::State state;
    for (unsigned reg = 0; reg < 36; ++reg) {
      const auto value = reg == 0   ? example.destination
                         : reg == 1 ? example.source
                         : reg < 32 ? 0x55U
                                    : 1U;
      auto bits = BitVector::from_u64(reg < 32 ? 64 : 1, value, 64, budget);
      ASSERT_TRUE(bits);
      state.cells.push_back({reg, std::move(*bits)});
    }

    EXPECT_EQ(eval::Execute(*result.group, state, budget), eval::Outcome::completed);
    EXPECT_EQ(state.cells[0].value.word(0), example.expected) << std::hex << example.word;
    EXPECT_EQ(state.cells[1].value.word(0), example.source);
    EXPECT_EQ(state.cells[kSp].value.word(0), 0x55);
    for (unsigned flag = kN; flag <= kV; ++flag) EXPECT_EQ(state.cells[flag].value.word(0), 1);
  }
}

TEST(A64Decode, BitfieldRegister31IsAlwaysZeroRatherThanStackPointer) {
  for (unsigned opc = 0; opc < 3; ++opc) {
    for (const unsigned width : {32U, 64U}) {
      for (const bool zero_source : {false, true}) {
        const auto word = 0x1300001fU | (opc << 29) | (width == 64 ? 0x80400000U : 0U) |
                          (5U << 16) | (9U << 10) | ((zero_source ? 31U : 1U) << 5);
        const auto result = LiftWord(word);
        ASSERT_TRUE(result.group);
        EXPECT_TRUE(result.group->writes().empty());
        for (const auto& node : result.group->nodes())
          if (node.op == ir::Op::read) {
            EXPECT_NE(node.storage, kSp);
          }
      }
    }
  }
}

TEST(A64Decode, FullBitfieldBudgetCutsPublishNoPartialGroup) {
  for (const auto word : {0x93781c20U, 0xb3781c00U, 0xd3782c20U}) {
    Budget budget({10000, 10000});
    ASSERT_TRUE(Decode(0x1000, Bytes(word), budget).group);
    const auto required = budget.used();
    for (std::uint64_t work = 0; work < required.work; ++work) {
      Budget limited({work, UINT64_MAX});
      const auto result = Decode(0x1000, Bytes(word), limited);
      EXPECT_FALSE(result.group);
      EXPECT_EQ(result.reason, DecodeDecline::work_limit);
    }

    for (std::uint64_t bytes = 0; bytes < required.bytes; ++bytes) {
      Budget limited({UINT64_MAX, bytes});
      const auto result = Decode(0x1000, Bytes(word), limited);
      EXPECT_FALSE(result.group);
      EXPECT_EQ(result.reason, DecodeDecline::byte_limit);
    }

    Budget exact(required);
    EXPECT_TRUE(Decode(0x1000, Bytes(word), exact).group);
  }
}

TEST(A64Decode, VariableShiftAdmissionKeepsNeighboringOpcodesUnsupported) {
  for (const auto sf : {0U, 1U}) {
    for (unsigned opcode = 0; opcode < 64; ++opcode) {
      const auto word = 0x1ac20020U | (sf << 31) | (opcode << 10);
      const auto result = LiftWord(word);

      // UDIV and SDIV sit beside the shifts.
      if ((opcode >= 8 && opcode <= 11) || opcode == 2 || opcode == 3) {
        ASSERT_TRUE(result.group) << std::hex << word;
        EXPECT_EQ(result.group->writes().size(), 1);
      } else {
        EXPECT_FALSE(result.group) << std::hex << word;
        EXPECT_EQ(result.reason, DecodeDecline::unsupported);
      }
    }

    for (const auto fixed_bit : {21U, 29U, 30U}) {
      const auto result = LiftWord((0x1ac22020U | (sf << 31)) ^ (1U << fixed_bit));
      EXPECT_FALSE(result.group);
      EXPECT_EQ(result.reason, DecodeDecline::unsupported);
    }
  }
}

// x0 after running `word` with x1 and x2 set and every other register 0x55.
std::uint64_t RunWithSources(std::uint32_t word, std::uint64_t x1, std::uint64_t x2) {
  const auto result = LiftWord(word);
  EXPECT_TRUE(result.group) << std::hex << word;
  if (!result.group) return 0;
  Budget budget({100000, 100000});
  eval::State state;
  for (unsigned reg = 0; reg < 36; ++reg) {
    const auto value = reg == 1 ? x1 : reg == 2 ? x2 : reg < 32 ? 0x55U : 1U;
    auto bits = BitVector::from_u64(reg < 32 ? 64 : 1, value, 64, budget);
    EXPECT_TRUE(bits);
    state.cells.push_back({reg, std::move(*bits)});
  }

  EXPECT_EQ(eval::Execute(*result.group, state, budget), eval::Outcome::completed);
  EXPECT_EQ(state.cells[1].value.word(0), x1);
  EXPECT_EQ(state.cells[2].value.word(0), x2);
  for (unsigned flag = kN; flag <= kV; ++flag) EXPECT_EQ(state.cells[flag].value.word(0), 1);
  return state.cells[0].value.word(0);
}

TEST(A64Decode, DivisionAndHighMultiplyFollowA64RatherThanC) {
  struct Example {
    std::uint32_t word;
    std::uint64_t x1, x2, expected;
  };

  constexpr auto kMin = UINT64_C(1) << 63;

  // GNU assembler encodings with rd = 0, rn = 1, rm = 2; the expected values
  // are worked by hand, not by a second implementation.
  const Example examples[] = {
      {0x1ac20820, 7, 2, 3},                             // udiv w
      {0x1ac20820, 7, 0, 0},                             // zero divisor
      {0x1ac20820, 0xffffffff00000009, 0x100000003, 3},  // W reads the low half
      {0x1ac20c20, 0xfffffff9, 2, 0xfffffffd},           // sdiv w: -7 / 2 = -3
      {0x1ac20c20, 7, 0xfffffffe, 0xfffffffd},           // 7 / -2 = -3
      {0x1ac20c20, 0x80000000, 0xffffffff, 0x80000000},  // INT_MIN / -1
      {0x1ac20c20, 0x80000000, 0, 0},
      {0x9ac20820, UINT64_MAX, 2, 0x7fffffffffffffff},  // udiv x
      {0x9ac20820, UINT64_MAX, 0, 0},
      {0x9ac20c20, 0xfffffffffffffff9, 2, 0xfffffffffffffffd},  // sdiv x
      {0x9ac20c20, kMin, UINT64_MAX, kMin},
      {0x9ac20c20, kMin, 0, 0},
      {0x9bc27c20, UINT64_MAX, UINT64_MAX, 0xfffffffffffffffe},  // umulh
      {0x9bc27c20, kMin, 4, 2},
      {0x9bc27c20, 0x123456789abcdef0, 0, 0},
      {0x9b427c20, UINT64_MAX, UINT64_MAX, 0},      // smulh: -1 * -1
      {0x9b427c20, UINT64_MAX, 5, UINT64_MAX},      // -5 extends its sign
      {0x9b427c20, kMin, kMin, UINT64_C(1) << 62},  // 2^126
      {0x9b427c20, kMin, 2, UINT64_MAX},            // -2^64
  };

  for (const auto& example : examples)
    EXPECT_EQ(RunWithSources(example.word, example.x1, example.x2), example.expected)
        << std::hex << example.word << ' ' << example.x1 << ' ' << example.x2;
}

TEST(A64Decode, BitCountsAndReversalsExecuteExactly) {
  struct Example {
    std::uint32_t word;
    std::uint64_t x1, expected;
  };

  const Example examples[] = {
      {0x5ac01020, 0, 32},  // clz w
      {0x5ac01020, 0x0000ffff00010000, 15},
      {0xdac01020, 0, 64},  // clz x
      {0xdac01020, 1, 63},
      {0xdac01020, UINT64_C(1) << 63, 0},
      {0x5ac01420, 0, 31},  // cls w
      {0x5ac01420, 0xffffffff, 31},
      {0x5ac01420, 0x40000000, 0},
      {0x5ac01420, 0x80000000, 0},
      {0x5ac01420, 0xc0000000, 1},
      {0x5ac01420, 1, 30},
      {0xdac01420, 0, 63},  // cls x
      {0xdac01420, UINT64_MAX, 63},
      {0xdac01420, 0xff00000000000000, 7},
      {0xdac01420, 1, 62},
      {0x5ac00020, 1, 0x80000000},  // rbit w
      {0x5ac00020, 0x12345678, 0x1e6a2c48},
      {0xdac00020, 1, UINT64_C(1) << 63},  // rbit x
      {0xdac00020, 0x0123456789abcdef, 0xf7b3d591e6a2c480},
      {0x5ac00420, 0xaabbccdd11223344, 0x22114433},          // rev16 w
      {0x5ac00820, 0xaabbccdd11223344, 0x44332211},          // rev w
      {0xdac00420, 0x1122334455667788, 0x2211443366558877},  // rev16 x
      {0xdac00820, 0x1122334455667788, 0x4433221188776655},  // rev32 x
      {0xdac00c20, 0x1122334455667788, 0x8877665544332211},  // rev x
  };

  for (const auto& example : examples)
    EXPECT_EQ(RunWithSources(example.word, example.x1, 0), example.expected)
        << std::hex << example.word << ' ' << example.x1;
}

TEST(A64Decode, OneSourceAndHighMultiplyAdmissionKeepsNeighboursUnsupported) {
  for (const auto sf : {0U, 1U}) {
    for (unsigned opcode = 0; opcode < 64; ++opcode) {
      const auto word = 0x5ac00020U | (sf << 31) | (opcode << 10);

      // A 32-bit REV of the whole register is REV itself at opcode 2; opcode 3
      // is unallocated there. CTZ, CNT and ABS are left out.
      const bool lifted = opcode < 6 && (sf || opcode != 3);
      EXPECT_EQ(static_cast<bool>(LiftWord(word).group), lifted) << std::hex << word;
    }

    // A nonzero opcode2 or S bit is not these instructions.
    EXPECT_FALSE(LiftWord(0x5ac10020U | (sf << 31)).group);
    EXPECT_FALSE(LiftWord(0x7ac01020U | (sf << 31)).group);
  }

  // Ra other than 31 and o0 set are unallocated for UMULH and SMULH, and the
  // 32-bit form does not exist.
  for (const auto word : {0x9bc27820U, 0x9b42fc20U, 0x9bc2fc20U, 0x1bc27c20U, 0x1b427c20U}) {
    const auto result = LiftWord(word);
    EXPECT_FALSE(result.group) << std::hex << word;
  }
}

TEST(A64Decode, ExtractRegisterUsesTwoEntryOperandsAndRejectsReservedForms) {
  for (const unsigned width : {32U, 64U}) {
    const auto base = width == 64 ? 0x93c00000U : 0x13800000U;
    const auto mask = width == 64 ? UINT64_MAX : UINT64_C(0xffffffff);
    for (const unsigned amount : {0U, 1U, width - 1}) {
      const auto word = base | (2U << 16) | (amount << 10) | (1U << 5);
      const auto result = LiftWord(word);
      ASSERT_TRUE(result.group) << std::hex << word;
      Budget budget({100000, 100000});
      eval::State state;
      for (unsigned reg = 0; reg < 36; ++reg) {
        const auto value = reg == 1   ? UINT64_C(0xfedcba9876543210)
                           : reg == 2 ? UINT64_C(0x0123456789abcdef)
                           : reg < 32 ? UINT64_C(0xa5)
                                      : reg & 1U;
        auto bits = BitVector::from_u64(reg < 32 ? 64 : 1, value, 64, budget);
        ASSERT_TRUE(bits);
        state.cells.push_back({reg, std::move(*bits)});
      }

      const auto high = UINT64_C(0xfedcba9876543210) & mask;
      const auto low = UINT64_C(0x0123456789abcdef) & mask;
      const auto expected = ((low >> amount) | (amount ? high << (width - amount) : 0)) & mask;
      EXPECT_EQ(eval::Execute(*result.group, state, budget), eval::Outcome::completed);
      EXPECT_EQ(state.cells[0].value.word(0), expected) << width << ':' << amount;
      EXPECT_EQ(state.cells[1].value.word(0), UINT64_C(0xfedcba9876543210));
      EXPECT_EQ(state.cells[2].value.word(0), UINT64_C(0x0123456789abcdef));
      EXPECT_EQ(state.cells[kSp].value.word(0), 0xa5);
    }

    EXPECT_EQ(LiftWord((base ^ (1U << 22)) | (2U << 16) | (1U << 5)).reason,
              DecodeDecline::invalid_encoding);
    if (width == 32) {
      EXPECT_EQ(LiftWord(base | (2U << 16) | (32U << 10) | (1U << 5)).reason,
                DecodeDecline::invalid_encoding);
    }
  }
}

TEST(A64Decode, VariableShiftCountsAreModuloWidthAndLeaveFlagsAndSpUnchanged) {
  // GNU encodings for lslv/lsrv/asrv/rorv w0,w1,w2; sf selects the X forms.
  for (const unsigned width : {32U, 64U}) {
    const auto high = UINT64_C(1) << (width - 1);
    const auto mask = width == 64 ? UINT64_MAX : UINT64_C(0xffffffff);
    const auto source = high | 1;
    const std::uint64_t counts[] = {0,          1,
                                    width - 1,  width,
                                    width + 1,  2 * width - 1,
                                    UINT64_MAX, (UINT64_C(1) << 63) | (width + 1)};
    for (unsigned kind = 0; kind < 4; ++kind) {
      const auto result = LiftWord(0x1ac22020U | (width == 64 ? 0x80000000U : 0U) | (kind << 10));
      ASSERT_TRUE(result.group);
      for (const auto count : counts) {
        Budget budget({100000, 100000});
        eval::State state;
        for (unsigned reg = 0; reg < 36; ++reg) {
          const auto value = reg == 1   ? source
                             : reg == 2 ? count
                             : reg < 32 ? UINT64_C(0xa5)
                                        : reg & 1U;
          auto bits = BitVector::from_u64(reg < 32 ? 64 : 1, value, 64, budget);
          ASSERT_TRUE(bits);
          state.cells.push_back({reg, std::move(*bits)});
        }

        const auto remainder = count & (width - 1);
        std::uint64_t expected = source;
        if (remainder == 1) {
          const std::uint64_t one[] = {2, high >> 1, high | (high >> 1), high | (high >> 1)};
          expected = one[kind];
        } else if (remainder == width - 1) {
          const std::uint64_t last[] = {high, 1, mask, 3};
          expected = last[kind];
        }

        EXPECT_EQ(eval::Execute(*result.group, state, budget), eval::Outcome::completed);
        EXPECT_EQ(state.cells[0].value.word(0), expected) << width << ':' << kind << ':' << count;
        EXPECT_EQ(state.cells[1].value.word(0), source);
        EXPECT_EQ(state.cells[2].value.word(0), count);
        EXPECT_EQ(state.cells[kSp].value.word(0), 0xa5);
        for (unsigned flag = kN; flag <= kV; ++flag)
          EXPECT_EQ(state.cells[flag].value.word(0), flag & 1U);
      }
    }
  }
}

TEST(A64Decode, VariableShiftZeroRegistersAndOverlapsUseEntryValues) {
  struct Registers {
    unsigned rd, rn, rm;
    std::uint64_t expected;
  };

  const Registers cases[] = {{0, 31, 2, 0}, {0, 1, 31, 3}, {31, 1, 2, 0xa5},
                             {1, 1, 2, 12}, {2, 1, 2, 12}, {1, 1, 1, 24}};
  for (const unsigned width : {32U, 64U}) {
    for (const auto& example : cases) {
      const auto word = 0x1ac02000U | (width == 64 ? 0x80000000U : 0U) | (example.rm << 16) |
                        (example.rn << 5) | example.rd;
      const auto result = LiftWord(word);
      ASSERT_TRUE(result.group);
      Budget budget({100000, 100000});
      eval::State state;
      for (unsigned reg = 0; reg < 36; ++reg) {
        const auto value = reg == 1 ? 3U : reg == 2 ? 2U : reg < 32 ? 0xa5U : 0U;
        auto bits = BitVector::from_u64(reg < 32 ? 64 : 1, value, 64, budget);
        ASSERT_TRUE(bits);
        state.cells.push_back({reg, std::move(*bits)});
      }

      EXPECT_EQ(eval::Execute(*result.group, state, budget), eval::Outcome::completed);
      EXPECT_EQ(state.cells[example.rd].value.word(0), example.expected);
      EXPECT_EQ(state.cells[kSp].value.word(0), 0xa5);
      for (const auto& node : result.group->nodes())
        if (node.op == ir::Op::read) {
          EXPECT_NE(node.storage, kSp);
        }
    }
  }
}

TEST(A64Decode, VariableShiftBudgetsDeclineWithoutPartialGroups) {
  for (const auto base : {0x1ac22020U, 0x9ac22020U}) {
    for (unsigned kind = 0; kind < 4; ++kind) {
      const auto word = base | (kind << 10);
      Budget budget({10000, 10000});
      ASSERT_TRUE(Decode(0x1000, Bytes(word), budget).group);
      const auto required = budget.used();
      for (std::uint64_t work = 0; work < required.work; ++work) {
        Budget limited({work, UINT64_MAX});
        const auto result = Decode(0x1000, Bytes(word), limited);
        EXPECT_FALSE(result.group);
        EXPECT_EQ(result.reason, DecodeDecline::work_limit);
      }

      for (std::uint64_t bytes = 0; bytes < required.bytes; ++bytes) {
        Budget limited({UINT64_MAX, bytes});
        const auto result = Decode(0x1000, Bytes(word), limited);
        EXPECT_FALSE(result.group);
        EXPECT_EQ(result.reason, DecodeDecline::byte_limit);
      }

      Budget exact(required);
      EXPECT_TRUE(Decode(0x1000, Bytes(word), exact).group);
    }
  }
}

// x0..x30, SP, NZCV, TPIDR_EL0 and q0..q31, each cell at the index of its
// storage. Q registers hold distinct nonzero patterns in both halves, so a
// narrower write that leaves upper bits behind shows.
eval::State VectorState(Budget& budget, std::uint64_t x1, unsigned nzcv = 0) {
  eval::State state;
  for (ir::StorageId id = 0; id < kQ0 + 32; ++id) {
    const std::array<std::uint64_t, 2> words{0xa7a6a5a4a3a2a1a0 + id, 0xafaeadacabaaa9a8 + id};
    auto bits = id >= kQ0 ? BitVector::from_words(128, words, 128, budget)
                : id >= kN && id <= kV
                    ? BitVector::from_u64(1, (nzcv >> (kV - id)) & 1, 64, budget)
                    : BitVector::from_u64(64, id == 1 ? x1 : 0x1000 + id, 64, budget);
    EXPECT_TRUE(bits);
    state.cells.push_back({id, std::move(*bits)});
  }

  return state;
}

unsigned Nzcv(const eval::State& state) {
  unsigned flags = 0;
  for (ir::StorageId id = kN; id <= kV; ++id) flags = (flags << 1) | state.cells[id].value.word(0);
  return flags;
}

TEST(A64Decode, ConditionalCompareSelectsComparisonFlagsOrTheImmediate) {
  for (const auto word :
       {0xfa420025U, 0x3a5fb86aU}) {  // ccmp x1, x2, #5, eq; ccmn w3, #31, #10, lt
    const auto result = LiftWord(word);
    ASSERT_TRUE(result.group) << std::hex << word;
    const auto writes = result.group->writes();
    ASSERT_EQ(writes.size(), 4);
    for (unsigned i = 0; i < 4; ++i) {
      EXPECT_EQ(writes[i].storage, kN + i);
      EXPECT_EQ(result.group->nodes()[writes[i].value].op, ir::Op::select);
    }
  }

  struct Case {
    std::uint32_t word;
    std::uint64_t x1, x2, x3;
    unsigned entry, expected;
  };

  const Case cases[] = {
      // Equal operands: the condition holding gives Z and C; failing gives #5.
      {0xfa420025U, 7, 7, 0, 0b0100, 0b0110},
      {0xfa420025U, 7, 7, 0, 0b0000, 0b0101},
      {0xfa420025U, 0, 1, 0, 0b0100, 0b1000},
      // 0x7fffffe1 + 31 overflows the W register into its sign bit.
      {0x3a5fb86aU, 0, 0, 0xffffffff7fffffe1, 0b1000, 0b1001},
      {0x3a5fb86aU, 0, 0, 0xffffffff7fffffe1, 0b1001, 0b1010},
      {0x3a5fb86aU, 0, 0, 0xffffffffffffffe1, 0b0001, 0b0110},
  };

  for (const auto& c : cases) {
    const auto result = LiftWord(c.word);
    ASSERT_TRUE(result.group);
    Budget budget({1000000, 1000000});
    auto state = VectorState(budget, c.x1, c.entry);
    for (const auto [reg, value] : {std::pair{2U, c.x2}, std::pair{3U, c.x3}}) {
      auto bits = BitVector::from_u64(64, value, 64, budget);
      ASSERT_TRUE(bits);
      state.cells[reg].value = std::move(*bits);
    }

    EXPECT_EQ(eval::Execute(*result.group, state, budget), eval::Outcome::completed);
    EXPECT_EQ(Nzcv(state), c.expected) << std::hex << c.word << ' ' << c.entry;
  }

  // o3 and a clear S bit are other encodings.
  for (const auto word : {0xfa420035U, 0xda420025U}) {
    EXPECT_EQ(LiftWord(word).reason, DecodeDecline::unsupported) << std::hex << word;
  }
}

TEST(A64Decode, SimdFpLoadsWriteWholeQRegistersAtEveryWidth) {
  struct Case {
    std::uint32_t word;
    unsigned rt, width, offset;
  };

  const Case loads[] = {{0x3d400c20U, 0, 8, 3},      // ldr b0, [x1, #3]
                        {0x7d400c21U, 1, 16, 6},     // ldr h1, [x1, #6]
                        {0xbd400c22U, 2, 32, 12},    // ldr s2, [x1, #12]
                        {0xfd400c23U, 3, 64, 24},    // ldr d3, [x1, #24]
                        {0x3dc00824U, 4, 128, 32}};  // ldr q4, [x1, #32]
  constexpr std::uint64_t base = 0x100000000;
  std::array<std::uint8_t, 64> bytes{};
  for (unsigned i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(0x40 + i);
  for (const auto& load : loads) {
    EXPECT_EQ(LiftWord(load.word).reason, DecodeDecline::unsupported);
    const auto result = LiftMemory(load.word);
    ASSERT_TRUE(result.group) << std::hex << load.word;
    EXPECT_EQ(result.group->memory_model(), ir::MemoryModel::atomic_scalar_reference);
    ASSERT_EQ(result.group->writes().size(), 1);
    EXPECT_EQ(result.group->writes()[0].storage, kQ0 + load.rt);
    const auto nodes = result.group->nodes();
    EXPECT_EQ(nodes[result.group->writes()[0].value].width, 128);
    const auto accesses = std::ranges::count_if(nodes, [&](const auto& node) {
      return node.op == ir::Op::load && node.width == std::min(load.width, 64U);
    });
    EXPECT_EQ(accesses, load.width == 128 ? 2 : 1);

    Budget budget({1000000, 1000000});
    auto state = VectorState(budget, base);
    const eval::RegionInput region[] = {{base, bytes}};
    auto memory = eval::Memory::Create(region, budget);
    ASSERT_TRUE(memory.memory);
    const auto run = eval::Execute(*result.group, state, *memory.memory, budget);
    EXPECT_EQ(run.outcome, eval::Outcome::completed);
    std::array<std::uint64_t, 2> expected{};
    for (unsigned i = 0; i < load.width / 8; ++i) {
      expected[i / 8] |= std::uint64_t(bytes[load.offset + i]) << ((i % 8) * 8);
    }

    EXPECT_EQ(state.cells[kQ0 + load.rt].value.word(0), expected[0]) << load.width;
    EXPECT_EQ(state.cells[kQ0 + load.rt].value.word(1), expected[1]) << load.width;
  }
}

TEST(A64Decode, SimdFpStoresWriteOnlyTheirWidthOfTheQRegister) {
  constexpr std::uint32_t word = 0xbd000c22U;  // str s2, [x1, #12]
  EXPECT_EQ(LiftWord(word).reason, DecodeDecline::unsupported);
  const auto group = LiftMemory(word).group;
  ASSERT_TRUE(group);
  EXPECT_TRUE(group->writes().empty());
  constexpr std::uint64_t base = 0x100000000;
  std::array<std::uint8_t, 24> bytes{};
  Budget budget({1000000, 1000000});
  auto state = VectorState(budget, base);
  const eval::RegionInput region[] = {{base, bytes}};
  auto memory = eval::Memory::Create(region, budget);
  ASSERT_TRUE(memory.memory);
  EXPECT_EQ(eval::Execute(*group, state, *memory.memory, budget).outcome, eval::Outcome::completed);
  std::array<std::uint8_t, 24> expected{};
  const auto low = state.cells[kQ0 + 2].value.word(0);
  for (unsigned i = 0; i < 4; ++i) expected[12 + i] = static_cast<std::uint8_t>(low >> (i * 8));
  EXPECT_TRUE(std::ranges::equal(memory.memory->Regions()[0].bytes, expected));
}

TEST(A64Decode, SimdFpAddressingFollowsTheScalarForms) {
  const auto post = LiftMemory(0xfc40842dU);  // ldr d13, [x1], #8
  ASSERT_TRUE(post.group);
  ASSERT_EQ(post.group->writes().size(), 2);
  EXPECT_EQ(post.group->writes()[0].storage, kQ0 + 13);
  EXPECT_EQ(post.group->writes()[1].storage, 1);
  const auto post_nodes = post.group->nodes();
  const auto load =
      std::ranges::find_if(post_nodes, [](const auto& node) { return node.op == ir::Op::load; });
  ASSERT_NE(load, post_nodes.end());
  EXPECT_EQ(post_nodes[load->inputs[0]].op, ir::Op::read);

  // A SIMD&FP transfer register is never the base, so writeback may name x1
  // alongside d1, which the general form rejects.
  const auto alias = LiftMemory(0xfc408421U);  // ldr d1, [x1], #8
  ASSERT_TRUE(alias.group);
  EXPECT_EQ(alias.group->writes()[0].storage, kQ0 + 1);
  EXPECT_EQ(alias.group->writes()[1].storage, 1);
  const auto pre = LiftMemory(0x3c9f0c2cU);  // str q12, [x1, #-16]!
  ASSERT_TRUE(pre.group);
  ASSERT_EQ(pre.group->writes().size(), 1);
  EXPECT_EQ(pre.group->writes()[0].storage, 1);
  EXPECT_EQ(std::ranges::count_if(
                pre.group->nodes(),
                [](const auto& node) { return node.op == ir::Op::store && node.width == 64; }),
            2);
  const auto unscaled = LiftMemory(0x3cdf7029U);  // ldur q9, [x1, #-9]
  ASSERT_TRUE(unscaled.group);
  EXPECT_TRUE(std::ranges::any_of(unscaled.group->nodes(), [](const auto& node) {
    return node.op == ir::Op::constant && node.immediate == std::uint64_t{0} - 9;
  }));

  // A Q register offset scales by 16, not by the size field's 1.
  const auto indexed = LiftMemory(0x3ce27834U);  // ldr q20, [x1, x2, lsl #4]
  ASSERT_TRUE(indexed.group);
  EXPECT_TRUE(std::ranges::any_of(indexed.group->nodes(), [](const auto& node) {
    return node.op == ir::Op::shl && node.width == 64;
  }));
  constexpr std::uint64_t base = 0x100000000;
  std::array<std::uint8_t, 32> bytes{};
  for (unsigned i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(i);
  Budget budget({1000000, 1000000});
  auto state = VectorState(budget, base);
  auto index = BitVector::from_u64(64, 1, 64, budget);
  ASSERT_TRUE(index);
  state.cells[2].value = std::move(*index);
  const eval::RegionInput region[] = {{base, bytes}};
  auto memory = eval::Memory::Create(region, budget);
  ASSERT_TRUE(memory.memory);
  EXPECT_EQ(eval::Execute(*indexed.group, state, *memory.memory, budget).outcome,
            eval::Outcome::completed);
  EXPECT_EQ(state.cells[kQ0 + 20].value.word(0), 0x1716151413121110);
  EXPECT_EQ(state.cells[kQ0 + 20].value.word(1), 0x1f1e1d1c1b1a1918);
}

TEST(A64Decode, SimdFpPairsAccessEachElementInTurn) {
  struct Case {
    std::uint32_t word;
    ir::Op op;
    unsigned width, accesses;
    std::vector<ir::StorageId> writes;
  };

  const Case cases[] = {
      {0xacc12428U, ir::Op::load, 64, 4, {kQ0 + 8, kQ0 + 9, 1}},  // ldp q8, q9, [x1], #32
      {0x6dbf1fe6U, ir::Op::store, 64, 2, {kSp}},                 // stp d6, d7, [sp, #-16]!
      {0x2d7f0420U, ir::Op::load, 32, 2, {kQ0, kQ0 + 1}}};        // ldp s0, s1, [x1, #-8]
  for (const auto& c : cases) {
    EXPECT_EQ(LiftWord(c.word).reason, DecodeDecline::unsupported);
    const auto result = LiftMemory(c.word);
    ASSERT_TRUE(result.group) << std::hex << c.word;
    EXPECT_EQ(result.group->memory_model(), ir::MemoryModel::atomic_scalar_reference);
    std::vector<ir::StorageId> writes;
    for (const auto& write : result.group->writes()) writes.push_back(write.storage);
    EXPECT_EQ(writes, c.writes);
    EXPECT_EQ(std::ranges::count_if(
                  result.group->nodes(),
                  [&](const auto& node) { return node.op == c.op && node.width == c.width; }),
              c.accesses);
  }

  constexpr std::uint64_t base = 0x100000000;
  std::array<std::uint8_t, 8> bytes{1, 2, 3, 4, 5, 6, 7, 8};
  Budget budget({1000000, 1000000});
  auto state = VectorState(budget, base + 8);
  const eval::RegionInput region[] = {{base, bytes}};
  auto memory = eval::Memory::Create(region, budget);
  ASSERT_TRUE(memory.memory);
  const auto group = LiftMemory(0x2d7f0420U).group;
  ASSERT_TRUE(group);
  EXPECT_EQ(eval::Execute(*group, state, *memory.memory, budget).outcome, eval::Outcome::completed);
  EXPECT_EQ(state.cells[kQ0].value.word(0), 0x04030201);
  EXPECT_EQ(state.cells[kQ0].value.word(1), 0);
  EXPECT_EQ(state.cells[kQ0 + 1].value.word(0), 0x08070605);
  EXPECT_EQ(state.cells[kQ0 + 1].value.word(1), 0);

  // LDP q0, q0 and opc 3 are reserved, as is a SIMD&FP opc<1> with a nonzero
  // size; LDNP is not modeled.
  for (const auto word : {0xad400040U, 0xed400440U, 0x7dc00000U}) {
    EXPECT_EQ(LiftMemory(word).reason, DecodeDecline::invalid_encoding) << std::hex << word;
  }

  EXPECT_EQ(LiftMemory(0xac400440U).reason, DecodeDecline::unsupported);
}

TEST(A64Decode, CmeqComparesEachIntegerLaneAndZeroesTheUnusedHalf) {
  for (unsigned q = 0; q < 2; ++q) {
    for (unsigned size = 0; size < 4; ++size) {
      const auto word = 0x2e228c20U | (q << 30) | (size << 22);
      const auto result = LiftWord(word);
      if (q == 0 && size == 3) {
        EXPECT_EQ(result.reason, DecodeDecline::invalid_encoding);
        continue;
      }

      ASSERT_TRUE(result.group) << std::hex << word;
      ASSERT_EQ(result.group->writes().size(), 1);
      EXPECT_EQ(result.group->writes()[0].storage, kQ0);
      Budget budget({1000000, 1000000});
      auto state = VectorState(budget, 0);
      const std::array<std::uint64_t, 2> lhs{0x8877665544332211, 0xfedcba9876543210};
      const std::array<std::uint64_t, 2> rhs{0x8877665544332211, 0xfedcba9876543211};
      auto left = BitVector::from_words(128, lhs, 128, budget);
      auto right = BitVector::from_words(128, rhs, 128, budget);
      ASSERT_TRUE(left);
      ASSERT_TRUE(right);
      state.cells[kQ0 + 1].value = std::move(*left);
      state.cells[kQ0 + 2].value = std::move(*right);
      EXPECT_EQ(eval::Execute(*result.group, state, budget), eval::Outcome::completed);
      EXPECT_EQ(state.cells[kQ0].value.word(0), UINT64_MAX);
      const unsigned lane_width = 8U << size;
      const auto different = UINT64_MAX >> (64 - lane_width);
      EXPECT_EQ(state.cells[kQ0].value.word(1), q ? ~different : 0);
    }
  }

  // Scalar CMEQ and vector CMTST do not enter the vector equality family.
  for (const auto word : {0x7ee28c20U, 0x0e228c20U}) {
    EXPECT_EQ(LiftWord(word).reason, DecodeDecline::unsupported);
  }

  Budget tiny({10000, 1});
  EXPECT_EQ(Decode(0x1000, Bytes(0x6e228c20U), tiny).reason, DecodeDecline::byte_limit);
}

TEST(A64Decode, UmovSelectsEveryLaneAndNeverNamesSp) {
  for (unsigned q = 0; q < 2; ++q) {
    for (unsigned imm5 = 0; imm5 < 32; ++imm5) {
      const auto word = 0x0e003c20U | (q << 30) | (imm5 << 16);
      const bool legal = q ? (imm5 & 15) == 8 : (imm5 & 7) != 0;
      const auto result = LiftWord(word);
      if (!legal) {
        EXPECT_EQ(result.reason, DecodeDecline::invalid_encoding) << std::hex << word;
        continue;
      }

      ASSERT_TRUE(result.group) << std::hex << word;
      Budget budget({1000000, 1000000});
      auto state = VectorState(budget, 0);
      const std::array<std::uint64_t, 2> source{0x8877665544332211, 0xfedcba9876543210};
      auto value = BitVector::from_words(128, source, 128, budget);
      ASSERT_TRUE(value);
      state.cells[kQ0 + 1].value = std::move(*value);
      EXPECT_EQ(eval::Execute(*result.group, state, budget), eval::Outcome::completed);
      unsigned size = 0;
      while (((imm5 >> size) & 1) == 0) ++size;
      const unsigned width = 8U << size;
      const unsigned at = (imm5 >> (size + 1)) * width;
      const auto expected = (source[at / 64] >> (at % 64)) & (UINT64_MAX >> (64 - width));
      EXPECT_EQ(state.cells[0].value.word(0), expected);
      const auto discard = LiftWord(word | 31);
      ASSERT_TRUE(discard.group);
      EXPECT_TRUE(discard.group->writes().empty());
    }
  }

  EXPECT_EQ(LiftWord(0x0e012c20U).reason, DecodeDecline::unsupported);  // SMOV
}

TEST(A64Decode, MoviAndMvniAreConstantsAndOtherModifiedImmediatesStayUnsupported) {
  struct Case {
    std::uint32_t word;
    unsigned rd;
    std::uint64_t low, high;
  };

  const Case cases[] = {
      {0x6f00e400U, 0, 0, 0},                                    // movi v0.2d, #0
      {0x2f00e401U, 1, 0, 0},                                    // movi d1, #0
      {0x6f05e542U, 2, 0xff00ff00ff00ff00, 0xff00ff00ff00ff00},  // movi v2.2d, #0xff00...
      {0x4f00c643U, 3, 0x000012ff000012ff, 0x000012ff000012ff},  // movi v3.4s, #0x12, msl #8
      {0x6f018684U, 4, 0xffcbffcbffcbffcb, 0xffcbffcbffcbffcb},  // mvni v4.8h, #0x34
      {0x2f018684U, 4, 0xffcbffcbffcbffcb, 0}};                  // mvni v4.4h, #0x34
  for (const auto& c : cases) {
    const auto result = LiftWord(c.word);
    ASSERT_TRUE(result.group) << std::hex << c.word;
    ASSERT_EQ(result.group->writes().size(), 1);
    EXPECT_EQ(result.group->writes()[0].storage, kQ0 + c.rd);
    EXPECT_TRUE(std::ranges::none_of(result.group->nodes(),
                                     [](const auto& node) { return node.op == ir::Op::read; }));
    Budget budget({1000000, 1000000});
    auto state = VectorState(budget, 0);
    EXPECT_EQ(eval::Execute(*result.group, state, budget), eval::Outcome::completed);
    EXPECT_EQ(state.cells[kQ0 + c.rd].value.word(0), c.low) << std::hex << c.word;
    EXPECT_EQ(state.cells[kQ0 + c.rd].value.word(1), c.high) << std::hex << c.word;
  }

  for (const auto word : {0x4f001645U, 0x4f03f606U}) {  // orr v5.4s, #0x12; fmov v6.4s, #1.0
    EXPECT_EQ(LiftWord(word).reason, DecodeDecline::unsupported) << std::hex << word;
  }
}

TEST(A64Decode, FmovBetweenGeneralAndSimdFpRegistersMovesBits) {
  Budget budget({1000000, 1000000});
  const auto entry = VectorState(budget, 0x8877665544332211);
  const auto q = [&](unsigned reg, unsigned half) {
    return entry.cells[kQ0 + reg].value.word(half);
  };

  const auto x = [&](unsigned reg) { return entry.cells[reg].value.word(0); };

  struct Case {
    std::uint32_t word;
    ir::StorageId storage;
    std::uint64_t low, high;
  };

  const Case cases[] = {{0x9e670020U, kQ0, 0x8877665544332211, 0},     // fmov d0, x1
                        {0x9e660062U, 2, q(3, 0), 0},                  // fmov x2, d3
                        {0x1e2700a4U, kQ0 + 4, x(5) & 0xffffffff, 0},  // fmov s4, w5
                        {0x1e2600e6U, 6, q(7, 0) & 0xffffffff, 0},     // fmov w6, s7
                        {0x9eaf0128U, kQ0 + 8, q(8, 0), x(9)},         // fmov v8.d[1], x9
                        {0x9eae016aU, 10, q(11, 1), 0}};               // fmov x10, v11.d[1]
  for (const auto& c : cases) {
    const auto result = LiftWord(c.word);
    ASSERT_TRUE(result.group) << std::hex << c.word;
    ASSERT_EQ(result.group->writes().size(), 1);
    EXPECT_EQ(result.group->writes()[0].storage, c.storage);
    auto state = VectorState(budget, 0x8877665544332211);
    EXPECT_EQ(eval::Execute(*result.group, state, budget), eval::Outcome::completed);
    EXPECT_EQ(state.cells[c.storage].value.word(0), c.low) << std::hex << c.word;
    if (c.storage >= kQ0) {
      EXPECT_EQ(state.cells[c.storage].value.word(1), c.high) << std::hex << c.word;
    }
  }

  // A general X with a single-precision register is not one of these moves.
  for (const auto word : {0x9e260000U, 0x9e270000U}) {
    EXPECT_EQ(LiftWord(word).reason, DecodeDecline::unsupported) << std::hex << word;
  }
}

TEST(A64Decode, FmovRegisterCopiesLowBitsAndZeroesTheRest) {
  for (const auto [word, width] : {std::pair{0x1e6041acU, 64U},     // fmov d12, d13
                                   std::pair{0x1e2041acU, 32U}}) {  // fmov s12, s13
    const auto result = LiftWord(word);
    ASSERT_TRUE(result.group) << std::hex << word;
    Budget budget({1000000, 1000000});
    auto state = VectorState(budget, 0);
    const auto source = state.cells[kQ0 + 13].value.word(0);
    EXPECT_EQ(eval::Execute(*result.group, state, budget), eval::Outcome::completed);
    EXPECT_EQ(state.cells[kQ0 + 12].value.word(0), width == 64 ? source : source & 0xffffffff);
    EXPECT_EQ(state.cells[kQ0 + 12].value.word(1), 0);
  }

  // FABS, FNEG and the half-precision copy are not bit copies of this kind.
  for (const auto word : {0x1e60c1acU, 0x1e6141acU, 0x1ee041acU}) {
    EXPECT_EQ(LiftWord(word).reason, DecodeDecline::unsupported) << std::hex << word;
  }
}

}  // namespace
}  // namespace nyx::a64
