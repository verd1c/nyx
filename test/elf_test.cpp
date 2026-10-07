#include "nyx/format/elf.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace nyx::format {
namespace {

void Put(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint64_t value,
         std::size_t width, bool big = false) {
  for (std::size_t i = 0; i < width; ++i) {
    bytes[offset + (big ? width - i - 1 : i)] = static_cast<std::uint8_t>(value >> (8 * i));
  }
}

std::vector<std::uint8_t> Fixture(bool big = false, std::size_t count = 1) {
  std::vector<std::uint8_t> bytes(512);
  bytes[0] = 0x7f;
  bytes[1] = 'E';
  bytes[2] = 'L';
  bytes[3] = 'F';
  bytes[4] = 2;
  bytes[5] = big ? 2 : 1;
  bytes[6] = 1;
  Put(bytes, 16, 3, 2, big);
  Put(bytes, 18, 183, 2, big);
  Put(bytes, 20, 1, 4, big);
  Put(bytes, 24, 0x1000, 8, big);
  Put(bytes, 32, 64, 8, big);
  Put(bytes, 52, 64, 2, big);
  Put(bytes, 54, 56, 2, big);
  Put(bytes, 56, count, 2, big);
  for (std::size_t i = 0; i < count; ++i) {
    const auto base = 64 + 56 * i;
    Put(bytes, base, 1, 4, big);
    Put(bytes, base + 4, 5, 4, big);
    Put(bytes, base + 8, 256, 8, big);
    Put(bytes, base + 16, 0x1000 + i * 0x100, 8, big);
    Put(bytes, base + 32, 64, 8, big);
    Put(bytes, base + 40, 128, 8, big);
    Put(bytes, base + 48, 1, 8, big);
  }

  bytes[256] = 0xab;
  return bytes;
}

// A PT_LOAD covering the dynamic array plus a PT_DYNAMIC naming it. `entries`
// are (tag, value) pairs; `extent` overrides p_filesz so a table can be cut
// short of its terminator, and `vaddr_shift` moves the mapped address away from
// the bytes p_offset names.
std::vector<std::uint8_t> DynamicFixture(
    std::vector<std::pair<std::uint64_t, std::uint64_t>> entries,
    std::optional<std::uint64_t> extent = {}, std::uint64_t vaddr_shift = 0) {
  std::vector<std::uint8_t> bytes(1024);
  bytes[0] = 0x7f;
  bytes[1] = 'E';
  bytes[2] = 'L';
  bytes[3] = 'F';
  bytes[4] = 2;
  bytes[5] = 1;
  bytes[6] = 1;
  Put(bytes, 16, 3, 2);
  Put(bytes, 18, 183, 2);
  Put(bytes, 20, 1, 4);
  Put(bytes, 32, 64, 8);
  Put(bytes, 52, 64, 2);
  Put(bytes, 54, 56, 2);
  Put(bytes, 56, 2, 2);
  constexpr std::uint64_t kDynOffset = 512;
  const auto size = entries.size() * 16;

  // PT_LOAD, read+execute, covering the whole file at vaddr 0.
  Put(bytes, 64, 1, 4);
  Put(bytes, 68, 5, 4);
  Put(bytes, 72, 0, 8);
  Put(bytes, 80, 0, 8);
  Put(bytes, 96, 1024, 8);
  Put(bytes, 104, 1024, 8);
  Put(bytes, 112, 1, 8);

  // PT_DYNAMIC.
  Put(bytes, 120, 2, 4);
  Put(bytes, 124, 6, 4);
  Put(bytes, 128, kDynOffset, 8);
  Put(bytes, 136, kDynOffset + vaddr_shift, 8);
  Put(bytes, 152, extent.value_or(size), 8);
  Put(bytes, 160, extent.value_or(size), 8);
  Put(bytes, 168, 8, 8);
  for (std::size_t i = 0; i < entries.size(); ++i) {
    Put(bytes, kDynOffset + i * 16, entries[i].first, 8);
    Put(bytes, kDynOffset + i * 16 + 8, entries[i].second, 8);
  }

  return bytes;
}

TEST(Elf, TextRelocationScanFollowsTheArrayTheLoaderWouldRead) {
  constexpr std::uint64_t kNeeded = 1, kNull = 0, kTextrel = 22, kFlags = 30;

  struct Case {
    const char* name;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> entries;
    std::optional<std::uint64_t> extent;
    std::uint64_t vaddr_shift;
    bool declares;
  };

  const std::vector<Case> cases{
      {"clean table terminated in extent", {{kNeeded, 7}, {kNull, 0}}, {}, 0, false},
      {"DT_TEXTREL present", {{kTextrel, 0}, {kNull, 0}}, {}, 0, true},
      {"DF_TEXTREL set in DT_FLAGS", {{kFlags, 4}, {kNull, 0}}, {}, 0, true},
      {"another DT_FLAGS bit is not DF_TEXTREL", {{kFlags, 8}, {kNull, 0}}, {}, 0, false},
      // The loader reads past p_filesz to DT_NULL; a scan bounded by the extent
      // would miss the entry beyond it, so an unterminated extent fails closed.
      {"extent hides the terminator", {{kNeeded, 7}, {kTextrel, 0}, {kNull, 0}}, 16, 0, true},
      {"extent is not a whole number of entries", {{kNeeded, 7}, {kNull, 0}}, 24, 0, true},
      {"empty extent", {{kNeeded, 7}, {kNull, 0}}, 0, 0, true},
      // p_vaddr naming bytes no load segment maps leaves nothing to read.
      {"dynamic array is unmapped", {{kNeeded, 7}, {kNull, 0}}, {}, 0x100000, true},
  };

  for (const auto& test : cases) {
    auto result = Load(DynamicFixture(test.entries, test.extent, test.vaddr_shift), {});
    ASSERT_TRUE(result.image) << test.name;
    EXPECT_EQ(result.image->declares_text_relocations(), test.declares) << test.name;
  }
}

TEST(Elf, RelativeRelocationsAreReadFromTheTableTheDynamicArrayNames) {
  constexpr std::uint64_t kNull = 0, kRela = 7, kRelaSz = 8, kRelaEnt = 9;
  constexpr std::uint64_t kTable = 640, kRelative = 1027, kJumpSlot = 1026;
  const auto build = [&](std::vector<std::array<std::uint64_t, 3>> entries,
                         std::uint64_t entry_size = 24) {
    auto bytes = DynamicFixture(
        {{kRela, kTable}, {kRelaSz, entries.size() * 24}, {kRelaEnt, entry_size}, {kNull, 0}});
    for (std::size_t i = 0; i < entries.size(); ++i) {
      Put(bytes, kTable + i * 24, entries[i][0], 8);
      Put(bytes, kTable + i * 24 + 8, entries[i][1], 8);
      Put(bytes, kTable + i * 24 + 16, entries[i][2], 8);
    }

    return bytes;
  };

  // Only the relative kind names an image location with no symbol to resolve,
  // and the results are sorted so a reader can find a slot by address.
  auto result = Load(
      build({{0x900, kRelative, 0x1234}, {0x800, kJumpSlot, 0x5}, {0x880, kRelative, 0x4321}}), {});
  ASSERT_TRUE(result.image);
  const auto relatives = result.image->relative_relocations();
  ASSERT_EQ(relatives.size(), 2);
  EXPECT_EQ(relatives[0].address, 0x880);
  EXPECT_EQ(relatives[0].target, 0x4321);
  EXPECT_EQ(relatives[1].address, 0x900);
  EXPECT_EQ(relatives[1].target, 0x1234);

  // An entry size this parser does not know, and a table the image does not
  // map, each leave nothing rather than something guessed.
  auto sized = Load(build({{0x900, kRelative, 0x1234}}, 16), {});
  ASSERT_TRUE(sized.image);
  EXPECT_TRUE(sized.image->relative_relocations().empty());
  auto unmapped =
      Load(DynamicFixture({{kRela, 0x900000}, {kRelaSz, 24}, {kRelaEnt, 24}, {kNull, 0}}), {});
  ASSERT_TRUE(unmapped.image);
  EXPECT_TRUE(unmapped.image->relative_relocations().empty());
}

TEST(Elf, EveryRelocationKindNamesBytesTheLoaderWrites) {
  constexpr std::uint64_t kNull = 0, kRela = 7, kRelaSz = 8, kRelaEnt = 9;
  constexpr std::uint64_t kJmpRel = 23, kPltRelSz = 2, kPltRel = 20;
  constexpr std::uint64_t kRelaTable = 640, kPltTable = 736;
  constexpr std::uint64_t kRelative = 1027, kGlobDat = 1025, kJumpSlot = 1026, kTlsDesc = 1031;
  auto bytes = DynamicFixture({{kRela, kRelaTable},
                               {kRelaSz, 72},
                               {kRelaEnt, 24},
                               {kJmpRel, kPltTable},
                               {kPltRelSz, 24},
                               {kPltRel, 7},
                               {kNull, 0}});
  const std::array<std::array<std::uint64_t, 3>, 4> entries{{{0x900, kRelative, 0x1234},
                                                             {0x940, kGlobDat, 0},
                                                             {0x9c0, kTlsDesc, 0},
                                                             {0x980, kJumpSlot, 0}}};
  for (std::size_t i = 0; i < entries.size(); ++i) {
    const auto at = i < 3 ? kRelaTable + i * 24 : kPltTable;
    for (unsigned j = 0; j < 3; ++j) Put(bytes, at + j * 8, entries[i][j], 8);
  }

  auto result = Load(bytes, {});
  ASSERT_TRUE(result.image);
  const auto& image = *result.image;

  // A symbol-resolved slot is written as surely as a relative one, and the
  // procedure linkage table's slots are written from their own table.
  EXPECT_TRUE(image.LoaderMayWrite(0x900, 8));
  EXPECT_TRUE(image.LoaderMayWrite(0x940, 1));
  EXPECT_TRUE(image.LoaderMayWrite(0x980, 8));

  // Overlap from either side counts, including a write that starts before the
  // range and reaches into it.
  EXPECT_TRUE(image.LoaderMayWrite(0x8f8, 9));
  EXPECT_TRUE(image.LoaderMayWrite(0x907, 1));

  // A doubleword slot ends where it ends: the word after it, where a table's
  // key can sit, is the file's. A TLS descriptor writes two doublewords.
  EXPECT_FALSE(image.LoaderMayWrite(0x908, 8));
  EXPECT_TRUE(image.LoaderMayWrite(0x9cf, 1));
  EXPECT_FALSE(image.LoaderMayWrite(0x9d0, 8));

  // Between slots, and past the widest write, nothing is written.
  EXPECT_FALSE(image.LoaderMayWrite(0x910, 8));
  EXPECT_FALSE(image.LoaderMayWrite(0x8f0, 8));
  EXPECT_FALSE(image.LoaderMayWrite(0xa00, 64));
}

// A kind this reader does not name could be as wide as any relocation, while
// ABS64 is one doubleword like the dynamic kinds.
TEST(Elf, UnknownRelocationKindReachesTheWidestWrite) {
  constexpr std::uint64_t kNull = 0, kRela = 7, kRelaSz = 8, kRelaEnt = 9, kTable = 640;
  constexpr std::uint64_t kAbs64 = 257, kUnknown = 4000;
  auto bytes = DynamicFixture({{kRela, kTable}, {kRelaSz, 48}, {kRelaEnt, 24}, {kNull, 0}});
  const std::array<std::array<std::uint64_t, 3>, 2> entries{
      {{0x900, kAbs64, 0}, {0x940, kUnknown, 0}}};
  for (std::size_t i = 0; i < entries.size(); ++i)
    for (unsigned j = 0; j < 3; ++j) Put(bytes, kTable + i * 24 + j * 8, entries[i][j], 8);
  auto result = Load(bytes, {});
  ASSERT_TRUE(result.image);
  const auto& image = *result.image;
  EXPECT_TRUE(image.LoaderMayWrite(0x907, 1));
  EXPECT_FALSE(image.LoaderMayWrite(0x908, 8));
  EXPECT_TRUE(image.LoaderMayWrite(0x948, 8));
  EXPECT_TRUE(image.LoaderMayWrite(0x94f, 1));
  EXPECT_FALSE(image.LoaderMayWrite(0x950, 8));
}

// A symbol this image defines gives its slot a value from the file, S + A, as
// surely as a relative relocation does; only whether the dynamic linker binds
// this definition or another is outside what the file can say.
TEST(Elf, SymbolRelocationsResolveOnlyToWhatTheImageDefines) {
  static constexpr std::uint64_t kNull = 0, kRela = 7, kRelaSz = 8, kRelaEnt = 9, kSymtab = 6,
                                 kSyment = 11, kFlags = 30;
  static constexpr std::uint64_t kTable = 640, kSymbols = 800;
  static constexpr std::uint64_t kAbs64 = 257, kGlobDat = 1025, kJumpSlot = 1026, kRelative = 1027;
  using Row = std::array<std::uint64_t, 3>;
  const std::vector<Row> rows{{0x900, 1ULL << 32 | kAbs64, 0x10},
                              {0x940, 2ULL << 32 | kGlobDat, 0},
                              {0x980, 3ULL << 32 | kJumpSlot, 0},
                              {0x9c0, 4ULL << 32 | kAbs64, 0},
                              {0xa00, 5ULL << 32 | kAbs64, 8},
                              // An index whose entry the image does not map names nothing.
                              {0xa40, 40ULL << 32 | kAbs64, 0}};
  // symbolic: 0 none, 1 DF_SYMBOLIC in DT_FLAGS, 2 the older DT_SYMBOLIC tag.
  const auto build = [](int symbolic, const std::vector<Row>& entries) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> dynamic{{kRela, kTable},
                                                                 {kRelaSz, entries.size() * 24},
                                                                 {kRelaEnt, 24},
                                                                 {kSymtab, kSymbols},
                                                                 {kSyment, 24}};
    if (symbolic == 1) dynamic.push_back({kFlags, 2});
    if (symbolic == 2) dynamic.push_back({16, 0});
    dynamic.push_back({kNull, 0});
    auto bytes = DynamicFixture(dynamic);

    // info = binding << 4 | type, other = visibility.
    struct Symbol {
      std::uint64_t info, other, section, value;
    };

    const std::array<Symbol, 7> symbols{
        {{0x12, 0, 5, 0x300},       // 1: global function, default visibility
         {0x11, 2, 5, 0x200},       // 2: global object, hidden
         {0x12, 0, 0, 0},           // 3: an import
         {0x16, 0, 5, 0x40},        // 4: TLS
         {0x01, 0, 5, 0x100},       // 5: local object
         {0x11, 0, 0xfff1, 0x500},  // 6: SHN_ABS, a number rather than a location
         {0x1a, 0, 5, 0x600}}};     // 7: an indirect function, whose value its resolver returns
    for (std::size_t i = 0; i < symbols.size(); ++i) {
      const auto at = kSymbols + (i + 1) * 24;
      Put(bytes, at + 4, symbols[i].info, 1);
      Put(bytes, at + 5, symbols[i].other, 1);
      Put(bytes, at + 6, symbols[i].section, 2);
      Put(bytes, at + 8, symbols[i].value, 8);
    }

    for (std::size_t i = 0; i < entries.size(); ++i)
      for (unsigned j = 0; j < 3; ++j) Put(bytes, kTable + i * 24 + j * 8, entries[i][j], 8);
    return bytes;
  };

  auto result = Load(build(0, rows), {});
  ASSERT_TRUE(result.image);
  const auto symbolics = result.image->symbolic_relocations();
  ASSERT_EQ(symbolics.size(), 3);
  EXPECT_EQ(symbolics[0].address, 0x900);
  EXPECT_EQ(symbolics[0].target, 0x310);
  EXPECT_TRUE(symbolics[0].interposable);
  EXPECT_EQ(symbolics[1].address, 0x940);
  EXPECT_EQ(symbolics[1].target, 0x200);
  EXPECT_FALSE(symbolics[1].interposable);
  EXPECT_EQ(symbolics[2].address, 0xa00);
  EXPECT_EQ(symbolics[2].target, 0x108);
  EXPECT_FALSE(symbolics[2].interposable);

  // The interposable slot has a value only under the declaration.
  const auto without = result.image->LocatedRelocations(false);
  ASSERT_EQ(without.size(), 2);
  EXPECT_EQ(without[0].address, 0x940);
  EXPECT_EQ(result.image->LocatedRelocations(true).size(), 3);

  // Every one of them is still a byte the loader writes.
  for (const auto address : {0x900, 0x940, 0x980, 0x9c0, 0xa00, 0xa40})
    EXPECT_TRUE(result.image->LoaderMayWrite(address, 8)) << address;

  // Linked symbolically, the image binds its own definitions first, whichever
  // of the two ways the dynamic array says so.
  for (const int symbolic : {1, 2}) {
    auto linked = Load(build(symbolic, rows), {});
    ASSERT_TRUE(linked.image);
    EXPECT_EQ(linked.image->LocatedRelocations(false).size(), 3) << symbolic;
  }

  // An absolute symbol, an indirect function and the null symbol name no
  // image location, declaration or not.
  auto others = Load(build(0, {{0x900, 6ULL << 32 | kAbs64, 0},
                               {0x940, 7ULL << 32 | kGlobDat, 0},
                               {0x980, 0ULL << 32 | kAbs64, 0x123}}),
                     {});
  ASSERT_TRUE(others.image);
  EXPECT_TRUE(others.image->symbolic_relocations().empty());
  EXPECT_TRUE(others.image->LocatedRelocations(true).empty());

  // A slot a relative and a symbolic relocation disagree about has no value.
  auto both = build(0, rows);
  Put(both, kTable + 5 * 24, 0x900, 8);
  Put(both, kTable + 5 * 24 + 8, kRelative, 8);
  Put(both, kTable + 5 * 24 + 16, 0x4444, 8);
  auto conflicted = Load(both, {});
  ASSERT_TRUE(conflicted.image);
  const auto located = conflicted.image->LocatedRelocations(true);
  EXPECT_TRUE(std::none_of(located.begin(), located.end(),
                           [](const RelativeRelocation& slot) { return slot.address == 0x900; }));
}

TEST(Elf, TablesThisLoaderCannotFollowLeaveEveryByteWritable) {
  constexpr std::uint64_t kNull = 0, kRela = 7, kRelaSz = 8, kRelaEnt = 9, kRel = 17, kRelr = 36,
                          kAndroidRela = 0x60000011, kJmpRel = 23, kPltRelSz = 2, kPltRel = 20;
  const auto writable = [](std::vector<std::pair<std::uint64_t, std::uint64_t>> entries,
                           std::optional<std::uint64_t> extent = {}) {
    auto result = Load(DynamicFixture(std::move(entries), extent), {});
    EXPECT_TRUE(result.image);
    return result.image->LoaderMayWrite(0xa00, 8);
  };

  EXPECT_FALSE(writable({{kNull, 0}}));

  // Formats the scan does not decode.
  EXPECT_TRUE(writable({{kRel, 640}, {kNull, 0}}));
  EXPECT_TRUE(writable({{kRelr, 640}, {kNull, 0}}));
  EXPECT_TRUE(writable({{kAndroidRela, 640}, {kNull, 0}}));
  EXPECT_TRUE(writable({{kJmpRel, 640}, {kPltRelSz, 24}, {kPltRel, 17}, {kNull, 0}}));

  // A copy relocation's reach is its symbol's size, which this scan does not know.
  {
    auto bytes = DynamicFixture({{kRela, 640}, {kRelaSz, 24}, {kRelaEnt, 24}, {kNull, 0}});
    Put(bytes, 640, 0x900, 8);
    Put(bytes, 648, 1024, 8);
    auto result = Load(bytes, {});
    ASSERT_TRUE(result.image);
    EXPECT_TRUE(result.image->LoaderMayWrite(0xa00, 8));
  }

  // A table the image does not map, an entry size it does not parse, and a
  // dynamic array whose terminator is out of reach.
  EXPECT_TRUE(writable({{kRela, 0x900000}, {kRelaSz, 24}, {kRelaEnt, 24}, {kNull, 0}}));
  EXPECT_TRUE(writable({{kRela, 640}, {kRelaSz, 32}, {kRelaEnt, 16}, {kNull, 0}}));
  EXPECT_TRUE(writable({{kRela, 640}, {kRelaSz, 0}, {kNull, 0}}, 16));
}

TEST(Elf, AnImageWithoutADynamicSectionHasNothingToRelocateThrough) {
  auto result = Load(Fixture(), {});
  ASSERT_TRUE(result.image);
  EXPECT_FALSE(result.image->declares_text_relocations());

  // Without a dynamic array nothing says which bytes startup code relocates.
  EXPECT_TRUE(result.image->LoaderMayWrite(0x100, 8));
}

TEST(Elf, LoadsBothByteOrdersWithoutSections) {
  for (bool big : {false, true}) {
    auto result = Load(Fixture(big));
    ASSERT_TRUE(result.image) << result.error;
    EXPECT_EQ(result.code, LoadError::None);
    EXPECT_EQ(result.image->machine(), 183);
    EXPECT_EQ(result.image->entry(), 0x1000);
    EXPECT_EQ(result.image->endianness(), big ? Endianness::Big : Endianness::Little);
    ASSERT_EQ(result.image->segments().size(), 1);
    ASSERT_TRUE(result.image->Read(0x1000, 1));
    EXPECT_EQ(result.image->Read(0x1000, 1)->front(), 0xab);
    ASSERT_EQ(result.image->executable_ranges().size(), 1);
    EXPECT_EQ(result.image->executable_ranges()[0], (FileRange{256, 64}));
  }
}

TEST(Elf, DoesNotMaterializeZeroFillOrReadAcrossFileBoundary) {
  auto bytes = Fixture();
  Put(bytes, 64 + 40, 1ULL << 50, 8);
  auto result = Load(std::move(bytes));
  ASSERT_TRUE(result.image);
  EXPECT_EQ(result.image->bytes().size(), 512);
  EXPECT_TRUE(result.image->Read(0x103f, 1));
  EXPECT_FALSE(result.image->Read(0x103f, 2));
  EXPECT_FALSE(result.image->Read(0x1040, 1));
  EXPECT_FALSE(result.image->Read(0x1000, 0));
  EXPECT_FALSE(result.image->Read(std::numeric_limits<std::uint64_t>::max(), 2));
}

TEST(Elf, RejectsAmbiguousMappingsIncludingZeroFill) {
  auto bytes = Fixture(false, 2);
  Put(bytes, 120 + 16, 0x1020, 8);
  Put(bytes, 120 + 32, 0, 8);
  auto result = Load(std::move(bytes));
  ASSERT_TRUE(result.image);
  EXPECT_TRUE(result.image->Read(0x1000, 32));
  EXPECT_FALSE(result.image->Read(0x101f, 2));
  EXPECT_FALSE(result.image->Read(0x1020, 1));
}

TEST(Elf, ExecutableManifestCountsFileUnionOnce) {
  auto bytes = Fixture(false, 3);
  Put(bytes, 120 + 8, 288, 8);
  Put(bytes, 176 + 8, 400, 8);
  auto result = Load(std::move(bytes));
  ASSERT_TRUE(result.image);
  ASSERT_EQ(result.image->executable_ranges().size(), 2);
  EXPECT_EQ(result.image->executable_ranges()[0], (FileRange{256, 96}));
  EXPECT_EQ(result.image->executable_ranges()[1], (FileRange{400, 64}));
}

TEST(Elf, RejectsEveryTruncatedHeaderAndProgramHeader) {
  for (std::size_t size = 0; size < 120; ++size) {
    auto bytes = Fixture();
    bytes.resize(size);
    EXPECT_FALSE(Load(std::move(bytes)).image) << size;
  }
}

TEST(Elf, RejectsMalformedExtentsAndAlignment) {
  struct Mutation {
    std::size_t offset;
    std::uint64_t value;
    std::size_t width;
  };

  const Mutation mutations[] = {{32, std::numeric_limits<std::uint64_t>::max() - 4, 8},
                                {32, 1, 8},
                                {52, 63, 2},
                                {54, 55, 2},
                                {20, 2, 4},
                                {64 + 8, std::numeric_limits<std::uint64_t>::max(), 8},
                                {64 + 32, 129, 8},
                                {64 + 8, 500, 8},
                                {64 + 16, std::numeric_limits<std::uint64_t>::max() - 64, 8},
                                {64 + 48, 3, 8},
                                {64 + 48, 4096, 8}};
  for (const auto& mutation : mutations) {
    auto bytes = Fixture();
    Put(bytes, mutation.offset, mutation.value, mutation.width);
    auto result = Load(std::move(bytes));
    EXPECT_FALSE(result.image) << mutation.offset;
    EXPECT_EQ(result.code, LoadError::Malformed) << mutation.offset;
  }
}

TEST(Elf, EnforcesBudgetsAndReportsUnsupportedForms) {
  EXPECT_EQ(Load(Fixture(), Limits{511, 4096}).code, LoadError::ResourceLimit);
  EXPECT_EQ(Load(Fixture(), Limits{512, 0}).code, LoadError::ResourceLimit);
  auto bytes = Fixture();
  bytes[4] = 1;
  EXPECT_EQ(Load(std::move(bytes)).code, LoadError::Unsupported);
  bytes = Fixture();
  Put(bytes, 56, 0xffff, 2);
  EXPECT_EQ(Load(std::move(bytes)).code, LoadError::Unsupported);
}

TEST(Elf, OptionalSectionMetadataDoesNotDetermineMapping) {
  auto bytes = Fixture();

  // Untrusted optional section annotations are not dereferenced by the loader.
  Put(bytes, 40, std::numeric_limits<std::uint64_t>::max(), 8);
  Put(bytes, 60, 40000, 2);
  auto result = Load(std::move(bytes));
  ASSERT_TRUE(result.image);
  EXPECT_TRUE(result.image->Read(0x1000, 64));
}

}  // namespace
}  // namespace nyx::format
