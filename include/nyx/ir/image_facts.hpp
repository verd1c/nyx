#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "nyx/ir/group.hpp"
#include "nyx/support/budget.hpp"

namespace nyx::ir {

// Image bytes the caller asserts do not change for the life of the run. This is
// a restriction the run declares, not something segment flags prove: a mapping
// can be made writable later or aliased by a writable one. The caller owns the
// assertion and must publish it; this library only reads what it is handed.
// Addresses are image locations, never runtime values, so a target resolved
// through them does not depend on where the image was placed.
struct ConstantImageRange {
  std::uint64_t address = 0;
  std::span<const std::uint8_t> bytes;

  // The bytes sit in a segment the loader maps without write permission, so a
  // store at an address nobody can place, which might be aimed anywhere, would
  // fault rather than change them. One placed on them still refutes the range:
  // that is the program contradicting the declaration. Declared with the range,
  // for the same reason: a mapping can still be made writable later.
  bool read_only = false;
};

// A slot the loader fills from a relative relocation. While its value stays
// stable it is an image location, so reading one keeps the load bias symbolic
// exactly as an ADRP does. Its file bytes are never the value the run sees.
struct RelocatedPointer {
  std::uint64_t address = 0;
  std::uint64_t target = 0;

  // False after a modeled store refutes the loader's value. The slot still
  // blocks reading file bytes: the loader wrote it even if code later did too.
  bool value_stable = true;
};

// What a run has established about its image: which bytes cannot change, and
// whether placement is constrained. Both are empty or false by default, so an
// analysis given no facts assumes nothing.
struct ImageFacts {
  std::span<const ConstantImageRange> constants;

  // Sorted by address. Stable values are a declared restriction: the loader
  // writes them once and the run is assumed not to replace them. Unstable
  // entries retain the loader-written byte location without claiming a value.
  std::span<const RelocatedPointer> pointers;

  // ADRP masks the program counter to its page, which commutes with the load
  // bias only when that bias is page-aligned. Real loaders map at page
  // granularity, but the execution contract admits arbitrary placement, so a
  // run must declare this before a PC-page target can resolve.
  bool page_aligned_placement = false;

  // Where the run declares the image sits. An address that is only a number
  // names a location only once this is known.
  std::optional<std::uint64_t> load_bias = std::nullopt;
};

// Separate from value stability. Disabling a load's memory effect also needs
// the declared bytes mapped and readable whenever the instruction executes.
struct ImageAccessContract {
  bool mapped_readable_lifetime = false;
  bool no_runtime_unmapping = false;
  bool ordinary_reads_unobservable = false;
};

struct ImageWrite {
  std::uint64_t address;
  unsigned width;
  ValueId operation;
};

struct ImageFactRefutation {
  std::size_t index;
  ImageWrite write;
};

struct ImageFactRefutations {
  std::vector<ImageFactRefutation> constants;
  std::vector<ImageFactRefutation> pointers;
};

// The bytes of one declared constant range a placed write reaches, which that
// write withdraws; `index` names the range in the facts it was checked
// against. A range is a claim about each of its bytes, so a store contradicts
// only the claims for the bytes it writes and the rest stand. Past
// kMaxRefutedSpans writes into one range it is withdrawn `whole` instead, the
// span covering all of it and naming the first write.
struct RefutedConstantSpan {
  std::size_t index;
  std::uint64_t address;
  std::uint64_t bytes;
  ImageWrite write;
  bool whole = false;
};

inline constexpr std::size_t kMaxRefutedSpans = 64;

// The byte spans in constants still refer to the caller's image storage.
// Keep that storage alive while using this reduced view.
struct RetainedImageFacts {
  std::vector<ConstantImageRange> constants;
  std::vector<RelocatedPointer> pointers;
  bool page_aligned_placement = false;
  std::optional<std::uint64_t> load_bias = std::nullopt;

  // Per retained constant, the index of the declared range it is a piece of.
  std::vector<std::size_t> origins;

  ImageFacts view() const { return {constants, pointers, page_aligned_placement, load_bias}; }
};

// Withdraws whole ranges.
[[nodiscard]] std::optional<RetainedImageFacts> RetainImageFacts(
    const ImageFacts&, std::span<const std::size_t> refuted_constants,
    std::span<const std::size_t> refuted_pointers, Budget&);
// Keeps each range's bytes no span covers, as one piece per unbroken run, with
// the range's read_only. Spans sorted by index, then address, and inside their
// range.
[[nodiscard]] std::optional<RetainedImageFacts> RetainImageFacts(
    const ImageFacts&, std::span<const RefutedConstantSpan> refuted_constants,
    std::span<const std::size_t> refuted_pointers, Budget&);

// Every placed write's bytes inside each declared constant range, sorted by
// index, then address. Empty optional means the resource budget was exhausted.
[[nodiscard]] std::optional<std::vector<RefutedConstantSpan>> RefuteConstantBytes(
    const ImageFacts&, std::span<const ImageWrite>, Budget&);

// Find the first modeled write into each declared value. Pointer slots must be
// sorted by address. Empty optional means the resource budget was exhausted.
[[nodiscard]] std::optional<ImageFactRefutations> RefuteImageFacts(const ImageFacts&,
                                                                   std::span<const ImageWrite>,
                                                                   Budget&);

// The number a read of `width` bits at image location `address` yields from
// declared constant bytes, assembled in `order`. Empty unless the width is a
// whole number of bytes up to 64 and the read lies wholly inside one range, and
// empty when it touches a relocated slot: the loader, not the file, supplies
// those bytes.
[[nodiscard]] std::optional<std::uint64_t> ReadConstant(const ImageFacts&, std::uint64_t address,
                                                        unsigned width, ByteOrder order);

// The image location a whole 64-bit read of a relocated slot yields. A partial
// read, or one starting anywhere but the slot, yields nothing.
[[nodiscard]] std::optional<std::uint64_t> ReadRelocated(const ImageFacts&, std::uint64_t address,
                                                         unsigned width);

}  // namespace nyx::ir
