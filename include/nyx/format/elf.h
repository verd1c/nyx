#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace nyx::format {

enum class Endianness { Little, Big };
enum class LoadError { None, Malformed, Unsupported, ResourceLimit };

struct Limits {
  std::size_t max_file_bytes = 512ULL * 1024 * 1024;
  std::size_t max_program_headers = 4096;
  std::size_t max_relocations = 1024 * 1024;
};

struct Segment {
  std::uint32_t type;
  std::uint32_t flags;
  std::uint64_t offset;
  std::uint64_t address;
  std::uint64_t file_size;
  std::uint64_t memory_size;
  std::uint64_t alignment;
};

// A slot the loader fills with an image location rather than with the bytes
// the file holds there. The file bytes at `address` are not that value, so a
// reader that ignores relocations reads something the run never sees.
struct RelativeRelocation {
  std::uint64_t address = 0;
  std::uint64_t target = 0;
};

// A slot the loader fills from a symbol this image defines (R_AARCH64_ABS64,
// GLOB_DAT or JUMP_SLOT): the symbol's image location plus the addend. The
// dynamic linker binds a default-visibility global to the first definition in
// its search order, which need not be this image's, so for such a symbol the
// value is this image's only if nothing interposes; `interposable` says so.
struct SymbolicRelocation {
  std::uint64_t address = 0;
  std::uint64_t target = 0;
  bool interposable = true;
};

struct FileRange {
  std::uint64_t offset;
  std::uint64_t size;
  bool operator==(const FileRange&) const = default;
};

struct LoadResult;

class Image {
 public:
  std::span<const std::uint8_t> bytes() const { return bytes_; }

  std::span<const Segment> segments() const { return segments_; }

  // These file bytes require classification; executable permission does not prove code.
  std::span<const FileRange> executable_ranges() const { return executable_ranges_; }

  std::uint16_t machine() const { return machine_; }

  Endianness endianness() const { return endianness_; }

  std::uint64_t entry() const { return entry_; }

  // True when the image declares DT_TEXTREL or DF_TEXTREL, or when the dynamic
  // array could not be followed to its terminator through its mapped bytes.
  // Its absence rules out one mutation channel only. Segment flags alone never
  // grant a lifetime constant: a mapping can be made writable later or aliased
  // by a writable one, so treating such bytes as fixed is an assumption of the
  // execution model, not something this loader checks.
  bool declares_text_relocations() const { return text_relocations_; }

  // Sorted by address. Empty when the table is absent or cannot be followed,
  // which is indistinguishable here from an image that has none.
  std::span<const RelativeRelocation> relative_relocations() const { return relatives_; }

  // Sorted by address. Only symbols the image defines in one of its own
  // sections; an import, an absolute, TLS or indirect-function symbol names no
  // image location the file can say.
  std::span<const SymbolicRelocation> symbolic_relocations() const { return symbolics_; }

  // Every slot the loader fills with an image location: the relative ones and
  // the symbolic ones, the interposable only when `own_binding` declares that
  // this image's definitions are the ones bound. Sorted by address; a slot two
  // relocations disagree about is left out.
  std::vector<RelativeRelocation> LocatedRelocations(bool own_binding) const;

  // Sorted offsets of every dynamic relocation this loader decoded, whatever
  // its kind. Complete only when loader_writes_known(); otherwise a table it
  // could not follow wrote more (see LoaderMayWrite).
  std::span<const std::uint64_t> loader_written() const { return loader_written_; }

  bool loader_writes_known() const { return loader_writes_known_; }

  // True when a dynamic relocation of any kind writes a byte of the range, and
  // also whenever the relocation tables could not all be followed or the image
  // has no dynamic array to name them, so that a file byte is never mistaken for
  // the value the loader leaves there.
  bool LoaderMayWrite(std::uint64_t address, std::uint64_t size) const;

  // File snapshot only: neither loader relocation values nor lifetime invariants.
  std::optional<std::span<const std::uint8_t>> Read(std::uint64_t address,
                                                    std::uint64_t size) const;

 private:
  friend LoadResult Load(std::vector<std::uint8_t>, Limits);
  std::vector<std::uint8_t> bytes_;
  std::vector<Segment> segments_;
  std::vector<FileRange> executable_ranges_;
  std::uint16_t machine_ = 0;
  Endianness endianness_ = Endianness::Little;
  std::uint64_t entry_ = 0;
  bool text_relocations_ = true;
  std::vector<RelativeRelocation> relatives_;
  std::vector<SymbolicRelocation> symbolics_;
  std::vector<std::uint64_t> loader_written_;

  // The offsets among them whose kind may write more than a doubleword.
  std::vector<std::uint64_t> loader_wide_;
  bool loader_writes_known_ = false;
};

struct LoadResult {
  std::optional<Image> image;
  LoadError code = LoadError::None;
  std::string error;
};

LoadResult Load(std::vector<std::uint8_t> bytes, Limits limits = {});

}  // namespace nyx::format
