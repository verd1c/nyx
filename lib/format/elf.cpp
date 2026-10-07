#include "nyx/format/elf.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

namespace nyx::format {
namespace {

bool Fits(std::uint64_t start, std::uint64_t size, std::uint64_t bound) {
  return start <= bound && size <= bound - start;
}

std::uint64_t Integer(std::span<const std::uint8_t> bytes, std::size_t offset, std::size_t size,
                      Endianness endian) {
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < size; ++i) {
    const auto index = endian == Endianness::Big ? i : size - i - 1;
    value = (value << 8) | bytes[offset + index];
  }

  return value;
}

LoadResult Failure(LoadError code, const char* message) { return {std::nullopt, code, message}; }

}  // namespace

LoadResult Load(std::vector<std::uint8_t> bytes, Limits limits) {
  if (bytes.size() > limits.max_file_bytes) {
    return Failure(LoadError::ResourceLimit, "ELF file exceeds byte budget");
  }

  if (bytes.size() < 16 || bytes[0] != 0x7f || bytes[1] != 'E' || bytes[2] != 'L' ||
      bytes[3] != 'F') {
    return Failure(LoadError::Malformed, "invalid ELF identification");
  }

  if (bytes[4] != 2) {
    return Failure(LoadError::Unsupported, "only ELF64 is supported");
  }

  if ((bytes[5] != 1 && bytes[5] != 2) || bytes[6] != 1 || bytes.size() < 64) {
    return Failure(LoadError::Malformed, "invalid ELF64 header");
  }

  const auto endian = bytes[5] == 1 ? Endianness::Little : Endianness::Big;
  const auto read = [&](std::size_t offset, std::size_t size) {
    return Integer(bytes, offset, size, endian);
  };

  if (read(20, 4) != 1 || read(52, 2) != 64) {
    return Failure(LoadError::Malformed, "invalid ELF version or header size");
  }

  if (read(16, 2) != 2 && read(16, 2) != 3) {
    return Failure(LoadError::Unsupported, "ELF must be executable or shared object");
  }

  const auto count = read(56, 2);
  if (count == 0xffff) {
    return Failure(LoadError::Unsupported, "extended program header numbering is unsupported");
  }

  if (count > limits.max_program_headers) {
    return Failure(LoadError::ResourceLimit, "ELF program headers exceed budget");
  }

  const auto phoff = read(32, 8);
  if ((count != 0 && (read(54, 2) != 56 || phoff < 64)) || !Fits(phoff, count * 56, bytes.size())) {
    return Failure(LoadError::Malformed, "invalid ELF program header table");
  }

  Image image;
  image.machine_ = static_cast<std::uint16_t>(read(18, 2));
  image.entry_ = read(24, 8);
  image.endianness_ = endian;
  image.segments_.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i) {
    const auto base = static_cast<std::size_t>(phoff + i * 56);
    Segment segment{static_cast<std::uint32_t>(read(base, 4)),
                    static_cast<std::uint32_t>(read(base + 4, 4)),
                    read(base + 8, 8),
                    read(base + 16, 8),
                    read(base + 32, 8),
                    read(base + 40, 8),
                    read(base + 48, 8)};
    // PT_NULL fields have no meaning and need not be valid extents.
    if (segment.type != 0 && !Fits(segment.offset, segment.file_size, bytes.size())) {
      return Failure(LoadError::Malformed, "segment file extent exceeds ELF bytes");
    }

    if (segment.type == 1) {
      if (segment.file_size > segment.memory_size ||
          !Fits(segment.address, segment.memory_size, std::numeric_limits<std::uint64_t>::max())) {
        return Failure(LoadError::Malformed, "invalid load segment memory extent");
      }

      const auto align = segment.alignment;
      if (align > 1 &&
          ((align & (align - 1)) != 0 || segment.address % align != segment.offset % align)) {
        return Failure(LoadError::Malformed, "invalid load segment alignment");
      }

      if ((segment.flags & 1) != 0 && segment.file_size != 0) {
        image.executable_ranges_.push_back({segment.offset, segment.file_size});
      }
    }

    image.segments_.push_back(segment);
  }

  auto& ranges = image.executable_ranges_;
  std::sort(ranges.begin(), ranges.end(), [](const FileRange& left, const FileRange& right) {
    return left.offset < right.offset;
  });
  std::size_t used = 0;
  for (const auto range : ranges) {
    if (used != 0 && range.offset <= ranges[used - 1].offset + ranges[used - 1].size) {
      auto& last = ranges[used - 1];
      last.size = std::max(last.offset + last.size, range.offset + range.size) - last.offset;
    } else {
      ranges[used++] = range;
    }
  }

  ranges.resize(used);
  image.bytes_ = std::move(bytes);

  // PT_DYNAMIC entries are pairs of 64-bit tag and value. Absent a dynamic
  // section there is nothing to relocate through. Every other outcome fails
  // closed, because a table this scan cannot follow is one whose contents it
  // cannot rule on.
  image.text_relocations_ = false;

  // A static image has no dynamic array to name its relocations, yet its startup
  // code may still apply some (IRELATIVE through __rela_iplt), so only a dynamic
  // array that is followed completely can say what the loader leaves alone.
  image.loader_writes_known_ =
      std::any_of(image.segments_.begin(), image.segments_.end(),
                  [](const Segment& segment) { return segment.type == 2; });
  for (const auto& segment : image.segments_) {
    if (segment.type != 2) continue;

    // A loader walks this array from its virtual address until DT_NULL and
    // never consults the file extent, so read the bytes it would read and
    // require the terminator inside the extent: an extent that stops short
    // hides whatever follows it.
    const auto mapped = image.Read(segment.address, segment.file_size);
    if (segment.file_size == 0 || segment.file_size % 16 != 0 || !mapped) {
      image.text_relocations_ = true;
      image.loader_writes_known_ = false;
      continue;
    }

    const auto word = [&](std::uint64_t at) {
      std::uint64_t value = 0;
      for (unsigned i = 0; i < 8; ++i) {
        const auto byte = static_cast<std::uint64_t>((*mapped)[static_cast<std::size_t>(at) + i]);
        value |= image.endianness_ == Endianness::Little ? byte << (8 * i) : byte << (8 * (7 - i));
      }

      return value;
    };

    bool terminated = false;
    for (std::uint64_t at = 0; at + 16 <= segment.file_size; at += 16) {
      const auto tag = word(at);
      const auto value = word(at + 8);
      if (tag == 0) {  // DT_NULL
        terminated = true;
        break;
      }

      if (tag == 22) image.text_relocations_ = true;                      // DT_TEXTREL
      if (tag == 30 && (value & 4) != 0) image.text_relocations_ = true;  // DF_TEXTREL
    }

    if (!terminated) image.text_relocations_ = true;

    // The same walk carries the relocation tables' locations. Every entry is a
    // byte range the loader writes, whatever its kind; only the relative kind
    // is also kept as a value, because it names an image location with no
    // symbol to resolve, so what the loader writes is known from the file alone.
    std::uint64_t rela = 0, rela_size = 0, rela_entry = 0;
    std::uint64_t jmprel = 0, jmprel_size = 0, jmprel_kind = 0;
    std::uint64_t symtab = 0, symbol_entry = 0;
    bool symbolic = false;
    bool foreign = false;
    for (std::uint64_t at = 0; at + 16 <= segment.file_size; at += 16) {
      const auto tag = word(at);
      const auto value = word(at + 8);
      if (tag == 0) break;
      if (tag == 6) symtab = value;                        // DT_SYMTAB
      if (tag == 11) symbol_entry = value;                 // DT_SYMENT
      if (tag == 16) symbolic = true;                      // DT_SYMBOLIC
      if (tag == 30 && (value & 2) != 0) symbolic = true;  // DF_SYMBOLIC
      if (tag == 7) rela = value;                          // DT_RELA
      if (tag == 8) rela_size = value;                     // DT_RELASZ
      if (tag == 9) rela_entry = value;                    // DT_RELAENT
      if (tag == 23) jmprel = value;                       // DT_JMPREL
      if (tag == 2) jmprel_size = value;                   // DT_PLTRELSZ
      if (tag == 20) jmprel_kind = value;                  // DT_PLTREL
      // DT_REL, DT_RELR and Android's packed forms write through tables this
      // scan does not decode, so it can no longer say what the loader leaves.
      if (tag == 17 || tag == 36 || tag == 0x6000000f || tag == 0x60000011 || tag == 0x6fffe000)
        foreign = true;
    }

    if (!terminated || foreign) image.loader_writes_known_ = false;

    // The symbol a relocation names, read at its index the way the loader
    // reads it, so no table size is needed: an index past what the image maps
    // simply yields nothing. Returns its image location and whether a
    // definition elsewhere could be bound in its place.
    const auto define = [&](std::uint64_t index) -> std::optional<std::pair<std::uint64_t, bool>> {
      if (index == 0 || symtab == 0 || symbol_entry != 24 ||
          index > (std::numeric_limits<std::uint64_t>::max() - symtab) / 24)
        return std::nullopt;
      const auto symbol = image.Read(symtab + index * 24, 24);
      if (!symbol) return std::nullopt;
      const auto field = [&](std::size_t at, std::size_t size) {
        return Integer(*symbol, at, size, image.endianness_);
      };

      const auto info = field(4, 1), other = field(5, 1), section = field(6, 2);

      // SHN_UNDEF is an import; the reserved range (SHN_ABS, SHN_COMMON, ...)
      // is not a location in this image. STT_TLS names a TLS-block offset and
      // STT_GNU_IFUNC a resolver whose result is the value, so neither is S.
      const auto type = info & 0xf;
      if (section == 0 || section >= 0xff00 || type == 6 || type == 10) return std::nullopt;
      const auto binding = info >> 4, visibility = other & 3;
      const bool interposable =
          !symbolic && binding != 0 && visibility == 0;  // STB_LOCAL, STV_DEFAULT
      return std::pair{field(8, 8), interposable};
    };

    const auto scan = [&](std::uint64_t address, std::uint64_t size, bool relatives) {
      if (size == 0) return true;
      if (size % 24 != 0 || size / 24 > limits.max_relocations) return false;
      const auto table = image.Read(address, size);
      if (!table) return false;
      const auto entry = [&](std::uint64_t at) {
        std::uint64_t value = 0;
        for (unsigned i = 0; i < 8; ++i) {
          const auto byte = static_cast<std::uint64_t>((*table)[static_cast<std::size_t>(at) + i]);
          value |=
              image.endianness_ == Endianness::Little ? byte << (8 * i) : byte << (8 * (7 - i));
        }

        return value;
      };

      image.loader_written_.reserve(image.loader_written_.size() +
                                    static_cast<std::size_t>(size / 24));
      for (std::uint64_t at = 0; at < size; at += 24) {
        // A copy relocation writes the symbol's whole size, not a slot, so the
        // bytes it reaches are not the ones its offset names.
        if ((entry(at + 8) & 0xffffffff) == 1024) return false;  // R_AARCH64_COPY
        image.loader_written_.push_back(entry(at));
        const auto kind = entry(at + 8) & 0xffffffff;

        // ABS64, GLOB_DAT, JUMP_SLOT, RELATIVE, the TLS module, offset and
        // thread-pointer words, and IRELATIVE each write one doubleword and
        // NONE writes nothing. Anything else, a TLS descriptor among them,
        // is taken to reach the widest any relocation does.
        if (kind != 0 && kind != 257 && (kind < 1025 || kind > 1032 || kind == 1031))
          image.loader_wide_.push_back(entry(at));
        if (relatives && kind == 1027) {  // R_AARCH64_RELATIVE
          image.relatives_.push_back({entry(at), entry(at + 16)});
        }

        // R_AARCH64_ABS64, GLOB_DAT and JUMP_SLOT all write S + A.
        if (kind == 257 || kind == 1025 || kind == 1026) {
          if (const auto located = define(entry(at + 8) >> 32))
            image.symbolics_.push_back(
                {entry(at), located->first + entry(at + 16), located->second});
        }
      }

      return true;
    };

    if ((rela_size != 0 && rela_entry != 24) || !scan(rela, rela_size, true)) {
      image.loader_writes_known_ = false;
    }

    // DT_PLTREL names the entry format; 7 is DT_RELA, the only one decoded.
    if (jmprel_size != 0 && (jmprel_kind != 7 || !scan(jmprel, jmprel_size, false))) {
      image.loader_writes_known_ = false;
    }

    std::sort(image.relatives_.begin(), image.relatives_.end(),
              [](const RelativeRelocation& left, const RelativeRelocation& right) {
                return left.address < right.address;
              });
    std::sort(image.symbolics_.begin(), image.symbolics_.end(),
              [](const SymbolicRelocation& left, const SymbolicRelocation& right) {
                return left.address < right.address;
              });
  }

  std::sort(image.loader_written_.begin(), image.loader_written_.end());
  std::sort(image.loader_wide_.begin(), image.loader_wide_.end());
  return {std::move(image), LoadError::None, {}};
}

std::optional<std::span<const std::uint8_t>> Image::Read(std::uint64_t address,
                                                         std::uint64_t size) const {
  if (size == 0 || !Fits(address, size, std::numeric_limits<std::uint64_t>::max())) {
    return std::nullopt;
  }

  const Segment* backing = nullptr;
  for (const auto& segment : segments_) {
    if (segment.type != 1 || segment.memory_size == 0) continue;
    const auto end = segment.address + segment.memory_size;
    if (address >= end || address + size <= segment.address) continue;

    // Partial overlap is ambiguous too, even if another mapping backs the full read.
    if (backing != nullptr || address < segment.address ||
        !Fits(address - segment.address, size, segment.file_size)) {
      return std::nullopt;
    }

    backing = &segment;
  }

  if (backing == nullptr) return std::nullopt;
  const auto offset = backing->offset + address - backing->address;
  return std::span<const std::uint8_t>(bytes_).subspan(static_cast<std::size_t>(offset),
                                                       static_cast<std::size_t>(size));
}

std::vector<RelativeRelocation> Image::LocatedRelocations(bool own_binding) const {
  std::vector<RelativeRelocation> located(relatives_.begin(), relatives_.end());
  for (const auto& symbolic : symbolics_)
    if (own_binding || !symbolic.interposable)
      located.push_back({symbolic.address, symbolic.target});
  std::stable_sort(located.begin(), located.end(),
                   [](const RelativeRelocation& left, const RelativeRelocation& right) {
                     return left.address < right.address;
                   });
  // Two relocations of one slot apply in table order, and which order that
  // was is not kept here; a slot they disagree about has no single value.
  std::vector<RelativeRelocation> unique;
  unique.reserve(located.size());
  for (std::size_t i = 0; i < located.size();) {
    std::size_t end = i + 1;
    bool agree = true;
    while (end < located.size() && located[end].address == located[i].address) {
      agree = agree && located[end].target == located[i].target;
      ++end;
    }

    if (agree) unique.push_back(located[i]);
    i = end;
  }

  return unique;
}

bool Image::LoaderMayWrite(std::uint64_t address, std::uint64_t size) const {
  if (!loader_writes_known_ || size == 0 ||
      address > std::numeric_limits<std::uint64_t>::max() - size) {
    return true;
  }

  // Copy relocations leave the writes unknown, and no other AArch64 dynamic
  // relocation writes more than sixteen bytes at its offset (a TLS descriptor is
  // the widest). A doubleword slot reaches the range only when it starts fewer
  // than eight bytes before it: counting sixteen for every kind refuses the
  // word just past each relocated slot, which is where a table's key sits.
  const auto reaches = [&](const std::vector<std::uint64_t>& starts, std::uint64_t width) {
    const auto low = address < width - 1 ? 0 : address - (width - 1);
    const auto first = std::lower_bound(starts.begin(), starts.end(), low);
    return first != starts.end() && *first < address + size;
  };

  return reaches(loader_written_, 8) || reaches(loader_wide_, 16);
}

}  // namespace nyx::format
