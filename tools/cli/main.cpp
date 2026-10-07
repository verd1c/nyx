#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unistd.h>
#include <utility>
#include <vector>

#include <sys/stat.h>

#include "nyx/analysis/cfg.hpp"
#include "nyx/analysis/entry_relations.hpp"
#include "nyx/analysis/image_writes.hpp"
#include "nyx/analysis/path_control.hpp"
#include "nyx/analysis/regions.hpp"
#include "nyx/analysis/ssa/build.hpp"
#include "nyx/analysis/unflatten.hpp"
#include "nyx/eval/program.hpp"
#include "nyx/format/elf.h"
#include "nyx/ir/print.hpp"
#include "nyx/passes/pipeline.hpp"
#include "nyx/recovery/control.hpp"
#include "nyx/recovery/image.hpp"
#include "nyx/recovery/mba.hpp"
#include "nyx/recovery/memory.hpp"
#include "nyx/recovery/overwritten_store.hpp"
#include "nyx/recovery/paired_load.hpp"
#include "nyx/support/sha256.hpp"
#include "nyx/target/a64/abi.hpp"
#include "nyx/target/a64/decode.hpp"
#include "nyx/target/a64/listing.hpp"
#include "nyx/verify/acceptance.hpp"

#include "ssa_state.hpp"

namespace {
std::string_view MemoryProfileName(nyx::a64::MemoryProfile profile) {
  switch (profile) {
    case nyx::a64::MemoryProfile::none:
      return "none";
    case nyx::a64::MemoryProfile::concrete_atomic_scalar:
      return "concrete_atomic_scalar";
    case nyx::a64::MemoryProfile::concrete_exclusive_scalar:
      return "concrete_exclusive_scalar";
  }

  return "none";
}

bool ParseMemoryProfile(std::string_view name, nyx::a64::MemoryProfile& profile) {
  if (profile != nyx::a64::MemoryProfile::none) return false;
  if (name == "concrete_atomic_scalar") {
    profile = nyx::a64::MemoryProfile::concrete_atomic_scalar;
  } else if (name == "concrete_exclusive_scalar") {
    profile = nyx::a64::MemoryProfile::concrete_exclusive_scalar;
  } else {
    return false;
  }

  return true;
}

bool ValidUtf8(std::string_view value) {
  for (std::size_t i = 0; i < value.size();) {
    const auto first = static_cast<unsigned char>(value[i++]);
    if (first < 0x80) continue;
    unsigned remaining = 0;
    std::uint32_t code = 0;
    std::uint32_t minimum = 0;
    if (first >= 0xc2 && first <= 0xdf) {
      remaining = 1;
      code = first & 0x1f;
      minimum = 0x80;
    } else if (first >= 0xe0 && first <= 0xef) {
      remaining = 2;
      code = first & 0x0f;
      minimum = 0x800;
    } else if (first >= 0xf0 && first <= 0xf4) {
      remaining = 3;
      code = first & 0x07;
      minimum = 0x10000;
    } else {
      return false;
    }

    if (remaining > value.size() - i) return false;
    while (remaining-- != 0) {
      const auto next = static_cast<unsigned char>(value[i++]);
      if ((next & 0xc0) != 0x80) return false;
      code = (code << 6) | (next & 0x3f);
    }

    if (code < minimum || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) {
      return false;
    }
  }

  return true;
}

std::string HexBytes(std::string_view value) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(value.size() * 2);
  for (const unsigned char ch : value) {
    out += kHex[ch >> 4];
    out += kHex[ch & 15];
  }

  return out;
}

std::string JsonString(std::string_view value) {
  // Linux filenames are arbitrary bytes; input_bytes_hex preserves invalid UTF-8.
  if (!ValidUtf8(value)) return "null";
  constexpr char kHex[] = "0123456789abcdef";
  std::size_t extent = 2;
  for (const unsigned char ch : value) extent += ch < 0x20 ? 6 : ch == '"' || ch == '\\' ? 2 : 1;
  std::string out;
  out.reserve(extent);
  out += '"';
  for (const unsigned char ch : value) {
    if (ch == '"' || ch == '\\') {
      out += '\\';
      out += static_cast<char>(ch);
    } else if (ch < 0x20) {
      out += "\\u00";
      out += kHex[ch >> 4];
      out += kHex[ch & 15];
    } else {
      out += static_cast<char>(ch);
    }
  }

  out += '"';
  return out;
}

int Error(std::string_view kind, std::string_view message) {
  std::cout << "{\"schema\":1,\"outcome\":\"refused\",\"reason\":" << JsonString(kind)
            << ",\"detail\":" << JsonString(message) << "}\n";
  return 1;
}

std::optional<nyx::format::Image> ReadImage(const char* path, nyx::Budget* budget = nullptr) {
  const nyx::format::Limits limits;

  // A pre-open path check races replacement with a FIFO; inspect the opened fd.
  const int fd = open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) {
    Error("input_io", "Cannot open input");
    return std::nullopt;
  }

  struct CloseFile {
    int fd;

    ~CloseFile() { close(fd); }
  } close_file{fd};
  struct stat metadata{};
  if (fstat(fd, &metadata) != 0 || !S_ISREG(metadata.st_mode) || metadata.st_size < 0) {
    Error("input_io", "Input must be a regular file with a known size");
    return std::nullopt;
  }

  const auto size = static_cast<std::uint64_t>(metadata.st_size);
  if (size > limits.max_file_bytes) {
    Error("resource_limit", "Input exceeds byte limit");
    return std::nullopt;
  }

  const auto allowance = size + limits.max_program_headers *
                                    (sizeof(nyx::format::Segment) + sizeof(nyx::format::FileRange));
  if (budget && budget->try_consume({size, allowance}) != nyx::BudgetDecline::none) {
    Error("resource_limit", "Input and loader allocation exceed lift budget");
    return std::nullopt;
  }

  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  std::size_t used = 0;
  while (used < bytes.size()) {
    const auto count = read(fd, bytes.data() + used, bytes.size() - used);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      Error("input_io", "Input changed or could not be read completely");
      return std::nullopt;
    }

    used += static_cast<std::size_t>(count);
  }

  auto loaded = nyx::format::Load(std::move(bytes), limits);
  if (!loaded.image) {
    const char* reason = "malformed_elf";
    if (loaded.code == nyx::format::LoadError::Unsupported) reason = "unsupported_elf";
    if (loaded.code == nyx::format::LoadError::ResourceLimit) reason = "resource_limit";
    Error(reason, loaded.error);
    return std::nullopt;
  }

  return std::move(loaded.image);
}

int Inspect(const char* path) {
  auto loaded = ReadImage(path);
  if (!loaded) return 1;
  const auto& image = *loaded;
  std::cout << "{\"schema\":1,\"outcome\":\"inspected\",\"input\":" << JsonString(path)
            << ",\"input_bytes_hex\":" << JsonString(HexBytes(path))
            << ",\"file_size\":" << image.bytes().size() << ",\"machine\":" << image.machine()
            << ",\"endianness\":\""
            << (image.endianness() == nyx::format::Endianness::Little ? "little" : "big")
            << "\",\"entry\":" << image.entry() << ",\"segments\":[";
  bool first = true;
  for (const auto& s : image.segments()) {
    if (!first) std::cout << ',';
    first = false;
    std::cout << "{\"type\":" << s.type << ",\"flags\":" << s.flags << ",\"offset\":" << s.offset
              << ",\"address\":" << s.address << ",\"file_size\":" << s.file_size
              << ",\"memory_size\":" << s.memory_size << ",\"alignment\":" << s.alignment << '}';
  }

  std::cout << "],\"executable_file_ranges\":[";
  first = true;
  std::uint64_t total = 0;
  for (const auto& range : image.executable_ranges()) {
    if (!first) std::cout << ',';
    first = false;
    total += range.size;
    std::cout << "{\"offset\":" << range.offset << ",\"size\":" << range.size << '}';
  }

  std::cout << "],\"executable_file_bytes\":" << total
            << ",\"classification\":\"not_checked\",\"relocations\":\"not_checked\"}\n";
  return 0;
}

bool Unsigned(std::string_view text, std::uint64_t& value, int base) {
  if (text.starts_with("0x") || text.starts_with("0X")) {
    text.remove_prefix(2);
    base = 16;
  }

  if (text.empty()) return false;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value, base);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

const char* DecodeReason(nyx::a64::DecodeDecline reason) {
  switch (reason) {
    case nyx::a64::DecodeDecline::none:
      return "none";
    case nyx::a64::DecodeDecline::unsupported:
      return "unsupported";
    case nyx::a64::DecodeDecline::invalid_encoding:
      return "invalid_encoding";
    case nyx::a64::DecodeDecline::invalid_location:
      return "invalid_location";
    case nyx::a64::DecodeDecline::work_limit:
      return "work_limit";
    case nyx::a64::DecodeDecline::byte_limit:
      return "byte_limit";
  }

  return "invalid_decline";
}

const char* RuleName(nyx::recovery::MbaRule rule) {
  switch (rule) {
    case nyx::recovery::MbaRule::or_xor_sum:
      return "or_xor_sum";
    case nyx::recovery::MbaRule::and_xor_union:
      return "and_xor_union";
    case nyx::recovery::MbaRule::or_and_sum:
      return "or_and_sum";
    case nyx::recovery::MbaRule::xor_and_sum:
      return "xor_and_sum";
    case nyx::recovery::MbaRule::xor_ones_not:
      return "xor_ones_not";
    case nyx::recovery::MbaRule::negated_add:
      return "negated_add";
    case nyx::recovery::MbaRule::linear_identity:
      return "linear_identity";
    case nyx::recovery::MbaRule::linear_direct:
      return "linear_direct";
    case nyx::recovery::MbaRule::constant_fold:
      return "constant_fold";
  }

  return "invalid";
}

// One maximal run of the requested region that a block admits: contiguous
// source groups carrying control only at the final one. A region the caller
// names may hold several of these, and cutting between them establishes
// nothing about whether the later runs are entered. Each is recovered as if
// entered at its own first instruction, which is the restriction a single-run
// request already declares and publishes.
struct SimplifiedRun {
  std::uint64_t address;
  std::uint64_t size;
  std::size_t source_groups;
  std::uint64_t mba;
  std::uint64_t folds;
  std::vector<nyx::recovery::MbaEdit> journal;
  std::string before;
  std::string after;
};

// A word whose effects this target does not model. It belongs to no run: it
// closes the run before it and the next run starts after it, so its unknown
// effects are never folded into either side.
struct UnmodeledWord {
  std::uint64_t address;
  std::array<std::uint8_t, 4> bytes;
  const char* reason;
};

void PrintJournal(std::span<const nyx::recovery::MbaEdit> journal) {
  bool first = true;
  for (const auto& edit : journal) {
    if (!first) std::cout << ',';
    first = false;
    std::cout << "{\"rule\":\"" << RuleName(edit.rule) << "\",\"node\":" << edit.node
              << ",\"width\":" << edit.width << ",\"from_revision\":" << edit.from_revision
              << ",\"to_revision\":" << edit.to_revision;
    for (const auto& entry :
         {std::pair{"original", &edit.original}, std::pair{"replacement", &edit.replacement}}) {
      const auto* descriptor = nyx::ir::Descriptor(entry.second->op);
      std::cout << ",\"" << entry.first << "\":{\"op\":\"" << descriptor->name << "\",\"inputs\":[";
      for (unsigned i = 0; i < descriptor->arity; ++i) {
        if (i) std::cout << ',';
        std::cout << entry.second->inputs[i];
      }

      std::cout << "],\"immediate\":" << entry.second->immediate << '}';
    }

    std::cout << '}';
  }
}

int Simplify(const char* path, std::uint64_t address, std::uint64_t size,
             std::span<const nyx::ir::Group> groups, nyx::Budget& budget,
             nyx::a64::MemoryProfile profile, std::span<const UnmodeledWord> unmodeled = {}) {
  if (groups.empty()) {
    return Error("unsupported_region",
                 "Region holds no instruction whose effects this target models");
  }

  if (budget.try_consume({groups.size(), groups.size() * sizeof(SimplifiedRun)}) !=
      nyx::BudgetDecline::none) {
    return Error("resource_limit", "Run table exceeds aggregate budget");
  }

  std::vector<SimplifiedRun> runs;
  std::uint64_t mba = 0, folds = 0, edits = 0;
  for (std::size_t first = 0; first < groups.size();) {
    // A group carrying a transfer ends its run, and so does a gap in the
    // source addresses, which an unmodeled word leaves behind.
    std::size_t last = first;
    while (last + 1 < groups.size() && !groups[last].transfer() &&
           groups[last].source_address() + groups[last].bytes().size() ==
               groups[last + 1].source_address())
      ++last;
    const auto run = groups.subspan(first, last + 1 - first);
    first = last + 1;
    auto normalized = nyx::ir::Normalize(run, budget);
    if (!normalized.block) {
      return Error(normalized.reason == nyx::ir::BlockDecline::resource_limit
                       ? "resource_limit"
                       : "unsupported_region",
                   "Require contiguous runs carrying control only at their final instruction");
    }

    auto simplified = nyx::recovery::SimplifyMba(*normalized.block, budget);
    if (!simplified.block) {
      const char* reason =
          simplified.reason == nyx::recovery::MbaDecline::resource_limit      ? "resource_limit"
          : simplified.reason == nyx::recovery::MbaDecline::revision_overflow ? "revision_overflow"
                                                                              : "invalid_group";
      return Error(reason, "No recovery artifact or partial journal published");
    }

    auto before = nyx::ir::PrintJson(*normalized.block, budget);
    auto after = nyx::ir::PrintJson(*simplified.block, budget);
    if (!before.json || !after.json) {
      const auto reason = !before.json ? before.reason : after.reason;
      return Error(
          reason == nyx::ir::PrintDecline::invalid_group ? "invalid_group" : "resource_limit",
          "Recovery serialization declined; no partial artifact published");
    }

    std::uint64_t run_mba = 0, run_folds = 0, run_bytes = 0;
    std::optional<nyx::ir::ValueId> last_mba;
    for (const auto& edit : simplified.journal) {
      if (edit.rule == nyx::recovery::MbaRule::constant_fold)
        ++run_folds;
      else if (!last_mba || *last_mba != edit.node) {
        ++run_mba;
        last_mba = edit.node;
      }
    }

    for (const auto& group : run) run_bytes += group.bytes().size();
    mba += run_mba;
    folds += run_folds;
    edits += simplified.journal.size();
    runs.push_back({run.front().source_address(), run_bytes, run.size(), run_mba, run_folds,
                    std::move(simplified.journal), std::move(*before.json),
                    std::move(*after.json)});
  }

  std::uint64_t journal_entries = 0, serialized = 0;
  for (const auto& run : runs) {
    journal_entries += run.journal.size();
    serialized += run.before.size() + run.after.size();
  }

  const auto envelope = std::string_view(path).size() * 16 + 4096 + journal_entries * 768 +
                        runs.size() * 256 + unmodeled.size() * 128;
  if (budget.try_consume({envelope + serialized, envelope}) != nyx::BudgetDecline::none) {
    return Error("resource_limit",
                 "Recovery publication exceeds budget; no partial artifact published");
  }

  std::cout
      << "{\"schema\":1,\"outcome\":\"" << (edits == 0 ? "unchanged" : "simplified")
      << "\",\"input\":" << JsonString(path)
      << ",\"input_bytes_hex\":" << JsonString(HexBytes(path)) << ",\"address\":" << address
      << ",\"size\":" << size << ",\"source_groups\":" << groups.size()
      << ",\"straight_line_runs\":" << runs.size()
      << ",\"unmodeled_source_groups\":" << unmodeled.size()
      << ",\"scope\":\"single_entry_straight_line\",\"interior_entry_analysis\":\"not_performed\","
         "\"execution_assumptions\":[\"entry at each run's first source group\",\"immutable "
         "code\",\"BTI inactive\",\"GCS inactive\",\"pointer authentication inactive\"],"
         "\"memory_profile\":\""
      << MemoryProfileName(profile)
      << "\",\"memory_invariance\":\"not_assumed\",\"architectural_fault_authority\":false,"
         "\"sp_alignment_check\":\"disabled\",\"memory_fault_scope\":\"experimental_QEMU_"
         "reference\","
         "\"mba_expressions_simplified\":"
      << mba << ",\"constant_folds\":" << folds << ",\"applied_edits\":" << edits
      << ",\"removed_source_groups\":0,\"removed_memory_effects\":0,\"machine_patching\":\"not_"
         "performed\","
         "\"proof_model\":\"closed_bitvector_identities\"";
  // A region that is one run covering every requested word keeps publishing at
  // the top level, the shape callers of the single-block form already read.
  // Anything else states each run's own extent, so what the region covers is
  // never left to be inferred from the requested range; a dropped word used to
  // refuse the whole request, so no caller could have read that shape before.
  if (runs.size() == 1 && unmodeled.empty()) {
    std::cout << ",\"journal\":[";
    PrintJournal(runs.front().journal);
    std::cout << "],\"before\":" << runs.front().before << ",\"after\":" << runs.front().after;
  } else {
    std::cout << ",\"runs\":[";
    bool first = true;
    for (const auto& run : runs) {
      if (!first) std::cout << ',';
      first = false;
      std::cout << "{\"address\":" << run.address << ",\"size\":" << run.size
                << ",\"source_groups\":" << run.source_groups << ",\"outcome\":\""
                << (run.journal.empty() ? "unchanged" : "simplified")
                << "\",\"mba_expressions_simplified\":" << run.mba
                << ",\"constant_folds\":" << run.folds
                << ",\"applied_edits\":" << run.journal.size() << ",\"journal\":[";
      PrintJournal(run.journal);
      std::cout << "],\"before\":" << run.before << ",\"after\":" << run.after << '}';
    }

    std::cout << ']';
  }

  if (!unmodeled.empty()) {
    std::cout << ",\"unmodeled\":[";
    bool first = true;
    for (const auto& word : unmodeled) {
      if (!first) std::cout << ',';
      first = false;
      char bytes[9];
      const auto length =
          std::snprintf(bytes, sizeof(bytes), "%02x%02x%02x%02x",
                        static_cast<unsigned>(word.bytes[0]), static_cast<unsigned>(word.bytes[1]),
                        static_cast<unsigned>(word.bytes[2]), static_cast<unsigned>(word.bytes[3]));
      if (length != 8)
        return Error("invalid_group", "Unmodeled word record exceeds its checked buffer");
      std::cout << "{\"address\":" << word.address << ",\"bytes\":\"" << bytes << "\",\"reason\":\""
                << word.reason << "\"}";
    }

    std::cout << ']';
  }

  std::cout << ",\"resources\":{\"work\":" << budget.used().work
            << ",\"bytes\":" << budget.used().bytes << "}}\n";
  return std::cout ? 0 : 1;
}

const char* EdgeName(nyx::analysis::CfgEdgeKind kind) {
  using nyx::analysis::CfgEdgeKind;
  switch (kind) {
    case CfgEdgeKind::fallthrough:
      return "fallthrough";
    case CfgEdgeKind::branch:
      return "branch";
    case CfgEdgeKind::callee:
      return "callee";
    case CfgEdgeKind::return_:
      return "return";
    case CfgEdgeKind::potential_return:
      return "potential_return";
    case CfgEdgeKind::opaque_unknown:
      return "opaque_unknown";
    case CfgEdgeKind::trap:
      return "trap";
  }

  return "invalid";
}

const char* ResolutionName(nyx::analysis::TargetResolution resolution) {
  using nyx::analysis::TargetResolution;
  switch (resolution) {
    case TargetResolution::block_entry:
      return "block_entry";
    case TargetResolution::opaque_entry:
      return "opaque_entry";
    case TargetResolution::mid_instruction:
      return "mid_instruction";
    case TargetResolution::outside_population:
      return "outside_population";
    case TargetResolution::absolute_runtime:
      return "absolute_runtime";
    case TargetResolution::unknown:
      return "unknown";
  }

  return "invalid";
}

const char* TransferName(nyx::ir::TransferKind kind) {
  switch (kind) {
    case nyx::ir::TransferKind::jump:
      return "jump";
    case nyx::ir::TransferKind::conditional:
      return "conditional";
    case nyx::ir::TransferKind::call:
      return "call";
    case nyx::ir::TransferKind::return_:
      return "return";
  }

  return "invalid";
}

void PrintJournalNode(std::ostream& out, const char* name, const nyx::ir::Node& node) {
  const auto* descriptor = nyx::ir::Descriptor(node.op);
  out << ",\"" << name << "\":{\"op\":\"" << descriptor->name << "\",\"inputs\":[";
  for (unsigned i = 0; i < descriptor->arity; ++i) {
    if (i) out << ',';
    out << node.inputs[i];
  }

  out << "],\"immediate\":" << node.immediate << '}';
}

const char* ImageRuleName(nyx::recovery::ImageRule rule) {
  switch (rule) {
    case nyx::recovery::ImageRule::constant_fold:
      return "constant_fold";
    case nyx::recovery::ImageRule::image_offset:
      return "image_offset";
    case nyx::recovery::ImageRule::page_base:
      return "page_base";
  }

  return "unknown";
}

void PrintRecoveryJournal(std::ostream& out,
                          const nyx::recovery::PathMemoryRecoveryResult& forwarded,
                          const nyx::recovery::MbaPathResult& simplified,
                          const nyx::recovery::ImagePathResult& folded) {
  bool first = true;
  for (const auto& fact : forwarded.facts) {
    if (!first) out << ',';
    first = false;
    const auto base = fact.address.base == nyx::recovery::AddressBase::absolute    ? "absolute"
                      : fact.address.base == nyx::recovery::AddressBase::load_bias ? "load_bias"
                                                                                   : "ssa_value";
    out << "{\"store\":" << fact.store << ",\"load\":" << fact.load << ",\"value\":" << fact.value
        << ",\"width\":" << fact.width << ",\"byte_order\":\""
        << (fact.byte_order == nyx::ir::ByteOrder::little ? "little" : "big")
        << "\",\"address\":{\"base\":\"" << base << "\",\"value\":" << fact.address.value
        << ",\"offset\":" << fact.address.offset
        << "},\"scope\":\"successful_itinerary_prefix\",\"rests_on\":["
        << (fact.entry_relation ? "\"entry_relation\"" : "")
        << "],\"from_revision\":" << fact.from_revision << ",\"to_revision\":" << fact.to_revision
        << '}';
  }

  out << "],\"memory_journal\":[";
  constexpr std::array use_names{"operand",   "final_write", "target",
                                 "condition", "alternative", "continuation"};
  first = true;
  for (const auto& edit : forwarded.journal) {
    if (!first) out << ',';
    first = false;
    out << "{\"fact\":" << edit.fact << ",\"use\":\"" << use_names[static_cast<unsigned>(edit.use)]
        << "\",\"owner\":" << edit.owner << ",\"slot\":" << edit.slot
        << ",\"original\":" << edit.original << ",\"replacement\":" << edit.replacement
        << ",\"from_revision\":" << edit.from_revision << ",\"to_revision\":" << edit.to_revision
        << '}';
  }

  out << "],\"arithmetic_journal\":[";
  first = true;
  for (const auto& edit : simplified.journal) {
    if (!first) out << ',';
    first = false;
    out << "{\"rule\":\"" << RuleName(edit.rule) << "\",\"node\":" << edit.node
        << ",\"width\":" << edit.width << ",\"from_revision\":" << edit.from_revision
        << ",\"to_revision\":" << edit.to_revision;
    PrintJournalNode(out, "original", edit.original);
    PrintJournalNode(out, "replacement", edit.replacement);
    out << '}';
  }

  // Each image edit names the declared restrictions it rests on, so a reader can
  // tell a fold that holds at any placement from one that holds only if the
  // caller's declarations do.
  out << "],\"image_journal\":[";
  first = true;
  for (const auto& edit : folded.journal) {
    if (!first) out << ',';
    first = false;
    out << "{\"rule\":\"" << ImageRuleName(edit.rule) << "\",\"node\":" << edit.node
        << ",\"from_revision\":" << edit.from_revision << ",\"to_revision\":" << edit.to_revision
        << ",\"rests_on\":[";
    const char* separator = "";
    for (const auto& [held, name] : {std::pair{edit.constant_bytes, "constant_bytes"},
                                     std::pair{edit.relocated_slot, "relocated_slot"},
                                     std::pair{edit.page_placement, "page_aligned_placement"}}) {
      if (!held) continue;
      out << separator << '"' << name << '"';
      separator = ",";
    }

    out << ']';
    PrintJournalNode(out, "original", edit.original);
    PrintJournalNode(out, "replacement", edit.replacement);
    out << '}';
  }

  // A declared value this path writes is dropped before any edit is published.
  out << "],\"declared_ranges_written\":[";
  first = true;
  for (const auto& written : folded.contradicted) {
    if (!first) out << ',';
    first = false;
    out << "{\"range\":" << written.range << ",\"store\":" << written.store
        << ",\"address\":" << written.address << ",\"width\":" << written.width << '}';
  }

  out << "],\"relocated_slots_written\":[";
  first = true;
  for (const auto& written : folded.contradicted_pointers) {
    if (!first) out << ',';
    first = false;
    out << "{\"slot\":" << written.slot << ",\"store\":" << written.store
        << ",\"address\":" << written.address << ",\"width\":" << written.width << '}';
  }

  out << ']';
}

const char* RegionStopName(nyx::analysis::RegionStop stop) {
  using nyx::analysis::RegionStop;
  switch (stop) {
    case RegionStop::unresolved:
      return "unresolved";
    case RegionStop::opaque:
      return "opaque";
    case RegionStop::call:
      return "call";
    case RegionStop::return_:
      return "return";
    case RegionStop::cycle:
      return "cycle";
    case RegionStop::second_fork:
      return "second_fork";
    case RegionStop::block_limit:
      return "block_limit";
    case RegionStop::source_limit:
      return "source_limit";
    case RegionStop::dispatch:
      return "dispatch";
  }

  return "invalid";
}

const char* MatchName(nyx::analysis::ExpectedMatch match) {
  using nyx::analysis::ExpectedMatch;
  switch (match) {
    case ExpectedMatch::not_applicable:
      return "not_applicable";
    case ExpectedMatch::always:
      return "always";
    case ExpectedMatch::never:
      return "never";
    case ExpectedMatch::unknown:
      return "unknown";
  }

  return "invalid";
}

const char* PathRoleName(nyx::analysis::PathEdgeRole role) {
  using nyx::analysis::PathEdgeRole;
  switch (role) {
    case PathEdgeRole::fallthrough:
      return "fallthrough";
    case PathEdgeRole::branch:
      return "branch";
    case PathEdgeRole::callee:
      return "callee";
    case PathEdgeRole::return_:
      return "return";
    case PathEdgeRole::potential_return:
      return "potential_return";
  }

  return "invalid";
}

template <typename T>
void OptionalNumber(std::ostream& out, std::optional<T> value) {
  if (value)
    out << *value;
  else
    out << "null";
}

void OptionalBool(std::ostream& out, std::optional<bool> value) {
  if (value)
    out << (*value ? "true" : "false");
  else
    out << "null";
}

void PrintPathControl(std::ostream& out, const nyx::analysis::PathControlFacts& facts) {
  out << "{\"path_revision\":" << facts.path_revision
      << ",\"total_boundaries\":" << facts.total_boundaries
      << ",\"proof_scope\":\"successful_itinerary_prefix\",\"proved_divergence\":";
  OptionalNumber(out, facts.proved_divergence);
  out << ",\"boundaries\":[";
  bool first = true;
  for (const auto& boundary : facts.boundaries) {
    if (!first) out << ',';
    first = false;
    out << "{\"boundary\":" << boundary.boundary
        << ",\"source_address\":" << boundary.source_address << ",\"transfer_kind\":";
    if (boundary.transfer_kind)
      out << '"' << TransferName(*boundary.transfer_kind) << '"';
    else
      out << "null";
    out << ",\"transfer_target\":";
    OptionalNumber(out, boundary.transfer_target);
    out << ",\"expected_image_successor\":";
    OptionalNumber(out, boundary.expected_image_successor);
    out << ",\"expected_match\":\"" << MatchName(boundary.expected_match)
        << "\",\"callee_return_unknown\":" << (boundary.callee_return_unknown ? "true" : "false")
        << ",\"load_dependencies\":[";
    for (std::size_t i = 0; i < boundary.load_dependencies.size(); ++i) {
      if (i) out << ',';
      out << boundary.load_dependencies[i];
    }

    out << "],\"dispatch\":";
    if (boundary.dispatch) {
      out << "{\"index\":" << boundary.dispatch->index << ",\"bound\":" << boundary.dispatch->bound
          << ",\"guard_boundary\":" << boundary.dispatch->guard_boundary
          << ",\"complete\":true,\"constant_image_dependency\":"
          << (boundary.dispatch->constant_image_dependency ? "true" : "false")
          << ",\"destinations\":[";
      for (std::size_t i = 0; i < boundary.dispatch->destinations.size(); ++i) {
        if (i) out << ',';
        out << boundary.dispatch->destinations[i];
      }

      out << "]}";
    } else
      out << "null";
    out << ",\"edges\":[";
    for (unsigned i = 0; i < boundary.edge_count; ++i) {
      if (i) out << ',';
      const auto& edge = boundary.edges[i];
      const char* kind =
          edge.target_kind == nyx::analysis::TargetKind::image_location     ? "image_location"
          : edge.target_kind == nyx::analysis::TargetKind::absolute_runtime ? "absolute_runtime"
                                                                            : "unknown";
      out << "{\"role\":\"" << PathRoleName(edge.role) << "\",\"target_kind\":\"" << kind
          << "\",\"target_address\":" << edge.target_address << ",\"target_value\":";
      OptionalNumber(out, edge.target_value);
      out << ",\"condition\":";
      OptionalNumber(out, edge.condition);
      out << ",\"when\":";
      OptionalBool(out, edge.when);
      out << ",\"known_condition\":";
      OptionalBool(out, edge.known_condition);
      out << ",\"constant_image_dependency\":"
          << (edge.constant_image_dependency ? "true" : "false") << '}';
    }

    out << "]}";
  }

  out << "]}";
}

struct RegionReports {
  std::vector<std::string> json;
  std::vector<std::optional<nyx::ir::RecoveredPath>> paths;
  std::uint64_t simplified = 0;
  std::uint64_t unchanged = 0;
  std::uint64_t declined = 0;
  std::uint64_t occurrences = 0;
  std::uint64_t memory_edits = 0;
  std::uint64_t removed_memory_effects = 0;
  std::uint64_t arithmetic_edits = 0;
  std::uint64_t image_edits = 0;

  // Declared ranges some path wrote itself. Not a count of edits: a nonzero
  // value means a declaration the run was given is contradicted by the code.
  std::uint64_t contradicted_ranges = 0;
  std::uint64_t contradicted_pointers = 0;
  std::uint64_t proved_divergences = 0;

  // Separate counts: one rule removes a conditional a literal predicate decided,
  // the other restores the conditional a table dispatch hid.
  std::uint64_t control_edits = 0;
  std::uint64_t recovered_branches = 0;

  // One entry per transfer at which some candidate proved a complete successor
  // set, summarising how far the examined candidates agree. `without` counts
  // candidates that reached the transfer without passing a guard that bounds
  // it, and so proved nothing about where it can go.
  struct Dispatch {
    std::uint64_t bound = 0;
    std::uint64_t reached = 0;
    std::uint64_t with_set = 0;
    std::uint64_t without = 0;
    bool agreeing = true;
    std::vector<std::uint64_t> destinations;
  };

  std::map<std::uint64_t, Dispatch> dispatches;
  std::vector<std::optional<nyx::analysis::PathControlFacts>> control;

  // Per candidate, whether some forwarding edit of its path used an entry relation.
  std::vector<std::uint8_t> entry_relation_dependent;
  std::uint64_t entry_relation_paths = 0;
};

std::optional<RegionReports> RecoverRegions(const nyx::analysis::Regions& regions,
                                            nyx::Budget& budget, nyx::analysis::ImageFacts facts,
                                            nyx::a64::MemoryProfile profile,
                                            const nyx::analysis::EntryRelations& relations,
                                            const char*& decline) {
  decline = "resource_limit";
  RegionReports reports;
  if (budget.try_consume({regions.candidates().size(),
                          regions.candidates().size() *
                              (sizeof(std::string) + sizeof(std::optional<nyx::ir::RecoveredPath>) +
                               1)}) != nyx::BudgetDecline::none)
    return {};
  reports.json.reserve(regions.candidates().size());
  reports.paths.resize(regions.candidates().size());
  reports.control.resize(regions.candidates().size());
  reports.entry_relation_dependent.resize(regions.candidates().size());
  for (std::size_t index = 0; index < regions.candidates().size(); ++index) {
    const auto& region = regions.candidates()[index];
    const auto entry = regions.graph()
                           .sources()[regions.graph().blocks()[region.entry_block].first_source]
                           .address;
    const auto& entry_relations = relations.blocks[region.entry_block];
    bool related = false;
    reports.occurrences += region.source_ids.size();
    auto normalized = nyx::analysis::NormalizeRegion(regions, index, budget);
    nyx::recovery::PathMemoryRecoveryResult forwarded;
    nyx::recovery::MbaPathResult simplified;
    nyx::recovery::ImagePathResult folded;
    nyx::analysis::PathControlResult control;
    nyx::recovery::ControlRecoveryResult recovered;
    std::string before, after;
    const char* outcome = "declined";
    const char* reason = "source_limit";
    if (normalized.path) {
      forwarded = nyx::recovery::ForwardMemoryValues(*normalized.path, budget, {}, entry_relations);
      if (!forwarded.path) {
        if (forwarded.reason != nyx::recovery::MemoryRecoveryDecline::resource_limit)
          decline = "invalid_group";
        return {};
      }

      related = std::any_of(forwarded.journal.begin(), forwarded.journal.end(),
                            [&](const nyx::recovery::MemoryEdit& edit) {
                              return forwarded.facts[edit.fact].entry_relation;
                            });
      reports.entry_relation_dependent[index] = related;
      reports.entry_relation_paths += related;
      simplified = nyx::recovery::SimplifyMba(*forwarded.path, budget);
      if (!simplified.path) {
        if (simplified.reason != nyx::recovery::MbaDecline::resource_limit)
          decline = "invalid_group";
        return {};
      }

      folded = nyx::recovery::FoldImageValues(*simplified.path, facts, budget);
      if (!folded.path) {
        if (folded.reason != nyx::recovery::ImageDecline::resource_limit) decline = "invalid_group";
        return {};
      }

      const auto folded_revision = folded.path->revision();
      if (budget.try_consume(
              {folded.journal.size(), folded.journal.size() * sizeof(nyx::ir::ValueId)}) !=
          nyx::BudgetDecline::none)
        return {};
      std::vector<nyx::ir::ValueId> image_dependent_nodes;
      image_dependent_nodes.reserve(folded.journal.size());
      for (const auto& edit : folded.journal) {
        if (edit.constant_bytes || edit.relocated_slot) image_dependent_nodes.push_back(edit.node);
      }

      // A range this path refutes is gone for every stage below, not just the
      // fold that found it.
      const nyx::analysis::ImageFacts held{folded.constants, folded.RetainedPointers(facts),
                                           facts.page_aligned_placement};
      recovered = nyx::recovery::RecoverControl(std::move(*folded.path), held, budget);
      if (!recovered.path) {
        if (recovered.reason != nyx::recovery::ControlRecoveryDecline::resource_limit)
          decline = "invalid_group";
        return {};
      }

      if (profile != nyx::a64::MemoryProfile::none) {
        auto loads = nyx::recovery::OmitForwardedPairLoads(
            std::move(*recovered.path), forwarded.facts, nyx::a64::kSp, budget, held);
        if (!loads.path) {
          if (loads.reason != nyx::recovery::PairedLoadDecline::resource_limit)
            decline = "invalid_group";
          return {};
        }

        auto stores = nyx::recovery::OmitOverwrittenStores(std::move(*loads.path), nyx::a64::kSp,
                                                           budget, held);
        if (!stores.path) {
          if (stores.reason != nyx::recovery::StoreCleanupDecline::resource_limit)
            decline = "invalid_group";
          return {};
        }

        recovered.path = std::move(*stores.path);
      }

      const auto removed =
          recovered.path->omissions().size() + 2 * recovered.path->paired_load_omissions().size();
      reports.removed_memory_effects += removed;
      for (const auto& rewrite : recovered.path->rewrites()) {
        if (rewrite.rule == nyx::ir::RewriteRule::dispatch_branch)
          ++reports.recovered_branches;
        else
          ++reports.control_edits;
      }

      control = nyx::analysis::AnalyzePathControl(
          *recovered.path, {folded_revision, image_dependent_nodes}, budget, {}, held);
      if (!control.facts) {
        if (control.reason != nyx::analysis::ControlDecline::resource_limit)
          decline = "invalid_group";
        return {};
      }

      if (control.facts->proved_divergence) ++reports.proved_divergences;
      for (const auto& boundary : control.facts->boundaries) {
        const bool indirect = boundary.edge_count != 0 &&
                              boundary.edges[0].target_kind == nyx::analysis::TargetKind::unknown;
        if (!boundary.dispatch && !indirect) continue;
        auto& entry = reports.dispatches[boundary.source_address];
        ++entry.reached;
        if (!boundary.dispatch) {
          ++entry.without;
          continue;
        }

        ++entry.with_set;
        if (entry.destinations.empty()) {
          entry.bound = boundary.dispatch->bound;
          entry.destinations = boundary.dispatch->destinations;
        } else if (entry.destinations != boundary.dispatch->destinations) {
          entry.agreeing = false;
        }
      }

      auto printed_after = nyx::ir::PrintJson(*recovered.path, budget, held);
      if (!printed_after.json) {
        if (printed_after.reason == nyx::ir::PrintDecline::invalid_group) decline = "invalid_group";
        return {};
      }

      after = std::move(*printed_after.json);
      if (!forwarded.journal.empty() || !simplified.journal.empty() || !folded.journal.empty() ||
          !recovered.path->rewrites().empty() || removed) {
        auto printed_before = nyx::ir::PrintJson(*normalized.path, budget);
        if (!printed_before.json) {
          if (printed_before.reason == nyx::ir::PrintDecline::invalid_group)
            decline = "invalid_group";
          return {};
        }

        before = std::move(*printed_before.json);
        outcome = "simplified";
        ++reports.simplified;
      } else {
        outcome = "unchanged";
        ++reports.unchanged;
      }

      reason = "none";
      reports.memory_edits += forwarded.journal.size();
      reports.arithmetic_edits += simplified.journal.size();
      reports.image_edits += folded.journal.size();
      reports.contradicted_ranges += folded.contradicted.size();
      reports.contradicted_pointers += folded.contradicted_pointers.size();
    } else {
      // Only an oversized entry block is a local refusal. Budget exhaustion or
      // invalid selections cannot masquerade as an exhaustive recovery report.
      if (!region.source_ids.empty() || region.stop != nyx::analysis::RegionStop::source_limit) {
        if (normalized.reason != nyx::ir::BlockDecline::resource_limit) decline = "invalid_group";
        return {};
      }

      ++reports.declined;
    }

    std::uint64_t control_extent = 0;
    if (control.facts) reports.control[index] = *control.facts;
    if (control.facts) {
      control_extent = 1024;
      for (const auto& boundary : control.facts->boundaries)
        control_extent += 512 + boundary.edge_count * 384 + boundary.load_dependencies.size() * 16;
    }

    const auto envelope = 4096 + region.source_ids.size() * 32 + region.block_ids.size() * 64 +
                          forwarded.facts.size() * 1024 + forwarded.journal.size() * 512 +
                          (simplified.journal.size() + folded.journal.size()) * 768 +
                          (folded.contradicted.size() + folded.contradicted_pointers.size()) * 192 +
                          (recovered.path ? recovered.path->omissions().size() +
                                                recovered.path->paired_load_omissions().size()
                                          : 0) *
                              512 +
                          control_extent + entry_relations.size() * 96 + 64;
    const auto extent = envelope + before.size() + after.size();

    // Cover stream growth and final retained storage before writing any artifact.
    if (budget.try_consume({extent, extent * 4}) != nyx::BudgetDecline::none) return {};
    std::ostringstream out;
    out << "{\"id\":" << index << ",\"entry_block\":" << region.entry_block
        << ",\"entry\":" << entry << ",\"outcome\":\"" << outcome << "\",\"reason\":\"" << reason
        << "\",\"stop\":\"" << RegionStopName(region.stop) << "\",\"stopped_edge\":";
    if (region.stopped_edge)
      out << *region.stopped_edge;
    else
      out << "null";
    out << ",\"block_ids\":[";
    for (std::size_t i = 0; i < region.block_ids.size(); ++i) {
      if (i) out << ',';
      out << region.block_ids[i];
    }

    out << "],\"source_ids\":[";
    for (std::size_t i = 0; i < region.source_ids.size(); ++i) {
      if (i) out << ',';
      out << region.source_ids[i];
    }

    out << "],\"transition_edges\":[";
    for (std::size_t i = 0; i < region.transition_edges.size(); ++i) {
      if (i) out << ',';
      out << region.transition_edges[i];
    }

    std::size_t folded_conditions = 0, dispatch_branches = 0;
    if (recovered.path) {
      for (const auto& rewrite : recovered.path->rewrites()) {
        if (rewrite.rule == nyx::ir::RewriteRule::dispatch_branch)
          ++dispatch_branches;
        else
          ++folded_conditions;
      }
    }

    const auto omitted_stores = recovered.path ? recovered.path->omissions().size() : 0;
    const auto omitted_pairs = recovered.path ? recovered.path->paired_load_omissions().size() : 0;
    out << "],\"applied_edits\":"
        << forwarded.journal.size() + simplified.journal.size() + folded.journal.size() +
               folded_conditions + dispatch_branches + omitted_stores + omitted_pairs
        << ",\"removed_conditional_transfers\":" << folded_conditions
        << ",\"recovered_dispatch_branches\":" << dispatch_branches
        << ",\"removed_memory_effects\":" << omitted_stores + 2 * omitted_pairs;
    // The relations proved at the entry block, whether or not any edit rests on them:
    // they describe every arrival the graph admits, which is what a checker needs.
    out << ",\"entry_relation_dependent\":" << (related ? "true" : "false");
    if (!entry_relations.empty()) {
      out << ",\"entry_relations\":[";
      for (std::size_t i = 0; i < entry_relations.size(); ++i) {
        const auto& relation = entry_relations[i];
        if (i) out << ',';
        out << "{\"storage\":" << relation.storage << ",\"root\":" << relation.root
            << ",\"offset\":" << relation.offset << '}';
      }

      out << ']';
    }

    if (!before.empty() || !folded.contradicted.empty() || !folded.contradicted_pointers.empty()) {
      out << ",\"forwarding_facts\":[";
      PrintRecoveryJournal(out, forwarded, simplified, folded);
      if (!before.empty()) out << ",\"before\":" << before;
    }

    if (!after.empty()) out << ",\"after\":" << after;
    if (control.facts) {
      out << ",\"control\":";
      PrintPathControl(out, *control.facts);
    }

    out << '}';
    reports.json.push_back(std::move(out).str());
    if (reports.json.back().size() > extent) {
      decline = "invalid_report";
      return {};
    }

    if (region.stop == nyx::analysis::RegionStop::dispatch && recovered.path) {
      reports.paths[index] = std::move(*recovered.path);
    }
  }

  return reports;
}

// The stitched graph's own share of the artifact, charged before it is written.
std::uint64_t RecoveredGraphExtent(const nyx::analysis::Unflattening& result) {
  std::uint64_t edges = 0;
  for (const auto& block : result.graph.blocks) edges += block.edges.size();
  return 512 + result.graph.entries.size() * 24 + result.graph.blocks.size() * 96 + edges * 256;
}

const char* EdgeKindName(nyx::analysis::CfgEdgeKind kind) {
  switch (kind) {
    case nyx::analysis::CfgEdgeKind::fallthrough:
      return "fallthrough";
    case nyx::analysis::CfgEdgeKind::branch:
      return "branch";
    case nyx::analysis::CfgEdgeKind::callee:
      return "callee";
    case nyx::analysis::CfgEdgeKind::return_:
      return "return";
    case nyx::analysis::CfgEdgeKind::potential_return:
      return "potential_return";
    case nyx::analysis::CfgEdgeKind::opaque_unknown:
      return "opaque_unknown";
    case nyx::analysis::CfgEdgeKind::trap:
      return "trap";
  }

  return "opaque_unknown";
}

// The stitched function: one graph, in addresses, with every edge saying what it
// rests on beyond the original graph. A block that is not listed is one this
// graph does not reach; whether that means it is gone is `retirement` above.
void PrintRecoveredGraph(std::ostream& out, const nyx::analysis::Regions& regions,
                         const nyx::analysis::Unflattening& result) {
  const auto graph = regions.graph();
  out << "\"recovered_graph\":{\"scope\":\"the blocks this graph reaches from the supplied"
         " known entries once every transition's source is joined straight to its destinations;"
         " closure is only within the supplied population\",\"entries\":[";
  for (std::size_t i = 0; i < result.graph.entries.size(); ++i) {
    const auto& block = graph.blocks()[result.graph.entries[i]];
    if (i) out << ',';
    out << graph.sources()[block.first_source].address;
  }

  out << "],\"blocks\":[";
  for (std::size_t i = 0; i < result.graph.blocks.size(); ++i) {
    const auto& block = result.graph.blocks[i];
    if (i) out << ',';
    out << "{\"address\":" << block.address << ",\"from_transition\":";
    if (block.transition)
      out << result.transitions[*block.transition].candidate;
    else
      out << "null";
    out << ",\"edges\":[";
    for (std::size_t j = 0; j < block.edges.size(); ++j) {
      const auto& edge = block.edges[j];
      if (j) out << ',';
      out << "{\"kind\":\"" << EdgeKindName(edge.kind) << "\",\"target\":";
      if (edge.target_block)
        out << graph.sources()[graph.blocks()[*edge.target_block].first_source].address;
      else
        out << "null";
      out << ",\"condition\":";
      if (edge.condition)
        out << *edge.condition;
      else
        out << "null";
      out << ",\"condition_owner\":";
      if (edge.condition) {
        out << '"'
            << (edge.condition_owner == nyx::analysis::ConditionOwner::recovered_path
                    ? "recovered_path"
                    : "original_block")
            << '"';
      } else
        out << "null";
      out << ",\"when\":";
      if (edge.when)
        out << (*edge.when ? "true" : "false");
      else
        out << "null";
      out << ",\"assumes\":[";
      bool first = true;
      const auto assume = [&](const char* text) {
        if (!first) out << ',';
        first = false;
        out << '"' << text << '"';
      };

      if (edge.assumptions.constant_image) assume("declared image values keep their stated values");
      if (edge.assumptions.callee_returns_to_continuation) {
        assume("an unknown callee returns only through its call's continuation");
      }

      if (edge.assumptions.return_leaves) assume("this return leaves for the caller");
      if (edge.assumptions.unresolved_target) assume("this target was never resolved to a block");
      if (edge.assumptions.declared_opaque_control) {
        assume("caller-declared opaque control behavior matches the source instruction");
      }

      if (edge.assumptions.entry_relations) {
        assume("storage relations proved from the known entries hold at the transition's entry");
      }

      if (edge.assumptions.declared_abi)
        assume("calls and returns keep the declared ABI's discipline");
      if (edge.assumptions.declared_return_leaves)
        assume("caller declares untargeted returns leave the selected population");
      if (edge.assumptions.declared_noreturn)
        assume("caller declares every possible callee of this call never returns");
      out << "]}";
    }

    out << "]}";
  }

  out << "]}";
}

// Addresses rather than block ids, so a reader can check a transition against
// the binary without the graph's numbering.
void PrintUnflattening(std::ostream& out, const nyx::analysis::Regions& regions,
                       const nyx::analysis::Unflattening& result,
                       const nyx::analysis::EntryRelations& relations) {
  const auto graph = regions.graph();
  const auto address = [&](std::uint32_t block) {
    return graph.sources()[graph.blocks()[block].first_source].address;
  };

  out << "\"unflattening\":{\"scope\":\"each transition replaces one entry block's route through a"
         " dispatcher by the destinations its recovered path proves; that path still executes every"
         " original instruction of the route, so a retired block no longer runs as a block of its "
         "own"
         " but its instructions still run inside transitions\",\"transitions\":[";
  for (std::size_t i = 0; i < result.transitions.size(); ++i) {
    const auto& transition = result.transitions[i];
    if (i) out << ',';
    out << "{\"candidate\":" << transition.candidate
        << ",\"entry\":" << address(transition.entry_block) << ",\"dispatch\":"
        << graph
               .sources()[graph.blocks()[transition.dispatch_block].first_source +
                          graph.blocks()[transition.dispatch_block].source_count - 1]
               .address
        << ",\"destinations\":[";
    for (std::size_t j = 0; j < transition.destinations.size(); ++j) {
      if (j) out << ',';
      out << address(transition.destinations[j]);
    }

    out << "],\"condition\":";
    if (transition.condition)
      out << *transition.condition;
    else
      out << "null";
    out << ",\"constant_image_dependency\":"
        << (transition.constant_image_dependency ? "true" : "false")
        << ",\"entry_relation_dependency\":"
        << (transition.entry_relation_dependency ? "true" : "false") << '}';
  }

  std::array<std::uint64_t, 3> refused{};
  for (const auto& refusal : result.refused) ++refused[static_cast<unsigned>(refusal.reason)];
  out << "],\"refused\":{\"unproved_route\":" << refused[0]
      << ",\"unresolved_target\":" << refused[1] << ",\"outside_dispatch_set\":" << refused[2]
      << "},\"retired_blocks\":[";
  for (std::size_t i = 0; i < result.retired_blocks.size(); ++i) {
    if (i) out << ',';
    out << address(result.retired_blocks[i]);
  }

  out << "],\"retirement\":\""
      << (result.unresolved_successor_blocks.empty() ? "established" : "not_established")
      << "\",\"unresolved_successors_from\":[";
  for (std::size_t i = 0; i < result.unresolved_successor_blocks.size(); ++i) {
    if (i) out << ',';
    out << address(result.unresolved_successor_blocks[i]);
  }

  out << "],\"reachable_unknown_callees\":" << result.reachable_unknown_callees
      << ",\"reachable_returns\":" << result.reachable_returns << ",\"retirement_assumes\":[";
  const char* separator = "";
  const auto assume = [&](bool holds, const char* text) {
    if (!holds || result.retired_blocks.empty()) return;
    out << separator << '"' << text << '"';
    separator = ",";
  };

  assume(result.reachable_unknown_callees != 0,
         "an unknown callee returns only through its call's continuation");
  assume(result.reachable_returns != 0,
         "a return the graph does not resolve leaves for the caller");
  assume(std::any_of(graph.sources().begin(), graph.sources().end(),
                     [](const auto& source) {
                       return source.opaque_control != nyx::analysis::OpaqueControl::unknown;
                     }),
         "caller-declared opaque control behavior matches the source instruction");
  const bool related = std::any_of(result.transitions.begin(), result.transitions.end(),
                                   [](const nyx::analysis::Transition& transition) {
                                     return transition.entry_relation_dependency;
                                   });
  assume(
      related,
      "storage relations proved from the known entries hold at each dependent transition's entry");
  assume(related && relations.calling_convention,
         "calls and returns keep the declared ABI's discipline");
  assume(related && relations.return_leaves,
         "caller declares untargeted returns leave the selected population");
  // What every published relation rests on, wherever in the graph the proof met it.
  out << "],\"entry_relations_rest_on\":[";
  separator = "";
  for (const auto& [holds, name] :
       {std::pair{relations.calling_convention, "declared_abi"},
        std::pair{relations.return_leaves, "declared_return_leaves"},
        std::pair{relations.constant_image, "constant_image"},
        std::pair{relations.declared_opaque_control, "declared_opaque_control"}}) {
    if (!holds) continue;
    out << separator << '"' << name << '"';
    separator = ",";
  }

  out << "],";
  PrintRecoveredGraph(out, regions, result);
  out << "}";
}

// Ranges the caller declares keep their file-initialized value for the run.
// Writable memory carries no such guarantee, so this is never inferred from the
// image: a caller states it, the artifact prints exactly what was stated, and
// every destination resolved through one depends on the caller being right.
struct DeclaredRange {
  std::uint64_t address, size;
};

// Declared bytes a recovered graph from an earlier pass of this run was seen
// writing: a store it places, in the block and at the node that make it.
// Nothing later reads them as constant.
struct SsaRefuted {
  std::uint64_t address, bytes, written_at, block;
  std::uint32_t node;
  unsigned width;
};

// Byte ranges [address, address + size) a whole-image scan saw stores reach.
using ReachedBytes = std::vector<std::pair<std::uint64_t, std::uint64_t>>;
constexpr std::size_t kMaxDeclaredRanges = 16;
constexpr std::size_t kMaxDeclaredNoreturn = 64;

// Everything the caller declares for a run, each printed with the artifact.
// Calls to `noreturn` targets are taken never to come back; like the ranges,
// that is the caller's claim, and an opaque call is otherwise never assumed so.
// With `abi`, code keeps the AAPCS64 call and return discipline: every callee
// returns only through its call's continuation with the callee-saved registers
// as it found them. Whether an untargeted return leaves this selected
// population is a separate restriction.
struct Declarations {
  std::vector<DeclaredRange> ranges;
  std::vector<std::uint64_t> noreturn;  // sorted, unique image locations
  bool abi = false;

  // Entry-SP-relative half-open range, for the SSA frame proposals only.
  std::optional<std::pair<std::int64_t, std::int64_t>> frame;
  bool frame_unreached_by_callees = false;
  bool image_access = false;
  bool closed_entries = false;
  bool return_leaves = false;

  // A trap never resumes inside the population except at a listed entry.
  bool trap_stops = false;

  // A fault or trap ends the run and nothing observes its registers.
  bool faults_terminal = false;

  // Every relocated slot keeps the loader's value, including those the loader
  // leaves writable.
  bool relocations_stable = false;

  // Every symbol the image defines is bound to its own definition, so a slot
  // the loader fills from one holds that definition's location.
  bool symbols_own = false;

  // Values observed by executing an initializer, offered as the contents of
  // the slots it wrote.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> derived;

  // What a call to a named target leaves in a register, observed by running it.
  std::vector<nyx::ir::SsaCallResult> call_results;

  // Slots a whole-image scan found written. Supplying the scan admits the
  // slots it did not find a writer for. That is bounded by what the scan
  // could see; it does not show that nothing writes them.
  std::optional<std::vector<std::uint64_t>> written_slots;

  // What that scan saw any placed store reach, when its record says.
  std::optional<ReachedBytes> written_data;

  // Every data word the image's code was not found writing keeps its file
  // value, the stores the scan could not place included.
  bool data_unwritten = false;

  // How many of `ranges` were admitted under that declaration, not named.
  std::size_t derived_ranges = 0;

  // Instructions and stores that scan could not account for. Admitting the
  // slots it omits rests on there being no writer among them.
  std::uint64_t written_slots_unexamined = 0;
  std::string record_digest;

  // Where the run says the image sits. Derived values are observations from an
  // execution at one placement, so admitting them needs this said out loud.
  std::optional<std::uint64_t> load_bias;

  // Slots a run saw written without the runs agreeing on what to. Seeing one
  // written is decisive on its own: the loader's value no longer stands.
  std::vector<std::uint64_t> derived_unstable;

  // Sorted by address. What earlier passes' graphs were seen writing, and
  // whether the passes stopped before a pass found nothing more.
  std::vector<SsaRefuted> ssa_refuted;

  // Relocated slots an earlier pass's graph placed a store onto: their
  // loader value no longer stands, exactly as the writer scan withdrawing one.
  std::vector<std::uint64_t> ssa_refuted_slots;
  bool ssa_refutation_unsettled = false;

  // Each routine run to obtain those values and what the run did. A derived
  // value is worth no more than the run behind it, so the artifact names it
  // whether the values came from a record or from this process.
  struct DerivationRun {
    std::uint64_t entry;
    std::string status;

    // Counts the run offered before the guesses filtered them. They separate a
    // run whose results were all dropped from one that produced nothing.
    std::uint64_t writes = 0;
    std::uint64_t results = 0;
    std::uint64_t results_dropped = 0;
  };

  std::vector<DerivationRun> derivation_runs;
};

// The values a run offered that the fact set carries, so a reader can check
// each against the image instead of taking the count on faith. A refuted slot,
// or one naming nothing, is not among them and is not printed.
std::string OfferedValues(const nyx::analysis::ImageFacts& facts,
                          const std::vector<std::pair<std::uint64_t, std::uint64_t>>& derived) {
  std::string text = "[";
  for (const auto& [address, value] : derived) {
    const auto at = std::lower_bound(facts.pointers.begin(), facts.pointers.end(), address,
                                     [](const nyx::analysis::RelocatedPointer& slot,
                                        std::uint64_t key) { return slot.address < key; });
    const bool carried = at != facts.pointers.end() && at->address == address && at->value_stable &&
                         at->target == value;
    text += (text.size() > 1 ? ",{\"address\":" : "{\"address\":") + std::to_string(address) +
            ",\"value\":" + std::to_string(value) + ",\"carried\":" + (carried ? "true" : "false") +
            "}";
  }

  return text + "]";
}

// A derived value is an observation, so the artifact says which run produced
// it and how that run ended: a run that stopped early saw fewer writes than
// the routine makes, and a reader can only judge that if it is printed.
std::string DerivationRuns(const std::vector<Declarations::DerivationRun>& runs) {
  std::string text = "[";
  for (std::size_t i = 0; i < runs.size(); ++i) {
    text += (i ? ",{\"entry\":" : "{\"entry\":") + std::to_string(runs[i].entry) +
            ",\"status\":" + JsonString(runs[i].status) +
            ",\"writes\":" + std::to_string(runs[i].writes) +
            ",\"results\":" + std::to_string(runs[i].results) +
            ",\"results_dropped\":" + std::to_string(runs[i].results_dropped) + "}";
  }

  return text + "]";
}

// What recovering costs when the caller says nothing. The fixed default this
// replaces stops at about four kilobytes, so anything larger carried an
// override, and measured cost runs from 110000 to 460000 per byte of region
// depending on the function with `--derive-entry auto` multiplying it, which
// is too wide a spread to derive a ceiling from the size. Deriving one from
// the size is also the wrong shape: a budget is what an input cannot spend
// its way past, and scaling it means a larger input grants itself a larger
// guarantee. A fixed ceiling costs a small region nothing, because a budget
// is spent as used rather than reserved.
nyx::Resources RecoveryResources(std::optional<std::uint64_t> work,
                                 std::optional<std::uint64_t> bytes) {
  return {work.value_or(20000000000ULL), bytes.value_or(12ULL << 30)};
}

// BEGIN:END in signed decimal, half-open and nonempty, at most 1 MiB wide.
std::optional<std::pair<std::int64_t, std::int64_t>> ParseFrame(std::string_view specification) {
  const auto colon = specification.find(':');
  if (colon == std::string_view::npos) return std::nullopt;
  std::int64_t begin = 0, end = 0;
  const auto parse = [](std::string_view text, std::int64_t& value) {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value, 10);
    return !text.empty() && result.ec == std::errc() && result.ptr == text.data() + text.size();
  };

  if (!parse(specification.substr(0, colon), begin) ||
      !parse(specification.substr(colon + 1), end) || begin >= end || end - begin > (1 << 20) ||
      begin < -(INT64_C(1) << 40) || end > (INT64_C(1) << 40))
    return std::nullopt;
  return std::pair{begin, end};
}

// HEX,... call targets, sorted and deduplicated.
std::optional<std::vector<std::uint64_t>> ParseNoreturn(std::string_view specification) {
  std::vector<std::uint64_t> targets;
  while (!specification.empty()) {
    const auto comma = specification.find(',');
    std::uint64_t target = 0;
    if (targets.size() == kMaxDeclaredNoreturn ||
        !Unsigned(specification.substr(0, comma), target, 16)) {
      return std::nullopt;
    }

    targets.push_back(target);
    if (comma == std::string_view::npos) {
      std::sort(targets.begin(), targets.end());
      targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
      return targets;
    }

    specification.remove_prefix(comma + 1);
    if (specification.empty()) return std::nullopt;
  }

  return std::nullopt;
}

// HEX:BYTES,... with each range at most a page, no wrap and no empty entry.
std::optional<std::vector<DeclaredRange>> ParseDeclaredRanges(std::string_view specification) {
  std::vector<DeclaredRange> ranges;
  while (!specification.empty()) {
    const auto comma = specification.find(',');
    const auto part = specification.substr(0, comma);
    const auto colon = part.find(':');
    DeclaredRange range{};
    if (ranges.size() == kMaxDeclaredRanges || colon == std::string_view::npos ||
        !Unsigned(part.substr(0, colon), range.address, 16) ||
        !Unsigned(part.substr(colon + 1), range.size, 10) || !range.size || range.size > 4096 ||
        range.size - 1 > UINT64_MAX - range.address)
      return std::nullopt;
    ranges.push_back(range);
    if (comma == std::string_view::npos) return ranges;
    specification.remove_prefix(comma + 1);
    if (specification.empty()) return std::nullopt;
  }

  return std::nullopt;
}

enum class GraphMode { none, plain_cfg, probe, recover };

// How a recover-regions run treats its SSA candidate. None of these is a
// declaration about the program: the pipeline selects registered passes, the
// state and stub files drive the candidate's evaluator, and the check file is
// the independent verifier's record for one exact candidate.
struct SsaRequest {
  // How many times to run the pipeline. One pass sees only what earlier
  // passes established, so a value a later pass makes knowable needs another
  // sweep to be used.
  unsigned rounds = 1;
  const char* state_path = nullptr;
  const char* stubs_path = nullptr;
  const char* check_path = nullptr;
  std::vector<std::size_t> passes;  // registry indices; empty is the default pipeline
  // Publish a reading view of the resulting graph; it is never an input.
  bool listing = false;

  // SHA-256 of the whole input file, so a record binds to the image it examined.
  std::string image_digest;
};

constexpr std::size_t kMaxCheckRecord = 4096;

// The whole file if it is at most `limit` bytes; null otherwise.
std::optional<std::string> ReadBounded(const char* path, std::size_t limit, nyx::Budget& budget) {
  std::FILE* file = std::fopen(path, "rb");
  if (!file) return std::nullopt;
  if (budget.try_consume({limit + 1, limit + 1}) != nyx::BudgetDecline::none) {
    std::fclose(file);
    return std::nullopt;
  }

  std::string data(limit + 1, '\0');
  const auto read = std::fread(data.data(), 1, data.size(), file);
  const bool failed = std::ferror(file);
  std::fclose(file);
  if (failed || read > limit) return std::nullopt;
  data.resize(read);
  return data;
}

// Which callees a pass settled on without anything having run them. A call
// result is what lets the load after the call resolve, so knowing these is
// what a second pass needs; finding them needs the graph the first pass built,
// which is why deriving cannot happen before recovery starts.
struct Discovery {
  // Callees the graph settled on that nothing has run.
  std::vector<std::uint64_t> callees;

  // Image locations the graph forms an address for. A slot among these that
  // the writer scan refuted is one this range reads and the image rewrites,
  // and the routine that rewrites it is the one worth running.
  std::vector<std::uint64_t> addressed;

  // Declared bytes still in the pass's fact set that a store its graph
  // places writes.
  std::vector<SsaRefuted> refuted;

  // Relocated slots a store its graph places writes.
  std::vector<std::uint64_t> refuted_slots;
};

// Every image location a large function names, code addresses included, runs
// to thousands; a cap that low stops before the loads of its later blocks.
constexpr std::size_t kMaxAddressed = std::size_t{1} << 16;

void CollectUnrunCallees(const nyx::ir::SsaGraph& graph, Discovery& into, nyx::Budget& budget) {
  const auto results = graph.call_results();
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    const auto handle = graph.Handle(slot);
    const auto* block = handle ? graph.Get(*handle) : nullptr;
    if (!block) continue;
    if (budget.try_consume({1 + block->nodes.size(), 0}) != nyx::BudgetDecline::none) return;
    if (into.addressed.size() < kMaxAddressed) {
      for (const auto& node : block->nodes) {
        // Under a declared bias a location folds to the number it is, so
        // both spellings name one. Which of these is a slot worth anything
        // is decided by the caller against the relocations, not here.
        std::optional<std::uint64_t> location;
        if (node.op == nyx::ir::Op::image_address)
          location = node.immediate;
        else if (node.op == nyx::ir::Op::constant && node.width == 64 && graph.load_bias() &&
                 node.immediate >= *graph.load_bias())
          location = node.immediate - *graph.load_bias();
        if (!location) continue;
        into.addressed.push_back(*location);
      }

      // A load names the location its address arithmetic forms, which before
      // simplification is a page plus an offset rather than one literal.
      for (nyx::ir::ValueId id = 0;
           id < block->nodes.size() && into.addressed.size() < kMaxAddressed; ++id) {
        if (block->nodes[id].op != nyx::ir::Op::load) continue;
        const auto location = nyx::ir::SsaLoadLocation(block->nodes, id, graph.load_bias());
        if (location) into.addressed.push_back(location->first);
      }
    }

    if (into.callees.size() >= 64) continue;
    const auto target = nyx::ir::SsaCallTarget(graph, *block, budget);
    if (!target) continue;
    const auto at = std::lower_bound(results.begin(), results.end(), *target,
                                     [](const nyx::ir::SsaCallResult& result, std::uint64_t key) {
                                       return result.target < key;
                                     });
    if (at != results.end() && at->target == *target) continue;
    if (std::find(into.callees.begin(), into.callees.end(), *target) == into.callees.end())
      into.callees.push_back(*target);
  }
}

int Graph(const char* path, std::uint64_t address, std::uint64_t size,
          std::span<const nyx::analysis::SourceRecord> sources, nyx::Budget& budget,
          nyx::a64::MemoryProfile profile, GraphMode mode, nyx::analysis::ImageFacts facts,
          const Declarations& declared, const SsaRequest& ssa_request = {},
          Discovery* discover = nullptr, const nyx::format::Image* image = nullptr) {
  const std::array<std::uint64_t, 1> entries{address};

  // Bytes an earlier pass's graph writes are withdrawn before anything reads
  // them, one span per declared range they fall in, each witnessed by the
  // store. What the CFG's own scan withdraws is then taken from the rest.
  struct Refutation {
    nyx::ir::RefutedConstantSpan span;
    const SsaRefuted* store;  // null for the CFG's own scan
  };

  std::vector<Refutation> ssa_spans;
  for (const auto& refuted : declared.ssa_refuted) {
    if (budget.try_consume({facts.constants.size() + 1, sizeof(Refutation)}) !=
        nyx::BudgetDecline::none)
      return Error("resource_limit", "Image fact withdrawal exceeds budget");
    for (std::size_t index = 0; index < facts.constants.size(); ++index) {
      const auto& range = facts.constants[index];
      const auto first = std::max(refuted.address, range.address);
      const auto last =
          std::min(refuted.address + refuted.bytes, range.address + range.bytes.size());
      if (first >= last) continue;
      ssa_spans.push_back(
          {{index, first, last - first, {refuted.written_at, refuted.width, refuted.node}},
           &refuted});
    }
  }

  const auto by_range = [](const Refutation& a, const Refutation& b) {
    return std::pair{a.span.index, a.span.address} < std::pair{b.span.index, b.span.address};
  };

  std::stable_sort(ssa_spans.begin(), ssa_spans.end(), by_range);
  std::vector<nyx::ir::RefutedConstantSpan> ssa_only;
  for (const auto& item : ssa_spans) ssa_only.push_back(item.span);
  auto before_cfg = nyx::ir::RetainImageFacts(facts, ssa_only, {}, budget);
  if (!before_cfg) return Error("resource_limit", "Image fact withdrawal exceeds budget");
  auto result = nyx::analysis::BuildCfg(sources, entries, budget, {}, before_cfg->view());
  if (!result.cfg) {
    const char* reason =
        result.reason == nyx::analysis::CfgDecline::resource_limit   ? "resource_limit"
        : result.reason == nyx::analysis::CfgDecline::invalid_entry  ? "invalid_entry"
        : result.reason == nyx::analysis::CfgDecline::invalid_source ? "invalid_source"
                                                                     : "invalid_group";
    return Error(reason, "CFG construction declined; no partial graph published");
  }

  // Every withdrawn span, named against the declared range it came from.
  std::vector<Refutation> refutations = ssa_spans;
  for (auto span : result.cfg->refuted_constant_spans()) {
    span.index = before_cfg->origins[span.index];
    refutations.push_back({span, nullptr});
  }

  std::stable_sort(refutations.begin(), refutations.end(), by_range);
  std::vector<nyx::ir::RefutedConstantSpan> refuted_spans;
  for (const auto& item : refutations) refuted_spans.push_back(item.span);
  auto retained = nyx::ir::RetainImageFacts(facts, refuted_spans,
                                            result.cfg->refuted_pointer_indices(), budget);
  if (!retained)
    return Error("resource_limit", "Image fact reduction declined; no partial graph published");
  if (mode == GraphMode::probe) {
    struct Site {
      std::uint64_t address, destinations;
      std::uint32_t block;
      bool constant_image_dependency;
    };

    if (budget.try_consume(
            {result.cfg->blocks().size(), result.cfg->blocks().size() * sizeof(Site)}) !=
        nyx::BudgetDecline::none)
      return Error("resource_limit", "CFG probe site table exceeds budget");
    std::vector<Site> sites;
    sites.reserve(result.cfg->blocks().size());
    for (std::uint32_t index = 0; index < result.cfg->blocks().size(); ++index) {
      const auto& block = result.cfg->blocks()[index];
      if (!block.dispatch) continue;
      const auto source = block.first_source + block.source_count - 1;

      // BuildCfg installs sorted, distinct dispatch destination addresses as
      // edges; a target block need not exist for its address to be in that set.
      sites.push_back({result.cfg->sources()[source].address, block.edges.size(), index,
                       block.edges.front().constant_image_dependency});
    }

    auto selected = nyx::analysis::BuildRegions(std::move(*result.cfg), budget);
    if (!selected.regions) {
      return Error(selected.reason == nyx::analysis::RegionDecline::resource_limit
                       ? "resource_limit"
                       : "invalid_graph",
                   "Region selection declined; no partial probe published");
    }

    const auto& regions = *selected.regions;
    if (budget.try_consume({regions.candidates().size() + sites.size(),
                            regions.graph().blocks().size() * sizeof(std::uint32_t) + 4096 +
                                sites.size() * 128}) != nyx::BudgetDecline::none) {
      return Error("resource_limit", "CFG probe publication exceeds budget");
    }

    std::vector<std::uint32_t> stopping(regions.graph().blocks().size());
    std::uint64_t dispatch_stops = 0, nominated_sites = 0;
    for (const auto& region : regions.candidates()) {
      if (region.stop != nyx::analysis::RegionStop::dispatch || region.block_ids.empty()) continue;
      ++dispatch_stops;
      ++stopping[region.block_ids.back()];
    }

    for (const auto& site : sites) {
      nominated_sites += stopping[site.block] != 0 && site.destinations >= 2;
    }

    const auto retained_facts = retained->view();
    if (budget.try_consume({retained_facts.pointers.size() + declared.ranges.size(),
                            declared.ranges.size() * 64}) != nyx::BudgetDecline::none) {
      return Error("resource_limit", "CFG probe fact summary exceeds budget");
    }

    const auto stable_pointers = std::count_if(
        retained_facts.pointers.begin(), retained_facts.pointers.end(),
        [](const nyx::ir::RelocatedPointer& pointer) { return pointer.value_stable; });
    std::cout << "{\"schema\":1,\"outcome\":\"cfg_probe\",\"scope\":\"recovery image facts"
                 " and graph-based region selection only; dispatch structure is not a recovered "
                 "transition\","
              << "\"address\":" << address << ",\"size\":" << size
              << ",\"source_groups\":" << regions.graph().sources().size()
              << ",\"block_count\":" << regions.graph().blocks().size()
              << ",\"dispatch_blocks\":" << sites.size()
              << ",\"region_candidates\":" << regions.candidates().size()
              << ",\"dispatch_stop_regions\":" << dispatch_stops
              << ",\"nominated_dispatch_sites\":" << nominated_sites << ",\"memory_profile\":\""
              << MemoryProfileName(profile) << "\""
              << ",\"dispatch_sites\":[";
    bool first = true;
    for (const auto& site : sites) {
      if (!first) std::cout << ',';
      first = false;
      std::cout << "{\"source_address\":" << site.address
                << ",\"destinations\":" << site.destinations
                << ",\"stopping_regions\":" << stopping[site.block]
                << ",\"constant_image_dependency\":"
                << (site.constant_image_dependency ? "true" : "false") << '}';
    }

    std::cout << "],\"declared_constant_ranges\":[";
    for (std::size_t i = 0; i < declared.ranges.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << "{\"address\":" << declared.ranges[i].address
                << ",\"bytes\":" << declared.ranges[i].size << '}';
    }

    std::cout
        << "],\"retained_constant_ranges\":" << retained_facts.constants.size()
        << ",\"retained_relocated_pointers\":" << retained_facts.pointers.size()
        << ",\"stable_relocated_pointers\":" << stable_pointers << ",\"page_aligned_placement\":"
        << (retained_facts.page_aligned_placement ? "true" : "false")
        << ",\"assumptions\":[\"read-only image values are restricted to their file values"
           " during this run\",\"only unrefuted relocated slots supply stable pointer values\","
           "\"page-aligned placement is assumed, not proved\"]"
        << ",\"resources\":{\"work\":" << budget.used().work << ",\"bytes\":" << budget.used().bytes
        << "}}\n";
    return std::cout ? 0 : 1;
  }

  const bool recover = mode == GraphMode::recover;
  std::optional<nyx::analysis::Regions> regions;
  std::optional<RegionReports> reports;
  nyx::analysis::UnflattenResult unflattened;
  nyx::analysis::SsaResult ssa_build;
  std::optional<std::string> ssa_proposals;
  std::optional<std::string> ssa_evaluation;
  nyx::analysis::EntryRelationsResult related;

  // Callees whose decoded instructions travel with the graph as bodies.
  std::size_t summarized_callees = 0;
  if (recover) {
    auto selected = nyx::analysis::BuildRegions(std::move(*result.cfg), budget);
    if (!selected.regions)
      return Error(selected.reason == nyx::analysis::RegionDecline::resource_limit
                       ? "resource_limit"
                       : "invalid_graph",
                   "Region selection declined; no partial artifact published");
    regions = std::move(selected.regions);

    // Without the declaration nearly every graph reaches a call or return, which
    // refuses every relation, so the proof is not run and cannot cost a budget.
    if (declared.abi) {
      related = nyx::analysis::ProveEntryRelations(
          regions->graph(), budget, {true, nyx::a64::kCalleeSaved, declared.return_leaves});
    } else {
      related.relations.emplace();
      related.relations->source_identity = regions->graph().identity();
      related.relations->blocks.resize(regions->graph().blocks().size());
    }

    if (!related.relations) {
      return Error(related.reason == nyx::analysis::RelationDecline::resource_limit
                       ? "resource_limit"
                       : "invalid_graph",
                   "Entry relation proof declined; no partial artifact published");
    }

    const char* decline = nullptr;
    reports =
        RecoverRegions(*regions, budget, retained->view(), profile, *related.relations, decline);
    if (!reports)
      return Error(
          decline,
          "Region recovery or its resource budget declined; no partial artifact published");
    unflattened = nyx::analysis::Unflatten(
        *regions, reports->control, budget, {declared.noreturn, declared.return_leaves},
        {reports->entry_relation_dependent, related.relations->constant_image,
         related.relations->declared_opaque_control, related.relations->calling_convention,
         related.relations->return_leaves});
    if (!unflattened.unflattening) {
      return Error(unflattened.reason == nyx::analysis::UnflattenDecline::resource_limit
                       ? "resource_limit"
                       : "invalid_graph",
                   "Unflattening declined; no partial artifact published");
    }

    auto observed = declared.abi ? nyx::a64::AbiObservability() : nyx::ir::SsaObservability{};
    observed.faults_terminal = declared.faults_terminal;
    ssa_build = nyx::analysis::BuildSsa(*regions, *unflattened.unflattening, reports->paths, budget,
                                        &*related.relations, retained->view(), std::move(observed),
                                        declared.closed_entries);
    if (ssa_build.reason == nyx::ir::SsaDecline::resource_limit)
      return Error("resource_limit",
                   "SSA construction exceeds budget; no partial artifact published");
    // A discovery pass has nothing to find without a graph, and the report
    // it would build next goes nowhere.
    if (discover && !ssa_build.graph) return 0;
    if (ssa_build.graph) {
      ssa_build.graph->SetLoadBias(declared.load_bias);
      ssa_build.graph->SetEntriesClosed(declared.closed_entries);
      if (!declared.call_results.empty()) ssa_build.graph->SetCallResults(declared.call_results);

      // What each callee the graph settles on runs, read from the image up to
      // its first transfer. Whether that makes it a leaf a call can skip is
      // for the graph to judge; this only decodes what is there.
      if (image) {
        std::vector<nyx::ir::SsaCalleeBody> bodies;
        const auto& graph = *ssa_build.graph;

        // A call reaches a callee either as the graph settles it now or, once
        // a pass folds what a run left, as one of those runs' entries.
        std::vector<std::uint64_t> targets;
        for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
          const auto handle = graph.Handle(slot);
          if (!handle) continue;
          if (const auto target = nyx::ir::SsaCallTarget(graph, *graph.Get(*handle), budget))
            targets.push_back(*target);
        }

        for (const auto& result : declared.call_results) targets.push_back(result.target);
        std::sort(targets.begin(), targets.end());
        targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
        if (budget.try_consume({targets.size(), targets.size() * sizeof(nyx::ir::SsaCalleeBody)}) !=
            nyx::BudgetDecline::none)
          return Error("resource_limit",
                       "Callee bodies exceed budget; no partial artifact published");
        for (const auto address : targets) {
          nyx::ir::SsaCalleeBody body{address, {}};
          for (auto at = address; body.groups.size() < nyx::ir::kMaxCalleeBodyGroups; at += 4) {
            // Only bytes an executable segment's file holds and the loader
            // leaves alone are instructions the callee runs as decoded; any
            // other would fault or differ, which is itself observable.
            bool executable = false;
            for (const auto& segment : image->segments())
              executable |= segment.type == 1 && (segment.flags & 1) != 0 &&
                            at >= segment.address && at - segment.address + 4 <= segment.file_size;
            const auto bytes = image->Read(at, 4);
            if (!executable || !bytes || image->LoaderMayWrite(at, 4)) break;
            std::array<std::uint8_t, 4> word{};
            std::copy(bytes->begin(), bytes->end(), word.begin());
            auto decoded = nyx::a64::Decode(at, word, budget, {profile});
            if (!decoded.group) break;
            const bool ends = decoded.group->transfer().has_value();
            body.groups.push_back(std::move(*decoded.group));
            if (ends) break;
          }

          if (!body.groups.empty() && body.groups.back().transfer())
            bodies.push_back(std::move(body));
        }

        std::sort(bodies.begin(), bodies.end(),
                  [](const auto& a, const auto& b) { return a.address < b.address; });
        summarized_callees = bodies.size();
        if (!bodies.empty()) ssa_build.graph->SetCalleeBodies(std::move(bodies));
      }

      nyx::passes::SsaDeclarations ssa_declared;
      ssa_declared.image_access = declared.image_access;
      ssa_declared.closed_entries = declared.closed_entries;
      ssa_declared.return_leaves = declared.return_leaves;

      // The ABI declaration is the one that says callees return only to their
      // continuation.
      ssa_declared.call_returns = declared.abi;
      ssa_declared.trap_stops = declared.trap_stops;
      if (declared.frame) {
        // Without the ABI, SP need not be aligned and a callee may move it.
        ssa_declared.frame = nyx::ir::PrivateFrameContract{nyx::a64::kSp,
                                                           declared.frame->first,
                                                           declared.frame->second,
                                                           true,
                                                           true,
                                                           true,
                                                           declared.abi ? 16U : 1U,
                                                           declared.frame_unreached_by_callees,
                                                           declared.abi};
      }

      std::vector<nyx::ir::Group> ssa_sources;
      if (declared.closed_entries || ssa_request.state_path) {
        const auto records = regions->graph().sources();
        if (records.size() > SIZE_MAX / sizeof(nyx::ir::Group) ||
            budget.try_consume({records.size(), records.size() * sizeof(nyx::ir::Group)}) !=
                nyx::BudgetDecline::none)
          return Error("resource_limit", "SSA source binding exceeds budget");
        ssa_sources.reserve(records.size());
        for (const auto& record : records) {
          const auto nodes = record.semantics ? record.semantics->nodes().size() : 0;
          const auto writes = record.semantics ? record.semantics->writes().size() : 0;
          if (nodes > UINT64_MAX / sizeof(nyx::ir::Node) ||
              writes > UINT64_MAX / sizeof(nyx::ir::Write) ||
              budget.try_consume({1 + record.bytes.size() + nodes + writes,
                                  record.bytes.size() + nodes * sizeof(nyx::ir::Node) +
                                      writes * sizeof(nyx::ir::Write)}) != nyx::BudgetDecline::none)
            return Error("resource_limit", "SSA source binding exceeds budget");
          if (record.semantics)
            ssa_sources.push_back(*record.semantics);
          else
            ssa_sources.emplace_back(record.address, record.bytes, std::vector<nyx::ir::Node>{},
                                     std::vector<nyx::ir::Write>{});
        }
      }

      auto run =
          nyx::passes::RunSsaPipeline(*ssa_build.graph, ssa_declared, ssa_sources, retained->view(),
                                      budget, ssa_request.passes, ssa_request.rounds);
      if (run.reason == nyx::passes::SsaPipelineDecline::invalid_selection)
        return Error("usage", "--passes names each registered pass at most once");
      if (run.reason == nyx::passes::SsaPipelineDecline::incomplete_record)
        return Error("invalid_graph",
                     "An SSA pass published no stage record; no partial artifact published");
      if (run.reason == nyx::passes::SsaPipelineDecline::retired_work_observed)
        return Error(
            "invalid_graph",
            "A retired loop or call left work a later stage reads; no partial artifact published");
      if (!run.result)
        return Error("resource_limit",
                     "SSA proposals exceed budget; no partial artifact published");
      auto& pipeline = *run.result;
      if (discover) {
        const auto& final_graph = pipeline.provisional ? *pipeline.provisional : *ssa_build.graph;
        CollectUnrunCallees(final_graph, *discover, budget);

        // Declared bytes still in this pass's facts that a store the graph
        // places writes. Stores nobody can place stay the writer scan's.
        const auto view = retained->view();
        const auto stores = nyx::ir::SsaImageStores::Collect(
            final_graph, view.page_aligned_placement, budget, &view);
        if (!stores) return Error("resource_limit", "SSA store collection exceeds budget");
        for (const auto& store : stores->placed()) {
          if (budget.try_consume({view.constants.size() + 1, sizeof(SsaRefuted)}) !=
              nyx::BudgetDecline::none)
            return Error("resource_limit", "SSA store collection exceeds budget");
          for (const auto& piece : view.constants) {
            const auto first = std::max(store.address, piece.address);
            const auto last =
                std::min(store.address + store.size, piece.address + piece.bytes.size());
            if (first < last)
              discover->refuted.push_back(
                  {first, last - first, store.address, store.block, store.node, store.size * 8});
          }

          if (budget.try_consume({view.pointers.size(), 0}) != nyx::BudgetDecline::none)
            return Error("resource_limit", "SSA store collection exceeds budget");
          for (const auto& slot : view.pointers) {
            if (!slot.value_stable) continue;
            if (store.address < slot.address + 8 && slot.address < store.address + store.size)
              discover->refuted_slots.push_back(slot.address);
          }
        }

        auto& addressed = discover->addressed;
        std::sort(addressed.begin(), addressed.end());
        addressed.erase(std::unique(addressed.begin(), addressed.end()), addressed.end());
        return 0;
      }

      // The candidate is the exact published batch of the exact image under
      // every declaration of the run and the exact stub declaration: change
      // any of them and it is another candidate, which a check record for
      // this one says nothing about. The stub bytes are read once, so the
      // evaluated and the hashed declaration cannot differ.
      std::optional<std::string> stubs;
      std::uint64_t stub_count = 0;
      if (ssa_request.stubs_path) {
        stubs = ReadBounded(ssa_request.stubs_path, nyx::cli::kMaxStubDeclaration, budget);
        const auto count =
            stubs ? nyx::cli::SsaStubCount(
                        {reinterpret_cast<const std::uint8_t*>(stubs->data()), stubs->size()})
                  : std::nullopt;
        if (!count)
          return Error("invalid_stubs",
                       "SSA stub declaration is unreadable or not a valid NYXSTB01 file");
        stub_count = *count;
      }

      std::string run_declarations =
          "profile=" + std::string(MemoryProfileName(profile)) +
          ";abi=" + (declared.abi ? "aapcs64" : "none") +
          ";returns=" + (declared.return_leaves ? "leave" : "none") +
          ";faults=" + (declared.faults_terminal ? "terminal" : "none") +
          ";entries=" + (declared.closed_entries ? "closed" : "none") +
          ";image_access=" + (declared.image_access ? "readable" : "none") +
          ";frame_reach=" + (declared.frame_unreached_by_callees ? "none" : "any") + ";frame=";
      if (declared.frame)
        run_declarations +=
            std::to_string(declared.frame->first) + ":" + std::to_string(declared.frame->second);
      run_declarations += ";noreturn=";
      for (const auto target : declared.noreturn) run_declarations += std::to_string(target) + ",";
      run_declarations += ";ranges=";
      for (const auto& range : declared.ranges)
        run_declarations += std::to_string(range.address) + ":" + std::to_string(range.size) + ",";
      // A candidate is the batch under every declaration of the run, and what
      // an execution was observed to leave is one. Two runs admitting
      // different observations are different candidates even where the
      // published graph happens to come out the same.
      run_declarations += ";load_bias=";
      if (declared.load_bias) run_declarations += std::to_string(*declared.load_bias);
      run_declarations +=
          ";relocations=" + std::string(declared.relocations_stable ? "stable" : "none");
      run_declarations += ";symbol_binding=" + std::string(declared.symbols_own ? "own" : "none");
      run_declarations +=
          ";image_data=" + std::string(declared.data_unwritten ? "unwritten" : "none");
      if (declared.written_slots)
        run_declarations += ";written_slots=" + std::to_string(declared.written_slots->size()) +
                            "/" + std::to_string(declared.written_slots_unexamined);
      run_declarations += ";derived=";
      for (const auto& [slot, value] : declared.derived)
        run_declarations += std::to_string(slot) + ":" + std::to_string(value) + ",";
      run_declarations += ";derived_unstable=";
      for (const auto slot : declared.derived_unstable)
        run_declarations += std::to_string(slot) + ",";
      // Which declared bytes survive the population's own stores is part of
      // the fact set the candidate rests on.
      run_declarations += ";constant_pieces=";
      for (const auto& piece : retained->constants)
        run_declarations += std::to_string(piece.address) + ":" +
                            std::to_string(piece.bytes.size()) + (piece.read_only ? "r," : ",");
      run_declarations += ";ssa_refuted=";
      for (const auto& refuted : declared.ssa_refuted)
        run_declarations +=
            std::to_string(refuted.address) + ":" + std::to_string(refuted.bytes) + ",";
      if (declared.ssa_refutation_unsettled) run_declarations += ";ssa_unsettled";
      run_declarations += ";call_results=";
      for (const auto& result : declared.call_results)
        run_declarations += std::to_string(result.target) + ":" + std::to_string(result.storage) +
                            ":" + std::to_string(result.value) + ",";
      std::optional<nyx::verify::CheckRecord> record;
      if (ssa_request.check_path) {
        const auto text = ReadBounded(ssa_request.check_path, kMaxCheckRecord, budget);
        if (!text)
          return Error("invalid_check_record", "SSA check record is unreadable or too large");
        record = nyx::verify::ParseCheckRecord(*text);
        if (!record)
          return Error("invalid_check_record",
                       "SSA check record is not a consistent NYXCHK01 record");
      }

      const auto digest_bytes = pipeline.head.size() + pipeline.body.size() + pipeline.text.size() +
                                (stubs ? stubs->size() : 0) + run_declarations.size() + 256;
      if (budget.try_consume({digest_bytes, digest_bytes}) != nyx::BudgetDecline::none)
        return Error("resource_limit", "SSA candidate identity exceeds budget");
      // Each field is length-prefixed so no two field sequences hash the same bytes.
      nyx::Sha256 hash;
      const auto field = [&](std::string_view text) {
        hash.Update(std::to_string(text.size()));
        hash.Update(std::string_view(":"));
        hash.Update(text);
      };

      field("nyx.ssa.candidate.v1");
      field(ssa_request.image_digest);
      field(std::to_string(address) + ":" + std::to_string(size));
      field(run_declarations);
      field(pipeline.head);
      field(pipeline.body);
      field(pipeline.text);
      field(stubs ? "stubs:" + *stubs : std::string("no_stubs"));
      const auto candidate = "sha256:" + hash.FinishHex();
      std::size_t proposed = 0;
      for (const auto& stage : pipeline.stages)
        proposed += stage.outcome == nyx::passes::SsaStageOutcome::proposed;
      std::uint64_t executable_edits = 0;
      for (const auto& stage : pipeline.stages) executable_edits += stage.executable_edits;
      const auto decision = nyx::verify::Decide(record ? &*record : nullptr, candidate,
                                                {proposed, executable_edits, stub_count});
      const bool bound = record && record->candidate == candidate;
      const char* check = !bound                                                ? "NOT CHECKED"
                          : record->status == nyx::verify::CheckStatus::matched ? "matched"
                          : record->status == nyx::verify::CheckStatus::refuted ? "refuted"
                          : record->status == nyx::verify::CheckStatus::inconclusive
                              ? "inconclusive"
                              : "NOT CHECKED";
      const char* status = decision.status == nyx::verify::Acceptance::accepted   ? "accepted"
                           : decision.status == nyx::verify::Acceptance::rejected ? "rejected"
                                                                                  : "provisional";
      std::string published = pipeline.head;
      published += ",\"proposals\":{\"status\":\"";
      published += status;
      published += "\",\"whole_function_check\":\"";
      published += check;
      published += "\",\"candidate_digest\":\"" + candidate + "\",\"acceptance\":{\"reason\":\"";
      published += decision.reason;
      published +=
          "\",\"required\":[\"stage proofs for every journaled edit\","
          "\"whole-function differential for this candidate: every examined state matched,"
          " none inconclusive, every executable edit and declared stub covered, and the"
          " ordered call-event trace compared\"],\"check\":";
      if (record) {
        published += "{\"candidate\":\"" + record->candidate + "\",\"bound\":";
        published += bound ? "true" : "false";
        for (const auto& [name, value] :
             {std::pair<const char*, std::uint64_t>{"states", record->states},
              {"matched", record->matched},
              {"refuted", record->refuted},
              {"inconclusive", record->inconclusive},
              {"edits_executable", record->edits_executable},
              {"edits_covered", record->edits_covered},
              {"stubs_declared", record->stubs_declared},
              {"stubs_called", record->stubs_called}})
          published += ",\"" + std::string(name) + "\":" + std::to_string(value);
        published += ",\"call_trace\":\"";
        published += record->call_trace == nyx::verify::CallTrace::matched    ? "matched"
                     : record->call_trace == nyx::verify::CallTrace::mismatch ? "mismatch"
                                                                              : "not_checked";
        published +=
            "\",\"scope\":\"bounded independent observations bound to this candidate"
            " digest; the record's producer is not authenticated and no universal proof"
            " follows from it\"}";
      } else {
        published += "null";
      }

      published +=
          "},\"tallies\":{\"stages\":" + std::to_string(pipeline.stages.size()) +
          ",\"proposed_stages\":" + std::to_string(proposed) + ",\"accepted_stages\":" +
          std::to_string(decision.status == nyx::verify::Acceptance::accepted ? proposed : 0) +
          "},\"scope\":\"each stage runs over the previous stage's provisional graph; the"
          " batch is accepted or rejected as a whole, and every stage inherits the SSA"
          " edges' assumptions\",";
      published += pipeline.body;

      // A refuted batch is rolled back: no transformed graph is published and
      // nothing downstream may start from it.
      published += ",\"text\":";
      published += decision.status == nyx::verify::Acceptance::rejected ? "null" : pipeline.text;
      published += "}";
      if (budget.try_consume({published.size(), published.size()}) != nyx::BudgetDecline::none)
        return Error("resource_limit",
                     "SSA proposals exceed budget; no partial artifact published");
      if (ssa_request.listing) {
        // A rolled-back batch leaves the base graph as what the run publishes.
        const auto& shown =
            decision.status != nyx::verify::Acceptance::rejected && pipeline.provisional
                ? *pipeline.provisional
                : *ssa_build.graph;
        auto target = nyx::a64::ListingTarget();
        target.sources = ssa_sources;
        const auto listed = nyx::ir::ListSsa(shown, target, budget);
        if (!listed.text)
          return Error(listed.reason == nyx::ir::SsaDecline::resource_limit ? "resource_limit"
                                                                            : "invalid_graph",
                       "SSA listing declined; no partial artifact published");
        published += ",\"listing\":" + JsonString(*listed.text);
      }

      ssa_proposals = std::move(published);
      if (ssa_request.state_path) {
        const auto& candidate_graph =
            pipeline.provisional ? *pipeline.provisional : *ssa_build.graph;
        auto evaluated = nyx::cli::EvaluateSsaRequest(
            candidate_graph, ssa_sources, address, ssa_request.state_path,
            stubs ? std::optional(std::span(reinterpret_cast<const std::uint8_t*>(stubs->data()),
                                            stubs->size()))
                  : std::nullopt,
            budget);
        if (!evaluated.json)
          return Error(evaluated.reason, "SSA initial state or evaluation declined");
        ssa_evaluation = std::move(evaluated.json);
      }
    } else if (ssa_request.state_path || ssa_request.check_path) {
      return Error("invalid_graph",
                   "SSA construction declined; no candidate to evaluate or accept");
    }
  }

  const auto& graph = regions ? regions->graph() : *result.cfg;
  const auto effective_facts = retained->view();

  // How many offered values the fact set actually carries: a slot the scan
  // refuted, or one naming nothing in the image, is not among them.
  const auto derived_slots = std::count_if(
      declared.derived.begin(), declared.derived.end(),
      [&](const std::pair<std::uint64_t, std::uint64_t>& item) {
        const auto at = std::lower_bound(effective_facts.pointers.begin(),
                                         effective_facts.pointers.end(), item.first,
                                         [](const nyx::analysis::RelocatedPointer& slot,
                                            std::uint64_t key) { return slot.address < key; });
        return at != effective_facts.pointers.end() && at->address == item.first &&
               at->value_stable && at->target == item.second;
      });
  const auto stable_pointers =
      std::count_if(effective_facts.pointers.begin(), effective_facts.pointers.end(),
                    [](const nyx::ir::RelocatedPointer& pointer) { return pointer.value_stable; });
  // A retained piece speaks for the declared range it was cut from.
  const auto declared_origin = [&](std::size_t origin) {
    const auto& fact = facts.constants[origin];
    return std::any_of(declared.ranges.begin(), declared.ranges.end(),
                       [&](const DeclaredRange& range) {
                         return fact.address == range.address && fact.bytes.size() == range.size;
                       });
  };

  const bool retained_auto_constant =
      std::any_of(retained->origins.begin(), retained->origins.end(),
                  [&](std::size_t origin) { return !declared_origin(origin); });
  const bool auto_cut = std::any_of(
      refuted_spans.begin(), refuted_spans.end(),
      [&](const nyx::ir::RefutedConstantSpan& span) { return !declared_origin(span.index); });
  std::vector<std::uint64_t> refuted_ranges;
  if (budget.try_consume(
          {refuted_spans.size() * 2, refuted_spans.size() * sizeof(std::uint64_t)}) !=
      nyx::BudgetDecline::none)
    return Error("resource_limit", "Refuted range table exceeds budget");
  for (const auto& span : refuted_spans)
    refuted_ranges.push_back(facts.constants[span.index].address);
  std::sort(refuted_ranges.begin(), refuted_ranges.end());
  refuted_ranges.erase(std::unique(refuted_ranges.begin(), refuted_ranges.end()),
                       refuted_ranges.end());
  if (budget.try_consume({graph.blocks().size(), graph.blocks().size() * sizeof(std::string)}) !=
      nyx::BudgetDecline::none) {
    return Error("resource_limit", "CFG serialization table exceeds budget");
  }

  std::vector<std::string> blocks;
  blocks.reserve(graph.blocks().size());
  std::uint64_t publication = 0, edge_count = 0, modeled = 0;
  if (reports)
    for (const auto& record : reports->json) publication += record.size();
  for (const auto& block : graph.blocks()) {
    edge_count += block.edges.size();
    if (block.ssa) {
      auto printed = nyx::ir::PrintJson(*block.ssa, budget);
      if (!printed.json)
        return Error(printed.reason == nyx::ir::PrintDecline::invalid_group ? "invalid_group"
                                                                            : "resource_limit",
                     "CFG serialization declined; no partial graph published");
      publication += printed.json->size();
      blocks.push_back(std::move(*printed.json));
    } else
      blocks.emplace_back("null");
  }

  for (const auto& source : graph.sources())
    if (source.semantics) ++modeled;
  const auto envelope = std::string_view(path).size() * 16 + 4096 + graph.sources().size() * 256 +
                        size * 2 + graph.blocks().size() * 1024 + edge_count * 768 +
                        refuted_spans.size() * 224 + graph.refuted_pointer_indices().size() * 24 +
                        (unflattened.unflattening
                             ? unflattened.unflattening->transitions.size() * 256 +
                                   (unflattened.unflattening->retired_blocks.size() +
                                    unflattened.unflattening->unresolved_successor_blocks.size()) *
                                       24 +
                                   RecoveredGraphExtent(*unflattened.unflattening)
                             : 0);
  if (budget.try_consume({publication + envelope, envelope}) != nyx::BudgetDecline::none) {
    return Error("resource_limit", "CFG publication exceeds budget; no partial graph published");
  }

  std::cout << "{\"schema\":1,\"outcome\":\"" << (recover ? "recovered_regions" : "cfg")
            << "\",\"input\":" << JsonString(path)
            << ",\"input_bytes_hex\":" << JsonString(HexBytes(path)) << ",\"address\":" << address
            << ",\"size\":" << size << ",\"source_groups\":" << graph.sources().size()
            << ",\"modeled_groups\":" << modeled
            << ",\"unmodeled_groups\":" << graph.sources().size() - modeled
            << ",\"block_count\":" << graph.blocks().size() << ",\"edge_count\":" << edge_count
            << ",\"passes\":" << graph.passes() << ",\"generation\":" << graph.generation()
            << ",\"refuted_constant_ranges\":" << refuted_ranges.size()
            << ",\"refuted_pointer_slots\":" << graph.refuted_pointer_indices().size()
            << ",\"refuted_constant_range_addresses\":[";
  // A range listed here lost at least one byte, not necessarily all of them:
  // refuted_constant_spans says which, and what wrote each.
  for (std::size_t i = 0; i < refuted_ranges.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << refuted_ranges[i];
  }

  std::cout << "],\"refuted_constant_spans\":[";

  // `refuted_by` says which scan saw the store: the CFG's, block by block, or
  // an earlier pass's whole graph, which also names the block and node.
  for (std::size_t i = 0; i < refutations.size(); ++i) {
    const auto& [span, store] = refutations[i];
    const auto& range = facts.constants[span.index];
    std::cout << (i ? "," : "") << "{\"range\":" << range.address
              << ",\"range_bytes\":" << range.bytes.size() << ",\"address\":" << span.address
              << ",\"bytes\":" << span.bytes << ",\"written_at\":" << span.write.address
              << ",\"width\":" << span.write.width
              << ",\"whole\":" << (span.whole ? "true" : "false") << ",\"refuted_by\":\""
              << (store ? "ssa_store" : "cfg_store") << '"';
    if (store) std::cout << ",\"block\":" << store->block << ",\"node\":" << store->node;
    std::cout << '}';
  }

  // False when the passes stopped at their bound with a graph still writing
  // declared bytes the published run reads.
  std::cout << "],\"ssa_refutation_settled\":"
            << (declared.ssa_refutation_unsettled ? "false" : "true")
            << ",\"refuted_pointer_slot_addresses\":[";
  for (std::size_t i = 0; i < graph.refuted_pointer_indices().size(); ++i) {
    if (i) std::cout << ',';
    std::cout << facts.pointers[graph.refuted_pointer_indices()[i]].address;
  }

  const auto& fact_scan = graph.image_fact_scan();

  // What the refutation could examine. A declaration it did not refute is not
  // thereby established: the scan reads the supplied population only, and only
  // the blocks whose instructions it models.
  std::cout
      << "],\"image_fact_scan\":{\"scope\":\"supplied_population\",\"scanned_blocks\":"
      << fact_scan.scanned_blocks << ",\"unscanned_blocks\":" << fact_scan.unscanned_blocks
      << ",\"unresolved_image_writes\":" << fact_scan.unresolved_writes
      << ",\"discharges_declaration\":false}"
      << ",\"classification\":\"user_requested_range\",\"scope\":\"supplied_population\","
         "\"partition\":\"known_entries_only\",\"reachability\":\"not_proved\",\"interior_"
         "entries\":\"not_excluded\","
         "\"whole_function_recovery\":false,\"successor_scope\":\"successful_terminal_"
         "instruction\","
         "\"execution_assumptions\":[\"immutable code\",\"BTI inactive\",\"GCS "
         "inactive\",\"pointer authentication inactive\""
      << (!retained_auto_constant
              ? ""
              : ",\"non-writable load segments are never made writable or aliased by a writable"
                " mapping\",\"no unmodeled writer changes them, including callees\"")
      << (!retained_auto_constant || !auto_cut
              ? ""
              : ",\"a load segment refuted_constant_spans cuts keeps its other bytes, each"
                " claimed on its own, so a store whose address rested on a refuted byte may"
                " have gone unseen\"")
      << (stable_pointers == 0 ? ""
          : declared.relocations_stable
              ? ",\"every relocated pointer slot keeps the value the loader wrote, the"
                " loader-writable ones included, except those a store in the supplied"
                " population refuted; writers elsewhere in the image, stores this range"
                " models but cannot place, and stores it does not model, were not"
                " examined; a slot this range itself writes gives no value to stores"
                " through it\""
              : ",\"the loader maps PT_GNU_RELRO read-only once relocation is done, so the"
                " slots inside it keep the value it wrote, except those a store in the"
                " supplied population refuted\"")
      << (declared.derived_ranges == 0
              ? ""
              : ",\"every data word the image's code was not found writing keeps its file"
                " value, including against the stores the whole-image scan could not place;"
                " the declared ranges it admitted are listed with the others\"")
      << (summarized_callees == 0
              ? ""
              : ",\"a callee whose body the graph carries runs the instructions decoded from"
                " the image at its address, and returns where that body says\"")
      << (!declared.symbols_own
              ? ""
              : ",\"every symbol the image defines is bound, eagerly, to its own definition, so a"
                " slot the loader fills from one holds that definition's location\"")
      << (!declared.written_slots
              ? ""
              : ",\"a whole-image scan found no writer for the relocated slots outside"
                " PT_GNU_RELRO that this run reads, so they still hold what the loader"
                " wrote, though " +
                    std::to_string(declared.written_slots_unexamined) +
                    " instructions and unplaced stores in that image went unexamined\"")
      << (declared.call_results.empty()
              ? ""
              : ",\"a call to a target an execution returned from leaves the register values"
                " that run produced, which is one observed execution and not a proof over"
                " every execution\"")
      << (derived_slots == 0
              ? ""
              : ",\"the slots an executed initializer was observed to write hold the values"
                " that run produced, which is one observed execution and not a proof over"
                " every execution\"")
      << (std::none_of(
              declared.derivation_runs.begin(), declared.derivation_runs.end(),
              [](const Declarations::DerivationRun& run) { return run.status != "from_record"; })
              ? ""
              : ",\"those runs were made here with the argument registers guessed, all zero,"
                " all 0xa5 and all 0x5a, and only what every guess produced was admitted;"
                " a value that varies with an argument no guess distinguished, or with"
                " memory this image does not start with, is not excluded\"")
      << (!declared.load_bias
              ? ""
              : ",\"the image is placed at the declared address, so a location is a number and"
                " arithmetic on one folds\"")
      << (facts.page_aligned_placement ? ",\"the image is placed at a page-aligned address\"" : "")
      << (!reports || !reports->removed_memory_effects
              ? ""
              : ",\"single-threaded normal memory without concurrent observers\","
                "\"omitted accesses mapped with required permissions\","
                "\"no resource-limit outcome\"");
  for (const auto& range : declared.ranges) {
    const auto of_range = [&](std::size_t origin) {
      return facts.constants[origin].address == range.address &&
             facts.constants[origin].bytes.size() == range.size;
    };

    if (std::none_of(retained->origins.begin(), retained->origins.end(), of_range)) continue;
    const bool cut =
        std::any_of(refuted_spans.begin(), refuted_spans.end(),
                    [&](const nyx::ir::RefutedConstantSpan& span) { return of_range(span.index); });
    std::cout << ",\"the declared range " << range.address << "+" << range.size
              << " keeps the value the image is loaded with, from its file or,"
                 " past the file, the zero the loader leaves"
              << (cut ? ", except the bytes refuted_constant_spans lists; each remaining byte"
                        " is claimed on its own, so a store whose address rested on a refuted"
                        " byte may have gone unseen"
                      : "")
              << "\"";
  }

  for (const auto target : declared.noreturn) {
    std::cout << ",\"a call to the declared target " << target << " never returns\"";
  }

  if (declared.abi) {
    std::cout
        << ",\"every callee returns only through its call's continuation, with AAPCS64 X19-X29"
           " and SP as it found them\"";
  }

  if (declared.return_leaves)
    std::cout << ",\"every untargeted return leaves the selected function population\"";
  if (declared.trap_stops)
    std::cout
        << ",\"a trap never resumes inside the selected population except at a listed entry\"";
  if (declared.faults_terminal)
    std::cout << ",\"a fault or trap ends the run and nothing observes its registers\"";
  std::cout << "],\"declared_constant_ranges\":[";
  for (std::size_t i = 0; i < declared.ranges.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << "{\"address\":" << declared.ranges[i].address
              << ",\"bytes\":" << declared.ranges[i].size << '}';
  }

  std::cout << "],\"declared_noreturn_targets\":[";
  for (std::size_t i = 0; i < declared.noreturn.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << declared.noreturn[i];
  }

  std::cout << "],\"declared_abi\":" << (declared.abi ? "\"aapcs64\"" : "null")
            << ",\"declared_return_leaves\":" << (declared.return_leaves ? "true" : "false")
            << ",\"constant_image_ranges\":" << effective_facts.constants.size()
            << ",\"derived_call_results\":" << declared.call_results.size()
            << ",\"derived_memory_slots\":" << derived_slots
            << ",\"derivation_runs\":" << DerivationRuns(declared.derivation_runs)
            << ",\"derived_values\":" << OfferedValues(effective_facts, declared.derived)
            << ",\"relocated_pointers\":" << stable_pointers
            << ",\"declared_constant_image_ranges\":" << facts.constants.size()
            << ",\"declared_relocated_pointer_slots\":" << facts.pointers.size()
            << ",\"memory_profile\":\"" << MemoryProfileName(profile)
            << "\",\"architectural_fault_authority\":false,\"sp_alignment_check\":\"disabled\","
               "\"memory_fault_scope\":\"experimental_QEMU_reference\",\"deobfuscation\":\""
            << (recover ? "entry_scoped_regions" : "not_performed") << "\",\"entries\":[";
  for (std::size_t i = 0; i < graph.entries().size(); ++i) {
    if (i) std::cout << ',';
    std::cout << graph.entries()[i];
  }

  std::cout << "],\"sources\":[";
  constexpr char hex[] = "0123456789abcdef";
  bool first = true;
  for (const auto& source : graph.sources()) {
    if (!first) std::cout << ',';
    first = false;
    std::cout << "{\"address\":" << source.address << ",\"bytes\":\"";
    for (auto byte : source.bytes) std::cout << hex[byte >> 4] << hex[byte & 15];
    const auto reason = source.opaque_reason;
    const char* status = source.semantics                                     ? "modeled"
                         : reason == nyx::analysis::OpaqueReason::unsupported ? "unsupported"
                         : reason == nyx::analysis::OpaqueReason::invalid_encoding
                             ? "invalid_encoding"
                             : "not_decoded";
    std::cout << "\",\"status\":\"" << status << "\",\"opaque_control\":\""
              << (source.opaque_control == nyx::analysis::OpaqueControl::normal_fallthrough
                      ? "declared_normal_fallthrough"
                  : source.opaque_control == nyx::analysis::OpaqueControl::trap ? "declared_trap"
                                                                                : "unknown")
              << "\"}";
  }

  std::cout << "],\"blocks\":[";
  for (std::size_t index = 0; index < graph.blocks().size(); ++index) {
    if (index) std::cout << ',';
    const auto& block = graph.blocks()[index];
    std::cout << "{\"id\":" << index << ",\"first_source\":" << block.first_source
              << ",\"source_count\":" << block.source_count
              << ",\"entry\":" << graph.sources()[block.first_source].address
              << ",\"ssa\":" << blocks[index] << ",\"control\":";
    if (block.control) {
      std::cout
          << "{\"block_revision\":" << block.control->block_revision
          << ",\"terminal_source\":" << block.control->terminal_source << ",\"kind\":\""
          << TransferName(block.control->kind)
          << "\",\"proof_scope\":\"all_inputs_successful_terminal\",\"callee_return_unknown\":"
          << (block.control->callee_return_unknown ? "true" : "false") << '}';
    } else
      std::cout << "null";
    std::cout << ",\"dispatch\":";
    if (block.dispatch) {
      std::cout << "{\"guard_block\":" << block.dispatch->guard_block
                << ",\"index\":" << block.dispatch->index << ",\"bound\":" << block.dispatch->bound
                << ",\"scope\":\"complete over the edges below\"}";
    } else
      std::cout << "null";
    std::cout << ",\"edges\":[";
    for (std::size_t i = 0; i < block.edges.size(); ++i) {
      if (i) std::cout << ',';
      const auto& edge = block.edges[i];
      const char* target_kind =
          edge.target.kind == nyx::analysis::TargetKind::image_location     ? "image_location"
          : edge.target.kind == nyx::analysis::TargetKind::absolute_runtime ? "absolute_runtime"
                                                                            : "unknown";
      std::cout << "{\"kind\":\"" << EdgeName(edge.kind) << "\",\"target\":{\"kind\":\""
                << target_kind << "\",\"address\":" << edge.target.address << ",\"value\":";
      if (edge.target.value)
        std::cout << *edge.target.value;
      else
        std::cout << "null";
      std::cout << "},\"resolution\":\"" << ResolutionName(edge.resolution)
                << "\",\"target_source\":";
      if (edge.target_source)
        std::cout << *edge.target_source;
      else
        std::cout << "null";
      std::cout << ",\"target_block\":";
      if (edge.target_block)
        std::cout << *edge.target_block;
      else
        std::cout << "null";
      std::cout << ",\"condition\":";
      if (edge.condition)
        std::cout << *edge.condition;
      else
        std::cout << "null";
      std::cout << ",\"when\":";
      if (edge.when)
        std::cout << (*edge.when ? "true" : "false");
      else
        std::cout << "null";
      std::cout << ",\"constant_image_dependency\":"
                << (edge.constant_image_dependency ? "true" : "false");
      std::cout << '}';
    }

    std::cout << "]}";
  }

  std::cout << ']';
  if (reports) {
    std::cout
        << ",\"region_contract\":{\"entry_scope\":\"first_source_only\","
           "\"edge_references\":\"owned_original_cfg\",\"selected_polarity\":\"provenance_not_"
           "assumption\","
           "\"continuation_guard\":\"actual_successor_address\",\"original_cfg\":\"unchanged\","
           "\"residual_edges\":\"all_original_edges_retained\",\"memory_invariance\":\"not_"
           "assumed\","
           "\"removed_source_groups\":0,\"removed_memory_effects\":"
        << reports->removed_memory_effects << ",\"memory_effect_scope\":\""
        << (reports->removed_memory_effects ? "conditional_atomic_scalar" : "none")
        << "\",\"architectural_fault_authority\":false,\"redirected_edges\":0,"
           "\"machine_patching\":\"not_performed\",\"candidate_source_population\":\"occurrences\","
           "\"starts\":\"every_modeled_block\",\"max_blocks\":8,\"max_sources\":512,\"max_forks\":"
           "1},"
           "\"region_counts\":{\"candidates\":"
        << reports->json.size() << ",\"simplified\":" << reports->simplified
        << ",\"unchanged\":" << reports->unchanged << ",\"declined\":" << reports->declined
        << ",\"source_occurrences\":" << reports->occurrences
        << ",\"proved_divergences\":" << reports->proved_divergences
        << ",\"removed_conditional_transfers\":" << reports->control_edits
        << ",\"recovered_dispatch_branches\":" << reports->recovered_branches
        << ",\"memory_edits\":" << reports->memory_edits
        << ",\"arithmetic_edits\":" << reports->arithmetic_edits
        << ",\"removed_memory_effects\":" << reports->removed_memory_effects
        << ",\"image_edits\":" << reports->image_edits
        << ",\"declared_ranges_written\":" << reports->contradicted_ranges
        << ",\"relocated_slots_written\":" << reports->contradicted_pointers
        << ",\"entry_relation_paths\":" << reports->entry_relation_paths
        << "},\"dispatch_summary\":{\"scope\":\"complete over the destinations one transfer"
           " can reach, summarised over the candidates examined; not a claim about paths this"
           " selection never followed\",\"transfers\":[";
    bool first_dispatch = true;
    for (const auto& [address, entry] : reports->dispatches) {
      if (entry.with_set == 0) continue;
      if (!first_dispatch) std::cout << ',';
      first_dispatch = false;
      std::cout << "{\"source_address\":" << address << ",\"index_bound\":" << entry.bound
                << ",\"reached_by\":" << entry.reached
                << ",\"proving_a_complete_set\":" << entry.with_set
                << ",\"reaching_without_the_guard\":" << entry.without
                << ",\"every_proof_agrees\":" << (entry.agreeing ? "true" : "false")
                << ",\"destinations\":[";
      for (std::size_t i = 0; i < entry.destinations.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << entry.destinations[i];
      }

      std::cout << "]}";
    }

    std::cout << "]},";
    PrintUnflattening(std::cout, *regions, *unflattened.unflattening, *related.relations);
    std::uint64_t ssa_phis = 0, ssa_unknown = 0;
    if (ssa_build.graph) {
      for (std::size_t slot = 0; slot < ssa_build.graph->slots(); ++slot) {
        const auto handle = ssa_build.graph->Handle(slot);
        if (!handle) continue;
        const auto& block = *ssa_build.graph->Get(*handle);
        ssa_phis += block.phis.size();
        for (const auto& edge : block.edges)
          ssa_unknown += !edge.target_block && edge.kind != nyx::ir::SsaEdgeKind::callee &&
                         edge.kind != nyx::ir::SsaEdgeKind::return_ &&
                         edge.kind != nyx::ir::SsaEdgeKind::trap;
      }
    }

    std::cout << ",\"ssa_graph\":{\"status\":\""
              << (ssa_build.graph                                           ? "built"
                  : ssa_build.reason == nyx::ir::SsaDecline::resource_limit ? "resource_limit"
                                                                            : "invalid_graph")
              << "\",\"blocks\":" << (ssa_build.graph ? ssa_build.graph->slots() : 0)
              << ",\"phis\":" << ssa_phis << ",\"unknown_successors\":" << ssa_unknown;
    if (ssa_proposals) std::cout << *ssa_proposals;
    if (ssa_evaluation) std::cout << ",\"evaluation\":" << *ssa_evaluation;
    std::cout << '}';
    std::cout << ",\"regions\":[";
    for (std::size_t i = 0; i < reports->json.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << reports->json[i];
    }

    std::cout << ']';
  }

  const auto remaining = budget.remaining();
  std::cout << ",\"resources\":{\"work\":" << budget.used().work
            << ",\"bytes\":" << budget.used().bytes
            << ",\"work_limit\":" << budget.used().work + remaining.work
            << ",\"byte_limit\":" << budget.used().bytes + remaining.bytes << "}}\n";
  return std::cout ? 0 : 1;
}

// An itinerary is one Path, so its instruction count is bounded by
// BlockLimits::max_groups; kMaxItineraryBytes is exactly that many AArch64
// instructions and the two only move together. The range count is a separate
// limit with different meaning: it bounds how many control transfers the
// itinerary may follow, which is what binds when the itinerary is a traced route
// through flattened code. Raising it costs array size and nothing else, because
// range boundaries do not survive the parse.
constexpr std::size_t kMaxItineraryRanges = 512;
constexpr std::uint64_t kMaxItineraryBytes = 16384;

int SimplifyPath(const char* filename, std::string_view specification,
                 nyx::a64::MemoryProfile profile, nyx::Resources resources) {
  struct Range {
    std::uint64_t address, size;
  };

  std::array<Range, kMaxItineraryRanges> ranges{};
  std::size_t count = 0;
  std::uint64_t total = 0;
  while (!specification.empty()) {
    const auto comma = specification.find(',');
    const auto part = specification.substr(0, comma);
    const auto colon = part.find(':');
    Range range{};
    if (count == ranges.size() || colon == std::string_view::npos ||
        !Unsigned(part.substr(0, colon), range.address, 16) ||
        !Unsigned(part.substr(colon + 1), range.size, 10) || range.address % 4 || !range.size ||
        range.size % 4 || range.size > kMaxItineraryBytes - total ||
        range.size - 1 > UINT64_MAX - range.address) {
      return Error("invalid_range",
                   "Require up to 512 HEX:BYTES ranges, aligned and totaling at most 16384 bytes");
    }

    ranges[count++] = range;
    total += range.size;
    if (comma == std::string_view::npos) break;
    specification.remove_prefix(comma + 1);
    if (specification.empty()) return Error("invalid_range", "Empty itinerary range");
  }

  if (!count) return Error("invalid_range", "An itinerary needs at least one source range");
  nyx::Budget budget(resources);
  auto image = ReadImage(filename, &budget);
  if (!image) return 1;
  if (image->machine() != 183 || image->endianness() != nyx::format::Endianness::Little) {
    return Error("unsupported_target", "Path recovery requires little-endian AArch64 ELF");
  }

  if (budget.try_consume({total / 4, total / 4 * sizeof(nyx::ir::Group)}) !=
      nyx::BudgetDecline::none) {
    return Error("resource_limit", "Path source table exceeds budget");
  }

  std::vector<nyx::ir::Group> groups;
  groups.reserve(total / 4);
  for (std::size_t i = 0; i < count; ++i) {
    const auto [address, size] = ranges[i];
    const auto bytes = image->Read(address, size);
    if (!bytes)
      return Error("unmapped_range", "Path range lacks an unambiguous file-backed mapping");
    bool executable = false;
    for (const auto& segment : image->segments()) {
      if (segment.type == 1 && (segment.flags & 1) && address >= segment.address &&
          address - segment.address <= segment.file_size &&
          size <= segment.file_size - (address - segment.address)) {
        executable = true;
      }
    }

    if (!executable)
      return Error("nonexecutable_range", "Path range is not in an executable load segment");
    for (std::size_t offset = 0; offset < size; offset += 4) {
      const std::array<std::uint8_t, 4> word{(*bytes)[offset], (*bytes)[offset + 1],
                                             (*bytes)[offset + 2], (*bytes)[offset + 3]};
      auto decoded = nyx::a64::Decode(address + offset, word, budget, {profile});
      if (!decoded.group) {
        const auto reason = decoded.reason == nyx::a64::DecodeDecline::work_limit ||
                                    decoded.reason == nyx::a64::DecodeDecline::byte_limit
                                ? "resource_limit"
                                : "unsupported_region";
        return Error(reason, "Path decoding declined; no recovery artifact published");
      }

      groups.push_back(std::move(*decoded.group));
    }
  }

  auto normalized = nyx::ir::NormalizePath(groups, budget);
  if (!normalized.path)
    return Error(normalized.reason == nyx::ir::BlockDecline::resource_limit ? "resource_limit"
                                                                            : "unsupported_region",
                 "Invalid or unsupported itinerary; no recovery artifact published");
  auto forwarded = nyx::recovery::ForwardMemoryValues(*normalized.path, budget);
  if (!forwarded.path)
    return Error(forwarded.reason == nyx::recovery::MemoryRecoveryDecline::resource_limit
                     ? "resource_limit"
                     : "invalid_group",
                 "Memory forwarding declined; no partial journal published");
  auto simplified = nyx::recovery::SimplifyMba(*forwarded.path, budget);
  if (!simplified.path)
    return Error(simplified.reason == nyx::recovery::MbaDecline::resource_limit ? "resource_limit"
                                                                                : "invalid_group",
                 "Path arithmetic recovery declined; no partial journal published");
  // Nothing is declared for an itinerary, so only placement-independent folds apply.
  auto folded = nyx::recovery::FoldImageValues(*simplified.path, {}, budget);
  if (!folded.path)
    return Error(folded.reason == nyx::recovery::ImageDecline::resource_limit ? "resource_limit"
                                                                              : "invalid_group",
                 "Path image folding declined; no partial journal published");
  nyx::recovery::StoreCleanupResult cleaned;
  if (profile != nyx::a64::MemoryProfile::none) {
    const auto folded_revision = folded.path->revision();
    auto loads = nyx::recovery::OmitForwardedPairLoads(
        nyx::ir::RecoveredPath(std::move(*folded.path), {}, folded_revision), forwarded.facts,
        nyx::a64::kSp, budget);
    if (!loads.path)
      return Error(loads.reason == nyx::recovery::PairedLoadDecline::resource_limit
                       ? "resource_limit"
                       : "invalid_group",
                   "Paired-load cleanup declined; no partial artifact published");
    cleaned = nyx::recovery::OmitOverwrittenStores(std::move(*loads.path), nyx::a64::kSp, budget);
    if (!cleaned.path)
      return Error(cleaned.reason == nyx::recovery::StoreCleanupDecline::resource_limit
                       ? "resource_limit"
                       : "invalid_group",
                   "Stack-store cleanup declined; no partial artifact published");
  }

  const auto omitted_stores = cleaned.path ? cleaned.path->omissions().size() : 0;
  const auto omitted_pairs = cleaned.path ? cleaned.path->paired_load_omissions().size() : 0;
  const auto omitted = omitted_stores + 2 * omitted_pairs;
  auto before = nyx::ir::PrintJson(*normalized.path, budget);
  auto after =
      omitted ? nyx::ir::PrintJson(*cleaned.path, budget)
              : nyx::ir::PrintJson(cleaned.path ? cleaned.path->basis() : *folded.path, budget);
  if (!before.json || !after.json)
    return Error("resource_limit", "Path serialization declined; no partial artifact published");
  const auto envelope = std::string_view(filename).size() * 16 + 8192 +
                        forwarded.facts.size() * 1024 + forwarded.journal.size() * 512 +
                        (simplified.journal.size() + folded.journal.size()) * 768 + omitted * 512;
  if (budget.try_consume({envelope + before.json->size() + after.json->size(), envelope}) !=
      nyx::BudgetDecline::none) {
    return Error("resource_limit",
                 "Path publication exceeds budget; no partial artifact published");
  }

  // NormalizePath admits an interior call only when the itinerary continues at
  // that call's settled target, so every one listed here is a followed branch,
  // not an elided callee. These are proved structural facts, so they are reported
  // on their own and never folded into execution_assumptions.
  std::string call_edges = "\"itinerary_call_edges\":[";
  bool first_call_edge = true;
  for (std::size_t i = 0; i + 1 < groups.size(); ++i) {
    const auto& transfer = groups[i].transfer();
    if (!transfer || transfer->kind != nyx::ir::TransferKind::call) continue;
    if (!first_call_edge) call_edges += ",";
    first_call_edge = false;
    call_edges += "{\"boundary\":" + std::to_string(i) +
                  ",\"source\":" + std::to_string(groups[i].source_address()) +
                  ",\"followed\":" + std::to_string(groups[i + 1].source_address()) + "}";
  }

  call_edges += first_call_edge ? "],\"itinerary_scope\":\"intraprocedural\","
                                : "],\"itinerary_scope\":\"crosses_call_edges\",";
  call_edges += "\"callee_return\":\"not_modeled\",";

  const auto edits = forwarded.journal.size() + simplified.journal.size() + folded.journal.size() +
                     omitted_stores + omitted_pairs;
  std::cout << "{\"schema\":1,\"outcome\":\"" << (edits ? "simplified" : "unchanged")
            << "\",\"input\":" << JsonString(filename)
            << ",\"input_bytes_hex\":" << JsonString(HexBytes(filename))
            << ",\"scope\":\"selected_itinerary\",\"source_population\":\"itinerary_occurrences\","
               "\"source_groups\":"
            << groups.size() << ",\"source_bytes\":" << total
            << ",\"entry\":" << groups.front().source_address()
            << ",\"reachability\":\"not_proved\",\"interior_entry_analysis\":\"not_performed\","
               "\"off_itinerary_successors\":\"retained\",\"whole_function_recovery\":false,"
            << call_edges
            << "\"execution_assumptions\":[\"entry at first source\",\"immutable code\",\"BTI "
               "inactive\",\"GCS inactive\",\"pointer authentication inactive\""
            << (omitted
                    ? ",\"single-threaded normal memory without concurrent observers\",\"omitted "
                      "accesses mapped with required permissions\",\"no resource-limit outcome\""
                    : "")
            << "],\"memory_effect_scope\":\"" << (omitted ? "conditional_atomic_scalar" : "none")
            << "\",\"memory_profile\":\"" << MemoryProfileName(profile)
            << "\",\"memory_invariance\":\"not_assumed\",\"sp_alignment_check\":\"disabled\","
               "\"memory_fault_scope\":\"experimental_QEMU_reference\",\"architectural_fault_"
               "authority\":false,"
               "\"removed_source_groups\":0,\"removed_memory_effects\":"
            << omitted
            << ",\"machine_patching\":\"not_performed\","
               "\"forwarding_facts\":[";
  PrintRecoveryJournal(std::cout, forwarded, simplified, folded);
  std::cout << ",\"applied_edits\":" << edits << ",\"before\":" << *before.json
            << ",\"after\":" << *after.json << ",\"resources\":{\"work\":" << budget.used().work
            << ",\"bytes\":" << budget.used().bytes << "}}\n";
  return std::cout ? 0 : 1;
}

// The bytes a run may treat as fixed: segments the loader maps readable and
// never writable, and never sharing a page with a writable one, since the
// loader maps whole pages. Zero-fill past the file extent is left out. The
// run declares this restriction and publishes it with the artifact.
std::optional<std::vector<nyx::analysis::ConstantImageRange>> CollectSegmentConstants(
    const nyx::format::Image& image, nyx::Budget& budget) {
  // A segment aligned past 4 KiB was linked for larger pages, and a system
  // using them maps both segments into one page if they share a page of that
  // size. The loader validated each alignment as a power of two.
  const auto page_of = [](const nyx::format::Segment& segment) {
    return std::max<std::uint64_t>(4096, segment.alignment);
  };

  const auto down = [](std::uint64_t value, std::uint64_t page) { return value & ~(page - 1); };
  const auto up = [](std::uint64_t value, std::uint64_t page) {
    return value > UINT64_MAX - (page - 1) ? UINT64_MAX : (value + page - 1) & ~(page - 1);
  };

  const auto end = [](std::uint64_t address, std::uint64_t size) {
    return size > UINT64_MAX - address ? UINT64_MAX : address + size;
  };

  std::vector<nyx::analysis::ConstantImageRange> constants;
  for (const auto& segment : image.segments()) {
    // Readable and not writable: a segment without read permission supplies
    // no value, and one with write permission is excluded outright.
    if (segment.type != 1 || (segment.flags & 6) != 4 || segment.file_size == 0) continue;
    bool shares_page_with_writable = false;
    for (const auto& other : image.segments()) {
      if (other.type != 1 || (other.flags & 2) == 0 || other.memory_size == 0) continue;
      const auto page = std::max(page_of(segment), page_of(other));
      if (down(segment.address, page) < up(end(other.address, other.memory_size), page) &&
          down(other.address, page) < up(end(segment.address, segment.file_size), page)) {
        shares_page_with_writable = true;
      }
    }

    if (shares_page_with_writable) continue;
    const auto bytes = image.Read(segment.address, segment.file_size);
    if (!bytes) continue;
    if (budget.try_consume({1, sizeof(nyx::analysis::ConstantImageRange)}) !=
        nyx::BudgetDecline::none)
      return std::nullopt;
    constants.push_back({segment.address, *bytes, true});
  }

  return constants;
}

// A relocated slot is an image location the loader writes once. Whether it
// then stays written is a property of where it sits. A slot inside
// PT_GNU_RELRO is mapped read-only once relocation is done, so keeping its
// value rests on the loader honouring that, the same structural argument as a
// non-writable segment. A slot outside it is ordinary writable data that the
// program may assign whenever it likes, so its loader value is not carried
// forward: the slot still blocks reading file bytes, because the loader did
// write it, but it supplies no value. `all_stable` re-admits those as a
// declaration the caller owns.
std::optional<std::vector<nyx::analysis::RelocatedPointer>> CollectRelocatedSlots(
    const nyx::format::Image& image, bool all_stable, bool own_binding,
    const std::optional<std::vector<std::uint64_t>>& written, nyx::Budget& budget) {
  constexpr std::uint32_t kRelro = 0x6474e552;
  std::uint64_t relro_begin = 0, relro_end = 0;
  for (const auto& segment : image.segments()) {
    if (segment.type != kRelro || segment.memory_size == 0) continue;
    if (segment.address > std::numeric_limits<std::uint64_t>::max() - segment.memory_size)
      return std::nullopt;
    relro_begin = segment.address;
    relro_end = segment.address + segment.memory_size;
  }

  const auto relatives = image.LocatedRelocations(own_binding);
  if (budget.try_consume(
          {relatives.size(), relatives.size() * sizeof(nyx::analysis::RelocatedPointer)}) !=
      nyx::BudgetDecline::none)
    return std::nullopt;
  std::vector<nyx::analysis::RelocatedPointer> pointers;
  pointers.reserve(relatives.size());
  for (const auto& relative : relatives) {
    const bool protected_after_relocation = relative.address >= relro_begin &&
                                            relative.address < relro_end &&
                                            relro_end - relative.address >= 8;
    // A scan that examined the whole image and found no writer for this slot
    // is why its loader value may still be there.
    const bool unwritten =
        written && !std::binary_search(written->begin(), written->end(), relative.address);
    pointers.push_back(
        {relative.address, relative.target, all_stable || protected_after_relocation || unwritten});
  }

  return pointers;
}

struct ImageFactTables {
  std::vector<nyx::analysis::ConstantImageRange> constants;
};

std::optional<ImageFactTables> CollectImageFacts(const nyx::format::Image& image,
                                                 nyx::Budget& budget) {
  auto constants = CollectSegmentConstants(image, budget);
  if (!constants) return std::nullopt;
  return ImageFactTables{std::move(*constants)};
}

// Execute from a caller-chosen entry over the loader-initialized image and
// record what it writes. The point is code whose result later code depends on:
// a lazy initializer fills a table, and nothing in the image's static data says
// what ends up there. The caller names the entry; nothing here is specific
// to one image.
//
// The result covers one execution. Determinism has to be argued separately,
// so the record lists what it examined and what it could not, and is
// published as discovery input only.
// A record `nyx derive-memory --record` wrote: address and value in hex, one
// pair per line. The values came from one observed execution, so admitting
// them is a declaration the caller makes and the artifact prints.
struct DerivedMemory {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
  std::vector<nyx::ir::SsaCallResult> results;
  std::uint64_t entry = 0;

  // How the run ended, and whether its memory rested on the image's own
  // symbol bindings. A run that stopped partway wrote real bytes but had not
  // finished, so they are not what the routine leaves.
  std::string status;
  bool own_binding = false;
};

std::optional<DerivedMemory> ReadDerivedMemory(const char* path, std::string_view digest,
                                               nyx::Budget& budget) {
  auto text = ReadBounded(path, 1 << 20, budget);
  if (!text) return std::nullopt;
  DerivedMemory derived;
  std::uint64_t entry = 0;
  bool has_entry = false;
  std::size_t at = 0;
  bool first = true;
  bool bound = false;
  while (at < text->size()) {
    auto end = text->find('\n', at);
    if (end == std::string::npos) end = text->size();
    const std::string line(text->data() + at, end - at);
    at = end + 1;
    if (first) {
      if (line != "NYXMEM01") return std::nullopt;
      first = false;
      continue;
    }

    if (line.empty()) continue;
    if (line.rfind("image ", 0) == 0) {
      if (line.substr(6) != digest) return std::nullopt;
      bound = true;
      continue;
    }

    if (line.rfind("entry ", 0) == 0) {
      if (!Unsigned(line.substr(6).c_str(), entry, 16)) return std::nullopt;
      has_entry = true;
      continue;
    }

    if (line.rfind("status ", 0) == 0) {
      if (!derived.status.empty()) return std::nullopt;
      derived.status = line.substr(7);
      continue;
    }

    if (line == "binding own") {
      derived.own_binding = true;
      continue;
    }

    // A result belongs to the entry that was run, which is the callee a call
    // to that address reaches.
    if (line.rfind("return ", 0) == 0) {
      if (!has_entry) return std::nullopt;
      const auto gap = line.find(' ', 7);
      if (gap == std::string::npos) return std::nullopt;
      std::uint64_t storage = 0, value = 0;
      if (!Unsigned(line.substr(7, gap - 7).c_str(), storage, 10) ||
          !Unsigned(line.substr(gap + 1).c_str(), value, 16) ||
          storage > std::numeric_limits<nyx::ir::StorageId>::max())
        return std::nullopt;
      if (budget.try_consume({1, sizeof(nyx::ir::SsaCallResult)}) != nyx::BudgetDecline::none)
        return std::nullopt;
      derived.results.push_back({entry, static_cast<nyx::ir::StorageId>(storage), value});
      continue;
    }

    const auto space = line.find(' ');
    if (space == std::string_view::npos) return std::nullopt;
    std::uint64_t address = 0, value = 0;
    if (!Unsigned(line.substr(0, space).c_str(), address, 16) ||
        !Unsigned(line.substr(space + 1).c_str(), value, 16))
      return std::nullopt;
    if (budget.try_consume({1, sizeof(std::pair<std::uint64_t, std::uint64_t>)}) !=
        nyx::BudgetDecline::none)
      return std::nullopt;
    derived.writes.push_back({address, value});
  }

  if (!bound || derived.status.empty()) return std::nullopt;
  std::sort(derived.results.begin(), derived.results.end(),
            [](const nyx::ir::SsaCallResult& a, const nyx::ir::SsaCallResult& b) {
              return std::pair{a.target, a.storage} < std::pair{b.target, b.storage};
            });
  derived.results.erase(std::unique(derived.results.begin(), derived.results.end(),
                                    [](const auto& a, const auto& b) {
                                      return a.target == b.target && a.storage == b.storage;
                                    }),
                        derived.results.end());
  derived.entry = entry;
  return derived;
}

// Adds what a pass's graph was seen writing to the run's withdrawn bytes and
// says how many are new. A byte stays withdrawn: the declared range it came
// from is still listed, so it is not admitted again either.
std::size_t Withdraw(const Discovery& found, Declarations& declared) {
  std::size_t withdrawn = 0;
  for (const auto slot : found.refuted_slots) {
    if (std::find(declared.ssa_refuted_slots.begin(), declared.ssa_refuted_slots.end(), slot) !=
        declared.ssa_refuted_slots.end())
      continue;
    declared.ssa_refuted_slots.push_back(slot);
    ++withdrawn;
  }

  std::sort(declared.ssa_refuted_slots.begin(), declared.ssa_refuted_slots.end());
  for (const auto& refuted : found.refuted) {
    const auto covered = std::any_of(
        declared.ssa_refuted.begin(), declared.ssa_refuted.end(), [&](const SsaRefuted& known) {
          return known.address <= refuted.address &&
                 refuted.address + refuted.bytes <= known.address + known.bytes;
        });
    if (covered) continue;
    declared.ssa_refuted.push_back(refuted);
    ++withdrawn;
  }

  std::sort(declared.ssa_refuted.begin(), declared.ssa_refuted.end(),
            [](const SsaRefuted& a, const SsaRefuted& b) {
              return std::pair{a.address, a.bytes} < std::pair{b.address, b.bytes};
            });
  return withdrawn;
}

// A record `nyx image-writers --record` wrote: one slot address per line.
// Under --assume-image-data unwritten: each word this range forms an address
// for that sits in writable image data, holds what the file or the loader's
// zero fill gives it, and that no store the whole-image scan placed reaches,
// is admitted as if named by --assume-initialized. It is what the per-range
// declarations of the key slots said by hand; recovery still checks it
// against the stores in its own population. Returns how many it admitted.
std::size_t AdmitUnwrittenData(const nyx::format::Image& image,
                               std::span<const std::uint64_t> addressed, Declarations& declared) {
  constexpr std::size_t kMaxAdmitted = 512;
  constexpr std::uint64_t kWord = 8;
  std::size_t admitted = 0;
  const auto& reached = *declared.written_data;
  for (const auto location : addressed) {
    if (declared.derived_ranges == kMaxAdmitted) break;
    if (location > UINT64_MAX - kWord) continue;
    bool writable = false;
    for (const auto& segment : image.segments())
      writable |= segment.type == 1 && (segment.flags & 2) != 0 && location >= segment.address &&
                  location - segment.address + kWord <= segment.memory_size;
    if (!writable || image.LoaderMayWrite(location, kWord)) continue;

    // A store reaches at most sixteen bytes, so only one starting that far
    // before the word can reach into it.
    const auto low = location < 16 ? 0 : location - 16;
    const auto from = std::lower_bound(reached.begin(), reached.end(),
                                       std::pair<std::uint64_t, std::uint64_t>{low, 0});
    bool touched = false;
    for (auto at = from; at != reached.end() && at->first < location + kWord; ++at)
      touched |= at->first + at->second > location;
    if (touched || std::any_of(declared.ranges.begin(), declared.ranges.end(),
                               [&](const DeclaredRange& range) {
                                 return range.address < location + kWord &&
                                        location < range.address + range.size;
                               }))
      continue;
    declared.ranges.push_back({location, kWord});
    ++declared.derived_ranges;
    ++admitted;
  }

  return admitted;
}

std::optional<std::vector<std::uint64_t>> ReadWrittenSlots(
    const char* path, std::string_view digest, std::uint64_t& unexamined, std::uint64_t& slots,
    std::optional<ReachedBytes>& reached, nyx::Budget& budget) {
  auto text = ReadBounded(path, 16 << 20, budget);
  if (!text) return std::nullopt;
  std::vector<std::uint64_t> written;
  std::size_t at = 0;
  bool first = true;
  bool bound = false;
  bool counted = false;
  bool sized = false;
  while (at < text->size()) {
    auto end = text->find('\n', at);
    if (end == std::string::npos) end = text->size();
    const std::string line(text->data() + at, end - at);
    at = end + 1;
    if (first) {
      if (line != "NYXWRT01") return std::nullopt;
      first = false;
      continue;
    }

    if (line.empty()) continue;

    // A record names the image it examined, so one from another build cannot
    // quietly apply by raw address.
    if (line.rfind("image ", 0) == 0) {
      if (line.substr(6) != digest) return std::nullopt;
      bound = true;
      continue;
    }

    if (line.rfind("slots ", 0) == 0) {
      if (sized || !Unsigned(line.substr(6).c_str(), slots, 10)) return std::nullopt;
      sized = true;
      continue;
    }

    if (line.rfind("wrote-count ", 0) == 0) {
      std::uint64_t count = 0;
      if (reached || !Unsigned(line.substr(12).c_str(), count, 10)) return std::nullopt;
      reached.emplace();
      continue;
    }

    if (line.rfind("wrote ", 0) == 0) {
      const auto gap = line.find(' ', 6);
      std::uint64_t address = 0, bits = 0;
      if (!reached || gap == std::string::npos ||
          !Unsigned(line.substr(6, gap - 6).c_str(), address, 16) ||
          !Unsigned(line.substr(gap + 1).c_str(), bits, 10))
        return std::nullopt;
      if (budget.try_consume({1, sizeof(ReachedBytes::value_type)}) != nyx::BudgetDecline::none)
        return std::nullopt;
      // A write of no stated width reaches as far as the widest store can.
      reached->push_back({address, bits ? (bits + 7) / 8 : 16});
      continue;
    }

    if (line.rfind("unexamined ", 0) == 0) {
      if (!Unsigned(line.substr(11).c_str(), unexamined, 10)) return std::nullopt;
      counted = true;
      continue;
    }

    std::uint64_t address = 0;
    if (!Unsigned(line.c_str(), address, 16)) return std::nullopt;
    if (budget.try_consume({1, sizeof(std::uint64_t)}) != nyx::BudgetDecline::none)
      return std::nullopt;
    written.push_back(address);
  }

  if (first || !bound || !counted || !sized) return std::nullopt;
  if (reached) std::sort(reached->begin(), reached->end());
  std::sort(written.begin(), written.end());
  return written;
}

std::string Hex64(std::uint64_t value) {
  constexpr char digits[] = "0123456789abcdef";
  std::string text;
  for (int shift = 60; shift >= 0; shift -= 4) {
    const auto nibble = (value >> shift) & 0xf;
    if (!text.empty() || nibble || shift == 0) text.push_back(digits[nibble]);
  }

  return text;
}

// What executing one routine from the loader's image left behind. A run is an
// observation at the placement it ran at, never a proof, so whoever uses it
// owes the reader that declaration. A run that cannot be made at all reports
// `decline` rather than printing: a caller may attempt one speculatively.
struct DerivedRun {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
  std::vector<nyx::ir::SsaCallResult> results;
  std::uint64_t relocations_applied = 0, relocations_unplaced = 0;
  std::uint64_t steps = 0, stopped_at = 0, undecodable = 0, windows = 0;
  const char* status = "invalid_program";
  bool returned = false;
  const char* decline = nullptr;
  const char* detail = nullptr;
};

DerivedRun Derive(const nyx::format::Image& image, std::uint64_t entry, std::uint64_t stack,
                  std::uint64_t step_limit, nyx::a64::MemoryProfile profile, nyx::Budget& budget,
                  bool own_binding, std::uint64_t arguments = 0) {
  DerivedRun run;
  const auto decline = [&](const char* code, const char* text) -> DerivedRun& {
    run.decline = code;
    run.detail = text;
    return run;
  };

  // The loader's starting state: every load segment at its own address, file
  // bytes then zero to its memory size, with the permissions it asks for.
  std::vector<std::vector<std::uint8_t>> owned;
  std::vector<nyx::eval::RegionInput> regions;
  std::uint64_t mapped = 0;
  for (const auto& segment : image.segments()) {
    if (segment.type != 1 || segment.memory_size == 0) continue;
    const auto bytes = image.Read(segment.address, segment.file_size);
    if (!bytes) continue;
    if (segment.memory_size > (128ULL << 20) ||
        budget.try_consume({1, segment.memory_size}) != nyx::BudgetDecline::none) {
      return decline("resource_limit", "Image mapping exceeds aggregate budget");
    }

    std::vector<std::uint8_t> region(static_cast<std::size_t>(segment.memory_size), 0);
    std::copy(bytes->begin(), bytes->end(), region.begin());
    owned.push_back(std::move(region));
    regions.push_back(
        {segment.address, owned.back(), (segment.flags & 4) != 0, (segment.flags & 2) != 0});
    mapped += segment.memory_size;
  }

  // The loader does not hand the program its file bytes: it first writes every
  // relocation. Executing without that reads zeros where the image expects
  // pointers, so the run starts where the loader would leave it: each slot
  // whose image location the file states holds it, and every other slot the
  // loader writes (an import, say) holds a value the run does not know, which
  // stops a run that reads it instead of letting the file's bytes stand in.
  if (!image.loader_writes_known())
    return decline("unknown_relocations",
                   "The image's relocation tables could not all be followed");
  const auto relatives = image.LocatedRelocations(own_binding);
  const auto written = image.loader_written();
  if (budget.try_consume(
          {relatives.size() + written.size(), written.size() * 16 * sizeof(std::uint64_t)}) !=
      nyx::BudgetDecline::none) {
    return decline("resource_limit", "Applying relocations exceeds aggregate budget");
  }

  std::vector<std::vector<std::uint64_t>> unknown(owned.size());
  for (const auto slot : written) {
    const auto located = std::lower_bound(relatives.begin(), relatives.end(), slot,
                                          [](const nyx::format::RelativeRelocation& relative,
                                             std::uint64_t key) { return relative.address < key; });
    if (located != relatives.end() && located->address == slot) continue;

    // No AArch64 dynamic relocation writes more than sixteen bytes; a TLS
    // descriptor is the widest, so every byte it could reach is marked, short
    // of a slot whose value is known, which the loader writes itself.
    std::uint64_t reach = 16;
    if (located != relatives.end() && located->address - slot < reach)
      reach = located->address - slot;
    for (std::size_t i = 0; i < regions.size(); ++i) {
      const auto begin = regions[i].address;
      if (slot < begin || slot - begin >= owned[i].size()) continue;
      for (std::uint64_t byte = slot - begin; byte < owned[i].size() && byte < slot - begin + reach;
           ++byte)
        unknown[i].push_back(byte);
      break;
    }
  }

  for (std::size_t i = 0; i < regions.size(); ++i) {
    std::sort(unknown[i].begin(), unknown[i].end());
    unknown[i].erase(std::unique(unknown[i].begin(), unknown[i].end()), unknown[i].end());
    regions[i].unknown = unknown[i];
  }

  for (const auto& relative : relatives) {
    for (std::size_t i = 0; i < regions.size(); ++i) {
      const auto begin = regions[i].address;

      // Subtracting keeps the comparison in range; adding eight to an address
      // near the top of the space would wrap and admit anything.
      if (owned[i].size() < 8 || relative.address < begin ||
          relative.address - begin > owned[i].size() - 8)
        continue;
      const auto at = static_cast<std::size_t>(relative.address - begin);
      for (unsigned byte = 0; byte < 8; ++byte)
        owned[i][at + byte] = static_cast<std::uint8_t>(relative.target >> (byte * 8));
      ++run.relocations_applied;
      break;
    }
  }

  run.relocations_unplaced = relatives.size() - run.relocations_applied;
  constexpr std::uint64_t kStackBytes = 1 << 16;
  std::vector<std::uint8_t> stack_bytes(kStackBytes, 0);
  const auto stack_region = regions.size();
  regions.push_back({stack, stack_bytes, true, true});

  // A return to this address is how the run says it finished.
  const std::uint64_t sentinel = stack + kStackBytes + 0x1000;
  nyx::eval::MemoryLimits memory_limits;
  memory_limits.max_total_bytes = mapped + kStackBytes + (1 << 20);
  memory_limits.max_regions = static_cast<std::uint32_t>(regions.size() + 8);
  auto built = nyx::eval::Memory::Create(regions, budget, memory_limits);
  if (!built.memory) {
    return decline(built.status == nyx::eval::MemoryStatus::resource_limit ? "resource_limit"
                                                                           : "invalid_mapping",
                   "Image segments do not form a usable mapping");
  }

  std::vector<nyx::eval::MemoryRegion> before(built.memory->Regions().begin(),
                                              built.memory->Regions().end());

  const auto initial_state = [&](nyx::eval::State& into) {
    if (budget.try_consume({36, 36 * sizeof(nyx::eval::Cell)}) != nyx::BudgetDecline::none)
      return false;
    for (unsigned i = 0; i < 36; ++i) {
      // Whatever a caller would pass goes in the argument registers, X8
      // among them: it carries the address of an indirect result, so a run
      // that leaves it alone leaves the caller's value, not a result of its
      // own. Running with different values is how a result that does not
      // depend on them is told from one that does.
      const auto initial = i == 30   ? sentinel
                           : i == 31 ? stack + kStackBytes / 2
                           : i <= 8  ? arguments
                                     : 0;
      auto value = nyx::BitVector::from_u64(i < 32 ? 64 : 1, initial, 64, budget);
      if (!value) return false;
      into.cells.push_back({i, std::move(*value)});
    }

    return true;
  };

  nyx::eval::State state;
  if (!initial_state(state))
    return decline("resource_limit", "Initial state exceeds aggregate budget");

  // Code is decoded where the run actually goes: a window at the entry, then a
  // window wherever it stops for want of one. Nothing pre-decodes an image.
  constexpr std::uint64_t kWindow = 8192;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> windows;
  std::vector<nyx::ir::Group> groups;

  // True only when this call added code; a repeat means the run is stuck for
  // some other reason than a missing window, and looping again would not help.
  const auto cover = [&](std::uint64_t address) {
    const auto base = address & ~(kWindow - 1);

    // A window recorded short does not cover an address past its end, so the
    // same base may be worth reading again once more of it is mapped.
    for (const auto& window : windows)
      if (window.first == base && address - base < window.second) return false;
    // A window is as much as is mapped from here: a small image, or the end
    // of a segment, gives less than the full one.
    std::optional<std::span<const std::uint8_t>> bytes;
    std::uint64_t span = kWindow;
    for (; span >= 4; span /= 2) {
      bytes = image.Read(base, span);
      if (bytes) break;
    }

    if (!bytes) return false;
    windows.push_back({base, span});
    for (std::size_t offset = 0; offset + 4 <= bytes->size(); offset += 4) {
      std::array<std::uint8_t, 4> word{};
      for (std::size_t i = 0; i < 4; ++i) word[i] = (*bytes)[offset + i];
      auto decoded = nyx::a64::Decode(base + offset, word, budget, {profile});
      if (decoded.group)
        groups.push_back(std::move(*decoded.group));
      else
        ++run.undecodable;
    }

    return true;
  };

  if (!cover(entry)) return decline("unmapped_range", "The entry is not file-backed");

  nyx::eval::ProgramLimits limits;
  limits.max_steps = static_cast<std::uint32_t>(step_limit);
  limits.max_groups = 65536;
  const std::array<std::uint64_t, 1> exits{sentinel};
  nyx::eval::ProgramResult result{};
  constexpr unsigned kRounds = 16;
  for (unsigned round = 0; round < kRounds; ++round) {
    result = nyx::eval::Run(groups, state, *built.memory, entry, exits, budget, limits,
                            {std::uint64_t{0}});
    if (result.status != nyx::eval::ProgramStatus::unresolved) break;

    // Whatever is reported must be the run that produced the mapping being
    // diffed, so the last round keeps its result rather than starting over.
    if (round + 1 == kRounds) break;

    // Stopped for want of code at this address: decode there and continue.
    if (!cover(result.runtime_pc)) break;
    nyx::eval::State fresh;
    if (!initial_state(fresh))
      return decline("resource_limit", "Initial state exceeds aggregate budget");
    state = std::move(fresh);
    auto again = nyx::eval::Memory::Create(regions, budget, memory_limits);
    if (!again.memory)
      return decline("invalid_mapping", "Image segments do not form a usable mapping");
    built = std::move(again);
  }

  run.status = [&] {
    switch (result.status) {
      case nyx::eval::ProgramStatus::exit:
        return "returned";
      case nyx::eval::ProgramStatus::unresolved:
        return "no_code";
      case nyx::eval::ProgramStatus::step_limit:
        return "step_limit";
      case nyx::eval::ProgramStatus::fault:
        return "fault";
      case nyx::eval::ProgramStatus::resource_limit:
        return "resource_limit";
      case nyx::eval::ProgramStatus::invalid_state:
        return "invalid_state";
      case nyx::eval::ProgramStatus::unsupported:
        return "unsupported";
      default:
        return "invalid_program";
    }
  }();
  run.returned = result.status == nyx::eval::ProgramStatus::exit;
  run.steps = result.committed_steps;
  run.stopped_at = result.runtime_pc;
  run.windows = windows.size();

  // What the callee leaves in the registers a result is returned in. Only a
  // run that returned says anything about them.
  if (run.returned) {
    for (unsigned storage = 0; storage <= 8 && storage < state.cells.size(); ++storage)
      run.results.push_back(
          {entry, static_cast<nyx::ir::StorageId>(storage), state.cells[storage].value.word(0)});
  }

  // What the run changed, by comparing the mapping it started from.
  const auto after = built.memory->Regions();
  for (std::size_t i = 0; i < after.size() && i < before.size(); ++i) {
    // The stack this run was given is not part of the image.
    if (after[i].address == regions[stack_region].address) continue;
    const auto& was = before[i].bytes;
    const auto& now = after[i].bytes;
    for (std::size_t at = 0; at + 8 <= now.size() && at + 8 <= was.size(); at += 8) {
      if (std::equal(now.begin() + at, now.begin() + at + 8, was.begin() + at)) continue;
      std::uint64_t value = 0;
      for (unsigned byte = 0; byte < 8; ++byte)
        value |= std::uint64_t{now[at + byte]} << (byte * 8);
      if (budget.try_consume({1, 16}) != nyx::BudgetDecline::none)
        return decline("resource_limit", "Derived writes exceed aggregate budget");
      run.writes.push_back({after[i].address + at, value});
    }
  }

  return run;
}

int DeriveMemory(const char* path, std::uint64_t entry, std::uint64_t stack, std::uint64_t steps,
                 const char* record, nyx::a64::MemoryProfile profile, bool own_binding,
                 nyx::Resources resources) {
  nyx::Budget budget(resources);
  auto loaded = ReadImage(path, &budget);
  if (!loaded) return 1;
  const auto& image = *loaded;
  if (image.machine() != 183 || image.endianness() != nyx::format::Endianness::Little) {
    return Error("unsupported_target", "Deriving memory requires a little-endian AArch64 ELF");
  }

  const auto run = Derive(image, entry, stack, steps, profile, budget, own_binding);
  if (run.decline) return Error(run.decline, run.detail);
  if (budget.try_consume({image.bytes().size(), 64}) != nyx::BudgetDecline::none)
    return Error("resource_limit", "Image identity exceeds aggregate budget");
  nyx::Sha256 derived_hash;
  derived_hash.Update(image.bytes());
  std::string lines = "NYXMEM01\nimage " + derived_hash.FinishHex() + "\nentry " + Hex64(entry) +
                      "\nstatus " + run.status + "\n" + (own_binding ? "binding own\n" : "");
  for (const auto& result : run.results)
    lines += "return " + std::to_string(result.storage) + " " + Hex64(result.value) + "\n";
  std::cout << "{\"schema\":1,\"outcome\":\"derived_memory\",\"input\":" << JsonString(path)
            << ",\"entry\":" << entry << ",\"relocations_applied\":" << run.relocations_applied
            << ",\"relocations_outside_the_mapping\":" << run.relocations_unplaced
            << ",\"status\":\"" << run.status << "\",\"steps\":" << run.steps
            << ",\"decoded_windows\":" << run.windows
            << ",\"undecodable_instructions\":" << run.undecodable
            << ",\"stopped_at\":" << run.stopped_at << ",\"writes\":[";
  for (std::size_t i = 0; i < run.writes.size(); ++i) {
    std::cout << (i ? "," : "") << "{\"address\":" << run.writes[i].first
              << ",\"value\":" << run.writes[i].second << '}';
    lines += Hex64(run.writes[i].first) + " " + Hex64(run.writes[i].second) + "\n";
  }

  std::cout << "],\"record\":\"" << (record ? record : "")
            << "\",\"write_count\":" << run.writes.size()
            << ",\"closed\":" << ((run.returned && run.undecodable == 0) ? "true" : "false")
            << ",\"rests_on\":[\"the run started from the loader's image with zeroed registers\","
               "\"one thread ran it, so every atomic took its uncontended path\","
               "\"a read of memory this mapping never wrote returns the loader's bytes, except a"
               " relocated slot whose value the image does not state, which stops the run\""
            << (own_binding ? ",\"every symbol the image defines is bound to its own definition\""
                            : "")
            << "]"
               ",\"authorizes_transformation\":false}\n";
  if (record) {
    std::ofstream out(record, std::ios::binary);
    out << lines;
    if (!out) return Error("write_failed", "The derived memory record could not be written");
  }

  return 0;
}

// Which of the image's own loader-initialized data its own code overwrites.
// A recovery run declares those bytes stable but only ever checks the range it
// was given, so a writer in another function is invisible to it. This reads
// every executable byte instead, and can only withdraw declarations.
int ImageWriters(const char* path, const char* record, nyx::a64::MemoryProfile profile,
                 nyx::Resources resources) {
  nyx::Budget budget(resources);
  auto loaded = ReadImage(path, &budget);
  if (!loaded) return 1;
  const auto& image = *loaded;
  if (image.machine() != 183 || image.endianness() != nyx::format::Endianness::Little) {
    return Error("unsupported_target",
                 "Image writer scanning requires a little-endian AArch64 ELF");
  }

  auto constants = image.declares_text_relocations()
                       ? std::optional(std::vector<nyx::analysis::ConstantImageRange>{})
                       : CollectSegmentConstants(image, budget);
  // Examining a slot for writers can only withdraw it, so every slot with a
  // stated location is examined, interposable or not.
  auto pointers = CollectRelocatedSlots(image, true, true, std::nullopt, budget);
  if (!constants || !pointers)
    return Error("resource_limit", "Image fact tables exceed aggregate budget");
  const auto declared_constants = constants->size();
  const auto declared_slots = pointers->size();
  if (budget.try_consume({image.bytes().size(), 64}) != nyx::BudgetDecline::none)
    return Error("resource_limit", "Image identity exceeds aggregate budget");
  nyx::Sha256 writers_hash;
  writers_hash.Update(image.bytes());
  const auto digest = writers_hash.FinishHex();

  // One window at a time: materializing every decoded instruction of a large
  // image at once costs more memory than the scan itself. A run split at a
  // window edge only loses resolution, which withdraws fewer declarations.
  constexpr std::size_t kWindow = 4096;
  std::vector<std::size_t> refuted_constants, refuted_slots;

  // A handful of refutations with the write that made them, so a reader can
  // check each refutation against the image.
  struct Witness {
    std::uint64_t fact;
    std::uint64_t write;
    unsigned width;
    std::uint64_t from;
  };

  std::vector<Witness> witnesses;
  std::uint64_t scanned = 0, unmodeled = 0, unresolved = 0, placed = 0;
  unsigned passes = 0;

  // Every image byte a placed store reaches, across all passes: a later pass
  // with fewer facts can place fewer, never un-write what an earlier found.
  std::vector<std::pair<std::uint64_t, unsigned>> reached;
  for (unsigned pass = 0; pass < 4; ++pass) {
    ++passes;
    const nyx::analysis::ImageFacts facts{*constants, *pointers, true, {}};
    std::vector<nyx::ir::ImageWrite> writes;
    std::vector<std::uint64_t> sites;
    scanned = unmodeled = unresolved = placed = 0;
    for (const auto& segment : image.segments()) {
      if (segment.type != 1 || (segment.flags & 1) == 0 || segment.file_size < 4) continue;
      const auto bytes = image.Read(segment.address, segment.file_size / 4 * 4);
      if (!bytes) continue;
      for (std::size_t base = 0; base < bytes->size(); base += kWindow * 4) {
        const auto count = std::min(kWindow, (bytes->size() - base) / 4);
        std::vector<nyx::analysis::SourceRecord> records;
        if (budget.try_consume({count, count * sizeof(nyx::analysis::SourceRecord)}) !=
            nyx::BudgetDecline::none)
          return Error("resource_limit", "Image writer scan exceeds aggregate budget");
        records.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
          const auto offset = base + i * 4;
          std::array<std::uint8_t, 4> word{};
          for (std::size_t b = 0; b < 4; ++b) word[b] = (*bytes)[offset + b];
          auto decoded = nyx::a64::Decode(segment.address + offset, word, budget, {profile});
          if (!decoded.group && decoded.reason != nyx::a64::DecodeDecline::unsupported &&
              decoded.reason != nyx::a64::DecodeDecline::invalid_encoding)
            return Error("resource_limit", "Image writer decode exceeds aggregate budget");
          records.push_back({segment.address + offset,
                             std::vector<std::uint8_t>(word.begin(), word.end()),
                             std::move(decoded.group), nyx::analysis::OpaqueReason::none,
                             nyx::analysis::OpaqueControl::unknown});
        }

        auto found = nyx::analysis::ScanImageWrites(records, facts, budget);
        if (!found) return Error("resource_limit", "Image writer scan exceeds aggregate budget");
        scanned += found->scanned_groups;
        unmodeled += found->unmodeled_groups;
        unresolved += found->unresolved_writes;
        placed += found->writes.size();
        if (budget.try_consume(
                {found->writes.size(), found->writes.size() * sizeof(nyx::ir::ImageWrite)}) !=
            nyx::BudgetDecline::none)
          return Error("resource_limit", "Image writer scan exceeds aggregate budget");
        writes.insert(writes.end(), found->writes.begin(), found->writes.end());
        sites.insert(sites.end(), found->write_instructions.begin(),
                     found->write_instructions.end());
        for (const auto& write : found->writes) reached.push_back({write.address, write.width});
      }
    }

    const auto refutations = nyx::ir::RefuteImageFacts(facts, writes, budget);
    if (!refutations)
      return Error("resource_limit", "Image fact refutation exceeds aggregate budget");
    if (refutations->constants.empty() && refutations->pointers.empty()) break;

    // Withdrawing a fact can hide a store that read it, so the scan repeats on
    // the reduced set and what it withdrew is kept.
    std::vector<std::size_t> constant_indices, slot_indices;
    for (const auto& conflict : refutations->constants) constant_indices.push_back(conflict.index);
    for (const auto& conflict : refutations->pointers) slot_indices.push_back(conflict.index);

    // The write a refutation names is one of possibly several; report the
    // first instruction found making it, so the claim can be looked up.
    const auto site = [&](const nyx::ir::ImageWrite& write) -> std::uint64_t {
      for (std::size_t i = 0; i < writes.size() && i < sites.size(); ++i)
        if (writes[i].address == write.address && writes[i].width == write.width) return sites[i];
      return 0;
    };

    for (const auto& conflict : refutations->constants) {
      refuted_constants.push_back((*constants)[conflict.index].address);
      if (witnesses.size() < 8192)
        witnesses.push_back({(*constants)[conflict.index].address, conflict.write.address,
                             conflict.write.width, site(conflict.write)});
    }

    for (const auto& conflict : refutations->pointers) {
      refuted_slots.push_back((*pointers)[conflict.index].address);
      if (witnesses.size() < 8192)
        witnesses.push_back({(*pointers)[conflict.index].address, conflict.write.address,
                             conflict.write.width, site(conflict.write)});
    }

    std::sort(constant_indices.begin(), constant_indices.end());
    std::sort(slot_indices.begin(), slot_indices.end());
    auto retained = nyx::ir::RetainImageFacts({*constants, *pointers, true}, constant_indices,
                                              slot_indices, budget);
    if (!retained) return Error("resource_limit", "Image fact retention exceeds aggregate budget");
    *constants = std::move(retained->constants);
    *pointers = std::move(retained->pointers);
  }

  std::sort(refuted_constants.begin(), refuted_constants.end());
  std::sort(refuted_slots.begin(), refuted_slots.end());
  std::cout << "{\"schema\":1,\"outcome\":\"image_writers\",\"input\":" << JsonString(path)
            << ",\"scope\":\"every executable load segment of the image\""
            << ",\"declared_constant_ranges\":" << declared_constants
            << ",\"declared_relocated_slots\":" << declared_slots
            << ",\"scanned_groups\":" << scanned << ",\"unmodeled_groups\":" << unmodeled
            << ",\"placed_image_writes\":" << placed
            << ",\"unresolved_image_writes\":" << unresolved << ",\"passes\":" << passes
            << ",\"refuted_constant_ranges\":[";
  for (std::size_t i = 0; i < refuted_constants.size(); ++i)
    std::cout << (i ? "," : "") << refuted_constants[i];
  std::cout << "],\"refuted_relocated_slots\":[";
  for (std::size_t i = 0; i < refuted_slots.size(); ++i)
    std::cout << (i ? "," : "") << refuted_slots[i];
  std::cout << "],\"refutation_witnesses\":[";
  for (std::size_t i = 0; i < witnesses.size(); ++i)
    std::cout << (i ? "," : "") << "{\"fact\":" << witnesses[i].fact
              << ",\"written_at\":" << witnesses[i].write << ",\"width\":" << witnesses[i].width
              << ",\"written_by\":" << witnesses[i].from << '}';
  std::cout << "],\"findings\":[";
  bool first = true;
  const auto finding = [&](const char* text) {
    std::cout << (first ? "" : ",") << JsonString(text);
    first = false;
  };

  if (!refuted_constants.empty() || !refuted_slots.empty())
    finding("the image's own code overwrites data a recovery run would treat as loader-stable");
  if (unmodeled != 0)
    finding(
        "instructions without modeled semantics were not examined, so their effects are unknown");
  if (unresolved != 0)
    finding(
        "stores through an image address the abstraction could not place may reach any declared "
        "location");
  std::cout << "],\"discharges_declaration\":false}\n";
  if (record) {
    // The record names its image and how much of it went unexamined, so a
    // reader admitting the slots it omits knows what that rests on.
    // It also says how many slots it examined: a slot it did not examine is
    // omitted for that reason and not because nothing writes it.
    std::string lines = "NYXWRT01\nimage " + digest + "\nslots " + std::to_string(declared_slots) +
                        "\nunexamined " + std::to_string(unmodeled + unresolved) + "\n";
    for (const auto address : refuted_slots) lines += Hex64(address) + "\n";

    // And every byte range a placed store reaches, so a reader can admit the
    // data nothing was found writing, not only the relocated slots.
    std::sort(reached.begin(), reached.end());
    reached.erase(std::unique(reached.begin(), reached.end()), reached.end());
    lines += "wrote-count " + std::to_string(reached.size()) + "\n";
    for (const auto& [address, width] : reached)
      lines += "wrote " + Hex64(address) + " " + std::to_string(width) + "\n";
    std::ofstream out(record, std::ios::binary);
    out << lines;
    if (!out) return Error("write_failed", "The image writer record could not be written");
  }

  return 0;
}

int Lift(const char* path, std::uint64_t address, std::uint64_t size, bool simplify = false,
         nyx::a64::MemoryProfile profile = nyx::a64::MemoryProfile::none,
         GraphMode mode = GraphMode::none,
         nyx::Resources resources = {100000000, 256ULL * 1024 * 1024},
         const Declarations& declared = {}, const SsaRequest& ssa_request = {},
         Discovery* discover = nullptr) {
  const bool cfg = mode != GraphMode::none;
  const bool recovery_facts = mode == GraphMode::probe || mode == GraphMode::recover;
  if ((address % 4) != 0 || size == 0 || (size % 4) != 0 || size > 4 * 1024 * 1024 ||
      address > std::numeric_limits<std::uint64_t>::max() - size) {
    return Error("invalid_range",
                 "Require aligned address, positive aligned size <= 4 MiB and no wrap");
  }

  nyx::Budget budget(resources);
  auto loaded = ReadImage(path, &budget);
  if (!loaded) return 1;
  const auto& image = *loaded;
  if (image.machine() != 183 || image.endianness() != nyx::format::Endianness::Little) {
    return Error("unsupported_target", "Lift currently requires little-endian AArch64 ELF");
  }

  const auto source = image.Read(address, size);
  if (!source)
    return Error("unmapped_range", "Requested bytes lack an unambiguous file-backed mapping");
  bool executable = false;
  for (const auto& segment : image.segments()) {
    if (segment.type == 1 && (segment.flags & 1) != 0 && address >= segment.address &&
        address - segment.address <= segment.file_size &&
        size <= segment.file_size - (address - segment.address))
      executable = true;
  }

  if (!executable)
    return Error("nonexecutable_range", "Requested range is not in an executable load segment");
  if (budget.try_consume({size / 4, (size / 4) * (cfg        ? sizeof(nyx::analysis::SourceRecord)
                                                  : simplify ? sizeof(nyx::ir::Group)
                                                             : sizeof(std::string))}) !=
      nyx::BudgetDecline::none) {
    return Error("resource_limit", "Lift output table exceeds aggregate budget");
  }

  std::vector<std::string> groups;
  std::vector<nyx::ir::Group> raw;
  std::vector<UnmodeledWord> skipped;
  std::vector<nyx::analysis::SourceRecord> records;
  if (cfg)
    records.reserve(static_cast<std::size_t>(size / 4));
  else if (simplify)
    raw.reserve(static_cast<std::size_t>(size / 4));
  else
    groups.reserve(static_cast<std::size_t>(size / 4));
  std::uint64_t supported = 0;
  std::uint64_t unmodeled = 0;
  for (std::size_t offset = 0; offset < source->size(); offset += 4) {
    std::array<std::uint8_t, 4> bytes{};
    for (std::size_t i = 0; i < 4; ++i) bytes[i] = (*source)[offset + i];
    auto decoded = nyx::a64::Decode(address + offset, bytes, budget, {profile});
    if (cfg) {
      if (!decoded.group && decoded.reason != nyx::a64::DecodeDecline::unsupported &&
          decoded.reason != nyx::a64::DecodeDecline::invalid_encoding) {
        return Error("resource_limit", "CFG lifting exhausted budget; no partial graph published");
      }

      if (budget.try_consume({1, 4}) != nyx::BudgetDecline::none)
        return Error("resource_limit", "CFG source bytes exceed budget");
      const auto reason = decoded.group ? nyx::analysis::OpaqueReason::none
                          : decoded.reason == nyx::a64::DecodeDecline::unsupported
                              ? nyx::analysis::OpaqueReason::unsupported
                              : nyx::analysis::OpaqueReason::invalid_encoding;
      auto control = nyx::analysis::OpaqueControl::unknown;
      if (reason == nyx::analysis::OpaqueReason::unsupported) {
        if (nyx::a64::OpaqueNormalFallthrough(bytes))
          control = nyx::analysis::OpaqueControl::normal_fallthrough;
        else if (nyx::a64::OpaqueTrap(bytes))
          control = nyx::analysis::OpaqueControl::trap;
      }

      records.push_back({address + offset, std::vector<std::uint8_t>(bytes.begin(), bytes.end()),
                         std::move(decoded.group), reason, control});
      continue;
    }

    if (decoded.group) {
      if (simplify) {
        raw.push_back(std::move(*decoded.group));
        continue;
      }

      auto printed = nyx::ir::PrintJson(*decoded.group, budget);
      if (!printed.json) {
        return Error(printed.reason == nyx::ir::PrintDecline::invalid_group ? "invalid_group"
                                                                            : "resource_limit",
                     "Lift serialization declined; no partial artifact published");
      }

      groups.push_back(std::move(*printed.json));
      ++supported;
    } else {
      if (simplify) {
        if (decoded.reason != nyx::a64::DecodeDecline::unsupported &&
            decoded.reason != nyx::a64::DecodeDecline::invalid_encoding) {
          return Error(DecodeReason(decoded.reason),
                       "Recovery budget exhausted; no recovery applied");
        }

        // The word's effects are unknown, so it joins no run: it ends the run
        // before it and the next run starts after it. Nothing is folded across
        // it, and it is published so the caller sees what was left out.
        if (budget.try_consume({1, sizeof(UnmodeledWord)}) != nyx::BudgetDecline::none) {
          return Error("resource_limit", "Unmodeled word table exceeds aggregate budget");
        }

        skipped.push_back({address + offset, bytes, DecodeReason(decoded.reason)});
        continue;
      }

      if (decoded.reason != nyx::a64::DecodeDecline::unsupported &&
          decoded.reason != nyx::a64::DecodeDecline::invalid_encoding) {
        return Error("resource_limit", "Lift budget exhausted; no partial artifact published");
      }

      char record[256];
      const auto length = std::snprintf(
          record, sizeof(record),
          "{\"schema\":1,\"kind\":\"nyx.ir.unmodeled\",\"source_address\":%llu,"
          "\"bytes\":\"%02x%02x%02x%02x\",\"reason\":\"%s\"}",
          static_cast<unsigned long long>(address + offset), static_cast<unsigned>(bytes[0]),
          static_cast<unsigned>(bytes[1]), static_cast<unsigned>(bytes[2]),
          static_cast<unsigned>(bytes[3]), DecodeReason(decoded.reason));
      if (length < 0 || static_cast<std::size_t>(length) >= sizeof(record)) {
        return Error("invalid_group", "Unmodeled source record exceeds its checked buffer");
      }

      if (budget.try_consume(
              {static_cast<std::uint64_t>(length), static_cast<std::uint64_t>(length) + 1}) !=
          nyx::BudgetDecline::none) {
        return Error("resource_limit",
                     "Lift output exceeds aggregate budget; no partial artifact published");
      }

      groups.emplace_back(record, static_cast<std::size_t>(length));
      ++unmodeled;
    }
  }

  // Treating these bytes as fixed is an assumption this run declares: a
  // mapping can be made writable later or aliased by a writable one,
  // and callee effects are not modeled. The absence of text relocations rules
  // out one channel; the rest are assumed away and published as assumptions.
  std::vector<nyx::analysis::ConstantImageRange> constants;

  // An image that relocates its own text has its file bytes disowned wholesale;
  // a declaration the run would then ignore is refused rather than dropped.
  if (!declared.ranges.empty() && image.declares_text_relocations()) {
    return Error("invalid_range",
                 "Declared ranges are not admitted for an image that relocates its text");
  }

  // Bytes the loader supplies have no file to live in, and the fact table
  // only points at them.
  std::vector<std::vector<std::uint8_t>> zero_filled;
  if (recovery_facts && !image.declares_text_relocations()) {
    auto collected = CollectImageFacts(image, budget);
    if (!collected) return Error("resource_limit", "Constant image table exceeds aggregate budget");
    constants = std::move(collected->constants);
    for (const auto& range : declared.ranges) {
      // The claim is about the value the run starts from, and for a segment
      // whose memory reaches past its file the loader supplies that: the
      // bytes between the two are zero, which is the format saying so rather
      // than this run assuming it. Bytes a relocation overwrites are still
      // not what the declaration names.
      auto bytes = image.Read(range.address, range.size);
      if (!bytes) {
        const auto zeroed =
            std::any_of(image.segments().begin(), image.segments().end(), [&](const auto& segment) {
              return segment.type == 1 && segment.memory_size > segment.file_size &&
                     range.address >= segment.address + segment.file_size &&
                     range.size <= segment.memory_size - (range.address - segment.address);
            });
        if (!zeroed)
          return Error(
              "invalid_range",
              "A declared constant range is neither file-backed nor zero-filled by the loader");
        if (budget.try_consume({range.size, range.size}) != nyx::BudgetDecline::none)
          return Error("resource_limit", "Constant image table exceeds aggregate budget");
        zero_filled.push_back(std::vector<std::uint8_t>(range.size, 0));
        bytes = std::span<const std::uint8_t>(zero_filled.back());
      }

      if (image.LoaderMayWrite(range.address, range.size)) {
        return Error("invalid_range", "The loader may write a declared constant range");
      }

      if (budget.try_consume({1, sizeof(nyx::analysis::ConstantImageRange)}) !=
          nyx::BudgetDecline::none) {
        return Error("resource_limit", "Constant image table exceeds aggregate budget");
      }

      constants.push_back({range.address, *bytes});
    }
  }

  // A relocated slot is an image location the loader writes once. The slots sit
  // in writable memory, so this is the same declared assumption as the bytes
  // above and is published alongside them.
  std::vector<nyx::analysis::RelocatedPointer> pointers;
  if (recovery_facts) {
    auto slots = CollectRelocatedSlots(image, declared.relocations_stable, declared.symbols_own,
                                       declared.written_slots, budget);
    if (!slots) return Error("resource_limit", "Relocated pointer table exceeds aggregate budget");
    pointers = std::move(*slots);

    // A derived value replaces the loader's in the slot it was seen written
    // to, but only when it names somewhere in this image: a slot may hold an
    // ordinary number, and calling that an image location would invent one.
    for (const auto& [address, value] : declared.derived) {
      const auto at = std::lower_bound(pointers.begin(), pointers.end(), address,
                                       [](const nyx::analysis::RelocatedPointer& slot,
                                          std::uint64_t key) { return slot.address < key; });
      if (at == pointers.end() || at->address != address) continue;

      // Seeing the slot written is itself decisive: whatever it now holds, it
      // is not what the loader put there, so the loader's value never stands.
      // Offering the observed value as a location needs more than that it is
      // file-backed, because an image whose first segment sits at zero makes
      // every small number look like one.
      if (value >= 0x1000 && image.Read(value, 1)) {
        at->target = value;
        at->value_stable = true;
      } else {
        at->value_stable = false;
      }
    }

    // A slot an earlier pass's graph was seen writing no longer holds the
    // loader's value, the same withdrawal as the writer scan makes.
    for (const auto address : declared.ssa_refuted_slots) {
      const auto at = std::lower_bound(pointers.begin(), pointers.end(), address,
                                       [](const nyx::analysis::RelocatedPointer& slot,
                                          std::uint64_t key) { return slot.address < key; });
      if (at != pointers.end() && at->address == address) at->value_stable = false;
    }

    // A slot two runs saw written differently is written, and that is all.
    for (const auto address : declared.derived_unstable) {
      const auto at = std::lower_bound(pointers.begin(), pointers.end(), address,
                                       [](const nyx::analysis::RelocatedPointer& slot,
                                          std::uint64_t key) { return slot.address < key; });
      if (at != pointers.end() && at->address == address) at->value_stable = false;
    }
  }

  SsaRequest request = ssa_request;
  if (mode == GraphMode::recover) {
    if (budget.try_consume({image.bytes().size(), 64}) != nyx::BudgetDecline::none)
      return Error("resource_limit", "Image identity exceeds aggregate budget");
    nyx::Sha256 hash;
    hash.Update(image.bytes());
    request.image_digest = hash.FinishHex();
  }

  if (cfg)
    return Graph(path, address, size, records, budget, profile, mode,
                 {constants, pointers, recovery_facts, declared.load_bias}, declared, request,
                 discover, &image);
  if (simplify) return Simplify(path, address, size, raw, budget, profile, skipped);

  // Path escaping and the fixed envelope are bounded separately from group JSON.
  const auto envelope = std::string_view(path).size() * 16 + 4096;
  std::uint64_t output_work = envelope + groups.size();
  for (const auto& group : groups) output_work += group.size();
  if (budget.try_consume({output_work, envelope}) != nyx::BudgetDecline::none) {
    return Error("resource_limit",
                 "Lift publication exceeds aggregate budget; no partial artifact published");
  }

  // Nothing is published until every requested word has a complete source record.
  std::cout << "{\"schema\":1,\"outcome\":\"" << (unmodeled == 0 ? "lifted" : "partial")
            << "\",\"input\":" << JsonString(path)
            << ",\"input_bytes_hex\":" << JsonString(HexBytes(path)) << ",\"address\":" << address
            << ",\"size\":" << size << ",\"source_groups\":" << groups.size()
            << ",\"supported_groups\":" << supported << ",\"unmodeled_groups\":" << unmodeled
            << ",\"supported_source_bytes\":" << supported * 4
            << ",\"unmodeled_source_bytes\":" << unmodeled * 4
            << ",\"scope\":\"register_and_control_semantics\",\"classification\":\"user_requested_"
               "range\","
               "\"callee_effects\":\"not_modeled\",\"control_protection\":\"not_modeled\","
               "\"execution_assumptions\":[\"BTI inactive\",\"GCS inactive\",\"pointer "
               "authentication inactive\"],"
               "\"deobfuscation\":\"not_performed\",\"groups\":[";
  bool first = true;
  for (const auto& group : groups) {
    if (!first) std::cout << ',';
    first = false;
    std::cout << group;
  }

  std::cout << "],\"resources\":{\"work\":" << budget.used().work
            << ",\"bytes\":" << budget.used().bytes << "}}\n";
  if (!std::cout) return 1;
  return unmodeled == 0 ? 0 : 2;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "help")) {
    std::cout
        << "Usage: nyx <command> <ELF> [options]\n"
           "\n"
           "Commands:\n"
           "  inspect <ELF>                          Print the ELF segment layout\n"
           "  lift <ELF> --address HEX --size N      Lift a range to IL\n"
           "  simplify <ELF> --address HEX --size N  Simplify straight-line code in a range\n"
           "  cfg <ELF> --address HEX --size N       Build a control-flow graph for a range\n"
           "  simplify-path <ELF> --ranges HEX:N,... Simplify one path through the given ranges\n"
           "  recover-regions <ELF> --address HEX --size N\n"
           "                                         Simplify guarded regions of a range\n"
           "  cfg-probe <ELF> --address HEX --size N Show the graph and regions recover-regions "
           "would use\n"
           "  image-writers <ELF>                    List image data the binary's own code "
           "overwrites\n"
           "  derive-memory <ELF> --entry HEX        Run from an entry and report the memory it "
           "writes\n"
           "  version                                Print the version\n"
           "\n"
           "Common options:\n"
           "  --memory-profile PROFILE   Allow memory instructions: concrete_atomic_scalar or\n"
           "                             concrete_exclusive_scalar\n"
           "  --work-budget N            Work limit (default 100000000)\n"
           "  --byte-budget N            Allocation limit in bytes (default 268435456)\n"
           "\n"
           "recover-regions assumptions (each one is echoed in the output):\n"
           "  --assume-initialized HEX:N,...    Writable ranges keep their load-time value\n"
           "  --assume-noreturn HEX,...         Calls to these addresses never return\n"
           "  --assume-abi aapcs64              Calls and returns follow the AArch64 ABI\n"
           "  --assume-returns leave            Returns leave the analyzed code\n"
           "  --assume-traps stop               Traps never resume inside the analyzed code\n"
           "  --assume-faults terminal          A fault ends execution\n"
           "  --assume-image-access readable    Every folded image byte is readable\n"
           "  --assume-set aapcs64-open-region-v1\n"
           "                                    Shorthand for the five assumptions above\n"
           "  --assume-relocations stable       Relocated slots keep the loader's value\n"
           "  --assume-symbol-binding own       Symbols bind to their own definitions\n"
           "  --assume-load-bias HEX            Load address of the image\n"
           "  --assume-private-frame BEGIN:END  This stack range is private to the function\n"
           "  --assume-frame-reach none         Callees never touch the private frame\n"
           "  --assume-entries closed           The listed entries are the only entry points\n"
           "\n"
           "recover-regions inputs and output:\n"
           "  --derived-memory FILE      Use a derive-memory record\n"
           "  --derive-entry HEX|auto    Run derive-memory from this entry first (repeatable)\n"
           "  --image-writers FILE       Use an image-writers record\n"
           "  --passes NAME,...          Run these passes instead of the default pipeline\n"
           "  --rounds N                 Pipeline rounds\n"
           "  --state-file FILE          Request a QEMU check of the result\n"
           "  --stubs-file FILE          Stubs for calls during that check\n"
           "  --check-file FILE          Accept or roll back using a verifier record\n"
           "  --listing text             Add a readable listing of the result\n";
    return 0;
  }

  if (argc == 2 && std::string_view(argv[1]) == "version") {
    std::cout << "{\"schema\":1,\"outcome\":\"version\",\"version\":\"0.1.0\"}\n";
    return 0;
  }

  if (argc == 3 && std::string_view(argv[1]) == "inspect") return Inspect(argv[2]);
  if (argc >= 5 && argc <= 19 && argc % 2 == 1 && std::string_view(argv[1]) == "derive-memory") {
    auto profile = nyx::a64::MemoryProfile::concrete_exclusive_scalar;
    std::optional<std::uint64_t> entry, stack, steps, work, bytes;
    const char* record = nullptr;
    bool own_binding = false;
    for (int i = 3; i < argc; i += 2) {
      const std::string_view option(argv[i]);
      std::uint64_t value = 0;
      if (option == "--entry" && !entry && Unsigned(argv[i + 1], value, 16))
        entry = value;
      else if (option == "--stack" && !stack && Unsigned(argv[i + 1], value, 16))
        stack = value;
      else if (option == "--steps" && !steps && Unsigned(argv[i + 1], value, 10))
        steps = value;
      else if (option == "--record" && !record)
        record = argv[i + 1];
      else if (option == "--work-budget" && !work && Unsigned(argv[i + 1], value, 10))
        work = value;
      else if (option == "--byte-budget" && !bytes && Unsigned(argv[i + 1], value, 10))
        bytes = value;
      else if (option == "--memory-profile" && ParseMemoryProfile(argv[i + 1], profile)) {
      } else if (option == "--assume-symbol-binding" && !own_binding &&
                 std::string_view(argv[i + 1]) == "own")
        own_binding = true;
      else
        return Error("usage",
                     "derive-memory requires --entry HEX and accepts --stack, --steps, "
                     "--assume-symbol-binding own, a profile and budgets");
    }

    if (!entry) return Error("usage", "derive-memory requires --entry HEX");
    if (steps.value_or(1) == 0 || steps.value_or(1) > 10000000)
      return Error("usage", "--steps is a positive count of at most 10000000");
    return DeriveMemory(
        argv[2], *entry, stack.value_or(0x70000000), steps.value_or(200000), record, profile,
        own_binding, {work.value_or(200000000000ULL), bytes.value_or(8ULL * 1024 * 1024 * 1024)});
  }

  if (argc >= 3 && argc <= 11 && argc % 2 == 1 && std::string_view(argv[1]) == "image-writers") {
    auto profile = nyx::a64::MemoryProfile::none;
    std::optional<std::uint64_t> work, bytes;
    const char* record = nullptr;
    for (int i = 3; i < argc; i += 2) {
      const std::string_view option(argv[i]);
      std::uint64_t value = 0;
      if (option == "--work-budget" && !work && Unsigned(argv[i + 1], value, 10))
        work = value;
      else if (option == "--byte-budget" && !bytes && Unsigned(argv[i + 1], value, 10))
        bytes = value;
      else if (option == "--record" && !record)
        record = argv[i + 1];
      else if (option == "--memory-profile" && ParseMemoryProfile(argv[i + 1], profile)) {
      } else
        return Error(
            "usage",
            "image-writers accepts a record path, an optional memory profile and decimal budgets");
    }

    return ImageWriters(
        argv[2], record, profile,
        {work.value_or(100000000000ULL), bytes.value_or(8ULL * 1024 * 1024 * 1024)});
  }

  if (argc >= 5 && argc <= 11 && argc % 2 == 1 && std::string_view(argv[1]) == "simplify-path") {
    std::optional<std::string_view> ranges;
    std::optional<std::uint64_t> work, bytes;
    auto profile = nyx::a64::MemoryProfile::none;
    for (int i = 3; i < argc; i += 2) {
      const std::string_view option(argv[i]);
      std::uint64_t value = 0;
      if (option == "--ranges" && !ranges)
        ranges = argv[i + 1];
      else if (option == "--work-budget" && !work && Unsigned(argv[i + 1], value, 10))
        work = value;
      else if (option == "--byte-budget" && !bytes && Unsigned(argv[i + 1], value, 10))
        bytes = value;
      else if (option == "--memory-profile" && ParseMemoryProfile(argv[i + 1], profile)) {
      } else
        return Error("usage",
                     "Require --ranges once and optional memory profile and decimal budgets");
    }

    if (ranges)
      return SimplifyPath(argv[2], *ranges, profile,
                          {work.value_or(100000000), bytes.value_or(256ULL * 1024 * 1024)});
  }

  if (argc >= 7 && argc <= 53 && argc % 2 == 1 &&
      (std::string_view(argv[1]) == "recover-regions" ||
       std::string_view(argv[1]) == "cfg-probe")) {
    const bool probe = std::string_view(argv[1]) == "cfg-probe";
    auto profile = nyx::a64::MemoryProfile::none;
    std::optional<std::uint64_t> address, size, work, bytes;
    std::optional<std::string_view> assumed, noreturn, abi, returns, frame, reach, image, entries,
        traps, faults, relocations, binding, image_data, derived_memory, image_writers,
        profile_name;
    std::optional<std::uint64_t> load_bias;
    std::vector<std::uint64_t> derive_entries;
    bool derive_auto = false;
    SsaRequest ssa_request;
    std::optional<std::string_view> ssa_passes;
    std::optional<std::uint64_t> rounds;
    for (int i = 3; i < argc; i += 2) {
      const std::string_view option(argv[i]);
      std::uint64_t value = 0;
      if (option == "--address" && !address && Unsigned(argv[i + 1], value, 16))
        address = value;
      else if (option == "--size" && !size && Unsigned(argv[i + 1], value, 10))
        size = value;
      else if (option == "--work-budget" && !work && Unsigned(argv[i + 1], value, 10))
        work = value;
      else if (option == "--byte-budget" && !bytes && Unsigned(argv[i + 1], value, 10))
        bytes = value;
      else if (option == "--assume-initialized" && !assumed)
        assumed = argv[i + 1];
      else if (option == "--assume-noreturn" && !noreturn)
        noreturn = argv[i + 1];
      else if (option == "--assume-abi" && !abi)
        abi = argv[i + 1];
      else if (option == "--assume-returns" && !returns)
        returns = argv[i + 1];
      else if (option == "--assume-private-frame" && !frame)
        frame = argv[i + 1];
      else if (option == "--assume-frame-reach" && !reach)
        reach = argv[i + 1];
      else if (option == "--assume-image-access" && !image)
        image = argv[i + 1];
      else if (option == "--assume-entries" && !entries)
        entries = argv[i + 1];
      else if (option == "--assume-traps" && !traps)
        traps = argv[i + 1];
      else if (option == "--assume-faults" && !faults)
        faults = argv[i + 1];
      else if (option == "--assume-relocations" && !relocations)
        relocations = argv[i + 1];
      else if (option == "--assume-symbol-binding" && !binding)
        binding = argv[i + 1];
      else if (option == "--assume-image-data" && !image_data)
        image_data = argv[i + 1];
      else if (option == "--assume-set" && !profile_name && !probe)
        profile_name = argv[i + 1];
      else if (option == "--derived-memory" && !derived_memory)
        derived_memory = argv[i + 1];
      else if (option == "--derive-entry" && !derive_auto &&
               std::string_view(argv[i + 1]) == "auto" && !probe)
        derive_auto = true;
      else if (option == "--derive-entry" && !probe && derive_entries.size() < 8 &&
               Unsigned(argv[i + 1], value, 16))
        derive_entries.push_back(value);
      else if (option == "--assume-load-bias" && !load_bias && Unsigned(argv[i + 1], value, 16))
        load_bias = value;
      else if (option == "--image-writers" && !image_writers)
        image_writers = argv[i + 1];
      else if (option == "--state-file" && !ssa_request.state_path && !probe)
        ssa_request.state_path = argv[i + 1];
      else if (option == "--stubs-file" && !ssa_request.stubs_path && !probe)
        ssa_request.stubs_path = argv[i + 1];
      else if (option == "--check-file" && !ssa_request.check_path && !probe)
        ssa_request.check_path = argv[i + 1];
      else if (option == "--passes" && !ssa_passes && !probe)
        ssa_passes = argv[i + 1];
      else if (option == "--rounds" && !rounds && !probe && Unsigned(argv[i + 1], value, 10) &&
               value >= 1 && value <= 8)
        rounds = value;
      else if (option == "--listing" && !ssa_request.listing && !probe &&
               std::string_view(argv[i + 1]) == "text")
        ssa_request.listing = true;
      else if (option == "--memory-profile" && ParseMemoryProfile(argv[i + 1], profile)) {
      } else
        return Error("usage",
                     "Require address/size once; optional memory profile, declarations and decimal "
                     "work/byte budgets");
    }

    // Five restrictions on how execution leaves a region, each of which takes
    // one value, so naming them one at a time is ceremony. The name carries a
    // version because adding a member later would silently change what an
    // older artifact's name meant. Entries are deliberately not among them:
    // the other five say how a run leaves, while closed entries is what makes
    // a block deletable, so it is the one whose falsity turns a published
    // edit into a wrong one and it stays typed.
    if (profile_name) {
      if (*profile_name != "aapcs64-open-region-v1")
        return Error("usage", "--assume-set names the set: aapcs64-open-region-v1");
      const std::array<std::pair<std::optional<std::string_view>*, std::string_view>, 5> set{
          {{&abi, "aapcs64"},
           {&returns, "leave"},
           {&traps, "stop"},
           {&faults, "terminal"},
           {&image, "readable"}}};
      for (const auto& [flag, value] : set) {
        if (*flag && **flag != value)
          return Error("usage", ("--assume-set settles " + std::string(value) +
                                 " and the command asks for " + std::string(**flag))
                                    .c_str());
        *flag = value;
      }
    }

    Declarations declared;
    if (assumed) {
      auto parsed = ParseDeclaredRanges(*assumed);
      if (!parsed)
        return Error("invalid_range",
                     "Require up to 16 HEX:BYTES declared ranges of at most 4096 bytes each");
      declared.ranges = std::move(*parsed);
    }

    if (noreturn) {
      if (probe) return Error("usage", "cfg-probe does not accept --assume-noreturn");
      auto parsed = ParseNoreturn(*noreturn);
      if (!parsed)
        return Error("invalid_range", "Require up to 64 HEX declared non-returning call targets");
      declared.noreturn = std::move(*parsed);
    }

    if (abi) {
      if (probe) return Error("usage", "cfg-probe does not accept --assume-abi");
      if (*abi != "aapcs64") return Error("usage", "--assume-abi names the ABI: aapcs64");
      declared.abi = true;
    }

    if (probe && (frame || reach || image || entries || returns || traps || faults))
      return Error("usage", "cfg-probe does not accept SSA proposal declarations");
    if (returns) {
      if (*returns != "leave") return Error("usage", "--assume-returns names the scope: leave");
      declared.return_leaves = true;
    }

    if (traps) {
      if (*traps != "stop") return Error("usage", "--assume-traps names the behavior: stop");
      declared.trap_stops = true;
    }

    if (faults) {
      if (*faults != "terminal")
        return Error("usage", "--assume-faults names the behavior: terminal");
      declared.faults_terminal = true;
    }

    declared.load_bias = load_bias;
    if ((derived_memory || derive_auto || !derive_entries.empty()) && !load_bias)
      return Error("usage",
                   "--derived-memory and --derive-entry need --assume-load-bias: their values were "
                   "observed at one placement");
    // The slots a writer scan of this image examines now, which a record
    // must have examined too for its omissions to mean anything.
    std::uint64_t examinable = 0;
    if (derived_memory || image_writers) {
      auto probed = ReadImage(argv[2]);
      if (!probed) return 1;
      nyx::Sha256 hash;
      hash.Update(probed->bytes());
      declared.record_digest = hash.FinishHex();
      examinable = probed->LocatedRelocations(true).size();
    }

    if (image_writers) {
      nyx::Budget probe({64ULL << 20, 64ULL << 20});
      std::uint64_t examined = 0;
      auto parsed = ReadWrittenSlots(std::string(*image_writers).c_str(), declared.record_digest,
                                     declared.written_slots_unexamined, examined,
                                     declared.written_data, probe);
      if (!parsed)
        return Error("invalid_record",
                     "The image writer record is unreadable, not NYXWRT01, or names another image");
      if (examined != examinable)
        return Error("invalid_record",
                     "The image writer record examined a different set of relocated slots than "
                     "this image declares, so what it omits is not what it found unwritten");
      declared.written_slots = std::move(*parsed);
    }

    if (relocations) {
      if (*relocations != "stable")
        return Error("usage", "--assume-relocations names the restriction: stable");
      declared.relocations_stable = true;
    }

    if (binding) {
      if (*binding != "own")
        return Error("usage", "--assume-symbol-binding names the restriction: own");
      declared.symbols_own = true;
    }

    if (image_data) {
      if (*image_data != "unwritten")
        return Error("usage", "--assume-image-data names the restriction: unwritten");
      // What nothing was found writing is only known from a record that says
      // what everything was found writing.
      if (!declared.written_data)
        return Error("usage",
                     "--assume-image-data unwritten needs an --image-writers record that lists the "
                     "bytes stores reach");
      declared.data_unwritten = true;
    }

    if (frame) {
      declared.frame = ParseFrame(*frame);
      if (!declared.frame)
        return Error(
            "invalid_range",
            "Require --assume-private-frame BEGIN:END, signed decimal, nonempty and at most 1 MiB");
    }

    if (reach) {
      if (!frame || *reach != "none")
        return Error("usage", "--assume-frame-reach none needs --assume-private-frame");
      declared.frame_unreached_by_callees = true;
    }

    // Recovery reads a PC page as a location on a page-aligned image; a bias
    // off a page boundary contradicts that, and a page-masked location folded
    // to its number would land elsewhere than the records made on it.
    if (load_bias && (*load_bias & 0xfff))
      return Error("usage", "--assume-load-bias must be a multiple of the 4096-byte page");
    if (image) {
      if (*image != "readable")
        return Error("usage", "--assume-image-access names the contract: readable");
      declared.image_access = true;
    }

    if (entries) {
      if (*entries != "closed")
        return Error("usage", "--assume-entries names the entry scope: closed");
      declared.closed_entries = true;
    }

    ssa_request.rounds = static_cast<unsigned>(rounds.value_or(1));
    if (ssa_request.stubs_path && !ssa_request.state_path && !ssa_request.check_path)
      return Error("usage", "--stubs-file requires --state-file or --check-file");
    // Evaluating a candidate is an input to the check; the check's record then
    // decides that same candidate in a separate run.
    if (ssa_request.state_path && ssa_request.check_path)
      return Error("usage", "--check-file cannot be combined with --state-file");
    if (ssa_passes) {
      auto names = *ssa_passes;
      while (true) {
        const auto comma = names.find(',');
        const auto index = nyx::passes::FindSsaPass(names.substr(0, comma));
        if (!index || std::find(ssa_request.passes.begin(), ssa_request.passes.end(), *index) !=
                          ssa_request.passes.end())
          return Error("usage",
                       "--passes names each registered pass at most once, comma-separated");
        ssa_request.passes.push_back(*index);
        if (comma == std::string_view::npos) break;
        names.remove_prefix(comma + 1);
      }
    }

    if (derived_memory) {
      nyx::Budget probe({8ULL << 20, 8ULL << 20});
      auto parsed =
          ReadDerivedMemory(std::string(*derived_memory).c_str(), declared.record_digest, probe);
      if (!parsed)
        return Error(
            "invalid_record",
            "The derived memory record is unreadable, not NYXMEM01, or names another image");
      if (parsed->status != "returned")
        return Error("invalid_record",
                     "The derived memory record is of a run that did not return, so its writes are "
                     "not what the routine leaves");
      if (parsed->own_binding && !declared.symbols_own)
        return Error("usage",
                     "The derived memory record rests on --assume-symbol-binding own, which this "
                     "run does not declare");
      declared.derived = std::move(parsed->writes);
      declared.call_results = std::move(parsed->results);

      // A record is the caller's declaration and is admitted as given: this
      // run made no execution of its own to compare it against.
      declared.derivation_runs.push_back(
          {parsed->entry, "from_record", declared.derived.size(), declared.call_results.size(), 0});
    }

    // Whether the last pass made found nothing more to withdraw.
    bool withdrawals_settled = false;
    if (derive_auto || !derive_entries.empty() || declared.data_unwritten) {
      // The same run `nyx derive-memory` makes, made here instead, so the
      // values reach recovery without a record passing between two commands.
      // It stays an observation: what it rests on is published either way.
      auto probed = ReadImage(argv[2]);
      if (!probed) return 1;
      if (probed->machine() != 183 || probed->endianness() != nyx::format::Endianness::Little)
        return Error("unsupported_target", "Deriving memory requires a little-endian AArch64 ELF");
      // Every derivation this command makes shares one budget, so asking for
      // more of them costs more of the same allowance. Without declared
      // budgets it is what `nyx derive-memory` uses, since recovery's own
      // defaults do not cover mapping an image.
      nyx::Budget derivation(
          {work.value_or(200000000000ULL), bytes.value_or(8ULL * 1024 * 1024 * 1024)});

      // What every run has said about a slot, decided once at the end rather
      // than as each run reports: a run that stopped early wrote real bytes
      // but had not finished, so it can contradict a value and cannot supply
      // one, and it must not retract a completed run that agrees with it.
      struct Seen {
        std::optional<std::uint64_t> settled;
        bool conflicted = false;
        bool completed = false;
      };

      std::map<std::uint64_t, Seen> observed;
      for (const auto& existing : declared.derived)
        observed[existing.first] = {existing.second, false, true};
      std::vector<std::uint64_t> attempted;
      const char* failure = nullptr;
      const char* failure_detail = nullptr;

      // A caller names an entry because it wants that run; a discovered one is
      // a guess this run made, so its declining is a fact about the guess.
      const auto derive = [&](std::span<const std::uint64_t> entries, bool named) {
        for (const auto entry : entries) {
          if (std::find(attempted.begin(), attempted.end(), entry) != attempted.end()) continue;
          attempted.push_back(entry);
          bool executable = false;
          for (const auto& segment : probed->segments())
            if (segment.type == 1 && (segment.flags & 1) != 0 && entry >= segment.address &&
                entry - segment.address < segment.file_size)
              executable = true;
          if (!executable) {
            declared.derivation_runs.push_back({entry, "not_executable", 0});
            continue;
          }

          // Nothing here knows what a caller would have passed, so the run
          // guesses. Guessing three ways and keeping only what all three
          // produced drops a result that varied with the guess; it does not
          // establish that none could. Both a fault and a return count, so a
          // routine that faults the same way each time still says nothing.
          constexpr std::array<std::uint64_t, 3> kGuesses{0, 0xa5a5a5a5a5a5a5a5,
                                                          0x5a5a5a5a5a5a5a5a};
          std::array<DerivedRun, kGuesses.size()> runs;
          bool declined = false;
          for (std::size_t i = 0; i < kGuesses.size() && !declined; ++i) {
            runs[i] = Derive(*probed, entry, 0x70000000, 200000, profile, derivation,
                             declared.symbols_own, kGuesses[i]);
            if (!runs[i].decline) continue;
            if (named) {
              failure = runs[i].decline;
              failure_detail = runs[i].detail;
              return;
            }

            declared.derivation_runs.push_back(
                {entry, std::string("declined_") + runs[i].decline, 0});
            declined = true;
          }

          if (declined) continue;
          const auto& first = runs.front();
          bool agrees = true;
          for (const auto& other : runs)
            agrees = agrees && other.status == std::string_view(first.status) &&
                     other.writes == first.writes;
          declared.derivation_runs.push_back(
              {entry, agrees ? first.status : std::string(first.status) + "_guess_dependent"});
          auto& reported = declared.derivation_runs.back();
          if (!agrees) continue;
          reported.writes = first.writes.size();
          for (const auto& [address, value] : first.writes) {
            auto& seen = observed[address];
            if (seen.settled && *seen.settled != value) seen.conflicted = true;
            if (!first.returned) continue;
            seen.settled = value;
            seen.completed = true;
          }

          for (const auto& result : first.results) {
            bool same = true;
            for (const auto& other : runs)
              same = same && std::any_of(other.results.begin(), other.results.end(),
                                         [&](const nyx::ir::SsaCallResult& candidate) {
                                           return candidate.storage == result.storage &&
                                                  candidate.value == result.value;
                                         });
            if (same)
              declared.call_results.push_back(result);
            else
              ++reported.results_dropped;
          }

          reported.results = first.results.size() - reported.results_dropped;
        }

        // Seeing a slot written is decisive on its own; agreeing on what it
        // now holds takes a run that finished and none that contradicts it.
        declared.derived.clear();
        declared.derived_unstable.clear();
        for (const auto& [address, seen] : observed) {
          if (seen.completed && !seen.conflicted && seen.settled)
            declared.derived.push_back({address, *seen.settled});
          else
            declared.derived_unstable.push_back(address);
        }

        std::sort(declared.call_results.begin(), declared.call_results.end(),
                  [](const nyx::ir::SsaCallResult& a, const nyx::ir::SsaCallResult& b) {
                    return std::pair(a.target, a.storage) < std::pair(b.target, b.storage);
                  });
        declared.call_results.erase(
            std::unique(declared.call_results.begin(), declared.call_results.end(),
                        [](const nyx::ir::SsaCallResult& a, const nyx::ir::SsaCallResult& b) {
                          return a.target == b.target && a.storage == b.storage;
                        }),
            declared.call_results.end());
      };

      derive(derive_entries, true);
      if (failure) return Error(failure, failure_detail);

      // Which callees are worth running is only visible in a graph, and
      // building one is what the run does, so the run happens more than once:
      // a pass that publishes nothing names them, they are run, and the next
      // pass sees further for it. It stops when a pass names nothing new. Each
      // pass costs a whole recovery under the declared budget.
      for (unsigned round = 0;
           (derive_auto || declared.data_unwritten) && address && size && round < 4; ++round) {
        Discovery found;
        std::ostringstream sink;
        auto* previous = std::cout.rdbuf(sink.rdbuf());
        const auto status = Lift(argv[2], *address, *size, false, profile, GraphMode::recover,
                                 RecoveryResources(work, bytes), declared, ssa_request, &found);
        std::cout.rdbuf(previous);

        // A pass that declined declined for a reason the caller can act on,
        // and it is the reason the published pass would give too. Suppressing
        // the report was for the case where there is a better one to print.
        if (status != 0) {
          std::cout << sink.str();
          return status;
        }

        // A slot this range addresses, that the loader filled and the image's
        // own code then overwrote, is one whose value nothing static can give.
        // What the loader put there is where the routine that rewrites it
        // begins: that is how a lazily initialized pointer works, and it is
        // the entry worth running. Both halves are already in hand, the reads
        // from the graph and the refutation from the writer scan.
        if (declared.written_slots) {
          const auto relatives = probed->LocatedRelocations(declared.symbols_own);
          for (const auto location : found.addressed) {
            if (!std::binary_search(declared.written_slots->begin(), declared.written_slots->end(),
                                    location))
              continue;
            const auto at =
                std::find_if(relatives.begin(), relatives.end(),
                             [&](const auto& relative) { return relative.address == location; });
            if (at == relatives.end()) continue;
            if (std::find(found.callees.begin(), found.callees.end(), at->target) ==
                found.callees.end())
              found.callees.push_back(at->target);
          }
        }

        const auto withdrawn = Withdraw(found, declared);
        withdrawals_settled = withdrawn == 0;
        const auto admitted =
            declared.data_unwritten ? AdmitUnwrittenData(*probed, found.addressed, declared) : 0;
        const auto before = attempted.size();
        if (derive_auto) derive(found.callees, false);
        if (attempted.size() == before && admitted == 0 && withdrawn == 0) break;
      }
    }

    // A graph can show a store to bytes the run declared fixed that the CFG's
    // block-by-block scan cannot place: through a register another block set.
    // Those bytes are withdrawn and the pass runs again, since what the CFG no
    // longer resolves from them can change what the graph places. The run
    // published is the one over the last fact set.
    // Whenever any declared constant or relocated slot is in play, a graph the
    // run builds can place a store on one, which the CFG's block-local scan
    // cannot. Those bytes are withdrawn and the run repeats, since what the CFG
    // no longer resolves from them changes what it places. Any byte withdrawn
    // means another pass: which layer read it -- a table row, a selected load,
    // the CFG's own resolution -- is not something a pass can list whole. The
    // run published is the one over the last fact set.
    constexpr unsigned kWithdrawalRounds = 8;
    if (address && size && !withdrawals_settled) {
      unsigned round = 0;
      for (; round < kWithdrawalRounds; ++round) {
        Discovery found;
        std::ostringstream sink;
        auto* previous = std::cout.rdbuf(sink.rdbuf());
        const auto status = Lift(argv[2], *address, *size, false, profile, GraphMode::recover,
                                 RecoveryResources(work, bytes), declared, ssa_request, &found);
        std::cout.rdbuf(previous);
        if (status != 0) {
          std::cout << sink.str();
          return status;
        }

        if (Withdraw(found, declared) == 0) break;
      }

      declared.ssa_refutation_unsettled = round == kWithdrawalRounds;
    }

    // The published run rests on the final fact set. If the passes never
    // settled, a placed store may still write a declared byte the graph
    // resolves from, so the run is refused rather than published unsound.
    if (address && size && declared.ssa_refutation_unsettled)
      return Error("unsettled_image_facts",
                   "Withdrawing the bytes the graph's own stores write did"
                   " not settle within the pass bound; no run published");
    if (address && size)
      return Lift(argv[2], *address, *size, false, profile,
                  probe ? GraphMode::probe : GraphMode::recover, RecoveryResources(work, bytes),
                  declared, ssa_request);
  }

  if ((argc == 7 && std::string_view(argv[1]) == "lift") ||
      ((argc == 7 || argc == 9) &&
       (std::string_view(argv[1]) == "simplify" || std::string_view(argv[1]) == "cfg"))) {
    const bool simplify = std::string_view(argv[1]) == "simplify";
    const bool cfg = std::string_view(argv[1]) == "cfg";
    auto profile = nyx::a64::MemoryProfile::none;
    std::optional<std::uint64_t> address;
    std::optional<std::uint64_t> size;
    for (int i = 3; i < argc; i += 2) {
      const std::string_view option(argv[i]);
      std::uint64_t value = 0;
      if (option == "--address" && !address && Unsigned(argv[i + 1], value, 16))
        address = value;
      else if (option == "--size" && !size && Unsigned(argv[i + 1], value, 10))
        size = value;
      else if ((simplify || cfg) && option == "--memory-profile" &&
               ParseMemoryProfile(argv[i + 1], profile)) {
      } else
        return Error("usage",
                     "Require --address HEX and --size BYTES once; simplify/cfg accept an explicit "
                     "memory profile");
    }

    if (address && size)
      return Lift(argv[2], *address, *size, simplify, profile,
                  cfg ? GraphMode::plain_cfg : GraphMode::none);
  }

  return Error("usage",
               "Use nyx --help; whole-function recovery and patching are not implemented yet");
}
