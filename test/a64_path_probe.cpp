#include <algorithm>
#include <array>
#include <iostream>
#include <string_view>

#include "nyx/eval/path.hpp"
#include "nyx/eval/program.hpp"
#include "nyx/recovery/mba.hpp"
#include "nyx/recovery/memory.hpp"
#include "nyx/target/a64/decode.hpp"

namespace {
bool Read(std::span<std::uint8_t> bytes) {
  std::cin.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  return static_cast<bool>(std::cin);
}

std::optional<std::uint64_t> Read64() {
  std::array<std::uint8_t, 8> bytes{};
  if (!Read(bytes)) return {};
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i) value |= std::uint64_t(bytes[i]) << (8 * i);
  return value;
}

void Hex(std::span<const std::uint8_t> bytes) {
  constexpr char digits[] = "0123456789abcdef";
  for (const auto byte : bytes) std::cout << digits[byte >> 4] << digits[byte & 15];
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string_view mode = argv[1];
  if (mode != "original" && mode != "normalized" && mode != "simplified" &&
      mode != "snapshot_mutant")
    return 2;
  std::array<std::uint8_t, 8> magic{};
  if (!Read(magic) ||
      std::string_view(reinterpret_cast<const char*>(magic.data()), 8) != "NYXSPR01")
    return 2;
  std::array<std::uint64_t, 39> header{};
  for (auto& value : header) {
    const auto input = Read64();
    if (!input) return 2;
    value = *input;
  }

  const auto flags = header[31], sp = header[32], scratch_base = header[33], bias = header[34];
  const auto entry = header[35], fragment_count = header[36], exit_count = header[37],
             page_count = header[38];
  if ((flags & ~UINT64_C(0xf0000000)) || !fragment_count || fragment_count > 16 || !exit_count ||
      exit_count > 16 || page_count > 8 || scratch_base % 4096)
    return 2;
  nyx::Budget budget({100000000, 256ULL * 1024 * 1024});
  std::vector<nyx::ir::Group> groups;
  std::size_t total = 0;
  for (std::size_t i = 0; i < fragment_count; ++i) {
    const auto source = Read64(), size = Read64();
    if (!source || !size || !*size || *size % 4 || *source % 4 || *size > 4096 - total ||
        *size - 1 > UINT64_MAX - *source || *size - 1 > UINT64_MAX - (*source + bias))
      return 2;
    total += *size;
    for (std::size_t offset = 0; offset < *size; offset += 4) {
      std::array<std::uint8_t, 4> word{};
      if (!Read(word)) return 2;
      auto decoded = nyx::a64::Decode(*source + offset, word, budget,
                                      {nyx::a64::MemoryProfile::concrete_atomic_scalar});
      if (!decoded.group) return 3;
      groups.push_back(std::move(*decoded.group));
    }
  }

  std::vector<std::uint64_t> exits;
  for (std::size_t i = 0; i < exit_count; ++i) {
    const auto address = Read64();
    if (!address) return 2;
    exits.push_back(*address + bias);
  }

  std::array<std::uint8_t, 4096> scratch{};
  if (!Read(scratch)) return 2;
  std::vector<std::array<std::uint8_t, 4096>> pages(page_count);
  std::vector<std::uint64_t> addresses;
  for (auto& page : pages) {
    const auto address = Read64();
    if (!address || *address % 4096 || !Read(page)) return 2;
    addresses.push_back(*address);
  }

  if (std::cin.get() != std::char_traits<char>::eof()) return 2;
  std::vector<nyx::eval::RegionInput> regions{{scratch_base, scratch, true, true}};
  for (std::size_t i = 0; i < pages.size(); ++i)
    regions.push_back({addresses[i], pages[i], true, false});
  auto memory = nyx::eval::Memory::Create(regions, budget);
  if (!memory.memory) return 4;
  nyx::eval::State state;
  for (unsigned i = 0; i < 36; ++i) {
    const auto initial = i < 31 ? header[i] : i == 31 ? sp : (flags >> (63 - i)) & 1;
    auto value = nyx::BitVector::from_u64(i < 32 ? 64 : 1, initial, 64, budget);
    if (!value) return 4;
    state.cells.push_back({i, std::move(*value)});
  }

  std::uint64_t pc = entry + bias;
  bool completed = false;
  std::vector<nyx::eval::ExecutionResult> trace;
  std::size_t facts = 0, memory_edits = 0, arithmetic_edits = 0;
  if (mode == "original") {
    auto result = nyx::eval::Run(groups, state, *memory.memory, pc, exits, budget, {}, {bias});
    completed = result.status == nyx::eval::ProgramStatus::exit;
    if (!completed && result.status != nyx::eval::ProgramStatus::fault) return 5;
    pc = result.runtime_pc;
    for (auto& step : result.trace) trace.push_back(std::move(step.execution));
  } else {
    const auto first = std::ranges::find_if(
        groups, [entry](const auto& group) { return group.source_address() == entry; });
    if (first == groups.end()) return 5;
    auto normalized = nyx::ir::NormalizePath(std::span(first, groups.end()), budget);
    if (!normalized.path) return 6;
    auto path = std::move(*normalized.path);
    if (mode == "simplified" || mode == "snapshot_mutant") {
      auto forwarded = nyx::recovery::ForwardMemoryValues(path, budget);
      if (!forwarded.path) return 7;
      facts = forwarded.facts.size();
      memory_edits = forwarded.journal.size();
      auto simplified = nyx::recovery::SimplifyMba(*forwarded.path, budget);
      if (!simplified.path) return 8;
      arithmetic_edits = simplified.journal.size();
      path = std::move(*simplified.path);
    }

    if (mode == "snapshot_mutant") {
      const auto transfer = path.boundaries().back().transfer;
      if (!transfer) return 12;
      auto nodes = std::vector<nyx::ir::Node>(path.nodes().begin(), path.nodes().end());
      auto& target = nodes[transfer->target];
      if (target.width != 64 || nyx::ir::Descriptor(target.op)->effect != nyx::ir::Effect::pure)
        return 12;
      // Deliberately wrong recovery: substitute a snapshot destination while
      // retaining the runtime table load and all its faults.
      target = {nyx::ir::Op::image_address, 64, {}, exits.front() - bias};
      path =
          nyx::ir::Path({path.sources().begin(), path.sources().end()}, std::move(nodes),
                        {path.origins().begin(), path.origins().end()},
                        {path.boundaries().begin(), path.boundaries().end()}, path.revision() + 1);
    }

    auto result = nyx::eval::ExecutePath(path, state, *memory.memory, budget, {}, {bias});
    completed = result.outcome == nyx::eval::Outcome::completed;
    if (!completed && result.outcome != nyx::eval::Outcome::fault) return 9;
    if (completed) {
      if (!result.runtime_next || std::ranges::find(exits, *result.runtime_next) == exits.end())
        return 10;
      pc = *result.runtime_next;
    } else
      pc = result.source_address + bias;
    trace = std::move(result.trace);
  }

  std::uint64_t output_flags = 0;
  for (unsigned i = 32; i < 36; ++i) output_flags |= state.cells[i].value.word(0) << (63 - i);
  std::cout << "{\"completed\":" << (completed ? "true" : "false") << ",\"pc\":" << pc
            << ",\"registers\":[";
  for (unsigned i = 0; i < 31; ++i) {
    if (i) std::cout << ',';
    std::cout << state.cells[i].value.word(0);
  }

  std::cout << "],\"nzcv\":" << output_flags << ",\"sp\":" << state.cells[31].value.word(0)
            << ",\"memory\":\"";
  const auto output_regions = memory.memory->Regions();
  for (const auto& region : output_regions)
    if (region.address == scratch_base) Hex(region.bytes);
  std::cout << "\",\"global_pages\":[";
  for (std::size_t i = 0; i < addresses.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << "{\"address\":" << addresses[i] << ",\"data\":\"";
    for (const auto& region : output_regions)
      if (region.address == addresses[i]) Hex(region.bytes);
    std::cout << "\"}";
  }

  std::cout << "],\"forwarding_facts\":" << facts << ",\"memory_edits\":" << memory_edits
            << ",\"arithmetic_edits\":" << arithmetic_edits << ",\"trace\":[";
  for (std::size_t i = 0; i < trace.size(); ++i) {
    if (i) std::cout << ',';
    const auto& step = trace[i];
    std::cout << "{\"fault_address\":";
    if (step.fault)
      std::cout << step.fault->address;
    else
      std::cout << "null";
    std::cout << ",\"events\":[";
    for (std::size_t j = 0; j < step.events.size(); ++j) {
      if (j) std::cout << ',';
      const auto& event = step.events[j];
      std::cout << "{\"operation\":" << event.operation << ",\"address\":" << event.address
                << ",\"write\":" << (event.write ? "true" : "false")
                << ",\"conditional\":" << (event.conditional ? "true" : "false")
                << ",\"performed\":" << (event.performed ? "true" : "false") << ",\"bytes\":\"";
      Hex(std::span(event.bytes).first(event.size));
      std::cout << "\"}";
    }

    std::cout << "]}";
  }

  std::cout << "]}\n";
  return std::cout ? 0 : 11;
}
