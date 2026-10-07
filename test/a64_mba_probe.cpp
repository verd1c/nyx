#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <string_view>

#include "nyx/eval/block.hpp"
#include "nyx/eval/program.hpp"
#include "nyx/recovery/mba.hpp"
#include "nyx/target/a64/decode.hpp"

namespace {
std::uint64_t Read64(std::span<const std::uint8_t> input, std::size_t index) {
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i) value |= std::uint64_t(input[8 + index * 8 + i]) << (8 * i);
  return value;
}

void Hex(std::span<const std::uint8_t> bytes) {
  constexpr char digits[] = "0123456789abcdef";
  for (auto byte : bytes) std::cout << digits[byte >> 4] << digits[byte & 15];
}

const char* Rule(nyx::recovery::MbaRule rule) {
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
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string_view mode = argv[1];
  if (mode != "original" && mode != "normalized" && mode != "simplified" && mode != "linear")
    return 2;
  std::array<std::uint8_t, 8 + 37 * 8> input{};
  std::cin.read(reinterpret_cast<char*>(input.data()), input.size());
  if (!std::cin || std::string_view(reinterpret_cast<const char*>(input.data()), 8) != "NYXMBA01")
    return 2;
  const auto flags = Read64(input, 31), sp = Read64(input, 32), base = Read64(input, 33);
  const auto source = Read64(input, 34), bias = Read64(input, 35), size = Read64(input, 36);

  // Placement is modular; only the byte extent must remain nonwrapping.
  const auto runtime = source + bias;
  if (size == 0 || size > 4096 || size % 4 || base % 4096 || (flags & ~UINT64_C(0xf0000000)) ||
      source > UINT64_MAX - size || runtime > UINT64_MAX - size)
    return 2;
  std::array<std::uint8_t, 4096> code{}, data{};
  std::cin.read(reinterpret_cast<char*>(code.data()), size);
  std::cin.read(reinterpret_cast<char*>(data.data()), data.size());
  if (!std::cin || std::cin.get() != std::char_traits<char>::eof()) return 2;
  nyx::Budget budget({100000000, 256ULL * 1024 * 1024});
  nyx::eval::State state;
  state.cells.reserve(36);
  for (unsigned i = 0; i < 36; ++i) {
    const auto initial = i < 31 ? Read64(input, i) : i == 31 ? sp : ((flags >> (63 - i)) & 1);
    auto value = nyx::BitVector::from_u64(i < 32 ? 64 : 1, initial, 64, budget);
    if (!value) return 3;
    state.cells.push_back({i, std::move(*value)});
  }

  const nyx::eval::RegionInput region{base, data, true, false};
  auto memory = nyx::eval::Memory::Create(std::span(&region, 1), budget);
  if (!memory.memory) return 3;
  std::vector<nyx::ir::Group> groups;
  groups.reserve(size / 4);
  for (std::size_t offset = 0; offset < size; offset += 4) {
    const std::array<std::uint8_t, 4> word{code[offset], code[offset + 1], code[offset + 2],
                                           code[offset + 3]};
    auto decoded = nyx::a64::Decode(source + offset, word, budget,
                                    {nyx::a64::MemoryProfile::concrete_atomic_scalar});
    if (!decoded.group) return 4;
    groups.push_back(std::move(*decoded.group));
  }

  nyx::eval::Outcome outcome;
  std::uint64_t pc = source + bias + size;
  std::vector<nyx::eval::ExecutionResult> trace;
  std::optional<nyx::ir::Block> block;
  std::vector<nyx::recovery::MbaEdit> journal;
  if (mode == "original") {
    const std::array<std::uint64_t, 1> exits{pc};
    auto result =
        nyx::eval::Run(groups, state, *memory.memory, source + bias, exits, budget, {}, {bias});
    if (result.status == nyx::eval::ProgramStatus::exit)
      outcome = nyx::eval::Outcome::completed;
    else if (result.status == nyx::eval::ProgramStatus::fault)
      outcome = nyx::eval::Outcome::fault;
    else
      return 5;
    pc = result.runtime_pc;
    for (auto& step : result.trace) trace.push_back(std::move(step.execution));
  } else {
    auto normalized = nyx::ir::Normalize(groups, budget);
    if (!normalized.block) return 6;
    block = std::move(normalized.block);
    if (mode == "simplified" || mode == "linear") {
      auto simplified = mode == "linear" ? nyx::recovery::SimplifyLinearMba(*block, budget)
                                         : nyx::recovery::SimplifyMba(*block, budget);
      if (!simplified.block) return 7;
      block = std::move(simplified.block);
      journal = std::move(simplified.journal);
    }

    auto result = nyx::eval::ExecuteBlock(*block, state, *memory.memory, budget, {}, {bias});
    outcome = result.outcome;
    if (outcome == nyx::eval::Outcome::fault)
      pc = result.source_address + bias;
    else if (outcome != nyx::eval::Outcome::completed)
      return 8;
    trace = std::move(result.trace);
  }

  std::uint64_t output_flags = 0;
  for (unsigned i = 32; i < 36; ++i) output_flags |= state.cells[i].value.word(0) << (63 - i);
  std::cout << "{\"completed\":" << (outcome == nyx::eval::Outcome::completed ? "true" : "false")
            << ",\"pc\":" << pc << ",\"registers\":[";
  for (unsigned i = 0; i < 31; ++i) {
    if (i) std::cout << ',';
    std::cout << state.cells[i].value.word(0);
  }

  std::cout << "],\"nzcv\":" << output_flags << ",\"sp\":" << state.cells[31].value.word(0)
            << ",\"memory\":\"";
  Hex(memory.memory->Regions()[0].bytes);
  std::cout << "\",\"rules\":[";
  bool first = true;
  for (const auto& edit : journal) {
    if (!first) std::cout << ',';
    first = false;
    std::cout << "{\"rule\":\"" << Rule(edit.rule) << "\",\"node\":" << edit.node
              << ",\"from_revision\":" << edit.from_revision
              << ",\"to_revision\":" << edit.to_revision << ",\"original_op\":\""
              << nyx::ir::Descriptor(edit.original.op)->name << "\",\"replacement_op\":\""
              << nyx::ir::Descriptor(edit.replacement.op)->name
              << "\",\"replacement_immediate\":" << edit.replacement.immediate << '}';
  }

  std::cout << "],\"trace\":[";
  first = true;
  for (const auto& step : trace) {
    if (!first) std::cout << ',';
    first = false;
    std::cout << "{\"events\":[";
    for (std::size_t i = 0; i < step.events.size(); ++i) {
      const auto& event = step.events[i];
      if (i) std::cout << ',';
      std::cout << "{\"operation\":" << event.operation << ",\"address\":" << event.address
                << ",\"write\":" << (event.write ? "true" : "false")
                << ",\"conditional\":" << (event.conditional ? "true" : "false")
                << ",\"performed\":" << (event.performed ? "true" : "false") << ",\"bytes\":\"";
      Hex(std::span(event.bytes).first(event.size));
      std::cout << "\"}";
    }

    std::cout << "],\"fault_address\":";
    if (step.fault)
      std::cout << step.fault->address;
    else
      std::cout << "null";
    std::cout << '}';
  }

  std::cout << "]}\n";
  return std::cout ? 0 : 9;
}
