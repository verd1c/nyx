#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>

#include "nyx/eval/program.hpp"
#include "nyx/target/a64/decode.hpp"

namespace {
constexpr std::size_t kPage = 4096;

std::uint64_t Read64(std::span<const std::uint8_t> input, std::size_t index) {
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i) value |= std::uint64_t(input[8 + index * 8 + i]) << (i * 8);
  return value;
}

const char* Status(nyx::eval::ProgramStatus status) {
  using nyx::eval::ProgramStatus;
  switch (status) {
    case ProgramStatus::exit:
      return "exit";
    case ProgramStatus::unresolved:
      return "unresolved";
    case ProgramStatus::step_limit:
      return "step_limit";
    case ProgramStatus::resource_limit:
      return "resource_limit";
    case ProgramStatus::fault:
      return "fault";
    case ProgramStatus::invalid_program:
      return "invalid_program";
    case ProgramStatus::invalid_state:
      return "invalid_state";
    case ProgramStatus::unsupported:
      return "unsupported";
  }

  return "invalid";
}

void Hex(std::span<const std::uint8_t> bytes) {
  constexpr char digits[] = "0123456789abcdef";
  for (auto byte : bytes) std::cout << digits[byte >> 4] << digits[byte & 15];
}
}  // namespace

int main() {
  // NYXPDIF1, x0..x30, NZCV, SP, scratch base, source base, load bias,
  // step limit, code size, original code bytes, one scratch page.
  std::array<std::uint8_t, 8 + 38 * 8> input{};
  std::cin.read(reinterpret_cast<char*>(input.data()), input.size());
  if (!std::cin || std::string_view(reinterpret_cast<const char*>(input.data()), 8) != "NYXPDIF1")
    return 2;
  const auto nzcv = Read64(input, 31);
  const auto sp = Read64(input, 32);
  const auto base = Read64(input, 33);
  const auto source = Read64(input, 34);
  const auto bias = Read64(input, 35);
  const auto steps = Read64(input, 36);
  const auto size = Read64(input, 37);
  const auto runtime = source + bias;
  if (nzcv & ~UINT64_C(0xf0000000) || base % kPage || size == 0 || size > kPage || size % 4 ||
      steps > 4096 || source > std::numeric_limits<std::uint64_t>::max() - size ||
      runtime > std::numeric_limits<std::uint64_t>::max() - size)
    return 2;
  std::array<std::uint8_t, kPage> code{};
  std::array<std::uint8_t, kPage> scratch{};
  std::cin.read(reinterpret_cast<char*>(code.data()), size);
  std::cin.read(reinterpret_cast<char*>(scratch.data()), scratch.size());
  if (!std::cin || std::cin.get() != std::char_traits<char>::eof()) return 2;
  nyx::Budget budget({100000000, 256ULL * 1024 * 1024});
  nyx::eval::State state;
  if (budget.try_consume({36, 36 * sizeof(nyx::eval::Cell)}) != nyx::BudgetDecline::none) return 3;
  state.cells.reserve(36);
  for (unsigned i = 0; i < 36; ++i) {
    const auto initial = i < 31 ? Read64(input, i) : i == 31 ? sp : ((nzcv >> (63 - i)) & 1);
    auto value = nyx::BitVector::from_u64(i < 32 ? 64 : 1, initial, 64, budget);
    if (!value) return 3;
    state.cells.push_back({i, std::move(*value)});
  }

  const std::array<nyx::eval::RegionInput, 1> regions{{{base, scratch, true, true}}};
  auto memory = nyx::eval::Memory::Create(regions, budget);
  if (!memory.memory) return 3;
  if (budget.try_consume({size / 4, size / 4 * sizeof(nyx::ir::Group)}) != nyx::BudgetDecline::none)
    return 3;
  std::vector<nyx::ir::Group> groups;
  groups.reserve(size / 4);
  for (std::size_t offset = 0; offset < size; offset += 4) {
    const std::array<std::uint8_t, 4> word{code[offset], code[offset + 1], code[offset + 2],
                                           code[offset + 3]};
    auto decoded = nyx::a64::Decode(source + offset, word, budget,
                                    {nyx::a64::MemoryProfile::concrete_atomic_scalar});
    if (!decoded.group) {
      std::cerr << "decode declined at " << offset << ": " << static_cast<unsigned>(decoded.reason)
                << '\n';
      return 4;
    }

    groups.push_back(std::move(*decoded.group));
  }

  const std::array<std::uint64_t, 1> exits{runtime + size};
  nyx::eval::ProgramLimits limits;
  limits.max_steps = steps;
  const auto result =
      nyx::eval::Run(groups, state, *memory.memory, runtime, exits, budget, limits, {bias});
  std::uint64_t flags = 0;
  for (unsigned i = 32; i < 36; ++i) flags |= state.cells[i].value.word(0) << (63 - i);
  std::cout << "{\"status\":\"" << Status(result.status) << "\",\"pc\":" << result.runtime_pc
            << ",\"committed_steps\":" << result.committed_steps << ",\"registers\":[";
  for (unsigned i = 0; i < 31; ++i) {
    if (i) std::cout << ',';
    std::cout << state.cells[i].value.word(0);
  }

  std::cout << "],\"nzcv\":" << flags << ",\"sp\":" << state.cells[31].value.word(0)
            << ",\"memory\":\"";
  Hex(memory.memory->Regions()[0].bytes);
  std::cout << "\",\"trace\":[";
  bool first = true;
  for (const auto& step : result.trace) {
    if (!first) std::cout << ',';
    first = false;
    std::cout << "{\"pc\":" << step.runtime_pc
              << ",\"outcome\":" << static_cast<unsigned>(step.execution.outcome)
              << ",\"events\":" << step.execution.events.size() << ",\"target\":";
    if (step.execution.transfer)
      std::cout << step.execution.transfer->target;
    else
      std::cout << "null";
    std::cout << ",\"fault_address\":";
    if (step.execution.fault)
      std::cout << step.execution.fault->address;
    else
      std::cout << "null";
    std::cout << '}';
  }

  std::cout << "]}\n";
  return std::cout ? 0 : 6;
}
