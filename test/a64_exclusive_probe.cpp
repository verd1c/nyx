#include <array>
#include <cstdint>
#include <iostream>

#include "nyx/eval/concrete.hpp"
#include "nyx/target/a64/decode.hpp"

namespace {
std::uint64_t Read(std::span<const std::uint8_t> bytes) {
  std::uint64_t value = 0;
  for (unsigned i = 0; i < bytes.size(); ++i) value |= std::uint64_t{bytes[i]} << (8 * i);
  return value;
}

bool ExecuteWord(std::uint32_t word, nyx::eval::State& state, nyx::eval::Memory& memory,
                 nyx::Budget& budget) {
  const std::array<std::uint8_t, 4> bytes = {
      static_cast<std::uint8_t>(word), static_cast<std::uint8_t>(word >> 8),
      static_cast<std::uint8_t>(word >> 16), static_cast<std::uint8_t>(word >> 24)};
  auto decoded =
      nyx::a64::Decode(0x2000, bytes, budget, {nyx::a64::MemoryProfile::concrete_exclusive_scalar});
  return decoded.group && nyx::eval::Execute(*decoded.group, state, memory, budget).outcome ==
                              nyx::eval::Outcome::completed;
}

int Run(unsigned width, unsigned mode) {
  constexpr std::uint64_t first32 = 42;
  constexpr std::uint64_t first64 = UINT64_C(0x1122334455667788);
  const auto first = width == 32 ? first32 : first64;
  const auto value = width == 32 ? UINT64_C(55) : UINT64_C(0x8877665544332211);
  const auto changed = width == 32 ? UINT64_C(77) : UINT64_C(0xfedcba9876543210);
  std::array<std::uint8_t, 16> bytes{};
  for (unsigned i = 0; i < width / 8; ++i) {
    bytes[i] = static_cast<std::uint8_t>(first >> (8 * i));
    bytes[8 + i] = static_cast<std::uint8_t>(std::uint64_t{17} >> (8 * i));
  }

  nyx::Budget budget({100000, 100000});
  const nyx::eval::RegionInput region{0x1000, bytes};
  auto memory = nyx::eval::Memory::Create(std::span(&region, 1), budget);
  if (!memory.memory) return 2;
  nyx::eval::State state;
  for (const auto [id, value] :
       {std::pair{8U, UINT64_C(0x1000)}, std::pair{9U, value}, std::pair{10U, UINT64_C(0)},
        std::pair{11U, UINT64_C(0)}, std::pair{12U, UINT64_C(0)},
        std::pair{13U, mode == 1 ? changed : first}, std::pair{14U, UINT64_C(0x1008)}}) {
    auto bits = nyx::BitVector::from_u64(64, value, 64, budget);
    if (!bits) return 3;
    state.cells.push_back({id, std::move(*bits)});
  }

  if (!ExecuteWord(width == 32 ? 0x885ffd0aU : 0xc85ffd0aU, state, *memory.memory, budget))
    return 4;
  if (mode == 1 || mode == 2) {
    if (!ExecuteWord(width == 32 ? 0xb900010dU : 0xf900010dU, state, *memory.memory, budget))
      return 5;
  }

  if (mode == 3) {
    if (!ExecuteWord(0xd5033f5fU, state, *memory.memory, budget)) return 6;
  }

  const auto first_store = (width == 32 ? 0x880bfd09U : 0xc80bfd09U) + (mode == 4 ? 0xc0U : 0U);
  const auto second_store = (width == 32 ? 0x880cfd09U : 0xc80cfd09U) + (mode == 4 ? 0xc0U : 0U);
  if (!ExecuteWord(first_store, state, *memory.memory, budget) ||
      !ExecuteWord(second_store, state, *memory.memory, budget))
    return 7;
  const auto memory_bytes = memory.memory->Regions()[0].bytes;
  std::cout << (width == 32 ? 'w' : 'x') << mode << ' ' << state.cells[2].value.word(0) << ' '
            << state.cells[3].value.word(0) << ' ' << state.cells[4].value.word(0) << ' '
            << Read(std::span(memory_bytes).first(width / 8)) << ' '
            << Read(std::span(memory_bytes).subspan(8, width / 8)) << '\n';
  return 0;
}

int RunFaults() {
  nyx::Budget budget({100000, 100000});
  const std::array<std::uint8_t, 4> bytes{42, 0, 0, 0};
  const nyx::eval::RegionInput region{0x1000, bytes, true, false};
  auto memory = nyx::eval::Memory::Create(std::span(&region, 1), budget);
  if (!memory.memory) return 9;
  nyx::eval::State state;
  for (const auto [id, value] :
       {std::pair{8U, UINT64_C(0x1000)}, std::pair{9U, UINT64_C(55)}, std::pair{10U, UINT64_C(0)},
        std::pair{11U, UINT64_C(7)}, std::pair{14U, UINT64_C(0)}}) {
    auto bits = nyx::BitVector::from_u64(64, value, 64, budget);
    if (!bits) return 10;
    state.cells.push_back({id, std::move(*bits)});
  }

  const auto execute = [&](std::uint32_t word) {
    const std::array<std::uint8_t, 4> instruction = {
        static_cast<std::uint8_t>(word), static_cast<std::uint8_t>(word >> 8),
        static_cast<std::uint8_t>(word >> 16), static_cast<std::uint8_t>(word >> 24)};
    auto decoded = nyx::a64::Decode(0x2000, instruction, budget,
                                    {nyx::a64::MemoryProfile::concrete_exclusive_scalar});
    return decoded.group ? nyx::eval::Execute(*decoded.group, state, *memory.memory, budget).outcome
                         : nyx::eval::Outcome::invalid_group;
  };

  std::cout << "f0 " << (execute(0x885ffdcaU) == nyx::eval::Outcome::fault ? "fault" : "unexpected")
            << '\n';
  if (execute(0x885ffd0aU) != nyx::eval::Outcome::completed) return 11;
  std::cout << "f1 " << (execute(0x880bfd09U) == nyx::eval::Outcome::fault ? "fault" : "unexpected")
            << '\n';
  if (execute(0x885ffd0aU) != nyx::eval::Outcome::completed) return 12;
  if (execute(0x880bfdc9U) != nyx::eval::Outcome::completed) return 13;
  std::cout << "f2 " << state.cells[3].value.word(0) << '\n';
  return 0;
}
}  // namespace

int main() {
  for (unsigned width : {32U, 64U}) {
    for (unsigned mode = 0; mode < 5; ++mode) {
      if (const auto status = Run(width, mode)) return status;
    }
  }

  if (const auto status = RunFaults()) return status;
  return std::cout ? 0 : 8;
}
