#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>

#include "nyx/eval/concrete.hpp"
#include "nyx/target/a64/decode.hpp"

namespace {

void Hex(std::span<const std::uint8_t> bytes) {
  constexpr char digits[] = "0123456789abcdef";
  for (auto byte : bytes) std::cout << digits[byte >> 4] << digits[byte & 15];
}

std::array<std::uint8_t, 16> VectorBytes(const nyx::BitVector& value) {
  std::array<std::uint8_t, 16> result{};
  for (unsigned i = 0; i < result.size(); ++i)
    result[i] = static_cast<std::uint8_t>(value.word(i / 8) >> ((i % 8) * 8));
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  const std::string_view operation(argv[1]), variant(argv[3]);
  if ((operation != "load" && operation != "store") ||
      (variant != "base" && variant != "max" && variant != "q31"))
    return 2;
  char* end = nullptr;
  const auto mapped = std::strtoul(argv[2], &end, 10);
  if (*end || mapped < 1 || mapped > 16) return 2;
  const bool load = operation == "load";
  const std::uint32_t word = variant == "max"   ? (load ? 0x3dfffd20 : 0x3dbffd20)
                             : variant == "q31" ? (load ? 0x3dc0013f : 0x3d80013f)
                                                : (load ? 0x3dc00120 : 0x3d800120);
  const std::array<std::uint8_t, 4> code{
      static_cast<std::uint8_t>(word), static_cast<std::uint8_t>(word >> 8),
      static_cast<std::uint8_t>(word >> 16), static_cast<std::uint8_t>(word >> 24)};
  nyx::Budget budget({1000000, 1000000});
  auto decoded =
      nyx::a64::Decode(0x1000, code, budget, {nyx::a64::MemoryProfile::concrete_atomic_scalar});
  if (!decoded.group) return 3;
  constexpr std::uint64_t address = 0x100000000;
  constexpr std::uint64_t offset = 65520;
  nyx::eval::State state;
  auto base =
      nyx::BitVector::from_u64(64, variant == "max" ? address - offset : address, 128, budget);
  const std::array<std::uint64_t, 2> seed{0xa7a6a5a4a3a2a1a0, 0xafaeadacabaaa9a8};
  auto vector = nyx::BitVector::from_words(128, seed, 128, budget);
  if (!base || !vector) return 3;
  state.cells.push_back({9, std::move(*base)});
  state.cells.push_back({nyx::a64::kQ0 + (variant == "q31" ? 31U : 0U), std::move(*vector)});
  std::array<std::uint8_t, 16> bytes{};
  if (load) {
    for (unsigned i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(i);
  }

  const nyx::eval::RegionInput region[] = {{address, std::span(bytes).first(mapped)}};
  auto memory = nyx::eval::Memory::Create(region, budget);
  if (!memory.memory) return 3;
  const auto result = nyx::eval::Execute(*decoded.group, state, *memory.memory, budget);
  if (result.outcome != nyx::eval::Outcome::completed &&
      result.outcome != nyx::eval::Outcome::fault)
    return 4;
  std::cout << "{\"completed\":"
            << (result.outcome == nyx::eval::Outcome::completed ? "true" : "false") << ",\"q\":\"";
  const auto q = VectorBytes(state.cells[1].value);
  Hex(q);
  std::cout << "\",\"memory\":\"";
  Hex(memory.memory->Regions()[0].bytes);
  std::cout << "\",\"fault_index\":";
  if (result.fault)
    std::cout << result.fault->address - address;
  else
    std::cout << "null";
  std::cout << "}\n";
  return std::cout ? 0 : 5;
}
