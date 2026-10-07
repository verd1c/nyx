#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <string_view>

#include "nyx/eval/concrete.hpp"
#include "nyx/target/a64/decode.hpp"

namespace {
constexpr std::size_t kPage = 4096;
constexpr std::size_t kVectors = 32 * 16;
constexpr std::size_t kInputSize = 8 + 35 * 8 + kVectors + kPage;

std::uint64_t Read64(std::span<const std::uint8_t> bytes, std::size_t at) {
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i) value |= std::uint64_t(bytes[at + i]) << (i * 8);
  return value;
}

void Hex(std::span<const std::uint8_t> bytes) {
  constexpr char digits[] = "0123456789abcdef";
  for (auto byte : bytes) std::cout << digits[byte >> 4] << digits[byte & 15];
}
}  // namespace

int main() {
  // The runner_vector.c payload: NYXVEC01, x0..x30, NZCV, SP, scratch base,
  // one instruction word, then q0..q31 and exactly one scratch page.
  std::array<std::uint8_t, kInputSize> input{};
  std::cin.read(reinterpret_cast<char*>(input.data()), input.size());
  if (!std::cin || std::cin.get() != std::char_traits<char>::eof() ||
      std::string_view(reinterpret_cast<const char*>(input.data()), 8) != "NYXVEC01")
    return 2;
  const auto nzcv = Read64(input, 8 + 31 * 8);
  const auto sp = Read64(input, 8 + 32 * 8);
  const auto base = Read64(input, 8 + 33 * 8);
  const auto word = Read64(input, 8 + 34 * 8);
  if ((nzcv & ~UINT64_C(0xf0000000)) || sp % 16 || base % kPage || word > UINT32_MAX ||
      base < UINT64_C(0x100000000) || base > UINT64_C(0x1000000000))
    return 2;
  nyx::Budget budget({100000000, 256ULL * 1024 * 1024});
  nyx::eval::State state;
  for (unsigned i = 0; i < 36; ++i) {
    const auto initial = i < 31    ? Read64(input, 8 + i * 8)
                         : i == 31 ? sp
                                   : ((nzcv >> (63 - i)) & 1);
    auto value = nyx::BitVector::from_u64(i < 32 ? 64 : 1, initial, 64, budget);
    if (!value) return 3;
    state.cells.push_back({i, std::move(*value)});
  }

  const auto vectors = std::span(input).subspan(8 + 35 * 8, kVectors);
  for (unsigned i = 0; i < 32; ++i) {
    const std::array<std::uint64_t, 2> words{Read64(vectors, i * 16), Read64(vectors, i * 16 + 8)};
    auto value = nyx::BitVector::from_words(128, words, 128, budget);
    if (!value) return 3;
    state.cells.push_back({nyx::a64::kQ0 + i, std::move(*value)});
  }

  const auto contents = std::span(input).last(kPage);
  const std::array<nyx::eval::RegionInput, 1> regions{{{base, contents, true, true}}};
  auto memory = nyx::eval::Memory::Create(regions, budget);
  if (!memory.memory) return 3;
  const std::array<std::uint8_t, 4> code{
      static_cast<std::uint8_t>(word), static_cast<std::uint8_t>(word >> 8),
      static_cast<std::uint8_t>(word >> 16), static_cast<std::uint8_t>(word >> 24)};
  auto decoded =
      nyx::a64::Decode(0x1000, code, budget, {nyx::a64::MemoryProfile::concrete_atomic_scalar});
  if (!decoded.group) {
    std::cerr << "decode declined: " << static_cast<unsigned>(decoded.reason) << '\n';
    return 4;
  }

  const auto result = nyx::eval::Execute(*decoded.group, state, *memory.memory, budget);
  if (result.outcome != nyx::eval::Outcome::completed &&
      result.outcome != nyx::eval::Outcome::fault) {
    std::cerr << "evaluation declined: " << static_cast<unsigned>(result.outcome) << '\n';
    return 5;
  }

  std::uint64_t flags = 0;
  for (unsigned i = 32; i < 36; ++i) flags |= state.cells[i].value.word(0) << (63 - i);
  std::cout << "{\"completed\":"
            << (result.outcome == nyx::eval::Outcome::completed ? "true" : "false")
            << ",\"registers\":[";
  for (unsigned i = 0; i < 31; ++i) std::cout << (i ? "," : "") << state.cells[i].value.word(0);
  std::cout << "],\"nzcv\":" << flags << ",\"sp\":" << state.cells[31].value.word(0)
            << ",\"vectors\":\"";
  for (unsigned i = 0; i < 32; ++i) {
    std::array<std::uint8_t, 16> bytes{};
    for (unsigned b = 0; b < 16; ++b)
      bytes[b] = static_cast<std::uint8_t>(state.cells[36 + i].value.word(b / 8) >> ((b % 8) * 8));
    Hex(bytes);
  }

  std::cout << "\",\"memory\":\"";
  Hex(memory.memory->Regions()[0].bytes);
  std::cout << "\",\"fault_address\":";
  if (result.fault)
    std::cout << result.fault->address;
  else
    std::cout << "null";
  std::cout << "}\n";
  return std::cout ? 0 : 6;
}
