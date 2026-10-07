#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <string_view>

#include "nyx/eval/concrete.hpp"
#include "nyx/target/a64/decode.hpp"

namespace {
constexpr std::size_t kPage = 4096;
constexpr std::size_t kInputSize = 8 + 36 * 8 + 4 + kPage;

std::uint64_t Read64(std::span<const std::uint8_t> input, std::size_t index) {
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i) value |= std::uint64_t(input[8 + index * 8 + i]) << (i * 8);
  return value;
}

void Hex(std::span<const std::uint8_t> bytes) {
  constexpr char digits[] = "0123456789abcdef";
  for (auto byte : bytes) std::cout << digits[byte >> 4] << digits[byte & 15];
}

const char* FaultName(nyx::eval::FaultKind kind) {
  switch (kind) {
    case nyx::eval::FaultKind::unmapped:
      return "unmapped";
    case nyx::eval::FaultKind::permission:
      return "permission";
    case nyx::eval::FaultKind::address_overflow:
      return "address_overflow";
    case nyx::eval::FaultKind::alignment:
      return "alignment";
  }

  return "unknown";
}
}  // namespace

int main() {
  // NYXMDIF1, x0..x30, NZCV, SP, scratch base, source PC, load bias,
  // one original little-endian instruction word, then exactly one scratch page.
  std::array<std::uint8_t, kInputSize> input{};
  std::cin.read(reinterpret_cast<char*>(input.data()), input.size());
  if (!std::cin || std::cin.get() != std::char_traits<char>::eof() ||
      std::string_view(reinterpret_cast<const char*>(input.data()), 8) != "NYXMDIF1")
    return 2;
  const auto nzcv = Read64(input, 31);
  const auto sp = Read64(input, 32);
  const auto base = Read64(input, 33);
  if ((nzcv & ~UINT64_C(0xf0000000)) || sp % 16 || base % kPage || base < UINT64_C(0x100000000) ||
      base > UINT64_C(0x1000000000))
    return 2;
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

  const auto contents = std::span(input).last(kPage);
  const std::array<nyx::eval::RegionInput, 1> regions{{{base, contents, true, true}}};
  auto memory = nyx::eval::Memory::Create(regions, budget);
  if (!memory.memory) return 3;
  std::array<std::uint8_t, 4> code{};
  for (unsigned i = 0; i < 4; ++i) code[i] = input[8 + 36 * 8 + i];
  auto decoded = nyx::a64::Decode(Read64(input, 34), code, budget,
                                  {nyx::a64::MemoryProfile::concrete_atomic_scalar});
  if (!decoded.group) {
    std::cerr << "decode declined: " << static_cast<unsigned>(decoded.reason) << '\n';
    return 4;
  }

  auto result =
      nyx::eval::Execute(*decoded.group, state, *memory.memory, budget, {}, {Read64(input, 35)});
  if (result.outcome != nyx::eval::Outcome::completed &&
      result.outcome != nyx::eval::Outcome::fault) {
    std::cerr << "evaluation declined: " << static_cast<unsigned>(result.outcome) << '\n';
    return 5;
  }

  std::uint64_t output_flags = 0;
  for (unsigned i = 32; i < 36; ++i) output_flags |= state.cells[i].value.word(0) << (63 - i);
  std::cout << "{\"schema\":1,\"profile\":\"concrete_atomic_scalar\",\"hardware_checked\":false,"
               "\"architectural_fault_authority\":false,\"completed\":"
            << (result.outcome == nyx::eval::Outcome::completed ? "true" : "false")
            << ",\"registers\":[";
  for (unsigned i = 0; i < 31; ++i) {
    if (i) std::cout << ',';
    std::cout << state.cells[i].value.word(0);
  }

  std::cout << "],\"nzcv\":" << output_flags << ",\"sp\":" << state.cells[31].value.word(0)
            << ",\"base\":" << base << ",\"memory\":\"";
  Hex(memory.memory->Regions()[0].bytes);
  std::cout << "\",\"fault\":";
  if (result.fault) {
    std::cout << "{\"kind\":\"" << FaultName(result.fault->kind)
              << "\",\"address\":" << result.fault->address
              << ",\"operation\":" << result.fault->operation
              << ",\"write\":" << (result.fault->write ? "true" : "false") << '}';
  } else {
    std::cout << "null";
  }

  std::cout << ",\"events\":[";
  bool first = true;
  for (const auto& event : result.events) {
    if (!first) std::cout << ',';
    first = false;
    std::cout << "{\"operation\":" << event.operation << ",\"address\":" << event.address
              << ",\"size\":" << event.size << ",\"write\":" << (event.write ? "true" : "false")
              << ",\"completed\":" << (event.completed ? "true" : "false")
              << ",\"conditional\":" << (event.conditional ? "true" : "false")
              << ",\"performed\":" << (event.performed ? "true" : "false") << ",\"bytes\":\"";
    Hex(std::span(event.bytes).first(event.size));
    std::cout << "\"}";
  }

  std::cout << "]}\n";
  return std::cout ? 0 : 6;
}
