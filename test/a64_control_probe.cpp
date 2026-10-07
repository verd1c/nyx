#include <array>
#include <cstdint>
#include <iostream>
#include <string_view>

#include "nyx/eval/concrete.hpp"
#include "nyx/target/a64/decode.hpp"

namespace {
constexpr std::size_t kInputSize = 8 + 35 * 8 + 4;

std::uint64_t Read64(const std::array<std::uint8_t, kInputSize>& input, std::size_t index) {
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i) value |= std::uint64_t(input[8 + index * 8 + i]) << (i * 8);
  return value;
}

const char* Kind(nyx::ir::TransferKind kind) {
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
}  // namespace

int main() {
  // NYXCDIF1, x0..x30, NZCV, SP, source PC, load bias, original instruction.
  std::array<std::uint8_t, kInputSize> input{};
  std::cin.read(reinterpret_cast<char*>(input.data()), input.size());
  if (!std::cin || std::cin.get() != std::char_traits<char>::eof() ||
      std::string_view(reinterpret_cast<const char*>(input.data()), 8) != "NYXCDIF1")
    return 2;
  const auto nzcv = Read64(input, 31);
  const auto sp = Read64(input, 32);
  if (nzcv & ~UINT64_C(0xf0000000)) return 2;
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

  std::array<std::uint8_t, 4> code{};
  for (unsigned i = 0; i < 4; ++i) code[i] = input[8 + 35 * 8 + i];
  auto decoded = nyx::a64::Decode(Read64(input, 33), code, budget);
  if (!decoded.group) {
    std::cerr << "decode declined: " << static_cast<unsigned>(decoded.reason) << '\n';
    return 4;
  }

  auto result = nyx::eval::ExecuteDetailed(*decoded.group, state, budget, {}, {Read64(input, 34)});
  if (result.outcome != nyx::eval::Outcome::completed || !result.transfer) {
    std::cerr << "evaluation declined or lacked transfer: " << static_cast<unsigned>(result.outcome)
              << '\n';
    return 5;
  }

  std::uint64_t output_flags = 0;
  for (unsigned i = 32; i < 36; ++i) output_flags |= state.cells[i].value.word(0) << (63 - i);
  std::cout << "{\"registers\":[";
  for (unsigned i = 0; i < 31; ++i) {
    if (i) std::cout << ',';
    std::cout << state.cells[i].value.word(0);
  }

  std::cout << "],\"nzcv\":" << output_flags << ",\"sp\":" << state.cells[31].value.word(0)
            << ",\"transfer\":{\"kind\":\"" << Kind(result.transfer->kind)
            << "\",\"target\":" << result.transfer->target << ",\"condition\":";
  if (result.transfer->condition)
    std::cout << (*result.transfer->condition ? "true" : "false");
  else
    std::cout << "null";
  std::cout << ",\"continuation\":";
  if (result.transfer->continuation)
    std::cout << *result.transfer->continuation;
  else
    std::cout << "null";
  std::cout << "}}\n";
  return std::cout ? 0 : 6;
}
