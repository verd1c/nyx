#include <charconv>
#include <iostream>
#include <string>
#include <string_view>

#include "nyx/eval/concrete.hpp"
#include "nyx/target/a64/decode.hpp"

namespace {
bool Hex(std::string_view text, std::uint64_t& value) {
  if (text.empty() || text.size() > 16) return false;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value, 16);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}
}  // namespace

int main(int argc, char** argv) {
  const bool with_sp = argc == 3 && std::string_view(argv[2]) == "--sp";
  if (argc != 2 && !with_sp) return 1;
  const std::string_view code(argv[1]);
  if (code.empty() || code.size() > 4096 * 8 || code.size() % 8 != 0) return 1;
  nyx::Budget decode_budget({10000000, 100000000});
  std::vector<nyx::ir::Group> groups;
  for (std::size_t offset = 0; offset < code.size(); offset += 8) {
    std::array<std::uint8_t, 4> bytes{};
    for (unsigned i = 0; i < 4; ++i) {
      std::uint64_t value = 0;
      if (!Hex(code.substr(offset + i * 2, 2), value)) return 1;
      bytes[i] = static_cast<std::uint8_t>(value);
    }

    auto result = nyx::a64::Decode(offset / 2, bytes, decode_budget);
    if (!result.group) {
      std::cout << "declined:" << static_cast<int>(result.reason) << '\n';
      return 2;
    }

    groups.push_back(std::move(*result.group));
  }

  unsigned cases = 0;
  std::string line;

  // Protocol is one bounded line of 31 GPRs and NZCV per initial state.
  while (true) {
    line.clear();
    char ch;
    while (std::cin.get(ch) && ch != '\n') {
      if (line.size() >= 1024) return 1;
      line += ch;
    }

    if (line.empty() && !std::cin) break;
    if (++cases > 4096) return 1;
    std::array<std::uint64_t, 33> input{};
    const unsigned field_count = with_sp ? 33 : 32;
    std::size_t pos = 0;
    for (unsigned field = 0; field < field_count; ++field) {
      const auto end = line.find(' ', pos);
      const auto size = (end == std::string::npos ? line.size() : end) - pos;
      if (pos > line.size() || !Hex(std::string_view(line).substr(pos, size), input[field]))
        return 1;
      pos += size + 1;
    }

    const auto nzcv = input[field_count - 1];
    if (pos != line.size() + 1 || (nzcv & ~std::uint64_t{0xf0000000}) != 0) return 1;
    nyx::Budget budget({1000000000, 128 * 1024 * 1024});
    nyx::eval::State state;
    for (unsigned i = 0; i < 36; ++i) {
      const unsigned width = i < 32 ? 64 : 1;
      const auto value = i < 31    ? input[i]
                         : i == 31 ? (with_sp ? input[31] : 0)
                                   : (nzcv >> (63 - i)) & 1;
      auto bits = nyx::BitVector::from_u64(width, value, 4096, budget);
      if (!bits) return 1;
      state.cells.push_back({i, std::move(*bits)});
    }

    for (const auto& group : groups) {
      const auto result = nyx::eval::Execute(group, state, budget);
      if (result != nyx::eval::Outcome::completed) {
        std::cout << "incomplete:" << static_cast<int>(result) << '\n';
        return 3;
      }
    }

    for (unsigned i = 0; i < 31; ++i) std::cout << std::hex << state.cells[i].value.word(0) << ' ';
    if (with_sp) std::cout << std::hex << state.cells[31].value.word(0) << ' ';
    std::uint64_t flags = 0;
    for (unsigned i = 32; i < 36; ++i) flags |= state.cells[i].value.word(0) << (63 - i);
    std::cout << std::hex << flags << '\n';
  }

  return cases == 0 ? 1 : 0;
}
