#include <array>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <string_view>

#include "nyx/support/bit_vector.hpp"

namespace {

constexpr unsigned max_bits = 4096;

bool parse_u64(std::string_view text, std::uint64_t& value, int base = 10) {
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value, base);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

const char* decline_name(nyx::BitVectorDecline decline) {
  switch (decline) {
    case nyx::BitVectorDecline::none:
      return "none";
    case nyx::BitVectorDecline::invalid_width:
      return "invalid_width";
    case nyx::BitVectorDecline::width_limit:
      return "width_limit";
    case nyx::BitVectorDecline::excess_words:
      return "excess_words";
    case nyx::BitVectorDecline::width_mismatch:
      return "width_mismatch";
    case nyx::BitVectorDecline::work_limit:
      return "work_limit";
    case nyx::BitVectorDecline::byte_limit:
      return "byte_limit";
  }

  return "invalid_decline";
}

bool parse_words(std::string_view text, std::array<std::uint64_t, max_bits / 64>& words,
                 unsigned& count) {
  if (text.empty() || text.size() > max_bits / 4) return false;
  count = 0;
  while (!text.empty()) {
    const auto size = text.size() > 16 ? 16 : text.size();
    if (!parse_u64(text.substr(text.size() - size), words[count++], 16)) return false;
    text.remove_suffix(size);
  }

  return true;
}

void emit(const nyx::BitVectorResult& result) {
  if (!result) {
    std::printf("error:%s\n", decline_name(result.decline()));
    return;
  }

  bool leading = true;
  for (unsigned i = result->word_count(); i > 0; --i) {
    const auto word = static_cast<unsigned long long>(result->word(i - 1));
    if (leading && word == 0 && i != 1) continue;
    std::printf(leading ? "%llx" : "%016llx", word);
    leading = false;
  }

  std::putchar('\n');
}

// Fixed five-field protocol: decimal width, operation, hex lhs, hex rhs,
// decimal count. Unused rhs/count fields must still be supplied as zero.
void evaluate(std::string_view line) {
  std::array<std::string_view, 5> fields{};
  unsigned field_count = 0;
  while (!line.empty()) {
    const auto start = line.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) break;
    line.remove_prefix(start);
    const auto end = line.find_first_of(" \t\r\n");
    if (field_count == fields.size()) {
      std::puts("error:parse");
      return;
    }

    fields[field_count++] = line.substr(0, end);
    if (end == std::string_view::npos) break;
    line.remove_prefix(end);
  }

  std::uint64_t width = 0;
  std::uint64_t count = 0;
  if (field_count != 5 || !parse_u64(fields[0], width) || !parse_u64(fields[4], count)) {
    std::puts("error:parse");
    return;
  }

  if (width == 0 || width > max_bits) {
    std::puts(width == 0 ? "error:invalid_width" : "error:width_limit");
    return;
  }

  std::array<std::uint64_t, max_bits / 64> lhs_words{}, rhs_words{};
  unsigned lhs_count = 0, rhs_count = 0;
  if (!parse_words(fields[2], lhs_words, lhs_count) ||
      !parse_words(fields[3], rhs_words, rhs_count)) {
    std::puts("error:parse");
    return;
  }

  nyx::Budget budget({std::uint64_t{max_bits} * max_bits + 3 * max_bits, 4096});
  auto lhs = nyx::BitVector::from_words(static_cast<unsigned>(width),
                                        std::span<const std::uint64_t>(lhs_words.data(), lhs_count),
                                        max_bits, budget);
  auto rhs = nyx::BitVector::from_words(static_cast<unsigned>(width),
                                        std::span<const std::uint64_t>(rhs_words.data(), rhs_count),
                                        max_bits, budget);
  if (!lhs) {
    emit(lhs);
    return;
  }

  if (!rhs) {
    emit(rhs);
    return;
  }

  const auto op = fields[1];
  if (op == "add")
    emit(lhs->add(*rhs, budget));
  else if (op == "sub")
    emit(lhs->sub(*rhs, budget));
  else if (op == "mul")
    emit(lhs->mul(*rhs, budget));
  else if (op == "and")
    emit(lhs->bit_and(*rhs, budget));
  else if (op == "or")
    emit(lhs->bit_or(*rhs, budget));
  else if (op == "xor")
    emit(lhs->bit_xor(*rhs, budget));
  else if (op == "not")
    emit(lhs->bit_not(budget));
  else if (op == "shl")
    emit(lhs->shl(count, budget));
  else if (op == "lshr")
    emit(lhs->lshr(count, budget));
  else if (op == "ashr")
    emit(lhs->ashr(count, budget));
  else if (op == "rotl")
    emit(lhs->rotl(count, budget));
  else if (op == "rotr")
    emit(lhs->rotr(count, budget));
  else
    std::puts("error:operation");
}

}  // namespace

int main() {
  std::array<char, 2304> line{};
  while (std::fgets(line.data(), static_cast<int>(line.size()), stdin)) {
    const auto size = std::strlen(line.data());
    if (size == line.size() - 1 && line[size - 1] != '\n') {
      int character;
      do {
        character = std::getchar();
      } while (character != '\n' && character != EOF);
      std::puts("error:line_limit");
      continue;
    }

    evaluate(std::string_view(line.data(), size));
  }

  return std::ferror(stdin) || std::ferror(stdout) ? 1 : 0;
}
