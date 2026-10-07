#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace nyx {

// FIPS 180-4 SHA-256, incremental. Identifies exact published content; it is
// not an authentication of who produced that content.
class Sha256 {
 public:
  Sha256();
  void Update(std::span<const std::uint8_t> bytes);
  void Update(std::string_view text);
  std::array<std::uint8_t, 32> Finish();

  // Lowercase hex of Finish().
  std::string FinishHex();

 private:
  void Block(const std::uint8_t* block);

  std::array<std::uint32_t, 8> state_;
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffered_ = 0;
  std::uint64_t length_ = 0;
};

}  // namespace nyx
