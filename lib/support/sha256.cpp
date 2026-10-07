#include "nyx/support/sha256.hpp"

#include <bit>
#include <cstring>

namespace nyx {
namespace {
constexpr std::array<std::uint32_t, 64> kRound = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
}  // namespace

Sha256::Sha256()
    : state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19} {}

void Sha256::Block(const std::uint8_t* block) {
  std::array<std::uint32_t, 64> w{};
  for (unsigned i = 0; i < 16; ++i)
    w[i] = std::uint32_t(block[4 * i]) << 24 | std::uint32_t(block[4 * i + 1]) << 16 |
           std::uint32_t(block[4 * i + 2]) << 8 | block[4 * i + 3];
  for (unsigned i = 16; i < 64; ++i) {
    const auto s0 = std::rotr(w[i - 15], 7) ^ std::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const auto s1 = std::rotr(w[i - 2], 17) ^ std::rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }

  auto [a, b, c, d, e, f, g, h] = state_;
  for (unsigned i = 0; i < 64; ++i) {
    const auto t1 = h + (std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25)) +
                    ((e & f) ^ (~e & g)) + kRound[i] + w[i];
    const auto t2 =
        (std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }

  const std::array<std::uint32_t, 8> add{a, b, c, d, e, f, g, h};
  for (unsigned i = 0; i < 8; ++i) state_[i] += add[i];
}

void Sha256::Update(std::span<const std::uint8_t> bytes) {
  length_ += bytes.size();
  for (const auto byte : bytes) {
    buffer_[buffered_++] = byte;
    if (buffered_ == buffer_.size()) {
      Block(buffer_.data());
      buffered_ = 0;
    }
  }
}

void Sha256::Update(std::string_view text) {
  Update({reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
}

std::array<std::uint8_t, 32> Sha256::Finish() {
  const auto bits = length_ * 8;
  const std::uint8_t one = 0x80, zero = 0;
  Update({&one, 1});
  while (buffered_ != 56) Update({&zero, 1});
  std::array<std::uint8_t, 8> length{};
  for (unsigned i = 0; i < 8; ++i) length[i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
  Update(length);
  std::array<std::uint8_t, 32> digest{};
  for (unsigned i = 0; i < 32; ++i)
    digest[i] = static_cast<std::uint8_t>(state_[i / 4] >> (24 - 8 * (i % 4)));
  return digest;
}

std::string Sha256::FinishHex() {
  constexpr char hex[] = "0123456789abcdef";
  std::string text;
  text.reserve(64);
  for (const auto byte : Finish()) {
    text += hex[byte >> 4];
    text += hex[byte & 15];
  }

  return text;
}

}  // namespace nyx
