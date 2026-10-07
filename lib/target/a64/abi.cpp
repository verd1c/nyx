#include "nyx/target/a64/abi.hpp"

#include <algorithm>
#include <array>

#include "nyx/target/a64/decode.hpp"

namespace nyx::a64 {
namespace {

template <std::size_t Count>
constexpr std::array<ir::StorageId, Count> Sorted(std::array<ir::StorageId, Count> ids) {
  std::sort(ids.begin(), ids.end());
  return ids;
}

// X18 is the platform register; Android keeps its shadow call stack there, so
// callees read it and callers rely on it.
constexpr auto kCall = Sorted([] {
  std::array<ir::StorageId, 9 + 1 + 12 + 1 + 16> ids{};
  std::size_t at = 0;
  for (ir::StorageId id = 0; id <= 8; ++id) ids[at++] = id;
  ids[at++] = 18;
  for (ir::StorageId id = 19; id <= 30; ++id) ids[at++] = id;
  ids[at++] = kSp;
  for (ir::StorageId id = 0; id < 16; ++id) ids[at++] = kQ0 + id;
  return ids;
}());

constexpr auto kReturn = Sorted([] {
  std::array<ir::StorageId, 9 + 1 + 11 + 1 + 16> ids{};
  std::size_t at = 0;
  for (ir::StorageId id = 0; id <= 8; ++id) ids[at++] = id;
  ids[at++] = 18;
  for (ir::StorageId id = 19; id <= 29; ++id) ids[at++] = id;
  ids[at++] = kSp;
  for (ir::StorageId id = 0; id < 16; ++id) ids[at++] = kQ0 + id;
  return ids;
}());

}  // namespace

std::span<const ir::StorageId> ObservedAtCall() { return kCall; }

std::span<const ir::StorageId> ObservedAtReturn() { return kReturn; }

ir::SsaObservability AbiObservability() {
  const auto preserved = Sorted(kCalleeSaved);
  return {true,
          false,
          {kCall.begin(), kCall.end()},
          {kReturn.begin(), kReturn.end()},
          {preserved.begin(), preserved.end()}};
}

}  // namespace nyx::a64
