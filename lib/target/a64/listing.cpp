#include "nyx/target/a64/listing.hpp"

#include <array>
#include <string_view>

#include "nyx/target/a64/abi.hpp"
#include "nyx/target/a64/decode.hpp"

namespace nyx::a64 {
namespace {

constexpr std::array<std::string_view, 32> kQ = {
    "q0",  "q1",  "q2",  "q3",  "q4",  "q5",  "q6",  "q7",  "q8",  "q9",  "q10",
    "q11", "q12", "q13", "q14", "q15", "q16", "q17", "q18", "q19", "q20", "q21",
    "q22", "q23", "q24", "q25", "q26", "q27", "q28", "q29", "q30", "q31"};
constexpr std::array<std::string_view, 31> kX = {
    "x0",  "x1",  "x2",  "x3",  "x4",  "x5",  "x6",  "x7",  "x8",  "x9",  "x10",
    "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21",
    "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30"};

constexpr auto kNames = [] {
  std::array<ir::SsaStorageName, 31 + 5 + 1 + 32> names{};
  std::size_t at = 0;
  for (ir::StorageId id = 0; id < 31; ++id) names[at++] = {id, kX[id]};
  names[at++] = {kSp, "sp"};
  names[at++] = {kN, "n"};
  names[at++] = {kZ, "z"};
  names[at++] = {kC, "c"};
  names[at++] = {kV, "v"};
  names[at++] = {kTpidrEl0, "tpidr_el0"};
  for (ir::StorageId id = 0; id < 32; ++id) names[at++] = {kQ0 + id, kQ[id]};
  return names;
}();

}  // namespace

ir::SsaListingTarget ListingTarget() { return {kNames, ObservedAtCall(), ObservedAtReturn()}; }

}  // namespace nyx::a64
