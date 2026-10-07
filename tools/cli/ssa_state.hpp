#pragma once

#include <optional>
#include <span>
#include <string>

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::cli {

struct SsaStateResult {
  std::optional<std::string> json;
  const char* reason = "none";
};

inline constexpr std::size_t kMaxStubDeclaration = 8 + 8 + 32 * (16 + 128);

// The number of stubs a NYXSTB01 declaration names, if its framing is valid:
// the magic, one to 32 aligned records of 4 to 128 bytes in the runtime window,
// no overlap and nothing after the last record. Whether the bytes match the
// mapped pages is checked only against a state file.
[[nodiscard]] std::optional<std::uint64_t> SsaStubCount(std::span<const std::uint8_t> declaration);

// Execute the exact published graph from a bounded NYXFUN01 initial-state
// envelope. `stubs` is the exact NYXSTB01 declaration the caller also bound
// into the candidate identity, so the evaluated and published stubs are the
// same bytes. This is an evaluator observation, not an independent comparison.
[[nodiscard]] SsaStateResult EvaluateSsaRequest(const ir::SsaGraph&,
                                                std::span<const ir::Group> decoded_sources,
                                                std::uint64_t image_entry, const char* request_path,
                                                std::optional<std::span<const std::uint8_t>> stubs,
                                                Budget&);

}  // namespace nyx::cli
