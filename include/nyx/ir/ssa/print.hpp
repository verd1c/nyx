#pragma once

#include <optional>
#include <string>

#include "nyx/ir/ssa/graph.hpp"

namespace nyx::ir {

struct SsaTextResult {
  std::optional<std::string> text;
  SsaDecline reason = SsaDecline::none;
};

[[nodiscard]] SsaTextResult PrintSsa(const SsaGraph&, Budget&);
[[nodiscard]] SsaTextResult PrintSsa(const SsaGraph&, std::span<const Group> decoded_sources,
                                     Budget&);

}  // namespace nyx::ir
