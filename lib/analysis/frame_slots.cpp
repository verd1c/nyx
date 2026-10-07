#include "nyx/analysis/frame_slots.hpp"

namespace nyx::analysis {

PrivateFrameResult ProvePrivateFrameSlots(const ir::SsaGraph& graph, PrivateFrameContract contract,
                                          Budget& budget) {
  const auto decline = [](PrivateFrameDecline reason) { return PrivateFrameResult{{}, reason}; };
  const auto valid = ir::ValidateSsa(graph, budget);
  if (valid != ir::SsaDecline::none)
    return decline(valid == ir::SsaDecline::resource_limit ? PrivateFrameDecline::resource_limit
                                                           : PrivateFrameDecline::invalid_graph);
  const auto scan = ir::ScanPrivateFrame(graph, contract, budget);
  switch (scan.reason) {
    case ir::FrameScanDecline::none:
      break;
    case ir::FrameScanDecline::invalid:
      return decline(PrivateFrameDecline::invalid_graph);
    case ir::FrameScanDecline::missing_contract:
      return decline(PrivateFrameDecline::missing_contract);
    case ir::FrameScanDecline::unknown_call:
      return decline(PrivateFrameDecline::unknown_call);
    case ir::FrameScanDecline::opaque_effect:
      return decline(PrivateFrameDecline::opaque_effect);
    case ir::FrameScanDecline::unknown_continuation:
      return decline(PrivateFrameDecline::unknown_continuation);
    case ir::FrameScanDecline::escaped_address:
      return decline(PrivateFrameDecline::escaped_address);
    case ir::FrameScanDecline::resource_limit:
      return decline(PrivateFrameDecline::resource_limit);
  }

  if (scan.dynamic) return decline(PrivateFrameDecline::unknown_offset);
  auto slots = ir::PrivateFrameSlots(scan, graph, contract, budget);
  if (!slots) return decline(PrivateFrameDecline::resource_limit);
  PrivateFrameFacts facts{graph.arena(), graph.revision(), contract, std::move(*slots)};
  const auto checked = ir::ValidatePrivateFrameFacts(facts, graph, budget);
  if (checked != ir::PrivateFrameFactDecline::none)
    return decline(checked == ir::PrivateFrameFactDecline::resource_limit
                       ? PrivateFrameDecline::resource_limit
                       : PrivateFrameDecline::invalid_graph);
  return {std::move(facts), PrivateFrameDecline::none};
}

}  // namespace nyx::analysis
