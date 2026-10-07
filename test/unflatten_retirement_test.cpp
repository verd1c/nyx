#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <string_view>

#include "nyx/analysis/cfg.hpp"
#include "nyx/analysis/path_control.hpp"
#include "nyx/analysis/regions.hpp"
#include "nyx/analysis/unflatten.hpp"
using namespace nyx;
using namespace nyx::analysis;

SourceRecord Record(ir::Group group) {
  return {group.source_address(),
          {group.bytes().begin(), group.bytes().end()},
          std::move(group),
          OpaqueReason::none};
}

SourceRecord Jump(std::uint64_t address, std::uint64_t target, bool image) {
  return Record(ir::Group(
      address, {5, 6, 7, 8}, {{image ? ir::Op::image_address : ir::Op::constant, 64, {}, target}},
      {}, ir::MemoryModel::unspecified, ir::Transfer{ir::TransferKind::jump, 0, {}, {}, {}}));
}

constexpr std::array<std::uint8_t, 8> kEntries = {0x10, 0x00, 0x20, 0x00, 0x30, 0x00, 0x40, 0x00};
constexpr std::array<std::uint8_t, 4> kState = {1, 0, 0, 0};

// The state is above three, or it indexes the table: the false arm is what
// bounds it, which is what makes the dispatch's successor set complete.
SourceRecord Guard(std::uint64_t address, std::uint64_t above, std::uint64_t next) {
  return Record(ir::Group(address, {1, 2, 3, 4},
                          {{ir::Op::read, 64, {}, 0, 8},
                           {ir::Op::extract, 32, {0}, 0},
                           {ir::Op::constant, 32, {}, 3},
                           {ir::Op::unsigned_less, 1, {1, 2}},
                           {ir::Op::bit_not, 1, {3}},
                           {ir::Op::sub, 32, {1, 2}},
                           {ir::Op::constant, 32, {}, 0},
                           {ir::Op::equal, 1, {5, 6}},
                           {ir::Op::bit_not, 1, {7}},
                           {ir::Op::bit_and, 1, {4, 8}},
                           {ir::Op::image_address, 64, {}, above},
                           {ir::Op::image_address, 64, {}, next}},
                          {}, ir::MemoryModel::unspecified,
                          ir::Transfer{ir::TransferKind::conditional, 10, 9, 11, {}}));
}

// The table jump itself. A block reached any other way carries no dispatch
// record, because the bound then depends on which path arrived.
SourceRecord Dispatch(std::uint64_t address) {
  return Record(ir::Group(address, {1, 2, 3, 4},
                          {{ir::Op::read, 64, {}, 0, 8},
                           {ir::Op::extract, 32, {0}, 0},
                           {ir::Op::zext, 64, {1}},
                           {ir::Op::constant, 64, {}, 1},
                           {ir::Op::shl, 64, {2, 3}},
                           {ir::Op::image_address, 64, {}, 0x5000},
                           {ir::Op::add, 64, {5, 4}},
                           {ir::Op::load, 16, {6}},
                           {ir::Op::zext, 64, {7}},
                           {ir::Op::image_address, 64, {}, 0x9000},
                           {ir::Op::add, 64, {9, 8}}},
                          {}, ir::MemoryModel::atomic_scalar_reference,
                          ir::Transfer{ir::TransferKind::jump, 10, {}, {}, {}}));
}

// One guarded dispatch, five cases, and a state the caller declares constant.
// Each mode asks the analysis for a different refusal or a different published
// result; none of them may end with a transition or a retirement the facts do
// not support.
enum class Mode {
  // Retirement: the case at 0x9020 leaves in one of three ways, and only an
  // exit the graph resolves to one of its blocks may let the dispatcher retire.
  absolute,
  outside,
  resolved,
  opaque_declared,
  opaque_unknown,
  // A trap never completes, so it leaves no successor, even as a destination.
  trap,
  opaque_destination,
  opaque_destination_unknown,
  // The stitched graph the surviving blocks form.
  stitched,
  // One refusal each: a route not proved to continue, a dispatch whose target
  // this path does not resolve, a destination the dispatch cannot reach, and
  // two routes from one entry that do not agree on where it goes.
  refuse_route,
  refuse_target,
  refuse_outside,
  refuse_conflict,
  refuse_predicate,
  refuse_assumption
};

int MixedNoreturn() {
  // A conditional call has one declared non-returning target and one target
  // that may return. Its continuation must survive either graph walk.
  std::vector<SourceRecord> sources{
      Record(ir::Group(0x100, {1, 2, 3, 4},
                       {{ir::Op::read, 1, {}, 0, 100},
                        {ir::Op::image_address, 64, {}, 0x200},
                        {ir::Op::image_address, 64, {}, 0x300},
                        {ir::Op::select, 64, {0, 1, 2}},
                        {ir::Op::image_address, 64, {}, 0x104}},
                       {}, ir::MemoryModel::unspecified,
                       ir::Transfer{ir::TransferKind::call, 3, {}, {}, 4})),
      Jump(0x104, 0x104, true)};
  const std::uint64_t entry[] = {0x100};
  Budget budget({UINT64_MAX, UINT64_MAX});
  auto built = BuildCfg(sources, entry, budget);
  if (!built.cfg || built.cfg->blocks().size() != 2 || built.cfg->blocks()[0].edges.size() != 3) {
    std::puts("FAIL: conditional call fixture did not produce two callees and a continuation");
    return 1;
  }

  Regions regions(std::move(*built.cfg), {});
  const std::uint64_t one[] = {0x200};
  const auto mixed = Unflatten(regions, {}, budget, {one});
  if (!mixed.unflattening || mixed.unflattening->graph.blocks.size() != 2 ||
      mixed.unflattening->graph.blocks[0].edges.size() != 3 ||
      mixed.unflattening->graph.blocks[0].edges.back().kind != CfgEdgeKind::potential_return ||
      !mixed.unflattening->graph.blocks[0].edges.back().target_block) {
    std::puts("FAIL: a possibly returning callee lost the continuation");
    return 1;
  }

  const std::uint64_t both[] = {0x200, 0x300};
  const auto never = Unflatten(regions, {}, budget, {both});
  if (!never.unflattening || never.unflattening->graph.blocks.size() != 1 ||
      never.unflattening->graph.blocks[0].edges.size() != 2) {
    std::puts("FAIL: all declared non-returning callees kept the continuation");
    return 1;
  }

  std::puts("ok: mixed-noreturn");
  return 0;
}

int main(int argc, char** argv) {
  const std::string_view mode = argc > 1 ? argv[1] : "";
  if (mode == "mixed-noreturn") return MixedNoreturn();
  const std::array<std::pair<std::string_view, Mode>, 15> modes{
      {{"absolute", Mode::absolute},
       {"outside", Mode::outside},
       {"resolved", Mode::resolved},
       {"opaque-declared", Mode::opaque_declared},
       {"opaque-unknown", Mode::opaque_unknown},
       {"trap", Mode::trap},
       {"opaque-destination", Mode::opaque_destination},
       {"opaque-destination-unknown", Mode::opaque_destination_unknown},
       {"stitched", Mode::stitched},
       {"refuse-route", Mode::refuse_route},
       {"refuse-target", Mode::refuse_target},
       {"refuse-outside", Mode::refuse_outside},
       {"refuse-conflict", Mode::refuse_conflict},
       {"refuse-predicate", Mode::refuse_predicate},
       {"refuse-assumption", Mode::refuse_assumption}}};
  const auto selected = std::find_if(modes.begin(), modes.end(),
                                     [&](const auto& entry) { return entry.first == mode; });
  if (selected == modes.end()) {
    std::puts(
        "usage: absolute | outside | resolved | opaque-declared | opaque-unknown | trap | "
        "opaque-destination"
        " | opaque-destination-unknown"
        " | stitched | refuse-route | refuse-target"
        " | refuse-outside | refuse-conflict | refuse-predicate | refuse-assumption | "
        "mixed-noreturn");
    return 2;
  }

  const auto chosen = selected->second;
  const bool unresolved_exit = chosen == Mode::absolute || chosen == Mode::outside;
  const bool opaque_exit =
      chosen == Mode::opaque_declared || chosen == Mode::opaque_unknown || chosen == Mode::trap;
  const SourceRecord breakpoint{
      0, {0x20, 0x00, 0x20, 0xd4}, {}, OpaqueReason::unsupported, OpaqueControl::trap};
  std::vector<SourceRecord> sources{
      Record(ir::Group(0x100, {1, 2, 3, 4},
                       {{ir::Op::image_address, 64, {}, 0x4000},
                        {ir::Op::load, 32, {0}},
                        {ir::Op::zext, 64, {1}}},
                       {{8, 2}}, ir::MemoryModel::atomic_scalar_reference)),
      Guard(0x104, 0x900, 0x108), Dispatch(0x108), Jump(0x900, 0x900, true),
      Jump(0x9010, 0x9010, true)};
  // Case 1: leaves through a jump the graph resolves only to an absolute
  // runtime value (or to an address outside the population).
  if (chosen == Mode::opaque_destination || chosen == Mode::opaque_destination_unknown) {
    sources.push_back(breakpoint);
    sources.back().address = 0x9020;
    if (chosen == Mode::opaque_destination_unknown)
      sources.back().opaque_control = OpaqueControl::unknown;
  } else {
    sources.push_back(opaque_exit                ? Jump(0x9020, 0x9050, true)
                      : !unresolved_exit         ? Jump(0x9020, 0x9020, true)
                      : chosen == Mode::absolute ? Jump(0x9020, 0x108, false)
                                                 : Jump(0x9020, 0x100000, true));
  }

  sources.push_back(Jump(0x9030, 0x9030, true));
  sources.push_back(Jump(0x9040, 0x9040, true));
  if (chosen == Mode::trap) {
    sources.push_back(breakpoint);
    sources.back().address = 0x9050;
  }

  if (chosen == Mode::opaque_declared || chosen == Mode::opaque_unknown) {
    sources.push_back({0x9050,
                       {0x0a, 0xfc, 0x5f, 0x88},
                       {},
                       OpaqueReason::unsupported,
                       chosen == Mode::opaque_declared ? OpaqueControl::normal_fallthrough
                                                       : OpaqueControl::unknown});
    sources.push_back(Jump(0x9054, 0x9054, true));
  }

  const std::uint64_t entries[] = {0x100};
  const std::array<ConstantImageRange, 2> ranges{ConstantImageRange{0x4000, kState},
                                                 ConstantImageRange{0x5000, kEntries}};
  const ImageFacts facts{ranges, {}, false, {}};
  Budget budget({UINT64_MAX, UINT64_MAX});
  auto graph = BuildCfg(sources, entries, budget, {}, facts);
  if (!graph.cfg) {
    std::puts("cfg declined");
    return 1;
  }

  auto built = BuildRegions(std::move(*graph.cfg), budget);
  if (!built.regions) {
    std::puts("regions declined");
    return 1;
  }

  auto regions = std::move(built.regions);
  std::vector<std::optional<PathControlFacts>> control(regions->candidates().size());
  for (std::size_t i = 0; i < control.size(); ++i) {
    auto path = NormalizeRegion(*regions, i, budget);
    if (!path.path) continue;
    auto c = AnalyzePathControl(*path.path, budget, {}, facts);
    if (c.facts) control[i] = *c.facts;
  }

  const auto& g = regions->graph();
  auto addr = [&](std::uint32_t b) { return g.sources()[g.blocks()[b].first_source].address; };
  const auto candidates = regions->candidates();

  // Set only where a mode needs a candidate list the selection will not produce.
  std::optional<Regions> twinned;

  // The candidate that starts at `entry` and stops at the dispatch at `at`.
  const auto candidate = [&](std::uint64_t entry, std::uint64_t at) {
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      if (candidates[i].stop == RegionStop::dispatch && addr(candidates[i].entry_block) == entry &&
          !candidates[i].block_ids.empty() && addr(candidates[i].block_ids.back()) == at)
        return i;
    }

    return candidates.size();
  };

  const auto route = candidate(0x100, 0x108);
  if (route == candidates.size() || !control[route]) {
    std::puts("FAIL: the fixture no longer produces a proved route to the dispatch");
    return 1;
  }

  // Each of these makes one fact of that route wrong, in the one way the rule
  // under test is meant to catch.
  if (chosen == Mode::refuse_route) {
    if (control[route]->boundaries.size() < 2) {
      std::puts("FAIL: the route has no internal transfer to leave unproved");
      return 1;
    }

    control[route]->boundaries.front().expected_match = ExpectedMatch::unknown;
  }

  if (chosen == Mode::refuse_outside) {
    auto& terminal = control[route]->boundaries.back();
    if (terminal.edge_count == 0) {
      std::puts("FAIL: the route resolves no destination");
      return 1;
    }

    // A block of this graph, but not one the dispatch can reach.
    terminal.edges[0].target_address = 0x900;
  }

  if (chosen == Mode::refuse_conflict || chosen == Mode::refuse_predicate ||
      chosen == Mode::refuse_assumption) {
    // One entry with two routes to a dispatch is not something this selection
    // produces: reaching a second dispatch needs a second fork, which stops the
    // region, and reaching the same one twice leaves it with another way in, so
    // it carries no dispatch record at all. The second candidate is therefore
    // supplied here, claiming the same proved route and a different answer
    // about where the entry goes. Neither may be published: they cannot both be
    // the route, and the rule is all that stands between them.
    std::vector<Region> both(candidates.begin(), candidates.end());
    both.push_back(both[route]);
    twinned.emplace(g, std::move(both));
    control.push_back(control[route]);
    auto& terminal = control.back()->boundaries.back();
    if (chosen == Mode::refuse_conflict) terminal.edges[0].target_address = 0x9030;
    if (chosen == Mode::refuse_predicate) {
      auto& first = control[route]->boundaries.back();
      first.edges[1] = first.edges[0];
      first.edge_count = 2;
      first.edges[0].condition = 1;
      first.edges[0].when = true;
      first.edges[1].target_address = 0x9030;
      first.edges[1].condition = 1;
      first.edges[1].when = false;
      terminal = first;
      terminal.edges[0].condition = 2;
      terminal.edges[1].condition = 2;
    }

    if (chosen == Mode::refuse_assumption) {
      for (auto& boundary : control[route]->boundaries) {
        for (unsigned i = 0; i < boundary.edge_count; ++i)
          boundary.edges[i].constant_image_dependency = false;
      }

      control.back() = control[route];
      control.back()->boundaries.back().edges[0].constant_image_dependency = true;
    }
  }

  auto result = Unflatten(twinned ? *twinned : *regions, control, budget);
  if (!result.unflattening) {
    std::puts("unflatten declined");
    return 1;
  }

  const auto& u = *result.unflattening;
  const bool dispatch_retired = std::any_of(u.retired_blocks.begin(), u.retired_blocks.end(),
                                            [&](std::uint32_t b) { return addr(b) == 0x108; });
  const auto refused_as = [&](std::size_t candidate, TransitionRefusal reason) {
    return std::any_of(u.refused.begin(), u.refused.end(), [&](const RefusedTransition& refusal) {
      return refusal.candidate == candidate && refusal.reason == reason;
    });
  };

  const auto transition_from = [&](std::uint64_t entry) {
    return std::any_of(
        u.transitions.begin(), u.transitions.end(),
        [&](const Transition& transition) { return addr(transition.entry_block) == entry; });
  };

  switch (chosen) {
    case Mode::opaque_declared: {
      const auto opaque =
          std::find_if(u.graph.blocks.begin(), u.graph.blocks.end(),
                       [](const RecoveredBlock& block) { return block.address == 0x9050; });
      if (!u.unresolved_successor_blocks.empty() || !dispatch_retired ||
          opaque == u.graph.blocks.end() || opaque->edges.size() != 1 ||
          !opaque->edges[0].assumptions.declared_opaque_control) {
        std::puts("FAIL: declared opaque successor must qualify retirement and its edge");
        return 1;
      }

      break;
    }
    case Mode::trap:
    case Mode::opaque_destination: {
      const std::uint64_t at = chosen == Mode::trap ? 0x9050 : 0x9020;
      const auto trap =
          std::find_if(u.graph.blocks.begin(), u.graph.blocks.end(),
                       [&](const RecoveredBlock& block) { return block.address == at; });
      if (!u.unresolved_successor_blocks.empty() || !dispatch_retired || !transition_from(0x100) ||
          trap == u.graph.blocks.end() || trap->edges.size() != 1 ||
          trap->edges[0].kind != CfgEdgeKind::trap || trap->edges[0].target_block ||
          trap->edges[0].assumptions.unresolved_target ||
          !trap->edges[0].assumptions.declared_opaque_control) {
        std::puts("FAIL: a declared trap must end its path without an unresolved successor");
        return 1;
      }

      break;
    }
    case Mode::opaque_destination_unknown: {
      // The destination is taken, but where control goes after it is not known.
      const bool names_opaque =
          std::any_of(u.unresolved_successor_blocks.begin(), u.unresolved_successor_blocks.end(),
                      [&](std::uint32_t b) { return addr(b) == 0x9020; });
      if (!transition_from(0x100) || !names_opaque || !u.retired_blocks.empty()) {
        std::puts("FAIL: an opaque destination of unknown control must prevent retirement");
        return 1;
      }

      break;
    }
    case Mode::opaque_unknown: {
      const bool names_opaque =
          std::any_of(u.unresolved_successor_blocks.begin(), u.unresolved_successor_blocks.end(),
                      [&](std::uint32_t b) { return addr(b) == 0x9050; });
      if (!names_opaque || !u.retired_blocks.empty()) {
        std::puts("FAIL: unknown opaque successor must prevent retirement");
        return 1;
      }

      break;
    }
    case Mode::resolved:
      if (!u.unresolved_successor_blocks.empty() || !dispatch_retired) {
        std::printf("FAIL: a fully resolved graph must retire the dispatcher (unresolved=%zu)\n",
                    u.unresolved_successor_blocks.size());
        return 1;
      }

      break;
    case Mode::absolute:
    case Mode::outside: {
      const bool names_case =
          std::any_of(u.unresolved_successor_blocks.begin(), u.unresolved_successor_blocks.end(),
                      [&](std::uint32_t b) { return addr(b) == 0x9020; });
      if (!u.retired_blocks.empty() || !names_case) {
        std::printf("FAIL: an exit not resolved to a block must block retirement (retired=%zu)\n",
                    u.retired_blocks.size());
        return 1;
      }

      break;
    }
    case Mode::stitched: {
      if (!dispatch_retired || u.graph.blocks.empty()) {
        std::puts("FAIL: the stitched graph must be published with the dispatcher retired");
        return 1;
      }

      // Every block it lists survived, and nothing it names is gone.
      for (const auto& block : u.graph.blocks) {
        if (std::find(u.retired_blocks.begin(), u.retired_blocks.end(), block.block) !=
            u.retired_blocks.end()) {
          std::printf("FAIL: the stitched graph lists the retired block %llx\n",
                      (unsigned long long)block.address);
          return 1;
        }

        for (const auto& edge : block.edges) {
          if (!edge.target_block) continue;
          const auto listed = std::any_of(
              u.graph.blocks.begin(), u.graph.blocks.end(),
              [&](const RecoveredBlock& other) { return other.block == *edge.target_block; });
          if (!listed) {
            std::printf("FAIL: an edge of %llx names %llx, which the graph does not list\n",
                        (unsigned long long)block.address,
                        (unsigned long long)addr(*edge.target_block));
            return 1;
          }
        }
      }

      // Reached from the entries it publishes, with nothing stranded.
      std::vector<std::uint32_t> stack(u.graph.entries.begin(), u.graph.entries.end());
      std::vector<std::uint32_t> seen = stack;
      while (!stack.empty()) {
        const auto id = stack.back();
        stack.pop_back();
        const auto at =
            std::find_if(u.graph.blocks.begin(), u.graph.blocks.end(),
                         [&](const RecoveredBlock& block) { return block.block == id; });
        if (at == u.graph.blocks.end()) continue;
        for (const auto& edge : at->edges) {
          if (!edge.target_block ||
              std::find(seen.begin(), seen.end(), *edge.target_block) != seen.end())
            continue;
          seen.push_back(*edge.target_block);
          stack.push_back(*edge.target_block);
        }
      }

      if (seen.size() != u.graph.blocks.size()) {
        std::printf("FAIL: %zu of %zu blocks are reachable from the published entries\n",
                    seen.size(), u.graph.blocks.size());
        return 1;
      }

      // A transition's source leaves straight for its destinations, and says so.
      for (const auto& block : u.graph.blocks) {
        if (!block.transition) continue;
        const auto& transition = u.transitions[*block.transition];
        if (block.block != transition.entry_block ||
            block.edges.size() != transition.destinations.size()) {
          std::printf("FAIL: %llx does not carry its transition's successors\n",
                      (unsigned long long)block.address);
          return 1;
        }

        for (std::size_t i = 0; i < block.edges.size(); ++i) {
          const auto& edge = block.edges[i];
          const bool guarded = transition.condition.has_value();
          if (edge.target_block != transition.destinations[i] ||
              edge.condition != transition.condition || edge.when.has_value() != guarded ||
              edge.condition_owner != ConditionOwner::recovered_path ||
              (guarded && *edge.when != (i == 0)) ||
              edge.assumptions.constant_image != transition.constant_image_dependency) {
            std::printf("FAIL: edge %zu of %llx does not match its transition\n", i,
                        (unsigned long long)block.address);
            return 1;
          }
        }
      }

      break;
    }
    case Mode::refuse_route:
      if (!refused_as(route, TransitionRefusal::unproved_route) || transition_from(0x100) ||
          dispatch_retired) {
        std::puts("FAIL: a route with an unproved internal transfer must publish no transition");
        return 1;
      }

      break;
    case Mode::refuse_target: {
      // The candidate that starts at the dispatch itself has no state store on
      // its path, so its jump resolves to nothing.
      const auto bare = [&]() -> std::size_t {
        for (std::size_t i = 0; i < candidates.size(); ++i) {
          if (candidates[i].stop == RegionStop::dispatch &&
              addr(candidates[i].entry_block) == 0x108)
            return i;
        }

        return candidates.size();
      }();
      if (bare == candidates.size()) {
        std::puts("FAIL: no candidate starts at the dispatch");
        return 1;
      }

      if (!refused_as(bare, TransitionRefusal::unresolved_target) || transition_from(0x108)) {
        std::puts("FAIL: a dispatch whose target is unresolved must publish no transition");
        return 1;
      }

      break;
    }
    case Mode::refuse_outside:
      if (!refused_as(route, TransitionRefusal::outside_dispatch_set) || transition_from(0x100) ||
          dispatch_retired) {
        std::puts(
            "FAIL: a destination outside the dispatch's own edges must publish no transition");
        return 1;
      }

      break;
    case Mode::refuse_conflict:
    case Mode::refuse_predicate:
    case Mode::refuse_assumption:
      if (transition_from(0x100) || dispatch_retired) {
        std::puts("FAIL: routes from one entry that disagree must publish no transition");
        return 1;
      }

      break;
  }

  std::printf("ok: %.*s\n", static_cast<int>(mode.size()), mode.data());
  return 0;
}
