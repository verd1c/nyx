#include <algorithm>
#include <array>
#include <iostream>
#include <sstream>
#include <string_view>

#include "nyx/analysis/path_control.hpp"
#include "nyx/analysis/regions.hpp"
#include "nyx/eval/path.hpp"
#include "nyx/eval/program.hpp"
#include "nyx/ir/print.hpp"
#include "nyx/ir/recovered_path.hpp"
#include "nyx/recovery/control.hpp"
#include "nyx/recovery/mba.hpp"
#include "nyx/recovery/memory.hpp"
#include "nyx/target/a64/decode.hpp"

namespace {
bool Read(std::span<std::uint8_t> bytes) {
  std::cin.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  return static_cast<bool>(std::cin);
}

std::optional<std::uint64_t> Read64() {
  std::array<std::uint8_t, 8> bytes{};
  if (!Read(bytes)) return {};
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i) value |= std::uint64_t(bytes[i]) << (8 * i);
  return value;
}

void Hex(std::span<const std::uint8_t> bytes) {
  constexpr char digits[] = "0123456789abcdef";
  for (const auto byte : bytes) std::cout << digits[byte >> 4] << digits[byte & 15];
}

std::string Topology(const nyx::analysis::Cfg& graph) {
  std::ostringstream out;
  const auto optional = [&](const auto& value) {
    out << value.has_value() << ':';
    if (value) out << *value;
    out << ',';
  };

  out << graph.generation() << ',' << graph.passes() << ';';
  for (auto entry : graph.entries()) out << entry << ',';
  out << ';';
  for (const auto& source : graph.sources()) {
    out << source.address << ':' << static_cast<unsigned>(source.opaque_reason) << ':';
    for (auto byte : source.bytes) out << static_cast<unsigned>(byte) << ',';
    out << ';';
  }

  for (const auto& block : graph.blocks()) {
    out << block.first_source << ',' << block.source_count << ',' << block.control.has_value()
        << ';';
    if (block.control)
      out << block.control->block_revision << ',' << block.control->terminal_source << ','
          << static_cast<unsigned>(block.control->kind) << ','
          << block.control->callee_return_unknown << ';';
    for (const auto& edge : block.edges) {
      out << static_cast<unsigned>(edge.kind) << ',' << static_cast<unsigned>(edge.target.kind)
          << ',' << edge.target.address << ',' << static_cast<unsigned>(edge.resolution) << ',';
      optional(edge.target.value);
      optional(edge.target_source);
      optional(edge.target_block);
      optional(edge.condition);
      optional(edge.when);
    }

    out << ';';
  }

  return out.str();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string_view mode = argv[1];
  if (mode != "original" && mode != "normalized" && mode != "simplified" &&
      mode != "snapshot_mutant" && mode != "control_recovered" && mode != "wrong_arm_mutant" &&
      mode != "forged_witness")
    return 2;
  std::array<std::uint8_t, 8> magic{};
  if (!Read(magic)) return 2;
  const std::string_view version(reinterpret_cast<const char*>(magic.data()), 8);
  if (version != "NYXSPR01" && version != "NYXSPR02") return 2;
  std::array<std::uint64_t, 39> header{};
  for (auto& value : header) {
    const auto input = Read64();
    if (!input) return 2;
    value = *input;
  }

  std::uint64_t scratch_size = 4096;
  if (version == "NYXSPR02") {
    const auto size = Read64();
    if (!size || *size != 8192) return 2;
    scratch_size = *size;
  }

  const auto flags = header[31], sp = header[32], scratch_base = header[33], bias = header[34];
  const auto entry = header[35], fragment_count = header[36], exit_count = header[37],
             page_count = header[38];
  if ((flags & ~UINT64_C(0xf0000000)) || !fragment_count || fragment_count > 16 || !exit_count ||
      exit_count > 16 || page_count > 8 || scratch_base % 4096)
    return 2;
  nyx::Budget budget({100000000, 256ULL * 1024 * 1024});
  std::vector<nyx::ir::Group> groups;
  std::size_t total = 0;
  for (std::size_t i = 0; i < fragment_count; ++i) {
    const auto source = Read64(), size = Read64();
    if (!source || !size || !*size || *size % 4 || *source % 4 || *size > 4096 - total ||
        *size - 1 > UINT64_MAX - *source || *size - 1 > UINT64_MAX - (*source + bias))
      return 2;
    total += *size;
    for (std::size_t offset = 0; offset < *size; offset += 4) {
      std::array<std::uint8_t, 4> word{};
      if (!Read(word)) return 2;
      auto decoded = nyx::a64::Decode(*source + offset, word, budget,
                                      {nyx::a64::MemoryProfile::concrete_atomic_scalar});
      if (!decoded.group) return 3;
      groups.push_back(std::move(*decoded.group));
    }
  }

  std::vector<std::uint64_t> exits;
  for (std::size_t i = 0; i < exit_count; ++i) {
    const auto address = Read64();
    if (!address) return 2;
    exits.push_back(*address + bias);
  }

  std::vector<std::uint8_t> scratch(scratch_size);
  if (!Read(scratch)) return 2;
  std::vector<std::array<std::uint8_t, 4096>> pages(page_count);
  std::vector<std::uint64_t> addresses;
  for (auto& page : pages) {
    const auto address = Read64();
    if (!address || *address % 4096 || !Read(page)) return 2;
    addresses.push_back(*address);
  }

  if (std::cin.get() != std::char_traits<char>::eof()) return 2;
  std::vector<nyx::eval::RegionInput> memory_regions{{scratch_base, scratch, true, true}};
  for (std::size_t i = 0; i < pages.size(); ++i)
    memory_regions.push_back({addresses[i], pages[i], true, false});
  auto memory = nyx::eval::Memory::Create(memory_regions, budget);
  if (!memory.memory) return 4;
  nyx::eval::State state;
  for (unsigned i = 0; i < 36; ++i) {
    const auto initial = i < 31 ? header[i] : i == 31 ? sp : (flags >> (63 - i)) & 1;
    auto value = nyx::BitVector::from_u64(i < 32 ? 64 : 1, initial, 64, budget);
    if (!value) return 4;
    state.cells.push_back({i, std::move(*value)});
  }

  std::uint64_t pc = entry + bias;
  bool completed = false;
  std::vector<nyx::eval::ExecutionResult> trace;
  std::size_t facts = 0, memory_edits = 0, arithmetic_edits = 0, control_edits = 0;
  std::optional<nyx::ir::RecoveredPath> recovered;
  std::optional<nyx::ir::Path> analysis_path;
  std::vector<nyx::analysis::SourceRecord> population;
  for (const auto& group : groups)
    population.push_back({group.source_address(),
                          {group.bytes().begin(), group.bytes().end()},
                          group,
                          nyx::analysis::OpaqueReason::none});
  const std::array<std::uint64_t, 1> entries{entry};
  auto graph = nyx::analysis::BuildCfg(population, entries, budget);
  if (!graph.cfg) return 13;
  const auto original_topology = Topology(*graph.cfg);
  std::vector<std::string> original_ssa;
  for (const auto& block : graph.cfg->blocks()) {
    if (!block.ssa) return 13;
    auto json = nyx::ir::PrintJson(*block.ssa, budget);
    if (!json.json) return 13;
    original_ssa.push_back(std::move(*json.json));
  }

  auto regions_result = nyx::analysis::BuildRegions(std::move(*graph.cfg), budget);
  if (!regions_result.regions) return 14;
  const auto& regions = *regions_result.regions;
  std::optional<std::size_t> selected;
  std::size_t entry_candidates = 0;
  for (std::size_t i = 0; i < regions.candidates().size(); ++i) {
    const auto& candidate = regions.candidates()[i];
    const auto& block = regions.graph().blocks()[candidate.entry_block];
    if (regions.graph().sources()[block.first_source].address != entry) continue;
    ++entry_candidates;
    if (!selected ||
        candidate.source_ids.size() > regions.candidates()[*selected].source_ids.size())
      selected = i;
  }

  if (!selected) return 15;
  const auto& selection = regions.candidates()[*selected];
  if (mode == "original") {
    auto result = nyx::eval::Run(groups, state, *memory.memory, pc, exits, budget, {}, {bias});
    completed = result.status == nyx::eval::ProgramStatus::exit;
    if (!completed && result.status != nyx::eval::ProgramStatus::fault) return 5;
    pc = result.runtime_pc;
    for (auto& step : result.trace) trace.push_back(std::move(step.execution));
  } else {
    auto normalized = nyx::analysis::NormalizeRegion(regions, *selected, budget);
    if (!normalized.path) return 6;
    auto path = std::move(*normalized.path);
    if (mode == "simplified" || mode == "snapshot_mutant" || mode == "control_recovered" ||
        mode == "wrong_arm_mutant" || mode == "forged_witness") {
      auto forwarded = nyx::recovery::ForwardMemoryValues(path, budget);
      if (!forwarded.path) return 7;
      facts = forwarded.facts.size();
      memory_edits = forwarded.journal.size();
      auto simplified = nyx::recovery::SimplifyMba(*forwarded.path, budget);
      if (!simplified.path) return 8;
      arithmetic_edits = simplified.journal.size();
      path = std::move(*simplified.path);
    }

    if (mode == "snapshot_mutant") {
      const auto transfer = path.boundaries().back().transfer;
      if (!transfer) return 12;
      auto nodes = std::vector<nyx::ir::Node>(path.nodes().begin(), path.nodes().end());
      auto& target = nodes[transfer->target];
      if (target.width != 64 || nyx::ir::Descriptor(target.op)->effect != nyx::ir::Effect::pure)
        return 12;
      // Deliberately wrong recovery: substitute a snapshot destination while
      // retaining the runtime table load and all its faults.
      target = {nyx::ir::Op::image_address, 64, {}, exits.front() - bias};
      path =
          nyx::ir::Path({path.sources().begin(), path.sources().end()}, std::move(nodes),
                        {path.origins().begin(), path.origins().end()},
                        {path.boundaries().begin(), path.boundaries().end()}, path.revision() + 1);
    }

    if (mode == "wrong_arm_mutant") {
      auto nodes = std::vector<nyx::ir::Node>(path.nodes().begin(), path.nodes().end());
      bool changed = false;
      for (const auto& boundary : path.boundaries()) {
        if (!boundary.transfer || boundary.transfer->kind != nyx::ir::TransferKind::conditional)
          continue;
        auto& condition = nodes[*boundary.transfer->condition];
        if (condition.op != nyx::ir::Op::constant || condition.width != 1) continue;
        condition.immediate ^= 1;
        changed = true;
        break;
      }

      if (!changed) return 18;

      // Deliberately corrupt the semantic basis, then obtain a valid certificate
      // for that wrong basis. This is distinct from forging a certificate.
      path =
          nyx::ir::Path({path.sources().begin(), path.sources().end()}, std::move(nodes),
                        {path.origins().begin(), path.origins().end()},
                        {path.boundaries().begin(), path.boundaries().end()}, path.revision() + 1);
    }

    if (mode == "control_recovered" || mode == "wrong_arm_mutant" || mode == "forged_witness") {
      auto folded = nyx::recovery::RecoverControl(std::move(path), {}, budget);
      if (!folded.path) return 18;
      recovered = std::move(*folded.path);
      control_edits = recovered->rewrites().size();
      if (mode == "forged_witness") {
        if (recovered->rewrites().empty()) return 18;
        std::vector<nyx::ir::ConditionalRewrite> rewrites(recovered->rewrites().begin(),
                                                          recovered->rewrites().end());
        auto& rewrite = rewrites.front();
        rewrite.replacement.target =
            rewrite.condition_value ? *rewrite.original.alternative : rewrite.original.target;
        nyx::ir::RecoveredPath forged(recovered->basis(), std::move(rewrites),
                                      recovered->revision());
        if (nyx::ir::ValidateRecoveredPath(forged, budget) != nyx::ir::BlockDecline::invalid_ir)
          return 19;
        std::cout << "{\"forged_witness_rejected\":true,\"native_executed\":false}\n";
        return 0;
      }

      analysis_path = recovered->basis();
    } else
      analysis_path = path;
    auto result =
        recovered
            ? nyx::eval::ExecuteRecoveredPath(*recovered, state, *memory.memory, budget, {}, {bias})
            : nyx::eval::ExecutePath(path, state, *memory.memory, budget, {}, {bias});
    completed = result.outcome == nyx::eval::Outcome::completed;
    if (!completed && result.outcome != nyx::eval::Outcome::fault) return 9;
    if (completed) {
      if (!result.runtime_next || std::ranges::find(exits, *result.runtime_next) == exits.end())
        return 10;
      pc = *result.runtime_next;
    } else
      pc = result.source_address + bias;
    trace = std::move(result.trace);
  }

  if (!analysis_path) {
    auto normalized = nyx::analysis::NormalizeRegion(regions, *selected, budget);
    if (!normalized.path) return 17;
    analysis_path = std::move(*normalized.path);
  }

  auto control = recovered ? nyx::analysis::AnalyzePathControl(
                                 *recovered, {recovered->basis().revision(), {}}, budget)
                           : nyx::analysis::AnalyzePathControl(*analysis_path, budget);
  auto printed_path = recovered ? nyx::ir::PrintJson(*recovered, budget)
                                : nyx::ir::PrintJson(*analysis_path, budget);
  if (!control.facts || !printed_path.json) return 17;
  if (Topology(regions.graph()) != original_topology) return 16;
  for (std::size_t i = 0; i < regions.graph().blocks().size(); ++i) {
    auto printed = nyx::ir::PrintJson(*regions.graph().blocks()[i].ssa, budget);
    if (!printed.json || *printed.json != original_ssa[i]) return 16;
  }

  std::uint64_t output_flags = 0;
  for (unsigned i = 32; i < 36; ++i) output_flags |= state.cells[i].value.word(0) << (63 - i);
  std::cout << "{\"completed\":" << (completed ? "true" : "false") << ",\"pc\":" << pc
            << ",\"registers\":[";
  for (unsigned i = 0; i < 31; ++i) {
    if (i) std::cout << ',';
    std::cout << state.cells[i].value.word(0);
  }

  std::cout << "],\"nzcv\":" << output_flags << ",\"sp\":" << state.cells[31].value.word(0)
            << ",\"memory\":\"";
  const auto output_regions = memory.memory->Regions();
  for (const auto& region : output_regions)
    if (region.address == scratch_base) Hex(region.bytes);
  std::cout << "\",\"global_pages\":[";
  for (std::size_t i = 0; i < addresses.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << "{\"address\":" << addresses[i] << ",\"data\":\"";
    for (const auto& region : output_regions)
      if (region.address == addresses[i]) Hex(region.bytes);
    std::cout << "\"}";
  }

  std::cout << "],\"selection_policy\":\"longest_candidate_for_entry\",\"entry_candidates\":"
            << entry_candidates
            << ",\"cfg_ssa_unchanged\":true,\"cfg_topology_unchanged\":true,\"selected_sources\":[";
  for (std::size_t i = 0; i < selection.source_ids.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << regions.graph().sources()[selection.source_ids[i]].address;
  }

  std::cout << "],\"transition_edges\":[";
  for (std::size_t i = 0; i < selection.transition_edges.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << selection.transition_edges[i];
  }

  std::cout << "],\"analyzed_path\":" << *printed_path.json
            << ",\"control\":{\"path_revision\":" << control.facts->path_revision
            << ",\"total_boundaries\":" << control.facts->total_boundaries
            << ",\"proof_scope\":\"successful_itinerary_prefix\",\"proved_divergence\":";
  const auto optional_number = [](const auto& value) {
    if (value)
      std::cout << *value;
    else
      std::cout << "null";
  };

  const auto optional_bool = [](const auto& value) {
    if (value)
      std::cout << (*value ? "true" : "false");
    else
      std::cout << "null";
  };

  optional_number(control.facts->proved_divergence);
  std::cout << ",\"boundaries\":[";
  constexpr const char* matches[] = {"not_applicable", "always", "never", "unknown"};
  constexpr const char* roles[] = {"fallthrough", "branch", "callee", "return", "potential_return"};
  constexpr const char* targets[] = {"unknown", "image_location", "absolute_runtime"};
  for (std::size_t i = 0; i < control.facts->boundaries.size(); ++i) {
    if (i) std::cout << ',';
    const auto& boundary = control.facts->boundaries[i];
    std::cout << "{\"boundary\":" << boundary.boundary
              << ",\"source_address\":" << boundary.source_address << ",\"transfer_target\":";
    optional_number(boundary.transfer_target);
    std::cout << ",\"expected_image_successor\":";
    optional_number(boundary.expected_image_successor);
    std::cout << ",\"expected_match\":\"" << matches[static_cast<unsigned>(boundary.expected_match)]
              << "\",\"callee_return_unknown\":"
              << (boundary.callee_return_unknown ? "true" : "false") << ",\"load_dependencies\":[";
    for (std::size_t j = 0; j < boundary.load_dependencies.size(); ++j) {
      if (j) std::cout << ',';
      std::cout << boundary.load_dependencies[j];
    }

    std::cout << "],\"edges\":[";
    for (unsigned j = 0; j < boundary.edge_count; ++j) {
      if (j) std::cout << ',';
      const auto& edge = boundary.edges[j];
      std::cout << "{\"role\":\"" << roles[static_cast<unsigned>(edge.role)]
                << "\",\"target_kind\":\"" << targets[static_cast<unsigned>(edge.target_kind)]
                << "\",\"target_address\":" << edge.target_address << ",\"target_value\":";
      optional_number(edge.target_value);
      std::cout << ",\"condition\":";
      optional_number(edge.condition);
      std::cout << ",\"when\":";
      optional_bool(edge.when);
      std::cout << ",\"known_condition\":";
      optional_bool(edge.known_condition);
      std::cout << '}';
    }

    std::cout << "]}";
  }

  std::cout << "]},\"forwarding_facts\":" << facts << ",\"memory_edits\":" << memory_edits
            << ",\"arithmetic_edits\":" << arithmetic_edits
            << ",\"control_edits\":" << control_edits << ",\"trace\":[";
  for (std::size_t i = 0; i < trace.size(); ++i) {
    if (i) std::cout << ',';
    const auto& step = trace[i];
    std::cout << "{\"fault_address\":";
    if (step.fault)
      std::cout << step.fault->address;
    else
      std::cout << "null";
    std::cout << ",\"events\":[";
    for (std::size_t j = 0; j < step.events.size(); ++j) {
      if (j) std::cout << ',';
      const auto& event = step.events[j];
      std::cout << "{\"operation\":" << event.operation << ",\"address\":" << event.address
                << ",\"write\":" << (event.write ? "true" : "false")
                << ",\"conditional\":" << (event.conditional ? "true" : "false")
                << ",\"performed\":" << (event.performed ? "true" : "false") << ",\"bytes\":\"";
      Hex(std::span(event.bytes).first(event.size));
      std::cout << "\"}";
    }

    std::cout << "]}";
  }

  std::cout << "]}\n";
  return std::cout ? 0 : 11;
}
