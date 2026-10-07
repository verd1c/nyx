#include "nyx/analysis/image_writes.hpp"

#include <limits>

namespace nyx::analysis {
namespace {

bool Charge(Budget& budget, std::size_t count, std::size_t size) {
  return count <= std::numeric_limits<std::uint64_t>::max() / (size ? size : 1) &&
         budget.try_consume({count, count * size}) == BudgetDecline::none;
}

bool ChargeGroup(const ir::Group& group, Budget& budget) {
  return Charge(budget, group.bytes().size(), 1) &&
         Charge(budget, group.nodes().size(), sizeof(ir::Node)) &&
         Charge(budget, group.writes().size(), sizeof(ir::Write));
}

// Long runs buy nothing: an address is formed from a page and an offset a few
// instructions apart, and normalizing a whole function to find that costs the
// rest of the function for nothing.
constexpr std::size_t kMaxRun = 64;

}  // namespace

std::optional<ImageWriterReport> ScanImageWrites(std::span<const SourceRecord> sources,
                                                 const ImageFacts& facts, Budget& budget,
                                                 ir::BlockLimits limits) {
  ImageWriterReport report;
  if (facts.constants.empty() && facts.pointers.empty()) return report;
  const auto run_limit = std::min<std::size_t>(kMaxRun, limits.max_groups);
  if (run_limit == 0) return report;
  std::vector<ir::Group> run;

  // One entry per group of the run, so a node can be traced back to the
  // instruction it came from.
  std::vector<std::uint64_t> run_addresses;
  std::uint64_t next_address = 0;

  const auto flush = [&]() -> bool {
    if (run.empty()) return true;
    auto normalized = ir::Normalize(run, budget, limits);
    const auto addresses = std::move(run_addresses);
    run.clear();
    run_addresses.clear();

    // A run this abstraction cannot normalize is one it did not examine, which
    // is the same admission as an unmodeled group.
    if (!normalized.block) {
      if (normalized.reason == ir::BlockDecline::resource_limit) return false;
      if (report.unmodeled_groups == std::numeric_limits<std::uint64_t>::max()) return false;
      ++report.unmodeled_groups;
      return true;
    }

    auto scan = KnownImageWrites(*normalized.block, facts, budget);
    if (!scan) return false;
    if (scan->unresolved > std::numeric_limits<std::uint64_t>::max() - report.unresolved_writes ||
        budget.try_consume({scan->writes.size(), scan->writes.size() * sizeof(ir::ImageWrite)}) !=
            BudgetDecline::none)
      return false;
    report.unresolved_writes += scan->unresolved;
    const auto boundaries = normalized.block->boundaries();
    for (const auto& write : scan->writes) {
      std::uint64_t instruction = addresses.empty() ? 0 : addresses.front();
      for (std::size_t index = 0; index < boundaries.size() && index < addresses.size(); ++index) {
        if (write.operation >= boundaries[index].first_node &&
            write.operation < boundaries[index].first_node + boundaries[index].node_count)
          instruction = addresses[index];
      }

      report.write_instructions.push_back(instruction);
    }

    report.writes.insert(report.writes.end(), scan->writes.begin(), scan->writes.end());
    return true;
  };

  for (const auto& record : sources) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return {};
    if (!record.semantics) {
      if (!flush()) return {};
      if (report.unmodeled_groups == std::numeric_limits<std::uint64_t>::max()) return {};
      ++report.unmodeled_groups;
      next_address = 0;
      continue;
    }

    if (!run.empty() && record.address != next_address) {
      if (!flush()) return {};
    }

    if (report.scanned_groups == std::numeric_limits<std::uint64_t>::max()) return {};
    ++report.scanned_groups;
    const bool transfer = record.semantics->transfer().has_value();
    if (!ChargeGroup(*record.semantics, budget)) return {};
    run.push_back(*record.semantics);
    run_addresses.push_back(record.address);
    next_address = record.address + record.bytes.size();
    if (transfer || run.size() >= run_limit) {
      if (!flush()) return {};
      next_address = 0;
    }
  }

  if (!flush()) return {};
  return report;
}

}  // namespace nyx::analysis
