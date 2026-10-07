#include "nyx/eval/program.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace nyx::eval {
namespace {

bool Fits(std::uint64_t address, std::uint64_t size) {
  return size != 0 && size - 1 <= std::numeric_limits<std::uint64_t>::max() - address;
}

unsigned SearchWork(std::size_t count) {
  unsigned result = 1;
  while (count != 0) {
    ++result;
    count >>= 1;
  }

  return result;
}

ProgramStatus Convert(Outcome outcome) {
  switch (outcome) {
    case Outcome::completed:
      return ProgramStatus::exit;
    case Outcome::fault:
      return ProgramStatus::fault;
    case Outcome::resource_limit:
      return ProgramStatus::resource_limit;
    case Outcome::invalid_state:
      return ProgramStatus::invalid_state;
    case Outcome::invalid_group:
      return ProgramStatus::invalid_program;
    case Outcome::unsupported:
      return ProgramStatus::unsupported;
  }

  return ProgramStatus::invalid_program;
}

}  // namespace

ProgramResult Run(std::span<const ir::Group> groups, State& state, Memory& memory,
                  std::uint64_t runtime_entry,
                  std::span<const std::uint64_t> runtime_exit_addresses, Budget& budget,
                  ProgramLimits limits, ExecutionContext context) {
  ProgramResult result{ProgramStatus::invalid_program, runtime_entry, 0, {}};
  if (!context.load_bias) {
    result.status = ProgramStatus::unsupported;
    return result;
  }

  if (groups.size() > limits.max_groups || runtime_exit_addresses.size() > limits.max_groups) {
    result.status = ProgramStatus::resource_limit;
    return result;
  }

  const auto bias = *context.load_bias;
  const auto charge = [&](std::uint64_t work, std::uint64_t bytes = 0) {
    if (budget.try_consume({work, bytes}) == BudgetDecline::none) return true;
    result.status = ProgramStatus::resource_limit;
    return false;
  };

  if (!charge(groups.size() * SearchWork(groups.size()) +
                  runtime_exit_addresses.size() * SearchWork(runtime_exit_addresses.size()),
              groups.size() * sizeof(const ir::Group*) +
                  runtime_exit_addresses.size() * sizeof(std::uint64_t))) {
    return result;
  }

  std::vector<const ir::Group*> ordered;
  ordered.reserve(groups.size());
  for (const auto& group : groups) {
    if (!Fits(group.source_address(), group.bytes().size()) ||
        !Fits(group.source_address() + bias, group.bytes().size()))
      return result;
    ordered.push_back(&group);
  }

  const auto location = [&](const ir::Group* group) { return group->source_address() + bias; };
  std::sort(ordered.begin(), ordered.end(),
            [&](const auto* a, const auto* b) { return location(a) < location(b); });
  for (std::size_t i = 1; i < ordered.size(); ++i) {
    if (location(ordered[i]) - location(ordered[i - 1]) < ordered[i - 1]->bytes().size()) {
      return result;
    }
  }

  const auto containing = [&](std::uint64_t pc) -> const ir::Group* {
    auto after = std::upper_bound(
        ordered.begin(), ordered.end(), pc,
        [&](auto address, const auto* group) { return address < location(group); });
    if (after == ordered.begin()) return nullptr;
    const auto* group = *std::prev(after);
    return pc - location(group) < group->bytes().size() ? group : nullptr;
  };

  std::vector<std::uint64_t> exits(runtime_exit_addresses.begin(), runtime_exit_addresses.end());
  std::sort(exits.begin(), exits.end());
  if (!charge(exits.size() * SearchWork(ordered.size()))) return result;
  for (std::size_t i = 0; i < exits.size(); ++i) {
    if ((i != 0 && exits[i] == exits[i - 1]) || containing(exits[i]) != nullptr) return result;
  }

  // Code/data aliases are checked before executing anything. Matching readonly
  // bytes are safe; a writable alias would invalidate the immutable lift arena.
  const auto regions = memory.Regions();
  if (!charge(ordered.size() + regions.size())) return result;
  std::size_t group_index = 0, region_index = 0;
  while (group_index < ordered.size() && region_index < regions.size()) {
    const auto* group = ordered[group_index];
    const auto start = location(group);
    const auto end = start + (group->bytes().size() - 1);
    const auto& region = regions[region_index];
    const auto region_end = region.address + (region.bytes.size() - 1);
    if (start <= region_end && region.address <= end) {
      if (region.writable) return result;
      const auto first = std::max(start, region.address);
      const auto last = std::min(end, region_end);
      if (!charge(last - first + 1)) return result;
      for (std::uint64_t i = 0; i <= last - first; ++i) {
        if (group->bytes()[first - start + i] != region.bytes[first - region.address + i])
          return result;
      }
    }

    if (end <= region_end)
      ++group_index;
    else
      ++region_index;
  }

  while (true) {
    if (!charge(SearchWork(exits.size()) + SearchWork(ordered.size()))) return result;
    if (std::binary_search(exits.begin(), exits.end(), result.runtime_pc)) {
      result.status = ProgramStatus::exit;
      return result;
    }

    const auto* group = containing(result.runtime_pc);
    if (group == nullptr || location(group) != result.runtime_pc) {
      result.status = ProgramStatus::unresolved;
      return result;
    }

    if (result.committed_steps == limits.max_steps) {
      result.status = ProgramStatus::step_limit;
      return result;
    }

    if (result.trace.size() == result.trace.capacity()) {
      const std::uint64_t capacity = std::min<std::uint64_t>(
          limits.max_steps, std::max<std::uint64_t>(1, 2 * result.trace.capacity()));
      if (!charge(result.trace.size() + 1, capacity * sizeof(ProgramStep))) return result;
      result.trace.reserve(static_cast<std::size_t>(capacity));
    }

    const auto pc = result.runtime_pc;
    auto step = Execute(*group, state, memory, budget, limits.instruction_limits, context);
    const auto outcome = step.outcome;
    std::optional<std::uint64_t> target;
    if (step.transfer) target = step.transfer->target;
    result.trace.push_back({pc, std::move(step)});
    if (outcome == Outcome::completed || outcome == Outcome::fault) ++result.committed_steps;
    if (outcome != Outcome::completed) {
      result.status = Convert(outcome);
      return result;
    }

    if (target) {
      result.runtime_pc = *target;
    } else {
      // Machine addresses, including fallthrough, use modulo-64 arithmetic.
      result.runtime_pc += group->bytes().size();
    }
  }
}

}  // namespace nyx::eval
