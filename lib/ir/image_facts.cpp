#include "nyx/ir/image_facts.hpp"

#include <algorithm>
#include <array>
#include <bit>

namespace nyx::ir {

namespace {
bool Overlaps(std::uint64_t first, std::uint64_t first_size, std::uint64_t second,
              std::uint64_t second_size) {
  return second - first < first_size || first - second < second_size;
}
}  // namespace

std::optional<ImageFactRefutations> RefuteImageFacts(const ImageFacts& facts,
                                                     std::span<const ImageWrite> writes,
                                                     Budget& budget) {
  const auto charge = [&](std::uint64_t count, std::uint64_t size) {
    return count <= UINT64_MAX / size &&
           budget.try_consume({count, count * size}) == BudgetDecline::none;
  };

  if (!charge(facts.constants.size() + facts.pointers.size(), sizeof(std::uint8_t))) return {};
  std::vector<std::uint8_t> ranges(facts.constants.size());
  std::vector<std::uint8_t> slots(facts.pointers.size());
  ImageFactRefutations result;
  for (const auto& write : writes) {
    if (write.width == 0 || write.width % 8 != 0) continue;
    const auto size = write.width / 8;
    if (budget.try_consume({facts.constants.size() + std::bit_width(facts.pointers.size()) + 2,
                            0}) != BudgetDecline::none)
      return {};
    for (std::size_t range = 0; range < facts.constants.size(); ++range) {
      if (ranges[range] || !Overlaps(write.address, size, facts.constants[range].address,
                                     facts.constants[range].bytes.size()))
        continue;
      ranges[range] = 1;
      if (!charge(1, sizeof(ImageFactRefutation))) return {};
      result.constants.push_back({range, write});
    }

    const auto visit = [&](std::size_t slot) {
      if (budget.try_consume({1, 0}) != BudgetDecline::none) return false;
      if (slots[slot] || !facts.pointers[slot].value_stable ||
          !Overlaps(write.address, size, facts.pointers[slot].address, 8))
        return true;
      slots[slot] = 1;
      if (!charge(1, sizeof(ImageFactRefutation))) return false;
      result.pointers.push_back({slot, write});
      return true;
    };

    const auto begin = facts.pointers.begin();
    const auto first = std::lower_bound(
        begin, facts.pointers.end(), write.address < 7 ? 0 : write.address - 7,
        [](const RelocatedPointer& slot, std::uint64_t key) { return slot.address < key; });
    const bool wraps = size > UINT64_MAX - write.address;
    const auto stop = write.address + size;
    for (auto at = first; at != facts.pointers.end() && (wraps || at->address < stop); ++at) {
      if (!visit(static_cast<std::size_t>(at - begin))) return {};
    }

    if (wraps) {
      const auto wrapped_end = size - (UINT64_MAX - write.address) - 1;
      for (auto at = begin; at != facts.pointers.end() && at->address < wrapped_end; ++at) {
        if (!visit(static_cast<std::size_t>(at - begin))) return {};
      }
    }

    if (write.address < 7) {
      if (budget.try_consume({static_cast<std::uint64_t>(std::bit_width(facts.pointers.size())) + 1,
                              0}) != BudgetDecline::none)
        return {};
      const auto suffix = std::lower_bound(
          begin, facts.pointers.end(), UINT64_MAX - 6,
          [](const RelocatedPointer& slot, std::uint64_t key) { return slot.address < key; });
      for (auto at = suffix; at != facts.pointers.end(); ++at) {
        if (!visit(static_cast<std::size_t>(at - begin))) return {};
      }
    }
  }

  const auto by_index = [](const ImageFactRefutation& a, const ImageFactRefutation& b) {
    return a.index < b.index;
  };

  const auto sort_work = [](std::size_t size) { return size * (std::bit_width(size) + 1); };
  if (budget.try_consume({sort_work(result.constants.size()) + sort_work(result.pointers.size()),
                          0}) != BudgetDecline::none)
    return {};
  std::sort(result.constants.begin(), result.constants.end(), by_index);
  std::sort(result.pointers.begin(), result.pointers.end(), by_index);
  return result;
}

std::optional<RetainedImageFacts> RetainImageFacts(const ImageFacts& facts,
                                                   std::span<const std::size_t> refuted_constants,
                                                   std::span<const std::size_t> refuted_pointers,
                                                   Budget& budget) {
  if (facts.constants.size() > UINT64_MAX - facts.pointers.size()) return {};
  const auto count = facts.constants.size() + facts.pointers.size();
  if (count > UINT64_MAX / std::max(sizeof(ConstantImageRange), sizeof(RelocatedPointer)) ||
      refuted_constants.size() > UINT64_MAX - count ||
      refuted_pointers.size() > UINT64_MAX - count - refuted_constants.size() ||
      budget.try_consume(
          {count + refuted_constants.size() + refuted_pointers.size(),
           facts.constants.size() * (sizeof(ConstantImageRange) + sizeof(std::size_t)) +
               facts.pointers.size() * sizeof(RelocatedPointer)}) != BudgetDecline::none)
    return {};
  if (!std::is_sorted(refuted_constants.begin(), refuted_constants.end()) ||
      !std::is_sorted(refuted_pointers.begin(), refuted_pointers.end()) ||
      std::adjacent_find(refuted_constants.begin(), refuted_constants.end()) !=
          refuted_constants.end() ||
      std::adjacent_find(refuted_pointers.begin(), refuted_pointers.end()) !=
          refuted_pointers.end() ||
      (!refuted_constants.empty() && refuted_constants.back() >= facts.constants.size()) ||
      (!refuted_pointers.empty() && refuted_pointers.back() >= facts.pointers.size()))
    return {};
  RetainedImageFacts retained;
  retained.page_aligned_placement = facts.page_aligned_placement;
  retained.load_bias = facts.load_bias;
  retained.constants.reserve(facts.constants.size() - refuted_constants.size());
  retained.origins.reserve(facts.constants.size() - refuted_constants.size());
  retained.pointers.reserve(facts.pointers.size());
  std::size_t next = 0;
  for (std::size_t i = 0; i < facts.constants.size(); ++i) {
    if (next < refuted_constants.size() && refuted_constants[next] == i) {
      ++next;
      continue;
    }

    retained.constants.push_back(facts.constants[i]);
    retained.origins.push_back(i);
  }

  next = 0;
  for (std::size_t i = 0; i < facts.pointers.size(); ++i) {
    retained.pointers.push_back(facts.pointers[i]);
    if (next < refuted_pointers.size() && refuted_pointers[next] == i) {
      retained.pointers.back().value_stable = false;
      ++next;
    }
  }

  return retained;
}

std::optional<std::uint64_t> ReadConstant(const ImageFacts& facts, std::uint64_t address,
                                          unsigned width, ByteOrder order) {
  if (width == 0 || width > 64 || width % 8 != 0) return std::nullopt;
  const auto size = width / 8;
  if (address > UINT64_MAX - size) return std::nullopt;

  // Slots are eight bytes, so only one starting fewer than eight before the
  // read can reach into it.
  const auto first = std::lower_bound(
      facts.pointers.begin(), facts.pointers.end(), address < 7 ? 0 : address - 7,
      [](const RelocatedPointer& slot, std::uint64_t key) { return slot.address < key; });
  if (first != facts.pointers.end() && first->address < address + size) return std::nullopt;
  if (address < 7) {
    const auto suffix = std::lower_bound(
        facts.pointers.begin(), facts.pointers.end(), UINT64_MAX - 6,
        [](const RelocatedPointer& slot, std::uint64_t key) { return slot.address < key; });
    for (auto at = suffix; at != facts.pointers.end(); ++at) {
      if (address - at->address < 8) return std::nullopt;
    }
  }

  for (const auto& range : facts.constants) {
    if (address < range.address) continue;
    const auto offset = address - range.address;
    if (offset > range.bytes.size() || range.bytes.size() - offset < size) continue;
    std::uint64_t bits = 0;
    for (unsigned i = 0; i < size; ++i) {
      const auto byte = std::uint64_t(range.bytes[offset + i]);
      bits |= order == ByteOrder::little ? byte << (8 * i) : byte << (8 * (size - 1 - i));
    }

    return bits;
  }

  return std::nullopt;
}

std::optional<std::uint64_t> ReadRelocated(const ImageFacts& facts, std::uint64_t address,
                                           unsigned width) {
  if (width != 64) return std::nullopt;
  const auto at = std::lower_bound(
      facts.pointers.begin(), facts.pointers.end(), address,
      [](const RelocatedPointer& slot, std::uint64_t key) { return slot.address < key; });
  if (at == facts.pointers.end() || at->address != address || !at->value_stable)
    return std::nullopt;
  return at->target;
}

std::optional<RetainedImageFacts> RetainImageFacts(
    const ImageFacts& facts, std::span<const RefutedConstantSpan> refuted_constants,
    std::span<const std::size_t> refuted_pointers, Budget& budget) {
  // A range keeps at most one piece more than the spans cutting it.
  const auto pieces = facts.constants.size() + refuted_constants.size();
  if (pieces < refuted_constants.size() ||
      pieces > UINT64_MAX / (sizeof(ConstantImageRange) + sizeof(std::size_t)) ||
      budget.try_consume({pieces + refuted_pointers.size() + facts.pointers.size(),
                          pieces * (sizeof(ConstantImageRange) + sizeof(std::size_t)) +
                              facts.pointers.size() * sizeof(RelocatedPointer)}) !=
          BudgetDecline::none)
    return {};
  for (std::size_t i = 0; i < refuted_constants.size(); ++i) {
    const auto& span = refuted_constants[i];
    if (span.index >= facts.constants.size() || !span.bytes) return {};
    const auto& range = facts.constants[span.index];
    if (span.address < range.address || span.address - range.address > range.bytes.size() ||
        span.bytes > range.bytes.size() - (span.address - range.address))
      return {};
    if (i && std::pair{refuted_constants[i - 1].index, refuted_constants[i - 1].address} >
                 std::pair{span.index, span.address})
      return {};
  }

  if (!std::is_sorted(refuted_pointers.begin(), refuted_pointers.end()) ||
      std::adjacent_find(refuted_pointers.begin(), refuted_pointers.end()) !=
          refuted_pointers.end() ||
      (!refuted_pointers.empty() && refuted_pointers.back() >= facts.pointers.size()))
    return {};
  RetainedImageFacts retained;
  retained.page_aligned_placement = facts.page_aligned_placement;
  retained.load_bias = facts.load_bias;
  std::size_t next = 0;
  for (std::size_t i = 0; i < facts.constants.size(); ++i) {
    const auto& range = facts.constants[i];
    std::uint64_t cursor = 0;  // offset into the range
    const auto keep = [&](std::uint64_t end) {
      if (end <= cursor) return;
      retained.constants.push_back(
          {range.address + cursor, range.bytes.subspan(cursor, end - cursor), range.read_only});
      retained.origins.push_back(i);
    };

    for (; next < refuted_constants.size() && refuted_constants[next].index == i; ++next) {
      const auto offset = refuted_constants[next].address - range.address;
      keep(offset);
      cursor = std::max(cursor, offset + refuted_constants[next].bytes);
    }

    keep(range.bytes.size());
  }

  std::size_t slot = 0;
  for (std::size_t i = 0; i < facts.pointers.size(); ++i) {
    retained.pointers.push_back(facts.pointers[i]);
    if (slot < refuted_pointers.size() && refuted_pointers[slot] == i) {
      retained.pointers.back().value_stable = false;
      ++slot;
    }
  }

  return retained;
}

std::optional<std::vector<RefutedConstantSpan>> RefuteConstantBytes(
    const ImageFacts& facts, std::span<const ImageWrite> writes, Budget& budget) {
  if (budget.try_consume({facts.constants.size(), facts.constants.size() * sizeof(std::size_t)}) !=
      BudgetDecline::none)
    return {};
  std::vector<std::size_t> reached(facts.constants.size());
  std::vector<RefutedConstantSpan> spans;
  for (const auto& write : writes) {
    if (write.width == 0 || write.width % 8 != 0) continue;
    if (budget.try_consume({facts.constants.size() + 2, 0}) != BudgetDecline::none) return {};
    const std::uint64_t size = write.width / 8;

    // A write past the top of the address space wraps to its bottom.
    const auto top = UINT64_MAX - write.address + 1 < size && write.address != 0
                         ? UINT64_MAX - write.address + 1
                         : size;
    const std::array<std::pair<std::uint64_t, std::uint64_t>, 2> parts{
        {{write.address, top}, {0, size - top}}};
    for (std::size_t index = 0; index < facts.constants.size(); ++index) {
      const auto& range = facts.constants[index];
      const auto end = range.bytes.size() > UINT64_MAX - range.address
                           ? UINT64_MAX
                           : range.address + range.bytes.size();
      for (const auto& [from, length] : parts) {
        if (!length || range.bytes.empty()) continue;
        const auto first = std::max(from, range.address);
        const auto last = std::min(length > UINT64_MAX - from ? UINT64_MAX : from + length, end);
        if (first >= last) continue;

        // Past the bound the range is kept whole; the count still records it.
        if (++reached[index] > kMaxRefutedSpans) continue;
        if (budget.try_consume({1, sizeof(RefutedConstantSpan)}) != BudgetDecline::none) return {};
        spans.push_back({index, first, last - first, write});
      }
    }
  }

  const auto sort_work = spans.size() * (std::bit_width(spans.size()) + 1);
  if (budget.try_consume({sort_work + spans.size(), 0}) != BudgetDecline::none) return {};
  std::stable_sort(spans.begin(), spans.end(), [](const auto& a, const auto& b) {
    return std::pair{a.index, a.address} < std::pair{b.index, b.address};
  });
  std::vector<RefutedConstantSpan> bounded;
  bounded.reserve(spans.size());
  for (std::size_t at = 0; at < spans.size();) {
    const auto index = spans[at].index;
    auto end = at;
    while (end < spans.size() && spans[end].index == index) ++end;
    if (reached[index] > kMaxRefutedSpans) {
      const auto& range = facts.constants[index];
      bounded.push_back({index, range.address, range.bytes.size(), spans[at].write, true});
    } else {
      bounded.insert(bounded.end(), spans.begin() + at, spans.begin() + end);
    }

    at = end;
  }

  return bounded;
}

}  // namespace nyx::ir
