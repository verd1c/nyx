#include "nyx/eval/memory.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace nyx::eval {
struct MemoryIdentity {};

namespace {

bool Wraps(std::uint64_t address, std::uint64_t size) {
  return size != 0 && address > std::numeric_limits<std::uint64_t>::max() - (size - 1);
}

}  // namespace

MemoryBuildResult Memory::Create(std::span<const RegionInput> regions, Budget& budget,
                                 MemoryLimits limits) {
  if (regions.size() > limits.max_regions) return {std::nullopt, MemoryStatus::resource_limit};
  const std::uint64_t count = regions.size();
  if (budget.try_consume({count * count, 0}) != BudgetDecline::none) {
    return {std::nullopt, MemoryStatus::resource_limit};
  }

  std::uint64_t total = 0;
  for (std::size_t i = 0; i < regions.size(); ++i) {
    const auto& region = regions[i];
    if (region.bytes.empty() || Wraps(region.address, region.bytes.size())) {
      return {std::nullopt, MemoryStatus::invalid_mapping};
    }

    if (region.bytes.size() > limits.max_total_bytes - total) {
      return {std::nullopt, MemoryStatus::resource_limit};
    }

    total += region.bytes.size();
    for (const auto offset : region.unknown)
      if (offset >= region.bytes.size()) return {std::nullopt, MemoryStatus::invalid_mapping};
    for (std::size_t j = 0; j < i; ++j) {
      const auto& other = regions[j];
      const bool overlap = region.address >= other.address
                               ? region.address - other.address < other.bytes.size()
                               : other.address - region.address < region.bytes.size();
      if (overlap) return {std::nullopt, MemoryStatus::invalid_mapping};
    }
  }

  if (budget.try_consume({count, count * sizeof(MemoryRegion)}) != BudgetDecline::none ||
      budget.try_consume({total, total}) != BudgetDecline::none ||
      budget.try_consume({1, sizeof(MemoryIdentity) + 2 * sizeof(void*)}) != BudgetDecline::none) {
    return {std::nullopt, MemoryStatus::resource_limit};
  }

  std::uint64_t flagged = 0;
  for (const auto& input : regions)
    if (!input.unknown.empty()) flagged += input.bytes.size();
  if (budget.try_consume({flagged, flagged}) != BudgetDecline::none)
    return {std::nullopt, MemoryStatus::resource_limit};
  Memory result;
  result.identity_ = std::make_shared<MemoryIdentity>();
  result.limits_ = limits;
  result.regions_.reserve(regions.size());
  for (const auto& input : regions) {
    result.regions_.push_back({input.address,
                               std::vector<std::uint8_t>(input.bytes.begin(), input.bytes.end()),
                               input.readable, input.writable});
    if (input.unknown.empty()) continue;
    auto& unknown = result.regions_.back().unknown;
    unknown.assign(input.bytes.size(), 0);
    for (const auto offset : input.unknown) unknown[static_cast<std::size_t>(offset)] = 1;
  }

  std::sort(result.regions_.begin(), result.regions_.end(),
            [](const auto& a, const auto& b) { return a.address < b.address; });
  return {std::move(result), MemoryStatus::ok};
}

const MemoryRegion* Memory::Find(std::uint64_t address) const {
  const auto after =
      std::upper_bound(regions_.begin(), regions_.end(), address,
                       [](auto value, const auto& region) { return value < region.address; });
  if (after == regions_.begin()) return nullptr;
  const auto& region = *std::prev(after);
  return address - region.address < region.bytes.size() ? &region : nullptr;
}

MemoryAccessResult Memory::Check(std::uint64_t address, std::uint64_t size, bool write,
                                 Budget& budget) const {
  if (size == 0 || size > 16) return {MemoryStatus::unsupported_access, address};
  if (Wraps(address, size)) return {MemoryStatus::address_overflow, address};
  if (budget.try_consume({size * (regions_.size() + 1), 0}) != BudgetDecline::none) {
    return {MemoryStatus::resource_limit, address};
  }

  for (std::uint64_t i = 0; i < size; ++i) {
    const auto* region = Find(address + i);
    if (region == nullptr) return {MemoryStatus::unmapped, address + i};
    if (write ? !region->writable : !region->readable) {
      return {MemoryStatus::permission, address + i};
    }
  }

  return {};
}

MemoryAccessResult Memory::CheckAccess(std::uint64_t address, std::uint64_t size, bool write,
                                       Budget& budget) const {
  return Check(address, size, write, budget);
}

MemoryReadResult Memory::Read(std::uint64_t address, std::uint64_t size, Budget& budget) const {
  const auto checked = Check(address, size, false, budget);
  if (checked.status != MemoryStatus::ok) return {checked.status, checked.fault_address};
  return Gather(address, size, {});
}

MemoryReadResult Memory::Gather(std::uint64_t address, std::uint64_t size,
                                std::span<const std::uint8_t> covered) const {
  MemoryReadResult result;
  result.size = static_cast<std::uint8_t>(size);
  for (std::uint64_t i = 0; i < size; ++i) {
    const auto* region = Find(address + i);
    const auto offset = address + i - region->address;

    // A byte a pending write covers takes that write's value, known or not here.
    if (!region->unknown.empty() && region->unknown[offset] && (covered.empty() || !covered[i]))
      return {MemoryStatus::unknown_value, address + i};
    result.bytes[i] = region->bytes[offset];
  }

  return result;
}

void Memory::Put(std::uint64_t address, std::span<const std::uint8_t> bytes) noexcept {
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    // Permission and mapping shape were checked before queuing this write.
    const auto* region = Find(address + i);
    auto& target = regions_[static_cast<std::size_t>(region - regions_.data())];
    target.bytes[address + i - region->address] = bytes[i];
    if (!target.unknown.empty()) target.unknown[address + i - region->address] = 0;
  }
}

Memory::Transaction Memory::Begin(Budget& budget) { return Transaction(*this, budget); }

Memory::Transaction::Transaction(Transaction&& other) noexcept
    : memory_(std::exchange(other.memory_, nullptr)),
      budget_(other.budget_),
      writes_(std::move(other.writes_)),
      resource_failed_(other.resource_failed_) {}

MemoryReadResult Memory::Transaction::Read(std::uint64_t address, std::uint64_t size) {
  if (memory_ == nullptr) return {MemoryStatus::invalid_transaction, address};
  if (resource_failed_) return {MemoryStatus::resource_limit, address};
  const auto checked = memory_->Check(address, size, false, *budget_);
  if (checked.status != MemoryStatus::ok) {
    resource_failed_ = checked.status == MemoryStatus::resource_limit;
    return {checked.status, checked.fault_address};
  }

  if (budget_->try_consume({size * writes_.size(), 0}) != BudgetDecline::none) {
    resource_failed_ = true;
    return {MemoryStatus::resource_limit, address};
  }

  std::array<std::uint8_t, 16> covered{};
  std::array<std::uint8_t, 16> pending{};
  for (const auto& write : writes_) {
    for (std::uint64_t i = 0; i < size; ++i) {
      if (address + i >= write.address && address + i - write.address < write.size) {
        pending[i] = write.bytes[address + i - write.address];
        covered[i] = 1;
      }
    }
  }

  auto result = memory_->Gather(address, size, std::span(covered).first(size));
  if (result.status != MemoryStatus::ok) return result;
  for (std::uint64_t i = 0; i < size; ++i)
    if (covered[i]) result.bytes[i] = pending[i];
  return result;
}

MemoryAccessResult Memory::Transaction::Write(std::uint64_t address,
                                              std::span<const std::uint8_t> bytes) {
  if (memory_ == nullptr) return {MemoryStatus::invalid_transaction, address};
  if (resource_failed_) return {MemoryStatus::resource_limit, address};
  const auto checked = memory_->Check(address, bytes.size(), true, *budget_);
  if (checked.status != MemoryStatus::ok) {
    resource_failed_ = checked.status == MemoryStatus::resource_limit;
    return checked;
  }

  if (writes_.size() == memory_->limits_.max_pending_writes) {
    resource_failed_ = true;
    return {MemoryStatus::resource_limit, address};
  }

  if (writes_.size() == writes_.capacity()) {
    const std::uint64_t capacity = std::min<std::uint64_t>(
        memory_->limits_.max_pending_writes, std::max<std::uint64_t>(1, writes_.capacity() * 2));
    if (budget_->try_consume({writes_.size() + 1, capacity * sizeof(PendingWrite)}) !=
        BudgetDecline::none) {
      resource_failed_ = true;
      return {MemoryStatus::resource_limit, address};
    }

    writes_.reserve(static_cast<std::size_t>(capacity));
  }

  PendingWrite pending{address, {}, static_cast<std::uint8_t>(bytes.size())};
  std::copy(bytes.begin(), bytes.end(), pending.bytes.begin());
  writes_.push_back(pending);
  return {};
}

void Memory::Transaction::Commit() noexcept {
  if (memory_ != nullptr && !resource_failed_) {
    for (const auto& write : writes_) {
      memory_->Put(write.address, std::span<const std::uint8_t>(write.bytes).first(write.size));
    }
  }

  memory_ = nullptr;
}

}  // namespace nyx::eval
