#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "nyx/support/budget.hpp"

namespace nyx::eval {

struct MemoryIdentity;

enum class MemoryStatus {
  ok,
  unmapped,
  permission,
  address_overflow,
  unsupported_access,
  resource_limit,
  invalid_mapping,
  invalid_transaction,
  // The bytes are mapped, but what they hold is not known: something outside
  // the model (the loader resolving an import, say) writes them first.
  unknown_value
};

struct MemoryLimits {
  std::uint32_t max_regions = 4096;
  std::uint64_t max_total_bytes = 64 * 1024 * 1024;
  std::uint32_t max_pending_writes = 4096;
};

struct RegionInput {
  std::uint64_t address;
  std::span<const std::uint8_t> bytes;
  bool readable = true;
  bool writable = true;

  // Offsets into `bytes` whose value is unknown until a write gives them one.
  std::span<const std::uint64_t> unknown = {};
};

struct MemoryAccessResult {
  MemoryStatus status = MemoryStatus::ok;
  std::uint64_t fault_address = 0;
};

struct MemoryReadResult {
  MemoryStatus status = MemoryStatus::ok;
  std::uint64_t fault_address = 0;
  std::array<std::uint8_t, 16> bytes{};
  std::uint8_t size = 0;
};

struct MemoryRegion {
  std::uint64_t address;
  std::vector<std::uint8_t> bytes;
  bool readable;
  bool writable;

  // One flag per byte when any byte is unknown; empty when every byte is known.
  std::vector<std::uint8_t> unknown = {};
};

struct MemoryBuildResult;

class Memory {
 public:
  class Transaction;

  Memory(const Memory&) = delete;
  Memory& operator=(const Memory&) = delete;
  Memory(Memory&&) noexcept = default;
  Memory& operator=(Memory&&) noexcept = default;

  static MemoryBuildResult Create(std::span<const RegionInput> regions, Budget& budget,
                                  MemoryLimits limits = {});

  std::span<const MemoryRegion> Regions() const { return regions_; }

  const std::shared_ptr<const MemoryIdentity>& identity() const { return identity_; }

  // Permission and mapping check without reading or queuing a write.
  MemoryAccessResult CheckAccess(std::uint64_t address, std::uint64_t size, bool write,
                                 Budget& budget) const;
  MemoryReadResult Read(std::uint64_t address, std::uint64_t size, Budget& budget) const;
  Transaction Begin(Budget& budget);

 private:
  Memory() = default;
  MemoryAccessResult Check(std::uint64_t address, std::uint64_t size, bool write,
                           Budget& budget) const;
  const MemoryRegion* Find(std::uint64_t address) const;

  // Reads checked bytes; `covered` (empty, or one flag per byte) marks those a
  // pending write supplies, which an unknown byte beneath does not block.
  MemoryReadResult Gather(std::uint64_t address, std::uint64_t size,
                          std::span<const std::uint8_t> covered) const;
  void Put(std::uint64_t address, std::span<const std::uint8_t> bytes) noexcept;

  std::vector<MemoryRegion> regions_;
  MemoryLimits limits_;
  std::shared_ptr<const MemoryIdentity> identity_;
};

struct MemoryBuildResult {
  std::optional<Memory> memory;
  MemoryStatus status = MemoryStatus::ok;
};

// The owner and its mapping shape must outlive the transaction. Commit publishes
// its ordered prefix even after a modeled access fault; destruction discards it.
class Memory::Transaction {
 public:
  Transaction(const Transaction&) = delete;
  Transaction& operator=(const Transaction&) = delete;
  Transaction(Transaction&& other) noexcept;
  Transaction& operator=(Transaction&&) = delete;

  MemoryReadResult Read(std::uint64_t address, std::uint64_t size);
  MemoryAccessResult Write(std::uint64_t address, std::span<const std::uint8_t> bytes);
  void Commit() noexcept;

 private:
  friend class Memory;

  Transaction(Memory& memory, Budget& budget) : memory_(&memory), budget_(&budget) {}

  struct PendingWrite {
    std::uint64_t address;
    std::array<std::uint8_t, 16> bytes;
    std::uint8_t size;
  };

  Memory* memory_;
  Budget* budget_;
  std::vector<PendingWrite> writes_;
  bool resource_failed_ = false;
};

}  // namespace nyx::eval
