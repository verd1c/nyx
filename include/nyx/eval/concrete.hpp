#pragma once

#include <vector>

#include "nyx/eval/memory.hpp"
#include "nyx/ir/group.hpp"
#include "nyx/support/bit_vector.hpp"

namespace nyx::eval {

struct Cell {
  ir::StorageId id;
  BitVector value;
};

struct ExclusiveReservation {
  std::uint64_t address;
  unsigned size;
  std::array<std::uint8_t, 8> bytes{};
  std::shared_ptr<const MemoryIdentity> memory_identity;
};

struct State {
  std::vector<Cell> cells;
  std::optional<ExclusiveReservation> exclusive;
};

enum class Outcome { completed, invalid_group, invalid_state, resource_limit, fault, unsupported };

struct AccessEvent {
  // A failed address-monitor check performs no memory access and emits no event.
  ir::ValueId operation;
  std::uint64_t address;
  unsigned size;
  bool write;
  bool completed;
  std::array<std::uint8_t, 16> bytes{};
  bool conditional = false;
  bool performed = true;
};

enum class FaultKind { unmapped, permission, address_overflow, alignment };

struct Fault {
  FaultKind kind;
  std::uint64_t address;
  ir::ValueId operation;
  bool write;
};

// This is one concrete instruction observation. It grants no authority to skip
// a callee, assume it returns, or treat an indirect destination as exhaustive.
struct ResolvedTransfer {
  ir::TransferKind kind;
  std::uint64_t target;
  std::optional<bool> condition;
  std::optional<std::uint64_t> continuation;
};

struct ExecutionResult {
  Outcome outcome;
  std::optional<Fault> fault;
  std::vector<AccessEvent> events;
  std::optional<ResolvedTransfer> transfer;

  ExecutionResult(Outcome status, std::optional<Fault> failure = {},
                  std::vector<AccessEvent> accesses = {},
                  std::optional<ResolvedTransfer> control = {})
      : outcome(status), fault(failure), events(std::move(accesses)), transfer(control) {}
};

struct Limits {
  unsigned max_width = 4096;
  unsigned max_nodes = 4096;
  unsigned max_cells = 256;
};

struct ExecutionContext {
  // Absence refuses location-derived values; it never assumes a zero load bias.
  std::optional<std::uint64_t> load_bias;
};

// Reads use the entry snapshot. All writes commit together after validation and
// evaluation succeed, so malformed groups and exhausted budgets leave state intact.
// This compatibility API declines control transfers rather than discarding them.
[[nodiscard]] Outcome Execute(const ir::Group& group, State& state, Budget& budget,
                              Limits limits = {}, ExecutionContext context = {});

// Returns terminal instruction effects without fetching or executing their target.
[[nodiscard]] ExecutionResult ExecuteDetailed(const ir::Group& group, State& state, Budget& budget,
                                              Limits limits = {}, ExecutionContext context = {});

// Ordinary byte memory with atomic scalar accesses. Completed stores and explicit
// register writes precede a modeled fault; final Group::writes run only on success.
// Invalid input/resource exhaustion discards all effects, including event output.
// Ordinary scalar accesses exclude device memory, atomics, concurrency and
// asynchronous faults. The separate QEMU-exclusive profile models its scalar
// reservation algorithm without concurrent observers or architectural authority.
// Reads still observe entry storage after an ordered write. This reference fault
// model is not an architecture-wide AArch64 post-abort state guarantee.
[[nodiscard]] ExecutionResult Execute(const ir::Group& group, State& state, Memory& memory,
                                      Budget& budget, Limits limits = {},
                                      ExecutionContext context = {});

}  // namespace nyx::eval
