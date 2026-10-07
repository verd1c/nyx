#pragma once

#include <cstdint>

namespace nyx {

enum class BudgetDecline { none, work_limit, byte_limit };

struct Resources {
  std::uint64_t work = 0;
  std::uint64_t bytes = 0;
};

// Bytes are cumulative allocation charges, not live memory; freeing an allocation
// does not permit an input to evade its deterministic resource limit.
class Budget {
 public:
  explicit constexpr Budget(Resources limit) noexcept : limit_(limit) {}

  [[nodiscard]] constexpr BudgetDecline try_consume(Resources amount) noexcept {
    if (amount.work > limit_.work - used_.work) {
      return BudgetDecline::work_limit;
    }

    if (amount.bytes > limit_.bytes - used_.bytes) {
      return BudgetDecline::byte_limit;
    }

    used_.work += amount.work;
    used_.bytes += amount.bytes;
    return BudgetDecline::none;
  }

  [[nodiscard]] constexpr Resources used() const noexcept { return used_; }

  [[nodiscard]] constexpr Resources remaining() const noexcept {
    return {limit_.work - used_.work, limit_.bytes - used_.bytes};
  }

 private:
  Resources limit_;
  Resources used_{};
};

}  // namespace nyx
