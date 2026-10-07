#pragma once

#include <array>

#include "nyx/ir/dispatch.hpp"
#include "nyx/ir/path.hpp"

namespace nyx::ir {

std::uint64_t NextRecoveredPathIdentity();

// What authorized a rewrite, and so what has to be rechecked for it. Two
// concern a conditional transfer: one removes the conditional a literal
// predicate already decided, the other restores the conditional a table
// dispatch hid. The third removes a restored conditional in turn, once the
// predicate behind it became a literal. That only happens well after
// recovery, so a path never carries it.
// bounded_exit: a block that branches to itself and to one other block, and
// that a bounded-loop fact showed leaves after `iterations` runs, jumps
// straight to where it leaves. Its only effects are storage, and whatever it
// changes is dead where it goes, so running it once is running it to the end.
// Over a dispatch it keeps both destinations as decided_dispatch does.
// retired_call: a call to a callee whose body the graph carries, that does
// nothing but compute and write registers and whose writes are all dead
// where the call returns, jumps straight to its continuation. The condition
// names the call's target and when_true the body's address.
// resolved_call: a call whose computed target the graph settles on one image
// location calls it directly, through a destination node, so what computed
// the target is no longer needed. The condition names the original target and
// when_true the location.
enum class RewriteRule {
  folded_condition,
  dispatch_branch,
  decided_dispatch,
  bounded_exit,
  retired_call,
  resolved_call
};

struct ConditionalRewrite {
  RewriteRule rule = RewriteRule::folded_condition;
  std::uint32_t boundary;
  Transfer original;
  Transfer replacement;
  ValueId condition;

  // folded_condition: the literal value of the predicate.
  bool condition_value = false;

  // dispatch_branch and decided_dispatch: the destinations the condition
  // selects between, in the order the replacement names them, and every
  // declared image read the two resolutions rest on. The reads stay in the
  // path and still execute. A decided dispatch keeps both, and both
  // destination nodes, as the provenance of the one it goes to.
  std::uint64_t when_true = 0;
  std::uint64_t when_false = 0;
  std::vector<ImageRead> witness;
  std::uint64_t from_revision;
  std::uint64_t to_revision;

  // bounded_exit: how many times the loop ran, the last one leaving.
  std::uint32_t iterations = 0;
};

inline constexpr std::uint32_t kMaxOverwriteBoundaryDistance = 16;

// A store whose entire write is replaced within a bounded source interval
// before any access or transfer can observe it. Execution still checks the
// computed address is mapped and writable when the omitted store would execute.
struct StoreOmission {
  ValueId store;
  ValueId overwriter;
  StorageId base_storage;
  std::uint64_t offset_bytes;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
};

struct StorageAddress {
  ValueId root;
  std::uint64_t offset;
};

// An affine address rooted in a read of one storage cell. The same bounded
// proof is used when proposing and validating a store omission.
[[nodiscard]] std::optional<StorageAddress> ResolveStorageAddress(std::span<const Node> nodes,
                                                                  ValueId id, StorageId storage);

// The two loads of one source instruction have no remaining value users in
// the basis. Their original readable accesses are checked together before
// either is skipped; the instruction's register writes and transfer remain.
struct PairedLoadOmission {
  std::array<ValueId, 2> stores;
  std::array<ValueId, 2> loads;
  StorageId base_storage;
  std::array<std::uint64_t, 2> offset_bytes;
  std::uint64_t from_revision;
  std::uint64_t to_revision;
};

// The basis already includes any preceding dataflow recovery. Effective control
// changes are separate from its immutable source groups and strict Path contract.
class RecoveredPath {
 public:
  RecoveredPath(Path basis, std::vector<ConditionalRewrite> rewrites, std::uint64_t revision,
                std::vector<Node> destinations = {}, std::vector<StoreOmission> omissions = {},
                std::vector<PairedLoadOmission> paired_load_omissions = {})
      : basis_(std::move(basis)),
        rewrites_(std::move(rewrites)),
        revision_(revision),
        destinations_(std::move(destinations)),
        omissions_(std::move(omissions)),
        paired_load_omissions_(std::move(paired_load_omissions)) {}

  const Path& basis() const { return basis_; }

  std::span<const ConditionalRewrite> rewrites() const { return rewrites_; }

  std::span<const StoreOmission> omissions() const { return omissions_; }

  std::span<const PairedLoadOmission> paired_load_omissions() const {
    return paired_load_omissions_;
  }

  std::uint64_t revision() const { return revision_; }

  std::uint64_t identity() const { return identity_; }

  // A flattened dispatch computes both of its destinations from one expression,
  // so neither is a node of any instruction and neither can be named by a basis
  // value. These are the values a dispatch rewrite introduced for them, with ids
  // running from `first_destination()` upward. They are pure literal places: no
  // basis node reads one, and nothing but a replacement transfer names one.
  std::span<const Node> destinations() const { return destinations_; }

  ValueId first_destination() const { return static_cast<ValueId>(basis_.nodes().size()); }

  // Consumers validate before interpreting overrides. Out-of-range access is
  // safe and returns no transfer; it never creates a synthetic source boundary.
  std::optional<Transfer> effective_transfer(std::size_t boundary) const;
  bool omits_store(ValueId id) const;
  const PairedLoadOmission* first_omitted_load(ValueId id) const;
  bool omits_second_load(ValueId id) const;

  RecoveredPath with_omissions(std::vector<StoreOmission> omissions, std::uint64_t revision) && {
    return {std::move(basis_),        std::move(rewrites_), revision,
            std::move(destinations_), std::move(omissions), std::move(paired_load_omissions_)};
  }

  RecoveredPath with_paired_load_omissions(std::vector<PairedLoadOmission> omissions,
                                           std::uint64_t revision) && {
    return {std::move(basis_),        std::move(rewrites_),  revision,
            std::move(destinations_), std::move(omissions_), std::move(omissions)};
  }

 private:
  std::uint64_t identity_ = NextRecoveredPathIdentity();
  Path basis_;
  std::vector<ConditionalRewrite> rewrites_;
  std::uint64_t revision_;
  std::vector<Node> destinations_;
  std::vector<StoreOmission> omissions_;
  std::vector<PairedLoadOmission> paired_load_omissions_;
};

// Re-derived from the final basis, not from a producer's forwarding journal.
bool ValidPairedLoadOmission(const Path&, const PairedLoadOmission&);

// Rechecks every witness against the path and the declared bytes; a journal is
// not trusted authority. A dispatch branch is re-resolved from the original
// transfer, so a record naming the wrong condition, the wrong destinations, the
// wrong order or reads that did not happen is refused. Without the facts that
// resolved it, it is refused too: the same bytes have to be declared to the
// checker as to the producer. Control rewrites, paired-load omissions and
// store omissions each advance the basis revision once when present, in that
// order. An empty wrapper keeps it unchanged, including at UINT64_MAX.
[[nodiscard]] BlockDecline ValidateRecoveredPath(const RecoveredPath&, Budget&, BlockLimits = {},
                                                 ImageFacts = {});

}  // namespace nyx::ir
