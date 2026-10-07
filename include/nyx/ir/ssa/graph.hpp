#pragma once

#include <array>
#include <utility>

#include "nyx/ir/image_facts.hpp"
#include "nyx/ir/recovered_path.hpp"
#include "nyx/ir/storage_relation.hpp"

namespace nyx::ir {

struct SsaHandle {
  std::uint64_t arena = 0;
  std::uint32_t slot = 0;
  std::uint32_t generation = 0;
  friend bool operator==(const SsaHandle&, const SsaHandle&) = default;
};

enum class SsaValueKind { input, phi, node, clobber, frame_phi, frame_input };

struct SsaValue {
  SsaValueKind kind = SsaValueKind::input;
  SsaHandle block{};
  std::uint32_t index = 0;
  friend bool operator==(const SsaValue&, const SsaValue&) = default;
};

struct SsaPhiInput {
  SsaHandle predecessor;
  SsaValue value;
};

struct SsaPhi {
  StorageId storage;
  unsigned width;
  bool external_entry = false;
  std::vector<SsaPhiInput> incoming;
};

struct SsaRead {
  ValueId node;
  std::uint32_t phi;
  SsaValue value;

  // A proved single-predecessor entry phi may read its predecessor's node.
  std::optional<SsaValue> predecessor_copy = std::nullopt;
  bool copy_closed_entries = false;
};

struct SsaDeadStorageWrite {
  std::uint32_t boundary;
  std::uint32_t index;
  bool closed_entries = false;
  friend bool operator==(const SsaDeadStorageWrite&, const SsaDeadStorageWrite&) = default;
};

struct SsaExitValue {
  StorageId storage;
  SsaValue value;
};

struct SsaFramePhi {
  std::int64_t offset;
  std::uint32_t size;
  bool external_entry = false;
  std::vector<SsaPhiInput> incoming;
};

struct SsaFrameAccess {
  ValueId node;
  std::uint32_t phi;

  // A load reads this SSA value; a store has no replacement.
  std::optional<SsaValue> replacement;
};

enum class SsaConstantKind { literal, image_location, bounded_table };

struct SsaConstantLoad {
  ValueId node;
  SsaConstantKind kind;
  std::uint64_t value;
  std::uint64_t source_address;

  // These caller declarations travel with the fold. Validation checks their
  // internal consistency and modeled writes, not their identity with ELF bytes.
  std::array<std::uint8_t, 8> declared_bytes{};
  std::uint64_t declared_target = 0;
  ImageAccessContract access{};
  bool page_aligned_placement = false;
  bool value_stable = false;
  bool skip_access = false;

  // Every range the fold reads was declared non-writable, so a store that
  // cannot be placed would fault there rather than change it.
  bool read_only = false;

  // A selected load uses value/source_address on true and these on false.
  std::optional<ValueId> condition = std::nullopt;
  std::uint64_t alternative_value = 0;
  std::uint64_t alternative_source_address = 0;
  std::array<std::uint8_t, 8> alternative_declared_bytes{};

  // For a bounded table, source_address is the image-relative base. The
  // index names an existing SSA read; each row contains one declared value.
  ValueId table_index = 0;
  std::uint64_t table_stride = 0;
  std::vector<std::array<std::uint8_t, 8>> table_bytes{};
  std::optional<SsaHandle> table_guard = std::nullopt;
  std::uint32_t table_guard_edge = 0;
  StorageId table_storage = 0;
  bool table_closed_entries = false;
};

// A load a block's control resolves through the run's declared
// image values: the bytes of a constant range, or a relocated slot's target.
// The declaration travels with the block so its successor rule can read it
// without the run's facts. Validation checks it against its load, not against
// ELF bytes; the successor rule ignores it after a write the block provably
// makes to the location. The load still runs.
struct SsaPathRead {
  ValueId node;
  std::uint64_t address;

  // A relocated slot holds `value` as an image location; otherwise `value` is
  // the declared bytes read at the load's width and byte order.
  bool relocated = false;
  std::uint64_t value = 0;

  // The address masked an image location to its page, so it names this
  // location only under the run's page-aligned placement declaration. The
  // declaration stays with the record after the address is rewritten as the
  // location it names.
  bool page_aligned_placement = false;
};

// Why a load whose value nothing uses may stop executing: an earlier store of
// the same block wrote every byte it reads, so they are mapped; or every
// address it can read lies in a declared read-only image range or relocated
// slot that the access contract keeps mapped and readable. The node is also a disabled
// effect; validation rechecks the basis and that nothing uses the value.
enum class SsaRetiredLoadBasis { written_before, declared_image };

struct SsaRetiredLoad {
  ValueId node;
  SsaRetiredLoadBasis basis;
  ValueId store = 0;
  std::uint64_t range_address = 0;
  std::uint64_t range_bytes = 0;
  ImageAccessContract access{};
};

enum class SsaEdgeKind {
  fallthrough,
  branch,
  callee,
  return_,
  potential_return,
  opaque_unknown,
  trap
};

enum class SsaTargetKind { image_location, absolute_runtime, unknown };

struct SsaEdgeAssumptions {
  bool constant_image = false;
  bool callee_returns_to_continuation = false;
  bool return_leaves = false;
  bool unresolved_target = false;
  bool declared_opaque_control = false;
  bool entry_relations = false;
  bool declared_abi = false;
  bool declared_return_leaves = false;
  bool declared_noreturn = false;
};

struct SsaEdge {
  SsaEdgeKind kind;
  SsaTargetKind target_kind;
  std::uint64_t address = 0;
  std::optional<SsaHandle> target_block;
  std::optional<ValueId> condition;
  std::optional<bool> when;

  // These bits preserve the proved edge's declared scope for later passes.
  SsaEdgeAssumptions assumptions;
};

struct SsaBlock {
  std::uint32_t original_block;
  std::uint64_t address;
  std::optional<std::uint32_t> transition;
  bool opaque = false;
  std::vector<Node> nodes;
  std::vector<Boundary> boundaries;
  std::vector<std::uint64_t> source_groups;

  // Exact original instruction bytes for binding a function comparison.
  std::vector<std::vector<std::uint8_t>> source_bytes;
  std::vector<std::uint32_t> original_sources;
  std::vector<StorageRelation> entry_relations;
  bool relation_declared_abi = false;
  bool relation_return_leaves = false;
  bool relation_constant_image = false;
  bool relation_declared_opaque_control = false;

  // Direct folded-control rewrites advance this local control revision too.
  std::uint64_t path_revision = 0;
  std::vector<ConditionalRewrite> control_rewrites;
  std::vector<StoreOmission> store_omissions;
  std::vector<PairedLoadOmission> paired_load_omissions;
  std::vector<Node> destination_nodes;

  // Effects disabled by SSA recovery. Nodes remain as provenance and stable IDs.
  std::vector<ValueId> disabled_effects;

  // Pure values retired by liveness. IDs remain stable for source provenance.
  std::vector<ValueId> dead_pure_nodes;
  std::vector<SsaPhi> phis;

  // One bit per phi. Opaque operations and unknown calls define fresh state.
  std::vector<std::uint8_t> clobbers;
  std::vector<SsaRead> reads;

  // Decoded writes remain in boundaries for provenance after retirement.
  std::vector<SsaDeadStorageWrite> dead_storage_writes;
  std::vector<SsaExitValue> exits;
  std::vector<SsaFramePhi> frame_phis;
  std::vector<SsaFrameAccess> frame_accesses;
  std::vector<SsaValue> frame_exits;
  std::vector<SsaConstantLoad> constant_loads;

  // Sorted by node.
  std::vector<SsaPathRead> path_reads;

  // Sorted by node.
  std::vector<SsaRetiredLoad> retired_loads;
  std::vector<SsaEdge> edges;
};

// One bit per node: whether the block's dispatch condition decides its value.
// A dispatcher's successors are proved by refolding its path under each value
// of that condition, so a pass that writes a value into one of these nodes
// leaves a block whose two records disagree and which no longer validates.
// They fold once the edge the condition decides has been retired. The nodes
// a refold fixes (the condition and what SsaImpliedBits reaches back
// through) are not marked: it writes over them before reading them. Empty
// where the block has no dispatch rewrite.
std::optional<std::vector<std::uint8_t>> SsaDispatchDependent(const SsaBlock&, SsaHandle, Budget&);
bool SsaEffectFree(const SsaBlock&, ValueId);
bool SsaNeedsInputs(const SsaBlock&, ValueId);
std::optional<ValueId> SsaFoldCondition(const SsaBlock&, ValueId);
std::optional<ValueId> SsaFoldIndex(const SsaBlock&, ValueId);

// What leaving the graph lets its environment observe. Undeclared, every
// storage is observed wherever control leaves, and wherever an effect may
// fault. Declared, a call observes `at_call` and a return `at_return`, a call
// returns `preserved` as it found it, and a run that never leaves observes no
// storage. `faults_terminal` further declares that a fault or trap ends the
// run with its storage unobserved.
struct SsaObservability {
  bool declared = false;
  bool faults_terminal = false;
  std::vector<StorageId> at_call;    // sorted
  std::vector<StorageId> at_return;  // sorted
  std::vector<StorageId> preserved;  // sorted
  friend bool operator==(const SsaObservability&, const SsaObservability&) = default;
};

// What a call to `target` leaves in `storage`. A callee's effect on registers
// is otherwise a clobber: the graph says fresh state appeared and nothing
// more. This says what appeared. The caller supplies it and the artifact prints it;
// recovery does not derive it.
struct SsaCallResult {
  std::uint64_t target;
  StorageId storage;
  std::uint64_t value;
  friend bool operator==(const SsaCallResult&, const SsaCallResult&) = default;
};

// The instructions a callee runs, decoded from the image at its address, for
// a call that can be told apart from running them. They travel with the graph
// so a check needs no image, in the same way a fold's declared bytes do: what
// is checked is what they compute, not that the image holds them.
struct SsaCalleeBody {
  std::uint64_t address = 0;
  std::vector<Group> groups;
};

// A body this long is not a leaf worth summarizing.
inline constexpr std::size_t kMaxCalleeBodyGroups = 32;

class SsaGraph {
 public:
  SsaGraph();
  SsaGraph(const SsaGraph&) = delete;
  SsaGraph& operator=(const SsaGraph&) = delete;
  SsaGraph(SsaGraph&&) noexcept;
  SsaGraph& operator=(SsaGraph&&) noexcept;

  SsaHandle Add(SsaBlock block);
  bool Erase(SsaHandle handle);
  const SsaBlock* Get(SsaHandle handle) const;
  std::optional<SsaBlock> CopyBlock(SsaHandle handle, Budget&) const;
  bool Replace(SsaHandle handle, SsaBlock block);

  template <class Function>
  bool Update(SsaHandle handle, Function function) {
    const auto* original = Get(handle);
    if (!original) return false;
    SsaBlock block = *original;
    function(block);
    return Replace(handle, std::move(block));
  }

  std::span<const SsaHandle> entries() const { return entries_; }

  void SetEntries(std::vector<SsaHandle> entries) {
    entries_ = std::move(entries);
    ++revision_;
  }

  const SsaObservability& observability() const { return observability_; }

  // Where the run declares the image is placed. Without it an image location
  // is a place, not a number, and arithmetic that would need its runtime
  // address does not fold. Declaring it is what makes the two the same.
  std::optional<std::uint64_t> load_bias() const { return load_bias_; }

  void SetLoadBias(std::optional<std::uint64_t> bias) {
    load_bias_ = bias;
    ++revision_;
  }

  // The run declares entries() lists every way into the graph, so a block's
  // registers arrive only along its listed predecessors. Without it an interior
  // block may be entered from outside holding anything.
  bool entries_closed() const { return entries_closed_; }

  void SetEntriesClosed(bool closed) {
    entries_closed_ = closed;
    ++revision_;
  }

  // Sorted by target, then storage, and unique.
  std::span<const SsaCallResult> call_results() const { return call_results_; }

  void SetCallResults(std::vector<SsaCallResult> results) {
    call_results_ = std::move(results);
    ++revision_;
  }

  // Sorted by address and unique.
  std::span<const SsaCalleeBody> callee_bodies() const { return callee_bodies_; }

  void SetCalleeBodies(std::vector<SsaCalleeBody> bodies) {
    callee_bodies_ = std::move(bodies);
    ++revision_;
  }

  void SetObservability(SsaObservability value) {
    observability_ = std::move(value);
    ++revision_;
  }

  std::uint64_t revision() const { return revision_; }

  std::uint64_t arena() const { return arena_; }

  std::size_t slots() const { return slots_.size(); }

  std::optional<SsaHandle> Handle(std::size_t slot) const;
  std::optional<SsaGraph> Clone(Budget&) const;

 private:
  struct Slot {
    std::optional<SsaBlock> block;
    std::uint32_t generation = 0;
  };

  std::uint64_t arena_;
  std::uint64_t revision_ = 0;
  std::vector<Slot> slots_;
  std::vector<SsaHandle> entries_;
  SsaObservability observability_;
  std::vector<SsaCallResult> call_results_;
  std::vector<SsaCalleeBody> callee_bodies_;
  std::optional<std::uint64_t> load_bias_;
  bool entries_closed_ = false;
};

enum class SsaDecline { none, invalid_graph, resource_limit };
SsaDecline ValidateSsa(const SsaGraph&, Budget&);

// A closed population declares that every architectural entry into this graph
// is listed in entries(). Recovery must not infer this from discovery alone.
enum class SsaEntryScope { discovered_only, closed_population };

struct SsaReachabilityFacts {
  std::uint64_t graph_arena = 0;
  std::uint64_t graph_revision = 0;
  SsaEntryScope entry_scope = SsaEntryScope::discovered_only;

  // A one bit over-approximation of blocks reachable from all declared entries.
  std::vector<std::uint8_t> reachable;
};

struct SsaDeadWriteFact {
  SsaHandle block;
  std::uint32_t boundary;
  std::uint32_t index;
  friend bool operator==(const SsaDeadWriteFact&, const SsaDeadWriteFact&) = default;
};

struct SsaDeadWriteFacts {
  std::uint64_t graph_arena = 0;
  std::uint64_t graph_revision = 0;
  std::vector<SsaDeadWriteFact> writes;
};

struct SsaPredecessorCopy {
  SsaHandle block;
  ValueId read;
  SsaValue source;
  friend bool operator==(const SsaPredecessorCopy&, const SsaPredecessorCopy&) = default;
};

struct SsaPredecessorCopyFacts {
  std::uint64_t graph_arena = 0;
  std::uint64_t graph_revision = 0;
  std::vector<SsaPredecessorCopy> copies;
};

struct SsaPhiConstant {
  SsaHandle block;
  std::uint32_t phi;
  std::uint64_t value;
  friend bool operator==(const SsaPhiConstant&, const SsaPhiConstant&) = default;
};

struct SsaPhiConstantFacts {
  std::uint64_t graph_arena = 0;
  std::uint64_t graph_revision = 0;

  // Only constants shared by every reachable incoming edge are listed.
  std::vector<SsaPhiConstant> constants;
};

struct SsaConstantValue {
  SsaHandle block;
  SsaValueKind kind;
  std::uint32_t index;
  std::uint64_t value;
  friend bool operator==(const SsaConstantValue&, const SsaConstantValue&) = default;
};

struct SsaConstantFacts {
  std::uint64_t graph_arena = 0;
  std::uint64_t graph_revision = 0;

  // Sorted by block slot, phi before node, then index.
  std::vector<SsaConstantValue> values;
};

struct SsaSccpFacts {
  std::uint64_t graph_arena = 0;
  std::uint64_t graph_revision = 0;
  SsaEntryScope entry_scope = SsaEntryScope::discovered_only;

  // A closed over-approximation of executable blocks and their source edges.
  // Only these blocks need every successor enumerated: an unresolved jump in
  // a block no executable edge reaches never runs, and requiring the whole
  // graph complete would let one such block keep the conditions that guard
  // it from ever being decided.
  std::vector<std::uint8_t> executable;
  std::vector<std::vector<std::uint8_t>> edges;
  SsaConstantFacts constants;
};

struct SsaDominanceFacts {
  std::uint64_t graph_arena = 0;
  std::uint64_t graph_revision = 0;
  SsaEntryScope entry_scope = SsaEntryScope::discovered_only;
  std::vector<std::uint8_t> reachable;

  // Row-major [dominator * slots + block]; unreachable blocks have zero bits.
  std::vector<std::uint8_t> dominates;
};

struct SsaGuardEdgeFact {
  std::uint64_t graph_arena = 0;
  std::uint64_t graph_revision = 0;
  SsaEntryScope entry_scope = SsaEntryScope::discovered_only;
  SsaHandle branch{};
  std::uint32_t edge_index = 0;
  SsaHandle guarded_block{};

  // This proves edge necessity in the graph, not decoded predicate semantics.
  // Proved taking unfolded table dispatches' edges as their successors
  // (ValidateSsaDispatchSuccessors); good only for proposing a table fold.
  bool through_dispatches = false;
};

struct SsaBranchPredicateFact {
  std::uint64_t graph_arena = 0;
  std::uint64_t graph_revision = 0;
  SsaHandle branch{};
  std::uint32_t edge_index = 0;
  ValueId condition = 0;
  bool when = false;
};

struct SsaIndexBoundFact {
  SsaGuardEdgeFact guard;
  SsaBranchPredicateFact predicate;
  StorageId storage = 0;
  std::uint64_t exclusive_upper = 0;
};

struct SsaBoundedTableAddressFact {
  SsaIndexBoundFact index_bound;
  ValueId load = 0;
  std::uint64_t base = 0;
  std::uint64_t stride = 0;
  bool placed = false;
};

// A bounded table load's address: an image location plus a register read
// times the row stride. The multiply is absent for a stride of one; the base
// may be a PC page plus an offset, which names it only under page-aligned
// placement (`placed`), or with the load bias declared, a plain number.
struct SsaTableAddress {
  std::uint64_t base;
  bool placed;
  ValueId index;
  std::uint64_t stride;
};

std::optional<SsaTableAddress> SsaTableLoadAddress(std::span<const Node>, ValueId load,
                                                   std::optional<std::uint64_t> bias);
// The node whose number `id` is: through widenings, and through low-bit
// extracts of a number that already fits them. At most kSsaCoreSteps steps,
// so a caller charges that much per call.
inline constexpr unsigned kSsaCoreSteps = 32;
ValueId SsaUnsignedCore(std::span<const Node>, ValueId id);

// What taking a guard's edge says about a value its block computed: that its
// number is below `exclusive_upper`. The edge is x < N taken, x >= N not
// taken, or x > N not taken, the last as flags test it: not below N and not
// equal to it. `value` is x's SsaUnsignedCore, so a compare at any width bounds
// every widening of the same number.
struct SsaEdgeBound {
  ValueId value;
  std::uint64_t exclusive_upper;
};

std::optional<SsaEdgeBound> SsaGuardEdgeBound(std::span<const Node>, ValueId condition, bool when);

// The exit by which `storage` leaves the block holding node `id`'s number: a
// node with that core, or for a read, what it read. Null otherwise.
const SsaExitValue* SsaExitHolding(const SsaBlock&, SsaHandle, StorageId storage, ValueId id);

// What a block that branches only to itself and to one other block leaves
// behind. Its loop-carried state is its own phis and promoted frame slots, so
// running it from constant entry values is the only way to learn its exit
// values: a lattice joins them and goes overdefined on the second iteration.
//
// The values are keyed on the successor, not on the loop. A phi input is not
// independently settable (ValidateSsa ties it to the predecessor's published
// exit, and the self edge reads that same exit), so an exit value stated
// about the loop block would also claim the accumulator is constant from the
// second iteration, which is false.
struct SsaBoundedLoopExit {
  SsaValueKind kind;  // phi or frame_phi
  std::uint32_t index;
  std::uint64_t value;
  friend bool operator==(const SsaBoundedLoopExit&, const SsaBoundedLoopExit&) = default;
};

struct SsaBoundedLoopFact {
  SsaHandle loop;
  SsaHandle successor;
  std::uint32_t self_edge = 0;
  std::uint32_t exit_edge = 0;

  // How many times the block ran, at least once. The run took the self edge
  // on all but the last.
  std::uint32_t iterations = 0;

  // Sorted by kind, then index; phi before frame_phi. Only values the run
  // determined are listed.
  std::vector<SsaBoundedLoopExit> exits;
  friend bool operator==(const SsaBoundedLoopFact&, const SsaBoundedLoopFact&) = default;
};

struct SsaBoundedLoopFacts {
  std::uint64_t graph_arena = 0;
  std::uint64_t graph_revision = 0;

  // Sorted by loop slot.
  std::vector<SsaBoundedLoopFact> loops;
};

// A loop this long is not one a run should be unrolling to answer a question
// about it; the obfuscator's hashes take eight. Hitting it is a refusal.
inline constexpr unsigned kMaxBoundedLoopIterations = 256;

struct SsaLivenessFacts {
  std::uint64_t graph_arena = 0;
  std::uint64_t graph_revision = 0;

  // Per-slot over-approximation of nodes needed by effects or control.
  std::vector<std::vector<std::uint8_t>> live_nodes;
};

struct SsaSelectedImageAddress {
  SsaHandle block;
  ValueId load;
  ValueId condition;
  std::uint64_t when_true;
  std::uint64_t when_false;
};

struct SsaImageAddressFacts {
  std::uint64_t graph_arena = 0;
  std::uint64_t graph_revision = 0;
  std::vector<SsaSelectedImageAddress> selected;
};

// `through_dispatches` admits a block ValidateSsaDispatchSuccessors admits as
// a member; see there for the one proof that may ask for it.
SsaDecline ValidateSsaReachabilityFacts(const SsaGraph&, const SsaReachabilityFacts&,
                                        std::span<const Group> decoded_sources, Budget&,
                                        bool through_dispatches = false);
SsaDecline ValidateSsaDeadWriteFacts(const SsaGraph&, const SsaReachabilityFacts&,
                                     const SsaDeadWriteFacts&,
                                     std::span<const Group> decoded_sources, Budget&);

// Which storage each block may still read or let be observed at its entry,
// under the graph's observability contract. Every decoded write overwrites,
// retired or not: a write is dead only if every path reaches an overwrite
// first, so the last write before any read or observation is a kept one and
// any set of individually dead writes can retire together. Undeclared, a
// path that cycles forever without an overwrite keeps the storage live.
// Caller validates graph shape first.
class SsaStorageLiveness {
 public:
  static std::optional<SsaStorageLiveness> Compute(const SsaGraph&, Budget&);

  // Checks all paths after the named write.
  SsaDecline CheckDeadWrite(SsaDeadWriteFact, Budget&) const;

  // Whether the block in `slot` may read or let be observed `storage` as it
  // holds it on entry. Storage the graph never names is not live.
  bool LiveIn(std::size_t slot, StorageId storage) const;

 private:
  explicit SsaStorageLiveness(const SsaGraph& graph) : graph_(&graph) {}

  bool Scan(const SsaBlock&, std::size_t first_boundary, std::vector<std::uint8_t>& states,
            Budget&) const;
  bool Leaves(const SsaBlock&, std::size_t slot, std::size_t storage) const;
  const SsaGraph* graph_;
  std::vector<StorageId> storages_;  // sorted
  // Per slot: whether its successor edges are complete, and per storage
  // whether the storage is live at the block's entry.
  std::vector<std::uint8_t> complete_;
  std::vector<std::vector<std::uint8_t>> live_in_;

  // Per slot: what a call to a leaf callee reads, when the block makes one.
  std::vector<std::optional<std::vector<StorageId>>> leaf_reads_;
};

// Which promoted private frame slots each block may still read as they hold
// them on entry. A slot's entry value is live where anything but the block's
// own pass-through names it, or where it passes through to a successor that
// reads it. No callee or caller can reach a private slot, so control leaving
// the function ends every one; a block whose successors are not proved
// complete keeps every slot live. Caller validates graph shape first.
class SsaFrameLiveness {
 public:
  static std::optional<SsaFrameLiveness> Compute(const SsaGraph&, Budget&);
  bool LiveIn(std::size_t slot, std::int64_t offset, std::uint32_t size) const;

 private:
  std::vector<std::pair<std::int64_t, std::uint32_t>> slots_;  // sorted
  std::vector<std::vector<std::uint8_t>> live_in_;             // per block slot, per frame slot
};

// One-off form of SsaStorageLiveness::CheckDeadWrite.
SsaDecline CheckSsaDeadStorageWrite(const SsaGraph&, SsaDeadWriteFact, Budget&);
SsaDecline ValidateSsaPredecessorCopyFacts(const SsaGraph&, const SsaReachabilityFacts&,
                                           const SsaPredecessorCopyFacts&,
                                           std::span<const Group> decoded_sources, Budget&);
SsaDecline ValidateSsaPhiConstantFacts(const SsaGraph&, const SsaReachabilityFacts&,
                                       const SsaPhiConstantFacts&,
                                       std::span<const Group> decoded_sources, Budget&);
// A value a bounded loop establishes is licensed by `loops`, which this
// rechecks first: the incoming that would otherwise have to agree is the
// loop's own accumulator, which disagrees with itself on every iteration.
SsaDecline ValidateSsaConstantFacts(const SsaGraph&, const SsaReachabilityFacts&,
                                    const SsaConstantFacts&, std::span<const Group> decoded_sources,
                                    Budget&, const SsaBoundedLoopFacts* loops = nullptr);
SsaDecline ValidateSsaSccpFacts(const SsaGraph&, const SsaSccpFacts&,
                                std::span<const Group> decoded_sources, Budget&,
                                const SsaBoundedLoopFacts* loops = nullptr);
SsaDecline ValidateSsaDominanceFacts(const SsaGraph&, const SsaDominanceFacts&,
                                     std::span<const Group> decoded_sources, Budget&);
SsaDecline ValidateSsaGuardEdgeFact(const SsaGraph&, const SsaGuardEdgeFact&,
                                    std::span<const Group> decoded_sources, Budget&);
SsaDecline ValidateSsaBranchPredicateFact(const SsaGraph&, const SsaBranchPredicateFact&,
                                          std::span<const Group> decoded_sources, Budget&);
SsaDecline ValidateSsaDirectBlockBinding(const SsaGraph&, SsaHandle,
                                         std::span<const Group> decoded_sources, Budget&);
SsaDecline ValidateSsaBoundedTableBlockBinding(const SsaGraph&, SsaHandle,
                                               std::span<const Group> decoded_sources, Budget&);
// ValidateSsa. Nothing a graph carries binds to the decoded sources any more:
// a bounded table fold re-derives its bound from the graph as it stands.
SsaDecline ValidateSsaWithSources(const SsaGraph&, std::span<const Group> decoded_sources, Budget&);

// ValidateSsa, plus that the work a retired loop or call no longer does is
// unobservable: everything a bounded_exit block changes is dead where it
// leaves, registers and promoted frame slots alike, and every register a
// retired_call's callee body writes is dead where the call returns. What it
// rechecks cannot come undone (no later edit adds a read), so a pass that
// retires either runs it on its candidate once, where ValidateSsa, which every
// pass runs, keeps only what the block and the body themselves show.
SsaDecline ValidateSsaRetiredWork(const SsaGraph&, Budget&);

// ValidateSsa, plus the evidence each retired_call rests on, as the graph can
// still show it: the call's target resolves to the body the rewrite names, the
// body is a leaf, and it returns through the register the call links. For a
// resolved_call, that its original target resolves to the location it names. It is
// evidence about the call when it was made, which a later edit elsewhere can
// stop the graph from re-deriving, so the pass that retires a call checks it
// on its candidate, for the blocks it just rewrote; ValidateSsa then keeps
// only that the body is carried.
SsaDecline ValidateSsaRetiredCalls(const SsaGraph&, std::span<const std::uint32_t> slots, Budget&);
SsaDecline ValidateSsaIndexBoundFact(const SsaGraph&, const SsaIndexBoundFact&,
                                     std::span<const Group> decoded_sources, Budget&);
SsaDecline ValidateSsaBoundedTableAddressFact(const SsaGraph&, const SsaBoundedTableAddressFact&,
                                              std::span<const Group> decoded_sources, Budget&);
SsaDecline ValidateSsaBoundedLoopFacts(const SsaGraph&, const SsaBoundedLoopFacts&,
                                       std::span<const Group> decoded_sources, Budget&);
SsaDecline ValidateSsaLivenessFacts(const SsaGraph&, const SsaLivenessFacts&, Budget&);
SsaDecline ValidateSsaImageAddressFacts(const SsaGraph&, const SsaImageAddressFacts&, Budget&);

// The image locations a graph's stores can write, gathered once for many
// queries. Under `page_aligned_placement`, a bitwise function of one image
// location that its load bias cannot change is a number, not a location. Under
// the graph's load bias, a number at or above it is the location it names.
//
// A load yields what the graph's own records say it reads (a fold, a path
// read) or, given `facts`, what a declared constant range or relocated slot
// holds; anything else it yields is a runtime value, which the whole-image
// writer scan accounts for. A checker passes no facts: a prover that has them
// then sees every store the checker sees, and more.
//
// A store at an image-derived address it cannot place may write anywhere the
// image is writable, so it conflicts with every read not declared `read_only`.
class SsaImageStores {
 public:
  static std::optional<SsaImageStores> Collect(const SsaGraph&, bool page_aligned_placement,
                                               Budget&, const ImageFacts* facts = nullptr);
  bool Conflicts(std::uint64_t address, unsigned size, bool read_only = false) const;

  // Only the stores it could place, or all of them if it could not read the
  // graph. A path read rests on the run's writer-scan declaration for stores
  // nobody can place, as the declaration it records was admitted under it.
  bool ConflictsPlaced(std::uint64_t address, unsigned size) const;

  // Each store it placed, with the block and node that make it, in graph order.
  struct Placed {
    std::uint64_t address;
    unsigned size;
    std::uint64_t block;
    ValueId node;
  };

  std::span<const Placed> placed() const { return placed_; }

 private:
  bool unknown_ = false;

  // Some store's image-derived address is unknown, which only a location the
  // loader maps without write permission is safe from.
  bool unplaced_ = false;
  unsigned widest_ = 0;
  std::vector<std::pair<std::uint64_t, unsigned>> stores_;  // sorted
  std::vector<Placed> placed_;
};

// Drops each path read a store the graph places writes: the load then reads
// whatever that store left, not the declared value. A builder passes the run's
// facts, so the checker, which has only the graph, finds no store it missed;
// an edit that lets the graph place a store it could not before passes none.
// Returns what it dropped, or empty when the budget ran out.
struct SsaDroppedPathRead {
  SsaHandle block;
  ValueId node;
  std::uint64_t address;
};

std::optional<std::vector<SsaDroppedPathRead>> SsaDropWrittenPathReads(
    SsaGraph&, Budget&, const ImageFacts* facts = nullptr);
// One-off form of SsaImageStores::Conflicts.
std::optional<bool> SsaConflictingImageStore(const SsaGraph&, std::uint64_t address, unsigned size,
                                             bool page_aligned_placement, Budget&,
                                             bool read_only = false,
                                             const ImageFacts* facts = nullptr);
// Recheck one fact after the containing graph's node shapes have been validated.
SsaDecline CheckSsaSelectedImageAddress(const SsaBlock&, ValueId load, ValueId condition,
                                        std::uint64_t when_true, std::uint64_t when_false, Budget&);
SsaDecline ValidateSsaDirectSuccessors(const SsaBlock&, Budget&);

// ValidateSsaDirectSuccessors, or a jump whose edges have the shape the CFG
// builder gives a table dispatch and which no fold proves yet. Only the proof
// of the guard such a fold rests on takes those edges as the block's
// successors: the fold it makes possible is what completes the block, and the
// graph that carries the fold is checked with every block complete
// (TableGuardStillBounds), so nothing that is kept rests on this.
SsaDecline ValidateSsaDispatchSuccessors(const SsaBlock&, Budget&);

// One block as a member of a closed population: bound to its sources, every
// successor enumerated, and each edge of a kind the population admits. Whether
// the successors are members too is the caller's to check.
SsaDecline ValidateSsaClosedMember(const SsaBlock&, std::span<const Group> decoded_sources, Budget&,
                                   bool through_dispatches = false);
// The image location a load reads when its address is image arithmetic
// (offsets, copies, a PC page), and whether naming it needs the run's
// page-aligned placement declaration.
std::optional<std::pair<std::uint64_t, bool>> SsaLoadLocation(std::span<const Node>, ValueId load,
                                                              std::optional<std::uint64_t> bias);
// The same for an address itself, at most 64 nodes deep.
std::optional<std::pair<std::uint64_t, bool>> SsaImageLocation(std::span<const Node>,
                                                               ValueId address,
                                                               std::optional<std::uint64_t> bias);
// Whether every value that can arrive as `value` is image arithmetic in the
// block that computes it: a page base carried in a register is as much a
// location as the ADRP that made it. A join met again round a loop adds no new
// value. Past 256 steps it returns false. Empty when the budget declines. The SCCP
// read fold spells such a register as a location, and SsaImageStores reads it as
// one before that rewrite, so a store through it is judged the same either way.
std::optional<bool> SsaImageBasedValue(const SsaGraph&, const SsaValue&, Budget&);

// Per slot, per register phi, private-slot phi and node, whether some way to
// the value carries an image location, possibly moved on by arithmetic such
// as a loop's own step. A load yields one where the graph records it does
// (a fold of a location, a path read of a relocated slot) or, given `facts`,
// where it reads a relocated slot they declare. A checker passes no facts; a
// prover that has them marks everything the checker marks, and more.
// Computed for the whole graph at once, so no walk is cut short. Empty when
// the budget declines.
struct SsaImageReach {
  std::vector<std::vector<std::uint8_t>> phis;
  std::vector<std::vector<std::uint8_t>> frames;
  std::vector<std::vector<std::uint8_t>> nodes;
};

std::optional<SsaImageReach> SsaImageReaching(const SsaGraph&, Budget&,
                                              const ImageFacts* facts = nullptr);
// What a block's call leaves in the storage a clobber names, when the graph
// carries a declared result for the callee it reaches. A block reaching more
// than one callee, or one whose target is not a known image location, has no
// answer.
std::optional<std::uint64_t> SsaClobberResult(const SsaGraph&, const SsaValue& clobber);

// The image location a value must hold, when every way of reaching it computes
// the same one. Phis are followed through predecessors; one that an entry path
// also reaches settles to nothing, because that path contributes no input.
std::optional<std::uint64_t> SsaSettledLocation(const SsaGraph&, const SsaValue&, Budget&);
std::optional<std::uint64_t> SsaSettledTarget(const SsaGraph&, const SsaBlock&, ValueId target,
                                              Budget&);
// The callee a block reaches, whether its edge names one or its transfer
// settles on one. A block reaching more than one, or none that is known, has
// no answer; that is the same rule a declared result is attributed under.
std::optional<std::uint64_t> SsaCallTarget(const SsaGraph&, const SsaBlock&, Budget&);

// The value a same-width zext or low extract copies, followed to its source.
ValueId SsaCopySource(std::span<const Node> nodes, ValueId id);

// The 1-bit values that fixing `condition` to `value` decides by itself: the
// condition, then what it negates or copies, in turn. A rewrite may use any of
// them where the condition was, so a check that fixes the condition fixes all.
std::vector<std::pair<ValueId, bool>> SsaImpliedBits(std::span<const Node> nodes, ValueId condition,
                                                     bool value);
// Per node of one block, whether execution uses its value. Roots are every
// access address (a retired one's is still checked), executed store values,
// kept writes, edges, executed transfers (a dispatch rewrite's replacement,
// not the jump it supersedes), fold and frame records, frame phis and reads
// from other blocks; register phis only name storage a kept write fills. Live
// pure nodes pass liveness to their inputs. A load's own
// value is live only through a live user. Empty when the budget runs out.
std::optional<std::vector<std::uint8_t>> SsaLiveValues(const SsaGraph&, SsaHandle, Budget&);

// The storages a block's call hands its callee to read, when the callee is a
// leaf whose body the graph carries: what the body reads before it writes it.
// Such a callee cannot fault, having no access to memory, so this is all of
// what it observes. Nothing when the callee is not settled or not a leaf.
std::optional<std::vector<StorageId>> SsaLeafCalleeReads(const SsaGraph&, const SsaBlock&, Budget&);

// Whether two values of one block are the same computation: equal literals,
// reads of the same value, or one operation over equal inputs.
bool SsaSameValue(const SsaBlock&, ValueId, ValueId);

// The image bytes a load can read, [first, end), when its address is an image
// location plus arithmetic over literals and at most one 0/1 choice.
std::optional<std::pair<std::uint64_t, std::uint64_t>> SsaLoadImageSpan(
    const SsaBlock&, ValueId load, std::optional<std::uint64_t> bias = std::nullopt);
// The declared value a path load reads, when its address is one image location
// and the facts cover the whole access: a stable relocated slot read as a
// little-endian 64-bit value, or constant bytes at the load's own byte order.
std::optional<SsaPathRead> ResolveSsaPathRead(std::span<const Node> nodes, ValueId load,
                                              ImageFacts facts);
SsaDecline ValidateSsaSourceBinding(const SsaBlock&, std::span<const Group>, Budget&);

}  // namespace nyx::ir
