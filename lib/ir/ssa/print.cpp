#include "nyx/ir/ssa/print.hpp"

#include <algorithm>
#include <charconv>
#include <limits>
#include <string_view>

namespace nyx::ir {
namespace {
class Counter {
 public:
  bool Text(std::string_view part) {
    if (part.size() > std::numeric_limits<std::size_t>::max() - size_) return false;
    size_ += part.size();
    return true;
  }

  std::size_t size() const { return size_; }

 private:
  std::size_t size_ = 0;
};

class Writer {
 public:
  explicit Writer(char* out) : out_(out) {}

  bool Text(std::string_view part) {
    for (char ch : part) *out_++ = ch;
    return true;
  }

 private:
  char* out_;
};

template <class Sink>
bool Number(Sink& out, std::uint64_t number) {
  char digits[20];
  const auto result = std::to_chars(digits, digits + sizeof(digits), number);
  return out.Text({digits, result.ptr});
}

template <class Sink>
bool Signed(Sink& out, std::int64_t number) {
  char digits[20];
  const auto result = std::to_chars(digits, digits + sizeof(digits), number);
  return out.Text({digits, result.ptr});
}

template <class Sink>
bool Handle(Sink& out, SsaHandle handle) {
  return out.Text("b") && Number(out, handle.slot) && out.Text(".") &&
         Number(out, handle.generation);
}

template <class Sink>
bool Value(Sink& out, SsaValue value) {
  const char* kind = nullptr;
  switch (value.kind) {
    case SsaValueKind::input:
      kind = "input";
      break;
    case SsaValueKind::phi:
      kind = "phi";
      break;
    case SsaValueKind::node:
      kind = "node";
      break;
    case SsaValueKind::clobber:
      kind = "clobber";
      break;
    case SsaValueKind::frame_phi:
      kind = "frame_phi";
      break;
    case SsaValueKind::frame_input:
      kind = "frame_input";
      break;
  }

  return out.Text(kind) && out.Text("(") && Handle(out, value.block) && out.Text(",") &&
         Number(out, value.index) && out.Text(")");
}

template <class Sink>
bool TransferText(Sink& out, const Transfer& transfer) {
  const char* kind = nullptr;
  switch (transfer.kind) {
    case TransferKind::jump:
      kind = "jump";
      break;
    case TransferKind::conditional:
      kind = "conditional";
      break;
    case TransferKind::call:
      kind = "call";
      break;
    case TransferKind::return_:
      kind = "return";
      break;
  }

  if (!out.Text(" transfer ") || !out.Text(kind) || !out.Text(" %") ||
      !Number(out, transfer.target))
    return false;
  if (transfer.condition && (!out.Text(" condition %") || !Number(out, *transfer.condition)))
    return false;
  if (transfer.alternative && (!out.Text(" alternative %") || !Number(out, *transfer.alternative)))
    return false;
  if (transfer.continuation &&
      (!out.Text(" continuation %") || !Number(out, *transfer.continuation)))
    return false;
  return true;
}

const char* EdgeName(SsaEdgeKind kind) {
  switch (kind) {
    case SsaEdgeKind::fallthrough:
      return "fallthrough";
    case SsaEdgeKind::branch:
      return "branch";
    case SsaEdgeKind::callee:
      return "callee";
    case SsaEdgeKind::return_:
      return "return";
    case SsaEdgeKind::potential_return:
      return "potential_return";
    case SsaEdgeKind::opaque_unknown:
      return "opaque_unknown";
    case SsaEdgeKind::trap:
      return "trap";
  }

  return "invalid";
}

const char* TargetName(SsaTargetKind kind) {
  switch (kind) {
    case SsaTargetKind::image_location:
      return "image_location";
    case SsaTargetKind::absolute_runtime:
      return "absolute_runtime";
    case SsaTargetKind::unknown:
      return "unknown";
  }

  return "invalid";
}

template <class Sink>
bool Emit(const SsaGraph& graph, Sink& out) {
  if (!out.Text("ssa revision ") || !Number(out, graph.revision()) || out.Text("\n") == false)
    return false;
  for (const auto entry : graph.entries())
    if (!out.Text("entry ") || !Handle(out, entry) || !out.Text("\n")) return false;
  // What the stores a check places rest on: with every way in declared, a
  // register carried in holds only what its listed predecessors leave.
  if (graph.entries_closed() && !out.Text("entries closed\n")) return false;
  const auto& observed = graph.observability();
  if (observed.declared || observed.faults_terminal) {
    const auto list = [&](const char* label, const std::vector<StorageId>& ids) {
      if (!out.Text(label)) return false;
      for (const auto id : ids)
        if (!out.Text(" ") || !Number(out, id)) return false;
      return true;
    };

    if (!out.Text("observability") ||
        !out.Text(observed.faults_terminal ? " faults_terminal" : "") ||
        (observed.declared &&
         (!list(" call", observed.at_call) || !list(" return", observed.at_return) ||
          !list(" preserved", observed.preserved))) ||
        !out.Text("\n"))
      return false;
  }

  for (const auto& result : graph.call_results()) {
    if (!out.Text("call_result @") || !Number(out, result.target) || !out.Text(" s") ||
        !Number(out, result.storage) || !out.Text(" = ") || !Number(out, result.value) ||
        !out.Text("\n"))
      return false;
  }

  for (const auto& body : graph.callee_bodies()) {
    if (!out.Text("callee_body @") || !Number(out, body.address) || !out.Text(" bytes"))
      return false;
    for (const auto& group : body.groups)
      for (const auto byte : group.bytes())
        if (!out.Text(" ") || !Number(out, byte)) return false;
    if (!out.Text("\n")) return false;
  }

  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto& block = *graph.Get(*handle);
    if (!out.Text("block ") || !Handle(out, *handle) || !out.Text(" @") ||
        !Number(out, block.address) || !out.Text(" original ") ||
        !Number(out, block.original_block))
      return false;
    if (block.opaque && !out.Text(" opaque")) return false;
    if (block.transition && (!out.Text(" transition ") || !Number(out, *block.transition)))
      return false;
    if (!out.Text(" path_revision ") || !Number(out, block.path_revision) ||
        !out.Text(block.relation_declared_abi ? " relation_abi" : "") ||
        !out.Text(block.relation_return_leaves ? " relation_returns_leave" : "") ||
        !out.Text(block.relation_constant_image ? " relation_image" : "") ||
        !out.Text(block.relation_declared_opaque_control ? " relation_opaque" : "") ||
        !out.Text("\n"))
      return false;
    for (std::size_t i = 0; i < block.original_sources.size(); ++i)
      if (!out.Text("  source ") || !Number(out, block.original_sources[i]) ||
          !out.Text(" group ") || !Number(out, block.source_groups[i]) || !out.Text("\n"))
        return false;
    for (const auto& relation : block.entry_relations)
      if (!out.Text("  relation s") || !Number(out, relation.storage) || !out.Text(" = s") ||
          !Number(out, relation.root) || !out.Text(" + ") || !Number(out, relation.offset) ||
          !out.Text("\n"))
        return false;
    for (std::size_t i = 0; i < block.phis.size(); ++i) {
      const auto& phi = block.phis[i];
      if (!out.Text("  phi ") || !Number(out, i) || !out.Text(" s") || !Number(out, phi.storage) ||
          !out.Text(":") || !Number(out, phi.width) ||
          !out.Text(phi.external_entry ? " entry" : "") ||
          !out.Text(block.clobbers[i] ? " clobber" : "") || !out.Text(" <-"))
        return false;
      for (const auto& input : phi.incoming)
        if (!out.Text(" ") || !Handle(out, input.predecessor) || !out.Text("=") ||
            !Value(out, input.value))
          return false;
      if (!out.Text("\n")) return false;
    }

    for (std::size_t i = 0; i < block.frame_phis.size(); ++i) {
      const auto& phi = block.frame_phis[i];
      if (!out.Text("  frame_phi ") || !Number(out, i) || !out.Text(" offset ") ||
          !Signed(out, phi.offset) || !out.Text(" size ") || !Number(out, phi.size) ||
          !out.Text(phi.external_entry ? " entry" : "") || !out.Text(" <-"))
        return false;
      for (const auto& input : phi.incoming)
        if (!out.Text(" ") || !Handle(out, input.predecessor) || !out.Text("=") ||
            !Value(out, input.value))
          return false;
      if (!out.Text("\n")) return false;
    }

    for (std::size_t i = 0; i < block.nodes.size(); ++i) {
      const auto& node = block.nodes[i];
      const auto* descriptor = Descriptor(node.op);
      if (!out.Text("  %") || !Number(out, i) || !out.Text(" = ") || !out.Text(descriptor->name) ||
          !out.Text(":") || !Number(out, node.width))
        return false;
      for (unsigned j = 0; j < descriptor->arity; ++j)
        if (!out.Text(" %") || !Number(out, node.inputs[j])) return false;
      if (node.op == Op::constant || node.op == Op::extract || node.op == Op::image_address)
        if (!out.Text(" #") || !Number(out, node.immediate)) return false;
      if (node.op == Op::read || node.op == Op::write)
        if (!out.Text(" s") || !Number(out, node.storage)) return false;
      if (node.op == Op::load || node.op == Op::store || node.op == Op::exclusive_load ||
          node.op == Op::exclusive_store)
        if (!out.Text(" align ") || !Number(out, node.access.alignment) ||
            !out.Text(node.access.byte_order == ByteOrder::little ? " le" : " be") ||
            !out.Text(node.access.decline_on_unaligned ? " strict" : " relaxed"))
          return false;
      if (std::binary_search(block.disabled_effects.begin(), block.disabled_effects.end(), i) &&
          !out.Text(" disabled"))
        return false;
      if (std::binary_search(block.dead_pure_nodes.begin(), block.dead_pure_nodes.end(), i) &&
          !out.Text(" dead"))
        return false;
      if (!out.Text("\n")) return false;
    }

    for (const auto& read : block.reads)
      if (!out.Text("  read %") || !Number(out, read.node) || !out.Text(" from phi ") ||
          !Number(out, read.phi) || !out.Text(" = ") || !Value(out, read.value) ||
          (read.predecessor_copy && (!out.Text(" copy ") || !Value(out, *read.predecessor_copy) ||
                                     !out.Text(" closed_entries"))) ||
          !out.Text("\n"))
        return false;
    for (const auto& access : block.frame_accesses) {
      if (!out.Text("  frame_access %") || !Number(out, access.node) || !out.Text(" frame ") ||
          !Number(out, access.phi))
        return false;
      if (access.replacement && (!out.Text(" -> ") || !Value(out, *access.replacement)))
        return false;
      if (!out.Text("\n")) return false;
    }

    for (const auto& retired : block.retired_loads) {
      if (!out.Text("  retired_load %") || !Number(out, retired.node)) return false;
      if (retired.basis == SsaRetiredLoadBasis::written_before) {
        if (!out.Text(" written_before %") || !Number(out, retired.store)) return false;
      } else if (!out.Text(" declared_image ") || !Number(out, retired.range_address) ||
                 !out.Text("+") || !Number(out, retired.range_bytes)) {
        return false;
      }

      if (!out.Text("\n")) return false;
    }

    for (const auto& read : block.path_reads)
      if (!out.Text("  path_read %") || !Number(out, read.node) || !out.Text(" @") ||
          !Number(out, read.address) || !out.Text(read.relocated ? " relocated " : " value ") ||
          !Number(out, read.value) ||
          !out.Text(read.page_aligned_placement ? " page_aligned\n" : "\n"))
        return false;
    for (const auto& fold : block.constant_loads) {
      if (fold.kind == SsaConstantKind::bounded_table) {
        if (!out.Text("  constant_load %") || !Number(out, fold.node) ||
            !out.Text(" bounded_table from ") || !Number(out, fold.source_address) ||
            !out.Text(" stride ") || !Number(out, fold.table_stride) || !out.Text(" index %") ||
            !Number(out, fold.table_index) || !out.Text(" rows ") ||
            !Number(out, fold.table_bytes.size()))
          return false;
        for (const auto& row : fold.table_bytes) {
          if (!out.Text(" [")) return false;
          for (unsigned byte = 0; byte < 8; ++byte)
            if ((byte && !out.Text(" ")) || !Number(out, row[byte])) return false;
          if (!out.Text("]")) return false;
        }

        if (!out.Text(" no_access") || (fold.read_only && !out.Text(" read_only")) ||
            !out.Text("\n"))
          return false;
        continue;
      }

      if (!out.Text("  constant_load %") || !Number(out, fold.node) ||
          !out.Text(fold.kind == SsaConstantKind::literal ? " literal " : " image_location ") ||
          !Number(out, fold.value) || !out.Text(" from ") || !Number(out, fold.source_address))
        return false;
      if (fold.condition && (!out.Text(" when %") || !Number(out, *fold.condition) ||
                             !out.Text(" else ") || !Number(out, fold.alternative_value) ||
                             !out.Text(" from ") || !Number(out, fold.alternative_source_address)))
        return false;
      if ((fold.skip_access && !out.Text(" no_access")) ||
          (fold.read_only && !out.Text(" read_only")) || !out.Text("\n"))
        return false;
    }

    for (const auto& boundary : block.boundaries) {
      if (!out.Text("  boundary ") || !Number(out, boundary.first_node) || !out.Text("+") ||
          !Number(out, boundary.node_count))
        return false;
      for (const auto& write : boundary.writes)
        if (!out.Text(" write s") || !Number(out, write.storage) || !out.Text("=%") ||
            !Number(out, write.value))
          return false;
      if (boundary.transfer && !TransferText(out, *boundary.transfer)) return false;
      if (!out.Text("\n")) return false;
    }

    for (const auto& rewrite : block.control_rewrites) {
      const auto* rule = rewrite.rule == RewriteRule::dispatch_branch    ? "dispatch_branch"
                         : rewrite.rule == RewriteRule::decided_dispatch ? "decided_dispatch"
                         : rewrite.rule == RewriteRule::bounded_exit     ? "bounded_exit"
                         : rewrite.rule == RewriteRule::retired_call     ? "retired_call"
                         : rewrite.rule == RewriteRule::resolved_call    ? "resolved_call"
                                                                         : "folded_condition";
      if (!out.Text("  rewrite ") || !out.Text(rule) || !out.Text(" boundary ") ||
          !Number(out, rewrite.boundary) || !out.Text(" condition %") ||
          !Number(out, rewrite.condition) ||
          !out.Text(rewrite.condition_value ? " true" : " false") || !out.Text(" destinations ") ||
          !Number(out, rewrite.when_true) || !out.Text(" ") || !Number(out, rewrite.when_false) ||
          !out.Text(" revisions ") || !Number(out, rewrite.from_revision) || !out.Text(" ") ||
          !Number(out, rewrite.to_revision) || !TransferText(out, rewrite.original) ||
          !TransferText(out, rewrite.replacement) ||
          (rewrite.rule == RewriteRule::bounded_exit &&
           (!out.Text(" iterations ") || !Number(out, rewrite.iterations))))
        return false;
      for (const auto& read : rewrite.witness)
        if (!out.Text(" witness %") || !Number(out, read.node) ||
            !out.Text(read.when ? " true " : " false ") || !Number(out, read.address) ||
            !out.Text(":") || !Number(out, read.width) || !out.Text("=") ||
            !Number(out, read.value) || !out.Text(read.relocated ? " relocated" : ""))
          return false;
      if (!out.Text("\n")) return false;
    }

    for (const auto& omission : block.store_omissions)
      if (!out.Text("  omit_store %") || !Number(out, omission.store) ||
          !out.Text(" overwritten_by %") || !Number(out, omission.overwriter) ||
          !out.Text(" base s") || !Number(out, omission.base_storage) || !out.Text(" offset ") ||
          !Number(out, omission.offset_bytes) || !out.Text(" revisions ") ||
          !Number(out, omission.from_revision) || !out.Text(" ") ||
          !Number(out, omission.to_revision) || !out.Text("\n"))
        return false;
    for (const auto& omission : block.paired_load_omissions)
      if (!out.Text("  omit_paired_loads %") || !Number(out, omission.loads[0]) ||
          !out.Text(" %") || !Number(out, omission.loads[1]) || !out.Text(" stores %") ||
          !Number(out, omission.stores[0]) || !out.Text(" %") || !Number(out, omission.stores[1]) ||
          !out.Text(" base s") || !Number(out, omission.base_storage) || !out.Text(" offsets ") ||
          !Number(out, omission.offset_bytes[0]) || !out.Text(" ") ||
          !Number(out, omission.offset_bytes[1]) || !out.Text(" revisions ") ||
          !Number(out, omission.from_revision) || !out.Text(" ") ||
          !Number(out, omission.to_revision) || !out.Text("\n"))
        return false;
    for (std::size_t i = 0; i < block.destination_nodes.size(); ++i)
      if (!out.Text("  destination ") || !Number(out, i) || !out.Text(" ") ||
          !out.Text(Descriptor(block.destination_nodes[i].op)->name) || !out.Text(" ") ||
          !Number(out, block.destination_nodes[i].immediate) || !out.Text("\n"))
        return false;
    for (const auto& mark : block.dead_storage_writes)
      if (!out.Text("  dead_write boundary ") || !Number(out, mark.boundary) ||
          !out.Text(" index ") || !Number(out, mark.index) || !out.Text(" closed_entries\n"))
        return false;
    for (const auto& exit : block.exits)
      if (!out.Text("  exit s") || !Number(out, exit.storage) || !out.Text("=") ||
          !Value(out, exit.value) || !out.Text("\n"))
        return false;
    for (std::size_t i = 0; i < block.frame_exits.size(); ++i)
      if (!out.Text("  frame_exit ") || !Number(out, i) || !out.Text("=") ||
          !Value(out, block.frame_exits[i]) || !out.Text("\n"))
        return false;
    for (const auto& edge : block.edges) {
      if (!out.Text("  edge ") || !out.Text(EdgeName(edge.kind)) || !out.Text(" target ") ||
          !out.Text(TargetName(edge.target_kind)) || !out.Text(" @") || !Number(out, edge.address))
        return false;
      if (edge.target_block && (!out.Text(" ") || !Handle(out, *edge.target_block))) return false;
      if (edge.condition && (!out.Text(" when %") || !Number(out, *edge.condition) ||
                             !out.Text(*edge.when ? " true" : " false")))
        return false;
      const auto& a = edge.assumptions;
      if (a.constant_image && !out.Text(" constant_image")) return false;
      if (a.callee_returns_to_continuation && !out.Text(" callee_returns")) return false;
      if (a.return_leaves && !out.Text(" return_leaves")) return false;
      if (a.unresolved_target && !out.Text(" unresolved")) return false;
      if (a.declared_opaque_control && !out.Text(" opaque_control")) return false;
      if (a.entry_relations && !out.Text(" entry_relations")) return false;
      if (a.declared_abi && !out.Text(" declared_abi")) return false;
      if (a.declared_return_leaves && !out.Text(" declared_return_leaves")) return false;
      if (a.declared_noreturn && !out.Text(" declared_noreturn")) return false;
      if (!out.Text("\n")) return false;
    }
  }

  return true;
}
}  // namespace

namespace {
SsaTextResult PrintValidatedSsa(const SsaGraph& graph, Budget& budget) {
  Counter counter;
  if (!Emit(graph, counter) || counter.size() == std::numeric_limits<std::size_t>::max())
    return {{}, SsaDecline::resource_limit};
  if (budget.try_consume({counter.size(), counter.size() + 1}) != BudgetDecline::none)
    return {{}, SsaDecline::resource_limit};
  std::string output(counter.size(), '\0');
  Writer writer(output.data());
  if (!Emit(graph, writer)) return {{}, SsaDecline::invalid_graph};
  return {std::move(output), SsaDecline::none};
}
}  // namespace

SsaTextResult PrintSsa(const SsaGraph& graph, Budget& budget) {
  const auto valid = ValidateSsa(graph, budget);
  return valid == SsaDecline::none ? PrintValidatedSsa(graph, budget) : SsaTextResult{{}, valid};
}

SsaTextResult PrintSsa(const SsaGraph& graph, std::span<const Group> sources, Budget& budget) {
  const auto valid = ValidateSsaWithSources(graph, sources, budget);
  return valid == SsaDecline::none ? PrintValidatedSsa(graph, budget) : SsaTextResult{{}, valid};
}

}  // namespace nyx::ir
