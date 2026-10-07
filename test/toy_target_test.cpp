#include <algorithm>
#include <map>
#include <random>
#include <set>
#include <string>
#include <tuple>

#include <gtest/gtest.h>

#include "nyx/analysis/cfg.hpp"
#include "nyx/analysis/entry_relations.hpp"
#include "nyx/analysis/path_control.hpp"
#include "nyx/analysis/regions.hpp"
#include "nyx/analysis/ssa/build.hpp"
#include "nyx/analysis/unflatten.hpp"
#include "nyx/eval/ssa.hpp"
#include "nyx/passes/pipeline.hpp"
#include "nyx/recovery/control.hpp"
#include "nyx/recovery/image.hpp"
#include "nyx/recovery/mba.hpp"
#include "nyx/recovery/memory.hpp"
#include "nyx/recovery/overwritten_store.hpp"
#include "nyx/recovery/paired_load.hpp"
#include "nyx/target/toy/decode.hpp"

// G-toy: the generic discovery, recovery, SSA and pass stages driven by a
// big-endian, variable-width target that shares nothing with AArch64. Every
// registered pass must propose on some toy program, and every proposed graph
// must agree with an interpreter written directly over the toy bytes.
namespace nyx::toy {
namespace {

constexpr std::uint64_t kImage = 0x1000;               // image location of the code
constexpr std::uint64_t kBias = 0x200000000 - kImage;  // runtime = location + bias
constexpr std::uint64_t kStack = 0x300000000;
constexpr std::uint64_t kLanding = 0x400000000;

// A two-pass toy assembler: relative fields are fixed up once labels are known.
class Asm {
 public:
  Asm& label(const std::string& name) {
    labels_[name] = code.size();
    return *this;
  }

  Asm& op3(std::uint8_t op, unsigned d, unsigned s, unsigned t) {
    return emit({op, Regs(d, s), std::uint8_t(t << 4)});
  }

  Asm& add(unsigned d, unsigned s, unsigned t) { return op3(0x10, d, s, t); }

  Asm& sub(unsigned d, unsigned s, unsigned t) { return op3(0x11, d, s, t); }

  Asm& and_(unsigned d, unsigned s, unsigned t) { return op3(0x12, d, s, t); }

  Asm& or_(unsigned d, unsigned s, unsigned t) { return op3(0x13, d, s, t); }

  Asm& xor_(unsigned d, unsigned s, unsigned t) { return op3(0x14, d, s, t); }

  Asm& mul(unsigned d, unsigned s, unsigned t) { return op3(0x15, d, s, t); }

  Asm& not_(unsigned d, unsigned s) { return emit({0x16, Regs(d, s)}); }

  Asm& mov(unsigned d, unsigned s) { return emit({0x17, Regs(d, s)}); }

  Asm& shl(unsigned d, unsigned s, std::uint8_t amount) { return emit({0x18, Regs(d, s), amount}); }

  Asm& movi(unsigned d, std::uint64_t value) {
    emit({0x20, Regs(d, 0)});
    for (int shift = 56; shift >= 0; shift -= 8) code.push_back(std::uint8_t(value >> shift));
    return *this;
  }

  Asm& nop() { return emit({0x02}); }

  Asm& bnez(unsigned s, const std::string& target) {
    return relative({0x51, Regs(0, s), 0, 0}, 2, target);
  }

  Asm& call(const std::string& target) { return relative({0x54, 0, 0}, 1, target); }

  Asm& jr(unsigned s) { return emit({0x55, Regs(0, s)}); }

  Asm& callr(unsigned s) { return emit({0x57, Regs(0, s)}); }

  Asm& movi16(unsigned d, std::uint16_t value) {
    return emit({0x21, Regs(d, 0), std::uint8_t(value >> 8), std::uint8_t(value)});
  }

  Asm& addi(unsigned d, unsigned s, std::int16_t value) {
    return emit({0x22, Regs(d, s), std::uint8_t(std::uint16_t(value) >> 8), std::uint8_t(value)});
  }

  Asm& ld(unsigned d, unsigned s, std::int16_t offset) {
    return emit({0x30, Regs(d, s), std::uint8_t(std::uint16_t(offset) >> 8), std::uint8_t(offset)});
  }

  Asm& st(unsigned base, std::int16_t offset, unsigned value) {
    return emit(
        {0x31, Regs(value, base), std::uint8_t(std::uint16_t(offset) >> 8), std::uint8_t(offset)});
  }

  Asm& ldx(unsigned d, unsigned s, unsigned t) { return op3(0x32, d, s, t); }

  Asm& seqz(unsigned d, unsigned s) { return emit({0x40, Regs(d, s)}); }

  Asm& lea(unsigned d, const std::string& target) {
    return relative({0x23, Regs(d, 0), 0, 0}, 2, target);
  }

  Asm& ldl(unsigned d, const std::string& target) {
    return relative({0x33, Regs(d, 0), 0, 0}, 2, target);
  }

  Asm& beqz(unsigned s, const std::string& target) {
    return relative({0x50, Regs(0, s), 0, 0}, 2, target);
  }

  Asm& bgeu(unsigned s, std::uint8_t bound, const std::string& target) {
    return relative({0x52, Regs(0, s), bound, 0, 0}, 3, target);
  }

  Asm& bgtu(unsigned s, std::uint8_t bound, const std::string& target) {
    return relative({0x56, Regs(0, s), bound, 0, 0}, 3, target);
  }

  Asm& jmp(const std::string& target) { return relative({0x53, 0, 0}, 1, target); }

  Asm& ret() { return emit({0x01}); }

  Asm& trap() { return emit({0x00}); }

  // The end of the selected population; what follows is data only.
  Asm& end() {
    size = code.size();
    return *this;
  }

  Asm& quad(std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8) code.push_back(std::uint8_t(value >> shift));
    return *this;
  }

  // A 64-bit table entry holding `target - base`, fixed up with the labels.
  Asm& offset(const std::string& target, const std::string& base) {
    offsets_.push_back({code.size(), target, base});
    return quad(0);
  }

  std::vector<std::uint8_t> Finish() {
    for (const auto& [at, start, target] : fixups_) {
      const auto offset = std::uint16_t(std::int64_t(labels_.at(target)) - std::int64_t(start));
      code[at] = std::uint8_t(offset >> 8);
      code[at + 1] = std::uint8_t(offset);
    }

    for (const auto& [at, target, base] : offsets_) {
      const auto value = std::uint64_t(labels_.at(target)) - std::uint64_t(labels_.at(base));
      for (unsigned i = 0; i < 8; ++i) code[at + i] = std::uint8_t(value >> (56 - 8 * i));
    }

    return code;
  }

  // The image location of a label.
  std::uint64_t Address(const std::string& name) const { return kImage + labels_.at(name); }

  std::vector<std::uint8_t> code;
  std::size_t size = 0;

 private:
  static std::uint8_t Regs(unsigned d, unsigned s) { return std::uint8_t(d << 4 | s); }

  Asm& emit(std::initializer_list<std::uint8_t> bytes) {
    code.insert(code.end(), bytes);
    return *this;
  }

  Asm& relative(std::initializer_list<std::uint8_t> bytes, std::size_t field,
                const std::string& target) {
    fixups_.push_back({code.size() + field, code.size(), target});
    return emit(bytes);
  }

  struct Fixup {
    std::size_t at, start;
    std::string target;
  };

  struct Offset {
    std::size_t at;
    std::string target, base;
  };

  std::map<std::string, std::size_t> labels_;
  std::vector<Fixup> fixups_;
  std::vector<Offset> offsets_;
};

// ---- The independent reference: the toy semantics read straight from bytes.

struct Machine {
  std::array<std::uint64_t, kRegisters> r{};
  std::vector<std::uint8_t> stack = std::vector<std::uint8_t>(4096);
  std::uint64_t pc = 0;  // runtime
  bool trapped = false;  // stopped at a trap instruction
};

bool Load(const Machine& m, std::span<const std::uint8_t> image, std::uint64_t address,
          std::uint64_t& value) {
  const std::uint8_t* bytes = nullptr;
  if (address >= kStack && address - kStack <= m.stack.size() - 8)
    bytes = &m.stack[address - kStack];
  const auto base = kImage + kBias;
  if (address >= base && address - base <= 4096 - 8) bytes = &image[address - base];
  if (!bytes) return false;
  value = 0;
  for (unsigned i = 0; i < 8; ++i) value = value << 8 | bytes[i];
  return true;
}

// What the reference saw: every opcode it executed, and for each conditional
// or seqz (by image location) the directions or results it took.
struct Coverage {
  std::set<std::uint8_t> opcodes;
  std::set<std::uint64_t> locations;
  std::map<std::uint64_t, std::set<bool>> directions;
};

// The one declared external callee: it lies outside the selected population,
// so neither side lifts it, and both apply this same summary. It keeps the
// toy convention: callee-saved registers and SP are left alone.
void ExternalCallee(std::array<std::uint64_t, kRegisters>& r) {
  r[0] = r[0] * 3 + r[1];
  r[2] = 0x5ca1ab1e;
}

// False on a fault, a trap or a step bound; true on reaching the landing.
// `jumped` receives the target of an indirect jump the run could not follow
// back into the image.
// A run may also start elsewhere and stop at a given address, which is how a
// leaf callee is run on its own.
bool Reference(Machine& m, std::span<const std::uint8_t> image, Coverage* coverage = nullptr,
               std::optional<std::uint64_t> callee = std::nullopt,
               std::optional<std::uint64_t> start = std::nullopt,
               std::optional<std::uint64_t> stop = std::nullopt) {
  m.pc = start.value_or(kImage + kBias);
  for (unsigned step = 0; step < 1000; ++step) {
    if (m.pc == stop.value_or(kLanding)) return true;
    if (callee && m.pc == *callee + kBias) {
      ExternalCallee(m.r);
      m.pc = m.r[kLink];
      continue;
    }

    const auto base = kImage + kBias;
    if (m.pc < base || m.pc - base >= 4096 - 10) return false;
    const auto* b = &image[m.pc - base];
    const unsigned d = b[1] >> 4, s = b[1] & 15, t = b[2] >> 4;
    const auto rel16 = [&](unsigned at) {
      return std::uint64_t(std::int64_t(std::int16_t(std::uint16_t(b[at] << 8 | b[at + 1]))));
    };

    auto& r = m.r;
    std::uint64_t value = 0;
    const auto location = m.pc - kBias;
    if (coverage) {
      coverage->opcodes.insert(b[0]);
      coverage->locations.insert(location);
    }

    const auto took = [&](bool direction) {
      if (coverage) coverage->directions[location].insert(direction);
      return direction;
    };

    switch (b[0]) {
      case 0x00:
        m.trapped = true;
        return false;
      case 0x01:
        m.pc = r[kLink];
        continue;
      case 0x02:
        m.pc += 1;
        continue;
      case 0x10:
        r[d] = r[s] + r[t];
        m.pc += 3;
        continue;
      case 0x11:
        r[d] = r[s] - r[t];
        m.pc += 3;
        continue;
      case 0x12:
        r[d] = r[s] & r[t];
        m.pc += 3;
        continue;
      case 0x13:
        r[d] = r[s] | r[t];
        m.pc += 3;
        continue;
      case 0x14:
        r[d] = r[s] ^ r[t];
        m.pc += 3;
        continue;
      case 0x15:
        r[d] = r[s] * r[t];
        m.pc += 3;
        continue;
      case 0x16:
        r[d] = ~r[s];
        m.pc += 2;
        continue;
      case 0x17:
        r[d] = r[s];
        m.pc += 2;
        continue;
      case 0x18:
        r[d] = r[s] << b[2];
        m.pc += 3;
        continue;
      case 0x20:
        r[d] = 0;
        for (unsigned i = 0; i < 8; ++i) r[d] = r[d] << 8 | b[2 + i];
        m.pc += 10;
        continue;
      case 0x21:
        r[d] = std::uint64_t(b[2]) << 8 | b[3];
        m.pc += 4;
        continue;
      case 0x22:
        r[d] = r[s] + rel16(2);
        m.pc += 4;
        continue;
      case 0x23:
        r[d] = m.pc + rel16(2);
        m.pc += 4;
        continue;
      case 0x30:
        if (!Load(m, image, r[s] + rel16(2), value)) return false;
        r[d] = value;
        m.pc += 4;
        continue;
      case 0x31: {
        const auto address = r[s] + rel16(2);
        if (address < kStack || address - kStack > m.stack.size() - 8) return false;
        for (unsigned i = 0; i < 8; ++i)
          m.stack[address - kStack + i] = std::uint8_t(r[d] >> (56 - 8 * i));
        m.pc += 4;
        continue;
      }
      case 0x32:
        if (!Load(m, image, r[s] + (r[t] << 3), value)) return false;
        r[d] = value;
        m.pc += 3;
        continue;
      case 0x33:
        if (!Load(m, image, m.pc + rel16(2), value)) return false;
        r[d] = value;
        m.pc += 4;
        continue;
      case 0x40:
        r[d] = took(r[s] == 0);
        m.pc += 2;
        continue;
      case 0x50:
        m.pc += took(r[s] == 0) ? rel16(2) : 4;
        continue;
      case 0x51:
        m.pc += took(r[s] != 0) ? rel16(2) : 4;
        continue;
      case 0x52:
        m.pc += took(r[s] >= b[2]) ? rel16(3) : 5;
        continue;
      case 0x56:
        m.pc += took(r[s] > b[2]) ? rel16(3) : 5;
        continue;
      case 0x53:
        m.pc += rel16(1);
        continue;
      case 0x54:
        r[kLink] = m.pc + 3;
        m.pc += rel16(1);
        continue;
      case 0x55:
        m.pc = r[s];
        continue;
      case 0x57:
        value = r[s];
        r[kLink] = m.pc + 2;
        m.pc = value;
        continue;
      default:
        return false;
    }
  }

  return false;
}

// ---- The generic pipeline, in the CLI's order, over toy sources.

struct Recovered {
  std::vector<ir::Group> groups;
  ir::SsaGraph base;
  std::optional<ir::SsaGraph> provisional;
  std::vector<passes::SsaStageSummary> stages;
  std::string body;
};

std::optional<Recovered> Recover(std::span<const std::uint8_t> image, std::size_t size,
                                 std::span<const std::size_t> selection, Budget& budget,
                                 std::string& why,
                                 std::optional<std::uint64_t> leaf = std::nullopt) {
  std::vector<analysis::SourceRecord> records;
  std::vector<ir::Group> groups;
  for (std::size_t offset = 0; offset < size;) {
    auto decoded = Decode(kImage + offset, image.subspan(offset, size - offset), budget);
    if (!decoded.length) {
      why = "undecodable population";
      return std::nullopt;
    }

    const std::vector<std::uint8_t> bytes(image.begin() + offset,
                                          image.begin() + offset + decoded.length);
    if (decoded.group) {
      groups.push_back(*decoded.group);
      records.push_back(
          {kImage + offset, bytes, std::move(decoded.group), analysis::OpaqueReason::none});
    } else if (decoded.reason == DecodeDecline::unsupported) {
      groups.emplace_back(kImage + offset, bytes, std::vector<ir::Node>{},
                          std::vector<ir::Write>{});
      records.push_back(
          {kImage + offset, bytes, std::nullopt, analysis::OpaqueReason::unsupported,
           OpaqueTrap(bytes) ? analysis::OpaqueControl::trap : analysis::OpaqueControl::unknown});
    } else {
      why = "toy decode declined";
      return std::nullopt;
    }

    offset += decoded.length;
  }

  // The image is declared read-only, as the CLI declares a read-only segment.
  const ir::ConstantImageRange whole{kImage, image};
  const ir::ImageFacts facts{std::span(&whole, 1), {}, true, {}};
  const std::array<std::uint64_t, 1> entries{kImage};
  auto cfg = analysis::BuildCfg(records, entries, budget, {}, facts);
  if (!cfg.cfg) {
    why = "cfg declined";
    return std::nullopt;
  }

  auto retained = ir::RetainImageFacts(facts, cfg.cfg->refuted_constant_spans(),
                                       cfg.cfg->refuted_pointer_indices(), budget);
  if (!retained) {
    why = "image facts declined";
    return std::nullopt;
  }

  auto selected = analysis::BuildRegions(std::move(*cfg.cfg), budget);
  if (!selected.regions) {
    why = "regions declined";
    return std::nullopt;
  }

  const auto& regions = *selected.regions;
  auto related = analysis::ProveEntryRelations(regions.graph(), budget, {true, kCalleeSaved, true});
  if (!related.relations) {
    why = "entry relations declined";
    return std::nullopt;
  }

  const auto view = retained->view();
  const auto count = regions.candidates().size();
  std::vector<std::optional<ir::RecoveredPath>> paths(count);
  std::vector<std::optional<analysis::PathControlFacts>> control(count);
  std::vector<std::uint8_t> dependent(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto& region = regions.candidates()[index];
    auto normalized = analysis::NormalizeRegion(regions, index, budget);
    if (!normalized.path) continue;
    auto forwarded = recovery::ForwardMemoryValues(*normalized.path, budget, {},
                                                   related.relations->blocks[region.entry_block]);
    if (!forwarded.path) {
      why = "memory forwarding declined";
      return std::nullopt;
    }

    dependent[index] = std::any_of(forwarded.journal.begin(), forwarded.journal.end(),
                                   [&](const recovery::MemoryEdit& edit) {
                                     return forwarded.facts[edit.fact].entry_relation;
                                   });
    auto simplified = recovery::SimplifyMba(*forwarded.path, budget);
    if (!simplified.path) {
      why = "mba declined";
      return std::nullopt;
    }

    auto folded = recovery::FoldImageValues(*simplified.path, view, budget);
    if (!folded.path) {
      why = "image fold declined";
      return std::nullopt;
    }

    const auto folded_revision = folded.path->revision();
    std::vector<ir::ValueId> image_nodes;
    for (const auto& edit : folded.journal)
      if (edit.constant_bytes || edit.relocated_slot) image_nodes.push_back(edit.node);
    const ir::ImageFacts held{folded.constants, folded.RetainedPointers(view),
                              view.page_aligned_placement};
    auto recovered = recovery::RecoverControl(std::move(*folded.path), held, budget);
    if (!recovered.path) {
      why = "control recovery declined";
      return std::nullopt;
    }

    auto loads = recovery::OmitForwardedPairLoads(std::move(*recovered.path), forwarded.facts, kSp,
                                                  budget, held);
    if (!loads.path) {
      why = "pair loads declined";
      return std::nullopt;
    }

    auto stores = recovery::OmitOverwrittenStores(std::move(*loads.path), kSp, budget, held);
    if (!stores.path) {
      why = "store cleanup declined";
      return std::nullopt;
    }

    auto facts_here = analysis::AnalyzePathControl(*stores.path, {folded_revision, image_nodes},
                                                   budget, {}, held);
    if (!facts_here.facts) {
      why = "path control declined";
      return std::nullopt;
    }

    control[index] = *facts_here.facts;
    paths[index] = std::move(*stores.path);
  }

  auto unflattened = analysis::Unflatten(
      regions, control, budget, {{}, true},
      {dependent, related.relations->constant_image, related.relations->declared_opaque_control,
       related.relations->calling_convention, related.relations->return_leaves});
  if (!unflattened.unflattening) {
    why = "unflatten declined";
    return std::nullopt;
  }

  auto ssa_build = analysis::BuildSsa(regions, *unflattened.unflattening, paths, budget,
                                      &*related.relations, view);
  if (!ssa_build.graph) {
    why = "SSA build declined";
    return std::nullopt;
  }

  // What the leaf runs, decoded up to its first transfer, as the CLI decodes
  // each callee the graph settles on.
  if (leaf) {
    ir::SsaCalleeBody body{*leaf, {}};
    for (auto offset = *leaf - kImage; body.groups.size() < ir::kMaxCalleeBodyGroups;) {
      auto decoded = Decode(kImage + offset, image.subspan(offset), budget);
      if (!decoded.group) break;
      const bool ends = decoded.group->transfer().has_value();
      offset += decoded.length;
      body.groups.push_back(std::move(*decoded.group));
      if (ends) break;
    }

    std::vector<ir::SsaCalleeBody> bodies;
    bodies.push_back(std::move(body));
    ssa_build.graph->SetCalleeBodies(std::move(bodies));
  }

  passes::SsaDeclarations declared;
  declared.closed_entries = declared.return_leaves = declared.image_access = true;

  // The toy callee summary returns to its continuation, and a toy trap ends the run.
  declared.call_returns = declared.trap_stops = true;
  declared.frame = ir::PrivateFrameContract{kSp, -64, 0, true, true, true, 8, true, true};
  auto run = passes::RunSsaPipeline(*ssa_build.graph, declared, groups, view, budget, selection);
  if (!run.result) {
    why = "pipeline declined";
    return std::nullopt;
  }

  return Recovered{std::move(groups), std::move(*ssa_build.graph),
                   std::move(run.result->provisional), std::move(run.result->stages),
                   std::move(run.result->body)};
}

// ---- Executing an SSA graph against the reference.

struct Observation {
  bool completed = false;
  eval::SsaStop stop = eval::SsaStop::declined;
  std::uint64_t runtime_pc = 0;
  std::array<std::uint64_t, kRegisters> r{};
  std::vector<std::uint8_t> stack;
};

Observation RunSsa(const ir::SsaGraph& graph, std::span<const ir::Group> groups,
                   std::span<const std::uint8_t> image, const Machine& start, Budget& budget,
                   std::optional<std::uint64_t> callee = std::nullopt,
                   std::optional<std::uint64_t> leaf = std::nullopt) {
  Observation seen;
  const std::array<eval::RegionInput, 2> regions{
      {{kImage + kBias, image, true, false}, {kStack, start.stack}}};
  auto memory = eval::Memory::Create(regions, budget);
  if (!memory.memory) return seen;
  eval::State state;
  for (unsigned reg = 0; reg < kRegisters; ++reg) {
    auto value = BitVector::from_u64(64, start.r[reg], 64, budget);
    if (!value) return seen;
    state.cells.push_back({reg, std::move(*value)});
  }

  eval::SsaCallee external;
  if (callee) {
    external = [&](std::uint64_t target, eval::State& now, eval::Memory&,
                   Budget& work) -> eval::SsaCalleeOutcome {
      std::array<std::uint64_t, kRegisters> r{};
      if (target != *callee + kBias || now.cells.size() < kRegisters)
        return {eval::Outcome::unsupported, std::nullopt};
      for (unsigned reg = 0; reg < kRegisters; ++reg) r[reg] = now.cells[reg].value.word(0);
      ExternalCallee(r);
      for (unsigned reg = 0; reg < kRegisters; ++reg) {
        auto value = BitVector::from_u64(64, r[reg], 64, work);
        if (!value) return {eval::Outcome::resource_limit, std::nullopt};
        now.cells[reg].value = std::move(*value);
      }

      return {eval::Outcome::completed, r[kLink]};
    };
  } else if (leaf) {
    external = [&](std::uint64_t target, eval::State& now, eval::Memory&,
                   Budget& work) -> eval::SsaCalleeOutcome {
      Machine m;
      if (target != *leaf + kBias || now.cells.size() < kRegisters)
        return {eval::Outcome::unsupported, std::nullopt};
      for (unsigned reg = 0; reg < kRegisters; ++reg) m.r[reg] = now.cells[reg].value.word(0);
      if (!Reference(m, image, nullptr, std::nullopt, target, m.r[kLink]))
        return {eval::Outcome::unsupported, std::nullopt};
      for (unsigned reg = 0; reg < kRegisters; ++reg) {
        auto value = BitVector::from_u64(64, m.r[reg], 64, work);
        if (!value) return {eval::Outcome::resource_limit, std::nullopt};
        now.cells[reg].value = std::move(*value);
      }

      return {eval::Outcome::completed, m.r[kLink]};
    };
  }

  const auto run = eval::ExecuteSsa(graph, groups, graph.entries()[0], state, *memory.memory,
                                    budget, {kBias}, external);
  seen.stop = run.stop;
  seen.runtime_pc = run.runtime_pc;
  if (run.outcome != eval::Outcome::completed || run.stop != eval::SsaStop::returned ||
      run.runtime_pc != kLanding)
    return seen;
  for (const auto& cell : state.cells)
    if (cell.id < kRegisters) seen.r[cell.id] = cell.value.word(0);
  for (const auto& region : memory.memory->Regions())
    if (region.address == kStack) seen.stack = region.bytes;
  seen.completed = true;
  return seen;
}

struct Program {
  std::string name;
  std::vector<std::uint8_t> image;  // padded to a page
  std::size_t size;

  // Registers the program reads as a small value rather than a random one, so
  // tests and table indices take every direction.
  std::vector<unsigned> small;

  // Registered pass names; empty runs the default pipeline.
  std::vector<std::string> pipeline;

  // Conditionals this program decides by construction, so each takes one
  // direction; every other conditional and seqz must take both.
  bool decided = false;

  // The image location of the one external callee, if the program calls it.
  std::optional<std::uint64_t> callee;

  // The image location of a leaf the program calls, in the image but outside
  // the selected population, which both sides run as the code it is.
  std::optional<std::uint64_t> leaf;

  // The program ends in an indirect jump to r0, which no stage resolves.
  bool indirect_exit = false;
};

Program Make(std::string name, Asm assembler, std::vector<unsigned> small = {},
             std::vector<std::string> pipeline = {}) {
  if (!assembler.size) assembler.end();
  auto image = assembler.Finish();
  image.resize(4096);
  Program program;
  program.name = std::move(name);
  program.image = std::move(image);
  program.size = assembler.size;
  program.small = std::move(small);
  program.pipeline = std::move(pipeline);
  return program;
}

Program Decided(Program program) {
  program.decided = true;
  return program;
}

std::vector<std::size_t> Selection(const Program& program) {
  std::vector<std::size_t> selection;
  for (const auto& name : program.pipeline) selection.push_back(passes::FindSsaPass(name).value());
  return selection;
}

std::vector<Program> Programs() {
  std::vector<Program> programs;

  // Both arms set r8 to 11: phi_constant_fold.
  programs.push_back(Make("equal_arms",
                          Asm()
                              .beqz(0, "left")
                              .movi16(8, 11)
                              .jmp("join")
                              .trap()
                              .label("left")
                              .movi16(8, 11)
                              .jmp("join")
                              .label("join")
                              .addi(0, 8, 1)
                              .ret(),
                          {0}));
  // One arm computes 10+1: constant_propagation.
  programs.push_back(Make("computed_arm",
                          Asm()
                              .beqz(0, "left")
                              .movi16(8, 10)
                              .addi(8, 8, 1)
                              .jmp("join")
                              .trap()
                              .label("left")
                              .movi16(8, 11)
                              .jmp("join")
                              .label("join")
                              .addi(0, 8, 1)
                              .ret(),
                          {0}));
  // A branch on a register a predecessor zeroed: the SCCP stages.
  const auto decided = [] {
    return Asm()
        .movi16(2, 0)
        .jmp("test")
        .label("test")
        .beqz(2, "taken")
        .movi16(8, 22)
        .jmp("join")
        .trap()
        .label("taken")
        .movi16(8, 10)
        .addi(8, 8, 1)
        .jmp("join")
        .label("join")
        .addi(0, 8, 12)
        .mov(3, 2)
        .ret();
  };

  programs.push_back(Decided(Make("decided_branch", decided())));

  // A freshly built graph holds only blocks its entries reach, so the first
  // unreachable-block stage can only act after an edit removes an edge.
  programs.push_back(Decided(Make(
      "decided_branch_retired_first", decided(), {},
      {"sccp_constant_fold", "sccp_storage_reads", "branch_retirement", "unreachable_blocks"})));
  // A literal load from the read-only image: constant_image_loads.
  programs.push_back(
      Make("literal", Asm().ldl(0, "value").ret().end().label("value").quad(0x123456789abcdef0)));
  // A two-row table selected by a comparison: selected_image_loads.
  programs.push_back(Make(
      "selected",
      Asm().seqz(8, 0).lea(9, "table").ldx(1, 9, 8).ret().end().label("table").quad(11).quad(22),
      {0}));
  // x0 + x1 as XOR plus twice AND: linear_mba.
  programs.push_back(
      Make("mba", Asm().xor_(2, 0, 1).and_(3, 0, 1).add(3, 3, 3).nop().add(0, 2, 3).ret()));
  // The same sum with the doubling as a shift, then x0 - x1 as x0 + ~x1 + 1,
  // a product by a 64-bit literal and an OR of the two terms.
  programs.push_back(Make("shift_not_mul", Asm()
                                               .xor_(2, 0, 1)
                                               .and_(3, 0, 1)
                                               .shl(3, 3, 1)
                                               .add(4, 2, 3)
                                               .not_(5, 1)
                                               .movi16(6, 1)
                                               .add(5, 5, 6)
                                               .add(5, 0, 5)
                                               .movi(7, 0x9e3779b97f4a7c15)
                                               .mul(7, 5, 7)
                                               .add(0, 4, 7)
                                               .or_(8, 2, 3)
                                               .ret()));
  // An unsigned-above test in both directions.
  programs.push_back(
      Make("above",
           Asm().movi16(1, 7).bgtu(0, 2, "big").movi16(1, 9).label("big").add(0, 0, 1).ret(), {0}));
  // A loop that counts to eight and whose count nothing reads after it, so
  // one run leaves the same observable state as eight: loop_exits.
  programs.push_back(Make("empty_loop", Asm()
                                            .movi16(1, 0)
                                            .movi16(2, 1)
                                            .label("count")
                                            .add(1, 1, 2)
                                            .movi16(3, 8)
                                            .sub(3, 3, 1)
                                            .bnez(3, "count")
                                            .movi16(1, 0)
                                            .movi16(3, 0)
                                            .ret()));
  // A call through a register an earlier block loads with the external
  // callee's location: resolved_calls calls it directly.
  {
    Asm assembler;
    assembler.mov(12, kLink)
        .lea(9, "callee")
        .jmp("call")
        .label("call")
        .callr(9)
        .add(0, 0, 1)
        .mov(kLink, 12)
        .ret()
        .end()
        .label("callee")
        .trap();
    const auto callee = kImage + assembler.size;
    auto program = Make("resolved_call", std::move(assembler));
    program.callee = callee;
    programs.push_back(std::move(program));
  }

  // A call to a leaf that only sets r5, which is overwritten before anything
  // reads it: leaf_calls goes straight to the continuation.
  {
    Asm assembler;
    assembler.mov(12, kLink)
        .call("leaf")
        .movi16(5, 0)
        .mov(kLink, 12)
        .ret()
        .end()
        .label("leaf")
        .movi16(5, 7)
        .ret();
    const auto leaf = kImage + assembler.size;
    auto program = Make("leaf_call", std::move(assembler));
    program.leaf = leaf;
    programs.push_back(std::move(program));
  }

  // A counted backward loop: r1 accumulates r2 r0 times.
  programs.push_back(Make("loop",
                          Asm()
                              .movi16(1, 0)
                              .beqz(0, "done")
                              .label("loop")
                              .add(1, 1, 2)
                              .addi(0, 0, -1)
                              .bnez(0, "loop")
                              .label("done")
                              .mov(0, 1)
                              .ret(),
                          {0}));
  // x0 + x1 + x2 interleaved: canonical_linear_mba.
  programs.push_back(Make("affine", Asm()
                                        .xor_(3, 0, 1)
                                        .xor_(4, 1, 2)
                                        .and_(5, 0, 1)
                                        .and_(6, 1, 2)
                                        .add(7, 5, 5)
                                        .add(8, 6, 6)
                                        .add(9, 3, 4)
                                        .add(10, 9, 7)
                                        .add(11, 10, 8)
                                        .sub(0, 11, 1)
                                        .ret()));
  // A store nothing reads, and a store read back: the frame stages. The store
  // above SP is outside the declared frame and must survive every stage.
  programs.push_back(
      Make("frame",
           Asm().st(kSp, -8, 0).st(kSp, -16, 1).st(kSp, 8, 2).ld(0, kSp, -16).ld(3, kSp, 8).ret()));
  // A value stored above SP and read back into a register the next instruction
  // replaces: the read is dead, and the store proves its bytes mapped.
  programs.push_back(
      Make("dead_load", Asm().st(kSp, 16, 2).ld(4, kSp, 16).movi16(4, 7).add(0, 0, 4).ret()));
  // A guarded four-row table: bounded_table_loads.
  programs.push_back(Make("bounded",
                          Asm()
                              .bgeu(0, 4, "out")
                              .lea(1, "table")
                              .ldx(0, 1, 0)
                              .ret()
                              .label("out")
                              .ret()
                              .end()
                              .label("table")
                              .quad(11)
                              .quad(22)
                              .quad(33)
                              .quad(44),
                          {0}));
  // A value copied through a single-predecessor join: predecessor_copies.
  programs.push_back(
      Make("copy", Asm().add(8, 0, 3).jmp("next").trap().label("next").add(1, 8, 2).ret()));
  // A register written on the way and overwritten after: dead_storage_writes.
  programs.push_back(Make(
      "dead_write",
      Asm().movi16(8, 11).jmp("next").trap().trap().label("next").movi16(8, 22).mov(0, 8).ret()));
  // A call to a declared external callee, with the link kept in a
  // callee-saved register across it.
  {
    Asm assembler;
    assembler.mov(12, kLink)
        .movi16(1, 5)
        .call("callee")
        .add(0, 0, 1)
        .mov(kLink, 12)
        .ret()
        .end()
        .label("callee")
        .trap();
    const auto callee = kImage + assembler.size;
    auto program = Make("external_call", std::move(assembler));
    program.callee = callee;
    programs.push_back(std::move(program));
  }

  // An indirect jump to a register value: the successor is unknown to every stage.
  {
    auto program = Make("indirect", Asm().addi(0, 1, 16).jr(0));
    program.indirect_exit = true;
    programs.push_back(std::move(program));
  }

  return programs;
}

// A flattened state machine. The state lives in a private frame slot; a
// dispatcher loads it, sends anything above 3 to a trap, and otherwise jumps
// through a table of offsets in declared image bytes:
//   0: r1 = r0 + r2, next 1
//   1: call the external callee, then r8 = 11 and next 2 + seqz(r3)
//   2: trap if r15 is zero, else r1 ^= r0, next 3
//   3: return r1 + r8 in r0; r8 is 11 on every way in.
// No state leaves the table's range, so the guard's trap arm is reachable in
// the graph but never taken.
struct Flattened {
  Program program;
  std::map<std::string, std::uint64_t> labels;  // case name -> image location
  std::uint64_t guard = 0;                      // the dispatcher's bound check
};

Flattened MakeFlattened() {
  Asm a;
  a.mov(12, kLink)
      .movi16(1, 0)
      .movi16(4, 0)
      .st(kSp, -8, 4)
      .jmp("head")
      .label("head")
      .ld(7, kSp, -8)
      .label("guard")
      .bgtu(7, 3, "default")
      .label("dispatch")
      .lea(9, "table")
      .ldx(10, 9, 7)
      .lea(11, "base")
      .add(11, 11, 10)
      .jr(11)
      .label("base")
      .label("case0")
      .add(1, 0, 2)
      .movi16(4, 1)
      .st(kSp, -8, 4)
      .jmp("head")
      .label("case1")
      .call("callee")
      .label("resume")
      .movi16(8, 11)
      .seqz(5, 3)
      .movi16(6, 2)
      .add(5, 5, 6)
      .st(kSp, -8, 5)
      .jmp("head")
      .label("case2")
      .beqz(15, "trap_arm")
      .label("case2_body")
      .xor_(1, 1, 0)
      .movi16(4, 3)
      .st(kSp, -8, 4)
      .jmp("head")
      .label("trap_arm")
      .trap()
      .label("case3")
      .add(0, 1, 8)
      .mov(kLink, 12)
      .ret()
      .label("default")
      .trap()
      .end()
      .label("table")
      .offset("case0", "base")
      .offset("case1", "base")
      .offset("case2", "base")
      .offset("case3", "base")
      .label("callee")
      .trap();
  Flattened flattened;
  for (const auto* name : {"case0", "case1", "resume", "case2", "case2_body", "trap_arm", "case3"})
    flattened.labels[name] = a.Address(name);
  flattened.guard = a.Address("guard");
  const auto callee = a.Address("callee");
  flattened.program = Make("flattened", std::move(a), {3, 15});
  flattened.program.callee = callee;
  return flattened;
}

Machine RandomStart(const Program& program, std::mt19937_64& random) {
  Machine m;
  for (auto& value : m.r) value = random();
  for (const auto reg : program.small) m.r[reg] = random() % 6;
  m.r[kSp] = kStack + 2048;
  m.r[kLink] = kLanding;
  return m;
}

TEST(ToyTarget, DecodesVariableWidthBigEndianInstructions) {
  Budget budget({100000, 1000000});
  const std::array<std::uint8_t, 10> movi{0x20, 0x30, 1, 2, 3, 4, 5, 6, 7, 8};
  auto decoded = Decode(0x40, movi, budget);
  ASSERT_TRUE(decoded.group);
  EXPECT_EQ(decoded.length, 10U);
  EXPECT_EQ(decoded.group->bytes().size(), 10U);
  ASSERT_EQ(decoded.group->writes().size(), 1U);
  EXPECT_EQ(decoded.group->writes()[0].storage, 3U);
  EXPECT_EQ(decoded.group->nodes()[decoded.group->writes()[0].value].immediate,
            0x0102030405060708U);

  const std::array<std::uint8_t, 4> store{0x31, 0x2d, 0xff, 0xf8};  // st [r13-8], r2
  decoded = Decode(0x40, store, budget);
  ASSERT_TRUE(decoded.group);
  EXPECT_EQ(decoded.length, 4U);
  const auto stored = std::find_if(decoded.group->nodes().begin(), decoded.group->nodes().end(),
                                   [](const ir::Node& node) { return node.op == ir::Op::store; });
  ASSERT_NE(stored, decoded.group->nodes().end());
  EXPECT_EQ(stored->access.byte_order, ir::ByteOrder::big);

  // A one-byte return reads the link register, not r30.
  const std::array<std::uint8_t, 1> ret{0x01};
  decoded = Decode(0x41, ret, budget);
  ASSERT_TRUE(decoded.group && decoded.group->transfer());
  EXPECT_EQ(decoded.length, 1U);
  EXPECT_EQ(decoded.group->nodes()[decoded.group->transfer()->target].storage, kLink);
  EXPECT_TRUE(ir::ValidTransfer(*decoded.group->transfer(), decoded.group->nodes()));

  // A conditional's fallthrough is its own length past it, here five bytes.
  const std::array<std::uint8_t, 5> bgeu{0x52, 0x01, 4, 0x00, 0x10};
  decoded = Decode(0x100, bgeu, budget);
  ASSERT_TRUE(decoded.group && decoded.group->transfer());
  EXPECT_TRUE(ir::ValidTransfer(*decoded.group->transfer(), decoded.group->nodes()));

  // A truncated instruction reports no length, never more than was supplied.
  decoded = Decode(0x40, std::span(movi).first(9), budget);
  EXPECT_EQ(decoded.reason, DecodeDecline::truncated);
  EXPECT_EQ(decoded.length, 0U);
  decoded = Decode(0x40, std::span(movi).first(0), budget);
  EXPECT_EQ(decoded.reason, DecodeDecline::truncated);
  EXPECT_EQ(decoded.length, 0U);

  // One that would wrap the address space is misplaced, not short.
  decoded = Decode(UINT64_MAX - 8, movi, budget);
  EXPECT_EQ(decoded.reason, DecodeDecline::invalid_location);
  EXPECT_EQ(decoded.length, 0U);
  EXPECT_TRUE(Decode(UINT64_MAX - 9, movi, budget).group);
  const std::array<std::uint8_t, 2> unknown{0xff, 0};
  EXPECT_EQ(Decode(0x40, unknown, budget).reason, DecodeDecline::invalid_encoding);
  const std::array<std::uint8_t, 3> dirty{0x10, 0x12, 0x35};  // add with a nonzero spare nibble
  EXPECT_EQ(Decode(0x40, dirty, budget).reason, DecodeDecline::invalid_encoding);
  const std::array<std::uint8_t, 1> trap{0x00};
  decoded = Decode(0x40, trap, budget);
  EXPECT_EQ(decoded.reason, DecodeDecline::unsupported);
  EXPECT_EQ(decoded.length, 1U);
  EXPECT_TRUE(OpaqueTrap(trap));
  Budget empty({0, 0});
  EXPECT_EQ(Decode(0x40, movi, empty).reason, DecodeDecline::work_limit);
}

// A loop is only run once in its stead when that changes nothing observable:
// not when something it computes from what the caller passed is read after
// it, and not when it stores to memory. Its count alone is no obstacle: the
// run knows it, so a read of it is folded to the number first.
TEST(ToyTarget, ALoopStaysWhereOneRunWouldBeObservable) {
  const auto counted = [](bool read_after, bool store) {
    Asm assembler;
    assembler.movi16(1, 0).movi16(2, 1).movi16(4, 0).label("count").add(1, 1, 2).add(4, 4, 0);
    if (store) assembler.st(kSp, 8, 1);
    assembler.movi16(3, 8).sub(3, 3, 1).bnez(3, "count");
    assembler.mov(0, read_after ? 4 : 1);
    assembler.movi16(1, 0).movi16(3, 0).movi16(4, 0).ret();
    return assembler;
  };

  const auto registry = passes::SsaPassRegistry();
  for (const auto& [read_after, store, retired] :
       {std::tuple{false, false, true}, {true, false, false}, {false, true, false}}) {
    SCOPED_TRACE(std::string(read_after ? "read after" : "") + (store ? "store" : ""));
    const auto program = Make("case", counted(read_after, store));
    Budget budget({50000000, 256ULL * 1024 * 1024});
    std::string why;
    const auto recovered = Recover(program.image, program.size, Selection(program), budget, why);
    ASSERT_TRUE(recovered) << why;
    bool proposed = false;
    for (const auto& stage : recovered->stages)
      proposed |= registry[stage.pass].name == "loop_exits" &&
                  stage.outcome == passes::SsaStageOutcome::proposed;
    EXPECT_EQ(proposed, retired);
  }
}

// A call is only skipped when that changes nothing observable: not when what
// the callee writes is read after it, not when it writes a register the
// caller never names but the caller's own caller sees, and not when the
// callee touches memory.
TEST(ToyTarget, ACallStaysWhereSkippingItWouldBeObservable) {
  const auto registry = passes::SsaPassRegistry();
  for (const auto& [read_after, store, unnamed, retired] : {std::tuple{false, false, false, true},
                                                            {true, false, false, false},
                                                            {false, true, false, false},
                                                            {false, false, true, false}}) {
    SCOPED_TRACE(std::string(read_after ? "read after" : "") + (store ? "store" : "") +
                 (unnamed ? "unnamed" : ""));
    Asm assembler;
    assembler.mov(12, kLink).call("leaf");
    if (read_after) assembler.mov(0, 5);
    assembler.movi16(5, 0).mov(kLink, 12).ret().end().label("leaf").movi16(unnamed ? 7 : 5, 7);
    if (store) assembler.st(kSp, 8, 5);
    assembler.ret();
    const auto leaf = kImage + assembler.size;
    const auto program = Make("case", std::move(assembler));
    Budget budget({50000000, 256ULL * 1024 * 1024});
    std::string why;
    const auto recovered =
        Recover(program.image, program.size, Selection(program), budget, why, leaf);
    ASSERT_TRUE(recovered) << why;
    bool proposed = false;
    for (const auto& stage : recovered->stages)
      proposed |= registry[stage.pass].name == "leaf_calls" &&
                  stage.outcome == passes::SsaStageOutcome::proposed;
    EXPECT_EQ(proposed, retired);
  }
}

// A call through a pointer a loop advances reaches a different place each
// time round, so it resolves to none of them (in particular, not the first).
TEST(ToyTarget, ACallThroughAnAdvancingPointerIsNotResolved) {
  // The pointer lives in a private frame slot, which the call leaves alone.
  Asm assembler;
  assembler.mov(12, kLink)
      .lea(9, "callee")
      .st(kSp, -8, 9)
      .movi16(3, 2)
      .label("again")
      .ld(9, kSp, -8)
      .callr(9)
      .ld(9, kSp, -8)
      .addi(9, 9, 1)
      .st(kSp, -8, 9)
      .addi(3, 3, -1)
      .bnez(3, "again")
      .mov(kLink, 12)
      .ret()
      .end()
      .label("callee")
      .ret()
      .ret();
  const auto program = Make("advancing", std::move(assembler));
  Budget budget({50000000, 256ULL * 1024 * 1024});
  std::string why;
  const auto recovered = Recover(program.image, program.size, Selection(program), budget, why);
  ASSERT_TRUE(recovered) << why;
  const auto registry = passes::SsaPassRegistry();
  for (const auto& stage : recovered->stages)
    EXPECT_FALSE(registry[stage.pass].name == "resolved_calls" &&
                 stage.outcome == passes::SsaStageOutcome::proposed);
}

TEST(ToyTarget, EveryRegisteredPassProposesAndAgreesWithTheReference) {
  std::map<std::string, std::string> proposers;  // pass -> first program
  std::set<std::uint8_t> executed;
  std::mt19937_64 random(0x746f79);
  for (const auto& program : Programs()) {
    SCOPED_TRACE(program.name);
    Budget budget({50000000, 256ULL * 1024 * 1024});
    std::string why;
    auto recovered =
        Recover(program.image, program.size, Selection(program), budget, why, program.leaf);
    ASSERT_TRUE(recovered) << why;
    const auto registry = passes::SsaPassRegistry();
    for (const auto& stage : recovered->stages)
      if (stage.outcome == passes::SsaStageOutcome::proposed)
        proposers.emplace(std::string(registry[stage.pass].name), program.name);
    Coverage coverage;
    for (unsigned trial = 0; trial < 64; ++trial) {
      auto start = RandomStart(program, random);
      auto expected = start;
      const bool returned = Reference(expected, program.image, &coverage, program.callee);
      for (const auto* graph :
           {&recovered->base, recovered->provisional ? &*recovered->provisional : nullptr}) {
        if (!graph) continue;
        const char* which = graph == &recovered->base ? "base" : "provisional";
        const auto seen = RunSsa(*graph, recovered->groups, program.image, start, budget,
                                 program.callee, program.leaf);
        if (program.indirect_exit) {
          // Neither side can follow the jump: the reference leaves the image
          // and SSA stops at the unresolved successor with the same target.
          ASSERT_FALSE(returned);
          EXPECT_FALSE(seen.completed) << which;
          EXPECT_EQ(seen.stop, eval::SsaStop::unresolved) << which;
          EXPECT_EQ(seen.runtime_pc, expected.pc) << which;
          continue;
        }

        ASSERT_TRUE(returned) << "reference did not return";
        ASSERT_TRUE(seen.completed) << which;
        EXPECT_EQ(seen.r, expected.r) << which;

        // The declared private frame is unobservable, so a provisional graph
        // may retire its stores; every byte outside it must still agree.
        auto seen_stack = seen.stack, expected_stack = expected.stack;
        if (graph != &recovered->base && seen_stack.size() == expected_stack.size()) {
          std::fill(seen_stack.begin() + 2048 - 64, seen_stack.begin() + 2048, 0);
          std::fill(expected_stack.begin() + 2048 - 64, expected_stack.begin() + 2048, 0);
        }

        EXPECT_EQ(seen_stack, expected_stack) << which;
      }
    }

    // A fold of a direction no state took would go untested.
    for (const auto& [location, taken] : coverage.directions)
      EXPECT_EQ(taken.size(), program.decided ? 1U : 2U)
          << "conditional at " << location << " took " << taken.size() << " direction(s)";
    executed.insert(coverage.opcodes.begin(), coverage.opcodes.end());
  }

  for (const auto& pass : passes::SsaPassRegistry())
    EXPECT_TRUE(proposers.count(std::string(pass.name)))
        << pass.name << " never proposed on a toy program";
  // Every opcode the decoder knows is executed against the reference, except
  // the trap, which is only ever an opaque stop.
  Budget budget({100000, 1000000});
  for (unsigned opcode = 0; opcode < 256; ++opcode) {
    std::array<std::uint8_t, 10> bytes{};
    bytes[0] = std::uint8_t(opcode);
    const auto decoded = Decode(0x40, bytes, budget);
    if (!decoded.length) continue;
    if (opcode == 0x00) {
      EXPECT_EQ(decoded.reason, DecodeDecline::unsupported);
      EXPECT_TRUE(OpaqueTrap(bytes));
      continue;
    }

    EXPECT_TRUE(executed.count(std::uint8_t(opcode))) << "opcode " << opcode << " never executed";
  }
}

// The comparison above can fail: a folded constant changed by one is caught.
TEST(ToyTarget, ReferenceRefutesAMutatedProvisionalGraph) {
  Program program;
  for (auto& candidate : Programs())
    if (candidate.name == "equal_arms") program = std::move(candidate);
  Budget budget({50000000, 256ULL * 1024 * 1024});
  std::string why;
  auto recovered = Recover(program.image, program.size, {}, budget, why);
  ASSERT_TRUE(recovered && recovered->provisional) << why;
  bool mutated = false;
  auto& graph = *recovered->provisional;
  for (std::size_t slot = 0; slot < graph.slots() && !mutated; ++slot) {
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto* block = graph.Get(*handle);
    for (ir::ValueId id = 0; id < block->nodes.size() && !mutated; ++id) {
      // The folded join value, which the block publishes as r0.
      if (block->nodes[id].op != ir::Op::constant || block->nodes[id].immediate != 12 ||
          std::none_of(block->exits.begin(), block->exits.end(), [&](const auto& exit) {
            return exit.value.kind == ir::SsaValueKind::node && exit.value.index == id;
          }))
        continue;
      mutated = graph.Update(*handle, [&](auto& changed) { changed.nodes[id].immediate = 13; });
    }
  }

  ASSERT_TRUE(mutated);
  std::mt19937_64 random(1);
  unsigned refuted = 0;
  for (unsigned trial = 0; trial < 16; ++trial) {
    auto start = RandomStart(program, random);
    auto expected = start;
    ASSERT_TRUE(Reference(expected, program.image));
    const auto seen = RunSsa(graph, recovered->groups, program.image, start, budget);
    ASSERT_TRUE(seen.completed);
    refuted += seen.r != expected.r;
  }

  EXPECT_EQ(refuted, 16U);
}

// A selected fold whose alternate row is wrong by one is refuted on exactly
// the states that take that row, and only when both rows run.
TEST(ToyTarget, ReferenceRefutesAMutatedSelectedRow) {
  Program program;
  for (auto& candidate : Programs())
    if (candidate.name == "selected") program = std::move(candidate);
  Budget budget({50000000, 256ULL * 1024 * 1024});
  std::string why;
  auto recovered = Recover(program.image, program.size, {}, budget, why);
  ASSERT_TRUE(recovered && recovered->provisional) << why;
  auto& graph = *recovered->provisional;
  bool mutated = false;
  for (std::size_t slot = 0; slot < graph.slots() && !mutated; ++slot) {
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto* block = graph.Get(*handle);
    for (std::size_t index = 0; index < block->constant_loads.size() && !mutated; ++index) {
      if (!block->constant_loads[index].condition) continue;
      mutated = graph.Update(*handle, [&](auto& changed) {
        auto& fold = changed.constant_loads[index];
        fold.alternative_value += 1;

        // Keep the declaration consistent, so only the value is wrong.
        for (unsigned i = 0; i < 8; ++i)
          fold.alternative_declared_bytes[i] = std::uint8_t(fold.alternative_value >> (56 - 8 * i));
      });
    }
  }

  ASSERT_TRUE(mutated);
  std::mt19937_64 random(2);
  unsigned alternate = 0, refuted = 0;
  for (unsigned trial = 0; trial < 64; ++trial) {
    auto start = RandomStart(program, random);
    auto expected = start;
    ASSERT_TRUE(Reference(expected, program.image));
    const auto seen = RunSsa(graph, recovered->groups, program.image, start, budget);
    ASSERT_TRUE(seen.completed);
    const bool took_alternate = start.r[0] != 0;  // seqz false selects the second operand's row
    alternate += took_alternate;
    const bool differs = seen.r != expected.r;
    refuted += differs;
    EXPECT_EQ(differs, took_alternate) << "r0=" << start.r[0];
  }

  EXPECT_GT(alternate, 0U);
  EXPECT_LT(alternate, 64U);
  EXPECT_EQ(refuted, alternate);
}

}  // namespace
}  // namespace nyx::toy

namespace nyx::toy {
namespace {

std::size_t Transitions(const ir::SsaGraph& graph) {
  std::size_t count = 0;
  for (std::size_t slot = 0; slot < graph.slots(); ++slot)
    if (const auto handle = graph.Handle(slot)) count += graph.Get(*handle)->transition.has_value();
  return count;
}

// The flattened program through the whole driver: Unflatten stitches the
// dispatch away, the reachability passes accept the transition, call and trap
// blocks, and both graphs agree with the reference on every case and trap.
TEST(ToyTarget, FlattenedDispatchRunsThroughTheReachabilityPasses) {
  const auto flat = MakeFlattened();
  const auto& program = flat.program;
  Budget budget({50000000, 256ULL * 1024 * 1024});
  std::string why;
  auto recovered = Recover(program.image, program.size, {}, budget, why);
  ASSERT_TRUE(recovered) << why;
  const auto& base = recovered->base;
  EXPECT_GE(Transitions(base), 4U);

  // One transition is a conditional dispatch: two edges selected by one condition.
  bool conditional = false, call = false, trap = false;
  for (std::size_t slot = 0; slot < base.slots(); ++slot) {
    const auto handle = base.Handle(slot);
    if (!handle) continue;
    const auto& block = *base.Get(*handle);
    conditional |= block.transition && block.edges.size() == 2 && block.edges[0].when &&
                   block.edges[0].assumptions.constant_image;
    for (const auto& edge : block.edges) {
      call |= edge.kind == ir::SsaEdgeKind::callee;
      trap |= block.opaque && edge.kind == ir::SsaEdgeKind::trap &&
              edge.assumptions.declared_opaque_control;
    }

    // The dispatcher's computed jump is gone from the recovered graph.
    for (const auto& edge : block.edges) EXPECT_FALSE(edge.assumptions.unresolved_target);
  }

  EXPECT_TRUE(conditional);
  EXPECT_TRUE(call);
  EXPECT_TRUE(trap);

  const auto registry = passes::SsaPassRegistry();
  bool reachability_proposed = false;
  for (const auto& stage : recovered->stages) {
    const auto& pass = registry[stage.pass];
    if (!(pass.requirements & passes::kNeedsClosedEntries)) continue;
    SCOPED_TRACE(std::string(pass.name));
    EXPECT_NE(stage.reason, "incomplete_successors");
    EXPECT_NE(stage.reason, "invalid_graph");
    EXPECT_NE(stage.outcome, passes::SsaStageOutcome::not_run);
    reachability_proposed |= stage.outcome == passes::SsaStageOutcome::proposed;
  }

  EXPECT_TRUE(reachability_proposed);
  ASSERT_TRUE(recovered->provisional);

  Coverage coverage;
  std::mt19937_64 random(0x666c6174);
  unsigned trapped = 0, returned = 0;
  for (unsigned trial = 0; trial < 128; ++trial) {
    const auto start = RandomStart(program, random);
    auto expected = start;
    const bool completed = Reference(expected, program.image, &coverage, program.callee);
    ASSERT_TRUE(completed || expected.trapped) << "reference faulted";
    trapped += expected.trapped;
    returned += completed;
    const std::array<const ir::SsaGraph*, 2> graphs{&base, &*recovered->provisional};
    for (const auto* graph : graphs) {
      const char* which = graph == &base ? "base" : "provisional";
      Budget run_budget({50000000, 256ULL * 1024 * 1024});
      const auto seen =
          RunSsa(*graph, recovered->groups, program.image, start, run_budget, program.callee);
      if (expected.trapped) {
        // A declared trap stops as a trap on its own location.
        EXPECT_EQ(seen.stop, eval::SsaStop::trap) << which;
        EXPECT_EQ(seen.runtime_pc, expected.pc) << which;
        continue;
      }

      ASSERT_TRUE(seen.completed) << which;
      EXPECT_EQ(seen.r, expected.r) << which;
      auto seen_stack = seen.stack, expected_stack = expected.stack;
      if (graph != &base) {
        std::fill(seen_stack.begin() + 2048 - 64, seen_stack.begin() + 2048, 0);
        std::fill(expected_stack.begin() + 2048 - 64, expected_stack.begin() + 2048, 0);
      }

      EXPECT_EQ(seen_stack, expected_stack) << which;
    }
  }

  EXPECT_GT(trapped, 0U);
  EXPECT_GT(returned, 0U);
  for (const auto& [name, location] : flat.labels)
    EXPECT_TRUE(coverage.locations.count(location)) << name << " never ran";
  // Every state stays in the table's range, so the guard takes one direction;
  // every other conditional, and the seqz that picks the next state, takes both.
  for (const auto& [location, taken] : coverage.directions) {
    if (location == flat.guard)
      EXPECT_EQ(taken, std::set<bool>{false});
    else
      EXPECT_EQ(taken.size(), 2U) << "conditional at " << location;
  }
}

// A plain switch: a bound check, then a jump through a table of offsets. The
// CFG builder enumerates the jump's targets, but the dispatch stays
// incomplete, and every reachability pass with it, until bounded_table_loads
// folds the table the targets come from.
TEST(ToyTarget, ASwitchDispatchIsCompletedByItsTableFold) {
  Asm a;
  a.addi(1, 0, -1)
      .bgeu(1, 4, "out")
      .lea(9, "table")
      .ldx(10, 9, 1)
      .lea(11, "base")
      .add(11, 11, 10)
      .jr(11)
      .label("base")
      .label("case0")
      .movi16(0, 11)
      .ret()
      .label("case1")
      .movi16(0, 22)
      .ret()
      .label("case2")
      .movi16(0, 33)
      .ret()
      .label("out")
      .movi16(0, 0)
      .ret()
      .end()
      .label("table")
      .offset("case0", "base")
      .offset("case1", "base")
      .offset("case2", "base")
      .offset("case1", "base");
  const auto program = Make("switch", std::move(a), {0});
  Budget budget({50000000, 256ULL * 1024 * 1024});
  std::string why;
  auto recovered = Recover(program.image, program.size, {}, budget, why);
  ASSERT_TRUE(recovered) << why;
  const auto dispatch = [](const ir::SsaGraph& graph) -> const ir::SsaBlock* {
    for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
      const auto handle = graph.Handle(slot);
      if (handle && graph.Get(*handle)->edges.size() == 3) return graph.Get(*handle);
    }

    return nullptr;
  };

  const auto* before = dispatch(recovered->base);
  ASSERT_TRUE(before);
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(*before, budget), ir::SsaDecline::invalid_graph);

  const auto registry = passes::SsaPassRegistry();
  bool folded = false, completed = false;
  for (const auto& stage : recovered->stages) {
    const auto& name = registry[stage.pass].name;
    SCOPED_TRACE(std::string(name));
    if (name == "bounded_table_loads" && stage.outcome == passes::SsaStageOutcome::proposed)
      folded = true;
    // Before the fold the reachability passes refuse the dispatch; after it
    // none does, and they go on to edit the graph.
    if (folded && (registry[stage.pass].requirements & passes::kNeedsClosedEntries)) {
      EXPECT_NE(stage.outcome, passes::SsaStageOutcome::declined);
      completed |= stage.outcome == passes::SsaStageOutcome::proposed;
    } else if (registry[stage.pass].requirements & passes::kNeedsClosedEntries) {
      EXPECT_EQ(stage.reason, "incomplete_successors");
    }
  }

  EXPECT_TRUE(folded);
  EXPECT_TRUE(completed);
  ASSERT_TRUE(recovered->provisional);
  const auto* after = dispatch(*recovered->provisional);
  ASSERT_TRUE(after);
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(*after, budget), ir::SsaDecline::none);
}

// Blocks of a recovered graph that jump through a table to three places.
std::vector<const ir::SsaBlock*> Dispatches(const ir::SsaGraph& graph) {
  std::vector<const ir::SsaBlock*> found;
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    const auto handle = graph.Handle(slot);
    if (handle && graph.Get(*handle)->edges.size() == 3) found.push_back(graph.Get(*handle));
  }

  return found;
}

// Each dispatch's fold stands only once the other is folded too, so the
// stage folds both tables in one candidate, and then every reachability
// pass after it runs.
void ExpectEveryDispatchFolded(Asm assembler, const char* name) {
  const auto program = Make(name, std::move(assembler), {0, 2});
  Budget budget({50000000, 256ULL * 1024 * 1024});
  std::string why;
  auto recovered = Recover(program.image, program.size, {}, budget, why);
  ASSERT_TRUE(recovered) << why;
  const auto before = Dispatches(recovered->base);
  ASSERT_EQ(before.size(), 2U);
  for (const auto* block : before)
    EXPECT_EQ(ir::ValidateSsaDirectSuccessors(*block, budget), ir::SsaDecline::invalid_graph);
  const auto registry = passes::SsaPassRegistry();
  bool folded = false, completed = false;
  for (const auto& stage : recovered->stages) {
    const auto& pass = registry[stage.pass];
    SCOPED_TRACE(std::string(pass.name));
    if (pass.name == "bounded_table_loads") {
      EXPECT_EQ(stage.outcome, passes::SsaStageOutcome::proposed);
      EXPECT_EQ(stage.executable_edits, 2U);
      folded = true;
    } else if (pass.requirements & passes::kNeedsClosedEntries) {
      if (folded) {
        EXPECT_NE(stage.outcome, passes::SsaStageOutcome::declined);
        completed |= stage.outcome == passes::SsaStageOutcome::proposed;
      } else {
        EXPECT_EQ(stage.reason, "incomplete_successors");
      }
    }
  }

  EXPECT_TRUE(folded);
  EXPECT_TRUE(completed);
  ASSERT_TRUE(recovered->provisional);
  const auto after = Dispatches(*recovered->provisional);
  ASSERT_EQ(after.size(), 2U);
  for (const auto* block : after)
    EXPECT_EQ(ir::ValidateSsaDirectSuccessors(*block, budget), ir::SsaDecline::none);
}

TEST(ToyTarget, TwoSwitchesAreFoldedTogether) {
  Asm a;
  a.beqz(2, "second")
      .addi(1, 0, -1)
      .bgeu(1, 4, "out")
      .lea(9, "table")
      .ldx(10, 9, 1)
      .lea(11, "base")
      .add(11, 11, 10)
      .jr(11)
      .label("second")
      .addi(1, 0, -1)
      .bgeu(1, 4, "out")
      .lea(9, "table2")
      .ldx(10, 9, 1)
      .lea(11, "base")
      .add(11, 11, 10)
      .jr(11)
      .label("base")
      .label("case0")
      .movi16(0, 11)
      .ret()
      .label("case1")
      .movi16(0, 22)
      .ret()
      .label("case2")
      .movi16(0, 33)
      .ret()
      .label("out")
      .movi16(0, 0)
      .ret()
      .end()
      .label("table")
      .offset("case0", "base")
      .offset("case1", "base")
      .offset("case2", "base")
      .offset("case1", "base")
      .label("table2")
      .offset("case2", "base")
      .offset("case0", "base")
      .offset("case1", "base")
      .offset("case0", "base");
  ExpectEveryDispatchFolded(std::move(a), "two_switches");
}

// The inner switch is one of the outer switch's cases.
TEST(ToyTarget, ANestedSwitchIsFoldedWithItsOuterSwitch) {
  Asm a;
  a.addi(1, 0, -1)
      .bgeu(1, 4, "out")
      .lea(9, "table")
      .ldx(10, 9, 1)
      .lea(11, "base")
      .add(11, 11, 10)
      .jr(11)
      .label("base")
      .label("case0")
      .movi16(0, 11)
      .ret()
      .label("case1")
      .movi16(0, 22)
      .ret()
      .label("inner")
      .addi(1, 2, -1)
      .bgeu(1, 4, "out")
      .lea(9, "table2")
      .ldx(10, 9, 1)
      .lea(11, "base")
      .add(11, 11, 10)
      .jr(11)
      .label("out")
      .movi16(0, 0)
      .ret()
      .end()
      .label("table")
      .offset("case0", "base")
      .offset("case1", "base")
      .offset("inner", "base")
      .offset("case1", "base")
      .label("table2")
      .offset("case0", "base")
      .offset("case1", "base")
      .offset("out", "base")
      .offset("case0", "base");
  ExpectEveryDispatchFolded(std::move(a), "nested_switch");
}

// The switch's guard compares a number it read from a table of its own, so
// one fold sits in the guard of the other. A fold no longer freezes its
// guard, so both fold in one candidate.
TEST(ToyTarget, ATableLoadInASwitchGuardFoldsWithTheSwitch) {
  Asm a;
  a.addi(1, 0, -1)
      .bgeu(1, 4, "out")
      .lea(9, "rows")
      .ldx(2, 9, 1)
      .bgeu(2, 4, "out")
      .lea(9, "table")
      .ldx(10, 9, 2)
      .lea(11, "base")
      .add(11, 11, 10)
      .jr(11)
      .label("base")
      .label("case0")
      .movi16(0, 11)
      .ret()
      .label("case1")
      .movi16(0, 22)
      .ret()
      .label("case2")
      .movi16(0, 33)
      .ret()
      .label("out")
      .movi16(0, 0)
      .ret()
      .end()
      .label("rows")
      .quad(2)
      .quad(0)
      .quad(1)
      .quad(3)
      .label("table")
      .offset("case0", "base")
      .offset("case1", "base")
      .offset("case2", "base")
      .offset("case1", "base");
  const auto program = Make("guard_table", std::move(a), {0});
  Budget budget({50000000, 256ULL * 1024 * 1024});
  std::string why;
  auto recovered = Recover(program.image, program.size, {}, budget, why);
  ASSERT_TRUE(recovered) << why;
  const auto registry = passes::SsaPassRegistry();
  bool folded = false;
  for (const auto& stage : recovered->stages)
    if (registry[stage.pass].name == "bounded_table_loads") {
      EXPECT_EQ(stage.outcome, passes::SsaStageOutcome::proposed);
      EXPECT_EQ(stage.executable_edits, 2U);
      folded = true;
    }
  EXPECT_TRUE(folded);
  ASSERT_TRUE(recovered->provisional);
  const auto after = Dispatches(*recovered->provisional);
  ASSERT_EQ(after.size(), 1U);
  EXPECT_EQ(ir::ValidateSsaDirectSuccessors(*after[0], budget), ir::SsaDecline::none);
}

// Mutations of what the reachability passes and Unflatten produced are caught.
TEST(ToyTarget, FlattenedMutationsAreRefuted) {
  const auto flat = MakeFlattened();
  const auto& program = flat.program;
  Budget budget({50000000, 256ULL * 1024 * 1024});
  std::string why;
  auto recovered = Recover(program.image, program.size, {}, budget, why);
  ASSERT_TRUE(recovered && recovered->provisional) << why;
  const auto& base = recovered->base;

  // A storage read a stage replaced with a constant, off by one. Folded control
  // values would divert the run instead of changing its result.
  auto folded = *recovered->provisional->Clone(budget);
  bool mutated = false;
  for (std::size_t slot = 0; slot < folded.slots() && !mutated; ++slot) {
    const auto handle = folded.Handle(slot);
    const auto original = base.Handle(slot);
    if (!handle || !original || !base.Get(*original)->transition) continue;
    const auto& before = base.Get(*original)->nodes;
    const auto& after = folded.Get(*handle)->nodes;
    for (ir::ValueId id = 0; id < before.size() && id < after.size() && !mutated; ++id) {
      if (before[id].op != ir::Op::read || after[id].op != ir::Op::constant) continue;
      mutated = folded.Update(*handle, [&](auto& block) { block.nodes[id].immediate += 1; });
    }

    if (mutated) break;
  }

  // No transition node was folded; fall back to any folded node.
  for (std::size_t slot = 0; slot < folded.slots() && !mutated; ++slot) {
    const auto handle = folded.Handle(slot);
    const auto original = base.Handle(slot);
    if (!handle || !original) continue;
    const auto& before = base.Get(*original)->nodes;
    const auto& after = folded.Get(*handle)->nodes;
    for (ir::ValueId id = 0; id < before.size() && id < after.size() && !mutated; ++id) {
      if (before[id].op != ir::Op::read || after[id].op != ir::Op::constant) continue;
      mutated = folded.Update(*handle, [&](auto& block) { block.nodes[id].immediate += 1; });
    }
  }

  ASSERT_TRUE(mutated);
  std::mt19937_64 random(3);
  unsigned compared = 0, refuted = 0;
  for (unsigned trial = 0; trial < 64; ++trial) {
    const auto start = RandomStart(program, random);
    auto expected = start;
    if (!Reference(expected, program.image, nullptr, program.callee)) continue;
    Budget run_budget({50000000, 256ULL * 1024 * 1024});
    const auto seen =
        RunSsa(folded, recovered->groups, program.image, start, run_budget, program.callee);
    ASSERT_TRUE(seen.completed);
    ++compared;
    refuted += seen.r != expected.r;
  }

  EXPECT_GT(compared, 0U);
  EXPECT_EQ(refuted, compared);

  // The conditional transition with its two destinations exchanged. Execution
  // rechecks the computed target against the edge it follows, so no state
  // that reaches it may complete.
  auto swapped = *base.Clone(budget);
  bool exchanged = false;
  for (std::size_t slot = 0; slot < swapped.slots() && !exchanged; ++slot) {
    const auto handle = swapped.Handle(slot);
    if (!handle) continue;
    const auto& block = *swapped.Get(*handle);
    if (!block.transition || block.edges.size() != 2 || !block.edges[0].when) continue;
    exchanged = swapped.Update(*handle, [](auto& changed) {
      std::swap(changed.edges[0].target_block, changed.edges[1].target_block);
      std::swap(changed.edges[0].address, changed.edges[1].address);
    });
  }

  ASSERT_TRUE(exchanged);
  unsigned reached = 0, caught = 0;
  for (unsigned trial = 0; trial < 64; ++trial) {
    const auto start = RandomStart(program, random);
    auto expected = start;
    Coverage coverage;
    const bool completed = Reference(expected, program.image, &coverage, program.callee);
    if (!coverage.locations.count(flat.labels.at("resume"))) continue;
    ++reached;
    Budget run_budget({50000000, 256ULL * 1024 * 1024});
    const auto seen =
        RunSsa(swapped, recovered->groups, program.image, start, run_budget, program.callee);
    caught += !seen.completed || !completed || seen.r != expected.r;
    EXPECT_FALSE(seen.completed && completed && seen.r == expected.r);
  }

  EXPECT_GT(reached, 0U);
  EXPECT_EQ(caught, reached);
}

}  // namespace
}  // namespace nyx::toy
