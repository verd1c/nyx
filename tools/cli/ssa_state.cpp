#include "ssa_state.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <fcntl.h>
#include <iterator>
#include <limits>
#include <unistd.h>

#include <sys/stat.h>

#include "nyx/eval/ssa.hpp"
#include "nyx/target/a64/decode.hpp"

namespace nyx::cli {
namespace {

constexpr std::uint64_t kPage = 4096;
constexpr std::uint64_t kLow = 0x100000000;
constexpr std::uint64_t kHigh = 0x1000000000;
constexpr std::uint64_t kMaxMemory = 64ULL << 20;
constexpr std::uint64_t kMaxEnvelope = kMaxMemory + 8 + 36 * 8 + 256 * 24 + 16 * 8;

struct Reader {
  std::span<const std::uint8_t> bytes;
  std::size_t offset = 0;

  std::optional<std::uint64_t> word() {
    if (bytes.size() - offset < 8) return {};
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value |= std::uint64_t{bytes[offset + index]} << (8 * index);
    offset += 8;
    return value;
  }

  std::optional<std::span<const std::uint8_t>> take(std::size_t count) {
    if (count > bytes.size() - offset) return {};
    const auto result = bytes.subspan(offset, count);
    offset += count;
    return result;
  }
};

struct RuntimeRegion {
  std::uint64_t address;
  std::span<const std::uint8_t> bytes;
  std::uint64_t protection;
};

struct Stub {
  std::uint64_t address;
  std::span<const std::uint8_t> bytes;
};

// The ordered crossings the QEMU runner records between the original code and
// the declared stubs, in its shape: a call's target, return address, SP and
// X0-X7, and a return's PC, SP, X0 and X1.
struct CallEvent {
  bool call;
  std::uint64_t pc, link, sp;
  std::array<std::uint64_t, 8> registers;
};

struct FileResult {
  std::optional<std::vector<std::uint8_t>> bytes;
  const char* reason = "none";
};

FileResult ReadRequest(const char* path, std::uint64_t maximum, Budget& budget) {
  const int fd = open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) return {{}, "input_io"};

  struct CloseFile {
    int fd;

    ~CloseFile() { close(fd); }
  } close_file{fd};
  struct stat metadata{};
  if (fstat(fd, &metadata) != 0 || !S_ISREG(metadata.st_mode) || metadata.st_size < 0)
    return {{}, "input_io"};
  const auto size = static_cast<std::uint64_t>(metadata.st_size);
  if (size > maximum || size > std::numeric_limits<std::size_t>::max())
    return {{}, "resource_limit"};
  if (budget.try_consume({size, size}) != BudgetDecline::none) return {{}, "resource_limit"};
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  std::size_t read_count = 0;
  while (read_count < bytes.size()) {
    const auto count = read(fd, bytes.data() + read_count, bytes.size() - read_count);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return {{}, "input_io"};
    read_count += static_cast<std::size_t>(count);
  }

  std::uint8_t extra = 0;
  ssize_t trailing;
  do trailing = read(fd, &extra, 1);
  while (trailing < 0 && errno == EINTR);
  if (trailing != 0) return {{}, "input_io"};
  return {std::move(bytes), "none"};
}

void Number(std::string& out, std::uint64_t value) {
  char digits[20];
  const auto result = std::to_chars(digits, digits + sizeof(digits), value);
  out.append(digits, result.ptr);
}

const char* OutcomeName(eval::Outcome outcome) {
  using O = eval::Outcome;
  switch (outcome) {
    case O::completed:
      return "completed";
    case O::invalid_group:
      return "invalid_group";
    case O::invalid_state:
      return "invalid_state";
    case O::resource_limit:
      return "resource_limit";
    case O::fault:
      return "fault";
    case O::unsupported:
      return "unsupported";
  }

  return "unsupported";
}

const char* StopName(eval::SsaStop stop) {
  using S = eval::SsaStop;
  switch (stop) {
    case S::returned:
      return "returned";
    case S::trap:
      return "trap";
    case S::unresolved:
      return "unresolved";
    case S::fault:
      return "fault";
    case S::declined:
      return "declined";
    case S::step_limit:
      return "step_limit";
  }

  return "declined";
}

const char* FaultName(eval::FaultKind kind) {
  using F = eval::FaultKind;
  switch (kind) {
    case F::unmapped:
      return "unmapped";
    case F::permission:
      return "permission";
    case F::address_overflow:
      return "address_overflow";
    case F::alignment:
      return "alignment";
  }

  return "unmapped";
}

}  // namespace

std::optional<std::uint64_t> SsaStubCount(std::span<const std::uint8_t> declaration) {
  if (declaration.size() > kMaxStubDeclaration) return {};
  Reader reader{declaration};
  const auto magic = reader.take(8);
  const auto count = reader.word();
  if (!magic || std::memcmp(magic->data(), "NYXSTB01", 8) != 0 || !count || !*count || *count > 32)
    return {};
  std::array<std::pair<std::uint64_t, std::uint64_t>, 32> spans{};
  for (std::uint64_t index = 0; index < *count; ++index) {
    const auto address = reader.word(), size = reader.word();
    if (!address || !size || *address % 4 || *address < kLow || *address >= kHigh || !*size ||
        *size > 128 || *size % 4 || *size > kHigh - *address ||
        !reader.take(static_cast<std::size_t>(*size)))
      return {};
    spans[index] = {*address, *address + *size};
  }

  if (reader.offset != declaration.size()) return {};
  std::sort(spans.begin(), spans.begin() + *count);
  for (std::uint64_t index = 1; index < *count; ++index)
    if (spans[index].first < spans[index - 1].second) return {};
  return count;
}

SsaStateResult EvaluateSsaRequest(const ir::SsaGraph& graph, std::span<const ir::Group> sources,
                                  std::uint64_t image_entry, const char* request_path,
                                  std::optional<std::span<const std::uint8_t>> stub_file,
                                  Budget& budget) {
  auto file = ReadRequest(request_path, kMaxEnvelope, budget);
  if (!file.bytes) return {{}, file.reason};
  Reader reader{*file.bytes};
  const auto magic = reader.take(8);
  if (!magic || std::memcmp(magic->data(), "NYXFUN01", 8) != 0) return {{}, "invalid_state"};
  std::array<std::uint64_t, 36> header{};
  for (auto& field : header) {
    const auto value = reader.word();
    if (!value) return {{}, "invalid_state"};
    field = *value;
  }

  const auto nzcv = header[31], sp = header[32], runtime_entry = header[33];
  const auto region_count = header[34], exit_count = header[35];
  if (nzcv & ~UINT64_C(0xf0000000) || runtime_entry < image_entry || region_count == 0 ||
      region_count > 256 || exit_count == 0 || exit_count > 16)
    return {{}, "invalid_state"};
  if (budget.try_consume({region_count + exit_count,
                          region_count * (sizeof(eval::RegionInput) + sizeof(RuntimeRegion)) +
                              exit_count * sizeof(std::uint64_t)}) != BudgetDecline::none)
    return {{}, "resource_limit"};
  std::vector<eval::RegionInput> regions;
  regions.reserve(static_cast<std::size_t>(region_count));
  std::vector<RuntimeRegion> runtime_regions;
  runtime_regions.reserve(static_cast<std::size_t>(region_count));
  std::uint64_t total = 0, writable = 0;
  for (std::uint64_t index = 0; index < region_count; ++index) {
    const auto address = reader.word(), size = reader.word(), protection = reader.word();
    if (!address || !size || !protection || *address % kPage || !*size || *size % kPage ||
        *address < kLow || *address >= kHigh || *size > kHigh - *address ||
        *size > kMaxMemory - total || !*protection || *protection > 15 ||
        ((*protection & 2) && (*protection & 4)) || ((*protection & 8) && !(*protection & 4)))
      return {{}, "invalid_state"};
    const auto bytes = reader.take(static_cast<std::size_t>(*size));
    if (!bytes) return {{}, "invalid_state"};
    total += *size;
    if (*protection & 2) writable += *size;
    runtime_regions.push_back({*address, *bytes, *protection});
    regions.push_back({*address, *bytes, bool(*protection & 1), bool(*protection & 2)});
  }

  if (budget.try_consume({region_count * std::bit_width(region_count), 0}) != BudgetDecline::none)
    return {{}, "resource_limit"};
  std::sort(runtime_regions.begin(), runtime_regions.end(),
            [](const RuntimeRegion& left, const RuntimeRegion& right) {
              return left.address < right.address;
            });
  for (std::size_t index = 1; index < runtime_regions.size(); ++index)
    if (runtime_regions[index].address <
        runtime_regions[index - 1].address + runtime_regions[index - 1].bytes.size())
      return {{}, "invalid_state"};
  std::vector<std::uint64_t> exits;
  exits.reserve(static_cast<std::size_t>(exit_count));
  for (std::uint64_t index = 0; index < exit_count; ++index) {
    const auto exit = reader.word();
    if (!exit || *exit % 4 || std::find(exits.begin(), exits.end(), *exit) != exits.end())
      return {{}, "invalid_state"};
    exits.push_back(*exit);
  }

  if (reader.offset != reader.bytes.size()) return {{}, "invalid_state"};
  bool lookup_exhausted = false;

  // Original code and landings never sit on STUB pages, as in the runner;
  // only a stub lookup may name one.
  const auto runtime = [&](std::uint64_t address, std::size_t size,
                           bool stub = false) -> std::optional<std::span<const std::uint8_t>> {
    if (budget.try_consume({std::uint64_t{1} + std::bit_width(runtime_regions.size()), 0}) !=
        BudgetDecline::none) {
      lookup_exhausted = true;
      return {};
    }

    const auto at = std::upper_bound(
        runtime_regions.begin(), runtime_regions.end(), address,
        [](std::uint64_t value, const RuntimeRegion& region) { return value < region.address; });
    if (at == runtime_regions.begin()) return {};
    const auto& region = *std::prev(at);
    if (!(region.protection & 4) || (!stub && (region.protection & 8)) ||
        address < region.address || address - region.address > region.bytes.size() ||
        size > region.bytes.size() - (address - region.address))
      return {};
    return region.bytes.subspan(static_cast<std::size_t>(address - region.address), size);
  };

  const auto bias = runtime_entry - image_entry;
  for (std::size_t slot = 0; slot < graph.slots(); ++slot) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return {{}, "resource_limit"};
    const auto handle = graph.Handle(slot);
    if (!handle) continue;
    const auto& block = *graph.Get(*handle);
    if (block.source_groups.size() != block.source_bytes.size()) return {{}, "invalid_graph"};
    for (std::size_t index = 0; index < block.source_groups.size(); ++index) {
      if (budget.try_consume({1, 0}) != BudgetDecline::none) return {{}, "resource_limit"};
      if (block.source_groups[index] > UINT64_MAX - bias) return {{}, "source_mismatch"};
      if (budget.try_consume({block.source_bytes[index].size(), 0}) != BudgetDecline::none)
        return {{}, "resource_limit"};
      const auto bytes =
          runtime(bias + block.source_groups[index], block.source_bytes[index].size());
      if (lookup_exhausted) return {{}, "resource_limit"};
      if (!bytes || !std::equal(bytes->begin(), bytes->end(), block.source_bytes[index].begin()))
        return {{}, "source_mismatch"};
    }
  }

  constexpr std::array<std::uint8_t, 4> breakpoint{0, 0, 0x20, 0xd4};
  for (const auto exit : exits) {
    const auto bytes = runtime(exit, breakpoint.size());
    if (lookup_exhausted) return {{}, "resource_limit"};
    if (!bytes || !std::equal(bytes->begin(), bytes->end(), breakpoint.begin()))
      return {{}, "invalid_state"};
  }

  std::vector<Stub> stubs;
  if (stub_file) {
    if (!SsaStubCount(*stub_file)) return {{}, "invalid_stubs"};
    Reader stub_reader{*stub_file};
    const auto stub_magic = stub_reader.take(8);
    const auto count = stub_reader.word();
    if (!stub_magic || std::memcmp(stub_magic->data(), "NYXSTB01", 8) != 0 || !count || !*count ||
        *count > 32)
      return {{}, "invalid_stubs"};
    if (budget.try_consume({*count, *count * sizeof(Stub)}) != BudgetDecline::none)
      return {{}, "resource_limit"};
    stubs.reserve(static_cast<std::size_t>(*count));
    for (std::uint64_t index = 0; index < *count; ++index) {
      const auto address = stub_reader.word(), size = stub_reader.word();
      if (!address || !size || *address % 4 || *address < kLow || *address >= kHigh || !*size ||
          *size > 128 || *size % 4 || *size > kHigh - *address)
        return {{}, "invalid_stubs"};
      const auto bytes = stub_reader.take(static_cast<std::size_t>(*size));
      if (!bytes) return {{}, "invalid_stubs"};
      if (budget.try_consume({*size, 0}) != BudgetDecline::none) return {{}, "resource_limit"};
      const auto mapped = runtime(*address, static_cast<std::size_t>(*size), true);
      if (lookup_exhausted) return {{}, "resource_limit"};
      if (!mapped || !std::equal(mapped->begin(), mapped->end(), bytes->begin()))
        return {{}, "stub_mismatch"};
      // With a traced STUB domain, every stub belongs to it, as in the runner.
      if (budget.try_consume({runtime_regions.size(), 0}) != BudgetDecline::none)
        return {{}, "resource_limit"};
      const bool traced =
          std::any_of(runtime_regions.begin(), runtime_regions.end(),
                      [](const RuntimeRegion& region) { return region.protection & 8; });
      if (traced && !std::any_of(runtime_regions.begin(), runtime_regions.end(),
                                 [&](const RuntimeRegion& region) {
                                   return (region.protection & 8) && *address >= region.address &&
                                          *address - region.address < region.bytes.size() &&
                                          *size <=
                                              region.bytes.size() - (*address - region.address);
                                 }))
        return {{}, "invalid_stubs"};
      for (const auto& source : sources) {
        if (budget.try_consume({1, 0}) != BudgetDecline::none) return {{}, "resource_limit"};
        if (source.source_address() > UINT64_MAX - bias) return {{}, "invalid_stubs"};
        const auto begin = bias + source.source_address();
        if (source.bytes().size() > UINT64_MAX - begin) return {{}, "invalid_stubs"};
        const auto end = begin + source.bytes().size();
        if (*address < end && begin < *address + *size) return {{}, "invalid_stubs"};
      }

      stubs.push_back({*address, *bytes});
    }

    if (stub_reader.offset != stub_reader.bytes.size()) return {{}, "invalid_stubs"};
    if (budget.try_consume({*count * std::bit_width(*count), 0}) != BudgetDecline::none)
      return {{}, "resource_limit"};
    std::sort(stubs.begin(), stubs.end(),
              [](const Stub& left, const Stub& right) { return left.address < right.address; });
    for (std::size_t index = 1; index < stubs.size(); ++index)
      if (stubs[index].address < stubs[index - 1].address + stubs[index - 1].bytes.size())
        return {{}, "invalid_stubs"};
  }

  auto memory = eval::Memory::Create(regions, budget);
  if (!memory.memory)
    return {
        {},
        memory.status == eval::MemoryStatus::resource_limit ? "resource_limit" : "invalid_state"};
  if (budget.try_consume({36, 36 * sizeof(eval::Cell)}) != BudgetDecline::none)
    return {{}, "resource_limit"};
  eval::State state;
  state.cells.reserve(36);
  for (unsigned index = 0; index < 36; ++index) {
    const auto value = index < 31    ? header[index]
                       : index == 31 ? sp
                                     : (nzcv >> (31 - (index - 32))) & 1;
    auto bits = BitVector::from_u64(index < 32 ? 64 : 1, value, 64, budget);
    if (!bits) return {{}, "resource_limit"};
    state.cells.push_back({index, std::move(*bits)});
  }

  std::optional<ir::SsaHandle> selected_entry;
  for (const auto handle : graph.entries()) {
    if (budget.try_consume({1, 0}) != BudgetDecline::none) return {{}, "resource_limit"};
    const auto* block = graph.Get(handle);
    if (block && block->address == image_entry) selected_entry = handle;
  }

  if (!selected_entry) return {{}, "invalid_entry"};
  std::size_t stub_calls = 0;
  std::vector<std::uint64_t> call_targets;
  std::vector<CallEvent> call_events;

  // Both vectors grow by one reserved element at a time, charged before growth.
  const auto record = [&](const eval::State& live, bool call, std::uint64_t pc, Budget& work) {
    if (work.try_consume(
            {1 + (call_events.size() == call_events.capacity() ? call_events.size() + 1 : 0),
             call_events.size() == call_events.capacity()
                 ? (call_events.size() + 1) * sizeof(CallEvent)
                 : 0}) != BudgetDecline::none)
      return false;
    if (call_events.size() == call_events.capacity()) call_events.reserve(call_events.size() + 1);
    CallEvent event{
        call, pc, call ? live.cells[30].value.word(0) : 0, live.cells[31].value.word(0), {}};
    for (unsigned index = 0; index < (call ? 8U : 2U); ++index)
      event.registers[index] = live.cells[index].value.word(0);
    call_events.push_back(event);
    return true;
  };

  const eval::SsaCallee callback =
      stubs.empty()
          ? eval::SsaCallee{}
          : eval::SsaCallee{[&](std::uint64_t target, eval::State& live, eval::Memory& live_memory,
                                Budget& work) -> eval::SsaCalleeOutcome {
              if (++stub_calls > 128 ||
                  work.try_consume({stubs.size() + (call_targets.size() == call_targets.capacity()
                                                        ? call_targets.size() + 1
                                                        : 0),
                                    call_targets.size() == call_targets.capacity()
                                        ? (call_targets.size() + 1) * sizeof(std::uint64_t)
                                        : 0}) != BudgetDecline::none)
                return {eval::Outcome::resource_limit, {}};
              if (call_targets.size() == call_targets.capacity())
                call_targets.reserve(call_targets.size() + 1);
              call_targets.push_back(target);
              if (!record(live, true, target, work)) return {eval::Outcome::resource_limit, {}};
              const auto found = std::find_if(stubs.begin(), stubs.end(), [&](const Stub& stub) {
                return stub.address == target;
              });
              if (found == stubs.end()) return {eval::Outcome::unsupported, {}};
              for (std::size_t offset = 0; offset < found->bytes.size(); offset += 4) {
                std::array<std::uint8_t, 4> bytes{};
                std::copy_n(found->bytes.begin() + offset, 4, bytes.begin());
                auto decoded = a64::Decode(found->address + offset, bytes, work,
                                           {a64::MemoryProfile::concrete_atomic_scalar});
                if (!decoded.group)
                  return {decoded.reason == a64::DecodeDecline::work_limit ||
                                  decoded.reason == a64::DecodeDecline::byte_limit
                              ? eval::Outcome::resource_limit
                              : eval::Outcome::unsupported,
                          {}};
                if (work.try_consume({decoded.group->writes().size() * a64::kCalleeSaved.size(),
                                      0}) != BudgetDecline::none)
                  return {eval::Outcome::resource_limit, {}};
                for (const auto& write : decoded.group->writes())
                  if (std::find(a64::kCalleeSaved.begin(), a64::kCalleeSaved.end(),
                                write.storage) != a64::kCalleeSaved.end())
                    return {eval::Outcome::unsupported, {}};
                const auto step = eval::Execute(*decoded.group, live, live_memory, work, {}, {0});
                if (step.outcome != eval::Outcome::completed)
                  return {step.outcome == eval::Outcome::resource_limit
                              ? step.outcome
                              : eval::Outcome::unsupported,
                          {}};
                if (offset + 4 == found->bytes.size()) {
                  if (!step.transfer || step.transfer->kind != ir::TransferKind::return_)
                    return {eval::Outcome::unsupported, {}};
                  if (!record(live, false, step.transfer->target, work))
                    return {eval::Outcome::resource_limit, {}};
                  return {eval::Outcome::completed, step.transfer->target};
                }

                if (step.transfer) return {eval::Outcome::unsupported, {}};
              }

              return {eval::Outcome::unsupported, {}};
            }};
  const auto run = eval::ExecuteSsa(graph, sources, *selected_entry, state, *memory.memory, budget,
                                    {runtime_entry - image_entry}, callback);
  if (run.outcome == eval::Outcome::resource_limit) return {{}, "resource_limit"};
  std::uint64_t stub_bytes = 0;
  for (const auto& stub : stubs) stub_bytes += stub.bytes.size();
  if (writable > (std::numeric_limits<std::uint64_t>::max() - 8192) / 2 ||
      region_count > (std::numeric_limits<std::uint64_t>::max() - 8192 - 2 * writable) / 128 ||
      run.visited_blocks.size() >
          (std::numeric_limits<std::uint64_t>::max() - 8192 - 2 * writable - 128 * region_count) /
              48 ||
      call_targets.size() > (std::numeric_limits<std::uint64_t>::max() - 8192 - 2 * writable -
                             128 * region_count - 48 * run.visited_blocks.size()) /
                                32 ||
      stub_bytes >
          (std::numeric_limits<std::uint64_t>::max() - 8192 - 2 * writable - 128 * region_count -
           48 * run.visited_blocks.size() - 32 * call_targets.size() - 128 * stubs.size()) /
              2 ||
      call_events.size() > (std::numeric_limits<std::uint64_t>::max() - 8192 - 2 * writable -
                            128 * region_count - 48 * run.visited_blocks.size() -
                            32 * call_targets.size() - 128 * stubs.size() - 2 * stub_bytes) /
                               384)
    return {{}, "resource_limit"};
  const auto capacity = 8192 + 2 * writable + 128 * region_count + 48 * run.visited_blocks.size() +
                        32 * call_targets.size() + 128 * stubs.size() + 2 * stub_bytes +
                        384 * call_events.size();
  if (capacity > std::numeric_limits<std::size_t>::max() ||
      budget.try_consume({2 * writable + region_count + 36 + 48 * run.visited_blocks.size() +
                              32 * call_targets.size() + 2 * stub_bytes + 128 * stubs.size() +
                              exit_count + 384 * call_events.size(),
                          capacity}) != BudgetDecline::none)
    return {{}, "resource_limit"};
  std::string json;
  json.reserve(static_cast<std::size_t>(capacity));
  const bool landed = run.outcome == eval::Outcome::completed &&
                      run.stop == eval::SsaStop::returned &&
                      std::find(exits.begin(), exits.end(), run.runtime_pc) != exits.end();
  json += "{\"status\":\"";
  json += landed ? "completed" : run.outcome == eval::Outcome::fault ? "fault" : "inconclusive";
  json += "\",\"outcome\":\"";
  json += OutcomeName(run.outcome);
  json += "\",\"stop\":\"";
  json += StopName(run.stop);
  json += "\",\"pc\":";
  Number(json, run.runtime_pc);
  json += ",\"completed_blocks\":";
  Number(json, run.completed_blocks);
  json += ",\"declared_stub_count\":";
  Number(json, stubs.size());
  json += ",\"declared_stubs\":[";
  constexpr char hex[] = "0123456789abcdef";
  for (std::size_t index = 0; index < stubs.size(); ++index) {
    if (index) json += ',';
    json += "{\"address\":";
    Number(json, stubs[index].address);
    json += ",\"bytes\":\"";
    for (const auto byte : stubs[index].bytes) {
      json += hex[byte >> 4];
      json += hex[byte & 15];
    }

    json += "\"}";
  }

  json += ']';
  json += ",\"external_calls\":";
  Number(json, stub_calls);
  json += ",\"call_targets\":[";
  for (std::size_t index = 0; index < call_targets.size(); ++index) {
    if (index) json += ',';
    Number(json, call_targets[index]);
  }

  json += "],\"call_events\":[";
  for (std::size_t index = 0; index < call_events.size(); ++index) {
    const auto& event = call_events[index];
    if (index) json += ',';
    json += event.call ? "{\"kind\":\"call\",\"target\":" : "{\"kind\":\"return\",\"pc\":";
    Number(json, event.pc);
    if (event.call) {
      json += ",\"return_address\":";
      Number(json, event.link);
    }

    json += ",\"sp\":";
    Number(json, event.sp);
    json += event.call ? ",\"arguments\":[" : ",\"results\":[";
    for (unsigned register_index = 0; register_index < (event.call ? 8U : 2U); ++register_index) {
      if (register_index) json += ',';
      Number(json, event.registers[register_index]);
    }

    json += "]}";
  }

  json += ']';
  json += ",\"stub_semantics\":\"decoded_nyxil_bound_runtime_bytes\"";
  json += ",\"visited_blocks\":[";
  for (std::size_t index = 0; index < run.visited_blocks.size(); ++index) {
    if (index) json += ',';
    json += "\"b";
    Number(json, run.visited_blocks[index].slot);
    json += '.';
    Number(json, run.visited_blocks[index].generation);
    json += '"';
  }

  json += ']';
  json += ",\"registers\":[";
  for (unsigned index = 0; index < 31; ++index) {
    if (index) json += ',';
    Number(json, state.cells[index].value.word(0));
  }

  json += "],\"nzcv\":";
  Number(json, (state.cells[32].value.word(0) << 31) | (state.cells[33].value.word(0) << 30) |
                   (state.cells[34].value.word(0) << 29) | (state.cells[35].value.word(0) << 28));
  json += ",\"sp\":";
  Number(json, state.cells[31].value.word(0));
  json += ",\"memory\":[";
  bool first = true;
  for (const auto& region : memory.memory->Regions()) {
    if (!region.writable) continue;
    if (!first) json += ',';
    first = false;
    json += "{\"address\":";
    Number(json, region.address);
    json += ",\"bytes\":\"";
    for (const auto byte : region.bytes) {
      json += hex[byte >> 4];
      json += hex[byte & 15];
    }

    json += "\"}";
  }

  json += "],\"fault\":";
  if (run.fault) {
    json += "{\"kind\":\"";
    json += FaultName(run.fault->kind);
    json += "\",\"address\":";
    Number(json, run.fault->address);
    json += '}';
  } else
    json += "null";
  json += ",\"independent_check\":\"NOT CHECKED\"}";
  return {std::move(json), "none"};
}

}  // namespace nyx::cli
