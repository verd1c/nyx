#define _GNU_SOURCE
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>

#include <sys/mman.h>

// NYXFUN02: page-aligned regions with their own permissions, one entry, BRK
// landings that end a run successfully, and declared stub entries. The whole
// original function runs once; any signal ends it, and the writable regions and
// the ordered call events are returned afterwards.
//
// STUB regions hold caller-written callees. Only one domain, the original code
// or the stubs, is executable at a time, so each crossing is an instruction
// fetch fault that records an event before the other domain is enabled. The
// events come from QEMU's own execution, not from the evaluator under test.
enum { MAX_REGIONS = 256, MAX_EXITS = 16, MAX_STUBS = 32, MAX_EVENTS = 64 };

enum { READ = 1, WRITE = 2, EXECUTE = 4, STUB = 8 };

enum { EVENT_CALL = 1, EVENT_RETURN = 2, EVENT_ENTER_OTHER = 3 };

// Outcomes past every signal number: a run that left the traced envelope.
enum { ENTERED_STUB_BODY = 256, EVENT_OVERFLOW = 257 };

#define LOW UINT64_C(0x100000000)
#define HIGH UINT64_C(0x1000000000)
#define MAX_BYTES (UINT64_C(64) << 20)

static uint64_t input[36], addresses[MAX_REGIONS], sizes[MAX_REGIONS], protections[MAX_REGIONS];
static uint64_t exits[MAX_EXITS], stubs[MAX_STUBS], stub_count;

// kind, target or PC, return address, SP, then X0-X7.
static uint64_t events[MAX_EVENTS][12], event_count;
static volatile sig_atomic_t in_stub;
static unsigned char alternate[65536] __attribute__((aligned(16)));
static volatile sig_atomic_t started;

static void emit(const void* data, size_t size) {
  const unsigned char* bytes = data;
  size_t sent = 0;
  while (sent < size) {
    ssize_t n = write(STDOUT_FILENO, bytes + sent, size - sent);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) _exit(22);
    sent += (size_t)n;
  }
}

static int region_of(uint64_t address) {
  for (unsigned i = 0; i < input[34]; ++i)
    if (address >= addresses[i] && address - addresses[i] < sizes[i]) return (int)i;
  return -1;
}

static int protect(unsigned i, int stub_domain) {
  int protection =
      (protections[i] & READ ? PROT_READ : 0) | (protections[i] & WRITE ? PROT_WRITE : 0);
  if (protections[i] & EXECUTE && !(protections[i] & STUB) == !stub_domain) protection |= PROT_EXEC;
  return mprotect((void*)(uintptr_t)addresses[i], sizes[i], protection);
}

// A fetch fault into the other executable domain is a crossing, not a fault.
// Returns nonzero when the run continues in the other domain.
static int cross(ucontext_t* context, siginfo_t* info, uint64_t* outcome) {
  uint64_t pc = context->uc_mcontext.pc;
  int region = region_of(pc);
  if ((uintptr_t)info->si_addr != pc || region < 0 || !(protections[region] & EXECUTE) ||
      !(protections[region] & STUB) == !in_stub)
    return 0;
  if (event_count == MAX_EVENTS) {
    *outcome = EVENT_OVERFLOW;
    return 0;
  }

  uint64_t* event = events[event_count];
  memset(event, 0, sizeof(events[0]));
  event[1] = pc;
  event[3] = context->uc_mcontext.sp;
  if (in_stub) {
    event[0] = EVENT_RETURN;
    memcpy(event + 4, context->uc_mcontext.regs, 2 * 8);
  } else {
    int declared = 0;
    for (unsigned i = 0; i < stub_count; ++i) declared |= stubs[i] == pc;
    event[0] = declared ? EVENT_CALL : EVENT_ENTER_OTHER;
    event[2] = context->uc_mcontext.regs[30];
    memcpy(event + 4, context->uc_mcontext.regs, 8 * 8);
    if (!declared) {
      ++event_count;
      *outcome = ENTERED_STUB_BODY;
      return 0;
    }
  }

  ++event_count;
  in_stub = !in_stub;
  for (unsigned i = 0; i < input[34]; ++i)
    if (protections[i] & EXECUTE && protect(i, in_stub)) _exit(23);
  return 1;
}

static void handle(int signal, siginfo_t* info, void* raw) {
  ucontext_t* context = raw;
  if (!started) {
    if (signal != SIGTRAP || info->si_code != SI_TKILL) _exit(20);
    started = 1;
    memcpy(context->uc_mcontext.regs, input, 31 * 8);
    context->uc_mcontext.pstate = (context->uc_mcontext.pstate & ~UINT64_C(0xf0000000)) | input[31];
    context->uc_mcontext.sp = input[32];
    context->uc_mcontext.pc = input[33];
    return;
  }

  uint64_t traced = 0;
  if (signal == SIGSEGV && cross(context, info, &traced)) return;
  uint64_t output[36];
  memcpy(output, context->uc_mcontext.regs, 31 * 8);
  output[31] = context->uc_mcontext.pstate & UINT64_C(0xf0000000);
  output[32] = context->uc_mcontext.sp;
  output[33] = context->uc_mcontext.pc;
  output[34] = (uint64_t)signal;
  output[35] = 0;
  if (signal == SIGTRAP) {
    // A BRK at a declared landing is completion; any other BRK is a trap.
    for (unsigned i = 0; i < input[35]; ++i)
      if (output[33] == exits[i] && info->si_code == TRAP_BRKPT) output[34] = 0;
  } else if (signal == SIGSEGV || signal == SIGBUS) {
    output[35] = (uintptr_t)info->si_addr;
  } else if (signal != SIGILL) {
    _exit(21);
  }

  if (traced) {
    output[34] = traced;
    output[35] = 0;
  }

  emit("NYXFUN02", 8);
  emit(output, sizeof(output));
  for (unsigned i = 0; i < input[34]; ++i)
    if (protections[i] & WRITE) emit((void*)(uintptr_t)addresses[i], sizes[i]);
  emit(&event_count, sizeof(event_count));
  emit(events, event_count * sizeof(events[0]));
  _exit(0);
}

int main(void) {
  // Python admits only caller-reviewed trusted fixtures. This runner is not a
  // security sandbox for arbitrary corpus code or caller-supplied addresses.
  unsigned char magic[8];
  if (fread(magic, 1, 8, stdin) != 8 || memcmp(magic, "NYXFUN02", 8)) return 2;
  if (fread(input, 8, 36, stdin) != 36) return 2;
  if (input[31] & ~UINT64_C(0xf0000000) || !input[34] || input[34] > MAX_REGIONS || !input[35] ||
      input[35] > MAX_EXITS)
    return 2;
  uint64_t total = 0;
  for (unsigned i = 0; i < input[34]; ++i) {
    if (fread(&addresses[i], 8, 1, stdin) != 1 || fread(&sizes[i], 8, 1, stdin) != 1 ||
        fread(&protections[i], 8, 1, stdin) != 1)
      return 2;
    if (addresses[i] % 4096 || !sizes[i] || sizes[i] % 4096 || addresses[i] < LOW ||
        addresses[i] > HIGH || sizes[i] > HIGH - addresses[i] || sizes[i] > MAX_BYTES - total ||
        !protections[i] || protections[i] & ~UINT64_C(15) ||
        (protections[i] & WRITE && protections[i] & EXECUTE) ||
        (protections[i] & STUB && !(protections[i] & EXECUTE)))
      return 2;
    total += sizes[i];
    for (unsigned j = 0; j < i; ++j)
      if (addresses[i] < addresses[j] + sizes[j] && addresses[j] < addresses[i] + sizes[i])
        return 2;
    void* mapping = mmap((void*)(uintptr_t)addresses[i], sizes[i], PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (mapping == MAP_FAILED || (uintptr_t)mapping != addresses[i]) return 3;
    if (fread(mapping, 1, sizes[i], stdin) != sizes[i]) return 2;
  }

  for (unsigned i = 0; i < input[35]; ++i) {
    if (fread(&exits[i], 8, 1, stdin) != 1 || exits[i] % 4) return 2;
    int landing = 0;
    for (unsigned j = 0; j < input[34]; ++j)
      landing |= protections[j] & EXECUTE && exits[i] >= addresses[j] &&
                 exits[i] - addresses[j] < sizes[j] &&
                 *(uint32_t*)(uintptr_t)exits[i] == UINT32_C(0xd4200000);
    if (!landing || protections[region_of(exits[i])] & STUB) return 2;
  }

  if (fread(&stub_count, 8, 1, stdin) != 1 || stub_count > MAX_STUBS) return 2;
  for (unsigned i = 0; i < stub_count; ++i) {
    if (fread(&stubs[i], 8, 1, stdin) != 1 || stubs[i] % 4) return 2;
    int region = region_of(stubs[i]);
    if (region < 0 || !(protections[region] & STUB)) return 2;
  }

  if (getchar() != EOF) return 2;
  int entry = 0;
  for (unsigned i = 0; i < input[34]; ++i) {
    char* begin = (char*)(uintptr_t)addresses[i];
    if (protections[i] & EXECUTE) __builtin___clear_cache(begin, begin + sizes[i]);
    if (protect(i, 0)) return 5;
    entry |= protections[i] & EXECUTE && !(protections[i] & STUB) && input[33] >= addresses[i] &&
             input[33] - addresses[i] < sizes[i];
  }

  if (!entry || input[33] % 4) return 2;
  stack_t stack = {.ss_sp = alternate, .ss_size = sizeof(alternate)};
  if (sigaltstack(&stack, NULL)) return 6;
  struct sigaction action = {.sa_sigaction = handle, .sa_flags = SA_SIGINFO | SA_ONSTACK};
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGTRAP, &action, NULL) || sigaction(SIGSEGV, &action, NULL) ||
      sigaction(SIGBUS, &action, NULL) || sigaction(SIGILL, &action, NULL))
    return 7;
  raise(SIGTRAP);
  return 8;
}
