#define _GNU_SOURCE
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>

#include <sys/mman.h>

static uint64_t input[41], sources[16], lengths[16], exits[16], addresses[8];
static unsigned char code[4096], initial[8192], pages[8][4096];
static unsigned char response[8 + 41 * 8 + 8192 + 8 * 4104];
static unsigned char alternate[65536] __attribute__((aligned(16)));
static volatile sig_atomic_t started;
static unsigned header_words;

static int in_source(uint64_t pc) {
  for (unsigned i = 0; i < input[36]; ++i) {
    uint64_t start = sources[i] + input[34];
    if (pc >= start && pc - start < lengths[i] && pc % 4 == 0) return 1;
  }

  return 0;
}

static void handle(int signal, siginfo_t* info, void* raw) {
  ucontext_t* context = raw;
  if (!started) {
    if (signal != SIGTRAP || info->si_code != SI_TKILL) _exit(20);
    started = 1;
    memcpy(context->uc_mcontext.regs, input, 31 * 8);
    context->uc_mcontext.pstate = (context->uc_mcontext.pstate & ~UINT64_C(0xf0000000)) | input[31];
    context->uc_mcontext.sp = input[32];
    context->uc_mcontext.pc = input[35] + input[34];
    return;
  }

  uint64_t output[41];
  memcpy(output, context->uc_mcontext.regs, 31 * 8);
  output[31] = context->uc_mcontext.pstate & UINT64_C(0xf0000000);
  output[32] = context->uc_mcontext.sp;
  output[33] = input[33];
  output[34] = input[34];
  output[35] = context->uc_mcontext.pc;
  output[38] = input[38];
  output[39] = input[39];
  output[40] = input[40];
  int completed = 0;
  for (unsigned i = 0; i < input[37]; ++i) {
    if (output[35] == exits[i] + input[34]) completed = 1;
  }

  if (signal == SIGTRAP && info->si_code == TRAP_BRKPT && completed) {
    output[36] = 0;
    output[37] = 0;
  } else if ((signal == SIGSEGV || signal == SIGBUS) && in_source(output[35])) {
    output[36] = signal;
    output[37] = (uintptr_t)info->si_addr;
  } else
    _exit(21);
  memcpy(response,
         header_words == 41   ? "NYXSPR03"
         : header_words == 40 ? "NYXSPR02"
                              : "NYXSPR01",
         8);
  size_t header_size = header_words * 8;
  memcpy(response + 8, output, header_size);
  memcpy(response + 8 + header_size, (void*)(uintptr_t)input[33], input[39]);
  size_t size = 8 + header_size + input[39];
  for (unsigned i = 0; i < input[38]; ++i) {
    memcpy(response + size, &addresses[i], 8);
    memcpy(response + size + 8, (void*)(uintptr_t)addresses[i], 4096);
    size += 4104;
  }

  size_t sent = 0;
  while (sent < size) {
    ssize_t n = write(STDOUT_FILENO, response + sent, size - sent);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) _exit(22);
    sent += (size_t)n;
  }

  _exit(0);
}

static int add_page(uint64_t* mapped, unsigned* count, uint64_t page) {
  for (unsigned i = 0; i < *count; ++i)
    if (mapped[i] == page) return 1;
  if (*count == 16) return 0;
  mapped[(*count)++] = page;
  return 1;
}

static void sort(uint64_t* values, unsigned count) {
  for (unsigned i = 1; i < count; ++i) {
    uint64_t value = values[i];
    unsigned j = i;
    while (j && values[j - 1] > value) {
      values[j] = values[j - 1];
      --j;
    }

    values[j] = value;
  }
}

int main(void) {
  // Python admits only caller-reviewed trusted fixtures. This runner is not a
  // security sandbox for arbitrary corpus code or caller-supplied addresses.
  unsigned char magic[8];
  if (fread(magic, 1, 8, stdin) != 8) return 2;
  header_words = memcmp(magic, "NYXSPR01", 8) == 0   ? 39
                 : memcmp(magic, "NYXSPR02", 8) == 0 ? 40
                 : memcmp(magic, "NYXSPR03", 8) == 0 ? 41
                                                     : 0;
  if (!header_words || fread(input, 8, header_words, stdin) != header_words) return 2;
  if (header_words == 39) input[39] = 4096;
  if (header_words < 41) input[40] = 0;
  if ((header_words >= 40 && input[39] != 8192) || input[31] & ~UINT64_C(0xf0000000) ||
      input[33] < UINT64_C(0x100000000) || input[33] > UINT64_C(0x1000000000) || input[33] % 4096 ||
      !input[36] || input[36] > 16 || !input[37] || input[37] > 16 || input[38] > 8 ||
      input[40] >> input[38])
    return 2;
  uint64_t code_pages[16], global_pages[8];
  unsigned code_count = 0;
  size_t offsets[16], used = 0;
  for (unsigned i = 0; i < input[36]; ++i) {
    if (fread(&sources[i], 8, 1, stdin) != 1 || fread(&lengths[i], 8, 1, stdin) != 1 ||
        !lengths[i] || lengths[i] > 4096 - used || sources[i] % 4 || lengths[i] % 4 ||
        sources[i] > UINT64_MAX - lengths[i])
      return 2;
    uint64_t runtime = sources[i] + input[34];
    if (runtime < UINT64_C(0x200000000) || runtime > UINT64_C(0x1000000000) ||
        runtime > UINT64_MAX - lengths[i] || runtime % 4)
      return 2;
    offsets[i] = used;
    if (fread(code + used, 1, lengths[i], stdin) != lengths[i]) return 2;
    used += lengths[i];
    for (uint64_t p = runtime & ~UINT64_C(4095);
         p <= ((runtime + lengths[i] - 1) & ~UINT64_C(4095)); p += 4096) {
      if (!add_page(code_pages, &code_count, p)) return 2;
    }

    for (unsigned j = 0; j < i; ++j) {
      uint64_t other = sources[j] + input[34];
      if (runtime < other + lengths[j] && other < runtime + lengths[i]) return 2;
    }
  }

  if (!in_source(input[35] + input[34])) return 2;
  for (unsigned i = 0; i < input[37]; ++i) {
    if (fread(&exits[i], 8, 1, stdin) != 1) return 2;
    uint64_t runtime = exits[i] + input[34];
    if (runtime % 4 || runtime < UINT64_C(0x200000000) || runtime > UINT64_C(0x1000000000) ||
        in_source(runtime) || !add_page(code_pages, &code_count, runtime & ~UINT64_C(4095)))
      return 2;
    for (unsigned j = 0; j < i; ++j)
      if (exits[j] == exits[i]) return 2;
  }

  if (fread(initial, 1, input[39], stdin) != input[39]) return 2;
  for (unsigned i = 0; i < input[38]; ++i) {
    if (fread(&addresses[i], 8, 1, stdin) != 1 || fread(pages[i], 1, 4096, stdin) != 4096 ||
        addresses[i] % 4096 || addresses[i] < UINT64_C(0x100000000) ||
        addresses[i] > UINT64_C(0x1000000000))
      return 2;
    global_pages[i] = addresses[i];
    for (unsigned j = 0; j < i; ++j)
      if (addresses[j] == addresses[i]) return 2;
  }

  if (getchar() != EOF) return 2;
  sort(code_pages, code_count);
  sort(global_pages, input[38]);
  uint64_t starts[25] = {input[33] - 4096}, ends[25] = {input[33] + input[39] + 4096};
  unsigned kinds[25] = {0}, arenas = 1;
  for (unsigned kind = 1; kind <= 2; ++kind) {
    uint64_t* items = kind == 1 ? code_pages : global_pages;
    unsigned count = kind == 1 ? code_count : input[38];
    for (unsigned i = 0; i < count; ++i) {
      // A page one past the last shares its guard page, so the arena grows over the gap.
      if (arenas > 1 && kinds[arenas - 1] == kind && items[i] <= ends[arenas - 1])
        ends[arenas - 1] = items[i] + 8192;
      else {
        starts[arenas] = items[i] - 4096;
        ends[arenas] = items[i] + 8192;
        kinds[arenas++] = kind;
      }
    }
  }

  for (unsigned i = 0; i < arenas; ++i) {
    for (unsigned j = 0; j < i; ++j)
      if (starts[i] < ends[j] && starts[j] < ends[i]) return 2;
    void* mapping = mmap((void*)(uintptr_t)starts[i], ends[i] - starts[i], PROT_NONE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (mapping == MAP_FAILED || (uintptr_t)mapping != starts[i]) return 3;
    if (mprotect((void*)(uintptr_t)(starts[i] + 4096), ends[i] - starts[i] - 8192,
                 PROT_READ | PROT_WRITE))
      return 4;
    if (kinds[i] == 1) {
      for (uint64_t at = starts[i] + 4096; at < ends[i] - 4096; at += 4)
        *(uint32_t*)(uintptr_t)at = UINT32_C(0xd4200000);
    }
  }

  memcpy((void*)(uintptr_t)input[33], initial, input[39]);
  for (unsigned i = 0; i < input[36]; ++i)
    memcpy((void*)(uintptr_t)(sources[i] + input[34]), code + offsets[i], lengths[i]);
  for (unsigned i = 0; i < input[38]; ++i) memcpy((void*)(uintptr_t)addresses[i], pages[i], 4096);
  for (unsigned i = 1; i < arenas; ++i) {
    char* begin = (char*)(uintptr_t)(starts[i] + 4096);
    size_t length = ends[i] - starts[i] - 8192;
    if (kinds[i] == 1) __builtin___clear_cache(begin, begin + length);
    if (mprotect(begin, length, PROT_READ | (kinds[i] == 1 ? PROT_EXEC : 0))) return 5;
  }

  for (unsigned i = 0; i < input[38]; ++i) {
    if (input[40] >> i & 1 &&
        mprotect((void*)(uintptr_t)addresses[i], 4096, PROT_READ | PROT_WRITE))
      return 5;
  }

  for (unsigned i = 1; i < arenas; ++i) {
    if (kinds[i] != 2) continue;
    for (uint64_t page = starts[i] + 4096; page < ends[i] - 4096; page += 4096) {
      int listed = 0;
      for (unsigned j = 0; j < input[38]; ++j) listed |= addresses[j] == page;
      if (!listed && mprotect((void*)(uintptr_t)page, 4096, PROT_NONE)) return 5;
    }
  }

  stack_t stack = {.ss_sp = alternate, .ss_size = sizeof(alternate)};
  if (sigaltstack(&stack, NULL)) return 6;
  struct sigaction action = {.sa_sigaction = handle, .sa_flags = SA_SIGINFO | SA_ONSTACK};
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGTRAP, &action, NULL) || sigaction(SIGSEGV, &action, NULL) ||
      sigaction(SIGBUS, &action, NULL))
    return 7;
  raise(SIGTRAP);
  return 8;
}
