#define _GNU_SOURCE
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>

#include <asm/sigcontext.h>
#include <sys/mman.h>

// runner_memory.c with the 32 Q registers added to the state: the signal
// context's FPSIMD record is both where they are seeded and where they are read.
static uint64_t input[35];
static unsigned char vectors[32 * 16];
static unsigned char initial[4096];
static unsigned char response[8 + 37 * 8 + 32 * 16 + 4096];
static unsigned char alternate[65536] __attribute__((aligned(16)));
static unsigned char* scratch;
static unsigned char* code;
static volatile sig_atomic_t started;

static struct fpsimd_context* fpsimd(ucontext_t* context) {
  struct _aarch64_ctx* record = (void*)context->uc_mcontext.__reserved;
  size_t remaining = sizeof(context->uc_mcontext.__reserved);
  while (remaining >= sizeof(*record) && record->size >= sizeof(*record) &&
         record->size <= remaining) {
    if (record->magic == FPSIMD_MAGIC && record->size >= sizeof(struct fpsimd_context))
      return (void*)record;
    remaining -= record->size;
    record = (void*)((unsigned char*)record + record->size);
  }

  _exit(23);
}

static void handle(int signal, siginfo_t* info, void* raw) {
  ucontext_t* context = raw;
  if (!started) {
    if (signal != SIGTRAP || info->si_code != SI_TKILL) _exit(20);
    started = 1;
    memcpy(context->uc_mcontext.regs, input, 31 * 8);
    context->uc_mcontext.pstate = (context->uc_mcontext.pstate & ~UINT64_C(0xf0000000)) | input[31];
    context->uc_mcontext.sp = input[32];
    context->uc_mcontext.pc = (uintptr_t)code;
    memcpy(fpsimd(context)->vregs, vectors, sizeof(vectors));
    return;
  }

  uint64_t output[37];
  memcpy(output, context->uc_mcontext.regs, 31 * 8);
  output[31] = context->uc_mcontext.pstate & UINT64_C(0xf0000000);
  output[32] = context->uc_mcontext.sp;
  output[33] = context->uc_mcontext.pc - (uintptr_t)code;
  if (signal == SIGTRAP && info->si_code == TRAP_BRKPT && output[33] == 4) {
    output[34] = 0;
    output[35] = 0;
  } else if ((signal == SIGSEGV || signal == SIGBUS) && output[33] == 0) {
    output[34] = signal;
    output[35] = (uintptr_t)info->si_addr;
  } else {
    _exit(21);
  }

  output[36] = (uintptr_t)code;
  memcpy(response, "NYXVEC01", 8);
  memcpy(response + 8, output, sizeof(output));
  memcpy(response + 8 + sizeof(output), fpsimd(context)->vregs, sizeof(vectors));
  memcpy(response + 8 + sizeof(output) + sizeof(vectors), scratch, 4096);
  size_t sent = 0;
  while (sent < sizeof(response)) {
    ssize_t count = write(STDOUT_FILENO, response + sent, sizeof(response) - sent);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) _exit(22);
    sent += (size_t)count;
  }

  _exit(0);
}

int main(void) {
  unsigned char magic[8];
  if (fread(magic, 1, 8, stdin) != 8 || memcmp(magic, "NYXVEC01", 8) ||
      fread(input, sizeof(input), 1, stdin) != 1 ||
      fread(vectors, sizeof(vectors), 1, stdin) != 1 ||
      fread(initial, sizeof(initial), 1, stdin) != 1 || getchar() != EOF ||
      input[31] & ~UINT64_C(0xf0000000) || input[34] > UINT32_MAX)
    return 2;
  uint64_t base = input[33];
  if (base < UINT64_C(0x100000000) || base > UINT64_C(0x1000000000) || base % 4096) return 3;
  void* mapping = mmap((void*)(uintptr_t)(base - 4096), 12288, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
  if (mapping == MAP_FAILED || (uintptr_t)mapping != base - 4096) return 4;
  scratch = (unsigned char*)(uintptr_t)base;
  if (mprotect(scratch, 4096, PROT_READ | PROT_WRITE)) return 5;
  memcpy(scratch, initial, 4096);
  code = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (code == MAP_FAILED) return 6;
  uint32_t words[2] = {(uint32_t)input[34], UINT32_C(0xd4200000)};
  memcpy(code, words, sizeof(words));
  __builtin___clear_cache((char*)code, (char*)code + sizeof(words));
  if (mprotect(code, 4096, PROT_READ | PROT_EXEC)) return 7;
  stack_t stack = {.ss_sp = alternate, .ss_size = sizeof(alternate)};
  if (sigaltstack(&stack, NULL)) return 8;
  struct sigaction action = {.sa_sigaction = handle, .sa_flags = SA_SIGINFO | SA_ONSTACK};
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGTRAP, &action, NULL) || sigaction(SIGSEGV, &action, NULL) ||
      sigaction(SIGBUS, &action, NULL))
    return 9;
  raise(SIGTRAP);
  return 10;
}
