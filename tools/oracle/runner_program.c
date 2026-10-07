#define _GNU_SOURCE
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>

#include <sys/mman.h>

static uint64_t input[37];
static unsigned char original[4096], initial[4096];
static uint64_t global_addresses[8];
static unsigned char global_inputs[8][4096];
static unsigned char response[8 + 39 * 8 + 4096 + 8 * (8 + 4096)];
static unsigned char alternate[65536] __attribute__((aligned(16)));
static unsigned char* scratch;
static volatile sig_atomic_t started;

static void handle(int signal, siginfo_t* info, void* raw) {
  ucontext_t* context = raw;
  if (!started) {
    if (signal != SIGTRAP || info->si_code != SI_TKILL) _exit(20);
    started = 1;
    memcpy(context->uc_mcontext.regs, input, 31 * 8);
    context->uc_mcontext.pstate = (context->uc_mcontext.pstate & ~UINT64_C(0xf0000000)) | input[31];
    context->uc_mcontext.sp = input[32];
    context->uc_mcontext.pc = input[34];
    return;
  }

  uint64_t output[39];
  memcpy(output, context->uc_mcontext.regs, 31 * 8);
  output[31] = context->uc_mcontext.pstate & UINT64_C(0xf0000000);
  output[32] = context->uc_mcontext.sp;
  output[33] = input[33];
  output[34] = input[34];
  output[35] = context->uc_mcontext.pc;
  output[38] = input[36];
  if (signal == SIGTRAP && info->si_code == TRAP_BRKPT && output[35] == input[34] + input[35]) {
    output[36] = 0;
    output[37] = 0;
  } else if ((signal == SIGSEGV || signal == SIGBUS) && output[35] >= input[34] &&
             output[35] < input[34] + input[35] && output[35] % 4 == 0) {
    output[36] = signal;
    output[37] = (uintptr_t)info->si_addr;
  } else {
    _exit(21);
  }

  memcpy(response, "NYXPRG02", 8);
  memcpy(response + 8, output, sizeof(output));
  memcpy(response + 8 + sizeof(output), scratch, 4096);
  size_t response_size = 8 + sizeof(output) + 4096;
  for (unsigned i = 0; i < input[36]; ++i) {
    memcpy(response + response_size, &global_addresses[i], 8);
    memcpy(response + response_size + 8, (const void*)(uintptr_t)global_addresses[i], 4096);
    response_size += 8 + 4096;
  }

  size_t sent = 0;
  while (sent < response_size) {
    ssize_t count = write(STDOUT_FILENO, response + sent, response_size - sent);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) _exit(22);
    sent += (size_t)count;
  }

  _exit(0);
}

int main(void) {
  // Only program.py's explicitly trusted, independently admitted fixtures use
  // this runner. This executable is not an arbitrary-code security sandbox.
  unsigned char magic[8];
  if (fread(magic, 1, 8, stdin) != 8 || memcmp(magic, "NYXPRG02", 8) ||
      fread(input, sizeof(input), 1, stdin) != 1 || (input[31] & ~UINT64_C(0xf0000000)) ||
      input[35] == 0 || input[35] > sizeof(original) || input[35] % 4 ||
      input[33] < UINT64_C(0x100000000) || input[33] >= UINT64_C(0x200000000) || input[33] % 4096 ||
      input[34] < UINT64_C(0x200000000) || input[34] > UINT64_C(0x1000000000) || input[34] % 4 ||
      input[36] > 8 || input[28] != input[33])
    return 2;
  if (fread(original, 1, input[35], stdin) != input[35] ||
      fread(initial, 1, sizeof(initial), stdin) != sizeof(initial))
    return 2;
  uint64_t code_page = input[34] & ~UINT64_C(4095);
  uint64_t starts[10] = {input[33] - 4096, code_page - 4096};
  uint64_t ends[10] = {input[33] + 8192, code_page + 12288};
  unsigned order[8];
  for (unsigned i = 0; i < input[36]; ++i) {
    if (fread(&global_addresses[i], 8, 1, stdin) != 1 ||
        fread(global_inputs[i], 1, 4096, stdin) != 4096 || global_addresses[i] % 4096 ||
        global_addresses[i] < UINT64_C(0x100000000) || global_addresses[i] > UINT64_C(0x1000000000))
      return 2;
    order[i] = i;
  }

  if (getchar() != EOF) return 2;

  // Sort indices so clustering cannot change input/output page identity order.
  for (unsigned i = 1; i < input[36]; ++i) {
    unsigned item = order[i], j = i;
    while (j && global_addresses[order[j - 1]] > global_addresses[item]) {
      order[j] = order[j - 1];
      --j;
    }

    order[j] = item;
  }

  unsigned arena_count = 2;
  for (unsigned i = 0; i < input[36]; ++i) {
    uint64_t address = global_addresses[order[i]];
    if (i && address == global_addresses[order[i - 1]]) return 2;
    if (arena_count > 2 && address == ends[arena_count - 1] - 4096) {
      ends[arena_count - 1] += 4096;
    } else {
      starts[arena_count] = address - 4096;
      ends[arena_count++] = address + 8192;
    }
  }

  for (unsigned i = 0; i < arena_count; ++i) {
    for (unsigned j = 0; j < i; ++j) {
      if (starts[i] < ends[j] && starts[j] < ends[i]) return 2;
    }
  }

  void* data_mapping = mmap((void*)(uintptr_t)(input[33] - 4096), 3 * 4096, PROT_NONE,
                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
  if (data_mapping == MAP_FAILED || (uintptr_t)data_mapping != input[33] - 4096) return 3;
  scratch = (unsigned char*)(uintptr_t)input[33];
  if (mprotect(scratch, 4096, PROT_READ | PROT_WRITE)) return 4;
  memcpy(scratch, initial, 4096);
  size_t code_size = (input[34] - code_page + input[35] + 4 + 4095) & ~(size_t)4095;
  void* code_mapping = mmap((void*)(uintptr_t)(code_page - 4096), code_size + 8192, PROT_NONE,
                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
  if (code_mapping == MAP_FAILED || (uintptr_t)code_mapping != code_page - 4096) return 5;
  unsigned char* code = (unsigned char*)(uintptr_t)input[34];
  if (mprotect((void*)(uintptr_t)code_page, code_size, PROT_READ | PROT_WRITE)) return 6;
  memcpy(code, original, input[35]);
  uint32_t sentinel = UINT32_C(0xd4200000);
  memcpy(code + input[35], &sentinel, sizeof(sentinel));
  __builtin___clear_cache((char*)code, (char*)code + input[35] + 4);
  if (mprotect((void*)(uintptr_t)code_page, code_size, PROT_READ | PROT_EXEC)) return 7;
  for (unsigned i = 2; i < arena_count; ++i) {
    void* mapping = mmap((void*)(uintptr_t)starts[i], ends[i] - starts[i], PROT_NONE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (mapping == MAP_FAILED || (uintptr_t)mapping != starts[i]) return 11;
    if (mprotect((void*)(uintptr_t)(starts[i] + 4096), ends[i] - starts[i] - 8192,
                 PROT_READ | PROT_WRITE))
      return 12;
  }

  for (unsigned i = 0; i < input[36]; ++i) {
    memcpy((void*)(uintptr_t)global_addresses[i], global_inputs[i], 4096);
  }

  for (unsigned i = 2; i < arena_count; ++i) {
    if (mprotect((void*)(uintptr_t)(starts[i] + 4096), ends[i] - starts[i] - 8192, PROT_READ))
      return 13;
  }

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
