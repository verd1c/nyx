#define _GNU_SOURCE
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>

#include <sys/mman.h>

static uint64_t input[35];
static unsigned char response[8 + 35 * 8];
static unsigned char alternate[65536] __attribute__((aligned(16)));
static volatile sig_atomic_t started;

static uint64_t sign_extend(uint64_t value, unsigned bits) {
  uint64_t sign = UINT64_C(1) << (bits - 1);
  return (value ^ sign) - sign;
}

static int landing(uint64_t target) {
  return target >= input[33] - 4096 && target < input[33] + 8192 && target != input[33] &&
         target % 4 == 0;
}

static int admitted(void) {
  uint32_t word = (uint32_t)input[34];
  uint64_t target;
  if ((word & UINT32_C(0x7c000000)) == UINT32_C(0x14000000)) {
    target = input[33] + (sign_extend(word & UINT32_C(0x3ffffff), 26) << 2);
  } else if ((word & UINT32_C(0xff000010)) == UINT32_C(0x54000000) ||
             (word & UINT32_C(0x7e000000)) == UINT32_C(0x34000000)) {
    target = input[33] + (sign_extend((word >> 5) & UINT32_C(0x7ffff), 19) << 2);
  } else if ((word & UINT32_C(0x7e000000)) == UINT32_C(0x36000000)) {
    target = input[33] + (sign_extend((word >> 5) & UINT32_C(0x3fff), 14) << 2);
  } else {
    uint32_t family = word & UINT32_C(0xfffffc1f);
    unsigned reg = (word >> 5) & 31;
    if ((family != UINT32_C(0xd61f0000) && family != UINT32_C(0xd63f0000) &&
         family != UINT32_C(0xd65f0000)) ||
        reg == 31)
      return 0;
    target = input[reg];
  }

  return landing(target);
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

  uint64_t destination = context->uc_mcontext.pc;
  if (signal != SIGTRAP || info->si_code != TRAP_BRKPT || !landing(destination) ||
      *(const uint32_t*)(uintptr_t)destination != UINT32_C(0xd4200000))
    _exit(21);
  uint64_t output[35];
  memcpy(output, context->uc_mcontext.regs, 31 * 8);
  output[31] = context->uc_mcontext.pstate & UINT64_C(0xf0000000);
  output[32] = context->uc_mcontext.sp;
  output[33] = input[33];
  output[34] = destination;
  memcpy(response, "NYXCTL01", 8);
  memcpy(response + 8, output, sizeof(output));
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
  if (fread(magic, 1, 8, stdin) != 8 || memcmp(magic, "NYXCTL01", 8) ||
      fread(input, sizeof(input), 1, stdin) != 1 || getchar() != EOF ||
      (input[31] & ~UINT64_C(0xf0000000)) || input[34] > UINT32_MAX ||
      input[33] < UINT64_C(0x200000000) || input[33] > UINT64_C(0x1000000000) || input[33] % 4096 ||
      !admitted())
    return 2;
  void* mapping = mmap((void*)(uintptr_t)(input[33] - 8192), 5 * 4096, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
  if (mapping == MAP_FAILED || (uintptr_t)mapping != input[33] - 8192) return 3;
  uint32_t* code = (uint32_t*)(uintptr_t)(input[33] - 4096);
  if (mprotect(code, 3 * 4096, PROT_READ | PROT_WRITE)) return 4;
  for (unsigned i = 0; i < 3 * 4096 / 4; ++i) code[i] = UINT32_C(0xd4200000);
  code[4096 / 4] = (uint32_t)input[34];
  __builtin___clear_cache((char*)code, (char*)code + 3 * 4096);
  if (mprotect(code, 3 * 4096, PROT_READ | PROT_EXEC)) return 5;
  stack_t stack = {.ss_sp = alternate, .ss_size = sizeof(alternate)};
  if (sigaltstack(&stack, NULL)) return 6;
  struct sigaction action = {.sa_sigaction = handle, .sa_flags = SA_SIGINFO | SA_ONSTACK};
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGTRAP, &action, NULL)) return 7;
  raise(SIGTRAP);
  return 8;
}
