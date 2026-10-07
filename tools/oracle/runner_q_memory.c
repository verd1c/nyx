#define _GNU_SOURCE
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>

#include <asm/sigcontext.h>
#include <sys/mman.h>

static sigjmp_buf fault_return;
static unsigned char observed_q[16];
static volatile sig_atomic_t fault_index;
static volatile sig_atomic_t found_fpsimd;
static unsigned char* target;
static int q_index;

static void on_fault(int signal, siginfo_t* info, void* context) {
  (void)signal;
  ucontext_t* uc = context;
  struct _aarch64_ctx* record = (void*)uc->uc_mcontext.__reserved;
  size_t remaining = sizeof(uc->uc_mcontext.__reserved);
  while (remaining >= sizeof(*record) && record->size >= sizeof(*record) &&
         record->size <= remaining) {
    if (record->magic == FPSIMD_MAGIC && record->size >= sizeof(struct fpsimd_context)) {
      struct fpsimd_context* fpsimd = (void*)record;
      memcpy(observed_q, &fpsimd->vregs[q_index], sizeof(observed_q));
      found_fpsimd = 1;
      break;
    }

    remaining -= record->size;
    record = (void*)((unsigned char*)record + record->size);
  }

  fault_index = (uintptr_t)info->si_addr - (uintptr_t)target;
  siglongjmp(fault_return, 1);
}

static void print_hex(const unsigned char* bytes, size_t size) {
  for (size_t i = 0; i < size; ++i) printf("%02x", bytes[i]);
}

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  int load = strcmp(argv[1], "load") == 0;
  if (!load && strcmp(argv[1], "store") != 0) return 2;
  char* end;
  unsigned long mapped = strtoul(argv[2], &end, 10);
  if (*end || mapped < 1 || mapped > 16) return 2;
  int max = strcmp(argv[3], "max") == 0;
  q_index = strcmp(argv[3], "q31") == 0 ? 31 : 0;
  if (!max && q_index != 31 && strcmp(argv[3], "base") != 0) return 2;
  long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) return 3;
  unsigned char* pages =
      mmap(NULL, (size_t)page_size * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (pages == MAP_FAILED || mprotect(pages + page_size, page_size, PROT_NONE)) return 3;
  target = pages + page_size - mapped;
  unsigned char seed[16];
  for (unsigned i = 0; i < 16; ++i) seed[i] = 0xa0 + i;
  if (load) {
    for (unsigned i = 0; i < mapped; ++i) target[i] = i;
  }

  memcpy(observed_q, seed, sizeof(seed));
  struct sigaction action = {.sa_sigaction = on_fault, .sa_flags = SA_SIGINFO};
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGSEGV, &action, NULL)) return 3;
  unsigned char* base = max ? (unsigned char*)((uintptr_t)target - 65520) : target;
  int completed = sigsetjmp(fault_return, 1) == 0;
  if (completed) {
    if (load && max) {
      __asm__ volatile("ldr q0, [%1]\n\tldr q0, [%2, #65520]\n\tstr q0, [%0]"
                       :
                       : "r"(observed_q), "r"(seed), "r"(base)
                       : "v0", "memory");
    } else if (load && q_index == 31) {
      __asm__ volatile("ldr q31, [%1]\n\tldr q31, [%2]\n\tstr q31, [%0]"
                       :
                       : "r"(observed_q), "r"(seed), "r"(base)
                       : "v31", "memory");
    } else if (load) {
      __asm__ volatile("ldr q0, [%1]\n\tldr q0, [%2]\n\tstr q0, [%0]"
                       :
                       : "r"(observed_q), "r"(seed), "r"(base)
                       : "v0", "memory");
    } else if (max) {
      __asm__ volatile("ldr q0, [%0]\n\tstr q0, [%1, #65520]\n\tstr q0, [%2]"
                       :
                       : "r"(seed), "r"(base), "r"(observed_q)
                       : "v0", "memory");
    } else if (q_index == 31) {
      __asm__ volatile("ldr q31, [%0]\n\tstr q31, [%1]\n\tstr q31, [%2]"
                       :
                       : "r"(seed), "r"(base), "r"(observed_q)
                       : "v31", "memory");
    } else {
      __asm__ volatile("ldr q0, [%0]\n\tstr q0, [%1]\n\tstr q0, [%2]"
                       :
                       : "r"(seed), "r"(base), "r"(observed_q)
                       : "v0", "memory");
    }
  }

  if (!completed && !found_fpsimd) return 4;
  printf("{\"completed\":%s,\"q\":\"", completed ? "true" : "false");
  print_hex(observed_q, sizeof(observed_q));
  printf("\",\"memory\":\"");
  print_hex(target, mapped);
  if (completed)
    printf("\",\"fault_index\":null}\n");
  else
    printf("\",\"fault_index\":%d}\n", fault_index);
  return 0;
}
