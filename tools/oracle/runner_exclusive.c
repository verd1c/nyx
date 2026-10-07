#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <sys/mman.h>

static sigjmp_buf fault_return;

static void on_fault(int signal_number) {
  (void)signal_number;
  siglongjmp(fault_return, 1);
}

static void run32(unsigned mode) {
  uint32_t memory[2] __attribute__((aligned(8))) = {42, 17};
  uint32_t loaded, status, again;
  uint32_t intervening = mode == 1 ? 77 : 42;
  uint32_t* target = mode == 4 ? memory + 1 : memory;
  uint32_t value = 55;
  __asm__ volatile(
      "ldaxr %w[loaded], [%[base]]\n\t"
      "cmp %w[mode], #1\n\tb.lo 1f\n\tcmp %w[mode], #2\n\tb.hi 1f\n\t"
      "str %w[intervening], [%[base]]\n1:\n\t"
      "cmp %w[mode], #3\n\tb.ne 2f\n\tclrex\n2:\n\t"
      "stlxr %w[status], %w[value], [%[target]]\n\t"
      "stlxr %w[again], %w[value], [%[target]]"
      : [loaded] "=&r"(loaded), [status] "=&r"(status), [again] "=&r"(again)
      : [base] "r"(memory), [target] "r"(target), [mode] "r"(mode), [intervening] "r"(intervening),
        [value] "r"(value)
      : "cc", "memory");
  printf("w%u %u %u %u %u %u\n", mode, loaded, status, again, memory[0], memory[1]);
}

static void run64(unsigned mode) {
  uint64_t memory[2] __attribute__((aligned(8))) = {UINT64_C(0x1122334455667788), 17};
  uint64_t loaded;
  uint32_t status, again;
  uint64_t intervening = mode == 1 ? UINT64_C(0xfedcba9876543210) : memory[0];
  uint64_t* target = mode == 4 ? memory + 1 : memory;
  uint64_t value = UINT64_C(0x8877665544332211);
  __asm__ volatile(
      "ldaxr %[loaded], [%[base]]\n\t"
      "cmp %w[mode], #1\n\tb.lo 1f\n\tcmp %w[mode], #2\n\tb.hi 1f\n\t"
      "str %[intervening], [%[base]]\n1:\n\t"
      "cmp %w[mode], #3\n\tb.ne 2f\n\tclrex\n2:\n\t"
      "stlxr %w[status], %[value], [%[target]]\n\t"
      "stlxr %w[again], %[value], [%[target]]"
      : [loaded] "=&r"(loaded), [status] "=&r"(status), [again] "=&r"(again)
      : [base] "r"(memory), [target] "r"(target), [mode] "r"(mode), [intervening] "r"(intervening),
        [value] "r"(value)
      : "cc", "memory");
  printf("x%u %llu %u %u %llu %llu\n", mode, (unsigned long long)loaded, status, again,
         (unsigned long long)memory[0], (unsigned long long)memory[1]);
}

int main(void) {
  for (unsigned mode = 0; mode < 5; ++mode) run32(mode);
  for (unsigned mode = 0; mode < 5; ++mode) run64(mode);
  struct sigaction action = {.sa_handler = on_fault};
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGSEGV, &action, NULL) || sigaction(SIGBUS, &action, NULL)) return 1;
  const long page = sysconf(_SC_PAGESIZE);
  if (page <= 0) return 2;
  uint32_t* memory =
      mmap(NULL, (size_t)page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (memory == MAP_FAILED) return 3;
  *memory = 42;
  if (mprotect(memory, (size_t)page, PROT_READ)) return 4;
  uint32_t loaded = 0, status = 7;
  if (sigsetjmp(fault_return, 1) == 0) {
    __asm__ volatile("ldaxr %w[out], [%[base]]"
                     : [out] "=r"(loaded)
                     : [base] "r"((uint32_t*)0)
                     : "memory");
    puts("f0 unexpected");
  } else {
    puts("f0 fault");
  }

  if (sigsetjmp(fault_return, 1) == 0) {
    __asm__ volatile(
        "ldaxr %w[out], [%[base]]\n\t"
        "stlxr %w[status], %w[value], [%[base]]"
        : [out] "=&r"(loaded), [status] "=&r"(status)
        : [base] "r"(memory), [value] "r"(55U)
        : "memory");
    printf("f1 status %u\n", status);
  } else {
    puts("f1 fault");
  }

  if (sigsetjmp(fault_return, 1) == 0) {
    __asm__ volatile(
        "ldaxr %w[out], [%[base]]\n\t"
        "stlxr %w[status], %w[value], [%[other]]"
        : [out] "=&r"(loaded), [status] "=&r"(status)
        : [base] "r"(memory), [other] "r"((uint32_t*)0), [value] "r"(55U)
        : "memory");
    printf("f2 %u\n", status);
  } else {
    puts("f2 fault");
  }

  if (munmap(memory, (size_t)page)) return 5;
  return 0;
}
