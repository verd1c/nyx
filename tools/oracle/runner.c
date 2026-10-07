#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern void nyx_execute(const uint64_t* input, uint64_t* output);

int main(void) {
  static const unsigned char magic[8] = {'N', 'Y', 'X', 'O', 'R', 'C', 'L', '1'};
  unsigned char header[8];
  uint64_t input[32], output[32];
  if (fread(header, 1, sizeof(header), stdin) != sizeof(header) ||
      memcmp(header, magic, sizeof(header)) != 0 || fread(input, sizeof(input), 1, stdin) != 1 ||
      getchar() != EOF || (input[31] & ~UINT64_C(0xf0000000)) != 0)
    return 2;
  nyx_execute(input, output);
  if (fwrite(magic, 1, sizeof(magic), stdout) != sizeof(magic) ||
      fwrite(output, sizeof(output), 1, stdout) != 1 || fflush(stdout) != 0)
    return 3;
  return 0;
}
