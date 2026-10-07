# Nyx Compiler Framework

Nyx is a toy project set of libraries for binary lifting and de-obfuscation. It lifts machine code into its own intermediate language (`NYXIL`), de-obfuscates it, simplifies it, and allows for re-compilation or analysis.

The deobfuscation mainly targets the obfuscation commonly found in protected Android libraries:

- mixed boolean-arithmetic (MBA) expressions
- encrypted calls
- opaque predicates
- control-flow flattening and dispatcher tables
- constants hidden behind loads from the image

See for example the following obfuscated code:

```asm
func:
    adrp    x9, basis 
    ldr     x9, [x9, :lo12:basis]
    movz    x11, #0x6f41   
    movk    x11, #0xd2b, lsl #16
    movk    x11, #0xc3e9, lsl #32
    movk    x11, #0x5a17, lsl #48
    eor     x9, x9, x11
    movz    x13, #0xf489                ; prime ^ mask
    movk    x13, #0xa54f, lsl #16
    movk    x13, #0xf272, lsl #32
    movk    x13, #0x3c6e, lsl #48
    movz    x14, #0xf53a                ; mask
    movk    x14, #0xa54f, lsl #16
    movk    x14, #0xf372, lsl #32
    movk    x14, #0x3c6e, lsl #48
    eor     x13, x13, x14

    ubfx    x10, x0, #0, #8             ; (h | b) - (h & b)
    orr     x11, x9, x10
    and     x12, x9, x10
    sub     x9, x11, x12
    mul     x9, x9, x13

    ubfx    x10, x0, #8, #8             ; (h + b) - 2(h & b)
    add     x11, x9, x10
    and     x12, x9, x10
    sub     x9, x11, x12, lsl #1
    mul     x9, x9, x13

    ubfx    x10, x0, #16, #8            ; (~h & b) | (h & ~b)
    bic     x11, x10, x9
    bic     x12, x9, x10
    orr     x9, x11, x12
    mul     x9, x9, x13

    ubfx    x10, x0, #24, #8            ; (h | b) & ~(h & b)
    orr     x11, x9, x10
    and     x12, x9, x10
    bic     x9, x11, x12
    mul     x9, x9, x13

    mov     x0, x9
    ret

basis:
    .quad   0x91e55f0d89094c64
```

Nyx is able to remove all four MBA identities, decrypt the offset basis, fold the prime and drop the scratch registers. What is left is a clear set of operations that show that the function clearly computes a FNV-1a hash:

```
L_4000d4:
    v56 = (0xcbf29ce484222325 ^ (x0 & 0xff)) * 0x100000001b3
    v68 = (v56 ^ ((x0 >> 8) & 0xff)) * 0x100000001b3
    v80 = (((x0 >> 0x10) & 0xff) ^ v68) * 0x100000001b3
    x0 = ((((x0 >> 0x18) & 0xff) ^ v80) * 0x100000001b3)
    return
```

Nyx is still totally a work in progress and its only proven to work on selected binaries

## Building

Requirements: CMake 3.28+, Ninja, and Clang 18+ or GCC 13+.

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
```

Which produces the binary at `build/nyx`.

## Usage

```sh
nyx inspect lib.so                                   # ELF segment summary as JSON
nyx lift lib.so --address 0x1000 --size 0x40         # lift a range to IL
nyx simplify lib.so --address 0x1000 --size 0x40     # simplify straight-line code
nyx cfg lib.so --address 0x1000 --size 0x200         # build a control-flow graph
nyx simplify-path lib.so --ranges 0x1000:0x20,0x1080:0x10
nyx recover-regions lib.so --address 0x1000 --size 0x800
```

## Tests

```sh
ctest --test-dir build --output-on-failure
```

Unit tests need GoogleTest and Python 3. The differential tests compare Nyx against real
execution and also need `aarch64-linux-gnu-gcc`, `aarch64-linux-gnu-objcopy`,
`aarch64-linux-gnu-objdump` and `qemu-aarch64`.

To build with sanitizers:

```sh
cmake -B out/asan -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DNYX_SANITIZE=address,undefined
cmake --build out/asan
ctest --test-dir out/asan --output-on-failure
```

## Layout

| path | contents |
| --- | --- |
| `include/nyx`, `lib` | the library: ELF loading, AArch64 decoding, IL, analyses and simplification passes |
| `tools/cli` | the `nyx` command |
| `tools/oracle` | QEMU-based reference execution used by the tests |
| `test` | unit and differential tests |
