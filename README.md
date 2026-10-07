# Nyx

Nyx is a toy project set of libraries for binary lifting and de-obfuscation. It lifts machine code into its own intermediate language (`NYXIL`), de-obfuscates it, simplifies it, and allows for re-compilation or analysis.

The deobfuscation mainly targets the obfuscation commonly found in protected Android libraries:

- mixed boolean-arithmetic (MBA) expressions
- encrypted calls
- opaque predicates
- control-flow flattening and dispatcher tables
- constants hidden behind loads from the image

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
