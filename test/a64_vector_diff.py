#!/usr/bin/env python3
import collections
import json
import pathlib
import random
import struct
import subprocess
import sys
import tempfile

PAGE_SIZE = 4096
BASES = (0x100000000, 0x100200000)
MAGIC = b"NYXVEC01"

# Every SIMD&FP load/store size in every addressing form the decoder admits,
# with SP and register-offset bases, and the general/FP moves and MOVI/MVNI
# constants. Each runs from random Q registers so that a narrower write which
# failed to zero the rest of its Q register is visible.
FIXTURES = [
    ("unsigned_offset", "ldr b0, [x1, #3]"),
    ("unsigned_offset", "str b0, [x1, #3]"),
    ("unsigned_offset", "ldr h1, [x1, #6]"),
    ("unsigned_offset", "str h1, [x1, #6]"),
    ("unsigned_offset", "ldr s2, [x1, #12]"),
    ("unsigned_offset", "str s2, [x1, #12]"),
    ("unsigned_offset", "ldr d3, [x1, #24]"),
    ("unsigned_offset", "str d3, [x1, #24]"),
    ("unsigned_offset", "ldr q4, [x1, #32]"),
    ("unsigned_offset", "str q4, [x1, #32]"),
    ("unscaled", "ldur b5, [x1, #-1]"),
    ("unscaled", "stur h6, [x1, #-3]"),
    ("unscaled", "ldur s7, [x1, #-5]"),
    ("unscaled", "stur d8, [x1, #-7]"),
    ("unscaled", "ldur q9, [x1, #-9]"),
    ("unscaled", "stur q10, [x1, #-15]"),
    ("unscaled", "ldur d31, [x1, #255]"),
    ("unscaled", "stur s31, [x1, #-256]"),
    ("writeback", "ldr q11, [x1, #16]!"),
    ("writeback", "str q12, [x1], #-16"),
    ("writeback", "ldr d13, [x1], #8"),
    ("writeback", "str s14, [x1, #-4]!"),
    ("writeback", "ldr b15, [x1], #1"),
    ("writeback", "str h16, [x1, #2]!"),
    ("writeback", "ldr q1, [x1, #-256]!"),
    ("writeback", "str d1, [x1], #255"),
    ("sp_base", "str q17, [sp, #-16]!"),
    ("sp_base", "ldr d18, [sp], #16"),
    ("sp_base", "ldr q19, [sp, #32]"),
    ("sp_base", "stur s20, [sp, #-4]"),
    ("register_offset", "ldr q20, [x1, x2, lsl #4]"),
    ("register_offset", "str d21, [x1, x2]"),
    ("register_offset", "ldr s22, [x1, w3, sxtw #2]"),
    ("register_offset", "str b23, [x1, x2]"),
    ("register_offset", "ldr h24, [x1, w2, uxtw #1]"),
    ("register_offset", "str q25, [x1, x2]"),
    ("pair", "ldp s0, s1, [x1, #-8]"),
    ("pair", "stp s2, s3, [x1, #8]"),
    ("pair", "ldp d4, d5, [x1, #16]"),
    ("pair", "stp d6, d7, [x1, #-16]!"),
    ("pair", "ldp q8, q9, [x1], #32"),
    ("pair", "stp q10, q11, [x1, #-32]!"),
    ("pair", "ldp q30, q31, [x1, #1008]"),
    ("pair", "stp s30, s31, [x1, #252]"),
    ("pair", "stp q0, q0, [x1, #-1024]"),
    ("pair_sp", "stp q0, q1, [sp, #-32]!"),
    ("pair_sp", "ldp d8, d9, [sp], #16"),
    ("fmov_general", "fmov d0, x1"),
    ("fmov_general", "fmov x2, d3"),
    ("fmov_general", "fmov s4, w5"),
    ("fmov_general", "fmov w6, s7"),
    ("fmov_general", "fmov v8.d[1], x9"),
    ("fmov_general", "fmov x10, v11.d[1]"),
    ("fmov_general", "fmov d12, xzr"),
    ("fmov_general", "fmov s13, wzr"),
    ("fmov_general", "fmov v14.d[1], xzr"),
    ("fmov_general", "fmov x30, d31"),
    ("fmov_register", "fmov d0, d1"),
    ("fmov_register", "fmov s2, s3"),
    ("fmov_register", "fmov d4, d4"),
    ("fmov_register", "fmov s31, s30"),
    ("movi", "movi v0.2d, #0"),
    ("movi", "movi d1, #0"),
    ("movi", "movi v2.2d, #0xff00ff00ff00ff00"),
    ("movi", "movi d3, #0xffffffffffffffff"),
    ("movi", "movi v4.16b, #0xa5"),
    ("movi", "movi v5.8b, #0x5a"),
    ("movi", "movi v6.4s, #0x12, lsl #24"),
    ("movi", "movi v7.2s, #0x34, lsl #8"),
    ("movi", "movi v8.8h, #0x56, lsl #8"),
    ("movi", "movi v9.4h, #0x78"),
    ("movi", "movi v10.4s, #0x9a, msl #16"),
    ("movi", "movi v11.2s, #0xbc, msl #8"),
    ("movi", "movi v31.4s, #0xff, lsl #16"),
    ("mvni", "mvni v12.4s, #0x12, lsl #16"),
    ("mvni", "mvni v13.8h, #0x34"),
    ("mvni", "mvni v14.2s, #0x56, msl #8"),
    ("mvni", "mvni v15.4h, #0x78, lsl #8"),
    ("mvni", "mvni v16.4s, #0"),
]

LANE_FIXTURES = (
    [
        ("cmeq", f"cmeq v0.{shape}, v1.{shape}, v2.{shape}")
        for shape in ("8b", "16b", "4h", "8h", "2s", "4s", "2d")
    ]
    + [
        ("cmeq_alias", f"cmeq v{rd}.16b, v1.16b, v{rm}.16b")
        for rd, rm in ((1, 2), (2, 2), (1, 1), (31, 2))
    ]
    + [
        ("umov", f"umov {reg}0, v1.{lane}[{index}]")
        for reg, lane, count in (("w", "b", 16), ("w", "h", 8), ("w", "s", 4), ("x", "d", 2))
        for index in range(count)
    ]
    + [("umov_discard", "umov wzr, v31.b[15]"), ("umov_discard", "umov xzr, v31.d[1]")]
)

# Accesses placed across the end of the scratch page, which QEMU faults on.
# A Q access and each pair element are separate accesses in the selected
# backend, so a store's earlier part stays written; writeback never commits.
FAULTS = [
    ("ldr q0, [x1]", 8),
    ("ldr q0, [x1]", 4),
    ("str q0, [x1]", 8),
    ("str q0, [x1]", 12),
    ("str d0, [x1]", 4),
    ("ldr s0, [x1]", 2),
    ("str h0, [x1]", 1),
    ("stp q0, q1, [x1]", 16),
    ("stp q0, q1, [x1]", 24),
    ("stp d0, d1, [x1]", 8),
    ("stp s0, s1, [x1]", 4),
    ("str q0, [x1, #16]!", 16),
    ("ldr d0, [x1], #8", 4),
]

# QEMU writes a SIMD&FP pair's first register before loading its second, so a
# second-element fault leaves q0 loaded. The architecture makes the destinations
# UNKNOWN there, and Nyx, which commits registers only on completion, keeps q0's
# entry value. Every other effect of these faults must still agree.
PARTIAL_LOADS = [
    ("ldp q0, q1, [x1], #32", 16),
    ("ldp q0, q1, [x1]", 24),
    ("ldp d0, d1, [x1]", 8),
    ("ldp s0, s1, [x1, #-4]!", 0),
]

MUTATIONS = [
    ("ldr s0, [x1]", "ldr d0, [x1]"),
    ("str d0, [x1]", "str s0, [x1]"),
    ("ldr q0, [x1], #16", "ldr q0, [x1]"),
    ("ldp q0, q1, [x1]", "ldp q1, q0, [x1]"),
    ("fmov d0, x1", "fmov v0.d[1], x1"),
    ("fmov d0, d1", "fmov s0, s1"),
    ("movi v0.2d, #0xff00ff00ff00ff00", "movi d0, #0xff00ff00ff00ff00"),
    ("mvni v0.4s, #1", "movi v0.4s, #1"),
    ("movi v0.4s, #0x12, msl #8", "movi v0.4s, #0x12, lsl #8"),
]

LANE_MUTATIONS = [
    ("cmeq v0.16b, v1.16b, v2.16b", "cmeq v0.8b, v1.8b, v2.8b"),
    ("cmeq v0.16b, v1.16b, v2.16b", "cmeq v0.8h, v1.8h, v2.8h"),
    ("umov w0, v1.b[0]", "umov w0, v1.b[15]"),
    ("umov x0, v1.d[1]", "umov w0, v1.s[2]"),
]


def assemble(instructions, directory):
    path = pathlib.Path(directory)
    (path / "words.S").write_text(".text\n" + "\n".join(instructions) + "\n")
    subprocess.run(
        ["aarch64-linux-gnu-gcc", "-c", "-o", str(path / "words.o"), str(path / "words.S")],
        check=True,
        timeout=30,
    )
    subprocess.run(
        [
            "aarch64-linux-gnu-objcopy",
            "-O",
            "binary",
            "-j",
            ".text",
            str(path / "words.o"),
            str(path / "words.bin"),
        ],
        check=True,
        timeout=30,
    )
    data = (path / "words.bin").read_bytes()
    if len(data) != 4 * len(instructions):
        raise AssertionError("assembler did not produce one word per instruction")
    return {
        text: int.from_bytes(data[i * 4 : i * 4 + 4], "little")
        for i, text in enumerate(instructions)
    }


def payload(state, word):
    registers, nzcv, sp, base, vectors, memory = state
    return MAGIC + struct.pack("<35Q", *registers, nzcv, sp, base, word) + vectors + memory


def reference(runner, state, word):
    data = subprocess.run(
        ["qemu-aarch64", "-cpu", "max,sve=off,sme=off", runner],
        input=payload(state, word),
        capture_output=True,
        check=True,
        timeout=10,
    ).stdout
    if len(data) != 8 + 37 * 8 + 32 * 16 + PAGE_SIZE or data[:8] != MAGIC:
        raise AssertionError("invalid vector runner response")
    values = struct.unpack("<37Q", data[8 : 8 + 37 * 8])
    completed = values[34] == 0
    if values[34] not in (0, 7, 11) or values[33] != (4 if completed else 0):
        raise AssertionError("invalid vector runner outcome")
    return {
        "completed": completed,
        "registers": list(values[:31]),
        "nzcv": values[31],
        "sp": values[32],
        "vectors": data[8 + 37 * 8 : 8 + 37 * 8 + 512].hex(),
        "memory": data[8 + 37 * 8 + 512 :].hex(),
        "fault_address": None if completed else values[35],
    }


def evaluate(probe, state, word):
    result = subprocess.run(
        [probe], input=payload(state, word), capture_output=True, check=True, timeout=10
    )
    return json.loads(result.stdout)


def initial(base, flags):
    rng = random.Random(0xF9D0 + base + flags)
    registers = [rng.getrandbits(64) for _ in range(31)]
    registers[1] = base + 1024
    registers[2] = 16
    registers[3] = (rng.getrandbits(32) << 32) | 0xFFFFFFFC
    vectors = bytes(rng.getrandbits(8) for _ in range(32 * 16))
    memory = bytes(rng.getrandbits(8) for _ in range(PAGE_SIZE))
    return (tuple(registers), flags << 28, base + 2048, base, vectors, memory)


def main():
    probe = sys.argv[1]
    source = pathlib.Path(__file__).resolve().parents[1] / "tools/oracle/runner_vector.c"
    counts = collections.Counter()
    with tempfile.TemporaryDirectory(prefix="nyx-vector-") as directory:
        runner = str(pathlib.Path(directory) / "runner")
        subprocess.run(
            [
                "aarch64-linux-gnu-gcc",
                "-std=gnu11",
                "-O2",
                "-static",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(source),
                "-o",
                runner,
            ],
            check=True,
            timeout=30,
        )
        texts = [text for _, text in FIXTURES + LANE_FIXTURES] + [
            text for text, _ in FAULTS + PARTIAL_LOADS
        ]
        texts += [text for pair in MUTATIONS + LANE_MUTATIONS for text in pair]
        words = assemble(list(dict.fromkeys(texts)), directory)
        states = [initial(base, flags) for base in BASES for flags in (0, 9)]
        lane_states = []
        for state in states:
            for difference in (None, 0, 8):
                vectors = bytearray(state[4])
                vectors[32:48] = vectors[16:32]
                if difference is not None:
                    vectors[32 + difference] ^= 1
                lane_states.append((*state[:4], bytes(vectors), state[5]))
        for family, text in LANE_FIXTURES:
            for state in lane_states:
                expected = reference(runner, state, words[text])
                actual = evaluate(probe, state, words[text])
                if not expected["completed"] or actual != expected:
                    raise AssertionError(f"{family}: {text}\nNyx  {actual}\nQEMU {expected}")
                counts[family] += 1
        for original, mutant in LANE_MUTATIONS:
            if not any(
                evaluate(probe, state, words[mutant]) != reference(runner, state, words[original])
                for state in lane_states
            ):
                raise AssertionError(f"unrefuted lane mutation: {original} -> {mutant}")
            counts["refuted_lane_mutations"] += 1
        for family, text in FIXTURES:
            for state in states:
                expected = reference(runner, state, words[text])
                if not expected["completed"]:
                    raise AssertionError(f"fixture faulted under QEMU: {text}")
                actual = evaluate(probe, state, words[text])
                if actual != expected:
                    raise AssertionError(
                        f"{family}: {text}, base={state[3]:x}\nNyx  {actual}\nQEMU {expected}"
                    )
                counts[family] += 1
        for text, mapped in FAULTS:
            for state in states:
                registers = list(state[0])
                registers[1] = state[3] + PAGE_SIZE - mapped
                state = (tuple(registers), *state[1:])
                expected = reference(runner, state, words[text])
                if expected["completed"]:
                    raise AssertionError(f"fault fixture completed under QEMU: {text}")
                actual = evaluate(probe, state, words[text])
                if actual != expected:
                    raise AssertionError(
                        f"fault: {text}, mapped={mapped}\nNyx  {actual}\nQEMU {expected}"
                    )
                counts["page_fault"] += 1
        for text, mapped in PARTIAL_LOADS:
            for state in states:
                registers = list(state[0])
                registers[1] = state[3] + PAGE_SIZE - mapped
                state = (tuple(registers), *state[1:])
                expected = reference(runner, state, words[text])
                entry_q0 = state[4][:16].hex()
                if expected["completed"] or expected["vectors"][:32] == entry_q0:
                    raise AssertionError(
                        f"QEMU no longer loads the first element before faulting: {text}"
                    )
                expected["vectors"] = entry_q0 + expected["vectors"][32:]
                actual = evaluate(probe, state, words[text])
                if actual != expected:
                    raise AssertionError(
                        f"partial load: {text}, mapped={mapped}\nNyx  {actual}\nQEMU {expected}"
                    )
                counts["pair_load_fault"] += 1
        for original, mutant in MUTATIONS:
            state = states[0]
            if evaluate(probe, state, words[mutant]) == reference(runner, state, words[original]):
                raise AssertionError(f"unrefuted mutation: {original} -> {mutant}")
            counts["refuted_mutations"] += 1
    controls = sum(n for family, n in counts.items() if family.startswith("refuted_"))
    comparisons = sum(n for family, n in counts.items() if not family.startswith("refuted_"))
    print(
        f"SIMD&FP registers: {comparisons} independent QEMU/Nyx comparisons; {controls} refuted mutations"
    )
    print("Family comparable counts:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
