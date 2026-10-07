#!/usr/bin/env python3
import collections
import pathlib
import random
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.oracle.aarch64 import Oracle, State


def evaluate(probe, code, states):
    payload = "".join(" ".join(f"{x:x}" for x in (*s.registers, s.nzcv)) + "\n" for s in states)
    result = subprocess.run(
        [probe, code.hex()], input=payload, text=True, capture_output=True, timeout=20, check=True
    )
    rows = result.stdout.splitlines()
    if len(rows) != len(states):
        raise AssertionError("semantic probe lost states")
    output = []
    for row in rows:
        values = tuple(int(x, 16) for x in row.split())
        if len(values) != 32:
            raise AssertionError("invalid semantic probe response")
        output.append(State(values[:31], values[31]))
    return output


def main():
    probe = sys.argv[1]
    oracle = Oracle()
    rng = random.Random(0xD019)
    states = []
    for flags in range(16):
        regs = tuple(rng.getrandbits(64) for _ in range(31))
        states.append(State(regs, flags << 28))
    for value in (
        0,
        1,
        (1 << 64) - 1,
        1 << 63,
        (1 << 63) - 1,
        (1 << 32) - 1,
        1 << 31,
        (1 << 31) - 1,
    ):
        regs = [value] * 31
        regs[1] = 1
        regs[2] = (1 << 64) - 1
        states.append(State(tuple(regs), 0))
    snippets = []
    for prefix, width in (("x", 64), ("w", 32)):
        for op in ("add", "adds", "sub", "subs"):
            snippets.append(
                ("arithmetic_immediate", [f"{op} {prefix}0, {prefix}1, #4095, lsl #12"])
            )
            for shift in ("lsl", "lsr", "asr"):
                snippets.append(
                    (
                        "arithmetic_shifted",
                        [f"{op} {prefix}0, {prefix}1, {prefix}2, {shift} #{width - 1}"],
                    )
                )
        # Every extend option at both ends of the shift range; a byte or halfword
        # source is read from a W register even in the 64-bit form.
        for op in ("add", "adds", "sub", "subs"):
            for extend in ("uxtb", "uxth", "uxtw", "uxtx", "sxtb", "sxth", "sxtw", "sxtx"):
                source = "x" if width == 64 and extend.endswith("x") else "w"
                for amount in (0, 4):
                    snippets.append(
                        (
                            "arithmetic_extended",
                            [f"{op} {prefix}0, {prefix}1, {source}2, {extend} #{amount}"],
                        )
                    )
        # Add/subtract with carry. The carry-in means the carry-out is not the
        # carry of a single addition, and the flag-setting forms are the only
        # instructions whose result and whose flags both depend on entry NZCV,
        # so every one of the sixteen entry flag states matters here.
        for op in ("adc", "adcs", "sbc", "sbcs"):
            snippets.append(("carry_arithmetic", [f"{op} {prefix}0, {prefix}1, {prefix}2"]))
            snippets.append(("carry_arithmetic", [f"{op} {prefix}0, {prefix}1, {prefix}0"]))
            snippets.append(("carry_arithmetic", [f"{op} {prefix}0, {prefix}zr, {prefix}2"]))
        for op in ("and", "ands", "orr", "orn", "eor", "eon", "bic", "bics"):
            snippets.append(
                ("logical_shifted", [f"{op} {prefix}30, {prefix}1, {prefix}2, ror #{width - 1}"])
            )
        for shift in ("lsl", "lsr", "asr"):
            for count in (0, 1, width - 1):
                snippets.append(("bitfield_shift", [f"{shift} {prefix}0, {prefix}1, #{count}"]))
        snippets.append(
            ("move_wide", [f"movn {prefix}0, #0x1234", f"movk {prefix}0, #0x5678, lsl #16"])
        )
        # A shifted MOVN must invert the whole shifted value, not the immediate
        # before shifting; with hw == 0 the two agree and nothing distinguishes them.
        for position in range(0, width, 16):
            snippets.append(
                (
                    "move_wide_shifted",
                    [
                        f"movn {prefix}0, #0x1234, lsl #{position}",
                        f"movz {prefix}1, #0xabcd, lsl #{position}",
                    ],
                )
            )
        snippets.append(
            (
                "multiply",
                [
                    f"madd {prefix}0, {prefix}1, {prefix}2, {prefix}3",
                    f"msub {prefix}4, {prefix}0, {prefix}1, {prefix}2",
                ],
            )
        )
        # Ra above the low four bits, and the zero-register accumulator that MUL
        # is an alias for, both distinguish a truncated accumulator field.
        snippets.append(
            (
                "multiply_accumulator",
                [
                    f"madd {prefix}0, {prefix}1, {prefix}2, {prefix}20",
                    f"msub {prefix}3, {prefix}1, {prefix}2, {prefix}30",
                    f"mul {prefix}4, {prefix}1, {prefix}2",
                    f"mneg {prefix}5, {prefix}1, {prefix}2",
                ],
            )
        )
        for condition in (
            "eq",
            "ne",
            "cs",
            "cc",
            "mi",
            "pl",
            "vs",
            "vc",
            "hi",
            "ls",
            "ge",
            "lt",
            "gt",
            "le",
            "al",
            "nv",
        ):
            snippets.append(
                (
                    "conditional_select",
                    [
                        f"csel {prefix}0, {prefix}1, {prefix}2, {condition}",
                        f"csinc {prefix}3, {prefix}1, {prefix}2, {condition}",
                        f"csinv {prefix}4, {prefix}1, {prefix}2, {condition}",
                        f"csneg {prefix}5, {prefix}1, {prefix}2, {condition}",
                    ],
                )
            )
        # Each condition against every entry NZCV, with an immediate NZCV that
        # differs per condition, so a flag taken from the wrong side shows. The
        # later compares in a sequence test the condition on the flags the
        # earlier ones left; CSET keeps every intermediate NZCV observable.
        for index, condition in enumerate(
            (
                "eq",
                "ne",
                "cs",
                "cc",
                "mi",
                "pl",
                "vs",
                "vc",
                "hi",
                "ls",
                "ge",
                "lt",
                "gt",
                "le",
                "al",
                "nv",
            )
        ):
            nzcv = (index * 7 + 3) % 16
            sequence = []
            for step, compare in enumerate(
                (
                    f"ccmp {prefix}1, {prefix}2, #{nzcv}, {condition}",
                    f"ccmn {prefix}1, #31, #{15 - nzcv}, {condition}",
                    f"ccmn {prefix}1, {prefix}2, #{15 - nzcv}, {condition}",
                    f"ccmp {prefix}2, #0, #{nzcv}, {condition}",
                )
            ):
                sequence.append(compare)
                sequence += [
                    f"cset x{10 + step * 4 + i}, {flag}"
                    for i, flag in enumerate(("mi", "eq", "cs", "vs"))
                ]
            snippets.append(("conditional_compare", sequence))
        snippets.append(
            (
                "conditional_compare",
                [
                    f"ccmp {prefix}zr, {prefix}1, #0, ne",
                    f"csel {prefix}0, {prefix}2, {prefix}3, lt",
                    f"ccmn {prefix}1, {prefix}zr, #15, hi",
                    f"ccmp {prefix}2, {prefix}2, #6, ge",
                ],
            )
        )
        # The edge states make x3 and x4 equal, so they divide zero by zero and
        # the most negative value by itself, and x2 is -1 for the signed overflow.
        # The shifted divisor keeps random quotients away from only 0 and 1.
        snippets.append(
            (
                "divide",
                [
                    f"udiv {prefix}0, {prefix}3, {prefix}4",
                    f"sdiv {prefix}5, {prefix}3, {prefix}4",
                    f"sdiv {prefix}6, {prefix}3, {prefix}2",
                    f"udiv {prefix}7, {prefix}3, {prefix}2",
                    f"sdiv {prefix}8, {prefix}3, {prefix}zr",
                    f"udiv {prefix}9, {prefix}3, {prefix}zr",
                    f"lsr {prefix}10, {prefix}4, #{width // 2 + 3}",
                    f"sdiv {prefix}11, {prefix}3, {prefix}10",
                    f"udiv {prefix}12, {prefix}3, {prefix}10",
                    f"asr {prefix}13, {prefix}4, #{width // 2 + 3}",
                    f"sdiv {prefix}14, {prefix}3, {prefix}13",
                ],
            )
        )
        # Shifted operands give CLZ and CLS long runs to count, not just the top bit.
        snippets.append(
            (
                "count_bits",
                [
                    f"clz {prefix}0, {prefix}1",
                    f"cls {prefix}2, {prefix}1",
                    f"clz {prefix}3, {prefix}4",
                    f"cls {prefix}5, {prefix}4",
                    f"clz {prefix}6, {prefix}zr",
                    f"cls {prefix}7, {prefix}zr",
                    f"lsr {prefix}8, {prefix}4, #{width // 2 + 3}",
                    f"clz {prefix}9, {prefix}8",
                    f"asr {prefix}10, {prefix}4, #{width // 2 + 3}",
                    f"cls {prefix}11, {prefix}10",
                    f"rbit {prefix}12, {prefix}4",
                    f"rbit {prefix}13, {prefix}1",
                ],
            )
        )
        reversals = [
            f"rev {prefix}0, {prefix}4",
            f"rev16 {prefix}1, {prefix}4",
            f"rev {prefix}2, {prefix}1",
        ]
        if width == 64:
            reversals.append("rev32 x3, x4")
        snippets.append(("byte_reverse", reversals))
    for flags in range(16):
        snippets.append(
            (
                "nzcv_transfer",
                [
                    "mov x1, #-1",
                    f"movk x1, #{(flags << 12) | 0xFFF}, lsl #16",
                    "msr nzcv, x1",
                    "mrs x30, nzcv",
                    "adc x2, xzr, xzr",
                ],
            )
        )
    snippets += [
        ("nzcv_transfer", ["mrs x0, nzcv", "mrs x30, nzcv", "mrs xzr, nzcv"]),
        (
            "nzcv_transfer",
            ["msr nzcv, x1", "mrs x2, nzcv", "adc x3, xzr, xzr", "msr nzcv, xzr", "mrs x4, nzcv"],
        ),
        (
            "nzcv_transfer",
            [
                "mrs x30, nzcv",
                "adds x0, x1, x2",
                "msr nzcv, x30",
                "cset x3, mi",
                "cset x4, eq",
                "cset x5, cs",
                "cset x6, vs",
            ],
        ),
        # The high half of the 128-bit product, with operands of every sign and
        # the -1 * -1 whose low half alone looks the same either way.
        (
            "multiply_high",
            [
                "umulh x0, x1, x2",
                "smulh x3, x1, x2",
                "umulh x4, x5, x6",
                "smulh x7, x5, x6",
                "umulh x8, x2, x2",
                "smulh x9, x2, x2",
                "smulh x10, x5, x5",
                "umulh x11, x5, xzr",
            ],
        ),
        (
            "flags_sequence",
            ["adds x0, x1, x2", "csel x3, x4, x5, vs", "subs x6, x0, x2", "csel x7, x8, x9, hi"],
        ),
        ("zero_register", ["adds xzr, x1, x2", "sub x0, xzr, x3", "orr w1, wzr, w2"]),
        ("identity", ["nop"]),
        # Widening multiplies, with a high accumulator register and the
        # zero-register accumulator their SMULL/UMULL aliases use.
        (
            "multiply_widening",
            [
                "smaddl x0, w1, w2, x20",
                "smsubl x3, w1, w2, x4",
                "umaddl x5, w1, w2, x6",
                "umsubl x7, w1, w2, x30",
                "smull x8, w1, w2",
                "umull x9, w1, w2",
            ],
        ),
        # A real-sample instance whose operands straddle the sign bit.
        ("multiply_widening", ["smaddl x9, w9, w10, x0"]),
        # A sign-extended halfword comparison from a real dispatcher case body.
        ("arithmetic_extended", ["cmp w8, w9, sxth", "csel w0, w10, w11, eq"]),
    ]
    counts = collections.Counter()
    for family, instructions in snippets:
        code = oracle.assemble(instructions)
        expected = oracle.run_bytes_many(code, states)
        actual = evaluate(probe, code, states)
        for index, (a, b) in enumerate(zip(actual, expected)):
            if a != b:
                raise AssertionError(f"{family} case {index}: {instructions}\nNyx {a}\nQEMU {b}")
        counts[family] += len(states)
    # Each deliberately wrong rewrite must be distinguished by the independent
    # original execution, including a condition mapping with the same value set.
    mutations = [
        (["add x0, x1, x2"], ["sub x0, x1, x2"]),
        (["adds x0, x1, x2"], ["add x0, x1, x2"]),
        (["add w0, w1, w2"], ["add x0, x1, x2"]),
        (["csel x0, x1, x2, eq"], ["csel x0, x2, x1, eq"]),
        (["add x0, x1, w2, sxth"], ["add x0, x1, w2, uxth"]),
        (["subs w0, w1, w2, uxtb #4"], ["subs w0, w1, w2, uxtb"]),
        (["smaddl x0, w1, w2, x3"], ["umaddl x0, w1, w2, x3"]),
        (["smaddl x0, w1, w2, x3"], ["smsubl x0, w1, w2, x3"]),
        (["ccmp x1, x2, #0, eq"], ["ccmn x1, x2, #0, eq"]),
        (["ccmp w1, w2, #5, ne"], ["ccmp x1, x2, #5, ne"]),
        (["ccmp x1, #3, #5, ge"], ["ccmp x1, #3, #10, ge"]),
        (["udiv x0, x5, x6"], ["sdiv x0, x5, x6"]),
        (["sdiv w0, w5, w6"], ["sdiv x0, x5, x6"]),
        (["umulh x0, x5, x6"], ["smulh x0, x5, x6"]),
        (["clz x0, x5"], ["cls x0, x5"]),
        (["rev x0, x5"], ["rev32 x0, x5"]),
        (["rev16 w0, w5"], ["rev w0, w5"]),
        (["rbit w0, w5"], ["rbit x0, x5"]),
        (["mrs x0, nzcv"], ["mrs x0, nzcv", "lsr x0, x0, #28"]),
        (["msr nzcv, x1"], ["msr nzcv, xzr"]),
    ]
    for original, mutant in mutations:
        expected = oracle.run_bytes_many(oracle.assemble(original), states)
        actual = evaluate(probe, oracle.assemble(mutant), states)
        if actual == expected:
            raise AssertionError(f"unrefuted mutation: {original} -> {mutant}")
    print(
        f"Independent AArch64 differential: {sum(counts.values())} states, {len(snippets)} sequences"
    )
    print("Family comparable counts:", dict(sorted(counts.items())))
    print(f"Mutations refuted: {len(mutations)}/{len(mutations)}")
    print("NOT CHECKED: memory, faults, SP, calls, branches, relocations, FP/SIMD")


if __name__ == "__main__":
    main()
