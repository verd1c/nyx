#!/usr/bin/env python3
import collections
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tools.oracle.aarch64 import Oracle, State
from tools.oracle.memory import MemoryOracle, MemoryState
from a64_semantic_diff import evaluate
from a64_sp_diff import assemble, evaluate as evaluate_sp, independent


def masks(width):
    # Construct repeated rotated runs mathematically; GNU as chooses the actual
    # N/immr/imms encoding independently of the production decoder.
    result = set()
    for element in (2, 4, 8, 16, 32, 64):
        if element > width:
            continue
        element_mask = (1 << element) - 1
        for ones in (1, element // 2, element - 1):
            run = (1 << ones) - 1
            for rotate in (0, 1, element - 1):
                rotated = ((run >> rotate) | (run << (element - rotate))) & element_mask
                result.add(sum(rotated << offset for offset in range(0, width, element)))
    return sorted(result)


def main():
    probe = sys.argv[1]
    oracle = Oracle()
    memory_oracle = MemoryOracle()
    rng = random.Random(0x10C1CA1)
    states = [
        State(tuple(rng.getrandbits(64) for _ in range(31)), flags << 28) for flags in range(16)
    ]
    states += [
        State((value,) * 31, 0xF0000000)
        for value in (0, 1, (1 << 64) - 1, 1 << 63, (1 << 32) - 1, 1 << 31)
    ]
    counts = collections.Counter()
    for width, prefix in ((32, "w"), (64, "x")):
        for mask in masks(width):
            for op in ("and", "orr", "eor", "ands"):
                instruction = f"{op} {prefix}0, {prefix}1, #{mask:#x}"
                code = assemble(memory_oracle, instruction)
                expected = oracle.run_bytes_many(code, states)
                actual = evaluate(probe, code, states)
                if actual != expected:
                    raise AssertionError(
                        (
                            instruction,
                            next(
                                i
                                for i, pair in enumerate(zip(actual, expected))
                                if pair[0] != pair[1]
                            ),
                        )
                    )
                counts[f"{op}{width}"] += len(states)
        for op in ("and", "orr", "eor", "ands"):
            instruction = (
                f"{op} {prefix}0, {prefix}zr, #0x55555555"
                if width == 32
                else f"{op} {prefix}0, {prefix}zr, #0x5555555555555555"
            )
            code = assemble(memory_oracle, instruction)
            if evaluate(probe, code, states) != oracle.run_bytes_many(code, states):
                raise AssertionError(instruction)
            counts["zero_source"] += len(states)
        instruction = f"ands {prefix}zr, {prefix}1, #0xff"
        code = assemble(memory_oracle, instruction)
        if evaluate(probe, code, states) != oracle.run_bytes_many(code, states):
            raise AssertionError(instruction)
        counts["flags_zero_destination"] += len(states)
    sp_states = [MemoryState(s.registers, s.nzcv, rng.getrandbits(64)) for s in states]
    for prefix, sp in (("w", "wsp"), ("x", "sp")):
        for op in ("and", "orr", "eor"):
            for mask in (1, 0xFF, 0x80000001 if prefix == "w" else 0x8000000000000001):
                instruction = f"{op} {sp}, {prefix}1, #{mask:#x}"
                code = assemble(memory_oracle, instruction)
                if evaluate_sp(probe, code, sp_states) != independent(
                    memory_oracle, code, sp_states
                ):
                    raise AssertionError(instruction)
                counts["sp_destination"] += len(sp_states)
    mutations = [
        ("and x0, x1, #0xff", "orr x0, x1, #0xff"),
        ("and x0, x1, #0xff", "and x0, x1, #0xfe"),
        ("ands x0, x1, #0xff", "and x0, x1, #0xff"),
        ("orr w0, w1, #0xff", "orr x0, x1, #0xff"),
    ]
    for original, mutant in mutations:
        expected = oracle.run_bytes_many(assemble(memory_oracle, original), states)
        if evaluate(probe, assemble(memory_oracle, mutant), states) == expected:
            raise AssertionError(("unrefuted mutation", original, mutant))
    expected = independent(memory_oracle, assemble(memory_oracle, "and sp, x1, #0xff"), sp_states)
    if evaluate_sp(probe, assemble(memory_oracle, "and x0, x1, #0xff"), sp_states) == expected:
        raise AssertionError("SP versus GPR destination mutation survived")
    print(
        json.dumps(
            {
                "comparisons": dict(counts),
                "total": sum(counts.values()),
                "mutations_refuted": len(mutations) + 1,
                "mask_counts": {str(width): len(masks(width)) for width in (32, 64)},
                "hardware_checked": False,
                "memory_instruction_semantics_checked": False,
                "tools": oracle.versions(),
                "sp_tools": memory_oracle.versions(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
