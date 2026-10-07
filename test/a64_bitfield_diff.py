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
from a64_sp_diff import evaluate as evaluate_sp, independent


def reference(state, op, width, rotate, end, destination, source):
    mask = (1 << width) - 1
    value = 0 if source == 31 else state.registers[source] & mask
    old = 0 if destination == 31 else state.registers[destination] & mask
    # Express extraction/insertion as intervals, independently of DecodeBitMasks
    # and the decoder's rotate-and-mask implementation.
    if rotate <= end:
        count, low = end - rotate + 1, 0
        field = (value >> rotate) & ((1 << count) - 1)
    else:
        count, low = end + 1, width - rotate
        field = (value & ((1 << count) - 1)) << low
    field_mask = ((1 << count) - 1) << low
    result = field
    if op == "bfm":
        result |= old & ~field_mask
    elif op == "sbfm" and (value >> end) & 1:
        result |= mask ^ ((1 << (low + count)) - 1)
    registers = list(state.registers)
    if destination != 31:
        registers[destination] = result & mask
    return State(tuple(registers), state.nzcv)


def states(rng, width, end):
    sign = 1 << end
    values = [
        0,
        (1 << 64) - 1,
        sign,
        sign - 1,
        sign | 1,
        1 << (width - 1),
        0xAAAAAAAAAAAAAAAA,
        0x5555555555555555,
    ]
    result = []
    for flags in range(16):
        registers = [rng.getrandbits(64) for _ in range(31)]
        registers[1] = values[flags] if flags < len(values) else rng.getrandbits(64)
        registers[0] = (0, (1 << 64) - 1, 0xAAAAAAAAAAAAAAAA, 0x5555555555555555)[flags % 4]
        result.append(State(tuple(registers), flags << 28))
    return result


def main():
    probe = sys.argv[1]
    oracle = Oracle()
    memory_oracle = MemoryOracle()
    rng = random.Random(0xB17F1E1D)
    counts = collections.Counter()
    instructions = 0
    sp_sentinels = 0
    for width, prefix in ((32, "w"), (64, "x")):
        fields = [
            (0, 0),
            (0, width - 1),
            (1, width - 1),
            (width - 1, width - 1),
            (width - 1, 0),
            (width // 2, width // 2 - 1),
            (width // 2, 3),
            (3, width // 2),
            (7, 7),
            (0, 7),
            (0, 15),
        ]
        if width == 64:
            fields.append((0, 31))
        for op in ("sbfm", "ubfm", "bfm"):
            for rotate, end in fields:
                fixtures = states(rng, width, end)
                text = f"{op} {prefix}0, {prefix}1, #{rotate}, #{end}"
                code = oracle.assemble([text])
                native = oracle.run_bytes_many(code, fixtures)
                actual = evaluate(probe, code, fixtures)
                modeled = [reference(s, op, width, rotate, end, 0, 1) for s in fixtures]
                if actual != native or modeled != native:
                    raise AssertionError((text, actual, native, modeled))
                counts[f"{op}{width}"] += len(fixtures)
                instructions += 1
            for destination, source in ((1, 1), (0, 31), (31, 1), (30, 30)):
                rotate, end = width - 5, 2
                fixtures = states(rng, width, end)
                rd = f"{prefix}zr" if destination == 31 else f"{prefix}{destination}"
                rn = f"{prefix}zr" if source == 31 else f"{prefix}{source}"
                text = f"{op} {rd}, {rn}, #{rotate}, #{end}"
                code = oracle.assemble([text])
                native = oracle.run_bytes_many(code, fixtures)
                if (
                    evaluate(probe, code, fixtures) != native
                    or [reference(s, op, width, rotate, end, destination, source) for s in fixtures]
                    != native
                ):
                    raise AssertionError(text)
                counts["overlap_and_zero_register"] += len(fixtures)
                instructions += 1
                if destination == 31 or source == 31:
                    sentinels = [
                        MemoryState(s.registers, s.nzcv, rng.getrandbits(64) | 1) for s in fixtures
                    ]
                    checked = evaluate_sp(probe, code, sentinels)
                    native_sp = independent(memory_oracle, code, sentinels)
                    if checked != native_sp:
                        raise AssertionError(("native SP sentinel mismatch", text))
                    for initial, actual, expected in zip(sentinels, checked, native, strict=True):
                        if (
                            actual.sp != initial.sp
                            or actual.registers != expected.registers
                            or actual.nzcv != expected.nzcv
                        ):
                            raise AssertionError(("ZR versus SP confusion", text))
                    sp_sentinels += len(sentinels)
    mutations = [
        ("sbfm x0, x1, #3, #7", "ubfm x0, x1, #3, #7"),
        ("bfm x0, x1, #61, #2", "ubfm x0, x1, #61, #2"),
        ("ubfm x0, x1, #61, #2", "ubfm x0, x1, #60, #2"),
        ("sbfm x0, x1, #61, #2", "sbfm x0, x1, #61, #3"),
        ("bfm w0, w1, #3, #7", "bfm x0, x1, #3, #7"),
        ("bfm x1, x1, #61, #2", "bfm x1, x0, #61, #2"),
        ("ubfm xzr, x1, #3, #7", "ubfm x0, x1, #3, #7"),
        ("ubfm x0, x1, #3, #7", "ands x0, x1, #0x1f"),
    ]
    for original, mutant in mutations:
        fixtures = states(rng, 64, 7) + states(rng, 64, 2)
        native = oracle.run_bytes_many(oracle.assemble([original]), fixtures)
        if evaluate(probe, oracle.assemble([mutant]), fixtures) == native:
            raise AssertionError(("unrefuted mutation", original, mutant))
    print(
        json.dumps(
            {
                "comparisons": dict(counts),
                "total": sum(counts.values()),
                "instruction_fixtures": instructions,
                "states_per_fixture": 16,
                "mutations_refuted": len(mutations),
                "python_interval_reference_checked": True,
                "all_nzcv_values_preserved": True,
                "hardware_checked": False,
                "native_sp_sentinel_comparisons": sp_sentinels,
                "native_sp_capture_checked": True,
                "scope": "stratified SBFM/UBFM/BFM register semantics; no memory or fault claims",
                "tools": oracle.versions(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
