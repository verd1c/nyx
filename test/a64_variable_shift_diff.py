#!/usr/bin/env python3
import collections
import json
import pathlib
import random
import struct
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tools.oracle.aarch64 import Oracle, State
from tools.oracle.memory import MemoryOracle, MemoryState
from a64_semantic_diff import evaluate
from a64_sp_diff import evaluate as evaluate_sp, independent

MASK64 = (1 << 64) - 1


def reference(state, op, width, destination, source, count_register, mutant=None):
    mask = (1 << width) - 1
    value = 0 if source == 31 else state.registers[source] & mask
    raw_count = 0 if count_register == 31 else state.registers[count_register] & mask
    count = raw_count % (64 if mutant == "wrong_modulus" else width)
    if mutant == "no_modulus":
        count = raw_count
    if op == "lslv":
        result = 0 if count >= width else value << count
    elif op == "lsrv":
        result = 0 if count >= width else value >> count
    elif op == "asrv":
        signed = value - (1 << width) if value & (1 << (width - 1)) else value
        result = (-1 if signed < 0 else 0) if count >= width else signed >> count
    else:
        rotation_width = 64 if mutant == "wide_rotation" else width
        count %= rotation_width
        result = (value >> count) | (value << (rotation_width - count))
    registers = list(state.registers)
    if destination != 31:
        registers[destination] = result & mask
    return State(tuple(registers), state.nzcv)


def states(rng, width, source, count_register):
    counts = [
        0,
        1,
        width - 1,
        width,
        width + 1,
        2 * width - 1,
        2 * width,
        MASK64,
        1 << 32,
        (1 << 32) + 1,
        (1 << 63) + width - 1,
        MASK64 - width,
        width // 2,
        width // 2 + 1,
        3,
        7,
    ]
    values = [
        0,
        MASK64,
        1 << (width - 1),
        (1 << (width - 1)) - 1,
        0xAAAAAAAAAAAAAAAA,
        0x5555555555555555,
        1,
        MASK64 - 1,
    ]
    result = []
    for flags, count in enumerate(counts):
        registers = [rng.getrandbits(64) for _ in range(31)]
        if source != 31:
            registers[source] = values[flags] if flags < len(values) else rng.getrandbits(64)
        if count_register != 31:
            registers[count_register] = count
        result.append(State(tuple(registers), flags << 28))
    return result


def recovery_checks(probe, oracle, rng):
    comparisons = collections.Counter()
    folds = 0
    source, bias, base = 0x1000, 0x200000000, 0x100000000
    page = bytes(index % 251 for index in range(4096))
    for width, prefix in ((32, "w"), (64, "x")):
        for op in ("lslv", "lsrv", "asrv", "rorv"):
            for count in (0, width, width + 1):
                code = oracle.assemble(
                    [
                        f"movn {prefix}1, #7",
                        f"movz {prefix}2, #{count}",
                        f"{op} {prefix}0, {prefix}1, {prefix}2",
                    ]
                )
                fixtures = [
                    State(tuple(rng.getrandbits(64) for _ in range(31)), flags << 28)
                    for flags in (0, 5, 10, 15)
                ]
                native = oracle.run_bytes_many(code, fixtures)
                for initial, expected in zip(fixtures, native, strict=True):
                    sp = rng.getrandbits(64)
                    payload = (
                        b"NYXMBA01"
                        + struct.pack(
                            "<37Q",
                            *initial.registers,
                            initial.nzcv,
                            sp,
                            base,
                            source,
                            bias,
                            len(code),
                        )
                        + code
                        + page
                    )
                    for mode in ("original", "normalized", "simplified"):
                        raw = subprocess.run(
                            [probe, mode],
                            input=payload,
                            capture_output=True,
                            timeout=10,
                            check=True,
                        )
                        output = json.loads(raw.stdout)
                        if (
                            not output["completed"]
                            or output["pc"] != source + bias + len(code)
                            or tuple(output["registers"]) != expected.registers
                            or output["nzcv"] != expected.nzcv
                            or output["sp"] != sp
                            or bytes.fromhex(output["memory"]) != page
                            or any(
                                step["events"] or step["fault_address"] is not None
                                for step in output["trace"]
                            )
                        ):
                            raise AssertionError(
                                (
                                    "constant-count recovery changed execution",
                                    op,
                                    width,
                                    count,
                                    mode,
                                )
                            )
                        if mode == "simplified":
                            accepted = [
                                edit for edit in output["rules"] if edit["rule"] == "constant_fold"
                            ]
                            required = {
                                "lslv": {"shl"},
                                "lsrv": {"lshr"},
                                "asrv": {"ashr"},
                                "rorv": {"shl", "lshr", "bit_or"},
                            }[op]
                            folded = {
                                edit["original_op"]
                                for edit in accepted
                                if edit["replacement_op"] == "constant"
                            }
                            final_op = "bit_or" if op == "rorv" else next(iter(required))
                            if not required <= folded or not any(
                                edit["original_op"] == final_op
                                and edit["replacement_op"] == "constant"
                                and edit["replacement_immediate"] == expected.registers[0]
                                for edit in accepted
                            ):
                                raise AssertionError(
                                    "actual shift or final rotate expression was not folded to native result"
                                )
                            folds += len(accepted)
                        comparisons[mode] += 1
    return {
        "comparisons": dict(comparisons),
        "constant_fold_journal_occurrences": folds,
        "scratch_and_sp_unchanged_checked": True,
        "native_sp_capture_checked": False,
    }


def main():
    probe = sys.argv[1]
    oracle, memory_oracle = Oracle(), MemoryOracle()
    rng = random.Random(0x5A1F7)
    counts = collections.Counter()
    native_sp = 0
    fixtures_count = 0
    layouts = [
        (0, 1, 2),
        (1, 1, 2),
        (2, 1, 2),
        (0, 1, 1),
        (1, 1, 1),
        (0, 31, 2),
        (0, 1, 31),
        (31, 1, 2),
        (30, 30, 2),
    ]
    for width, prefix in ((32, "w"), (64, "x")):
        for op in ("lslv", "lsrv", "asrv", "rorv"):
            for destination, source, count_register in layouts:
                registers = [
                    f"{prefix}zr" if r == 31 else f"{prefix}{r}"
                    for r in (destination, source, count_register)
                ]
                text = f"{op} " + ", ".join(registers)
                code = oracle.assemble([text])
                fixtures = states(rng, width, source, count_register)
                native = oracle.run_bytes_many(code, fixtures)
                model = [
                    reference(s, op, width, destination, source, count_register) for s in fixtures
                ]
                actual = evaluate(probe, code, fixtures)
                if actual != native or model != native:
                    raise AssertionError((text, actual, native, model))
                counts[f"{op}{width}"] += len(fixtures)
                fixtures_count += 1
                if 31 in (destination, source, count_register):
                    sentinels = [
                        MemoryState(s.registers, s.nzcv, rng.getrandbits(64) | 1) for s in fixtures
                    ]
                    checked = evaluate_sp(probe, code, sentinels)
                    expected = independent(memory_oracle, code, sentinels)
                    if checked != expected or any(
                        a.sp != b.sp for a, b in zip(sentinels, checked, strict=True)
                    ):
                        raise AssertionError(("SP/ZR mismatch", text))
                    native_sp += len(fixtures)
    mutations = [
        ("lslv x0, x1, x2", "lsrv x0, x1, x2"),
        ("asrv x0, x1, x2", "lsrv x0, x1, x2"),
        ("rorv w0, w1, w2", "rorv x0, x1, x2"),
        ("lslv x2, x1, x2", "lslv x2, x2, x1"),
        ("lslv xzr, x1, x2", "lslv x0, x1, x2"),
    ]
    for original, mutant in mutations:
        fixtures = states(rng, 64, 1, 2)
        native = oracle.run_bytes_many(oracle.assemble([original]), fixtures)
        if evaluate(probe, oracle.assemble([mutant]), fixtures) == native:
            raise AssertionError(("unrefuted instruction mutant", original, mutant))
    reference_mutants = [
        ("lslv", 64, "no_modulus"),
        ("lsrv", 32, "wrong_modulus"),
        ("rorv", 32, "wide_rotation"),
    ]
    for op, width, mutant in reference_mutants:
        prefix = "w" if width == 32 else "x"
        fixtures = states(rng, width, 1, 2)
        native = oracle.run_bytes_many(
            oracle.assemble([f"{op} {prefix}0, {prefix}1, {prefix}2"]), fixtures
        )
        if [reference(s, op, width, 0, 1, 2, mutant) for s in fixtures] == native:
            raise AssertionError(("unrefuted independent-reference mutant", mutant))
    recovery = (
        recovery_checks(sys.argv[2], oracle, rng) if len(sys.argv) > 2 else {"checked": False}
    )
    print(
        json.dumps(
            {
                "instruction_fixtures": fixtures_count,
                "comparisons": dict(counts),
                "total": sum(counts.values()),
                "native_sp_sentinel_comparisons": native_sp,
                "instruction_mutations_refuted": len(mutations),
                "independent_reference_mutations_refuted": len(reference_mutants),
                "python_integer_reference_checked": True,
                "all_nzcv_values_preserved": True,
                "hardware_checked": False,
                "memory_instruction_semantics_checked": False,
                "constant_count_recovery": recovery,
                "tools": oracle.versions(),
                "sp_tools": memory_oracle.versions(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
