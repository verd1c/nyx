#!/usr/bin/env python3
import collections
import dataclasses
import json
import pathlib
import random
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.oracle.control import ControlOracle, ControlState, DEFAULT_PC

MASK64 = (1 << 64) - 1


def assemble(oracle, instruction):
    with tempfile.TemporaryDirectory(prefix="nyx-control-diff-") as directory:
        path = pathlib.Path(directory)
        (path / "word.S").write_text(".text\n" + instruction + "\n")
        backend = oracle.backend
        backend._command([backend.compiler, "-c", "-o", str(path / "word.o"), str(path / "word.S")])
        backend._command(
            [
                backend.objcopy,
                "-O",
                "binary",
                "-j",
                ".text",
                str(path / "word.o"),
                str(path / "word.bin"),
            ]
        )
        code = (path / "word.bin").read_bytes()
        if len(code) != 4:
            raise AssertionError("fixture must assemble to one original instruction")
        return code


def evaluate(probe, code, state, source=0x400124, bias=None):
    if bias is None:
        bias = (state.pc - source) & MASK64
    payload = (
        b"NYXCDIF1"
        + struct.pack("<35Q", *state.registers, state.nzcv, state.sp, source, bias)
        + code
    )
    result = subprocess.run([probe], input=payload, capture_output=True, timeout=10, check=True)
    return json.loads(result.stdout)


def matches(outcome, output):
    return (
        tuple(output["registers"]) == outcome.state.registers
        and output["nzcv"] == outcome.state.nzcv
        and output["sp"] == outcome.state.sp
        and output["transfer"]["target"] == outcome.landing_pc
    )


def initial(flags=0, pc=DEFAULT_PC):
    rng = random.Random(0xC071F0 + flags)
    return ControlState(tuple(rng.getrandbits(64) for _ in range(31)), flags << 28, 0x123456780, pc)


def with_register(state, register, value):
    values = list(state.registers)
    values[register] = value
    return dataclasses.replace(state, registers=tuple(values))


def main():
    probe = sys.argv[1]
    oracle = ControlOracle()
    counts = collections.Counter()
    mutations = collections.Counter()
    fixtures = []
    conditions = (
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
    for condition in conditions:
        for offset in (64, -64):
            fixtures.append(
                (
                    "flags",
                    f"b.{condition} .{offset:+d}",
                    "conditional",
                    [initial(flags) for flags in range(16)],
                )
            )
    for opcode in ("cbz", "cbnz"):
        for register in ("w0", "x0", "wzr", "xzr"):
            fixtures.append(
                (
                    "compare",
                    f"{opcode} {register}, .+128",
                    "conditional",
                    [
                        with_register(initial(), 0, value)
                        for value in (0, 1, 1 << 32, 1 << 63, MASK64)
                    ],
                )
            )
    for opcode in ("tbz", "tbnz"):
        for register, bit in (
            ("w0", 0),
            ("w0", 31),
            ("x0", 32),
            ("x0", 63),
            ("wzr", 31),
            ("xzr", 63),
        ):
            fixtures.append(
                (
                    "test_bit",
                    f"{opcode} {register}, #{bit}, .-128",
                    "conditional",
                    [
                        with_register(initial(), 0, value)
                        for value in (0, 1 << bit, MASK64 ^ (1 << bit), MASK64)
                    ],
                )
            )
    for opcode in ("b", "bl"):
        for offset in (-4096, -4, 4, 8188):
            fixtures.append(
                (
                    "direct",
                    f"{opcode} .{offset:+d}",
                    "call" if opcode == "bl" else "jump",
                    [
                        dataclasses.replace(initial(pc=pc), sp=sp)
                        for pc in (DEFAULT_PC, 0xF00001000)
                        for sp in (0, 1, MASK64, 0x123456780)
                    ],
                )
            )
    for opcode in ("br", "blr", "ret"):
        for register in (0, 16, 30):
            fixtures.append(
                (
                    "indirect",
                    f"{opcode} x{register}",
                    "call" if opcode == "blr" else "return" if opcode == "ret" else "jump",
                    [
                        with_register(initial(pc=pc), register, pc + offset)
                        for pc in (DEFAULT_PC, 0xF00001000)
                        for offset in (-4, 128)
                    ],
                )
            )
    for family, instruction, kind, states in fixtures:
        code = assemble(oracle, instruction)
        outcomes = oracle.run_many(code, states)
        for state, outcome in zip(states, outcomes, strict=True):
            output = evaluate(probe, code, state)
            if not matches(outcome, output) or output["transfer"]["kind"] != kind:
                raise AssertionError((instruction, state, outcome, output))
            continuation = output["transfer"]["continuation"]
            if continuation != (state.pc + 4 if kind == "call" else None):
                raise AssertionError("call continuation is not the observed link address")
            if kind == "call" and outcome.state.registers[30] != continuation:
                raise AssertionError("native link register differs from continuation")
            condition = output["transfer"]["condition"]
            if kind == "conditional":
                # These fixtures keep the two destinations distinct, so native PC
                # independently identifies the selected arm without a flag model.
                if type(condition) is not bool or condition != (outcome.landing_pc != state.pc + 4):
                    raise AssertionError("condition-to-destination relation differs")
            elif condition is not None:
                raise AssertionError("unconditional transfer invented a condition")
            counts[family] += 1
        # Inverting EQ/NE keeps the identical successor set. A target-set-only
        # verifier would miss this mutation; the observed relation must refute it.
        if instruction in ("b.eq .+64", "b.ne .+64"):
            mutant = (int.from_bytes(code, "little") ^ 1).to_bytes(4, "little")
            for state, outcome in zip(states, outcomes, strict=True):
                if matches(outcome, evaluate(probe, mutant, state)):
                    raise AssertionError("inverted condition was not refuted")
                mutations["same_successors_wrong_relation"] += 1
        if instruction == "bl .+4":
            mutant = (int.from_bytes(code, "little") & ~(1 << 31)).to_bytes(4, "little")
            for state, outcome in zip(states, outcomes, strict=True):
                if matches(outcome, evaluate(probe, mutant, state)):
                    raise AssertionError("missing link write was not refuted")
                mutations["dropped_link_write"] += 1
        if family == "direct":
            state, outcome = states[0], outcomes[0]
            if matches(outcome, evaluate(probe, code, state, bias=0)):
                raise AssertionError("missing runtime placement was not refuted")
            mutations["dropped_load_bias"] += 1
    print(
        json.dumps(
            {
                "comparisons": sum(counts.values()),
                "families": dict(counts),
                "mutations_refuted": sum(mutations.values()),
                "mutations": dict(mutations),
                "tools": oracle.versions(),
                "hardware_checked": False,
                "returning_calls_checked": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
