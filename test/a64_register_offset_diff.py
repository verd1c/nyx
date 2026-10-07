#!/usr/bin/env python3
import collections
import dataclasses
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tools.oracle.memory import MemoryOracle, MemoryState, DEFAULT_BASE
from a64_memory_diff import assemble, evaluate, same


def initial(extension, negative=False, base=DEFAULT_BASE, flags=0):
    rng = random.Random(0x0FF5E7 + flags)
    registers = [rng.getrandbits(64) for _ in range(31)]
    registers[1] = base + 1024
    value = -2 if negative else 3
    registers[2] = value & ((1 << 64) - 1)
    if extension in ("uxtw", "sxtw"):
        registers[2] = (0xABCDEF01 << 32) | (value & 0xFFFFFFFF)
    memory = bytes(rng.getrandbits(8) for _ in range(4096))
    return MemoryState(tuple(registers), flags << 28, base + 2048, memory, base)


def main():
    probe = sys.argv[1]
    oracle = MemoryOracle()
    counts = collections.Counter()
    faults = 0

    def compare(instruction, states, family):
        nonlocal faults
        code = assemble(oracle, instruction)
        outcomes = oracle.run_many(code, states)
        for state, outcome in zip(states, outcomes, strict=True):
            output, actual = evaluate(probe, code, state)
            if not same(outcome, output, actual):
                raise AssertionError((instruction, state, outcome, output))
            if len(output["events"]) != 1:
                raise AssertionError("scalar register-offset instruction lost its modeled access")
            counts[family] += 1
            faults += not outcome.completed
        return outcomes

    forms = [
        ("ldr x0", 3),
        ("str x0", 3),
        ("ldr w0", 2),
        ("str w0", 2),
        ("ldrb w0", 0),
        ("strb w0", 0),
        ("ldrh w0", 1),
        ("strh w0", 1),
        ("ldrsb w0", 0),
        ("ldrsb x0", 0),
        ("ldrsh w0", 1),
        ("ldrsh x0", 1),
        ("ldrsw x0", 2),
    ]
    for operation, scale in forms:
        for extension in ("uxtw", "sxtw", "lsl", "sxtx"):
            index = "w2" if extension.endswith("tw") else "x2"
            states = [
                initial(extension, negative, base, flags)
                for base, flags in ((DEFAULT_BASE, 0), (DEFAULT_BASE + 0x200000, 15))
                for negative in ((False, True) if extension.startswith("s") else (False,))
            ]
            for shift in sorted({0, scale}):
                compare(f"{operation}, [x1, {index}, {extension} #{shift}]", states, extension)
    for instruction in (
        "ldr x0, [sp, x2, lsl #3]",
        "str x0, [sp, w2, sxtw #3]",
        "ldr x1, [x1, x2]",
        "ldr x2, [x1, x2]",
        "str x2, [x1, x2]",
        "ldrsw xzr, [x1, x2, lsl #2]",
        "str xzr, [x1, xzr]",
        "ldr x0, [x1, wzr, sxtw #3]",
    ):
        compare(instruction, [initial("lsl")], "sp_zr_alias")
    state = initial("lsl")
    registers = list(state.registers)
    # Both operands use the same original register; their modular sum is mapped.
    registers[1] = (1 << 63) + (state.base + 512) // 2
    compare(
        "ldr x1, [x1, x1]",
        [dataclasses.replace(state, registers=tuple(registers))],
        "base_index_destination_alias",
    )
    for operation in ("ldr x0", "str x0", "ldrsw x0", "ldrb w0", "ldr xzr", "str xzr"):
        states = []
        for address in (state.base - 1, state.base + 4096):
            registers = list(state.registers)
            registers[1] = address
            registers[2] = 0
            states.append(dataclasses.replace(state, registers=tuple(registers)))
        compare(f"{operation}, [x1, x2]", states, "guard_fault")
    registers = list(state.registers)
    registers[1] = state.base + 4092
    registers[2] = 0
    for operation in ("ldr x0", "str x0"):
        compare(
            f"{operation}, [x1, x2]",
            [dataclasses.replace(state, registers=tuple(registers))],
            "cross_guard_fault",
        )
    # Actual DEVELOPMENT dispatcher encoding; source address affects provenance,
    # not this instruction's register-derived address or loaded signed value.
    code = bytes.fromhex("2b79a8b8")
    if code != assemble(oracle, "ldrsw x11, [x9, x8, lsl #2]"):
        raise AssertionError("development original word differs from independent assembly")
    registers = list(state.registers)
    registers[9] = state.base + 1024
    registers[8] = 3
    compare(
        "ldrsw x11, [x9, x8, lsl #2]",
        [dataclasses.replace(state, registers=tuple(registers))],
        "development_dispatcher",
    )
    mutations = [
        ("ldr x0, [x1, x2]", "ldr x0, [x1, x2, lsl #3]", initial("lsl")),
        ("ldr x0, [x1, w2, sxtw]", "ldr x0, [x1, w2, uxtw]", initial("sxtw", True)),
        ("ldrsw x0, [x1, x2]", "ldr w0, [x1, x2]", initial("lsl")),
        ("ldr x0, [x1, x2]", "ldr x0, [x1, xzr]", initial("lsl")),
    ]
    for original, mutant, current in mutations:
        if original.startswith("ldrsw"):
            memory = bytearray(current.memory)
            memory[1027:1031] = bytes.fromhex("00000080")
            current = dataclasses.replace(current, memory=bytes(memory))
        expected = oracle.run(assemble(oracle, original), current)
        output, actual = evaluate(probe, assemble(oracle, mutant), current)
        if same(expected, output, actual):
            raise AssertionError(("unrefuted mutation", original, mutant))
    print(
        json.dumps(
            {
                "comparisons": dict(counts),
                "total": sum(counts.values()),
                "fault_cases": faults,
                "mutations_refuted": len(mutations),
                "hardware_checked": False,
                "architectural_fault_authority": False,
                "memory_events_independently_checked": False,
                "tools": oracle.versions(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
