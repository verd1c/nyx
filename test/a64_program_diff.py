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
from tools.oracle.program import ProgramOracle, ProgramState

MASK64 = (1 << 64) - 1


def assemble(oracle, body):
    with tempfile.TemporaryDirectory(prefix="nyx-program-diff-") as directory:
        path = pathlib.Path(directory)
        (path / "fixture.S").write_text(".text\n" + body + "\n")
        backend = oracle.backend
        backend._command(
            [backend.compiler, "-c", "-o", str(path / "fixture.o"), str(path / "fixture.S")]
        )
        backend._command(
            [
                backend.objcopy,
                "-O",
                "binary",
                "-j",
                ".text",
                str(path / "fixture.o"),
                str(path / "fixture.bin"),
            ]
        )
        code = (path / "fixture.bin").read_bytes()
        if not code or len(code) > 4096 or len(code) % 4:
            raise AssertionError("invalid synthetic program extent")
        return code


def initial(seed, code_base, scratch_base):
    rng = random.Random(seed)
    registers = [rng.getrandbits(64) for _ in range(31)]
    registers[2] = seed % 3
    registers[28] = scratch_base
    return ProgramState(
        tuple(registers),
        (seed % 16) << 28,
        scratch_base + 2048,
        bytes(rng.getrandbits(8) for _ in range(4096)),
        scratch_base,
        code_base,
    )


def evaluate(probe, code, state, source=0x400124, steps=256):
    payload = (
        b"NYXPDIF1"
        + struct.pack(
            "<38Q",
            *state.registers,
            state.nzcv,
            state.sp,
            state.scratch_base,
            source,
            (state.code_base - source) & MASK64,
            steps,
            len(code),
        )
        + code
        + state.memory
    )
    result = subprocess.run([probe], input=payload, capture_output=True, timeout=10, check=True)
    return json.loads(result.stdout)


def matches(outcome, output):
    if outcome.signal:
        return (
            output["status"] == "fault"
            and output["pc"] == outcome.exit_pc
            and output["trace"][-1]["fault_address"] == outcome.fault_address
            and tuple(output["registers"]) == outcome.state.registers
            and output["nzcv"] == outcome.state.nzcv
            and output["sp"] == outcome.state.sp
            and bytes.fromhex(output["memory"]) == outcome.state.memory
        )
    return (
        output["status"] == "exit"
        and output["pc"] == outcome.exit_pc
        and tuple(output["registers"]) == outcome.state.registers
        and output["nzcv"] == outcome.state.nzcv
        and output["sp"] == outcome.state.sp
        and bytes.fromhex(output["memory"]) == outcome.state.memory
    )


def main():
    probe = sys.argv[1]
    oracle = ProgramOracle()
    fixtures = {
        "direct_call": """
            bl .Lcallee
            add x0, x0, #7
            b .Lexit
        .Lcallee:
            add x0, x0, #9
            str x0, [x28, #24]
            ret
        .Lexit:
        """,
        "indirect_call_lr": """
            adr x30, .Lcallee
            blr x30
            add x0, x0, #7
            b .Lexit
        .Lcallee:
            eor x0, x0, x2
            str x0, [x28, #24]
            ret
        .Lexit:
        """,
        "callee_flags": """
            bl .Lcallee
            b.eq .Lequal
            mov x0, #1
            b .Lexit
        .Lequal:
            mov x0, #2
            b .Lexit
        .Lcallee:
            subs x2, x2, #1
            ret
        .Lexit:
        """,
        "escaped_stack_alias": """
            sub sp, sp, #16
            str x0, [sp]
            bl .Lcallee
            ldr x3, [sp]
            add sp, sp, #16
            b .Lexit
        .Lcallee:
            str x2, [x28, #2032]
            ldr x0, [sp]
            ret
        .Lexit:
        """,
        "callee_guard_fault": """
            bl .Lcallee
            mov x0, #123
            b .Lexit
        .Lcallee:
            str x2, [x28]
            ldr x0, [x28, #4096]
            ret
        .Lexit:
        """,
        "callee_pair_writeback": """
            bl .Lcallee
            ldr x3, [sp, #-16]
            b .Lexit
        .Lcallee:
            stp x0, x2, [sp, #-16]!
            ldp x4, x5, [sp], #16
            ret
        .Lexit:
        """,
    }
    counts = collections.Counter()
    mutations = collections.Counter()
    for name, body in fixtures.items():
        code = assemble(oracle, body)
        states = [
            initial(seed, code_base, scratch_base)
            for code_base, scratch_base in ((0x200001000, 0x100000000), (0x600001000, 0x100004000))
            for seed in range(12)
        ]
        outcomes = oracle.run_many(code, states, trusted_fixture=True)
        for index, (state, outcome) in enumerate(zip(states, outcomes, strict=True)):
            # A second source coordinate exercises negative/modular load biases.
            source = 0x700000000 if index % 2 else 0x400124
            output = evaluate(probe, code, state, source=source)
            if not matches(outcome, output) or output["committed_steps"] == 0:
                raise AssertionError((name, state, outcome, output))
            counts[name] += 1
        if name == "direct_call":
            # Keep BL, RET and LR intact so refutation depends on callee effects.
            mutant = code[:12] + bytes.fromhex("1f2003d51f2003d5") + code[20:]
            mutation_name = "erased_callee_effects"
        elif name == "escaped_stack_alias":
            mutant = code[:24] + bytes.fromhex("1f2003d5") + code[28:]
            mutation_name = "dropped_callee_stack_alias_store"
        elif name == "callee_flags":
            word = int.from_bytes(code[4:8], "little") ^ 1
            mutant = code[:4] + word.to_bytes(4, "little") + code[8:]
            mutation_name = "swapped_returned_flag_branch"
        else:
            continue
        for state, outcome in zip(states, outcomes, strict=True):
            changed = evaluate(probe, mutant, state)
            if mutation_name == "erased_callee_effects":
                if (
                    changed["registers"][30] != outcome.state.registers[30]
                    or changed["pc"] != outcome.exit_pc
                ):
                    raise AssertionError(
                        "callee-effect mutation unexpectedly changed LR or exit PC"
                    )
            if matches(outcome, changed):
                raise AssertionError(f"{mutation_name} escaped detection")
            mutations[mutation_name] += 1
    # A step budget yields an incomplete prefix; it says nothing about
    # termination or equivalence.
    code = assemble(oracle, fixtures["direct_call"])
    state = initial(0, 0x200001000, 0x100000000)
    output = evaluate(probe, code, state, steps=1)
    if output["status"] != "step_limit" or output["committed_steps"] != 1:
        raise AssertionError("program budget did not preserve a typed incomplete call prefix")
    if output["registers"][30] != state.code_base + 4 or output["pc"] != state.code_base + 12:
        raise AssertionError("incomplete call prefix lost target or link write")
    print(
        json.dumps(
            {
                "comparisons": sum(counts.values()),
                "families": dict(counts),
                "mutations_refuted": sum(mutations.values()),
                "mutations": dict(mutations),
                "tools": oracle.versions(),
                "hardware_checked": False,
                "memory_events_independently_checked": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
