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
from tools.oracle.memory import DEFAULT_BASE, MemoryOracle, MemoryState, PAGE_SIZE, _parse


def assemble(oracle, instruction):
    with tempfile.TemporaryDirectory(prefix="nyx-memory-diff-") as directory:
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
            raise AssertionError("fixture must assemble to exactly one original word")
        return code


def evaluate(probe, code, state, source_address=0x400000, load_bias=0):
    payload = (
        b"NYXMDIF1"
        + struct.pack(
            "<36Q", *state.registers, state.nzcv, state.sp, state.base, source_address, load_bias
        )
        + code
        + state.memory
    )
    result = subprocess.run([probe], input=payload, capture_output=True, timeout=10, check=True)
    output = json.loads(result.stdout)
    if (
        output["profile"] != "concrete_atomic_scalar"
        or output["hardware_checked"]
        or output["architectural_fault_authority"]
    ):
        raise AssertionError("probe changed its declared reference profile")
    actual = MemoryState(
        tuple(output["registers"]),
        output["nzcv"],
        output["sp"],
        bytes.fromhex(output["memory"]),
        output["base"],
    )
    return output, actual


def same(outcome, output, actual):
    if outcome.completed != output["completed"] or outcome.state != actual:
        return False
    if outcome.completed:
        return output["fault"] is None
    return (
        outcome.signal == 11
        and output["fault"] is not None
        and output["fault"]["kind"] == "unmapped"
        and output["fault"]["address"] == outcome.fault_address
    )


def initial(base=DEFAULT_BASE, flags=0):
    rng = random.Random(0xD019F0 + flags)
    registers = [rng.getrandbits(64) for _ in range(31)]
    registers[1] = base + 1024
    memory = bytes(rng.getrandbits(8) for _ in range(PAGE_SIZE))
    return MemoryState(tuple(registers), flags << 28, base + 2048, memory, base)


def main():
    probe = sys.argv[1]
    oracle = MemoryOracle()
    counts = collections.Counter()
    fixtures = [
        ("unsigned_offset", "ldr x0, [x1, #24]"),
        ("unsigned_offset", "str x0, [x1, #24]"),
        ("release_store", "stlr w0, [x1]"),
        ("release_store", "stlr x0, [x1]"),
        ("release_store", "stlr wzr, [x1]"),
        ("release_store", "stlr xzr, [x1]"),
        ("unsigned_offset", "ldr w30, [x1, #12]"),
        ("unsigned_offset", "str w30, [x1, #12]"),
        ("byte_halfword", "ldrb w0, [x1, #3]"),
        ("byte_halfword", "strb w0, [x1, #3]"),
        ("byte_halfword", "ldrh w0, [x1, #6]"),
        ("byte_halfword", "strh w0, [x1, #6]"),
        ("unscaled", "ldur x30, [x1, #-7]"),
        ("unscaled", "stur w30, [x1, #-3]"),
        ("unscaled_small", "ldurb w0, [x1, #-1]"),
        ("unscaled_small", "sturb w0, [x1, #-1]"),
        ("unscaled_small", "ldurh w0, [x1, #-2]"),
        ("unscaled_small", "sturh w0, [x1, #-2]"),
        ("signed_load", "ldrsb x0, [x1, #1]"),
        ("signed_load", "ldrsb w0, [x1, #1]"),
        ("signed_load", "ldrsh x0, [x1, #2]"),
        ("signed_load", "ldrsh w0, [x1, #2]"),
        ("signed_load", "ldrsw x0, [x1, #4]"),
        ("signed_unscaled", "ldursb x30, [x1, #-1]"),
        ("signed_unscaled", "ldursb w30, [x1, #-1]"),
        ("signed_unscaled", "ldursh x30, [x1, #-2]"),
        ("signed_unscaled", "ldursh w30, [x1, #-2]"),
        ("signed_unscaled", "ldursw x30, [x1, #-4]"),
        ("signed_writeback", "ldrsb x0, [x1, #1]!"),
        ("signed_writeback", "ldrsh w0, [x1], #2"),
        ("signed_writeback", "ldrsw x30, [x1], #4"),
        ("writeback", "ldr x0, [x1, #8]!"),
        ("writeback", "str w0, [x1], #4"),
        ("writeback", "strb w0, [x1, #-1]!"),
        ("writeback", "ldrh w0, [x1], #2"),
        ("sp_writeback", "ldr x0, [sp, #16]!"),
        ("sp_writeback", "str x0, [sp], #16"),
        ("zero_register", "ldr xzr, [x1], #8"),
        ("zero_register", "str wzr, [x1, #4]"),
        ("base_destination_alias", "ldr x1, [x1, #8]"),
        ("pair_sp", "stp x0, x30, [sp, #-16]!", 2),
        ("pair_sp", "ldp x0, x30, [sp], #16", 2),
        ("pair_width", "ldp w0, w30, [x1, #-8]", 1),
        ("pair_width", "stp w0, w30, [x1, #8]", 1),
        ("pair_signed", "ldpsw x0, x30, [x1, #-8]", 1),
        ("pair_zero", "ldp xzr, x30, [x1]", 2),
        ("pair_zero", "stp xzr, xzr, [x1]", 2),
        ("pair_base_alias", "ldp x1, x2, [x1]", 2),
    ]
    states = [
        initial(base, flags)
        for base in (DEFAULT_BASE, DEFAULT_BASE + 0x200000)
        for flags in (0, 5, 10, 15)
    ]
    ordered = [
        f"{mnemonic} {register}, [x1]"
        for mnemonic, registers in (
            ("ldar", ("w0", "x0", "wzr", "xzr")),
            ("ldarb", ("w0", "wzr")),
            ("ldarh", ("w0", "wzr")),
            ("stlrb", ("w0", "wzr")),
            ("stlrh", ("w0", "wzr")),
        )
        for register in registers
    ]
    ordered += [
        "ldar x1, [x1]",
        "ldarb w1, [x1]",
        "stlr x1, [x1]",
        "ldar x0, [sp]",
        "ldarb w0, [sp]",
        "ldarh w0, [sp]",
        "stlrb w0, [sp]",
        "stlrh w0, [sp]",
    ]
    fixtures += [("acquire_release", instruction) for instruction in ordered]
    for fixture in fixtures:
        family, instruction = fixture[:2]
        event_count = fixture[2] if len(fixture) == 3 else 1
        code = assemble(oracle, instruction)
        expected = oracle.run_many(code, states)
        for state, outcome in zip(states, expected, strict=True):
            output, actual = evaluate(probe, code, state)
            if not same(outcome, output, actual):
                raise AssertionError(
                    f"{family}: {instruction}, base={state.base:x}, NZCV={state.nzcv:x}"
                )
            if len(output["events"]) != event_count or not all(
                event["completed"] for event in output["events"]
            ):
                raise AssertionError(
                    "completed memory instruction did not produce its access events"
                )
            # The recorded events are the project's own account of where it
            # touched memory. QEMU supplies no event trace, so compare them
            # against the address the disassembled text implies instead of
            # leaving them unchecked.
            access = _parse(instruction)
            address = access.effective(state)
            write = instruction.split()[0].startswith("st")
            covered = 0
            for event in output["events"]:
                if event["address"] != address + covered or event["write"] != write:
                    raise AssertionError(
                        f"recorded access event disagrees with the instruction's address or "
                        f"direction: {instruction}, event={event}, expected={address + covered:x}"
                    )
                covered += event["size"]
            if covered != access.size:
                raise AssertionError(
                    f"recorded access events do not cover the access: {instruction}"
                )
            counts[family] += 1

    # Compare QEMU-observed guard faults for this pinned backend. These cases do
    # not establish architecture-wide post-abort atomicity or SP alignment rules.
    for instruction in (
        "ldr x0, [x1], #8",
        "str x0, [x1], #8",
        "ldr x0, [x1, #8]!",
        "str x0, [x1, #8]!",
    ):
        code = assemble(oracle, instruction)
        state = initial()
        registers = list(state.registers)
        registers[1] = state.base + PAGE_SIZE - (8 if "!" in instruction else 0)
        state = dataclasses.replace(state, registers=tuple(registers))
        outcome = oracle.run(code, state)
        output, actual = evaluate(probe, code, state)
        if outcome.completed or not same(outcome, output, actual) or actual != state:
            raise AssertionError(f"guard fault or writeback disagreement: {instruction}")
        if len(output["events"]) != 1 or output["events"][0]["completed"]:
            raise AssertionError("faulting access event lost")
        # The fault record's direction is the project's own account of what it
        # was doing when it faulted; nothing else here compares it.
        if output["fault"]["write"] != instruction.startswith("st"):
            raise AssertionError(f"fault record reports the wrong direction: {instruction}")
        counts["guard_fault_reference"] += 1

    for instruction in ("stlr w0, [x1]", "stlr x0, [x1]", *ordered[:12]):
        code = assemble(oracle, instruction)
        state = initial()
        registers = list(state.registers)
        registers[1] = state.base + PAGE_SIZE
        state = dataclasses.replace(state, registers=tuple(registers))
        outcome = oracle.run(code, state)
        output, actual = evaluate(probe, code, state)
        if outcome.completed or not same(outcome, output, actual) or actual != state:
            raise AssertionError(f"acquire/release guard fault disagreement: {instruction}")
        if len(output["events"]) != 1 or output["events"][0]["completed"]:
            raise AssertionError("acquire/release guard fault lost its access event")
        if output["fault"]["write"] != instruction.startswith("st"):
            raise AssertionError("acquire/release fault direction differs")
        counts["acquire_release_guard_fault_reference"] += 1

    # The pinned QEMU profile packs W pairs into one 64-bit access. X pairs
    # retain two accesses; only their first store can commit before this fault.
    for instruction, width, completed_events in (
        ("stp x0, x2, [x1]", 8, 1),
        ("ldp x0, x2, [x1]", 8, 1),
        ("stp w0, w2, [x1], #8", 4, 0),
        ("ldpsw x0, x2, [x1], #8", 4, 0),
    ):
        code = assemble(oracle, instruction)
        state = initial()
        registers = list(state.registers)
        registers[1] = state.base + PAGE_SIZE - width
        state = dataclasses.replace(state, registers=tuple(registers))
        outcome = oracle.run(code, state)
        output, actual = evaluate(probe, code, state)
        if outcome.completed or not same(outcome, output, actual):
            raise AssertionError(f"pair second-access fault prefix disagrees: {instruction}")
        if (
            len(output["events"]) != completed_events + 1
            or any(not event["completed"] for event in output["events"][:completed_events])
            or output["events"][-1]["completed"]
        ):
            raise AssertionError("pair fault did not retain its profile-specific access prefix")
        if output["fault"]["write"] != instruction.startswith("st"):
            raise AssertionError(f"pair fault record reports the wrong direction: {instruction}")
        counts["pair_fault_reference"] += 1

    state = initial()
    reference_state = state
    for instruction in ("str x0, [x1]", "ldr w2, [x1, #4]", "strh w3, [x1, #2]", "ldr x30, [x1]"):
        code = assemble(oracle, instruction)
        outcome = oracle.run(code, reference_state)
        output, state = evaluate(probe, code, state)
        if not same(outcome, output, state):
            raise AssertionError("overlapping successive accesses disagree")
        reference_state = outcome.state
        counts["overlapping_sequence"] += 1

    mutations = [
        ("ldar x0, [x1]", "ldar w0, [x1]", "acquire_load_width"),
        ("ldarb w0, [x1]", "ldarh w0, [x1]", "acquire_byte_width"),
        ("stlrb w0, [x1]", "stlrh w0, [x1]", "release_byte_width"),
        ("ldarb wzr, [x1]", "ldarb w0, [x1]", "acquire_discard"),
        ("str x0, [x1]", "str w0, [x1]", "store_width"),
        ("ldr x0, [x1]", "ldr w0, [x1]", "load_width"),
        ("str x0, [x1], #8", "str x0, [x1]", "writeback"),
        ("stp x0, x2, [x1]", "str x0, [x1]", "dropped_second_store"),
        ("ldp x0, x2, [x1]", "ldr x0, [x1]", "dropped_second_load"),
        ("stp x0, x2, [x1], #16", "stp x0, x2, [x1]", "pair_writeback"),
    ]
    for original, mutation, name in mutations:
        state = initial()
        if name == "release_byte_width":
            # The fixed random seed happens to leave this second byte equal to
            # the wider store's value. Make the extra write observable.
            registers = list(state.registers)
            registers[0] = (registers[0] & ~0xFF00) | ((state.memory[1025] ^ 0xFF) << 8)
            state = dataclasses.replace(state, registers=tuple(registers))
        expected = oracle.run(assemble(oracle, original), state)
        output, actual = evaluate(probe, assemble(oracle, mutation), state)
        if same(expected, output, actual):
            raise AssertionError("negative control failed to refute " + name)
        counts["refuted_mutations"] += 1

    # These words are independently admitted and executed by the reference oracle.
    # Rebase the identical bytes using the runner's actual code mapping, including
    # non-page-aligned biases that distinguish ADRP's runtime-PC masking order.
    mask64 = (1 << 64) - 1
    for page in (False, True):
        for immediate in (0, 1, -1, (1 << 20) - 1, -(1 << 20)):
            encoded = immediate & ((1 << 21) - 1)
            word = (
                (0x90000000 if page else 0x10000000)
                | ((encoded & 3) << 29)
                | ((encoded >> 2) << 5)
                | (30 if page else 0)
            )
            code = word.to_bytes(4, "little")
            state = initial()
            outcome = oracle.run(code, state)
            for source_address in (0x1000, 0x1234, 0x1FFC):
                bias = (outcome.code_address - source_address) & mask64
                output, actual = evaluate(probe, code, state, source_address, bias)
                if not same(outcome, output, actual) or output["events"]:
                    raise AssertionError(
                        f"PC-relative mismatch: page={page}, offset={immediate}, source={source_address:x}"
                    )
                counts["adrp_runtime_bias" if page else "adr_runtime_bias"] += 1
                wrong, wrong_state = evaluate(
                    probe, code, state, source_address, (bias + PAGE_SIZE) & mask64
                )
                if same(outcome, wrong, wrong_state):
                    raise AssertionError("wrong load bias was not refuted")
                counts["refuted_bias_mutations"] += 1

    state = initial()
    valid_payload = (
        b"NYXMDIF1"
        + struct.pack("<36Q", *state.registers, state.nzcv, state.sp, state.base, 0x400000, 0)
        + bytes.fromhex("200040f9")
        + state.memory
    )
    for payload in (b"NYXMDIF1", valid_payload + b"extra", b"INVALID!" + valid_payload[8:]):
        malformed = subprocess.run([probe], input=payload, capture_output=True, timeout=10)
        if malformed.returncode == 0 or malformed.stdout:
            raise AssertionError("invalid probe input produced evidence")
    if sum(counts.values()) == 0:
        raise AssertionError("zero evidence")
    print(
        json.dumps(
            {
                "profile": "concrete_atomic_scalar",
                "hardware_checked": False,
                "architectural_fault_authority": False,
                "memory_events": "addresses, sizes and directions checked against the disassembled instruction; QEMU supplies no event trace, and values are not independently traced",
                "families": dict(counts),
                "oracle": oracle.versions(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
