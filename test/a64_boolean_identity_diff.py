#!/usr/bin/env python3
import collections
import hashlib
import json
import pathlib
import random
import struct
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.oracle.aarch64 import Oracle, State
from tools.oracle.program import GlobalPage, ProgramOracle, ProgramState

MASK64 = (1 << 64) - 1
SCRATCH = 0x100000000
GLOBAL_PAGE = 0x100010000
# Original bytes of the development body at 0x74b1b4, matched against the ELF's
# PT_LOAD contents. This is not a reassembly of the recovered expression.
BODY = 0x74B1B4
BODY_CODE = bytes.fromhex("4a010b8a6baa46f96c9246f98b21cb9a4c010b8a4a010bca8a010aaa")
BODY_SHA256 = "01443468423eba64366b9fc952f2f29066ba2c7cd45728feba711a8761778974"
COUNT_OFFSET, VALUE_OFFSET = 3408, 3360


def union(a, b, width):
    return (a | b) & ((1 << width) - 1)


def total(a, b, width):
    return (a + b) & ((1 << width) - 1)


def complement(a, _, width):
    return ~a & ((1 << width) - 1)


# Each case is one obfuscated spelling of an identity, the rule that must commit
# for it, and an independent model of the value the whole sequence produces in x0.
# Both widths are covered: normalizing a W form reads each operand through its own
# extract of the zero-extended write, and recovery sees through that pair and
# numbers equal values, so the W spellings commit exactly as the X ones do.
def cases(width):
    prefix = "x" if width == 64 else "w"
    d, a, b, s, t, u = (f"{prefix}{r}" for r in (0, 1, 2, 3, 4, 5))
    shared = [
        ("xor_ones_not", complement, [f"movn {s}, #0", f"eor {d}, {a}, {s}"]),
        ("xor_ones_not", complement, [f"movn {s}, #0", f"eor {d}, {s}, {a}"]),
    ]
    return shared + [
        (
            "and_xor_union",
            union,
            [f"and {s}, {a}, {b}", f"eor {t}, {a}, {b}", f"orr {d}, {s}, {t}"],
        ),
        (
            "and_xor_union",
            union,
            [f"and {s}, {a}, {b}", f"eor {t}, {b}, {a}", f"orr {d}, {t}, {s}"],
        ),
        (
            "and_xor_union",
            union,
            [f"and {s}, {a}, {b}", f"eor {t}, {a}, {b}", f"orr {d}, {t}, {s}"],
        ),
        ("or_and_sum", total, [f"orr {s}, {a}, {b}", f"and {t}, {a}, {b}", f"add {d}, {s}, {t}"]),
        ("or_and_sum", total, [f"orr {s}, {a}, {b}", f"and {t}, {b}, {a}", f"add {d}, {t}, {s}"]),
        (
            "xor_and_sum",
            total,
            [f"eor {s}, {a}, {b}", f"and {t}, {a}, {b}", f"add {d}, {s}, {t}, lsl #1"],
        ),
        # The shifted carries reach the addition as its first operand here.
        (
            "xor_and_sum",
            total,
            [f"eor {s}, {b}, {a}", f"and {t}, {a}, {b}", f"lsl {u}, {t}, #1", f"add {d}, {u}, {s}"],
        ),
    ]


# Every mutation keeps the surrounding shape and breaks one requirement of the
# identity, so the rule must decline while execution still matches the hardware.
def declines(width):
    prefix = "x" if width == 64 else "w"
    d, a, b, s, t = (f"{prefix}{r}" for r in (0, 1, 2, 3, 4))
    shared = [("xor_ones_not", [f"movn {s}, #1", f"eor {d}, {a}, {s}"])]
    if width != 64:
        return shared
    other = f"{prefix}5"
    return shared + [
        ("and_xor_union", [f"bic {s}, {a}, {b}", f"eor {t}, {a}, {b}", f"orr {d}, {s}, {t}"]),
        ("and_xor_union", [f"and {s}, {a}, {b}", f"eor {t}, {a}, {a}", f"orr {d}, {s}, {t}"]),
        # A disjunction where the exclusive-or belongs computes the same value,
        # so only the committed rule distinguishes it from the real union.
        ("and_xor_union", [f"and {s}, {a}, {b}", f"orr {t}, {a}, {b}", f"orr {d}, {s}, {t}"]),
        ("or_and_sum", [f"orr {s}, {a}, {b}", f"and {t}, {b}, {b}", f"add {d}, {s}, {t}"]),
        # Each of these swaps one operation class of the union-plus-intersection
        # sum: `(a^b)+(a&b)` and `(a|b)+(a|b)` are both different functions.
        ("or_and_sum", [f"eor {s}, {a}, {b}", f"and {t}, {a}, {b}", f"add {d}, {s}, {t}"]),
        ("or_and_sum", [f"orr {s}, {a}, {b}", f"orr {t}, {a}, {b}", f"add {d}, {s}, {t}"]),
        ("xor_and_sum", [f"eor {s}, {a}, {b}", f"and {t}, {a}, {b}", f"add {d}, {s}, {t}, lsl #2"]),
        ("xor_and_sum", [f"eor {s}, {a}, {b}", f"orr {t}, {a}, {b}", f"add {d}, {s}, {t}, lsl #1"]),
        # The carry sum must agree on both operands, not only on the conjunction.
        (
            "xor_and_sum",
            [f"eor {s}, {a}, {b}", f"and {t}, {a}, {other}", f"add {d}, {s}, {t}, lsl #1"],
        ),
    ]


# An incorrect recovery of the same sequence, written as machine code so the
# independent backend can refute it. These never reach the recovery libraries.
def wrong_recoveries(width):
    prefix = "x" if width == 64 else "w"
    d, a, b = (f"{prefix}{r}" for r in (0, 1, 2))
    return {
        "and_xor_union": [f"orr {d}, {a}, {a}"],
        "or_and_sum": [f"add {d}, {a}, {a}"],
        "xor_and_sum": [f"sub {d}, {a}, {b}"],
        "xor_ones_not": [f"mov {d}, {a}"],
    }


def register_states(rng, width):
    values = [
        (0, 0),
        (MASK64, MASK64),
        (0, MASK64),
        (MASK64, 0),
        (1, MASK64),
        (0xAAAAAAAAAAAAAAAA, 0x5555555555555555),
        (0xFFFFFFFF, 0x100000000),
        (1 << (width - 1), (1 << (width - 1)) - 1),
    ]
    values += [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(8)]
    states = []
    for flags, (a, b) in enumerate(values):
        registers = [rng.getrandbits(64) for _ in range(31)]
        registers[1], registers[2] = a, b
        states.append(State(tuple(registers), (flags % 16) << 28))
    return states


def evaluate(probe, mode, code, state, sp, page, source=0x1000, bias=0x200000000, base=SCRATCH):
    payload = (
        b"NYXMBA01"
        + struct.pack("<37Q", *state.registers, state.nzcv, sp, base, source, bias, len(code))
        + code
        + page
    )
    process = subprocess.run(
        [probe, mode], input=payload, capture_output=True, check=True, timeout=10
    )
    return json.loads(process.stdout)


def matches(output, expected, pc, sp, page):
    return (
        output["completed"]
        and output["pc"] == pc
        and output["sp"] == sp
        and tuple(output["registers"]) == expected.registers
        and output["nzcv"] == expected.nzcv
        and bytes.fromhex(output["memory"]) == page
    )


def register_checks(probe, oracle, rng):
    comparisons = collections.Counter()
    committed = collections.Counter()
    declined = 0
    refuted = 0
    source, bias = 0x1000, 0x200000000
    page = bytes(index % 251 for index in range(4096))
    for width in (32, 64):
        mask = (1 << width) - 1
        for rule, model, text in cases(width):
            code = oracle.assemble(text)
            states = register_states(rng, width)
            native = oracle.run_bytes_many(code, states)
            for state, expected in zip(states, native, strict=True):
                a, b = state.registers[1] & mask, state.registers[2] & mask
                if expected.registers[0] != model(a, b, width):
                    raise AssertionError(
                        ("fixture does not compute the claimed identity", rule, text)
                    )
                sp = rng.getrandbits(64)
                for mode in ("original", "normalized", "simplified"):
                    output = evaluate(probe, mode, code, state, sp, page, source, bias)
                    if not matches(output, expected, source + bias + len(code), sp, page):
                        raise AssertionError((rule, text, mode, "native comparison failed", output))
                    if any(
                        step["events"] or step["fault_address"] is not None
                        for step in output["trace"]
                    ):
                        raise AssertionError(
                            "register-only identity fixture reported a memory effect"
                        )
                    if mode == "simplified":
                        committed.update(edit["rule"] for edit in output["rules"])
                        applied = [edit for edit in output["rules"] if edit["rule"] == rule]
                        if len(applied) != 1:
                            raise AssertionError(
                                (
                                    "identity was not committed exactly once",
                                    rule,
                                    text,
                                    output["rules"],
                                )
                            )
                        if applied[0]["from_revision"] != 0 or applied[0]["to_revision"] != 1:
                            raise AssertionError(
                                "committed edit is not bound to the published revision"
                            )
                    comparisons[mode] += 1
        for rule, text in declines(width):
            code = oracle.assemble(text)
            states = register_states(rng, width)
            native = oracle.run_bytes_many(code, states)
            for state, expected in zip(states, native, strict=True):
                sp = rng.getrandbits(64)
                output = evaluate(probe, "simplified", code, state, sp, page, source, bias)
                if not matches(output, expected, source + bias + len(code), sp, page):
                    raise AssertionError((rule, text, "near miss changed execution", output))
                if any(edit["rule"] == rule for edit in output["rules"]):
                    raise AssertionError(("near miss was accepted as the identity", rule, text))
                declined += 1
        for rule, model, text in cases(width):
            # Prove the comparison above can refute a wrong answer of this shape.
            wrong = oracle.assemble(wrong_recoveries(width)[rule])
            states = register_states(rng, width)
            native = oracle.run_bytes_many(oracle.assemble(text), states)
            mutated = oracle.run_bytes_many(wrong, states)
            if all(
                actual.registers[0] == expected.registers[0]
                for actual, expected in zip(mutated, native, strict=True)
            ):
                raise AssertionError(("incorrect recovery was not refuted", rule, text))
            refuted += 1
    return comparisons, committed, declined, refuted


# A sum identity over a negated operand: the addition it publishes still carries
# `0 - x`, which the existing negation rule then commits. 83 occurrences on the
# development range take this form, so the composition is checked natively too.
def compositions():
    return [
        ("or_and_sum", ["neg x1, x5", "orr x3, x1, x2", "and x4, x1, x2", "add x0, x3, x4"]),
        (
            "xor_and_sum",
            ["neg x1, x5", "eor x3, x1, x2", "and x4, x1, x2", "add x0, x3, x4, lsl #1"],
        ),
        (
            "or_xor_sum",
            ["neg x1, x5", "orr x3, x1, x2", "eor x4, x1, x2", "lsl x3, x3, #1", "sub x0, x3, x4"],
        ),
    ]


def composition_checks(probe, oracle, rng):
    comparisons = 0
    source, bias = 0x1000, 0x200000000
    page = bytes(index % 251 for index in range(4096))
    for rule, text in compositions():
        code = oracle.assemble(text)
        states = register_states(rng, 64)
        native = oracle.run_bytes_many(code, states)
        for state, expected in zip(states, native, strict=True):
            if expected.registers[0] != (state.registers[2] - state.registers[5]) & MASK64:
                raise AssertionError(("composition fixture does not compute b - a", rule))
            sp = rng.getrandbits(64)
            for mode in ("original", "normalized", "simplified"):
                output = evaluate(probe, mode, code, state, sp, page, source, bias)
                if not matches(output, expected, source + bias + len(code), sp, page):
                    raise AssertionError((rule, mode, "composition comparison failed", output))
                if mode == "simplified":
                    at_node = collections.defaultdict(list)
                    for edit in output["rules"]:
                        at_node[edit["node"]].append(edit["rule"])
                    if not any(rules == [rule, "negated_add"] for rules in at_node.values()):
                        raise AssertionError(
                            ("sum identity did not expose the negation", rule, output["rules"])
                        )
                comparisons += 1
    return comparisons


def linear_subexpression_checks(probe, oracle, rng):
    text = [
        "eor x4, x1, x2",
        "mvn x5, x3",
        "and x6, x4, x5",
        "mvn x7, x4",
        "and x8, x7, x3",
        "orr x0, x6, x8",
    ]
    code = oracle.assemble(text)
    wrong = oracle.assemble(["eor x4, x1, x2", "orr x0, x4, x3"])
    states = register_states(rng, 64)
    native = oracle.run_bytes_many(code, states)
    mutated = oracle.run_bytes_many(wrong, states)
    page = bytes(index % 251 for index in range(4096))
    compared, refuted = 0, 0
    for state, expected, incorrect in zip(states, native, mutated, strict=True):
        model = (state.registers[1] ^ state.registers[2] ^ state.registers[3]) & MASK64
        if expected.registers[0] != model:
            raise AssertionError("three-input fixture does not compute XOR")
        for mode in ("original", "normalized", "linear"):
            actual = evaluate(probe, mode, code, state, SCRATCH + 2048, page)
            if not matches(
                actual, expected, 0x200000000 + 0x1000 + len(code), SCRATCH + 2048, page
            ):
                raise AssertionError(("linear subexpression differs from QEMU", mode, actual))
            if mode == "linear" and not any(
                edit["rule"] == "linear_direct" and edit["replacement_op"] == "bit_xor"
                for edit in actual["rules"]
            ):
                raise AssertionError(
                    ("the three-input rule did not propose an edit", actual["rules"])
                )
            compared += 1
        refuted += incorrect.registers[0] != expected.registers[0]
    if not refuted:
        raise AssertionError("incorrect OR mutation did not refute")
    return compared, refuted


def body_state(rng, bias, count, value, flags, base_register=None):
    registers = [rng.getrandbits(64) for _ in range(31)]
    registers[28] = SCRATCH
    registers[19] = GLOBAL_PAGE if base_register is None else base_register
    page = bytearray(rng.getrandbits(8) for _ in range(4096))
    struct.pack_into("<Q", page, COUNT_OFFSET, count)
    struct.pack_into("<Q", page, VALUE_OFFSET, value)
    return ProgramState(
        tuple(registers),
        flags << 28,
        SCRATCH + 2048,
        bytes(rng.getrandbits(8) for _ in range(4096)),
        SCRATCH,
        bias + BODY,
        (GlobalPage(GLOBAL_PAGE, bytes(page)),),
    )


def body_model(state, count, value):
    conjunction = state.registers[10] & state.registers[11]
    shifted = (value << (count & 63)) & MASK64
    return conjunction | shifted


def body_checks(probe, oracle, rng):
    comparisons = collections.Counter()
    mutations = collections.Counter()
    cases_run = 0
    loads = None
    for bias in (0x200000000, 0x600000000):
        for count in (0, 1, 31, 63, 64, MASK64):
            for flags in (0, 5, 10, 15):
                value = rng.getrandbits(64)
                state = body_state(rng, bias, count, value, flags)
                expected = oracle.run(BODY_CODE, state, trusted_fixture=True)
                if not expected.completed or expected.exit_pc != bias + BODY + len(BODY_CODE):
                    raise AssertionError("original body did not reach its declared exit")
                if expected.state.registers[10] != body_model(state, count, value):
                    raise AssertionError(
                        "native body disagrees with the independent identity model"
                    )
                if expected.state.global_pages[0].data != state.global_pages[0].data:
                    raise AssertionError("read-only body unexpectedly changed its global page")
                outputs = {}
                for mode in ("original", "normalized", "simplified"):
                    output = evaluate(
                        probe,
                        mode,
                        BODY_CODE,
                        expected_input(state),
                        state.sp,
                        state.global_pages[0].data,
                        BODY,
                        bias,
                        GLOBAL_PAGE,
                    )
                    if not matches(
                        output,
                        expected.state,
                        expected.exit_pc,
                        state.sp,
                        state.global_pages[0].data,
                    ):
                        raise AssertionError(
                            (mode, count, flags, bias, "body comparison failed", output)
                        )
                    if mode == "simplified":
                        applied = [edit["rule"] for edit in output["rules"]]
                        if applied.count("and_xor_union") != 1:
                            raise AssertionError(("body identity was not committed once", applied))
                    outputs[mode] = output
                    comparisons[mode] += 1
                if not (
                    outputs["original"]["trace"]
                    == outputs["normalized"]["trace"]
                    == outputs["simplified"]["trace"]
                ):
                    raise AssertionError("recovery changed the modeled memory access sequence")
                events = sum(len(step["events"]) for step in outputs["simplified"]["trace"])
                if events != 2 or any(
                    event["write"]
                    for step in outputs["simplified"]["trace"]
                    for event in step["events"]
                ):
                    raise AssertionError(("recovery did not retain both original loads", events))
                loads = min(loads, events) if loads is not None else events
                cases_run += 1
        # The union's own operand must fault before the body publishes a result.
        # Both loads are placed in the guard page below the input, so confinement
        # admits the fixture and the first access is the one that faults.
        state = body_state(
            rng, bias, 1, rng.getrandbits(64), 0, base_register=GLOBAL_PAGE - COUNT_OFFSET - 8
        )
        expected = oracle.run(BODY_CODE, state, trusted_fixture=True)
        if expected.completed or expected.signal != 11 or expected.exit_pc != bias + BODY + 4:
            raise AssertionError("fault control did not stop at the first original load")
        for mode in ("original", "normalized", "simplified"):
            output = evaluate(
                probe,
                mode,
                BODY_CODE,
                expected_input(state),
                state.sp,
                state.global_pages[0].data,
                BODY,
                bias,
                GLOBAL_PAGE,
            )
            if (
                output["completed"]
                or output["pc"] != expected.exit_pc
                or tuple(output["registers"]) != expected.state.registers
            ):
                raise AssertionError((mode, "fault control diverged", output))
            comparisons[mode] += 1
        cases_run += 1
        for name, offset, word in (
            ("wrong_union_operand", 24, 0xAA0C018A),
            ("conjunction_instead_of_exclusive", 20, 0x8A0B014A),
        ):
            state = body_state(rng, bias, 5, rng.getrandbits(64), 0)
            expected = oracle.run(BODY_CODE, state, trusted_fixture=True)
            mutant = bytearray(BODY_CODE)
            struct.pack_into("<I", mutant, offset, word)
            changed = oracle.run(bytes(mutant), state, trusted_fixture=True)
            if changed.state.registers == expected.state.registers:
                raise AssertionError(("body mutation was not refuted natively", name))
            mutations[name] += 1
        # The union's two operands are disjoint, so replacing the combining ORR
        # with EOR computes the same value. A combiner-only mutation therefore
        # cannot refute this body; recovery must still decline the changed shape.
        state = body_state(rng, bias, 5, rng.getrandbits(64), 0)
        expected = oracle.run(BODY_CODE, state, trusted_fixture=True)
        equivalent = bytearray(BODY_CODE)
        struct.pack_into("<I", equivalent, 24, 0xCA0A018A)
        changed = oracle.run(bytes(equivalent), state, trusted_fixture=True)
        if changed.state.registers != expected.state.registers:
            raise AssertionError("disjoint-operand combiner control unexpectedly diverged")
        output = evaluate(
            probe,
            "simplified",
            bytes(equivalent),
            expected_input(state),
            state.sp,
            state.global_pages[0].data,
            BODY,
            bias,
            GLOBAL_PAGE,
        )
        if not matches(
            output, changed.state, changed.exit_pc, state.sp, state.global_pages[0].data
        ):
            raise AssertionError("equivalent combiner control changed execution")
        if any(edit["rule"] == "and_xor_union" for edit in output["rules"]):
            raise AssertionError("union rule was claimed for an exclusive-or combiner")
        mutations["equivalent_combiner_controls"] += 1
        # Breaking the conjunction must make the rule decline, not rewrite.
        state = body_state(rng, bias, 5, rng.getrandbits(64), 0)
        mutant = bytearray(BODY_CODE)
        struct.pack_into("<I", mutant, 16, 0x8A2B014C)
        expected = oracle.run(bytes(mutant), state, trusted_fixture=True)
        output = evaluate(
            probe,
            "simplified",
            bytes(mutant),
            expected_input(state),
            state.sp,
            state.global_pages[0].data,
            BODY,
            bias,
            GLOBAL_PAGE,
        )
        if not matches(
            output, expected.state, expected.exit_pc, state.sp, state.global_pages[0].data
        ):
            raise AssertionError("body near miss changed execution")
        if any(edit["rule"] == "and_xor_union" for edit in output["rules"]):
            raise AssertionError("body near miss was accepted as the identity")
        mutations["conjunction_near_miss_declined"] += 1
    return comparisons, mutations, cases_run, loads


def expected_input(state):
    return State(state.registers, state.nzcv)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: a64_boolean_identity_diff.py MBA_PROBE")
    probe = sys.argv[1]
    if len(BODY_CODE) != 28 or hashlib.sha256(BODY_CODE).hexdigest() != BODY_SHA256:
        raise AssertionError("original development body provenance changed")
    oracle, program = Oracle(), ProgramOracle()
    rng = random.Random(0x0B001EA4)
    comparisons, committed, declined, refuted = register_checks(probe, oracle, rng)
    compositions_checked = composition_checks(probe, oracle, rng)
    linear_compared, linear_refuted = linear_subexpression_checks(probe, oracle, rng)
    body_comparisons, mutations, body_cases, body_loads = body_checks(probe, program, rng)
    print(
        json.dumps(
            {
                "register_comparisons": dict(comparisons),
                "register_committed_rules": dict(committed),
                "near_miss_declines": declined,
                "incorrect_recovery_shapes_refuted": refuted,
                "negation_composition_comparisons": compositions_checked,
                "linear_subexpression_comparisons": linear_compared,
                "linear_subexpression_mutations_refuted": linear_refuted,
                "body_source_address": BODY,
                "body_source_bytes": len(BODY_CODE),
                "body_source_sha256": BODY_SHA256,
                "body_cases": body_cases,
                "body_comparisons": dict(body_comparisons),
                "body_mutations": dict(mutations),
                "body_loads_retained": body_loads,
                "python_integer_reference_checked": True,
                "scope": "closed Boolean identities over one straight-line region; no whole-function claim",
                "hardware_checked": False,
                "memory_events_independently_checked": False,
                "tools": oracle.versions(),
                "program_tools": program.versions(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
