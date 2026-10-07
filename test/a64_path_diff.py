#!/usr/bin/env python3
import collections
import dataclasses
import json
import pathlib
import struct
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "oracle"))
from tools.oracle.sparse import MAGIC, Fragment, SparseOracle
from test_sparse import TAIL, DISPATCH, EXITS, fixture


def evaluate(probe, mode, fragments, entry, exits, state):
    payload = (
        MAGIC
        + struct.pack(
            "<39Q",
            *state.registers,
            state.nzcv,
            state.sp,
            state.scratch_base,
            state.load_bias,
            entry,
            len(fragments),
            len(exits),
            len(state.global_pages),
        )
        + b"".join(
            struct.pack("<QQ", fragment.source_address, len(fragment.code)) + fragment.code
            for fragment in fragments
        )
        + struct.pack("<" + "Q" * len(exits), *exits)
        + state.memory
        + b"".join(struct.pack("<Q", page.address) + page.data for page in state.global_pages)
    )
    result = subprocess.run(
        [probe, mode], input=payload, capture_output=True, timeout=15, check=True
    )
    return json.loads(result.stdout)


def same(expected, actual):
    state = expected.state
    if (
        actual["completed"] != expected.completed
        or actual["pc"] != expected.exit_pc
        or tuple(actual["registers"]) != state.registers
        or actual["nzcv"] != state.nzcv
        or actual["sp"] != state.sp
        or bytes.fromhex(actual["memory"]) != state.memory
        or [(page["address"], bytes.fromhex(page["data"])) for page in actual["global_pages"]]
        != [(page.address, page.data) for page in state.global_pages]
    ):
        return False
    faults = [
        step["fault_address"] for step in actual["trace"] if step["fault_address"] is not None
    ]
    return faults == ([] if expected.completed else [expected.fault_address])


def main():
    probe = sys.argv[1]
    oracle = SparseOracle()
    counts = collections.Counter()
    refuted = collections.Counter()
    positive_controls = collections.Counter()
    case_families = collections.Counter()
    cases = []
    synthetic_tail = Fragment(
        TAIL.source_address, struct.pack("<I", 0x52800009 | (648 << 5)) + TAIL.code[4:]
    )
    for bias in (0x200000000, 0x600000000):
        cases.append(
            (
                "synthetic_out_of_range_predecessor",
                (synthetic_tail, DISPATCH),
                TAIL.source_address,
                fixture(bias, index=648),
            )
        )
        for target in EXITS[:2]:
            cases.append(
                ("tail", (TAIL, DISPATCH), TAIL.source_address, fixture(bias, target=target))
            )
        for index in (0, 358, 647, 648, 0xFFFFFFFF):
            cases.append(
                ("dispatcher", (DISPATCH,), DISPATCH.source_address, fixture(bias, index, EXITS[1]))
            )
        cases.append(
            (
                "table_fault",
                (TAIL, DISPATCH),
                TAIL.source_address,
                dataclasses.replace(fixture(bias), global_pages=()),
            )
        )
        state = fixture(bias)
        registers = list(state.registers)
        registers[19] = state.scratch_base - 4096
        cases.append(
            (
                "fault",
                (TAIL, DISPATCH),
                TAIL.source_address,
                dataclasses.replace(state, registers=tuple(registers)),
            )
        )
    for family, fragments, entry, state in cases:
        case_families[family] += 1
        expected = oracle.run(fragments, entry, EXITS, state, trusted_fixture=True)
        if family == "table_fault" and (
            expected.completed
            or expected.signal != 11
            or expected.exit_pc != state.load_bias + 0x752E64
            or expected.fault_address != state.load_bias + 0x1F8EB0
        ):
            raise AssertionError("table fault fixture did not reach the retained dispatcher load")
        outputs = {}
        for mode in ("original", "normalized", "simplified"):
            actual = evaluate(probe, mode, fragments, entry, EXITS, state)
            if not same(expected, actual):
                raise AssertionError((family, mode, state, expected, actual))
            outputs[mode] = actual
            counts[mode] += 1
        if (
            not outputs["original"]["trace"]
            == outputs["normalized"]["trace"]
            == outputs["simplified"]["trace"]
        ):
            raise AssertionError(
                "path rewrite changed modeled access sequence/source correspondence"
            )
        if family in ("tail", "table_fault") and not outputs["simplified"]["forwarding_facts"]:
            raise AssertionError("tail state store/load was never forwarded")
        if family == "synthetic_out_of_range_predecessor":
            simplified = outputs["simplified"]
            if (
                not expected.completed
                or expected.exit_pc != state.load_bias + EXITS[2]
                or not simplified["forwarding_facts"]
                or not simplified["memory_edits"]
                or not simplified["arithmetic_edits"]
                or len(simplified["trace"]) != 8
            ):
                raise AssertionError(
                    "synthetic predecessor did not diverge after forwarding and folding"
                )
        mutant = evaluate(probe, "snapshot_mutant", fragments, entry, EXITS, state)
        if mutant["trace"] != outputs["original"]["trace"]:
            raise AssertionError("snapshot mutation removed or changed retained memory accesses")
        control = (
            family
            if not expected.completed
            else "guard_default"
            if expected.exit_pc == state.load_bias + EXITS[2]
            else "matching_snapshot"
            if expected.exit_pc == state.load_bias + EXITS[0]
            else None
        )
        if control is not None:
            if not same(expected, mutant):
                raise AssertionError(("snapshot mutation positive control differs", control))
            positive_controls[control] += 1
        else:
            if same(expected, mutant):
                raise AssertionError("executed snapshot target mutation survived")
            refuted["snapshot_target_ssa_mutations"] += 1
        if (
            family == "dispatcher"
            and expected.completed
            and struct.unpack_from("<I", state.memory, 84)[0] <= 647
        ):
            altered = bytearray(DISPATCH.code)
            altered[8] ^= 1
            changed = evaluate(
                probe,
                "original",
                (Fragment(DISPATCH.source_address, bytes(altered)),),
                entry,
                EXITS,
                state,
            )
            if same(expected, changed):
                raise AssertionError("unsigned guard polarity mutation survived")
            refuted["guard_instruction_mutations"] += 1
    print(
        json.dumps(
            {
                "native_cases": len(cases),
                "case_families": dict(case_families),
                "comparisons": dict(counts),
                "refutations": dict(refuted),
                "snapshot_positive_controls": dict(positive_controls),
                "scope": "declared sparse itinerary and runtime table snapshots; not whole-function recovery",
                "table_invariance_proved": False,
                "memory_events_independently_checked": False,
                "hardware_checked": False,
                "tools": oracle.versions(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
