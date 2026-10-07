#!/usr/bin/env python3
import collections
import copy
import dataclasses
import hashlib
import json
import pathlib
import random
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from a64_regions_diff import (
    SparseOracle,
    Fragment,
    evaluate,
    same,
    consume_facts,
    fact_matches_native,
    check_control_wrapper,
)
from tools.oracle.sparse import SparseState

# Verified against the original development ELF's PT_LOAD bytes, not reassembled.
BODY = Fragment(
    0x747FC0,
    bytes.fromhex(
        "68aa4bf9080540f969ba4bf96aa64bf94a25c99ae903292a08f97fd3"
        "0821c99a08010aaa688e0bf9a80e8052685600b9688e4bf9685a05f9eb240014"
    ),
)
BODY_SHA256 = "b9dea326cb5e292c614c7114779861021a3dfc939328045f35ef84c5ac61d583"
EXIT = 0x7513A4
MASK64 = (1 << 64) - 1


def fixture(count, flags, bias):
    rng = random.Random(0x747FC0 + count + flags)
    registers = [rng.getrandbits(64) for _ in range(31)]
    base = 0x100000000
    registers[19] = base
    memory = bytearray(rng.getrandbits(8) for _ in range(8192))
    struct.pack_into("<Q", memory, 5968, base + 128)
    struct.pack_into("<Q", memory, 136, 0x4123456789ABCDEF)
    struct.pack_into("<Q", memory, 6000, count)
    struct.pack_into("<Q", memory, 5960, 0xF0E1D2C3B4A59687)
    return SparseState(tuple(registers), flags << 28, 0x123456789, bytes(memory), base, bias, ())


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: a64_variable_shift_regions_diff.py REGIONS_PROBE")
    probe = sys.argv[1]
    if len(BODY.code) != 60 or hashlib.sha256(BODY.code).hexdigest() != BODY_SHA256:
        raise AssertionError("original development body provenance changed")
    oracle = SparseOracle()
    cases = []
    for bias in (0x200000000, 0x600000000):
        for count in (0, 1, 31, 32, 63, 64, MASK64):
            for flags in (0, 5, 10, 15):
                cases.append(("completed", count, fixture(count, flags, bias)))
        state = fixture(1, 10, bias)
        registers = list(state.registers)
        registers[19] = state.scratch_base - 4096 - 5968
        cases.append(
            ("first_load_fault", 1, dataclasses.replace(state, registers=tuple(registers)))
        )
        memory = bytearray(state.memory)
        struct.pack_into("<Q", memory, 5968, state.scratch_base - 4096 - 8)
        cases.append(("second_load_fault", 1, dataclasses.replace(state, memory=bytes(memory))))
    comparisons = collections.Counter()
    families = collections.Counter()
    fact_checks = 0
    mutations = collections.Counter()
    for family, count, state in cases:
        families[family] += 1
        expected = oracle.run((BODY,), BODY.source_address, (EXIT,), state, trusted_fixture=True)
        if family == "completed":
            if not expected.completed or expected.exit_pc != state.load_bias + EXIT:
                raise AssertionError("original body did not reach its declared direct exit")
            if struct.unpack_from("<I", expected.state.memory, 84)[0] != 117:
                raise AssertionError("original body did not publish its dispatcher state")
            if (
                expected.state.memory[5912:5920] != expected.state.memory[2736:2744]
                or expected.state.memory[5912:5920] == state.memory[5912:5920]
            ):
                raise AssertionError("original store/load fixture lacks a changed observable value")
        else:
            fault_pc = BODY.source_address + (0 if family == "first_load_fault" else 4)
            if (
                expected.completed
                or expected.signal != 11
                or expected.exit_pc != state.load_bias + fault_pc
                or expected.fault_address != state.scratch_base - 4096
                or expected.state.memory != state.memory
            ):
                raise AssertionError("fault control did not stop at its reviewed load boundary")
        outputs = {}
        for mode in ("original", "normalized", "simplified", "control_recovered"):
            output = evaluate(probe, mode, (BODY,), BODY.source_address, (EXIT,), state)
            if not same(expected, output):
                raise AssertionError(
                    (
                        family,
                        count,
                        state.nzcv,
                        state.load_bias,
                        mode,
                        "native full-state comparison failed",
                        expected,
                        output,
                    )
                )
            if (
                output["selected_sources"]
                != list(range(BODY.source_address, BODY.source_address + 60, 4))
                or output["entry_candidates"] != 1
                or output["selection_policy"] != "longest_candidate_for_entry"
                or not output["cfg_ssa_unchanged"]
                or not output["cfg_topology_unchanged"]
            ):
                raise AssertionError(
                    "automatic selection failed to join the complete original body"
                )
            path = output["analyzed_path"]
            if (
                len(path["sources"]) != 15
                or b"".join(bytes.fromhex(source["bytes"]) for source in path["sources"])
                != BODY.code
            ):
                raise AssertionError("selected IR changed original source bytes or boundaries")
            original_loads = sum(
                node["op"] == "load" for source in path["sources"] for node in source["nodes"]
            )
            retained_loads = sum(node["op"] == "load" for node in path["nodes"])
            if original_loads != 5 or retained_loads != original_loads:
                raise AssertionError("recovery erased an original faulting load")
            facts = output["control"]
            if facts["proved_divergence"] is not None or len(facts["boundaries"]) != 15:
                raise AssertionError("straight-line body acquired a spurious divergence")
            for record in facts["boundaries"]:
                if len(record["edges"]) != 1:
                    raise AssertionError("straight-line body acquired invented alternatives")
                edge = record["edges"][0]
                target = EXIT if record["boundary"] == 14 else record["source_address"] + 4
                if (
                    edge["target_kind"] != "image_location"
                    or edge["target_address"] != target
                    or edge["condition"] is not None
                    or record["load_dependencies"]
                ):
                    raise AssertionError("control facts invented an indirect target or guard")
            consumed = consume_facts(output, state)
            if not fact_matches_native(consumed, expected):
                raise AssertionError("extracted control facts disagree with native execution")
            if len(consumed[3]) != (
                15 if family == "completed" else 0 if family == "first_load_fault" else 1
            ):
                raise AssertionError("fact consumer assumed success past a faulting instruction")
            fact_checks += 1
            if mode in ("simplified", "control_recovered"):
                if output["forwarding_facts"] < 1 or output["memory_edits"] < 1:
                    raise AssertionError("real store-to-load forwarding was not exercised")
                if output["control_edits"] != 0:
                    raise AssertionError("direct branch was reported as a removed conditional")
            outputs[mode] = output
            comparisons[mode] += 1
        check_control_wrapper(outputs["simplified"], outputs["control_recovered"])
        if any(output["trace"] != outputs["original"]["trace"] for output in outputs.values()):
            raise AssertionError("recovery changed the modeled memory access sequence")
        if family == "completed":
            wrong_facts = copy.deepcopy(outputs["simplified"]["control"])
            wrong_facts["boundaries"][-1]["edges"][0]["target_address"] += 4
            if fact_matches_native(
                consume_facts(outputs["simplified"], state, wrong_facts, False), expected
            ):
                raise AssertionError("corrupted exit fact survived native comparison")
            mutations["destination_facts_refuted"] += 1
        if family == "completed" and count == 1 and state.nzcv == 0:
            changed = bytearray(BODY.code)
            # Change the real LSLV at 0x747fdc to LSRV, preserving all accesses.
            word = struct.unpack_from("<I", changed, 28)[0]
            struct.pack_into("<I", changed, 28, word ^ (1 << 10))
            mutant = evaluate(
                probe,
                "original",
                (Fragment(BODY.source_address, bytes(changed)),),
                BODY.source_address,
                (EXIT,),
                state,
            )
            if same(expected, mutant):
                raise AssertionError(
                    "wrong variable-shift direction survived full-state comparison"
                )
            mutations["instruction_direction_refuted"] += 1
    print(
        json.dumps(
            {
                "native_cases": len(cases),
                "case_families": dict(families),
                "comparisons": dict(comparisons),
                "control_fact_comparisons": fact_checks,
                "mutations": dict(mutations),
                "source_address": BODY.source_address,
                "source_bytes": len(BODY.code),
                "source_sha256": BODY_SHA256,
                "selected_source_groups": 15,
                "scratch_bytes": 8192,
                "declared_exit": EXIT,
                "selection_input": "original source population and entry only; no itinerary supplied",
                "scope": "complete development body through its direct exit; no downstream dispatcher claim",
                "original_cfg_unchanged": True,
                "whole_function_recovery": False,
                "memory_events_independently_checked": False,
                "hardware_checked": False,
                "tools": oracle.versions(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
