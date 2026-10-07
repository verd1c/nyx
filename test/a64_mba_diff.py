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
from tools.oracle.program import GlobalPage, ProgramOracle, ProgramState

# Original bytes from the already-exposed development diagnostic, not a
# reassembly or a fixture synthesized from the proposed recovered expression.
SOURCE = 0xC687B0
GLOBAL = 0x1CB8358
K = 0x8EA4F57646BF7523
MASK64 = (1 << 64) - 1
CODE = bytes.fromhex(
    "8982009029610d91290140f9e90309cb6ba48ed2ebd7a8f2"
    "cbaedef28bd4f1f22b010baa6bf97fd36ca48ed2ecd7a8f2"
    "ccaedef28cd4f1f229010cca690109cb0b0180d2"
)


def evaluate(probe, mode, code, state, bias):
    page = state.global_pages[0]
    payload = (
        b"NYXMBA01"
        + struct.pack(
            "<37Q", *state.registers, state.nzcv, state.sp, page.address, SOURCE, bias, len(code)
        )
        + code
        + page.data
    )
    process = subprocess.run(
        [probe, mode], input=payload, capture_output=True, check=True, timeout=10
    )
    return json.loads(process.stdout)


def same(expected, actual):
    return (
        expected.completed == actual["completed"]
        and expected.exit_pc == actual["pc"]
        and expected.state.registers == tuple(actual["registers"])
        and expected.state.nzcv == actual["nzcv"]
        and expected.state.sp == actual["sp"]
        and expected.state.global_pages[0].data == bytes.fromhex(actual["memory"])
    )


def main():
    probe = sys.argv[1]
    if (
        hashlib.sha256(CODE).hexdigest()
        != "fddeb461b9f105230c22f63acd1ce817099d63039bc9705846821d3619c63dd1"
    ):
        raise AssertionError("original development slice bytes changed")
    oracle = ProgramOracle()
    rng = random.Random(0xC687B0)
    values = [0, 1, K, K - 1, MASK64, 1 << 63, (1 << 63) - 1] + [
        rng.getrandbits(64) for _ in range(25)
    ]
    comparisons = collections.Counter()
    refuted = 0
    matching_snapshot_cases = 0
    rule_counts = None
    for bias in (0x200000000, 0x600000000):
        states = []
        for index, value in enumerate(values):
            registers = [rng.getrandbits(64) for _ in range(31)]
            registers[28] = 0x100000000
            page = bytearray(rng.getrandbits(8) for _ in range(4096))
            struct.pack_into("<Q", page, GLOBAL % 4096, value)
            states.append(
                ProgramState(
                    tuple(registers),
                    (index % 16) << 28,
                    0x100000800,
                    bytes(rng.getrandbits(8) for _ in range(4096)),
                    0x100000000,
                    bias + SOURCE,
                    (GlobalPage(bias + (GLOBAL & ~4095), bytes(page)),),
                )
            )
        native = oracle.run_many(CODE, states, trusted_fixture=True)
        for value, state, expected in zip(values, states, native, strict=True):
            if not expected.completed or expected.state.registers[9] != (K - value) & MASK64:
                raise AssertionError(
                    "native source bytes do not demonstrate the claimed arithmetic relation"
                )
            if expected.state.memory != state.memory:
                raise AssertionError("diagnostic unexpectedly changed scratch memory")
            outputs = {}
            for mode in ("original", "normalized", "simplified"):
                actual = evaluate(probe, mode, CODE, state, bias)
                if not same(expected, actual):
                    raise AssertionError((mode, value, expected, actual))
                outputs[mode] = actual
                comparisons[mode] += 1
            if not (
                outputs["original"]["trace"]
                == outputs["normalized"]["trace"]
                == outputs["simplified"]["trace"]
            ):
                raise AssertionError(
                    "normalization/recovery changed modeled instruction access boundaries"
                )
            rules = collections.Counter(edit["rule"] for edit in outputs["simplified"]["rules"])
            if rules["or_xor_sum"] != 1 or rules["negated_add"] != 1:
                raise AssertionError(("diagnostic MBA was not actually rewritten", rules))
            if any(
                edit["from_revision"] != 0 or edit["to_revision"] != 1
                for edit in outputs["simplified"]["rules"]
            ):
                raise AssertionError("rewrite journal is not tied to the committed revision")
            if rule_counts is None:
                rule_counts = dict(rules)
            elif dict(rules) != rule_counts:
                raise AssertionError("recovery unexpectedly depends on concrete global contents")
            # The incorrect literal-one recovery retains the original load and
            # effects, but replaces only the final arithmetic result instruction.
            mutant = CODE[:60] + bytes.fromhex("290080d2") + CODE[64:]
            changed = evaluate(probe, "original", mutant, state, bias)
            if value == K - 1:
                if not same(expected, changed):
                    raise AssertionError("snapshot-specific matching control should match")
                matching_snapshot_cases += 1
            else:
                if same(expected, changed):
                    raise AssertionError("unjustified snapshot constant was not refuted")
                refuted += 1
    print(
        json.dumps(
            {
                "native_states": len(values) * 2,
                "comparisons": dict(comparisons),
                "committed_rules_per_region": rule_counts,
                "snapshot_constant_mutations_refuted": refuted,
                "snapshot_specific_matching_controls": matching_snapshot_cases,
                "source_address": SOURCE,
                "source_bytes": len(CODE),
                "unknown_global_preserved": True,
                "tools": oracle.versions(),
                "hardware_checked": False,
                "memory_events_independently_checked": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
