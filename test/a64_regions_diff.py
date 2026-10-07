#!/usr/bin/env python3
import collections
import copy
import dataclasses
import json
import hashlib
import pathlib
import struct
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "oracle"))
from tools.oracle.sparse import MAGIC, MAGIC_LARGE, Fragment, SparseOracle
from test_sparse import TAIL, DISPATCH, EXITS, fixture, FULL_PREDECESSOR, full_fixture


def evaluate(probe, mode, fragments, entry, exits, state):
    large = len(state.memory) == 8192
    header = (
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
    if large:
        header += (len(state.memory),)
    payload = (
        (MAGIC_LARGE if large else MAGIC)
        + struct.pack("<" + str(len(header)) + "Q", *header)
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


def consume_facts(output, state, facts=None, check_claims=True):
    path = output["analyzed_path"]
    facts = output["control"] if facts is None else facts
    if (
        facts["path_revision"] != path["revision"]
        or facts["proof_scope"] != "successful_itinerary_prefix"
    ):
        raise AssertionError("fact references lack the analyzed path revision/prefix scope")
    if facts["total_boundaries"] != len(path["boundaries"]):
        raise AssertionError("fact boundary extent changed")
    records = {record["boundary"]: record for record in facts["boundaries"]}
    if facts["proved_divergence"] is not None and max(records) != facts["proved_divergence"]:
        raise AssertionError("facts extend beyond their proved prefix divergence")
    if check_claims:
        for index, record in records.items():
            transfer = path["boundaries"][index]["transfer"]
            roots = (
                []
                if transfer is None
                else [
                    value
                    for key, value in transfer.items()
                    if key in ("target", "condition", "alternative", "continuation")
                    and value is not None
                ]
            )
            dependencies = set()
            visited = set()
            pending = list(roots)
            while pending:
                value = pending.pop()
                if value in visited:
                    continue
                visited.add(value)
                node = path["nodes"][value]
                if node["op"] == "load":
                    dependencies.add(value)
                pending.extend(node["inputs"])
            if record["load_dependencies"] != sorted(dependencies):
                raise AssertionError(
                    "control fact load provenance differs from its complete structural dependency DAG"
                )
            if record["transfer_target"] != (None if transfer is None else transfer["target"]):
                raise AssertionError("fact replaced original transfer target provenance")
    cells = dict(enumerate(state.registers))
    cells[31] = state.sp
    cells.update({32 + i: (state.nzcv >> (31 - i)) & 1 for i in range(4)})
    memory = {state.scratch_base + i: byte for i, byte in enumerate(state.memory)}
    writable = set(memory)
    for page in state.global_pages:
        memory.update({page.address + i: byte for i, byte in enumerate(page.data)})
    values = []
    consumed = []
    selected_targets = {}
    mask64 = (1 << 64) - 1
    for index, boundary in enumerate(path["boundaries"]):
        source = path["sources"][index]["source_address"]
        entry = dict(cells)
        for node in path["nodes"][
            boundary["first_node"] : boundary["first_node"] + boundary["node_count"]
        ]:
            args = [values[value] for value in node["inputs"]]
            op = node["op"]
            width = node["width"]
            if op == "constant":
                value = node["immediate"]
            elif op == "image_address":
                value = state.load_bias + node["immediate"]
            elif op == "read":
                value = entry[node["storage"]]
            elif op == "write":
                cells[node["storage"]] = args[0]
                value = 0
            elif op == "add":
                value = args[0] + args[1]
            elif op == "sub":
                value = args[0] - args[1]
            elif op == "mul":
                value = args[0] * args[1]
            elif op == "bit_and":
                value = args[0] & args[1]
            elif op == "bit_or":
                value = args[0] | args[1]
            elif op == "bit_xor":
                value = args[0] ^ args[1]
            elif op == "bit_not":
                value = ~args[0]
            elif op == "extract":
                value = args[0] >> node["immediate"]
            elif op == "zext":
                value = args[0]
            elif op == "select":
                value = args[1] if args[0] else args[2]
            elif op == "equal":
                value = int(args[0] == args[1])
            elif op == "unsigned_less":
                value = int(args[0] < args[1])
            elif op == "signed_less":
                bits = path["nodes"][node["inputs"][0]]["width"]
                sign = 1 << (bits - 1)
                value = int((args[0] ^ sign) < (args[1] ^ sign))
            elif op in ("shl", "lshr", "ashr"):
                bits = path["nodes"][node["inputs"][0]]["width"]
                count = args[1]
                if op == "ashr":
                    signed = args[0] - (1 << bits) if args[0] >> (bits - 1) else args[0]
                    value = signed >> min(count, bits)
                elif count >= bits:
                    value = 0
                else:
                    value = args[0] << count if op == "shl" else args[0] >> count
            elif op in ("udiv", "sdiv", "umulh", "smulh", "clz", "rbit"):
                bits = width
                signed = lambda x: x - (1 << bits) if x >> (bits - 1) else x
                if op == "udiv":
                    value = args[0] // args[1] if args[1] else 0
                elif op == "sdiv":
                    a, b = signed(args[0]), signed(args[1])
                    value = 0 if b == 0 else (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
                elif op == "umulh":
                    value = (args[0] * args[1]) >> bits
                elif op == "smulh":
                    value = (signed(args[0]) * signed(args[1])) >> bits
                elif op == "clz":
                    value = bits - args[0].bit_length()
                else:
                    value = int(format(args[0], "0%db" % bits)[::-1], 2)
            elif op in ("load", "store"):
                address = args[0]
                size = width // 8
                allowed = writable if op == "store" else memory
                missing = next(
                    (address + i for i in range(size) if address + i not in allowed), None
                )
                if missing is not None:
                    return (
                        False,
                        (source + state.load_bias) & mask64,
                        missing,
                        consumed,
                        selected_targets,
                    )
                order = node["access"]["byte_order"]
                if op == "load":
                    value = int.from_bytes(bytes(memory[address + i] for i in range(size)), order)
                else:
                    memory.update(
                        {address + i: byte for i, byte in enumerate(args[1].to_bytes(size, order))}
                    )
                    value = 0
            else:
                raise AssertionError(("unmodeled independent fact dependency", op))
            if node["id"] != len(values):
                raise AssertionError("noncontiguous SSA ids")
            values.append(value & ((1 << width) - 1))
        for write in boundary["writes"]:
            cells[write["storage"]] = values[write["value"]]
        if index not in records:
            raise AssertionError("reached boundary lacks prefix-scoped facts")
        record = records[index]
        if record["source_address"] != source:
            raise AssertionError("fact source mismatch")
        consumed.append(index)
        targets = []
        for edge in record["edges"]:
            if edge["role"] == "potential_return":
                continue
            condition = None if edge["condition"] is None else bool(values[edge["condition"]])
            if (
                check_claims
                and edge["known_condition"] is not None
                and edge["known_condition"] != condition
            ):
                raise AssertionError("claimed constant guard disagrees with runtime dependency")
            if condition is not None and condition != edge["when"]:
                continue
            if edge["target_kind"] == "image_location":
                target = (state.load_bias + edge["target_address"]) & mask64
            elif edge["target_kind"] == "absolute_runtime":
                target = edge["target_address"]
            elif edge["target_kind"] == "unknown":
                target = values[edge["target_value"]]
            else:
                raise AssertionError("invalid target domain")
            targets.append(target)
        if len(targets) != 1:
            raise AssertionError("facts did not select exactly one executed successor")
        target = targets[0]
        selected_targets[index] = target
        expected = record["expected_image_successor"]
        matches = expected is not None and target == (expected + state.load_bias) & mask64
        if check_claims and (
            (record["expected_match"] == "always" and not matches)
            or (record["expected_match"] == "never" and matches)
        ):
            raise AssertionError(
                "claimed itinerary match disagrees with native-bound dependency evaluation"
            )
        if expected is None or not matches:
            return (True, target, None, consumed, selected_targets)
    raise AssertionError("facts did not terminate their selected region")


def fact_matches_native(result, expected):
    return result[:3] == (
        expected.completed,
        expected.exit_pc,
        None if expected.completed else expected.fault_address,
    )


def check_control_wrapper(basis, output):
    path = output["analyzed_path"]
    original = basis["analyzed_path"]
    rewrites = path["control_rewrites"]
    if (
        path["kind"] != "nyx.ir.recovered_path"
        or path["basis_revision"] != original["revision"]
        or path["revision"] != original["revision"] + bool(rewrites)
        or path["sources"] != original["sources"]
        or path["nodes"] != original["nodes"]
    ):
        raise AssertionError("control rewrite changed basis state/effects/source provenance")
    expected = copy.deepcopy(original["boundaries"])
    for witness in rewrites:
        index = witness["boundary"]
        condition = path["nodes"][witness["condition"]]
        if (
            witness["original"] != original["boundaries"][index]["transfer"]
            or witness["original"]["kind"] != "conditional"
            or witness["original"]["condition"] != witness["condition"]
            or condition["op"] != "constant"
            or condition["width"] != 1
            or witness["condition_value"] != bool(condition["immediate"] & 1)
            or witness["from_revision"] != original["revision"]
            or witness["to_revision"] != path["revision"]
        ):
            raise AssertionError("control witness lacks literal-condition and revision provenance")
        replacement = witness["replacement"]
        target = witness["original"]["target" if witness["condition_value"] else "alternative"]
        if (
            replacement["kind"] != "jump"
            or replacement["target"] != target
            or any(
                replacement.get(key) is not None
                for key in ("condition", "alternative", "continuation")
            )
        ):
            raise AssertionError("effective jump selected the wrong witnessed arm")
        expected[index]["transfer"] = replacement
    if path["boundaries"] != expected or output["control_edits"] != len(rewrites):
        raise AssertionError("effective transfer overrides changed other boundary effects")


def main():
    probe = sys.argv[1]
    oracle = SparseOracle()
    counts = collections.Counter()
    refuted = collections.Counter()
    positive_controls = collections.Counter()
    case_families = collections.Counter()
    fact_checks = collections.Counter()
    control_checks = collections.Counter()
    cases = []
    full_hash = hashlib.sha256(FULL_PREDECESSOR.code).hexdigest()
    if full_hash != "48705b25a687fb17ca141111a41483ae620dddc89d03ee73b7845d3b117fe3f6":
        raise AssertionError("full development predecessor provenance changed")
    decoy = Fragment(0x74A000, bytes.fromhex("1f2003d5"))
    synthetic_tail = Fragment(
        TAIL.source_address, struct.pack("<I", 0x52800009 | (648 << 5)) + TAIL.code[4:]
    )
    for bias in (0x200000000, 0x600000000):
        for target in EXITS[:2]:
            for bit in (0, 1):
                cases.append(
                    (
                        "full_predecessor",
                        (FULL_PREDECESSOR, DISPATCH),
                        FULL_PREDECESSOR.source_address,
                        full_fixture(bias, target, bit),
                    )
                )
        cases.append(
            (
                "full_table_fault",
                (FULL_PREDECESSOR, DISPATCH),
                FULL_PREDECESSOR.source_address,
                dataclasses.replace(full_fixture(bias), global_pages=()),
            )
        )
        full = full_fixture(bias)
        registers = list(full.registers)
        registers[23] = full.scratch_base - 4096 - 868
        cases.append(
            (
                "full_prefix_fault",
                (FULL_PREDECESSOR, DISPATCH),
                FULL_PREDECESSOR.source_address,
                dataclasses.replace(full, registers=tuple(registers)),
            )
        )
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
        # Supply an unordered population and an unrelated fragment. Only entry,
        # never an itinerary, is passed to automatic region selection. Include
        # the predecessor even for dispatcher entry to catch leaked entry facts.
        predecessor = fragments[0] if entry != DISPATCH.source_address else TAIL
        population = (DISPATCH, decoy, predecessor)
        expected = oracle.run(population, entry, EXITS, state, trusted_fixture=True)
        if family in ("table_fault", "full_table_fault") and (
            expected.completed
            or expected.signal != 11
            or expected.exit_pc != state.load_bias + 0x752E64
            or expected.fault_address != state.load_bias + 0x1F8EB0
        ):
            raise AssertionError("table fault fixture did not reach the retained dispatcher load")
        if family == "full_prefix_fault" and (
            expected.completed
            or expected.signal != 11
            or expected.exit_pc != state.load_bias + 0x751378
            or expected.fault_address != state.scratch_base - 4096
        ):
            raise AssertionError("full prefix fault did not preserve the original LDRB boundary")
        outputs = {}
        for mode in ("original", "normalized", "simplified", "control_recovered"):
            actual = evaluate(probe, mode, population, entry, EXITS, state)
            if not same(expected, actual):
                raise AssertionError((family, mode, state, expected, actual))
            expected_sources = (
                list(
                    range(
                        predecessor.source_address,
                        predecessor.source_address + len(predecessor.code),
                        4,
                    )
                )
                if entry != DISPATCH.source_address
                else []
            )
            expected_sources += list(
                range(DISPATCH.source_address, DISPATCH.source_address + len(DISPATCH.code), 4)
            )
            if (
                actual["selected_sources"] != expected_sources
                or not actual["cfg_ssa_unchanged"]
                or not actual["cfg_topology_unchanged"]
                or actual["selection_policy"] != "longest_candidate_for_entry"
                or actual["entry_candidates"] < 2
            ):
                raise AssertionError(
                    ("automatic source correspondence or entry alternatives lost", actual)
                )
            consumed = consume_facts(actual, state)
            if not fact_matches_native(consumed, expected):
                raise AssertionError(
                    (
                        "extracted facts disagree with native execution",
                        family,
                        mode,
                        consumed,
                        expected,
                    )
                )
            fact_checks["native_comparisons"] += 1
            if not expected.completed and any(
                actual["control"]["boundaries"][i]["source_address"]
                == expected.exit_pc - state.load_bias
                for i in consumed[3]
            ):
                raise AssertionError("faulting boundary was incorrectly consumed as successful")
            outputs[mode] = actual
            counts[mode] += 1
        check_control_wrapper(outputs["simplified"], outputs["control_recovered"])
        if outputs["control_recovered"]["control_edits"] != (0 if family == "dispatcher" else 1):
            raise AssertionError(
                "unknown entry condition was folded or known condition left unchanged"
            )
        control_checks[
            "unchanged_unknown_conditions"
            if family == "dispatcher"
            else "emitted_literal_condition_folds"
        ] += 1
        if (
            not outputs["original"]["trace"]
            == outputs["normalized"]["trace"]
            == outputs["simplified"]["trace"]
            == outputs["control_recovered"]["trace"]
        ):
            raise AssertionError(
                "path rewrite changed modeled access sequence/source correspondence"
            )
        if (
            family in ("tail", "table_fault", "full_predecessor", "full_table_fault")
            and not outputs["simplified"]["forwarding_facts"]
        ):
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
        analyzed = outputs["simplified"]
        consumed = consume_facts(analyzed, state)
        reached = set(consumed[3])
        guarded = [
            record
            for record in analyzed["control"]["boundaries"]
            if record["boundary"] in reached
            and len(record["edges"]) == 2
            and record["edges"][0]["condition"] is not None
        ]
        if guarded:
            record = guarded[-1]
            swapped = copy.deepcopy(analyzed["control"])
            changed = next(
                item for item in swapped["boundaries"] if item["boundary"] == record["boundary"]
            )
            for edge in changed["edges"]:
                edge["when"] = not edge["when"]
            try:
                wrong = consume_facts(analyzed, state, swapped, False)
            except AssertionError as error:
                if (
                    str(error) != "reached boundary lacks prefix-scoped facts"
                    or analyzed["control"]["proved_divergence"] != record["boundary"]
                ):
                    raise
                wrong = None
                fact_checks["guard_mutation_prefix_inconsistencies"] += 1
            if wrong is not None and fact_matches_native(wrong, expected):
                raise AssertionError("swapped extracted guard polarity survived native comparison")
            if wrong is not None:
                fact_checks["guard_mutation_native_exit_mismatches"] += 1
            fact_checks["guard_polarity_mutations_refuted"] += 1
            equal_targets = copy.deepcopy(analyzed["control"])
            equal_record = next(
                item
                for item in equal_targets["boundaries"]
                if item["boundary"] == record["boundary"]
            )
            for edge in equal_record["edges"]:
                edge["target_kind"] = "absolute_runtime"
                edge["target_address"] = consumed[4][record["boundary"]]
                edge["when"] = not edge["when"]
            if not fact_matches_native(
                consume_facts(analyzed, state, equal_targets, False), expected
            ):
                raise AssertionError("equal-target polarity positive control changed execution")
            fact_checks["equal_target_polarity_positive_controls"] += 1
        if expected.completed:
            destination = copy.deepcopy(analyzed["control"])
            last = next(
                item for item in destination["boundaries"] if item["boundary"] == consumed[3][-1]
            )
            for edge in last["edges"]:
                edge["target_kind"] = "absolute_runtime"
                edge["target_address"] = (expected.exit_pc + 4) & ((1 << 64) - 1)
            if fact_matches_native(consume_facts(analyzed, state, destination, False), expected):
                raise AssertionError("corrupted extracted destination survived native comparison")
            fact_checks["destination_mutations_refuted"] += 1
        if family not in ("dispatcher", "synthetic_out_of_range_predecessor"):
            wrong = evaluate(probe, "wrong_arm_mutant", population, entry, EXITS, state)
            if not expected.completed and family in ("fault", "full_prefix_fault"):
                if not same(expected, wrong):
                    raise AssertionError("fault-before-guard positive control changed")
                control_checks["wrong_basis_fault_positive_controls"] += 1
            else:
                if same(expected, wrong):
                    raise AssertionError("executed wrong literal guard basis survived")
                control_checks["wrong_basis_native_refutations"] += 1
        if family == "tail":
            forged = evaluate(probe, "forged_witness", population, entry, EXITS, state)
            if forged != {"forged_witness_rejected": True, "native_executed": False}:
                raise AssertionError(
                    "forged certificate was accepted or misreported as native evidence"
                )
            control_checks["forged_witness_validator_rejections"] += 1
        mutant = evaluate(probe, "snapshot_mutant", population, entry, EXITS, state)
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
    # Both arms name the next instruction. Folding must preserve all flags and
    # data effects even though the arm choice cannot change the destination.
    equal = Fragment(0x3000, bytes.fromhex("000080d21f0000f120000054"))
    for bias in (0x200000000, 0x600000000):
        state = fixture(bias)
        native = oracle.run((equal,), 0x3000, (0x300C,), state, trusted_fixture=True)
        basis = evaluate(probe, "simplified", (equal,), 0x3000, (0x300C,), state)
        recovered = evaluate(probe, "control_recovered", (equal,), 0x3000, (0x300C,), state)
        check_control_wrapper(basis, recovered)
        if not same(native, recovered) or recovered["control_edits"] != 1:
            raise AssertionError("equal-target control-fold positive case changed native behavior")
        control_checks["equal_target_native_positive_controls"] += 1
    print(
        json.dumps(
            {
                "native_cases": len(cases),
                "case_families": dict(case_families),
                "control_rewrite_checks": dict(control_checks),
                "control_fact_checks": dict(fact_checks),
                "full_predecessor_sha256": full_hash,
                "comparisons": dict(counts),
                "refutations": dict(refuted),
                "snapshot_positive_controls": dict(positive_controls),
                "scope": "automatically selected entry-scoped region from unordered population; runtime table snapshots",
                "selection_input": "source population plus entry only; no itinerary supplied",
                "original_cfg_ssa_unchanged": True,
                "original_cfg_topology_unchanged": True,
                "whole_function_recovery": False,
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
