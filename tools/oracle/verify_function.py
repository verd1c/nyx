"""Bounded QEMU differential for the SSA graph the CLI prints.

The manifest describes trusted, hand-reviewed executable pages.
Stubs on STUB pages give QEMU's own ordered call-event trace, which must equal
the CLI's; stubs elsewhere leave the call trace unchecked.
"""

import argparse
import json
import pathlib
import re
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tools.oracle.aarch64 import Declined, OracleError
from tools.oracle.function import FunctionOracle, FunctionState, Region


def _integer(value):
    if type(value) is int:
        return value
    if type(value) is str:
        return int(value, 0)
    raise ValueError("expected integer or base-prefixed integer string")


def _region(spec):
    return Region(
        _integer(spec["address"]), bytes.fromhex(spec["hex"]), _integer(spec["protection"])
    )


def _state(spec):
    return FunctionState(
        tuple(_integer(x) for x in spec["registers"]),
        _integer(spec["nzcv"]),
        _integer(spec["sp"]),
        tuple(_region(region) for region in spec["regions"]),
    )


def _request(state, entry, exits):
    header = (*state.registers, state.nzcv, state.sp, entry, len(state.regions), len(exits))
    chunks = [b"NYXFUN01", struct.pack("<36Q", *header)]
    for region in state.regions:
        chunks.extend(
            (struct.pack("<3Q", region.address, len(region.data), region.protection), region.data)
        )
    chunks.append(struct.pack(f"<{len(exits)}Q", *exits))
    return b"".join(chunks)


def _stub_request(stubs):
    chunks = [b"NYXSTB01", struct.pack("<Q", len(stubs))]
    for address, code in stubs:
        chunks.extend((struct.pack("<2Q", address, len(code)), code))
    return b"".join(chunks)


# What the caller can observe of X0-X30 and NZCV when the run completes. Under
# the declared AAPCS64 contract a return exposes only its results X0-X8, the
# platform register X18 and the preserved X19-X29; X9-X17, X30 and the flags
# are dead once it leaves. Written here from the standard, not taken from the
# CLI under test.
_ALL_REGISTERS = tuple(range(31))
_AAPCS64_RETURN = (*range(9), *range(18, 30))


def _masked(memory, frame):
    """Memory with the declared private frame's bytes cleared: nothing outside
    the function, and no asynchronous observer, can read them."""
    if frame is None:
        return memory
    begin, end = frame
    masked = {}
    for address, data in memory.items():
        data = bytearray(data)
        for index in range(max(begin, address), min(end, address + len(data))):
            data[index - address] = 0
        masked[address] = bytes(data)
    return masked


def _compare(original, proposed, declarations=None, return_address=None, frame=None):
    declarations = declarations or {}
    # The ABI speaks for a return to the caller only; any other stop, and a
    # fault mid-function, still expose every register unless faults are
    # declared terminal.
    returned = (
        original.completed
        and declarations.get("abi") == "aapcs64"
        and return_address is not None
        and original.pc == return_address
    )
    observed = _AAPCS64_RETURN if returned else _ALL_REGISTERS
    flags, stack = not returned, True
    status = proposed.get("status")
    if original.completed and status == "fault" or not original.completed and status == "completed":
        return "refuted"
    if status not in ("completed", "fault"):
        return "inconclusive"
    if not original.completed:
        kind = (proposed.get("fault") or {}).get("kind")
        expected = {"unmapped": 11, "permission": 11, "alignment": 7}.get(kind)
        if expected is None or original.signal not in (7, 11):
            return "inconclusive"
        if (
            original.signal != expected
            or (proposed.get("fault") or {}).get("address") != original.fault_address
        ):
            return "refuted"
    try:
        memory = {entry["address"]: bytes.fromhex(entry["bytes"]) for entry in proposed["memory"]}
        registers = tuple(proposed["registers"])
        if len(registers) != len(original.registers):
            return "refuted"
        # A terminal fault's register state, SP included, is never observed.
        if not original.completed and declarations.get("faults") == "terminal":
            observed, flags, stack = (), False, False
        agrees = (
            all(registers[index] == original.registers[index] for index in observed)
            and (not flags or proposed["nzcv"] == original.nzcv)
            and (not stack or proposed["sp"] == original.sp)
            and proposed["pc"] == original.pc
            and _masked(memory, frame) == _masked(original.memory, frame)
        )
    except (KeyError, TypeError, ValueError):
        return "refuted"
    return "matched" if agrees else "refuted"


def _slot(handle):
    match = re.fullmatch(r"b([0-9]+)\.[0-9]+", handle) if type(handle) is str else None
    return int(match.group(1)) if match else None


def _edit_targets(proposal):
    executable, removed = [], []
    for stage in proposal["stages"]:
        if stage["outcome"] != "proposed":
            continue
        if not stage.get("journal") and not stage.get("phi_journal"):
            executable.append((stage["pass"], "journal", 0, None))
        for journal_name in ("journal", "phi_journal"):
            for index, edit in enumerate(stage.get(journal_name, ())):
                identity = (stage["pass"], journal_name, index)
                if edit.get("kind") == "removed_block":
                    removed.append(identity)
                    continue
                handle = edit.get("result_block") or edit.get("block")
                executable.append((*identity, _slot(handle)))
    return tuple(executable), tuple(removed)


_EVENT_FIELDS = {
    "call": {"kind", "target", "return_address", "sp", "arguments"},
    "return": {"kind", "pc", "sp", "results"},
}


def _word(value):
    return type(value) is int and 0 <= value <= (1 << 64) - 1


def _call_trace(evaluated, call_targets):
    """The CLI's events, rejected as a CLI defect unless each has the exact shape."""
    events = evaluated.get("call_events")
    if (
        type(events) is not list
        or any(
            type(event) is not dict
            or set(event) != _EVENT_FIELDS.get(event.get("kind"))
            or not all(_word(event[key]) for key in event.keys() - {"kind", "arguments", "results"})
            or not all(
                type(event.get(key, [])) is list
                and len(event.get(key, [])) == (8 if key == "arguments" else 2)
                and all(_word(value) for value in event.get(key, []))
                for key in {"arguments", "results"} & event.keys()
            )
            for event in events
        )
        or [event.get("target") for event in events if event.get("kind") == "call"] != call_targets
    ):
        raise RuntimeError("CLI reported an inconsistent call trace")
    return events


def _visited_slots(evaluated):
    return {
        _slot(handle)
        for handle in evaluated.get("visited_blocks", ())[: evaluated.get("completed_blocks", 0)]
    }


def verify(executable, manifest):
    if manifest.get("trusted_fixture") is not True:
        raise ValueError("trusted_fixture=true and reviewed executable pages required")
    states = manifest["states"]
    if type(states) is not list or len(states) > 256:
        raise ValueError("expected at most 256 states")
    image = pathlib.Path(manifest["image"])
    address = _integer(manifest["address"])
    size = _integer(manifest["size"])
    exits = tuple(_integer(value) for value in manifest["exits"])
    stubs = tuple(
        (_integer(item["address"]), bytes.fromhex(item["hex"]))
        for item in manifest.get("stubs", ())
    )
    oracle = FunctionOracle()
    for item in manifest["admitted"]:
        oracle.admit(_integer(item["address"]), bytes.fromhex(item["hex"]))
    for item in manifest.get("data", ()):
        oracle.admit_data(_integer(item["address"]), bytes.fromhex(item["hex"]))
    command = [
        str(executable),
        "recover-regions",
        str(image),
        "--address",
        f"{address:x}",
        "--size",
        str(size),
        "--memory-profile",
        manifest.get("memory_profile", "concrete_atomic_scalar"),
    ]
    declarations = manifest.get("declarations", {})
    if declarations.get("abi"):
        command.extend(("--assume-abi", declarations["abi"]))
    if declarations.get("entries"):
        command.extend(("--assume-entries", declarations["entries"]))
    if declarations.get("returns"):
        command.extend(("--assume-returns", declarations["returns"]))
    if declarations.get("image_access"):
        command.extend(("--assume-image-access", declarations["image_access"]))
    if declarations.get("faults"):
        command.extend(("--assume-faults", declarations["faults"]))
    frame_bounds = None
    if declarations.get("frame"):
        command.extend(("--assume-private-frame", declarations["frame"]))
        frame_bounds = tuple(int(bound) for bound in declarations["frame"].split(":"))
    if declarations.get("frame_reach"):
        command.extend(("--assume-frame-reach", declarations["frame_reach"]))
    counts = {
        "matched": 0,
        "refuted": 0,
        "inconclusive": 0,
        "negative_controls_refuted": 0,
        "examined": 0,
    }
    details = []
    edit_targets = None
    proposal_fingerprint = None
    covered_edits = set()
    declared_stubs = [{"address": address, "bytes": code.hex()} for address, code in sorted(stubs)]
    declared_stub_targets = {address for address, _ in stubs}
    called_stubs = set()
    candidate = None
    trace = {
        "compared_states": 0,
        "compared_events": 0,
        "untraced_states": 0,
        "mismatched_states": 0,
    }
    with tempfile.TemporaryDirectory(prefix="nyx-ssa-function-check-") as temp:
        request_path = pathlib.Path(temp) / "state.bin"
        if stubs:
            stub_path = pathlib.Path(temp) / "stubs.bin"
            stub_path.write_bytes(_stub_request(stubs))
            command.extend(("--stubs-file", str(stub_path)))
        for index, spec in enumerate(states):
            state = _state(spec)
            entry = _integer(spec["entry"])
            frame = (
                None
                if frame_bounds is None
                else (state.sp + frame_bounds[0], state.sp + frame_bounds[1])
            )
            try:
                original = oracle.run(state, entry, exits, stubs=stubs, trusted_fixture=True)
            except OracleError as error:
                if "exceeded its deadline" not in str(error):
                    raise
                counts["inconclusive"] += 1
                details.append(
                    {"index": index, "result": "inconclusive", "reason": "original_timeout"}
                )
                continue
            if original.left_trace:
                counts["inconclusive"] += 1
                details.append(
                    {"index": index, "result": "inconclusive", "reason": "original_left_trace"}
                )
                continue
            request_path.write_bytes(_request(state, entry, exits))
            try:
                process = subprocess.run(
                    command + ["--state-file", str(request_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
            except subprocess.TimeoutExpired:
                counts["inconclusive"] += 1
                details.append(
                    {"index": index, "result": "inconclusive", "reason": "candidate_timeout"}
                )
                continue
            try:
                report = json.loads(process.stdout)
            except json.JSONDecodeError as error:
                raise RuntimeError("CLI returned invalid JSON") from error
            if process.returncode:
                if report.get("outcome") != "refused":
                    raise RuntimeError("CLI failed without a typed refusal")
                counts["inconclusive"] += 1
                details.append(
                    {"index": index, "result": "inconclusive", "reason": report["reason"]}
                )
                continue
            if (
                "ssa_graph" not in report
                or "proposals" not in report["ssa_graph"]
                or "evaluation" not in report["ssa_graph"]
            ):
                counts["inconclusive"] += 1
                details.append(
                    {"index": index, "result": "inconclusive", "reason": "no_candidate_evaluation"}
                )
                continue
            proposal = report["ssa_graph"]["proposals"]
            evaluated = report["ssa_graph"]["evaluation"]
            if (
                proposal["whole_function_check"] != "NOT CHECKED"
                or evaluated["independent_check"] != "NOT CHECKED"
            ):
                raise RuntimeError("CLI claimed an independent check before comparison")
            if (
                evaluated.get("declared_stub_count") != len(stubs)
                or evaluated.get("declared_stubs") != declared_stubs
            ):
                raise RuntimeError("CLI used a different external stub declaration")
            call_targets = evaluated.get("call_targets")
            if (
                type(call_targets) is not list
                or evaluated.get("external_calls") != len(call_targets)
                or any(
                    type(target) is not int or target < 0 or target > (1 << 64) - 1
                    for target in call_targets
                )
            ):
                raise RuntimeError("CLI reported an inconsistent call trace")
            events = _call_trace(evaluated, call_targets)
            undeclared_calls = sorted(set(call_targets) - declared_stub_targets)
            if proposal["text"] is None or not any(
                stage["outcome"] == "proposed" for stage in proposal["stages"]
            ):
                counts["inconclusive"] += 1
                details.append(
                    {"index": index, "result": "inconclusive", "reason": "no_proposed_edits"}
                )
                continue
            fingerprint = json.dumps(
                {
                    key: report["ssa_graph"].get(key)
                    for key in ("revision", "text", "declarations", "proposals")
                },
                sort_keys=True,
            )
            if proposal_fingerprint is None:
                proposal_fingerprint = fingerprint
                candidate = proposal.get("candidate_digest")
            elif proposal_fingerprint != fingerprint:
                raise RuntimeError("published SSA batch changed across input states")
            targets = _edit_targets(proposal)
            if edit_targets is None:
                edit_targets = targets
            elif edit_targets != targets:
                raise RuntimeError("proposal journals changed across input states")
            result = (
                ("refuted" if evaluated.get("status") == "completed" else "inconclusive")
                if undeclared_calls
                else _compare(original, evaluated, declarations, state.registers[30], frame)
            )
            witnessed = {
                target
                for target in targets[0]
                if target[3] is not None and target[3] in _visited_slots(evaluated)
            }
            if undeclared_calls:
                reason = "undeclared_call"
            elif result == "matched" and targets[0] and not witnessed:
                result = "inconclusive"
                reason = "no_executed_edit"
            else:
                reason = evaluated["stop"] if result == "inconclusive" else "none"
            # Without declared stubs every executable word QEMU may run is admitted
            # original code or a landing BRK, so there is no external callee and
            # an empty trace is a compared one, not an untraced run.
            if result == "matched" and original.events is None:
                trace["untraced_states"] += 1
            elif result == "matched":
                trace["compared_states"] += 1
                trace["compared_events"] += len(original.events)
                if events != list(original.events):
                    trace["mismatched_states"] += 1
                    result, reason = "refuted", "call_trace_mismatch"
            counts[result] += 1
            if result == "matched":
                covered_edits.update(witnessed)
                called_stubs.update(call_targets)
                wrong = dict(evaluated)
                wrong["registers"] = list(evaluated["registers"])
                wrong["registers"][0] ^= 1
                if not original.completed and declarations.get("faults") == "terminal":
                    wrong["fault"] = dict(evaluated["fault"])
                    wrong["fault"]["address"] ^= 1
                if _compare(original, wrong, declarations, state.registers[30], frame) != "refuted":
                    raise RuntimeError("negative control did not refute")
                counts["negative_controls_refuted"] += 1
            details.append({"index": index, "result": result, "reason": reason})
    counts["examined"] = counts["matched"] + counts["refuted"]
    executable, removed = edit_targets if edit_targets is not None else ((), ())
    uncovered = [target for target in executable if target not in covered_edits]
    uncalled_stubs = sorted(declared_stub_targets - called_stubs)
    call_trace = (
        "mismatch"
        if trace["mismatched_states"]
        else "matched"
        if trace["compared_states"] and not trace["untraced_states"]
        else "not_checked"
    )
    status = (
        "refuted"
        if counts["refuted"]
        else "NOT CHECKED"
        if counts["examined"] == 0
        else "inconclusive"
        if (counts["inconclusive"] or uncovered or uncalled_stubs or call_trace != "matched")
        else "matched"
    )
    return {
        "schema": 1,
        "status": status,
        "counts": counts,
        "states": details,
        "edit_coverage": {
            "executable": len(executable),
            "covered": len(covered_edits),
            "uncovered": [
                {"pass": name, "journal": journal, "journal_index": index}
                for name, journal, index, _ in uncovered
            ],
            "removed_unreachable_blocks": len(removed),
        },
        "stub_coverage": {
            "declared": len(stubs),
            "called": len(called_stubs),
            "uncalled_targets": uncalled_stubs,
        },
        "call_trace": call_trace,
        "call_trace_counts": trace,
        "candidate": candidate,
        "scope": "bounded complete-state QEMU comparison of the exact CLI SSA proposal",
        "universal_proof": "NOT CHECKED",
        "declarations": declarations,
    }


def check_record(report):
    """The NYXCHK01 text record binding this result to the published candidate."""
    counts = report["counts"]
    edits = report["edit_coverage"]
    stubs = report["stub_coverage"]
    candidate = report["candidate"] if report["candidate"] is not None else "none"
    if type(candidate) is not str or not candidate or any(ch.isspace() for ch in candidate):
        raise ValueError("candidate digest must be one printable token")
    fields = (
        ("candidate", candidate),
        ("scope", "whole_function"),
        ("states", counts["examined"] + counts["inconclusive"]),
        ("matched", counts["matched"]),
        ("refuted", counts["refuted"]),
        ("inconclusive", counts["inconclusive"]),
        ("edits_executable", edits["executable"]),
        ("edits_covered", edits["covered"]),
        ("stubs_declared", stubs["declared"]),
        ("stubs_called", stubs["called"]),
        ("call_trace", report["call_trace"]),
        ("status", "not_checked" if report["status"] == "NOT CHECKED" else report["status"]),
    )
    return "NYXCHK01\n" + "".join(f"{name} {value}\n" for name, value in fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=pathlib.Path)
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument(
        "--check-record",
        type=pathlib.Path,
        help="write the NYXCHK01 record for the checked candidate here",
    )
    args = parser.parse_args()
    try:
        report = verify(args.executable, json.loads(args.manifest.read_text()))
        if args.check_record:
            args.check_record.write_text(check_record(report))
    except (KeyError, ValueError, RuntimeError, Declined, OracleError) as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True))
    return int(report["status"] != "matched")


if __name__ == "__main__":
    sys.exit(main())
