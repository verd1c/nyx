"""Run the CLI's published SSA batch and explicit call stubs against QEMU.

The stubs live on their own STUB page, so QEMU records its own ordered call and
return events and the verifier compares them with the CLI's trace.
"""

import json
import pathlib
import random
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.oracle.verify_function import _request, _state, _stub_request, check_record, verify
from tools.oracle.function import BRK, EXECUTE, READ, WRITE, STUB as STUB_PAGE


CODE = 0x200000000
LANDING = CODE + 0x100
CALLEE = CODE + 0x1000
SECOND = CALLEE + 0x10
STACK = 0x300000000
SOURCE = 0x400080
FUNCTION = (
    0xAA1E03F3,
    0xCA010002,
    0x8A010003,
    0x8B030063,
    0x8B030040,
    0x940003FB,
    0x91000400,
    0xAA1303FE,
    0xD65F03C0,
)
STUB = (0x91001C00, 0xF9000080, 0xD65F03C0)
# The same MBA, then two calls whose order the trace must preserve.
TWO_CALLS = (
    0xAA1E03F3,
    0xCA010002,
    0x8A010003,
    0x8B030063,
    0x8B030040,
    0x940003FB,
    0x940003FE,
    0xAA1303FE,
    0xD65F03C0,
)
SECOND_STUB = (0xD1000C00, 0xD65F03C0)
# The first call enters the stub one instruction past its declared entry.
MID_BODY = FUNCTION[:5] + (0x940003FC,) + FUNCTION[6:]


def elf(code):
    data = bytearray(128 + len(code))
    data[:16] = b"\x7fELF\x02\x01\x01" + bytes(9)
    struct.pack_into("<HHIQQQIHHHHHH", data, 16, 3, 183, 1, SOURCE, 64, 0, 0, 64, 56, 1, 0, 0, 0)
    struct.pack_into("<IIQQQQQQ", data, 64, 1, 5, 0, 0x400000, 0, len(data), len(data), 4096)
    data[128:] = code
    return bytes(data)


def words(values):
    return struct.pack(f"<{len(values)}I", *values)


def page(placed):
    data = bytearray(struct.pack("<I", BRK) * 1024)
    for offset, code in placed:
        data[offset : offset + len(code)] = code
    return bytes(data)


def fixture(temp, name, function_words, stubs, pairs):
    """A manifest with the function on its code page and every stub on one STUB page."""
    function = words(function_words)
    code = page(((0, function),))
    stub_page = page(tuple((address - CALLEE, code) for address, code in stubs))
    image = pathlib.Path(temp) / f"{name}.so"
    image.write_bytes(elf(function))
    states = []
    for left, right in pairs:
        registers = [index + 3 for index in range(31)]
        registers[0], registers[1], registers[4], registers[30] = (
            left,
            right,
            STACK + 1024,
            LANDING,
        )
        states.append(
            {
                "entry": CODE,
                "registers": registers,
                "nzcv": 0xA0000000,
                "sp": STACK + 2048,
                "regions": [
                    {"address": CODE, "protection": READ | EXECUTE, "hex": code.hex()},
                    {
                        "address": CALLEE,
                        "protection": READ | EXECUTE | STUB_PAGE,
                        "hex": stub_page.hex(),
                    },
                    {"address": STACK, "protection": READ | WRITE, "hex": bytes(4096).hex()},
                ],
            }
        )
    return {
        "trusted_fixture": True,
        "image": str(image),
        "address": SOURCE,
        "size": len(function),
        "exits": [LANDING],
        "admitted": [{"address": CODE, "hex": function.hex()}],
        "stubs": [{"address": address, "hex": code.hex()} for address, code in stubs],
        "declarations": {"abi": "aapcs64", "entries": "closed", "returns": "leave"},
        "states": states,
    }


def wrap(temp, name, executable, change):
    """A CLI wrapper that applies `change` to a successful evaluation's JSON."""
    wrapper = pathlib.Path(temp) / f"{name}.py"
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import json, subprocess, sys\n"
        f"p = subprocess.run([{str(executable)!r}, *sys.argv[1:]], "
        "capture_output=True, text=True)\n"
        "report = json.loads(p.stdout)\n"
        "if p.returncode == 0:\n"
        "    evaluation = report['ssa_graph']['evaluation']\n"
        + "".join(f"    {line}\n" for line in change)
        + "print(json.dumps(report))\n"
        "sys.exit(p.returncode)\n"
    )
    wrapper.chmod(0o755)
    return wrapper


def main(executable):
    function = words(FUNCTION)
    stub = words(STUB)
    rng = random.Random(0x4D324353)
    pairs = [(0, 0), (1, 7), ((1 << 64) - 1, 1)] + [
        (rng.getrandbits(64), rng.getrandbits(64)) for _ in range(13)
    ]
    with tempfile.TemporaryDirectory() as temp:
        manifest = fixture(temp, "function", FUNCTION, ((CALLEE, stub),), pairs)
        image = pathlib.Path(manifest["image"])
        states = manifest["states"]
        report = verify(executable, manifest)
        counts = report["counts"]
        if (
            report["status"] != "matched"
            or counts["matched"] != len(pairs)
            or counts["refuted"]
            or counts["inconclusive"]
            or counts["negative_controls_refuted"] != len(pairs)
            or report["edit_coverage"]["uncovered"]
            or report["call_trace"] != "matched"
            or report["call_trace_counts"]["compared_events"] != 2 * len(pairs)
        ):
            raise AssertionError(f"call stub differential missed the fixture: {report}")
        record = check_record(report)
        if (
            "\ncall_trace matched\n" not in record
            or not record.endswith("\nstatus matched\n")
            or f"\nstates {len(pairs)}\n" not in record
        ):
            raise AssertionError(f"check record does not describe the match: {record}")

        arguments = wrap(
            temp,
            "changed_argument",
            executable,
            ("evaluation['call_events'][0]['arguments'][2] ^= 1",),
        )
        altered = verify(arguments, {**manifest, "states": states[:1]})
        if (
            altered["status"] != "refuted"
            or altered["call_trace"] != "mismatch"
            or altered["states"][0]["reason"] != "call_trace_mismatch"
        ):
            raise AssertionError("changed call argument was not refuted")
        results = wrap(
            temp, "changed_result", executable, ("evaluation['call_events'][1]['results'][0] ^= 1",)
        )
        altered = verify(results, {**manifest, "states": states[:1]})
        if (
            altered["status"] != "refuted"
            or altered["states"][0]["reason"] != "call_trace_mismatch"
        ):
            raise AssertionError("changed stub result was not refuted")
        malformed = wrap(
            temp, "malformed_event", executable, ("evaluation['call_events'][1]['extra'] = 0",)
        )
        try:
            verify(malformed, {**manifest, "states": states[:1]})
        except RuntimeError:
            pass
        else:
            raise AssertionError("malformed CLI call event was classified, not rejected")
        dropped = wrap(temp, "dropped_return", executable, ("evaluation['call_events'].pop()",))
        altered = verify(dropped, {**manifest, "states": states[:1]})
        if altered["status"] != "refuted" or altered["call_trace"] != "mismatch":
            raise AssertionError("missing return event was not refuted")

        second = words(SECOND_STUB)
        two = fixture(temp, "two_calls", TWO_CALLS, ((CALLEE, stub), (SECOND, second)), pairs[:4])
        ordered = verify(executable, two)
        if (
            ordered["status"] != "matched"
            or ordered["call_trace"] != "matched"
            or ordered["call_trace_counts"]["compared_events"] != 4 * 4
            or ordered["stub_coverage"]["called"] != 2
        ):
            raise AssertionError(f"two-call trace did not match: {ordered}")
        swapped = wrap(
            temp,
            "swapped_calls",
            executable,
            (
                "events = evaluation['call_events']",
                "evaluation['call_events'] = events[2:] + events[:2]",
                "evaluation['call_targets'] = evaluation['call_targets'][::-1]",
            ),
        )
        altered = verify(swapped, {**two, "states": two["states"][:1]})
        if (
            altered["status"] != "refuted"
            or altered["states"][0]["reason"] != "call_trace_mismatch"
        ):
            raise AssertionError("swapped call order was not refuted")

        middle = fixture(temp, "mid_body", MID_BODY, ((CALLEE, stub),), pairs[:1])
        entered = verify(executable, middle)
        if (
            entered["status"] != "NOT CHECKED"
            or entered["counts"]["inconclusive"] != 1
            or entered["states"][0]["reason"] != "original_left_trace"
        ):
            raise AssertionError(f"mid-body stub entry was not inconclusive: {entered}")

        stub_path = pathlib.Path(temp) / "stubs.bin"
        stub_path.write_bytes(_stub_request(((CALLEE, stub),)))
        request_path = pathlib.Path(temp) / "state.bin"
        request_path.write_bytes(_request(_state(states[0]), CODE, (LANDING,)))
        command = (
            executable,
            "recover-regions",
            str(image),
            "--address",
            f"{SOURCE:x}",
            "--size",
            str(len(function)),
            "--memory-profile",
            "concrete_atomic_scalar",
            "--assume-abi",
            "aapcs64",
            "--assume-entries",
            "closed",
            "--assume-returns",
            "leave",
            "--state-file",
            str(request_path),
        )
        cli_report = json.loads(
            subprocess.run(
                (*command, "--stubs-file", str(stub_path)),
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        observed = cli_report["ssa_graph"]["evaluation"]
        edits = sum(
            len(stage["journal"]) + len(stage.get("phi_journal", ()))
            for stage in cli_report["ssa_graph"]["proposals"]["stages"]
            if stage["outcome"] == "proposed"
        )
        if (
            observed["declared_stub_count"] != 1
            or observed["declared_stubs"] != [{"address": CALLEE, "bytes": stub.hex()}]
            or observed["call_targets"] != [CALLEE]
            or observed["external_calls"] != 1
            or [event["kind"] for event in observed["call_events"]] != ["call", "return"]
            or observed["call_events"][0]["return_address"] != CODE + 0x18
            or observed["status"] != "completed"
        ):
            raise AssertionError("declared external stub was not executed exactly once")
        # Original code on a STUB page is refused, as the runner refuses it.
        misplaced = dict(
            states[0],
            regions=[
                dict(
                    region,
                    protection=region["protection"]
                    | (STUB_PAGE if region["address"] == CODE else 0),
                )
                for region in states[0]["regions"]
            ],
        )
        request_path.write_bytes(_request(_state(misplaced), CODE, (LANDING,)))
        refused = subprocess.run(
            (*command, "--stubs-file", str(stub_path)), capture_output=True, text=True
        )
        if not refused.returncode or json.loads(refused.stdout).get("reason") != "source_mismatch":
            raise AssertionError(f"code on a STUB page was evaluated: {refused.stdout}")
        request_path.write_bytes(_request(_state(states[0]), CODE, (LANDING,)))
        wrapper = wrap(
            temp,
            "undeclared_call",
            executable,
            (
                f"evaluation['call_targets'] = [{CALLEE + 4}]",
                f"evaluation['call_events'] = [dict(evaluation['call_events'][0], target={CALLEE + 4})]",
                "evaluation['status'] = 'inconclusive'",
                "evaluation['outcome'] = 'unsupported'",
                "evaluation['stop'] = 'unresolved'",
            ),
        )
        unlisted = verify(wrapper, {**manifest, "states": states[:1]})
        if (
            unlisted["status"] != "NOT CHECKED"
            or unlisted["states"][0]["reason"] != "undeclared_call"
        ):
            raise AssertionError("undeclared attempted call was not inconclusive")
        missing_store = wrap(
            temp,
            "missing_store",
            executable,
            (
                "page = evaluation['memory'][0]",
                "page['bytes'] = page['bytes'][:2048] + '0' * 16 + page['bytes'][2064:]",
            ),
        )
        altered = verify(missing_store, {**manifest, "states": states[:1]})
        if altered["status"] != "refuted" or altered["counts"]["refuted"] != 1:
            raise AssertionError("lost external-call store was not refuted")
        without = json.loads(
            subprocess.run(command, check=True, capture_output=True, text=True).stdout
        )
        if without["ssa_graph"]["evaluation"]["status"] != "inconclusive":
            raise AssertionError("missing stub did not make the call inconclusive")
        wrong = bytearray(stub_path.read_bytes())
        wrong[-1] ^= 1
        stub_path.write_bytes(wrong)
        refused = subprocess.run(
            (*command, "--stubs-file", str(stub_path)), capture_output=True, text=True
        )
        if refused.returncode == 0 or json.loads(refused.stdout).get("reason") != "stub_mismatch":
            raise AssertionError("changed stub bytes were not refused")
        for invalid in (
            b"NYXSTB01" + struct.pack("<Q", 0),
            _stub_request(((CALLEE, stub), (CALLEE, stub))),
            _stub_request(((CODE, function[:4]),)),
        ):
            stub_path.write_bytes(invalid)
            refused = subprocess.run(
                (*command, "--stubs-file", str(stub_path)), capture_output=True, text=True
            )
            if (
                refused.returncode == 0
                or json.loads(refused.stdout).get("reason") != "invalid_stubs"
            ):
                raise AssertionError("malformed or overlapping stubs were admitted")
        no_state = subprocess.run(
            (
                executable,
                "recover-regions",
                str(image),
                "--address",
                f"{SOURCE:x}",
                "--size",
                str(len(function)),
                "--stubs-file",
                str(stub_path),
            ),
            capture_output=True,
            text=True,
        )
        if no_state.returncode == 0 or json.loads(no_state.stdout).get("reason") != "usage":
            raise AssertionError("stub declaration without state file was admitted")
        for code_words, expected_stop in (
            ((*STUB[:-1], 0xD65F0260), "unresolved"),
            ((0xAA0003F3, 0xD65F03C0), "declined"),
        ):
            changed_stub = words(code_words)
            changed_state = dict(states[0])
            changed_state["regions"] = [
                states[0]["regions"][0],
                {**states[0]["regions"][1], "hex": page(((0, changed_stub),)).hex()},
                states[0]["regions"][2],
            ]
            request_path.write_bytes(_request(_state(changed_state), CODE, (LANDING,)))
            stub_path.write_bytes(_stub_request(((CALLEE, changed_stub),)))
            process = subprocess.run(
                (*command, "--stubs-file", str(stub_path)),
                check=True,
                capture_output=True,
                text=True,
            )
            evaluation = json.loads(process.stdout)["ssa_graph"]["evaluation"]
            if evaluation["status"] != "inconclusive" or evaluation["stop"] != expected_stop:
                raise AssertionError("bad continuation or ABI clobber was accepted")
    print(
        f"SSA CLI external call: matched={len(pairs)} inconclusive=0 "
        f"negative_controls_refuted={len(pairs)} stubs=1 "
        f"call_events={report['call_trace_counts']['compared_events']} "
        f"two_call_events={ordered['call_trace_counts']['compared_events']} "
        f"edits={edits} blocks={observed['completed_blocks']}"
    )


if __name__ == "__main__":
    main(sys.argv[1])
