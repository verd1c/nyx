"""Compare the CLI's actual provisional SSA batch with an original under QEMU."""

import copy
import dataclasses
import json
import pathlib
import random
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.oracle.function import BRK, EXECUTE, READ, WRITE, FunctionOracle, FunctionState, Region
from tools.oracle.verify_function import _compare, verify


CODE = 0x200000000
LANDING = CODE + 0x100
STACK = 0x300000000
SOURCE = 0x400080


def elf(code):
    data = bytearray(128 + len(code))
    data[:16] = b"\x7fELF\x02\x01\x01" + bytes(9)
    struct.pack_into("<HHIQQQIHHHHHH", data, 16, 3, 183, 1, SOURCE, 64, 0, 0, 64, 56, 1, 0, 0, 0)
    struct.pack_into("<IIQQQQQQ", data, 64, 1, 5, 0, 0x400000, 0, len(data), len(data), 4096)
    data[128:] = code
    return bytes(data)


def request(state):
    header = (*state.registers, state.nzcv, state.sp, CODE, len(state.regions), 1)
    payload = [b"NYXFUN01", struct.pack("<36Q", *header)]
    for region in state.regions:
        payload.extend(
            (struct.pack("<3Q", region.address, len(region.data), region.protection), region.data)
        )
    payload.append(struct.pack("<Q", LANDING))
    return b"".join(payload)


def matched(actual, evaluated):
    return (
        evaluated["status"] == "completed"
        and evaluated["outcome"] == "completed"
        and evaluated["stop"] == "returned"
        and tuple(evaluated["registers"]) == actual.registers
        and evaluated["nzcv"] == actual.nzcv
        and evaluated["sp"] == actual.sp
        and evaluated["pc"] == actual.pc
        and {entry["address"]: bytes.fromhex(entry["bytes"]) for entry in evaluated["memory"]}
        == actual.memory
    )


def main(executable):
    words = (0xD2800168, 0x14000003, BRK, BRK, 0xD28002C8, 0xAA0803E0, 0xD65F03C0)
    source_code = b"".join(struct.pack("<I", word) for word in words)
    code_page = bytearray(struct.pack("<I", BRK) * 1024)
    code_page[: len(source_code)] = source_code
    code_page = bytes(code_page)
    oracle = FunctionOracle()
    oracle.admit(CODE, code_page[:8])
    oracle.admit(CODE + 16, code_page[16:28])
    rng = random.Random(0x4D32434C)
    values = [0, 11, 22, (1 << 64) - 1] + [rng.getrandbits(64) for _ in range(12)]
    matched_count = refuted = 0
    edit_counts = None
    completed_blocks = None
    manifest_states = []
    with tempfile.TemporaryDirectory() as temp:
        image = pathlib.Path(temp) / "function.so"
        image.write_bytes(elf(source_code))
        request_path = pathlib.Path(temp) / "request.bin"
        for initial in values:
            registers = [index + 3 for index in range(31)]
            registers[8], registers[30] = initial, LANDING
            state = FunctionState(
                tuple(registers),
                0xA0000000,
                STACK + 2048,
                (Region(CODE, code_page, READ | EXECUTE), Region(STACK, bytes(4096), READ | WRITE)),
            )
            manifest_states.append(
                {
                    "entry": CODE,
                    "registers": registers,
                    "nzcv": state.nzcv,
                    "sp": state.sp,
                    "regions": [
                        {
                            "address": region.address,
                            "protection": region.protection,
                            "hex": region.data.hex(),
                        }
                        for region in state.regions
                    ],
                }
            )
            actual = oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
            if not actual.completed or actual.pc != LANDING:
                raise AssertionError("original did not complete")
            request_path.write_bytes(request(state))
            command = (
                executable,
                "recover-regions",
                str(image),
                "--address",
                "400080",
                "--size",
                "28",
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
            process = subprocess.run(command, check=True, capture_output=True, text=True)
            report = json.loads(process.stdout)
            if report["ssa_graph"]["proposals"]["whole_function_check"] != "NOT CHECKED":
                raise AssertionError("CLI claimed an independent comparison it did not run")
            if not any(
                stage["outcome"] == "proposed"
                for stage in report["ssa_graph"]["proposals"]["stages"]
            ):
                raise AssertionError("fixture did not exercise a proposed SSA batch")
            evaluation = report["ssa_graph"]["evaluation"]
            current_edits = {
                stage["pass"]: len(stage["journal"])
                for stage in report["ssa_graph"]["proposals"]["stages"]
                if stage["outcome"] == "proposed"
            }
            if edit_counts is None:
                edit_counts = current_edits
                completed_blocks = evaluation["completed_blocks"]
            elif current_edits != edit_counts or evaluation["completed_blocks"] != completed_blocks:
                raise AssertionError("proposal or executed block count changed by input state")
            if evaluation["independent_check"] != "NOT CHECKED" or not matched(actual, evaluation):
                raise AssertionError(f"CLI SSA/QEMU disagreement for x8={initial}")
            matched_count += 1
            wrong = copy.deepcopy(evaluation)
            wrong["registers"][0] ^= 1
            if matched(actual, wrong):
                raise AssertionError("wrong-value observation was not refuted")
            refuted += 1
        valid = request(state)
        refusals = {
            "invalid_state": (
                b"BADMAGIC" + valid[8:],
                valid[:-1],
                valid[: 8 + 35 * 8]
                + struct.pack("<Q", 2)
                + valid[8 + 36 * 8 :]
                + struct.pack("<Q", LANDING),
                valid[: 8 + 36 * 8 + 24 + 0x100]
                + bytes([valid[8 + 36 * 8 + 24 + 0x100] ^ 1])
                + valid[8 + 36 * 8 + 24 + 0x101 :],
            ),
            "source_mismatch": (
                valid[: 8 + 36 * 8 + 24]
                + bytes([valid[8 + 36 * 8 + 24] ^ 1])
                + valid[8 + 36 * 8 + 25 :],
            ),
        }
        for reason, inputs in refusals.items():
            for invalid in inputs:
                request_path.write_bytes(invalid)
                process = subprocess.run(command, capture_output=True, text=True)
                report = json.loads(process.stdout)
                if process.returncode == 0 or report.get("reason") != reason:
                    raise AssertionError(f"bad state did not refuse as {reason}: {report}")
        manifest = {
            "trusted_fixture": True,
            "image": str(image),
            "address": SOURCE,
            "size": len(source_code),
            "exits": [LANDING],
            "admitted": [
                {"address": CODE, "hex": code_page[:8].hex()},
                {"address": CODE + 16, "hex": code_page[16:28].hex()},
            ],
            "declarations": {"abi": "aapcs64", "entries": "closed", "returns": "leave"},
            "states": manifest_states,
        }
        checked = verify(executable, manifest)
        counts = checked["counts"]
        if (
            checked["status"] != "matched"
            or counts["examined"] != len(values)
            or counts["matched"] != len(values)
            or counts["refuted"]
            or counts["inconclusive"]
            or counts["negative_controls_refuted"] != len(values)
        ):
            raise AssertionError(f"production verifier missed the fixture: {checked}")
        manifest["states"] = manifest_states[:1]
        manifest["stubs"] = [{"address": CODE + 0x200, "hex": struct.pack("<I", BRK).hex()}]
        uncalled = verify(executable, manifest)
        if uncalled["status"] != "inconclusive" or uncalled["stub_coverage"][
            "uncalled_targets"
        ] != [CODE + 0x200]:
            raise AssertionError("unused declared stub was accepted")
        del manifest["stubs"]
        manifest["states"] = []
        if verify(executable, manifest)["status"] != "NOT CHECKED":
            raise AssertionError("zero-evidence verifier claimed a check")
        manifest_path = pathlib.Path(temp) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        tool = pathlib.Path(__file__).resolve().parents[1] / "tools/oracle/verify_function.py"
        process = subprocess.run(
            (sys.executable, str(tool), str(executable), str(manifest_path)),
            capture_output=True,
            text=True,
        )
        if process.returncode == 0 or json.loads(process.stdout)["status"] != "NOT CHECKED":
            raise AssertionError("zero-evidence verifier command exited successfully")
        manifest["trusted_fixture"] = False
        try:
            verify(executable, manifest)
        except ValueError:
            pass
        else:
            raise AssertionError("untrusted fixture was admitted")
        manifest["trusted_fixture"] = True
        manifest["states"] = manifest_states[:1]
        mutations = {
            "wrong_value": "    report['ssa_graph']['evaluation']['registers'][0] ^= 1\n",
            "hide_visit": "    report['ssa_graph']['evaluation']['visited_blocks'] = []\n",
            "hide_first_visit": (
                "    if struct.unpack_from('<Q', open(sys.argv[-1], 'rb').read(), 8 + 8 * 8)[0] == 0:\n"
                "        report['ssa_graph']['evaluation']['visited_blocks'] = []\n"
            ),
            "extra_unvisited_edit": (
                "    next(s for s in report['ssa_graph']['proposals']['stages'] "
                "if s['outcome'] == 'proposed')['journal'].append({'block': 'b999.1'})\n"
            ),
            "extra_unvisited_phi": (
                "    next(s for s in report['ssa_graph']['proposals']['stages'] "
                "if s['outcome'] == 'proposed')['phi_journal'] = "
                "[{'result_block': 'b999.1'}]\n"
            ),
            "changed_batch": (
                "    if struct.unpack_from('<Q', open(sys.argv[-1], 'rb').read(), 8 + 8 * 8)[0] == 11:\n"
                "        next(s for s in report['ssa_graph']['proposals']['stages'] "
                "if s['outcome'] == 'proposed' and s['journal'])"
                "['journal'][0]['from_revision'] += 1\n"
            ),
        }
        for mode, expected, controls in (
            ("wrong_value", "refuted", 0),
            ("hide_visit", "NOT CHECKED", 0),
            ("hide_first_visit", "inconclusive", 1),
            ("extra_unvisited_edit", "inconclusive", 1),
            ("extra_unvisited_phi", "inconclusive", 1),
            ("changed_batch", "error", 0),
        ):
            manifest["states"] = (
                manifest_states[:2]
                if mode in ("hide_first_visit", "changed_batch")
                else manifest_states[:1]
            )
            wrapper = pathlib.Path(temp) / f"{mode}.py"
            wrapper.write_text(
                "#!/usr/bin/env python3\n"
                "import json, struct, subprocess, sys\n"
                f"p = subprocess.run([{str(executable)!r}, *sys.argv[1:]], "
                "capture_output=True, text=True)\n"
                "report = json.loads(p.stdout)\n"
                "if p.returncode == 0:\n" + mutations[mode] + "print(json.dumps(report))\n"
                "sys.exit(p.returncode)\n"
            )
            wrapper.chmod(0o755)
            if mode == "changed_batch":
                try:
                    verify(wrapper, manifest)
                except RuntimeError:
                    continue
                raise AssertionError("different proposal journals were accepted")
            result = verify(wrapper, manifest)
            if (
                result["status"] != expected
                or result["counts"]["negative_controls_refuted"] != controls
            ):
                raise AssertionError(f"{mode} wrapper was not classified: {result}")
            if (
                mode in ("extra_unvisited_edit", "extra_unvisited_phi")
                and not result["edit_coverage"]["uncovered"]
            ):
                raise AssertionError("unvisited journal edit was not reported")
        faulty = dataclasses.replace(actual, signal=11, fault_address=STACK)
        if (
            _compare(faulty, evaluation) != "refuted"
            or _compare(
                actual,
                {**evaluation, "status": "fault", "fault": {"kind": "unmapped", "address": STACK}},
            )
            != "refuted"
        ):
            raise AssertionError("completion/fault mismatch was not refuted")
    print(
        f"SSA CLI batch: matched={matched_count} refuted={refuted} "
        f"edits={sum(edit_counts.values())} blocks={completed_blocks} "
        f"stages={edit_counts}"
    )


if __name__ == "__main__":
    main(sys.argv[1])
