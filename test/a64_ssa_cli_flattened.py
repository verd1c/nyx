"""Check the SSA passes that need reachability through recovered transitions and a call.

The flattened function keeps its state in its frame and dispatches through a
halfword table; case 1 calls out. Reachability now crosses the transitions and
the call, so the closed-population passes run. QEMU compares the CLI's exact
batch on complete states, and the matched record accepts it.
"""

import json
import pathlib
import random
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tools.oracle.verify_function import _stub_request, check_record, verify
from tools.oracle.function import BRK, EXECUTE, READ, STUB, WRITE
from a64_ssa_cli_calls import CALLEE, CODE, LANDING, SOURCE, STACK, elf, page, words

FLATTENED = bytes.fromhex(
    "fd7bbfa9fd030091ff0301d1f30300917b4200917f0a00b90d000014"
    "28008052680a00b97f0700f909000014"
    "f503009448008052680a00b97f0300f904000014"
    "bf030091fd7bc1a8c0035fd6"
    "680a40b90909007168ffff54a90000100afeff102b7968784a090b8b40011fd6"
)
TABLE = bytes.fromhex("0000040009000000")
# add x0, x0, #7; ret: no callee-saved register or SP is touched.
STUB_WORDS = (0x91001C00, 0xD65F03C0)
READS_REACHABILITY = (
    "unreachable_blocks",
    "phi_constant_fold",
    "constant_propagation",
    "sccp_constant_fold",
    "sccp_storage_reads",
    "branch_retirement",
    "unreachable_after_sccp",
    "predecessor_copies",
    "dead_storage_writes",
)


def main(executable):
    code = FLATTENED + TABLE
    stub = words(STUB_WORDS)
    code_page = page(((0, code),))
    if CODE + len(code) > LANDING or code_page[LANDING - CODE : LANDING - CODE + 4] != BRK.to_bytes(
        4, "little"
    ):
        raise AssertionError("landing is not a BRK past the function")
    stub_page = page(((CALLEE - CALLEE, stub),))
    rng = random.Random(0x4D32464C)
    states = []
    for _ in range(12):
        registers = [rng.getrandbits(64) for _ in range(31)]
        registers[30] = LANDING
        states.append(
            {
                "entry": CODE,
                "registers": registers,
                "nzcv": 0x60000000,
                "sp": STACK + 2048,
                "regions": [
                    {"address": CODE, "protection": READ | EXECUTE, "hex": code_page.hex()},
                    {
                        "address": CALLEE,
                        "protection": READ | EXECUTE | STUB,
                        "hex": stub_page.hex(),
                    },
                    {"address": STACK, "protection": READ | WRITE, "hex": bytes(4096).hex()},
                ],
            }
        )
    with tempfile.TemporaryDirectory() as temp:
        root = pathlib.Path(temp)
        image = root / "flattened.so"
        image.write_bytes(elf(code))
        manifest = {
            "trusted_fixture": True,
            "image": str(image),
            "address": SOURCE,
            "size": len(code),
            "exits": [LANDING],
            "admitted": [{"address": CODE, "hex": FLATTENED.hex()}],
            "data": [{"address": CODE + len(FLATTENED), "hex": TABLE.hex()}],
            "stubs": [{"address": CALLEE, "hex": stub.hex()}],
            "declarations": {
                "abi": "aapcs64",
                "entries": "closed",
                "returns": "leave",
                "image_access": "readable",
            },
            "states": states,
        }
        checked = verify(executable, manifest)
        counts = checked["counts"]
        if (
            checked["status"] != "matched"
            or counts["matched"] != len(states)
            or counts["refuted"]
            or counts["inconclusive"]
            or checked["call_trace"] != "matched"
            or checked["edit_coverage"]["uncovered"]
        ):
            raise AssertionError(f"flattened batch did not match QEMU: {checked}")

        stubs = root / "stubs.bin"
        stubs.write_bytes(_stub_request(((CALLEE, stub),)))
        record = root / "record.txt"
        record.write_text(check_record(checked))
        command = (
            executable,
            "recover-regions",
            str(image),
            "--address",
            f"{SOURCE:x}",
            "--size",
            str(len(code)),
            "--memory-profile",
            "concrete_atomic_scalar",
            "--assume-abi",
            "aapcs64",
            "--assume-entries",
            "closed",
            "--assume-returns",
            "leave",
            "--assume-image-access",
            "readable",
            "--stubs-file",
            str(stubs),
            "--check-file",
            str(record),
        )
        report = json.loads(
            subprocess.run(command, capture_output=True, text=True, check=True).stdout
        )
        proposals = report["ssa_graph"]["proposals"]
        stages = {stage["pass"]: stage for stage in proposals["stages"]}
        declined = [
            name
            for name in READS_REACHABILITY
            if stages[name]["reason"] in ("incomplete_successors", "invalid_graph")
        ]
        proposed = [name for name in READS_REACHABILITY if stages[name]["outcome"] == "proposed"]
        text = report["ssa_graph"]["text"]
        if (
            declined
            or not proposed
            or proposals["status"] != "accepted"
            or " transition " not in text
            or "callee_returns" not in text
        ):
            raise AssertionError(
                f"reachability passes did not run through transitions: "
                f"{declined} {proposed} {proposals['status']}"
            )
        # Without the ABI declaration nothing says a callee returns to its
        # continuation, so every closed-population pass waits for it.
        undeclared = json.loads(
            subprocess.run(
                tuple(
                    item
                    for item in command[: command.index("--stubs-file")]
                    if item not in ("--assume-abi", "aapcs64")
                ),
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )["ssa_graph"]["proposals"]["stages"]
        waiting = {
            stage["pass"]: stage["reason"]
            for stage in undeclared
            if stage["pass"] in READS_REACHABILITY
        }
        if set(waiting.values()) != {"no_call_return_declaration"}:
            raise AssertionError(f"calls were admitted without a declaration: {waiting}")

        # The prologue moves SP and keeps its frame base in X19 across the
        # call; declaring the 80-byte frame private promotes its slots, and
        # QEMU still agrees outside that dead region.
        framed = dict(
            manifest,
            declarations={**manifest["declarations"], "frame": "-80:0", "frame_reach": "none"},
        )
        promoted = verify(executable, framed)
        frame_counts = promoted["counts"]
        if (
            promoted["status"] != "matched"
            or frame_counts["matched"] != len(states)
            or frame_counts["refuted"]
            or frame_counts["inconclusive"]
            or promoted["edit_coverage"]["uncovered"]
        ):
            raise AssertionError(f"promoted frame did not match QEMU: {promoted}")
        framed_report = json.loads(
            subprocess.run(
                command[: command.index("--stubs-file")]
                + ("--assume-private-frame", "-80:0", "--assume-frame-reach", "none"),
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        framed_stages = {
            stage["pass"]: stage for stage in framed_report["ssa_graph"]["proposals"]["stages"]
        }
        if (
            framed_stages["frame_promotion"]["outcome"] != "proposed"
            or not framed_stages["frame_promotion"]["journal"]
        ):
            raise AssertionError(f"frame was not promoted: {framed_stages['frame_promotion']}")
        print(
            json.dumps(
                {
                    "matched": counts["matched"],
                    "promoted_accesses": len(framed_stages["frame_promotion"]["journal"]),
                    "call_events": checked["call_trace_counts"]["compared_events"],
                    "reachability_passes_proposed": proposed,
                    "status": proposals["status"],
                }
            )
        )


if __name__ == "__main__":
    main(sys.argv[1])
