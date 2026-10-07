"""Accept, reject or keep provisional the CLI's SSA batch from an independent check record."""

import json
import pathlib
import random
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tools.oracle.verify_function import _stub_request, check_record, verify
from a64_ssa_cli_calls import CALLEE, FUNCTION, SOURCE, STUB, fixture, words, wrap


def run(command, expect_ok=True):
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    report = json.loads(process.stdout)
    if bool(process.returncode) == expect_ok:
        raise AssertionError(f"unexpected exit {process.returncode}: {report}")
    return report


def main(executable):
    stub = words(STUB)
    rng = random.Random(0x4D32414B)
    pairs = [(0, 0), (1, 7), ((1 << 64) - 1, 1)] + [
        (rng.getrandbits(64), rng.getrandbits(64)) for _ in range(9)
    ]
    with tempfile.TemporaryDirectory() as temp:
        root = pathlib.Path(temp)
        manifest = fixture(temp, "function", FUNCTION, ((CALLEE, stub),), pairs)
        checked = verify(executable, manifest)
        if checked["status"] != "matched" or not str(checked["candidate"]).startswith("sha256:"):
            raise AssertionError(f"fixture did not produce a matched, bound check: {checked}")
        record = root / "record.txt"
        record.write_text(check_record(checked))
        stubs = root / "stubs.bin"
        stubs.write_bytes(_stub_request(((CALLEE, stub),)))
        command = (
            executable,
            "recover-regions",
            manifest["image"],
            "--address",
            f"{SOURCE:x}",
            "--size",
            str(len(FUNCTION) * 4),
            "--memory-profile",
            "concrete_atomic_scalar",
            "--assume-abi",
            "aapcs64",
            "--assume-entries",
            "closed",
            "--assume-returns",
            "leave",
        )

        plain = run(command)["ssa_graph"]["proposals"]
        if (
            plain["status"] != "provisional"
            or plain["whole_function_check"] != "NOT CHECKED"
            or plain["acceptance"]["reason"] != "whole_function_check_not_run"
            or plain["acceptance"]["check"] is not None
            or plain["text"] is None
        ):
            raise AssertionError(f"unchecked batch was not provisional: {plain}")

        accepted = run((*command, "--stubs-file", str(stubs), "--check-file", str(record)))[
            "ssa_graph"
        ]["proposals"]
        proposed = sum(stage["outcome"] == "proposed" for stage in accepted["stages"])
        if (
            accepted["status"] != "accepted"
            or accepted["whole_function_check"] != "matched"
            or accepted["candidate_digest"] != checked["candidate"]
            or accepted["acceptance"]["reason"] != "none"
            or not accepted["acceptance"]["check"]["bound"]
            or accepted["acceptance"]["check"]["call_trace"] != "matched"
            or accepted["text"] is None
            or not proposed
            or accepted["tallies"]
            != {
                "stages": len(accepted["stages"]),
                "proposed_stages": proposed,
                "accepted_stages": proposed,
            }
        ):
            raise AssertionError(f"matched record did not accept the batch: {accepted}")

        # The stub declaration is part of the candidate: without it the same
        # record names another candidate.
        unbound = run((*command, "--check-file", str(record)))["ssa_graph"]["proposals"]
        if (
            unbound["status"] != "provisional"
            or unbound["acceptance"]["reason"] != "check_for_other_candidate"
            or unbound["acceptance"]["check"]["bound"]
            or unbound["whole_function_check"] != "NOT CHECKED"
            or unbound["tallies"]["accepted_stages"]
        ):
            raise AssertionError(f"record for another candidate changed status: {unbound}")
        # So is the pipeline.
        named = run(
            (
                *command,
                "--passes",
                "linear_mba,canonical_linear_mba",
                "--stubs-file",
                str(stubs),
                "--check-file",
                str(record),
            )
        )
        named = named["ssa_graph"]["proposals"]
        if (
            named["status"] != "provisional"
            or named["pipeline"]["name"] != "named"
            or [stage["pass"] for stage in named["stages"]]
            != ["linear_mba", "canonical_linear_mba"]
            or named["acceptance"]["reason"] != "check_for_other_candidate"
        ):
            raise AssertionError(f"named pipeline reused the default candidate record: {named}")

        # A real refutation: the verifier compares a wrong call trace for this
        # candidate, and its record rolls the batch back.
        changed = wrap(
            temp,
            "changed_argument",
            executable,
            ("evaluation['call_events'][0]['arguments'][2] ^= 1",),
        )
        refuted = verify(changed, {**manifest, "states": manifest["states"][:2]})
        if refuted["status"] != "refuted" or refuted["candidate"] != checked["candidate"]:
            raise AssertionError(f"wrong call trace was not refuted: {refuted}")
        refuted_record = root / "refuted.txt"
        refuted_record.write_text(check_record(refuted))
        rejected = run((*command, "--stubs-file", str(stubs), "--check-file", str(refuted_record)))[
            "ssa_graph"
        ]["proposals"]
        if (
            rejected["status"] != "rejected"
            or rejected["whole_function_check"] != "refuted"
            or rejected["acceptance"]["reason"] != "whole_function_refuted"
            or rejected["text"] is not None
            or rejected["tallies"]["accepted_stages"]
        ):
            raise AssertionError(f"refuting record did not roll the batch back: {rejected}")

        # A partial check keeps the batch provisional.
        second = wrap(
            temp,
            "second_unresolved",
            executable,
            (
                "import os",
                "marker = __file__ + '.seen'",
                "if os.path.exists(marker):",
                "    evaluation['status'] = 'inconclusive'",
                "    evaluation['stop'] = 'unresolved'",
                "open(marker, 'a').close()",
            ),
        )
        partial = verify(second, {**manifest, "states": manifest["states"][:2]})
        if (
            partial["status"] != "inconclusive"
            or partial["counts"]["matched"] != 1
            or partial["counts"]["inconclusive"] != 1
        ):
            raise AssertionError(f"wrapper did not leave one state inconclusive: {partial}")
        partial_record = root / "partial.txt"
        partial_record.write_text(check_record(partial))
        pending = run(
            (*command, "--stubs-file", str(stubs), "--check-file", str(partial_record)),
            expect_ok=True,
        )
        pending = pending["ssa_graph"]["proposals"]
        if (
            pending["status"] != "provisional"
            or pending["text"] is None
            or pending["whole_function_check"] != "inconclusive"
            or pending["acceptance"]["reason"] != "whole_function_check_inconclusive"
        ):
            raise AssertionError(f"partial check changed the batch status: {pending}")

        # Every run declaration is part of the candidate, even one the SSA text
        # does not show.
        profiled = run(tuple(item for item in command if item not in ("--assume-abi", "aapcs64")))
        if profiled["ssa_graph"]["proposals"]["candidate_digest"] == plain["candidate_digest"]:
            raise AssertionError("dropping --assume-abi kept the candidate digest")

        good = check_record(checked)
        edits = checked["edit_coverage"]["executable"]
        if not edits:
            raise AssertionError(f"fixture has no executable edit to cover: {checked}")
        for text, reason in (
            (
                good.replace("\nstatus matched\n", "\nstatus not_checked\n")
                .replace("\nmatched 12\n", "\nmatched 0\n")
                .replace("\ninconclusive 0\n", "\ninconclusive 12\n"),
                "whole_function_check_not_run",
            ),
            (good.replace(checked["candidate"], "none"), "check_for_other_candidate"),
            # A complete record over another denominator is not this candidate's check.
            (
                good.replace(
                    f"\nedits_executable {edits}\nedits_covered {edits}\n",
                    "\nedits_executable 0\nedits_covered 0\n",
                ),
                "check_coverage_differs",
            ),
            (
                good.replace(
                    "\nstubs_declared 1\nstubs_called 1\n", "\nstubs_declared 0\nstubs_called 0\n"
                ),
                "check_coverage_differs",
            ),
        ):
            if text == good:
                raise AssertionError(f"record edit for {reason} did not apply: {good}")
            edited = root / "edited.txt"
            edited.write_text(text)
            held = run((*command, "--stubs-file", str(stubs), "--check-file", str(edited)))[
                "ssa_graph"
            ]["proposals"]
            if held["status"] != "provisional" or held["acceptance"]["reason"] != reason:
                raise AssertionError(f"expected provisional {reason}: {held}")
        oversized = root / "oversized.txt"
        oversized.write_text(good + " " * 4096)
        bad_stubs = root / "bad_stubs.bin"
        bad_stubs.write_bytes(_stub_request(((CALLEE, stub),)) + b"\0")
        for extra, reason in (
            (("--check-file", str(oversized)), "invalid_check_record"),
            (("--check-file", str(record), "--stubs-file", str(bad_stubs)), "invalid_stubs"),
        ):
            refused = run((*command, *extra), expect_ok=False)
            if refused.get("reason") != reason:
                raise AssertionError(f"{extra} was not refused as {reason}: {refused}")

        # A record whose status disagrees with its counts is refused outright.
        forged = root / "forged.txt"
        forged.write_text(check_record(refuted).replace("status refuted", "status matched"))
        refused = run(
            (*command, "--stubs-file", str(stubs), "--check-file", str(forged)),
            expect_ok=False,
        )
        if refused.get("reason") != "invalid_check_record":
            raise AssertionError(f"inconsistent record was not refused: {refused}")
        for extra, reason in (
            (("--passes", "no_such_pass"), "usage"),
            (("--passes", "linear_mba,linear_mba"), "usage"),
            (("--check-file", str(record), "--state-file", str(record)), "usage"),
        ):
            refused = run((*command, *extra), expect_ok=False)
            if refused.get("reason") != reason:
                raise AssertionError(f"{extra} was not refused as {reason}: {refused}")

        print(
            json.dumps(
                {
                    "accepted_stages": accepted["tallies"]["accepted_stages"],
                    "matched_states": checked["counts"]["matched"],
                    "compared_call_events": checked["call_trace_counts"]["compared_events"],
                    "rejected_refuted_states": refuted["counts"]["refuted"],
                    "candidate": accepted["candidate_digest"],
                }
            )
        )


if __name__ == "__main__":
    main(sys.argv[1])
