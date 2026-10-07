import json
import pathlib
import random
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.oracle.function import BRK, EXECUTE, READ, WRITE, FunctionOracle, FunctionState, Region
from tools.oracle.verify_function import verify
from a64_ssa_cli_diff import CODE, LANDING, SOURCE, STACK, elf, request
from a64_ssa_function_diff import check_state


# An interleaved three-input MBA for x0 + x1 + x2, independently assembled.
WORDS = (
    0xCA010003,
    0xCA020024,
    0x8A010005,
    0x8A020026,
    0x8B0500A7,
    0x8B0600C8,
    0x8B040069,
    0x8B07012A,
    0x8B08014B,
    0xCB010160,
    0xD65F03C0,
)
# 2*x0 + x1 - x2 from XOR, AND and BIC terms, so the chain needs a doubled
# and a subtracted value.
SIGNED_WORDS = (
    0xCA010003,
    0x8A010004,
    0xCA020005,
    0x8A200046,
    0x8B040067,
    0x8B0400E7,
    0x8B0500E7,
    0xCB0600E7,
    0xCB0600E0,
    0xD65F03C0,
)


# Each mutation changes the result by twice or once one input, so it is
# invisible exactly when that change vanishes modulo 2^64.
def unchanged_by(scale, value):
    return scale * value % (1 << 64) == 0


FORMS = {
    "plus": (
        WORDS,
        "--affine-mba-probe",
        "add",
        1,
        True,
        ((1, lambda values: unchanged_by(2, values[2])),),
    ),
    "signed": (
        SIGNED_WORDS,
        "--signed-affine-mba-probe",
        "sub",
        2,
        False,
        (
            (1, lambda values: unchanged_by(2, values[2])),
            (2, lambda values: unchanged_by(2, values[0])),
        ),
    ),
}


def check(executable, probe_executable, name):
    words, probe_mode, replacement, inserted, direct_unchanged, mutations = FORMS[name]
    code = b"".join(struct.pack("<I", word) for word in words)
    page = bytearray(struct.pack("<I", BRK) * 1024)
    page[: len(code)] = code
    page = bytes(page)
    rng = random.Random(0x4D324146)
    inputs = [(0, 0, 0), (1, 0, 1), ((1 << 64) - 1, 1, 2), ((1 << 64) - 1,) * 3]
    inputs += [tuple(rng.getrandbits(64) for _ in range(3)) for _ in range(12)]
    oracle = FunctionOracle()
    oracle.admit(CODE, code)
    states = []
    candidate_refuted = 0
    expected_refuted = 0
    with tempfile.TemporaryDirectory() as temp:
        root = pathlib.Path(temp)
        image = root / "mba_affine.so"
        image.write_bytes(elf(code))
        request_path = root / "request.bin"
        for values in inputs:
            registers = [index + 3 for index in range(31)]
            registers[:3] = values
            registers[30] = LANDING
            state = FunctionState(
                tuple(registers),
                0xA0000000,
                STACK + 2048,
                (Region(CODE, page, READ | EXECUTE), Region(STACK, bytes(4096), READ | WRITE)),
            )
            states.append(
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
            if len(states) == 1:
                request_path.write_bytes(request(state))
            actual = oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
            if not actual.completed or actual.pc != LANDING:
                raise AssertionError("original affine fixture did not return")
            for mutate, invisible in ((0, None), *mutations):
                probe = json.loads(
                    subprocess.check_output(
                        (
                            probe_executable,
                            probe_mode,
                            *(str(value) for value in values),
                            str(mutate),
                        )
                    )
                )
                if probe["edits"] != 1:
                    raise AssertionError(f"{name} affine probe did not edit one node")
                if mutate:
                    refuted = probe["registers"][0] != actual.registers[0]
                    if refuted == invisible(values):
                        raise AssertionError(
                            f"{name} affine mutation {mutate} result changed: {values}"
                        )
                    candidate_refuted += refuted
                    expected_refuted += not invisible(values)
                else:
                    check_state(probe, actual, f"{name} affine MBA inputs={values}")
        command = (
            executable,
            "recover-regions",
            str(image),
            "--address",
            format(SOURCE, "x"),
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
            "--state-file",
            str(request_path),
        )
        report = json.loads(subprocess.check_output(command))
        stages = report["ssa_graph"]["proposals"]["stages"]
        direct = next(item for item in stages if item["pass"] == "linear_mba")
        stage = next(item for item in stages if item["pass"] == "canonical_linear_mba")
        if (
            (direct_unchanged and direct["outcome"] != "unchanged")
            or stage["outcome"] != "proposed"
            or len(stage["journal"]) != 1
            or len(stage["journal"][0]["inserted"]) != inserted
            or stage["journal"][0]["replacement_op"] != replacement
            or report["ssa_graph"]["proposals"]["whole_function_check"] != "NOT CHECKED"
        ):
            raise AssertionError(f"{name} affine edit was not published: {stage}")
        manifest = {
            "trusted_fixture": True,
            "image": str(image),
            "address": SOURCE,
            "size": len(code),
            "exits": [LANDING],
            "admitted": [{"address": CODE, "hex": code.hex()}],
            "declarations": {"abi": "aapcs64", "entries": "closed", "returns": "leave"},
            "states": states,
        }
        checked = verify(executable, manifest)
        counts = checked["counts"]
        if (
            checked["status"] != "matched"
            or counts["examined"] != len(inputs)
            or counts["matched"] != len(inputs)
            or counts["refuted"]
            or counts["inconclusive"]
            or counts["negative_controls_refuted"] != len(inputs)
            or candidate_refuted != expected_refuted
        ):
            raise AssertionError(f"{name} affine QEMU comparison failed: {checked}")
        print(
            json.dumps(
                {
                    "form": name,
                    "matched": counts["matched"],
                    "candidate_mutations": len(inputs) * len(mutations),
                    "candidate_mutations_refuted": candidate_refuted,
                    "synthesis_edits": len(stage["journal"]),
                    "completed_blocks": report["ssa_graph"]["evaluation"]["completed_blocks"],
                }
            )
        )


def main(executable, probe_executable):
    for name in FORMS:
        check(executable, probe_executable, name)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
