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


# DNF of x0 ^ x1 ^ x2, followed by RET. The bits are from an independently
# assembled AArch64 fixture; this script does not derive them from NyxIL.
WORDS = (
    0xAA2003E3,
    0xAA2103E4,
    0xAA2203E5,
    0x8A040066,
    0x8A0200C6,
    0x8A010067,
    0x8A0500E7,
    0x8A040008,
    0x8A050108,
    0x8A010009,
    0x8A020129,
    0xAA0700C6,
    0xAA090108,
    0xAA0800C0,
    0xD65F03C0,
)


def main(executable, probe_executable):
    code = b"".join(struct.pack("<I", word) for word in WORDS)
    page = bytearray(struct.pack("<I", BRK) * 1024)
    page[: len(code)] = code
    page = bytes(page)
    random_values = random.Random(0x4D324D4241)
    inputs = [(0, 0, 0), (1, 0, 1), ((1 << 64) - 1, 0, 0), ((1 << 64) - 1,) * 3]
    inputs += [tuple(random_values.getrandbits(64) for _ in range(3)) for _ in range(12)]
    states = []
    oracle = FunctionOracle()
    oracle.admit(CODE, code)
    candidate_refuted = 0
    with tempfile.TemporaryDirectory() as temp:
        root = pathlib.Path(temp)
        image = root / "mba3.so"
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
            actual = oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
            if not actual.completed or actual.pc != LANDING:
                raise AssertionError("original synthesis fixture did not return")
            for mutate in (False, True):
                probe = json.loads(
                    subprocess.check_output(
                        (
                            probe_executable,
                            "--canonical-mba-probe",
                            *(str(value) for value in values),
                            str(int(mutate)),
                        )
                    )
                )
                if probe["edits"] != 1:
                    raise AssertionError("synthesis probe did not edit one node")
                if mutate:
                    candidate_refuted += probe["registers"][0] != actual.registers[0]
                else:
                    check_state(probe, actual, f"canonical MBA inputs={values}")
            if len(states) == 1:
                request_path.write_bytes(request(state))
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
        stage = next(
            item
            for item in report["ssa_graph"]["proposals"]["stages"]
            if item["pass"] == "canonical_linear_mba"
        )
        if (
            stage["outcome"] != "proposed"
            or len(stage["journal"]) != 1
            or len(stage["journal"][0]["inserted"]) != 1
            or stage["journal"][0]["replacement_op"] != "bit_xor"
            or report["ssa_graph"]["proposals"]["whole_function_check"] != "NOT CHECKED"
        ):
            raise AssertionError(f"new-node edit was not published: {stage}")
        manifest = {
            "trusted_fixture": True,
            "image": str(image),
            "address": SOURCE,
            "size": len(code),
            "exits": [LANDING],
            "admitted": [{"address": CODE, "hex": page[: len(code)].hex()}],
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
        ):
            raise AssertionError(f"QEMU comparison failed: {checked}")
        if candidate_refuted < 1:
            raise AssertionError("mutated synthesized node was not refuted")
        print(
            json.dumps(
                {
                    "matched": counts["matched"],
                    "mutations_refuted": counts["negative_controls_refuted"],
                    "candidate_mutations_refuted": candidate_refuted,
                    "synthesis_edits": len(stage["journal"]),
                    "completed_blocks": report["ssa_graph"]["evaluation"]["completed_blocks"],
                }
            )
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
