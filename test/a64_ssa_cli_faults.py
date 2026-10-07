"""Check register writes retired under the terminal-fault declaration against QEMU.

`mov x9, #11; b 1f; 1: ldr x0, [x1]; ret`: under AAPCS64 alone the load may fault while
X9 still holds 11, so the write stays. Declaring faults terminal retires it,
and the verifier then compares a faulting state's fault, PC and memory but not
its registers. States fault on an unmapped X1 or complete on a mapped one; the
branch lets the retiring block complete first, so a faulting state witnesses it.
"""

import json
import pathlib
import random
import subprocess
import sys
import tempfile
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tools.oracle.verify_function import _compare, verify
from tools.oracle.function import EXECUTE, READ, WRITE
from a64_ssa_cli_calls import CODE, LANDING, SOURCE, STACK, elf, page, words

FUNCTION = (0xD2800169, 0x14000001, 0xF9400020, 0xD65F03C0)
UNMAPPED = 0x500000000
BASE = {"abi": "aapcs64", "entries": "closed", "returns": "leave"}


def retired_writes(executable, image, size, declarations):
    command = [
        executable,
        "recover-regions",
        str(image),
        "--address",
        f"{SOURCE:x}",
        "--size",
        str(size),
        "--memory-profile",
        "concrete_atomic_scalar",
        "--assume-abi",
        "aapcs64",
        "--assume-entries",
        "closed",
        "--assume-returns",
        "leave",
    ]
    if declarations.get("faults"):
        command += ["--assume-faults", declarations["faults"]]
    report = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
    stages = {stage["pass"]: stage for stage in report["ssa_graph"]["proposals"]["stages"]}
    return [item["storage"] for item in stages["dead_storage_writes"]["journal"]]


def main(executable):
    code = words(FUNCTION)
    code_page = page(((0, code),))
    rng = random.Random(0x4641554C)
    states = []
    for index in range(16):
        registers = [rng.getrandbits(64) for _ in range(31)]
        registers[1] = UNMAPPED + 8 * index if index % 2 else STACK + 8 * index
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
                        "address": STACK,
                        "protection": READ | WRITE,
                        "hex": bytes(rng.getrandbits(8) for _ in range(4096)).hex(),
                    },
                ],
            }
        )
    with tempfile.TemporaryDirectory() as temp:
        image = pathlib.Path(temp) / "faults.so"
        image.write_bytes(elf(code))
        if retired_writes(executable, image, len(code), BASE) != []:
            raise AssertionError("a write a fault may observe was retired without the declaration")
        declared = {**BASE, "faults": "terminal"}
        retired = retired_writes(executable, image, len(code), declared)
        if len(retired) != 1:
            raise AssertionError(f"expected the X9 write retired, got {retired}")
        manifest = {
            "trusted_fixture": True,
            "image": str(image),
            "address": SOURCE,
            "size": len(code),
            "exits": [LANDING],
            "admitted": [{"address": CODE, "hex": code.hex()}],
            "declarations": declared,
            "states": states,
        }
        checked = verify(executable, manifest)
        counts = checked["counts"]
        if (
            checked["status"] != "matched"
            or counts["matched"] != len(states)
            or counts["refuted"]
            or counts["inconclusive"]
            or counts["negative_controls_refuted"] != len(states)
            or checked["edit_coverage"]["uncovered"]
        ):
            raise AssertionError(f"terminal-fault batch did not match QEMU: {checked}")

    # The declaration is what lets a faulting state's registers differ.
    original = types.SimpleNamespace(
        completed=False,
        signal=11,
        fault_address=UNMAPPED,
        registers=tuple(range(31)),
        nzcv=0,
        sp=STACK,
        pc=CODE + 8,
        memory={},
    )
    proposed = {
        "status": "fault",
        "fault": {"kind": "unmapped", "address": UNMAPPED},
        "registers": [9 ^ 1 if index == 9 else index for index in range(31)],
        "nzcv": 0,
        "sp": STACK,
        "pc": CODE + 8,
        "memory": [],
    }
    if _compare(original, proposed, BASE, LANDING) != "refuted":
        raise AssertionError("a register difference at a fault matched without the declaration")
    if _compare(original, proposed, {**BASE, "faults": "terminal"}, LANDING) != "matched":
        raise AssertionError("a terminal fault compared its registers")
    moved = dict(proposed, pc=CODE + 12)
    if _compare(original, moved, {**BASE, "faults": "terminal"}, LANDING) != "refuted":
        raise AssertionError("a terminal fault at another PC matched")
    print(json.dumps({"states": len(states), "retired": retired, "status": "matched"}))


if __name__ == "__main__":
    main(sys.argv[1])
