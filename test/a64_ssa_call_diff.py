"""QEMU and SSA whole-function comparison with the same external call stub."""

import json
import pathlib
import random
import struct
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.oracle.function import BRK, EXECUTE, READ, WRITE, FunctionOracle, FunctionState, Region


CODE = 0x200000000
LANDING = CODE + 0x100
CALLEE = CODE + 0x200
STACK = 0x300000000
FUNCTION = (
    0xAA1E03F3,
    0xCA010002,
    0x8A010003,
    0x8B030063,
    0x8B030040,
    0x9400007B,
    0x91000400,
    0xAA1303FE,
    0xD65F03C0,
)
STUB = (0x91001C00, 0xF9000080, 0xD65F03C0)


def probe(executable, left, right, mode):
    process = subprocess.run(
        [executable, "--probe", str(left), str(right), str(mode)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(process.stdout)


def main(executable):
    code = bytearray(struct.pack("<I", BRK) * 1024)
    for index, word in enumerate(FUNCTION):
        struct.pack_into("<I", code, index * 4, word)
    for index, word in enumerate(STUB):
        struct.pack_into("<I", code, 0x200 + index * 4, word)
    code = bytes(code)
    stub = struct.pack("<III", *STUB)
    oracle = FunctionOracle()
    oracle.admit(CODE, code[: len(FUNCTION) * 4])
    rng = random.Random(0x4D324341)
    pairs = [(0, 0), (1, 7), ((1 << 64) - 1, 1)] + [
        (rng.getrandbits(64), rng.getrandbits(64)) for _ in range(13)
    ]
    matched = refused = refuted = 0
    for left, right in pairs:
        registers = [index + 3 for index in range(31)]
        registers[0], registers[1], registers[4], registers[30] = (
            left,
            right,
            STACK + 1024,
            LANDING,
        )
        state = FunctionState(
            tuple(registers),
            0xA0000000,
            STACK + 2048,
            (Region(CODE, code, READ | EXECUTE), Region(STACK, bytes(4096), READ | WRITE)),
        )
        actual = oracle.run(state, CODE, (LANDING,), stubs=((CALLEE, stub),), trusted_fixture=True)
        if not actual.completed or actual.pc != LANDING:
            raise AssertionError("original did not complete after the stub")
        # The stub shares the code page, so QEMU records no call trace here.
        if actual.events is not None:
            raise AssertionError("same-page stub claimed a call trace")
        evaluated = probe(executable, left, right, 0)
        if (
            tuple(evaluated["registers"]) != actual.registers
            or evaluated["nzcv"] != actual.nzcv
            or evaluated["sp"] != actual.sp
            or evaluated["pc"] != actual.pc
            or evaluated["calls"] != 1
            or evaluated["edits"] != 1
            or bytes.fromhex(evaluated["memory_hex"]) != actual.memory[STACK]
        ):
            raise AssertionError(f"SSA/QEMU call disagreement for x0={left}, x1={right}")
        matched += 1
        for mode in (1, 2):
            wrong = probe(executable, left, right, mode)
            if wrong["registers"][0] == actual.registers[0]:
                raise AssertionError(f"call mutation {mode} was not refuted")
            refuted += 1
        missing_store = probe(executable, left, right, 5)
        if (
            missing_store["registers"][0] != actual.registers[0]
            or bytes.fromhex(missing_store["memory_hex"]) == actual.memory[STACK]
        ):
            raise AssertionError("missing stub store was not refuted by memory comparison")
        refuted += 1
        for mode in (3, 4):
            if not probe(executable, left, right, mode).get("declined"):
                raise AssertionError(f"call refusal {mode} was accepted")
            refused += 1
    print(f"SSA call stub: matched={matched} refused={refused} refuted={refuted}")


if __name__ == "__main__":
    main(sys.argv[1])
