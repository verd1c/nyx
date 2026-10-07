"""Independent QEMU comparison for a decoded cross-block SSA copy."""

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
STACK = 0x300000000


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
    for offset, word in ((0, 0x8B030008), (4, 0x14000002), (12, 0x8B020101), (16, 0xD65F03C0)):
        struct.pack_into("<I", code, offset, word)
    code = bytes(code)
    oracle = FunctionOracle()
    oracle.admit(CODE, code[:8])
    oracle.admit(CODE + 12, code[12:20])
    rng = random.Random(0x4D324350)
    pairs = [(0, 0), (1, 7), ((1 << 64) - 1, 1)] + [
        (rng.getrandbits(64), rng.getrandbits(64)) for _ in range(13)
    ]
    matched = refused = refuted = 0
    for left, right in pairs:
        registers = [index + 3 for index in range(31)]
        registers[0], registers[2], registers[30] = left, right, LANDING
        state = FunctionState(
            tuple(registers),
            0xA0000000,
            STACK + 2048,
            (Region(CODE, code, READ | EXECUTE), Region(STACK, bytes(4096), READ | WRITE)),
        )
        actual = oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
        if not actual.completed or actual.pc != LANDING:
            raise AssertionError("original did not complete at landing")
        evaluated = probe(executable, left, right, 0)
        if (
            tuple(evaluated["registers"]) != actual.registers
            or evaluated["nzcv"] != actual.nzcv
            or evaluated["sp"] != actual.sp
            or evaluated["pc"] != actual.pc
            or not evaluated["memory_unchanged"]
            or evaluated["edits"] != 1
            or actual.memory[STACK] != bytes(4096)
        ):
            raise AssertionError(f"SSA/QEMU disagreement for x0={left}, x2={right}")
        matched += 1
        for mode in (1, 2, 3):
            if not probe(executable, left, right, mode).get("declined"):
                raise AssertionError(f"copy mutation {mode} was accepted")
            refused += 1
        wrong = probe(executable, left, right, 4)
        if wrong["registers"][1] == actual.registers[1]:
            raise AssertionError("wrong-value mutation was not refuted")
        refuted += 1
    print(f"SSA predecessor copy: matched={matched} refused={refused} refuted={refuted}")


if __name__ == "__main__":
    main(sys.argv[1])
