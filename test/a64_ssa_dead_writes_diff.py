"""Independent QEMU comparison for a retired cross-block storage write."""

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


def probe(executable, initial, mode):
    process = subprocess.run(
        [executable, "--probe", str(initial), str(mode)], check=True, capture_output=True, text=True
    )
    return json.loads(process.stdout)


def main(executable):
    code = bytearray(struct.pack("<I", BRK) * 1024)
    for offset, word in (
        (0, 0xD2800168),
        (4, 0x14000003),
        (16, 0xD28002C8),
        (20, 0xAA0803E0),
        (24, 0xD65F03C0),
    ):
        struct.pack_into("<I", code, offset, word)
    code = bytes(code)
    oracle = FunctionOracle()
    oracle.admit(CODE, code[:8])
    oracle.admit(CODE + 16, code[16:28])
    rng = random.Random(0x4D324457)
    values = [0, 11, 22, (1 << 64) - 1] + [rng.getrandbits(64) for _ in range(12)]
    matched = refused = refuted = 0
    for initial in values:
        registers = [index + 3 for index in range(31)]
        registers[8], registers[30] = initial, LANDING
        state = FunctionState(
            tuple(registers),
            0xA0000000,
            STACK + 2048,
            (Region(CODE, code, READ | EXECUTE), Region(STACK, bytes(4096), READ | WRITE)),
        )
        actual = oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
        if not actual.completed or actual.pc != LANDING:
            raise AssertionError("original did not complete at landing")
        evaluated = probe(executable, initial, 0)
        if (
            tuple(evaluated["registers"]) != actual.registers
            or evaluated["nzcv"] != actual.nzcv
            or evaluated["sp"] != actual.sp
            or evaluated["pc"] != actual.pc
            or not evaluated["memory_unchanged"]
            or evaluated["edits"] != 1
            or actual.memory[STACK] != bytes(4096)
        ):
            raise AssertionError(f"SSA/QEMU disagreement for x8={initial}")
        matched += 1
        for mode in (1, 2, 3):
            if not probe(executable, initial, mode).get("declined"):
                raise AssertionError(f"dead-write mutation {mode} was accepted")
            refused += 1
        wrong = probe(executable, initial, 4)
        if wrong["registers"][0] == actual.registers[0]:
            raise AssertionError("wrong-value mutation was not refuted")
        refuted += 1
    fault_code = bytearray(struct.pack("<I", BRK) * 1024)
    for offset, word in ((0, 0xD2800168), (4, 0xF9400020), (8, 0xD28002C8), (12, 0xD65F03C0)):
        struct.pack_into("<I", fault_code, offset, word)
    fault_code = bytes(fault_code)
    fault_oracle = FunctionOracle()
    fault_oracle.admit(CODE, fault_code[:16])
    for address, expected in ((0x400000000, 11), (STACK + 2048, 22)):
        registers = [index + 3 for index in range(31)]
        registers[1], registers[30] = address, LANDING
        state = FunctionState(
            tuple(registers),
            0xA0000000,
            STACK + 2048,
            (Region(CODE, fault_code, READ | EXECUTE), Region(STACK, bytes(4096), READ | WRITE)),
        )
        actual = fault_oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
        if (
            actual.registers[8] != expected
            or (address == 0x400000000 and actual.completed)
            or (address != 0x400000000 and not actual.completed)
        ):
            raise AssertionError("fault fixture did not expose expected x8 state")
        if not probe(executable, address, 5).get("declined"):
            raise AssertionError("potentially faulting write was retired")
        refused += 1
    print(f"SSA dead writes: matched={matched} refused={refused} refuted={refuted}")


if __name__ == "__main__":
    main(sys.argv[1])
