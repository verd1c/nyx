import json
import pathlib
import random
import struct
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.oracle.function import BRK, EXECUTE, READ, WRITE, FunctionOracle, FunctionState, Region

CODE, LANDING, STACK = 0x200000000, 0x200000100, 0x300000000


def main(executable):
    code = bytearray(struct.pack("<I", BRK) * 1024)
    words = (0xAA1E03E9, 0x94000003, 0xD2800C60, 0xD65F0120, 0x91000400, 0xAA1E03E1, 0xD65F0120)
    for index, word in enumerate(words):
        struct.pack_into("<I", code, index * 4, word)
    oracle = FunctionOracle()
    oracle.admit(CODE, bytes(code[: len(words) * 4]))
    rng = random.Random(0x1CA11)
    inputs = [0, 1, (1 << 64) - 1] + [rng.getrandbits(64) for _ in range(13)]
    matched = refuted = unresolved = 0
    for value in inputs:
        registers = [index + 3 for index in range(31)]
        registers[0], registers[30] = value, LANDING
        state = FunctionState(
            tuple(registers),
            0xA0000000,
            STACK + 2048,
            (Region(CODE, bytes(code), READ | EXECUTE), Region(STACK, bytes(4096), READ | WRITE)),
        )
        actual = oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
        if not actual.completed or actual.pc != LANDING:
            raise AssertionError("original internal-call fixture did not return")
        reports = []
        for mode in range(3):
            run = subprocess.run(
                [executable, "--internal-call-probe", str(value), str(mode)],
                capture_output=True,
                text=True,
                check=True,
            )
            reports.append(json.loads(run.stdout))
        good, internal_return, wrong_link = reports
        if (
            tuple(good["registers"]) != actual.registers
            or good["nzcv"] != actual.nzcv
            or good["sp"] != actual.sp
            or good["pc"] != actual.pc
            or not good["memory_unchanged"]
            or actual.memory[STACK] != bytes(4096)
        ):
            raise AssertionError("SSA state differs from QEMU")
        matched += 1
        if not wrong_link.get("declined") and (
            wrong_link["registers"][1] == actual.registers[1]
            or wrong_link["registers"][30] == actual.registers[30]
        ):
            raise AssertionError("wrong link mutation was not refuted")
        refuted += 1
        if not internal_return.get("declined") or internal_return["pc"] != CODE + 8:
            raise AssertionError("internal return was falsely reported as function completion")
        unresolved += 1
    print(
        json.dumps(
            {
                "qemu_comparisons": matched,
                "wrong_link_mutations_refuted": refuted,
                "unresolved_internal_returns": unresolved,
                "scope": "direct internal call, GPR/NZCV/SP/PC and writable memory",
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
