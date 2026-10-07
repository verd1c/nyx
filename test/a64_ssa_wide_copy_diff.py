import json
import pathlib
import random
import struct
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.oracle.function import BRK, EXECUTE, READ, WRITE, FunctionOracle, FunctionState, Region
from tools.oracle.aarch64 import Declined

CODE, LANDING, STACK = 0x200000000, 0x200000100, 0x300000000


def main(executable):
    code = bytearray(struct.pack("<I", BRK) * 1024)
    words = (0x3DC00000, 0x3D800020, 0xD65F03C0)
    for index, word in enumerate(words):
        struct.pack_into("<I", code, index * 4, word)
    oracle = FunctionOracle()
    oracle.admit_vector_copy(CODE, bytes(code[:12]))
    for bad in (
        (words[1], words[0], words[2]),
        (words[0], words[1] | 1, words[2]),
        (words[0], words[1], 0x14000000),
    ):
        try:
            FunctionOracle().admit_vector_copy(CODE, struct.pack("<III", *bad))
        except Declined:
            pass
        else:
            raise AssertionError(
                "vector admission accepted an uninitialized register or wrong control"
            )
    rng = random.Random(0x128C0)
    inputs = [1, 1 << 63, (1 << 64) - 1] + [rng.getrandbits(64) | 1 for _ in range(13)]
    matched = refuted = 0
    for high in inputs:
        data = bytearray(4096)
        struct.pack_into("<QQ", data, 0, 0x123456789ABCDEF0, high)
        registers = [index + 3 for index in range(31)]
        registers[0], registers[1], registers[30] = STACK, STACK + 32, LANDING
        state = FunctionState(
            tuple(registers),
            0xA0000000,
            STACK + 2048,
            (Region(CODE, bytes(code), READ | EXECUTE), Region(STACK, bytes(data), READ | WRITE)),
        )
        actual = oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
        if not actual.completed:
            raise AssertionError("QEMU wide copy did not complete")
        reports = []
        for mode in range(2):
            run = subprocess.run(
                [executable, "--wide-copy-probe", str(high), str(mode)],
                capture_output=True,
                text=True,
                check=True,
            )
            reports.append(json.loads(run.stdout))
        good, wrong = reports
        if (
            tuple(good["registers"]) != actual.registers
            or good["nzcv"] != actual.nzcv
            or good["sp"] != actual.sp
            or good["pc"] != actual.pc
            or bytes(good["memory"]) != actual.memory[STACK]
        ):
            raise AssertionError("simplified wide copy differs from QEMU")
        matched += 1
        if bytes(wrong["memory"]) == actual.memory[STACK]:
            raise AssertionError("zeroed upper-half mutation was not refuted")
        refuted += 1
    print(
        json.dumps(
            {
                "qemu_comparisons": matched,
                "upper_half_mutations_refuted": refuted,
                "scope": "128-bit load/store copy; GPR/NZCV/SP/PC and all writable memory",
                "vector_register_comparison": "not checked by this oracle",
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
