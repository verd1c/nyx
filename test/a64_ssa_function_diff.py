"""Independent QEMU checks of SSA control and a provisional MBA rewrite."""

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


def code_page():
    code = bytearray(struct.pack("<I", BRK) * 1024)
    for offset, word in (
        (0, 0xB4000080),  # cbz x0, +0x10
        (4, 0xD28002C0),  # mov x0, #22
        (8, 0xD65F03C0),  # ret
        (16, 0xD2800160),  # mov x0, #11
        (20, 0xD65F03C0),
        (32, 0x14000000),
    ):  # unreachable self-loop
        struct.pack_into("<I", code, offset, word)
    return bytes(code)


def mba_code_page():
    code = bytearray(struct.pack("<I", BRK) * 1024)
    for offset, word in enumerate((0xCA010002, 0x8A010003, 0x8B030063, 0x8B030040, 0xD65F03C0)):
        struct.pack_into("<I", code, offset * 4, word)
    return bytes(code)


def pure_dce_code_page():
    code = bytearray(struct.pack("<I", BRK) * 1024)
    for offset, word in enumerate((0x8B010000, 0x8B01001F, 0xD65F03C0)):
        struct.pack_into("<I", code, offset * 4, word)
    return bytes(code)


def phi_constant_code_page():
    code = bytearray(struct.pack("<I", BRK) * 1024)
    for offset, word in (
        (0, 0xB4000080),
        (4, 0xD2800168),
        (8, 0x14000004),
        (16, 0xD2800168),
        (20, 0x14000001),
        (24, 0x91000500),
        (28, 0xD65F03C0),
    ):
        struct.pack_into("<I", code, offset, word)
    return bytes(code)


def computed_phi_code_page():
    code = bytearray(struct.pack("<I", BRK) * 1024)
    for offset, word in (
        (0, 0xB40000A0),
        (4, 0xD2800148),
        (8, 0x91000508),
        (12, 0x14000004),
        (20, 0xD2800168),
        (24, 0x14000001),
        (28, 0x91000500),
        (32, 0xD65F03C0),
    ):
        struct.pack_into("<I", code, offset, word)
    return bytes(code)


def sccp_code_page():
    code = bytearray(struct.pack("<I", BRK) * 1024)
    for offset, word in (
        (0, 0xB40000BF),
        (4, 0xD2800148),
        (8, 0x91000508),
        (12, 0x14000004),
        (20, 0xD28002C8),
        (24, 0x14000001),
        (28, 0x91000500),
        (32, 0xD65F03C0),
    ):
        struct.pack_into("<I", code, offset, word)
    return bytes(code)


def read_condition_code_page():
    code = bytearray(struct.pack("<I", BRK) * 1024)
    for offset, word in (
        (0, 0xD2800000),
        (4, 0x14000002),
        (12, 0xAA0003E1),
        (16, 0xB4000060),
        (20, 0xD2800160),
        (24, 0xD65F03C0),
        (28, 0xD28002C0),
        (32, 0xD65F03C0),
    ):
        struct.pack_into("<I", code, offset, word)
    return bytes(code)


def frame_dead_code_page():
    code = bytearray(struct.pack("<I", BRK) * 1024)
    for offset, word in enumerate((0xF81F03E0, 0xF85F03FF, 0xD65F03C0)):
        struct.pack_into("<I", code, offset * 4, word)
    return bytes(code)


def frame_promotion_code_page():
    code = bytearray(struct.pack("<I", BRK) * 1024)
    for offset, word in enumerate(
        (
            0xB4000080,
            0xD28002C1,
            0xF81F03E1,
            0x14000003,
            0xD2800161,
            0xF81F03E1,
            0xF85F03E0,
            0xD65F03C0,
        )
    ):
        struct.pack_into("<I", code, offset * 4, word)
    return bytes(code)


def literal_load_code_page(live):
    code = bytearray(struct.pack("<I", BRK) * 1024)
    struct.pack_into("<I", code, 0, 0x58008000 if live else 0x5800801F)
    struct.pack_into("<I", code, 4, 0xD65F03C0)
    return bytes(code)


def literal_load_data_page():
    data = bytearray(4096)
    struct.pack_into("<Q", data, 0, 0x12345678)
    return bytes(data)


def selected_load_code_page():
    code = bytearray(struct.pack("<I", BRK) * 1024)
    for offset, word in enumerate((0xF100001F, 0x9A9F17E8, 0x10007FC9, 0xF8687921, 0xD65F03C0)):
        struct.pack_into("<I", code, offset * 4, word)
    return bytes(code)


def selected_load_data_page():
    data = bytearray(4096)
    struct.pack_into("<QQ", data, 0, 11, 22)
    return bytes(data)


def bounded_table_code_page():
    code = bytearray(struct.pack("<I", BRK) * 1024)
    for offset, word in enumerate(
        (0xF100101F, 0x54000082, 0x1000FFC1, 0xF8607820, 0xD65F03C0, 0xD65F03C0)
    ):
        struct.pack_into("<I", code, offset * 4, word)
    return bytes(code)


def bounded_table_data_page():
    data = bytearray(4096)
    struct.pack_into("<QQQQ", data, 0, 11, 22, 33, 44)
    return bytes(data)


def two_bounded_table_code_page():
    code = bytearray(struct.pack("<I", BRK) * 1024)
    for offset, word in enumerate(
        (0xF100101F, 0x540000A2, 0x1000FFC1, 0xF8607823, 0xF8607822, 0xD65F03C0, 0xD65F03C0)
    ):
        struct.pack_into("<I", code, offset * 4, word)
    return bytes(code)


def probe(executable, value, mutation):
    result = subprocess.run(
        [executable, "--probe", str(value), str(int(mutation))],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def dce_probe(executable, value, mutation):
    result = subprocess.run(
        [executable, "--dce-probe", str(value), str(int(mutation))],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def mba_probe(executable, left, right, mutation):
    result = subprocess.run(
        [executable, "--mba-probe", str(left), str(right), str(int(mutation))],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def pure_dce_probe(executable, left, right, mutation):
    result = subprocess.run(
        [executable, "--pure-dce-probe", str(left), str(right), str(int(mutation))],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def phi_constant_probe(executable, value, mode):
    result = subprocess.run(
        [executable, "--phi-constant-probe", str(value), str(mode)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def computed_phi_probe(executable, value, mode):
    result = subprocess.run(
        [executable, "--computed-phi-probe", str(value), str(mode)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def sccp_probe(executable, value, mode):
    result = subprocess.run(
        [executable, "--sccp-probe", str(value), str(mode)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def read_condition_probe(executable, value, mode):
    result = subprocess.run(
        [executable, "--read-condition-probe", str(value), str(mode)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def frame_dead_probe(executable, value, mode):
    result = subprocess.run(
        [executable, "--frame-dead-probe", str(value), str(mode)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def frame_promotion_probe(executable, value, mutation):
    result = subprocess.run(
        [executable, "--frame-promotion-probe", str(value), str(int(mutation))],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def literal_load_probe(executable, value, live, mode):
    result = subprocess.run(
        [executable, "--literal-load-probe", str(value), str(int(live)), str(mode)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def selected_load_probe(executable, value, mode):
    result = subprocess.run(
        [executable, "--selected-load-probe", str(value), str(mode)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def bounded_table_probe(executable, value, mode):
    result = subprocess.run(
        [executable, "--bounded-table-probe", str(value), str(mode)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def two_bounded_table_probe(executable, value, mode):
    result = subprocess.run(
        [executable, "--two-bounded-table-probe", str(value), str(mode)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def check_state(evaluated, actual, value):
    if (
        tuple(evaluated["registers"]) != actual.registers
        or evaluated["nzcv"] != actual.nzcv
        or evaluated["sp"] != actual.sp
        or evaluated["pc"] != actual.pc
        or not evaluated["memory_unchanged"]
        or actual.memory[STACK] != bytes(4096)
    ):
        raise AssertionError(f"SSA/QEMU disagreement for {value}")


def main(executable):
    code = code_page()
    oracle = FunctionOracle()
    oracle.admit(CODE, code[:12])
    oracle.admit(CODE + 16, code[16:24])
    oracle.admit(CODE + 32, code[32:36])
    rng = random.Random(0x4D325146)
    inputs = [0, 1, (1 << 64) - 1] + [rng.getrandbits(64) for _ in range(13)]
    matched = refuted = dce_matched = dce_refuted = dce_target_refused = 0
    for value in inputs:
        registers = [index + 3 for index in range(31)]
        registers[0] = value
        registers[30] = LANDING
        state = FunctionState(
            tuple(registers),
            0xA0000000,
            STACK + 2048,
            (Region(CODE, code, READ | EXECUTE), Region(STACK, bytes(4096), READ | WRITE)),
        )
        actual = oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
        if not actual.completed or actual.pc != LANDING:
            raise AssertionError("original did not complete at the declared landing")
        evaluated = probe(executable, value, False)
        check_state(evaluated, actual, f"x0={value}")
        matched += 1
        wrong = probe(executable, value, True)
        if wrong.get("declined") or wrong["registers"][0] != actual.registers[0]:
            refuted += 1
        pruned = dce_probe(executable, value, False)
        check_state(pruned, actual, f"DCE x0={value}")
        if pruned["removed_blocks"] != 1:
            raise AssertionError("expected one journaled unreachable block deletion")
        dce_matched += 1
        wrong = dce_probe(executable, value, True)
        if wrong.get("declined") or wrong["registers"][0] != actual.registers[0]:
            dce_refuted += 1
        wrong_target = dce_probe(executable, value, 2)
        if wrong_target.get("declined"):
            dce_target_refused += 1
    if refuted != len(inputs):
        raise AssertionError("swapped-edge mutation was not refuted for every state")
    if dce_refuted != len(inputs):
        raise AssertionError("DCE missing-successor mutation was not refused for every state")
    if dce_target_refused != len(inputs):
        raise AssertionError("DCE changed-target mutation was not refused for every state")
    mba_code = mba_code_page()
    mba_oracle = FunctionOracle()
    mba_oracle.admit(CODE, mba_code[:20])
    mba_matched = mba_refuted = edits = 0
    pairs = [(0, 1), (7, 3), ((1 << 64) - 1, 5)] + [
        (rng.getrandbits(64), rng.getrandbits(64) | 1) for _ in range(13)
    ]
    for left, right in pairs:
        registers = [index + 3 for index in range(31)]
        registers[0], registers[1], registers[30] = left, right, LANDING
        state = FunctionState(
            tuple(registers),
            0xA0000000,
            STACK + 2048,
            (Region(CODE, mba_code, READ | EXECUTE), Region(STACK, bytes(4096), READ | WRITE)),
        )
        actual = mba_oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
        if not actual.completed or actual.pc != LANDING:
            raise AssertionError("MBA original did not reach landing")
        evaluated = mba_probe(executable, left, right, False)
        check_state(evaluated, actual, f"MBA x0={left}, x1={right}")
        if evaluated["edits"] != 1:
            raise AssertionError("expected one journaled SSA MBA edit")
        edits += evaluated["edits"]
        mba_matched += 1
        wrong = mba_probe(executable, left, right, True)
        if wrong["registers"][0] != actual.registers[0]:
            mba_refuted += 1
    if mba_refuted != len(pairs):
        raise AssertionError("MBA rewrite mutation was not refuted for every state")
    pure_code = pure_dce_code_page()
    pure_oracle = FunctionOracle()
    pure_oracle.admit(CODE, pure_code[:12])
    pure_matched = pure_refused = pure_edits = 0
    for left, right in pairs:
        registers = [index + 3 for index in range(31)]
        registers[0], registers[1], registers[30] = left, right, LANDING
        state = FunctionState(
            tuple(registers),
            0xA0000000,
            STACK + 2048,
            (Region(CODE, pure_code, READ | EXECUTE), Region(STACK, bytes(4096), READ | WRITE)),
        )
        actual = pure_oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
        if not actual.completed or actual.pc != LANDING:
            raise AssertionError("pure DCE original did not reach landing")
        evaluated = pure_dce_probe(executable, left, right, False)
        check_state(evaluated, actual, f"pure DCE x0={left}, x1={right}")
        if evaluated["edits"] < 1:
            raise AssertionError("expected a journaled dead pure node")
        pure_edits += evaluated["edits"]
        pure_matched += 1
        if pure_dce_probe(executable, left, right, True).get("declined"):
            pure_refused += 1
    if pure_refused != len(pairs):
        raise AssertionError("live pure node deletion was not refused for every state")
    phi_code = phi_constant_code_page()
    phi_oracle = FunctionOracle()
    phi_oracle.admit(CODE, phi_code[:12])
    phi_oracle.admit(CODE + 16, phi_code[16:32])
    phi_matched = phi_refused = phi_refuted = 0
    for value in inputs:
        registers = [index + 3 for index in range(31)]
        registers[0], registers[30] = value, LANDING
        state = FunctionState(
            tuple(registers),
            0,
            STACK + 2048,
            (Region(CODE, phi_code, READ | EXECUTE), Region(STACK, bytes(4096), READ | WRITE)),
        )
        actual = phi_oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
        if not actual.completed or actual.pc != LANDING:
            raise AssertionError("phi-constant original did not reach landing")
        evaluated = phi_constant_probe(executable, value, 0)
        check_state(evaluated, actual, f"phi-constant x0={value}")
        if evaluated["edits"] != 1:
            raise AssertionError("expected one journaled phi-constant fold")
        phi_matched += 1
        if phi_constant_probe(executable, value, 1).get("declined"):
            phi_refused += 1
        wrong = phi_constant_probe(executable, value, 2)
        if wrong["registers"][0] != actual.registers[0]:
            phi_refuted += 1
    if phi_refused != len(inputs) or phi_refuted != len(inputs):
        raise AssertionError("phi-constant mutation coverage is incomplete")
    computed_code = computed_phi_code_page()
    computed_oracle = FunctionOracle()
    computed_oracle.admit(CODE, computed_code[:16])
    computed_oracle.admit(CODE + 20, computed_code[20:36])
    computed_matched = computed_refused = computed_refuted = 0
    for value in inputs:
        registers = [index + 3 for index in range(31)]
        registers[0], registers[30] = value, LANDING
        state = FunctionState(
            tuple(registers),
            0,
            STACK + 2048,
            (Region(CODE, computed_code, READ | EXECUTE), Region(STACK, bytes(4096), READ | WRITE)),
        )
        actual = computed_oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
        if not actual.completed or actual.pc != LANDING:
            raise AssertionError("computed-phi original did not reach landing")
        evaluated = computed_phi_probe(executable, value, 0)
        check_state(evaluated, actual, f"computed-phi x0={value}")
        if evaluated["edits"] != 2:
            raise AssertionError("expected two journaled computed-phi folds")
        computed_matched += 1
        if computed_phi_probe(executable, value, 1).get("declined"):
            computed_refused += 1
        wrong = computed_phi_probe(executable, value, 2)
        if wrong["registers"][0] != actual.registers[0]:
            computed_refuted += 1
    if computed_refused != len(inputs) or computed_refuted != len(inputs):
        raise AssertionError("computed-phi mutation coverage is incomplete")
    sccp_code = sccp_code_page()
    sccp_oracle = FunctionOracle()
    sccp_oracle.admit(CODE, sccp_code[:16])
    sccp_oracle.admit(CODE + 20, sccp_code[20:36])
    sccp_matched = sccp_refused = sccp_refuted = sccp_edge_refused = 0
    for value in inputs:
        registers = [index + 3 for index in range(31)]
        registers[0], registers[30] = value, LANDING
        state = FunctionState(
            tuple(registers),
            0,
            STACK + 2048,
            (Region(CODE, sccp_code, READ | EXECUTE), Region(STACK, bytes(4096), READ | WRITE)),
        )
        actual = sccp_oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
        if not actual.completed or actual.pc != LANDING:
            raise AssertionError("SCCP original did not reach landing")
        evaluated = sccp_probe(executable, value, 0)
        check_state(evaluated, actual, f"SCCP x0={value}")
        if (
            evaluated["edits"] < 1
            or evaluated["retired_edges"] != 1
            or evaluated["removed_blocks"] != 1
        ):
            raise AssertionError("expected SCCP fold, edge retirement and block removal")
        sccp_matched += 1
        if sccp_probe(executable, value, 1).get("declined"):
            sccp_refused += 1
        wrong = sccp_probe(executable, value, 2)
        if wrong["registers"][0] != actual.registers[0]:
            sccp_refuted += 1
        if sccp_probe(executable, value, 3).get("declined"):
            sccp_edge_refused += 1
    if (
        sccp_refused != len(inputs)
        or sccp_refuted != len(inputs)
        or sccp_edge_refused != len(inputs)
    ):
        raise AssertionError("SCCP mutation coverage is incomplete")
    read_code = read_condition_code_page()
    read_oracle = FunctionOracle()
    read_oracle.admit(CODE, read_code[:8])
    read_oracle.admit(CODE + 12, read_code[12:36])
    read_matched = read_refused = read_refuted = 0
    for value in inputs:
        registers = [index + 3 for index in range(31)]
        registers[0], registers[30] = value, LANDING
        state = FunctionState(
            tuple(registers),
            0,
            STACK + 2048,
            (Region(CODE, read_code, READ | EXECUTE), Region(STACK, bytes(4096), READ | WRITE)),
        )
        actual = read_oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
        if not actual.completed or actual.pc != LANDING:
            raise AssertionError("storage-read original did not reach landing")
        evaluated = read_condition_probe(executable, value, 0)
        check_state(evaluated, actual, f"storage-read x0={value}")
        if evaluated["read_folds"] != 1:
            raise AssertionError("expected a proved storage-read fold")
        read_matched += 1
        if read_condition_probe(executable, value, 1).get("declined"):
            read_refused += 1
        wrong = read_condition_probe(executable, value, 2)
        if wrong["registers"][1] != actual.registers[1]:
            read_refuted += 1
    if read_refused != len(inputs) or read_refuted != len(inputs):
        raise AssertionError("storage-read mutation coverage is incomplete")
    frame_code = frame_dead_code_page()
    frame_oracle = FunctionOracle()
    frame_oracle.admit(CODE, frame_code[:12])
    frame_matched = frame_refused = frame_forged_refused = 0
    for value in inputs:
        registers = [index + 3 for index in range(31)]
        registers[0], registers[30] = value, LANDING
        state = FunctionState(
            tuple(registers),
            0xA0000000,
            STACK + 2048,
            (Region(CODE, frame_code, READ | EXECUTE), Region(STACK, bytes(4096), READ | WRITE)),
        )
        actual = frame_oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
        if not actual.completed or actual.pc != LANDING:
            raise AssertionError("private-frame original did not reach landing")
        outside = bytearray(actual.memory[STACK])
        if outside[2032:2040] != value.to_bytes(8, "little"):
            raise AssertionError("private-frame original did not write its declared slot")
        outside[2032:2040] = bytes(8)
        if outside != bytes(4096):
            raise AssertionError("private-frame original changed external memory")
        evaluated = frame_dead_probe(executable, value, 0)
        if (
            tuple(evaluated["registers"]) != actual.registers
            or evaluated["nzcv"] != actual.nzcv
            or evaluated["sp"] != actual.sp
            or evaluated["pc"] != actual.pc
            or not evaluated["memory_unchanged"]
            or evaluated["edits"] != 2
            or evaluated["slot_offset"] != -16
            or evaluated["slot_size"] != 8
        ):
            raise AssertionError(f"private-frame SSA/QEMU disagreement for x0={value}")
        frame_matched += 1
        if frame_dead_probe(executable, value, 1).get("declined"):
            frame_refused += 1
        if frame_dead_probe(executable, value, 2).get("declined"):
            frame_forged_refused += 1
    if frame_refused != len(inputs):
        raise AssertionError("private-frame escape was not refused for every state")
    if frame_forged_refused != len(inputs):
        raise AssertionError("forged private-frame access was not refused for every state")
    promotion_code = frame_promotion_code_page()
    promotion_oracle = FunctionOracle()
    promotion_oracle.admit(CODE, promotion_code[:32])
    promotion_matched = promotion_refused = promotion_refuted = 0
    for value in inputs:
        registers = [index + 3 for index in range(31)]
        registers[0], registers[30] = value, LANDING
        state = FunctionState(
            tuple(registers),
            0xA0000000,
            STACK + 2048,
            (
                Region(CODE, promotion_code, READ | EXECUTE),
                Region(STACK, bytes(4096), READ | WRITE),
            ),
        )
        actual = promotion_oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
        if not actual.completed or actual.pc != LANDING:
            raise AssertionError("frame-promotion original did not reach landing")
        expected = 11 if value == 0 else 22
        outside = bytearray(actual.memory[STACK])
        if outside[2032:2040] != expected.to_bytes(8, "little"):
            raise AssertionError("frame-promotion original did not write its declared slot")
        outside[2032:2040] = bytes(8)
        if outside != bytes(4096):
            raise AssertionError("frame-promotion original changed external memory")
        evaluated = frame_promotion_probe(executable, value, False)
        if (
            tuple(evaluated["registers"]) != actual.registers
            or evaluated["nzcv"] != actual.nzcv
            or evaluated["sp"] != actual.sp
            or evaluated["pc"] != actual.pc
            or not evaluated["memory_unchanged"]
            or evaluated["edits"] != 3
            or evaluated["join_inputs"] != 2
        ):
            raise AssertionError(f"frame-promotion SSA/QEMU disagreement for x0={value}")
        promotion_matched += 1
        if frame_promotion_probe(executable, value, True).get("declined"):
            promotion_refused += 1
        wrong = frame_promotion_probe(executable, value, 2)
        if not wrong.get("declined") and tuple(wrong["registers"]) != actual.registers:
            promotion_refuted += 1
    if promotion_refused != len(inputs):
        raise AssertionError("corrupt frame phi was not refused for every state")
    if promotion_refuted != len(inputs):
        raise AssertionError("wrong promoted frame value was not refuted for every state")
    literal_matched = literal_refused = literal_refuted = live_load_refused = 0
    for live in (False, True):
        literal_code = literal_load_code_page(live)
        literal_data = literal_load_data_page()
        literal_oracle = FunctionOracle()
        literal_oracle.admit(CODE, literal_code[:8])
        for value in inputs:
            registers = [index + 3 for index in range(31)]
            registers[0], registers[30] = value, LANDING
            state = FunctionState(
                tuple(registers),
                0xA0000000,
                STACK + 2048,
                (
                    Region(CODE, literal_code, READ | EXECUTE),
                    Region(CODE + 4096, literal_data, READ),
                    Region(STACK, bytes(4096), READ | WRITE),
                ),
            )
            actual = literal_oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
            if not actual.completed or actual.pc != LANDING:
                raise AssertionError("literal-load original did not reach landing")
            evaluated = literal_load_probe(executable, value, live, 0)
            check_state(evaluated, actual, f"literal-load x0={value}, live={live}")
            if not evaluated["retired_access"] or evaluated["dead_nodes"] != (1 if live else 2):
                raise AssertionError("literal-load effect or dead values were not retired")
            literal_matched += 1
            if literal_load_probe(executable, value, live, 1).get("declined"):
                literal_refused += 1
            if live:
                wrong = literal_load_probe(executable, value, True, 2)
                if not wrong.get("declined") and wrong["registers"][0] != actual.registers[0]:
                    literal_refuted += 1
                if literal_load_probe(executable, value, True, 3).get("declined"):
                    live_load_refused += 1
    if (
        literal_refused != 2 * len(inputs)
        or literal_refuted != len(inputs)
        or live_load_refused != len(inputs)
    ):
        raise AssertionError("literal-load mutation coverage is incomplete")
    selected_code = selected_load_code_page()
    selected_data = selected_load_data_page()
    selected_oracle = FunctionOracle()
    selected_oracle.admit(CODE, selected_code[:20])
    selected_matched = selected_refused = selected_refuted = selected_address_refused = 0
    selected_condition_refused = 0
    for value in inputs:
        registers = [index + 3 for index in range(31)]
        registers[0], registers[30] = value, LANDING
        state = FunctionState(
            tuple(registers),
            0xA0000000,
            STACK + 2048,
            (
                Region(CODE, selected_code, READ | EXECUTE),
                Region(CODE + 4096, selected_data, READ),
                Region(STACK, bytes(4096), READ | WRITE),
            ),
        )
        actual = selected_oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
        if not actual.completed or actual.pc != LANDING:
            raise AssertionError("selected-load original did not reach landing")
        evaluated = selected_load_probe(executable, value, 0)
        check_state(evaluated, actual, f"selected-load x0={value}")
        if not evaluated["retired_access"] or evaluated["dead_nodes"] != 3:
            raise AssertionError("selected load or address was not retired")
        selected_matched += 1
        if selected_load_probe(executable, value, 1).get("declined"):
            selected_refused += 1
        wrong = selected_load_probe(executable, value, 2)
        if not wrong.get("declined") and wrong["registers"][1] != actual.registers[1]:
            selected_refuted += 1
        if selected_load_probe(executable, value, 3).get("declined"):
            selected_address_refused += 1
        if selected_load_probe(executable, value, 4).get("declined"):
            selected_condition_refused += 1
    if (
        selected_refused != len(inputs)
        or selected_refuted != len(inputs)
        or selected_address_refused != len(inputs)
        or selected_condition_refused != len(inputs)
    ):
        raise AssertionError("selected-load mutation coverage is incomplete")
    table_code = bounded_table_code_page()
    table_data = bounded_table_data_page()
    table_oracle = FunctionOracle()
    table_oracle.admit(CODE, table_code[:24])
    table_matched = table_refused = table_refuted = table_stale_refused = 0
    table_inputs = [0, 1, 2, 3, 4, (1 << 64) - 1]
    for value in table_inputs:
        registers = [index + 3 for index in range(31)]
        registers[0], registers[30] = value, LANDING
        state = FunctionState(
            tuple(registers),
            0,
            STACK + 2048,
            (
                Region(CODE, table_code, READ | EXECUTE),
                Region(CODE + 8192, table_data, READ),
                Region(STACK, bytes(4096), READ | WRITE),
            ),
        )
        actual = table_oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
        if not actual.completed or actual.pc != LANDING:
            raise AssertionError("bounded-table original did not reach landing")
        evaluated = bounded_table_probe(executable, value, 0)
        check_state(evaluated, actual, f"bounded-table x0={value}")
        if not evaluated["retired_access"] or evaluated["dead_nodes"] != 3:
            raise AssertionError("bounded-table load and address were not retired")
        table_matched += 1
        if bounded_table_probe(executable, value, 1).get("declined"):
            table_refused += 1
        if value < 4:
            wrong = bounded_table_probe(executable, value, 2)
            if wrong["registers"][0] != actual.registers[0]:
                table_refuted += 1
        if bounded_table_probe(executable, value, 3).get("declined"):
            table_stale_refused += 1
    if (
        table_refused != len(table_inputs)
        or table_refuted != 4
        or table_stale_refused != len(table_inputs)
    ):
        raise AssertionError("bounded-table mutation coverage is incomplete")
    two_code = two_bounded_table_code_page()
    two_oracle = FunctionOracle()
    two_oracle.admit(CODE, two_code[:28])
    two_matched = two_refused = two_second_refuted = two_first_refuted = 0
    for value in table_inputs:
        registers = [index + 3 for index in range(31)]
        registers[0], registers[30] = value, LANDING
        state = FunctionState(
            tuple(registers),
            0,
            STACK + 2048,
            (
                Region(CODE, two_code, READ | EXECUTE),
                Region(CODE + 8192, table_data, READ),
                Region(STACK, bytes(4096), READ | WRITE),
            ),
        )
        actual = two_oracle.run(state, CODE, (LANDING,), trusted_fixture=True)
        if not actual.completed or actual.pc != LANDING:
            raise AssertionError("two-table original did not reach landing")
        evaluated = two_bounded_table_probe(executable, value, 0)
        check_state(evaluated, actual, f"two-table x0={value}")
        if evaluated["folds"] != 2:
            raise AssertionError("expected two journaled table folds")
        two_matched += 1
        if two_bounded_table_probe(executable, value, 1).get("declined"):
            two_refused += 1
        if value < 4:
            wrong = two_bounded_table_probe(executable, value, 2)
            if wrong["registers"][2] != actual.registers[2]:
                two_second_refuted += 1
            wrong = two_bounded_table_probe(executable, value, 3)
            if wrong["registers"][3] != actual.registers[3]:
                two_first_refuted += 1
    if two_refused != len(table_inputs) or two_second_refuted != 4 or two_first_refuted != 4:
        raise AssertionError("two-table mutation coverage is incomplete")
    print(
        json.dumps(
            {
                "whole_function_ssa_qemu_comparisons": matched,
                "swapped_edge_mutations_refuted": refuted,
                "unreachable_dce_qemu_comparisons": dce_matched,
                "dce_missing_successor_mutations_refused": dce_refuted,
                "dce_changed_target_mutations_refused": dce_target_refused,
                "mba_candidate_qemu_comparisons": mba_matched,
                "mba_candidate_edits_seen": edits,
                "mba_mutations_refuted": mba_refuted,
                "pure_dce_qemu_comparisons": pure_matched,
                "pure_dce_edits_seen": pure_edits,
                "pure_dce_live_node_mutations_refused": pure_refused,
                "phi_constant_qemu_comparisons": phi_matched,
                "phi_constant_open_scope_mutations_refused": phi_refused,
                "phi_constant_wrong_value_mutations_refuted": phi_refuted,
                "computed_phi_qemu_comparisons": computed_matched,
                "computed_phi_open_scope_mutations_refused": computed_refused,
                "computed_phi_wrong_value_mutations_refuted": computed_refuted,
                "sccp_qemu_comparisons": sccp_matched,
                "sccp_open_scope_mutations_refused": sccp_refused,
                "sccp_wrong_value_mutations_refuted": sccp_refuted,
                "sccp_wrong_edge_mutations_refused": sccp_edge_refused,
                "storage_read_qemu_comparisons": read_matched,
                "storage_read_open_scope_mutations_refused": read_refused,
                "storage_read_wrong_value_mutations_refuted": read_refuted,
                "private_frame_dead_state_qemu_comparisons": frame_matched,
                "private_frame_escape_mutations_refused": frame_refused,
                "private_frame_forged_access_mutations_refused": frame_forged_refused,
                "frame_promotion_qemu_comparisons": promotion_matched,
                "frame_phi_mutations_refused": promotion_refused,
                "frame_promotion_value_mutations_refuted": promotion_refuted,
                "literal_load_qemu_comparisons": literal_matched,
                "literal_load_missing_contract_mutations_refused": literal_refused,
                "literal_load_wrong_value_mutations_refuted": literal_refuted,
                "literal_load_live_deletion_mutations_refused": live_load_refused,
                "selected_load_qemu_comparisons": selected_matched,
                "selected_load_missing_contract_mutations_refused": selected_refused,
                "selected_load_wrong_value_mutations_refuted": selected_refuted,
                "selected_load_forged_address_mutations_refused": selected_address_refused,
                "selected_load_live_condition_mutations_refused": selected_condition_refused,
                "bounded_table_qemu_comparisons": table_matched,
                "bounded_table_missing_contract_mutations_refused": table_refused,
                "bounded_table_wrong_row_mutations_refuted": table_refuted,
                "bounded_table_changed_source_mutations_refused": table_stale_refused,
                "two_bounded_table_qemu_comparisons": two_matched,
                "two_bounded_table_missing_contract_mutations_refused": two_refused,
                "two_bounded_table_second_row_mutations_refuted": two_second_refuted,
                "two_bounded_table_first_row_mutations_refuted": two_first_refuted,
                "frame_declarations": [
                    "SP-32..SP+0 fresh mapped writable",
                    "no external aliases",
                    "no asynchronous observers",
                ],
                "dce_declarations": ["closed_population", "returns_leave"],
                "scope": "synthetic functions, GPR/NZCV/SP and writable page outside the private slot; no calls",
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
