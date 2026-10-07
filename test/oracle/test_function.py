import pathlib
import struct
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tools.oracle.aarch64 import Declined, OracleError
from tools.oracle.function import (
    BRK,
    ENTERED_STUB_BODY,
    EVENT_OVERFLOW,
    EXECUTE,
    READ,
    STUB,
    WRITE,
    FunctionOracle,
    FunctionState,
    Region,
    _check_events,
    _event,
)


CODE = 0x200000000
STACK = 0x300000000
LANDING = CODE + 0x100
RET = 0xD65F03C0


def page(words):
    data = bytearray(struct.pack("<I", BRK) * 1024)
    for offset, word in words.items():
        struct.pack_into("<I", data, offset, word)
    return bytes(data)


def state(code):
    registers = [index + 3 for index in range(31)]
    registers[30] = LANDING
    return FunctionState(
        tuple(registers),
        0xA0000000,
        STACK + 2048,
        (Region(CODE, code, READ | EXECUTE), Region(STACK, bytes(4096), READ | WRITE)),
    )


class FunctionOracleTest(unittest.TestCase):
    def test_admitted_function_reaches_landing_and_preserves_memory(self):
        oracle = FunctionOracle()
        code = page({0: 0xD28000E0, 4: RET})  # mov x0, #7; ret
        original = state(code)
        with self.assertRaises(Declined):
            oracle.run(original, CODE, (LANDING,), trusted_fixture=True)
        oracle.admit(CODE, code[:8])
        with self.assertRaises(Declined):
            oracle.run(original, LANDING, (LANDING,), trusted_fixture=True)
        outcome = oracle.run(original, CODE, (LANDING,), trusted_fixture=True)
        self.assertTrue(outcome.completed)
        self.assertEqual(
            (outcome.pc, outcome.sp, outcome.nzcv), (LANDING, original.sp, original.nzcv)
        )
        self.assertEqual(outcome.registers, (7,) + original.registers[1:])
        self.assertEqual(outcome.memory, {STACK: bytes(4096)})

        changed = state(page({0: 0xD2800100, 4: RET}))  # mov x0, #8
        with self.assertRaises(Declined):
            oracle.run(changed, CODE, (LANDING,), trusted_fixture=True)
        replaced_ret = state(page({0: 0xD28000E0}))
        with self.assertRaises(Declined):
            oracle.run(replaced_ret, CODE, (CODE + 4,), trusted_fixture=True)

    def test_stub_bytes_are_explicit_and_checked(self):
        oracle = FunctionOracle()
        stub = struct.pack("<II", 0xD2800120, RET)  # mov x0, #9; ret
        code = page({0: 0x14000020, 0x80: 0xD2800120, 0x84: RET})
        oracle.admit(CODE, code[:4])
        original = state(code)
        with self.assertRaises(Declined):
            oracle.run(original, CODE, (LANDING,), trusted_fixture=True)
        with self.assertRaises(Declined):
            oracle.run(
                original,
                CODE + 0x80,
                (LANDING,),
                stubs=((CODE + 0x80, stub),),
                trusted_fixture=True,
            )
        outcome = oracle.run(
            original, CODE, (LANDING,), stubs=((CODE + 0x80, stub),), trusted_fixture=True
        )
        self.assertTrue(outcome.completed)
        self.assertEqual(outcome.registers[0], 9)
        with self.assertRaises(Declined):
            oracle.run(
                original,
                CODE,
                (LANDING,),
                stubs=((CODE + 0x80, stub[:-4] + b"\0" * 4),),
                trusted_fixture=True,
            )
        self.assertIsNone(outcome.events)

    def test_stub_pages_record_ordered_calls_and_returns(self):
        oracle = FunctionOracle()
        # mov x19, x30; bl A; bl B; mov x30, x19; ret. A sets x0 to 9, B adds 1.
        code = page({0: 0xAA1E03F3, 4: 0x940003FF, 8: 0x94000402, 12: 0xAA1303FE, 16: RET})
        first, second = CODE + 0x1000, CODE + 0x1010
        stubs = (
            (first, struct.pack("<II", 0xD2800120, RET)),
            (second, struct.pack("<II", 0x91000400, RET)),
        )
        stub_page = page({0: 0xD2800120, 4: RET, 0x10: 0x91000400, 0x14: RET})
        oracle.admit(CODE, code[:20])
        original = state(code)
        traced = FunctionState(
            original.registers,
            original.nzcv,
            original.sp,
            (
                original.regions[0],
                Region(first, stub_page, READ | EXECUTE | STUB),
                original.regions[1],
            ),
        )
        outcome = oracle.run(traced, CODE, (LANDING,), stubs=stubs, trusted_fixture=True)
        self.assertTrue(outcome.completed)
        self.assertEqual(outcome.registers[0], 10)
        arguments = list(original.registers[:8])
        self.assertEqual(
            outcome.events,
            (
                {
                    "kind": "call",
                    "target": first,
                    "return_address": CODE + 8,
                    "sp": original.sp,
                    "arguments": arguments,
                },
                {"kind": "return", "pc": CODE + 8, "sp": original.sp, "results": [9, arguments[1]]},
                {
                    "kind": "call",
                    "target": second,
                    "return_address": CODE + 12,
                    "sp": original.sp,
                    "arguments": [9] + arguments[1:],
                },
                {
                    "kind": "return",
                    "pc": CODE + 12,
                    "sp": original.sp,
                    "results": [10, arguments[1]],
                },
            ),
        )

        # Entering a stub past its declared entry leaves the traced envelope.
        middle = page({0: 0xAA1E03F3, 4: 0x94000400, 8: 0x94000402, 12: 0xAA1303FE, 16: RET})
        oracle = FunctionOracle()
        oracle.admit(CODE, middle[:20])
        entered = FunctionState(
            original.registers,
            original.nzcv,
            original.sp,
            (Region(CODE, middle, READ | EXECUTE),) + traced.regions[1:],
        )
        outcome = oracle.run(entered, CODE, (LANDING,), stubs=stubs, trusted_fixture=True)
        self.assertFalse(outcome.completed)
        self.assertTrue(outcome.left_trace)
        self.assertEqual(outcome.signal, ENTERED_STUB_BODY)
        self.assertEqual(outcome.events[-1], {"kind": "enter_other", "pc": first + 4})

        # A traced stub must not share a page with original code or a landing.
        with self.assertRaises(Declined):
            oracle.run(
                entered,
                CODE,
                (LANDING,),
                stubs=stubs + ((CODE + 0x80, stubs[0][1]),),
                trusted_fixture=True,
            )
        with self.assertRaises(Declined):
            oracle.run(entered, CODE, (first + 0x20,), stubs=stubs, trusted_fixture=True)
        with self.assertRaises(ValueError):
            Region(first, stub_page, READ | STUB)

    def test_runner_events_must_alternate_and_end_the_run_consistently(self):
        call = _event((1, CODE + 0x1000, CODE + 8, STACK, *range(8)))
        back = _event((2, CODE + 8, 0, STACK, 5, 6, *(0,) * 6))
        other = _event((3, CODE + 0x1004, CODE + 8, STACK, *range(8)))
        _check_events((call, back, call), 11)
        _check_events((call, back, other), ENTERED_STUB_BODY)
        _check_events((call, back) * 32, EVENT_OVERFLOW)
        _check_events((call, back) * 32, 0)
        for events, signal in (
            ((back,), 0),
            ((call, call), 0),
            ((other, call), ENTERED_STUB_BODY),
            ((call, back), ENTERED_STUB_BODY),
            ((call, back), EVENT_OVERFLOW),
        ):
            with self.assertRaises(OracleError):
                _check_events(events, signal)
        for words in (
            (2, CODE, 1, STACK, *(0,) * 8),
            (2, CODE, 0, STACK, 0, 0, 1, *(0,) * 5),
            (4, *(0,) * 11),
        ):
            with self.assertRaises(OracleError):
                _event(words)

    def test_literal_load_reads_separate_read_only_page(self):
        oracle = FunctionOracle()
        code = page({0: 0x58008000, 4: RET})
        data = struct.pack("<Q", 0x12345678) + bytes(4088)
        original = state(code)
        mapped = FunctionState(
            original.registers,
            original.nzcv,
            original.sp,
            (original.regions[0], Region(CODE + 4096, data, READ), original.regions[1]),
        )
        oracle.admit(CODE, code[:8])
        outcome = oracle.run(mapped, CODE, (LANDING,), trusted_fixture=True)
        self.assertTrue(outcome.completed)
        self.assertEqual(outcome.registers[0], 0x12345678)
        self.assertEqual(outcome.memory, {STACK: bytes(4096)})


if __name__ == "__main__":
    unittest.main()
