import dataclasses
import itertools
import pathlib
import random
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tools.oracle.aarch64 import Declined, OracleError
from tools.oracle.memory import DEFAULT_BASE, MemoryOracle, MemoryState, PAGE_SIZE, MAGIC, _parse


class MemoryOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = MemoryOracle()

    def state(self, base=DEFAULT_BASE):
        rng = random.Random(123)
        registers = [rng.getrandbits(64) for _ in range(31)]
        registers[1] = base + 32
        return MemoryState(
            tuple(registers),
            0xA0000000,
            base + PAGE_SIZE,
            bytes(i % 256 for i in range(PAGE_SIZE)),
            base,
        )

    def word(self, instruction):
        with tempfile.TemporaryDirectory(prefix="nyx-test-memory-") as directory:
            path = pathlib.Path(directory)
            (path / "word.S").write_text(".text\n" + instruction + "\n")
            backend = self.oracle.backend
            backend._command(
                [backend.compiler, "-c", "-o", str(path / "word.o"), str(path / "word.S")]
            )
            backend._command(
                [
                    backend.objcopy,
                    "-O",
                    "binary",
                    "-j",
                    ".text",
                    str(path / "word.o"),
                    str(path / "word.bin"),
                ]
            )
            return (path / "word.bin").read_bytes()

    def test_fixed_original_load_preserves_other_state(self):
        state = self.state()
        outcome = self.oracle.run(bytes.fromhex("200040f9"), state)
        self.assertTrue(outcome.completed)
        expected = list(state.registers)
        expected[0] = int.from_bytes(state.memory[32:40], "little")
        self.assertEqual(outcome.state, dataclasses.replace(state, registers=tuple(expected)))
        self.assertEqual(outcome.pc_offset, 4)
        self.assertEqual(outcome.fault_address, 0)

    def test_register_offset_confinement_and_observed_signed_load(self):
        state = self.state()
        registers = list(state.registers)
        registers[2] = 0xABCDEF00FFFFFFFC
        state = dataclasses.replace(state, registers=tuple(registers))
        outcome = self.oracle.run(self.word("ldrsw x0, [x1, w2, sxtw #2]"), state)
        expected = int.from_bytes(state.memory[16:20], "little", signed=True) & ((1 << 64) - 1)
        registers[0] = expected
        self.assertEqual(outcome.state, dataclasses.replace(state, registers=tuple(registers)))
        for instruction in ("ldr x0, [x1, w2, uxtw]", "ldr x0, [x1, x2]"):
            with self.assertRaises(Declined):
                self.oracle.run(self.word(instruction), state)
        for instruction in ("ldr x0, [x1, w2, lsl #3]", "ldrb w0, [x1, x2, lsl #1]"):
            with self.assertRaises(Declined):
                _parse(instruction)
        unchanged = self.oracle.run(self.word("ldr xzr, [x1, xzr]"), state)
        self.assertTrue(unchanged.completed)
        self.assertEqual(unchanged.state, state)

    def test_acquire_release_admission_requires_base_only_and_correct_width(self):
        for mnemonic, width in (
            ("ldar", 4),
            ("stlr", 4),
            ("ldarb", 1),
            ("stlrb", 1),
            ("ldarh", 2),
            ("stlrh", 2),
        ):
            self.assertEqual(_parse(f"{mnemonic} w0, [x1]").size, width)
            for operands in ("w0, [x1, #0]", "w0, [x1], #4", "w0, [x1, #4]!", "w0, w2, [x1]"):
                with (
                    self.subTest(mnemonic=mnemonic, operands=operands),
                    self.assertRaises(Declined),
                ):
                    _parse(f"{mnemonic} {operands}")
        for mnemonic in ("ldarb", "stlrb", "ldarh", "stlrh"):
            with self.assertRaises(Declined):
                _parse(f"{mnemonic} x0, [x1]")

    def test_pair_sp_writeback_and_aliasing(self):
        state = self.state()
        stored = self.oracle.run(self.word("stp x0, x30, [sp, #-16]!"), state)
        self.assertTrue(stored.completed)
        self.assertEqual(stored.state.sp, state.sp - 16)
        self.assertEqual(
            stored.state.memory[-16:], struct.pack("<QQ", state.registers[0], state.registers[30])
        )
        loaded = self.oracle.run(self.word("ldp x2, x3, [sp], #16"), stored.state)
        self.assertTrue(loaded.completed)
        self.assertEqual(loaded.state.registers[2:4], (state.registers[0], state.registers[30]))
        self.assertEqual(loaded.state.sp, state.sp)
        self.assertEqual(loaded.state.nzcv, state.nzcv)
        # A differently sized overlapping read observes the bytes written by STP.
        overlap = self.oracle.run(self.word("ldr w4, [sp, #4]"), stored.state)
        self.assertEqual(overlap.state.registers[4], state.registers[0] >> 32)

    def test_scalar_writeback_and_width_mutation(self):
        state = self.state()
        stored = self.oracle.run(self.word("str w0, [x1], #4"), state)
        self.assertEqual(stored.state.registers[1], state.registers[1] + 4)
        self.assertEqual(
            stored.state.memory[32:36], struct.pack("<I", state.registers[0] & 0xFFFFFFFF)
        )
        wide = self.oracle.run(self.word("str x0, [x1], #4"), state)
        self.assertNotEqual(stored.state.memory, wide.state.memory)
        self.assertEqual(stored.state.nzcv, state.nzcv)
        self.assertEqual(stored.state.sp, state.sp)

    def test_guard_fault_reports_state_and_fault_address(self):
        state = self.state()
        registers = list(state.registers)
        registers[1] = state.base + PAGE_SIZE
        state = dataclasses.replace(state, registers=tuple(registers))
        outcome = self.oracle.run(self.word("ldr x0, [x1], #8"), state)
        self.assertFalse(outcome.completed)
        self.assertEqual(outcome.signal, 11)
        self.assertEqual(outcome.fault_address, state.base + PAGE_SIZE)
        self.assertEqual(outcome.pc_offset, 0)
        self.assertEqual(outcome.state, state)

    def test_pair_crossing_guard_captures_actual_fault(self):
        state = self.state()
        registers = list(state.registers)
        registers[1] = state.base + PAGE_SIZE - 8
        state = dataclasses.replace(state, registers=tuple(registers))
        outcome = self.oracle.run(self.word("stp x0, x2, [x1]"), state)
        self.assertEqual(outcome.signal, 11)
        self.assertEqual(outcome.fault_address, state.base + PAGE_SIZE)
        self.assertEqual(outcome.state.registers, state.registers)
        # The oracle reports backend-observed bytes, without inventing pair atomicity.
        self.assertEqual(outcome.state.memory[:-8], state.memory[:-8])

    def test_guest_sp_may_be_unmapped_and_placements_vary(self):
        for base in [DEFAULT_BASE, DEFAULT_BASE + 0x200000]:
            state = dataclasses.replace(self.state(base), sp=0)
            outcome = self.oracle.run(self.word("ldr x0, [x1]"), state)
            self.assertTrue(outcome.completed)
            self.assertEqual(outcome.state.sp, 0)
            self.assertEqual(
                outcome.state.registers[0], int.from_bytes(state.memory[32:40], "little")
            )

    def test_unsafe_unsupported_and_unpredictable_decline(self):
        for instruction in [
            "ret",
            "svc #0",
            "nop",
            "ldr x0, [x1, x2]",
            "ldp x0, x0, [x1]",
            "ldr x1, [x1], #8",
        ]:
            with self.subTest(instruction=instruction), self.assertRaises(Declined):
                self.oracle.run(self.word(instruction), self.state())
        state = self.state()
        registers = list(state.registers)
        registers[1] = 0x400000
        with self.assertRaisesRegex(Declined, "outside scratch"):
            self.oracle.run(
                bytes.fromhex("200040f9"), dataclasses.replace(state, registers=tuple(registers))
            )

    def test_immediate_arithmetic_sp_zr_width_and_flags(self):
        maximum = (1 << 64) - 1
        cases = [
            ("add sp, sp, #32", maximum - 15, 3, 16, 3, 0xA0000000),
            ("sub wsp, wsp, #1", 0x100000000, 3, 0xFFFFFFFF, 3, 0xA0000000),
            ("add x0, sp, #1", maximum, 3, maximum, 0, 0xA0000000),
            ("add w0, wsp, #1", maximum, 3, maximum, 0, 0xA0000000),
            ("add sp, x0, #16", 0, 3, 19, 3, 0xA0000000),
            ("sub wsp, w0, #1", maximum, 3, 2, 3, 0xA0000000),
            (
                "adds x0, sp, #1",
                0x7FFFFFFFFFFFFFFF,
                3,
                0x7FFFFFFFFFFFFFFF,
                0x8000000000000000,
                0x90000000,
            ),
            ("adds wzr, wsp, #1", maximum, 3, maximum, 3, 0x60000000),
            ("subs xzr, sp, #1", 0, 3, 0, 3, 0x80000000),
            ("cmp wsp, #0", 0x100000000, 3, 0x100000000, 3, 0x60000000),
            ("add sp, sp, #0", 0, 3, 0, 3, 0xA0000000),
            ("add sp, sp, #1, lsl #12", 0, 3, 4096, 3, 0xA0000000),
            ("mov sp, x0", 0, maximum, maximum, maximum, 0xA0000000),
            ("mov w0, wsp", maximum, 3, maximum, 0xFFFFFFFF, 0xA0000000),
        ]
        for instruction, sp, x0, expected_sp, expected_x0, expected_flags in cases:
            with self.subTest(instruction=instruction):
                state = self.state()
                registers = list(state.registers)
                registers[0] = x0
                state = dataclasses.replace(state, registers=tuple(registers), sp=sp)
                outcome = self.oracle.run(self.word(instruction), state)
                self.assertTrue(outcome.completed)
                registers[0] = expected_x0
                self.assertEqual(
                    outcome.state,
                    dataclasses.replace(
                        state, registers=tuple(registers), sp=expected_sp, nzcv=expected_flags
                    ),
                )

    def test_unscaled_narrow_stores_and_loads(self):
        state = self.state()
        for store, load, size in [("sturb", "ldurb", 1), ("sturh", "ldurh", 2)]:
            with self.subTest(store=store):
                stored = self.oracle.run(self.word(f"{store} w0, [x1, #-1]"), state)
                self.assertTrue(stored.completed)
                expected = bytearray(state.memory)
                expected[31 : 31 + size] = (state.registers[0] & ((1 << (size * 8)) - 1)).to_bytes(
                    size, "little"
                )
                self.assertEqual(stored.state.memory, bytes(expected))
                loaded = self.oracle.run(self.word(f"{load} w2, [x1, #-1]"), stored.state)
                self.assertEqual(
                    loaded.state.registers[2], state.registers[0] & ((1 << (size * 8)) - 1)
                )

    def test_signed_load_access_size_and_destination_extension(self):
        for mnemonic, size in [
            ("ldrsb", 1),
            ("ldursb", 1),
            ("ldrsh", 2),
            ("ldursh", 2),
            ("ldrsw", 4),
            ("ldursw", 4),
        ]:
            for register in ["x0"] if size == 4 else ["w0", "x0"]:
                with self.subTest(mnemonic=mnemonic, register=register):
                    state = self.state()
                    registers = list(state.registers)
                    registers[1] = state.base + PAGE_SIZE - size
                    state = dataclasses.replace(state, registers=tuple(registers))
                    access = _parse(f"{mnemonic} {register}, [x1]")
                    self.assertEqual(access.size, size)
                    result = self.oracle.run(self.word(f"{mnemonic} {register}, [x1]"), state)
                    self.assertTrue(result.completed)
                    value = int.from_bytes(state.memory[-size:], "little", signed=True)
                    mask = (1 << (64 if register.startswith("x") else 32)) - 1
                    self.assertEqual(result.state.registers[0], value & mask)
                    self.assertEqual(result.state.memory, state.memory)

    def test_pc_relative_original_words_and_independent_code_address(self):
        for page in (False, True):
            for immediate in (0, 1, -1, (1 << 20) - 1, -(1 << 20)):
                encoded = immediate & ((1 << 21) - 1)
                word = (
                    (0x90000000 if page else 0x10000000)
                    | ((encoded & 3) << 29)
                    | ((encoded >> 2) << 5)
                )
                with self.subTest(page=page, immediate=immediate):
                    state = self.state()
                    result = self.oracle.run(word.to_bytes(4, "little"), state)
                    self.assertTrue(result.completed)
                    expected = (result.code_address + (immediate << (12 if page else 0))) & (
                        (1 << 64) - 1
                    )
                    self.assertEqual(result.state.registers[0], expected)
                    self.assertEqual(result.state.registers[1:], state.registers[1:])
                    self.assertEqual(result.state.sp, state.sp)
                    self.assertEqual(result.state.memory, state.memory)
        # Discarding the computed address still returns runner-owned mapping metadata.
        state = self.state()
        result = self.oracle.run((0x1000001F).to_bytes(4, "little"), state)
        self.assertEqual(result.state, state)
        self.assertGreater(result.code_address, 0)
        self.assertEqual(result.code_address % PAGE_SIZE, 0)

    def test_signed_word_pair_width_and_second_access_fault(self):
        state = self.state()
        memory = bytearray(state.memory)
        memory[32:40] = struct.pack("<II", 0x80000000, 0x7FFFFFFF)
        state = dataclasses.replace(state, memory=bytes(memory))
        outcome = self.oracle.run(self.word("ldpsw x0, x30, [x1], #8"), state)
        self.assertTrue(outcome.completed)
        self.assertEqual(outcome.state.registers[0], 0xFFFFFFFF80000000)
        self.assertEqual(outcome.state.registers[30], 0x7FFFFFFF)
        self.assertEqual(outcome.state.registers[1], state.registers[1] + 8)
        registers = list(state.registers)
        registers[1] = state.base + PAGE_SIZE - 4
        fault_state = dataclasses.replace(state, registers=tuple(registers))
        fault = self.oracle.run(self.word("ldpsw x0, x30, [x1], #8"), fault_state)
        self.assertFalse(fault.completed)
        self.assertEqual(fault.fault_address, state.base + PAGE_SIZE)
        self.assertEqual(fault.state, fault_state)
        for instruction in ("ldpsw x0, x0, [x1]", "ldpsw x1, x2, [x1], #8"):
            with self.subTest(instruction=instruction), self.assertRaises(Declined):
                self.oracle.run(self.word(instruction), state)

    def test_code_address_protocol_rejects_missing_or_guest_scratch_metadata(self):
        state = self.state()
        values = [*state.registers, state.nzcv, state.sp, 4, 0, 0, 0x80000000]
        valid = MAGIC + struct.pack("<37Q", *values) + state.memory
        self.assertEqual(self.oracle._response(valid, state.base).code_address, 0x80000000)
        for address in (0, state.base, 0x80000001):
            values[-1] = address
            with self.assertRaisesRegex(OracleError, "code mapping"):
                self.oracle._response(
                    MAGIC + struct.pack("<37Q", *values) + state.memory, state.base
                )
        with self.assertRaises(OracleError):
            self.oracle._response(b"NYXMEM01" + valid[8:], state.base)

    def test_resource_and_response_boundaries(self):
        with self.assertRaisesRegex(ValueError, "immutable"):
            MemoryState([0] * 31)
        oracle = MemoryOracle(max_cases=1)
        code = bytes.fromhex("200040f9")
        with self.assertRaisesRegex(Declined, "budget"):
            oracle.run_many(code, itertools.repeat(self.state()))
        with self.assertRaisesRegex(Declined, "no evidence"):
            oracle.run_many(code, [])
        with self.assertRaises(OracleError):
            oracle._response(b"bad", DEFAULT_BASE)


if __name__ == "__main__":
    unittest.main()
