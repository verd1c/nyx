import itertools
import pathlib
import random
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tools.oracle.aarch64 import Declined, MASK64, Oracle, OracleError, State, validate


class OracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = Oracle()

    def state(self, seed=42, nzcv=0):
        rng = random.Random(seed)
        return State(tuple(rng.getrandbits(64) for _ in range(31)), nzcv)

    def test_roundtrip_all_registers_and_flags(self):
        states = [self.state(i, i << 28) for i in range(16)]
        self.assertEqual(self.oracle.run_many(["nop"], states), states)

    def test_flag_dependency_and_conditional_mapping(self):
        registers = list(self.state().registers)
        registers[1] = MASK64
        registers[2] = 1
        state = State(tuple(registers))
        block = ["adds x0, x1, x2", "adc x3, xzr, xzr", "csel x4, x1, x2, eq"]
        result = self.oracle.run(block, state)
        self.assertEqual(result.registers[0], 0)
        self.assertEqual(result.registers[3], 1)
        self.assertEqual(result.registers[4], MASK64)
        self.assertEqual(result.nzcv, 0x60000000)
        swapped = self.oracle.run(block[:2] + ["csel x4, x2, x1, eq"], state)
        self.assertNotEqual(result, swapped)
        dropped_flags = self.oracle.run(["add x0, x1, x2"] + block[1:], state)
        self.assertNotEqual(result, dropped_flags)

    def test_width_shift_and_x30(self):
        registers = list(self.state().registers)
        registers[1] = 0xFFFFFFFF80000001
        registers[2] = 33
        state = State(tuple(registers), 0xA0000000)
        result = self.oracle.run(["lsl w0, w1, w2", "asr w3, w1, #31", "mov x30, #9"], state)
        self.assertEqual(result.registers[0], 2)
        self.assertEqual(result.registers[3], 0xFFFFFFFF)
        self.assertEqual(result.registers[30], 9)
        self.assertEqual(result.nzcv, state.nzcv)
        widened = self.oracle.run(["lsl x0, x1, x2"], state)
        self.assertNotEqual(result.registers[0], widened.registers[0])

    def test_signed_overflow_and_borrow(self):
        registers = list(self.state().registers)
        registers[1] = 0x7FFFFFFFFFFFFFFF
        registers[2] = 1
        state = State(tuple(registers))
        result = self.oracle.run(["adds x0, x1, x2"], state)
        self.assertEqual(result.registers[0], 0x8000000000000000)
        self.assertEqual(result.nzcv, 0x90000000)
        result = self.oracle.run(["subs x0, xzr, x2"], state)
        self.assertEqual(result.registers[0], MASK64)
        self.assertEqual(result.nzcv, 0x80000000)

    def test_declines_unsafe_and_out_of_scope(self):
        for line in [
            "svc #0",
            "br x0",
            "ret",
            "ldr x0, [x1]",
            "mov sp, x0",
            "mov x0, sp",
            "add x0, x0, #1; svc #0",
            "nop\nsvc #0",
            ".inst 0xd4000001",
            "b label",
            "msr daif, x0",
            "mov v0.16b, v1.16b",
            "mrs x0, tpidr_el0",
            "msr tpidr_el0, x0",
            "mrs w0, nzcv",
            "msr nzcv, sp",
            "msr nzcv, #15",
            "mrs x0, S3_3_C4_C2_0",
        ]:
            with self.subTest(line=line), self.assertRaises(Declined):
                validate([line])

    def test_nzcv_transfers_keep_other_registers_and_ignore_reserved_bits(self):
        for flags in range(16):
            initial = self.state(flags, flags << 28)
            result = self.oracle.run_bytes(bytes.fromhex("01423bd5"), initial)
            expected = list(initial.registers)
            expected[1] = initial.nzcv
            self.assertEqual(result, State(tuple(expected), initial.nzcv))
            regs = list(initial.registers)
            regs[30] = (MASK64 ^ 0xF0000000) | (flags << 28)
            result = self.oracle.run(["msr nzcv, x30", "mrs xzr, nzcv"], State(tuple(regs)))
            self.assertEqual(result, State(tuple(regs), flags << 28))
            self.assertEqual(self.oracle.run(["msr nzcv, xzr"], result), State(tuple(regs), 0))

    def test_invalid_assembly_is_not_execution(self):
        with self.assertRaises(OracleError):
            self.oracle.run(["add x0"], self.state())

    def test_encoding_and_process_deadline(self):
        self.assertEqual(
            self.oracle.assemble(["nop", "add x0, x1, x2"]), bytes.fromhex("1f2003d52000028b")
        )
        oracle = Oracle(timeout=0.01)
        with self.assertRaisesRegex(OracleError, "deadline"):
            oracle._command([sys.executable, "-c", "import time; time.sleep(10)"])

    def test_case_budget_and_zero_evidence(self):
        oracle = Oracle(max_cases=2)
        with mock.patch.object(oracle, "_command") as command:
            with self.assertRaisesRegex(Declined, "budget"):
                oracle.run_many(["nop"], itertools.repeat(self.state()))
            with self.assertRaisesRegex(Declined, "no evidence"):
                oracle.run_many(["nop"], [])
            with self.assertRaisesRegex(Declined, "no evidence"):
                oracle.run_bytes_many(bytes.fromhex("1f2003d5"), [])
            command.assert_not_called()
        states = [self.state(0), self.state(1)]
        self.assertEqual(oracle.run_many(["nop"], states), states)

    def test_original_bytes_and_mutation(self):
        state = self.state()
        # Fixed encodings keep this check independent of assemble().
        original = bytes.fromhex("2000028b")  # add x0, x1, x2
        changed = bytes.fromhex("200002cb")  # sub x0, x1, x2
        result = self.oracle.run_bytes(original, state)
        self.assertEqual(result.registers[0], (state.registers[1] + state.registers[2]) & MASK64)
        self.assertNotEqual(result, self.oracle.run_bytes(changed, state))
        self.assertEqual(self.oracle.run_bytes(bytes.fromhex("1f2003d5"), state), state)

    def test_original_bytes_decline_unknown_unsafe_and_changed_reassembly(self):
        for code in [
            b"",
            b"x",
            bytes.fromhex("00000000"),
            bytes.fromhex("c0035fd6"),  # ret
            bytes.fromhex("00000014"),  # b .
            bytes.fromhex("e0030091"),  # mov x0, sp
            bytes.fromhex("200040f9"),
        ]:  # ldr x0, [x1]
            with self.subTest(code=code.hex()), self.assertRaises(Declined):
                self.oracle.run_bytes(code, self.state())
        with mock.patch.object(self.oracle, "assemble", return_value=b"different"):
            with self.assertRaisesRegex(Declined, "reassembly differs"):
                self.oracle.run_bytes(bytes.fromhex("1f2003d5"), self.state())

    def test_invalid_backend_response_and_exit(self):
        response = bytearray(self.state().encode())
        response[256] = 1  # An invalid NZCV bit, with a valid response envelope.
        with self.assertRaisesRegex(OracleError, "response state"):
            State.decode(response)
        with self.assertRaisesRegex(OracleError, r"failed \(7\)"):
            self.oracle._command([sys.executable, "-c", "raise SystemExit(7)"])

    def test_protocol_rejects_invalid_state(self):
        with self.assertRaises(ValueError):
            State((0,) * 31, 1)
        with self.assertRaises(ValueError):
            State((0,) * 30)
        with self.assertRaises(OracleError):
            State.decode(b"bad")


if __name__ == "__main__":
    unittest.main()
