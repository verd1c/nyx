import dataclasses
import itertools
import pathlib
import random
import struct
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tools.oracle.aarch64 import Declined, OracleError
from tools.oracle.control import ControlOracle, ControlState, DEFAULT_PC, MAGIC


class ControlOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = ControlOracle()

    def state(self, flags=0, pc=DEFAULT_PC):
        rng = random.Random(0xC017)
        return ControlState(tuple(rng.getrandbits(64) for _ in range(31)), flags << 28, 0, pc)

    def run_word(self, word, state):
        return self.oracle.run(word.to_bytes(4, "little"), state)

    def test_direct_forward_backward_and_placements(self):
        for pc in (DEFAULT_PC, DEFAULT_PC + 0x200000):
            state = self.state(15, pc)
            for word, delta in ((0x14000008, 32), (0x17FFFFF8, -32)):
                result = self.run_word(word, state)
                self.assertEqual(result.landing_pc, pc + delta)
                self.assertEqual(result.state, state)

    def test_all_conditions_and_flag_states(self):
        states = [self.state(flags) for flags in range(16)]
        for condition in range(16):
            word = 0x54000100 | condition  # B.cond +32
            outcomes = self.oracle.run_many(word.to_bytes(4, "little"), states)
            for flags, state, outcome in zip(range(16), states, outcomes, strict=True):
                n, z, c, v = (bool(flags & bit) for bit in (8, 4, 2, 1))
                taken = [
                    z,
                    not z,
                    c,
                    not c,
                    n,
                    not n,
                    v,
                    not v,
                    c and not z,
                    not c or z,
                    n == v,
                    n != v,
                    not z and n == v,
                    z or n != v,
                    True,
                    True,
                ][condition]
                with self.subTest(condition=condition, flags=flags):
                    self.assertEqual(outcome.landing_pc, state.pc + (32 if taken else 4))
                    self.assertEqual(outcome.state, state)

    def test_swapped_condition_is_refuted_even_with_same_target_set(self):
        states = [self.state(0), self.state(4)]
        original = self.oracle.run_many((0x54000100).to_bytes(4, "little"), states)
        swapped = self.oracle.run_many((0x54000101).to_bytes(4, "little"), states)
        self.assertEqual({x.landing_pc for x in original}, {x.landing_pc for x in swapped})
        self.assertNotEqual([x.landing_pc for x in original], [x.landing_pc for x in swapped])

    def test_backward_conditional_offsets(self):
        for base, bits, value in (
            (0x54000000, 19, 0),
            (0xB4000000, 19, 0),
            (0xB5000000, 19, 1),
            (0x36000000, 14, 0),
            (0x37000000, 14, 1),
        ):
            state = self.state(4)
            registers = list(state.registers)
            registers[0] = value
            state = dataclasses.replace(state, registers=tuple(registers))
            word = base | (((-8) & ((1 << bits) - 1)) << 5)
            outcome = self.run_word(word, state)
            self.assertEqual(outcome.landing_pc, state.pc - 32)
            self.assertEqual(outcome.state, state)

    def test_compare_zero_width_and_test_bit_boundaries(self):
        state = self.state(10)
        registers = list(state.registers)
        for value in (0, 1, 1 << 31, 1 << 32, 1 << 63):
            registers[0] = value
            current = dataclasses.replace(state, registers=tuple(registers))
            for word, zero in (
                (0xB4000100, value == 0),
                (0x34000100, value & 0xFFFFFFFF == 0),
                (0xB5000100, value != 0),
                (0x35000100, value & 0xFFFFFFFF != 0),
            ):
                outcome = self.run_word(word, current)
                self.assertEqual(outcome.landing_pc, current.pc + (32 if zero else 4))
                self.assertEqual(outcome.state, current)
            for bit in (0, 31, 32, 63):
                word = 0x36000100 | ((bit & 31) << 19) | ((bit >> 5) << 31)
                for nonzero in (False, True):
                    outcome = self.run_word(word | (int(nonzero) << 24), current)
                    taken = bool(value & (1 << bit)) == nonzero
                    self.assertEqual(outcome.landing_pc, current.pc + (32 if taken else 4))
                    self.assertEqual(outcome.state, current)

    def test_indirect_jump_return_and_call_landings(self):
        state = self.state(5)
        registers = list(state.registers)
        registers[3] = state.pc + 96
        registers[30] = state.pc - 64
        state = dataclasses.replace(state, registers=tuple(registers), sp=0xFFFFFFFFFFFFFFF0)
        for word, destination, link in (
            (0xD61F0060, state.pc + 96, False),  # BR x3
            (0xD65F03C0, state.pc - 64, False),  # RET x30
            (0xD65F0060, state.pc + 96, False),  # RET x3
            (0x94000008, state.pc + 32, True),  # BL +32
            (0xD63F0060, state.pc + 96, True),  # BLR x3
            (0xD63F03C0, state.pc - 64, True),
        ):  # BLR x30
            result = self.run_word(word, state)
            expected = list(state.registers)
            if link:
                expected[30] = state.pc + 4
            self.assertEqual(result.landing_pc, destination)
            self.assertEqual(result.state, dataclasses.replace(state, registers=tuple(expected)))
        # This observes a call's entry/LR only; no callee or continuation ran.
        branch = self.run_word(0x14000008, state)
        call = self.run_word(0x94000008, state)
        self.assertEqual(branch.landing_pc, call.landing_pc)
        self.assertNotEqual(branch.state, call.state)

    def test_unsafe_targets_and_unsupported_instructions_decline(self):
        state = self.state()
        for word in (
            0xD4000001,
            0xD4200000,
            0xF9400020,
            0x14000000,
            0x14000800,
            0xD61F03E0,
            0xD61F0060,
        ):
            with self.subTest(word=hex(word)), self.assertRaises(Declined):
                self.run_word(word, state)
        # Even an untaken conditional target must stay in the controlled arena.
        with self.assertRaises(Declined):
            self.run_word(0x54010000, self.state(0))

    def test_protocol_and_resource_refusals(self):
        state = self.state()
        with self.assertRaisesRegex(Declined, "no evidence"):
            self.oracle.run_many((0x14000008).to_bytes(4, "little"), [])
        with self.assertRaisesRegex(Declined, "budget"):
            ControlOracle(max_cases=1).run_many(
                (0x14000008).to_bytes(4, "little"), itertools.repeat(state)
            )
        for landing in (state.pc, state.pc + 2 * 4096, state.pc + 5):
            response = MAGIC + struct.pack(
                "<35Q", *state.registers, state.nzcv, state.sp, state.pc, landing
            )
            with self.assertRaises(OracleError):
                self.oracle._response(response, state)
        with self.assertRaises(OracleError):
            self.oracle._response(b"bad", state)


if __name__ == "__main__":
    unittest.main()
