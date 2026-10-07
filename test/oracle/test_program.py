import dataclasses
import hashlib
import itertools
import pathlib
import random
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tools.oracle.aarch64 import Declined, OracleError
from tools.oracle.program import (
    DEFAULT_CODE,
    DEFAULT_SCRATCH,
    GlobalPage,
    MAGIC,
    ProgramOracle,
    ProgramState,
)


class ProgramOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = ProgramOracle()

    def state(self, scratch=DEFAULT_SCRATCH, code=DEFAULT_CODE):
        rng = random.Random(0xCA11)
        regs = [rng.getrandbits(64) for _ in range(31)]
        regs[0] = 5
        regs[28] = scratch
        return ProgramState(
            tuple(regs),
            0xA0000000,
            scratch + 2048,
            bytes(rng.getrandbits(8) for _ in range(4096)),
            scratch,
            code,
        )

    def assemble(self, source):
        with tempfile.TemporaryDirectory(prefix="nyx-program-fixture-") as directory:
            path = pathlib.Path(directory)
            (path / "fixture.S").write_text(".text\n.global _start\n_start:\n" + source + "\n")
            backend = self.oracle.backend
            backend._command(
                [
                    backend.compiler,
                    "-nostdlib",
                    "-no-pie",
                    "-Wl,-Ttext=0x400000,-e,_start,--build-id=none",
                    "-o",
                    str(path / "fixture.elf"),
                    str(path / "fixture.S"),
                ]
            )
            backend._command(
                [
                    backend.objcopy,
                    "-O",
                    "binary",
                    "-j",
                    ".text",
                    str(path / "fixture.elf"),
                    str(path / "fixture.bin"),
                ]
            )
            return (path / "fixture.bin").read_bytes()

    def execute(self, code, state):
        return self.oracle.run(code, state, trusted_fixture=True)

    def test_direct_real_callee_returns_and_preserves_full_state(self):
        code = self.assemble("""
            bl .Lcallee
            add x0, x0, #1
            b .Lexit
        .Lcallee:
            add x0, x0, #7
            ret
        .Lexit:
        """)
        for scratch, address in (
            (DEFAULT_SCRATCH, DEFAULT_CODE),
            (DEFAULT_SCRATCH + 0x200000, DEFAULT_CODE + 0x300000),
        ):
            state = self.state(scratch, address)
            outcome = self.execute(code, state)
            expected = list(state.registers)
            expected[0] = 13
            expected[30] = address + 4
            self.assertTrue(outcome.completed)
            self.assertEqual(outcome.exit_pc, address + len(code))
            self.assertEqual(outcome.state, dataclasses.replace(state, registers=tuple(expected)))

    def test_blr_x30_uses_old_target_then_returns_to_new_lr(self):
        code = self.assemble("""
            adr x30, .Lcallee
            blr x30
            add x0, x0, #3
            b .Lexit
        .Lcallee:
            mov x0, #11
            ret
        .Lexit:
        """)
        state = self.state()
        outcome = self.execute(code, state)
        expected = list(state.registers)
        expected[0] = 14
        expected[30] = state.code_base + 8
        self.assertTrue(outcome.completed)
        self.assertEqual(outcome.state, dataclasses.replace(state, registers=tuple(expected)))

    def test_callee_memory_stack_and_caller_aliases_are_observed(self):
        code = self.assemble("""
            str x2, [x28, #2032]
            bl .Lcallee
            ldr x4, [x28, #2032]
            b .Lexit
        .Lcallee:
            sub sp, sp, #16
            ldr x0, [sp]
            add x0, x0, #5
            str x0, [sp]
            str x3, [x28, #32]
            add sp, sp, #16
            ret
        .Lexit:
        """)
        state = self.state()
        outcome = self.execute(code, state)
        expected = list(state.registers)
        expected[0] = (expected[2] + 5) & ((1 << 64) - 1)
        expected[4] = expected[0]
        expected[30] = state.code_base + 8
        memory = bytearray(state.memory)
        memory[2032:2040] = struct.pack("<Q", expected[0])
        memory[32:40] = struct.pack("<Q", expected[3])
        self.assertTrue(outcome.completed)
        self.assertEqual(
            outcome.state,
            dataclasses.replace(state, registers=tuple(expected), memory=bytes(memory)),
        )

    def test_callee_flags_and_conditional_return_paths_reach_caller(self):
        code = self.assemble("""
            bl .Lcallee
            csel x2, x3, x4, eq
            b .Lexit
        .Lcallee:
            cmp x0, #0
            b.eq .Lzero
            mov x1, #9
            ret
        .Lzero:
            mov x1, #4
            ret
        .Lexit:
        """)
        for value in (0, 1):
            state = self.state()
            registers = list(state.registers)
            registers[0] = value
            state = dataclasses.replace(state, registers=tuple(registers))
            outcome = self.execute(code, state)
            expected = list(state.registers)
            expected[1] = 4 if value == 0 else 9
            expected[2] = expected[3 if value == 0 else 4]
            expected[30] = state.code_base + 4
            self.assertTrue(outcome.completed)
            self.assertEqual(
                outcome.state,
                dataclasses.replace(
                    state, registers=tuple(expected), nzcv=0x60000000 if value == 0 else 0x20000000
                ),
            )

    def test_fault_in_real_callee_does_not_invent_return(self):
        code = self.assemble("""
            bl .Lcallee
            mov x0, #55
            b .Lexit
        .Lcallee:
            str x1, [x28, #4096]
            ret
        .Lexit:
        """)
        state = self.state()
        outcome = self.execute(code, state)
        self.assertFalse(outcome.completed)
        self.assertEqual(outcome.signal, 11)
        self.assertEqual(outcome.exit_pc, state.code_base + 12)
        self.assertEqual(outcome.fault_address, state.scratch_base + 4096)
        expected = list(state.registers)
        expected[30] = state.code_base + 4
        self.assertEqual(outcome.state, dataclasses.replace(state, registers=tuple(expected)))

    def test_untrusted_unsafe_unknown_and_nested_lr_restore_decline(self):
        state = self.state()
        nop = bytes.fromhex("1f2003d5")
        with self.assertRaisesRegex(Declined, "trusted_fixture"):
            self.oracle.run(nop, state)
        for source in (
            "svc #0",
            "brk #0",
            "ldr x0, [x1]",
            "ldr x30, [sp]",
            "mov x28, x0",
            "ret",
            "br x0",
            "str x0, [x28], #8",
        ):
            with self.subTest(source=source), self.assertRaises(Declined):
                self.execute(self.assemble(source), state)
        with self.assertRaisesRegex(Declined, "outside fixture"):
            self.execute((0x14000008).to_bytes(4, "little"), state)

    def test_nontermination_is_deadline_not_success(self):
        oracle = ProgramOracle(timeout=1)
        commands = []
        execute = oracle.backend._command

        def record(command, **kwargs):
            commands.append(command)
            return execute(command, **kwargs)

        oracle.backend._command = record
        with self.assertRaisesRegex(OracleError, "deadline"):
            oracle.run((0x14000000).to_bytes(4, "little"), self.state(), trusted_fixture=True)
        self.assertEqual(commands[-1][0], oracle.backend.qemu)

    def test_resource_and_response_boundaries(self):
        state = self.state()
        nop = bytes.fromhex("1f2003d5")
        with self.assertRaisesRegex(Declined, "no evidence"):
            self.oracle.run_many(nop, [], trusted_fixture=True)
        with self.assertRaisesRegex(Declined, "budget"):
            ProgramOracle(max_cases=1).run_many(nop, itertools.repeat(state), trusted_fixture=True)
        with self.assertRaisesRegex(Declined, "1024"):
            self.execute(nop * 1025, state)
        response = (
            MAGIC
            + struct.pack(
                "<39Q",
                *state.registers,
                state.nzcv,
                state.sp,
                state.scratch_base,
                state.code_base,
                state.code_base,
                0,
                0,
                0,
            )
            + state.memory
        )
        with self.assertRaisesRegex(OracleError, "designated exit"):
            self.oracle._response(response, state, 4)
        with self.assertRaises(OracleError):
            self.oracle._response(b"bad", state, 4)

    def test_sp_derived_pointer_outside_admitted_arena_declines(self):
        code = self.assemble("add x9, sp, #0x2, lsl #12\nstr x1, [x9]")
        with self.assertRaisesRegex(Declined, "outside declared input arenas"):
            self.execute(code, self.state())

    def test_original_development_global_slice_keeps_g_unknown(self):
        # Original 68 bytes at source/file offset 0xc687b0.
        code = bytes.fromhex(
            "8982009029610d91290140f9e90309cb6ba48ed2ebd7a8f2cbaedef28bd4f1f2"
            "2b010baa6bf97fd36ca48ed2ecd7a8f2ccaedef28cd4f1f229010cca690109cb0b0180d2"
        )
        self.assertEqual(
            hashlib.sha256(code).hexdigest(),
            "fddeb461b9f105230c22f63acd1ce817099d63039bc9705846821d3619c63dd1",
        )
        constant = 0x8EA4F57646BF7523
        values = [0, 1, constant, constant - 1, (1 << 64) - 1, 1 << 63, (1 << 63) - 1]
        rng = random.Random(0xF100)
        values += [rng.getrandbits(64) for _ in range(9)]
        cases = []
        for bias in (0x200000000, 0x600000000):
            for value in values:
                page = bytearray(rng.getrandbits(8) for _ in range(4096))
                struct.pack_into("<Q", page, 0x358, value)
                state = self.state(code=bias + 0xC687B0)
                cases.append(
                    dataclasses.replace(
                        state, global_pages=(GlobalPage(bias + 0x1CB8000, bytes(page)),)
                    )
                )
        outcomes = self.oracle.run_many(code, cases, trusted_fixture=True)
        for state, outcome in zip(cases, outcomes, strict=True):
            value = struct.unpack_from("<Q", state.global_pages[0].data, 0x358)[0]
            expected = list(state.registers)
            expected[9] = (constant - value) & ((1 << 64) - 1)
            expected[11] = 8
            expected[12] = constant
            self.assertTrue(outcome.completed)
            self.assertEqual(outcome.exit_pc, state.code_base + len(code))
            self.assertEqual(outcome.state, dataclasses.replace(state, registers=tuple(expected)))
        self.assertGreater(len({outcome.state.registers[9] for outcome in outcomes}), 1)

    def test_real_adjacent_stack_overwrite_has_same_final_state_without_first_store(self):
        # Original bytes at 0xb55528.
        original = bytes.fromhex("ea0300f9e10300f9")  # str x10, [sp]; str x1, [sp]
        without_first = bytes.fromhex("1f2003d5e10300f9")
        without_second = bytes.fromhex("ea0300f91f2003d5")
        rng = random.Random(0xB55528)
        for code_base in (DEFAULT_CODE, DEFAULT_CODE + 0x300000):
            for _ in range(8):
                state = self.state(code=code_base)
                registers = list(state.registers)
                registers[1] = rng.getrandbits(64)
                registers[10] = registers[1] ^ (1 << 63)
                state = dataclasses.replace(state, registers=tuple(registers))
                expected = self.execute(original, state)
                actual = self.execute(without_first, state)
                mutant = self.execute(without_second, state)
                self.assertTrue(expected.completed)
                self.assertEqual(expected.state, actual.state)
                self.assertEqual(expected.exit_pc, actual.exit_pc)
                self.assertNotEqual(expected.state.memory, mutant.state.memory)

    def test_original_adjusted_stack_overwrite_matches_without_first_store(self):
        # Original sample bytes at 0x152869c.
        original = bytes.fromhex("ffc300d1fdfb01a9fd630091e10300f9e20300f9")
        self.assertEqual(
            hashlib.sha256(original).hexdigest(),
            "232fb95c4d1498dd57d1f6f70a7595a3842216cfc671cf99d3e10c413d51778c",
        )
        without_first = original[:12] + bytes.fromhex("1f2003d5") + original[16:]
        without_second = original[:16] + bytes.fromhex("1f2003d5")
        rng = random.Random(0x152869C)
        cases = []
        for scratch, code in (
            (DEFAULT_SCRATCH, DEFAULT_CODE),
            (DEFAULT_SCRATCH + 0x200000, DEFAULT_CODE + 0x300000),
        ):
            for _ in range(8):
                state = self.state(scratch, code)
                registers = list(state.registers)
                registers[1] = rng.getrandbits(64)
                registers[2] = registers[1] ^ (1 << 63)
                cases.append(
                    dataclasses.replace(
                        state,
                        registers=tuple(registers),
                        sp=scratch + 128,
                        memory=bytes(rng.getrandbits(8) for _ in range(4096)),
                    )
                )
        before = self.oracle.run_many(original, cases, trusted_fixture=True)
        after = self.oracle.run_many(without_first, cases, trusted_fixture=True)
        wrong = self.oracle.run_many(without_second, cases, trusted_fixture=True)
        for state, expected, actual, mutant in zip(cases, before, after, wrong, strict=True):
            self.assertTrue(expected.completed)
            self.assertEqual(expected, actual)
            self.assertEqual(expected.state.sp, state.scratch_base + 80)
            self.assertEqual(
                struct.unpack_from("<Q", expected.state.memory, 80)[0], state.registers[2]
            )
            self.assertNotEqual(expected.state.memory, mutant.state.memory)
        guard = dataclasses.replace(cases[0], sp=cases[0].scratch_base + 32)
        fault = self.execute(original, guard)
        shifted_fault = self.execute(without_first, guard)
        self.assertEqual((fault.signal, shifted_fault.signal), (11, 11))
        self.assertEqual(fault.fault_address, guard.scratch_base - 16)
        self.assertEqual(shifted_fault.fault_address, fault.fault_address)
        self.assertNotEqual(shifted_fault.exit_pc, fault.exit_pc)

    def test_original_nonadjacent_stack_overwrite_matches_without_first_store(self):
        # Original sample bytes at 0x81a934.
        original = bytes.fromhex("e90f00f90a058092ea0f00f9")
        self.assertEqual(
            hashlib.sha256(original).hexdigest(),
            "7183d6aebf9fd5a1fbed86cad7f5e64e9330a842d1cc7fcfdd4da3e0b96452ad",
        )
        without_first = bytes.fromhex("1f2003d5") + original[4:]
        without_second = original[:8] + bytes.fromhex("1f2003d5")
        rng = random.Random(0x81A934)
        cases = []
        for scratch, code in (
            (DEFAULT_SCRATCH, DEFAULT_CODE),
            (DEFAULT_SCRATCH + 0x200000, DEFAULT_CODE + 0x300000),
        ):
            for _ in range(8):
                state = self.state(scratch, code)
                registers = list(state.registers)
                registers[9] = rng.getrandbits(64)
                cases.append(
                    dataclasses.replace(
                        state,
                        registers=tuple(registers),
                        sp=scratch + 128,
                        memory=bytes(rng.getrandbits(8) for _ in range(4096)),
                    )
                )
        before = self.oracle.run_many(original, cases, trusted_fixture=True)
        after = self.oracle.run_many(without_first, cases, trusted_fixture=True)
        wrong = self.oracle.run_many(without_second, cases, trusted_fixture=True)
        for state, expected, actual, mutant in zip(cases, before, after, wrong, strict=True):
            self.assertTrue(expected.completed)
            self.assertEqual(expected, actual)
            self.assertEqual(
                struct.unpack_from("<Q", expected.state.memory, 152)[0], 0xFFFFFFFFFFFFFFD7
            )
            self.assertNotEqual(expected.state.memory, mutant.state.memory)

    def test_original_intervening_stack_read_refutes_store_omission(self):
        # Original sample bytes at 0x1559e40.
        original = bytes.fromhex("e00300f989198092e90340f9e90300f9")
        self.assertEqual(
            hashlib.sha256(original).hexdigest(),
            "dd1f0a0014bef485b19c202cae229661bef9f67c88d26502a9b70d9a07e036c1",
        )
        without_first = bytes.fromhex("1f2003d5") + original[4:]
        rng = random.Random(0x1559E40)
        cases = []
        for scratch, code in (
            (DEFAULT_SCRATCH, DEFAULT_CODE),
            (DEFAULT_SCRATCH + 0x200000, DEFAULT_CODE + 0x300000),
        ):
            for _ in range(8):
                state = self.state(scratch, code)
                registers = list(state.registers)
                registers[0] = rng.getrandbits(64)
                memory = bytearray(state.memory)
                struct.pack_into("<Q", memory, 128, registers[0] ^ 1)
                cases.append(
                    dataclasses.replace(
                        state, registers=tuple(registers), sp=scratch + 128, memory=bytes(memory)
                    )
                )
        before = self.oracle.run_many(original, cases, trusted_fixture=True)
        after = self.oracle.run_many(without_first, cases, trusted_fixture=True)
        for state, expected, omitted in zip(cases, before, after, strict=True):
            self.assertTrue(expected.completed)
            self.assertTrue(omitted.completed)
            self.assertEqual(expected.state.registers[9], state.registers[0])
            self.assertNotEqual(expected.state.registers[9], omitted.state.registers[9])
            self.assertNotEqual(expected.state.memory, omitted.state.memory)

    def test_original_nonadjacent_stack_store_after_read_matches_omission(self):
        # Original sample bytes at 0x16beac4.
        original = bytes.fromhex("ffc300d1fdfb01a9fd63009149198092ea0b40f9ea0b00f9ab078092eb0b00f9")
        self.assertEqual(
            hashlib.sha256(original).hexdigest(),
            "081c0036d8592a6cb2c02ec7d8b5f3dbf0c035d0c27c069b23687aba33ce012c",
        )
        without_first = original[:20] + bytes.fromhex("1f2003d5") + original[24:]
        without_second = original[:28] + bytes.fromhex("1f2003d5")
        rng = random.Random(0x16BEAC4)
        cases = []
        for scratch, code in (
            (DEFAULT_SCRATCH, DEFAULT_CODE),
            (DEFAULT_SCRATCH + 0x200000, DEFAULT_CODE + 0x300000),
        ):
            for _ in range(8):
                state = self.state(scratch, code)
                registers = list(state.registers)
                registers[10] = rng.getrandbits(64)
                memory = bytearray(state.memory)
                struct.pack_into("<Q", memory, 96, rng.getrandbits(64) & ~(1 << 63))
                cases.append(
                    dataclasses.replace(
                        state, registers=tuple(registers), sp=scratch + 128, memory=bytes(memory)
                    )
                )
        before = self.oracle.run_many(original, cases, trusted_fixture=True)
        after = self.oracle.run_many(without_first, cases, trusted_fixture=True)
        wrong = self.oracle.run_many(without_second, cases, trusted_fixture=True)
        for expected, omitted, mutant in zip(before, after, wrong, strict=True):
            self.assertTrue(expected.completed)
            self.assertEqual(expected, omitted)
            self.assertEqual(
                struct.unpack_from("<Q", expected.state.memory, 96)[0], 0xFFFFFFFFFFFFFFC2
            )
            self.assertNotEqual(expected.state.memory, mutant.state.memory)

    def test_selected_development_stack_pair_requires_successful_global_load(self):
        # First 76 bytes of frozen 0x78a3c4+84, ending at the proved branch
        # to its epilogue.
        original = bytes.fromhex(
            "fd7bbfa9fd03009149e083d2eb031f2ac8a500b089b8aef28a0380520959c2f22930f5f2"
            "ec030b2a2b008052ccffff349f050071c10000540b4947f92b010bcbaa2f00a94b008052f7ffff17"
        )
        self.assertEqual(
            hashlib.sha256(original).hexdigest(),
            "d855cfbd8812e2d6a9c4450a0c2c03cd52f091964fdf084611c58a020492075a",
        )
        # SUB retains STP's SP writeback while removing its two memory effects.
        without_save = bytes.fromhex("ff4300d1") + original[4:]
        rng = random.Random(0x78A3C4)
        successful = []
        faults = []
        for bias in (0x200000000, 0x600000000):
            code_base = bias + 0x78A3C4
            for _ in range(8):
                registers = [rng.getrandbits(64) for _ in range(31)]
                registers[28] = DEFAULT_SCRATCH
                page = bytearray(rng.getrandbits(8) for _ in range(4096))
                struct.pack_into("<Q", page, 0xE90, rng.getrandbits(64))
                successful.append(
                    ProgramState(
                        tuple(registers),
                        0,
                        DEFAULT_SCRATCH + 2048,
                        bytes(rng.getrandbits(8) for _ in range(4096)),
                        DEFAULT_SCRATCH,
                        code_base,
                        (GlobalPage(bias + 0x1C43000, bytes(page)),),
                    )
                )
            # The preceding guard page makes the same LDR fault after the save.
            faults.append(
                dataclasses.replace(
                    successful[-1], global_pages=(GlobalPage(bias + 0x1C44000, bytes(4096)),)
                )
            )
        original_success = self.oracle.run_many(original, successful, trusted_fixture=True)
        omitted_success = self.oracle.run_many(without_save, successful, trusted_fixture=True)
        for expected, actual in zip(original_success, omitted_success, strict=True):
            self.assertTrue(expected.completed)
            self.assertEqual(expected.exit_pc, expected.state.code_base + len(original))
            self.assertEqual(expected.state, actual.state)
            self.assertEqual(expected.exit_pc, actual.exit_pc)
        original_fault = self.oracle.run_many(original, faults, trusted_fixture=True)
        omitted_fault = self.oracle.run_many(without_save, faults, trusted_fixture=True)
        for expected, actual in zip(original_fault, omitted_fault, strict=True):
            self.assertEqual((expected.signal, actual.signal), (11, 11))
            self.assertEqual(expected.exit_pc, actual.exit_pc)
            self.assertEqual(expected.fault_address, actual.fault_address)
            self.assertNotEqual(expected.state.memory, actual.state.memory)

    def test_explicit_global_permissions_provenance_and_mapping_limits(self):
        state = self.state(code=DEFAULT_CODE + 0xFFC)
        address = DEFAULT_CODE + 0x100000
        page = GlobalPage(address, bytes(range(256)) * 16)
        registers = list(state.registers)
        registers[9] = address
        state = dataclasses.replace(state, registers=tuple(registers), global_pages=(page,))
        # Cross a code-page boundary while retaining exact source entry alignment.
        code = self.assemble("ldr x0, [x9]\nadd x0, x0, #1")
        outcome = self.execute(code, state)
        self.assertTrue(outcome.completed)
        self.assertEqual(outcome.state.registers[0], int.from_bytes(page.data[:8], "little") + 1)
        self.assertEqual(outcome.state.global_pages, state.global_pages)
        fault = self.execute(self.assemble("str x0, [x9]"), state)
        self.assertFalse(fault.completed)
        self.assertEqual(fault.signal, 11)
        self.assertEqual(fault.fault_address, address)
        self.assertEqual(fault.state, state)
        with self.assertRaisesRegex(Declined, "provenance"):
            self.execute(self.assemble("ldr x9, [x9]\nldr x0, [x9]"), state)
        with self.assertRaisesRegex(Declined, "provenance"):
            self.execute(self.assemble("eor x9, x9, x0\nldr x0, [x9]"), state)
        with self.assertRaisesRegex(ValueError, "overlap"):
            dataclasses.replace(
                state, global_pages=(GlobalPage(state.code_base & ~4095, bytes(4096)),)
            )
        with self.assertRaisesRegex(ValueError, "eight"):
            dataclasses.replace(state, global_pages=(page,) * 9)

    def test_adjacent_readonly_pages_support_cross_page_load_and_keep_page_identity(self):
        address = DEFAULT_CODE + 0x100000
        first = GlobalPage(address, bytes(range(256)) * 16)
        second = GlobalPage(address + 4096, bytes(reversed(range(256))) * 16)
        state = self.state()
        registers = list(state.registers)
        registers[9] = address + 4092
        state = dataclasses.replace(state, registers=tuple(registers), global_pages=(second, first))
        outcome = self.execute(self.assemble("ldr x0, [x9]"), state)
        expected = list(state.registers)
        expected[0] = int.from_bytes(first.data[-4:] + second.data[:4], "little")
        self.assertTrue(outcome.completed)
        self.assertEqual(outcome.state, dataclasses.replace(state, registers=tuple(expected)))
        self.assertEqual(outcome.state.global_pages, (second, first))
        fault = self.execute(self.assemble("str x0, [x9]"), state)
        self.assertEqual(fault.signal, 11)
        self.assertEqual(fault.fault_address, address + 4092)
        self.assertEqual(fault.state, state)
        for offset in (-4, 8188):
            with self.subTest(offset=offset):
                registers[9] = address + offset
                guarded = dataclasses.replace(state, registers=tuple(registers))
                fault = self.execute(self.assemble("ldr x0, [x9]"), guarded)
                self.assertEqual(fault.signal, 11)
                self.assertEqual(fault.state, guarded)
                self.assertEqual(fault.fault_address, address - 4 if offset < 0 else address + 8192)

    def test_global_clusters_reject_duplicates_guard_conflicts_and_code_aliases(self):
        state = self.state()
        address = DEFAULT_CODE + 0x100000
        page = GlobalPage(address, bytes(4096))
        for pages in (
            (page, page),
            (page, GlobalPage(address + 8192, bytes(4096))),
            (
                GlobalPage(DEFAULT_CODE + 4096, bytes(4096)),
                GlobalPage(DEFAULT_CODE + 8192, bytes(4096)),
            ),
            (GlobalPage(DEFAULT_SCRATCH + 4096, bytes(4096)),),
        ):
            with self.subTest(addresses=[page.address for page in pages]):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    dataclasses.replace(state, global_pages=pages)
        pages = tuple(GlobalPage(address + i * 4096, bytes([i]) * 4096) for i in reversed(range(8)))
        admitted = dataclasses.replace(state, global_pages=pages)
        outcome = self.execute(bytes.fromhex("1f2003d5"), admitted)
        self.assertTrue(outcome.completed)
        self.assertEqual(outcome.state, admitted)
        with self.assertRaisesRegex(ValueError, "eight"):
            dataclasses.replace(
                state, global_pages=pages + (GlobalPage(address + 8 * 4096, bytes(4096)),)
            )

    def test_native_runner_independently_rejects_duplicate_global_pages(self):
        oracle = ProgramOracle()
        command = oracle.backend._command
        code = bytes.fromhex("1f2003d5")
        address = DEFAULT_CODE + 0x100000
        state = dataclasses.replace(
            self.state(),
            global_pages=(
                GlobalPage(address, bytes(4096)),
                GlobalPage(address + 4096, bytes(4096)),
            ),
        )

        def duplicate_page(args, **kwargs):
            if args[0] == oracle.backend.qemu:
                payload = bytearray(kwargs["input"])
                first = 8 + 37 * 8 + len(code) + 4096
                payload[first + 4104 : first + 4112] = payload[first : first + 8]
                kwargs["input"] = bytes(payload)
            return command(args, **kwargs)

        oracle.backend._command = duplicate_page
        with self.assertRaisesRegex(OracleError, r"failed \(2\)"):
            oracle.run(code, state, trusted_fixture=True)

    def test_native_runner_maps_the_code_page_without_write_permission(self):
        # program.py advertises "no self modification" in its execution model.
        # _confine declines a store into the code arena, so the only way to show
        # the mapping itself enforces it is to put one there after admission.
        oracle = ProgramOracle()
        command = oracle.backend._command
        code = bytes.fromhex("1f2003d51f2003d5")
        # adr x0, . ; str x1, [x0] -- a store into the page being executed.
        self_modify = bytes.fromhex("00000010010000f9")
        state = self.state()

        def overwrite(args, **kwargs):
            if args[0] == oracle.backend.qemu:
                payload = bytearray(kwargs["input"])
                first = 8 + 37 * 8
                payload[first : first + len(code)] = self_modify
                kwargs["input"] = bytes(payload)
            return command(args, **kwargs)

        oracle.backend._command = overwrite
        outcome = oracle.run(code, state, trusted_fixture=True)
        self.assertFalse(outcome.completed)
        self.assertEqual(outcome.signal, 11)
        self.assertEqual(outcome.exit_pc, state.code_base + 4)
        self.assertEqual(outcome.fault_address, state.code_base)


if __name__ == "__main__":
    unittest.main()
