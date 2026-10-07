import dataclasses
import hashlib
import pathlib
import random
import struct
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tools.oracle.aarch64 import Declined, OracleError
from tools.oracle.program import GlobalPage
from tools.oracle.sparse import Fragment, SparseOracle, SparseState, MAGIC_LARGE

TAIL = Fragment(0x751394, bytes.fromhex("c92c8052695600b97fea03f968e603f9aa060014"))
DISPATCH = Fragment(
    0x752E4C,
    bytes.fromhex("685640b9091d0a7108fdff5429d5ffd0296124910a0000102b79a8b84a010b8b40011fd6"),
)
FULL_PREDECESSOR = Fragment(
    0x75136C,
    bytes.fromhex(
        "68ee43f969f243f969fa00f9e9924d39290100124a0180d20b0280d23f01007249118b9a696a09f9c92c8052695600b97fea03f968e603f9aa060014"
    ),
)
EXITS = (0x748B84, 0x749100, 0x752DF4)
MASK = (1 << 64) - 1


def fixture(bias=0x200000000, index=358, target=EXITS[0]):
    rng = random.Random(0x751394 + index)
    registers = [rng.getrandbits(64) for _ in range(31)]
    registers[19] = 0x100000000
    memory = bytearray(rng.getrandbits(8) for _ in range(4096))
    struct.pack_into("<I", memory, 84, index)
    table = bytearray(8192)
    if index <= 647:
        struct.pack_into("<i", table, 0x918 + index * 4, target - 0x752E60)
    pages = (
        GlobalPage(bias + 0x1F9000, bytes(table[4096:])),
        GlobalPage(bias + 0x1F8000, bytes(table[:4096])),
    )
    return SparseState(
        tuple(registers), 0xF0000000, 0x123456789, bytes(memory), 0x100000000, bias, pages
    )


def full_fixture(bias=0x200000000, target=EXITS[0], bit=0):
    state = fixture(bias, target=target)
    registers = list(state.registers)
    registers[23] = state.scratch_base + 128
    memory = bytearray(state.memory) + bytearray((i * 29 + 7) & 255 for i in range(4096))
    memory[996] = bit
    return dataclasses.replace(state, registers=tuple(registers), memory=bytes(memory))


class SparseOracleTest(unittest.TestCase):
    def setUp(self):
        self.oracle = SparseOracle()

    def test_selected_development_table_route_matches_direct_successor(self):
        # Frozen development route at 0x15a7d94. Its source groups occur in
        # three discontiguous fragments, with a table read before the final BR.
        head = Fragment(0x15A7D94, bytes.fromhex("48008052685e00b9f6ffff17"))
        bridge = Fragment(0x15A7D74, bytes.fromhex("f9020014"))
        tail = Fragment(
            0x15A8958,
            bytes.fromhex(
                "685e40b909e500716893ff542967ff9029591591ea88ff102b7968784a090b8b40011fd6"
            ),
        )
        self.assertEqual(
            hashlib.sha256(head.code + bridge.code + tail.code).hexdigest(),
            "9df02e7c1d00a1c82b767ddb13831081bfd644697ad8104a3d69164ab129ea71",
        )
        original = (head, bridge, tail)
        direct = (
            head,
            bridge,
            Fragment(tail.source_address, tail.code[:-4] + bytes.fromhex("98fdff17")),
        )
        wrong = (
            head,
            bridge,
            Fragment(tail.source_address, tail.code[:-4] + bytes.fromhex("95fcff17")),
        )
        exits = (0x15A7FD8, 0x15A7BCC)
        rng = random.Random(0x15A7D94)
        states = []
        faults = []
        for bias in (0x200000000, 0x600000000):
            for offset in (1024, 2048):
                table = bytearray(4096)
                struct.pack_into("<H", table, 0x55A, 340)
                registers = [rng.getrandbits(64) for _ in range(31)]
                registers[19] = 0x100000000 + offset
                state = SparseState(
                    tuple(registers),
                    0x90000000,
                    0x100000800,
                    bytes(rng.getrandbits(8) for _ in range(4096)),
                    0x100000000,
                    bias,
                    (GlobalPage(bias + 0x28C000, bytes(table)),),
                )
                states.append(state)
                faults.append(dataclasses.replace(state, global_pages=()))
        before = self.oracle.run_many(
            original, head.source_address, exits, states, trusted_fixture=True
        )
        after = self.oracle.run_many(
            direct, head.source_address, exits, states, trusted_fixture=True
        )
        mutated = self.oracle.run_many(
            wrong, head.source_address, exits, states, trusted_fixture=True
        )
        for state, left, right, bad in zip(states, before, after, mutated, strict=True):
            self.assertTrue(left.completed)
            self.assertEqual(left, right)
            self.assertEqual(left.exit_pc, state.load_bias + exits[0])
            self.assertEqual(bad.exit_pc, state.load_bias + exits[1])
            self.assertNotEqual(left, bad)
            address = state.registers[19] - state.scratch_base + 92
            self.assertEqual(struct.unpack_from("<I", left.state.memory, address)[0], 2)
        original_faults = self.oracle.run_many(
            original, head.source_address, exits, faults, trusted_fixture=True
        )
        direct_faults = self.oracle.run_many(
            direct, head.source_address, exits, faults, trusted_fixture=True
        )
        for state, left, right in zip(faults, original_faults, direct_faults, strict=True):
            self.assertEqual(left, right)
            self.assertEqual(left.signal, 11)
            self.assertEqual(left.fault_address, state.load_bias + 0x28C55A)

    def test_development_route_through_a_frame_alias_needs_its_entry_relation(self):
        # Development route at 0x15a7ac4: it stores state 7 at [x19+0x5c], then
        # writes [x27+0x80]. The prologue sets x27 = x19+0x90, which is what
        # keeps that write off the state; Nyx reports 0x15a7d24 only under it.
        head = Fragment(0x15A7AC4, bytes.fromhex("e8008052685e00b97f4300f9a9000014"))
        bridge = Fragment(0x15A7D74, bytes.fromhex("f9020014"))
        tail = Fragment(
            0x15A8958,
            bytes.fromhex(
                "685e40b909e500716893ff542967ff9029591591ea88ff102b7968784a090b8b40011fd6"
            ),
        )
        original = (head, bridge, tail)
        direct = (
            head,
            bridge,
            Fragment(tail.source_address, tail.code[:-4] + bytes.fromhex("ebfcff17")),
        )
        exits = (0x15A7D24, 0x15A7DBC, 0x15A7BCC)
        rng = random.Random(0x15A7AC4)
        for displacement, state_value, exit_pc in ((0x90, 7, exits[0]), (-0x24, 0, exits[1])):
            states = []
            for bias in (0x200000000, 0x600000000):
                for offset in (1024, 2048):
                    table = bytearray(4096)
                    struct.pack_into("<H", table, 0x556, 205)
                    struct.pack_into("<H", table, 0x556 + 14, 167)
                    registers = [rng.getrandbits(64) for _ in range(31)]
                    registers[19] = 0x100000000 + offset
                    registers[27] = (registers[19] + displacement) & MASK
                    states.append(
                        SparseState(
                            tuple(registers),
                            0x90000000,
                            0x100000800,
                            bytes(rng.getrandbits(8) for _ in range(4096)),
                            0x100000000,
                            bias,
                            (GlobalPage(bias + 0x28C000, bytes(table)),),
                        )
                    )
            before = self.oracle.run_many(
                original, head.source_address, exits, states, trusted_fixture=True
            )
            after = self.oracle.run_many(
                direct, head.source_address, exits, states, trusted_fixture=True
            )
            for state, left, right in zip(states, before, after, strict=True):
                self.assertTrue(left.completed)
                self.assertEqual(left.exit_pc, state.load_bias + exit_pc)
                address = state.registers[19] - state.scratch_base + 0x5C
                self.assertEqual(
                    struct.unpack_from("<I", left.state.memory, address)[0], state_value
                )
                if state_value:
                    self.assertEqual(left, right)
                else:
                    self.assertEqual(right.exit_pc, state.load_bias + exits[0])

    def test_writable_global_page_captures_a_store_and_readonly_still_faults(self):
        store = (Fragment(0x1000, bytes.fromhex("010000f9")),)  # str x1, [x0]
        page = 0x300000000
        registers = [0] * 31
        registers[0] = page + 8
        registers[1] = 0x1122334455667788
        base = SparseState(
            tuple(registers),
            0,
            0x100000800,
            bytes(8192),
            0x100000000,
            0x200000000,
            (GlobalPage(page, bytes(4096), True),),
        )
        written = self.oracle.run(store, 0x1000, (0x1004,), base, trusted_fixture=True)
        self.assertTrue(written.completed)
        self.assertTrue(written.state.global_pages[0].writable)
        self.assertEqual(
            struct.unpack_from("<Q", written.state.global_pages[0].data, 8)[0], 0x1122334455667788
        )
        readonly = dataclasses.replace(base, global_pages=(GlobalPage(page, bytes(4096)),))
        faulted = self.oracle.run(store, 0x1000, (0x1004,), readonly, trusted_fixture=True)
        self.assertEqual((faulted.signal, faulted.fault_address), (11, page + 8))
        with self.assertRaises(ValueError):
            dataclasses.replace(base, memory=bytes(4096))

    def test_pages_one_apart_share_an_arena_and_the_gap_still_faults(self):
        load = (Fragment(0x1000, bytes.fromhex("000040f9")),)  # ldr x0, [x0]
        first, second = 0x300000000, 0x300002000
        pages = (
            GlobalPage(first, bytes(4096)),
            GlobalPage(second, (7).to_bytes(8, "little") + bytes(4088)),
        )
        registers = [0] * 31
        registers[0] = second
        state = SparseState(
            tuple(registers), 0, 0x100000800, bytes(4096), 0x100000000, 0x200000000, pages
        )
        # The exit pages 0x1000 and 0x3000 share a guard page too.
        loaded = self.oracle.run(load, 0x1000, (0x1004, 0x3000), state, trusted_fixture=True)
        self.assertTrue(loaded.completed)
        self.assertEqual(loaded.state.registers[0], 7)
        registers[0] = first + 4096
        gap = self.oracle.run(
            load,
            0x1000,
            (0x1004,),
            dataclasses.replace(state, registers=tuple(registers)),
            trusted_fixture=True,
        )
        self.assertEqual((gap.signal, gap.fault_address), (11, first + 4096))

    def test_original_development_tail_runtime_table_and_bias(self):
        self.assertEqual(
            hashlib.sha256(TAIL.code).hexdigest(),
            "282f0c1dd7c3ef61a20be4600a88b92fa6cf40198204e1b94f48b420b7bcab5f",
        )
        self.assertEqual(
            hashlib.sha256(DISPATCH.code).hexdigest(),
            "72d278301d22350a0de8cf680f3d4a65668f7baf8185a01df11cb713794ff952",
        )
        states = [
            fixture(bias, target=target)
            for bias in (0x200000000, 0x600000000)
            for target in EXITS[:2]
        ]
        results = self.oracle.run_many(
            (TAIL, DISPATCH), TAIL.source_address, EXITS, states, trusted_fixture=True
        )
        for state, outcome in zip(states, results, strict=True):
            table = state.global_pages[1].data
            delta = struct.unpack_from("<i", table, 0x918 + 358 * 4)[0]
            target = state.load_bias + 0x752E60 + delta
            registers = list(state.registers)
            registers[8] = 358
            registers[9] = state.load_bias + 0x1F8918
            registers[10] = target
            registers[11] = delta & MASK
            memory = bytearray(state.memory)
            struct.pack_into("<I", memory, 84, 358)
            struct.pack_into("<Q", memory, 1992, state.registers[8])
            struct.pack_into("<Q", memory, 2000, 0)
            self.assertTrue(outcome.completed)
            self.assertEqual(outcome.exit_pc, target)
            self.assertEqual(
                outcome.state,
                dataclasses.replace(
                    state, registers=tuple(registers), nzcv=0x80000000, memory=bytes(memory)
                ),
            )
        self.assertNotEqual(results[0].exit_pc, results[1].exit_pc)

    def test_full_real_predecessor_with_large_scratch(self):
        self.assertEqual(
            hashlib.sha256(FULL_PREDECESSOR.code).hexdigest(),
            "48705b25a687fb17ca141111a41483ae620dddc89d03ee73b7845d3b117fe3f6",
        )
        states = [
            full_fixture(bias, target, bit)
            for bias in (0x200000000, 0x600000000)
            for target in EXITS[:2]
            for bit in (0, 1)
        ]
        outcomes = self.oracle.run_many(
            (FULL_PREDECESSOR, DISPATCH),
            FULL_PREDECESSOR.source_address,
            EXITS,
            states,
            trusted_fixture=True,
        )
        for state, outcome in zip(states, outcomes, strict=True):
            delta = struct.unpack_from("<i", state.global_pages[1].data, 0x918 + 358 * 4)[0]
            target = state.load_bias + 0x752E60 + delta
            registers = list(state.registers)
            registers[8] = 358
            registers[9] = state.load_bias + 0x1F8918
            registers[10] = target
            registers[11] = delta & MASK
            memory = bytearray(state.memory)
            memory[496:504] = memory[2016:2024]
            memory[1992:2000] = memory[2008:2016]
            struct.pack_into("<Q", memory, 4816, 10 if memory[996] & 1 else 16)
            struct.pack_into("<I", memory, 84, 358)
            struct.pack_into("<Q", memory, 2000, 0)
            self.assertTrue(outcome.completed)
            self.assertEqual(outcome.exit_pc, target)
            self.assertEqual(
                outcome.state,
                dataclasses.replace(
                    state, registers=tuple(registers), nzcv=0x80000000, memory=bytes(memory)
                ),
            )
        with self.assertRaises(ValueError):
            dataclasses.replace(states[0], memory=bytes(12288))
        collision = dataclasses.replace(
            states[0], global_pages=(GlobalPage(states[0].scratch_base + 8192, bytes(4096)),)
        )
        with self.assertRaises(Declined):
            self.oracle.run(
                (FULL_PREDECESSOR, DISPATCH),
                FULL_PREDECESSOR.source_address,
                EXITS,
                collision,
                trusted_fixture=True,
            )

    def test_selected_development_pair_loads_have_register_only_surrogate(self):
        original = bytes.fromhex(
            "fd7bbfa9fd03009149e083d2eb031f2ac8a500b089b8aef28a0380520959c2f22930f5f2"
            "ec030b2a2b008052ccffff349f050071c10000540b4947f92b010bcbaa2f00a94b008052"
            "f7ffff17fd7bc1a8c0035fd6"
        )
        self.assertEqual(
            hashlib.sha256(original).hexdigest(),
            "a2d72e7ea7e9933978a2fe2cc4aec284721f23322db5979fb2105c2d29b403f8",
        )
        # The bounded surrogate redirects after STP to preserve its x11 value in
        # x29 before the loop changes x11. Its epilogue reads only registers.
        replaced = bytearray(original)
        struct.pack_into("<I", replaced, 0x78A408 - 0x78A3C4, 0x1400003E)
        struct.pack_into("<I", replaced, 0x78A410 - 0x78A3C4, 0x14000044)
        original_fragments = (Fragment(0x78A3C4, original),)
        surrogate_fragments = (
            Fragment(0x78A3C4, bytes(replaced)),
            Fragment(0x78A500, bytes.fromhex("fd030baa4b008052b8ffff17")),
            Fragment(0x78A520, bytes.fromhex("fe031daafd030aaaff430091c0035fd6")),
        )
        exits = (0x78A418, 0x78A41C)
        rng = random.Random(0x78A3C4)
        states = []
        faults = []
        for bias in (0x200000000, 0x600000000):
            for index in range(8):
                exit_address = bias + exits[index % 2]
                page = bytearray(rng.getrandbits(8) for _ in range(4096))
                struct.pack_into("<Q", page, 0xE90, (0xA98112C875C41F02 - exit_address) & MASK)
                state = SparseState(
                    tuple(rng.getrandbits(64) for _ in range(31)),
                    0xF0000000,
                    0x100000800,
                    bytes(rng.getrandbits(8) for _ in range(4096)),
                    0x100000000,
                    bias,
                    (GlobalPage(bias + 0x1C43000, bytes(page)),),
                )
                states.append(state)
            faults.append(dataclasses.replace(states[-1], global_pages=()))
        expected = self.oracle.run_many(
            original_fragments, 0x78A3C4, exits, states, trusted_fixture=True
        )
        actual = self.oracle.run_many(
            surrogate_fragments, 0x78A3C4, exits, states, trusted_fixture=True
        )
        for state, left, right in zip(states, expected, actual, strict=True):
            self.assertTrue(left.completed)
            self.assertEqual(left, right)
            self.assertEqual(left.state.sp, state.sp)
            self.assertEqual(left.state.registers[11], 1)
            self.assertEqual(struct.unpack_from("<QQ", left.state.memory, 2032), (28, left.exit_pc))
        original_faults = self.oracle.run_many(
            original_fragments, 0x78A3C4, exits, faults, trusted_fixture=True
        )
        surrogate_faults = self.oracle.run_many(
            surrogate_fragments, 0x78A3C4, exits, faults, trusted_fixture=True
        )
        for left, right in zip(original_faults, surrogate_faults, strict=True):
            self.assertEqual(left.signal, 11)
            self.assertEqual(left, right)

    def test_sample_pair_loads_have_register_only_surrogate(self):
        # Sample bytes from 0x1656244 through 0x1656268, ending before the next prologue.
        original = bytes.fromhex(
            "ffc300d1fdfb01a9fd630091691c80922901098be90340f9e90340f9e90b40f9fdfb41a9ffc30091"
        )
        self.assertEqual(
            hashlib.sha256(original).hexdigest(),
            "9c6d563b31df3932ab022dd9412054e00c8f2850c59e69ed0b06b50483f46235",
        )
        surrogate = bytearray(original)
        # x29 is otherwise dead between the prologue store and paired reload.
        struct.pack_into("<I", surrogate, 8, 0xD503201F)
        struct.pack_into("<I", surrogate, 32, 0xD503201F)
        mutant = bytearray(original)
        struct.pack_into("<I", mutant, 32, 0xD503201F)
        exits = (0x165626C,)
        rng = random.Random(0x1656244)
        cases = []
        faults = []
        for bias in (0x200000000, 0x600000000):
            for _ in range(8):
                cases.append(
                    SparseState(
                        tuple(rng.getrandbits(64) for _ in range(31)),
                        0x90000000,
                        0x100000800,
                        bytes(rng.getrandbits(8) for _ in range(4096)),
                        0x100000000,
                        bias,
                    )
                )
            faults.append(dataclasses.replace(cases[-1], sp=0x100000000))
        before = self.oracle.run_many(
            (Fragment(0x1656244, original),), 0x1656244, exits, cases, trusted_fixture=True
        )
        after = self.oracle.run_many(
            (Fragment(0x1656244, bytes(surrogate)),), 0x1656244, exits, cases, trusted_fixture=True
        )
        wrong = self.oracle.run_many(
            (Fragment(0x1656244, bytes(mutant)),), 0x1656244, exits, cases, trusted_fixture=True
        )
        for left, right, bad in zip(before, after, wrong, strict=True):
            self.assertTrue(left.completed)
            self.assertEqual(left, right)
            self.assertNotEqual(left.state.registers[29], bad.state.registers[29])
        original_faults = self.oracle.run_many(
            (Fragment(0x1656244, original),), 0x1656244, exits, faults, trusted_fixture=True
        )
        surrogate_faults = self.oracle.run_many(
            (Fragment(0x1656244, bytes(surrogate)),), 0x1656244, exits, faults, trusted_fixture=True
        )
        for left, right in zip(original_faults, surrogate_faults, strict=True):
            self.assertEqual(left.signal, 11)
            self.assertEqual(left, right)

    def test_selected_development_census_pair_loads_have_register_only_surrogate(self):
        original = bytes.fromhex(
            "fd7bbfa9fd030091e50304aae40303aae84000d008210591080140f9097d4092"
            "08fd6093081140920ab082d2aafeb8f2aa01c7f26a75faf208010a8b08114092"
            "2a0080d24a21c89a29010a8a2829c89a1f0500f18000005483c841b9fd7bc1a8"
        )
        self.assertEqual(
            hashlib.sha256(original).hexdigest(),
            "a134fd55784ff31d6c9b8d9ee34958e5ddaa05f2fab81f94a7b9c756225b266f",
        )
        surrogate = bytearray(original)
        # Preserve the incoming x29 and use register-only SP adjustment at the
        # selected epilogue; the earlier STP remains an observable write.
        struct.pack_into("<I", surrogate, 4, 0xD503201F)
        struct.pack_into("<I", surrogate, 92, 0x910043FF)
        mutant = bytearray(surrogate)
        struct.pack_into("<I", mutant, 92, 0xD503201F)
        exits = (0x1503D80, 0x1503D84)
        rng = random.Random(0x1503D20)
        cases = []
        faults = []
        for bias in (0x200000000, 0x600000000):
            for _ in range(8):
                registers = [rng.getrandbits(64) for _ in range(31)]
                registers[3] = 0x100000000 + 512
                page = bytearray(rng.getrandbits(8) for _ in range(4096))
                struct.pack_into("<Q", page, 0x148, rng.getrandbits(32) << 32)
                cases.append(
                    SparseState(
                        tuple(registers),
                        0x90000000,
                        0x100000800,
                        bytes(rng.getrandbits(8) for _ in range(4096)),
                        0x100000000,
                        bias,
                        (GlobalPage(bias + 0x1D21000, bytes(page)),),
                    )
                )
            faults.append(dataclasses.replace(cases[-1], sp=0x100000000))
        before = self.oracle.run_many(
            (Fragment(0x1503D20, original),), 0x1503D20, exits, cases, trusted_fixture=True
        )
        after = self.oracle.run_many(
            (Fragment(0x1503D20, bytes(surrogate)),), 0x1503D20, exits, cases, trusted_fixture=True
        )
        wrong = self.oracle.run_many(
            (Fragment(0x1503D20, bytes(mutant)),), 0x1503D20, exits, cases, trusted_fixture=True
        )
        for left, right, bad in zip(before, after, wrong, strict=True):
            self.assertTrue(left.completed)
            self.assertEqual(left.exit_pc, 0x1503D80 + left.state.load_bias)
            self.assertEqual(left, right)
            self.assertNotEqual(left.state.sp, bad.state.sp)
        original_faults = self.oracle.run_many(
            (Fragment(0x1503D20, original),), 0x1503D20, exits, faults, trusted_fixture=True
        )
        surrogate_faults = self.oracle.run_many(
            (Fragment(0x1503D20, bytes(surrogate)),), 0x1503D20, exits, faults, trusted_fixture=True
        )
        for left, right in zip(original_faults, surrogate_faults, strict=True):
            self.assertEqual(left.signal, 11)
            self.assertEqual(left, right)

    def test_sample_global_mba_preserves_indirect_call_target(self):
        # Sample bytes from 0x13c665c through 0x13c66ac, stopping before BLR.
        original = bytes.fromhex(
            "484900f0097946f9e903096baa289552ca6eb5722b010a2a29010a0a6901090b"
            "087946f9e803086b0a259552ca6eb5720b010a2a08010a0a6801080b0a4c8052"
            "08012a9b08d969f8099487526901a072080109ca"
        )
        self.assertEqual(
            hashlib.sha256(original).hexdigest(),
            "8b3482d451c6b32230b7d3a3810e51cad14c5c0e6ab6ac1d3279714594e7d070",
        )
        simplified = bytearray(original)
        for offset, word in (
            (24, 0xD503201F),
            (28, 0x0B090149),
            (52, 0xD503201F),
            (56, 0x0B080148),
        ):
            struct.pack_into("<I", simplified, offset, word)
        mutant = bytearray(simplified)
        struct.pack_into("<I", mutant, 28, 0xD503201F)
        exits = (0x13C66B0,)
        rng = random.Random(0x13C665C)
        cases = []
        faults = []
        for bias in (0x200000000, 0x600000000):
            for delta in (-2, -1, 0, 1):
                for _ in range(2):
                    registers = [rng.getrandbits(64) for _ in range(31)]
                    registers[0] = 0x100000000 + 2048
                    page = bytearray(rng.getrandbits(8) for _ in range(4096))
                    struct.pack_into(
                        "<Q", page, 0xCF0, (rng.getrandbits(32) << 32) | (0xAB76A928 + delta)
                    )
                    cases.append(
                        SparseState(
                            tuple(registers),
                            0x90000000,
                            0x100000800,
                            bytes(rng.getrandbits(8) for _ in range(4096)),
                            0x100000000,
                            bias,
                            (GlobalPage(bias + 0x1CF1000, bytes(page)),),
                        )
                    )
            faults.append(dataclasses.replace(cases[-1], global_pages=()))
            faults.append(
                dataclasses.replace(
                    cases[-1], registers=(0x100000000 + 256,) + cases[-1].registers[1:]
                )
            )
        before = self.oracle.run_many(
            (Fragment(0x13C665C, original),), 0x13C665C, exits, cases, trusted_fixture=True
        )
        after = self.oracle.run_many(
            (Fragment(0x13C665C, bytes(simplified)),), 0x13C665C, exits, cases, trusted_fixture=True
        )
        wrong = self.oracle.run_many(
            (Fragment(0x13C665C, bytes(mutant)),), 0x13C665C, exits, cases, trusted_fixture=True
        )
        for state, left, right in zip(cases, before, after, strict=True):
            self.assertTrue(left.completed)
            self.assertEqual(left, right)
            global_value = struct.unpack_from("<Q", state.global_pages[0].data, 0xCF0)[0]
            delta = (global_value & 0xFFFFFFFF) - 0xAB76A928
            table_offset = 2048 + 232 - 616 * delta
            self.assertEqual(
                left.state.registers[8],
                struct.unpack_from("<Q", state.memory, table_offset)[0] ^ 0xB3CA0,
            )
        self.assertTrue(any(left != bad for left, bad in zip(before, wrong, strict=True)))
        original_faults = self.oracle.run_many(
            (Fragment(0x13C665C, original),), 0x13C665C, exits, faults, trusted_fixture=True
        )
        simplified_faults = self.oracle.run_many(
            (Fragment(0x13C665C, bytes(simplified)),),
            0x13C665C,
            exits,
            faults,
            trusted_fixture=True,
        )
        for left, right in zip(original_faults, simplified_faults, strict=True):
            self.assertEqual(left.signal, 11)
            self.assertEqual(left, right)

    def test_sample_repeated_global_mba_preserves_live_x21(self):
        # Sample bytes from 0x144d07c through 0x144d0d4, stopping before BLR.
        original = bytes.fromhex(
            "fa3b40b9e83f40b9e80f00b9fb2340f9284500f008810991080140f9e80308cb"
            "099480d209cdbcf2a9e8c6f2a99aecf2090109aa0a9480d20acdbcf2aae8c6f2"
            "aa9aecf208010a8a2801088b7523c89a284500f008010b91080140f9"
        )
        self.assertEqual(
            hashlib.sha256(original).hexdigest(),
            "85667c8650426f56f351b7cee616f7ffb8f173f4780c6d58b93918b561bddefb",
        )
        simplified = bytearray(original)
        struct.pack_into("<I", simplified, 68, 0xD503201F)
        struct.pack_into("<I", simplified, 72, 0x8B080148)
        mutant = bytearray(simplified)
        struct.pack_into("<I", mutant, 72, 0xD503201F)
        exits = (0x144D0D8,)
        rng = random.Random(0x144D07C)
        cases = []
        faults = []
        constant = 0x64D53745E66804A0
        for bias in (0x200000000, 0x600000000):
            for global_value in (
                0,
                1,
                constant,
                constant + 1,
                constant - 1,
                0xFFFFFFFFFFFFFFFF,
                rng.getrandbits(64),
                rng.getrandbits(64),
            ):
                registers = [rng.getrandbits(64) for _ in range(31)]
                page = bytearray(rng.getrandbits(8) for _ in range(4096))
                struct.pack_into("<Q", page, 0x260, global_value)
                cases.append(
                    SparseState(
                        tuple(registers),
                        0x90000000,
                        0x100000800,
                        bytes(rng.getrandbits(8) for _ in range(4096)),
                        0x100000000,
                        bias,
                        (GlobalPage(bias + 0x1CF4000, bytes(page)),),
                    )
                )
            faults.append(dataclasses.replace(cases[-1], global_pages=()))
            faults.append(dataclasses.replace(cases[-1], sp=0x100000FF0))
        before = self.oracle.run_many(
            (Fragment(0x144D07C, original),), 0x144D07C, exits, cases, trusted_fixture=True
        )
        after = self.oracle.run_many(
            (Fragment(0x144D07C, bytes(simplified)),), 0x144D07C, exits, cases, trusted_fixture=True
        )
        wrong = self.oracle.run_many(
            (Fragment(0x144D07C, bytes(mutant)),), 0x144D07C, exits, cases, trusted_fixture=True
        )
        for state, left, right in zip(cases, before, after, strict=True):
            self.assertTrue(left.completed)
            self.assertEqual(left, right)
            global_value = struct.unpack_from("<Q", state.global_pages[0].data, 0x260)[0]
            source = struct.unpack_from("<Q", state.memory, 0x800 + 64)[0]
            self.assertEqual(
                left.state.registers[21], (source << ((constant - global_value) & 63)) & MASK
            )
        self.assertTrue(any(left != bad for left, bad in zip(before, wrong, strict=True)))
        original_faults = self.oracle.run_many(
            (Fragment(0x144D07C, original),), 0x144D07C, exits, faults, trusted_fixture=True
        )
        simplified_faults = self.oracle.run_many(
            (Fragment(0x144D07C, bytes(simplified)),),
            0x144D07C,
            exits,
            faults,
            trusted_fixture=True,
        )
        for left, right in zip(original_faults, simplified_faults, strict=True):
            self.assertEqual(left.signal, 11)
            self.assertEqual(left, right)

    def test_selected_conditional_dispatches_preserve_both_arms_and_state(self):
        routes = (
            (
                0xBDCA84,
                0xBDC634,
                0xBDCAD4,
                "e8018052090280521f04007108b1891a680a00b9e7feff17",
                "28010014",
                "680a40b9098d0071c8e1ff54c9b2ffd029a13b918adaff102b7968784a090b8b40011fd6",
                "47646a562c2621d2b585375e7747cdc6cabdf000a7a7a39b1bd10277aa388fac",
                0x236000,
                0x236EE8,
                ((0xF06, 322), (0xF08, 596)),
                (0xBDCB40, 0xBDCF88),
                0x37000268,
                0x14000124,
                0,
                19,
                (0, 1, 2, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF),
            ),
            (
                0x144D064,
                0x144D5F4,
                0x144CEAC,
                "28008052490280523f0300720811891ae81300b95f010014",
                "2efeff17",
                "e81340b909490071a8e3ff54c971ffd029b11b918aceff102b7968784a090b8b40011fd6",
                "683cfd4a473457274582d2f1d8707fce246904709aedf51eb7020354b557c73d",
                0x286000,
                0x2866EC,
                ((0x6EE, 677), (0x710, 352)),
                (0x144D324, 0x144CE10),
                0x370022C8,
                0x17FFFFD0,
                25,
                31,
                (0, 1, 2, 3, 0xFFFFFFFE, 0xFFFFFFFF),
            ),
        )
        for (
            entry,
            hub,
            dispatch,
            head_hex,
            hub_hex,
            tail_hex,
            digest,
            table_page,
            table_address,
            table_entries,
            exits,
            direct_word,
            fallthrough_word,
            control_register,
            pointer_register,
            values,
        ) in routes:
            with self.subTest(entry=hex(entry)):
                head, bridge, tail = (bytes.fromhex(x) for x in (head_hex, hub_hex, tail_hex))
                self.assertEqual(hashlib.sha256(head + bridge + tail).hexdigest(), digest)
                table = bytearray(4096)
                for offset, value in table_entries:
                    struct.pack_into("<H", table, offset, value)
                original = (Fragment(entry, head), Fragment(hub, bridge), Fragment(dispatch, tail))
                direct = (
                    Fragment(entry, head),
                    Fragment(hub, bridge),
                    Fragment(dispatch, tail[:-4] + struct.pack("<I", direct_word)),
                    Fragment(dispatch + len(tail), struct.pack("<I", fallthrough_word)),
                )
                swapped = (
                    Fragment(entry, head),
                    Fragment(hub, bridge),
                    Fragment(dispatch, tail[:-4] + struct.pack("<I", direct_word ^ 0x01000000)),
                    Fragment(dispatch + len(tail), struct.pack("<I", fallthrough_word)),
                )
                rng = random.Random(entry)
                states = []
                expected = []
                faults = []
                for bias in (0x200000000, 0x600000000):
                    for case, value in enumerate(values):
                        deterministic = bias == 0x200000000 and case < 2
                        registers = (
                            [id + 1 for id in range(31)]
                            if deterministic
                            else [rng.getrandbits(64) for _ in range(31)]
                        )
                        registers[control_register] = value
                        if pointer_register != 31:
                            registers[pointer_register] = 0x100000000 + 1024
                        state = SparseState(
                            tuple(registers),
                            0x90000000,
                            0x100000800,
                            bytes(4096)
                            if deterministic
                            else bytes(rng.getrandbits(8) for _ in range(4096)),
                            0x100000000,
                            bias,
                            (GlobalPage(bias + table_page, bytes(table)),),
                        )
                        states.append(state)
                        signed = value if value < 0x80000000 else value - 0x100000000
                        predicate = signed < 1 if control_register == 0 else bool(value & 1)
                        expected.append(
                            (
                                exits[0 if predicate else 1],
                                (15 if predicate else 16)
                                if control_register == 0
                                else (1 if predicate else 18),
                            )
                        )
                    faults.append(dataclasses.replace(states[-1], global_pages=()))
                before = self.oracle.run_many(original, entry, exits, states, trusted_fixture=True)
                after = self.oracle.run_many(direct, entry, exits, states, trusted_fixture=True)
                wrong = self.oracle.run_many(swapped, entry, exits, states, trusted_fixture=True)
                for case, (state, (destination, index), left, right, bad) in enumerate(
                    zip(states, expected, before, after, wrong, strict=True)
                ):
                    self.assertTrue(left.completed)
                    self.assertEqual(left.exit_pc, state.load_bias + destination)
                    self.assertEqual(left, right)
                    self.assertEqual(
                        bad.exit_pc, state.load_bias + exits[0 if destination == exits[1] else 1]
                    )
                    offset = 1032 if control_register == 0 else 2064
                    self.assertEqual(struct.unpack_from("<I", left.state.memory, offset)[0], index)
                    if case < 2:
                        arm = exits.index(destination)
                        self.assertEqual(left.state.registers[8], index)
                        self.assertEqual(left.state.registers[9], state.load_bias + table_address)
                        self.assertEqual(left.state.registers[10], left.exit_pc)
                        self.assertEqual(left.state.registers[11], table_entries[arm][1])
                        self.assertEqual(
                            left.state.nzcv,
                            0x60000000 if control_register == 25 and arm == 1 else 0x80000000,
                        )
                original_faults = self.oracle.run_many(
                    original, entry, exits, faults, trusted_fixture=True
                )
                direct_faults = self.oracle.run_many(
                    direct, entry, exits, faults, trusted_fixture=True
                )
                for left, right in zip(original_faults, direct_faults, strict=True):
                    self.assertEqual(left.signal, 11)
                    self.assertEqual(left, right)

    def test_sample_global_mba_retains_unknown_load(self):
        original = bytes.fromhex(
            "fd7bbfa9fd030091a83f00b0090986d28b00805208e544f949b2b0f2494ed1f2"
            "69bbe9f2e80308cb0a0109aa080109ca4af97fd3480108cb0a028052a90308aa"
            "a803088a2a6928f8ab0300f9fd7bc1a8c0035fd6"
        )
        self.assertEqual(
            hashlib.sha256(original).hexdigest(),
            "40f868f22b90b7f8e11d85c5a3329e106d3626bacac6a8a241874882b394e6a7",
        )
        # After NEG, 2*(x8 OR x9) - (x8 XOR x9) is x8+x9. The later
        # MOV W10 overwrites the intermediate x10 before any memory access.
        replacement = bytes.fromhex("2801088b1f2003d51f2003d51f2003d5")
        simplified = original[:40] + replacement + original[56:]
        exits = (0x152D970,)
        rng = random.Random(0x152D91C)
        cases = []
        for bias in (0x200000000, 0x600000000):
            for delta in (0, 16, 24, 32, -2048, 2064):
                registers = [rng.getrandbits(64) for _ in range(31)]
                registers[30] = bias + exits[0]
                page = bytearray(rng.getrandbits(8) for _ in range(4096))
                struct.pack_into("<Q", page, 0x9C8, (0x4DDB8A7285923048 - delta) & MASK)
                cases.append(
                    SparseState(
                        tuple(registers),
                        0x90000000,
                        0x100000800,
                        bytes(rng.getrandbits(8) for _ in range(4096)),
                        0x100000000,
                        bias,
                        (GlobalPage(bias + 0x1D22000, bytes(page)),),
                    )
                )
        before = self.oracle.run_many(
            (Fragment(0x152D91C, original),), 0x152D91C, exits, cases, trusted_fixture=True
        )
        after = self.oracle.run_many(
            (Fragment(0x152D91C, simplified),), 0x152D91C, exits, cases, trusted_fixture=True
        )
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            self.assertEqual(left, right)
            self.assertEqual(left.signal, 11 if index % 6 >= 4 else 0)
        wrong = original[:40] + bytes.fromhex("280108cb1f2003d51f2003d51f2003d5") + original[56:]
        changed = self.oracle.run(
            (Fragment(0x152D91C, wrong),), 0x152D91C, exits, cases[0], trusted_fixture=True
        )
        self.assertNotEqual(changed, before[0])

    def test_mixed_scratch_versions_upper_guard_and_response_bounds(self):
        fragment = Fragment(0x1000, bytes.fromhex("200040f9"))
        states = []
        for current, offset in ((fixture(), 32), (full_fixture(), 4096)):
            registers = list(current.registers)
            registers[1] = current.scratch_base + offset
            states.append(dataclasses.replace(current, registers=tuple(registers)))
        outcomes = self.oracle.run_many(
            (fragment,), 0x1000, (0x1004,), states, trusted_fixture=True
        )
        for state, offset, outcome in zip(states, (32, 4096), outcomes, strict=True):
            registers = list(state.registers)
            registers[0] = int.from_bytes(state.memory[offset : offset + 8], "little")
            self.assertEqual(outcome.state, dataclasses.replace(state, registers=tuple(registers)))
        state = states[1]
        registers = list(state.registers)
        registers[1] = state.scratch_base + 8192
        fault_state = dataclasses.replace(state, registers=tuple(registers))
        fault = self.oracle.run((fragment,), 0x1000, (0x1004,), fault_state, trusted_fixture=True)
        self.assertFalse(fault.completed)
        self.assertEqual(fault.fault_address, state.scratch_base + 8192)
        self.assertEqual(fault.state, fault_state)
        payload = (
            MAGIC_LARGE
            + struct.pack(
                "<40Q",
                *state.registers,
                state.nzcv,
                state.sp,
                state.scratch_base,
                state.load_bias,
                state.load_bias + 0x1004,
                0,
                0,
                len(state.global_pages),
                8192,
            )
            + state.memory
            + b"".join(struct.pack("<Q", page.address) + page.data for page in state.global_pages)
        )
        self.oracle._response(payload, state, (fragment,), (0x1004,))
        with self.assertRaises(OracleError):
            self.oracle._response(payload[:-1], state, (fragment,), (0x1004,))
        wrong = bytearray(payload)
        struct.pack_into("<Q", wrong, 8 + 39 * 8, 4096)
        with self.assertRaisesRegex(OracleError, "scratch length"):
            self.oracle._response(bytes(wrong), state, (fragment,), (0x1004,))

    def test_dispatch_guard_unsigned_bounds_and_second_table_page(self):
        states = [fixture(index=index, target=EXITS[1]) for index in (0, 647, 648, 0xFFFFFFFF)]
        outcomes = self.oracle.run_many(
            (DISPATCH,), DISPATCH.source_address, EXITS, states, trusted_fixture=True
        )
        for index, state, outcome in zip((0, 647, 648, 0xFFFFFFFF), states, outcomes, strict=True):
            registers = list(state.registers)
            registers[8] = index
            result = (index - 647) & 0xFFFFFFFF
            flags = ((result >> 31) << 31) | (int(result == 0) << 30) | (int(index >= 647) << 29)
            if index > 647:
                registers[9] = result
                target = state.load_bias + EXITS[2]
            else:
                registers[9] = state.load_bias + 0x1F8918
                registers[10] = state.load_bias + EXITS[1]
                registers[11] = (EXITS[1] - 0x752E60) & MASK
                target = registers[10]
            self.assertTrue(outcome.completed)
            self.assertEqual(outcome.exit_pc, target)
            self.assertEqual(
                outcome.state, dataclasses.replace(state, registers=tuple(registers), nzcv=flags)
            )
        altered = bytearray(DISPATCH.code)
        altered[8] ^= 1
        mutant = Fragment(DISPATCH.source_address, bytes(altered))
        wrong = self.oracle.run(
            (mutant,), DISPATCH.source_address, EXITS, states[0], trusted_fixture=True
        )
        self.assertNotEqual(wrong.exit_pc, outcomes[0].exit_pc)

    def test_data_and_sp_faults_preserve_observed_prefix(self):
        state = fixture()
        registers = list(state.registers)
        registers[19] = state.scratch_base - 4096
        state = dataclasses.replace(state, registers=tuple(registers))
        outcome = self.oracle.run(
            (TAIL, DISPATCH), TAIL.source_address, EXITS, state, trusted_fixture=True
        )
        registers[9] = 358
        self.assertFalse(outcome.completed)
        self.assertEqual(outcome.signal, 11)
        self.assertEqual(outcome.exit_pc, state.load_bias + TAIL.source_address + 4)
        self.assertEqual(outcome.fault_address, state.scratch_base - 4096 + 84)
        self.assertEqual(outcome.state, dataclasses.replace(state, registers=tuple(registers)))
        sp = dataclasses.replace(fixture(), sp=0x100000000 - 4096)
        fragment = Fragment(0x1000, bytes.fromhex("e00340f9"))
        fault = self.oracle.run((fragment,), 0x1000, (0x1004,), sp, trusted_fixture=True)
        self.assertFalse(fault.completed)
        self.assertEqual(fault.fault_address, sp.sp)
        self.assertEqual(fault.state, sp)

    def test_unlisted_trap_unknown_pc_system_and_bounds_refuse(self):
        state = fixture()
        # B+8 reaches a BRK-filled mapped word, but not the declared B+4 exit.
        fragment = Fragment(0x1000, bytes.fromhex("02000014"))
        with self.assertRaises(OracleError):
            self.oracle.run((fragment,), 0x1000, (0x1004,), state, trusted_fixture=True)
        with self.assertRaises(Declined):
            self.oracle.run((fragment,), 0x1000, (0x1004,), state)
        registers = list(state.registers)
        registers[3] = 0
        unknown = dataclasses.replace(state, registers=tuple(registers))
        with self.assertRaises(OracleError):
            self.oracle.run(
                (Fragment(0x1000, bytes.fromhex("60001fd6")),),
                0x1000,
                (0x1004,),
                unknown,
                trusted_fixture=True,
            )
        with self.assertRaises(Declined):
            self.oracle.run(
                (Fragment(0x1000, bytes.fromhex("010000d4")),),
                0x1000,
                (0x1004,),
                state,
                trusted_fixture=True,
            )
        with self.assertRaises(Declined):
            self.oracle.run_many((fragment,), 0x1000, (0x1004,), [], trusted_fixture=True)
        with self.assertRaises(Declined):
            self.oracle.run((fragment, fragment), 0x1000, (0x1004,), state, trusted_fixture=True)
        with self.assertRaises(OracleError):
            self.oracle._response(b"", state, (fragment,), (0x1004,))

    def test_nontermination_is_incomplete(self):
        oracle = SparseOracle(timeout=1)
        commands = []
        command = oracle.backend._command

        def record(args, **kwargs):
            commands.append(args)
            return command(args, **kwargs)

        oracle.backend._command = record
        fragment = Fragment(0x1000, bytes.fromhex("00000014"))
        with self.assertRaisesRegex(OracleError, "deadline"):
            oracle.run((fragment,), 0x1000, (0x1004,), fixture(), trusted_fixture=True)
        self.assertEqual(commands[-1][0], oracle.backend.qemu)

    def test_stack_pointer_is_restored_and_captured_from_the_live_context(self):
        # Every pinned fragment addresses through x19/x23 and leaves SP alone,
        # so nothing else here would notice a runner that echoed the input SP
        # instead of capturing the executed one.
        state = fixture()
        entry = state.scratch_base + 2048
        state = dataclasses.replace(state, sp=entry)
        spill = Fragment(0x760000, bytes.fromhex("e10f1ff8"))  # str x1, [sp, #-16]!
        outcome = self.oracle.run(
            (spill,), spill.source_address, (spill.source_address + 4,), state, trusted_fixture=True
        )
        self.assertTrue(outcome.completed)
        self.assertEqual(outcome.state.sp, entry - 16)
        self.assertNotEqual(outcome.state.sp, state.sp)
        stored = int.from_bytes(
            outcome.state.memory[entry - 16 - state.scratch_base :][:8], "little"
        )
        self.assertEqual(stored, state.registers[1])

    def test_global_input_pages_are_not_writable(self):
        # sparse.py advertises global pages as read-only snapshots; without this
        # the mapping could be made writable and nothing would fail.
        state = fixture()
        registers = list(state.registers)
        registers[0] = state.global_pages[0].address
        state = dataclasses.replace(state, registers=tuple(registers))
        store = Fragment(0x760000, bytes.fromhex("010000f9"))  # str x1, [x0]
        outcome = self.oracle.run(
            (store,), store.source_address, (store.source_address + 4,), state, trusted_fixture=True
        )
        self.assertFalse(outcome.completed)
        self.assertEqual(outcome.signal, 11)
        self.assertEqual(outcome.fault_address, state.global_pages[0].address)
        self.assertEqual(outcome.state.global_pages, state.global_pages)


if __name__ == "__main__":
    unittest.main()
