import dataclasses
import pathlib
import re
import struct
import tempfile

from tools.oracle.aarch64 import Declined, MASK64, Oracle, OracleError, State, validate
from tools.oracle.control import _family
from tools.oracle.memory import _MEMORY, _parse


MAGIC = b"NYXPRG02"
PAGE_SIZE = 4096
DEFAULT_SCRATCH = 0x100000000
DEFAULT_CODE = 0x200000000
MAX_WORDS = 1024
MAX_ADMISSION_STATES = 16384
MAX_GLOBAL_PAGES = 8


@dataclasses.dataclass(frozen=True)
class GlobalPage:
    address: int
    data: bytes
    # Only the sparse runner's NYXSPR03 protocol maps a page writable.
    writable: bool = False

    def __post_init__(self):
        if (
            type(self.address) is not int
            or self.address % PAGE_SIZE
            or not 0x100000000 <= self.address <= 0x1000000000
        ):
            raise ValueError("global page must be aligned in the admitted mapping range")
        if type(self.data) is not bytes or len(self.data) != PAGE_SIZE:
            raise ValueError("global input must contain exactly one immutable page")
        if type(self.writable) is not bool:
            raise ValueError("global page writability must be a bool")


def _global_regions(pages):
    if any(not isinstance(page, GlobalPage) for page in pages):
        raise ValueError("expected GlobalPage inputs")
    regions = []
    for address in sorted(page.address for page in pages):
        if regions and address < regions[-1][1]:
            raise ValueError("global pages overlap or repeat")
        if regions and address == regions[-1][1]:
            regions[-1] = (regions[-1][0], address + PAGE_SIZE)
        else:
            regions.append((address, address + PAGE_SIZE))
    return regions


@dataclasses.dataclass(frozen=True)
class ProgramState:
    registers: tuple[int, ...]
    nzcv: int = 0
    sp: int = DEFAULT_SCRATCH + PAGE_SIZE
    memory: bytes = bytes(PAGE_SIZE)
    scratch_base: int = DEFAULT_SCRATCH
    code_base: int = DEFAULT_CODE
    global_pages: tuple[GlobalPage, ...] = ()

    def __post_init__(self):
        if type(self.registers) is not tuple:
            raise ValueError("registers must be an immutable tuple")
        State(self.registers, self.nzcv)
        if type(self.sp) is not int or not 0 <= self.sp <= MASK64:
            raise ValueError("SP must be an unsigned 64-bit value")
        if type(self.memory) is not bytes or len(self.memory) != PAGE_SIZE:
            raise ValueError("memory must be exactly one immutable scratch page")
        if (
            type(self.scratch_base) is not int
            or self.scratch_base % PAGE_SIZE
            or not 0x100000000 <= self.scratch_base < 0x200000000
        ):
            raise ValueError("scratch base must be page-aligned in the scratch range")
        if (
            type(self.code_base) is not int
            or self.code_base % 4
            or not 0x200000000 <= self.code_base <= 0x1000000000
        ):
            raise ValueError("code entry must be instruction-aligned in the code range")
        if self.registers[28] != self.scratch_base:
            raise ValueError("x28 must equal the immutable scratch base")
        if (
            type(self.global_pages) is not tuple
            or len(self.global_pages) > MAX_GLOBAL_PAGES
            or any(isinstance(page, GlobalPage) and page.writable for page in self.global_pages)
        ):
            raise ValueError("global inputs must be a tuple of at most eight pages")
        code_page = self.code_base & ~(PAGE_SIZE - 1)
        # Reserve room for the maximum admitted code extent plus the exit word.
        arenas = [
            (self.scratch_base - PAGE_SIZE, self.scratch_base + 2 * PAGE_SIZE),
            (code_page - PAGE_SIZE, code_page + 3 * PAGE_SIZE),
        ]
        arenas += [
            (start - PAGE_SIZE, end + PAGE_SIZE)
            for start, end in _global_regions(self.global_pages)
        ]
        arenas.sort()
        if any(right[0] < left[1] for left, right in zip(arenas, arenas[1:])):
            raise ValueError("code, scratch or global guard arenas overlap")


@dataclasses.dataclass(frozen=True)
class ProgramOutcome:
    state: ProgramState
    exit_pc: int
    signal: int
    fault_address: int

    @property
    def completed(self):
        return self.signal == 0


@dataclasses.dataclass(frozen=True)
class _Instruction:
    kind: str
    offset: int = 0
    access: object = None
    writeback: bool = False
    destination: int | None = None
    source: int | None = None
    width: int = 64
    destinations: tuple[int, ...] = ()


def _instruction(word, text, offset):
    try:
        family, displacement, register = _family(word)
    except Declined:
        family = None
    if family is not None:
        if family in ("br", "blr", "ret"):
            if family == "br" or register != 30:
                raise Declined("fixture indirect control is restricted to BLR/RET x30")
            expected = "ret" if family == "ret" else "blr x30"
            if " ".join(text.split()) != expected:
                raise Declined("indirect transfer differs from independent disassembly")
        else:
            mnemonic = text.split()[0]
            if (family != "b.cond" and mnemonic != family) or (
                family == "b.cond"
                and not re.fullmatch(
                    r"b\.(?:eq|ne|cs|hs|cc|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le|al|nv)", mnemonic
                )
            ):
                raise Declined("direct control family differs from independent disassembly")
            target = re.search(r"\b0x([0-9a-f]{1,16})\Z", text)
            if target is None or int(target[1], 16) != (offset + displacement) & MASK64:
                raise Declined("direct target differs from independent disassembly")
        return _Instruction(family, displacement or 0)
    if word & 0x1F000000 == 0x10000000:
        reg = word & 31
        if reg == 28:
            raise Declined("fixture ADR cannot overwrite immutable x28")
        page = bool(word & (1 << 31))
        immediate = ((word >> 29) & 3) | (((word >> 5) & 0x7FFFF) << 2)
        if immediate & (1 << 20):
            immediate -= 1 << 21
        mnemonic = "adrp" if page else "adr"
        destination = "xzr" if reg == 31 else f"x{reg}"
        match = re.fullmatch(rf"{mnemonic}\s+{destination},\s*0x([0-9a-f]{{1,16}})", text)
        displacement = immediate << (12 if page else 0)
        expected = ((offset & ~4095) if page else offset) + displacement
        if not match or int(match[1], 16) != expected & MASK64:
            raise Declined("PC-relative target differs from independent disassembly")
        return _Instruction(mnemonic, displacement, destination=reg if reg != 31 else None)
    match = _MEMORY.fullmatch(text)
    if match:
        access = _parse(text)
        mnemonic, first, second, base, _, pre, post = match.groups()
        if base == "x28" and (pre or post):
            raise Declined("fixture memory cannot write back immutable x28")
        if mnemonic.startswith("ld") and any(
            reg and reg[1:] in ("28", "30") for reg in (first, second)
        ):
            raise Declined("fixture loads cannot replace immutable x28 or tracked LR")
        destinations = (
            tuple(int(reg[1:]) for reg in (first, second) if reg and reg[1:] != "zr")
            if mnemonic.startswith("ld")
            else ()
        )
        return _Instruction(
            "memory", access.offset, access, bool(pre or post), destinations=destinations
        )
    match = re.fullmatch(
        r"(add|sub)\s+sp,\s*sp,\s*#(0x[0-9a-f]+|[0-9]+)(?:,\s*lsl\s+#(0|12))?", text
    )
    if match:
        value = int(match[2], 0) << int(match[3] or "0")
        return _Instruction("sp", value if match[1] == "add" else -value)
    if " ".join(text.split()) == "mov sp, sp":
        return _Instruction("sp")
    match = re.fullmatch(
        r"(add|sub)\s+x([0-9]+),\s*sp,\s*#(0x[0-9a-f]+|[0-9]+)(?:,\s*lsl\s+#(0|12))?", text
    )
    if match:
        destination = int(match[2])
        if destination >= 31 or destination == 28:
            raise Declined("SP-derived pointer cannot overwrite immutable x28")
        value = int(match[3], 0) << int(match[4] or "0")
        return _Instruction(
            "sp_copy", value if match[1] == "add" else -value, destination=destination
        )
    match = re.fullmatch(r"mov\s+x([0-9]+),\s*sp", text)
    if match:
        destination = int(match[1])
        if destination >= 31 or destination == 28:
            raise Declined("SP copy cannot overwrite immutable x28")
        return _Instruction("sp_copy", destination=destination)
    validate([text])
    if re.search(r"\b[wx](?:28|30)\b", text):
        raise Declined("fixture integer operations cannot reference reserved x28 or LR")
    match = re.fullmatch(
        r"(add|sub|adds|subs)\s+([xw])([0-9]+),\s*([xw])([0-9]+),\s*"
        r"#(0x[0-9a-f]+|[0-9]+)(?:,\s*lsl\s+#(0|12))?",
        text,
    )
    if match and match[2] == match[4]:
        value = int(match[6], 0) << int(match[7] or "0")
        return _Instruction(
            "pointer_add",
            value if match[1].startswith("add") else -value,
            destination=int(match[3]),
            source=int(match[5]),
            width=64 if match[2] == "x" else 32,
        )
    mnemonic = text.split()[0]
    destination = None
    if mnemonic not in ("cmp", "cmn", "tst", "nop"):
        match = re.match(r"\S+\s+[xw]([0-9]+)\b", text)
        if match:
            destination = int(match[1])
    return _Instruction("integer", destination=destination)


def _confine(instructions, state):
    # Only seeded/PC-derived addresses and immediate offsets retain provenance.
    # Loaded data and general arithmetic results become unknown; this never
    # predicts memory values, branch conditions or call results for validation.
    pending = [(0, state.sp, state.registers)]
    arenas = [(state.scratch_base - PAGE_SIZE, state.scratch_base + 2 * PAGE_SIZE)]
    arenas += [
        (start - PAGE_SIZE, end + PAGE_SIZE) for start, end in _global_regions(state.global_pages)
    ]
    visited = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        if len(visited) >= MAX_ADMISSION_STATES:
            raise Declined("program confinement work budget exhausted")
        visited.add(current)
        index, sp, incoming = current
        if index == len(instructions):
            continue
        if not 0 <= index < len(instructions):
            raise Declined("control target outside fixture and designated exit")
        instruction = instructions[index]
        known = list(incoming)
        pc = state.code_base + index * 4
        successors = [index + 1]
        if instruction.kind in ("b", "bl", "b.cond", "cbz", "cbnz", "tbz", "tbnz"):
            target = index + instruction.offset // 4
            successors = [target] if instruction.kind in ("b", "bl") else [index + 1, target]
            if instruction.kind == "bl":
                known[30] = pc + 4
        elif instruction.kind in ("blr", "ret"):
            lr = known[30]
            if (
                lr is None
                or lr % 4
                or not state.code_base <= lr <= state.code_base + len(instructions) * 4
            ):
                raise Declined("indirect callee/return target outside controlled fixture")
            successors = [(lr - state.code_base) // 4]
            if instruction.kind == "blr":
                known[30] = pc + 4
        elif instruction.kind in ("adr", "adrp"):
            if instruction.destination is not None:
                known[instruction.destination] = (
                    ((pc & ~4095) if instruction.kind == "adrp" else pc) + instruction.offset
                ) & MASK64
        elif instruction.kind == "pointer_add":
            source = known[instruction.source]
            known[instruction.destination] = (
                None
                if source is None
                else (source + instruction.offset) & ((1 << instruction.width) - 1)
            )
        elif instruction.kind == "integer":
            if instruction.destination is not None:
                known[instruction.destination] = None
        elif instruction.kind == "sp":
            sp = (sp + instruction.offset) & MASK64
        elif instruction.kind == "sp_copy":
            known[instruction.destination] = (sp + instruction.offset) & MASK64
        elif instruction.kind == "memory":
            access = instruction.access
            base_register = None if access.base_register == "sp" else int(access.base_register[1:])
            base = sp if base_register is None else known[base_register]
            if base is None:
                raise Declined("memory address lost its admitted provenance")
            address = (base + (0 if access.post_indexed else access.offset)) & MASK64
            if not any(start <= address and address + access.size <= end for start, end in arenas):
                raise Declined("fixture access outside declared input arenas and faulting guards")
            for destination in instruction.destinations:
                known[destination] = None
            if instruction.writeback:
                if base_register is None:
                    sp = (base + instruction.offset) & MASK64
                else:
                    known[base_register] = (base + instruction.offset) & MASK64
        pending.extend((successor, sp, tuple(known)) for successor in successors)


class ProgramOracle:
    def __init__(self, **kwargs):
        self.backend = Oracle(**kwargs)

    def versions(self):
        result = self.backend.versions()
        result["protocol"] = MAGIC.decode()
        result["scope"] = (
            "trusted bounded original fixtures, real leaf callee/return, GPR/NZCV/SP, full scratch and explicit global pages"
        )
        result["execution_model"] = {
            "admission": "static PC/SP/LR/address provenance confinement; explicit trusted_fixture opt-in",
            "isolation": "not a general security sandbox or permission to execute arbitrary corpus code",
            "landing_pages": "RX without PROT_BTI; code and scratch disjoint; no self modification",
            "gcs": "not enabled by runner",
            "omitted_state": [
                "PSTATE.BTYPE",
                "FP/SIMD",
                "external events",
                "address tagging (TBI applies)",
            ],
            "timeout": "incomplete OracleError, never a successful outcome",
            "hardware_checked": False,
            "architectural_fault_authority": False,
            "global_pages": "read-only input snapshots; no lifetime invariance claim",
        }
        return result

    def _decode(self, code):
        if not isinstance(code, bytes) or not code or len(code) % 4 or len(code) > MAX_WORDS * 4:
            raise Declined("fixture requires 1 to 1024 complete original instruction words")
        with tempfile.TemporaryDirectory(prefix="nyx-program-decode-") as directory:
            path = pathlib.Path(directory) / "code.bin"
            path.write_bytes(code)
            output = self.backend._command(
                [self.backend.objdump, "-D", "-b", "binary", "-m", "aarch64", "-EL", str(path)]
            ).stdout.decode()
        instructions = []
        checked = []
        for line in output.splitlines():
            match = re.fullmatch(r"\s*([0-9a-f]+):\s+([0-9a-f]{8})\s+(.+)", line)
            if not match:
                continue
            offset = len(instructions) * 4
            word = int.from_bytes(code[offset : offset + 4], "little")
            if int(match[1], 16) != offset or int(match[2], 16) != word:
                raise Declined("program disassembly does not match original bytes")
            text = match[3].split("//", 1)[0].strip()
            instruction = _instruction(word, text, offset)
            # Control transfers and PC-relative forms are already checked against
            # their own immediates above; every other kind is read from this text
            # alone, including the address arithmetic _confine bounds.
            if instruction.kind in ("memory", "sp", "sp_copy", "pointer_add", "integer"):
                checked.append((text, word))
            instructions.append(instruction)
        if len(instructions) * 4 != len(code):
            raise Declined("program disassembly did not cover every original word")
        self.backend.reassembles(checked)
        return instructions

    @staticmethod
    def _response(data, original, code_size):
        header = 8 + 39 * 8
        if (
            len(data) != header + PAGE_SIZE + len(original.global_pages) * (8 + PAGE_SIZE)
            or data[:8] != MAGIC
        ):
            raise OracleError("invalid program response envelope")
        values = struct.unpack("<39Q", data[8:header])
        if values[33:35] != (original.scratch_base, original.code_base):
            raise OracleError("program response changed its mappings")
        exit_pc, signal, fault_address = values[35:38]
        if values[38] != len(original.global_pages):
            raise OracleError("program response changed global page count")
        if signal == 0:
            if exit_pc != original.code_base + code_size or fault_address:
                raise OracleError("program did not reach designated exit")
        elif (
            signal not in (7, 11)
            or exit_pc % 4
            or not original.code_base <= exit_pc < original.code_base + code_size
        ):
            raise OracleError("program fault is outside original instruction bytes")
        try:
            pages = []
            offset = header + PAGE_SIZE
            for original_page in original.global_pages:
                address = struct.unpack_from("<Q", data, offset)[0]
                if address != original_page.address:
                    raise OracleError("program response changed global page placement")
                pages.append(GlobalPage(address, data[offset + 8 : offset + 8 + PAGE_SIZE]))
                offset += 8 + PAGE_SIZE
            state = ProgramState(
                values[:31],
                values[31],
                values[32],
                data[header : header + PAGE_SIZE],
                values[33],
                values[34],
                tuple(pages),
            )
        except ValueError as error:
            raise OracleError("invalid program output state") from error
        return ProgramOutcome(state, exit_pc, signal, fault_address)

    def run_many(self, code, states, *, trusted_fixture=False):
        if trusted_fixture is not True:
            raise Declined("program oracle requires explicit trusted_fixture=True")
        instructions = self._decode(code)
        cases = []
        for state in states:
            if len(cases) == self.backend.max_cases:
                raise Declined("program case budget exhausted")
            if not isinstance(state, ProgramState):
                raise Declined("expected ProgramState")
            _confine(instructions, state)
            cases.append(state)
        if not cases:
            raise Declined("zero program cases: no evidence")
        with tempfile.TemporaryDirectory(prefix="nyx-program-oracle-") as directory:
            executable = pathlib.Path(directory) / "runner"
            self.backend._command(
                [
                    self.backend.compiler,
                    "-static",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-o",
                    str(executable),
                    str(pathlib.Path(__file__).with_name("runner_program.c")),
                ]
            )
            outcomes = []
            for state in cases:
                payload = (
                    MAGIC
                    + struct.pack(
                        "<37Q",
                        *state.registers,
                        state.nzcv,
                        state.sp,
                        state.scratch_base,
                        state.code_base,
                        len(code),
                        len(state.global_pages),
                    )
                    + code
                    + state.memory
                    + b"".join(
                        struct.pack("<Q", page.address) + page.data for page in state.global_pages
                    )
                )
                data = self.backend._command(
                    [self.backend.qemu, "-cpu", "max", str(executable)], input=payload
                ).stdout
                outcomes.append(self._response(data, state, len(code)))
            return outcomes

    def run(self, code, state, *, trusted_fixture=False):
        return self.run_many(code, [state], trusted_fixture=trusted_fixture)[0]
