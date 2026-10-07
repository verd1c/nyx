import dataclasses
import pathlib
import re
import struct
import tempfile

from tools.oracle.aarch64 import Declined, MASK64, Oracle, OracleError, State


PAGE_SIZE = 4096
DEFAULT_BASE = 0x100000000
MAGIC = b"NYXMEM02"
_REGISTER = r"(?:[xw](?:[0-9]|[12][0-9]|30)|[xw]zr)"
_ARITH_REGISTER = rf"(?:{_REGISTER}|sp|wsp)"
_BASE = r"(?:x(?:[0-9]|[12][0-9]|30)|sp)"
_IMMEDIATE = r"#[+-]?(?:0x[0-9a-f]+|[0-9]+)"
_MEMORY = re.compile(
    rf"(ldr|str|ldar|stlr|ldarb|stlrb|ldarh|stlrh|ldur|stur|ldrb|strb|ldrh|strh|ldurb|sturb|ldurh|sturh|ldrsb|ldrsh|ldrsw|ldursb|ldursh|ldursw|ldp|ldpsw|stp)\s+({_REGISTER}),\s*"
    rf"(?:({_REGISTER}),\s*)?\[({_BASE})(?:,\s*({_IMMEDIATE}))?\](!)?"
    rf"(?:,\s*({_IMMEDIATE}))?\Z"
)
_REGISTER_MEMORY = re.compile(
    rf"(ldr|str|ldrb|strb|ldrh|strh|ldrsb|ldrsh|ldrsw)\s+({_REGISTER}),\s*"
    rf"\[({_BASE}),\s*({_REGISTER})(?:,\s*(uxtw|sxtw|lsl|sxtx)(?:\s+#([0-3]))?)?\]\Z"
)

_PC_RELATIVE = re.compile(r"(adr|adrp)\s+(x(?:[0-9]|[12][0-9]|30)|xzr),\s*(0x[0-9a-f]{1,16})\Z")

_ARITHMETIC = re.compile(
    rf"(?:add|adds|sub|subs)\s+{_ARITH_REGISTER},\s*{_ARITH_REGISTER},\s*{_IMMEDIATE}"
    rf"(?:,\s*lsl\s+#12)?\Z|"
    rf"(?:cmp|cmn)\s+{_ARITH_REGISTER},\s*{_IMMEDIATE}(?:,\s*lsl\s+#12)?\Z|"
    rf"mov\s+(?:sp,\s*sp|wsp,\s*wsp|sp,\s*x(?:[0-9]|[12][0-9]|30)|x(?:[0-9]|[12][0-9]|30),\s*sp|"
    rf"wsp,\s*w(?:[0-9]|[12][0-9]|30)|w(?:[0-9]|[12][0-9]|30),\s*wsp)\Z"
)

_LOGICAL_SP = re.compile(
    rf"(?:and|orr|eor)\s+(?:sp|wsp),\s*{_REGISTER},\s*{_IMMEDIATE}\Z|"
    rf"mov\s+(?:sp|wsp),\s*{_IMMEDIATE}\Z"
)

_BITFIELD = re.compile(
    rf"(?:sbfm|ubfm|bfm|sbfx|ubfx|sbfiz|ubfiz|bfi|bfxil)\s+{_REGISTER},\s*{_REGISTER},\s*{_IMMEDIATE},\s*{_IMMEDIATE}\Z|"
    rf"bfc\s+{_REGISTER},\s*{_IMMEDIATE},\s*{_IMMEDIATE}\Z"
)

_VARIABLE_SHIFT = re.compile(
    rf"(?:lsl|lsr|asr|ror|lslv|lsrv|asrv|rorv)\s+{_REGISTER},\s*{_REGISTER},\s*{_REGISTER}\Z"
)


@dataclasses.dataclass(frozen=True)
class MemoryState:
    registers: tuple[int, ...]
    nzcv: int = 0
    sp: int = DEFAULT_BASE + PAGE_SIZE
    memory: bytes = bytes(PAGE_SIZE)
    base: int = DEFAULT_BASE

    def __post_init__(self):
        if type(self.registers) is not tuple:
            raise ValueError("registers must be an immutable tuple")
        State(self.registers, self.nzcv)
        if type(self.sp) is not int or not 0 <= self.sp <= MASK64:
            raise ValueError("SP must be an unsigned 64-bit value")
        if not isinstance(self.memory, bytes) or len(self.memory) != PAGE_SIZE:
            raise ValueError("scratch memory must contain exactly 4096 bytes")
        if (
            type(self.base) is not int
            or not DEFAULT_BASE <= self.base <= 0x1000000000
            or self.base % PAGE_SIZE
        ):
            raise ValueError("scratch base must be page-aligned in the admitted placement range")


@dataclasses.dataclass(frozen=True)
class MemoryOutcome:
    state: MemoryState
    signal: int
    fault_address: int
    pc_offset: int
    code_address: int

    @property
    def completed(self):
        return self.signal == 0


@dataclasses.dataclass(frozen=True)
class _Access:
    base_register: str
    offset: int
    size: int
    post_indexed: bool
    index_register: str | None = None
    extension: str = "lsl"
    shift: int = 0

    def effective(self, state):
        # The address this instruction accesses, derived from the disassembler's
        # text and the input state. Callers compare recorded access events
        # against it; the values read or written still come from execution.
        base = (
            state.sp if self.base_register == "sp" else state.registers[int(self.base_register[1:])]
        )
        offset = self.offset
        if self.index_register is not None:
            index = self.index_register
            offset = 0 if index[1:] == "zr" else state.registers[int(index[1:])]
            width = 32 if self.extension in ("uxtw", "sxtw") else 64
            offset &= (1 << width) - 1
            if self.extension in ("sxtw", "sxtx") and offset & (1 << (width - 1)):
                offset -= 1 << width
            offset <<= self.shift
        return (base + (0 if self.post_indexed else offset)) & MASK64

    def admit(self, state):
        address = self.effective(state)
        # This is an isolation check, not the semantic result. Include guard pages
        # so QEMU, rather than this calculation, determines fault and partial effects.
        if address < state.base - PAGE_SIZE or address + self.size > state.base + 2 * PAGE_SIZE:
            raise Declined("memory access is outside scratch and its unmapped guard pages")


def _parse(instruction):
    if (
        _ARITHMETIC.fullmatch(instruction)
        or _LOGICAL_SP.fullmatch(instruction)
        or _BITFIELD.fullmatch(instruction)
        or _VARIABLE_SHIFT.fullmatch(instruction)
    ):
        return None
    indexed = _REGISTER_MEMORY.fullmatch(instruction)
    if indexed:
        mnemonic, register, base, index, extension, shift = indexed.groups()
        access = _parse(f"{mnemonic} {register}, [{base}]")
        extension = extension or "lsl"
        shift = int(shift or 0)
        if index[0] != ("w" if extension in ("uxtw", "sxtw") else "x") or shift not in (
            0,
            access.size.bit_length() - 1,
        ):
            raise Declined("invalid register-offset extension or access scale")
        return dataclasses.replace(access, index_register=index, extension=extension, shift=shift)
    match = _MEMORY.fullmatch(instruction)
    if not match:
        raise Declined(
            "oracle admits immediate/register-offset scalar memory, immediate pairs, add/subtract and logical-immediate SP operations"
        )
    mnemonic, first, second, base, offset, pre, post = match.groups()
    pair = mnemonic in ("ldp", "ldpsw", "stp")
    if mnemonic in ("ldar", "stlr", "ldarb", "stlrb", "ldarh", "stlrh") and (
        offset or pre or post or second
    ):
        raise Declined("acquire/release transfers require a plain base address")
    if pair != (second is not None) or (pre and post) or (post and offset):
        raise Declined("invalid scalar/pair addressing form")
    if pair and first[0] != second[0]:
        raise Declined("mixed pair register widths")
    writeback = bool(pre or post)
    registers = [first] + ([second] if second else [])
    if writeback and any(register[1:] == base[1:] for register in registers):
        raise Declined("constrained-unpredictable base/writeback overlap")
    if mnemonic in ("ldp", "ldpsw") and first == second:
        raise Declined("constrained-unpredictable overlapping load destinations")
    widths = {
        "ldarb": 1,
        "stlrb": 1,
        "ldarh": 2,
        "stlrh": 2,
        "ldrb": 1,
        "strb": 1,
        "ldurb": 1,
        "sturb": 1,
        "ldrh": 2,
        "strh": 2,
        "ldurh": 2,
        "sturh": 2,
        "ldrsb": 1,
        "ldursb": 1,
        "ldrsh": 2,
        "ldursh": 2,
        "ldrsw": 4,
        "ldursw": 4,
        "ldpsw": 4,
    }
    width = widths.get(mnemonic, 8 if first.startswith("x") else 4)
    if mnemonic in (
        "ldarb",
        "stlrb",
        "ldarh",
        "stlrh",
        "ldrb",
        "strb",
        "ldurb",
        "sturb",
        "ldrh",
        "strh",
        "ldurh",
        "sturh",
    ) and not first.startswith("w"):
        raise Declined("byte and halfword unsigned transfers require W register")
    if mnemonic in ("ldrsw", "ldursw", "ldpsw") and not first.startswith("x"):
        raise Declined("signed word loads require X register")
    immediate = post or offset
    value = int(immediate[1:], 0) if immediate else 0
    return _Access(base, value, width * (2 if pair else 1), bool(post))


class MemoryOracle:
    def __init__(self, **kwargs):
        self.backend = Oracle(**kwargs)

    def versions(self):
        result = self.backend.versions()
        result["protocol"] = MAGIC.decode()
        result["scope"] = (
            "single original immediate/register-offset scalar, base-only LDAR/STLR including byte/halfword forms or immediate pair memory, ADR/ADRP, bitfield, variable shift, immediate add/subtract or logical-immediate SP instruction; GPR/NZCV/SP; scratch + guards"
        )
        result["execution_model"] = {
            "admission": "one word, independently disassembled and exactly reassembled",
            "isolation": "not a security sandbox or permission to execute arbitrary corpus code",
            "timeout": "incomplete OracleError, never a successful outcome",
            "hardware_checked": False,
            "architectural_fault_authority": False,
            "sp_alignment": "QEMU omits the stack-alignment check; fixtures align SP explicitly",
            "omitted_state": [
                "FP/SIMD",
                "PSTATE.BTYPE",
                "GCS",
                "external events",
                "address tagging (TBI applies)",
            ],
        }
        return result

    def _decode(self, code):
        if not isinstance(code, bytes) or len(code) != 4:
            raise Declined("memory oracle requires exactly one original instruction word")
        with tempfile.TemporaryDirectory(prefix="nyx-memory-decode-") as directory:
            path = pathlib.Path(directory)
            (path / "word.bin").write_bytes(code)
            output = self.backend._command(
                [
                    self.backend.objdump,
                    "-D",
                    "-b",
                    "binary",
                    "-m",
                    "aarch64",
                    "-EL",
                    str(path / "word.bin"),
                ]
            ).stdout.decode()
            matches = re.findall(r"^\s*0:\s+([0-9a-f]{8})\s+(.+)$", output, re.MULTILINE)
            if len(matches) != 1 or int(matches[0][0], 16) != int.from_bytes(code, "little"):
                raise Declined("independent disassembly did not cover original word")
            instruction = matches[0][1].split("//", 1)[0].strip()
            pc_relative = _PC_RELATIVE.fullmatch(instruction)
            if pc_relative:
                mnemonic, destination, target = pc_relative.groups()
                word = int.from_bytes(code, "little")
                family = 0x90000000 if mnemonic == "adrp" else 0x10000000
                register = 31 if destination == "xzr" else int(destination[1:])
                immediate = ((word >> 29) & 3) | (((word >> 5) & 0x7FFFF) << 2)
                if immediate & (1 << 20):
                    immediate -= 1 << 21
                expected = (immediate << (12 if mnemonic == "adrp" else 0)) & MASK64
                if (
                    word & 0x9F000000 != family
                    or word & 31 != register
                    or int(target, 16) != expected
                ):
                    raise Declined(
                        "PC-relative original word disagrees with independent disassembly"
                    )
                # Objdump used PC zero. Assembling its absolute target may create
                # an unresolved relocation; run the original checked word instead.
                return None
            access = _parse(instruction)
            # Only a fully parsed admitted instruction reaches the assembler.
            (path / "word.S").write_text(".text\n" + instruction + "\n")
            self.backend._command(
                [self.backend.compiler, "-c", "-o", str(path / "word.o"), str(path / "word.S")]
            )
            self.backend._command(
                [
                    self.backend.objcopy,
                    "-O",
                    "binary",
                    "-j",
                    ".text",
                    str(path / "word.o"),
                    str(path / "roundtrip.bin"),
                ]
            )
            if (path / "roundtrip.bin").read_bytes() != code:
                raise Declined("memory instruction reassembly differs from original bytes")
            return access

    def run_many(self, code, states):
        access = self._decode(code)
        cases = []
        for state in states:
            if len(cases) == self.backend.max_cases:
                raise Declined("memory oracle case budget exhausted")
            if not isinstance(state, MemoryState):
                raise Declined("expected MemoryState values")
            if access is not None:
                access.admit(state)
            cases.append(state)
        if not cases:
            raise Declined("zero memory oracle cases: no evidence")
        with tempfile.TemporaryDirectory(prefix="nyx-memory-oracle-") as directory:
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
                    str(pathlib.Path(__file__).with_name("runner_memory.c")),
                ]
            )
            results = []
            for state in cases:
                payload = (
                    MAGIC
                    + struct.pack(
                        "<35Q",
                        *state.registers,
                        state.nzcv,
                        state.sp,
                        state.base,
                        int.from_bytes(code, "little"),
                    )
                    + state.memory
                )
                data = self.backend._command(
                    [self.backend.qemu, "-cpu", "max", str(executable)], input=payload
                ).stdout
                results.append(self._response(data, state.base))
            return results

    @staticmethod
    def _response(data, base):
        if len(data) != 8 + 37 * 8 + PAGE_SIZE or data[:8] != MAGIC:
            raise OracleError("invalid memory oracle response envelope")
        values = struct.unpack("<37Q", data[8 : 8 + 37 * 8])
        if (
            (values[34] == 0 and (values[33] != 4 or values[35] != 0))
            or (values[34] in (7, 11) and values[33] != 0)
            or values[34] not in (0, 7, 11)
        ):
            raise OracleError("invalid memory oracle exit/fault outcome")
        if (
            values[36] == 0
            or values[36] % PAGE_SIZE
            or values[36] > MASK64 - PAGE_SIZE
            or values[36] < base + 2 * PAGE_SIZE
            and values[36] + PAGE_SIZE > base - PAGE_SIZE
        ):
            raise OracleError("invalid independent code mapping address")
        try:
            state = MemoryState(values[:31], values[31], values[32], data[8 + 37 * 8 :], base)
        except ValueError as error:
            raise OracleError("invalid memory oracle response state") from error
        return MemoryOutcome(state, values[34], values[35], values[33], values[36])

    def run(self, code, state):
        return self.run_many(code, [state])[0]
