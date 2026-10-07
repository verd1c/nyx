import dataclasses
import pathlib
import re
import struct
import tempfile

from tools.oracle.aarch64 import Declined, MASK64, Oracle, OracleError, State


MAGIC = b"NYXCTL01"
PAGE_SIZE = 4096
DEFAULT_PC = 0x200001000


@dataclasses.dataclass(frozen=True)
class ControlState:
    registers: tuple[int, ...]
    nzcv: int = 0
    sp: int = 0
    # Fixed placement of the original instruction; the observed exit PC lives in
    # ControlOutcome.landing_pc, including when state contains exit registers.
    pc: int = DEFAULT_PC

    def __post_init__(self):
        if type(self.registers) is not tuple:
            raise ValueError("registers must be an immutable tuple")
        State(self.registers, self.nzcv)
        if type(self.sp) is not int or not 0 <= self.sp <= MASK64:
            raise ValueError("SP must be an unsigned 64-bit value")
        if (
            type(self.pc) is not int
            or not 0x200000000 <= self.pc <= 0x1000000000
            or self.pc % PAGE_SIZE
        ):
            raise ValueError("PC must be page-aligned in the admitted placement range")


@dataclasses.dataclass(frozen=True)
class ControlOutcome:
    state: ControlState
    landing_pc: int


def _signed(value, width):
    return value - (1 << width) if value & (1 << (width - 1)) else value


def _family(word):
    if word & 0x7C000000 == 0x14000000:
        return ("bl" if word >> 31 else "b"), _signed(word & 0x3FFFFFF, 26) * 4, None
    if word & 0xFF000010 == 0x54000000:
        return "b.cond", _signed((word >> 5) & 0x7FFFF, 19) * 4, None
    if word & 0x7E000000 == 0x34000000:
        return ("cbnz" if word & (1 << 24) else "cbz"), _signed((word >> 5) & 0x7FFFF, 19) * 4, None
    if word & 0x7E000000 == 0x36000000:
        return ("tbnz" if word & (1 << 24) else "tbz"), _signed((word >> 5) & 0x3FFF, 14) * 4, None
    for family, encoding in (("br", 0xD61F0000), ("blr", 0xD63F0000), ("ret", 0xD65F0000)):
        if word & 0xFFFFFC1F == encoding:
            reg = (word >> 5) & 31
            if reg == 31:
                raise Declined("zero-register indirect target is outside the landing arena")
            return family, None, reg
    raise Declined("unsupported control instruction")


def _landing(state, target):
    return (
        state.pc - PAGE_SIZE <= target < state.pc + 2 * PAGE_SIZE
        and target != state.pc
        and target % 4 == 0
    )


class ControlOracle:
    def __init__(self, **kwargs):
        self.backend = Oracle(**kwargs)

    def versions(self):
        result = self.backend.versions()
        result["protocol"] = MAGIC.decode()
        result["scope"] = (
            "single original control transfer to mapped trap landing; GPR/NZCV/SP/LR; no callee body or returning call"
        )
        result["execution_model"] = {
            "landing_pages": "RX without PROT_BTI",
            "gcs": "not enabled by the runner",
            "omitted_state": ["PSTATE.BTYPE", "GCS", "FP/SIMD", "memory events"],
            "hardware_checked": False,
        }
        return result

    def _admit(self, code):
        if not isinstance(code, bytes) or len(code) != 4:
            raise Declined("control oracle requires one original instruction word")
        word = int.from_bytes(code, "little")
        family, offset, register = _family(word)
        with tempfile.TemporaryDirectory(prefix="nyx-control-decode-") as directory:
            path = pathlib.Path(directory) / "word.bin"
            path.write_bytes(code)
            output = self.backend._command(
                [self.backend.objdump, "-D", "-b", "binary", "-m", "aarch64", "-EL", str(path)]
            ).stdout.decode()
        matches = re.findall(r"^\s*0:\s+([0-9a-f]{8})\s+(.+)$", output, re.MULTILINE)
        if len(matches) != 1 or int(matches[0][0], 16) != word:
            raise Declined("independent disassembly does not cover original control word")
        instruction = matches[0][1].split("//", 1)[0].strip()
        mnemonic = instruction.split()[0]
        if family == "b.cond":
            if not re.fullmatch(
                r"b\.(?:eq|ne|cs|hs|cc|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le|al|nv)", mnemonic
            ):
                raise Declined("conditional branch disagrees with independent disassembly")
        elif mnemonic != family:
            raise Declined("control family disagrees with independent disassembly")
        if offset is not None:
            target = re.search(r"\b0x([0-9a-f]{1,16})\Z", instruction)
            if target is None or int(target[1], 16) != offset & MASK64:
                raise Declined("direct target disagrees with independent disassembly")
        else:
            expected = "ret" if family == "ret" and register == 30 else f"{family} x{register}"
            if " ".join(instruction.split()) != expected:
                raise Declined("indirect register disagrees with independent disassembly")
        return offset, register

    @staticmethod
    def _response(data, original):
        if len(data) != 8 + 35 * 8 or data[:8] != MAGIC:
            raise OracleError("invalid control oracle response envelope")
        values = struct.unpack("<35Q", data[8:])
        if values[33] != original.pc or not _landing(original, values[34]):
            raise OracleError("control response is not a mapped landing")
        try:
            state = ControlState(values[:31], values[31], values[32], values[33])
        except ValueError as error:
            raise OracleError("invalid control response state") from error
        return ControlOutcome(state, values[34])

    def run_many(self, code, states):
        offset, register = self._admit(code)
        cases = []
        for state in states:
            if len(cases) == self.backend.max_cases:
                raise Declined("control oracle case budget exhausted")
            if not isinstance(state, ControlState):
                raise Declined("expected ControlState values")
            target = (state.pc + offset) & MASK64 if register is None else state.registers[register]
            # This checks only isolation. QEMU independently decides which landing
            # is reached; no condition is evaluated or expected result manufactured.
            if not _landing(state, target):
                raise Declined("control target is outside mapped landings or loops to the source")
            cases.append(state)
        if not cases:
            raise Declined("zero control cases: no evidence")
        with tempfile.TemporaryDirectory(prefix="nyx-control-oracle-") as directory:
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
                    str(pathlib.Path(__file__).with_name("runner_control.c")),
                ]
            )
            outcomes = []
            for state in cases:
                payload = MAGIC + struct.pack(
                    "<35Q",
                    *state.registers,
                    state.nzcv,
                    state.sp,
                    state.pc,
                    int.from_bytes(code, "little"),
                )
                data = self.backend._command(
                    [self.backend.qemu, "-cpu", "max", str(executable)], input=payload
                ).stdout
                outcomes.append(self._response(data, state))
            return outcomes

    def run(self, code, state):
        return self.run_many(code, [state])[0]
