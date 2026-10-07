import dataclasses
import math
import pathlib
import re
import shutil
import struct
import subprocess
import tempfile


MAGIC = b"NYXORCL1"
MASK64 = (1 << 64) - 1
NZCV_MASK = 0xF0000000
# This boundary deliberately excludes memory, SP, PC-relative values, control flow,
# FP/SIMD and system instructions other than NZCV transfers. A register-only block cannot reach
# the runner's syscalls or overwrite its context. Assembly is never accepted raw.
MNEMONICS = frozenset(
    """
    add adds sub subs adc adcs sbc sbcs neg negs ngc ngcs cmp cmn
    and ands orr orn eor eon bic bics tst mov movz movn movk mvn
    lsl lsr asr ror lslv lsrv asrv rorv extr clz cls rbit rev rev16 rev32
    ubfm sbfm bfm ubfx sbfx ubfiz sbfiz bfi bfxil bfc uxtb uxth sxtb sxth sxtw
    csel csinc csinv csneg cset csetm cinc cinv cneg ccmp ccmn
    mul mneg madd msub smull umull smaddl smsubl umaddl umsubl smulh umulh sdiv udiv nop
""".split()
)
TOKEN = re.compile(
    r"(?:[xw](?:[0-9]|[12][0-9]|30)|[xw]zr|"
    r"#[+-]?(?:0x[0-9a-f]+|[0-9]+)|"
    r"lsl|lsr|asr|ror|uxtb|uxth|uxtw|uxtx|sxtb|sxth|sxtw|sxtx|"
    r"eq|ne|cs|hs|cc|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le|al|nv)\Z"
)


class Declined(ValueError):
    pass


class OracleError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class State:
    registers: tuple[int, ...]
    nzcv: int = 0

    def __post_init__(self):
        if len(self.registers) != 31 or any(
            type(x) is not int or not 0 <= x <= MASK64 for x in self.registers
        ):
            raise ValueError("expected 31 unsigned 64-bit registers (x0 through x30)")
        if type(self.nzcv) is not int or self.nzcv & ~NZCV_MASK:
            raise ValueError("NZCV contains bits outside the four architectural flags")

    def encode(self):
        return MAGIC + struct.pack("<32Q", *self.registers, self.nzcv)

    @classmethod
    def decode(cls, data):
        if len(data) != 264 or data[:8] != MAGIC:
            raise OracleError("invalid oracle response header or length")
        values = struct.unpack("<32Q", data[8:])
        try:
            return cls(values[:31], values[31])
        except ValueError as error:
            raise OracleError("invalid oracle response state") from error


def validate(instructions):
    if not isinstance(instructions, (tuple, list)) or len(instructions) > 4096:
        raise Declined("expected a list of at most 4096 straight-line instructions")
    result = []
    for line in instructions:
        if not isinstance(line, str) or len(line) > 200:
            raise Declined("invalid instruction text")
        line = line.strip().lower()
        if not line or any(c in line for c in "\n\r;:/\\[]{}\x00"):
            raise Declined("only one register instruction is allowed per entry")
        parts = line.split(None, 1)
        if parts[0] in ("mrs", "msr"):
            reg = r"x(?:[0-9]|[12][0-9]|30|zr)"
            pattern = rf"{reg}\s*,\s*nzcv" if parts[0] == "mrs" else rf"nzcv\s*,\s*{reg}"
            if len(parts) != 2 or not re.fullmatch(pattern, parts[1]):
                raise Declined("only NZCV transfers with an X register are admitted")
            result.append(line)
            continue
        if parts[0] not in MNEMONICS:
            raise Declined(f"unsupported instruction: {parts[0]}")
        if len(parts) > 1:
            tokens = re.split(r"[, \t]+", parts[1])
            if any(not TOKEN.fullmatch(token) for token in tokens):
                raise Declined("unsupported operand; SP, memory and symbols are excluded")
        result.append(line)
    return result


def assembly(instructions):
    return _assembly_validated(validate(instructions))


def _assembly_validated(instructions):
    # SP stays private to the harness. Saving every guest register before using a
    # scratch register means x30 is observable too, with no reserved guest GPR.
    lines = [
        ".text",
        ".global nyx_execute",
        ".type nyx_execute, %function",
        "nyx_execute:",
        "sub sp, sp, #384",
        "str x1, [sp, #256]",
    ]
    for index in range(19, 31, 2):
        lines.append(f"stp x{index}, x{index + 1}, [sp, #{272 + (index - 19) * 8}]")
    lines += ["mov x30, x0", "ldr x0, [x30, #248]", "msr nzcv, x0"]
    for index in range(0, 30, 2):
        lines.append(f"ldp x{index}, x{index + 1}, [x30, #{index * 8}]")
    lines += ["ldr x30, [x30, #240]", *instructions]
    for index in range(0, 30, 2):
        lines.append(f"stp x{index}, x{index + 1}, [sp, #{index * 8}]")
    lines += ["str x30, [sp, #240]", "mrs x0, nzcv", "str x0, [sp, #248]", "ldr x1, [sp, #256]"]
    for offset in range(0, 256, 16):
        lines += [f"ldp x2, x3, [sp, #{offset}]", f"stp x2, x3, [x1, #{offset}]"]
    for index in range(19, 31, 2):
        lines.append(f"ldp x{index}, x{index + 1}, [sp, #{272 + (index - 19) * 8}]")
    lines += [
        "add sp, sp, #384",
        "ret",
        ".size nyx_execute, .-nyx_execute",
        '.section .note.GNU-stack,"",%progbits',
        "",
    ]
    return "\n".join(lines)


class Oracle:
    def __init__(
        self,
        compiler="aarch64-linux-gnu-gcc",
        qemu="qemu-aarch64",
        timeout=5,
        objcopy="aarch64-linux-gnu-objcopy",
        objdump="aarch64-linux-gnu-objdump",
        max_cases=4096,
    ):
        self.compiler = shutil.which(compiler)
        self.qemu = shutil.which(qemu)
        self.objcopy = shutil.which(objcopy)
        self.objdump = shutil.which(objdump)
        if not self.compiler or not self.qemu or not self.objcopy or not self.objdump:
            raise OracleError("AArch64 GCC, objcopy, objdump and qemu-aarch64 are required")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if type(max_cases) is not int or max_cases <= 0:
            raise ValueError("max_cases must be a positive integer")
        self.timeout = timeout
        self.max_cases = max_cases

    def _command(self, args, **kwargs):
        try:
            result = subprocess.run(args, capture_output=True, timeout=self.timeout, **kwargs)
        except subprocess.TimeoutExpired as error:
            raise OracleError("oracle process exceeded its deadline") from error
        except OSError as error:
            raise OracleError("oracle process could not start") from error
        if result.returncode:
            raise OracleError(
                f"oracle process failed ({result.returncode}): "
                + result.stderr.decode(errors="replace")[:4096]
            )
        return result

    def versions(self):
        return {
            "compiler": self._command([self.compiler, "--version"]).stdout.decode().splitlines()[0],
            "qemu": self._command([self.qemu, "--version"]).stdout.decode().splitlines()[0],
            "objcopy": self._command([self.objcopy, "--version"]).stdout.decode().splitlines()[0],
            "objdump": self._command([self.objdump, "--version"]).stdout.decode().splitlines()[0],
            "cpu": "max",
            "protocol": MAGIC.decode(),
            "scope": "x0-x30,NZCV; register-only straight-line; no guest SP/memory/FP/SIMD",
        }

    def assemble(self, instructions):
        lines = validate(instructions)
        with tempfile.TemporaryDirectory(prefix="nyx-oracle-assemble-") as directory:
            path = pathlib.Path(directory)
            (path / "snippet.S").write_text(".text\n" + "\n".join(lines) + "\n")
            self._command(
                [self.compiler, "-c", "-o", str(path / "snippet.o"), str(path / "snippet.S")]
            )
            self._command(
                [
                    self.objcopy,
                    "-O",
                    "binary",
                    "-j",
                    ".text",
                    str(path / "snippet.o"),
                    str(path / "snippet.bin"),
                ]
            )
            data = (path / "snippet.bin").read_bytes()
            if len(data) != 4 * len(lines):
                raise OracleError("assembler did not produce one word per instruction")
            return data

    def _payloads(self, states):
        payloads = []
        for state in states:
            if len(payloads) == self.max_cases:
                raise Declined("oracle case budget exhausted")
            if not isinstance(state, State):
                raise Declined("oracle input must contain State values")
            payloads.append(state.encode())
        if not payloads:
            raise Declined("zero oracle cases: no evidence")
        return payloads

    def instructions_from_bytes(self, code):
        if not isinstance(code, bytes) or not code or len(code) % 4 or len(code) > 4096 * 4:
            raise Declined("expected 1 to 4096 complete little-endian instruction words")
        with tempfile.TemporaryDirectory(prefix="nyx-oracle-decode-") as directory:
            path = pathlib.Path(directory) / "snippet.bin"
            path.write_bytes(code)
            output = self._command(
                [self.objdump, "-D", "-b", "binary", "-m", "aarch64", "-EL", str(path)]
            ).stdout.decode()
        instructions = []
        for line in output.splitlines():
            match = re.fullmatch(r"\s*([0-9a-f]+):\s+([0-9a-f]{8})\s+(.+)", line)
            if not match:
                continue
            offset = len(instructions) * 4
            if int(match[1], 16) != offset or int(match[2], 16) != int.from_bytes(
                code[offset : offset + 4], "little"
            ):
                raise Declined("disassembly does not match original instruction bytes")
            instructions.append(match[3].split("//", 1)[0].strip())
        if len(instructions) * 4 != len(code):
            raise Declined("disassembly did not cover every original instruction")
        instructions = validate(instructions)
        if self.assemble(instructions) != code:
            raise Declined("disassembly reassembly differs from original instruction bytes")
        return instructions

    def reassembles(self, pairs):
        # The admission logic in the program and sparse oracles reads an
        # instruction from the disassembler's text, so confirm the assembler
        # turns that text back into the original word. Only position-independent
        # forms belong here: a PC-relative encoding assembles to a relocation
        # rather than a resolved displacement, and is cross-checked against its
        # own immediate by the caller instead.
        if not pairs:
            return
        try:
            with tempfile.TemporaryDirectory(prefix="nyx-oracle-reassemble-") as directory:
                path = pathlib.Path(directory)
                (path / "check.S").write_text(".text\n" + "\n".join(t for t, _ in pairs) + "\n")
                self._command(
                    [self.compiler, "-c", "-o", str(path / "check.o"), str(path / "check.S")]
                )
                self._command(
                    [
                        self.objcopy,
                        "-O",
                        "binary",
                        "-j",
                        ".text",
                        str(path / "check.o"),
                        str(path / "check.bin"),
                    ]
                )
                data = (path / "check.bin").read_bytes()
        except OracleError as error:
            raise Declined("disassembly is not assemblable text") from error
        if len(data) != 4 * len(pairs):
            raise Declined("reassembly did not produce one word per instruction")
        for index, (_, word) in enumerate(pairs):
            if int.from_bytes(data[index * 4 : index * 4 + 4], "little") != word:
                raise Declined("disassembly reassembly differs from original instruction bytes")

    def run_bytes_many(self, code, states):
        payloads = self._payloads(states)
        self.instructions_from_bytes(code)
        # Only words admitted by the independent disassembler and exact reassembly
        # check reach .inst. Execute their original encoding, not an alias spelling.
        words = [
            f".inst 0x{int.from_bytes(code[i : i + 4], 'little'):08x}"
            for i in range(0, len(code), 4)
        ]
        return self._run_source(_assembly_validated(words), payloads)

    def run_bytes(self, code, state):
        return self.run_bytes_many(code, [state])[0]

    def run_many(self, instructions, states):
        payloads = self._payloads(states)
        # Reuse assemble's one-word-per-instruction check, so a line the
        # assembler expands or drops cannot reach .text unnoticed.
        self.assemble(instructions)
        return self._run_source(assembly(instructions), payloads)

    def _run_source(self, source, payloads):
        with tempfile.TemporaryDirectory(prefix="nyx-oracle-") as directory:
            path = pathlib.Path(directory)
            (path / "snippet.S").write_text(source)
            executable = path / "runner"
            self._command(
                [
                    self.compiler,
                    "-static",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-o",
                    str(executable),
                    str(path / "snippet.S"),
                    str(pathlib.Path(__file__).with_name("runner.c")),
                ]
            )
            return [
                State.decode(
                    self._command([self.qemu, "-cpu", "max", str(executable)], input=payload).stdout
                )
                for payload in payloads
            ]

    def run(self, instructions, state):
        return self.run_many(instructions, [state])[0]
