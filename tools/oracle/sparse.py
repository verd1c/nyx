import dataclasses
import pathlib
import re
import struct
import tempfile

from tools.oracle.aarch64 import Declined, MASK64, Oracle, OracleError, State, validate
from tools.oracle.control import _family
from tools.oracle.memory import _parse
from tools.oracle.program import GlobalPage, _global_regions

MAGIC = b"NYXSPR01"
MAGIC_LARGE = b"NYXSPR02"
# NYXSPR02 plus a bitmask of global pages mapped writable, in input order.
MAGIC_WRITABLE = b"NYXSPR03"
PAGE = 4096


@dataclasses.dataclass(frozen=True)
class Fragment:
    source_address: int
    code: bytes

    def __post_init__(self):
        if (
            type(self.source_address) is not int
            or not 0 <= self.source_address <= MASK64
            or self.source_address % 4
            or type(self.code) is not bytes
            or not self.code
            or len(self.code) > PAGE
            or len(self.code) % 4
            or self.source_address > MASK64 - len(self.code)
        ):
            raise ValueError("invalid bounded original code fragment")


@dataclasses.dataclass(frozen=True)
class SparseState:
    registers: tuple[int, ...]
    nzcv: int = 0
    sp: int = 0x100000800
    memory: bytes = bytes(PAGE)
    scratch_base: int = 0x100000000
    load_bias: int = 0x200000000
    global_pages: tuple[GlobalPage, ...] = ()

    def __post_init__(self):
        if type(self.registers) is not tuple:
            raise ValueError("registers must be immutable")
        State(self.registers, self.nzcv)
        if any(
            type(value) is not int or not 0 <= value <= MASK64
            for value in (self.sp, self.load_bias)
        ):
            raise ValueError("SP and bias must be unsigned64")
        if (
            type(self.scratch_base) is not int
            or not 0x100000000 <= self.scratch_base <= 0x1000000000
            or self.scratch_base % PAGE
            or type(self.memory) is not bytes
            or len(self.memory) not in (PAGE, 2 * PAGE)
        ):
            raise ValueError("invalid scratch mapping")
        if type(self.global_pages) is not tuple or len(self.global_pages) > 8:
            raise ValueError("at most eight readonly global pages")
        _global_regions(self.global_pages)
        if _writable(self) and len(self.memory) != 2 * PAGE:
            raise ValueError("writable global pages need the two-page scratch protocol")


@dataclasses.dataclass(frozen=True)
class SparseOutcome:
    state: SparseState
    exit_pc: int
    signal: int
    fault_address: int

    @property
    def completed(self):
        return self.signal == 0


def _regions(pages):
    # Pages one apart share their guard page, so they join one arena; the runner
    # leaves the gap between two data pages unmapped and fills a code gap with BRK.
    regions = []
    for page in sorted(pages):
        if regions and page <= regions[-1][1] + PAGE:
            regions[-1] = (regions[-1][0], page + PAGE)
        else:
            regions.append((page, page + PAGE))
    return regions


def _writable(state):
    return sum(1 << index for index, page in enumerate(state.global_pages) if page.writable)


def _layout(fragments, entry, exits, state):
    spans = sorted(
        ((fragment.source_address + state.load_bias) & MASK64, len(fragment.code))
        for fragment in fragments
    )
    pages = set()
    for index, (start, size) in enumerate(spans):
        if (
            not 0x200000000 <= start <= 0x1000000000
            or start % 4
            or start > MASK64 - size
            or index
            and start < spans[index - 1][0] + spans[index - 1][1]
        ):
            raise Declined("invalid or overlapping runtime fragments")
        pages.update(range(start & ~(PAGE - 1), (start + size - 1 & ~(PAGE - 1)) + PAGE, PAGE))

    def contains(pc):
        return any(start <= pc < start + size for start, size in spans)

    if not contains((entry + state.load_bias) & MASK64):
        raise Declined("entry outside original code")
    for exit_address in exits:
        pc = (exit_address + state.load_bias) & MASK64
        if pc % 4 or not 0x200000000 <= pc <= 0x1000000000 or contains(pc):
            raise Declined("exit must be separate aligned controlled landing")
        pages.add(pc & ~(PAGE - 1))
    if len(pages) > 16:
        raise Declined("code page budget exceeded")
    arenas = [(state.scratch_base - PAGE, state.scratch_base + len(state.memory) + PAGE)]
    arenas += [(start - PAGE, end + PAGE) for start, end in _regions(pages)]
    arenas += [
        (start - PAGE, end + PAGE)
        for start, end in _regions(page.address for page in state.global_pages)
    ]
    arenas.sort()
    if any(right[0] < left[1] for left, right in zip(arenas, arenas[1:])):
        raise Declined("code, scratch or global guarded mappings overlap")


class SparseOracle:
    def __init__(self, **kwargs):
        self.backend = Oracle(**kwargs)

    def versions(self):
        result = self.backend.versions()
        result["protocol"] = MAGIC.decode()
        result["protocols"] = [MAGIC.decode(), MAGIC_LARGE.decode(), MAGIC_WRITABLE.decode()]
        result["scope"] = (
            "caller-reviewed sparse original fixtures; full GPR/NZCV/SP/scratch/global input pages, readonly unless flagged writable; actual exits and faults"
        )
        result["execution_model"] = {
            "admission": "trusted_fixture=True; bounded independent instruction whitelist",
            "isolation": "no generic static target/address confinement; not a security sandbox or arbitrary corpus runner",
            "inputs": "caller reviews all addresses, table values and intended control landings",
            "global_pages": "runtime snapshots, readonly unless flagged writable, not invariant file data",
            "protection": "RX code without BTI; GCS inactive",
            "timeout": "incomplete OracleError, never successful outcome",
            "hardware_checked": False,
            "architectural_fault_authority": False,
            "omitted_state": [
                "FP/SIMD",
                "PSTATE.BTYPE",
                "GCS",
                "external events",
                "address tagging (TBI applies)",
            ],
        }
        return result

    def _admit(self, fragment):
        with tempfile.TemporaryDirectory(prefix="nyx-sparse-disassembly-") as directory:
            path = pathlib.Path(directory) / "code.bin"
            path.write_bytes(fragment.code)
            output = self.backend._command(
                [self.backend.objdump, "-D", "-b", "binary", "-m", "aarch64", "-EL", str(path)]
            ).stdout.decode()
        offset = 0
        checked = []
        for line in output.splitlines():
            match = re.fullmatch(r"\s*([0-9a-f]+):\s+([0-9a-f]{8})\s+(.+)", line)
            if not match:
                continue
            word = int.from_bytes(fragment.code[offset : offset + 4], "little")
            if int(match[1], 16) != offset or int(match[2], 16) != word:
                raise Declined("disassembly disagrees with original bytes")
            instruction = match[3].split("//", 1)[0].strip()
            mnemonic = instruction.split()[0]
            try:
                family, _, _ = _family(word)
            except Declined:
                family = None
            # A disagreement between the independent decoder and the disassembler
            # must refuse the fragment. Catching it here would hand the word to
            # the text whitelist instead, which is the opposite of a cross-check.
            if family is not None:
                if (
                    family == "b.cond"
                    and not re.fullmatch(
                        r"b\.(eq|ne|cs|hs|cc|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le|al|nv)", mnemonic
                    )
                    or family != "b.cond"
                    and mnemonic != family
                ):
                    raise Declined("control family mismatch")
            elif word & 0x1F000000 == 0x10000000 and mnemonic in ("adr", "adrp"):
                pass
            elif word & 0xFF000000 == 0x58000000 and re.fullmatch(
                r"ldr\s+x(?:[0-9]|[12][0-9]|30|zr),\s*0x[0-9a-f]+", instruction
            ):
                checked.append((instruction, word))
            else:
                try:
                    _parse(instruction)
                except Declined:
                    validate([instruction])
                # Read from this text alone, so confirm it reassembles to the word.
                checked.append((instruction, word))
            offset += 4
        if offset != len(fragment.code):
            raise Declined("disassembly did not cover original code")
        self.backend.reassembles(checked)

    @staticmethod
    def _response(data, state, fragments, exits):
        large = len(state.memory) == 2 * PAGE
        writable = _writable(state)
        words = 41 if writable else 40 if large else 39
        header = 8 + words * 8
        magic = MAGIC_WRITABLE if writable else MAGIC_LARGE if large else MAGIC
        if (
            len(data) != header + len(state.memory) + len(state.global_pages) * 4104
            or data[:8] != magic
        ):
            raise OracleError("invalid sparse response envelope")
        values = struct.unpack("<" + str(words) + "Q", data[8:header])
        pc, signal, fault = values[35:38]
        if large and values[39] != len(state.memory):
            raise OracleError("scratch length changed")
        if writable and values[40] != writable:
            raise OracleError("writable page mask changed")
        if values[33:35] != (state.scratch_base, state.load_bias) or values[38] != len(
            state.global_pages
        ):
            raise OracleError("sparse response changed mapping metadata")
        if signal == 0:
            if fault or pc not in {
                (exit_address + state.load_bias) & MASK64 for exit_address in exits
            }:
                raise OracleError("unlisted sparse exit")
        elif (
            signal not in (7, 11)
            or pc % 4
            or not any(
                (fragment.source_address + state.load_bias) & MASK64
                <= pc
                < ((fragment.source_address + state.load_bias) & MASK64) + len(fragment.code)
                for fragment in fragments
            )
        ):
            raise OracleError("fault PC outside declared original code")
        pages = []
        offset = header + len(state.memory)
        for original in state.global_pages:
            address = struct.unpack_from("<Q", data, offset)[0]
            if address != original.address:
                raise OracleError("global page identity changed")
            pages.append(GlobalPage(address, data[offset + 8 : offset + 4104], original.writable))
            offset += 4104
        try:
            result = SparseState(
                values[:31],
                values[31],
                values[32],
                data[header : header + len(state.memory)],
                values[33],
                values[34],
                tuple(pages),
            )
        except ValueError as error:
            raise OracleError("invalid sparse captured state") from error
        return SparseOutcome(result, pc, signal, fault)

    def run_many(self, fragments, entry_image, exit_images, states, *, trusted_fixture=False):
        if trusted_fixture is not True:
            raise Declined("explicit trusted_fixture=True required; no arbitrary corpus execution")
        if (
            type(fragments) is not tuple
            or not 1 <= len(fragments) <= 16
            or any(not isinstance(fragment, Fragment) for fragment in fragments)
            or sum(len(fragment.code) for fragment in fragments) > PAGE
        ):
            raise Declined("bounded immutable original fragments required")
        if (
            type(entry_image) is not int
            or not 0 <= entry_image <= MASK64
            or entry_image % 4
            or type(exit_images) is not tuple
            or not 1 <= len(exit_images) <= 16
            or any(
                type(value) is not int or not 0 <= value <= MASK64 or value % 4
                for value in exit_images
            )
            or len(set(exit_images)) != len(exit_images)
        ):
            raise Declined("invalid sparse entry/exit declarations")
        for fragment in fragments:
            self._admit(fragment)
        cases = []
        for state in states:
            if len(cases) == self.backend.max_cases:
                raise Declined("case budget exhausted")
            if not isinstance(state, SparseState):
                raise Declined("expected SparseState")
            _layout(fragments, entry_image, exit_images, state)
            cases.append(state)
        if not cases:
            raise Declined("zero cases: no evidence")
        with tempfile.TemporaryDirectory(prefix="nyx-sparse-oracle-") as directory:
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
                    str(pathlib.Path(__file__).with_name("runner_sparse.c")),
                ]
            )
            outputs = []
            for state in cases:
                large = len(state.memory) == 2 * PAGE
                writable = _writable(state)
                header = (
                    *state.registers,
                    state.nzcv,
                    state.sp,
                    state.scratch_base,
                    state.load_bias,
                    entry_image,
                    len(fragments),
                    len(exit_images),
                    len(state.global_pages),
                )
                if large:
                    header += (len(state.memory),)
                if writable:
                    header += (writable,)
                magic = MAGIC_WRITABLE if writable else MAGIC_LARGE if large else MAGIC
                payload = (
                    magic
                    + struct.pack("<" + str(len(header)) + "Q", *header)
                    + b"".join(
                        struct.pack("<QQ", fragment.source_address, len(fragment.code))
                        + fragment.code
                        for fragment in fragments
                    )
                    + struct.pack("<" + "Q" * len(exit_images), *exit_images)
                    + state.memory
                    + b"".join(
                        struct.pack("<Q", page.address) + page.data for page in state.global_pages
                    )
                )
                raw = self.backend._command(
                    [self.backend.qemu, "-cpu", "max", str(executable)], input=payload
                ).stdout
                outputs.append(self._response(raw, state, fragments, exit_images))
            return outputs

    def run(self, fragments, entry_image, exit_images, state, *, trusted_fixture=False):
        return self.run_many(
            fragments, entry_image, exit_images, [state], trusted_fixture=trusted_fixture
        )[0]
