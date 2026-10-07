"""Run a whole original function once from a supplied state under QEMU.

The caller lays out every page the run may touch: the original code, stubs
standing in for its callees, landings that end the run, and data. Nothing
else is mapped, so an access anywhere else faults. Only the original code is
admitted through the independent disassembler whitelist; the caller writes
the stubs and must review every address it supplies.

Stubs placed on STUB pages are traced: the runner keeps the original code and
the stubs in separate executable domains and records each crossing in order.
Stubs sharing a page with other code run untraced, and their events are None.
"""

import dataclasses
import pathlib
import struct
import tempfile

from tools.oracle.aarch64 import Declined, MASK64, Oracle, OracleError, State
from tools.oracle.sparse import Fragment, SparseOracle

MAGIC = b"NYXFUN02"
PAGE = 4096
LOW, HIGH = 0x100000000, 0x1000000000
READ, WRITE, EXECUTE, STUB = 1, 2, 4, 8
BRK = 0xD4200000
MAX_REGIONS, MAX_EXITS, MAX_BYTES = 256, 16, 64 << 20
# Runner outcomes past every signal number: the run left the traced envelope.
ENTERED_STUB_BODY, EVENT_OVERFLOW = 256, 257


@dataclasses.dataclass(frozen=True)
class Region:
    address: int
    data: bytes
    protection: int

    def __post_init__(self):
        if (
            type(self.address) is not int
            or self.address % PAGE
            or not LOW <= self.address < HIGH
            or type(self.data) is not bytes
            or not self.data
            or len(self.data) % PAGE
            or self.address + len(self.data) > HIGH
            or type(self.protection) is not int
            or not 0 < self.protection <= 15
            or self.protection & WRITE
            and self.protection & EXECUTE
            or self.protection & STUB
            and not self.protection & EXECUTE
        ):
            raise ValueError("invalid page-aligned region")


@dataclasses.dataclass(frozen=True)
class FunctionState:
    registers: tuple[int, ...]
    nzcv: int
    sp: int
    regions: tuple[Region, ...]

    def __post_init__(self):
        State(self.registers, self.nzcv)
        if type(self.sp) is not int or not 0 <= self.sp <= MASK64:
            raise ValueError("SP must be unsigned64")
        if (
            type(self.regions) is not tuple
            or not 0 < len(self.regions) <= MAX_REGIONS
            or any(not isinstance(region, Region) for region in self.regions)
            or sum(len(region.data) for region in self.regions) > MAX_BYTES
        ):
            raise ValueError("invalid region list")
        spans = sorted(
            (region.address, region.address + len(region.data)) for region in self.regions
        )
        if any(right[0] < left[1] for left, right in zip(spans, spans[1:])):
            raise ValueError("regions overlap")


@dataclasses.dataclass(frozen=True)
class FunctionOutcome:
    registers: tuple[int, ...]
    nzcv: int
    sp: int
    pc: int
    # 0 at a declared landing; otherwise the signal that ended the run.
    signal: int
    fault_address: int
    # The final bytes of every writable region, by address.
    memory: dict
    # Ordered call and return events across the STUB domain; None if untraced.
    events: tuple = None

    @property
    def completed(self):
        return self.signal == 0

    @property
    def left_trace(self):
        return self.signal in (ENTERED_STUB_BODY, EVENT_OVERFLOW)


def _event(words):
    kind, pc, link, sp, *registers = words
    if kind == 1:
        return {
            "kind": "call",
            "target": pc,
            "return_address": link,
            "sp": sp,
            "arguments": list(registers),
        }
    if kind == 2 and not link and not any(registers[2:]):
        return {"kind": "return", "pc": pc, "sp": sp, "results": list(registers[:2])}
    if kind == 3:
        return {"kind": "enter_other", "pc": pc}
    raise OracleError("invalid call event")


def _check_events(events, signal):
    """Calls and returns alternate from a call; enter_other and a full list end the run."""
    for index, event in enumerate(events):
        expected = "call" if index % 2 == 0 else "return"
        if event["kind"] == "enter_other":
            if index != len(events) - 1 or signal != ENTERED_STUB_BODY:
                raise OracleError("stub-body entry not at the end of its run")
        elif event["kind"] != expected:
            raise OracleError("call events do not alternate")
    if (signal == ENTERED_STUB_BODY) != bool(events and events[-1]["kind"] == "enter_other"):
        raise OracleError("stub-body outcome without its event")
    if signal == EVENT_OVERFLOW and len(events) != 64:
        raise OracleError("event overflow before the event list was full")


class FunctionOracle:
    def __init__(self, **kwargs):
        self.backend = Oracle(**kwargs)
        self.admission = SparseOracle(**kwargs)
        self.admitted = {}
        # Reviewed data words, such as a jump table, that share an executable
        # page with original code. They are read, never meant to run; one that
        # did would execute as whatever it encodes, which is why each is named.
        self.data = {}
        self.executable = None
        self.directory = None

    def versions(self):
        result = self.backend.versions()
        result["protocol"] = MAGIC.decode()
        result["scope"] = (
            "caller-laid-out whole-function runs: GPR/NZCV/SP and every mapped page;"
            " original code admitted by the disassembler whitelist or a defined-Q copy,"
            " stubs caller-written; vector register results are not compared"
        )
        result["execution_model"] = {
            "admission": "trusted_fixture=True; every executable word is admitted original code, an explicit stub, or BRK",
            "isolation": "not a security sandbox or arbitrary corpus runner",
            "protection": "RX code without BTI; GCS inactive; W^X per region",
            "call_trace": "STUB pages form a separate executable domain; each crossing is an ordered event",
            "timeout": "incomplete OracleError, never an outcome",
            "hardware_checked": False,
            "architectural_fault_authority": False,
            "omitted_state": [
                "FP/SIMD",
                "TPIDR_EL0",
                "PSTATE.BTYPE",
                "GCS",
                "external events",
                "address tagging (TBI applies)",
            ],
        }
        return result

    def admit(self, address, code):
        """Admit original code, one page-bounded fragment at a time."""
        if (
            type(address) is not int
            or type(code) is not bytes
            or not code
            or len(code) % 4
            or address % 4
            or address < LOW
            or address + len(code) > HIGH
        ):
            raise Declined("original code must be whole aligned words")
        words = {
            address + index * 4: word[0]
            for index, word in enumerate(struct.iter_unpack("<I", code))
        }
        if any(pc in self.admitted and self.admitted[pc] != word for pc, word in words.items()):
            raise Declined("original code changed after admission")
        offset = 0
        while offset < len(code):
            length = min(PAGE - (address + offset) % PAGE, len(code) - offset)
            self.admission._admit(Fragment(address + offset, code[offset : offset + length]))
            offset += length
        self.admitted.update(words)

    def admit_vector_copy(self, address, code):
        # The runner does not initialize or report vector registers. This
        # bounded sequence defines its entire Q register before using it,
        # so its memory result is independent of omitted entry vector state.
        if (
            type(address) is not int
            or address % 4
            or address < LOW
            or address + 12 > HIGH
            or type(code) is not bytes
            or len(code) != 12
        ):
            raise Declined("vector copy requires three aligned instructions")
        load, store, ret = struct.unpack("<III", code)
        if (
            load & 0xFFC00000 != 0x3DC00000
            or store & 0xFFC00000 != 0x3D800000
            or load & 31 != store & 31
            or ret & 0xFFFFFC1F != 0xD65F0000
        ):
            raise Declined("vector copy requires LDR Q, STR of the same Q, and RET")
        words = {address + index * 4: word for index, word in enumerate((load, store, ret))}
        if any(pc in self.admitted and self.admitted[pc] != word for pc, word in words.items()):
            raise Declined("original code changed after admission")
        self.admitted.update(words)

    def admit_data(self, address, data):
        """Admit reviewed data words that sit in an executable page."""
        if (
            type(address) is not int
            or type(data) is not bytes
            or not data
            or len(data) % 4
            or address % 4
            or address < LOW
            or address + len(data) > HIGH
        ):
            raise Declined("data must be whole aligned words")
        words = {
            address + index * 4: word[0]
            for index, word in enumerate(struct.iter_unpack("<I", data))
        }
        if any(
            pc in self.admitted or pc in self.data and self.data[pc] != word
            for pc, word in words.items()
        ):
            raise Declined("data overlaps original code or changed after admission")
        self.data.update(words)

    def _runner(self):
        if self.executable is None:
            self.directory = tempfile.TemporaryDirectory(prefix="nyx-function-oracle-")
            executable = pathlib.Path(self.directory.name) / "runner"
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
                    str(pathlib.Path(__file__).with_name("runner_function.c")),
                ]
            )
            self.executable = executable
        return self.executable

    def run(self, state, entry, exits, *, stubs=(), trusted_fixture=False):
        if trusted_fixture is not True:
            raise Declined("explicit trusted_fixture=True required; no arbitrary corpus execution")
        if not isinstance(state, FunctionState):
            raise Declined("expected FunctionState")
        if (
            type(exits) is not tuple
            or not 0 < len(exits) <= MAX_EXITS
            or len(set(exits)) != len(exits)
            or type(entry) is not int
            or entry % 4
        ):
            raise Declined("invalid entry or landings")
        executable = [region for region in state.regions if region.protection & EXECUTE]
        if type(stubs) is not tuple or len(stubs) > 32:
            raise Declined("expected at most 32 explicit stubs")
        stub_words = {}
        for stub in stubs:
            if (
                type(stub) is not tuple
                or len(stub) != 2
                or type(stub[0]) is not int
                or type(stub[1]) is not bytes
                or not stub[1]
                or len(stub[1]) % 4
                or stub[0] % 4
                or stub[0] < LOW
                or stub[0] + len(stub[1]) > HIGH
            ):
                raise Declined("invalid stub")
            for index, (word,) in enumerate(struct.iter_unpack("<I", stub[1])):
                pc = stub[0] + index * 4
                if pc in self.admitted or pc in stub_words or pc in self.data:
                    raise Declined("overlapping original code or stubs")
                stub_words[pc] = word

        def word(pc):
            for region in executable:
                if region.address <= pc < region.address + len(region.data):
                    return int.from_bytes(
                        region.data[pc - region.address : pc - region.address + 4], "little"
                    )
            return None

        if word(entry) is None or any(
            type(pc) is not int or pc % 4 or word(pc) != BRK for pc in exits
        ):
            raise Declined("entry must be mapped code and every landing a mapped BRK")

        def stub_page(pc):
            return any(
                region.protection & STUB
                and region.address <= pc < region.address + len(region.data)
                for region in executable
            )

        traced = any(region.protection & STUB for region in executable)
        if traced and (
            any(not stub_page(pc) for pc in stub_words)
            or any(stub_page(pc) for pc in (entry, *exits, *self.admitted))
        ):
            raise Declined("traced stubs need STUB pages apart from original code and landings")
        if entry not in self.admitted or entry in exits:
            raise Declined("entry must name admitted original code outside the landings")
        for region in executable:
            for index, (actual,) in enumerate(struct.iter_unpack("<I", region.data)):
                pc = region.address + index * 4
                if (
                    pc in self.admitted
                    and actual != self.admitted[pc]
                    or pc in stub_words
                    and actual != stub_words[pc]
                    or pc in self.data
                    and actual != self.data[pc]
                    or pc not in self.admitted
                    and pc not in stub_words
                    and pc not in self.data
                    and actual != BRK
                ):
                    raise Declined("executable word lacks matching original admission or stub")
        if any(word(pc) != expected for pc, expected in stub_words.items()):
            raise Declined("stub bytes differ from mapped code")
        header = (*state.registers, state.nzcv, state.sp, entry, len(state.regions), len(exits))
        payload = [MAGIC, struct.pack("<36Q", *header)]
        for region in state.regions:
            payload += [
                struct.pack("<3Q", region.address, len(region.data), region.protection),
                region.data,
            ]
        payload.append(struct.pack("<%dQ" % len(exits), *exits))
        entries = sorted(stub[0] for stub in stubs) if traced else []
        payload.append(struct.pack("<%dQ" % (len(entries) + 1), len(entries), *entries))
        data = self.backend._command(
            [self.backend.qemu, "-cpu", "max", str(self._runner())], input=b"".join(payload)
        ).stdout
        writable = [region for region in state.regions if region.protection & WRITE]
        size = 8 + 36 * 8 + sum(len(region.data) for region in writable)
        if len(data) < size + 8 or data[:8] != MAGIC:
            raise OracleError("invalid function response envelope")
        count = struct.unpack_from("<Q", data, size)[0]
        if count > 64 or len(data) != size + 8 + count * 96:
            raise OracleError("invalid function response envelope")
        values = struct.unpack_from("<36Q", data, 8)
        signal = values[34]
        if signal not in (0, 4, 5, 7, 11, ENTERED_STUB_BODY, EVENT_OVERFLOW):
            raise OracleError("unexpected function signal")
        events = tuple(
            _event(struct.unpack_from("<12Q", data, size + 8 + index * 96))
            for index in range(count)
        )
        if not traced and events:
            raise OracleError("untraced run reported call events")
        _check_events(events, signal)
        if signal == 0 and values[33] not in exits:
            raise OracleError("completion away from a landing")
        memory, offset = {}, 8 + 36 * 8
        for region in writable:
            memory[region.address] = data[offset : offset + len(region.data)]
            offset += len(region.data)
        return FunctionOutcome(
            values[:31],
            values[31],
            values[32],
            values[33],
            signal,
            values[35],
            memory,
            events if traced or not stubs else None,
        )
