#!/usr/bin/env python3
"""Check every unflattening transition of a recover-regions artifact against QEMU.

For each transition the original bytes of its route run under the sparse QEMU
oracle, and the recovered path the artifact publishes runs under an evaluator
written here from the IR definition, not Nyx's own. Both start from the same
state, built to satisfy what the transition declares: its entry relations and
the image values the loader writes. A transition is verified only if every
completed original run matches the recovered path in registers, flags, SP,
memory and next PC. Both arms of a condition must be seen at least four times,
each on the destination the condition selects. A single-destination route's
final BR, replaced by a direct branch to its destination, must also match the
original exactly. Anything the harness cannot check is inconclusive, never
verified. A verdict rests on the loader's image values, as the artifact does.
The routes are corpus bytes: the oracle admits each instruction independently,
but its runner is no sandbox, so run this only on images you trust.
"""

import argparse
import concurrent.futures
import itertools
import json
import pathlib
import random
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tools.oracle.aarch64 import Declined, OracleError
from tools.oracle.program import GlobalPage
from tools.oracle.sparse import Fragment, SparseOracle, SparseState

MASK64 = (1 << 64) - 1
PAGE = 4096
SCRATCH = 0x100000000
SCRATCH_SIZE = 2 * PAGE
BIASES = (0x200000000, 0x600000000)
SP, N, Z, C, V = 31, 32, 33, 34, 35
MAX_GLOBAL_PAGES = 8


def mask(width):
    return (1 << width) - 1


class Image:
    """File-backed load segments and relative relocations of a little-endian ELF64."""

    def __init__(self, path):
        self.data = pathlib.Path(path).read_bytes()
        if self.data[:6] != b"\x7fELF\x02\x01":
            raise ValueError("expected a little-endian ELF64 image")
        (phoff,) = struct.unpack_from("<Q", self.data, 32)
        phentsize, phnum = struct.unpack_from("<HH", self.data, 54)
        self.segments, dynamic = [], None
        # RELRO, and ranges the run declares constant: never runtime state to vary.
        self.frozen = []
        for index in range(phnum):
            kind, flags, offset, vaddr, _, filesz, memsz, _ = struct.unpack_from(
                "<IIQQQQQQ", self.data, phoff + index * phentsize
            )
            if kind == 1:
                self.segments.append((vaddr, offset, filesz, memsz, bool(flags & 2)))
            elif kind == 2:
                dynamic = (offset, filesz)
            elif kind == 0x6474E552:
                self.frozen.append((vaddr, memsz))
        self.relative = {}
        if dynamic:
            tags = {}
            for at in range(dynamic[0], dynamic[0] + dynamic[1], 16):
                tag, value = struct.unpack_from("<qQ", self.data, at)
                if tag == 0:
                    break
                tags[tag] = value
            if 7 in tags and 8 in tags:
                start = self.offset(tags[7])
                for at in range(start, start + tags[8], 24):
                    slot, info, addend = struct.unpack_from("<QQq", self.data, at)
                    if info & 0xFFFFFFFF == 1027:
                        self.relative[slot] = addend

    def offset(self, address):
        for vaddr, offset, filesz, _, _ in self.segments:
            if vaddr <= address < vaddr + filesz:
                return offset + address - vaddr
        raise ValueError("address %#x has no file bytes" % address)

    def page(self, address, bias):
        """One page as the loader leaves it at `bias`: file bytes, zero fill, relocations."""
        content = bytearray(PAGE)
        for vaddr, offset, filesz, _, _ in self.segments:
            low, high = max(address, vaddr), min(address + PAGE, vaddr + filesz)
            if low < high:
                content[low - address : high - address] = self.data[
                    offset + low - vaddr : offset + high - vaddr
                ]
        for slot in range(address, address + PAGE, 8):
            if slot in self.relative:
                struct.pack_into(
                    "<Q", content, slot - address, (bias + self.relative[slot]) & MASK64
                )
        return bytes(content)

    def mapped(self, address):
        return any(vaddr <= address < vaddr + memsz for vaddr, _, _, memsz, _ in self.segments)

    def writable(self, address):
        return any(
            vaddr <= address < vaddr + memsz and write
            for vaddr, _, _, memsz, write in self.segments
        )

    def variable(self, address, size):
        """Writable bytes that neither the loader relocates nor the run declares constant."""
        return all(
            self.writable(byte)
            and not any(start <= byte < start + length for start, length in self.frozen)
            for byte in range(address, address + size)
        ) and not any(slot in self.relative for slot in range(address & ~7, address + size, 8))


class Fault(Exception):
    pass


class Unsupported(Exception):
    pass


class Tagged(Exception):
    """A data address with a nonzero top byte. The hardware ignores that byte
    (TBI); the IR does not model it and Nyx declines such an access, so a state
    that reaches one is outside what the transition claims."""


def evaluate(path, registers, nzcv, sp, memory, bias, accesses=None, sources=None, loaded=None):
    """Run a published recovered path. Returns (registers, nzcv, sp, next_pc, values).
    `accesses`, when given, records the address of each load and store. `sources`,
    when given, maps each value to the inputs it came from: storage the path
    read, and loads of bytes the path had not itself written."""
    nodes, boundaries = path["nodes"], path["boundaries"]
    named = {
        item["value"]: (bias + item["image_address"]) & MASK64
        for item in path.get("destinations", [])
    }
    omitted = {item["store"] for item in path.get("store_omissions", [])}
    for item in path.get("paired_load_omissions", []):
        omitted.update(item["loads"])
    state = {index: registers[index] for index in range(31)}
    state[SP] = sp
    for bit, storage in zip((31, 30, 29, 28), (N, Z, C, V)):
        state[storage] = (nzcv >> bit) & 1
    values, written = {}, {}

    def value(id):
        return named[id] if id in named else values[id]

    next_pc = None
    for index, boundary in enumerate(boundaries):
        writes = []
        for id in range(boundary["first_node"], boundary["first_node"] + boundary["node_count"]):
            if id in omitted:
                continue
            node = nodes[id]
            op, width, inputs = node["op"], node["width"], node["inputs"]
            args = [value(i) for i in inputs]
            widths = [nodes[i]["width"] for i in inputs]
            if op == "constant":
                result = node["immediate"] & mask(width)
            elif op == "image_address":
                result = (bias + node["immediate"]) & MASK64
            elif op == "read":
                if node["storage"] not in state:
                    raise Unsupported("reads storage %d" % node["storage"])
                result = state[node["storage"]] & mask(width)
            elif op in ("add", "sub", "mul", "bit_and", "bit_or", "bit_xor"):
                a, b = args
                result = {
                    "add": a + b,
                    "sub": a - b,
                    "mul": a * b,
                    "bit_and": a & b,
                    "bit_or": a | b,
                    "bit_xor": a ^ b,
                }[op] & mask(width)
            elif op == "bit_not":
                result = ~args[0] & mask(width)
            elif op in ("udiv", "sdiv", "umulh", "smulh", "clz", "rbit"):
                # AArch64 rules: a zero divisor gives zero and the quotient
                # truncates toward zero; the high products keep the upper half.
                def signed(x):
                    return x - (1 << width) if x >> (width - 1) else x

                a = args[0]
                b = args[1] if len(args) > 1 else 0
                if op == "udiv":
                    result = a // b if b else 0
                elif op == "sdiv":
                    sa, sb = signed(a), signed(b)
                    result = (
                        0 if sb == 0 else (abs(sa) // abs(sb)) * (-1 if (sa < 0) != (sb < 0) else 1)
                    )
                elif op == "umulh":
                    result = (a * b) >> width
                elif op == "smulh":
                    result = (signed(a) * signed(b)) >> width
                elif op == "clz":
                    result = width - a.bit_length()
                else:
                    result = int(format(a, "0%db" % width)[::-1], 2)
                result &= mask(width)
            elif op in ("shl", "lshr", "ashr"):
                a, count = args
                if op == "shl":
                    result = (a << count) & mask(width) if count < width else 0
                elif op == "lshr":
                    result = a >> count if count < width else 0
                else:
                    signed = a - (1 << width) if a >> (width - 1) else a
                    result = (signed >> min(count, width)) & mask(width)
            elif op == "extract":
                result = (args[0] >> node["immediate"]) & mask(width)
            elif op == "zext":
                result = args[0]
            elif op == "select":
                result = args[1] if args[0] & 1 else args[2]
            elif op in ("equal", "unsigned_less", "signed_less"):
                a, b = args
                if op == "signed_less":
                    w = widths[0]
                    a, b = (x - (1 << w) if x >> (w - 1) else x for x in (a, b))
                result = int(a == b if op == "equal" else a < b)
            elif op in ("load", "store"):
                size = width // 8
                order = node.get("access", {}).get("byte_order", "little")
                if accesses is not None:
                    accesses[id] = args[0]
                if op == "load":
                    raw = memory.read(args[0], size)
                    result = int.from_bytes(raw, order)
                    if loaded is not None:
                        loaded[id] = result
                    if sources is not None:
                        origin = set()
                        for byte in range(args[0], args[0] + size):
                            origin |= written.get(byte, {("load", id)})
                        sources[id] = frozenset(origin)
                else:
                    memory.write(args[0], args[1].to_bytes(size, order))
                    if sources is not None:
                        for byte in range(args[0], args[0] + size):
                            written[byte] = sources[inputs[1]]
                    continue
            else:
                raise Unsupported("op " + op)
            values[id] = result
            if sources is not None and op != "load":
                sources[id] = (
                    frozenset({("storage", node["storage"])})
                    if op == "read"
                    else frozenset().union(*(sources.get(i, frozenset()) for i in inputs))
                )
        for write in boundary["writes"]:
            writes.append((write["storage"], value(write["value"])))
        for storage, result in writes:
            state[storage] = result
        transfer = boundary["transfer"]
        if transfer is None:
            next_pc = (
                bias
                + path["sources"][index]["source_address"]
                + len(bytes.fromhex(path["sources"][index]["bytes"]))
            ) & MASK64
        elif transfer["kind"] in ("jump", "return"):
            next_pc = value(transfer["target"])
        elif transfer["kind"] == "conditional":
            next_pc = (
                value(transfer["target"])
                if value(transfer["condition"]) & 1
                else value(transfer["alternative"])
            )
        else:
            raise Unsupported("transfer " + transfer["kind"])
        expected = boundary.get("expected_successor_image")
        if expected is not None and next_pc != (bias + expected) & MASK64:
            raise Fault("diverged at boundary %d" % index)
    flags = (state[N] << 31) | (state[Z] << 30) | (state[C] << 29) | (state[V] << 28)
    return tuple(state[index] & MASK64 for index in range(31)), flags, state[SP], next_pc, values


class Memory:
    def __init__(self, scratch, pages):
        self.regions = [(SCRATCH, bytearray(scratch), True)]
        self.regions += [(page.address, bytearray(page.data), page.writable) for page in pages]

    def locate(self, address, size, write):
        if address >> 56:
            raise Tagged()
        for base, content, writable in self.regions:
            if base <= address and address + size <= base + len(content):
                if write and not writable:
                    raise Fault("write to readonly page")
                return content, address - base
        raise Fault("unmapped access %#x" % address)

    # An access may span two adjacent mappings, as the oracle's merged arenas allow.
    def read(self, address, size):
        result = bytearray()
        for byte in range(address, address + size):
            content, offset = self.locate(byte, 1, False)
            result.append(content[offset])
        return bytes(result)

    def write(self, address, data):
        for position, value in enumerate(data):
            content, offset = self.locate(address + position, 1, True)
            content[offset] = value


def root(nodes, id):
    """An address as (kind, key, offset): a storage read, an image location, or opaque."""
    node = nodes[id]
    if node["op"] == "read":
        return "read", node["storage"], 0
    if node["op"] == "image_address":
        return "image", node["immediate"], 0
    if node["op"] == "constant":
        return "absolute", 0, node["immediate"]
    if node["op"] in ("add", "sub"):
        a, b = (root(nodes, i) for i in node["inputs"])
        if b[0] == "absolute":
            return a[0], a[1], (a[2] + b[2] if node["op"] == "add" else a[2] - b[2]) & MASK64
        if a[0] == "absolute" and node["op"] == "add":
            return b[0], b[1], (a[2] + b[2]) & MASK64
        if a[0] in ("image", "image_indexed"):
            return "image_indexed", a[1], a[2]
    return "opaque", id, 0


def signed(value):
    return value - (1 << 64) if value >> 63 else value


def pointers(nodes, id, registers, loads, memo, pairs=None):
    """Storage and loaded values an address is built from, however obfuscated.
    An address formed from an image location is a table: what indexes it is
    data, not a pointer, and is left alone. A sum of two such inputs is a
    pointer and an offset, but which is which is not known, so it is recorded."""
    if id not in memo:
        memo[id] = (frozenset(), frozenset())
        node = nodes[id]
        found_registers, found_loads = set(), set()
        if node["op"] == "read" and node["storage"] <= SP:
            found_registers.add(node["storage"])
        elif node["op"] == "load":
            found_loads.add(id)
        elif node["op"] != "image_address" and root(nodes, id)[0] not in ("image", "image_indexed"):
            sides = []
            for input in node["inputs"]:
                side = pointers(nodes, input, set(), set(), memo, pairs)
                sides.append(side)
                found_registers |= side[0]
                found_loads |= side[1]
            if (
                pairs is not None
                and node["op"] == "add"
                and len(sides) == 2
                and all(r or l for r, l in sides)
                and tuple(sides) not in pairs
            ):
                pairs.append(tuple(sides))
        memo[id] = (frozenset(found_registers), frozenset(found_loads))
    registers |= memo[id][0]
    loads |= memo[id][1]
    return memo[id]


def cone(nodes, condition):
    """The loads, storage and literals a condition reads."""
    loads, storage, literals, pending, seen = set(), set(), set(), [condition], set()
    while pending:
        id = pending.pop()
        if id in seen or id >= len(nodes):
            continue
        seen.add(id)
        node = nodes[id]
        if node["op"] == "read":
            storage.add(node["storage"])
        elif node["op"] == "load":
            loads.add(id)
        elif node["op"] == "constant":
            literals.update(
                value & MASK64
                for value in (node["immediate"] - 1, node["immediate"], node["immediate"] + 1)
            )
        pending.extend(node["inputs"])
    return sorted(loads), sorted(storage), sorted(literals)


def plan(transition, region, image, relations_held=None):
    """Everything a state needs: pointer placements, global pages, biased values.
    Proved entry relations are imposed only where the claim rests on them, or
    when the caller asks; otherwise aliasing they would exclude is explored."""
    path = region["after"]
    nodes = path["nodes"]
    if relations_held is None:
        relations_held = region.get("entry_relation_dependent", "entry_relations" in region)
    relations = (
        {
            item["storage"]: (item["root"], item["offset"])
            for item in region.get("entry_relations", [])
        }
        if relations_held
        else {}
    )
    registers, pointer_loads, pages, literals = set(), set(), set(), {0, 1}
    index_registers, index_loads, pairs, memo, offsets = set(), set(), [], {}, {}
    for node in nodes:
        if node["op"] == "constant" and node["immediate"] < 4096:
            literals.update({node["immediate"], node["immediate"] - 1, node["immediate"] + 1})
        if node["op"] == "read" and node["storage"] > SP and node["storage"] not in (N, Z, C, V):
            raise Unsupported("reads storage %d the runner does not control" % node["storage"])
        if node["op"] not in ("load", "store"):
            continue
        kind, key, offset = root(nodes, node["inputs"][0])
        if kind in ("image", "image_indexed"):
            address = (key + offset) & MASK64
            pages.add(address & ~(PAGE - 1))
            if kind == "image_indexed":
                pages.add((address & ~(PAGE - 1)) + PAGE)
                # A table index is small, or the access leaves the table.
                for input in nodes[node["inputs"][0]]["inputs"]:
                    if root(nodes, input)[0] not in ("image", "image_indexed"):
                        pointers(nodes, input, index_registers, index_loads, {})
        elif kind == "absolute":
            raise Unsupported("absolute memory address")
        else:
            pointers(nodes, node["inputs"][0], registers, pointer_loads, memo, pairs)
            if kind == "read":
                offsets.setdefault(key, set()).add(signed(offset))
    pages = sorted(page for page in pages if image.mapped(page))
    if len(pages) > MAX_GLOBAL_PAGES:
        raise Unsupported("needs more than eight global pages")
    condition = transition["condition"]
    loads, storage, compared = cone(nodes, condition) if condition is not None else ([], [], [])
    return {
        "relations": relations,
        "registers": sorted(registers),
        "pointer_loads": sorted(pointer_loads),
        "index_registers": sorted(index_registers - registers),
        "index_loads": sorted(index_loads),
        "pairs": pairs,
        "offsets": {storage: sorted(values) for storage, values in offsets.items()},
        "related": sorted(set(relations) | {root for root, _ in relations.values()}),
        "pages": pages,
        "literals": sorted({value & MASK64 for value in literals} | set(compared)),
        "compared": sorted(set(compared) | {0, 1}),
        "cone_loads": loads,
        "cone_storage": storage,
    }


CENTER = SCRATCH + SCRATCH_SIZE // 2
# Where a pointer may land. Choosing among them per state covers both pointers
# that alias and pointers that do not.
SLOTS = tuple(SCRATCH + 1024 + 768 * k for k in range(8))


def make_state(setup, image, bias, rng, path=None, forces=()):
    """A state meeting the transition's declarations. Pointers land mid-scratch,
    related storage keeps its relation, and loaded pointers are rewritten to
    point there too, found by running the recovered path until they settle."""
    small = setup["literals"]

    def pick():
        return rng.choice(small) if rng.random() < 0.6 else rng.getrandbits(64)

    registers = [pick() for _ in range(31)]
    # Of a sum of two pointer inputs, one side may be an offset in this state.
    offset_registers, offset_loads = set(setup["index_registers"]), set(setup["index_loads"])
    for sides in setup["pairs"]:
        side_registers, side_loads = rng.choice(sides + ((frozenset(), frozenset()),))
        offset_registers |= side_registers
        offset_loads |= side_loads
    sp = CENTER
    for storage in setup["registers"]:
        if storage in offset_registers and storage < 31:
            registers[storage] = rng.randrange(256)
        elif storage < 31:
            registers[storage] = rng.choice(SLOTS)
        elif storage == SP:
            sp = rng.choice(SLOTS)
    for storage in setup["index_registers"]:
        if storage < 31:
            registers[storage] = rng.randrange(256)
    # Slots alone keep distinct bases far apart; a base placed so that one of
    # its accesses lands on or next to another base's access tests aliasing
    # at the displacements the path actually uses.
    bases = [
        storage for storage in setup["offsets"] if storage < 31 and storage not in offset_registers
    ]
    for index, storage in enumerate(bases[1:], 1):
        if rng.random() < 0.4:
            other = rng.choice(bases[:index])
            placed = (
                registers[other]
                + rng.choice(setup["offsets"][other])
                - rng.choice(setup["offsets"][storage])
                + rng.randrange(-8, 9)
            )
            if SCRATCH <= placed < SCRATCH + SCRATCH_SIZE:
                registers[storage] = placed
    for storage, (owner, displacement) in setup["relations"].items():
        base = sp if owner == SP else registers[owner]
        if storage == SP:
            sp = (base + displacement) & MASK64
        elif storage < 31:
            registers[storage] = (base + displacement) & MASK64
    nzcv = rng.randrange(16) << 28
    memory = bytearray(rng.getrandbits(8) for _ in range(SCRATCH_SIZE))
    # Writable data is runtime state, not file content: a forced value may land there.
    written = {}

    def global_pages():
        return tuple(
            GlobalPage(
                bias + page, bytes(written.get(page, image.page(page, bias))), image.writable(page)
            )
            for page in sorted(setup["pages"])
        )

    pages = global_pages()
    forced_loads, pointed = {}, {}
    for kind, key, value in forces:
        if kind == "load":
            forced_loads[key] = value
        elif key in (N, Z, C, V):
            nzcv = (nzcv & ~(1 << (31 - (key - N)))) | ((value & 1) << (31 - (key - N)))
        elif key < 31 and key not in setup["registers"] and key not in setup["related"]:
            registers[key] = value
    if path is not None:
        # Each pass settles at least one more pointer, forced value or image
        # page the route touches. Placements that overlap can keep displacing
        # each other, so the passes are bounded; a state that did not settle
        # is still a state, and only the original's run under QEMU is judged.
        for _ in range(12):
            seen, loaded = {}, {}
            try:
                evaluate(
                    path,
                    list(registers),
                    nzcv,
                    sp,
                    Memory(bytes(memory), pages),
                    bias,
                    accesses=seen,
                    loaded=loaded,
                )
            except (Fault, Tagged, Unsupported, KeyError):
                pass
            changed = False

            def target(id, width):
                # A value forced relative to another load follows what that load read.
                value = forced_loads[id]
                if isinstance(value, tuple):
                    other, delta = value
                    if other not in loaded:
                        return None
                    value = loaded[other] + delta
                return value & mask(8 * width)

            for address in seen.values():
                page = (address - bias) & ~(PAGE - 1) & MASK64
                if (
                    not SCRATCH <= address < SCRATCH + SCRATCH_SIZE
                    and page not in setup["pages"]
                    and image.mapped(page)
                ):
                    if len(setup["pages"]) == MAX_GLOBAL_PAGES:
                        raise Unsupported("needs more than eight global pages")
                    setup["pages"] = sorted(setup["pages"] + [page])
                    pages = global_pages()
                    changed = True
            for id, address in seen.items():
                if path["nodes"][id]["op"] != "load":
                    continue
                width = path["nodes"][id]["width"] // 8
                if id in forced_loads:
                    wanted = target(id, width)
                elif id in offset_loads:
                    # An offset may be small, or be whatever the image or chance left there.
                    wanted = pointed.setdefault(
                        id, rng.randrange(256) if rng.random() < 0.5 else None
                    )
                elif id in setup["pointer_loads"]:
                    wanted = pointed.setdefault(id, rng.choice(SLOTS)) & mask(8 * width)
                else:
                    continue
                if wanted is None:
                    continue
                if SCRATCH <= address <= SCRATCH + SCRATCH_SIZE - width:
                    content, at = memory, address - SCRATCH
                else:
                    page = (address - bias) & ~(PAGE - 1) & MASK64
                    at = address - bias - page
                    if (
                        page not in setup["pages"]
                        or at > PAGE - width
                        or not image.variable(page + at, width)
                    ):
                        continue
                    content = written.setdefault(page, bytearray(image.page(page, bias)))
                if int.from_bytes(content[at : at + width], "little") != wanted:
                    content[at : at + width] = wanted.to_bytes(width, "little")
                    if content is not memory:
                        pages = global_pages()
                    changed = True
            if not changed:
                break
    return SparseState(tuple(registers), nzcv, sp, bytes(memory), SCRATCH, bias, pages)


class Search:
    """States whose recovered path reaches a wanted destination. The recovered
    path only proposes; the original's run under QEMU is what gets judged.
    Inputs that decide the condition, found by following its value back
    through memory, are tried against the literals it meets."""

    def __init__(self, setup, image, rng, path, condition, budget=4000, reach=None):
        self.setup, self.image, self.rng, self.path, self.condition = (
            setup,
            image,
            rng,
            path,
            condition,
        )
        self.reach = reach or (
            lambda state: (
                (
                    evaluate(
                        path,
                        list(state.registers),
                        state.nzcv,
                        state.sp,
                        Memory(state.memory, state.global_pages),
                        state.load_bias,
                    )[3]
                    - state.load_bias
                )
                & MASK64
            )
        )
        self.inputs = [("load", id) for id in setup["cone_loads"]] + [
            ("storage", storage) for storage in setup["cone_storage"]
        ]
        self.roots, self.combinations, self.searched, self.budget = None, [], 0, budget

    def forces(self):
        """One input set to a literal, or two inputs set equal, or nothing."""
        rng, inputs = self.rng, self.inputs
        choice = rng.random()
        if not inputs or choice < 0.3:
            return ()
        value = rng.choice(self.setup["literals"]) if rng.random() < 0.7 else rng.getrandbits(64)
        if choice < 0.7 or len(inputs) < 2:
            return ((*rng.choice(inputs), value),)
        return tuple((*item, value) for item in rng.sample(inputs, 2))

    def trace(self):
        # Which inputs reach the condition depends on aliasing, so the roots
        # are gathered over several states.
        found, traced_states = set(), 0
        for _ in range(32):
            traced = {}
            probe = make_state(self.setup, self.image, BIASES[0], self.rng, self.path)
            try:
                evaluate(
                    self.path,
                    list(probe.registers),
                    probe.nzcv,
                    probe.sp,
                    Memory(probe.memory, probe.global_pages),
                    probe.load_bias,
                    sources=traced,
                )
            except (Fault, Tagged, KeyError):
                continue
            found |= traced.get(self.condition, frozenset())
            traced_states += 1
            if traced_states == 8:
                break
        self.roots = sorted(found)
        if 0 < len(self.roots) <= 4:
            # The condition's own constants first, every literal after.
            for pool in (self.setup["literals"], self.setup["compared"]):
                tried = list(
                    itertools.islice(itertools.product(pool, repeat=len(self.roots)), 4000)
                )
                self.rng.shuffle(tried)
                self.combinations += tried
            # Two inputs a small distance apart: a counter against its bound.
            if len(self.roots) == 2 and all(kind == "load" for kind, _ in self.roots):
                (_, first), (_, second) = self.roots
                near = [((first, second), (second, (first, delta))) for delta in (-2, -1, 0, 1, 2)]
                near += [((second, first), (first, (second, delta))) for delta in (-2, -1, 0, 1, 2)]
                self.combinations += near * 4

    def states(self, wanted, count):
        if self.condition is not None and self.roots is None:
            self.trace()
        found = []
        while wanted and len(found) < count and self.searched < self.budget:
            self.searched += 1
            chosen = self.forces()
            if self.combinations:
                combination = self.combinations.pop()
                if isinstance(combination[0], tuple):
                    # Only the dependent load is forced; the other keeps its value.
                    chosen = (("load", *combination[1]),)
                else:
                    chosen = tuple((*root, value) for root, value in zip(self.roots, combination))
            state = make_state(
                self.setup, self.image, BIASES[self.searched % 2], self.rng, self.path, chosen
            )
            try:
                reached = self.reach(state)
            except (Fault, Tagged, KeyError):
                continue
            if reached in wanted:
                found.append(state)
        return found


def fragments_of(path):
    spans = sorted(
        {(source["source_address"], bytes.fromhex(source["bytes"])) for source in path["sources"]}
    )
    merged = []
    for address, code in spans:
        if merged and merged[-1][0] + len(merged[-1][1]) == address:
            merged[-1] = (merged[-1][0], merged[-1][1] + code)
        else:
            merged.append((address, code))
    return tuple(Fragment(address, code) for address, code in merged)


def branch(source, target):
    delta = target - source
    if delta % 4 or not -(1 << 27) <= delta < (1 << 27):
        raise Unsupported("direct branch out of range")
    return struct.pack("<I", 0x14000000 | ((delta >> 2) & 0x3FFFFFF))


def verify(job):
    artifact_path, image_path, index, target_states, seed = job
    artifact = json.loads(pathlib.Path(artifact_path).read_text())
    image = Image(image_path)
    image.frozen += [
        (item["address"], item["bytes"]) for item in artifact.get("declared_constant_ranges", [])
    ]
    transition = artifact["unflattening"]["transitions"][index]
    region = artifact["regions"][transition["candidate"]]
    record = {
        "candidate": transition["candidate"],
        "entry": transition["entry"],
        "destinations": transition["destinations"],
        "conditional": transition["condition"] is not None,
        "entry_relation_dependency": transition.get("entry_relation_dependency", False),
    }
    try:
        path = region["after"]
        setup = plan(transition, region, image)
        fragments = fragments_of(path)
        if len(fragments) > 16 or sum(len(fragment.code) for fragment in fragments) > PAGE:
            raise Unsupported("route exceeds the oracle fragment budget")
        inside = lambda address: any(
            f.source_address <= address < f.source_address + len(f.code) for f in fragments
        )
        exits = list(transition["destinations"])
        for boundary in region["control"]["boundaries"]:
            for edge in boundary["edges"]:
                if edge["target_kind"] == "image_location" and not inside(edge["target_address"]):
                    exits.append(edge["target_address"])
        # When the route ends in a register BR with no other effect, the
        # original stops on the BR and its register is held to the recovered
        # destination: a destination the route passes through, or a wrong
        # route that loops back into itself, cannot be a landing.
        last = path["sources"][-1]
        word = int.from_bytes(bytes.fromhex(last["bytes"]), "little")
        stop = None
        full_route, full_exits = fragments, tuple(dict.fromkeys(exits))[:16]
        if (
            word & 0xFFFFFC1F == 0xD61F0000
            and not path["boundaries"][-1]["writes"]
            and sum(
                source["source_address"] == last["source_address"] for source in path["sources"]
            )
            == 1
        ):
            stop = (last["source_address"], (word >> 5) & 31)
            fragments = fragments_of({"sources": path["sources"][:-1]})
            inside = lambda address: any(
                f.source_address <= address < f.source_address + len(f.code) for f in fragments
            )
            exits = [last["source_address"]] + [address for address in exits if not inside(address)]
        elif any(inside(destination) for destination in transition["destinations"]):
            raise Unsupported("a destination lies inside the route")
        exits = tuple(dict.fromkeys(exits))[:16]
        oracle = SparseOracle()
        entry = path["sources"][0]["source_address"]
        rng = random.Random(seed * 1000003 + transition["entry"])
        matched = {destination: 0 for destination in transition["destinations"]}
        completed = faulted = 0
        surrogate_states = []

        def missing():
            return [
                destination
                for destination, count in matched.items()
                if count < (4 if record["conditional"] else 1)
            ]

        search = Search(setup, image, rng, path, transition["condition"])
        faults, timed_out, tagged = set(), 0, 0
        for batch in range(10):
            if batch < 2:
                states = [
                    make_state(setup, image, BIASES[k % 2], rng, path, search.forces())
                    for k in range(target_states)
                ]
            else:
                states = search.states(missing(), target_states)
                if not states:
                    break
            try:
                outcomes = oracle.run_many(fragments, entry, exits, states, trusted_fixture=True)
            except OracleError:
                # One state at a time. Only the runner's exit 21 means the
                # original trapped somewhere no exit names; a deadline is an
                # unfinished run, and any other failure is the harness's.
                outcomes = []
                for state in states:
                    try:
                        outcomes.append(
                            oracle.run(fragments, entry, exits, state, trusted_fixture=True)
                        )
                    except OracleError as error:
                        if "deadline" in str(error):
                            outcomes.append(None)
                            continue
                        if "failed (21)" in str(error) or "unlisted sparse exit" in str(error):
                            record.update(
                                disposition="refuted",
                                reason="the original left through a place the transition does not name",
                            )
                        else:
                            record.update(
                                disposition="inconclusive",
                                reason="oracle failure: " + str(error)[:200],
                            )
                        return record
            for state, outcome in zip(states, outcomes):
                if outcome is None:
                    timed_out += 1
                    continue
                if not outcome.completed:
                    faulted += 1
                    faults.add((outcome.exit_pc - state.load_bias) & MASK64)
                    continue
                completed += 1
                memory = Memory(state.memory, state.global_pages)
                try:
                    registers, flags, sp, pc, values = evaluate(
                        path, list(state.registers), state.nzcv, state.sp, memory, state.load_bias
                    )
                except Tagged:
                    tagged += 1
                    completed -= 1
                    continue
                except Fault as error:
                    record.update(
                        disposition="refuted",
                        reason="recovered path failed where the original completed: " + str(error),
                    )
                    return record
                except KeyError as error:
                    raise Unsupported(
                        "the recovered path names a value it never computes: %s" % error
                    )
                expected = outcome.state
                reached = outcome.exit_pc if stop is None else expected.registers[stop[1]]
                if stop is not None and outcome.exit_pc != (state.load_bias + stop[0]) & MASK64:
                    record.update(
                        disposition="refuted", reason="the original never reached its final BR"
                    )
                    return record
                if (
                    registers != expected.registers
                    or flags != expected.nzcv
                    or sp != expected.sp
                    or bytes(memory.regions[0][1]) != expected.memory
                    or pc != reached
                    or any(
                        bytes(content) != page.data
                        for (_, content, _), page in zip(memory.regions[1:], expected.global_pages)
                    )
                ):
                    record.update(
                        disposition="refuted",
                        reason="state or next PC differs",
                        detail={
                            "bias": state.load_bias,
                            "expected_pc": reached,
                            "pc": pc,
                            "registers": [
                                k for k in range(31) if registers[k] != expected.registers[k]
                            ],
                            "nzcv": [flags, expected.nzcv],
                            "sp": [sp, expected.sp],
                        },
                    )
                    return record
                taken = (pc - state.load_bias) & MASK64
                if taken not in matched:
                    record.update(
                        disposition="refuted",
                        reason="the next PC is not a published destination",
                        detail={"taken": taken},
                    )
                    return record
                if record["conditional"]:
                    holds = values.get(transition["condition"], 0) & 1
                    if taken != transition["destinations"][0 if holds else 1]:
                        record.update(
                            disposition="refuted",
                            reason="the condition does not select the published destination",
                        )
                        return record
                matched[taken] += 1
                surrogate_states.append((state, outcome))
            if completed >= 8 and not missing():
                break
        record.update(
            completed=completed, faulted=faulted, arms={str(k): v for k, v in matched.items()}
        )
        record["timed_out"] = timed_out
        record["tagged"] = tagged
        # A direct branch in place of the BR must leave the full outcome of the
        # original unchanged, which needs every destination outside the route.
        landing = not any(
            f.source_address <= destination < f.source_address + len(f.code)
            for f in full_route
            for destination in transition["destinations"]
        )
        if not record["conditional"] and surrogate_states and landing:
            source = last["source_address"]
            patched = []
            for fragment in full_route:
                code = fragment.code
                if fragment.source_address <= source < fragment.source_address + len(code):
                    at = source - fragment.source_address
                    code = (
                        code[:at] + branch(source, transition["destinations"][0]) + code[at + 4 :]
                    )
                patched.append(Fragment(fragment.source_address, code))
            states = [state for state, _ in surrogate_states[:16]]
            try:
                original = oracle.run_many(
                    full_route, entry, full_exits, states, trusted_fixture=True
                )
                direct = oracle.run_many(
                    tuple(patched), entry, full_exits, states, trusted_fixture=True
                )
            except OracleError as error:
                raise Unsupported("the direct-branch check did not run: " + str(error)[:200])
            record["direct_branch_matches"] = original == direct
            if not record["direct_branch_matches"]:
                record.update(
                    disposition="refuted",
                    reason="a direct branch to the destination changed the outcome",
                )
                return record
        record["searched"] = search.searched
        if completed == 0 and faulted:
            record.update(
                disposition="inconclusive",
                reason="every original run faulted",
                fault_sources=sorted(faults)[:4],
            )
        elif completed < 8:
            record.update(
                disposition="inconclusive", reason="fewer than eight completed original runs"
            )
        elif any(count == 0 for count in matched.values()):
            record.update(
                disposition="inconclusive",
                reason="a destination was never taken, even in %d searched states"
                % search.searched,
            )
        elif record["conditional"] and min(matched.values()) < 4:
            record.update(
                disposition="inconclusive", reason="a destination was taken fewer than four times"
            )
        else:
            record.update(disposition="verified", reason=None)
    except (Unsupported, Declined) as error:
        record.update(disposition="inconclusive", reason=str(error))
    return record


def verify_chain(job):
    """Two transitions in a row: the original runs from A's entry through B under
    QEMU, while A's recovered path runs and then B's from whatever A left. B's
    entry is never constructed, so a precondition of B that A does not actually
    establish, such as an entry relation, shows up as a mismatch."""
    artifact_path, image_path, (first, second), target_states, seed = job
    artifact = json.loads(pathlib.Path(artifact_path).read_text())
    image = Image(image_path)
    image.frozen += [
        (item["address"], item["bytes"]) for item in artifact.get("declared_constant_ranges", [])
    ]
    transitions = artifact["unflattening"]["transitions"]
    a, b = transitions[first], transitions[second]
    ra, rb = artifact["regions"][a["candidate"]], artifact["regions"][b["candidate"]]
    pa, pb = ra["after"], rb["after"]
    record = {"first": a["entry"], "second": b["entry"], "final_destinations": b["destinations"]}
    try:
        # A's proved relations hold on every arrival the graph admits, so A's
        # state keeps them; whether B's hold is then up to A.
        setup, second_setup = plan(a, ra, image, True), plan(b, rb, image, False)
        # B's own pointers usually pass through A untouched, so they are placed too.
        setup["pages"] = sorted(set(setup["pages"]) | set(second_setup["pages"]))
        setup["registers"] = sorted(set(setup["registers"]) | set(second_setup["registers"]))
        setup["offsets"] = {
            storage: sorted(
                set(setup["offsets"].get(storage, []))
                | set(second_setup["offsets"].get(storage, []))
            )
            for storage in set(setup["offsets"]) | set(second_setup["offsets"])
        }
        if len(setup["pages"]) > MAX_GLOBAL_PAGES:
            raise Unsupported("needs more than eight global pages")
        fragments = fragments_of({"sources": pa["sources"] + pb["sources"]})
        if len(fragments) > 16 or sum(len(fragment.code) for fragment in fragments) > PAGE:
            raise Unsupported("the chain exceeds the oracle fragment budget")
        inside = lambda address: any(
            f.source_address <= address < f.source_address + len(f.code) for f in fragments
        )
        if any(inside(destination) for destination in b["destinations"]):
            raise Unsupported("a destination of the second transition lies inside the chain")
        exits = list(b["destinations"]) + [
            d for d in a["destinations"] if d != b["entry"] and not inside(d)
        ]
        for region in (ra, rb):
            for boundary in region["control"]["boundaries"]:
                for edge in boundary["edges"]:
                    if edge["target_kind"] == "image_location" and not inside(
                        edge["target_address"]
                    ):
                        exits.append(edge["target_address"])
        exits = tuple(dict.fromkeys(exits))[:16]
        oracle, rng = SparseOracle(), random.Random(seed * 1000003 + a["entry"] * 31 + b["entry"])
        entry = pa["sources"][0]["source_address"]
        matched = tried = 0

        def chained(state):
            # Where B's recovered path goes, for a state in which A's heads for B.
            memory = Memory(state.memory, state.global_pages)
            registers, flags, sp, pc, _ = evaluate(
                pa, list(state.registers), state.nzcv, state.sp, memory, state.load_bias
            )
            if (pc - state.load_bias) & MASK64 != b["entry"]:
                return None
            return (
                evaluate(pb, list(registers), flags, sp, memory, state.load_bias)[3]
                - state.load_bias
            ) & MASK64

        # Every arm of B is wanted, each from states in which A heads for B.
        arms = {destination: 0 for destination in b["destinations"]}
        search = Search(setup, image, rng, pa, a["condition"], reach=chained)
        matched = tried = 0
        for _ in range(16):
            states = search.states([d for d, count in arms.items() if count < 2], target_states)
            if not states:
                break
            outcomes = []
            for state in states:
                # One state at a time: only exit 21 is a landing nothing names.
                try:
                    outcomes.append(
                        oracle.run(fragments, entry, exits, state, trusted_fixture=True)
                    )
                except OracleError as error:
                    if "deadline" in str(error):
                        outcomes.append(None)
                        continue
                    if "failed (21)" in str(error) or "unlisted sparse exit" in str(error):
                        record.update(
                            disposition="refuted",
                            reason="the original left the chain through a place neither transition names",
                        )
                        return record
                    raise Unsupported("oracle failure: " + str(error)[:200])
            for state, outcome in zip(states, outcomes):
                if outcome is None or not outcome.completed:
                    continue
                tried += 1
                memory = Memory(state.memory, state.global_pages)
                try:
                    registers, flags, sp, pc, _ = evaluate(
                        pa, list(state.registers), state.nzcv, state.sp, memory, state.load_bias
                    )
                    registers, flags, sp, pc, _ = evaluate(
                        pb, list(registers), flags, sp, memory, state.load_bias
                    )
                except Tagged:
                    tried -= 1
                    continue
                except KeyError as error:
                    raise Unsupported(
                        "a recovered path names a value it never computes: %s" % error
                    )
                except Fault as error:
                    record.update(
                        disposition="refuted",
                        reason="the chained paths failed where the original"
                        " completed: " + str(error),
                    )
                    return record
                expected = outcome.state
                if (
                    registers != expected.registers
                    or flags != expected.nzcv
                    or sp != expected.sp
                    or bytes(memory.regions[0][1]) != expected.memory
                    or pc != outcome.exit_pc
                    or any(
                        bytes(content) != page.data
                        for (_, content, _), page in zip(memory.regions[1:], expected.global_pages)
                    )
                ):
                    record.update(
                        disposition="refuted",
                        reason="the chain differs from the original",
                        detail={
                            "pc": pc,
                            "expected_pc": outcome.exit_pc,
                            "registers": [
                                k for k in range(31) if registers[k] != expected.registers[k]
                            ],
                        },
                    )
                    return record
                taken = (pc - state.load_bias) & MASK64
                if taken not in arms:
                    record.update(
                        disposition="refuted",
                        reason="the chain ends where the second transition does not say it goes",
                        detail={"taken": taken},
                    )
                    return record
                matched += 1
                arms[taken] += 1
            if matched >= 8 and min(arms.values()) >= 2:
                break
        record["arms"] = {str(k): v for k, v in arms.items()}
        record.update(completed=tried, matched=matched)
        if matched < 8:
            record.update(
                disposition="inconclusive", reason="fewer than eight completed chained runs"
            )
        elif min(arms.values()) < 2:
            record.update(
                disposition="inconclusive",
                reason="an arm of the second transition was not exercised",
            )
        else:
            record.update(disposition="verified", reason=None)
    except (Unsupported, Declined) as error:
        record.update(disposition="inconclusive", reason=str(error))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("artifact")
    parser.add_argument("--image", help="defaults to the artifact input")
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--chains",
        action="store_true",
        help="check each pair of transitions where one leads straight into the next",
    )
    arguments = parser.parse_args()
    artifact = json.loads(pathlib.Path(arguments.artifact).read_text())
    image = arguments.image or artifact["input"]
    transitions = artifact["unflattening"]["transitions"]
    if arguments.chains:
        by_entry = {t["entry"]: index for index, t in enumerate(transitions)}
        pairs = [
            (index, by_entry[d])
            for index, t in enumerate(transitions)
            for d in t["destinations"]
            if d in by_entry and d != t["entry"]
        ]
        jobs = [
            (arguments.artifact, image, pair, arguments.states, arguments.seed) for pair in pairs
        ]
        check = verify_chain
    else:
        jobs = [
            (arguments.artifact, image, index, arguments.states, arguments.seed)
            for index in range(len(transitions))
        ]
        check = verify
    with concurrent.futures.ProcessPoolExecutor(arguments.jobs) as pool:
        records = list(pool.map(check, jobs))
    summary = {}
    for record in records:
        summary[record["disposition"]] = summary.get(record["disposition"], 0) + 1
    json.dump(
        {
            "schema": 1,
            "artifact": arguments.artifact,
            "address": artifact["address"],
            "size": artifact["size"],
            "seed": arguments.seed,
            "summary": summary,
            "assumes": [
                "relocated pointer slots hold the value the loader writes and writable data"
                " starts from its file bytes, except inputs deliberately varied",
                "the artifact's declared constant ranges and RELRO keep their values",
            ],
            "transitions": records,
        },
        sys.stdout,
        indent=1,
    )
    print()


if __name__ == "__main__":
    main()
