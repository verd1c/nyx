#!/usr/bin/env python3
import collections
import copy
import json
import pathlib
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.oracle.control import ControlOracle, ControlState, DEFAULT_PC

SOURCE = 0x400080
MASK64 = (1 << 64) - 1


def assemble(oracle, text):
    with tempfile.TemporaryDirectory(prefix="nyx-cfg-assembly-") as directory:
        path = pathlib.Path(directory)
        (path / "source.S").write_text(".text\n" + text + "\n")
        oracle.backend._command(
            [oracle.backend.compiler, "-c", "-o", str(path / "source.o"), str(path / "source.S")]
        )
        oracle.backend._command(
            [
                oracle.backend.objcopy,
                "-O",
                "binary",
                "-j",
                ".text",
                str(path / "source.o"),
                str(path / "source.bin"),
            ]
        )
        return (path / "source.bin").read_bytes()


def graph(binary, code):
    data = bytearray(128 + len(code))
    data[:16] = b"\x7fELF\x02\x01\x01" + bytes(9)
    struct.pack_into("<HHIQQQIHHHHHH", data, 16, 3, 183, 1, SOURCE, 64, 0, 0, 64, 56, 1, 0, 0, 0)
    struct.pack_into("<IIQQQQQQ", data, 64, 1, 5, 0, SOURCE - 128, 0, len(data), len(data), 4096)
    data[128:] = code
    with tempfile.TemporaryDirectory(prefix="nyx-cfg-input-") as directory:
        path = pathlib.Path(directory) / "fixture.so"
        path.write_bytes(data)
        result = subprocess.run(
            [binary, "cfg", str(path), "--address", hex(SOURCE), "--size", str(len(code))],
            capture_output=True,
            text=True,
            timeout=10,
        )
    if result.returncode != 0:
        raise AssertionError((result.returncode, result.stdout, result.stderr))
    report = json.loads(result.stdout)
    if b"".join(bytes.fromhex(source["bytes"]) for source in report["sources"]) != code:
        raise AssertionError("CFG changed original instruction provenance")
    return report


def values(block, state, bias):
    # A small independent integer interpreter for printed predicate DAGs only;
    # no Nyx decoder/evaluator imports and no memory/callee execution model.
    cells = dict(enumerate(state.registers))
    cells[31] = state.sp
    cells.update({32 + i: (state.nzcv >> (31 - i)) & 1 for i in range(4)})
    result = []
    for node in block["ssa"]["nodes"]:
        op = node["op"]
        args = [result[index] for index in node["inputs"]]
        if op == "constant":
            value = node["immediate"]
        elif op == "image_address":
            value = bias + node["immediate"]
        elif op == "read":
            value = cells[node["storage"]]
        elif op == "add":
            value = args[0] + args[1]
        elif op == "sub":
            value = args[0] - args[1]
        elif op == "bit_and":
            value = args[0] & args[1]
        elif op == "bit_or":
            value = args[0] | args[1]
        elif op == "bit_xor":
            value = args[0] ^ args[1]
        elif op == "bit_not":
            value = ~args[0]
        elif op == "equal":
            value = int(args[0] == args[1])
        elif op == "extract":
            value = args[0] >> node["immediate"]
        elif op == "zext":
            value = args[0]
        elif op == "select":
            value = args[1] if args[0] else args[2]
        else:
            raise AssertionError(("unmodeled independent predicate operation", op))
        result.append(value & ((1 << node["width"]) - 1))
    return result


def selected(block, state, bias, edges=None):
    evaluated = values(block, state, bias)
    targets = []
    for edge in block["edges"] if edges is None else edges:
        if edge["kind"] not in ("branch", "callee", "return"):
            continue
        if edge["condition"] is not None and bool(evaluated[edge["condition"]]) != edge["when"]:
            continue
        target = edge["target"]
        if target["kind"] == "image_location":
            targets.append((target["address"] + bias) & MASK64)
        elif target["kind"] == "absolute_runtime":
            targets.append(target["address"])
        else:
            targets.append(None)
    return targets


def state(flags=0, value=0):
    registers = [0x128301 + index for index in range(31)]
    registers[0] = value
    registers[3] = DEFAULT_PC + 96
    registers[30] = DEFAULT_PC + 64
    return ControlState(tuple(registers), flags << 28, 0, DEFAULT_PC)


def main():
    binary = sys.argv[1]
    oracle = ControlOracle()
    counts = collections.Counter()
    mutations = 0
    conditions = (
        "eq",
        "ne",
        "cs",
        "cc",
        "mi",
        "pl",
        "vs",
        "vc",
        "hi",
        "ls",
        "ge",
        "lt",
        "gt",
        "le",
        "al",
        "nv",
    )
    cases = [
        (f"b.{condition} .+32", [state(flags) for flags in range(16)]) for condition in conditions
    ]
    cases += [
        (
            f"{mnemonic} {width}0, .+32",
            [state(value=value) for value in (0, 1, 1 << 31, 1 << 32, 1 << 63)],
        )
        for mnemonic in ("cbz", "cbnz")
        for width in ("w", "x")
    ]
    cases += [
        (f"{mnemonic} x0, #{bit}, .+32", [state(value=value) for value in (0, 1 << bit)])
        for mnemonic in ("tbz", "tbnz")
        for bit in (0, 31, 32, 63)
    ]
    cases += [
        (text, [state(0), state(15, MASK64)])
        for text in (
            "b.eq .-32",
            "cbnz x0, .-32",
            "tbnz x0, #63, .-32",
            "b .+32",
            "b .-32",
            "bl .+32",
        )
    ]
    bias = DEFAULT_PC - SOURCE
    for text, states in cases:
        code = assemble(oracle, text)
        block = graph(binary, code)["blocks"][0]
        observed = oracle.run_many(code, states)
        for current, outcome in zip(states, observed, strict=True):
            if selected(block, current, bias) != [outcome.landing_pc]:
                raise AssertionError((text, current, block, outcome))
            counts[text.split()[0]] += 1
        guarded = [edge for edge in block["edges"] if edge["condition"] is not None]
        if len(guarded) == 2 and guarded[0]["target"]["address"] != guarded[1]["target"]["address"]:
            mutant = copy.deepcopy(block["edges"])
            for edge in mutant:
                if edge["when"] is not None:
                    edge["when"] = not edge["when"]
            if not any(
                selected(block, current, bias, mutant) != [outcome.landing_pc]
                for current, outcome in zip(states, observed, strict=True)
            ):
                raise AssertionError(
                    "swapped predicates retained the same condition-to-destination relation"
                )
            mutations += 1
        if text.startswith("bl "):
            continuation = [edge for edge in block["edges"] if edge["kind"] == "potential_return"]
            if len(continuation) != 1 or continuation[0]["target"]["address"] != SOURCE + 4:
                raise AssertionError("call continuation link metadata missing")
            if not block["control"]["callee_return_unknown"]:
                raise AssertionError("call incorrectly proves return")
            if any(outcome.state.registers[30] != DEFAULT_PC + 4 for outcome in observed):
                raise AssertionError("native call link capture failed")
    for text in ("b.eq .+4", "cbz x0, .+4", "tbz x0, #63, .+4"):
        code = assemble(oracle, text)
        block = graph(binary, code)["blocks"][0]
        edges = [edge for edge in block["edges"] if edge["condition"] is not None]
        if len(edges) != 2 or {edge["when"] for edge in edges} != {False, True}:
            raise AssertionError("same-destination conditional arms were collapsed")
        states = [state(0), state(15, MASK64)]
        for current, outcome in zip(states, oracle.run_many(code, states), strict=True):
            if selected(block, current, bias) != [outcome.landing_pc]:
                raise AssertionError(text)
            counts["same_target_arms"] += 1
    for text, role in (("br x3", "branch"), ("blr x3", "callee"), ("ret", "return")):
        code = assemble(oracle, text)
        block = graph(binary, code)["blocks"][0]
        edges = [edge for edge in block["edges"] if edge["kind"] == role]
        if len(edges) != 1 or edges[0]["target"]["kind"] != "unknown":
            raise AssertionError("unseeded indirect target was guessed")
        observed = oracle.run(code, state())
        expected = state().registers[30 if text == "ret" else 3]
        if observed.landing_pc != expected:
            raise AssertionError("indirect native reference mismatch")
        counts["unresolved_indirect"] += 1
    code = assemble(oracle, "adr x3, .+32\nbr x3")
    block = graph(binary, code)["blocks"][0]
    registers = list(state().registers)
    registers[3] = DEFAULT_PC + 28
    at_branch = ControlState(tuple(registers), state().nzcv, state().sp, DEFAULT_PC)
    observed = oracle.run(code[4:], at_branch)
    if selected(block, state(), bias - 4) != [observed.landing_pc]:
        raise AssertionError("local ADR provenance failed to resolve BR target")
    print(
        json.dumps(
            {
                "native_condition_destination_comparisons": dict(counts),
                "swapped_guard_mutations_refuted": mutations,
                "local_adr_br_checked": True,
                "local_adr_br_native_scope": "terminal BR with independently derived ADR target",
                "graph_reachability_proved": False,
                "callee_execution_checked": False,
                "hardware_checked": False,
                "tools": oracle.versions(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
