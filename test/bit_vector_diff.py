#!/usr/bin/env python3
import random
import subprocess
import sys


def reference(width, op, lhs, rhs, count):
    mask = (1 << width) - 1
    if op == "add":
        value = lhs + rhs
    elif op == "sub":
        value = lhs - rhs
    elif op == "mul":
        value = lhs * rhs
    elif op == "and":
        value = lhs & rhs
    elif op == "or":
        value = lhs | rhs
    elif op == "xor":
        value = lhs ^ rhs
    elif op == "not":
        value = ~lhs
    elif op == "shl":
        value = 0 if count >= width else lhs << count
    elif op == "lshr":
        value = 0 if count >= width else lhs >> count
    elif op == "ashr":
        signed = lhs - (1 << width) if lhs & (1 << (width - 1)) else lhs
        value = (-1 if signed < 0 else 0) if count >= width else signed >> count
    elif op == "rotl":
        shift = count % width
        value = (lhs << shift) | (lhs >> (width - shift))
    elif op == "rotr":
        shift = count % width
        value = (lhs >> shift) | (lhs << (width - shift))
    else:
        raise AssertionError(op)
    return f"{value & mask:x}"


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: bit_vector_diff.py PROBE_EXECUTABLE")
    rng = random.Random(0x7AE395)
    widths = sorted(
        set(
            [
                1,
                2,
                7,
                8,
                9,
                31,
                32,
                33,
                63,
                64,
                65,
                127,
                128,
                129,
                255,
                256,
                257,
                511,
                512,
                513,
                1023,
                1024,
                1025,
                2047,
                2048,
                2049,
                4095,
                4096,
            ]
            + [rng.randint(1, 4096) for _ in range(24)]
        )
    )
    cases = []
    expected = []
    for width in widths:
        mask = (1 << width) - 1
        # (0, 1) and (1, 0) make a borrow propagate out of the lowest limb and
        # through every limb above it; random limbs reach that with probability
        # about 2**-64 per limb, so the pairs above never exercise it.
        operands = [
            (0, mask),
            (mask, 1),
            (1 << (width - 1), mask),
            (0, 1),
            (1, 0),
            (1 << 64 if width > 64 else 0, (1 << 64) - 1 if width > 64 else mask),
            (rng.getrandbits(width), rng.getrandbits(width)),
            (rng.getrandbits(width), rng.getrandbits(width)),
        ]
        for lhs, rhs in operands:
            for op in ("add", "sub", "mul", "and", "or", "xor", "not"):
                cases.append(f"{width} {op} {lhs:x} {rhs:x} 0")
                expected.append(reference(width, op, lhs, rhs, 0))
            for count in sorted(
                set([0, 1, 63, 64, 65, width - 1, width, width + 1, (1 << 64) - 1])
            ):
                for op in ("shl", "lshr", "ashr", "rotl", "rotr"):
                    cases.append(f"{width} {op} {lhs:x} 0 {count}")
                    expected.append(reference(width, op, lhs, 0, count))
    malformed = [
        ("0 add 0 0 0", "error:invalid_width"),
        ("4097 add 0 0 0", "error:width_limit"),
        ("64 add 10000000000000000 0 0", "error:excess_words"),
        ("64 add xyz 0 0", "error:parse"),
        ("64 nope 0 0 0", "error:operation"),
        ("64 shl 0 0 18446744073709551616", "error:parse"),
        ("64 add 0 0 0 extra", "error:parse"),
        ("x" * 3000, "error:line_limit"),
        ("64 add 1 2 0", "3"),
    ]
    for request, answer in malformed:
        cases.append(request)
        expected.append(answer)
    process = subprocess.run(
        [sys.argv[1]],
        input="\n".join(cases) + "\n",
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )
    actual = process.stdout.splitlines()
    if len(actual) != len(expected):
        raise AssertionError(f"expected {len(expected)} lines, got {len(actual)}; {process.stderr}")
    for index, (wanted, got) in enumerate(zip(expected, actual)):
        if wanted != got:
            raise AssertionError(f"case {index}: {cases[index]}\nexpected {wanted}\nactual {got}")
    print(
        f"Python integer differential: {len(cases)} cases passed across {len(widths)} widths (1..4096 bits)"
    )


if __name__ == "__main__":
    main()
