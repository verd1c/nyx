#!/usr/bin/env python3
import pathlib
import subprocess
import sys
import tempfile


def run(command):
    return subprocess.run(command, capture_output=True, text=True, check=True, timeout=30).stdout


def main():
    source = pathlib.Path(__file__).resolve().parents[1] / "tools/oracle/runner_exclusive.c"
    with tempfile.TemporaryDirectory(prefix="nyx-exclusive-") as directory:
        runner = pathlib.Path(directory) / "runner"
        subprocess.run(
            [
                "aarch64-linux-gnu-gcc",
                "-std=gnu11",
                "-O0",
                "-static",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(source),
                "-o",
                str(runner),
            ],
            check=True,
            timeout=30,
        )
        expected = run(["qemu-aarch64", str(runner)]).splitlines()
        actual = run([sys.argv[1]]).splitlines()
        labels = [f"{width}{mode}" for width in "wx" for mode in range(5)] + [
            f"f{mode}" for mode in range(3)
        ]
        if [row.split(" ", 1)[0] for row in expected] != labels:
            raise AssertionError(f"QEMU fixture did not exercise every sequence: {expected}")
        if [row.split(" ", 1)[0] for row in actual] != labels:
            raise AssertionError(f"Nyx probe did not exercise every sequence: {actual}")
        if expected != actual:
            raise AssertionError(f"exclusive reference mismatch: QEMU={expected}, Nyx={actual}")
        print(f"exclusive scalar: {len(actual)} independent QEMU/Nyx sequences")


if __name__ == "__main__":
    main()
