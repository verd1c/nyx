#!/usr/bin/env python3
import json
import pathlib
import subprocess
import sys
import tempfile


def run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
    return json.loads(result.stdout)


def main():
    probe = sys.argv[1]
    source = pathlib.Path(__file__).resolve().parents[1] / "tools/oracle/runner_q_memory.c"
    with tempfile.TemporaryDirectory(prefix="nyx-q-memory-") as directory:
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
        count = 0
        for operation in ("load", "store"):
            for mapped in range(1, 17):
                for variant in ("base", "max", "q31"):
                    args = [operation, str(mapped), variant]
                    expected = run(["qemu-aarch64", str(runner), *args])
                    actual = run([probe, *args])
                    if actual != expected:
                        raise AssertionError(
                            f"QEMU mismatch {args}: expected={expected}, actual={actual}"
                        )
                    count += 1
        print(f"Q memory: {count} independent QEMU/Nyx comparisons")


if __name__ == "__main__":
    main()
