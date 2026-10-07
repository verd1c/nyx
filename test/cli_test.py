import json
import copy
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

BINARY = sys.argv.pop(1)


class CliTest(unittest.TestCase):
    def run_cli(self, *args):
        result = subprocess.run(
            [BINARY, *args], capture_output=True, text=True, check=False, timeout=5
        )
        return result.returncode, json.loads(result.stdout)

    def elf(self, code, machine=183, flags=5, memory_extra=0, big=False):
        data = bytearray(128 + len(code))
        data[:16] = b"\x7fELF\x02" + bytes([2 if big else 1, 1]) + bytes(9)
        endian = ">" if big else "<"
        struct.pack_into(
            endian + "HHIQQQIHHHHHH",
            data,
            16,
            3,
            machine,
            1,
            0x400080,
            64,
            0,
            0,
            64,
            56,
            1,
            0,
            0,
            0,
        )
        struct.pack_into(
            endian + "IIQQQQQQ",
            data,
            64,
            1,
            flags,
            0,
            0x400000,
            0,
            len(data),
            len(data) + memory_extra,
            4096,
        )
        data[128:] = code
        return data

    def lift_fixture(self, data, address="0x400080", size="12"):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lift.so"
            path.write_bytes(data)
            return self.run_cli("lift", str(path), "--address", address, "--size", size)

    def cfg_fixture(self, data, address="0x400080", size="12"):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cfg.so"
            path.write_bytes(data)
            return self.run_cli("cfg", str(path), "--address", address, "--size", size)

    def test_cfg_retains_opaque_sources_without_inventing_fallthrough(self):
        code = bytes.fromhex("2000028b200040f91f2003d5")  # add, unadmitted ldr, nop
        status, report = self.cfg_fixture(self.elf(code))
        self.assertEqual(status, 0)
        self.assertEqual(report["outcome"], "cfg")
        self.assertEqual(report["source_groups"], 3)
        self.assertEqual(report["modeled_groups"], 2)
        self.assertEqual(report["unmodeled_groups"], 1)
        self.assertEqual(
            [source["status"] for source in report["sources"]],
            ["modeled", "unsupported", "modeled"],
        )
        self.assertEqual(
            b"".join(bytes.fromhex(source["bytes"]) for source in report["sources"]), code
        )
        self.assertEqual(
            [source["address"] for source in report["sources"]], [0x400080, 0x400084, 0x400088]
        )
        opaque = next(block for block in report["blocks"] if block["entry"] == 0x400084)
        self.assertIsNone(opaque["ssa"])
        self.assertIsNone(opaque["control"])
        self.assertEqual(len(opaque["edges"]), 1)
        self.assertEqual(opaque["edges"][0]["kind"], "opaque_unknown")
        self.assertEqual(opaque["edges"][0]["resolution"], "unknown")
        self.assertIsNone(opaque["edges"][0]["target_block"])
        self.assertEqual(report["blocks"][0]["edges"][0]["resolution"], "opaque_entry")
        self.assertEqual(report["partition"], "known_entries_only")
        self.assertEqual(report["reachability"], "not_proved")
        self.assertEqual(report["interior_entries"], "not_excluded")
        self.assertFalse(report["whole_function_recovery"])
        self.assertEqual(report["deobfuscation"], "not_performed")

    def test_exclusive_scalar_accesses_declare_only_control_fallthrough(self):
        for name, word in [("ldaxr", "0afc5f88"), ("stxr", "097d0b88"), ("stlxr", "09fd0b88")]:
            with self.subTest(name=name):
                code = bytes.fromhex("1f2003d5" + word + "1f2003d5")
                image = self.elf(code)
                status, cfg = self.cfg_fixture(image)
                self.assertEqual(status, 0)
                self.assertEqual(cfg["sources"][1]["status"], "unsupported")
                self.assertEqual(cfg["sources"][1]["opaque_control"], "declared_normal_fallthrough")
                opaque = cfg["blocks"][1]
                self.assertIsNone(opaque["ssa"])
                self.assertIsNone(opaque["control"])
                self.assertEqual(opaque["edges"][0]["kind"], "fallthrough")
                self.assertEqual(opaque["edges"][0]["target"]["address"], 0x400088)
                self.assertEqual(opaque["edges"][0]["target_block"], 2)
                with tempfile.TemporaryDirectory() as temp:
                    path = Path(temp) / (name + ".so")
                    path.write_bytes(image)
                    recovered_status, recovered = self.run_cli(
                        "recover-regions", str(path), "--address", "400080", "--size", "12"
                    )
                self.assertEqual(recovered_status, 0)
                blocks = recovered["unflattening"]["recovered_graph"]["blocks"]
                declared = next(block for block in blocks if block["address"] == 0x400084)
                self.assertIn(
                    "caller-declared opaque control behavior matches the source instruction",
                    declared["edges"][0]["assumes"],
                )

    def test_breakpoint_declares_a_trap_without_successor(self):
        code = bytes.fromhex("1f2003d5" + "200020d4" + "1f2003d5")  # nop, brk #1, nop
        image = self.elf(code)
        status, cfg = self.cfg_fixture(image)
        self.assertEqual(status, 0)
        self.assertEqual(cfg["sources"][1]["status"], "unsupported")
        self.assertEqual(cfg["sources"][1]["opaque_control"], "declared_trap")
        trap = cfg["blocks"][1]
        self.assertIsNone(trap["ssa"])
        self.assertEqual(
            [(edge["kind"], edge["target_block"]) for edge in trap["edges"]], [("trap", None)]
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "brk.so"
            path.write_bytes(image)
            status, recovered = self.run_cli(
                "recover-regions", str(path), "--address", "400080", "--size", "12"
            )
        self.assertEqual(status, 0)
        unflattening = recovered["unflattening"]
        self.assertEqual(unflattening["unresolved_successors_from"], [])
        declared = next(
            block
            for block in unflattening["recovered_graph"]["blocks"]
            if block["address"] == 0x400084
        )
        self.assertEqual(
            declared["edges"],
            [
                {
                    "kind": "trap",
                    "target": None,
                    "condition": None,
                    "condition_owner": None,
                    "when": None,
                    "assumes": [
                        "caller-declared opaque control behavior matches the source instruction"
                    ],
                }
            ],
        )

    def test_exclusive_profile_publishes_explicit_effects(self):
        code = bytes.fromhex("0afd5f8809fd0b885f3f03d5")  # ldaxr, stlxr, clrex
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "exclusive.so"
            path.write_bytes(self.elf(code))
            status, report = self.run_cli(
                "cfg",
                str(path),
                "--address",
                "400080",
                "--size",
                "12",
                "--memory-profile",
                "concrete_exclusive_scalar",
            )
        self.assertEqual(status, 0)
        self.assertEqual(report["memory_profile"], "concrete_exclusive_scalar")
        self.assertEqual([source["status"] for source in report["sources"]], ["modeled"] * 3)
        self.assertEqual(report["unmodeled_groups"], 0)
        operations = [node["op"] for node in report["blocks"][0]["ssa"]["nodes"]]
        self.assertEqual(
            [op for op in operations if op.startswith("exclusive_")],
            ["exclusive_load", "exclusive_store", "exclusive_clear"],
        )

    def test_cfg_keeps_guarded_same_destination_edges(self):
        code = bytes.fromhex("200000541f2003d5")  # b.eq .+4; nop
        status, report = self.cfg_fixture(self.elf(code), size="8")
        self.assertEqual(status, 0)
        edges = report["blocks"][0]["edges"]
        self.assertEqual(len(edges), 2)
        self.assertEqual([edge["kind"] for edge in edges], ["branch", "branch"])
        self.assertEqual([edge["when"] for edge in edges], [True, False])
        self.assertIsInstance(edges[0]["condition"], int)
        self.assertEqual(edges[0]["condition"], edges[1]["condition"])
        self.assertEqual([edge["target"]["address"] for edge in edges], [0x400084, 0x400084])
        self.assertEqual(edges[0]["target_block"], edges[1]["target_block"])
        self.assertEqual(
            report["blocks"][0]["control"]["proof_scope"], "all_inputs_successful_terminal"
        )

    def test_cfg_call_continuation_does_not_certify_callee_return(self):
        code = bytes.fromhex("020000941f2003d5c0035fd6")  # bl .+8; nop; ret
        status, report = self.cfg_fixture(self.elf(code))
        self.assertEqual(status, 0)
        caller = report["blocks"][0]
        self.assertEqual(caller["control"]["kind"], "call")
        self.assertTrue(caller["control"]["callee_return_unknown"])
        self.assertEqual([edge["kind"] for edge in caller["edges"]], ["callee", "potential_return"])
        self.assertEqual(
            [edge["target"]["address"] for edge in caller["edges"]], [0x400088, 0x400084]
        )
        self.assertNotEqual(caller["edges"][0]["target_block"], caller["edges"][1]["target_block"])
        callee = report["blocks"][caller["edges"][0]["target_block"]]
        self.assertEqual(callee["control"]["kind"], "return")
        self.assertEqual(callee["edges"][0]["kind"], "return")
        self.assertEqual(callee["edges"][0]["target"]["kind"], "unknown")
        self.assertEqual(callee["edges"][0]["resolution"], "unknown")
        self.assertIsNone(callee["edges"][0]["target_block"])

    def test_cfg_malformed_inputs_publish_only_refusal(self):
        code = bytes.fromhex("1f2003d5")
        for data, address, size in [
            (b"\x7fELF", "0x400080", "4"),
            (self.elf(code), "0x400081", "4"),
            (self.elf(code), "0x400080", "3"),
            (self.elf(code), "0x400080", "8"),
            (self.elf(code), "0x400080junk", "4"),
            (self.elf(code, flags=4), "0x400080", "4"),
        ]:
            with self.subTest(address=address, size=size, data_size=len(data)):
                status, report = self.cfg_fixture(data, address, size)
                self.assertEqual(status, 1)
                self.assertEqual(report["outcome"], "refused")
                self.assertNotIn("blocks", report)
                self.assertNotIn("sources", report)
                self.assertNotIn("edges", report)

    def test_cfg_resource_refusal_publishes_no_partial_graph(self):
        code = bytes.fromhex("1f2003d5") * 4097
        status, report = self.cfg_fixture(self.elf(code), size=str(len(code)))
        self.assertEqual(status, 1)
        self.assertEqual(report["outcome"], "refused")
        self.assertEqual(report["reason"], "resource_limit")
        self.assertNotIn("blocks", report)
        self.assertNotIn("sources", report)
        self.assertNotIn("edges", report)

    def test_missing_input_fails(self):
        status, report = self.run_cli("inspect", "/this-input-must-not-exist/nyx.so")
        self.assertNotEqual(status, 0)
        self.assertEqual(report["reason"], "input_io")

    def test_unimplemented_verb_does_not_claim_success(self):
        status, report = self.run_cli("deobfuscate", "input.so")
        self.assertNotEqual(status, 0)
        self.assertEqual(report["outcome"], "refused")

    def test_manifest_without_section_headers(self):
        data = bytearray(256)
        data[:16] = b"\x7fELF\x02\x01\x01" + bytes(9)
        struct.pack_into(
            "<HHIQQQIHHHHHH", data, 16, 3, 183, 1, 0x400000, 64, 0, 0, 64, 56, 1, 0, 0, 0
        )
        struct.pack_into("<IIQQQQQQ", data, 64, 1, 5, 0, 0x400000, 0, 256, 512, 4096)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'quoted"é-€-😀-image.so'
            path.write_bytes(data)
            status, report = self.run_cli("inspect", str(path))
        self.assertEqual(status, 0)
        self.assertEqual(report["input"], str(path))
        self.assertEqual(bytes.fromhex(report["input_bytes_hex"]), os.fsencode(path))
        self.assertEqual(report["executable_file_bytes"], 256)
        self.assertEqual(report["segments"][0]["memory_size"], 512)
        self.assertEqual(report["classification"], "not_checked")
        self.assertEqual(report["relocations"], "not_checked")

    def test_truncated_input_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "truncated.so"
            path.write_bytes(b"\x7fELF")
            status, report = self.run_cli("inspect", str(path))
        self.assertNotEqual(status, 0)
        self.assertEqual(report["reason"], "malformed_elf")

    def test_nonregular_input_does_not_block(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.pipe"
            os.mkfifo(path)
            status, report = self.run_cli("inspect", str(path))
        self.assertNotEqual(status, 0)
        self.assertEqual(report["reason"], "input_io")

    def test_invalid_utf8_path_is_lossless(self):
        with tempfile.TemporaryDirectory() as temp:
            for invalid in [b"\xff", b"\xc0\x80", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"\xe2"]:
                with self.subTest(invalid=invalid):
                    path = Path(temp) / os.fsdecode(b"invalid-" + invalid + b".so")
                    path.write_bytes(Path(BINARY).read_bytes())
                    status, report = self.run_cli("inspect", str(path))
                    self.assertEqual(status, 0)
                    self.assertIsNone(report["input"])
                    self.assertEqual(bytes.fromhex(report["input_bytes_hex"]), os.fsencode(path))

    def test_lift_call_preserves_link_and_reports_execution_scope(self):
        code = bytes.fromhex("02000094")  # bl .+8
        status, report = self.lift_fixture(self.elf(code), size="4")
        self.assertEqual(status, 0)
        self.assertEqual(report["scope"], "register_and_control_semantics")
        self.assertEqual(report["callee_effects"], "not_modeled")
        self.assertEqual(report["control_protection"], "not_modeled")
        self.assertEqual(
            report["execution_assumptions"],
            ["BTI inactive", "GCS inactive", "pointer authentication inactive"],
        )
        self.assertEqual(report["deobfuscation"], "not_performed")
        group = report["groups"][0]
        self.assertEqual(group["bytes"], code.hex())
        self.assertEqual(group["transfer"]["kind"], "call")
        self.assertEqual(
            group["writes"], [{"storage": 30, "value": group["transfer"]["continuation"]}]
        )
        self.assertTrue(
            any(
                node["op"] == "image_address" and node["immediate"] == 0x400080
                for node in group["nodes"]
            )
        )

    def test_simplify_real_mba_preserves_unknown_load_and_source_boundaries(self):
        code = bytes.fromhex(
            "8982009029610d91290140f9e90309cb6ba48ed2ebd7a8f2"
            "cbaedef28bd4f1f22b010baa6bf97fd36ca48ed2ecd7a8f2"
            "ccaedef28cd4f1f229010cca690109cb0b0180d2"
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mba.so"
            path.write_bytes(self.elf(code))
            args = ("simplify", str(path), "--address", "0x400080", "--size", str(len(code)))
            # Without the profile the load has no modelled effect, so it joins
            # no run: it ends the run before it and the next one starts after
            # it. The arithmetic on either side is still recovered, and the
            # word left out is named rather than folded over.
            status, split = self.run_cli(*args)
            self.assertEqual(status, 0)
            self.assertEqual(split["straight_line_runs"], 2)
            self.assertEqual(split["unmodeled_source_groups"], 1)
            self.assertEqual(
                split["unmodeled"],
                [{"address": 0x400088, "bytes": "290140f9", "reason": "unsupported"}],
            )
            self.assertEqual(
                [(run["address"], run["size"]) for run in split["runs"]],
                [(0x400080, 8), (0x40008C, 56)],
            )
            self.assertNotIn("after", split)
            status, report = self.run_cli(*args, "--memory-profile", "concrete_atomic_scalar")
        self.assertEqual(status, 0)
        self.assertEqual(report["outcome"], "simplified")
        self.assertEqual(report["source_groups"], 17)
        self.assertEqual(report["straight_line_runs"], 1)
        self.assertEqual(report["unmodeled_source_groups"], 0)
        self.assertEqual(report["mba_expressions_simplified"], 1)
        self.assertEqual(report["constant_folds"], 12)
        self.assertEqual(report["applied_edits"], len(report["journal"]))
        self.assertEqual(report["memory_invariance"], "not_assumed")
        self.assertEqual(report["removed_memory_effects"], 0)
        self.assertFalse(report["architectural_fault_authority"])
        self.assertEqual(report["sp_alignment_check"], "disabled")
        self.assertEqual(report["removed_source_groups"], 0)
        self.assertEqual(report["interior_entry_analysis"], "not_performed")
        before, after = report["before"], report["after"]
        self.assertEqual(before["sources"], after["sources"])
        self.assertEqual(before["boundaries"], after["boundaries"])
        self.assertEqual(b"".join(bytes.fromhex(g["bytes"]) for g in after["sources"]), code)
        final_write = [
            w for boundary in after["boundaries"] for w in boundary["writes"] if w["storage"] == 9
        ][-1]
        expression = after["nodes"][final_write["value"]]
        self.assertEqual(expression["op"], "sub")
        constant, unknown = [after["nodes"][i] for i in expression["inputs"]]
        self.assertEqual(constant["op"], "constant")
        self.assertEqual(constant["immediate"], 0x8EA4F57646BF7523)
        self.assertEqual(unknown["op"], "load")
        self.assertEqual(after["sources"][unknown["origin"]["boundary"]]["bytes"], "290140f9")
        self.assertEqual(
            [edit["rule"] for edit in report["journal"] if edit["node"] == final_write["value"]],
            ["or_xor_sum", "negated_add"],
        )
        self.assertEqual(after["revision"], 1)
        self.assertEqual(
            {edit["rule"] for edit in report["journal"]},
            {"constant_fold", "or_xor_sum", "negated_add"},
        )

    def test_simplify_counts_negated_add_without_or_xor_pattern(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "negative-add.so"
            path.write_bytes(self.elf(bytes.fromhex("e00300cb0000018b")))
            status, report = self.run_cli(
                "simplify", str(path), "--address", "0x400080", "--size", "8"
            )
        self.assertEqual(status, 0)
        self.assertEqual(report["mba_expressions_simplified"], 1)
        self.assertEqual([edit["rule"] for edit in report["journal"]], ["negated_add"])

    def test_simplify_source_limit_publishes_no_partial_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "large-region.so"
            code = bytes.fromhex("1f2003d5") * 4097
            path.write_bytes(self.elf(code))
            status, report = self.run_cli(
                "simplify", str(path), "--address", "0x400080", "--size", str(len(code))
            )
        self.assertEqual(status, 1)
        self.assertEqual(report["reason"], "resource_limit")
        self.assertNotIn("after", report)
        self.assertNotIn("journal", report)

    def test_simplify_reports_no_gain_and_cuts_runs_at_interior_control(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "region.so"
            path.write_bytes(self.elf(bytes.fromhex("1f2003d5")))
            status, report = self.run_cli(
                "simplify", str(path), "--address", "0x400080", "--size", "4"
            )
            self.assertEqual(status, 0)
            self.assertEqual(report["outcome"], "unchanged")
            self.assertEqual(report["straight_line_runs"], 1)
            self.assertEqual(report["applied_edits"], 0)
            self.assertEqual(report["before"], report["after"])
            self.assertNotIn("runs", report)

            # Control before the region's final instruction ends a run rather
            # than refusing the region. Nothing is folded across the cut, and
            # the run after it is recovered as if entered at its first
            # instruction, which no cut establishes.
            path.write_bytes(self.elf(bytes.fromhex("010000941f2003d5")))
            status, report = self.run_cli(
                "simplify", str(path), "--address", "0x400080", "--size", "8"
            )
            self.assertEqual(status, 0)
            self.assertEqual(report["outcome"], "unchanged")
            self.assertEqual(report["straight_line_runs"], 2)
            self.assertEqual(report["unmodeled_source_groups"], 0)
            self.assertIn("entry at each run's first source group", report["execution_assumptions"])
            self.assertNotIn("before", report)
            self.assertNotIn("journal", report)
            self.assertEqual(
                [(run["address"], run["size"], run["source_groups"]) for run in report["runs"]],
                [(0x400080, 4, 1), (0x400084, 4, 1)],
            )
            self.assertEqual(
                [
                    [source["bytes"] for source in run["before"]["sources"]]
                    for run in report["runs"]
                ],
                [["01000094"], ["1f2003d5"]],
            )
            self.assertTrue(all(run["before"] == run["after"] for run in report["runs"]))

            # One run that does not reach every requested word still states its
            # own extent, so what was recovered is never inferred from the range.
            path.write_bytes(self.elf(bytes.fromhex("ffffffff1f2003d51f2003d5")))
            status, report = self.run_cli(
                "simplify", str(path), "--address", "0x400080", "--size", "12"
            )
            self.assertEqual(status, 0)
            self.assertEqual(report["straight_line_runs"], 1)
            self.assertEqual(report["unmodeled_source_groups"], 1)
            self.assertNotIn("before", report)
            self.assertEqual(
                [(run["address"], run["size"]) for run in report["runs"]], [(0x400084, 8)]
            )

            # A region whose every word is unmodelled leaves no run to recover.
            path.write_bytes(self.elf(bytes.fromhex("ffffffff")))
            status, report = self.run_cli(
                "simplify", str(path), "--address", "0x400080", "--size", "4"
            )
            self.assertEqual(status, 1)
            self.assertEqual(report["outcome"], "refused")
            self.assertEqual(report["reason"], "unsupported_region")
            self.assertNotIn("after", report)
            self.assertNotIn("journal", report)

    def test_lift_mixed_words_preserves_complete_source_accounting(self):
        code = bytes.fromhex("2000028b200040f91f2003d5")  # add, ldr, nop
        status, report = self.lift_fixture(self.elf(code))
        self.assertEqual(status, 2)
        self.assertEqual(report["outcome"], "partial")
        self.assertEqual(report["source_groups"], 3)
        self.assertEqual(report["supported_groups"], 2)
        self.assertEqual(report["unmodeled_groups"], 1)
        self.assertEqual(report["supported_source_bytes"], 8)
        self.assertEqual(report["unmodeled_source_bytes"], 4)
        self.assertEqual(report["deobfuscation"], "not_performed")
        self.assertEqual(
            [g["source_address"] for g in report["groups"]], [0x400080, 0x400084, 0x400088]
        )
        self.assertEqual(b"".join(bytes.fromhex(g["bytes"]) for g in report["groups"]), code)
        self.assertEqual(report["groups"][1]["reason"], "unsupported")
        self.assertEqual(report["groups"][1]["kind"], "nyx.ir.unmodeled")
        self.assertEqual(report["groups"][0]["kind"], "nyx.ir.group")
        self.assertTrue(report["groups"][0]["writes"])
        self.assertEqual(report["groups"][2]["nodes"], [])

    def test_lift_supported_and_reserved_words_distinguish_outcomes(self):
        status, report = self.lift_fixture(self.elf(bytes.fromhex("2000028b")), size="4")
        self.assertEqual(status, 0)
        self.assertEqual(report["outcome"], "lifted")
        self.assertEqual(report["supported_groups"], 1)
        self.assertEqual(report["unmodeled_groups"], 0)
        status, report = self.lift_fixture(self.elf(bytes.fromhex("2080020b")), size="4")
        self.assertEqual(status, 2)
        self.assertEqual(report["groups"][0]["reason"], "invalid_encoding")
        self.assertEqual(report["unmodeled_source_bytes"], 4)

    def test_lift_range_and_target_refusals(self):
        code = bytes.fromhex("1f2003d5") * 3
        for address, size in [
            ("400081", "4"),
            ("400080", "0"),
            ("400080", "3"),
            ("400080", "4194308"),
            ("fffffffffffffffc", "8"),
            ("-1", "4"),
            ("0x", "4"),
            ("400080junk", "4"),
        ]:
            with self.subTest(address=address, size=size):
                status, report = self.lift_fixture(self.elf(code), address, size)
                self.assertEqual(status, 1)
                self.assertEqual(report["outcome"], "refused")
                self.assertNotIn("groups", report)
        for data in [self.elf(code, machine=62), self.elf(code, big=True)]:
            status, report = self.lift_fixture(data)
            self.assertEqual(status, 1)
            self.assertEqual(report["reason"], "unsupported_target")
        status, report = self.lift_fixture(self.elf(code, flags=4))
        self.assertEqual(status, 1)
        self.assertEqual(report["reason"], "nonexecutable_range")
        status, report = self.lift_fixture(self.elf(code, memory_extra=16), size="16")
        self.assertEqual(status, 1)
        self.assertEqual(report["reason"], "unmapped_range")

    def test_lift_overlapping_mapping_is_refused(self):
        data = self.elf(bytes.fromhex("1f2003d5") * 32)
        struct.pack_into("<H", data, 56, 2)
        struct.pack_into("<IIQQQQQQ", data, 120, 1, 4, 128, 0x400080, 0, 16, 16, 1)
        status, report = self.lift_fixture(data)
        self.assertEqual(status, 1)
        self.assertEqual(report["reason"], "unmapped_range")

    def test_lift_budget_exhaustion_publishes_no_partial_artifact(self):
        code = bytes.fromhex("1f2003d5") * 120000
        status, report = self.lift_fixture(self.elf(code), size=str(len(code)))
        self.assertEqual(status, 1)
        self.assertEqual(report["reason"], "resource_limit")
        self.assertEqual(report["outcome"], "refused")
        self.assertNotIn("groups", report)

    def replay_recovery(self, report):
        replay = copy.deepcopy(report["before"])
        for edit in report["memory_journal"]:
            if edit["use"] == "operand":
                container, key = replay["nodes"][edit["owner"]]["inputs"], edit["slot"]
            elif edit["use"] == "final_write":
                container, key = (
                    replay["boundaries"][edit["owner"]]["writes"][edit["slot"]],
                    "value",
                )
            else:
                container, key = replay["boundaries"][edit["owner"]]["transfer"], edit["use"]
            self.assertEqual(container[key], edit["original"])
            container[key] = edit["replacement"]
            replay["revision"] = edit["to_revision"]
        for edit in report["arithmetic_journal"] + report["image_journal"]:
            node = replay["nodes"][edit["node"]]
            self.assertEqual(node["op"], edit["original"]["op"])
            self.assertEqual(node["inputs"], edit["original"]["inputs"])
            self.assertEqual(node.get("immediate", 0), edit["original"]["immediate"])
            node["op"] = edit["replacement"]["op"]
            node["inputs"] = edit["replacement"]["inputs"]
            if node["op"] in ("constant", "extract", "image_address"):
                node["immediate"] = edit["replacement"]["immediate"]
            else:
                node.pop("immediate", None)
            replay["revision"] = edit["to_revision"]
        if report["after"]["kind"] == "nyx.ir.recovered_path":
            basis_revision = replay["revision"]
            self.assertEqual(report["after"]["basis_revision"], basis_revision)
            edits = report["after"]["control_rewrites"]
            omissions = report["after"].get("store_omissions", [])
            paired_loads = report["after"].get("paired_load_omissions", [])
            # A destination the dispatch rule names is no node of the basis, so
            # it takes the next value id after it.
            destinations = []
            first_destination = len(replay["nodes"])
            for edit in edits:
                original = replay["boundaries"][edit["boundary"]]["transfer"]
                self.assertEqual(original, edit["original"])
                if edit["rule"] == "folded_condition":
                    self.assertEqual(original["kind"], "conditional")
                    self.assertEqual(original["condition"], edit["condition"])
                    condition = replay["nodes"][edit["condition"]]
                    self.assertEqual((condition["op"], condition["width"]), ("constant", 1))
                    self.assertEqual(bool(condition["immediate"] & 1), edit["condition_value"])
                    selected = (
                        original["target"] if edit["condition_value"] else original["alternative"]
                    )
                    self.assertEqual(edit["replacement"], {"kind": "jump", "target": selected})
                    self.assertEqual(edit["witness"] if "witness" in edit else [], [])
                else:
                    self.assertEqual(edit["rule"], "dispatch_branch")
                    self.assertEqual(original["kind"], "jump")
                    self.assertFalse(edit["condition_value"])
                    self.assertNotEqual(edit["when_true"], edit["when_false"])
                    named = first_destination + len(destinations)
                    self.assertEqual(
                        edit["replacement"],
                        {
                            "kind": "conditional",
                            "target": named,
                            "condition": edit["condition"],
                            "alternative": named + 1,
                        },
                    )
                    self.assertEqual(replay["nodes"][edit["condition"]]["width"], 1)
                    # Every read the two destinations rest on is named, and each
                    # one stays a load of the basis that still executes.
                    self.assertTrue(edit["witness"])
                    for read in edit["witness"]:
                        self.assertEqual(replay["nodes"][read["node"]]["op"], "load")
                        self.assertEqual(replay["nodes"][read["node"]]["width"], read["width"])
                    self.assertEqual(
                        sorted({read["when"] for read in edit["witness"]}), [False, True]
                    )
                    destinations.append({"value": named, "image_address": edit["when_true"]})
                    destinations.append({"value": named + 1, "image_address": edit["when_false"]})
                self.assertEqual(
                    (edit["from_revision"], edit["to_revision"]),
                    (basis_revision, basis_revision + 1),
                )
                replay["boundaries"][edit["boundary"]]["transfer"] = copy.deepcopy(
                    edit["replacement"]
                )
            replay["kind"] = "nyx.ir.recovered_path"
            replay["basis_revision"] = basis_revision
            replay["revision"] = basis_revision + bool(edits) + bool(paired_loads) + bool(omissions)
            replay["control_rewrites"] = copy.deepcopy(edits)
            replay["destinations"] = destinations
            replay["store_omissions"] = copy.deepcopy(omissions)
            replay["paired_load_omissions"] = copy.deepcopy(paired_loads)
            for omission in omissions:
                self.assertEqual(
                    (omission["from_revision"], omission["to_revision"]),
                    (basis_revision + bool(edits) + bool(paired_loads), replay["revision"]),
                )
                self.assertEqual(replay["nodes"][omission["store"]]["op"], "store")
                self.assertEqual(replay["nodes"][omission["overwriter"]]["op"], "store")
            for omission in paired_loads:
                self.assertEqual(
                    (omission["from_revision"], omission["to_revision"]),
                    (basis_revision + bool(edits), basis_revision + bool(edits) + 1),
                )
                self.assertEqual(
                    [replay["nodes"][i]["op"] for i in omission["stores"]], ["store", "store"]
                )
                self.assertEqual(
                    [replay["nodes"][i]["op"] for i in omission["loads"]], ["load", "load"]
                )
            if omissions:
                replay["store_omission_scope"] = report["after"]["store_omission_scope"]
            if paired_loads:
                replay["paired_load_omission_scope"] = report["after"]["paired_load_omission_scope"]
            folded = [edit for edit in edits if edit["rule"] == "folded_condition"]
            self.assertEqual(report.get("removed_conditional_transfers", 0), len(folded))
            self.assertEqual(report.get("recovered_dispatch_branches", 0), len(edits) - len(folded))
            if "removed_memory_effects" in report:
                self.assertEqual(
                    report["removed_memory_effects"], len(omissions) + 2 * len(paired_loads)
                )
            self.assertEqual(
                report["applied_edits"],
                len(report["memory_journal"])
                + len(report["arithmetic_journal"])
                + len(report["image_journal"])
                + len(edits)
                + len(omissions)
                + len(paired_loads),
            )
        self.assertEqual(replay, report["after"])
        return replay

    def test_path_recovery_forwards_state_preserving_sources_and_replayable_journals(self):
        tail = bytes.fromhex("c92c8052695600b97fea03f968e603f9aa060014")
        dispatcher = bytes.fromhex(
            "685640b9091d0a7108fdff5429d5ffd0296124910a0000102b79a8b84a010b8b40011fd6"
        )
        gap = 0x752E4C - 0x751394
        code = tail + bytes(gap - len(tail)) + dispatcher
        ranges = f"0x400080:20,{0x400080 + gap:x}:36"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "path.so"
            path.write_bytes(self.elf(code))
            args = ["simplify-path", str(path), "--ranges", ranges]
            refused, report = self.run_cli(*args)
            self.assertEqual(refused, 1)
            self.assertEqual(report["reason"], "unsupported_region")
            self.assertNotIn("after", report)
            status, report = self.run_cli(*args, "--memory-profile", "concrete_atomic_scalar")
        self.assertEqual(status, 0)
        self.assertEqual(report["outcome"], "simplified")
        self.assertEqual(report["source_groups"], 14)
        self.assertEqual(report["source_bytes"], 56)
        self.assertEqual(report["off_itinerary_successors"], "retained")
        self.assertFalse(report["whole_function_recovery"])
        self.assertEqual(report["removed_memory_effects"], 0)
        self.assertEqual(report["before"]["sources"], report["after"]["sources"])
        self.assertEqual(
            b"".join(bytes.fromhex(s["bytes"]) for s in report["after"]["sources"]),
            tail + dispatcher,
        )
        self.assertEqual(len(report["forwarding_facts"]), 1)
        self.assertEqual(report["forwarding_facts"][0]["address"]["offset"], 84)
        self.assertEqual(report["forwarding_facts"][0]["scope"], "successful_itinerary_prefix")
        replay = self.replay_recovery(report)
        conditional = next(
            b["transfer"]
            for b in replay["boundaries"]
            if b["transfer"] and b["transfer"]["kind"] == "conditional"
        )
        self.assertEqual(replay["nodes"][conditional["condition"]]["op"], "constant")
        self.assertEqual(replay["nodes"][conditional["condition"]]["immediate"], 0)
        self.assertEqual(
            report["applied_edits"],
            len(report["memory_journal"])
            + len(report["arithmetic_journal"])
            + len(report["image_journal"]),
        )
        # Nothing is declared for an itinerary: the image folds it makes hold at
        # any placement and rest on no restriction.
        self.assertTrue(report["image_journal"])
        for edit in report["image_journal"]:
            self.assertEqual((edit["rule"], edit["rests_on"]), ("image_offset", []))
            self.assertEqual(edit["replacement"]["op"], "image_address")

    def test_path_cleanup_reports_conditional_store_omission(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stack.so"
            path.write_bytes(self.elf(bytes.fromhex("e01f00f9e01f00f9")))
            status, report = self.run_cli(
                "simplify-path",
                str(path),
                "--ranges",
                "0x400080:8",
                "--memory-profile",
                "concrete_atomic_scalar",
            )
        self.assertEqual(status, 0)
        self.assertEqual(report["removed_memory_effects"], 1)
        self.assertEqual(report["memory_effect_scope"], "conditional_atomic_scalar")
        self.assertIn(
            "omitted accesses mapped with required permissions", report["execution_assumptions"]
        )
        self.assertIn("no resource-limit outcome", report["execution_assumptions"])
        self.assertEqual(report["applied_edits"], 1)
        self.assertFalse(report["architectural_fault_authority"])
        self.assertFalse(report["whole_function_recovery"])
        self.assertEqual(report["after"]["kind"], "nyx.ir.recovered_path")
        self.assertEqual(
            report["after"]["store_omissions"],
            [
                {
                    "rule": "overwritten_store",
                    "store": 4,
                    "overwriter": 7,
                    "base_storage": 31,
                    "offset_bytes": 56,
                    "from_revision": 0,
                    "to_revision": 1,
                }
            ],
        )
        self.assertIn("mapped writable", report["after"]["store_omission_scope"])
        self.replay_recovery(report)

    def test_selected_stack_itinerary_replays_forwarding_and_paired_load_omission(self):
        code = bytes.fromhex(
            "fd7bbfa9fd03009149e083d2eb031f2ac8a500b089b8aef28a0380520959c2f22930f5f2"
            "ec030b2a2b008052ccffff349f050071c10000540b4947f92b010bcbaa2f00a94b008052"
            "f7ffff17fd7bc1a8c0035fd6"
        )
        image = bytearray(0x3C4 + len(code))
        image[:16] = b"\x7fELF\x02\x01\x01" + bytes(9)
        struct.pack_into(
            "<HHIQQQIHHHHHH", image, 16, 3, 183, 1, 0x78A3C4, 64, 0, 0, 64, 56, 1, 0, 0, 0
        )
        struct.pack_into("<IIQQQQQQ", image, 64, 1, 5, 0, 0x78A000, 0, len(image), len(image), 4096)
        image[0x3C4:] = code
        ranges = "0x78a3c4:48,0x78a3e8:12,0x78a3f4:8,0x78a3fc:20,0x78a3e8:12,0x78a3f4:8,0x78a410:8"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "selected-stack.so"
            path.write_bytes(image)
            status, report = self.run_cli(
                "simplify-path",
                str(path),
                "--ranges",
                ranges,
                "--memory-profile",
                "concrete_atomic_scalar",
            )
        self.assertEqual(status, 0)
        self.assertEqual(report["source_groups"], 29)
        self.assertEqual(report["removed_memory_effects"], 2)
        self.assertEqual(report["memory_effect_scope"], "conditional_atomic_scalar")
        self.assertIn(
            "single-threaded normal memory without concurrent observers",
            report["execution_assumptions"],
        )
        self.assertEqual(report["after"]["paired_load_omissions"][0]["stores"], [89, 90])
        self.assertEqual(report["after"]["paired_load_omissions"][0]["loads"], [132, 133])
        self.assertFalse(report["architectural_fault_authority"])
        self.replay_recovery(report)

    def test_sample_stack_itinerary_replays_paired_load_omission(self):
        code = bytes.fromhex(
            "ffc300d1fdfb01a9fd630091691c80922901098be90340f9e90340f9e90b40f9"
            "fdfb41a9ffc30091fd7bbea9f30b00f9fd030091f30300aa008003915c1a0a94"
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample-stack.so"
            path.write_bytes(self.elf(code))
            status, report = self.run_cli(
                "simplify-path",
                str(path),
                "--ranges",
                "0x400080:64",
                "--memory-profile",
                "concrete_atomic_scalar",
            )
            region_status, regions = self.run_cli(
                "recover-regions",
                str(path),
                "--address",
                "400080",
                "--size",
                str(len(code)),
                "--memory-profile",
                "concrete_atomic_scalar",
            )
        self.assertEqual(status, 0)
        self.assertEqual(report["source_groups"], 16)
        self.assertEqual(report["removed_memory_effects"], 2)
        self.assertEqual(report["memory_effect_scope"], "conditional_atomic_scalar")
        self.assertEqual(report["after"]["paired_load_omissions"][0]["stores"], [9, 10])
        self.assertEqual(report["after"]["paired_load_omissions"][0]["loads"], [28, 29])
        self.assertFalse(report["architectural_fault_authority"])
        self.replay_recovery(report)
        self.assertEqual(region_status, 0)
        selected = next(
            r
            for r in regions["regions"]
            if r["entry"] == 0x400080 and r["source_ids"] == list(range(16))
        )
        self.assertEqual(selected["removed_memory_effects"], 2)
        self.assertEqual(regions["region_counts"]["removed_memory_effects"], 2)
        self.assertEqual(regions["region_contract"]["removed_memory_effects"], 2)
        self.assertFalse(regions["region_contract"]["architectural_fault_authority"])
        self.assertIn(
            "omitted accesses mapped with required permissions", regions["execution_assumptions"]
        )
        self.assertIn("no resource-limit outcome", regions["execution_assumptions"])
        self.assertEqual(selected["after"]["paired_load_omissions"][0]["loads"], [28, 29])
        self.assertEqual(selected["control"]["path_revision"], selected["after"]["revision"])
        self.replay_recovery(selected)

    def test_sample_repeated_global_mba_keeps_load(self):
        code = bytes.fromhex(
            "fa3b40b9e83f40b9e80f00b9fb2340f9284500f008810991080140f9e80308cb"
            "099480d209cdbcf2a9e8c6f2a99aecf2090109aa0a9480d20acdbcf2aae8c6f2"
            "aa9aecf208010a8a2801088b7523c89a284500f008010b91080140f900013fd6"
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample-mba.so"
            path.write_bytes(self.elf(code))
            status, report = self.run_cli(
                "simplify-path",
                str(path),
                "--ranges",
                "400080:96",
                "--memory-profile",
                "concrete_atomic_scalar",
            )
        self.assertEqual(status, 0)
        self.assertEqual(report["source_groups"], 24)
        self.assertEqual(report["removed_memory_effects"], 0)
        self.assertEqual(
            [
                (edit["rule"], edit["node"])
                for edit in report["arithmetic_journal"]
                if edit["rule"] != "constant_fold"
            ],
            [("or_and_sum", 56), ("negated_add", 56)],
        )
        self.assertEqual(
            (
                report["after"]["nodes"][25]["op"],
                report["after"]["nodes"][56]["op"],
                report["after"]["nodes"][56]["inputs"][1],
            ),
            ("load", "sub", 25),
        )
        self.replay_recovery(report)

    def test_automatic_regions_preserve_graph_and_replay_every_changed_candidate(self):
        tail = bytes.fromhex("c92c8052695600b97fea03f968e603f902000014")
        dispatcher = bytes.fromhex(
            "685640b9091d0a7108fdff5429d5ffd0296124910a0000102b79a8b84a010b8b40011fd6"
        )
        code = tail + bytes.fromhex("ffffffff") + dispatcher
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "regions.so"
            path.write_bytes(self.elf(code))
            args = [
                str(path),
                "--address",
                "400080",
                "--size",
                str(len(code)),
                "--memory-profile",
                "concrete_atomic_scalar",
            ]
            status, report = self.run_cli("recover-regions", *args)
            original_status, original = self.run_cli("cfg", *args)
            work, allocation = report["resources"]["work"], report["resources"]["bytes"]
            exact_status, exact = self.run_cli(
                "recover-regions",
                *args,
                "--work-budget",
                str(work),
                "--byte-budget",
                str(allocation),
            )
            self.assertEqual(exact_status, 0)
            self.assertEqual(exact["regions"], report["regions"])
            for work_cap, byte_cap in (
                (work - 1, allocation),
                (work, allocation - 1),
                (work // 2, allocation),
            ):
                refused, failure = self.run_cli(
                    "recover-regions",
                    *args,
                    "--work-budget",
                    str(work_cap),
                    "--byte-budget",
                    str(byte_cap),
                )
                self.assertEqual(refused, 1)
                self.assertEqual(failure["reason"], "resource_limit")
                self.assertNotIn("regions", failure)
                self.assertNotIn("blocks", failure)
        self.assertEqual((status, original_status), (0, 0))
        self.assertEqual(report["outcome"], "recovered_regions")
        for key in ("sources", "blocks", "entries", "block_count", "edge_count", "generation"):
            self.assertEqual(report[key], original[key])
        self.assertFalse(report["whole_function_recovery"])
        self.assertEqual(report["region_contract"]["redirected_edges"], 0)
        self.assertEqual(
            report["region_contract"]["selected_polarity"], "provenance_not_assumption"
        )
        self.assertEqual(report["unmodeled_groups"], 1)
        modeled_starts = {b["id"] for b in report["blocks"] if b["ssa"] is not None}
        self.assertEqual({r["entry_block"] for r in report["regions"]}, modeled_starts)
        self.assertEqual(len(report["regions"]), report["region_counts"]["candidates"])
        self.assertEqual(
            sum(len(r["source_ids"]) for r in report["regions"]),
            report["region_counts"]["source_occurrences"],
        )
        changed = [r for r in report["regions"] if r["outcome"] == "simplified"]
        self.assertEqual(len(changed), report["region_counts"]["simplified"])
        for region in changed:
            self.replay_recovery(region)
            expected = [report["sources"][i] for i in region["source_ids"]]
            self.assertEqual(
                [(v["address"], v["bytes"]) for v in expected],
                [(v["source_address"], v["bytes"]) for v in region["before"]["sources"]],
            )
            self.assertEqual(region["before"]["sources"], region["after"]["sources"])
        candidates = [r for r in changed if r["entry"] == 0x400080]
        self.assertEqual(len(candidates), 2)
        selected = max(candidates, key=lambda r: len(r["source_ids"]))
        self.assertEqual(len(selected["source_ids"]), 14)
        self.assertEqual(len(selected["forwarding_facts"]), 1)
        self.assertEqual(selected["removed_conditional_transfers"], 1)
        rewrite = selected["after"]["control_rewrites"][0]
        node = selected["after"]["nodes"][rewrite["condition"]]
        self.assertEqual((node["op"], node["immediate"]), ("constant", 0))
        self.assertEqual(
            selected["after"]["boundaries"][rewrite["boundary"]]["transfer"]["kind"], "jump"
        )
        # Entering the shared dispatcher must not inherit the predecessor's store.
        for region in changed:
            if region["entry"] == 0x400098:
                self.assertEqual(region["forwarding_facts"], [])
                guard = next(
                    b["transfer"]
                    for b in region["after"]["boundaries"]
                    if b["transfer"] and b["transfer"]["kind"] == "conditional"
                )
                self.assertNotEqual(region["after"]["nodes"][guard["condition"]]["op"], "constant")

    def test_cfg_probe_matches_recovery_structure_without_publishing_rewrites(self):
        code = bytes.fromhex("2000028b200040f91f2003d5")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "probe.so"
            path.write_bytes(self.elf(code))
            args = [
                str(path),
                "--address",
                "400080",
                "--size",
                str(len(code)),
                "--memory-profile",
                "concrete_atomic_scalar",
            ]
            probe_status, probe = self.run_cli("cfg-probe", *args)
            recovered_status, recovered = self.run_cli("recover-regions", *args)
            self.assertEqual((probe_status, recovered_status), (0, 0))
            self.assertEqual(probe["outcome"], "cfg_probe")
            self.assertEqual(probe["source_groups"], recovered["source_groups"])
            self.assertEqual(probe["block_count"], recovered["block_count"])
            self.assertEqual(
                probe["dispatch_blocks"],
                sum(block["dispatch"] is not None for block in recovered["blocks"]),
            )
            self.assertEqual(
                probe["dispatch_stop_regions"],
                sum(region["stop"] == "dispatch" for region in recovered["regions"]),
            )
            self.assertEqual(probe["memory_profile"], "concrete_atomic_scalar")
            self.assertTrue(
                all(
                    type(site["constant_image_dependency"]) is bool
                    for site in probe["dispatch_sites"]
                )
            )
            self.assertLessEqual(
                probe["stable_relocated_pointers"], probe["retained_relocated_pointers"]
            )
            self.assertEqual(probe["declared_constant_ranges"], [])
            self.assertNotIn("regions", probe)
            self.assertNotIn("unflattening", probe)
            self.assertNotIn("blocks", probe)
            writable = Path(temp) / "writable.so"
            writable.write_bytes(self.dynamic_elf(code))
            declared_status, declared = self.run_cli(
                "cfg-probe",
                str(writable),
                "--address",
                "400100",
                "--size",
                str(len(code)),
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-initialized",
                "400110:4",
            )
            self.assertEqual(declared_status, 0)
            self.assertEqual(
                declared["declared_constant_ranges"], [{"address": 0x400110, "bytes": 4}]
            )
            refused_status, refused = self.run_cli(
                "cfg-probe", *args, "--work-budget", str(probe["resources"]["work"] - 1)
            )
            self.assertEqual(refused_status, 1)
            self.assertEqual(refused["reason"], "resource_limit")
            self.assertNotIn("dispatch_sites", refused)
            rejected_status, rejected = self.run_cli(
                "cfg-probe", *args, "--assume-noreturn", "400080"
            )
            self.assertEqual(rejected_status, 1)
            self.assertEqual(rejected["reason"], "usage")

            # Bounded table dispatch with four distinct destinations; the
            # halfword table is image data after the scanned instruction span.
            dispatcher = bytes.fromhex(
                "49010010280140b91f0d0071c8000054e90000102a596878"
                "eb0000106b010a8b60011fd6c0035fd60200000000000400"
                "08000c00c0035fd6c0035fd6c0035fd6c0035fd6"
            )
            path.write_bytes(self.elf(dispatcher))
            positive_args = [
                str(path),
                "--address",
                "400080",
                "--size",
                "40",
                "--memory-profile",
                "concrete_atomic_scalar",
            ]
            positive_status, positive = self.run_cli("cfg-probe", *positive_args)
            positive_recovery_status, positive_recovery = self.run_cli(
                "recover-regions", *positive_args
            )
            self.assertEqual((positive_status, positive_recovery_status), (0, 0))
            self.assertEqual(
                (positive["dispatch_blocks"], positive["nominated_dispatch_sites"]), (1, 1)
            )
            self.assertEqual(
                positive["dispatch_sites"],
                [
                    {
                        "source_address": 0x4000A0,
                        "destinations": 4,
                        "stopping_regions": 2,
                        "constant_image_dependency": True,
                    }
                ],
            )
            self.assertEqual(
                positive["dispatch_stop_regions"],
                sum(region["stop"] == "dispatch" for region in positive_recovery["regions"]),
            )

    def test_region_control_stops_after_proved_divergence_and_keeps_runtime_load(self):
        tail = bytes.fromhex("c92c8052695600b97fea03f968e603f902000014")
        dispatcher = bytearray.fromhex(
            "685640b9091d0a7108fdff5429d5ffd0296124910a0000102b79a8b84a010b8b40011fd6"
        )
        struct.pack_into("<I", dispatcher, 8, 0x540000E8)  # B.HI to the default block after BR.
        code = tail + bytes.fromhex("ffffffff") + dispatcher + bytes.fromhex("1f2003d5c0031fd6")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "control.so"
            path.write_bytes(self.elf(code))
            status, report = self.run_cli(
                "recover-regions",
                str(path),
                "--address",
                "400080",
                "--size",
                str(len(code)),
                "--memory-profile",
                "concrete_atomic_scalar",
            )
        self.assertEqual(status, 0)
        candidates = [r for r in report["regions"] if r["entry"] == 0x400080]
        divergent = next(r for r in candidates if r["control"]["proved_divergence"] is not None)
        control = divergent["control"]
        self.assertEqual(control["proof_scope"], "successful_itinerary_prefix")
        self.assertEqual(control["path_revision"], divergent["after"]["revision"])
        self.assertEqual(control["proved_divergence"], 7)
        self.assertEqual(len(control["boundaries"]), 8)
        self.assertEqual(control["total_boundaries"], len(divergent["after"]["boundaries"]))
        self.assertGreater(control["total_boundaries"], len(control["boundaries"]))
        self.assertEqual(control["boundaries"][-1]["expected_match"], "never")
        self.assertEqual(control["boundaries"][-1]["transfer_kind"], "jump")
        self.assertEqual(len(control["boundaries"][-1]["edges"]), 1)
        self.assertFalse(divergent["after"]["control_rewrites"][0]["condition_value"])
        body = next(r for r in candidates if r["control"]["proved_divergence"] is None)
        control = body["control"]
        self.assertEqual(control["boundaries"][7]["expected_match"], "always")
        terminal = control["boundaries"][-1]
        self.assertEqual(terminal["expected_match"], "not_applicable")
        self.assertEqual(terminal["edges"][0]["target_kind"], "unknown")
        self.assertEqual(terminal["transfer_target"], terminal["edges"][0]["target_value"])
        self.assertEqual(len(terminal["load_dependencies"]), 1)
        self.assertEqual(body["after"]["nodes"][terminal["load_dependencies"][0]]["op"], "load")
        self.assertEqual(body["before"]["sources"], body["after"]["sources"])
        self.assertEqual(
            sum(
                r.get("control", {}).get("proved_divergence") is not None for r in report["regions"]
            ),
            report["region_counts"]["proved_divergences"],
        )
        direct = next(
            r for r in report["regions"] if r["entry"] == 0x400098 and len(r["source_ids"]) == 9
        )
        self.assertEqual(direct["control"]["boundaries"][2]["expected_match"], "unknown")
        self.assertEqual(
            [e["known_condition"] for e in direct["control"]["boundaries"][2]["edges"]],
            [None, None],
        )
        for region in report["regions"]:
            if region["outcome"] != "declined":
                self.assertIn("after", region)
                self.assertEqual(region["control"]["path_revision"], region["after"]["revision"])
                for fact in region["control"]["boundaries"]:
                    self.assertEqual(
                        fact["source_address"],
                        region["after"]["sources"][fact["boundary"]]["source_address"],
                    )

    def test_automatic_regions_report_local_selection_limit_and_global_budget_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "regions.so"
            path.write_bytes(self.elf(bytes.fromhex("1f2003d5") * 513))
            args = [str(path), "--address", "400080", "--size", str(513 * 4)]
            status, report = self.run_cli("recover-regions", *args)
            self.assertEqual(status, 0)
            self.assertEqual(report["region_counts"]["declined"], 1)
            self.assertEqual(report["regions"][0]["reason"], "source_limit")
            self.assertEqual(report["regions"][0]["source_ids"], [])
            self.assertEqual(report["source_groups"], 513)
            for option in ("--work-budget", "--byte-budget"):
                status, report = self.run_cli("recover-regions", *args, option, "0")
                self.assertEqual(status, 1)
                self.assertEqual(report["outcome"], "refused")
                self.assertNotIn("regions", report)
                self.assertNotIn("sources", report)
                status, report = self.run_cli(
                    "recover-regions", *args, option, "100", option, "200"
                )
                self.assertEqual(status, 1)
                self.assertEqual(report["reason"], "usage")

    def dynamic_elf(self, code, dynamic=((0, 0),)):
        # One writable load segment at 0x400000 holding `code` at 0x400100, and a
        # dynamic array at 0x400180: only a dynamic array followed to its end
        # says which bytes the loader leaves as the file has them.
        data = bytearray(0x200)
        data[:16] = b"\x7fELF\x02\x01\x01" + bytes(9)
        struct.pack_into(
            "<HHIQQQIHHHHHH", data, 16, 3, 183, 1, 0x400100, 64, 0, 0, 64, 56, 2, 0, 0, 0
        )
        struct.pack_into(
            "<IIQQQQQQ", data, 64, 1, 7, 0, 0x400000, 0, len(data), len(data) + 64, 4096
        )
        struct.pack_into(
            "<IIQQQQQQ",
            data,
            120,
            2,
            6,
            0x180,
            0x400180,
            0,
            16 * len(dynamic),
            16 * len(dynamic),
            8,
        )
        data[0x100 : 0x100 + len(code)] = code
        for index, (tag, value) in enumerate(dynamic):
            struct.pack_into("<QQ", data, 0x180 + 16 * index, tag, value)
        return data

    def test_a_store_into_a_read_only_segment_withdraws_only_the_bytes_it_writes(self):
        # adr x11, .; str wzr, [x11]; adr x9, .; ldr w10, [x9, #16];
        # add x9, x9, x10; br x9; .word 0x14; ret
        # The store writes the segment's own first four code bytes. The offset
        # the jump reads, further on, is still declared, so the jump resolves.
        code = bytes.fromhex("0b0000107f0100b9090000102a1140b929010a8b20011fd614000000c0035fd6")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stored.so"
            data = self.elf(code)
            path.write_bytes(data)
            status, report = self.run_cli(
                "recover-regions",
                str(path),
                "--address",
                "400080",
                "--size",
                str(len(code)),
                "--memory-profile",
                "concrete_atomic_scalar",
            )
            self.assertEqual(status, 0, report)
            self.assertEqual(report["refuted_constant_ranges"], 1)
            self.assertEqual(report["refuted_constant_range_addresses"], [0x400000])
            self.assertEqual(
                report["refuted_constant_spans"],
                [
                    {
                        "range": 0x400000,
                        "range_bytes": len(data),
                        "address": 0x400080,
                        "bytes": 4,
                        "written_at": 0x400080,
                        "width": 32,
                        "whole": False,
                        "refuted_by": "cfg_store",
                    }
                ],
            )
            block = next(
                b
                for b in report["blocks"]
                if b["control"] and b["control"]["terminal_source"] == 0x400094
            )
            self.assertEqual(len(block["edges"]), 1)
            self.assertEqual(block["edges"][0]["target"]["kind"], "image_location")
            self.assertEqual(block["edges"][0]["target"]["address"], 0x40009C)
            # The remaining bytes are each asserted individually.
            self.assertIn(
                "a load segment refuted_constant_spans cuts keeps its other bytes, each"
                " claimed on its own, so a store whose address rested on a refuted byte"
                " may have gone unseen",
                report["execution_assumptions"],
            )

    def test_read_only_bytes_sharing_a_large_page_with_writable_data_are_not_fixed(self):
        # adr x9, .; ldr w10, [x9, #16]; add x9, x9, x10; br x9; .word 0x14; ret
        # The offset sits in a read-only segment at 0x400000; a writable one
        # starts at 0x401000. Linked for 4 KiB pages the two never share one;
        # linked for 64 KiB pages they do, and the loader maps them together.
        code = bytes.fromhex("090000102a1140b929010a8b20011fd614000000c0035fd6")

        def image(align):
            data = bytearray(0x1010)
            data[:16] = b"\x7fELF\x02\x01\x01" + bytes(9)
            struct.pack_into(
                "<HHIQQQIHHHHHH", data, 16, 3, 183, 1, 0x400100, 64, 0, 0, 64, 56, 2, 0, 0, 0
            )
            struct.pack_into("<IIQQQQQQ", data, 64, 1, 5, 0, 0x400000, 0, 0x200, 0x200, align)
            struct.pack_into("<IIQQQQQQ", data, 120, 1, 6, 0x1000, 0x401000, 0, 0x10, 0x10, align)
            data[0x100 : 0x100 + len(code)] = code
            return data

        kinds = []
        with tempfile.TemporaryDirectory() as temp:
            for align in (0x1000, 0x10000):
                path = Path(temp) / f"paged{align}.so"
                path.write_bytes(image(align))
                status, report = self.run_cli(
                    "recover-regions",
                    str(path),
                    "--address",
                    "400100",
                    "--size",
                    str(len(code)),
                    "--memory-profile",
                    "concrete_atomic_scalar",
                )
                self.assertEqual(status, 0, report)
                block = next(
                    b
                    for b in report["blocks"]
                    if b["control"] and b["control"]["terminal_source"] == 0x40010C
                )
                self.assertEqual(len(block["edges"]), 1)
                kinds.append(block["edges"][0]["target"]["kind"])
        self.assertEqual(kinds, ["image_location", "unknown"])

    def test_a_declared_word_a_carried_register_writes_resolves_no_jump(self):
        # adr x9, word; b next; mov w10, #4; str w10, [x9]; adr x11, word;
        # ldr w12, [x11]; adr x13, .+8; add x13, x13, x12; br x13; ret ...
        # The word is declared to keep its file value, 0x10, which sends the
        # jump to 0x400130. The store that makes it 4 goes through x9, set in
        # another block, so only the whole graph places it.
        code = bytes.fromhex(
            "09020010"
            "01000014"
            "8a008052"
            "2a0100b9"
            "8b010010"
            "6c0140b9"
            "4d000010"
            "ad010c8b"
            "a0011fd6"
            "c0035fd6"
            "00000000"
            "00000000"
            "c0035fd6"
            "c0035fd6"
            "00000000"
            "00000000"
            "10000000"
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "carried.so"
            path.write_bytes(self.dynamic_elf(code))
            args = [
                str(path),
                "--address",
                "400100",
                "--size",
                "56",
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-initialized",
                "400140:4",
            ]

            def jump(report):
                block = next(
                    b
                    for b in report["blocks"]
                    if b["control"] and b["control"]["terminal_source"] == 0x400120
                )
                return [
                    (edge["target"]["kind"], edge["target"]["address"]) for edge in block["edges"]
                ]

            # With every way in listed, the graph says where x9 points.
            status, report = self.run_cli("recover-regions", *args, "--assume-entries", "closed")
            self.assertEqual(status, 0, report)
            self.assertEqual(
                report["refuted_constant_spans"],
                [
                    {
                        "range": 0x400140,
                        "range_bytes": 4,
                        "address": 0x400140,
                        "bytes": 4,
                        "written_at": 0x400140,
                        "width": 32,
                        "whole": False,
                        "refuted_by": "ssa_store",
                        "block": 0x400108,
                        "node": 6,
                    }
                ],
            )
            self.assertTrue(report["ssa_refutation_settled"])
            self.assertEqual(jump(report), [("unknown", 0)])
            self.assertFalse(any("declared range" in a for a in report["execution_assumptions"]))
            # Without it, x9 could arrive holding anything; that store is the
            # declaration's to answer for, and the word still resolves the jump.
            status, report = self.run_cli("recover-regions", *args)
            self.assertEqual(status, 0, report)
            self.assertEqual(report["refuted_constant_spans"], [])
            self.assertEqual(jump(report), [("image_location", 0x400130)])

    def test_declared_initialized_range_is_the_only_way_writable_bytes_resolve_a_jump(self):
        # adr x9, .; ldr w10, [x9, #16]; add x9, x9, x10; br x9; .word 0x14; ret
        # The offset lives in a writable segment, so nothing about the image
        # makes it fixed; only the caller's declaration does.
        code = bytes.fromhex("090000102a1140b929010a8b20011fd614000000c0035fd6")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "declared.so"
            path.write_bytes(self.dynamic_elf(code))
            args = [
                str(path),
                "--address",
                "400100",
                "--size",
                str(len(code)),
                "--memory-profile",
                "concrete_atomic_scalar",
            ]

            def jump(report):
                block = next(
                    b
                    for b in report["blocks"]
                    if b["control"] and b["control"]["terminal_source"] == 0x40010C
                )
                self.assertEqual(len(block["edges"]), 1)
                return block["edges"][0]

            status, report = self.run_cli("recover-regions", *args)
            self.assertEqual(status, 0)
            self.assertEqual(report["declared_constant_ranges"], [])
            self.assertFalse(any("declared range" in a for a in report["execution_assumptions"]))
            self.assertEqual(jump(report)["target"]["kind"], "unknown")

            status, report = self.run_cli(
                "recover-regions", *args, "--assume-initialized", "400110:4"
            )
            self.assertEqual(status, 0)
            self.assertEqual(
                report["declared_constant_ranges"], [{"address": 0x400110, "bytes": 4}]
            )
            self.assertIn(
                "the declared range %d+4 keeps the value the image is loaded with,"
                " from its file or, past the file, the zero the loader leaves" % 0x400110,
                report["execution_assumptions"],
            )
            edge = jump(report)
            self.assertEqual(
                edge["target"],
                {"kind": "image_location", "address": 0x400114, "value": edge["target"]["value"]},
            )
            self.assertEqual(edge["resolution"], "block_entry")

            # Past the file extent the loader leaves zero, which the segment
            # header states, so a range there can be assumed initialized.
            beyond = "%x:4" % (0x400000 + 0x200 + 8)
            status, report = self.run_cli("recover-regions", *args, "--assume-initialized", beyond)
            self.assertEqual(status, 0, report)
            self.assertIn("zero the loader leaves", " ".join(report["execution_assumptions"]))
            # Every malformed declaration is still refused.
            for spec in (
                "400110",
                "400110:0",
                "400110:4,",
                "400110:4097",
                "zz:4",
                ",".join(["400110:4"] * 17),
            ):
                status, report = self.run_cli(
                    "recover-regions", *args, "--assume-initialized", spec
                )
                self.assertEqual(status, 1, spec)
                self.assertEqual(report["reason"], "invalid_range", spec)
                self.assertNotIn("regions", report)
            status, report = self.run_cli(
                "recover-regions",
                *args,
                "--assume-initialized",
                "400110:4",
                "--assume-initialized",
                "400110:4",
            )
            self.assertEqual(report["reason"], "usage")
            # Without a dynamic array nothing names what startup code relocates,
            # so the same declaration over the same bytes is refused.
            path.write_bytes(self.elf(code, flags=7, memory_extra=64))
            status, report = self.run_cli(
                "recover-regions",
                str(path),
                "--address",
                "400080",
                "--size",
                str(len(code)),
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-initialized",
                "400090:4",
            )
            self.assertEqual(status, 1)
            self.assertIn("loader may write", report["detail"])

    def test_declared_noreturn_targets_are_printed_sorted_and_malformed_ones_refused(self):
        code = bytes.fromhex("c0035fd6")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "noreturn.so"
            path.write_bytes(self.dynamic_elf(code))
            args = [str(path), "--address", "400100", "--size", "4"]
            status, report = self.run_cli("recover-regions", *args)
            self.assertEqual(status, 0)
            self.assertEqual(report["declared_noreturn_targets"], [])
            self.assertFalse(any("never returns" in a for a in report["execution_assumptions"]))
            status, report = self.run_cli(
                "recover-regions", *args, "--assume-noreturn", "9000,1ba78d0,9000"
            )
            self.assertEqual(status, 0)
            self.assertEqual(report["declared_noreturn_targets"], [0x9000, 0x1BA78D0])
            self.assertEqual(
                [a for a in report["execution_assumptions"] if "never returns" in a],
                [
                    "a call to the declared target %d never returns" % 0x9000,
                    "a call to the declared target %d never returns" % 0x1BA78D0,
                ],
            )
            for spec in ("", ",", "9000,", "zz", ",".join(["9000"] * 65)):
                status, report = self.run_cli("recover-regions", *args, "--assume-noreturn", spec)
                self.assertEqual(status, 1, spec)
                self.assertNotIn("regions", report)
            status, report = self.run_cli(
                "recover-regions", *args, "--assume-noreturn", "9000", "--assume-noreturn", "9000"
            )
            self.assertEqual(report["reason"], "usage")

    def test_reachability_through_a_trap_needs_the_trap_declaration(self):
        # cbz x0, L; ret; L: brk #0. The BRK block is complete only if a trap
        # never resumes inside the population, which the caller must declare.
        code = bytes.fromhex("400000b4c0035fd6000020d4")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trap.so"
            path.write_bytes(self.elf(code))
            args = [
                str(path),
                "--address",
                "400080",
                "--size",
                "12",
                "--assume-abi",
                "aapcs64",
                "--assume-returns",
                "leave",
                "--assume-entries",
                "closed",
            ]
            status, report = self.run_cli("recover-regions", *args)
            self.assertEqual(status, 0)
            self.assertFalse(report["ssa_graph"]["declarations"]["trap_stops"])
            stages = {stage["pass"]: stage for stage in report["ssa_graph"]["proposals"]["stages"]}
            self.assertEqual(
                (stages["unreachable_blocks"]["outcome"], stages["unreachable_blocks"]["reason"]),
                ("not_run", "no_trap_stop_declaration"),
            )
            status, report = self.run_cli("recover-regions", *args, "--assume-traps", "stop")
            self.assertEqual(status, 0)
            self.assertTrue(report["ssa_graph"]["declarations"]["trap_stops"])
            self.assertIn(
                "a trap never resumes inside the selected population except at a listed entry",
                report["execution_assumptions"],
            )
            stages = {stage["pass"]: stage for stage in report["ssa_graph"]["proposals"]["stages"]}
            self.assertEqual(stages["unreachable_blocks"]["outcome"], "unchanged")
            self.assertIn(
                "a trap never resumes inside the population except at a listed entry",
                stages["unreachable_blocks"]["assumptions"],
            )
            for extra in (
                ("--assume-traps", "resume"),
                ("--assume-traps", "stop", "--assume-traps", "stop"),
            ):
                status, report = self.run_cli("recover-regions", *args, *extra)
                self.assertEqual((status, report["reason"]), (1, "usage"), extra)
            status, report = self.run_cli(
                "cfg-probe",
                str(path),
                "--address",
                "400080",
                "--size",
                "12",
                "--assume-traps",
                "stop",
            )
            self.assertEqual((status, report["reason"]), (1, "usage"))

    def test_declared_abi_lets_a_case_store_through_a_frame_alias(self):
        # A flattened function keeps its state at [x19+8] and case data through
        # x27 = x19+16. Case 0 stores the next state and then writes [x27+8];
        # only the entry relation places that write apart from the state. Case
        # 1 calls out first, so the relation reaches it only across the call.
        code = bytes.fromhex(
            "fd7bbfa9fd030091ff0301d1f30300917b4200917f0a00b90d000014"
            "28008052680a00b97f0700f909000014"
            "f503009448008052680a00b97f0300f904000014"
            "bf030091fd7bc1a8c0035fd6"
            "680a40b90909007168ffff54a90000100afeff102b7968784a090b8b40011fd6"
            "0000040009000000"
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "flattened.so"
            path.write_bytes(self.elf(code))
            args = [
                str(path),
                "--address",
                "400080",
                "--size",
                "108",
                "--memory-profile",
                "concrete_atomic_scalar",
            ]
            status, plain = self.run_cli("recover-regions", *args)
            self.assertEqual(status, 0)
            self.assertIsNone(plain["declared_abi"])
            self.assertEqual(plain["unflattening"]["entry_relations_rest_on"], [])
            self.assertEqual([t["entry"] for t in plain["unflattening"]["transitions"]], [0x400080])
            self.assertEqual(plain["unflattening"]["retired_blocks"], [])
            self.assertEqual(plain["region_counts"]["entry_relation_paths"], 0)

            status, abi_only = self.run_cli("recover-regions", *args, "--assume-abi", "aapcs64")
            self.assertEqual(status, 0)
            self.assertFalse(abi_only["declared_return_leaves"])
            self.assertEqual(abi_only["unflattening"]["retired_blocks"], [])

            status, report = self.run_cli(
                "recover-regions", *args, "--assume-abi", "aapcs64", "--assume-returns", "leave"
            )
            self.assertEqual(status, 0)
            self.assertEqual(report["ssa_graph"]["status"], "built")
            self.assertEqual(report["ssa_graph"]["blocks"], 5)
            self.assertIn("relation_returns_leave", report["ssa_graph"]["text"])
            self.assertIn("declared_return_leaves", report["ssa_graph"]["text"])
            self.assertEqual(report["declared_abi"], "aapcs64")
            self.assertIn(
                "every callee returns only through its call's continuation, with AAPCS64"
                " X19-X29 and SP as it found them",
                report["execution_assumptions"],
            )
            self.assertIn(
                "every untargeted return leaves the selected function population",
                report["execution_assumptions"],
            )
            unflattening = report["unflattening"]
            self.assertEqual(
                [
                    (t["entry"], t["destinations"], t["entry_relation_dependency"])
                    for t in unflattening["transitions"]
                ],
                [
                    (0x400080, [0x40009C], False),
                    (0x40009C, [0x4000AC], True),
                    (0x4000B0, [0x4000C0], True),
                ],
            )
            self.assertEqual(unflattening["retirement"], "established")
            self.assertEqual(unflattening["retired_blocks"], [0x4000CC, 0x4000D8])
            self.assertEqual(
                unflattening["entry_relations_rest_on"],
                ["declared_abi", "declared_return_leaves", "constant_image"],
            )
            self.assertIn(
                "storage relations proved from the known entries hold at each dependent"
                " transition's entry",
                unflattening["retirement_assumes"],
            )
            relation = (
                "storage relations proved from the known entries hold at the transition's entry"
            )
            graph = {block["address"]: block for block in unflattening["recovered_graph"]["blocks"]}
            self.assertEqual(sorted(graph), [0x400080, 0x40009C, 0x4000AC, 0x4000B0, 0x4000C0])
            self.assertNotIn(relation, graph[0x400080]["edges"][0]["assumes"])
            self.assertEqual(unflattening["reachable_returns"], 1)
            self.assertIn(
                "a return the graph does not resolve leaves for the caller",
                unflattening["retirement_assumes"],
            )
            self.assertEqual(
                graph[0x4000C0]["edges"][0]["assumes"], ["this return leaves for the caller"]
            )
            self.assertIn(relation, graph[0x40009C]["edges"][0]["assumes"])
            self.assertIn(
                "caller declares untargeted returns leave the selected population",
                graph[0x40009C]["edges"][0]["assumes"],
            )
            case = next(
                region
                for region in report["regions"]
                if region["entry"] == 0x40009C and region["stop"] == "dispatch"
            )
            self.assertEqual(
                case["entry_relations"],
                [
                    {"storage": 27, "root": 19, "offset": 16},
                    {"storage": 29, "root": 19, "offset": 64},
                    {"storage": 31, "root": 19, "offset": 0},
                ],
            )
            self.assertEqual(
                [fact["rests_on"] for fact in case["forwarding_facts"]], [["entry_relation"]]
            )

            for extra in (
                ["--assume-abi", "sysv"],
                ["--assume-abi", "aapcs64", "--assume-abi", "aapcs64"],
            ):
                status, refused = self.run_cli("recover-regions", *args, *extra)
                self.assertEqual((status, refused["reason"]), (1, "usage"))
            status, refused = self.run_cli("cfg-probe", *args, "--assume-abi", "aapcs64")
            self.assertEqual((status, refused["reason"]), (1, "usage"))

    def test_ssa_proposals_publish_what_each_edit_rests_on(self):
        # The flattened fixture above: its dispatch table is read-only image
        # bytes, so a declared access contract lets its loads fold.
        code = bytes.fromhex(
            "fd7bbfa9fd030091ff0301d1f30300917b4200917f0a00b90d000014"
            "28008052680a00b97f0700f909000014"
            "f503009448008052680a00b97f0300f904000014"
            "bf030091fd7bc1a8c0035fd6"
            "680a40b90909007168ffff54a90000100afeff102b7968784a090b8b40011fd6"
            "0000040009000000"
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "flattened.so"
            path.write_bytes(self.elf(code))
            args = [
                str(path),
                "--address",
                "400080",
                "--size",
                "108",
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-abi",
                "aapcs64",
            ]
            status, plain = self.run_cli("recover-regions", *args)
            self.assertEqual(status, 0)
            ssa_graph = plain["ssa_graph"]
            self.assertTrue(
                ssa_graph["text"].startswith("ssa revision %d\n" % ssa_graph["revision"])
            )
            # --assume-abi is what declares that callees return to their continuation.
            self.assertEqual(
                ssa_graph["declarations"],
                {
                    "private_frame": None,
                    "image_access": False,
                    "closed_entries": False,
                    "return_leaves": False,
                    "call_returns": True,
                    "trap_stops": False,
                },
            )
            proposals = ssa_graph["proposals"]
            self.assertEqual(
                (proposals["status"], proposals["whole_function_check"]),
                ("provisional", "NOT CHECKED"),
            )
            stages = {stage["pass"]: stage for stage in proposals["stages"]}
            self.assertEqual(
                list(stages),
                [
                    "unreachable_blocks",
                    "phi_constant_fold",
                    "constant_propagation",
                    "sccp_constant_fold",
                    "constant_image_loads",
                    "selected_image_loads",
                    "linear_mba",
                    "canonical_linear_mba",
                    "dead_private_state",
                    "frame_promotion",
                    "bounded_table_loads",
                    "sccp_storage_reads",
                    "branch_retirement",
                    "loop_exits",
                    "leaf_calls",
                    "resolved_calls",
                    "unreachable_after_sccp",
                    "predecessor_copies",
                    "dead_storage_writes",
                    "algebraic_simplify",
                    "dead_loads",
                    "dead_effect_free_nodes",
                ],
            )
            loads = stages["constant_image_loads"]
            self.assertEqual(
                (stages["unreachable_blocks"]["outcome"], stages["unreachable_blocks"]["reason"]),
                ("not_run", "no_closed_entry_declaration"),
            )
            self.assertEqual(
                (stages["phi_constant_fold"]["outcome"], stages["phi_constant_fold"]["reason"]),
                ("not_run", "no_closed_entry_declaration"),
            )
            self.assertEqual(
                (
                    stages["constant_propagation"]["outcome"],
                    stages["constant_propagation"]["reason"],
                ),
                ("not_run", "no_closed_entry_declaration"),
            )
            self.assertEqual(
                (stages["sccp_constant_fold"]["outcome"], stages["sccp_constant_fold"]["reason"]),
                ("not_run", "no_closed_entry_declaration"),
            )
            self.assertEqual((loads["outcome"], loads["journal"]), ("unchanged", []))
            self.assertTrue(loads["refused"])
            self.assertEqual({item["reason"] for item in loads["refused"]}, {"not_nonfaulting"})
            for name in ("frame_promotion", "dead_private_state"):
                self.assertEqual(
                    (stages[name]["outcome"], stages[name]["reason"]),
                    ("not_run", "no_private_frame_declaration"),
                )
            self.assertEqual(stages["dead_effect_free_nodes"]["outcome"], "proposed")
            self.assertIn(" dead\n", proposals["text"])

            status, report = self.run_cli(
                "recover-regions",
                *args,
                "--assume-image-access",
                "readable",
                "--assume-private-frame",
                "-64:0",
                "--assume-frame-reach",
                "none",
            )
            self.assertEqual(status, 0)
            ssa_graph = report["ssa_graph"]
            self.assertEqual(ssa_graph["text"], plain["ssa_graph"]["text"])
            self.assertEqual(
                ssa_graph["declarations"],
                {
                    "private_frame": {
                        "begin": -64,
                        "end": 0,
                        "sp_alignment": 16,
                        "fresh_mapped_writable": True,
                        "no_external_aliases": True,
                        "no_async_observers": True,
                        "callees_cannot_touch": True,
                        "callees_preserve_sp": True,
                    },
                    "image_access": True,
                    "closed_entries": False,
                    "return_leaves": False,
                    "call_returns": True,
                    "trap_stops": False,
                },
            )
            stages = {stage["pass"]: stage for stage in ssa_graph["proposals"]["stages"]}
            loads = stages["constant_image_loads"]
            self.assertEqual(loads["outcome"], "proposed")
            self.assertEqual(loads["from_revision"], ssa_graph["revision"])
            self.assertIn(
                "every folded image byte stays mapped readable for the run, and reading"
                " it has no observable effect",
                loads["assumptions"],
            )
            image = self.elf(code)
            for edit in loads["journal"]:
                self.assertEqual((edit["fact"], edit["kind"]), ("constant_range", "literal"))
                self.assertIn(
                    "the image range %d+%d keeps its file-initialized value"
                    % (edit["fact_address"], edit["fact_bytes"]),
                    loads["assumptions"],
                )
                self.assertLessEqual(edit["fact_address"], edit["source_address"])
                # The fixture maps file offset 0 at 0x400000, so the fold is its file bytes.
                offset, size = edit["source_address"] - 0x400000, edit["width"] // 8
                self.assertEqual(
                    edit["value"], int.from_bytes(image[offset : offset + size], edit["byte_order"])
                )
            self.assertIn(16, {edit["width"] for edit in loads["journal"]})
            self.assertEqual(stages["linear_mba"]["from_revision"], loads["to_revision"])
            # The frame proof follows the prologue's SP move and the frame base
            # it keeps in X19 through every block.
            for name in ("frame_promotion", "dead_private_state"):
                self.assertEqual(
                    (stages[name]["outcome"], stages[name]["reason"]), ("proposed", "none"), name
                )
                self.assertTrue(stages[name]["journal"], name)
            text = ssa_graph["proposals"]["text"]
            self.assertGreaterEqual(text.count(" disabled"), len(loads["journal"]))
            self.assertEqual(text.count("  constant_load %"), len(loads["journal"]))

            for extra in (
                ["--assume-private-frame", "0:0"],
                ["--assume-private-frame", "-8"],
                ["--assume-private-frame", "x:0"],
                ["--assume-frame-reach", "none"],
                ["--assume-private-frame", "-8:0", "--assume-frame-reach", "some"],
                ["--assume-image-access", "mapped"],
                ["--assume-image-access", "readable", "--assume-image-access", "readable"],
            ):
                status, refused = self.run_cli("recover-regions", *args, *extra)
                self.assertEqual(status, 1, extra)
                self.assertIn(refused["reason"], ("usage", "invalid_range"), extra)
            status, refused = self.run_cli(
                "cfg-probe",
                str(path),
                "--address",
                "400080",
                "--size",
                "108",
                "--assume-image-access",
                "readable",
            )
            self.assertEqual((status, refused["reason"]), (1, "usage"))

    def test_ssa_retires_dead_private_state_before_frame_promotion(self):
        code = bytes.fromhex("e0031ff8ff035ff8c0035fd6")  # stur, ldur xzr, ret
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "private.so"
            path.write_bytes(self.elf(code))
            status, report = self.run_cli(
                "recover-regions",
                str(path),
                "--address",
                "400080",
                "--size",
                "12",
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-abi",
                "aapcs64",
                "--assume-private-frame",
                "-32:0",
            )
        self.assertEqual(status, 0)
        stages = report["ssa_graph"]["proposals"]["stages"]
        self.assertEqual(
            [stage["pass"] for stage in stages],
            [
                "unreachable_blocks",
                "phi_constant_fold",
                "constant_propagation",
                "sccp_constant_fold",
                "constant_image_loads",
                "selected_image_loads",
                "linear_mba",
                "canonical_linear_mba",
                "dead_private_state",
                "frame_promotion",
                "bounded_table_loads",
                "sccp_storage_reads",
                "branch_retirement",
                "loop_exits",
                "leaf_calls",
                "resolved_calls",
                "unreachable_after_sccp",
                "predecessor_copies",
                "dead_storage_writes",
                "algebraic_simplify",
                "dead_loads",
                "dead_effect_free_nodes",
            ],
        )
        dead = stages[8]
        self.assertEqual(dead["outcome"], "proposed")
        self.assertEqual(len(dead["journal"]), 2)
        self.assertEqual({item["op"] for item in dead["journal"]}, {"store", "load"})
        self.assertTrue(any("entry SP-32 to entry SP+0" in item for item in dead["assumptions"]))
        self.assertEqual(stages[9]["from_revision"], dead["to_revision"])

    def test_ssa_retires_declared_nonfaulting_literal_load(self):
        code = bytes.fromhex("9f000058c0035fd6") + bytes(8) + struct.pack("<Q", 0x12345678)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "literal.so"
            path.write_bytes(self.elf(code))
            status, report = self.run_cli(
                "recover-regions",
                str(path),
                "--address",
                "400080",
                "--size",
                "8",
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-abi",
                "aapcs64",
                "--assume-image-access",
                "readable",
            )
        self.assertEqual(status, 0)
        stages = {stage["pass"]: stage for stage in report["ssa_graph"]["proposals"]["stages"]}
        folded = stages["constant_image_loads"]
        self.assertEqual(folded["outcome"], "proposed")
        self.assertEqual(len(folded["journal"]), 1)
        self.assertTrue(folded["journal"][0]["retired_access"])
        dead = stages["dead_effect_free_nodes"]
        self.assertEqual(dead["outcome"], "proposed")
        self.assertEqual({edit["op"] for edit in dead["journal"]}, {"image_address", "load"})
        self.assertTrue(any("constant-image stage" in item for item in dead["assumptions"]))
        self.assertIn("no_access", report["ssa_graph"]["proposals"]["text"])

    def test_ssa_retires_selected_table_load(self):
        words = (0xF100001F, 0x9A9F17E8, 0x100007C9, 0xF8687921, 0xD65F03C0)
        code = b"".join(struct.pack("<I", word) for word in words)
        code += bytes(0x100 - len(code)) + struct.pack("<QQ", 11, 22)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "selected.so"
            path.write_bytes(self.elf(code))
            status, report = self.run_cli(
                "recover-regions",
                str(path),
                "--address",
                "400080",
                "--size",
                "20",
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-abi",
                "aapcs64",
                "--assume-image-access",
                "readable",
            )
        self.assertEqual(status, 0)
        stages = {stage["pass"]: stage for stage in report["ssa_graph"]["proposals"]["stages"]}
        selected = stages["selected_image_loads"]
        self.assertEqual(selected["outcome"], "proposed")
        self.assertEqual(len(selected["journal"]), 1)
        edit = selected["journal"][0]
        self.assertTrue(edit["retired_access"])
        self.assertEqual({edit["when_true"]["value"], edit["when_false"]["value"]}, {11, 22})
        self.assertEqual(
            {edit["when_true"]["address"], edit["when_false"]["address"]}, {0x400180, 0x400188}
        )
        self.assertTrue(
            any("every selected image byte" in item for item in selected["assumptions"])
        )
        dead = stages["dead_effect_free_nodes"]
        self.assertEqual(dead["outcome"], "proposed")
        # The folded address is rechecked by value, so simplification may
        # reshape it before its unused nodes retire.
        self.assertLessEqual({"constant", "shl", "add"}, {item["op"] for item in dead["journal"]})
        self.assertIn(" when %", report["ssa_graph"]["proposals"]["text"])

    def test_ssa_bounded_table_fold_requires_declared_scope_and_prints_edits(self):
        words = (0xF100101F, 0x54000082, 0x100007C1, 0xF8607820, 0xD65F03C0, 0xD65F03C0)
        code = b"".join(struct.pack("<I", word) for word in words)
        code += bytes(0x100 - len(code)) + struct.pack("<QQQQ", 11, 22, 33, 44)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bounded.so"
            path.write_bytes(self.elf(code))
            args = (
                str(path),
                "--address",
                "400080",
                "--size",
                "24",
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-abi",
                "aapcs64",
            )
            status, plain = self.run_cli(
                "recover-regions", *args, "--assume-image-access", "readable"
            )
            self.assertEqual(status, 0)
            stages = {stage["pass"]: stage for stage in plain["ssa_graph"]["proposals"]["stages"]}
            self.assertEqual(
                (stages["bounded_table_loads"]["outcome"], stages["bounded_table_loads"]["reason"]),
                ("not_run", "no_closed_entry_declaration"),
            )
            status, no_access = self.run_cli("recover-regions", *args, "--assume-entries", "closed")
            self.assertEqual(status, 0)
            stages = {
                stage["pass"]: stage for stage in no_access["ssa_graph"]["proposals"]["stages"]
            }
            self.assertEqual(
                (stages["bounded_table_loads"]["outcome"], stages["bounded_table_loads"]["reason"]),
                ("not_run", "no_image_access_declaration"),
            )
            status, no_return_scope = self.run_cli(
                "recover-regions",
                *args,
                "--assume-image-access",
                "readable",
                "--assume-entries",
                "closed",
            )
            self.assertEqual(status, 0)
            stage = next(
                item
                for item in no_return_scope["ssa_graph"]["proposals"]["stages"]
                if item["pass"] == "bounded_table_loads"
            )
            self.assertEqual(
                (stage["outcome"], stage["reason"]), ("not_run", "no_return_leaves_declaration")
            )
            status, report = self.run_cli(
                "recover-regions",
                *args,
                "--assume-image-access",
                "readable",
                "--assume-entries",
                "closed",
                "--assume-returns",
                "leave",
            )
            self.assertEqual(status, 0)
            self.assertTrue(report["ssa_graph"]["declarations"]["closed_entries"])
            self.assertTrue(report["ssa_graph"]["declarations"]["return_leaves"])
            stages = {stage["pass"]: stage for stage in report["ssa_graph"]["proposals"]["stages"]}
            bounded = stages["bounded_table_loads"]
            self.assertEqual(
                (bounded["outcome"], bounded["reason"], len(bounded["journal"])),
                ("proposed", "none", 1),
            )
            edit = bounded["journal"][0]
            self.assertEqual(
                (edit["base"], edit["stride"], edit["rows"], edit["retired_access"]),
                (0x400180, 8, 4, True),
            )
            self.assertTrue(
                any("listed entries exhaust" in item for item in bounded["assumptions"])
            )
            self.assertTrue(any("image range" in item for item in bounded["assumptions"]))
            dead = stages["dead_effect_free_nodes"]
            # The table address's three nodes, and one the guard computes
            # for nothing: a fold no longer freezes its guard.
            self.assertEqual((dead["outcome"], len(dead["journal"])), ("proposed", 4))
            self.assertEqual(sum(item["block"] != edit["block"] for item in dead["journal"]), 1)
            self.assertIn("bounded_table", report["ssa_graph"]["proposals"]["text"])
            self.assertEqual(
                report["ssa_graph"]["proposals"]["whole_function_check"], "NOT CHECKED"
            )
            dead_words = (*words[:3], 0xF860783F, *words[4:])
            dead_code = b"".join(struct.pack("<I", word) for word in dead_words)
            dead_code += bytes(0x100 - len(dead_code)) + struct.pack("<QQQQ", 11, 22, 33, 44)
            path.write_bytes(self.elf(dead_code))
            status, dead_report = self.run_cli(
                "recover-regions",
                *args,
                "--assume-image-access",
                "readable",
                "--assume-entries",
                "closed",
                "--assume-returns",
                "leave",
            )
            self.assertEqual(status, 0)
            dead_stages = {
                stage["pass"]: stage for stage in dead_report["ssa_graph"]["proposals"]["stages"]
            }
            self.assertEqual(len(dead_stages["bounded_table_loads"]["journal"]), 1)
            self.assertTrue(
                any(
                    item["op"] == "load"
                    for item in dead_stages["dead_effect_free_nodes"]["journal"]
                )
            )
            self.assertTrue(
                any(
                    "bounded-table stage" in item
                    for item in dead_stages["dead_effect_free_nodes"]["assumptions"]
                )
            )
            two_words = (
                0xF100101F,
                0x540000A2,
                0x100007C1,
                0xF8607823,
                0xF8607822,
                0xD65F03C0,
                0xD65F03C0,
            )
            two_code = b"".join(struct.pack("<I", word) for word in two_words)
            two_code += bytes(0x100 - len(two_code)) + struct.pack("<QQQQ", 11, 22, 33, 44)
            path.write_bytes(self.elf(two_code))
            two_args = (
                str(path),
                "--address",
                "400080",
                "--size",
                "28",
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-abi",
                "aapcs64",
            )
            status, two_report = self.run_cli(
                "recover-regions",
                *two_args,
                "--assume-image-access",
                "readable",
                "--assume-entries",
                "closed",
                "--assume-returns",
                "leave",
            )
            self.assertEqual(status, 0)
            two_stages = {
                stage["pass"]: stage for stage in two_report["ssa_graph"]["proposals"]["stages"]
            }
            self.assertEqual(len(two_stages["bounded_table_loads"]["journal"]), 2)
            self.assertEqual(two_stages["bounded_table_loads"]["refused"], [])
            partial_words = (
                0xF100101F,
                0x540000C2,
                0x100007C1,
                0x100008A4,
                0xF8607883,
                0xF8607822,
                0xD65F03C0,
                0xD65F03C0,
            )
            partial_code = b"".join(struct.pack("<I", word) for word in partial_words)
            partial_code += bytes(0x100 - len(partial_code)) + struct.pack("<QQQQ", 11, 22, 33, 44)
            path.write_bytes(self.elf(partial_code))
            partial_args = (
                str(path),
                "--address",
                "400080",
                "--size",
                "32",
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-abi",
                "aapcs64",
            )
            status, partial_report = self.run_cli(
                "recover-regions",
                *partial_args,
                "--assume-image-access",
                "readable",
                "--assume-entries",
                "closed",
                "--assume-returns",
                "leave",
            )
            self.assertEqual(status, 0)
            partial = next(
                stage
                for stage in partial_report["ssa_graph"]["proposals"]["stages"]
                if stage["pass"] == "bounded_table_loads"
            )
            self.assertEqual(len(partial["journal"]), 1)
            self.assertEqual(
                [(item["phase"], item["reason"]) for item in partial["refused"]],
                [("constant_load", "no_invariant")],
            )
            status, refused = self.run_cli("recover-regions", *args, "--assume-entries", "open")
            self.assertEqual((status, refused["reason"]), (1, "usage"))
            status, refused = self.run_cli("cfg-probe", *args, "--assume-entries", "closed")
            self.assertEqual((status, refused["reason"]), (1, "usage"))
            path.write_bytes(self.elf(code[:0x100]))
            status, missing = self.run_cli(
                "recover-regions",
                *args,
                "--assume-image-access",
                "readable",
                "--assume-entries",
                "closed",
                "--assume-returns",
                "leave",
            )
            self.assertEqual(status, 0)
            stage = next(
                item
                for item in missing["ssa_graph"]["proposals"]["stages"]
                if item["pass"] == "bounded_table_loads"
            )
            self.assertEqual(
                (stage["outcome"], stage["reason"], stage["journal"]),
                ("declined", "candidate_refused", []),
            )
            self.assertEqual(
                [(item["phase"], item["reason"]) for item in stage["refused"]],
                [("constant_load", "no_invariant")],
            )

    def test_ssa_unreachable_stage_declares_entries_and_refuses_unknown_successor(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "control.so"
            path.write_bytes(self.elf(struct.pack("<I", 0xD65F03C0)))
            args = (
                str(path),
                "--address",
                "400080",
                "--size",
                "4",
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-abi",
                "aapcs64",
            )
            status, open_report = self.run_cli("recover-regions", *args)
            self.assertEqual(status, 0)
            open_stage = open_report["ssa_graph"]["proposals"]["stages"][0]
            self.assertEqual(
                (open_stage["pass"], open_stage["outcome"], open_stage["reason"]),
                ("unreachable_blocks", "not_run", "no_closed_entry_declaration"),
            )
            status, closed_report = self.run_cli(
                "recover-regions", *args, "--assume-entries", "closed", "--assume-returns", "leave"
            )
            self.assertEqual(status, 0)
            closed_stage = closed_report["ssa_graph"]["proposals"]["stages"][0]
            self.assertEqual(
                (closed_stage["outcome"], closed_stage["reason"], closed_stage["journal"]),
                ("unchanged", "none", []),
            )
            self.assertTrue(
                any("listed entries exhaust" in item for item in closed_stage["assumptions"])
            )
            path.write_bytes(self.elf(struct.pack("<I", 0xD61F0000)))  # br x0
            status, unknown_report = self.run_cli(
                "recover-regions", *args, "--assume-entries", "closed", "--assume-returns", "leave"
            )
            self.assertEqual(status, 0)
            unknown_stage = unknown_report["ssa_graph"]["proposals"]["stages"][0]
            self.assertEqual(
                (unknown_stage["outcome"], unknown_stage["reason"], unknown_stage["journal"]),
                ("declined", "incomplete_successors", []),
            )
            self.assertEqual(
                unknown_stage["from_revision"], unknown_report["ssa_graph"]["revision"]
            )
            self.assertIsNone(unknown_stage["to_revision"])
            unknown_phi = unknown_report["ssa_graph"]["proposals"]["stages"][1]
            self.assertEqual(
                (
                    unknown_phi["pass"],
                    unknown_phi["outcome"],
                    unknown_phi["reason"],
                    unknown_phi["journal"],
                ),
                ("phi_constant_fold", "declined", "incomplete_successors", []),
            )
            path.write_bytes(self.elf(struct.pack("<I", 0xD65F03C0)))
            status, abi_only = self.run_cli("recover-regions", *args, "--assume-entries", "closed")
            self.assertEqual(status, 0)
            self.assertFalse(abi_only["declared_return_leaves"])
            self.assertEqual(
                (
                    abi_only["ssa_graph"]["proposals"]["stages"][0]["outcome"],
                    abi_only["ssa_graph"]["proposals"]["stages"][0]["reason"],
                ),
                ("not_run", "no_return_leaves_declaration"),
            )
            no_abi_args = (
                str(path),
                "--address",
                "400080",
                "--size",
                "4",
                "--memory-profile",
                "concrete_atomic_scalar",
            )
            status, no_abi = self.run_cli(
                "recover-regions", *no_abi_args, "--assume-entries", "closed"
            )
            self.assertEqual(status, 0)
            self.assertFalse(no_abi["ssa_graph"]["declarations"]["return_leaves"])
            return_assumptions = no_abi["unflattening"]["recovered_graph"]["blocks"][0]["edges"][0][
                "assumes"
            ]
            self.assertEqual(return_assumptions, ["this target was never resolved to a block"])
            self.assertEqual(
                (
                    no_abi["ssa_graph"]["proposals"]["stages"][0]["outcome"],
                    no_abi["ssa_graph"]["proposals"]["stages"][0]["reason"],
                ),
                ("not_run", "no_return_leaves_declaration"),
            )
            status, bad_scope = self.run_cli(
                "recover-regions", *args, "--assume-returns", "unknown"
            )
            self.assertEqual((status, bad_scope["reason"]), (1, "usage"))

    def test_ssa_phi_constant_fold_declares_scope_and_journals_rewrite(self):
        words = (
            0xB4000080,
            0xD2800168,
            0x14000004,
            0xD4200000,
            0xD2800168,
            0x14000001,
            0x91000500,
            0xD65F03C0,
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "phi.so"
            path.write_bytes(self.elf(b"".join(struct.pack("<I", word) for word in words)))
            args = (
                str(path),
                "--address",
                "400080",
                "--size",
                "32",
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-abi",
                "aapcs64",
            )
            stage = lambda report: next(
                item
                for item in report["ssa_graph"]["proposals"]["stages"]
                if item["pass"] == "phi_constant_fold"
            )
            status, open_report = self.run_cli("recover-regions", *args)
            self.assertEqual(status, 0)
            self.assertEqual(
                (stage(open_report)["outcome"], stage(open_report)["reason"]),
                ("not_run", "no_closed_entry_declaration"),
            )
            status, no_return = self.run_cli("recover-regions", *args, "--assume-entries", "closed")
            self.assertEqual(status, 0)
            self.assertEqual(
                (stage(no_return)["outcome"], stage(no_return)["reason"]),
                ("not_run", "no_return_leaves_declaration"),
            )
            status, report = self.run_cli(
                "recover-regions", *args, "--assume-entries", "closed", "--assume-returns", "leave"
            )
            self.assertEqual(status, 0)
            folded = stage(report)
            self.assertEqual(folded["outcome"], "proposed")
            self.assertGreaterEqual(folded["proved_phis"], 1)
            self.assertTrue(
                any(
                    edit["original_op"] == "add" and edit["value"] == 12
                    for edit in folded["journal"]
                )
            )
            self.assertTrue(any("entries exhaust" in item for item in folded["assumptions"]))
            unreachable = next(
                item
                for item in report["ssa_graph"]["proposals"]["stages"]
                if item["pass"] == "unreachable_blocks"
            )
            self.assertEqual(
                folded["from_revision"], unreachable["to_revision"] or unreachable["from_revision"]
            )

            unequal = list(words)
            unequal[4] = 0xD28002C8  # mov x8, #22
            path.write_bytes(self.elf(b"".join(struct.pack("<I", word) for word in unequal)))
            status, report = self.run_cli(
                "recover-regions", *args, "--assume-entries", "closed", "--assume-returns", "leave"
            )
            self.assertEqual(status, 0)
            self.assertEqual(
                (stage(report)["outcome"], stage(report)["journal"]), ("unchanged", [])
            )

    def test_ssa_predecessor_copy_publishes_scope_and_journal(self):
        words = (0x8B030008, 0x14000002, 0xD4200000, 0x8B020101, 0xD65F03C0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predecessor-copy.so"
            path.write_bytes(self.elf(b"".join(struct.pack("<I", word) for word in words)))
            args = (
                str(path),
                "--address",
                "400080",
                "--size",
                "20",
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-abi",
                "aapcs64",
            )
            stage = lambda report: next(
                item
                for item in report["ssa_graph"]["proposals"]["stages"]
                if item["pass"] == "predecessor_copies"
            )
            status, report = self.run_cli("recover-regions", *args)
            self.assertEqual(status, 0)
            self.assertEqual(
                (stage(report)["outcome"], stage(report)["reason"]),
                ("not_run", "no_closed_entry_declaration"),
            )
            status, report = self.run_cli("recover-regions", *args, "--assume-entries", "closed")
            self.assertEqual(status, 0)
            self.assertEqual(
                (stage(report)["outcome"], stage(report)["reason"]),
                ("not_run", "no_return_leaves_declaration"),
            )
            status, report = self.run_cli(
                "recover-regions", *args, "--assume-entries", "closed", "--assume-returns", "leave"
            )
            self.assertEqual(status, 0)
            copied = stage(report)
            self.assertEqual((copied["outcome"], len(copied["journal"])), ("proposed", 1))
            self.assertEqual(copied["journal"][0]["from_revision"], copied["from_revision"])
            self.assertEqual(copied["journal"][0]["to_revision"], copied["to_revision"])
            self.assertTrue(any("entries exhaust" in item for item in copied["assumptions"]))
            self.assertIn("closed_entries", report["ssa_graph"]["proposals"]["text"])

    def test_ssa_dead_write_publishes_scope_and_journal(self):
        words = (0xD2800168, 0x14000003, 0xD4200000, 0xD4200000, 0xD28002C8, 0xAA0803E0, 0xD65F03C0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dead-write.so"
            path.write_bytes(self.elf(b"".join(struct.pack("<I", word) for word in words)))
            args = (
                str(path),
                "--address",
                "400080",
                "--size",
                "28",
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-abi",
                "aapcs64",
            )
            stage = lambda report: next(
                item
                for item in report["ssa_graph"]["proposals"]["stages"]
                if item["pass"] == "dead_storage_writes"
            )
            status, report = self.run_cli("recover-regions", *args)
            self.assertEqual(status, 0)
            self.assertEqual(
                (stage(report)["outcome"], stage(report)["reason"]),
                ("not_run", "no_closed_entry_declaration"),
            )
            status, report = self.run_cli("recover-regions", *args, "--assume-entries", "closed")
            self.assertEqual(status, 0)
            self.assertEqual(
                (stage(report)["outcome"], stage(report)["reason"]),
                ("not_run", "no_return_leaves_declaration"),
            )
            status, report = self.run_cli(
                "recover-regions", *args, "--assume-entries", "closed", "--assume-returns", "leave"
            )
            self.assertEqual(status, 0)
            retired = stage(report)
            self.assertEqual((retired["outcome"], len(retired["journal"])), ("proposed", 1))
            self.assertEqual(retired["journal"][0]["storage"], 8)
            self.assertEqual(retired["journal"][0]["from_revision"], retired["from_revision"])
            self.assertEqual(retired["journal"][0]["to_revision"], retired["to_revision"])
            self.assertTrue(
                any("overwritten on every path" in item for item in retired["assumptions"])
            )
            self.assertTrue(
                any("not observed asynchronously" in item for item in retired["assumptions"])
            )
            self.assertIn("dead_write boundary", report["ssa_graph"]["proposals"]["text"])

    def test_ssa_constant_propagation_follows_computed_predecessor(self):
        words = (
            0xB40000A0,
            0xD2800148,
            0x91000508,
            0x14000004,
            0xD4200000,
            0xD2800168,
            0x14000001,
            0x91000500,
            0xD65F03C0,
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "computed-phi.so"
            path.write_bytes(self.elf(b"".join(struct.pack("<I", word) for word in words)))
            args = (
                str(path),
                "--address",
                "400080",
                "--size",
                "36",
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-abi",
                "aapcs64",
            )
            stages = lambda report: {
                item["pass"]: item for item in report["ssa_graph"]["proposals"]["stages"]
            }
            status, open_report = self.run_cli("recover-regions", *args)
            self.assertEqual(status, 0)
            self.assertEqual(
                (
                    stages(open_report)["constant_propagation"]["outcome"],
                    stages(open_report)["constant_propagation"]["reason"],
                ),
                ("not_run", "no_closed_entry_declaration"),
            )
            status, no_return = self.run_cli("recover-regions", *args, "--assume-entries", "closed")
            self.assertEqual(status, 0)
            self.assertEqual(
                (
                    stages(no_return)["constant_propagation"]["outcome"],
                    stages(no_return)["constant_propagation"]["reason"],
                ),
                ("not_run", "no_return_leaves_declaration"),
            )
            status, report = self.run_cli(
                "recover-regions", *args, "--assume-entries", "closed", "--assume-returns", "leave"
            )
            self.assertEqual(status, 0)
            self.assertEqual(stages(report)["phi_constant_fold"]["outcome"], "unchanged")
            folded = stages(report)["constant_propagation"]
            self.assertEqual(folded["outcome"], "proposed")
            self.assertGreaterEqual(folded["proved_phis"], 1)
            self.assertEqual(len(folded["journal"]), 2)
            self.assertEqual({edit["value"] for edit in folded["journal"]}, {11, 12})

            unequal = list(words)
            unequal[5] = 0xD28002C8  # mov x8, #22
            path.write_bytes(self.elf(b"".join(struct.pack("<I", word) for word in unequal)))
            status, report = self.run_cli(
                "recover-regions", *args, "--assume-entries", "closed", "--assume-returns", "leave"
            )
            self.assertEqual(status, 0)
            folded = stages(report)["constant_propagation"]
            self.assertEqual([edit["value"] for edit in folded["journal"]], [11])

    def test_ssa_sccp_folds_a_join_reached_through_a_constant_branch(self):
        words = (
            0xB40000BF,
            0xD2800148,
            0x91000508,
            0x14000004,
            0xD4200000,
            0xD28002C8,
            0x14000001,
            0x91000500,
            0xD65F03C0,
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "constant-branch.so"
            path.write_bytes(self.elf(b"".join(struct.pack("<I", word) for word in words)))
            args = (
                str(path),
                "--address",
                "400080",
                "--size",
                "36",
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-abi",
                "aapcs64",
            )
            stages = lambda report: {
                item["pass"]: item for item in report["ssa_graph"]["proposals"]["stages"]
            }
            status, open_report = self.run_cli("recover-regions", *args)
            self.assertEqual(status, 0)
            self.assertEqual(
                stages(open_report)["sccp_constant_fold"]["reason"], "no_closed_entry_declaration"
            )
            status, report = self.run_cli(
                "recover-regions", *args, "--assume-entries", "closed", "--assume-returns", "leave"
            )
            self.assertEqual(status, 0)
            self.assertEqual(stages(report)["constant_propagation"]["outcome"], "proposed")
            folded = stages(report)["sccp_constant_fold"]
            self.assertEqual(folded["outcome"], "proposed")
            self.assertGreaterEqual(folded["proved_phis"], 1)
            self.assertTrue(any(edit["value"] == 23 for edit in folded["journal"]))
            reads = stages(report)["sccp_storage_reads"]
            self.assertEqual(reads["outcome"], "proposed")
            self.assertTrue(
                any(
                    edit["original_op"] == "read" and edit["read_phi"] >= 0
                    for edit in reads["journal"]
                )
            )
            self.assertGreater(folded["executable_blocks"], 0)
            self.assertGreater(folded["selected_edges"], 0)
            self.assertEqual(len(folded["excluded_edges"]), 1)
            self.assertEqual(
                (
                    folded["excluded_edges"][0]["condition_value"],
                    folded["excluded_edges"][0]["when"],
                ),
                (1, False),
            )
            self.assertEqual(folded["excluded_edges"][0]["graph_revision"], folded["from_revision"])
            retired = stages(report)["branch_retirement"]
            self.assertEqual((retired["outcome"], len(retired["journal"])), ("proposed", 1))
            cleaned = stages(report)["unreachable_after_sccp"]
            self.assertEqual(cleaned["outcome"], "proposed")
            self.assertEqual(sum(item["kind"] == "removed_block" for item in cleaned["journal"]), 1)

    def test_declared_range_the_loader_relocates_is_refused(self):
        # The same jump, now with a relative relocation over the offset word:
        # the file byte there is not the value the run sees.
        data = bytearray(0x200)
        data[:16] = b"\x7fELF\x02\x01\x01" + bytes(9)
        struct.pack_into(
            "<HHIQQQIHHHHHH", data, 16, 3, 183, 1, 0x400100, 64, 0, 0, 64, 56, 2, 0, 0, 0
        )
        struct.pack_into("<IIQQQQQQ", data, 64, 1, 7, 0, 0x400000, 0, len(data), len(data), 4096)
        struct.pack_into("<IIQQQQQQ", data, 120, 2, 6, 0x180, 0x400180, 0, 80, 80, 8)
        data[0x100:0x118] = bytes.fromhex("090000102a1140b929010a8b20011fd614000000c0035fd6")
        struct.pack_into("<QQQQQQQQ", data, 0x180, 7, 0x4001D0, 8, 24, 9, 24, 0, 0)
        struct.pack_into("<QQq", data, 0x1D0, 0x400110, 1027, 0x14)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "relocated.so"
            path.write_bytes(data)
            args = [
                str(path),
                "--address",
                "400100",
                "--size",
                "24",
                "--memory-profile",
                "concrete_atomic_scalar",
            ]
            # The slot is eight bytes wide, so a range starting inside it is
            # written as surely as one starting at it.
            for spec in ("400110:4", "400114:4", "40010c:8"):
                status, report = self.run_cli(
                    "recover-regions", *args, "--assume-initialized", spec
                )
                self.assertEqual(status, 1, spec)
                self.assertEqual(report["reason"], "invalid_range", spec)
                self.assertIn("loader may write", report["detail"])
            status, report = self.run_cli(
                "recover-regions", *args, "--assume-initialized", "400120:4"
            )
            self.assertEqual(status, 0)
            self.assertEqual(
                report["declared_constant_ranges"], [{"address": 0x400120, "bytes": 4}]
            )
            # An image that relocates its own text disowns its file bytes, so no
            # declaration over them is admitted at all. The array stays terminated.
            struct.pack_into("<QQQQ", data, 0x1B0, 22, 0, 0, 0)
            path.write_bytes(data)
            status, report = self.run_cli(
                "recover-regions", *args, "--assume-initialized", "400120:4"
            )
            self.assertEqual(status, 1)
            self.assertIn("relocates its text", report["detail"])

    def relocated_slot_image(self, relro):
        """One relocated slot at 0x400120, optionally inside PT_GNU_RELRO."""
        data = bytearray(0x200)
        data[:16] = b"\x7fELF\x02\x01\x01" + bytes(9)
        headers = 3 if relro else 2
        struct.pack_into(
            "<HHIQQQIHHHHHH", data, 16, 3, 183, 1, 0x400100, 64, 0, 0, 64, 56, headers, 0, 0, 0
        )
        struct.pack_into("<IIQQQQQQ", data, 64, 1, 7, 0, 0x400000, 0, len(data), len(data), 4096)
        struct.pack_into("<IIQQQQQQ", data, 120, 2, 6, 0x180, 0x400180, 0, 80, 80, 8)
        if relro:
            struct.pack_into("<IIQQQQQQ", data, 176, 0x6474E552, 4, 0x120, 0x400120, 0, 8, 8, 1)
        data[0x100:0x104] = bytes.fromhex("c0035fd6")
        struct.pack_into("<QQQQQQQQ", data, 0x180, 7, 0x4001D0, 8, 24, 9, 24, 0, 0)
        struct.pack_into("<QQq", data, 0x1D0, 0x400120, 1027, 0x14)
        return bytes(data)

    def test_only_relocation_protected_slots_supply_a_value_by_default(self):
        # The loader writes every relocated slot, but only keeps the ones it
        # then maps read-only. A slot in writable data is an ordinary global.
        with tempfile.TemporaryDirectory() as temp:
            for relro, stable in ((False, 0), (True, 1)):
                path = Path(temp) / f"slot{int(relro)}.so"
                path.write_bytes(self.relocated_slot_image(relro))
                args = [
                    str(path),
                    "--address",
                    "400100",
                    "--size",
                    "4",
                    "--memory-profile",
                    "concrete_atomic_scalar",
                ]
                status, report = self.run_cli("recover-regions", *args)
                self.assertEqual(status, 0)
                self.assertEqual(report["declared_relocated_pointer_slots"], 1)
                self.assertEqual(report["relocated_pointers"], stable, f"relro={relro}")
                names = " ".join(report["execution_assumptions"])
                if stable:
                    self.assertIn("PT_GNU_RELRO", names)
                # The caller may take on the wider restriction, and it is printed.
                status, declared = self.run_cli(
                    "recover-regions", *args, "--assume-relocations", "stable"
                )
                self.assertEqual(status, 0)
                self.assertEqual(declared["relocated_pointers"], 1)
                self.assertIn(
                    "the loader-writable ones included", " ".join(declared["execution_assumptions"])
                )
                # A store it cannot place is one it did not examine, like one elsewhere.
                self.assertIn(
                    "stores this range models but cannot place, and stores it does"
                    " not model, were not examined; a slot this range itself writes"
                    " gives no value to stores through it",
                    " ".join(declared["execution_assumptions"]),
                )
            status, refused = self.run_cli("recover-regions", *args, "--assume-relocations", "yes")
            self.assertEqual((status, refused["reason"]), (1, "usage"))

    def initializer_image(self):
        """Executable code in one segment writing a relocated slot in another."""
        data = bytearray(0x2000)
        data[:16] = b"\x7fELF\x02\x01\x01" + bytes(9)
        struct.pack_into(
            "<HHIQQQIHHHHHH", data, 16, 3, 183, 1, 0x400100, 64, 0, 0, 64, 56, 3, 0, 0, 0
        )
        struct.pack_into("<IIQQQQQQ", data, 64, 1, 5, 0, 0x400000, 0, 0x1000, 0x1000, 4096)
        struct.pack_into("<IIQQQQQQ", data, 120, 1, 6, 0x1000, 0x401000, 0, 0x1000, 0x1000, 4096)
        struct.pack_into("<IIQQQQQQ", data, 176, 2, 6, 0x1180, 0x401180, 0, 80, 80, 8)
        # mov x8,#42 ; adrp x9,0x401000 ; add x9,x9,#0x120 ; str x8,[x9] ; ret
        data[0x100:0x114] = bytes.fromhex("480580d2090000b029810491280100f9c0035fd6")
        struct.pack_into("<QQQQQQQQ", data, 0x1180, 7, 0x4011D0, 8, 24, 9, 24, 0, 0)
        struct.pack_into("<QQq", data, 0x11D0, 0x401120, 1027, 0x14)
        return bytes(data)

    def test_derive_memory_runs_an_initializer_and_reports_what_it_rests_on(self):
        # The slot the code writes is one the loader relocated, so the loader's
        # value and the run's differ and the record has something to say.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "init.so"
            path.write_bytes(self.initializer_image())
            record = Path(temp) / "derived.nyxmem"
            status, report = self.run_cli(
                "derive-memory", str(path), "--entry", "400100", "--record", str(record)
            )
            self.assertEqual(status, 0, report)
            self.assertEqual(report["status"], "returned")
            self.assertEqual(report["relocations_applied"], 1)
            # One observed execution never authorizes a rewrite by itself.
            self.assertFalse(report["authorizes_transformation"])
            self.assertTrue(report["rests_on"])
            written = {item["address"]: item["value"] for item in report["writes"]}
            self.assertEqual(written.get(0x401120), 42)
            text = record.read_text()
            self.assertIn("NYXMEM01", text)
            # The record names the image it came from, so one made from another
            # build cannot apply by raw address.
            self.assertIn("image ", text)
            self.assertNotIn("70000000", text)  # the run's own stack is not image memory

            # The recovery run admits the observed value for that slot.
            args = [
                str(path),
                "--address",
                "400100",
                "--size",
                "20",
                "--memory-profile",
                "concrete_atomic_scalar",
            ]
            status, plain = self.run_cli("recover-regions", *args)
            self.assertEqual(status, 0)
            # Observed values came from an execution at one placement, so
            # admitting them without saying which is refused.
            status, refused = self.run_cli(
                "recover-regions", *args, "--derived-memory", str(record)
            )
            self.assertEqual((status, refused["reason"]), (1, "usage"))
            status, derived = self.run_cli(
                "recover-regions", *args, "--derived-memory", str(record), "--assume-load-bias", "0"
            )
            self.assertEqual(status, 0, derived)
            # 42 is not an address in this image, so it is not offered as one.
            self.assertEqual(derived["derived_memory_slots"], 0)
            for content in (
                "NOTAHEADER\n",
                "NYXMEM01\n1c4c598 7c8c2c\n",  # no image named
                "NYXMEM01\nimage %s\n" % ("0" * 64),
            ):  # names another image
                bad = Path(temp) / "bad.nyxmem"
                bad.write_text(content)
                status, refused = self.run_cli(
                    "recover-regions",
                    *args,
                    "--derived-memory",
                    str(bad),
                    "--assume-load-bias",
                    "0",
                )
                self.assertEqual((status, refused["reason"]), (1, "invalid_record"), content)

            # A writer record admits the slots it omits, so it must name its
            # image and say how much of that image went unexamined.
            writers = Path(temp) / "w.nyxwrt"
            digest = [l for l in text.splitlines() if l.startswith("image ")][0]
            # It must also say how many relocated slots it examined, and that
            # must be the set this image declares: one relative slot here.
            for content, ok in (
                (f"NYXWRT01\n{digest}\nslots 1\nunexamined 0\n", True),
                (f"NYXWRT01\n{digest}\nunexamined 0\n", False),
                (f"NYXWRT01\n{digest}\nslots 0\nunexamined 0\n", False),
                (f"NYXWRT01\n{digest}\nslots 1\n", False),
                ("NYXWRT01\nslots 1\nunexamined 0\n", False),
            ):
                writers.write_text(content)
                status, report = self.run_cli(
                    "recover-regions", *args, "--image-writers", str(writers)
                )
                if ok:
                    self.assertEqual(status, 0, report)
                else:
                    self.assertEqual((status, report["reason"]), (1, "invalid_record"), content)

    def symbol_image(self, defined):
        """Code copying a slot the loader fills from symbol 1 into another slot."""
        data = bytearray(self.initializer_image())
        # adrp x9,0x401000 ; ldr x8,[x9,#0x128] ; str x8,[x9,#0x120] ; ret
        data[0x100:0x110] = bytes.fromhex("090000b0289540f9289100f9c0035fd6")
        data[0x100 + 0x10 : 0x114] = bytes(4)
        struct.pack_into("<IIQQQQQQ", data, 176, 2, 6, 0x1180, 0x401180, 0, 96, 96, 8)
        struct.pack_into(
            "<QQQQQQQQQQQQ", data, 0x1180, 7, 0x4011E0, 8, 24, 9, 24, 6, 0x401200, 11, 24, 0, 0
        )
        data[0x11D0:0x11E0] = bytes(16)
        struct.pack_into("<QQq", data, 0x11E0, 0x401128, 1 << 32 | 257, 0)  # R_AARCH64_ABS64
        # Symbol 1: a default-visibility global function at the entry, or an import.
        struct.pack_into(
            "<IBBHQQ", data, 0x1218, 0, 0x12, 0, 1 if defined else 0, 0x400100 if defined else 0, 0
        )
        return bytes(data)

    def test_a_run_never_reads_a_slot_whose_loader_value_it_does_not_know(self):
        with tempfile.TemporaryDirectory() as temp:
            imported = Path(temp) / "import.so"
            imported.write_bytes(self.symbol_image(False))
            defined = Path(temp) / "defined.so"
            defined.write_bytes(self.symbol_image(True))
            record = Path(temp) / "derived.nyxmem"
            # An import's slot has no value the file states, declared or not:
            # the run stops where it reads it instead of reading the file's
            # zero, and a record of that run is not admitted.
            for extra in ((), ("--assume-symbol-binding", "own")):
                status, report = self.run_cli(
                    "derive-memory",
                    str(imported),
                    "--entry",
                    "400100",
                    "--record",
                    str(record),
                    *extra,
                )
                self.assertEqual(status, 0, report)
                self.assertEqual(report["status"], "unsupported", extra)
                self.assertEqual(report["writes"], [], extra)
            args = [
                str(imported),
                "--address",
                "400100",
                "--size",
                "20",
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-load-bias",
                "0",
            ]
            status, refused = self.run_cli(
                "recover-regions", *args, "--derived-memory", str(record)
            )
            self.assertEqual((status, refused["reason"]), (1, "invalid_record"))
            self.assertIn("did not return", refused["detail"])

            # A symbol the image defines names a location, but whether the
            # dynamic linker binds this definition is a declaration.
            status, report = self.run_cli("derive-memory", str(defined), "--entry", "400100")
            self.assertEqual((status, report["status"]), (0, "unsupported"))
            status, report = self.run_cli(
                "derive-memory",
                str(defined),
                "--entry",
                "400100",
                "--assume-symbol-binding",
                "own",
                "--record",
                str(record),
            )
            self.assertEqual((status, report["status"]), (0, "returned"), report)
            written = {item["address"]: item["value"] for item in report["writes"]}
            self.assertEqual(written.get(0x401120), 0x400100)
            self.assertTrue(any("own definition" in line for line in report["rests_on"]))
            text = record.read_text()
            self.assertIn("status returned\n", text)
            self.assertIn("binding own\n", text)
            args = [
                str(defined),
                "--address",
                "400100",
                "--size",
                "20",
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-load-bias",
                "0",
                "--derived-memory",
                str(record),
            ]
            status, refused = self.run_cli("recover-regions", *args)
            self.assertEqual((status, refused["reason"]), (1, "usage"))
            self.assertIn("--assume-symbol-binding own", refused["detail"])
            status, admitted = self.run_cli(
                "recover-regions", *args, "--assume-symbol-binding", "own"
            )
            self.assertEqual(status, 0, admitted)
            self.assertTrue(
                any("own definition" in line for line in admitted["execution_assumptions"])
            )
            status, refused = self.run_cli(
                "recover-regions", *args, "--assume-symbol-binding", "yes"
            )
            self.assertEqual((status, refused["reason"]), (1, "usage"))

    def test_a_declared_range_may_name_bytes_the_loader_zeroes(self):
        # A segment whose memory reaches past its file is zero there, which
        # the format states rather than the run assuming it. Declaring that
        # the zero stays is the same restriction as declaring a file value
        # stays, and a range in neither is still refused.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bss.so"
            path.write_bytes(self.initializer_image())
            args = [
                str(path),
                "--address",
                "400100",
                "--size",
                "20",
                "--memory-profile",
                "concrete_atomic_scalar",
            ]
            # Nothing in this fixture reaches past its file, so a range in
            # neither the file nor a zero tail is refused.
            for spec in ("402000:8", "500000:8"):
                status, refused = self.run_cli(
                    "recover-regions", *args, "--assume-initialized", spec
                )
                self.assertEqual((status, refused["reason"]), (1, "invalid_range"), spec)
                self.assertIn("zero-filled", refused["detail"])

            # A tail the loader does zero is admitted, and carries the value
            # it names into the fact table rather than only being accepted.
            image = bytearray(self.initializer_image())
            struct.pack_into("<Q", image, 160, 0x2000)  # second LOAD memsz
            tail = Path(temp) / "tail.so"
            tail.write_bytes(bytes(image))
            status, report = self.run_cli(
                "recover-regions",
                str(tail),
                "--address",
                "400100",
                "--size",
                "20",
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-initialized",
                "402040:8",
            )
            self.assertEqual(status, 0, report)
            self.assertEqual(
                report["declared_constant_ranges"], [{"address": 0x402040, "bytes": 8}]
            )
            self.assertEqual(
                report["constant_image_ranges"], report["declared_constant_image_ranges"]
            )

    def test_recovery_derives_in_process_what_a_record_would_have_carried(self):
        # The same observation the two-command route carries in a record, made
        # by the run that needs it, so the two must agree on what they admit.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "init.so"
            path.write_bytes(self.initializer_image())
            args = [
                str(path),
                "--address",
                "400100",
                "--size",
                "20",
                "--memory-profile",
                "concrete_atomic_scalar",
            ]
            status, refused = self.run_cli("recover-regions", *args, "--derive-entry", "400100")
            self.assertEqual((status, refused["reason"]), (1, "usage"))
            status, report = self.run_cli(
                "recover-regions", *args, "--derive-entry", "400100", "--assume-load-bias", "0"
            )
            self.assertEqual(status, 0, report)
            runs = report["derivation_runs"]
            self.assertEqual(
                [(run["entry"], run["status"]) for run in runs], [(0x400100, "returned")]
            )
            self.assertEqual(runs[0]["writes"], 1)
            # 42 is not an address in this image, so it is not offered as one,
            # exactly as the record route reports it.
            self.assertEqual(report["derived_memory_slots"], 0)
            # A target outside any executable segment is reported, not run.
            status, outside = self.run_cli(
                "recover-regions", *args, "--derive-entry", "401120", "--assume-load-bias", "0"
            )
            self.assertEqual(status, 0, outside)
            self.assertEqual(outside["derivation_runs"][0]["status"], "not_executable")
            # Nothing in this population calls anything, so the pass that looks
            # for callees to run finds none and stops.
            status, auto = self.run_cli(
                "recover-regions", *args, "--derive-entry", "auto", "--assume-load-bias", "0"
            )
            self.assertEqual(status, 0, auto)
            self.assertEqual(auto["derivation_runs"], [])
            # A bias off a page boundary contradicts the page placement a PC
            # page is read under, so it is refused rather than half applied.
            status, unaligned = self.run_cli("recover-regions", *args, "--assume-load-bias", "800")
            self.assertEqual((status, unaligned["reason"]), (1, "usage"))
            status, aligned = self.run_cli("recover-regions", *args, "--assume-load-bias", "1000")
            self.assertEqual(status, 0, aligned)

    def test_path_bad_ranges_and_internal_calls_publish_no_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "path.so"
            path.write_bytes(self.elf(bytes.fromhex("020000941f2003d5")))
            for ranges in (
                "",
                "400080:0",
                "400081:4",
                "400080:3",
                "400080:4,",
                ",400080:4",
                "fffffffffffffffc:8",
                "400080:16388",
                "400080:8",
            ):
                with self.subTest(ranges=ranges):
                    status, report = self.run_cli("simplify-path", str(path), "--ranges", ranges)
                    self.assertEqual(status, 1)
                    self.assertEqual(report["outcome"], "refused")
                    self.assertNotIn("after", report)
                    self.assertNotIn("memory_journal", report)


if __name__ == "__main__":
    unittest.main()
