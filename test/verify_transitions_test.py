import copy
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.oracle import verify_transitions

BINARY = sys.argv.pop(1)

# A flattened function whose cases keep their state at [x19+8] and write data
# through x27 = x19+16, with its jump table three pages past the code so the
# oracle can map it apart. Case 1 calls out before storing its state.
CODE = bytes.fromhex(
    "fd7bbfa9fd030091ff0301d1f30300917b4200917f0a00b90d000014"
    "28008052680a00b97f0700f909000014"
    "f503009448008052680a00b97f0300f904000014"
    "bf030091fd7bc1a8c0035fd6"
    "680a40b90909007168ffff54497901100afeff102b7968784a090b8b40011fd6"
)
TABLE = bytes.fromhex("0000040009000000")


def image():
    code = bytearray(0x2F80 + len(TABLE))
    code[: len(CODE)] = CODE
    code[0x2F80:] = TABLE
    data = bytearray(128 + len(code))
    data[:16] = b"\x7fELF\x02" + bytes([1, 1]) + bytes(9)
    struct.pack_into("<HHIQQQIHHHHHH", data, 16, 3, 183, 1, 0x400080, 64, 0, 0, 64, 56, 1, 0, 0, 0)
    struct.pack_into("<IIQQQQQQ", data, 64, 1, 5, 0, 0x400000, 0, len(data), len(data), 4096)
    data[128:] = code
    return bytes(data)


class VerifyTransitionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = pathlib.Path(cls.directory.name) / "flattened.so"
        cls.path.write_bytes(image())
        result = subprocess.run(
            [
                BINARY,
                "recover-regions",
                str(cls.path),
                "--address",
                "400080",
                "--size",
                str(len(CODE)),
                "--memory-profile",
                "concrete_atomic_scalar",
                "--assume-abi",
                "aapcs64",
                "--assume-returns",
                "leave",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        cls.artifact = json.loads(result.stdout)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def verify(self, artifact, index):
        path = pathlib.Path(self.directory.name) / ("artifact-%d.json" % index)
        path.write_text(json.dumps(artifact))
        return verify_transitions.verify((str(path), str(self.path), index, 16, 1))

    def test_every_transition_matches_the_original_under_qemu(self):
        transitions = self.artifact["unflattening"]["transitions"]
        self.assertEqual([t["entry"] for t in transitions], [0x400080, 0x40009C, 0x4000B0])
        for index in range(len(transitions)):
            record = self.verify(self.artifact, index)
            self.assertEqual(record["disposition"], "verified", record)
            self.assertGreaterEqual(record["completed"], 8)
            self.assertTrue(record["direct_branch_matches"])

    def chain(self, artifact, first, second):
        path = pathlib.Path(self.directory.name) / "chain.json"
        path.write_text(json.dumps(artifact))
        return verify_transitions.verify_chain((str(path), str(self.path), (first, second), 16, 1))

    def test_the_entry_establishes_what_the_next_case_rests_on(self):
        # The entry block computes x27 = x19+16 itself; case 0 relies on it
        # without constructing it, so only a correct composition matches.
        record = self.chain(self.artifact, 0, 1)
        self.assertEqual(record["disposition"], "verified", record)
        wrong = copy.deepcopy(self.artifact)
        path = wrong["regions"][wrong["unflattening"]["transitions"][1]["candidate"]]["after"]
        store = next(
            node for node in path["nodes"] if node["op"] == "store" and node["width"] == 32
        )
        path["nodes"][store["inputs"][1]]["immediate"] = 2
        self.assertEqual(self.chain(wrong, 0, 1)["disposition"], "refuted")

    def test_a_wrong_destination_is_refuted(self):
        wrong = copy.deepcopy(self.artifact)
        wrong["unflattening"]["transitions"][1]["destinations"] = [0x4000C0]
        record = self.verify(wrong, 1)
        self.assertEqual(
            (record["disposition"], record["reason"]),
            ("refuted", "the next PC is not a published destination"),
        )

    def test_a_wrong_recovered_value_is_refuted(self):
        wrong = copy.deepcopy(self.artifact)
        path = wrong["regions"][wrong["unflattening"]["transitions"][1]["candidate"]]["after"]
        # The next state the case stores, as the recovered path holds it.
        store = next(
            node for node in path["nodes"] if node["op"] == "store" and node["width"] == 32
        )
        state = path["nodes"][store["inputs"][1]]
        self.assertEqual((state["op"], state["immediate"]), ("constant", 1))
        state["immediate"] = 2
        record = self.verify(wrong, 1)
        self.assertEqual(record["disposition"], "refuted", record)

    def test_states_that_break_the_entry_relation_refute_what_rests_on_it(self):
        # Without the relation, x27 may land on x19; the path forwarded the
        # state past [x27+8] only because the relation says it cannot.
        broken = copy.deepcopy(self.artifact)
        transition = broken["unflattening"]["transitions"][1]
        self.assertTrue(transition["entry_relation_dependency"])
        del broken["regions"][transition["candidate"]]["entry_relations"]
        record = self.verify(broken, 1)
        self.assertEqual(
            (record["disposition"], record["reason"]), ("refuted", "state or next PC differs")
        )


if __name__ == "__main__":
    unittest.main()
