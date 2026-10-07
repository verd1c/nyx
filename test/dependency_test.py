import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
ALLOW = {
    "support": {"support"},
    "ir": {"ir", "support"},
    "format": {"format", "support"},
    "target": {"target", "ir", "support"},
    "eval": {"eval", "ir", "support"},
    "recovery": {"recovery", "ir", "support"},
    "analysis": {"analysis", "ir", "support"},
    "passes": {"passes", "analysis", "recovery", "ir", "support"},
    "verify": {"verify", "support"},
}


def violations(module, text):
    return [
        dependency
        for dependency in re.findall(r'^\s*#\s*include\s*[<"]nyx/([^/]+)/', text, re.M)
        if dependency not in ALLOW[module]
    ]


class DependencyTest(unittest.TestCase):
    def test_production_include_boundaries(self):
        checked = 0
        for directory in (ROOT / "include/nyx", ROOT / "lib"):
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or path.suffix not in (".h", ".hpp", ".cpp"):
                    continue
                module = path.relative_to(directory).parts[0]
                self.assertIn(module, ALLOW, f"declare boundaries for new module {module}")
                self.assertEqual(violations(module, path.read_text()), [], str(path))
                checked += 1
        self.assertGreater(checked, 0)

    def test_generic_module_cannot_include_target(self):
        for directive in (
            '#include "nyx/target/a64/decode.hpp"',
            "# include <nyx/target/a64/decode.hpp>",
        ):
            self.assertEqual(violations("eval", directive), ["target"])
            self.assertEqual(violations("ir", directive), ["target"])
            self.assertEqual(violations("passes", directive), ["target"])


if __name__ == "__main__":
    unittest.main()
