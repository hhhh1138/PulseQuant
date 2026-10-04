"""CPU-only tests of the bundled prompt and seed policy contracts."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from pulsequant_utils.seed_policy import resolve_prompt_seed
from pulsequant_utils.prompt_policy import load_nonempty_prompts, select_calibration_prompts


class HelperTests(unittest.TestCase):
    def test_seed_modes(self):
        self.assertEqual(resolve_prompt_seed(42, "fixed", "absolute", 10, 2), 42)
        self.assertEqual(resolve_prompt_seed(42, "per_prompt", "absolute", 10, 2), 54)
        self.assertEqual(resolve_prompt_seed(42, "per_prompt", "local", 10, 2), 44)

    def test_shard_invariance(self):
        self.assertEqual(resolve_prompt_seed(42, "per_prompt", "absolute", 10, 2),
                         resolve_prompt_seed(42, "per_prompt", "absolute", 0, 12))

    def test_prompt_selection(self):
        self.assertEqual(select_calibration_prompts(["a", "b", "c"], [2, 0], 2), ["c", "a"])
        with self.assertRaises(ValueError):
            select_calibration_prompts(["a"], [3], 1)
        with self.assertRaises(ValueError):
            select_calibration_prompts(["a"], [0], 2)

    def test_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.txt"
            path.write_text(" a \n\n b\n", encoding="utf-8")
            self.assertEqual(load_nonempty_prompts(path), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
