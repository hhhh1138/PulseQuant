"""CPU regression tests for the PulseQuant-only release boundary."""
import os
import ast
import re
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from pulsequant_utils.release_policy import validate_release_settings


class ReleasePolicyTest(unittest.TestCase):
    def test_packaged_launcher_arguments(self):
        root = Path(__file__).resolve().parents[1]
        def flags(path):
            return {arg.value for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                    for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
        extra = flags(root / "tools/select_calibration_indices.py") | flags(
            root / "runtime/wan22/wan22_validate_pulse_calibration.py")
        for model, runner in (("wan21_1_3b", "wan21/wan_runner/run_method.py"),
                              ("wan21_14b", "wan21/wan_runner/run_method.py"),
                              ("wan22_a14b", "wan22/wan_runner/run_method.py"),
                              ("minimax_h3", "h3/run_h3_w4_quant_suite.py"),
                              ("self_forcing", "self_forcing/self_forcing_quant_runner_walsh.py")):
            known = flags(root / "runtime" / runner) | extra
            for script in (root / "models" / model).glob("*.sh"):
                with self.subTest(script=script.name, model=model):
                    used = set(re.findall(r"--[a-z][a-z0-9-]+", script.read_text()))
                    self.assertFalse(used - known, sorted(used - known))

    def test_default_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            validate_release_settings()

    def test_paper_settings(self):
        with patch.dict(os.environ, {"PULSE_WALSH2_AXIS_MODE": "activation_pca",
                                     "PULSE_ORBIT_RADIAL_SPAN": ".08",
                                     "PULSE_TEMPORAL_DELTA_WEIGHT": "0"}, clear=True):
            validate_release_settings()

    def test_retired_switch_rejected(self):
        for env, value in (("PULSE_ACTIVATION_CODEBOOK", "gaussian_max"),
                           ("PULSE_WALSH2_AXIS_MODE", "random_orthogonal"),
                           ("PULSE_ORBIT_RADIAL_LS", "1"),
                           ("PULSE_TEMPORAL_DELTA_WEIGHT", "1")):
            with self.subTest(env=env), patch.dict(os.environ, {env: value}, clear=True):
                with self.assertRaisesRegex(ValueError, "retired experiment"):
                    validate_release_settings()


if __name__ == "__main__":
    unittest.main()
