"""CPU tests for dataset-independent calibration and video utilities."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selection = load_file("selection", ROOT / "tools/select_calibration_indices.py")
assembly = load_file("assembly", ROOT / "runtime/h3/assemble_config.py")


class ReleaseInputsTest(unittest.TestCase):
    def test_selection_for_arbitrary_lengths(self):
        for size in (1, 2, 7, 10, 23, 501):
            indices = selection.select_indices(size)
            self.assertEqual(len(indices), min(10, size))
            self.assertEqual(len(set(indices)), len(indices))
            self.assertEqual(indices[0], 0)
            self.assertEqual(indices[-1], size - 1)
        self.assertEqual(selection.select_indices(25, "3"), [0, 12, 24])
        self.assertEqual(selection.select_indices(25, indices="7,1,20"), [7, 1, 20])

    def test_invalid_selection(self):
        for size, count, indices in ((0, "auto", ""), (2, "3", ""),
                                      (5, "auto", "1,1"), (5, "auto", "-1"),
                                      (5, "auto", "5"), (5, "3", "0,1")):
            with self.subTest(size=size, count=count, indices=indices):
                with self.assertRaises(ValueError):
                    selection.select_indices(size, count, indices)

    def test_prompt_cli_ignores_empty_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.txt"
            path.write_text("first\n\n  \nsecond\n", encoding="utf-8")
            result = subprocess.check_output([sys.executable, str(ROOT / "tools/select_calibration_indices.py"),
                                               "--prompts", str(path)], text=True)
            self.assertEqual(result.strip(), "2 0,1")

    def make_shard(self, root, shard, index):
        videos = root / shard / "videos"
        videos.mkdir(parents=True, exist_ok=True)
        path = videos / f"{index:02d}_example.mp4"
        path.write_bytes(b"placeholder")
        path.with_suffix(".json").write_text(json.dumps({"index": index}))

    def test_assembly_multiple_shards_and_large_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(1002):
                self.make_shard(root, f"shard_{index % 3}", index)
            records, sources = assembly.collect_records(root, [{}] * 1002)
            self.assertEqual(len(records), 1002)
            self.assertEqual(len(sources), 2004)

    def test_assembly_missing_duplicate_extra(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_shard(root, "shard_a", 0)
            with self.assertRaisesRegex(RuntimeError, "missing"):
                assembly.collect_records(root, [{}, {}])
            self.make_shard(root, "shard_b", 1)
            with self.assertRaisesRegex(RuntimeError, "extra"):
                assembly.collect_records(root, [{}])
            self.make_shard(root, "shard_c", 1)
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                assembly.collect_records(root, [{}, {}])

    def test_video_audit_with_custom_dimensions(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = cv2.VideoWriter(str(root / "custom_name.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 8, (32, 24))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
            writer.release()
            command = [sys.executable, str(ROOT / "runtime/h3/audit_videos.py"), str(root),
                       "--output", str(root / "audit.json"), "--width", "32", "--height", "24"]
            subprocess.run(command, check=True, capture_output=True)
            report = json.loads((root / "audit.json").read_text())
            self.assertEqual(report["count"], 1)
            self.assertEqual(report["videos"][0]["decoded_frames"], 3)
            failed = subprocess.run(command + ["--expected-count", "2"], capture_output=True)
            self.assertNotEqual(failed.returncode, 0)

    def test_wan22_calibration_validation_uses_actual_prompt_count(self):
        import torch
        for count in (1, 3, 10):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                names = [f"{expert}.linear_{i}" for expert in ("high_noise", "low_noise")
                         for i in range(400)]
                config = {"model": "/path/to/custom-checkpoint", "dit_quant_method": "pulseprofile",
                          "dit_weight_bits": 4, "dit_activation_bits": 4,
                          "dit_quant_calibration_prompt_count": count,
                          "dit_quant_calibration_prompt_indices": list(range(count))}
                experts = {"high_noise": {"calibration_step_indices": [0, 1, 2, 3],
                                          "required_calibration_calls": 8 * count},
                           "low_noise": {"calibration_step_indices": [4, 5],
                                         "required_calibration_calls": 4 * count}}
                payloads = {"config.json": config, "dit_quant_expert_calibration.json": experts,
                            "dit_quant_cache_save.json": {"modules": 800},
                            "dit_quant_audit.json": {"modules": [
                                {"module": name, "weight_bits": 4, "activation_bits": 4}
                                for name in names]}}
                for name, payload in payloads.items():
                    (root / name).write_text(json.dumps(payload))
                cache = root / "cache.pt"
                torch.save({"format": "pulse-finalized-v2", "modules": dict.fromkeys(names)}, cache)
                result = subprocess.run([sys.executable,
                    str(ROOT / "runtime/wan22/wan22_validate_pulse_calibration.py"),
                    "--output", str(root), "--cache", str(cache), "--bits", "4"],
                    text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["high_noise_calibration_calls"], 8 * count)


if __name__ == "__main__":
    unittest.main()
