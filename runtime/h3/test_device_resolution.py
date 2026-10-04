from __future__ import annotations

import unittest
from unittest.mock import patch

from run_h3_w4_quant_suite import resolve_devices


class DeviceResolutionTest(unittest.TestCase):
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.is_available", return_value=True)
    def test_single_gpu_shares_cuda_zero(self, _available, _count) -> None:
        self.assertEqual(resolve_devices(None, None), ("cuda:0", "cuda:0"))

    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.is_available", return_value=True)
    def test_two_gpus_split_text_and_dit(self, _available, _count) -> None:
        self.assertEqual(resolve_devices(None, None), ("cuda:0", "cuda:1"))

    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.is_available", return_value=True)
    def test_invalid_explicit_device_is_rejected(self, _available, _count) -> None:
        with self.assertRaisesRegex(ValueError, "only 1 CUDA device"):
            resolve_devices("cuda:0", "cuda:1")


if __name__ == "__main__":
    unittest.main()
