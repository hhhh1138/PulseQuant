"""Regression tests for Wan checkpoint directory routing (no weights needed)."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]


def load_adapter(family):
    path = ROOT / 'runtime' / family / 'wan_runner' / 'local_loader.py'
    spec = importlib.util.spec_from_file_location(f'{family}_adapter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LoaderLayoutTests(unittest.TestCase):
    def check_layout(self, family, model_path, expected_paths):
        adapter = load_adapter(family)
        transformer = MagicMock()
        with patch.object(adapter.AutoTokenizer, 'from_pretrained'), \
             patch.object(adapter, 'load_text_encoder') as text_load, \
             patch.object(adapter, 'load_vae') as vae_load, \
             patch.object(adapter, 'FlowMatchEulerDiscreteScheduler'), \
             patch.object(adapter, 'load_transformer', return_value=transformer) as load, \
             patch.object(adapter, 'WanPipeline') as pipeline:
            adapter.load_local_pipeline(model_path)
        assets = ROOT / 'configs' / 'wan_assets' / family
        text_load.assert_called_once_with(Path(model_path), assets)
        vae_load.assert_called_once_with(Path(model_path), assets)
        self.assertEqual([c.args[0] for c in load.call_args_list], expected_paths)
        kwargs = pipeline.call_args.kwargs
        self.assertIs(kwargs['transformer'], transformer)
        if family == 'wan21':
            self.assertNotIn('transformer_2', kwargs)
            self.assertNotIn('boundary_ratio', kwargs)
        else:
            self.assertIs(kwargs['transformer_2'], transformer)
            self.assertEqual(kwargs['boundary_ratio'], 0.875)

    def test_wan21_13b_single_transformer(self):
        self.check_layout('wan21', '/models/Wan2.1-1.3B', [Path('/models/Wan2.1-1.3B')])

    def test_wan21_14b_single_transformer(self):
        self.check_layout('wan21', '/models/Wan2.1-14B', [Path('/models/Wan2.1-14B')])

    def test_wan22_keeps_dual_experts(self):
        base = Path('/models/Wan2.2-A14B')
        self.check_layout('wan22', str(base), [base / 'high_noise_model', base / 'low_noise_model'])


if __name__ == '__main__':
    unittest.main()
