from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from .int_quant_linear import (
    QuantSpec,
    WholeDiTQuantLinear,
    build_quant_cache_context,
    load_finalized_quant_cache,
    save_finalized_quant_cache,
)


class CacheContractTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {
            "PULSE_ALLOW_LEGACY_CACHE": "1",
            "PULSE_LEGACY_WEIGHT_BITS": "4",
            "PULSE_LEGACY_ACTIVATION_BITS": "4",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def make_module(self, risk_lambda: float = 0.75) -> WholeDiTQuantLinear:
        module = WholeDiTQuantLinear(
            nn.Linear(8, 8, bias=True),
            QuantSpec(
                method="minmax",
                weight_bits=4,
                activation_bits=4,
                risk_lambda=risk_lambda,
            ),
            "blocks.0.attn.to_q",
        )
        module.finalized = True
        return module

    def test_contract_accepts_exact_match_and_rejects_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text('{"hidden_size":8}\n')
            profile = root / "profile.json"
            profile.write_text('{"version":1,"weights":[1.0]}\n')
            prompts = root / "prompts.json"
            prompts.write_text('["test"]\n')
            context = build_quant_cache_context(
                model,
                propagation_profile=profile,
                calibration_prompts=prompts,
                extra={"axis_mode": "activation_pca"},
            )
            cache = root / "cache.pt"
            save_finalized_quant_cache(
                [self.make_module()], cache, cache_context=context
            )
            loaded = load_finalized_quant_cache(
                [self.make_module()], cache, cache_context=context
            )
            self.assertEqual(loaded["format"], "pulse-finalized-v2")

            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                load_finalized_quant_cache(
                    [self.make_module(risk_lambda=0.5)],
                    cache,
                    cache_context=context,
                )

            profile.write_text('{"version":2,"weights":[2.0]}\n')
            changed_context = build_quant_cache_context(
                model,
                propagation_profile=profile,
                calibration_prompts=prompts,
                extra={"axis_mode": "activation_pca"},
            )
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                load_finalized_quant_cache(
                    [self.make_module()], cache, cache_context=changed_context
                )

    def test_external_assets_cache_accepts_bundled_configs_by_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text('{}')
            external = root / "external-diffusers"
            bundled = root / "bundled-configs"
            for assets in (external, bundled):
                for component in ("text_encoder", "vae"):
                    target = assets / component / "config.json"
                    target.parent.mkdir(parents=True)
                    target.write_text('{"width":8}')
            # The historical external assets directory may contain a complete
            # Diffusers checkpoint, although the loader consumes only two configs.
            (external / "model_index.json").write_text('{"pipeline":"Wan"}')
            (external / "unused.safetensors").write_bytes(b"unused weight shard")

            def context(assets):
                return build_quant_cache_context(model, extra={
                    "runner": "wan21",
                    "diffusers_assets": build_quant_cache_context(assets)["model"],
                })

            cache = root / "cache.pt"
            source = self.make_module()
            save_finalized_quant_cache([source], cache, cache_context=context(external))
            restored = self.make_module()
            loaded = load_finalized_quant_cache([restored], cache, cache_context=context(bundled))
            self.assertEqual(loaded["format"], "pulse-finalized-v2")
            torch.testing.assert_close(restored.weight, source.weight)
            for component in ("text_encoder", "vae"):
                target = bundled / component / "config.json"
                target.write_text('{"width":9}')
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    load_finalized_quant_cache([self.make_module()], cache,
                                               cache_context=context(bundled))
                target.write_text('{"width":8}')

            (model / "config.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                load_finalized_quant_cache([self.make_module()], cache,
                                           cache_context=context(bundled))

    def test_assets_path_tampering_still_requires_valid_stored_contract_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.pt"
            context = {"model": "unit-test", "extra": {"diffusers_assets": {
                "kind": "directory", "identifier": "/old",
                "configuration": [
                    {"path": f"{component}/config.json", "bytes": 2, "sha256": "test"}
                    for component in ("text_encoder", "vae")
                ],
            }}}
            save_finalized_quant_cache([self.make_module()], cache, cache_context=context)
            payload = torch.load(cache, weights_only=False)
            payload["cache_contract"]["context"]["extra"]["diffusers_assets"]["identifier"] = "/tampered"
            torch.save(payload, cache)
            with self.assertRaisesRegex(ValueError, "contract payload is corrupted"):
                load_finalized_quant_cache([self.make_module()], cache, cache_context=context)

    def test_v1_cache_loads_with_best_effort_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "legacy.pt"
            module = self.make_module()
            context = {"model": "unit-test"}
            save_finalized_quant_cache([module], cache, cache_context=context)
            payload = torch.load(cache, map_location="cpu", weights_only=False)
            payload["format"] = "pulse-finalized-v1"
            payload.pop("cache_contract", None)
            payload.pop("cache_contract_sha256", None)
            torch.save(payload, cache)

            restored = self.make_module()
            with self.assertWarnsRegex(RuntimeWarning, "best-effort"):
                loaded = load_finalized_quant_cache(
                    [restored], cache, cache_context=context
                )
            torch.testing.assert_close(restored.weight, module.weight)
            torch.testing.assert_close(restored.bias, module.bias)
            x = torch.randn(2, 8)
            torch.testing.assert_close(restored(x), module(x))
            self.assertEqual(loaded["format"], "pulse-finalized-v1")
            self.assertEqual(loaded["contract_validation"], "legacy_best_effort")
            self.assertIn("propagation_profile", loaded["legacy_unverified_fields"])

            with patch.dict("os.environ"):
                import os
                os.environ.pop("PULSE_ALLOW_LEGACY_CACHE", None)
                with self.assertWarnsRegex(RuntimeWarning, "best-effort"):
                    default_loaded = load_finalized_quant_cache(
                        [self.make_module()], cache, cache_context=context
                    )
                self.assertEqual(default_loaded["contract_validation"], "legacy_best_effort")

            with patch.dict(
                "os.environ", {"PULSE_ALLOW_LEGACY_CACHE": "0"}
            ):
                with self.assertRaisesRegex(ValueError, "disabled"):
                    load_finalized_quant_cache(
                        [self.make_module()], cache, cache_context=context
                    )

    def test_correction_environment_is_part_of_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.pt"
            context = {"model": "unit-test"}
            with patch.dict(
                "os.environ", {"PULSE_WALSH2_PCA_ITERS": "2"}
            ):
                save_finalized_quant_cache(
                    [self.make_module()], cache, cache_context=context
                )
            with patch.dict(
                "os.environ", {"PULSE_WALSH2_PCA_ITERS": "3"}
            ):
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    load_finalized_quant_cache(
                        [self.make_module()], cache, cache_context=context
                    )

    def test_corrupt_buffers_rejected_before_any_module_is_changed(self):
        cases = {
            "missing_weight": lambda s: s.pop("weight"),
            "missing_bias": lambda s: s.pop("bias"),
            "missing_mask": lambda s: s.pop("channel_mask"),
            "none_weight": lambda s: s.update(weight=None),
            "weight_shape": lambda s: s.update(weight=torch.zeros(7, 8)),
            "bias_shape": lambda s: s.update(bias=torch.zeros(7)),
            "mask_shape": lambda s: s.update(channel_mask=torch.ones(7)),
            "schedule_shape": lambda s: s.update(activation_bit_schedule=torch.ones(9)),
            "packed_shape": lambda s: s.update(weight={"format": "rowwise_int8",
                "codes": torch.zeros(8, 8, dtype=torch.int8), "scale": torch.ones(8)}),
        }
        for legacy in (False, True):
            for name, corrupt in cases.items():
                with self.subTest(legacy=legacy, corruption=name), tempfile.TemporaryDirectory() as directory:
                    cache = Path(directory) / "cache.pt"
                    sources = [self.make_module(), self.make_module()]
                    sources[1].module_name = "blocks.1.attn.to_q"
                    context = {"model": "test"}
                    save_finalized_quant_cache(sources, cache, cache_context=context)
                    payload = torch.load(cache, weights_only=False)
                    if legacy:
                        payload["format"] = "pulse-finalized-v1"
                    corrupt(payload["modules"][sources[1].module_name])
                    torch.save(payload, cache)
                    targets = [self.make_module(), self.make_module()]
                    targets[1].module_name = sources[1].module_name
                    targets[0].finalized = False
                    before = targets[0].weight.clone()
                    calls = targets[0].calls
                    with self.assertRaises(ValueError):
                        load_finalized_quant_cache(targets, cache, cache_context=context)
                    torch.testing.assert_close(targets[0].weight, before)
                    self.assertFalse(targets[0].finalized)
                    self.assertEqual(targets[0].calls, calls)

    def test_legacy_controls_and_optional_buffers(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.pt"
            save_finalized_quant_cache([self.make_module()], cache, cache_context={})
            payload = torch.load(cache, weights_only=False)
            payload["format"] = "pulse-finalized-v1"
            state = next(iter(payload["modules"].values()))
            state.pop("activation_rescue_score")
            torch.save(payload, cache)
            with self.assertWarns(RuntimeWarning):
                load_finalized_quant_cache([self.make_module()], cache)
            for env, value in (("PULSE_ALLOW_LEGACY_CACHE", "0"),
                               ("PULSE_LEGACY_WEIGHT_BITS", None),
                               ("PULSE_LEGACY_ACTIVATION_BITS", None),
                               ("PULSE_LEGACY_WEIGHT_BITS", "6"),
                               ("PULSE_LEGACY_ACTIVATION_BITS", "6")):
                with self.subTest(env=env, value=value), patch.dict("os.environ"):
                    import os
                    if value is None:
                        os.environ.pop(env, None)
                    else:
                        os.environ[env] = value
                    with self.assertRaises(ValueError):
                        load_finalized_quant_cache([self.make_module()], cache)
            state["activation_bit_schedule"].fill_(6)
            torch.save(payload, cache)
            with self.assertRaisesRegex(ValueError, "precision mismatch"):
                load_finalized_quant_cache([self.make_module()], cache)

    def test_legacy_rowwise_int8(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.pt"
            source = self.make_module()
            save_finalized_quant_cache([source], cache, cache_context={})
            payload = torch.load(cache, weights_only=False)
            payload["format"] = "pulse-finalized-v1"
            codes = torch.randint(-8, 8, (8, 8), dtype=torch.int8)
            scale = torch.rand(8, 1)
            payload["modules"][source.module_name]["weight"] = {
                "format": "rowwise_int8", "codes": codes, "scale": scale}
            torch.save(payload, cache)
            target = self.make_module()
            with self.assertWarns(RuntimeWarning):
                load_finalized_quant_cache([target], cache)
            torch.testing.assert_close(target.weight, codes.float() * scale)


if __name__ == "__main__":
    unittest.main()
