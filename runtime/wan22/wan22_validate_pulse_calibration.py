#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--cache", type=Path, required=True)
p.add_argument("--bits", type=int, required=True)
a = p.parse_args()

config = json.loads((a.output / "config.json").read_text())
audit = json.loads((a.output / "dit_quant_audit.json").read_text())
experts = json.loads((a.output / "dit_quant_expert_calibration.json").read_text())
saved = json.loads((a.output / "dit_quant_cache_save.json").read_text())
cache = torch.load(a.cache, map_location="cpu", weights_only=False)
rows = audit["modules"]
names = [row["module"] for row in rows]

assert config["dit_quant_method"] == "pulseprofile"
assert config["dit_weight_bits"] == 4
assert config["dit_activation_bits"] == a.bits
prompt_count = config["dit_quant_calibration_prompt_count"]
assert prompt_count > 0
assert len(config["dit_quant_calibration_prompt_indices"]) == prompt_count
assert set(experts) == {"high_noise", "low_noise"}, experts
assert experts["high_noise"]["calibration_step_indices"] == [0, 1, 2, 3]
assert experts["low_noise"]["calibration_step_indices"] == [4, 5]
assert experts["high_noise"]["required_calibration_calls"] == 2 * 4 * prompt_count
assert experts["low_noise"]["required_calibration_calls"] == 2 * 2 * prompt_count
assert len(rows) == 800, len(rows)
assert sum(name.startswith("high_noise.") for name in names) == 400
assert sum(name.startswith("low_noise.") for name in names) == 400
assert all(row["weight_bits"] == 4 for row in rows)
assert all(row["activation_bits"] == a.bits for row in rows)
assert cache["format"] == "pulse-finalized-v2"
assert set(cache["modules"]) == set(names)
assert saved["modules"] == 800

result = {
    "status": "verified",
    "model": config["model"],
    "weight_bits": 4,
    "activation_bits": a.bits,
    "module_count": len(rows),
    "high_noise_modules": 400,
    "low_noise_modules": 400,
    "calibration_prompt_count": prompt_count,
    "high_noise_calibration_calls": 2 * 4 * prompt_count,
    "low_noise_calibration_calls": 2 * 2 * prompt_count,
    "cache": str(a.cache),
    "cache_bytes": a.cache.stat().st_size,
}
(a.output / "calibration_verification.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
