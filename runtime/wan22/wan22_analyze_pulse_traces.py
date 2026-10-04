#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch


RUN_PATTERN = re.compile(r"^b(?P<block>\d+)_t(?P<step>\d+)$")


def relative_mse(value, reference):
    value, reference = value.float(), reference.float()
    return float((value-reference).square().mean()/reference.square().mean().clamp_min(1e-20))


def motion_relative_mse(value, reference):
    value_motion, reference_motion = value.float().diff(dim=2), reference.float().diff(dim=2)
    return float((value_motion-reference_motion).square().mean()/reference_motion.square().mean().clamp_min(1e-20))


def load_trace(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {int(step): tensor for step, tensor in payload["latents"].items()}


def local_scores(run_dir):
    rows = json.loads((run_dir/"dit_quant_audit.json").read_text())["modules"]
    weights = np.asarray([row["weight_numel"] for row in rows], dtype=np.float64)
    return tuple(float(np.average([row[key] for row in rows], weights=weights)) for key in (
        "weight_relative_rmse", "activation_relative_rmse", "output_relative_rmse"
    ))


p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--reference", type=Path, required=True)
p.add_argument("--steps", type=int, default=40)
p.add_argument("--horizon", type=int, default=4)
p.add_argument("--reference-prompt-offset", type=int, default=280)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
final_step = a.steps - 1
rows = []
for run_dir in sorted(a.root.iterdir()):
    match = RUN_PATTERN.match(run_dir.name)
    trace_path = run_dir/"latent_traces"/"00.pt"
    if match is None or not trace_path.exists() or not (run_dir/"dit_quant_audit.json").exists():
        continue
    block, injection = int(match.group("block")), int(match.group("step"))
    cfg = json.loads((run_dir/"config.json").read_text())
    reference_index = int(cfg["prompt_offset"]) - a.reference_prompt_offset
    reference = load_trace(a.reference/"latent_traces"/f"{reference_index:02d}.pt")
    trace = load_trace(trace_path)
    horizon = min(injection+a.horizon, final_step)
    if not {injection,horizon,final_step}.issubset(reference) or not {injection,horizon,final_step}.issubset(trace):
        continue
    weight_score, activation_score, output_score = local_scores(run_dir)
    immediate = relative_mse(trace[injection],reference[injection])
    propagated = relative_mse(trace[horizon],reference[horizon])
    final = relative_mse(trace[final_step],reference[final_step])
    immediate_motion = motion_relative_mse(trace[injection],reference[injection])
    propagated_motion = motion_relative_mse(trace[horizon],reference[horizon])
    final_motion = motion_relative_mse(trace[final_step],reference[final_step])
    gain = math.sqrt((propagated+1e-20)/(immediate+1e-20))
    rows.append({
        "run":run_dir.name,"block":block,"injection_step":injection,
        "source_prompt_index":int(cfg["prompt_offset"]),"horizon_step":horizon,
        "weight_relative_rmse":weight_score,"local_activation_relative_rmse":activation_score,
        "local_output_relative_rmse":output_score,"immediate_latent_relative_mse":immediate,
        "propagated_latent_relative_mse":propagated,"final_latent_relative_mse":final,
        "immediate_motion_relative_mse":immediate_motion,
        "propagated_motion_relative_mse":propagated_motion,"final_motion_relative_mse":final_motion,
        "finite_difference_jvp_gain":gain,"closed_loop_log_norm_gain":math.log(gain+1e-20),
        "expert":"high_noise" if injection < 26 else "low_noise",
    })
result={"schema_version":1,"protocol":{"intervention":"one W4A4 block at one denoising step; both CFG calls; later calls BF16","horizon":a.horizon,"target_step":final_step},"runs":rows}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"runs":len(rows),"high":sum(r["expert"]=="high_noise" for r in rows),"low":sum(r["expert"]=="low_noise" for r in rows)},indent=2))
