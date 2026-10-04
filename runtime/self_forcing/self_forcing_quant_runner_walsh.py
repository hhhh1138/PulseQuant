#!/usr/bin/env python3
"""Four-step Self-Forcing AR benchmark with whole-backbone W4A6 quantization."""

from __future__ import annotations

import argparse
import json
import os
import time
import types
from pathlib import Path

import torch

from attention_adapters import build_adapter


class ForwardTimer:
    def __init__(self, module):
        self.starts = []
        self.ends = []
        original = module.forward

        def wrapped(_module, *args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            end.record()
            self.starts.append(start)
            self.ends.append(end)
            return result

        module.forward = types.MethodType(wrapped, module)

    def reset(self):
        self.starts.clear()
        self.ends.clear()

    def result(self):
        torch.cuda.synchronize()
        return sum(a.elapsed_time(b) for a, b in zip(self.starts, self.ends)) / 1000, len(self.starts)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--method",
        choices=('dense', 'fouranchor'),
        required=True,
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--prompt-offset", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--calibration-prompts", default="")
    parser.add_argument("--calibration-indices", default="0,120")
    parser.add_argument("--calibration-seed", type=int, default=991)
    parser.add_argument("--precision-profile", default="")
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--activation-bits", type=int, default=6)
    parser.add_argument("--a6-constants", default="")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--dense-self-calls", type=int, default=0,
                        help="Dense self-attention layer calls before sparse dispatch")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    repo = Path(args.self_repo).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    prompt_path = Path(args.prompts).resolve()
    output_dir = Path(args.output_dir).resolve()
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    if args.overwrite and metrics_path.exists():
        metrics_path.unlink()

    # The repository compiles a training-only FlexAttention path at import.
    # It is unused in cached AR inference and incompatible with Sol's isolated
    # Triton overlay, so disable that decoration consistently for all methods.
    def no_compile(function=None, *compile_args, **compile_kwargs):
        return (lambda target: target) if function is None else function

    torch.compile = no_compile
    import sys

    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from einops import rearrange
    from omegaconf import OmegaConf
    from torchvision.io import write_video
    from pipeline import CausalInferencePipeline
    from wan.modules import causal_model

    prompts = [line.strip() for line in prompt_path.read_text().splitlines() if line.strip()]
    selected = list(enumerate(prompts))[args.prompt_offset:args.prompt_offset + args.limit]
    completed = set()
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            if line.strip():
                completed.add(json.loads(line)["prompt_index"])

    original_attention = causal_model.attention
    adapter = build_adapter("dense", original_attention)
    causal_model.attention = adapter

    config = OmegaConf.merge(
        OmegaConf.load(repo / "configs/default_config.yaml"),
        OmegaConf.load(config_path),
    )
    pipeline = CausalInferencePipeline(config, device=torch.device("cuda"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_key = "generator" if "generator" in checkpoint else "generator_ema"
    pipeline.generator.load_state_dict(checkpoint[state_key])
    del checkpoint
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device="cuda")
    pipeline.generator.to(device="cuda")
    pipeline.vae.to(device="cuda")
    backbone = pipeline.generator.model
    fused_blocks = 0

    quant_modules = []
    quant_started = time.perf_counter()
    quant_metadata = {"method": args.method, "quantized_modules": 0}
    if args.method != "dense":
        from dataclasses import replace
        from spatial_single_step.int_quant_linear import QuantSpec, WholeDiTQuantLinear

        activation_codebook = "uniform".lower()

        canonical_targets = []
        for block_index, block in enumerate(backbone.blocks):
            canonical_targets.extend([
                (block.self_attn, "q", f"blocks.{block_index}.attn1.to_q"),
                (block.self_attn, "k", f"blocks.{block_index}.attn1.to_k"),
                (block.self_attn, "v", f"blocks.{block_index}.attn1.to_v"),
                (block.self_attn, "o", f"blocks.{block_index}.attn1.to_out.0"),
                (block.cross_attn, "q", f"blocks.{block_index}.attn2.to_q"),
                (block.cross_attn, "k", f"blocks.{block_index}.attn2.to_k"),
                (block.cross_attn, "v", f"blocks.{block_index}.attn2.to_v"),
                (block.cross_attn, "o", f"blocks.{block_index}.attn2.to_out.0"),
                (block.ffn, 0, f"blocks.{block_index}.ffn.net.0.proj"),
                (block.ffn, 2, f"blocks.{block_index}.ffn.net.2"),
            ])

        trajectory_coordinates = tuple(
            value for _ in range(7) for value in (1.0, 0.9375, 0.8333333, 0.625, 0.0)
        )
        calibration_indices = [int(x) for x in args.calibration_indices.split(",") if x.strip()]
        calibration_count = len(calibration_indices) if args.method in ('fouranchor',) else 1
        profile = None
        propagation = None
        absolute_risks = None
        precision_profile = None
        if args.method in ('fouranchor',):
            if not args.precision_profile:
                raise ValueError("--precision-profile is required for our methods")
            profile = json.loads(Path(args.precision_profile).read_text())
            precision_profile = {
                row["module"]: (args.weight_bits, args.activation_bits)
                for row in profile["modules"]
            }
            propagation = {
                row["module"]: tuple(float(v) for v in row["propagation_state_weights"])
                for row in profile["modules"] if row.get("propagation_state_weights")
            }
            # The source Wan profile contains 12 CFG states.  Self-Forcing has
            # 35 AR states (7 blocks x 5 calls), so transfer the normalized
            # risk curve by linear phase interpolation before calibration.
            propagation = {
                name: tuple(
                    torch.nn.functional.interpolate(
                        torch.tensor(values).reshape(1, 1, -1),
                        size=35,
                        mode="linear",
                        align_corners=True,
                    ).flatten().tolist()
                )
                for name, values in propagation.items()
            }
            absolute_risks = {
                row["module"]: float(row.get("absolute_propagation_risk", 1.0))
                for row in profile["modules"]
            }
        internal_method = "pulseprofile"
        calibration_calls = 35 * calibration_count
        spec = QuantSpec(
            method=internal_method,
            weight_bits=args.weight_bits,
            activation_bits=args.activation_bits,
            calibration_calls=calibration_calls,
            alpha=0.5665,
            hadamard_max_group=128,
            bgr_shrink_steps=6,
            bgr_refine_steps=2,
            risk_tail_fraction=0.25,
            samples_per_call=4,
            trajectory_trust_ratio=0.05,
            trajectory_segment_calls=35,
            inference_segment_calls=35,
            calibration_coordinates=trajectory_coordinates,
            inference_coordinates=trajectory_coordinates,
            precision_profile=precision_profile,
            propagation_state_weights=propagation,
            absolute_propagation_risks=absolute_risks,
        )
        expected_calls = {}
        for parent, key, name in canonical_targets:
            source = parent[key] if isinstance(key, int) else getattr(parent, key)
            is_cached_cross_kv = name.endswith("attn2.to_k") or name.endswith("attn2.to_v")
            expected = calibration_count if is_cached_cross_kv else calibration_calls
            local_spec = spec
            if calibration_calls:
                local_propagation = spec.propagation_state_weights
                if local_propagation is not None:
                    transferred = local_propagation[name]
                    local_propagation = {
                        name: (sum(transferred) / len(transferred),)
                        if is_cached_cross_kv else transferred
                    }
                local_spec = replace(
                    spec,
                    # Keep every module in BF16 collection mode until the
                    # complete calibration set has finished, then finalize all
                    # modules together using their true call counts.
                    calibration_calls=expected + 1,
                    precision_profile=(
                        None if spec.precision_profile is None
                        else {name: spec.precision_profile[name]}
                    ),
                    propagation_state_weights=local_propagation,
                    absolute_propagation_risks=(
                        None if spec.absolute_propagation_risks is None
                        else {name: spec.absolute_propagation_risks[name]}
                    ),
                )
            replacement = WholeDiTQuantLinear(source, local_spec, name)
            expected_calls[name] = expected
            if isinstance(key, int):
                parent[key] = replacement
            else:
                setattr(parent, key, replacement)
            quant_modules.append(replacement)
        quant_metadata.update({
            "implementation": "shared WholeDiTQuantLinear adapter",
            "internal_method": internal_method,
            "quantized_modules": len(quant_modules),
            "calibration_calls_per_module": calibration_calls,
            "trajectory": "7 AR blocks x [4 denoise anchors + 1 clean-cache refresh]",
        })

        if calibration_calls:
            calibration_path = Path(args.calibration_prompts or args.prompts)
            calibration_prompts = [x.strip() for x in calibration_path.read_text().splitlines() if x.strip()]
            use_indices = calibration_indices[:calibration_count]
            for calibration_number, calibration_index in enumerate(use_indices):
                torch.manual_seed(args.calibration_seed + calibration_number)
                calibration_noise = torch.randn((1, 21, 16, 60, 104), device="cuda", dtype=torch.bfloat16)
                with torch.inference_mode():
                    calibration_video, calibration_latents = pipeline.inference(
                        noise=calibration_noise,
                        text_prompts=[calibration_prompts[calibration_index]],
                        return_latents=True,
                        initial_latent=None,
                        low_memory=False,
                    )
                del calibration_video, calibration_latents
                pipeline.vae.model.clear_cache()
                torch.cuda.empty_cache()
            # Optional Self-Forcing-specific risk refinement.  The transferred
            # Wan diffusion curve is only a prior: actual AR calibration
            # activations estimate a per-state local Jacobian gain, while a
            # causal remaining-horizon factor protects errors that can enter
            # more future clean-KV blocks.  Everything is folded into the
            # offline candidate objective; deployment stays frozen W4A6.
            if os.environ.get("PULSE_SELF_AR_LOCAL_RISK", "0") == "1":
                beta = max(
                    0.0, float(os.environ.get("PULSE_SELF_AR_LOCAL_RISK_BETA", ".5"))
                )
                causal_power = max(
                    0.0, float(os.environ.get("PULSE_SELF_AR_CAUSAL_POWER", ".5"))
                )
                module_absolute_scores = []
                for module in quant_modules:
                    samples = torch.stack(module.calibration_samples).float()
                    outputs = torch.matmul(samples, module.weight.float().transpose(0, 1))
                    input_energy = samples.square().mean((1, 2)).clamp_min(1e-12)
                    output_energy = outputs.square().mean((1, 2)).clamp_min(1e-12)
                    gain = (output_energy / input_energy).sqrt()
                    gain = gain / gain.log().mean().exp().clamp_min(1e-12)
                    is_cached_cross_kv = (
                        module.module_name.endswith("attn2.to_k")
                        or module.module_name.endswith("attn2.to_v")
                    )
                    states_per_prompt = 1 if is_cached_cross_kv else 35
                    if gain.numel() % states_per_prompt == 0:
                        # Calibration samples are prompt-major.  Keep a single
                        # per-trajectory state curve and aggregate prompts with
                        # a geometric mean, so downstream guard code can repeat
                        # it for each prompt without double-counting prompts.
                        gain = gain.reshape(-1, states_per_prompt)
                        gain = gain.clamp_min(1e-12).log().mean(0).exp()
                    base = module.propagation_state_weight
                    if base is None:
                        base = torch.ones_like(gain)
                    else:
                        base = base.float().to(gain)
                        if base.numel() != gain.numel():
                            base = torch.nn.functional.interpolate(
                                base.reshape(1, 1, -1), size=gain.numel(),
                                mode="linear", align_corners=True,
                            ).flatten()
                    if gain.numel() >= 35 and gain.numel() % 35 == 0:
                        one_trajectory = torch.tensor(
                            [
                                float(7 - state // 5) ** causal_power
                                for state in range(35)
                            ],
                            device=gain.device,
                        )
                        causal = one_trajectory.repeat(gain.numel() // 35)
                        causal = causal / causal.mean().clamp_min(1e-12)
                    else:
                        # Cross-attention K/V is clean-cached and observed once
                        # per AR trajectory.  Its state curve is scalar, but its
                        # absolute score below still reflects measured gain.
                        causal = torch.ones_like(gain)
                    refined = base * gain.pow(beta) * causal
                    refined = refined / refined.mean().clamp_min(1e-12)
                    module.propagation_state_weight = refined
                    module_absolute_scores.append(
                        (output_energy.mean() / input_energy.mean()).sqrt()
                    )
                absolute = torch.stack(module_absolute_scores)
                absolute = absolute / absolute.mean().clamp_min(1e-12)
                for module, score in zip(quant_modules, absolute):
                    module.absolute_propagation_risk.copy_(
                        module.absolute_propagation_risk.float()
                        * score.to(module.absolute_propagation_risk)
                    )
                quant_metadata.update({
                    "self_ar_local_risk": True,
                    "self_ar_local_risk_beta": beta,
                    "self_ar_causal_power": causal_power,
                    "self_ar_risk_source": "measured local gain x causal remaining horizon",
                })
            for module in quant_modules:
                expected = expected_calls[module.module_name]
                if module.calls != expected:
                    raise RuntimeError(
                        f"unexpected calibration calls for {module.module_name}: "
                        f"observed={module.calls}, expected={expected}"
                    )
                module.spec = replace(module.spec, calibration_calls=expected)
                module._finalize()
    quant_metadata["setup_seconds"] = time.perf_counter() - quant_started
    quant_metadata["w_bits"] = args.weight_bits if args.method != "dense" else 16
    quant_metadata["a_bits"] = args.activation_bits if args.method != "dense" else 16
    (output_dir / "quantization.json").write_text(json.dumps(quant_metadata, indent=2) + "\n")
    timer = ForwardTimer(pipeline.generator)

    for prompt_index, prompt in selected:
        if prompt_index in completed:
            continue
        if hasattr(adapter, "reset_routes"):
            adapter.reset_routes()
        timer.reset()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(args.seed + prompt_index)
        noise = torch.randn(
            (1, 21, 16, 60, 104),
            device="cuda",
            dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            video, latents = pipeline.inference(
                noise=noise,
                text_prompts=[prompt],
                return_latents=True,
                initial_latent=None,
                low_memory=False,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        dit_seconds, dit_calls = timer.result()
        video_u8 = 255.0 * rearrange(video, "b t c h w -> b t h w c").cpu()
        video_path = video_dir / f"{prompt_index:02d}.mp4"
        write_video(str(video_path), video_u8[0], fps=16)
        pipeline.vae.model.clear_cache()
        row = {
            "prompt_index": prompt_index,
            "prompt": prompt,
            "seed": args.seed + prompt_index,
            "method": args.method,
            "generation_seconds": elapsed,
            "dit_forward_seconds": dit_seconds,
            "dit_forward_calls": dit_calls,
            "mean_dit_forward_ms": 1000 * dit_seconds / max(dit_calls, 1),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "adapter_stats": adapter.stats.as_dict(),
            "fused_glue_blocks": fused_blocks,
            "latent_mean_abs": float(latents.float().abs().mean()),
            "video": str(video_path),
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
