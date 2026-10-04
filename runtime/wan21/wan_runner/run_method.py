#!/usr/bin/env python3
"""PulseQuant calibration, cache loading and dense-attention video inference."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

from local_loader import bundled_assets_root, load_local_pipeline
from pulsequant_utils.seed_policy import resolve_prompt_seed
from pulsequant_utils.prompt_policy import load_nonempty_prompts, select_calibration_prompts


DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

SVG_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def parse_index_set(spec: str) -> frozenset[int]:
    """Parse comma-separated non-negative indices and inclusive ranges."""
    result: set[int] = set()
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, stop = (int(value) for value in item.split("-", 1))
        else:
            start = stop = int(item)
        if start < 0 or stop < start:
            raise ValueError(f"invalid non-negative index range: {item!r}")
        result.update(range(start, stop + 1))
    return frozenset(result)


class DiTForwardTimer:
    def __init__(
        self,
        transformer,
        profile_path=None,
        profile_call=3,
        *,
        cudagraph_mark_steps=False,
    ):
        self.original_forward = transformer.forward
        self.events = []
        self.total_calls = 0
        self.profile_path = Path(profile_path) if profile_path else None
        self.profile_call = int(profile_call)
        self.cudagraph_mark_steps = bool(cudagraph_mark_steps)

        def timed(*args, **kwargs):
            if self.cudagraph_mark_steps:
                # Regional compilation may capture repeated Wan blocks with
                # CUDA Graphs.  Each top-level transformer invocation is a new
                # graph step; without this boundary PyTorch can report that a
                # previous graph output was overwritten on the next denoising
                # call.
                torch.compiler.cudagraph_mark_step_begin()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.total_calls += 1
            if self.profile_path is not None and self.total_calls == self.profile_call:
                self.profile_path.parent.mkdir(parents=True, exist_ok=True)
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                ) as prof:
                    output = self.original_forward(*args, **kwargs)
                prof.export_chrome_trace(str(self.profile_path.with_suffix(".trace.json")))
                self.profile_path.write_text(
                    prof.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=120
                    ) + "\n"
                )
            else:
                output = self.original_forward(*args, **kwargs)
            end.record()
            self.events.append((start, end))
            return output

        transformer.forward = timed

    def reset(self):
        self.events.clear()

    def elapsed(self):
        torch.cuda.synchronize()
        return sum(a.elapsed_time(b) for a, b in self.events) / 1000.0, len(self.events)


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument(
        "--calibration-prompts",
        help="Optional disjoint prompt file used only for quantization warmup/calibration",
    )
    p.add_argument(
        "--prompt-offset",
        type=int,
        default=0,
        help="Skip this many non-empty prompts before applying --limit",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--method", choices=('dense',), required=True
    )
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--seed-mode", choices=("fixed", "per_prompt"), default="fixed")
    p.add_argument(
        "--seed-index-origin", choices=("local", "absolute"), default="local",
        help="For per_prompt seeds, use the shard-local index or prompt-offset + local index",
    )
    p.add_argument(
        "--warmup-prompt-offset", type=int, default=-1,
        help="Use a disjoint prompt for warmup/calibration; -1 reuses the first eval prompt",
    )
    p.add_argument(
        "--warmup-seed", type=int, default=-1,
        help="Optional disjoint warmup/calibration seed; -1 reuses --seed",
    )
    p.add_argument(
        "--dit-quant-calibration-prompt-count", type=int, default=1,
        help="Use consecutive disjoint warmup prompts for whole-DiT calibration",
    )
    p.add_argument(
        "--dit-quant-calibration-steps",
        type=int,
        default=0,
        help=(
            "Denoising steps per quantization calibration prompt; 0 reuses --steps. "
            "Target videos always use --steps."
        ),
    )
    p.add_argument(
        "--dit-quant-calibration-prompt-indices",
        default="",
        help=(
            "optional comma-separated prompt indices for calibration; when set, "
            "its length must equal --dit-quant-calibration-prompt-count"
        ),
    )
    p.add_argument("--negative-prompt-style", choices=("sol", "svg"), default="sol")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--flow-shift", type=float, default=3.0)
    p.add_argument("--dense-calls", type=int, default=10)
    p.add_argument("--warmup-passes", type=int, default=1)
    p.add_argument(
        "--calibration-only",
        action="store_true",
        help="Run quantization warmup, finalize/save the cache, and skip video generation",
    )
    p.add_argument(
        "--linear-input-audit",
        action="store_true",
        help="record current-forward max magnitudes at Wan-1.3B projection/FFN linears",
    )
    p.add_argument(
        "--dit-quant-method",
        choices=('none', 'pulseprofile'),
        default="none",
        help=(
            "algorithmic whole-backbone integer Q/DQ over Wan attention and FFN "
            "linears; quality simulation, not a packed-kernel latency claim"
        ),
    )
    p.add_argument("--dit-weight-bits", type=int, default=4)
    p.add_argument("--dit-activation-bits", type=int, default=8)
    p.add_argument(
        "--dit-quant-scope",
        choices=("all", "self_attn", "cross_attn", "ffn"),
        default="all",
    )
    p.add_argument(
        "--dit-quant-block-indices",
        default="",
        help=(
            "optional comma-separated block indices/ranges to quantize; empty "
            "selects every block"
        ),
    )
    p.add_argument(
        "--dit-quant-pulse-step-indices",
        default="",
        help=(
            "causal intervention mode: after calibration, enable quantization only "
            "for both CFG transformer calls of these zero-based denoising steps"
        ),
    )
    p.add_argument(
        "--latent-trace-steps",
        default="",
        help=(
            "save callback latents for comma-separated zero-based denoising steps; "
            "use 'all' to save the complete trajectory"
        ),
    )
    p.add_argument("--dit-quant-alpha", type=float, default=0.5665)
    p.add_argument("--dit-quant-hadamard-group", type=int, default=128)
    p.add_argument("--dit-quant-bgr-shrink-steps", type=int, default=12)
    p.add_argument("--dit-quant-bgr-refine-steps", type=int, default=3)
    p.add_argument("--dit-quant-bgr-max-shrink-fraction", type=float, default=0.44)
    p.add_argument("--dit-quant-bgr-convergence-epsilon", type=float, default=0.0)
    p.add_argument("--dit-quant-arq-block-size", type=int, default=1)
    p.add_argument(
        "--dit-quant-risk-lambda",
        type=float,
        default=0.75,
        help="TWGR blend of mean and worst-timestep rotated activation energy",
    )
    p.add_argument(
        "--dit-quant-risk-tail-fraction",
        type=float,
        default=0.0,
        help=(
            "CVaR tail fraction over calibration states; 0 uses the single "
            "worst state"
        ),
    )
    p.add_argument(
        "--dit-quant-samples-per-call",
        type=int,
        default=2,
        help="tiny token sketch size per denoising call for TAOR/TAJOR",
    )
    p.add_argument(
        "--dit-quant-covariance-shrinkage",
        type=float,
        default=0.0,
        help="TAOR-SH blend toward an isotropic covariance prior",
    )
    p.add_argument(
        "--dit-quant-trajectory-delta-weight",
        type=float,
        default=1.0,
        help="TAOR-DELTA weight on same-CFG denoising-increment reconstruction",
    )
    p.add_argument(
        "--dit-quant-trajectory-trust-ratio",
        type=float,
        default=0.05,
        help="TIDE-TR maximum relative state-risk increase before delta-risk ranking",
    )
    p.add_argument(
        "--dit-quant-activation-scale-momentum",
        type=float,
        default=0.0,
        help="optional same-CFG activation-range momentum for frozen clipping",
    )
    p.add_argument(
        "--dit-quant-activation-fixed-clipping-ratio",
        type=float,
        default=0.0,
        help="override a frozen clipping schedule with one fixed token ratio",
    )
    p.add_argument("--dit-quant-activation-mse-min-ratio", type=float, default=0.75)
    p.add_argument("--dit-quant-activation-mse-candidates", type=int, default=5)
    p.add_argument("--dit-quant-activation-error-audit", action="store_true")
    p.add_argument(
        "--dit-quant-state-audit",
        action="store_true",
        help=(
            "save compact per-layer channel masks and deterministic weight samples; "
            "used to measure calibration-prompt stability without dumping the model"
        ),
    )
    p.add_argument(
        "--dit-quant-activation-high-bits", type=int, default=0,
        help="Optional higher activation precision for a functional module scope.",
    )
    p.add_argument(
        "--dit-quant-activation-high-scope",
        choices=(
            "none", "cross_attn", "cross_out", "self_attn", "attention",
            "ffn", "ffn_out", "output_projections",
        ),
        default="none",
    )
    p.add_argument(
        "--dit-quant-bit-sensitivity-threshold",
        type=float,
        default=1.5,
        help="TAOR-AB keeps W4 when local W3/W4 functional-risk ratio exceeds this value",
    )
    p.add_argument(
        "--dit-quant-w4-module-list",
        default="",
        help="TAOR-Profile JSON audit or newline list containing modules retained at W4",
    )
    p.add_argument(
        "--dit-quant-precision-profile",
        default="",
        help="allocator JSON with per-module weight_bits and activation_bits for TIDE-Profile",
    )
    p.add_argument("--profile-one-dit", default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--regional-compile", action="store_true")
    p.add_argument("--regional-compile-mode", default="default")
    p.add_argument(
        "--temporal-mode",
        choices=('none',),
        default="none",
    )
    return p.parse_args()


def main():
    args = args_parser()
    out = Path(args.output_dir)
    videos = out / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    metrics = out / "metrics.jsonl"
    if args.overwrite and metrics.exists():
        metrics.unlink()
    (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=args.flow_shift,
    )
    scheduler.set_timesteps(args.steps)
    temporal_coordinates = tuple(float(x) for x in scheduler.sigmas[: args.steps])
    pipe = load_local_pipeline(args.model)
    pipe.scheduler = scheduler
    pipe.to("cuda")

    # Load a small, explicitly audited parameter overlay after the base model
    # reaches its execution device.  This avoids materializing a duplicate
    # 5.3 GiB Wan1.3B checkpoint and, critically, avoids rewriting all six
    # Wan14B safetensor shards for 160 tiny Q/K RMSNorm gain tensors.

    if args.regional_compile:
        pipe.transformer.compile_repeated_blocks(
            fullgraph=False, dynamic=False, mode=args.regional_compile_mode
        )
    int_quant_modules = []
    quant_cache_loaded = False
    quant_cache_context = None
    if args.dit_quant_method != "none":
        from spatial_single_step.int_quant_linear import (
            QuantSpec,
            apply_residual_a8_schedule,
            build_quant_cache_context,
            install_wan_whole_dit_quant,
            load_finalized_quant_cache,
            save_finalized_quant_cache,
        )
        import spatial_single_step.int_quant_linear as loaded_int_quant_linear
        (out / "activation_codebook_runtime.json").write_text(
            json.dumps(
                {
                    "requested": "uniform",
                    "module_file": str(loaded_int_quant_linear.__file__),
                    "module_constant": getattr(
                        loaded_int_quant_linear, "_PULSE_ACTIVATION_CODEBOOK", None
                    ),
                    "token_qdq_module": loaded_int_quant_linear.token_symmetric_qdq.__module__,
                },
                indent=2,
            ) + "\n"
        )

        # Optional calibration-free activation-codebook experiments.  The
        # Hadamard/RPBH inputs are close to Gaussian, so a fixed scalar
        # Lloyd-Max codebook is a natural alternative to uniform A6.  Patch
        # the module global used by IntQuantLinear before wrappers are built;
        # weights, Pareto radii/profile, and the online per-token statistic are
        # otherwise unchanged.

        if args.dit_weight_bits not in (2, 3, 4, 5, 6, 8):
            raise ValueError("unsupported --dit-weight-bits")
        if args.dit_activation_bits not in (3, 4, 6, 8):
            raise ValueError("unsupported --dit-activation-bits")
        if args.dit_quant_activation_high_bits not in (0, 4, 5, 6, 8):
            raise ValueError("unsupported --dit-quant-activation-high-bits")
        w4_module_names = frozenset()
        precision_profile = None
        propagation_state_weights = None
        absolute_propagation_risks = None
        if args.dit_quant_precision_profile:
            precision_payload = json.loads(Path(args.dit_quant_precision_profile).read_text())
            if precision_payload.get("uses_evaluation_scores") is not False:
                raise ValueError("precision profile must explicitly declare uses_evaluation_scores=false")
            # Reuse the frozen module selection and propagation weights while
            # enforcing the experiment-wide W4A4 contract.  The historical
            # profile was authored for W4A6 and must not silently restore A6.
            precision_profile = {
                row["module"]: (args.dit_weight_bits, args.dit_activation_bits)
                for row in precision_payload["modules"]
            }
            propagation_state_weights = {
                row["module"]: tuple(
                    float(value) for value in row["propagation_state_weights"]
                )
                for row in precision_payload["modules"]
                if row.get("propagation_state_weights")
            }
            absolute_propagation_risks = {
                row["module"]: float(row.get("absolute_propagation_risk", 1.0))
                for row in precision_payload["modules"]
            }
            if args.dit_quant_method == "pulseprofile" and not propagation_state_weights:
                raise ValueError(
                    "pulseprofile requires per-module propagation_state_weights"
                )
        if args.dit_quant_w4_module_list:
            profile_path = Path(args.dit_quant_w4_module_list)
            if profile_path.suffix == ".json":
                profile_payload = json.loads(profile_path.read_text())
                w4_module_names = frozenset(
                    row["module"] for row in profile_payload["modules"]
                    if int(row["weight_bits"]) == 4
                )
            else:
                w4_module_names = frozenset(
                    line.strip() for line in profile_path.read_text().splitlines() if line.strip()
                )
        calibration_steps = args.dit_quant_calibration_steps or args.steps
        if calibration_steps <= 0:
            raise ValueError("--dit-quant-calibration-steps must be positive or zero")
        calibration_scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=args.flow_shift,
        )
        calibration_scheduler.set_timesteps(calibration_steps)
        calibration_coordinates = tuple(
            float(x) for x in calibration_scheduler.sigmas[:calibration_steps]
        )
        quant_block_indices = parse_index_set(args.dit_quant_block_indices)
        pulse_step_indices = parse_index_set(args.dit_quant_pulse_step_indices)
        if pulse_step_indices and not quant_block_indices:
            raise ValueError(
                "--dit-quant-pulse-step-indices requires explicit "
                "--dit-quant-block-indices to bound preserved BF16 weights"
            )
        if any(step >= args.steps for step in pulse_step_indices):
            raise ValueError(
                "pulse step indices must be smaller than --steps="
                f"{args.steps}: {sorted(pulse_step_indices)}"
            )
        spec = QuantSpec(
            method=args.dit_quant_method,
            weight_bits=args.dit_weight_bits,
            activation_bits=args.dit_activation_bits,
            calibration_calls=(
                2 * calibration_steps * args.dit_quant_calibration_prompt_count
                if args.dit_quant_method in ('pulseprofile',)
                else 0
            ),
            alpha=args.dit_quant_alpha,
            hadamard_max_group=args.dit_quant_hadamard_group,
            bgr_shrink_steps=args.dit_quant_bgr_shrink_steps,
            bgr_refine_steps=args.dit_quant_bgr_refine_steps,
            bgr_max_shrink_fraction=args.dit_quant_bgr_max_shrink_fraction,
            bgr_convergence_epsilon=args.dit_quant_bgr_convergence_epsilon,
            arq_block_size=args.dit_quant_arq_block_size,
            risk_lambda=args.dit_quant_risk_lambda,
            risk_tail_fraction=args.dit_quant_risk_tail_fraction,
            samples_per_call=args.dit_quant_samples_per_call,
            covariance_shrinkage=args.dit_quant_covariance_shrinkage,
            trajectory_delta_weight=args.dit_quant_trajectory_delta_weight,
            trajectory_trust_ratio=args.dit_quant_trajectory_trust_ratio,
            trajectory_segment_calls=2 * calibration_steps,
            inference_segment_calls=2 * args.steps,
            calibration_coordinates=calibration_coordinates,
            inference_coordinates=temporal_coordinates,
            activation_scale_momentum=args.dit_quant_activation_scale_momentum,
            activation_fixed_clipping_ratio=(
                args.dit_quant_activation_fixed_clipping_ratio
            ),
            activation_mse_min_ratio=args.dit_quant_activation_mse_min_ratio,
            activation_mse_candidate_count=args.dit_quant_activation_mse_candidates,
            activation_error_audit=args.dit_quant_activation_error_audit,
            activation_high_bits=args.dit_quant_activation_high_bits,
            activation_high_scope=args.dit_quant_activation_high_scope,
            bit_sensitivity_threshold=args.dit_quant_bit_sensitivity_threshold,
            w4_module_names=w4_module_names,
            precision_profile=precision_profile,
            propagation_state_weights=propagation_state_weights,
            absolute_propagation_risks=absolute_propagation_risks,
            pulse_call_indices=frozenset(
                call
                for step in pulse_step_indices
                for call in (2 * step, 2 * step + 1)
            ),
        )
        int_quant_modules = install_wan_whole_dit_quant(
            pipe.transformer,
            spec,
            include=args.dit_quant_scope,
            block_indices=(quant_block_indices or None),
        )
        diffusers_assets_context = build_quant_cache_context(
            bundled_assets_root()
        )["model"]
        quant_cache_context = build_quant_cache_context(
            args.model,
            propagation_profile=(args.dit_quant_precision_profile or None),
            calibration_prompts=(args.calibration_prompts or args.prompts),
            extra={
                "runner": "wan21",
                "diffusers_assets": diffusers_assets_context,
                "quant_cli": {
                    key: value
                    for key, value in vars(args).items()
                    if key.startswith("dit_quant_")
                },
                "protocol": {
                    "height": args.height,
                    "width": args.width,
                    "frames": args.frames,
                    "steps": args.steps,
                    "flow_shift": args.flow_shift,
                    "guidance_scale": args.guidance_scale,
                    "negative_prompt_style": args.negative_prompt_style,
                },
            },
        )
        cache_load_path = os.environ.get("PULSE_QUANT_CACHE_LOAD", "")
        if cache_load_path:
            cache_metadata = load_finalized_quant_cache(
                int_quant_modules,
                cache_load_path,
                cache_context=quant_cache_context,
            )
            (out / "dit_quant_cache_load.json").write_text(
                json.dumps(cache_metadata, indent=2) + "\n"
            )
            quant_cache_loaded = True

        # Trajectory-closed radial refit (TCR): perturb only the already finalized
        # dequantized W4 radii.  Codes, rotations, A6, and the precision profile
        # remain unchanged, and the scalar is folded into the cached weights so
        # evaluation has no extra runtime operator.
        tcr_global_scale = float("1")
        tcr_family_scales = {
            "qkv": float(tcr_global_scale),
            "attn_out": float(tcr_global_scale),
            "ffn": float(tcr_global_scale),
        }
        if any(value <= 0 for value in tcr_family_scales.values()):
            raise ValueError("all PULSE_TCR_*_SCALE values must be positive")
        tcr_family_counts = {key: 0 for key in tcr_family_scales}
        with torch.no_grad():
            for module in int_quant_modules:
                name = module.module_name
                if ".ffn." in name:
                    family = "ffn"
                elif ".to_out." in name:
                    family = "attn_out"
                elif name.endswith((".to_q", ".to_k", ".to_v")):
                    family = "qkv"
                else:
                    raise ValueError(f"unclassified TCR linear: {name}")
                module.weight.mul_(tcr_family_scales[family])
                tcr_family_counts[family] += 1
        (out / "dit_quant_tcr_scale.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "global_scale": tcr_global_scale,
                    "family_scales": tcr_family_scales,
                    "family_counts": tcr_family_counts,
                    "linear_count": len(int_quant_modules),
                    "folded_into_weight": True,
                },
                indent=2,
            )
            + "\n"
        )

    linear_input_audit = {}
    linear_input_hooks = []
    if args.linear_input_audit:
        target_shapes = {(1536, 1536), (1536, 8960), (8960, 1536)}

        def make_linear_audit_hook(module_name):
            def hook(module, inputs):
                value = inputs[0].detach()
                linear_input_audit[module_name]["maxima"].append(value.abs().amax())
            return hook

        for module_name, module in pipe.transformer.named_modules():
            if isinstance(module, torch.nn.Linear) and (
                module.in_features, module.out_features
            ) in target_shapes:
                linear_input_audit[module_name] = {
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "maxima": [],
                }
                linear_input_hooks.append(
                    module.register_forward_pre_hook(make_linear_audit_hook(module_name))
                )


    timer = DiTForwardTimer(
        pipe.transformer,
        args.profile_one_dit,
        cudagraph_mark_steps=False,
    )
    all_prompts = load_nonempty_prompts(args.prompts)
    calibration_source_prompts = (
        load_nonempty_prompts(args.calibration_prompts)
        if args.calibration_prompts else all_prompts
    )
    prompts = all_prompts[args.prompt_offset : args.prompt_offset + args.limit]
    if not prompts and not args.calibration_only:
        raise ValueError(
            f"No prompts selected: offset={args.prompt_offset}, limit={args.limit}, "
            f"available={len(all_prompts)}"
        )

    def generate(
        prompt,
        seed,
        *,
        decode=True,
        inference_steps=None,
        latent_trace_path: Path | None = None,
    ):
        timer.reset()
        torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        trace_payload = None
        callback = None
        if latent_trace_path is not None:
            trace_all = args.latent_trace_steps.strip().lower() == "all"
            trace_steps = (
                frozenset()
                if trace_all
                else parse_index_set(args.latent_trace_steps)
            )
            trace_payload = {
                "format": "wan-latent-trajectory-v1",
                "seed": int(seed),
                "requested_steps": "all" if trace_all else sorted(trace_steps),
                "latents": {},
                "timesteps": {},
            }

            def callback(_pipe, step_index, timestep, callback_kwargs):
                if trace_all or step_index in trace_steps:
                    trace_payload["latents"][int(step_index)] = (
                        callback_kwargs["latents"].detach().to("cpu", torch.float16)
                    )
                    trace_payload["timesteps"][int(step_index)] = float(
                        timestep.detach().float().cpu()
                        if torch.is_tensor(timestep)
                        else timestep
                    )
                return callback_kwargs

        torch.cuda.synchronize()
        start = time.perf_counter()
        pipeline_kwargs = dict(
            prompt=prompt,
            negative_prompt=(
                SVG_NEGATIVE
                if args.negative_prompt_style == "svg"
                else DEFAULT_NEGATIVE
            ),
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=(args.steps if inference_steps is None else inference_steps),
            generator=generator,
            output_type="np" if decode else "latent",
        )
        frames = pipe(**pipeline_kwargs).frames[0]
        if latent_trace_path is not None:
            latent_trace_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(trace_payload, latent_trace_path)
        dit_s, calls = timer.elapsed()
        return frames, time.perf_counter() - start, dit_s, calls

    warmup_seed = args.warmup_seed if args.warmup_seed >= 0 else args.seed
    calibration_steps = args.dit_quant_calibration_steps or args.steps
    if calibration_steps <= 0:
        raise ValueError("--dit-quant-calibration-steps must be positive or zero")
    calibration_prompt_count = (
        args.dit_quant_calibration_prompt_count
        if args.dit_quant_method != "none"
        else 1
    )
    if calibration_prompt_count < 1:
        raise ValueError("--dit-quant-calibration-prompt-count must be positive")
    calibration_prompt_indices = [
        int(item.strip())
        for item in args.dit_quant_calibration_prompt_indices.split(",")
        if item.strip()
    ]
    if calibration_prompt_indices:
        if len(calibration_prompt_indices) != calibration_prompt_count:
            raise ValueError(
                "--dit-quant-calibration-prompt-indices length must equal "
                "--dit-quant-calibration-prompt-count"
            )
        warmup_prompts = select_calibration_prompts(
            calibration_source_prompts, calibration_prompt_indices,
            calibration_prompt_count,
        )
    else:
        calibration_prompt_indices = [
            args.warmup_prompt_offset + index
            if args.warmup_prompt_offset >= 0
            else args.prompt_offset + min(index, len(prompts) - 1)
            for index in range(calibration_prompt_count)
        ]
        warmup_prompts = select_calibration_prompts(
            calibration_source_prompts, calibration_prompt_indices,
            calibration_prompt_count,
        )
    for warm in range(0 if quant_cache_loaded else args.warmup_passes):
        for calibration_index, warmup_prompt in enumerate(warmup_prompts):
            calibration_output, wall, dit, calls = generate(
                warmup_prompt,
                warmup_seed + calibration_index,
                decode=False,
                inference_steps=calibration_steps,
            )
            del calibration_output
            torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "warmup": warm,
                        "calibration_prompt_index": calibration_index,
                        "warmup_prompt_offset": calibration_prompt_indices[calibration_index],
                        "warmup_seed": warmup_seed + calibration_index,
                        "generation_seconds": wall,
                        "dit_forward_seconds": dit,
                        "calls": calls,
                        "calibration_steps": calibration_steps,
                    }
                ),
                flush=True,
            )
    if int_quant_modules:
        residual_a8_metadata = apply_residual_a8_schedule(
            int_quant_modules,
            float(os.environ.get("PULSE_RESIDUAL_A8_FRACTION", "0")),
        )
        (out / "dit_quant_residual_a8.json").write_text(
            json.dumps(residual_a8_metadata, indent=2) + "\n"
        )
    # For exact A16/selective-bypass execution, cache loading deliberately
    # keeps weights in the rotated basis until every offline correction (Walsh,
    # TCR, residual scheduling) is complete.  Fold only at this final boundary;
    # inference tensors produced here are never mutated afterwards.
    if os.environ.get("PULSE_DEFER_SELECTIVE_BYPASS_FOLD", "0") == "1":
        for module in int_quant_modules:
            module._fold_selective_bypass_weight()
    cache_save_path = os.environ.get("PULSE_QUANT_CACHE_SAVE", "")
    if cache_save_path and int_quant_modules and not quant_cache_loaded:
        cache_metadata = save_finalized_quant_cache(
            int_quant_modules,
            cache_save_path,
            cache_context=quant_cache_context,
        )
        (out / "dit_quant_cache_save.json").write_text(
            json.dumps(cache_metadata, indent=2) + "\n"
        )

    completed = set()
    old_rows = []
    if metrics.exists() and not args.overwrite:
        for line in metrics.read_text().splitlines():
            try:
                row = json.loads(line)
                old_rows.append(row)
                completed.add(int(row["prompt_index"]))
            except (ValueError, KeyError):
                pass

    for index, prompt in enumerate(prompts):
        video = videos / f"{index:02d}.mp4"
        if index in completed and video.exists() and not args.overwrite:
            continue
        # The released runner uses one fixed seed; strict cross-method audits use
        # the SVG protocol, seed+i, so every method sees identical latent noise.
        prompt_seed = resolve_prompt_seed(
            args.seed, args.seed_mode, args.seed_index_origin, args.prompt_offset, index
        )
        trace_only = False
        frames, wall, dit, calls = generate(
            prompt,
            prompt_seed,
            decode=not trace_only,
            latent_trace_path=(
                out / "latent_traces" / f"{index:02d}.pt"
                if args.latent_trace_steps
                else None
            ),
        )
        export_to_video(frames, str(video), fps=args.fps)
        row = {
            "prompt_index": index,
            "source_prompt_index": args.prompt_offset + index,
            "prompt": prompt,
            "seed": prompt_seed,
            "seed_mode": args.seed_mode,
            "seed_index_origin": args.seed_index_origin,
            "calibration_prompt_source": str(args.calibration_prompts or args.prompts),
            "method": args.method,
            "generation_seconds": wall,
            "dit_forward_seconds": dit,
            "dit_forward_calls": calls,
            "mean_dit_forward_ms": dit * 1000 / max(calls, 1),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "video": str(video),
            "file": str(video.resolve()),
            "status": "complete",
            "exact_cross_kv_cache": False,
            "dit_quant_method": args.dit_quant_method,
            "dit_weight_bits": args.dit_weight_bits,
            "dit_activation_bits": args.dit_activation_bits,
            "dit_quant_scope": args.dit_quant_scope,
            "dit_quant_alpha": args.dit_quant_alpha,
            "dit_quant_hadamard_group": args.dit_quant_hadamard_group,
            "dit_quant_bgr_shrink_steps": args.dit_quant_bgr_shrink_steps,
            "dit_quant_bgr_refine_steps": args.dit_quant_bgr_refine_steps,
            "dit_quant_bgr_max_shrink_fraction": args.dit_quant_bgr_max_shrink_fraction,
            "dit_quant_bgr_convergence_epsilon": args.dit_quant_bgr_convergence_epsilon,
            "dit_quant_arq_block_size": args.dit_quant_arq_block_size,
            "dit_quant_risk_lambda": args.dit_quant_risk_lambda,
            "dit_quant_risk_tail_fraction": args.dit_quant_risk_tail_fraction,
            "dit_quant_samples_per_call": args.dit_quant_samples_per_call,
            "dit_quant_covariance_shrinkage": args.dit_quant_covariance_shrinkage,
            "dit_quant_trajectory_delta_weight": args.dit_quant_trajectory_delta_weight,
            "dit_quant_trajectory_trust_ratio": args.dit_quant_trajectory_trust_ratio,
            "dit_quant_activation_scale_momentum": args.dit_quant_activation_scale_momentum,
            "dit_quant_activation_fixed_clipping_ratio": args.dit_quant_activation_fixed_clipping_ratio,
            "dit_quant_activation_mse_min_ratio": args.dit_quant_activation_mse_min_ratio,
            "dit_quant_activation_mse_candidates": args.dit_quant_activation_mse_candidates,
            "dit_quant_activation_error_audit": args.dit_quant_activation_error_audit,
            "dit_quant_trajectory_segment_calls": 2 * calibration_steps,
            "dit_quant_inference_segment_calls": 2 * args.steps,
            "dit_quant_calibration_prompt_count": args.dit_quant_calibration_prompt_count,
            "dit_quant_calibration_prompt_indices": calibration_prompt_indices,
            "dit_quant_linears": len(int_quant_modules),
            "dit_quant_calls": sum(module.calls for module in int_quant_modules),
            "dit_quant_finalized": all(module.finalized for module in int_quant_modules),
            "dit_quant_deploy_fp8": False,
            "regional_compile": args.regional_compile,
            "regional_compile_mode": args.regional_compile_mode,
            "fuse_qkv_projections": False,
            "bf16_block_glue": 'none',
            "bf16_output_glue": False,
            "fused_fp32_glue": False,
            "fused_fp32_glue_part": 'all',
            "fused_fp32_inplace_residual": False,
            "fused_fp32_dual_stage": False,
            "fused_qk_rope": False,
            "fused_qk_rope_inplace": False,
            "fused_qk_rms_rope": False,
            "kv_splits": 'auto',
            "ffn_input_corrector_scale": 0.0,
            "ffn_sketch_tokens": 0,
            "cfg_self_share": False,
            "cfg_diff_beta": -1.0,
            "cfg_precompute_slopes": False,
            
        }
        with metrics.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        old_rows = [item for item in old_rows if int(item["prompt_index"]) != index] + [
            row
        ]
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "idea": out.name,
                    "method": args.method,
                    "videos": sorted(old_rows, key=lambda x: x["prompt_index"]),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if int_quant_modules:
        quant_audit = []
        for module in int_quant_modules:
            signal = float(module.weight_signal_sq.float().cpu())
            error = float(module.weight_error_sq.float().cpu())
            quant_audit.append(
                {
                    "module": module.module_name,
                    "block": module.block_index,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "weight_bits": module.spec.weight_bits,
                    "activation_bits": module.spec.activation_bits,
                    "activation_granularity": (
                        "channel"
                        if int(module.activation_granularity.item()) == 1
                        else "token"
                    ),
                    "activation_bulk_ratio": float(
                        module.activation_bulk_ratio.float().cpu()
                    ),
                    "activation_bulk_schedule": [
                        float(v) for v in module.activation_bulk_schedule.float().cpu()
                    ],
                    "activation_bit_schedule": [
                        int(v) for v in module.activation_bit_schedule.cpu()
                    ],
                    "activation_rescue_score": [
                        float(v) for v in module.activation_rescue_score.float().cpu()
                    ],
                    "activation_a8_ratio_schedule": [
                        float(v) for v in module.activation_a8_ratio_schedule.float().cpu()
                    ],
                    "activation_scale_momentum": module.spec.activation_scale_momentum,
                    "activation_fixed_clipping_ratio": (
                        module.spec.activation_fixed_clipping_ratio
                    ),
                    "hadamard_group": module.group,
                    "calls": module.calls,
                    "weight_relative_rmse": (error / max(signal, 1e-30)) ** 0.5,
                    "activation_relative_rmse": (
                        float(module.activation_error_sq.float().cpu())
                        / max(float(module.activation_signal_sq.float().cpu()), 1e-30)
                    ) ** 0.5,
                    "output_relative_rmse": (
                        float(module.output_error_sq.float().cpu())
                        / max(float(module.output_signal_sq.float().cpu()), 1e-30)
                    ) ** 0.5,
                    "importance_cv": float(module.importance_cv.float().cpu()),
                    "importance_p99_over_median": float(
                        module.importance_tail_ratio.float().cpu()
                    ),
                    "bit_sensitivity_ratio": float(
                        module.bit_sensitivity_ratio.float().cpu()
                    ),
                    "trajectory_delta_normalizer": float(
                        module.trajectory_delta_normalizer.float().cpu()
                    ),
                    "propagation_state_weights": (
                        None
                        if module.propagation_state_weight is None
                        else [
                            float(value)
                            for value in module.propagation_state_weight.float().cpu()
                        ]
                    ),
                    "absolute_propagation_risk": float(
                        module.absolute_propagation_risk.float().cpu()
                    ),
                    "propagation_clipping_weights": [
                        float(value)
                        for value in module.propagation_clipping_weights.float().cpu()
                    ],
                    "linear_gain_schedule": [
                        float(value)
                        for value in module.linear_gain_schedule.float().cpu()
                    ],
                    "activation_candidate_scores": [
                        [float(value) for value in row]
                        for row in module.activation_candidate_scores.float().cpu()
                    ],
                    "weight_numel": module.weight.numel(),
                    "activation_cost": (
                        module.calibration_activation_elements
                        / max(module.calibration_activation_calls, 1)
                    ),
                    "activation_cost_definition": "mean_input_elements_per_calibration_call",
                    "calibration_activation_calls": module.calibration_activation_calls,
                    "candidates": module.candidate_risks,
                    "mechanism_audit": dict(module.path_coupling_metadata),
                }
            )
        (out / "dit_quant_audit.json").write_text(
            json.dumps(
                {
                    "method": args.dit_quant_method,
                    "scope": args.dit_quant_scope,
                    "weight_bits": args.dit_weight_bits,
                    "activation_bits": args.dit_activation_bits,
                    "bgr_shrink_steps": args.dit_quant_bgr_shrink_steps,
                    "bgr_refine_steps": args.dit_quant_bgr_refine_steps,
                    "bgr_max_shrink_fraction": args.dit_quant_bgr_max_shrink_fraction,
                    "bgr_convergence_epsilon": args.dit_quant_bgr_convergence_epsilon,
                    "arq_block_size": args.dit_quant_arq_block_size,
                    "risk_lambda": args.dit_quant_risk_lambda,
                    "risk_tail_fraction": args.dit_quant_risk_tail_fraction,
                    "samples_per_call": args.dit_quant_samples_per_call,
                    "covariance_shrinkage": args.dit_quant_covariance_shrinkage,
                    "trajectory_delta_weight": args.dit_quant_trajectory_delta_weight,
                    "candidate_risk_normalization": None,
                    "trajectory_trust_ratio": args.dit_quant_trajectory_trust_ratio,
                    "activation_scale_momentum": args.dit_quant_activation_scale_momentum,
                    "activation_fixed_clipping_ratio": args.dit_quant_activation_fixed_clipping_ratio,
                    "activation_mse_min_ratio": args.dit_quant_activation_mse_min_ratio,
                    "activation_mse_candidates": args.dit_quant_activation_mse_candidates,
                    "activation_error_audit": args.dit_quant_activation_error_audit,
                    "trajectory_segment_calls": 2 * calibration_steps,
                    "inference_segment_calls": 2 * args.steps,
                    "calibration_scheduler_coordinates": list(calibration_coordinates),
                    "inference_scheduler_coordinates": list(temporal_coordinates),
                    "scheduler_mapping_policy": "cfg_pair_then_sigma_interpolation_to_runtime_bins",
                    "prop_clip_scheduler_align": os.getenv(
                        "PULSE_PROP_SCHEDULER_ALIGN", "1"
                    ),
                    "prop_clip_smooth_penalty": os.getenv(
                        "PULSE_PROP_CLIP_SMOOTH", "0"
                    ),
                    "prop_clip_beta": os.getenv("PULSE_PROP_CLIP_BETA", "1"),
                    "prop_linear_beta": os.getenv("PULSE_PROP_LINEAR_BETA", "0"),
                    "prop_clip_cvar": os.getenv("PULSE_PROP_CLIP_CVAR", "0"),
                    "bit_sensitivity_threshold": args.dit_quant_bit_sensitivity_threshold,
                    "modules": quant_audit,
                },
                indent=2,
            )
            + "\n"
        )

        if args.dit_quant_state_audit:
            state_audit = {
                "schema_version": 1,
                "method": args.dit_quant_method,
                "calibration_prompt_count": args.dit_quant_calibration_prompt_count,
                "warmup_prompt_offset": args.warmup_prompt_offset,
                "warmup_seed": warmup_seed,
                "calibration_prompt_indices": calibration_prompt_indices,
                "modules": {},
            }
            # A fixed uniform sketch preserves error direction and code agreement
            # information while keeping a 300-linear Wan snapshot only a few MB.
            sample_count = 2048
            for module in int_quant_modules:
                flat_weight = module.weight.detach().reshape(-1)
                count = min(sample_count, flat_weight.numel())
                indices = torch.linspace(
                    0,
                    flat_weight.numel() - 1,
                    count,
                    device=flat_weight.device,
                ).round().long()
                state_audit["modules"][module.module_name] = {
                    "weight_numel": flat_weight.numel(),
                    "weight_bits": module.spec.weight_bits,
                    "activation_bits": module.spec.activation_bits,
                    "weight_sample": flat_weight[indices].float().cpu(),
                    "channel_mask": (
                        None
                        if module.channel_mask is None
                        else module.channel_mask.detach().float().cpu()
                    ),
                    "act_max": (
                        None
                        if module.act_max is None
                        else module.act_max.detach().float().cpu()
                    ),
                }
            torch.save(state_audit, out / "dit_quant_state.pt")

    if linear_input_audit:
        audit_payload = {}
        for module_name, item in linear_input_audit.items():
            maxima = torch.stack(item["maxima"]).float().cpu() if item["maxima"] else torch.empty(0)
            audit_payload[module_name] = {
                "in_features": item["in_features"],
                "out_features": item["out_features"],
                "calls": int(maxima.numel()),
                "maximum_abs": float(maxima.max()) if maxima.numel() else None,
                "mean_call_max_abs": float(maxima.mean()) if maxima.numel() else None,
                "e4m3_unit_scale_saturated": bool(maxima.numel() and maxima.max() > 448),
            }
        (out / "linear_input_audit.json").write_text(
            json.dumps({"scope": "current_forward_only", "modules": audit_payload}, indent=2) + "\n"
        )
        for hook in linear_input_hooks:
            hook.remove()


if __name__ == "__main__":
    main()
