#!/usr/bin/env python3
"""MiniMax-H3 N-prompt whole-transformer W4A{4,6} fake-quant suite.

The runner keeps one loaded pipeline per method/precision configuration and
generates every prompt in the supplied manifest.  Quantization is restricted
to ``nn.Linear`` modules in the main T2VA transformer; VAE, text encoder,
normalization and convolutional projections remain in their checkpoint dtype.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
import types
from pathlib import Path

# H3 and Wan share the same PulseQuant linear implementation.  Put the
# canonical runtime ahead of this script directory so a stale H3-local copy
# cannot shadow the cache-v2 contract merely because this file is executed by
# pathname (Python otherwise places ``runtime/h3`` at ``sys.path[0]``).
_CANONICAL_QUANT_RUNTIME = Path(__file__).resolve().parents[1] / "wan21"
if not _CANONICAL_QUANT_RUNTIME.is_dir():
    raise RuntimeError(
        f"missing canonical PulseQuant runtime: {_CANONICAL_QUANT_RUNTIME}"
    )
sys.path.insert(0, str(_CANONICAL_QUANT_RUNTIME))

import torch
import torch.nn as nn
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils import export_to_video


def replace_linear_modules(
    transformer: nn.Module,
    spec,
    include_names: set[str] | None = None,
) -> list[nn.Module]:
    from spatial_single_step.int_quant_linear import WholeDiTQuantLinear

    targets = [
        (name, module)
        for name, module in transformer.named_modules()
        if name
        and isinstance(module, nn.Linear)
        and (include_names is None or name in include_names)
    ]
    replacements = []
    for name, source in targets:
        parent_name, key = name.rsplit(".", 1) if "." in name else ("", name)
        parent = transformer.get_submodule(parent_name) if parent_name else transformer
        replacement = WholeDiTQuantLinear(source, spec, name)
        # Diffusers' get_parameter_dtype() checks parameters before buffers.
        # WholeDiTQuantLinear stores its weight as a buffer, while its first
        # auxiliary floating buffer is FP32; without an anchor H3 therefore
        # casts selected BF16 projection inputs to FP32.  The empty parameter
        # preserves the source Linear's public dtype semantics without adding
        # storage or a higher-precision compute path.
        replacement.register_parameter(
            "_dtype_anchor",
            nn.Parameter(
                torch.empty(0, device=source.weight.device, dtype=source.weight.dtype),
                requires_grad=False,
            ),
        )
        if key.isdigit():
            parent[int(key)] = replacement
        else:
            setattr(parent, key, replacement)
        replacements.append(replacement)
    if not replacements:
        raise RuntimeError("no MiniMax-H3 transformer Linear modules found")
    return replacements


def resolve_devices(dit_device: str | None, text_device: str | None) -> tuple[str, str]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("MiniMax-H3 requires at least one visible CUDA device")
    resolved_dit = dit_device or "cuda:0"
    resolved_text = text_device or (
        "cuda:1" if torch.cuda.device_count() >= 2 else resolved_dit
    )
    for label, value in (("DiT", resolved_dit), ("text encoder", resolved_text)):
        device = torch.device(value)
        if device.type != "cuda":
            raise ValueError(f"{label} device must be CUDA, got {value!r}")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"{label} requested {value}, but only {torch.cuda.device_count()} "
                "CUDA device(s) are visible"
            )
    return resolved_dit, resolved_text


def load_pipeline(model: str, dit_device: str, text_device: str):
    workflow = ModularPipeline.from_pretrained(model).blocks.get_workflow("t2va")
    text_manager = ComponentsManager()
    text_manager.enable_auto_cpu_offload(device=text_device)
    conditioner = workflow.sub_blocks.pop("text_encoder").init_pipeline(
        model, components_manager=text_manager
    )
    for component in conditioner._component_specs.values():
        if component.pretrained_model_name_or_path is not None:
            component.pretrained_model_name_or_path = model
    conditioner.load_components(dtype=torch.bfloat16)

    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device=dit_device)
    pipe = workflow.init_pipeline(model, components_manager=manager)
    for component in pipe._component_specs.values():
        if component.pretrained_model_name_or_path is not None:
            component.pretrained_model_name_or_path = model
    pipe.load_components(dtype=torch.bfloat16)
    return conditioner, pipe, manager


def install_quantization(pipe, args):

    from spatial_single_step.int_quant_linear import QuantSpec

    internal_method = args.walsh_internal_method
    calibration_prompt_count = len(args.calibration_prompts)
    precision_profile = None
    propagation_state_weights = None
    absolute_propagation_risks = None
    propagation_profile_audit = None
    if not args.propagation_profile:
        raise ValueError(
            "PulseQuant requires --propagation-profile; refusing to "
            "silently fall back to an unweighted mean-tail objective"
        )
    profile_path = Path(args.propagation_profile)
    payload = json.loads(profile_path.read_text())
    if payload.get("uses_evaluation_scores") is not False:
        raise ValueError(
            "propagation profile must explicitly declare "
            "uses_evaluation_scores=false"
        )
    rows = payload.get("modules") or []
    if not rows:
        raise ValueError("propagation profile contains no module entries")
    missing_weights = [
        row.get("module", "<unnamed>")
        for row in rows
        if not row.get("propagation_state_weights")
    ]
    if missing_weights:
        raise ValueError(
            "propagation profile has entries without state weights: "
            + ", ".join(missing_weights[:8])
        )
    precision_profile = {
        row["module"]: (4, args.activation_bits) for row in rows
    }
    propagation_state_weights = {
        row["module"]: tuple(
            float(value) for value in row["propagation_state_weights"]
        )
        for row in rows
    }
    absolute_propagation_risks = {
        row["module"]: float(row.get("absolute_propagation_risk", 1.0))
        for row in rows
    }
    state_lengths = sorted(
        {len(values) for values in propagation_state_weights.values()}
    )
    total_calls = args.calibration_calls * calibration_prompt_count
    incompatible = [length for length in state_lengths if total_calls % length]
    if incompatible:
        raise ValueError(
            "profile state-weight lengths must divide the number of "
            f"calibration calls ({total_calls}); got {incompatible}"
        )
    propagation_profile_audit = {
        "path": str(profile_path.resolve()),
        "module_count": len(rows),
        "state_weight_lengths": state_lengths,
        "selected_blocks": payload.get("selected_blocks"),
        "selected_anchors": payload.get("selected_anchors"),
        "calibration_source": payload.get("calibration_source"),
    }
    spec = QuantSpec(
        method=internal_method,
        weight_bits=4,
        activation_bits=args.activation_bits,
        calibration_calls=args.calibration_calls * calibration_prompt_count,
        alpha=args.alpha,
        hadamard_max_group=args.hadamard_group,
        bgr_shrink_steps=args.shrink_steps,
        bgr_refine_steps=args.refine_steps,
        risk_lambda=args.risk_lambda,
        samples_per_call=args.samples_per_call,
        trajectory_delta_weight=args.trajectory_delta_weight,
        trajectory_segment_calls=args.calibration_calls,
        risk_tail_fraction=args.risk_tail_fraction,
        precision_profile=precision_profile,
        propagation_state_weights=propagation_state_weights,
        absolute_propagation_risks=absolute_propagation_risks,
    )
    policy_audit = None
    include_names = None
    if os.environ.get("H3_MATCH_ORBIT_POLICY", "0") == "1":
        import orbitquant

        policy_recipe = orbitquant.recipe(
            f"w4a{args.activation_bits}",
            target_policy="universal",
            runtime_mode="dequant_bf16",
        )
        policy_audit = orbitquant.inspect_linear_module_policy(
            pipe.transformer, policy_recipe
        )
        include_names = {
            row["name"]
            for row in policy_audit["modules"]
            if row["action"] == "orbitquant"
        }
    modules = replace_linear_modules(pipe.transformer, spec, include_names)
    special_summary = None
    if include_names is not None:
        # The custom Walsh2 wrappers now occupy the ordinary projection sites.
        # Running the official recipe over the remaining nn.Linear modules
        # preserves H3's dedicated AdaLN INT4 path and boundary policy.
        special_summary = orbitquant.quantize_model(
            pipe.transformer,
            policy_recipe,
            quantization_device="cuda",
            staging_mode="streaming",
        )
    return {
        "method": args.method,
        "internal_method": internal_method,
        "implementation": "PulseQuant RPBH + propagation-weighted radius reconstruction + response-subspace correction",
        "weight_bits": 4,
        "activation_bits": args.activation_bits,
        "runtime_mode": "BF16 fake quant",
        "linear_count": len(modules),
        "h3_policy_match": (
            None
            if policy_audit is None
            else {
                "custom_orbit_projection_count": len(include_names),
                "original_action_counts": policy_audit["action_counts"],
                "remaining_official_summary": str(special_summary),
            }
        ),
        "calibration_calls_per_prompt": args.calibration_calls,
        "calibration_prompt_count": calibration_prompt_count,
        "calibration_calls": args.calibration_calls * calibration_prompt_count,
        "calibration_uses_test_prompts": False,
        "risk_lambda": args.risk_lambda,
        "risk_tail_fraction": args.risk_tail_fraction,
        "trajectory_delta_weight": args.trajectory_delta_weight,
        "trajectory_stride": int(os.environ.get("PULSE_TRAJECTORY_STRIDE", "2")),
        "propagation_profile": propagation_profile_audit,
    }, modules


def install_device_guards(transformer: nn.Module) -> int:
    """Keep non-persistent RoPE buffers aligned with their runtime input.

    MiniMax-H3's auto-offload hook does not track ``inv_freq`` after a
    quantizer mutates the transformer in place.  The buffer is tiny, so move
    it lazily immediately before RoPE rather than changing the model's
    offload policy.
    """

    guarded = 0
    for module in transformer.modules():
        if not hasattr(module, "inv_freq"):
            continue

        def sync_inv_freq(target, positional):
            if not positional or not torch.is_tensor(positional[0]):
                return
            wanted = positional[0].device
            inv_freq = target.inv_freq
            if inv_freq.device != wanted:
                target.inv_freq = inv_freq.to(device=wanted, non_blocking=True)

        module.register_forward_pre_hook(sync_inv_freq)
        guarded += 1
    return guarded


def install_h3_sampling(transformer, modules):
    """Resolve neighborhoods from actual packed video coordinates, never token counts."""
    if os.environ.get("H3_STRUCTURED_SAMPLE", "0") != "1":
        return

    def prepare(_module, positional, kwargs):
        if all(module.finalized for module in modules):
            return
        positions = kwargs["position_ids"].detach().cpu()
        video = kwargs["video_indices"].detach().cpu().tolist()
        coords = [tuple(positions[i].tolist()) for i in video]
        lookup = {xyz: index for index, xyz in enumerate(coords)}
        axes = [sorted(set(x[d] for x in coords)) for d in range(3)]
        successors = [{a:b for a,b in zip(axis, axis[1:])} for axis in axes]
        neighborhoods = []
        for i, xyz in enumerate(coords):
            near = []
            for d in (2, 1, 0):
                if xyz[d] not in successors[d]:
                    break
                target = list(xyz)
                target[d] = successors[d][xyz[d]]
                if tuple(target) not in lookup:
                    break
                near.append(lookup[tuple(target)])
            if len(near) == 3:
                neighborhoods.append([i] + near)
        if len(neighborhoods) < 8:
            raise RuntimeError("H3 sampling: insufficient verified video neighborhoods")
        selected = [neighborhoods[i] for i in torch.linspace(0, len(neighborhoods)-1, 8).round().long().tolist()]
        local = [i for neighborhood in selected for i in neighborhood]
        packed = [video[i] for i in local]
        edges = tuple((4*i, 4*i+j) for i in range(8) for j in (1,2,3))
        for module in modules:
            name = module.module_name
            is_packed = (name.startswith("transformer_blocks.") and (".attn." in name or ".ff." in name)) or name in ("proj_out", "audio_proj_out")
            is_video = name == "proj_in"
            if not (is_packed or is_video):
                continue
            total = len(positions) if is_packed else len(video)
            coverage = torch.linspace(0, total-1, 16).round().long().tolist()
            module.h3_sample_indices = torch.tensor((packed if is_packed else local) + coverage)
            module.h3_edges = edges
        if not getattr(_module, "h3_sampling_logged", False):
            print(json.dumps({"event":"h3_neighborhoods_verified", "video_tokens":len(video), "axes":[len(a) for a in axes], "neighborhoods":8, "samples":48}), flush=True)
            _module.h3_sampling_logged = True

    transformer.register_forward_pre_hook(prepare, with_kwargs=True)


def pipeline_call(pipe, state, args, seed: int, steps: int):
    return pipe(
        state=state,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_inference_steps=steps,
        generator=torch.Generator().manual_seed(seed),
        output=["videos", "audio", "sampling_rate"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dit-device",
        help="CUDA device for the H3 transformer (default: cuda:0).",
    )
    parser.add_argument(
        "--text-device",
        help=(
            "CUDA device for the text encoder. Defaults to cuda:1 when two "
            "devices are visible, otherwise shares the DiT device."
        ),
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=('walsh2',), required=True)
    parser.add_argument("--activation-bits", type=int, choices=(4, 6), required=True)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=31)
    parser.add_argument("--calibration-calls", type=int, default=5)
    parser.add_argument("--calibration-prompt", default="A ceramic teapot rotating slowly on a wooden table in soft studio light")
    parser.add_argument("--calibration-prompts-file")
    parser.add_argument(
        "--propagation-profile",
        help=(
            "Model-specific pulse-propagation profile. Required when "
            "--walsh-internal-method=pulseprofile."
        ),
    )
    parser.add_argument("--calibration-seed", type=int, default=1701)
    parser.add_argument("--alpha", type=float, default=0.5665)
    parser.add_argument("--hadamard-group", type=int, default=128)
    parser.add_argument("--shrink-steps", type=int, default=12)
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--risk-lambda", type=float, default=0.75)
    parser.add_argument("--risk-tail-fraction", type=float, default=0.0)
    parser.add_argument("--samples-per-call", type=int, default=2)
    parser.add_argument("--trajectory-delta-weight", type=float, default=1.0)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--indices", default="")
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Finalize and optionally save the quantizer without generating videos.",
    )
    parser.add_argument(
        "--save-quant-cache",
        help="Persist finalized PulseQuant tensors after calibration.",
    )
    parser.add_argument(
        "--load-quant-cache",
        help="Load finalized PulseQuant tensors and skip calibration.",
    )
    parser.add_argument(
        "--walsh-internal-method",
        choices=('pulseprofile',),
        default="pulseprofile",
    )
    args = parser.parse_args()

    args.calibration_prompts = [args.calibration_prompt]
    if args.calibration_prompts_file:
        args.calibration_prompts = json.loads(
            Path(args.calibration_prompts_file).read_text()
        )
        if not args.calibration_prompts:
            raise ValueError("calibration prompt list must not be empty")

    args.dit_device, args.text_device = resolve_devices(
        args.dit_device, args.text_device
    )
    torch.cuda.set_device(torch.device(args.dit_device))
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    conditioner, pipe, manager = load_pipeline(
        args.model, args.dit_device, args.text_device
    )
    audit, quant_modules = install_quantization(pipe, args)
    audit["device_placement"] = {
        "visible_cuda_devices": torch.cuda.device_count(),
        "dit": args.dit_device,
        "text_encoder": args.text_device,
    }
    install_h3_sampling(pipe.transformer, quant_modules)
    audit["h3_experiment"] = {key: os.environ.get(key) for key in ("H3_STRUCTURED_SAMPLE", "H3_EDGE_WEIGHT", "H3_A_CLIP", "PULSE_RPBH_SEED")}
    audit["alpha"] = args.alpha
    # Quantizers mutate parameters/buffers after ComponentsManager attached
    # its model hook.  Refresh the hook so its device test and model footprint
    # reflect the quantized transformer rather than the pre-quantization one.
    manager.enable_auto_cpu_offload(device=args.dit_device)
    audit["rope_device_guards"] = install_device_guards(pipe.transformer)
    print(
        json.dumps(
            {
                "event": "quantization_installed",
                "method": args.method,
                "activation_bits": args.activation_bits,
                "quant_linear_count": len(quant_modules),
                "rope_device_guards": audit["rope_device_guards"],
            }
        ),
        flush=True,
    )

    calibration_seconds = 0.0
    quant_cache_context = None
    if quant_modules:
        from spatial_single_step.int_quant_linear import build_quant_cache_context

        quant_cache_context = build_quant_cache_context(
            args.model,
            propagation_profile=args.propagation_profile,
            calibration_prompts=args.calibration_prompts_file,
            extra={
                "runner": "minimax_h3",
                "method": args.method,
                "walsh_internal_method": args.walsh_internal_method,
                "weight_bits": 4,
                "activation_bits": args.activation_bits,
                "calibration_prompts": args.calibration_prompts,
                "calibration_calls_per_prompt": args.calibration_calls,
                "alpha": args.alpha,
                "hadamard_group": args.hadamard_group,
                "shrink_steps": args.shrink_steps,
                "refine_steps": args.refine_steps,
                "risk_lambda": args.risk_lambda,
                "risk_tail_fraction": args.risk_tail_fraction,
                "samples_per_call": args.samples_per_call,
                "trajectory_delta_weight": args.trajectory_delta_weight,
                "protocol": {
                    "height": args.height,
                    "width": args.width,
                    "frames": args.frames,
                    "steps": args.steps,
                },
                "h3_match_orbit_policy": os.environ.get(
                    "H3_MATCH_ORBIT_POLICY", "0"
                ),
            },
        )
    if quant_modules and args.load_quant_cache:
        from spatial_single_step.int_quant_linear import load_finalized_quant_cache

        audit["quant_cache_load"] = load_finalized_quant_cache(
            quant_modules,
            args.load_quant_cache,
            cache_context=quant_cache_context,
        )
        print(
            json.dumps({"event": "quant_cache_loaded", **audit["quant_cache_load"]}),
            flush=True,
        )
    elif quant_modules:
        start = time.perf_counter()
        for prompt_index, calibration_prompt in enumerate(args.calibration_prompts):
            calibration_state = conditioner(prompt=calibration_prompt)
            # Diffusers performs steps-1 transformer calls for this workflow.
            pipeline_call(
                pipe,
                calibration_state,
                args,
                args.calibration_seed + prompt_index,
                args.calibration_calls + 1,
            )
        torch.cuda.synchronize()
        calibration_seconds = time.perf_counter() - start
        unfinished = [module.module_name for module in quant_modules if not module.finalized]
        if unfinished:
            raise RuntimeError(f"quant calibration did not finalize {len(unfinished)} modules")
        walsh2_rows = [
            {"module": module.module_name, **module.path_coupling_metadata}
            for module in quant_modules
            if "protected_mse_ratio" in module.path_coupling_metadata
        ]
        if walsh2_rows:
            (output_dir / "walsh2_correction_audit.json").write_text(
                json.dumps(walsh2_rows, indent=2) + "\n"
            )
            audit["walsh2_correction"] = {
                "module_count": len(walsh2_rows),
                "changed_fraction_mean": sum(
                    row["changed_fraction"] for row in walsh2_rows
                ) / len(walsh2_rows),
                "weight_mse_ratio_mean": sum(
                    row["weight_mse_ratio"] for row in walsh2_rows
                ) / len(walsh2_rows),
                "protected_mse_ratio_mean": sum(
                    row["protected_mse_ratio"] for row in walsh2_rows
                ) / len(walsh2_rows),
            }
        print(
            json.dumps(
                {
                    "event": "calibration_complete",
                    "method": args.method,
                    "activation_bits": args.activation_bits,
                    "seconds": calibration_seconds,
                }
            ),
            flush=True,
        )
        if args.save_quant_cache:
            from spatial_single_step.int_quant_linear import save_finalized_quant_cache

            audit["quant_cache_save"] = save_finalized_quant_cache(
                quant_modules,
                args.save_quant_cache,
                cache_context=quant_cache_context,
            )
            print(
                json.dumps(
                    {"event": "quant_cache_saved", **audit["quant_cache_save"]}
                ),
                flush=True,
            )

    if args.calibration_only:
        suite = {
            "model": args.model,
            "protocol": {
                "height": args.height,
                "width": args.width,
                "frames": args.frames,
                "steps": args.steps,
            },
            "quantization": audit,
            "calibration_seconds": calibration_seconds,
            "records": [],
        }
        (output_dir / "suite.json").write_text(json.dumps(suite, indent=2) + "\n")
        (output_dir / "DONE").write_text("ok\n")
        return

    records = []
    if args.indices:
        selected_indices = [int(value) for value in args.indices.split(",")]
        indexed_manifest = [(index, manifest[index]) for index in selected_indices]
    else:
        selected_manifest = manifest[: args.max_prompts] if args.max_prompts > 0 else manifest
        indexed_manifest = list(enumerate(selected_manifest))
    for index, row in indexed_manifest:
        name = row["name"]
        destination = video_dir / f"{index:02d}_{name}.mp4"
        if (
            destination.exists()
            and destination.stat().st_size > 100_000
            and destination.with_suffix(".json").exists()
        ):
            records.append(json.loads(destination.with_suffix(".json").read_text()))
            continue
        state_start = time.perf_counter()
        state = conditioner(prompt=row["prompt"])
        conditioned = time.perf_counter()
        events = []
        original = pipe.transformer.forward

        @functools.wraps(original)
        def timed(_module, *positional, **keywords):
            begin_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            begin_event.record()
            result = original(*positional, **keywords)
            end_event.record()
            events.append((begin_event, end_event))
            return result

        pipe.transformer.forward = types.MethodType(timed, pipe.transformer)
        torch.cuda.set_device(torch.device(args.dit_device))
        torch.cuda.reset_peak_memory_stats()
        result = pipeline_call(pipe, state, args, int(row["seed"]), args.steps)
        torch.cuda.synchronize()
        finished = time.perf_counter()
        pipe.transformer.forward = original
        dit_calls = [start.elapsed_time(end) / 1000 for start, end in events]
        export_to_video(result["videos"][0], str(destination), fps=24)
        torch.save(
            {"audio": result["audio"][0], "sampling_rate": result["sampling_rate"]},
            destination.with_suffix(".audio.pt"),
        )
        record = {
            **row,
            "index": index,
            "output": str(destination),
            "method": args.method,
            "weight_bits": 4,
            "activation_bits": args.activation_bits,
            "conditioner_seconds": conditioned - state_start,
            "generation_seconds": finished - conditioned,
            "dit_seconds": sum(dit_calls),
            "dit_calls": len(dit_calls),
            "peak_allocated_gib_gpu0": torch.cuda.max_memory_allocated() / 2**30,
        }
        destination.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print(json.dumps(record), flush=True)

    suite = {
        "model": args.model,
        "protocol": {"height": args.height, "width": args.width, "frames": args.frames, "steps": args.steps},
        "quantization": audit,
        "calibration_seconds": calibration_seconds,
        "records": records,
    }
    (output_dir / "suite.json").write_text(json.dumps(suite, indent=2) + "\n")
    (output_dir / "DONE").write_text("ok\n")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    main()
