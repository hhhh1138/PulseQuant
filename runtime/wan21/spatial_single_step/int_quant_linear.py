#!/usr/bin/env python3
"""PulseQuant calibration and quantize/dequantize inference for video transformers.

The released path combines RPBH coordinates, propagation-weighted radius selection,
and activation-adapted response-subspace correction. Uniform per-row quantization
is retained only as a low-level profiling primitive. Cache field names and the
internal taorgwfmse identifier are retained for historical checkpoint compatibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pulsequant_utils.release_policy import validate_release_settings
import time
import warnings
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F



def _safe_scale(value: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return value.clamp_min(eps)


_PULSE_ACTIVATION_CODEBOOK = "uniform".lower()


def _mean_tail_risk(
    per_state: torch.Tensor, risk_lambda: float, tail_fraction: float = 0.0
) -> torch.Tensor:
    """Blend mean risk with max risk or a sample-count-stable CVaR tail."""
    if per_state.shape[0] == 0:
        # A very short synthetic/calibration segment can contain no valid
        # stride-two within-segment delta. Treat the absent delta term as zero
        # instead of crashing in ``amax``; real profiles use much longer
        # denoising segments but the primitive should still be total.
        return per_state.sum(0)
    if tail_fraction > 0.0:
        tail_count = max(1, math.ceil(per_state.shape[0] * tail_fraction))
        tail = per_state.topk(tail_count, dim=0, sorted=False).values.mean(0)
    else:
        tail = per_state.amax(0)
    return (1.0 - risk_lambda) * per_state.mean(0) + risk_lambda * tail


def _propagation_weighted_risk(
    per_state: torch.Tensor,
    state_weights: torch.Tensor | None,
    risk_lambda: float,
    tail_fraction: float,
) -> torch.Tensor:
    if state_weights is None:
        return _mean_tail_risk(per_state, risk_lambda, tail_fraction)
    weights = state_weights.to(device=per_state.device, dtype=per_state.dtype).flatten()
    if per_state.shape[0] % weights.numel() != 0:
        raise ValueError(
            "propagation state-weight count must divide calibration calls: "
            f"{weights.numel()} vs {per_state.shape[0]}"
        )
    weights = weights.repeat(per_state.shape[0] // weights.numel())
    weights = weights / weights.mean().clamp_min(1e-12)
    weighted_state = per_state * weights.reshape(-1, *([1] * (per_state.ndim - 1)))
    return _mean_tail_risk(weighted_state, risk_lambda, tail_fraction)


def _phase_aligned_bin_indices(
    total_calls: int,
    trajectory_period: int,
    bins: int,
    bin_index: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Select the same denoising phase from every calibration trajectory."""
    if trajectory_period <= 0 or total_calls <= trajectory_period:
        start = bin_index * total_calls // bins
        stop = (bin_index + 1) * total_calls // bins
        return torch.arange(start, stop, device=device)
    indices = []
    for trajectory_start in range(0, total_calls, trajectory_period):
        trajectory_stop = min(trajectory_start + trajectory_period, total_calls)
        length = trajectory_stop - trajectory_start
        phase_start = trajectory_start + bin_index * length // bins
        phase_stop = trajectory_start + (bin_index + 1) * length // bins
        indices.append(torch.arange(phase_start, phase_stop, device=device))
    return torch.cat(indices)


def _interp_phase_values(
    values: torch.Tensor,
    source_coordinates: torch.Tensor,
    target_coordinates: torch.Tensor,
) -> torch.Tensor:
    """Linearly resample values whose final axis follows descending sigma."""
    source = source_coordinates.to(device=values.device, dtype=values.dtype).flatten()
    target = target_coordinates.to(device=values.device, dtype=values.dtype).flatten()
    if source.numel() != values.shape[-1]:
        raise ValueError(f"coordinate/value mismatch: {source.numel()} vs {values.shape[-1]}")
    if source.numel() == 1:
        return values[..., :1].expand(*values.shape[:-1], target.numel())
    order = torch.argsort(source)
    source = source[order]
    values = values[..., order]
    target = target.clamp(source[0], source[-1])
    right = torch.searchsorted(source, target).clamp(1, source.numel() - 1)
    left = right - 1
    x0, x1 = source[left], source[right]
    mix = (target - x0) / (x1 - x0).clamp_min(1e-12)
    return values[..., left] * (1.0 - mix) + values[..., right] * mix


def _scheduler_bin_coordinates(
    inference_coordinates: tuple[float, ...], bins: int, *, device: torch.device
) -> torch.Tensor:
    coordinates = torch.tensor(inference_coordinates, device=device, dtype=torch.float32)
    if coordinates.numel() == 0:
        return torch.linspace(1.0, 0.0, bins, device=device)
    return torch.stack(
        [
            coordinates[index * coordinates.numel() // bins : (index + 1) * coordinates.numel() // bins].mean()
            for index in range(bins)
        ]
    )


def _scheduler_aligned_candidate_scores(
    error_matrix: torch.Tensor,
    trajectory_period: int,
    bins: int,
    calibration_coordinates: tuple[float, ...],
    inference_coordinates: tuple[float, ...],
    risk_lambda: float,
    tail_fraction: float,
) -> torch.Tensor:
    """Return [bin, candidate] scores without splitting the two CFG calls."""
    if trajectory_period <= 0 or trajectory_period % 2:
        raise ValueError("scheduler-aligned calibration requires an even trajectory period")
    steps = trajectory_period // 2
    source = torch.tensor(calibration_coordinates, device=error_matrix.device, dtype=torch.float32)
    if source.numel() == 0:
        source = torch.linspace(1.0, 0.0, steps, device=error_matrix.device)
    if source.numel() != steps:
        raise ValueError(f"calibration sigma count must equal denoise steps: {source.numel()} vs {steps}")
    target = _scheduler_bin_coordinates(inference_coordinates, bins, device=error_matrix.device)
    trajectories = []
    for start in range(0, error_matrix.shape[1], trajectory_period):
        segment = error_matrix[:, start : start + trajectory_period]
        if segment.shape[1] != trajectory_period:
            continue
        paired = segment.reshape(segment.shape[0], steps, 2).mean(-1)
        trajectories.append(_interp_phase_values(paired, source, target))
    if not trajectories:
        raise ValueError("no complete calibration trajectory for scheduler alignment")
    per_prompt = torch.stack(trajectories, dim=0)  # [prompt, candidate, bin]
    return torch.stack(
        [
            _mean_tail_risk(per_prompt[:, :, index], risk_lambda, tail_fraction)
            for index in range(bins)
        ],
        dim=0,
    )


def _resample_propagation_weights(
    weights: torch.Tensor | None,
    bins: int,
    calibration_coordinates: tuple[float, ...],
    inference_coordinates: tuple[float, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    if weights is None:
        return torch.ones(bins, device=device)
    values = weights.to(device=device, dtype=torch.float32).flatten()
    steps = len(calibration_coordinates) or max(1, values.numel() // 2)
    if values.numel() == 2 * steps:
        values = values.reshape(steps, 2).mean(-1)
    elif values.numel() != steps:
        values = F.interpolate(values.reshape(1, 1, -1), size=steps, mode="linear", align_corners=True).flatten()
    source = torch.tensor(calibration_coordinates, device=device, dtype=torch.float32)
    if source.numel() == 0:
        source = torch.linspace(1.0, 0.0, steps, device=device)
    target = _scheduler_bin_coordinates(inference_coordinates, bins, device=device)
    result = _interp_phase_values(values, source, target)
    return result / result.mean().clamp_min(1e-12)


def _smooth_prop_clipping_schedule(
    scores: torch.Tensor,
    propagation: torch.Tensor,
    candidate_ratios: tuple[float, ...],
    smooth_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chain DP over timestep bins; zero penalty is the independent argmin."""
    baseline = scores.amin(dim=1, keepdim=True).clamp_min(1e-12)
    regret = (scores / baseline - 1.0).clamp_min(0.0)
    cost = regret * propagation[:, None]
    if smooth_penalty <= 0.0:
        indices = cost.argmin(dim=1)
        return indices, cost
    ratios = torch.tensor(candidate_ratios, device=scores.device, dtype=scores.dtype)
    transition = (ratios[:, None] - ratios[None, :]).abs() / 0.05 * smooth_penalty
    dynamic = cost[0]
    parents = []
    for index in range(1, cost.shape[0]):
        choices = dynamic[:, None] + transition
        best, parent = choices.min(dim=0)
        dynamic = cost[index] + best
        parents.append(parent)
    selected = [int(dynamic.argmin())]
    for parent in reversed(parents):
        selected.append(int(parent[selected[-1]]))
    selected.reverse()
    return torch.tensor(selected, device=scores.device), cost


@torch.inference_mode()
def token_symmetric_qdq(value: torch.Tensor, bits: int) -> torch.Tensor:
    """Dynamic per-token symmetric quantization, returned in input dtype."""
    qmax = 2 ** (bits - 1) - 1
    dtype = value.dtype
    flat = value.float().reshape(-1, value.shape[-1])
    scale = _safe_scale(flat.abs().amax(dim=1, keepdim=True) / qmax)
    quant = torch.round(flat / scale).clamp(-qmax, qmax)
    return (quant * scale).to(dtype).reshape_as(value)


@torch.inference_mode()
def clipped_token_qdq(value: torch.Tensor, bits: int, ratio: float) -> torch.Tensor:
    """Token-wise symmetric Q/DQ with a fixed clipping ratio."""
    qmax = 2 ** (bits - 1) - 1
    dtype = value.dtype
    flat = value.float().reshape(-1, value.shape[-1])
    maximum = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) * ratio
    scale = _safe_scale(maximum / qmax)
    quant = torch.round(flat / scale).clamp(-qmax, qmax)
    return (quant * scale).to(dtype).reshape_as(value)


@torch.inference_mode()
def frame_consistent_clipped_token_qdq(
    value: torch.Tensor,
    bits: int,
    previous_maximum: torch.Tensor | None,
    denoising_momentum: float,
    clipping_ratio: float | torch.Tensor,
    latent_frames: int,
    tube_share: float,
    scale_smoothing: float,
    scale_second_smoothing: float = 0.0,
    motion_gate_tau: float = 0.0,
    moment_strength: float = 0.0,
    moment_group: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token W4 Q/DQ with a distinct, explicit video-frame axis.

    Wan flattens ``T x H x W`` into the token dimension.  This path restores
    that layout, smooths *log range* only along T, and optionally removes the
    temporally low-pass component of group-mean quantization bias.  Denoising
    momentum remains a separate axis and is applied before frame smoothing.
    """
    qmax = 2 ** (bits - 1) - 1
    dtype = value.dtype
    if value.ndim == 2:
        shaped_value = value.unsqueeze(0)
        squeeze_batch = True
    elif value.ndim == 3:
        shaped_value = value
        squeeze_batch = False
    else:
        flat = value.float().reshape(-1, value.shape[-1])
        current = flat.abs().amax(1, keepdim=True).clamp_min(1e-8)
        ratio = torch.as_tensor(clipping_ratio, device=flat.device, dtype=flat.dtype)
        scale = (current * ratio / qmax).clamp_min(1e-8)
        quantized = (torch.round(flat / scale).clamp(-qmax, qmax) * scale).to(dtype).reshape_as(value)
        return quantized, current.detach()

    batch, tokens, channels = shaped_value.shape
    if latent_frames <= 1 or tokens % latent_frames or tokens // latent_frames < 4:
        current = shaped_value.float().abs().amax(-1, keepdim=True)
        ratio = torch.as_tensor(
            clipping_ratio, device=shaped_value.device, dtype=torch.float32
        )
        scale = (current * ratio / qmax).clamp_min(1e-8)
        quantized = (
            torch.round(shaped_value.float() / scale).clamp(-qmax, qmax) * scale
        ).to(dtype)
        if squeeze_batch:
            quantized = quantized.squeeze(0)
        return quantized, current.detach()

    work = shaped_value.float()
    current = work.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    maximum = current
    if previous_maximum is not None and previous_maximum.shape == current.shape:
        maximum = current.lerp(previous_maximum, denoising_momentum)

    spatial = tokens // latent_frames
    temporal_max = maximum.reshape(batch, latent_frames, spatial, 1)
    log_range = temporal_max.log()
    if tube_share > 0.0:
        # A tube is one spatial token followed through latent time.  Sharing
        # its log-domain range gives every frame identical A4 bin boundaries
        # at strength=1, while retaining a continuous calibration knob.
        tube_log_range = log_range.mean(dim=1, keepdim=True)
        log_range = log_range.lerp(
            tube_log_range, min(max(float(tube_share), 0.0), 1.0)
        )
    if scale_smoothing > 0.0:
        padded = torch.cat((log_range[:, :1], log_range, log_range[:, -1:]), dim=1)
        filtered = (padded[:, :-2] + 2.0 * padded[:, 1:-1] + padded[:, 2:]) * 0.25
        gate = torch.ones_like(log_range)
        if motion_gate_tau > 0.0:
            rms = work.reshape(batch, latent_frames, spatial, channels).square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt().log()
            change = torch.zeros_like(rms)
            change[:, 1:] = (rms[:, 1:] - rms[:, :-1]).abs()
            gate = torch.exp(-change / motion_gate_tau)
        log_range = log_range.lerp(filtered, (scale_smoothing * gate).clamp(0.0, 1.0))
    if scale_second_smoothing > 0.0 and latent_frames >= 3:
        padded = torch.cat((log_range[:, :1], log_range[:, :1], log_range, log_range[:, -1:], log_range[:, -1:]), dim=1)
        filtered = (padded[:, :-4] + 4.0 * padded[:, 1:-3] + 6.0 * padded[:, 2:-2] + 4.0 * padded[:, 3:-1] + padded[:, 4:]) / 16.0
        log_range = log_range.lerp(filtered, scale_second_smoothing)

    smoothed_maximum = log_range.exp().reshape(batch, tokens, 1)
    ratio = torch.as_tensor(clipping_ratio, device=work.device, dtype=work.dtype)
    scale = (smoothed_maximum * ratio / qmax).clamp_min(1e-8)
    dequant = torch.round(work / scale).clamp(-qmax, qmax) * scale

    if moment_strength > 0.0 and channels % moment_group == 0:
        groups = channels // moment_group
        residual = (dequant - work).reshape(batch, latent_frames, spatial, groups, moment_group)
        # One scalar per frame and channel group: a genuinely low-rank side
        # correction, not restoration of the full BF16 activation residual.
        bias = residual.mean((2, 4), keepdim=True)
        padded = torch.cat((bias[:, :1], bias, bias[:, -1:]), dim=1)
        lowpass_bias = (padded[:, :-2] + 2.0 * padded[:, 1:-1] + padded[:, 2:]) * 0.25
        dequant = dequant - moment_strength * lowpass_bias.expand(-1, -1, spatial, -1, moment_group).reshape_as(dequant)

    result = dequant.to(dtype)
    if squeeze_batch:
        result = result.squeeze(0)
    return result, current.detach()


def _compiled_clipped_token_qdq_graph(
    value: torch.Tensor, ratio: torch.Tensor, qmax: torch.Tensor
) -> torch.Tensor:
    """Inductor-fused FP32 token Q/DQ with tensor-valued runtime controls."""
    dtype = value.dtype
    flat = value.float().reshape(-1, value.shape[-1])
    maximum = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) * ratio
    scale = (maximum / qmax).clamp_min(1e-8)
    quant = torch.round(flat / scale).clamp(-qmax, qmax)
    return (quant * scale).to(dtype).reshape_as(value)


_COMPILED_CLIPPED_TOKEN_QDQ = None
_COMPILED_CACHED_SCALE_QDQ = None
_COMPILED_MASKED_HADAMARD = None


def compiled_clipped_token_qdq(
    value: torch.Tensor, ratio: torch.Tensor, qmax: torch.Tensor
) -> torch.Tensor:
    """Lazily build one shape-dynamic fused activation Q/DQ callable."""
    global _COMPILED_CLIPPED_TOKEN_QDQ
    if _COMPILED_CLIPPED_TOKEN_QDQ is None:
        _COMPILED_CLIPPED_TOKEN_QDQ = torch.compile(
            _compiled_clipped_token_qdq_graph,
            fullgraph=True,
            # Wan uses a small, fixed family of token/feature shapes. Static
            # kernels are materially faster here; ratio/qmax remain tensor
            # inputs, so ten schedule values do not trigger recompilation.
            dynamic=False,
            mode="default",
        )
    return _COMPILED_CLIPPED_TOKEN_QDQ(value, ratio, qmax)


def _compiled_cached_scale_qdq_graph(
    value: torch.Tensor,
    maximum: torch.Tensor,
    ratio: torch.Tensor,
    qmax: torch.Tensor,
) -> torch.Tensor:
    """Q/DQ with a previously measured token range (no online reduction)."""
    dtype = value.dtype
    flat = value.float().reshape(-1, value.shape[-1])
    scale = (maximum * ratio / qmax).clamp_min(1e-8)
    quant = torch.round(flat / scale).clamp(-qmax, qmax)
    return (quant * scale).to(dtype).reshape_as(value)


def compiled_cached_scale_qdq(
    value: torch.Tensor,
    maximum: torch.Tensor,
    ratio: torch.Tensor,
    qmax: torch.Tensor,
) -> torch.Tensor:
    """Lazily compile the reduction-free activation Q/DQ path."""
    global _COMPILED_CACHED_SCALE_QDQ
    if _COMPILED_CACHED_SCALE_QDQ is None:
        _COMPILED_CACHED_SCALE_QDQ = torch.compile(
            _compiled_cached_scale_qdq_graph,
            fullgraph=True,
            dynamic=False,
            mode=os.environ.get("PULSE_COMPILE_MODE", "default"),
        )
    return _COMPILED_CACHED_SCALE_QDQ(value, maximum, ratio, qmax)


def _compiled_masked_hadamard_graph(
    value: torch.Tensor, mask: torch.Tensor, matrix: torch.Tensor
) -> torch.Tensor:
    group = matrix.shape[0]
    work = (value * mask).reshape(-1, value.shape[-1] // group, group)
    return torch.matmul(work, matrix).reshape_as(value)


def compiled_masked_hadamard(
    value: torch.Tensor, mask: torch.Tensor, matrix: torch.Tensor
) -> torch.Tensor:
    """Fuse channel balancing into the tensor-core Hadamard input path."""
    global _COMPILED_MASKED_HADAMARD
    if _COMPILED_MASKED_HADAMARD is None:
        _COMPILED_MASKED_HADAMARD = torch.compile(
            _compiled_masked_hadamard_graph,
            fullgraph=True,
            dynamic=False,
            mode="default",
        )
    return _COMPILED_MASKED_HADAMARD(value, mask, matrix)


@torch.inference_mode()
def trajectory_anchored_clipped_token_qdq(
    value: torch.Tensor,
    bits: int,
    previous_maximum: torch.Tensor | None,
    momentum: float,
    clipping_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same-CFG range prediction plus frozen clipping in one integer Q/DQ."""
    qmax = 2 ** (bits - 1) - 1
    dtype = value.dtype
    flat = value.float().reshape(-1, value.shape[-1])
    current = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    if previous_maximum is not None and previous_maximum.shape == current.shape:
        maximum = current.lerp(previous_maximum, momentum)
    else:
        maximum = current
    scale = _safe_scale(maximum.mul(clipping_ratio) / qmax)
    quant = torch.round(flat / scale).clamp(-qmax, qmax)
    return (quant * scale).to(dtype).reshape_as(value), current.detach()


def _hadamard_group(width: int, maximum: int = 128) -> int:
    group = min(maximum, width)
    while group > 1 and (group & (group - 1) or width % group):
        group //= 2
    return max(group, 1)


@torch.inference_mode()
def block_hadamard(value: torch.Tensor, group: int) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform over contiguous channel groups."""
    if group <= 1:
        return value
    shape = value.shape
    work = value.reshape(-1, shape[-1] // group, group)
    step = 1
    while step < group:
        base_shape = work.shape
        paired = work.reshape(
            *base_shape[:-1], group // (2 * step), 2, step
        )
        left = paired[..., 0, :]
        right = paired[..., 1, :]
        work = torch.cat((left + right, left - right), dim=-1).reshape(base_shape)
        step *= 2
    return (work / math.sqrt(group)).reshape(shape)


@torch.inference_mode()
def minmax_weight_qdq(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Official ViDiT-Q-style per-output-channel asymmetric weight Q/DQ."""
    levels = 2**bits - 1
    work = weight.float()
    low = torch.minimum(work.amin(1, keepdim=True), torch.zeros(1, device=work.device))
    high = torch.maximum(work.amax(1, keepdim=True), torch.zeros(1, device=work.device))
    scale = _safe_scale((high - low) / levels)
    zero = torch.round(-low / scale).clamp(0, levels)
    quant = torch.round(work / scale + zero).clamp(0, levels)
    return (quant - zero) * scale


@torch.inference_mode()
def orbit_codebook_weight_refine(
    rotated_weight: torch.Tensor,
    bits: int,
    *,
    calibration_samples: torch.Tensor,
    propagation_state_weights: torch.Tensor | None,
    risk_lambda: float,
    risk_tail_fraction: float,
    trajectory_delta_weight: float = 0.0,
    trajectory_trust_ratio: float = -1.0,
    trajectory_segment_calls: int = 0,
    absolute_propagation_risk: float = 1.0,
    module_name: str = "",
    row_chunk: int = 256,
) -> torch.Tensor:
    """Select each row radius on a multiplicative grid using propagated response risk."""
    if trajectory_delta_weight != 0.0 or trajectory_trust_ratio >= 0.0:
        raise ValueError("Temporal/Pareto radius objectives are not part of this release")
    from orbitquant.codebooks import get_codebook

    codebook = get_codebook(rotated_weight.shape[1], bits, 2)
    source = rotated_weight.float()
    row_norms = source.norm(dim=1, keepdim=True).clamp_min(1e-8)
    directions = codebook.quantize(source / row_norms)
    base = directions * row_norms
    output = torch.empty_like(source)
    span = max(0.0, float(os.environ.get("PULSE_ORBIT_RADIAL_SPAN", ".10")))
    levels = max(3, int(os.environ.get("PULSE_ORBIT_RADIAL_LEVELS", "5")))
    if levels % 2 == 0:
        levels += 1
    gains = torch.linspace(1.0 - span, 1.0 + span, steps=levels,
                           device=source.device, dtype=torch.float32)
    for start in range(0, source.shape[0], row_chunk):
        stop = min(start + row_chunk, source.shape[0])
        target = source[start:stop]
        best = base[start:stop].clone()
        best_error = torch.full((stop - start,), float("inf"),
                                device=source.device, dtype=torch.float32)
        for gain in gains:
            trial = directions[start:stop] * (row_norms[start:stop] * gain)
            projected = torch.matmul(
                calibration_samples.float(), (target - trial).transpose(0, 1))
            error = _propagation_weighted_risk(
                projected.square().mean(1), propagation_state_weights,
                risk_lambda, risk_tail_fraction)
            take = error < best_error
            best_error = torch.where(take, error, best_error)
            best[take] = trial[take]
        output[start:stop] = best
    return output


@torch.inference_mode()
def joint_walsh2_code_correction(
    source: torch.Tensor,
    base_weight: torch.Tensor,
    codebook,
    axes: torch.Tensor,
    *,
    group_size: int = 128,
    passes: int = 1,
    weight_trust: float = 0.25,
    row_chunk: int = 64,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Jointly repair two protected responses using only legal W4 codes.

    The original Walsh2 implementation repaired one moment and then the other,
    so its second greedy pass could undo the first.  Here every adjacent-code
    proposal is accepted only when the *joint* two-mode objective decreases.
    The row radius remains frozen, hence the result stays in the same W4
    spherical codebook and adds no inference-time operation.
    """
    source = source.float()
    base_weight = base_weight.float()
    width = source.shape[1]
    group = min(max(1, group_size), width)
    while width % group:
        group //= 2
    axes = axes.float()
    if axes.ndim != 2 or axes.shape[1] != width or axes.shape[0] != 2:
        raise ValueError(f"Walsh2 axes must have shape (2, {width})")
    centroids = codebook.centroids.to(device=source.device, dtype=torch.float32)

    # Recover the exact discrete directions underlying the calibrated radial
    # solution.  Alternating code assignment and scalar LS is stable because
    # the deployed representation has one positive radius per output row.
    radius = base_weight.norm(dim=1, keepdim=True).clamp_min(1e-8)
    for _ in range(4):
        indices = codebook.quantize_indices(base_weight / radius).long()
        direction = centroids[indices]
        radius = (
            (base_weight * direction).sum(1, keepdim=True)
            / direction.square().sum(1, keepdim=True).clamp_min(1e-12)
        ).clamp_min(1e-8)
    recovered = centroids[indices] * radius
    recovery_rel = (
        (base_weight - recovered).square().sum()
        / base_weight.square().sum().clamp_min(1e-30)
    ).sqrt()

    updated = recovered.clone()
    updated_indices = indices.clone()
    protected_before = source.new_zeros(())
    protected_after = source.new_zeros(())
    base_error = (source - recovered).square().sum()
    changed = 0
    for row_start in range(0, source.shape[0], row_chunk):
        row_stop = min(row_start + row_chunk, source.shape[0])
        target = source[row_start:row_stop].reshape(-1, group)
        work = updated[row_start:row_stop].reshape(-1, group).clone()
        idx = updated_indices[row_start:row_stop].reshape(-1, group).clone()
        row_radius = radius[row_start:row_stop].expand(-1, width).reshape(-1, group)
        axis_bank = axes[:, None, :].expand(
            -1, row_stop - row_start, -1
        ).reshape(2, -1, group).transpose(0, 1).contiguous()
        axis_norm = axis_bank.square().sum(2).clamp_min(1e-12)
        residual = ((target - work)[:, None, :] * axis_bank).sum(2)
        protected_before += (residual.square() / axis_norm).sum()

        for _ in range(max(1, passes)):
            choices = torch.stack(
                (
                    (idx - 1).clamp_min(0),
                    (idx + 1).clamp_max(centroids.numel() - 1),
                ),
                dim=2,
            )
            delta_options = row_radius[:, :, None] * (
                centroids[choices] - centroids[idx][:, :, None]
            )
            response_change = (
                delta_options[:, :, :, None]
                * axis_bank.transpose(1, 2)[:, :, None, :]
            )
            # response_change is [group-row, coordinate, down/up, mode].
            revised = residual[:, None, None, :] - response_change
            protected_delta = (
                (revised.square() - residual[:, None, None, :].square())
                / axis_norm[:, None, None, :]
            ).mean(3)
            old_error = work - target
            mse_delta = (
                2.0 * old_error[:, :, None] * delta_options
                + delta_options.square()
            ) / group
            individual_delta = protected_delta + weight_trust * mse_delta
            direction = individual_delta.argmin(2)
            proposal_delta = delta_options.gather(
                2, direction[:, :, None]
            ).squeeze(2)
            proposal_index = choices.gather(
                2, direction[:, :, None]
            ).squeeze(2)
            proposal_score = individual_delta.gather(
                2, direction[:, :, None]
            ).squeeze(2)
            order = proposal_score.argsort(1)
            sorted_delta = proposal_delta.gather(1, order)
            sorted_index = proposal_index.gather(1, order)
            sorted_axis = axis_bank.gather(
                2, order[:, None, :].expand(-1, 2, -1)
            ).transpose(1, 2)
            cumulative_response = (
                sorted_delta[:, :, None] * sorted_axis
            ).cumsum(1)
            cumulative_mse = (
                2.0 * old_error.gather(1, order) * sorted_delta
                + sorted_delta.square()
            ).cumsum(1) / group
            prefix_delta = (
                (
                    (residual[:, None, :] - cumulative_response).square()
                    - residual[:, None, :].square()
                )
                / axis_norm[:, None, :]
            ).mean(2) + weight_trust * cumulative_mse
            prefix_delta = torch.cat(
                (torch.zeros_like(prefix_delta[:, :1]), prefix_delta), dim=1
            )
            selected_count = prefix_delta.argmin(1)
            selected_sorted = (
                torch.arange(group, device=work.device)[None, :]
                < selected_count[:, None]
            )
            selected_mask = torch.zeros_like(selected_sorted).scatter(
                1, order, selected_sorted
            )
            selected_delta = torch.zeros_like(sorted_delta).scatter(
                1,
                order,
                torch.where(selected_sorted, sorted_delta, torch.zeros_like(sorted_delta)),
            )
            selected_indices = idx.clone().scatter(1, order, sorted_index)
            idx = torch.where(selected_mask, selected_indices, idx)
            work += selected_delta
            residual -= (
                selected_delta[:, None, :] * axis_bank
            ).sum(2)

        protected_after += (residual.square() / axis_norm).sum()
        updated[row_start:row_stop].copy_(work.reshape(row_stop - row_start, width))
        updated_indices[row_start:row_stop].copy_(idx.reshape(row_stop - row_start, width))
        changed += int(idx.ne(indices[row_start:row_stop].reshape(-1, group)).sum().cpu())

    new_error = (source - updated).square().sum()
    moments = source.shape[0] * (width // group) * 2
    return updated, {
        "changed_codes": changed,
        "changed_fraction": changed / max(1, source.numel()),
        "recovery_relative_rmse": float(recovery_rel.cpu()),
        "weight_mse_ratio": float((new_error / base_error.clamp_min(1e-30)).cpu()),
        "protected_mse_ratio": float((protected_after / protected_before.clamp_min(1e-30)).cpu()),
        # Keep additive sufficient statistics so a model-level ratio can be
        # computed without averaging heterogeneous per-layer ratios.
        "weight_error_sq_before": float(base_error.cpu()),
        "weight_error_sq_after": float(new_error.cpu()),
        "protected_error_sq_before": float(protected_before.cpu()),
        "protected_error_sq_after": float(protected_after.cpu()),
        "protected_moments": moments,
    }


@dataclass(frozen=True)
class QuantSpec:
    method: str
    weight_bits: int
    activation_bits: int
    calibration_calls: int = 0
    alpha: float = 0.5665
    hadamard_max_group: int = 128
    bgr_shrink_steps: int = 12
    bgr_refine_steps: int = 3
    bgr_max_shrink_fraction: float = 0.44
    bgr_convergence_epsilon: float = 0.0
    arq_block_size: int = 1
    risk_lambda: float = 0.75
    samples_per_call: int = 2
    covariance_shrinkage: float = 0.0
    trajectory_delta_weight: float = 1.0
    trajectory_trust_ratio: float = -1.0
    trajectory_segment_calls: int = 0
    inference_segment_calls: int = 0
    calibration_coordinates: tuple[float, ...] = ()
    inference_coordinates: tuple[float, ...] = ()
    activation_scale_momentum: float = 0.0
    activation_fixed_clipping_ratio: float = 0.0
    activation_mse_min_ratio: float = 0.75
    activation_mse_candidate_count: int = 5
    activation_error_audit: bool = False
    activation_high_bits: int = 0
    activation_high_scope: str = "none"
    risk_tail_fraction: float = 0.0
    bit_sensitivity_threshold: float = 1.5
    w4_module_names: frozenset[str] = frozenset()
    precision_profile: Mapping[str, tuple[int, int]] | None = None
    # Optional causal-intervention mode.  After calibration, quantization is
    # enabled only on these zero-based transformer calls; all other calls use
    # the preserved BF16 weight.  Wan CFG normally makes two transformer calls
    # per denoising step, so a step pulse contains ``{2*t, 2*t+1}``.
    pulse_call_indices: frozenset[int] = frozenset()
    propagation_state_weights: Mapping[str, tuple[float, ...]] | None = None
    absolute_propagation_risks: Mapping[str, float] | None = None


class WholeDiTQuantLinear(nn.Module):
    """Quantized replacement for a Wan backbone ``nn.Linear``."""

    def __init__(
        self,
        source: nn.Linear,
        spec: QuantSpec,
        module_name: str,
        *,
        selective_activation_bypass: bool = False,
    ):
        super().__init__()
        validate_release_settings()
        if spec.method in ('pulseprofile',):
            if spec.precision_profile is None or module_name not in spec.precision_profile:
                raise ValueError(f"missing TIDE precision profile entry for {module_name}")
            profiled_weight_bits, profiled_activation_bits = spec.precision_profile[module_name]
            spec = replace(
                spec,
                method="taorgwfmse",
                weight_bits=int(profiled_weight_bits),
                activation_bits=int(profiled_activation_bits),
            )
        high_precision_match = {
            "none": False,
            "cross_attn": ".attn2." in module_name,
            "cross_out": ".attn2.to_out.0" in module_name,
            "self_attn": ".attn1." in module_name,
            "attention": ".attn1." in module_name or ".attn2." in module_name,
            "ffn": ".ffn." in module_name,
            "ffn_out": ".ffn.net.2" in module_name,
            "output_projections": (
                ".attn2.to_out.0" in module_name or ".ffn.net.2" in module_name
            ),
        }.get(spec.activation_high_scope)
        if high_precision_match is None:
            raise ValueError(
                f"unsupported activation_high_scope={spec.activation_high_scope!r}"
            )
        if high_precision_match and spec.activation_high_bits > 0:
            spec = replace(spec, activation_bits=spec.activation_high_bits)
        self.spec = spec
        self.module_name = module_name
        self.selective_activation_bypass = selective_activation_bypass
        self.selective_weight_folded = False
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.group = _hadamard_group(source.in_features, spec.hadamard_max_group)
        # The reference implementation realizes a 128-point FWHT as seven
        # reshape/add/cat stages.  On large Wan token grids those launches and
        # temporaries dominate the fake-quant forward.  A tiny dense 128x128
        # Hadamard uses tensor cores and is numerically equivalent up to BF16
        # accumulation order.  Keep this deployment optimization opt-in so the
        # original calibration and audit path remains reproducible.
        self.fast_hadamard = False
        self.triton_hadamard = False
        self.compiled_qdq = False
        self.compiled_rotation = False
        # The stock RPBH path performs permutation, sign balancing and every
        # FWHT stage as separate PyTorch operations.  This opt-in path keeps
        # the exact frozen Pareto representation but dispatches the rotation
        # through one Triton kernel with device-cached constants.
        self.triton_rpbh_rotation = (
            os.environ.get("PULSE_TRITON_RPBH_ROTATION", "0") == "1"
        )
        self.precompose_rotation = (
            os.environ.get("PULSE_PRECOMPOSE_ROTATION", "0") == "1"
        )
        legacy_orbit_codebook = os.environ.get("PULSE_ORBIT_CODEBOOK", "0") == "1"
        self.orbit_weight_codebook = (
            os.environ.get(
                "PULSE_ORBIT_WEIGHT_CODEBOOK",
                "1" if legacy_orbit_codebook else "0",
            )
            == "1"
        )
        self.orbit_activation_codebook_enabled = (
            os.environ.get(
                "PULSE_ORBIT_ACTIVATION_CODEBOOK",
                "1" if legacy_orbit_codebook else "0",
            )
            == "1"
        )
        # Retain the old public/cache field for backward compatibility.  New
        # experiments persist and validate the two components independently.
        self.orbit_codebook_hybrid = (
            self.orbit_weight_codebook
            and self.orbit_activation_codebook_enabled
        )
        self.rotation_mode = os.environ.get("PULSE_ROTATION_MODE", "block_hadamard")
        if self.rotation_mode not in {"block_hadamard", "orbit_rpbh"}:
            raise ValueError(f"unsupported PULSE_ROTATION_MODE={self.rotation_mode!r}")
        if self.rotation_mode == "orbit_rpbh":
            if self.precompose_rotation:
                raise ValueError("orbit_rpbh does not support PULSE_PRECOMPOSE_ROTATION")
            from orbitquant.rotations import get_rpbh_rotation

            rpbh_block_value = os.environ.get("PULSE_RPBH_BLOCK_SIZE", "128")
            rpbh_block = "paper" if rpbh_block_value == "paper" else int(rpbh_block_value)
            self.rpbh_rotation = get_rpbh_rotation(
                source.in_features,
                seed=int(os.environ.get("PULSE_RPBH_SEED", "0")),
                block_size=rpbh_block,
            )
        else:
            self.rpbh_rotation = None
        if self.spec.method == "taorgwfmse" and not self.orbit_weight_codebook:
            raise ValueError("PulseQuant requires PULSE_ORBIT_WEIGHT_CODEBOOK=1")
        if self.orbit_weight_codebook or self.orbit_activation_codebook_enabled:
            if self.rotation_mode != "orbit_rpbh":
                raise ValueError("Orbit weight/activation codebooks require orbit_rpbh rotation")
        if self.orbit_weight_codebook or self.orbit_activation_codebook_enabled:
            from orbitquant.codebooks import get_codebook
        if self.orbit_weight_codebook:
            self.orbit_weight_codebook_object = get_codebook(
                source.in_features, spec.weight_bits, 2
            )
        else:
            self.orbit_weight_codebook_object = None
        if self.orbit_activation_codebook_enabled:
            self.orbit_activation_codebook = get_codebook(
                source.in_features, spec.activation_bits, 2
            )
        else:
            self.orbit_activation_codebook = None
        self.activation_scale_reuse_interval = max(
            1, int("1")
        )
        self.register_buffer(
            "runtime_hadamard_matrix",
            (
                block_hadamard(
                    torch.eye(
                        self.group,
                        device=source.weight.device,
                        dtype=source.weight.dtype,
                    ),
                    self.group,
                )
                if self.fast_hadamard
                else None
            ),
        )
        self.register_buffer(
            "runtime_activation_qmax",
            torch.tensor(
                float(2 ** (spec.activation_bits - 1) - 1),
                device=source.weight.device,
                dtype=torch.float32,
            ),
        )
        self.calls = 0
        self.finalized = False
        # Calibration-time input traffic is kept as Python integers so the
        # mixed-precision allocator can weight activation bits by the actual
        # tensors being quantized.  ``in_features`` alone cannot distinguish
        # long video-token paths from short text K/V paths.
        self.calibration_activation_elements = 0
        self.calibration_activation_calls = 0
        self.candidate_risks: list[dict[str, object]] = []
        self.path_coupling_metadata: dict[str, object] = {}
        self._path_coupling_left_weight: torch.Tensor | None = None
        self._path_coupling_role = ""
        self._activation_scale_history: list[torch.Tensor | None] = [None, None]
        self.register_buffer("weight", source.weight.detach().clone())
        self.register_buffer(
            "fp_weight",
            source.weight.detach().clone() if spec.pulse_call_indices else None,
        )
        profile_state_weights = (
            None
            if spec.propagation_state_weights is None
            else spec.propagation_state_weights.get(module_name)
        )
        self.register_buffer(
            "propagation_state_weight",
            None
            if profile_state_weights is None
            else torch.tensor(
                profile_state_weights,
                device=source.weight.device,
                dtype=torch.float32,
            ),
        )
        absolute_risk = (
            1.0
            if spec.absolute_propagation_risks is None
            else float(spec.absolute_propagation_risks.get(module_name, 1.0))
        )
        self.register_buffer(
            "absolute_propagation_risk",
            torch.tensor(absolute_risk, device=source.weight.device, dtype=torch.float32),
        )
        self.register_buffer(
            "bias", None if source.bias is None else source.bias.detach().clone()
        )
        self.register_buffer("channel_mask", None)
        self.register_buffer("weight_error_sq", torch.zeros((), device=source.weight.device))
        self.register_buffer("weight_signal_sq", torch.zeros((), device=source.weight.device))
        self.register_buffer("activation_error_sq", torch.zeros((), device=source.weight.device))
        self.register_buffer("activation_signal_sq", torch.zeros((), device=source.weight.device))
        self.register_buffer("output_error_sq", torch.zeros((), device=source.weight.device))
        self.register_buffer("output_signal_sq", torch.zeros((), device=source.weight.device))
        self.register_buffer("importance_cv", torch.full((), float("nan"), device=source.weight.device))
        self.register_buffer("importance_tail_ratio", torch.full((), float("nan"), device=source.weight.device))
        self.register_buffer("bit_sensitivity_ratio", torch.full((), float("nan"), device=source.weight.device))
        self.register_buffer(
            "trajectory_delta_normalizer",
            torch.ones((), device=source.weight.device, dtype=torch.float32),
        )
        self.register_buffer(
            "activation_granularity", torch.zeros((), device=source.weight.device, dtype=torch.int8)
        )
        self.register_buffer(
            "activation_bulk_ratio", torch.zeros((), device=source.weight.device, dtype=torch.float32)
        )
        self.register_buffer(
            "activation_bulk_schedule",
            torch.zeros(10, device=source.weight.device, dtype=torch.float32),
        )
        # Allow a cheap post-calibration sensitivity split.  Attention and FFN
        # see materially different distributions after RPBH, so forcing one
        # global A4 range can make the locally best compromise suboptimal for
        # the denoising trajectory.  Class-specific values fall back to the
        # global schedule and remain runtime-only (the frozen weights/cache do
        # not need to be rebuilt for these sweeps).
        override_text = "".strip()
        if ".attn" in module_name:
            override_text = override_text.strip()
        elif ".ffn." in module_name:
            override_text = override_text.strip()
        if override_text:
            override_values = [float(item) for item in override_text.split(",")]
            if len(override_values) != 10 or any(
                not (0.0 < item <= 1.0) for item in override_values
            ):
                raise ValueError(
                    "PULSE_ACTIVATION_CLIP_SCHEDULE requires 10 comma-separated "
                    "ratios in (0, 1]"
                )
            override_schedule = torch.tensor(
                override_values, device=source.weight.device, dtype=torch.float32
            )
        else:
            override_schedule = None
        self.register_buffer("activation_override_schedule", override_schedule)
        self.register_buffer(
            "propagation_clipping_weights",
            torch.ones(10, device=source.weight.device, dtype=torch.float32),
        )
        self.register_buffer(
            "linear_gain_schedule",
            torch.ones(10, device=source.weight.device, dtype=torch.float32),
        )
        self.register_buffer(
            "activation_candidate_scores",
            torch.full((10, 5), float("nan"), device=source.weight.device, dtype=torch.float32),
        )
        # Offline residual-aware mixed-activation schedule.  The default stays
        # at the requested activation precision; after calibration a global
        # allocator may promote only the highest-risk Linear x time bins to A8.
        self.register_buffer(
            "activation_bit_schedule",
            torch.full((10,), spec.activation_bits, device=source.weight.device, dtype=torch.int8),
        )
        self.register_buffer(
            "activation_qmax_schedule",
            torch.full(
                (10,),
                float(2 ** (spec.activation_bits - 1) - 1),
                device=source.weight.device,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "activation_rescue_score",
            torch.full((10,), float("-inf"), device=source.weight.device, dtype=torch.float32),
        )
        self.register_buffer(
            "activation_a8_ratio_schedule",
            torch.ones(10, device=source.weight.device, dtype=torch.float32),
        )
        calibration_methods = {"taorgwfmse"}
        if spec.method in calibration_methods:
            if spec.calibration_calls <= 0:
                raise ValueError(f"{spec.method} requires positive calibration_calls")
            self.register_buffer(
                "act_max",
                torch.zeros(source.in_features, device=source.weight.device, dtype=torch.float32),
            )
            self.register_buffer(
                "act_energy_sum",
                torch.zeros(source.in_features, device=source.weight.device, dtype=torch.float32),
            )
            self.register_buffer(
                "act_energy_max",
                torch.zeros(source.in_features, device=source.weight.device, dtype=torch.float32),
            )
            self.energy_calls = 0
            self.calibration_samples: list[torch.Tensor] = []
        else:
            self.register_buffer("act_max", None)
            self.register_buffer("act_energy_sum", None)
            self.register_buffer("act_energy_max", None)
            self.energy_calls = 0
            self.calibration_samples = []
            self._finalize()

    @torch.inference_mode()
    def _finalize(self) -> None:
        work = self.weight
        if self.spec.method in ('taorgwfmse',):
            self.channel_mask = torch.ones_like(self.act_max, dtype=work.dtype)
            work = work / self.channel_mask[None, :]
            unrotated_samples = torch.stack(self.calibration_samples, dim=0)
            unrotated_samples = (
                unrotated_samples * self.channel_mask.float()[None, None, :]
            )
            adaptive_seed_text = "".strip()
            work = self._offline_rotation(work)
            target = work.float()
            samples = self._offline_rotation(unrotated_samples)
            quantized_samples = None
            local_risk_lambda = self.spec.risk_lambda
            reference: torch.Tensor | None = None
            effective_propagation = self.propagation_state_weight
            linear_beta = float("0")
            selective_threshold = float(
                "0"
            )
            objective_samples = samples
            adversarial_epsilon = float(
                "0"
            )
            temporal_delta_weight = float(
                os.environ.get("PULSE_TEMPORAL_DELTA_WEIGHT", "0")
            )
            two_stage_trust_ratio = float(
                "-1"
            )
            absolute_risk = float(self.absolute_propagation_risk.float().cpu())
            search_threshold = float(
                "0"
            )
            bgr_shrink_steps = self.spec.bgr_shrink_steps
            work = orbit_codebook_weight_refine(
                work,
                self.spec.weight_bits,
                calibration_samples=objective_samples,
                propagation_state_weights=effective_propagation,
                risk_lambda=local_risk_lambda,
                risk_tail_fraction=self.spec.risk_tail_fraction,
                trajectory_delta_weight=temporal_delta_weight,
                trajectory_trust_ratio=two_stage_trust_ratio,
                trajectory_segment_calls=self.spec.trajectory_segment_calls,
                absolute_propagation_risk=absolute_risk,
                module_name=self.module_name,
            )
            if (
                self.orbit_weight_codebook
                and os.environ.get("PULSE_WALSH2_CORRECTION", "0") == "1"
            ):
                axis_mode = "activation_pca".lower()
                width = target.shape[1]
                flat_samples = samples.float().reshape(-1, width)
                maximum = max(
                    32, int(os.environ.get("PULSE_WALSH2_PCA_SAMPLES", "512"))
                )
                if flat_samples.shape[0] > maximum:
                    chosen = torch.linspace(
                        0,
                        flat_samples.shape[0] - 1,
                        maximum,
                        device=flat_samples.device,
                    ).round().long()
                    flat_samples = flat_samples[chosen]
                coordinate = torch.arange(width, device=work.device)
                starts = (
                    torch.ones(width, device=work.device),
                    1.0 - 2.0 * coordinate.bitwise_and(1).float(),
                )
                axes_list = []
                iterations = max(
                        1,
                        int(
                            os.environ.get(
                                "PULSE_WALSH2_PCA_ITERS", "2"
                            )
                        ),
                    )
                for start in starts:
                    vector = start / start.norm().clamp_min(1e-12)
                    for prior in axes_list:
                        vector -= (vector * prior).sum() * prior
                    vector /= vector.norm().clamp_min(1e-12)
                    for _ in range(iterations):
                        vector = flat_samples.transpose(0, 1).matmul(
                            flat_samples.matmul(vector)
                        )
                        for prior in axes_list:
                            vector -= (vector * prior).sum() * prior
                        vector /= vector.norm().clamp_min(1e-12)
                    axes_list.append(vector)
                axes = torch.stack(axes_list)
                work, walsh2_audit = joint_walsh2_code_correction(
                    target,
                    work,
                    self.orbit_weight_codebook_object,
                    axes,
                    group_size=int(
                        os.environ.get("PULSE_WALSH2_GROUP", "128")
                    ),
                    passes=int(os.environ.get("PULSE_WALSH2_PASSES", "1")),
                    weight_trust=float(
                        os.environ.get("PULSE_WALSH2_WEIGHT_TRUST", ".25")
                    ),
                )
                self.path_coupling_metadata.update(
                    {
                        "walsh2_axis_mode": axis_mode,
                        **{},
                        **walsh2_audit,
                    }
                )
            if (
                self.spec.method == "taorgwfmse"
                and self.spec.activation_fixed_clipping_ratio > 0.0
            ):
                # A fixed ratio is an explicit request to bypass activation
                # FMSE calibration.  Previously we still evaluated all five
                # candidates for every linear layer and only overwrote the
                # result afterwards, so the option saved no finalize time.
                self.activation_bulk_schedule.fill_(
                    self.spec.activation_fixed_clipping_ratio
                )
            if (
                self.spec.method == "taorgwfmse"
                and self.spec.activation_fixed_clipping_ratio <= 0.0
            ):
                if reference is None:
                    reference = torch.matmul(samples.float(), target.transpose(0, 1))
                candidate_ratio_text = "".strip()
                candidate_ratios = (1.0, 0.95, 0.90, 0.85, 0.75)
                if len(candidate_ratios) != self.activation_candidate_scores.shape[1] or any(
                    not (0.0 < ratio <= 1.0) for ratio in candidate_ratios
                ):
                    raise ValueError(
                        "PULSE_FMSE_CANDIDATE_RATIOS must contain exactly five "
                        "comma-separated ratios in (0, 1]"
                    )
                step_errors = []
                temporal_step_errors = []
                frame_step_errors = []
                frame_second_step_errors = []
                latent_frames = int(os.environ.get("PULSE_VIDEO_LATENT_FRAMES", "0"))
                frame_tube_share = float("0")
                frame_scale_smoothing = float("0")
                frame_scale_second = float("0")
                frame_motion_tau = float("0")
                frame_moment_strength = float("0")
                frame_moment_group = int("128")
                for ratio in candidate_ratios:
                    if (
                        self.spec.activation_scale_momentum > 0.0
                        or False
                        or False
                        or False
                        or False
                    ):
                        histories: list[torch.Tensor | None] = [None, None]
                        quantized_calls = []
                        period = max(2, self.spec.trajectory_segment_calls)
                        for call_index, sample in enumerate(samples):
                            if call_index % period == 0:
                                histories = [None, None]
                            branch = call_index % 2
                            quantized, current_maximum = frame_consistent_clipped_token_qdq(
                                sample,
                                self.spec.activation_bits,
                                histories[branch],
                                self.spec.activation_scale_momentum,
                                ratio,
                                latent_frames,
                                frame_tube_share,
                                frame_scale_smoothing,
                                frame_scale_second,
                                frame_motion_tau,
                                frame_moment_strength,
                                frame_moment_group,
                            )
                            histories[branch] = current_maximum
                            quantized_calls.append(quantized)
                        quantized = torch.stack(quantized_calls, dim=0)
                    else:
                        quantized = clipped_token_qdq(
                            samples, self.spec.activation_bits, ratio
                        )
                    prediction = torch.matmul(
                        quantized.float(), work.float().transpose(0, 1)
                    )
                    residual = reference - prediction
                    step_errors.append(residual.square().mean((1, 2)))
                    if (
                        latent_frames > 1
                        and residual.shape[1] % latent_frames == 0
                        and residual.shape[1] // latent_frames >= 1
                    ):
                        frame_residual = residual.reshape(
                            residual.shape[0], latent_frames, -1, residual.shape[-1]
                        )
                        frame_delta = frame_residual[:, 1:] - frame_residual[:, :-1]
                        frame_step_errors.append(frame_delta.square().mean((1, 2, 3)))
                        if latent_frames >= 3:
                            frame_second = frame_delta[:, 1:] - frame_delta[:, :-1]
                            frame_second_step_errors.append(
                                frame_second.square().mean((1, 2, 3))
                            )
                        else:
                            frame_second_step_errors.append(
                                torch.zeros(residual.shape[0], device=residual.device)
                            )
                    else:
                        frame_step_errors.append(
                            torch.zeros(residual.shape[0], device=residual.device)
                        )
                        frame_second_step_errors.append(
                            torch.zeros(residual.shape[0], device=residual.device)
                        )
                    temporal_segments = []
                    period = max(2, self.spec.trajectory_segment_calls)
                    steps = period // 2
                    for start in range(0, residual.shape[0], period):
                        segment = residual[start : start + period]
                        if segment.shape[0] != period:
                            continue
                        if period % 2 == 0:
                            paired = segment.reshape(steps, 2, *segment.shape[1:])
                            delta = (paired[1:] - paired[:-1]).square().mean((2, 3))
                            # Wan calibration alternates CFG branches.  Compare
                            # adjacent scheduler states within each branch.
                            delta = torch.cat((delta[:1], delta), dim=0)
                        else:
                            # Self-Forcing has one call per AR state (35 calls),
                            # not interleaved CFG pairs.  Preserve all state
                            # coordinates by differencing consecutive calls.
                            delta = (segment[1:] - segment[:-1]).square().mean((1, 2))
                            delta = torch.cat((delta[:1], delta), dim=0)
                        temporal_segments.append(delta.reshape(-1))
                    if temporal_segments:
                        temporal_step_errors.append(torch.cat(temporal_segments))
                    else:
                        # Clean-cached cross-attention K/V can be observed only
                        # once per prompt, shorter than the 35-state AR period.
                        # It has no within-trajectory temporal increment.
                        temporal_step_errors.append(
                            torch.zeros(
                                residual.shape[0],
                                device=residual.device,
                                dtype=residual.dtype,
                            )
                        )
                error_matrix = torch.stack(step_errors)
                temporal_error_matrix = torch.stack(temporal_step_errors)
                frame_error_matrix = torch.stack(frame_step_errors)
                frame_second_error_matrix = torch.stack(frame_second_step_errors)
                calls = error_matrix.shape[1]
                bins = self.activation_bulk_schedule.numel()
                scheduler_align = os.environ.get("PULSE_PROP_SCHEDULER_ALIGN", "0") == "1"
                clip_tail = 0.0
                if scheduler_align:
                    scores = _scheduler_aligned_candidate_scores(
                        error_matrix,
                        self.spec.trajectory_segment_calls,
                        bins,
                        self.spec.calibration_coordinates,
                        self.spec.inference_coordinates,
                        self.spec.risk_lambda,
                        clip_tail,
                    )
                    temporal_scores = _scheduler_aligned_candidate_scores(
                        temporal_error_matrix,
                        self.spec.trajectory_segment_calls,
                        bins,
                        self.spec.calibration_coordinates,
                        self.spec.inference_coordinates,
                        self.spec.risk_lambda,
                        clip_tail,
                    )
                    frame_scores = _scheduler_aligned_candidate_scores(
                        frame_error_matrix,
                        self.spec.trajectory_segment_calls,
                        bins,
                        self.spec.calibration_coordinates,
                        self.spec.inference_coordinates,
                        self.spec.risk_lambda,
                        clip_tail,
                    )
                    frame_second_scores = _scheduler_aligned_candidate_scores(
                        frame_second_error_matrix,
                        self.spec.trajectory_segment_calls,
                        bins,
                        self.spec.calibration_coordinates,
                        self.spec.inference_coordinates,
                        self.spec.risk_lambda,
                        clip_tail,
                    )
                else:
                    score_rows = []
                    for bin_index in range(bins):
                        indices = _phase_aligned_bin_indices(
                            calls,
                            self.spec.trajectory_segment_calls,
                            bins,
                            bin_index,
                            device=error_matrix.device,
                        )
                        segment = error_matrix[:, indices]
                        score_rows.append(
                            _mean_tail_risk(
                                segment.transpose(0, 1),
                                self.spec.risk_lambda,
                                clip_tail,
                            )
                        )
                    scores = torch.stack(score_rows, dim=0)
                    temporal_scores = scores.new_zeros(scores.shape)
                    frame_score_rows = []
                    frame_second_score_rows = []
                    for bin_index in range(bins):
                        indices = _phase_aligned_bin_indices(
                            calls,
                            self.spec.trajectory_segment_calls,
                            bins,
                            bin_index,
                            device=error_matrix.device,
                        )
                        frame_score_rows.append(
                            _mean_tail_risk(
                                frame_error_matrix[:, indices].transpose(0, 1),
                                self.spec.risk_lambda,
                                clip_tail,
                            )
                        )
                        frame_second_score_rows.append(
                            _mean_tail_risk(
                                frame_second_error_matrix[:, indices].transpose(0, 1),
                                self.spec.risk_lambda,
                                clip_tail,
                            )
                        )
                    frame_scores = torch.stack(frame_score_rows, dim=0)
                    frame_second_scores = torch.stack(frame_second_score_rows, dim=0)

                state_scores = scores.clone()

                temporal_weight = float(
                    "0"
                )

                frame_weight = float("0")
                frame_second_weight = float(
                    "0"
                )
                frame_state_trust = float(
                    "-1"
                )

                propagation = _resample_propagation_weights(
                    self.propagation_state_weight,
                    bins,
                    self.spec.calibration_coordinates,
                    self.spec.inference_coordinates,
                    device=scores.device,
                )
                prop_beta = float("1")
                propagation = torch.exp(
                    prop_beta * (propagation.log() - propagation.log().mean())
                )
                linear_beta = float("0")
                linear_gain = torch.ones_like(propagation)
                propagation = propagation / propagation.mean().clamp_min(1e-12)
                smooth_penalty = float("0")
                selected, normalized_cost = _smooth_prop_clipping_schedule(
                    scores,
                    propagation,
                    candidate_ratios,
                    smooth_penalty,
                )
                ratios = torch.tensor(candidate_ratios, device=scores.device)
                self.activation_bulk_schedule.copy_(ratios[selected])
                self.propagation_clipping_weights.copy_(propagation)
                self.linear_gain_schedule.copy_(linear_gain)
                self.activation_candidate_scores.copy_(scores.float())
                if float(os.environ.get("PULSE_RESIDUAL_A8_FRACTION", "0")) > 0.0:
                    # Compare A6 and A8 at the *same* selected clipping ratio,
                    # so the score measures only the value of extra activation
                    # precision rather than silently changing two policies.
                    a8_step_errors = []
                    for ratio in candidate_ratios:
                        quantized8 = clipped_token_qdq(samples, 8, ratio)
                        prediction8 = torch.matmul(
                            quantized8.float(), work.float().transpose(0, 1)
                        )
                        a8_step_errors.append(
                            (reference - prediction8).square().mean((1, 2))
                        )
                    a8_matrix = torch.stack(a8_step_errors)
                    reference_energy = reference.square().mean((1, 2))[None, :]
                    if scheduler_align:
                        scores8 = _scheduler_aligned_candidate_scores(
                            a8_matrix,
                            self.spec.trajectory_segment_calls,
                            bins,
                            self.spec.calibration_coordinates,
                            self.spec.inference_coordinates,
                            self.spec.risk_lambda,
                            clip_tail,
                        )
                        signal = _scheduler_aligned_candidate_scores(
                            reference_energy,
                            self.spec.trajectory_segment_calls,
                            bins,
                            self.spec.calibration_coordinates,
                            self.spec.inference_coordinates,
                            0.0,
                            0.0,
                        ).squeeze(1)
                    else:
                        score8_rows = []
                        signal_rows = []
                        for bin_index in range(bins):
                            indices = _phase_aligned_bin_indices(
                                calls,
                                self.spec.trajectory_segment_calls,
                                bins,
                                bin_index,
                                device=error_matrix.device,
                            )
                            score8_rows.append(
                                _mean_tail_risk(
                                    a8_matrix[:, indices].transpose(0, 1),
                                    self.spec.risk_lambda,
                                    clip_tail,
                                )
                            )
                            signal_rows.append(reference_energy[0, indices].mean())
                        scores8 = torch.stack(score8_rows)
                        signal = torch.stack(signal_rows)
                    chosen6 = scores.gather(1, selected[:, None]).squeeze(1)
                    selected8 = selected
                    chosen8 = scores8.gather(1, selected8[:, None]).squeeze(1)
                    self.activation_a8_ratio_schedule.copy_(ratios[selected8])
                    relative_gain = (chosen6 - chosen8).clamp_min(0.0) / signal.clamp_min(1e-12)
                    self.activation_rescue_score.copy_((relative_gain * propagation).float())
            self.calibration_samples.clear()
        elif self.spec.method == "minmax":
            target = work.float()
            work = minmax_weight_qdq(work, self.spec.weight_bits)
        else:
            raise ValueError(f"unknown whole-DiT quant method: {self.spec.method}")
        dequant = work.float()
        self.weight_error_sq.copy_((target - dequant).square().sum())
        self.weight_signal_sq.copy_(target.square().sum())
        # Deployment-only approximation: move the orthogonal transform to the
        # frozen weight and quantize activations in the original channel basis.
        # This removes one online Hadamard per Linear.  It is intentionally
        # opt-in because Q(Hx) and H Q(x) are not algebraically identical.
        rotated_methods = {"taorgwfmse"}
        if self.precompose_rotation and self.spec.method in rotated_methods:
            work = block_hadamard(work.to(self.weight.dtype), self.group)
        self.weight = work.to(self.weight.dtype)
        self.finalized = True

    @torch.inference_mode()
    def _offline_rotation(self, value: torch.Tensor) -> torch.Tensor:
        if self.rpbh_rotation is not None:
            return self.rpbh_rotation.apply_to_activations(value.float()).to(value.dtype)
        return block_hadamard(value, self.group)

    @torch.inference_mode()
    def _runtime_hadamard(self, value: torch.Tensor) -> torch.Tensor:
        if self.rpbh_rotation is not None:
            return self.rpbh_rotation.apply_to_activations(value.float()).to(value.dtype)
        if not self.fast_hadamard:
            return block_hadamard(value, self.group)
        shape = value.shape
        work = value.reshape(-1, shape[-1] // self.group, self.group)
        return torch.matmul(work, self.runtime_hadamard_matrix).reshape(shape)

    @torch.inference_mode()
    def _fold_selective_bypass_weight(self) -> None:
        """Fold balance+Hadamard into W4 BF16 weight for exact A16 bypass.

        With activation Q/DQ disabled, ``linear(H(x*m), W)`` is algebraically
        identical to ``linear(x, (W H)*m)``.  This removes mask, rotation, and
        Q/DQ from low-propagation-risk modules without a custom kernel.
        """
        if not self.selective_activation_bypass or self.selective_weight_folded:
            return
        if self.rpbh_rotation is not None:
            # Runtime computes (x * mask) @ R against a weight represented in
            # the rotated basis.  Folding back to an ordinary Linear therefore
            # requires W_rot @ R.T, i.e. the explicit inverse RPBH transform.
            # A plain block Hadamard is only valid for the non-permuted,
            # self-inverse rotation and silently corrupts RPBH weights.
            folded = self.rpbh_rotation.apply_inverse_to_weight(self.weight.float())
            folded = folded.to(self.weight.dtype)
        else:
            folded = block_hadamard(self.weight, self.group)
        folded = folded * self.channel_mask.to(folded.dtype)
        self.weight = folded
        self.selective_weight_folded = True

    @torch.compiler.disable
    @torch.inference_mode()
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if not self.finalized:
            self.calibration_activation_elements += int(value.numel())
            self.calibration_activation_calls += 1
            flat = value.reshape(-1, value.shape[-1])
            self.act_max.copy_(torch.maximum(self.act_max, flat.float().abs().amax(0)))
            if self.spec.method in ('taorgwfmse',):
                count = min(self.spec.samples_per_call, flat.shape[0])
                h3_indices = getattr(self, "h3_sample_indices", None)
                if h3_indices is not None and value.ndim == 3:
                    self.calibration_samples.append(value[0, h3_indices.to(value.device)].detach().clone())
                    output = F.linear(value, self.weight, self.bias)
                    if self.calls >= self.spec.calibration_calls:
                        self._finalize()
                    return output
                # H3 has a few conditioning projections whose flattened
                # token count varies between denoising calls (sometimes
                # one token, sometimes two).  Keep the calibration tensor
                # rectangular by sampling the requested count with
                # deterministic repetition when the input is shorter.
                count = max(1, self.spec.samples_per_call)
                indices = torch.linspace(
                    0, flat.shape[0] - 1, count, device=flat.device
                ).round().long()
                self.calibration_samples.append(flat[indices].detach().clone())
            output = F.linear(value, self.weight, self.bias)
            if self.calls >= self.spec.calibration_calls:
                self._finalize()
            return output

        if self.spec.pulse_call_indices:
            post_calibration_call = self.calls - self.spec.calibration_calls - 1
            if post_calibration_call not in self.spec.pulse_call_indices:
                return F.linear(value, self.fp_weight, self.bias)

        if self.selective_activation_bypass:
            self._fold_selective_bypass_weight()
            return F.linear(value, self.weight, self.bias)
        activation_reference = None
        if self.spec.method in ('taorgwfmse',):
            runtime_mask = self.channel_mask.to(value.dtype)
            if (
                not self.orbit_activation_codebook_enabled
                and
                self.compiled_rotation
                and self.fast_hadamard
                and not self.precompose_rotation
                and self.rpbh_rotation is None
            ):
                quant_input = compiled_masked_hadamard(
                    value, runtime_mask, self.runtime_hadamard_matrix
                )
            else:
                quant_input = value * runtime_mask
                if not self.precompose_rotation and not self.orbit_activation_codebook_enabled:
                    quant_input = self._runtime_hadamard(quant_input)
            activation_reference = quant_input
            if self.orbit_activation_codebook_enabled:
                from orbitquant.kernels import quantize_activations_kernel

                quant_input = quantize_activations_kernel(
                    quant_input,
                    rotation=self.rpbh_rotation,
                    codebook=self.orbit_activation_codebook,
                    eps=1e-8,
                    backend="auto",
                )
            elif self.spec.method == "taorgwfmse":
                period = max(
                    1,
                    self.spec.inference_segment_calls
                    or self.spec.trajectory_segment_calls,
                )
                post_call = self.calls - self.spec.calibration_calls - 1
                phase = post_call % period
                bin_index = min(
                    self.activation_bulk_schedule.numel() - 1,
                    phase * self.activation_bulk_schedule.numel() // period,
                )
                # Keep the frozen schedule value on-device.  Calling ``.item``
                # here serialized the CPU with CUDA once per quantized Linear
                # (300/400 modules × 100 transformer calls per video).
                # A fixed ratio is also useful for post-calibration A/B sweeps:
                # cached FMSE schedules were historically searched with an A6-
                # oriented range, which is often too conservative for A4.
                ratio = (
                    quant_input.new_tensor(self.spec.activation_fixed_clipping_ratio)
                    if self.spec.activation_fixed_clipping_ratio > 0.0
                    else self.activation_override_schedule[bin_index]
                    if self.activation_override_schedule is not None
                    else self.activation_bulk_schedule[bin_index]
                )
                qmax = self.activation_qmax_schedule[bin_index]
                if (
                    float("0") > 0.0
                    or float("0") > 0.0
                    or float("0") > 0.0
                    or float("0") > 0.0
                ):
                    if post_call % period == 0:
                        self._activation_scale_history = [None, None]
                    branch = post_call % 2
                    quant_input, current_maximum = frame_consistent_clipped_token_qdq(
                        quant_input,
                        self.spec.activation_bits,
                        self._activation_scale_history[branch],
                        self.spec.activation_scale_momentum,
                        ratio,
                        int(os.environ.get("PULSE_VIDEO_LATENT_FRAMES", "0")),
                        float("0"),
                        float("0"),
                        float("0"),
                        float("0"),
                        float("0"),
                        int("128"),
                    )
                    self._activation_scale_history[branch] = current_maximum
                elif self.activation_scale_reuse_interval > 1:
                    if post_call % period == 0:
                        self._activation_scale_history = [None, None]
                    branch = post_call % 2
                    prior = self._activation_scale_history[branch]
                    flat = quant_input.float().reshape(-1, quant_input.shape[-1])
                    same_branch_step = post_call // 2
                    refresh = (
                        prior is None
                        or prior.shape[0] != flat.shape[0]
                        or same_branch_step % self.activation_scale_reuse_interval == 0
                    )
                    if refresh:
                        prior = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
                        self._activation_scale_history[branch] = prior.detach()
                    if self.compiled_qdq:
                        quant_input = compiled_cached_scale_qdq(
                            quant_input,
                            prior,
                            ratio,
                            qmax,
                        )
                    else:
                        scale = (
                            prior * ratio / qmax
                        ).clamp_min(1e-8)
                        quant = torch.round(flat / scale).clamp(
                            -qmax,
                            qmax,
                        )
                        quant_input = (quant * scale).to(
                            quant_input.dtype
                        ).reshape_as(quant_input)
                elif self.spec.activation_scale_momentum > 0.0:
                    if post_call % period == 0:
                        self._activation_scale_history = [None, None]
                    branch = post_call % 2
                    quant_input, current_maximum = (
                        trajectory_anchored_clipped_token_qdq(
                            quant_input,
                            self.spec.activation_bits,
                            self._activation_scale_history[branch],
                            self.spec.activation_scale_momentum,
                            ratio,
                        )
                    )
                    self._activation_scale_history[branch] = current_maximum
                else:
                    if self.compiled_qdq:
                        quant_input = compiled_clipped_token_qdq(
                            quant_input,
                            ratio,
                            qmax,
                        )
                    else:
                        flat = quant_input.float().reshape(-1, quant_input.shape[-1])
                        scale = (flat.abs().amax(dim=1, keepdim=True) * ratio / qmax).clamp_min(1e-8)
                        quant_input = (
                            torch.round(flat / scale).clamp(-qmax, qmax) * scale
                        ).to(quant_input.dtype).reshape_as(quant_input)
            else:
                quant_input = token_symmetric_qdq(quant_input, self.spec.activation_bits)
        else:
            activation_reference = value
            quant_input = token_symmetric_qdq(value, self.spec.activation_bits)
        if self.spec.activation_error_audit and activation_reference is not None:
            reference_float = activation_reference.float()
            quant_float = quant_input.float()
            self.activation_signal_sq.add_(reference_float.square().sum())
            self.activation_error_sq.add_((reference_float - quant_float).square().sum())
        quant_output = F.linear(quant_input, self.weight, self.bias)
        if self.spec.pulse_call_indices:
            reference_output = F.linear(value, self.fp_weight, self.bias)
            self.output_signal_sq.add_(reference_output.float().square().sum())
            self.output_error_sq.add_(
                (reference_output.float() - quant_output.float()).square().sum()
            )
        return quant_output


def _wan_main_linear_names(transformer: nn.Module) -> list[str]:
    names: list[str] = []
    for block_index, block in enumerate(transformer.blocks):
        prefix = f"blocks.{block_index}"
        candidates = (
            ("attn1.to_q", block.attn1.to_q),
            ("attn1.to_k", block.attn1.to_k),
            ("attn1.to_v", block.attn1.to_v),
            ("attn1.to_out.0", block.attn1.to_out[0]),
            ("attn2.to_q", block.attn2.to_q),
            ("attn2.to_k", block.attn2.to_k),
            ("attn2.to_v", block.attn2.to_v),
            ("attn2.to_out.0", block.attn2.to_out[0]),
            ("ffn.net.0.proj", block.ffn.net[0].proj),
            ("ffn.net.2", block.ffn.net[2]),
        )
        for suffix, module in candidates:
            if not isinstance(module, nn.Linear):
                raise TypeError(f"unexpected Wan linear at {prefix}.{suffix}: {type(module)}")
            names.append(f"{prefix}.{suffix}")
    return names


def install_wan_whole_dit_quant(
    transformer: nn.Module,
    spec: QuantSpec,
    *,
    include: str = "all",
    block_indices: frozenset[int] | None = None,
) -> list[WholeDiTQuantLinear]:
    """Replace compute-dominant attention and FFN linears in every Wan block."""
    allowed = {"all", "self_attn", "cross_attn", "ffn"}
    if include not in allowed:
        raise ValueError(f"include must be one of {sorted(allowed)}")
    replacements: list[WholeDiTQuantLinear] = []
    activation_fraction = float(
        "1.0"
    )
    activation_fraction = min(1.0, max(0.0, activation_fraction))
    selected_for_a6: set[str] | None = None
    if activation_fraction < 1.0:
        weights = spec.propagation_state_weights or {}
        scores: list[tuple[float, str]] = []
        for name in _wan_main_linear_names(transformer):
            values = sorted((float(v) for v in weights.get(name, (1.0,))), reverse=True)
            tail_count = max(1, round(len(values) * max(spec.risk_tail_fraction, 0.25)))
            score = sum(values[:tail_count]) / tail_count
            scores.append((score, name))
        keep = max(1, round(len(scores) * activation_fraction)) if activation_fraction else 0
        selected_for_a6 = {
            name for _, name in sorted(scores, key=lambda item: (item[0], item[1]), reverse=True)[:keep]
        }
    for block_index, block in enumerate(transformer.blocks):
        if block_indices is not None and block_index not in block_indices:
            continue
        targets: list[tuple[nn.Module, str | int, str]] = []
        if include in {"all", "self_attn"}:
            targets.extend(
                [
                    (block.attn1, "to_q", "attn1.to_q"),
                    (block.attn1, "to_k", "attn1.to_k"),
                    (block.attn1, "to_v", "attn1.to_v"),
                    (block.attn1.to_out, 0, "attn1.to_out.0"),
                ]
            )
        if include in {"all", "cross_attn"}:
            targets.extend(
                [
                    (block.attn2, "to_q", "attn2.to_q"),
                    (block.attn2, "to_k", "attn2.to_k"),
                    (block.attn2, "to_v", "attn2.to_v"),
                    (block.attn2.to_out, 0, "attn2.to_out.0"),
                ]
            )
        if include in {"all", "ffn"}:
            targets.extend(
                [
                    (block.ffn.net[0], "proj", "ffn.net.0.proj"),
                    (block.ffn.net, 2, "ffn.net.2"),
                ]
            )
        for parent, key, suffix in targets:
            source = parent[key] if isinstance(key, int) else getattr(parent, key)
            if not isinstance(source, nn.Linear):
                raise TypeError(f"unexpected Wan linear at block {block_index} {suffix}")
            name = f"blocks.{block_index}.{suffix}"
            replacement = WholeDiTQuantLinear(
                source,
                spec,
                name,
                selective_activation_bypass=(
                    selected_for_a6 is not None and name not in selected_for_a6
                ),
            )
            replacement.block_index = block_index
            if isinstance(key, int):
                parent[key] = replacement
            else:
                setattr(parent, key, replacement)
            replacements.append(replacement)
    return replacements


@torch.inference_mode()
def apply_residual_a8_schedule(
    modules: list[WholeDiTQuantLinear], fraction: float
) -> dict[str, object]:
    """Promote a global top fraction of Linear x time bins from A6 to A8."""
    fraction = min(1.0, max(0.0, float(fraction)))
    entries: list[tuple[float, str, int, WholeDiTQuantLinear]] = []
    scope = "all"
    for module in modules:
        module.activation_bit_schedule.fill_(module.spec.activation_bits)
        module.activation_qmax_schedule.fill_(
            float(2 ** (module.spec.activation_bits - 1) - 1)
        )
        suffix = module.module_name.split(".", 2)[-1]
        in_scope = (
            True
            or (False and suffix.startswith("attn1."))
            or (False and suffix.startswith("attn2."))
            or (False and suffix.startswith("ffn."))
        )
        if not in_scope:
            continue
        for bin_index, score in enumerate(module.activation_rescue_score.float().cpu()):
            value = float(score)
            if math.isfinite(value) and value > 0.0:
                entries.append((value, module.module_name, bin_index, module))
    requested = round(len(modules) * 10 * fraction)
    keep = min(len(entries), max(0, requested))
    selected = sorted(entries, key=lambda row: (row[0], row[1], row[2]), reverse=True)[:keep]
    for _, _, bin_index, module in selected:
        module.activation_bit_schedule[bin_index] = 8
        module.activation_qmax_schedule[bin_index] = 127.0
    return {
        "policy": "global_residual_prop_top_fraction",
        "fraction": fraction,
        "scope": scope,
        "ratio_mode": "same",
        "eligible_positive_bins": len(entries),
        "selected_bins": keep,
        "total_bins": len(modules) * 10,
        "threshold": (selected[-1][0] if selected else None),
    }


_RUNTIME_CACHE_BUFFERS = (
    "weight",
    "bias",
    "channel_mask",
    "activation_granularity",
    "activation_bulk_ratio",
    "activation_bulk_schedule",
    "activation_bit_schedule",
    "activation_qmax_schedule",
    "activation_rescue_score",
    "activation_a8_ratio_schedule",
    "trajectory_delta_normalizer",
)


_CACHE_ENV_EXCLUDE = {
    "PULSE_QUANT_CACHE_LOAD",
    "PULSE_QUANT_CACHE_SAVE",
    "PULSE_ALLOW_ACTIVATION_CODEBOOK_CACHE_MISMATCH",
    "PULSE_ALLOW_LEGACY_CACHE",
    "PULSE_LEGACY_WEIGHT_BITS",
    "PULSE_LEGACY_ACTIVATION_BITS",
}


def _cache_jsonable(value):
    """Convert a calibration contract to deterministic JSON primitives."""
    if is_dataclass(value):
        return _cache_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _cache_jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted((_cache_jsonable(item) for item in value), key=repr)
    if isinstance(value, (tuple, list)):
        return [_cache_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if torch.is_tensor(value):
        return _cache_jsonable(value.detach().cpu().tolist())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        _cache_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_manifest(model: str | os.PathLike[str]) -> dict[str, object]:
    """Fingerprint model configuration and checkpoint shard identity cheaply.

    JSON/configuration files are content-hashed. Large checkpoint shards are
    identified by relative path, byte size, and nanosecond mtime so inference
    does not reread tens of GiB merely to validate a cache.
    """
    root = Path(model)
    if not root.exists():
        return {"identifier": str(model), "kind": "external_identifier"}
    if root.is_file():
        return {
            "identifier": str(root.resolve()),
            "kind": "file",
            "bytes": root.stat().st_size,
            "sha256": _file_sha256(root),
        }
    config_rows = []
    shard_rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in {".json", ".txt", ".model"}:
            config_rows.append(
                {"path": relative, "bytes": path.stat().st_size, "sha256": _file_sha256(path)}
            )
        elif path.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"}:
            stat = path.stat()
            shard_rows.append(
                {"path": relative, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            )
    manifest = {
        "identifier": str(root.resolve()),
        "kind": "directory",
        "configuration": config_rows,
        "checkpoint_shards": shard_rows,
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    return manifest


def build_quant_cache_context(
    model: str | os.PathLike[str],
    *,
    propagation_profile: str | os.PathLike[str] | None = None,
    calibration_prompts: str | os.PathLike[str] | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build caller context that is not recoverable from ``QuantSpec``."""
    context: dict[str, object] = {"model": _model_manifest(model)}
    for key, value in (
        ("propagation_profile", propagation_profile),
        ("calibration_prompts", calibration_prompts),
    ):
        if value is None:
            context[key] = None
            continue
        path = Path(value)
        context[key] = (
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            if path.is_file()
            else {"path": str(value), "missing": True}
        )
    context["extra"] = _cache_jsonable(extra or {})
    return context


def _cache_contract_for_comparison(contract):
    """Compare consumed Wan asset configs by content, independent of directory.

    Historical v2 contracts fingerprinted the entire external Diffusers model
    directory. Only the T5 and VAE config files are consumed by the Wan loader;
    its other files and checkpoint shards are unrelated to this asset identity.
    The original stored contract hash is still verified before comparison.
    """
    normalized = _cache_jsonable(contract)
    if not isinstance(normalized, dict):
        return normalized
    context = normalized.get("context")
    if not isinstance(context, dict):
        return normalized
    extra = context.get("extra")
    if not isinstance(extra, dict):
        return normalized
    assets = extra.get("diffusers_assets")
    if not isinstance(assets, dict) or assets.get("kind") != "directory":
        return normalized
    rows = assets.get("configuration")
    if not isinstance(rows, list):
        return normalized
    required = {"text_encoder/config.json", "vae/config.json"}
    selected = [row for row in rows if isinstance(row, dict) and row.get("path") in required]
    if len(selected) != 2 or {row["path"] for row in selected} != required:
        return normalized
    if any("bytes" not in row or "sha256" not in row for row in selected):
        return normalized
    extra["diffusers_assets"] = {
        "kind": "directory",
        "configuration": [
            {key: row[key] for key in ("path", "bytes", "sha256")}
            for row in sorted(selected, key=lambda row: row["path"])
        ],
    }
    return normalized


def _quant_cache_contract(
    modules: list[WholeDiTQuantLinear],
    cache_context: Mapping[str, object] | None,
) -> dict[str, object]:
    if not modules:
        raise ValueError("cannot build a quant cache contract without modules")
    first_spec = _cache_jsonable(modules[0].spec)
    module_rows = []
    for module in modules:
        module_rows.append(
            {
                "name": module.module_name,
                "weight_shape": list(module.weight.shape),
                "weight_dtype": str(module.weight.dtype),
                "method": module.spec.method,
                "weight_bits": module.spec.weight_bits,
                "activation_bits": module.spec.activation_bits,
                "hadamard_group": module.group,
                "rotation_mode": module.rotation_mode,
                "precompose_rotation": bool(module.precompose_rotation),
            }
        )
    environment = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("PULSE_") and key not in _CACHE_ENV_EXCLUDE
    }
    return {
        "schema": "pulse-calibration-contract-v2",
        "quant_spec": first_spec,
        "modules": module_rows,
        "environment": environment,
        "context": _cache_jsonable(cache_context or {}),
    }


@torch.inference_mode()
def save_finalized_quant_cache(
    modules: list[WholeDiTQuantLinear],
    path: str | os.PathLike[str],
    *,
    cache_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Persist only the frozen tensors needed by the inference hot path.

    This deliberately omits calibration samples, statistics, source biases and
    other tensors already supplied by the base checkpoint.  The first version
    stores the finalized BF16 weights losslessly; a later packed-kernel backend
    can replace that payload without changing the caller contract.
    """
    if not modules or not all(module.finalized for module in modules):
        raise RuntimeError("quant cache can only be saved after every module finalizes")
    if cache_context is None:
        raise ValueError(
            "pulse-finalized-v2 requires cache_context with model and calibration identity"
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache_contract = _quant_cache_contract(modules, cache_context)
    cache_contract_sha256 = _canonical_sha256(cache_contract)
    payload: dict[str, object] = {
        "format": "pulse-finalized-v2",
        "cache_contract": cache_contract,
        "cache_contract_sha256": cache_contract_sha256,
        "precompose_rotation": bool(modules[0].precompose_rotation),
        "rotation_mode": modules[0].rotation_mode,
        "orbit_codebook_hybrid": bool(modules[0].orbit_codebook_hybrid),
        "orbit_weight_codebook": bool(modules[0].orbit_weight_codebook),
        "orbit_activation_codebook": bool(
            modules[0].orbit_activation_codebook_enabled
        ),
        "rpbh_block_size": os.environ.get("PULSE_RPBH_BLOCK_SIZE", "128"),
        "rpbh_seed": int(os.environ.get("PULSE_RPBH_SEED", "0")),
        "orbit_radial_ls": float(os.environ.get("PULSE_ORBIT_RADIAL_LS", "0")),
        "orbit_radial_clip": float(os.environ.get("PULSE_ORBIT_RADIAL_CLIP", ".08")),
        "analytic_transport_lambda": float(os.environ.get("PULSE_ANALYTIC_TRANSPORT_LAMBDA", "0")),
        "analytic_transport_threshold": float(os.environ.get("PULSE_ANALYTIC_TRANSPORT_THRESHOLD", "0")),
        "orbit_lite_radial_base": float(os.environ.get("PULSE_ORBIT_LITE_RADIAL_BASE", "0")),
        "orbit_lite_weight_trust": float(os.environ.get("PULSE_ORBIT_LITE_WEIGHT_TRUST", "-1")),
        "orbit_radial_span": float(os.environ.get("PULSE_ORBIT_RADIAL_SPAN", ".10")),
        "orbit_radial_levels": int(os.environ.get("PULSE_ORBIT_RADIAL_LEVELS", "5")),
        "orbit_adaptive_span_power": float(os.environ.get("PULSE_ORBIT_ADAPTIVE_SPAN_POWER", "0")),
        "orbit_adaptive_span_min": float(os.environ.get("PULSE_ORBIT_ADAPTIVE_SPAN_MIN", ".5")),
        "orbit_adaptive_span_max": float(os.environ.get("PULSE_ORBIT_ADAPTIVE_SPAN_MAX", "1.5")),
        "output_bias_correction": float(os.environ.get("PULSE_OUTPUT_BIAS_CORRECTION", "0")),
        "direction_risk_beta": float(os.environ.get("PULSE_DIRECTION_RISK_BETA", "0")),
        "temporal_delta_weight": float(os.environ.get("PULSE_TEMPORAL_DELTA_WEIGHT", "0")),
        "two_stage_trust_ratio": float(os.environ.get("PULSE_TWO_STAGE_TRUST_RATIO", "-1")),
        "pareto_stage_weight": float(os.environ.get("PULSE_PARETO_STAGE_WEIGHT", "0")),
        "pareto_stage_risk_threshold": float(os.environ.get("PULSE_PARETO_STAGE_RISK_THRESHOLD", "0")),
        "pareto_subspace_weight": float(os.environ.get("PULSE_PARETO_SUBSPACE_WEIGHT", "0")),
        "pareto_stage_delta_trust": float(os.environ.get("PULSE_PARETO_STAGE_DELTA_TRUST", "-1")),
        "pareto_stage_min_margin": float(os.environ.get("PULSE_PARETO_STAGE_MIN_MARGIN", "0")),
        "pareto_module_scope": os.environ.get("PULSE_PARETO_MODULE_SCOPE", "all"),
        "pareto_block_cutoff": int(os.environ.get("PULSE_PARETO_BLOCK_CUTOFF", "15")),
        "selective_risk_threshold": float(os.environ.get("PULSE_SELECTIVE_RISK_THRESHOLD", "0")),
        "guard_selection": float(os.environ.get("PULSE_GUARD_SELECTION", "0")),
        "guard_local_tolerance": float(os.environ.get("PULSE_GUARD_LOCAL_TOLERANCE", ".005")),
        "guard_win_rate": float(os.environ.get("PULSE_GUARD_WIN_RATE", ".75")),
        "guard_min_margin": float(os.environ.get("PULSE_GUARD_MIN_MARGIN", ".001")),
        "guard_proposal_prompts": float(os.environ.get("PULSE_GUARD_PROPOSAL_PROMPTS", "0")),
        "adaptive_search_risk_threshold": float(os.environ.get("PULSE_ADAPTIVE_SEARCH_RISK_THRESHOLD", "0")),
        "adaptive_search_low_steps": float(os.environ.get("PULSE_ADAPTIVE_SEARCH_LOW_STEPS", "2")),
        "adaptive_search_low_levels": float(os.environ.get("PULSE_ADAPTIVE_SEARCH_LOW_LEVELS", "3")),
        "modules": {
            module.module_name: {
                name: (
                    None
                    if getattr(module, name) is None
                    else getattr(module, name).detach().cpu()
                )
                for name in _RUNTIME_CACHE_BUFFERS
            }
            for module in modules
        },
    }
    torch.save(payload, destination)
    return {
        "format": payload["format"],
        "path": str(destination),
        "modules": len(modules),
        "bytes": destination.stat().st_size,
        "cache_contract_sha256": cache_contract_sha256,
    }


@torch.inference_mode()
def _validate_cache_states(modules, states, *, legacy: bool) -> None:
    """Preflight every module before modifying any live inference state."""
    if not isinstance(states, Mapping):
        raise ValueError("quant cache modules must be a mapping")
    required = {"weight", "bias", "channel_mask"} if legacy else set(_RUNTIME_CACHE_BUFFERS)
    for module in modules:
        state = states[module.module_name]
        if not isinstance(state, Mapping):
            raise ValueError(f"invalid state for {module.module_name}")
        missing = required - state.keys()
        if missing:
            raise ValueError(f"{module.module_name}: missing required buffers {sorted(missing)}")
        for name in _RUNTIME_CACHE_BUFFERS:
            if name not in state:
                continue
            value = state[name]
            current = getattr(module, name)
            label = f"{module.module_name}.{name}"
            if value is None:
                if name == "weight" or current is not None:
                    raise ValueError(f"{label}: unexpected None")
                if name == "channel_mask" and module.act_max is not None and module.spec.method != "twgr":
                    raise ValueError(f"{label}: calibrated channel mask is required")
                continue
            if name == "bias" and current is None:
                raise ValueError(f"{label}: expected no bias")
            shape = (module.in_features,) if name == "channel_mask" else tuple(current.shape)
            if name == "weight" and isinstance(value, Mapping):
                codes, scale = value.get("codes"), value.get("scale")
                if (value.get("format") != "rowwise_int8"
                        or not isinstance(codes, torch.Tensor)
                        or codes.dtype != torch.int8 or tuple(codes.shape) != shape
                        or not isinstance(scale, torch.Tensor)
                        or not scale.is_floating_point()
                        or tuple(scale.shape) != (module.out_features, 1)):
                    raise ValueError(f"{label}: invalid rowwise_int8 codes/scale")
            elif not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"{label}: expected tensor shape {shape}")
            if legacy and name == "activation_bit_schedule":
                # A uniform schedule provides direct evidence of the cache's
                # activation precision; mixed rescue schedules remain supported.
                if value.numel() and bool(torch.all(value == value.flatten()[0])):
                    if int(value.flatten()[0]) != module.spec.activation_bits:
                        raise ValueError(f"{label}: legacy activation precision mismatch")


@torch.inference_mode()
def load_finalized_quant_cache(
    modules: list[WholeDiTQuantLinear],
    path: str | os.PathLike[str],
    *,
    cache_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Restore a frozen cache and make calibration a true offline operation."""
    started = time.perf_counter()
    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("quant cache payload must be a mapping")
    cache_format = payload.get("format")
    if cache_format not in {"pulse-finalized-v1", "pulse-finalized-v2"}:
        raise ValueError(
            "unsafe or unsupported quant cache format "
            f"{cache_format!r}; expected pulse-finalized-v1 or pulse-finalized-v2"
        )
    legacy_cache = cache_format == "pulse-finalized-v1"
    if legacy_cache:
        if os.environ.get("PULSE_ALLOW_LEGACY_CACHE", "1") != "1":
            raise ValueError(
                "pulse-finalized-v1 cache loading is disabled; "
                "set PULSE_ALLOW_LEGACY_CACHE=1 to enable it"
            )
        for field, env in (("weight_bits", "PULSE_LEGACY_WEIGHT_BITS"),
                           ("activation_bits", "PULSE_LEGACY_ACTIVATION_BITS")):
            declared = os.environ.get(env)
            if declared is None:
                raise ValueError(f"legacy cache requires explicit {env} confirmation")
            if any(int(declared) != getattr(module.spec, field) for module in modules):
                raise ValueError(f"legacy cache precision mismatch: {env}={declared}")
            if field in payload and int(payload[field]) != int(declared):
                raise ValueError(f"legacy cache metadata differs from {env}")
        cached_sha256 = None
        warnings.warn(
            "loading pulse-finalized-v1 with best-effort compatibility checks; "
            "the cache does not contain model/profile/prompt hashes or the full "
            "calibration contract",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        if cache_context is None:
            raise ValueError(
                "pulse-finalized-v2 requires cache_context with model and calibration identity"
            )
        expected_contract = _quant_cache_contract(modules, cache_context)
        cached_contract = payload.get("cache_contract")
        cached_sha256 = payload.get("cache_contract_sha256")
        if cached_sha256 != _canonical_sha256(cached_contract):
            raise ValueError("quant cache contract payload is corrupted")
        expected_comparison_sha256 = _canonical_sha256(
            _cache_contract_for_comparison(expected_contract)
        )
        cached_comparison_sha256 = _canonical_sha256(
            _cache_contract_for_comparison(cached_contract)
        )
        if cached_comparison_sha256 != expected_comparison_sha256:
            raise ValueError(
                "quant cache calibration contract mismatch: "
                f"cached={cached_comparison_sha256}, requested={expected_comparison_sha256}; "
                "recalibration required"
            )
    states = payload.get("modules")
    if not isinstance(states, Mapping):
        raise ValueError("quant cache modules must be a mapping")
    cached_precompose = bool(payload.get("precompose_rotation", False))
    if any(module.precompose_rotation != cached_precompose for module in modules):
        raise ValueError(
            "quant cache rotation mode differs from PULSE_PRECOMPOSE_ROTATION"
        )
    cached_rotation_mode = payload.get("rotation_mode", "block_hadamard")
    if any(module.rotation_mode != cached_rotation_mode for module in modules):
        raise ValueError(
            "quant cache rotation mode differs from PULSE_ROTATION_MODE"
        )
    cached_orbit_hybrid = bool(payload.get("orbit_codebook_hybrid", False))
    cached_orbit_weight = bool(payload.get("orbit_weight_codebook", cached_orbit_hybrid))
    cached_orbit_activation = bool(
        payload.get("orbit_activation_codebook", cached_orbit_hybrid)
    )
    if any(module.orbit_weight_codebook != cached_orbit_weight for module in modules):
        raise ValueError("quant cache differs from PULSE_ORBIT_WEIGHT_CODEBOOK")
    activation_codebook_mismatch = any(
        module.orbit_activation_codebook_enabled != cached_orbit_activation
        for module in modules
    )
    allow_activation_codebook_mismatch = (
        os.environ.get("PULSE_ALLOW_ACTIVATION_CODEBOOK_CACHE_MISMATCH", "0") == "1"
    )
    if activation_codebook_mismatch and not allow_activation_codebook_mismatch:
        raise ValueError("quant cache differs from PULSE_ORBIT_ACTIVATION_CODEBOOK")
    cached_rpbh_block = payload.get("rpbh_block_size")
    if cached_rpbh_block is not None:
        requested_block = os.environ.get("PULSE_RPBH_BLOCK_SIZE", "128")
        if str(cached_rpbh_block) != requested_block:
            raise ValueError("quant cache differs from PULSE_RPBH_BLOCK_SIZE")
    cached_rpbh_seed = payload.get("rpbh_seed")
    if cached_rpbh_seed is not None:
        requested_seed = int(os.environ.get("PULSE_RPBH_SEED", "0"))
        if int(cached_rpbh_seed) != requested_seed:
            raise ValueError("quant cache differs from PULSE_RPBH_SEED")
    cached_radial_span = payload.get("orbit_radial_span")
    if cached_radial_span is not None:
        requested_span = float(os.environ.get("PULSE_ORBIT_RADIAL_SPAN", ".10"))
        if not math.isclose(float(cached_radial_span), requested_span, abs_tol=1e-12):
            raise ValueError("quant cache differs from PULSE_ORBIT_RADIAL_SPAN")
    cached_radial_levels = payload.get("orbit_radial_levels")
    if cached_radial_levels is not None:
        requested_levels = int(os.environ.get("PULSE_ORBIT_RADIAL_LEVELS", "5"))
        if int(cached_radial_levels) != requested_levels:
            raise ValueError("quant cache differs from PULSE_ORBIT_RADIAL_LEVELS")
    cached_pareto_scope = payload.get("pareto_module_scope")
    if (
        cached_pareto_scope is not None
        and cached_pareto_scope
        != os.environ.get("PULSE_PARETO_MODULE_SCOPE", "all")
    ):
        raise ValueError("quant cache differs from PULSE_PARETO_MODULE_SCOPE")
    cached_pareto_cutoff = payload.get("pareto_block_cutoff")
    if (
        cached_pareto_cutoff is not None
        and int(cached_pareto_cutoff)
        != int(os.environ.get("PULSE_PARETO_BLOCK_CUTOFF", "15"))
    ):
        raise ValueError("quant cache differs from PULSE_PARETO_BLOCK_CUTOFF")
    for key, environment, default in (
        ("orbit_radial_ls", "PULSE_ORBIT_RADIAL_LS", "0"),
        ("orbit_radial_clip", "PULSE_ORBIT_RADIAL_CLIP", ".08"),
        ("analytic_transport_lambda", "PULSE_ANALYTIC_TRANSPORT_LAMBDA", "0"),
        ("analytic_transport_threshold", "PULSE_ANALYTIC_TRANSPORT_THRESHOLD", "0"),
        ("orbit_lite_radial_base", "PULSE_ORBIT_LITE_RADIAL_BASE", "0"),
        ("orbit_lite_weight_trust", "PULSE_ORBIT_LITE_WEIGHT_TRUST", "-1"),
        ("direction_risk_beta", "PULSE_DIRECTION_RISK_BETA", "0"),
        ("temporal_delta_weight", "PULSE_TEMPORAL_DELTA_WEIGHT", "0"),
        ("two_stage_trust_ratio", "PULSE_TWO_STAGE_TRUST_RATIO", "-1"),
        ("pareto_stage_weight", "PULSE_PARETO_STAGE_WEIGHT", "0"),
        ("pareto_stage_risk_threshold", "PULSE_PARETO_STAGE_RISK_THRESHOLD", "0"),
        ("pareto_subspace_weight", "PULSE_PARETO_SUBSPACE_WEIGHT", "0"),
        ("pareto_stage_delta_trust", "PULSE_PARETO_STAGE_DELTA_TRUST", "-1"),
        ("pareto_stage_min_margin", "PULSE_PARETO_STAGE_MIN_MARGIN", "0"),
        ("selective_risk_threshold", "PULSE_SELECTIVE_RISK_THRESHOLD", "0"),
        ("guard_selection", "PULSE_GUARD_SELECTION", "0"),
        ("guard_local_tolerance", "PULSE_GUARD_LOCAL_TOLERANCE", ".005"),
        ("guard_win_rate", "PULSE_GUARD_WIN_RATE", ".75"),
        ("guard_min_margin", "PULSE_GUARD_MIN_MARGIN", ".001"),
        ("guard_proposal_prompts", "PULSE_GUARD_PROPOSAL_PROMPTS", "0"),
        ("adaptive_search_risk_threshold", "PULSE_ADAPTIVE_SEARCH_RISK_THRESHOLD", "0"),
        ("adaptive_search_low_steps", "PULSE_ADAPTIVE_SEARCH_LOW_STEPS", "2"),
        ("adaptive_search_low_levels", "PULSE_ADAPTIVE_SEARCH_LOW_LEVELS", "3"),
        ("orbit_adaptive_span_power", "PULSE_ORBIT_ADAPTIVE_SPAN_POWER", "0"),
        ("orbit_adaptive_span_min", "PULSE_ORBIT_ADAPTIVE_SPAN_MIN", ".5"),
        ("orbit_adaptive_span_max", "PULSE_ORBIT_ADAPTIVE_SPAN_MAX", "1.5"),
        ("output_bias_correction", "PULSE_OUTPUT_BIAS_CORRECTION", "0"),
    ):
        if key in payload:
            requested = float(os.environ.get(environment, default))
            if not math.isclose(float(payload[key]), requested, abs_tol=1e-12):
                raise ValueError(f"quant cache differs from {environment}")
    expected = {module.module_name for module in modules}
    if set(states) != expected:
        missing = sorted(expected - set(states))
        extra = sorted(set(states) - expected)
        raise ValueError(
            f"quant cache module mismatch: missing={missing[:4]}, extra={extra[:4]}"
        )
    _validate_cache_states(modules, states, legacy=legacy_cache)
    for module in modules:
        state = states[module.module_name]
        for name in _RUNTIME_CACHE_BUFFERS:
            # v1 caches created before a new optional runtime schedule was
            # introduced remain valid; retain the module's initialized default.
            if name not in state:
                continue
            cached = state[name]
            current = getattr(module, name)
            if cached is None:
                setattr(module, name, None)
            elif (
                name == "weight"
                and isinstance(cached, Mapping)
                and cached.get("format") == "rowwise_int8"
            ):
                codes = cached["codes"].to(
                    device=module.weight.device, dtype=torch.float32
                )
                scale = cached["scale"].to(
                    device=module.weight.device, dtype=torch.float32
                )
                restored = codes * scale
                if current is None or current.shape != restored.shape:
                    setattr(module, name, restored.to(module.weight.dtype))
                else:
                    current.copy_(restored.to(dtype=current.dtype))
            elif current is None or current.shape != cached.shape:
                setattr(module, name, cached.to(module.weight.device))
            else:
                current.copy_(cached.to(device=current.device, dtype=current.dtype))
        module.calls = module.spec.calibration_calls
        module.energy_calls = 0
        module.calibration_samples.clear()
        module._activation_scale_history = [None, None]
        module.finalized = True
        # Moment-conserving Walsh correction must operate on the cached,
        # rotated weight representation.  For A16 ablations, postpone folding
        # the inverse rotation into the weight until the runner has applied
        # that correction; otherwise the folded inference tensor cannot be
        # updated safely and the correction is mathematically in the wrong
        # basis.
        if os.environ.get("PULSE_DEFER_SELECTIVE_BYPASS_FOLD", "0") != "1":
            module._fold_selective_bypass_weight()
    return {
        "format": payload["format"],
        "contract_validation": (
            "legacy_best_effort" if legacy_cache else "v2_exact"
        ),
        "legacy_unverified_fields": (
            [
                "model_identity",
                "propagation_profile",
                "calibration_prompts",
                "quant_spec",
                "correction_environment",
            ]
            if legacy_cache
            else []
        ),
        "path": str(source),
        "modules": len(modules),
        "bytes": source.stat().st_size,
        "load_seconds": time.perf_counter() - started,
        "cache_contract_sha256": cached_sha256,
    }


__all__ = [
    "QuantSpec",
    "WholeDiTQuantLinear",
    "install_wan_whole_dit_quant",
    "apply_residual_a8_schedule",
    "save_finalized_quant_cache",
    "load_finalized_quant_cache",
    "build_quant_cache_context",
    "minmax_weight_qdq",
    "token_symmetric_qdq",
]
