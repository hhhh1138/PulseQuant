"""Reject retired experimental switches while preserving default cache metadata."""
import math
import os

_RETIRED_DEFAULTS = {
    "PULSE_ACTIVATION_CODEBOOK": [
        "uniform"
    ],
    "PULSE_SIGMA_DELTA_GROUP": [
        "32"
    ],
    "PULSE_SIGMA_DELTA_REVERSE": [
        "0"
    ],
    "PULSE_CONSERVATION_GROUP": [
        "32"
    ],
    "PULSE_RMS_SHELL_STRENGTH": [
        "1"
    ],
    "PULSE_PROMPT_UNCERTAINTY_BETA": [
        "0"
    ],
    "PULSE_PROMPT_WORST_WEIGHT": [
        "0"
    ],
    "PULSE_GUARD_LOCAL_TOLERANCE": [
        ".005"
    ],
    "PULSE_GUARD_WIN_RATE": [
        ".75"
    ],
    "PULSE_GUARD_MIN_MARGIN": [
        ".001"
    ],
    "PULSE_DIRECTION_RISK_BETA": [
        "0"
    ],
    "PULSE_COMPILE_MODE": [
        "default"
    ],
    "PULSE_ACTIVE_PROBE_OUTPUT": [
        ""
    ],
    "PULSE_ACTIVE_PROBE_SENTINEL_BLOCKS": [
        "0,7,14,21,29"
    ],
    "PULSE_ACTIVE_PROBE_PROMPT_INDICES": [
        ""
    ],
    "PULSE_GUARD_SELECTION": [
        "0"
    ],
    "PULSE_GUARD_PROPOSAL_PROMPTS": [
        "0"
    ],
    "PULSE_PATH_COUPLING_BASIS": [
        "random"
    ],
    "PULSE_SYNTHETIC_GAUGE_ALPHAS": [
        "0,.125,.25,.375,.5,.75,1"
    ],
    "PULSE_SYNTHETIC_GAUGE_PROBES": [
        "16"
    ],
    "PULSE_SYNTHETIC_GAUGE_SEED": [
        "2718"
    ],
    "PULSE_SYNTHETIC_GAUGE_VALUE_WEIGHT": [
        ".25"
    ],
    "PULSE_SYNTHETIC_GAUGE_AUDIT": [
        ""
    ],
    "PULSE_WEIGHT_GAUGE_VALUE_WEIGHT": [
        ".25"
    ],
    "PULSE_WEIGHT_GAUGE_MIN_GAIN": [
        "0"
    ],
    "PULSE_WEIGHT_GAUGE_AUDIT": [
        ""
    ],
    "PULSE_STAGE_CANCELLATION_BLOCKS": [
        "5"
    ],
    "PULSE_ORBIT_ANGULAR_ETA": [
        "0"
    ],
    "PULSE_ORBIT_LITE_RADIAL_BASE": [
        "0"
    ],
    "PULSE_ORBIT_RADIAL_LS": [
        "0",
        "1"
    ],
    "PULSE_ORBIT_RADIAL_CLIP": [
        ".08"
    ],
    "PULSE_PARETO_MODULE_SCOPE": [
        "all"
    ],
    "PULSE_PARETO_BLOCK_CUTOFF": [
        "15"
    ],
    "PULSE_ORBIT_ADAPTIVE_SPAN_POWER": [
        "0"
    ],
    "PULSE_ORBIT_ADAPTIVE_SPAN_MIN": [
        ".5"
    ],
    "PULSE_ORBIT_ADAPTIVE_SPAN_MAX": [
        "1.5"
    ],
    "PULSE_ADAPTIVE_SEARCH_RISK_THRESHOLD": [
        "0"
    ],
    "PULSE_ADAPTIVE_SEARCH_LOW_LEVELS": [
        "3"
    ],
    "PULSE_ORBIT_SUBSPACE_WEIGHT": [
        "0"
    ],
    "PULSE_ORBIT_SUBSPACE_GROUP": [
        "32"
    ],
    "PULSE_STAGE_CANCELLATION_WEIGHT": [
        "0"
    ],
    "PULSE_PARETO_STAGE_WEIGHT": [
        "0"
    ],
    "PULSE_PARETO_STAGE_RISK_THRESHOLD": [
        "0"
    ],
    "PULSE_PARETO_SUBSPACE_WEIGHT": [
        "0"
    ],
    "PULSE_ORBIT_LITE_WEIGHT_TRUST": [
        "-1"
    ],
    "PULSE_PARETO_STAGE_DELTA_TRUST": [
        "-1"
    ],
    "PULSE_ORBIT_SUBSPACE_RANK": [
        "4"
    ],
    "PULSE_PARETO_STAGE_MIN_MARGIN": [
        "0"
    ],
    "PULSE_STAGE_CANCELLATION_DECAY": [
        ".8"
    ],
    "PULSE_FAST_HADAMARD": [
        "0"
    ],
    "PULSE_TRITON_HADAMARD": [
        "0"
    ],
    "PULSE_COMPILED_QDQ": [
        "0"
    ],
    "PULSE_COMPILED_ROTATION": [
        "0"
    ],
    "PULSE_ACTIVATION_SCALE_REUSE_INTERVAL": [
        "1"
    ],
    "PULSE_ACTIVATION_CLIP_SCHEDULE": [
        ""
    ],
    "PULSE_PATH_COUPLED_WEIGHT": [
        "0"
    ],
    "PULSE_PATH_GAUGE": [
        "0"
    ],
    "PULSE_RPBH_ADAPTIVE_SEEDS": [
        ""
    ],
    "PULSE_RPBH_ADAPTIVE_CORE_COUNT": [
        "2"
    ],
    "PULSE_RPBH_ADAPTIVE_EXTRA_SCOPE": [
        "all"
    ],
    "PULSE_RPBH_ADAPTIVE_EXTRA_MIN_GAIN": [
        "0"
    ],
    "PULSE_RPBH_ADAPTIVE_AUDIT": [
        ""
    ],
    "PULSE_PATH_COUPLING_STRENGTH": [
        ".25"
    ],
    "PULSE_PATH_COUPLING_SKETCH_DIM": [
        "8"
    ],
    "PULSE_PATH_COUPLING_ROW_CHUNK": [
        "256"
    ],
    "PULSE_PATH_COUPLING_SEED": [
        "1701"
    ],
    "PULSE_TEMPORAL_RPBH_SEEDS": [
        ""
    ],
    "PULSE_RPBH_TEMPORAL_WEIGHT": [
        ".5"
    ],
    "PULSE_RPBH_PROXY_CLIP": [
        ".70"
    ],
    "PULSE_ACTIVE_PROBE_ONLY": [
        "0"
    ],
    "PULSE_PROP_LINEAR_BETA": [
        "0"
    ],
    "PULSE_SELECTIVE_RISK_THRESHOLD": [
        "0"
    ],
    "PULSE_ADVERSARIAL_CALIB_EPS": [
        "0"
    ],
    "PULSE_TWO_STAGE_TRUST_RATIO": [
        "-1"
    ],
    "PULSE_ADAPTIVE_SEARCH_LOW_STEPS": [
        "2"
    ],
    "PULSE_PROP_W4_CLIP_PENALTY": [
        "0"
    ],
    "PULSE_MANIFOLD_NORMAL_WEIGHT": [
        "0"
    ],
    "PULSE_LYAPUNOV_WEIGHT": [
        "0"
    ],
    "PULSE_TEMPORAL_FREQUENCY_WEIGHT": [
        "0"
    ],
    "PULSE_WALSH2_AXIS_MODE": [
        "activation_pca"
    ],
    "PULSE_WALSH2_RANDOM_SEED": [
        "20260918"
    ],
    "PULSE_FMSE_CANDIDATE_RATIOS": [
        ""
    ],
    "PULSE_FRAME_TUBE_SHARE": [
        "0"
    ],
    "PULSE_FRAME_SCALE_SMOOTH": [
        "0"
    ],
    "PULSE_FRAME_SCALE_SECOND_SMOOTH": [
        "0"
    ],
    "PULSE_FRAME_MOTION_GATE_TAU": [
        "0"
    ],
    "PULSE_FRAME_MOMENT_STRENGTH": [
        "0"
    ],
    "PULSE_FRAME_MOMENT_GROUP": [
        "128"
    ],
    "PULSE_PROP_CLIP_CVAR": [
        "0"
    ],
    "PULSE_A6_TEMPORAL_WEIGHT": [
        "0"
    ],
    "PULSE_FRAME_DELTA_WEIGHT": [
        "0"
    ],
    "PULSE_FRAME_SECOND_DELTA_WEIGHT": [
        "0"
    ],
    "PULSE_FRAME_STATE_TRUST_RATIO": [
        "-1"
    ],
    "PULSE_PROP_CLIP_BETA": [
        "1"
    ],
    "PULSE_PROP_CLIP_SMOOTH": [
        "0"
    ],
    "PULSE_RESIDUAL_A8_RATIO_MODE": [
        "same"
    ],
    "PULSE_OUTPUT_BIAS_CORRECTION": [
        "0"
    ],
    "PULSE_ANALYTIC_TRANSPORT_BLOCKS": [
        "30"
    ],
    "PULSE_ANALYTIC_TRANSPORT_DEPTH_BETA": [
        ".7"
    ],
    "PULSE_ANALYTIC_TRANSPORT_LAMBDA": [
        "0"
    ],
    "PULSE_ANALYTIC_TRANSPORT_THRESHOLD": [
        "0"
    ],
    "PULSE_TEMPORAL_A6_MOMENTUM": [
        "0"
    ],
    "PULSE_HIERARCHICAL_RPBH_SEEDS": [
        "0,1,5"
    ],
    "PULSE_HIERARCHICAL_BLOCKS_PER_STAGE": [
        "5"
    ],
    "PULSE_HIERARCHICAL_GLOBAL_PENALTY": [
        "0"
    ],
    "PULSE_HIERARCHICAL_SWITCH_PENALTY": [
        "0"
    ],
    "PULSE_HIERARCHICAL_RISK_MULTIPLIERS": [
        ""
    ],
    "PULSE_HIERARCHICAL_RPBH_AUDIT": [
        ""
    ],
    "PULSE_PROPAGATION_A6_FRACTION": [
        "1.0"
    ],
    "PULSE_HIERARCHICAL_RPBH": [
        "0"
    ],
    "PULSE_PATH_GAUGE_SCOPE": [
        "all"
    ],
    "PULSE_PATH_GAUGE_ALPHA": [
        ".5"
    ],
    "PULSE_PATH_GAUGE_ATTENTION": [
        "all"
    ],
    "PULSE_PATH_GAUGE_CLIP": [
        "4"
    ],
    "PULSE_PATH_GAUGE_BLOCK_START": [
        "0"
    ],
    "PULSE_PATH_GAUGE_OBJECTIVE": [
        "norm"
    ],
    "PULSE_PATH_GAUGE_GATE": [
        "none"
    ],
    "PULSE_PATH_COUPLING_SCOPE": [
        "all"
    ],
    "PULSE_RESIDUAL_A8_SCOPE": [
        "all"
    ],
    "PULSE_MODEL_STATE_OVERLAY": [
        ""
    ],
    "PULSE_ACTIVATION_NATIVE_DISPATCH": [
        "0"
    ],
    "PULSE_MOMENT_CONSERVE_GROUP": [
        "0"
    ],
    "PULSE_MOMENT_CONSERVE_AXIS": [
        "ones"
    ],
    "PULSE_MOMENT_CONSERVE_MODES": [
        "1"
    ],
    "PULSE_MOMENT_CONSERVE_MODE_OFFSET": [
        "0"
    ],
    "PULSE_MOMENT_CONSERVE_MODE_LIST": [
        ""
    ],
    "PULSE_MOMENT_CONSERVE_PASSES": [
        "1"
    ],
    "PULSE_MOMENT_CONSERVE_STRENGTH": [
        "1"
    ],
    "PULSE_MOMENT_CONSERVE_STRATEGY": [
        "fixed"
    ],
    "PULSE_MOMENT_ROW_ENABLE": [
        "1"
    ],
    "PULSE_MOMENT_COLUMN_GROUP": [
        "0"
    ],
    "PULSE_MOMENT_MSE_REFIT": [
        "0"
    ],
    "PULSE_TCR_GLOBAL_SCALE": [
        "1"
    ],
    "PULSE_TRACE_ONLY": [
        "0"
    ],
    "PULSE_VAE_CPU_OFFLOAD": [
        "0"
    ],
    "PULSE_TEMPORAL_DELTA_WEIGHT": [
        "0"
    ]
}

def validate_release_settings():
    # The LS-center experiment used different fallback defaults in old branches;
    # the released row-norm-centered search always uses the disabled setting.
    _RETIRED_DEFAULTS["PULSE_ORBIT_RADIAL_LS"] = ["0"]
    for name, allowed in _RETIRED_DEFAULTS.items():
        value = os.environ.get(name)
        if value is None or value in allowed:
            continue
        try:
            numeric_match = any(math.isclose(float(value), float(item), rel_tol=0.0, abs_tol=1e-12)
                                for item in allowed)
        except ValueError:
            numeric_match = False
        if not numeric_match:
            raise ValueError(
                f"{name}={value!r} selects a retired experiment; "
                f"this release only accepts {allowed!r}. Unset it to use PulseQuant.")
