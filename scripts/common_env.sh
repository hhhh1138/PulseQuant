#!/usr/bin/env bash
# Shared final PulseQuant policy. Model scripts may override values before use.
export REPO="${REPO:-$PKG_ROOT}"
export ORBITQUANT_ROOT="${ORBITQUANT_ROOT:-$PKG_ROOT/third_party/orbitquant}"
export H3_DIFFUSERS_ROOT="${H3_DIFFUSERS_ROOT:-$PKG_ROOT/third_party/diffusers_h3/src}"
export SELF_REPO="${SELF_REPO:-$PKG_ROOT/third_party/self_forcing}"
export PYTHONPATH="$PKG_ROOT/runtime${PYTHONPATH:+:$PYTHONPATH}"
function pulse_require_paths() {
    local name value
    for name in "$@"; do
        value="${!name:-}"
        if [[ -z "$value" || ! -e "$value" ]]; then
            printf 'Missing input: %s=%s. Configure your local paths; see README.md.\n' "$name" "$value" >&2
            return 2
        fi
    done
    if ! command -v "${PYTHON_BIN:-python}" >/dev/null 2>&1; then
        printf 'Python interpreter not found: %s\n' "${PYTHON_BIN:-python}" >&2
        return 2
    fi
}
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export ORBITQUANT_KERNELS_AUTOFETCH="${ORBITQUANT_KERNELS_AUTOFETCH:-0}"
export PULSE_ALLOW_LEGACY_CACHE="${PULSE_ALLOW_LEGACY_CACHE:-1}"
export PULSE_ROTATION_MODE="${PULSE_ROTATION_MODE:-orbit_rpbh}"
export PULSE_RPBH_BLOCK_SIZE="${PULSE_RPBH_BLOCK_SIZE:-paper}"
export PULSE_RPBH_SEED="${PULSE_RPBH_SEED:-11}"
export PULSE_TRITON_RPBH_ROTATION="${PULSE_TRITON_RPBH_ROTATION:-1}"
export PULSE_ORBIT_WEIGHT_CODEBOOK="${PULSE_ORBIT_WEIGHT_CODEBOOK:-1}"
export PULSE_ORBIT_ACTIVATION_CODEBOOK="${PULSE_ORBIT_ACTIVATION_CODEBOOK:-1}"
export PULSE_WALSH2_CORRECTION="${PULSE_WALSH2_CORRECTION:-1}"
export PULSE_WALSH2_AXIS_MODE="${PULSE_WALSH2_AXIS_MODE:-activation_pca}"
export PULSE_WALSH2_GROUP="${PULSE_WALSH2_GROUP:-128}"
export PULSE_WALSH2_PASSES="${PULSE_WALSH2_PASSES:-1}"
export PULSE_WALSH2_WEIGHT_TRUST="${PULSE_WALSH2_WEIGHT_TRUST:-.25}"
export PULSE_WALSH2_PCA_SAMPLES="${PULSE_WALSH2_PCA_SAMPLES:-512}"
export PULSE_WALSH2_PCA_ITERS="${PULSE_WALSH2_PCA_ITERS:-2}"
export PULSE_ORBIT_RADIAL_SPAN="${PULSE_ORBIT_RADIAL_SPAN:-.08}"
export PULSE_ORBIT_RADIAL_LEVELS="${PULSE_ORBIT_RADIAL_LEVELS:-5}"
export PULSE_TEMPORAL_DELTA_WEIGHT="${PULSE_TEMPORAL_DELTA_WEIGHT:-0}"
export PULSE_RESIDUAL_A8_FRACTION="${PULSE_RESIDUAL_A8_FRACTION:-0}"
