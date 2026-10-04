#!/usr/bin/env bash
set -euo pipefail
ABITS="${1:-4}"; [[ "$ABITS" == 4 || "$ABITS" == 6 ]]
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; source "$PKG_ROOT/scripts/common_env.sh"
REPO="${REPO:?Set REPO to the integration dependency directory; see README.md}"; PYTHON_BIN="${PYTHON_BIN:-python}"; SELF_REPO="${SELF_REPO:?Set SELF_REPO to your local input path; see README.md}"; CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to your local input path; see README.md}"
PROMPTS="${PROMPTS:?Set PROMPTS to your local input path; see README.md}"; CALIB_PROMPTS="${CALIB_PROMPTS:-$PKG_ROOT/configs/calibration_prompts/h3_complementary_3.txt}"; PROFILE="${PROFILE:-$PKG_ROOT/configs/profiles/self_flat_prior.json}"; OUT="${OUT:-$PKG_ROOT/runs/self_forcing/w4a$ABITS/inference}"
pulse_require_paths REPO SELF_REPO SELF_WAN_MODEL_ROOT CHECKPOINT PROMPTS CALIB_PROMPTS PROFILE; mkdir -p "$OUT"; export PYTHONPATH="$PKG_ROOT/runtime/self_forcing:$PKG_ROOT/runtime/wan21:${ORBITQUANT_ROOT:-$REPO/third_party/orbitquant}:$REPO${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}" PULSE_SELF_AR_LOCAL_RISK=1 PULSE_SELF_AR_LOCAL_RISK_BETA=1 PULSE_SELF_AR_CAUSAL_POWER=.5
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}" "$PYTHON_BIN" "$PKG_ROOT/runtime/self_forcing/self_forcing_quant_runner_walsh.py" --self-repo "$SELF_REPO" --config "$SELF_REPO/configs/self_forcing_dmd.yaml" --checkpoint "$CHECKPOINT" \
  --prompts "$PROMPTS" --calibration-prompts "$CALIB_PROMPTS" --calibration-indices 0,1,2 --calibration-seed 991 --precision-profile "$PROFILE" --output-dir "$OUT" --method fouranchor \
  --prompt-offset "${OFFSET:-0}" --limit "${LIMIT:-1}" --seed "${SEED:-1024}" --weight-bits 4 --activation-bits "$ABITS" --overwrite
