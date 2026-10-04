#!/usr/bin/env bash
set -euo pipefail
ABITS="${1:-4}"; [[ "$ABITS" == 4 || "$ABITS" == 6 ]]
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; source "$PKG_ROOT/scripts/common_env.sh"
REPO="${REPO:?Set REPO to the integration dependency directory; see README.md}"; PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:?Set MODEL to your local input path; see README.md}"; PROFILE="${PROFILE:-$PKG_ROOT/configs/profiles/wan22_model_specific.json}"
RUN_ROOT="${RUN_ROOT:-$PKG_ROOT/runs/wan22_a14b/w4a$ABITS}"; CACHE="${CACHE:-$RUN_ROOT/cache/pulsequant.pt}"; pulse_require_paths REPO MODEL PROFILE; mkdir -p "$RUN_ROOT/cache" "$RUN_ROOT/calibration"
CALIB_PROMPTS="${CALIB_PROMPTS:-$PKG_ROOT/configs/calibration_prompts/h3_complementary_3.txt}"
pulse_require_paths CALIB_PROMPTS
export PYTHONPATH="$H3_DIFFUSERS_ROOT:$PKG_ROOT/runtime/wan21:${ORBITQUANT_ROOT:-$REPO/third_party/orbitquant}:$REPO${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}" PULSE_QUANT_CACHE_SAVE="$CACHE" PULSE_MODEL_CPU_OFFLOAD="${PULSE_MODEL_CPU_OFFLOAD:-1}" PULSE_PROP_SCHEDULER_ALIGN=1
"$PYTHON_BIN" "$PKG_ROOT/runtime/wan22/wan_runner/run_method.py" --model "$MODEL" --prompts "$CALIB_PROMPTS" --calibration-prompts "$CALIB_PROMPTS" \
  --steps 40 --height 480 --width 832 --frames 81 --seed 23001 --seed-mode per_prompt --seed-index-origin absolute --negative-prompt-style svg --guidance-scale 4 --guidance-scale-2 3 --flow-shift 12 --method dense --temporal-mode none \
  --dit-weight-bits 4 --dit-activation-bits "$ABITS" --dit-quant-scope all --dit-quant-method pulseprofile --dit-quant-precision-profile "$PROFILE" --dit-quant-calibration-prompt-count 3 --dit-quant-calibration-prompt-indices 0,1,2 \
  --dit-quant-calibration-steps 6 --dit-quant-samples-per-call 4 --dit-quant-risk-lambda .75 --dit-quant-risk-tail-fraction .25 --dit-quant-trajectory-delta-weight 1 --dit-quant-trajectory-trust-ratio .05 --dit-quant-bgr-shrink-steps 6 --dit-quant-bgr-refine-steps 2 --dit-quant-activation-fixed-clipping-ratio 0 \
  --prompt-offset 0 --limit 0 --calibration-only --output-dir "$RUN_ROOT/calibration"
test -s "$CACHE"; "$PYTHON_BIN" "$PKG_ROOT/runtime/wan22/wan22_validate_pulse_calibration.py" --output "$RUN_ROOT/calibration" --cache "$CACHE" --bits "$ABITS"; sha256sum "$CACHE" > "$CACHE.sha256"
