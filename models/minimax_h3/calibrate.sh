#!/usr/bin/env bash
set -euo pipefail
ABITS="${1:-4}"; [[ "$ABITS" == 4 || "$ABITS" == 6 ]]
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; source "$PKG_ROOT/scripts/common_env.sh"
REPO="${REPO:?Set REPO to the integration dependency directory; see README.md}"; PYTHON_BIN="${PYTHON_BIN:-python}"; MODEL="${MODEL:?Set MODEL to your local input path; see README.md}"
MANIFEST="${MANIFEST:-$PKG_ROOT/configs/calibration_prompts/h3_complementary_3.json}"; CALIB_PROMPTS="${CALIB_PROMPTS:-$PKG_ROOT/configs/calibration_prompts/h3_complementary_3.json}"; PROFILE="${PROFILE:-$PKG_ROOT/configs/profiles/h3_10x8_h4.json}"; RUN_ROOT="${RUN_ROOT:-$PKG_ROOT/runs/minimax_h3/w4a$ABITS}"; CACHE="${CACHE:-$RUN_ROOT/cache/pulsequant.pt}"
test -s "$PROFILE" || { echo "missing model-specific H3 propagation profile: $PROFILE" >&2; echo "generate the 10-block x 8-timestep, H=4 pulse profile before calibration" >&2; exit 20; }
pulse_require_paths REPO MODEL MANIFEST CALIB_PROMPTS PROFILE; mkdir -p "$RUN_ROOT/cache" "$RUN_ROOT/calibration"; export PYTHONPATH="$PKG_ROOT/runtime/wan21:${ORBITQUANT_ROOT:-$REPO/third_party/orbitquant}:${H3_DIFFUSERS_ROOT:-$REPO/third_party/diffusers_h3/src}:$REPO${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}" H3_MATCH_ORBIT_POLICY=1 H3_STRUCTURED_SAMPLE=0 H3_EDGE_WEIGHT=0 H3_A_CLIP=1
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-0}" "$PYTHON_BIN" "$PKG_ROOT/runtime/h3/run_h3_w4_quant_suite.py" --model "$MODEL" --manifest "$MANIFEST" --method walsh2 --walsh-internal-method pulseprofile --propagation-profile "$PROFILE" --activation-bits "$ABITS" \
  --height 544 --width 960 --frames 124 --steps 31 --calibration-calls 8 --calibration-prompts-file "$CALIB_PROMPTS" --samples-per-call 4 --risk-lambda .75 --risk-tail-fraction .5 --trajectory-delta-weight 1 \
  --output-dir "$RUN_ROOT/calibration" --calibration-only --save-quant-cache "$CACHE"
test -s "$CACHE"; sha256sum "$CACHE" > "$CACHE.sha256"
