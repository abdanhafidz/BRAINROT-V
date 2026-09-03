#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1
cd "$(dirname "$0")"

MODAL_COMMAND=(modal)
if [[ -x ".modal-client/Scripts/python.exe" ]]; then
  MODAL_COMMAND=(".modal-client/Scripts/python.exe" -m modal)
elif [[ -x ".modal-client/bin/python" ]]; then
  MODAL_COMMAND=(".modal-client/bin/python" -m modal)
fi

PROMPT="${1:-A lone astronaut walking through a neon market in the rain}"
OUTPUT="${2:-output/brainrot-v.mp4}"
NUM_FRAMES="${3:-21}"
SAMPLING_STEPS="${4:-50}"
MODE="${5:-brainrot_v}"

"${MODAL_COMMAND[@]}" run modal_brainrot.py::app.main --prompt "$PROMPT" \
  --output "$OUTPUT" --num-frames "$NUM_FRAMES" \
  --sampling-steps "$SAMPLING_STEPS" --mode "$MODE"
