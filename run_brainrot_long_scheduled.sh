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

PROMPT_PATH="${1:-prompts/brainrot_v_long_horizon_60s.txt}"
RUN_ID="${2:-long-horizon-60s-scheduled}"
DURATION_SECONDS="${3:-60}"
SAMPLING_STEPS="${4:-50}"
SEED="${5:-0}"

COMMON=(run --detach --quiet modal_brainrot.py::app.submit_one_long
  --prompt-path "$PROMPT_PATH"
  --duration-seconds "$DURATION_SECONDS"
  --sampling-steps "$SAMPLING_STEPS"
  --seed "$SEED")

"${MODAL_COMMAND[@]}" "${COMMON[@]}" --mode baseline \
  --run-id "$RUN_ID-baseline"
"${MODAL_COMMAND[@]}" "${COMMON[@]}" --mode brainrot_v \
  --run-id "$RUN_ID-brainrot-v"

printf 'Submitted independent persistent workers.\n'
printf 'Baseline volume path: %s-baseline/baseline.mp4\n' "$RUN_ID"
printf 'Brainrot-V volume path: %s-brainrot-v/brainrot_v.mp4\n' "$RUN_ID"
