#!/usr/bin/env bash
set -uo pipefail
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")"

MODAL_COMMAND=(modal)
if [[ -x ".modal-client/Scripts/python.exe" ]]; then
  MODAL_COMMAND=(".modal-client/Scripts/python.exe" -m modal)
elif [[ -x ".modal-client/bin/python" ]]; then
  MODAL_COMMAND=(".modal-client/bin/python" -m modal)
fi

RUN_ID="${1:-long-horizon-60s-scheduled}"
OUTPUT_DIR="${2:-output/brainrot-v-long-horizon-60s}"
SUFFIX="${3:-}"

mkdir -p "$OUTPUT_DIR"

# Video, log and manifest are all part of the record. A run whose MP4 arrives
# without its log and manifest cannot be audited later.
ARTIFACTS=(
  "baseline|baseline.mp4|baseline-causal-forcing-60s${SUFFIX}.mp4"
  "baseline|baseline.log|baseline-causal-forcing-60s${SUFFIX}.log"
  "baseline|baseline.manifest.json|baseline-causal-forcing-60s${SUFFIX}.manifest.json"
  "baseline|baseline.schedule.json|baseline-causal-forcing-60s${SUFFIX}.schedule.json"
  "brainrot-v|brainrot_v.mp4|proposed-brainrot-v-60s${SUFFIX}.mp4"
  "brainrot-v|brainrot_v.log|proposed-brainrot-v-60s${SUFFIX}.log"
  "brainrot-v|brainrot_v.manifest.json|proposed-brainrot-v-60s${SUFFIX}.manifest.json"
  "brainrot-v|brainrot_v.schedule.json|proposed-brainrot-v-60s${SUFFIX}.schedule.json"
)

failed=()
for entry in "${ARTIFACTS[@]}"; do
  IFS='|' read -r arm remote local <<<"$entry"
  if ! "${MODAL_COMMAND[@]}" volume get brainrot-v-outputs \
      "$RUN_ID-$arm/$remote" "$OUTPUT_DIR/$local" --force; then
    failed+=("$RUN_ID-$arm/$remote")
  fi
done

if ((${#failed[@]})); then
  printf 'Warning: could not download: %s\n' "${failed[*]}" >&2
  # A missing MP4 is fatal; a missing log only degrades the record.
  for item in "${failed[@]}"; do
    [[ "$item" == *.mp4 ]] && { printf 'Required video artifact missing\n' >&2; exit 1; }
  done
fi

printf "Downloaded run '%s' into %s\n" "$RUN_ID" "$OUTPUT_DIR"
