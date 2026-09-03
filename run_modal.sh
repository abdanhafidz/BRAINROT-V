#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1

cd "$(dirname "$0")"

MODAL_COMMAND=(modal)
if [[ -x ".modal-client/Scripts/python.exe" ]]; then
  MODAL_COMMAND=(".modal-client/Scripts/python.exe" -m modal)
elif [[ -x ".modal-client/bin/python" ]]; then
  MODAL_COMMAND=(".modal-client/bin/python" -m modal)
elif ! command -v modal >/dev/null 2>&1; then
  echo "Modal CLI is missing. Install/authenticate once with: pip install modal && modal setup" >&2
  exit 1
fi

PROMPT="${1:-A cinematic tracking shot of a red fox running through a misty pine forest at sunrise}"
OUTPUT="${2:-output/modal-causal-forcing.mp4}"

"${MODAL_COMMAND[@]}" run modal_app.py --prompt "$PROMPT" --output "$OUTPUT"
