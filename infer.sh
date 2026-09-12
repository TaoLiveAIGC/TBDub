#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash infer.sh SOURCE_VIDEO DRIVING_AUDIO [OUTPUT_DIR] [--cpu-offload]" >&2
  exit 2
}

[[ $# -ge 2 ]] || usage

source_video="$1"
driving_audio="$2"
shift 2
output_dir="results"
if [[ $# -gt 0 && "$1" != --* ]]; then
  output_dir="$1"
  shift
fi
memory_args=()
if [[ $# -gt 0 ]]; then
  [[ $# -eq 1 && "$1" == "--cpu-offload" ]] || usage
  memory_args=(--cpu-offload)
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
checkpoint_dir="${TBDUB_CHECKPOINT_DIR:-$script_dir/checkpoints}"

python "$script_dir/inference.py" \
  --video "$source_video" \
  --audio "$driving_audio" \
  --checkpoint-dir "$checkpoint_dir" \
  --inference-mode teacher \
  --seed 42 \
  --output-dir "$output_dir" \
  "${memory_args[@]}"
