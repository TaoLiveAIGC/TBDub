#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash infer.sh SOURCE_VIDEO DRIVING_AUDIO [OUTPUT_DIR]" >&2
  exit 2
fi

source_video="$1"
driving_audio="$2"
output_dir="${3:-results}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
checkpoint_dir="${TBDUB_CHECKPOINT_DIR:-$script_dir/checkpoints}"

python "$script_dir/inference.py" \
  --video "$source_video" \
  --audio "$driving_audio" \
  --dit-checkpoint \
    "$checkpoint_dir/tbdub_base.safetensors" \
    "$checkpoint_dir/tbdub_finetune.safetensors" \
  --ref-cfg-scale 2.0 \
  --audio-cfg-scale 6.0 \
  --num-inference-steps 30 \
  --seed 42 \
  --output-dir "$output_dir"
