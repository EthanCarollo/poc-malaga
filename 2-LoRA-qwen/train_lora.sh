#!/usr/bin/env bash
set -euo pipefail
SHARM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARM_PYTHON="$SHARM_DIR/../1-local-generation/.venv-qwen/bin/python"
SHARM_RESOLUTION="${TRAINING_RESOLUTION:-512}"
SHARM_STEPS="${MAX_TRAIN_STEPS:-500}"
SHARM_OUTPUT="${LORA_OUTPUT_DIR:-$SHARM_DIR/../1-local-generation/weights/sharm-image-2.1-sprites-128-lora}"
"$SHARM_PYTHON" "$SHARM_DIR/02-sprite-lora/prepare_training_cache.py" --resolution "$SHARM_RESOLUTION"
exec "$SHARM_PYTHON" "$SHARM_DIR/02-sprite-lora/train_lora.py" --resolution "$SHARM_RESOLUTION" --steps "$SHARM_STEPS" --output "$SHARM_OUTPUT" "$@"
