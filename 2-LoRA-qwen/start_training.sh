#!/usr/bin/env bash
set -euo pipefail
SHARM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARM_PYTHON="$SHARM_DIR/../1-local-generation/.venv-qwen/bin/python"
exec "$SHARM_PYTHON" "$SHARM_DIR/run_training.py" --detach "$@"
