#!/usr/bin/env bash
set -euo pipefail
SHARM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARM_PROJECT="$(cd "$SHARM_DIR/.." && pwd)"
SHARM_PYTHON="$SHARM_PROJECT/1-local-generation/.venv-qwen/bin/python"
SHARM_DIFFUSERS="$SHARM_PROJECT/1-local-generation/.runtime/diffusers-qwen21-lora"
SHARM_DIFFUSERS_REVISION="b800f828f3951c4d206e154411bcbfa693cd322e"
if [ ! -x "$SHARM_PYTHON" ]; then
  "$SHARM_PROJECT/1-local-generation/setup-wsl.sh"
fi
if [ ! -d "$SHARM_DIFFUSERS/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/huggingface/diffusers.git "$SHARM_DIFFUSERS"
  git -C "$SHARM_DIFFUSERS" checkout "$SHARM_DIFFUSERS_REVISION"
fi
if [ "$(git -C "$SHARM_DIFFUSERS" rev-parse HEAD)" != "$SHARM_DIFFUSERS_REVISION" ]; then
  echo "Le clone Diffusers doit être à la révision $SHARM_DIFFUSERS_REVISION. Le checkout existant est conservé." >&2
  exit 1
fi
"$SHARM_PYTHON" -m pip install -r "$SHARM_DIR/requirements-sharm.txt"
"$SHARM_PYTHON" -m pip install -e "$SHARM_DIFFUSERS"
"$SHARM_PYTHON" -c 'from diffusers import QwenImage21Pipeline; from transformers import Qwen3VLForConditionalGeneration; import kernels'
echo "Runtime Sharm prêt. Lancement : $SHARM_DIR/start_training.sh"
