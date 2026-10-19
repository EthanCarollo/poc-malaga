#!/usr/bin/env bash
set -euo pipefail
SHARM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARM_PROJECT="$(cd "$SHARM_DIR/.." && pwd)"
SHARM_PYTHON="$SHARM_PROJECT/1-local-generation/.venv-qwen/bin/python"
SHARM_DIFFUSERS="$SHARM_PROJECT/1-local-generation/.runtime/diffusers-qwen21-lora"
SHARM_DIFFUSERS_REVISION="b800f828f3951c4d206e154411bcbfa693cd322e"
if ! command -v "${CC:-cc}" >/dev/null; then
  echo "Un compilateur C est requis par Triton FP8 : installer build-essential dans Linux/WSL." >&2
  exit 1
fi
if [ ! -x "$SHARM_PYTHON" ]; then
  if command -v python3.12 >/dev/null; then
    python3.12 -m venv "$SHARM_PROJECT/1-local-generation/.venv-qwen"
  elif command -v uv >/dev/null; then
    uv venv --python 3.12 --seed "$SHARM_PROJECT/1-local-generation/.venv-qwen"
  else
    echo "Installer Python 3.12 ou uv dans WSL pour créer le runtime Sharm." >&2
    exit 1
  fi
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
"$SHARM_PYTHON" -c 'from diffusers import QwenImage21Pipeline; from transformers import Qwen3VLForConditionalGeneration; import kernels, torchvision'
SHARM_KERNEL_READY=0
for SHARM_KERNEL_ATTEMPT in 1 2 3; do
  if "$SHARM_PYTHON" - <<'PY'
import os
from huggingface_hub import snapshot_download
from kernels import get_kernel
repo = "kernels-community/finegrained-fp8"
get_kernel(repo, version=4)
# The complete v4 snapshot is small (<1 MB). Cache its version alias and
# completion record too, which the Hub requires for fully offline loading.
snapshot_download(repo_id=repo, repo_type="kernel", revision="v4",
                  cache_dir=os.environ.get("KERNELS_CACHE"), max_workers=2)
PY
  then
    SHARM_KERNEL_READY=1
    break
  fi
  sleep 3
done
if [ "$SHARM_KERNEL_READY" != 1 ]; then
  echo "Impossible de préparer le kernel FP8 vérifié depuis Hugging Face." >&2
  exit 1
fi
echo "Runtime Sharm prêt. Lancement : $SHARM_DIR/start_training.sh"
