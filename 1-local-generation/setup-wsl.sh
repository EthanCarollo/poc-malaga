#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="$ROOT/.runtime/flux-3-image-klein"

command -v python3.12 >/dev/null || { echo "Python 3.12 est requis dans WSL." >&2; exit 1; }
command -v git >/dev/null || { echo "Git est requis dans WSL." >&2; exit 1; }

mkdir -p "$ROOT/.runtime" "$ROOT/weights" "$ROOT/outputs"

if [ ! -d "$RUNTIME/.git" ]; then
  git clone https://github.com/black-forest-labs/flux-3-image-klein.git "$RUNTIME"
fi

python3.12 -m venv "$ROOT/.venv-flux3"
python3.12 -m venv "$ROOT/.venv-qwen"

"$ROOT/.venv-flux3/bin/python" -m pip install --upgrade pip
"$ROOT/.venv-flux3/bin/python" -m pip install -e "$RUNTIME" --extra-index-url https://download.pytorch.org/whl/cu130

"$ROOT/.venv-qwen/bin/python" -m pip install --upgrade pip
"$ROOT/.venv-qwen/bin/python" -m pip install \
  "torch>=2.4.0" \
  "transformers>=5.17" \
  accelerate \
  pillow \
  huggingface_hub \
  "git+https://github.com/huggingface/diffusers.git" \
  --extra-index-url https://download.pytorch.org/whl/cu130

cat > "$ROOT/runtime-config.json" <<'JSON'
{
  "runtimes": {
    "flux3": ".venv-flux3/bin/python",
    "qwen21": ".venv-qwen/bin/python"
  },
  "weights_dir": "weights",
  "outputs_dir": "outputs"
}
JSON

echo
echo "Environnements installés. Accepte les licences des modèles puis lance :"
echo "  hf auth login"
echo "  ./download-models.sh"
echo "  python3 server.py"
