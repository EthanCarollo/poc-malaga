#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS="$ROOT/weights"
HF="$ROOT/.venv-qwen/bin/hf"

if [ ! -x "$HF" ]; then
  HF="$ROOT/.venv-flux3/bin/hf"
fi
if [ ! -x "$HF" ]; then
  echo "Lance d’abord ./setup-wsl.sh pour installer Hugging Face Hub." >&2
  exit 1
fi

mkdir -p "$WEIGHTS/flux-3-image-klein" "$WEIGHTS/pup" "$WEIGHTS/qwen-image-2.1"

echo "Téléchargement de FLUX 3 Image Klein, y compris la variante FP8…"
"$HF" download black-forest-labs/flux-3-image-klein --local-dir "$WEIGHTS/flux-3-image-klein"

echo "Téléchargement de l’agrandisseur de prompt FLUX (facultatif dans l’interface)…"
"$HF" download black-forest-labs/flux-3-image-klein-pup --local-dir "$WEIGHTS/pup"

echo "Téléchargement de Qwen-Image-2.1…"
"$HF" download Qwen/Qwen-Image-2.1 --local-dir "$WEIGHTS/qwen-image-2.1"

echo "Poids enregistrés dans $WEIGHTS"
