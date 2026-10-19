#!/usr/bin/env bash
set -euo pipefail
umask 077
SHARM_REMOTE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SHARM_REMOTE_ARTIFACTS="$SHARM_REMOTE_ROOT/2-LoRA-qwen/artifacts/spatial"
mkdir -p "$SHARM_REMOTE_ARTIFACTS" "$SHARM_REMOTE_ROOT/.tmp"
function sharm_remote_failed() {
  SHARM_REMOTE_EXIT=$?
  python3 -c 'import json,sys,time; from pathlib import Path; p=Path(sys.argv[1]); r=json.loads(p.read_text()) if p.exists() else {}; r.update(state="failed",exit_code=int(sys.argv[2]),updated_at=time.time()); p.write_text(json.dumps(r))' "$SHARM_REMOTE_ARTIFACTS/remote-status.json" "$SHARM_REMOTE_EXIT"
  exit "$SHARM_REMOTE_EXIT"
}
trap sharm_remote_failed ERR
exec 9>"$SHARM_REMOTE_ARTIFACTS/remote-bootstrap.lock"
flock -n 9 || { echo "La chaîne distante est déjà active."; exit 1; }
export PATH="$SHARM_REMOTE_ROOT/tools:$PATH"
export UV_PYTHON_INSTALL_DIR="$SHARM_REMOTE_ROOT/.python"
export UV_CACHE_DIR="$SHARM_REMOTE_ROOT/.cache/uv"
export PIP_NO_CACHE_DIR=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export TMPDIR="$SHARM_REMOTE_ROOT/.tmp"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$SHARM_REMOTE_ROOT"
python3 -c 'import json,time; from pathlib import Path; p=Path("2-LoRA-qwen/artifacts/spatial/remote-status.json"); p.write_text(json.dumps({"state":"running","stage":"install_runtime","updated_at":time.time()}))'
SHARM_REMOTE_PYTHON="$SHARM_REMOTE_ROOT/1-local-generation/.venv-qwen/bin/python"
if [ ! -f "$SHARM_REMOTE_ARTIFACTS/runtime-ready.json" ]; then
  uv python install 3.12
  bash "$SHARM_REMOTE_ROOT/2-LoRA-qwen/setup_sharm.sh"
  "$SHARM_REMOTE_PYTHON" -m pip check
  "$SHARM_REMOTE_PYTHON" -c 'import json,torch; from pathlib import Path; assert torch.cuda.is_available(); Path("2-LoRA-qwen/artifacts/spatial/runtime-ready.json").write_text(json.dumps({"torch":torch.__version__,"gpu":torch.cuda.get_device_name(),"cuda":torch.version.cuda}))'
fi
python3 -c 'import json,time; from pathlib import Path; p=Path("2-LoRA-qwen/artifacts/spatial/remote-status.json"); p.write_text(json.dumps({"state":"running","stage":"cpu_tests","updated_at":time.time()}))'
"$SHARM_REMOTE_PYTHON" -m unittest discover -s 2-LoRA-qwen/tests -v
exec "$SHARM_REMOTE_PYTHON" -u "$SHARM_REMOTE_ROOT/2-LoRA-qwen/03-spatial-distillation/run_remote.py"
