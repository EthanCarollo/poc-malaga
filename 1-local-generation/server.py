from __future__ import annotations

import atexit
import base64
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "runtime-config.json"
DEFAULT_CONFIG: dict[str, Any] = {
    "runtimes": {
        "flux3": ".venv-flux3/bin/python",
        "qwen21": ".venv-qwen/bin/python",
    },
    "weights_dir": "weights",
    "outputs_dir": "outputs",
}
MODEL_INFO = {
    "flux3": {"label": "FLUX 3 Image Klein", "precision": "FP8", "steps": 16},
    "qwen21": {"label": "Qwen-Image-2.1", "precision": "BF16", "steps": 40},
}
RESOLUTIONS = {
    "512x512": (512, 512),
    "1024x1024": (1024, 1024),
    "832x1216": (832, 1216),
    "1216x832": (1216, 832),
}
MAX_REQUEST_BYTES = 32 * 1024 * 1024
MAX_REFERENCE_BYTES = 20 * 1024 * 1024


def load_config() -> dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if CONFIG_PATH.exists():
        saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        for key in ("runtimes", "weights_dir", "outputs_dir"):
            if key in saved:
                config[key] = saved[key]
    if sys.platform == "win32":
        for key, value in list(config["runtimes"].items()):
            if value.endswith("/bin/python"):
                config["runtimes"][key] = value.replace("/bin/python", "/Scripts/python.exe")
    return config


CONFIG = load_config()
OUTPUT_DIR = (PROJECT_DIR / CONFIG["outputs_dir"]).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INCOMING_DIR = OUTPUT_DIR / ".incoming"
INCOMING_DIR.mkdir(exist_ok=True)
HISTORY_PATH = OUTPUT_DIR / "history.json"


def configured_python(model: str) -> str | None:
    value = str(CONFIG["runtimes"].get(model, "")).strip()
    if not value:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_DIR / candidate
    if candidate.is_file():
        return str(candidate.resolve())
    return shutil.which(value)


def gpu_info() -> list[dict[str, Any]]:
    command = shutil.which("nvidia-smi")
    if not command:
        return []
    try:
        result = subprocess.run(
            [command, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        devices = []
        for line in result.stdout.splitlines():
            name, memory = (part.strip() for part in line.split(",", 1))
            devices.append({"name": name, "memory_mb": int(memory)})
        return devices
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


class WorkerManager:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.model: str | None = None
        self.port: int | None = None
        self.log_file: Any = None

    @staticmethod
    def free_port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def stop(self) -> None:
        process, self.process = self.process, None
        self.model, self.port = None, None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def ensure(self, model: str) -> int:
        if model not in MODEL_INFO:
            raise ValueError("Modèle inconnu.")
        with self.lock:
            if self.process and self.process.poll() is None and self.model == model and self.port:
                return self.port
            self.stop()
            python = configured_python(model)
            if not python:
                shown = CONFIG["runtimes"].get(model, "")
                raise RuntimeError(
                    f"Runtime Python introuvable pour {MODEL_INFO[model]['label']}. "
                    f"Configure le chemin « {shown} » dans runtime-config.json après l’installation."
                )

            port = self.free_port()
            log_path = OUTPUT_DIR / f"worker-{model}.log"
            self.log_file = log_path.open("a", encoding="utf-8")
            env = os.environ.copy()
            weights_dir = Path(CONFIG["weights_dir"]).expanduser()
            if not weights_dir.is_absolute():
                weights_dir = PROJECT_DIR / weights_dir
            env["WEIGHTS_DIR"] = str(weights_dir.resolve())
            env["HF_HOME"] = env.get("HF_HOME", str((PROJECT_DIR / ".runtime" / "huggingface").resolve()))
            script = PROJECT_DIR / "worker.py"
            self.process = subprocess.Popen(
                [python, str(script), "--model", model, "--port", str(port), "--outputs", str(OUTPUT_DIR)],
                cwd=PROJECT_DIR,
                env=env,
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.model, self.port = model, port
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    log = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
                    self.stop()
                    raise RuntimeError(f"Le moteur {MODEL_INFO[model]['label']} n’a pas démarré.\n{log}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as response:
                        if response.status == 200:
                            return port
                except (OSError, urllib.error.URLError):
                    time.sleep(0.2)
            log = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
            self.stop()
            raise RuntimeError(f"Le moteur local met trop de temps à démarrer.\n{log}")


WORKERS = WorkerManager()
INFERENCE_LOCK = threading.Lock()
atexit.register(WORKERS.stop)


def read_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def save_history(entries: list[dict[str, Any]]) -> None:
    HISTORY_PATH.write_text(json.dumps(entries[:100], ensure_ascii=False, indent=2), encoding="utf-8")


class AppHandler(BaseHTTPRequestHandler):
    server_version = "Tests/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def send_file(self, path: Path) -> None:
        raw = path.read_bytes()
        mime = "image/png" if path.suffix.lower() == ".png" else "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-cache" if path.suffix in {".html", ".css", ".js"} else "public, max-age=3600")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        if path == "/api/health":
            devices = gpu_info()
            self.send_json(200, {
                "application": "poc-malaga",
                "ok": True,
                "gpu": devices,
                "runtimes": {model: configured_python(model) is not None for model in MODEL_INFO},
                "active_model": WORKERS.model if WORKERS.process and WORKERS.process.poll() is None else None,
                "flux_vram_warning": any(device["memory_mb"] < 26 * 1024 for device in devices),
            })
            return
        if path == "/api/history":
            self.send_json(200, {"items": read_history()})
            return
        if path.startswith("/outputs/"):
            name = Path(urllib.parse.unquote(path.removeprefix("/outputs/"))).name
            target = (OUTPUT_DIR / name).resolve()
            if target.is_file() and target.parent == OUTPUT_DIR:
                self.send_file(target)
            else:
                self.send_error(404)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        if path == "/api/shutdown":
            self.send_json(200, {"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if path != "/api/generate":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("La requête est vide ou trop volumineuse (24 Mo maximum).")
            payload = json.loads(self.rfile.read(length))
            model = str(payload.get("model", ""))
            if model not in MODEL_INFO:
                raise ValueError("Choisis un modèle disponible.")
            prompt = str(payload.get("prompt", "")).strip()
            if not prompt:
                raise ValueError("Écris un prompt avant de lancer la génération.")
            if len(prompt) > 1000:
                raise ValueError("Le prompt est limité à 1 000 caractères.")
            resolution = str(payload.get("resolution", "1024x1024"))
            if resolution not in RESOLUTIONS:
                raise ValueError("Format d’image invalide.")
            width, height = RESOLUTIONS[resolution]
            seed = int(payload.get("seed", -1))
            if seed < 0:
                seed = random.SystemRandom().randrange(0, 2**32)
            reference_path = None
            reference_data = payload.get("reference_image")
            if reference_data:
                if not isinstance(reference_data, str) or "," not in reference_data:
                    raise ValueError("Image de référence invalide.")
                header, encoded = reference_data.split(",", 1)
                if len(encoded) > MAX_REFERENCE_BYTES * 4 // 3:
                    raise ValueError("L’image de référence dépasse 20 Mo.")
                raw = base64.b64decode(encoded, validate=True)
                if len(raw) > MAX_REFERENCE_BYTES:
                    raise ValueError("L’image de référence dépasse 20 Mo.")
                mime = header.split(";", 1)[0].removeprefix("data:")
                ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(mime)
                if not ext:
                    raise ValueError("Formats acceptés : PNG, JPEG ou WebP.")
                reference_path = INCOMING_DIR / f"{uuid.uuid4().hex}{ext}"
                reference_path.write_bytes(raw)

            request_payload = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "seed": seed,
                "reference": str(reference_path) if reference_path else None,
                "prompt_upsampling": bool(payload.get("prompt_upsampling", False)),
            }
            try:
                with INFERENCE_LOCK:
                    port = WORKERS.ensure(model)
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{port}/generate",
                        data=json.dumps(request_payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=60 * 60) as response:
                        generated = json.loads(response.read())
            finally:
                if reference_path:
                    reference_path.unlink(missing_ok=True)

            entry = {
                "id": uuid.uuid4().hex,
                "model": model,
                "model_label": MODEL_INFO[model]["label"],
                "precision": MODEL_INFO[model]["precision"],
                "prompt": prompt,
                "resolution": resolution,
                "seed": seed,
                "image": f"/outputs/{generated['filename']}",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            entries = [entry, *read_history()]
            save_history(entries)
            self.send_json(200, {"item": entry})
        except urllib.error.HTTPError as error:
            try:
                details = json.loads(error.read())
                message = details.get("error", str(error))
            except (json.JSONDecodeError, OSError):
                message = str(error)
            self.send_json(502, {"error": message})
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.send_json(400, {"error": str(error)})
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            self.send_json(503, {"error": str(error)})
        except Exception as error:
            traceback.print_exc()
            self.send_json(500, {"error": f"La génération a échoué : {error}"})


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    server = LocalServer(("127.0.0.1", 7860), AppHandler)
    print("Tests — API locale prête sur 127.0.0.1:7860")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt de l’application Tests.")
    finally:
        server.server_close()
        WORKERS.stop()


if __name__ == "__main__":
    main()
