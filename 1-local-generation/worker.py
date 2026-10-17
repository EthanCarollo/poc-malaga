from __future__ import annotations

import argparse
import json
import os
import threading
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image


PIPELINE: Any = None
MODEL_LOCK = threading.Lock()
OUTPUT_DIR = Path("outputs")


def load_flux_pipeline(prompt_upsampling: bool) -> Any:
    from flux3.generate import generate
    from flux3.pipeline import MODELS, WEIGHTS, Pipeline, dit_path

    model_name = "distilled"
    dit = dit_path(model_name, "fp8")
    if not dit.exists():
        raise FileNotFoundError(f"Variante FP8 introuvable : {dit}")
    shared = WEIGHTS / MODELS[model_name]["dit"]
    upsampler = None
    if prompt_upsampling:
        from flux3.pup import Upsampler

        pup_path = WEIGHTS / "pup"
        if not pup_path.exists():
            raise FileNotFoundError(f"Agrandisseur de prompt absent : {pup_path}")
        upsampler = Upsampler(str(pup_path), memory=0.35)
    pipeline = Pipeline(
        {model_name: str(dit)},
        str(shared / "text_encoder"),
        str(shared / "ae.safetensors"),
        offload=True,
    )
    return SimpleNamespace(generate=generate, models=SimpleNamespace(upsampler=upsampler, pipeline=pipeline))


def load_qwen_pipeline() -> Any:
    import torch
    from diffusers import QwenImage21Pipeline

    weights_dir = Path(os.environ.get("WEIGHTS_DIR", "weights"))
    pipeline = QwenImage21Pipeline.from_pretrained(
        str(weights_dir / "qwen-image-2.1"),
        torch_dtype=torch.bfloat16,
    )
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory
        if total_vram < 20 * 1024**3:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to("cuda")
    return pipeline


class WorkerHandler(BaseHTTPRequestHandler):
    server_version = "POC-Malaga-Worker/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {"ok": True})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        global PIPELINE
        if self.path != "/generate":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            prompt = str(payload["prompt"]).strip()
            width, height = int(payload["width"]), int(payload["height"])
            seed = int(payload["seed"])
            reference_path = payload.get("reference")
            references = [Image.open(reference_path).convert("RGB")] if reference_path else []
            with MODEL_LOCK:
                if PIPELINE is None:
                    print("Chargement des poids du modèle…", flush=True)
                    if self.server.model == "flux3":
                        PIPELINE = load_flux_pipeline(bool(payload.get("prompt_upsampling")))
                    else:
                        PIPELINE = load_qwen_pipeline()
                if self.server.model == "flux3":
                    image, record = PIPELINE.generate(
                        PIPELINE.models,
                        "distilled",
                        prompt,
                        references,
                        width=width,
                        height=height,
                        steps=16,
                        seed=seed,
                        pup=bool(payload.get("prompt_upsampling")),
                    )
                else:
                    import torch

                    kwargs: dict[str, Any] = {
                        "prompt": prompt,
                        "width": width,
                        "height": height,
                        "num_inference_steps": 40,
                        "generator": torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed),
                    }
                    if references:
                        kwargs["image"] = references[0]
                    image = PIPELINE(**kwargs).images[0]
                    record = {"model": "Qwen-Image-2.1", "seed": seed, "width": width, "height": height}

            created = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            filename = f"{self.server.model}-{created}-{uuid.uuid4().hex[:8]}.png"
            image.save(OUTPUT_DIR / filename)
            (OUTPUT_DIR / f"{Path(filename).stem}.json").write_text(
                json.dumps({"record": record, "input_prompt": prompt}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.send_json(200, {"filename": filename, "record": record})
        except Exception as error:
            traceback.print_exc()
            self.send_json(500, {"error": f"{type(error).__name__}: {error}"})


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("flux3", "qwen21"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    args = parser.parse_args()
    OUTPUT_DIR = args.outputs.resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = WorkerServer(("127.0.0.1", args.port), WorkerHandler)
    server.model = args.model
    print(f"Moteur {args.model} disponible sur le port local {args.port}.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
