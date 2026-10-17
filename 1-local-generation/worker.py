from __future__ import annotations

import argparse
import gc
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


class SequentialFluxPipeline:
    """Load the FLUX text encoder, DiT, and autoencoder in separate stages."""

    def __init__(self) -> None:
        from flux3.pipeline import MODELS, WEIGHTS, dit_path

        self.model_name = "distilled"
        self.dit_path = dit_path(self.model_name, "fp8")
        if not self.dit_path.exists():
            raise FileNotFoundError(f"Variante FP8 introuvable : {self.dit_path}")
        shared = WEIGHTS / MODELS[self.model_name]["dit"]
        self.text_encoder_path = shared / "text_encoder"
        self.autoencoder_path = shared / "ae.safetensors"

    def _load_autoencoder(self) -> Any:
        import torch
        from flux3.autoencoder import AutoEncoder
        from flux3.checkpoint import load_checkpoint

        return load_checkpoint(AutoEncoder(), str(self.autoencoder_path), "cpu", torch.float32)

    def _encode_references(self, references: list[Image.Image]) -> Any:
        if not references:
            return None
        import torch
        from flux3.pipeline import scaled_ids, to_tensor

        autoencoder = self._load_autoencoder()
        tokens, ids = [], []
        with torch.inference_mode():
            for index, reference in enumerate(references):
                encoded, positions = autoencoder.encode(to_tensor(reference).to("cpu"))
                positions = scaled_ids(positions)
                positions[:, 3] = -(1 + index)
                tokens.append(encoded.to(torch.bfloat16))
                ids.append(positions)
        result = (torch.cat(tokens), torch.cat(ids))
        del autoencoder, tokens, ids
        gc.collect()
        return result

    def generate(
        self,
        model: str,
        prompt: str,
        references: list[Image.Image],
        width: int,
        height: int,
        steps: int,
        guidance: float | None,
        seed: int,
        alpha: float = 6.93,
    ) -> Image.Image:
        import torch
        from flux3.autoencoder import latent_ids, latent_shape
        from flux3.dit import DiT, IN_CHANNELS, Packed
        from flux3.pipeline import MODELS, pack, scaled_ids, to_pil
        from flux3.sampling import denoise, denoise_flow_map, timesteps
        from flux3.text_encoder import TextEncoder
        from flux3.checkpoint import load_checkpoint

        if not torch.cuda.is_available():
            raise RuntimeError("FLUX 3 FP8 nécessite un GPU CUDA. Qwen reste disponible sur CPU/GPU.")

        # Encode the prompt on CPU, then release the large text encoder before touching VRAM.
        text_encoder = TextEncoder(str(self.text_encoder_path), "cpu")
        flow_map = MODELS[model]["guidance"] is None
        with torch.inference_mode():
            if flow_map:
                text = text_encoder.encode(prompt)
                text_samples = [text]
            else:
                text_samples = [text_encoder.encode(""), text_encoder.encode(prompt)]
        del text_encoder
        gc.collect()

        condition = self._encode_references(references)
        latent_height, latent_width = latent_shape(height, width)
        noise = torch.randn(
            (latent_height * latent_width, IN_CHANNELS["image"]),
            generator=torch.Generator().manual_seed(seed),
        )
        noise = noise.to(torch.bfloat16).float()
        image_tokens = (noise, scaled_ids(latent_ids(latent_height, latent_width, "cpu")))
        samples = [[text_samples[-1], condition, image_tokens]] if flow_map else [
            [text_samples[0], condition, image_tokens],
            [text_samples[1], condition, image_tokens],
        ]
        packed = pack(samples)
        schedule = timesteps(steps, alpha)

        # Keep the text encoder and VAE on CPU. Only the quantized DiT and working tensors
        # are moved to the GPU for the repeated denoising pass.
        with torch.device("meta"):
            dit = DiT(flow_map)
        dit = load_checkpoint(dit, str(self.dit_path), "cpu", torch.bfloat16)
        gpu_packed = None
        tokens = None
        try:
            dit.to("cuda")
            gpu_packed = Packed(
                data={name: tensor.to("cuda") for name, tensor in packed.data.items()},
                ids={name: tensor.to("cuda") for name, tensor in packed.ids.items()},
                timesteps={name: tensor.to("cuda") for name, tensor in packed.timesteps.items()},
                mode_indices=packed.mode_indices,
            )
            if flow_map:
                tokens = denoise_flow_map(dit, gpu_packed, schedule)
            else:
                if guidance is None:
                    raise ValueError("Le modèle FLUX Base nécessite un niveau de guidance.")
                tokens = denoise(dit, gpu_packed, schedule, guidance)
            tokens = tokens.to("cpu")
        finally:
            if gpu_packed is not None:
                del gpu_packed
            try:
                dit.to("cpu")
            except RuntimeError:
                pass
            del dit
            gc.collect()
            torch.cuda.empty_cache()

        # Decode after the DiT has left VRAM.
        autoencoder = self._load_autoencoder()
        with torch.inference_mode():
            decoded = autoencoder.decode(tokens, image_tokens[1])
        image = to_pil(decoded)
        del autoencoder, packed, condition, text_samples, samples, noise, tokens, decoded
        gc.collect()
        return image


def load_flux_pipeline() -> Any:
    from flux3.generate import generate as generate_image

    pipeline = SequentialFluxPipeline()
    return SimpleNamespace(generate=generate_image, models=SimpleNamespace(upsampler=None, pipeline=pipeline))


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
    server_version = "Tests-Worker/0.1"

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
            if self.server.model == "flux3" and payload.get("prompt_upsampling"):
                raise ValueError("L’agrandisseur de prompt n’est pas disponible dans le mode mémoire séquentiel.")
            reference_path = payload.get("reference")
            references = [Image.open(reference_path).convert("RGB")] if reference_path else []
            with MODEL_LOCK:
                if PIPELINE is None:
                    print("Chargement des poids du modèle…", flush=True)
                    if self.server.model == "flux3":
                        PIPELINE = load_flux_pipeline()
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
