#!/usr/bin/env python3
"""Use a local spatial candidate with the actual frozen FP8 encoder."""
import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
import torch
from sharm_image21.core import FrozenEncoder, Projector, file_hash
from sharm_image21.diffusion import load_diffuser_pipeline
from sharm_image21.image_io import save_png
from sharm_image21.spatial_data import layout_prompt


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--layout", type=Path, help="JSON : objects [{label,box}], mêmes numéros que dans le prompt.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    if args.resolution < 64 or args.resolution % 32 or args.steps < 1:
        raise ValueError("Résolution multiple de 32 et nombre de pas positif requis.")
    config = json.loads((args.candidate / "candidate.json").read_text())
    if file_hash(args.candidate / "projector/model.safetensors") != config["projector_sha256"]:
        raise ValueError("Poids du projecteur modifiés.")
    prompt, boxes = args.prompt, None
    if args.layout:
        objects = json.loads(args.layout.read_text(encoding="utf-8"))["objects"]
        prompt, coordinates = layout_prompt(prompt, objects)
        boxes = torch.tensor(coordinates, device="cuda", dtype=torch.bfloat16)
    encoder = FrozenEncoder("student")
    if encoder.identity != config["identity"]["student"]["encoder"]:
        raise ValueError("Encodeur différent de celui de l'entraînement.")
    source, _ = encoder.encode([prompt])[0]
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    projector = Projector.load(args.candidate / "projector", "cuda").eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        embeddings = projector(source.cuda(), layout=boxes)[None].to(torch.bfloat16)
    del projector
    gc.collect()
    torch.cuda.empty_cache()
    pipeline = load_diffuser_pipeline()
    if config["lora_sha256"]:
        if file_hash(args.candidate / "lora/pytorch_lora_weights.safetensors") != config["lora_sha256"]:
            raise ValueError("Poids du LoRA modifiés.")
        pipeline.load_lora_weights(args.candidate / "lora", adapter_name="spatial")
    image = pipeline(prompt_embeds=embeddings, width=args.resolution, height=args.resolution, output_resolution=args.resolution,
                     num_inference_steps=args.steps, true_cfg_scale=1.0, use_kv_cache=False,
                     generator=torch.Generator(device="cuda").manual_seed(args.seed)).images[0]
    if image.mode != "RGBA":
        raise ValueError("Sortie RGBA native requise.")
    save_png(image, args.output, preview_path=args.output.with_name(args.output.stem+"-preview.png"), require_transparency=True)


if __name__ == "__main__":
    main()
