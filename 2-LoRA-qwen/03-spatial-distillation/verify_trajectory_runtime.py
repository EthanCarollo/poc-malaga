#!/usr/bin/env python3
"""Execute a complete 20-step real NF4 teacher trajectory across Euler updates."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
import numpy as np
import torch
from sharm_image21.core import FeatureCache, ORIGINAL, atomic_json, file_hash, read_jsonl
from sharm_image21.diffusion import load_transformer, predict_flow


@torch.no_grad()
def main():
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps
    rows = read_jsonl(ROOT / "artifacts/spatial/dataset/scenes.jsonl")
    row = sorted((r for r in rows if r["split"] == "train"), key=lambda r: r["id"])[0]
    resolution = json.loads((ROOT / "artifacts/spatial/dataset/manifest.json").read_text())["resolution"]
    reference = FeatureCache(ROOT / "cache/spatial/teacher").get(row["caption_plain_id"])[None].cuda()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ORIGINAL, subfolder="scheduler", local_files_only=True)
    mu = calculate_shift((resolution//16)**2, scheduler.config.get("base_image_seq_len", 256),
                         scheduler.config.get("max_image_seq_len", 4096), scheduler.config.get("base_shift", 0.5),
                         scheduler.config.get("max_shift", 1.15))
    timesteps = retrieve_timesteps(scheduler, 20, "cuda", sigmas=np.linspace(1, 1/20, 20), mu=mu)[0]
    model = load_transformer()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    generator = torch.Generator(device="cuda").manual_seed(700+int(row["id"][:8], 16))
    latent = torch.randn((1, 64, 1, resolution//16, resolution//16), device="cuda", dtype=torch.bfloat16, generator=generator)
    for step, timestep in enumerate(timesteps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            target = predict_flow(model, latent, reference, timestep[None].to(torch.bfloat16)/1000)
        if target.dtype != reference.dtype or not torch.isfinite(target).all():
            raise ValueError("Prédiction NF4 incompatible avec la précision des embeddings.")
        latent = scheduler.step(target, timestep, latent, return_dict=False)[0]
        if latent.dtype != reference.dtype or not torch.isfinite(latent).all():
            raise ValueError("Le scheduler a changé la précision ou produit des latents non finis.")
        print(f"Euler BF16 : {step+1}/20", flush=True)
    report = {"passed": True, "steps": 20, "resolution": resolution, "scene_id": row["id"],
              "latent_dtype": str(latent.dtype), "prediction_dtype": str(target.dtype),
              "seconds": time.monotonic()-started, "peak_gib": torch.cuda.max_memory_allocated()/2**30,
              "diffusion_sha256": file_hash(ROOT / "common/sharm_image21/diffusion.py"),
              "scope": "Full real NF4 Euler trajectory regression; not generated spatial quality"}
    atomic_json(ROOT / "artifacts/spatial/trajectory-runtime-audit.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
