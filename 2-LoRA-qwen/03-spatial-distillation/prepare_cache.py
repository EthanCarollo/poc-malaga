#!/usr/bin/env python3
"""Cache RGBA VAE targets or teacher flows along actual Euler trajectories."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
import numpy as np
import torch
from PIL import Image
from sharm_image21.core import ORIGINAL, TEACHER_REVISION, FeatureCache, atomic_json, digest, file_hash, read_jsonl
from sharm_image21.diffusion import load_transformer, predict_flow


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("latents", "trajectories"))
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/spatial/dataset")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache/spatial")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--points", type=int, default=3)
    parser.add_argument("--max-scenes", type=int, default=512, help="Budget des trajectoires, indépendant du corpus supervisé.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA et les poids Qwen locaux sont requis pour ce cache.")
    rows = read_jsonl(args.dataset / "scenes.jsonl")
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    if digest(rows) != manifest["dataset_sha256"]:
        raise ValueError("Scènes modifiées depuis le manifeste.")
    identity = {"dataset_sha256": digest(rows), "teacher_revision": TEACHER_REVISION,
                "resolution": manifest["resolution"]}
    resolution = manifest["resolution"]
    if args.kind == "latents":
        from diffusers import AutoencoderKLQwenImage21
        identity.update(vae_config=file_hash(ORIGINAL / "vae/config.json"), sampling="mode-normalized-rgba-v1")
        cache = FeatureCache(args.cache / "latents", identity)
        pending = [r for r in rows if r["id"] not in cache]
        if not pending:
            return
        vae = AutoencoderKLQwenImage21.from_pretrained(ORIGINAL, subfolder="vae", torch_dtype=torch.bfloat16,
                                                      local_files_only=True).cuda().eval().requires_grad_(False)
        mean = torch.tensor(vae.config.latents_mean, device="cuda", dtype=torch.bfloat16).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device="cuda", dtype=torch.bfloat16).view(1, -1, 1, 1, 1)
        for index, row in enumerate(pending):
            path = args.dataset / row["file_name"]
            if file_hash(path) != row["image_sha256"]:
                raise ValueError(f"Image modifiée : {path}")
            with Image.open(path) as image:
                if image.mode != "RGBA" or image.size != (resolution, resolution):
                    raise ValueError(f"Image RGBA à {resolution}px requise : {path}")
                pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)[None, :, None]
            tensor = pixels.cuda().to(torch.bfloat16)/127.5-1
            values = ((vae.encode(tensor).latent_dist.mode()-mean)/std)[0].cpu().contiguous()
            if not torch.isfinite(values).all():
                raise RuntimeError("Latents VAE non finis.")
            cache.write([(row["id"], values, [])])
            if index % 25 == 0:
                print(f"Latents RGBA : {len(cache.index['entries'])}/{len(rows)}", flush=True)
        return
    if not 1 <= args.points <= args.steps or args.steps < 2 or args.max_scenes < 2:
        raise ValueError("De 1 à steps points ; au moins deux pas de trajectoire.")
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps
    teacher = FeatureCache(args.cache / "teacher")
    teacher.require([{"id": r.get("caption_plain_id", r["caption_id"])} for r in rows])
    # Cache teacher trajectories on a fixed subset rather than regenerating the
    # entire corpus. All compositions still contribute supervised image targets.
    train_rows = sorted((r for r in rows if r["split"] == "train"), key=lambda r: r["id"])
    valid_rows = sorted((r for r in rows if r["split"] == "validation"), key=lambda r: r["id"])
    train_count = max(1, int(args.max_scenes*0.8))
    scenes = train_rows[:train_count]+valid_rows[:max(1, args.max_scenes-train_count)]
    identity.update(teacher_index_sha256=file_hash(teacher.path), steps=args.steps, points=args.points,
                    selected_scenes=[r["id"] for r in scenes],
                    teacher_prompt_variant="natural-text-without-coordinates",
                    transformer_config=file_hash(ORIGINAL / "transformer/config.json"),
                    precision="NF4-double-BF16", sampler="Euler-CFG1-no-KV-cache-v2-bf16-output", seed_policy="700+scene-id",
                    diffusion_sha256=file_hash(ROOT / "common/sharm_image21/diffusion.py"))
    noisy = FeatureCache(args.cache / "trajectory-latents", identity)
    flows = FeatureCache(args.cache / "trajectory-flows", identity)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ORIGINAL, subfolder="scheduler", local_files_only=True)
    mu = calculate_shift((resolution//16)**2, scheduler.config.get("base_image_seq_len", 256),
                         scheduler.config.get("max_image_seq_len", 4096), scheduler.config.get("base_shift", 0.5),
                         scheduler.config.get("max_shift", 1.15))
    def reset_schedule():
        return retrieve_timesteps(scheduler, args.steps, "cuda", sigmas=np.linspace(1, 1/args.steps, args.steps), mu=mu)[0]
    schedule = reset_schedule()
    # Test prompts are never used to prepare training targets or select checkpoints.
    records = []
    for row in scenes:
        selected = sorted(random.Random(int(row["id"][:8], 16)).sample(range(args.steps), args.points))
        records.extend({"id": f"{row['id']}-{step}", "scene_id": row["id"], "step": step,
                        "split": row["split"], "timestep": float((schedule[step].to(torch.bfloat16)/1000).item())}
                       for step in selected)
    if all(r["id"] in noisy and r["id"] in flows for r in records):
        atomic_json(args.cache / "trajectories.json", {"identity": identity, "rows": records})
        return
    model = load_transformer()
    by_scene = {}
    for record in records:
        by_scene.setdefault(record["scene_id"], {})[record["step"]] = record
    for index, row in enumerate(scenes):
        wanted = by_scene[row["id"]]
        if all(r["id"] in noisy and r["id"] in flows for r in wanted.values()):
            continue
        reference = teacher.get(row.get("caption_plain_id", row["caption_id"]))[None].cuda()
        generator = torch.Generator(device="cuda").manual_seed(700+int(row["id"][:8], 16))
        latent = torch.randn((1, 64, 1, resolution//16, resolution//16), device="cuda", dtype=torch.bfloat16, generator=generator)
        timesteps = reset_schedule()
        for step, timestep in enumerate(timesteps):
            t = timestep.expand(1).to(device="cuda", dtype=torch.bfloat16)/1000
            with torch.autocast("cuda", dtype=torch.bfloat16):
                target = predict_flow(model, latent, reference, t)
            if not torch.isfinite(target).all():
                raise RuntimeError("Prédiction enseignant non finie.")
            if step in wanted:
                record = wanted[step]
                record["timestep"] = float(t.item())
                noisy.write([(record["id"], latent[0].cpu().contiguous(), [])])
                flows.write([(record["id"], target[0].cpu().contiguous(), [])])
            latent = scheduler.step(target, timestep, latent, return_dict=False)[0]
        atomic_json(args.cache / "trajectories.json", {"identity": identity, "rows": records})
        print(f"Trajectoires enseignant : {index+1}/{len(scenes)}", flush=True)
    noisy.require(records)
    flows.require(records)


if __name__ == "__main__":
    main()
