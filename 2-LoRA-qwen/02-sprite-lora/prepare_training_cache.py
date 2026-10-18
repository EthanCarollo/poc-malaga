#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import numpy as np
import torch
from PIL import Image
from sharm_image21.core import (ARTIFACTS, CACHE, ORIGINAL, ROOT, TEACHER_REVISION,
                                FeatureCache, Projector, atomic_json, digest, file_hash, read_jsonl)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()
    resolution = args.resolution
    if resolution < 64 or resolution % 32:
        raise ValueError("Résolution multiple de 32 requise.")
    projector_dir = ARTIFACTS / "projector"
    fingerprint = file_hash(projector_dir / "model.safetensors")
    gate = json.loads((ARTIFACTS / "base-validation.json").read_text())
    if not gate.get("passed") or gate["projector_sha256"] != fingerprint:
        raise RuntimeError("La base Sharm doit être validée avant le cache LoRA.")
    dataset = ROOT / f"dataset/train_{resolution}"
    if not (dataset / "metadata.jsonl").is_file():
        raise FileNotFoundError(f"Préparer le dataset à {resolution}px avant l'entraînement.")
    with (ROOT / "dataset/manifest.csv").open() as stream:
        species = {Path(row["file"]).name: row["species_id"] for row in csv.DictReader(stream)}
    rows = []
    for row in read_jsonl(dataset / "metadata.jsonl"):
        image_path = dataset / row["file_name"]
        image_sha = file_hash(image_path)
        species_id = species[row["file_name"]]
        rows.append({**row, "id": digest({"name": row["file_name"], "sha256": image_sha}),
                     "image_sha256": image_sha, "species_id": species_id, "caption_id": digest(row["text"]),
                     "split": "validation" if int(digest(species_id)[:8], 16) % 10 == 0 else "train"})
    identity = {"dataset_sha256": digest(rows), "resolution": resolution,
                "projector_sha256": fingerprint, "teacher_revision": TEACHER_REVISION}
    student = FeatureCache(CACHE / "student")
    caption_rows = list({row["caption_id"]: row for row in rows}.values())
    missing = [row["caption_id"] for row in caption_rows if row["caption_id"] not in student]
    if missing:
        raise RuntimeError(f"{len(missing)} légendes sprites sont absentes du cache du 4B ; compléter le corpus.")
    projected = FeatureCache(CACHE / f"sprites-{resolution}-text", {**identity, "student": student.index["identity"]})
    model = Projector.load(projector_dir, "cuda").eval()
    buffer = []
    for row in caption_rows:
        key = row["caption_id"]
        if key in projected:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values = model(student.get(key).cuda()).to(torch.bfloat16).cpu()
        if not torch.isfinite(values).all():
            raise RuntimeError("Embeddings projetés non finis.")
        buffer.append((key, values, student.index["entries"][key]["token_ids"]))
        if len(buffer) == 32:
            projected.write(buffer)
            buffer = []
    projected.write(buffer)
    del model
    torch.cuda.empty_cache()
    latents = FeatureCache(CACHE / f"sprites-{resolution}-latents",
                           {**identity, "vae_config": file_hash(ORIGINAL / "vae/config.json"),
                            "sampling": "per-image-seed-42-bf16-rgba-normalized-v1"})
    pending = [row for row in rows if row["id"] not in latents]
    if pending:
        from diffusers import AutoencoderKLQwenImage21
        vae = AutoencoderKLQwenImage21.from_pretrained(ORIGINAL, subfolder="vae", torch_dtype=torch.bfloat16,
                                                      local_files_only=True).to("cuda").eval()
        vae.requires_grad_(False)
        mean = torch.tensor(vae.config.latents_mean, device="cuda", dtype=torch.bfloat16).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device="cuda", dtype=torch.bfloat16).view(1, -1, 1, 1, 1)
        buffer = []
        for index, row in enumerate(pending):
            with Image.open(dataset / row["file_name"]) as image:
                if image.mode != "RGBA" or image.size != (resolution, resolution):
                    raise ValueError(f"Image RGBA à la taille attendue requise : {row['file_name']}")
                pixels = np.array(image, copy=True)
            blue = (pixels[..., :3] == (165, 235, 255)).all(axis=-1) & (pixels[..., 3] != 0)
            if blue.any() or not (pixels[..., 3] == 0).any():
                raise ValueError(f"Fond bleu opaque ou alpha absent : {row['file_name']}")
            tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
            tensor = tensor.to("cuda", dtype=torch.bfloat16)/127.5-1
            generator = torch.Generator(device="cuda").manual_seed(42+int(row["id"][:8], 16))
            encoded = vae.encode(tensor).latent_dist.sample(generator=generator)
            normalized = ((encoded-mean)/std)[0].cpu().contiguous()
            if not torch.isfinite(normalized).all():
                raise RuntimeError("Latents non finis.")
            buffer.append((row["id"], normalized, []))
            if len(buffer) == 16 or index+1 == len(pending):
                latents.write(buffer)
                buffer = []
                print(f"Latents sprites : {len(latents.index['entries'])}/{len(rows)}", flush=True)
    latents.require(rows)
    result = {"identity": identity, "rows": rows, "text_cache": str(projected.directory),
              "latent_cache": str(latents.directory), "dataset_dir": str(dataset)}
    atomic_json(ARTIFACTS / f"sprite-training-{resolution}.json", result)
    print(json.dumps({"images": len(rows), "captions": len(caption_rows), "identity": identity}), flush=True)


if __name__ == "__main__":
    main()
