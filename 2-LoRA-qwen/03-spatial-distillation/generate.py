#!/usr/bin/env python3
"""Generate an auditable spatial benchmark using paired seeds and shared diffusion."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
import torch
from sharm_image21.core import FeatureCache, Projector, atomic_json, digest, file_hash, read_jsonl
from sharm_image21.diffusion import load_diffuser_pipeline
from sharm_image21.image_io import save_png


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True, help="Dossier best contenant candidate.json et projector/.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/spatial/dataset")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache/spatial")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--baseline-projector", type=Path)
    parser.add_argument("--include-teacher", action="store_true")
    args = parser.parse_args()
    if min(args.limit, args.steps) < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Limite/pas positifs et graines distinctes requis.")
    candidate = json.loads((args.candidate / "candidate.json").read_text())
    if candidate["projector_sha256"] != file_hash(args.candidate / "projector/model.safetensors"):
        raise ValueError("Projecteur candidat modifié.")
    if candidate["lora_sha256"] and candidate["lora_sha256"] != file_hash(args.candidate / "lora/pytorch_lora_weights.safetensors"):
        raise ValueError("LoRA candidat modifié.")
    all_rows = read_jsonl(args.dataset / "scenes.jsonl")
    if candidate["identity"]["dataset_sha256"] != digest(all_rows):
        raise ValueError("Candidat entraîné sur un autre dataset.")
    rows = [r for r in all_rows if r["split"] == args.split][:args.limit]
    if not rows:
        raise ValueError("Aucun prompt dans cet ensemble.")
    student, teacher = (FeatureCache(args.cache / k) for k in ("student", "teacher"))
    for name, cache in (("student", student), ("teacher", teacher)):
        if file_hash(cache.path) != candidate["identity"][f"{name}_index"]:
            raise ValueError("Features différentes de celles de l'entraînement.")
    dataset = json.loads((args.dataset / "manifest.json").read_text())
    methods = (["baseline"] if args.baseline_projector else []) + (["teacher_text", "teacher_layout"] if args.include_teacher else [])
    methods += ["candidate_text", "candidate_layout"]
    config = {"candidate_sha256": file_hash(args.candidate / "candidate.json"), "dataset_sha256": digest(all_rows),
              "baseline_sha256": file_hash(args.baseline_projector / "model.safetensors") if args.baseline_projector else None,
              "rows": [r["id"] for r in rows], "seeds": args.seeds, "steps": args.steps, "split": args.split,
              "resolution": dataset["resolution"], "methods": methods, "cfg": 1.0, "kv_cache": False,
              "generator_sha256": file_hash(Path(__file__)), "core_sha256": file_hash(ROOT / "common/sharm_image21/core.py")}
    signature = digest(config)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "predictions.json"
    previous = json.loads(path.read_text()) if path.exists() else {"signature": signature, "rows": []}
    if previous["signature"] != signature:
        raise ValueError("Générations incompatibles dans ce dossier.")
    predictions = {r["id"]: r for r in previous["rows"]}
    embeddings = FeatureCache(args.output / "embeddings", {"signature": signature})
    # Prepare embeddings before loading the diffuser: no encoder resides on GPU.
    for method in methods:
        projector = None if method.startswith("teacher") else Projector.load(
            args.baseline_projector if method == "baseline" else args.candidate / "projector", "cuda").eval()
        for row in rows:
            key = digest({"method": method, "scene": row["id"]})
            if key in embeddings:
                continue
            caption = row["caption_id"] if method.endswith("layout") else row.get("caption_plain_id", row["caption_id"])
            if method.startswith("teacher"):
                values = teacher.get(caption)
            else:
                boxes = torch.tensor([o["box"] for o in row["objects"]], device="cuda", dtype=torch.bfloat16) if method == "candidate_layout" else None
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    values = projector(student.get(caption).cuda(), layout=boxes).to(torch.bfloat16).cpu()
            embeddings.write([(key, values.contiguous(), [])])
        del projector
        gc.collect()
        torch.cuda.empty_cache()
    pipeline = load_diffuser_pipeline()
    lora_loaded = False
    resolution = dataset["resolution"]
    for method in methods:
        if method.startswith("candidate") and candidate["lora_sha256"] and not lora_loaded:
            pipeline.load_lora_weights(args.candidate / "lora", adapter_name="spatial")
            lora_loaded = True
        for row in rows:
            values = embeddings.get(digest({"method": method, "scene": row["id"]}))[None].cuda()
            for seed in args.seeds:
                key = digest({"method": method, "scene": row["id"], "seed": seed})
                if key in predictions and file_hash(args.output / predictions[key]["file_name"]) == predictions[key]["image_sha256"]:
                    continue
                image = pipeline(prompt_embeds=values, width=resolution, height=resolution, output_resolution=resolution,
                                 num_inference_steps=args.steps, true_cfg_scale=1.0, use_kv_cache=False,
                                 generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]
                if image.mode != "RGBA":
                    raise ValueError("Le générateur n'a pas retourné un RGBA natif.")
                name = f"images/{key}.png"
                # Keep failed opaque examples: spatial evaluation must count them.
                save_png(image, args.output / name, preview_path=args.output / f"previews/{key}.png")
                predictions[key] = {"id": key, "scene_id": row["id"], "pair_group": row["pair_group"],
                    "method": method, "seed": seed, "file_name": name, "preview": f"previews/{key}.png",
                    "image_sha256": file_hash(args.output / name), "objects": row["objects"],
                    "relation": row["relation"], "prompt": row["text"] if method.endswith("layout") else row.get("text_plain", row["text"])}
                atomic_json(path, {"signature": signature, "config": config, "rows": list(predictions.values())})
                print(f"{method} : {len(predictions)}/{len(methods)*len(rows)*len(args.seeds)}", flush=True)


if __name__ == "__main__":
    main()
