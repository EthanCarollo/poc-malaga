#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import numpy as np
import torch
from sharm_image21.core import (ARTIFACTS, CACHE, FeatureCache, Projector, atomic_json,
                                file_hash, read_jsonl)
from sharm_image21.diffusion import load_transformer, load_diffuser_pipeline, predict_flow


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    projector_path = ARTIFACTS / "projector"
    fingerprint = file_hash(projector_path / "model.safetensors")
    report_path = ARTIFACTS / "base-validation.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get("projector_sha256") == fingerprint and previous.get("passed"):
            print("Base déjà validée à cette révision.", flush=True)
            return
    training = json.loads((projector_path / "training-complete.json").read_text())
    metrics = training["validation"]
    if not (metrics["relative_mse"] < 0.5 and metrics["cosine"] > 0.65 and 0.75 < metrics["norm_ratio"] < 1.25):
        atomic_json(report_path, {"passed": False, "reason": "embedding_alignment_below_gate", "metrics": metrics,
                                  "projector_sha256": fingerprint})
        raise RuntimeError("Alignement insuffisant ; le LoRA ne sera pas lancé sur cette base.")
    rows = [row for row in read_jsonl(ARTIFACTS / "corpus.jsonl") if row["split"] == "validation"]
    chosen = []
    for marker in ("alpha channel", "exact text", "to the left", "Une illustration", "Exactly", "photograph"):
        match = next((row for row in rows if marker in row["text"] and row not in chosen), None)
        if match:
            chosen.append(match)
    chosen += [row for row in rows if row not in chosen][:max(0, 6-len(chosen))]
    student, teacher = FeatureCache(CACHE / "student"), FeatureCache(CACHE / "teacher")
    projector = Projector.load(projector_path, "cuda").eval()
    pairs = []
    for row in chosen:
        if student.index["entries"][row["id"]]["token_ids"] != teacher.index["entries"][row["id"]]["token_ids"]:
            raise ValueError("Tokens de validation désalignés.")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            projected = projector(student.get(row["id"]).cuda()).to(torch.bfloat16)
        pairs.append((teacher.get(row["id"]).unsqueeze(0).cuda(), projected.unsqueeze(0)))
    del projector
    transformer = load_transformer()
    errors = []
    for index, (reference, adapted) in enumerate(pairs):
        generator = torch.Generator(device="cuda").manual_seed(700+index)
        latents = torch.randn((1, 64, 1, 32, 32), generator=generator, device="cuda", dtype=torch.bfloat16)
        for timestep in (0.2, 0.5, 0.8):
            t = torch.tensor([timestep], device="cuda", dtype=torch.bfloat16)
            reference_prediction = predict_flow(transformer, latents, reference, t).float()
            adapted_prediction = predict_flow(transformer, latents, adapted, t).float()
            error = ((reference_prediction-adapted_prediction).square().mean()/reference_prediction.square().mean().clamp_min(1e-8)).item()
            errors.append(error)
            if not np.isfinite(error):
                raise RuntimeError("Prédiction du diffuseur non finie.")
    report = {"projector_sha256": fingerprint, "embedding_metrics": metrics,
              "denoiser_relative_mse": float(np.mean(errors)), "denoiser_worst_mse": max(errors),
              "scope": "experimental text-to-image; image editing not validated", "updated_at": time.time()}
    if report["denoiser_relative_mse"] > 0.25 or report["denoiser_worst_mse"] > 0.5:
        atomic_json(report_path, {**report, "passed": False, "reason": "denoiser_deviation_above_gate"})
        raise RuntimeError("Déviation du diffuseur trop forte ; arrêt avant le LoRA.")
    pipeline = load_diffuser_pipeline(transformer=transformer)
    output = ARTIFACTS / "base-comparisons"
    output.mkdir(exist_ok=True)
    samples = []
    for index, (row, pair) in enumerate(zip(chosen[:args.samples], pairs[:args.samples])):
        for label, embeddings in zip(("teacher", "sharm"), pair):
            image = pipeline(prompt_embeds=embeddings, prompt_embeds_mask=None, width=512, height=512,
                             output_resolution=512, num_inference_steps=args.steps, true_cfg_scale=1.0,
                             generator=torch.Generator(device="cuda").manual_seed(1000+index)).images[0]
            if image.mode != "RGBA" or np.asarray(image)[..., :3].std() < 1 or np.asarray(image)[..., 3].max() == 0:
                raise RuntimeError(f"Échantillon {label} vide ou format incorrect.")
            filename = f"{index:02d}-{label}.png"
            image.save(output / filename)
            samples.append({"file": filename, "prompt": row["text"], "seed": 1000+index})
    atomic_json(output / "samples.json", samples)
    atomic_json(report_path, {**report, "passed": True, "samples": samples})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
