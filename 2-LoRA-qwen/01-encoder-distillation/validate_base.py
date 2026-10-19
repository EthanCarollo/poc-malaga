#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import numpy as np
import torch
from sharm_image21.core import (ARTIFACTS, CACHE, PROJECT, FeatureCache, Projector, atomic_json,
                                file_hash, model_identity, read_jsonl)
from sharm_image21.diffusion import load_transformer, load_diffuser_pipeline, predict_flow
from sharm_image21.image_io import require_transparent_background
from sharm_image21.model_cards import (LABELS, comparison_manifest, generation_kwargs,
                                       load_comparisons, save_member)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.samples <= 6 or args.steps < 1:
        raise ValueError("Prévoir de 1 à 6 comparaisons et un nombre de pas positif.")
    projector_path = ARTIFACTS / "projector"
    fingerprint = file_hash(projector_path / "model.safetensors")
    report_path = ARTIFACTS / "base-validation.json"
    output = ARTIFACTS / "base-comparisons"
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get("projector_sha256") == fingerprint and previous.get("passed"):
            try:
                existing = load_comparisons(output, "encoder", fingerprint)
            except (OSError, ValueError, KeyError):
                existing = None
            if (existing and len(existing["pairs"]) == args.samples
                    and existing["settings"]["num_inference_steps"] == args.steps):
                print("Base déjà validée, comparaisons appariées présentes à cette révision.", flush=True)
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
    output.mkdir(exist_ok=True)
    comparison = comparison_manifest(
        "encoder", pipeline, steps=args.steps, projector_sha256=fingerprint,
        controls={"base_revision": model_identity("teacher")["revision"],
                  "teacher_identity": model_identity("teacher"), "student_identity": model_identity("student"),
                  "teacher_cache_index_sha256": file_hash(teacher.path),
                  "student_cache_index_sha256": file_hash(student.path),
                  "lora": "absent on both sides", "shared_diffusion_and_vae": True},
        selection={"split": "projector validation", "method": "first matching prompt per fixed category, then corpus order",
                   "markers": ["alpha channel", "exact text", "to the left", "Une illustration", "Exactly", "photograph"],
                   "selected_before_generation": True, "count": args.samples},
        script=Path(__file__).resolve(), project=PROJECT)
    samples = []
    for index, (row, pair) in enumerate(zip(chosen[:args.samples], pairs[:args.samples])):
        record = {"id": row["id"], "prompt": row["text"], "seed": 1000+index}
        for side, label, embeddings in zip(("left", "right"), ("teacher", "sharm"), pair):
            image = pipeline(prompt_embeds=embeddings, prompt_embeds_mask=None, **generation_kwargs(args.steps),
                             generator=torch.Generator(device="cuda").manual_seed(1000+index)).images[0]
            if image.mode != "RGBA" or np.asarray(image)[..., :3].std() < 1 or np.asarray(image)[..., 3].max() == 0:
                raise RuntimeError(f"Échantillon {label} vide ou format incorrect.")
            if any(marker in row["text"].lower() for marker in
                   ("alpha channel", "transparent background", "with transparency")):
                require_transparent_background(image)
            member = save_member(image, output, f"{index:02d}-{label}", LABELS["encoder"][0 if side == "left" else 1])
            record[side] = member
            samples.append({"file": member["image"]["file"], "prompt": row["text"],
                            "seed": 1000+index, "conditioner": label})
        comparison["pairs"].append(record)
    atomic_json(output / "comparison.json", comparison)
    load_comparisons(output, "encoder", fingerprint)
    atomic_json(output / "samples.json", samples)
    atomic_json(report_path, {**report, "passed": True, "samples": samples,
                             "comparison_sha256": file_hash(output / "comparison.json")})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
