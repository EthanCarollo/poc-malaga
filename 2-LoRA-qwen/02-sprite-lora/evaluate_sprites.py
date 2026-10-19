#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import numpy as np
import torch
from sharm_image21.core import ARTIFACTS, PROJECT, WEIGHTS, FeatureCache, atomic_json, file_hash
from sharm_image21.diffusion import load_diffuser_pipeline
from sharm_image21.image_io import require_transparent_background
from sharm_image21.model_cards import (LABELS, comparison_manifest, generation_kwargs, load_comparisons,
                                       lora_card, package_comparisons, save_member)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-weight", type=float, default=1.0)
    args = parser.parse_args()
    if not math.isfinite(args.lora_weight) or not 0 < args.lora_weight <= 2:
        raise ValueError("Choisir un poids LoRA fini entre 0 et 2.")
    lora_dir = WEIGHTS / "sharm-image-2.1-sprites-128-lora"
    if not (lora_dir / "training-complete.json").exists():
        raise RuntimeError("Le LoRA n'est pas terminé.")
    data = json.loads((ARTIFACTS / "sprite-training-512.json").read_text())
    base = WEIGHTS / "sharm-image-2.1-vl4b-fp8"
    base_config = json.loads((base / "sharm_config.json").read_text())
    training = json.loads((lora_dir / "training-complete.json").read_text())
    fingerprint = data["identity"]["projector_sha256"]
    if (base_config["projector_sha256"] != fingerprint
            or file_hash(base / "projector/model.safetensors") != fingerprint
            or training["config"]["cache_identity"] != data["identity"]):
        raise ValueError("Les comparaisons doivent utiliser la base et le cache exacts du LoRA.")
    load_comparisons(base / "comparisons/encoder", "encoder", fingerprint)
    lora_fingerprint = file_hash(lora_dir / "pytorch_lora_weights.safetensors")
    rows = [row for row in data["rows"] if row["split"] == "validation"]
    samples, species = [], set()
    for row in rows:
        if row["species_id"] not in species:
            samples.append(row)
            species.add(row["species_id"])
        if len(samples) == 6:
            break
    if len(samples) != 6:
        raise ValueError("Six espèces de validation distinctes sont requises pour la galerie.")
    cache = FeatureCache(Path(data["text_cache"]))
    if cache.index["identity"]["projector_sha256"] != fingerprint:
        raise ValueError("Les embeddings ne correspondent pas au projecteur de la base.")
    pipeline = load_diffuser_pipeline(base=base)
    output = ARTIFACTS / "sprite-samples"
    output.mkdir(exist_ok=True)
    existing = None
    if (output / "comparison.json").exists():
        candidate = load_comparisons(output, "lora", fingerprint, lora_fingerprint)
        if (candidate["settings"]["num_inference_steps"] == 30
                and [pair["species_id"] for pair in candidate["pairs"]] == [row["species_id"] for row in samples]):
            existing = candidate
    previous_report = json.loads((output / "evaluation.json").read_text()) if existing else None
    comparison = comparison_manifest(
        "lora", pipeline, steps=30, projector_sha256=fingerprint,
        controls={"base_model": base_config["hf_repo_intent"], "base_revision": base_config["base_revision"],
                  "student_identity": base_config["student_identity"], "same_conditioning_embeddings": True,
                  "text_cache_index_sha256": file_hash(cache.path), "dataset_identity": data["identity"],
                  "lora_sha256": lora_fingerprint, "lora_weight": args.lora_weight,
                  "training": {key: training["config"][key] for key in
                               ("steps", "resolution", "rank", "learning_rate", "accumulation")},
                  "training_image_count": sum(row["split"] == "train" for row in data["rows"]),
                  "validation_image_count": len(rows),
                  "left_adapter": "absent", "right_adapter": "sharm_sprites"},
        selection={"split": "LoRA validation by species", "method": "first six distinct species in metadata order",
                   "selected_before_generation": True, "count": len(samples)},
        script=Path(__file__).resolve(), project=PROJECT)
    comparison["pairs"] = [{"id": f"species-{row['species_id']}", "species_id": row["species_id"],
                            "caption_id": row["caption_id"],
                            "prompt": row["text"], "seed": 2000+index}
                           for index, row in enumerate(samples)]
    report = [{"species_id": row["species_id"], "prompt": row["text"], "seed": 2000+index}
              for index, row in enumerate(samples)]
    # Complete the whole baseline pass before any adapter is loaded.
    for side, name, label in zip(("left", "right"), ("before", "after"), LABELS["lora"]):
        if side == "left" and existing:
            for index, pair in enumerate(existing["pairs"]):
                comparison["pairs"][index]["left"] = pair["left"]
                report[index]["before"] = previous_report["samples"][index]["before"]
            continue
        if side == "right":
            pipeline.load_lora_weights(lora_dir, adapter_name="sharm_sprites")
            pipeline.set_adapters("sharm_sprites", adapter_weights=args.lora_weight)
        for index, row in enumerate(samples):
            embeds = cache.get(row["caption_id"]).unsqueeze(0).cuda()
            image = pipeline(prompt_embeds=embeds, prompt_embeds_mask=None, **generation_kwargs(30),
                             generator=torch.Generator(device="cuda").manual_seed(2000+index)).images[0]
            pixels = np.asarray(image)
            if image.mode != "RGBA" or pixels[..., :3].std() < 1 or pixels[..., 3].max() == 0:
                raise RuntimeError(f"Échantillon sprite {name} vide ou mauvais format.")
            alpha_fraction = require_transparent_background(image)
            comparison["pairs"][index][side] = save_member(
                image, output, f"{row['species_id']}-{name}", label, final_size=128)
            report[index][name] = {"transparent_fraction": float((pixels[..., 3] == 0).mean()),
                                   "near_transparent_fraction": alpha_fraction,
                                   "image": comparison["pairs"][index][side]["image"]["file"]}
    atomic_json(output / "comparison.json", comparison)
    load_comparisons(output, "lora", fingerprint, lora_fingerprint)
    atomic_json(output / "evaluation.json", {"samples": report, "manual_visual_review_required": True,
                                            "projector_sha256": fingerprint, "lora_sha256": lora_fingerprint,
                                            "comparison_sha256": file_hash(output / "comparison.json"),
                                            "scope": "automatic checks verify file and numerical integrity, not identity quality"})
    atomic_json(lora_dir / "sharm_lora_config.json", {"projector_sha256": fingerprint,
        "base_model": "ethanzxv/Sharm-Image-2.1-VL4B-FP8", "private": True, "published": False,
        "recommended_weight": args.lora_weight,
        "base_diffusion_revision": base_config["base_revision"],
        "lora_sha256": lora_fingerprint, "comparison_sha256": file_hash(output / "comparison.json")})
    packaged_lora = package_comparisons(output, lora_dir / "comparisons/lora", "lora", fingerprint, lora_fingerprint)
    packaged_encoder = package_comparisons(base / "comparisons/encoder", lora_dir / "comparisons/encoder", "encoder", fingerprint)
    footprint = json.loads((base / "footprint.json").read_text())
    atomic_json(lora_dir / "footprint.json", footprint)
    atomic_json(lora_dir / "evaluation.json", json.loads((output / "evaluation.json").read_text()))
    (lora_dir / "README.md").write_text(lora_card(packaged_lora, packaged_encoder, footprint), encoding="utf-8")
    for filename in ("LICENSE", "LICENSE.encoder", "NOTICE", "requirements-inference.txt"):
        (lora_dir / filename).write_bytes((base / filename).read_bytes())
    notice = lora_dir / "NOTICE"
    notice.write_text(notice.read_text(encoding="utf-8") +
                      "Sharm Sprites 128 adds trained LoRA adapters to the diffusion transformer.\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
