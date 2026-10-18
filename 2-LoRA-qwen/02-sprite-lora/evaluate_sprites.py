#!/usr/bin/env python3
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import numpy as np
import torch
from sharm_image21.core import ARTIFACTS, PROJECT, WEIGHTS, FeatureCache, atomic_json, file_hash
from sharm_image21.diffusion import load_diffuser_pipeline
from sharm_image21.model_cards import (LABELS, comparison_manifest, generation_kwargs, load_comparisons,
                                       lora_card, package_comparisons, save_member)


@torch.inference_mode()
def main():
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
    comparison = comparison_manifest(
        "lora", pipeline, steps=30, projector_sha256=fingerprint,
        controls={"base_model": base_config["hf_repo_intent"], "base_revision": base_config["base_revision"],
                  "student_identity": base_config["student_identity"], "same_conditioning_embeddings": True,
                  "text_cache_index_sha256": file_hash(cache.path), "dataset_identity": data["identity"],
                  "lora_sha256": lora_fingerprint, "lora_weight": 1.0,
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
        if side == "right":
            pipeline.load_lora_weights(lora_dir, adapter_name="sharm_sprites")
            pipeline.set_adapters("sharm_sprites", adapter_weights=1.0)
        for index, row in enumerate(samples):
            embeds = cache.get(row["caption_id"]).unsqueeze(0).cuda()
            image = pipeline(prompt_embeds=embeds, prompt_embeds_mask=None, **generation_kwargs(30),
                             generator=torch.Generator(device="cuda").manual_seed(2000+index)).images[0]
            pixels = np.asarray(image)
            if image.mode != "RGBA" or pixels[..., :3].std() < 1 or pixels[..., 3].max() == 0:
                raise RuntimeError(f"Échantillon sprite {name} vide ou mauvais format.")
            comparison["pairs"][index][side] = save_member(
                image, output, f"{row['species_id']}-{name}", label, final_size=128)
            report[index][name] = {"transparent_fraction": float((pixels[..., 3] == 0).mean()),
                                   "image": comparison["pairs"][index][side]["image"]["file"]}
    atomic_json(output / "comparison.json", comparison)
    load_comparisons(output, "lora", fingerprint, lora_fingerprint)
    atomic_json(output / "evaluation.json", {"samples": report, "manual_visual_review_required": True,
                                            "projector_sha256": fingerprint, "lora_sha256": lora_fingerprint,
                                            "comparison_sha256": file_hash(output / "comparison.json"),
                                            "scope": "automatic checks verify file and numerical integrity, not identity quality"})
    atomic_json(lora_dir / "sharm_lora_config.json", {"projector_sha256": fingerprint,
        "base_model": "ethanzxv/Sharm-Image-2.1-VL4B-FP8", "private": True, "published": False,
        "lora_sha256": lora_fingerprint, "comparison_sha256": file_hash(output / "comparison.json")})
    packaged_lora = package_comparisons(output, lora_dir / "comparisons/lora", "lora", fingerprint, lora_fingerprint)
    packaged_encoder = package_comparisons(base / "comparisons/encoder", lora_dir / "comparisons/encoder", "encoder", fingerprint)
    footprint = json.loads((base / "footprint.json").read_text())
    atomic_json(lora_dir / "footprint.json", footprint)
    atomic_json(lora_dir / "evaluation.json", json.loads((output / "evaluation.json").read_text()))
    (lora_dir / "README.md").write_text(lora_card(packaged_lora, packaged_encoder, footprint), encoding="utf-8")
    for filename in ("LICENSE", "LICENSE.encoder", "NOTICE"):
        (lora_dir / filename).write_bytes((base / filename).read_bytes())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
