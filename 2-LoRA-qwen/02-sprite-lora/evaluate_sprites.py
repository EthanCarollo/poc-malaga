#!/usr/bin/env python3
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import numpy as np
import torch
from PIL import Image
from sharm_image21.core import ARTIFACTS, WEIGHTS, FeatureCache, atomic_json, file_hash
from sharm_image21.diffusion import load_diffuser_pipeline


@torch.inference_mode()
def main():
    lora_dir = WEIGHTS / "sharm-image-2.1-sprites-128-lora"
    if not (lora_dir / "training-complete.json").exists():
        raise RuntimeError("Le LoRA n'est pas terminé.")
    data = json.loads((ARTIFACTS / "sprite-training-512.json").read_text())
    rows = [row for row in data["rows"] if row["split"] == "validation"]
    samples, species = [], set()
    for row in rows:
        if row["species_id"] not in species:
            samples.append(row)
            species.add(row["species_id"])
        if len(samples) == 6:
            break
    cache = FeatureCache(Path(data["text_cache"]))
    pipeline = load_diffuser_pipeline()
    pipeline.load_lora_weights(lora_dir, adapter_name="sharm_sprites")
    output = ARTIFACTS / "sprite-samples"
    output.mkdir(exist_ok=True)
    report = []
    for index, row in enumerate(samples):
        embeds = cache.get(row["caption_id"]).unsqueeze(0).cuda()
        image = pipeline(prompt_embeds=embeds, prompt_embeds_mask=None, width=512, height=512,
                         output_resolution=512, num_inference_steps=30,
                         generator=torch.Generator(device="cuda").manual_seed(2000+index)).images[0]
        pixels = np.asarray(image)
        if image.mode != "RGBA" or pixels[..., :3].std() < 1 or pixels[..., 3].max() == 0:
            raise RuntimeError("Échantillon sprite vide ou mauvais format.")
        image.save(output / f"{row['species_id']}-512.png")
        image.resize((128, 128), Image.Resampling.NEAREST).save(output / f"{row['species_id']}-128.png")
        report.append({"species_id": row["species_id"], "prompt": row["text"],
                       "transparent_fraction": float((pixels[..., 3] == 0).mean()),
                       "seed": 2000+index})
    atomic_json(output / "evaluation.json", {"samples": report, "manual_visual_review_required": True,
                                            "scope": "automatic checks verify file and numerical integrity, not identity quality"})
    atomic_json(lora_dir / "sharm_lora_config.json", {"projector_sha256": data["identity"]["projector_sha256"],
        "base_model": "ethanzxv/Sharm-Image-2.1-VL4B-FP8", "private": True, "published": False,
        "lora_sha256": file_hash(lora_dir / "pytorch_lora_weights.safetensors")})
    (lora_dir / "README.md").write_text(
        "---\nlicense: other\nlicense_name: qwen-research\nlicense_link: LICENSE\n"
        "base_model: ethanzxv/Sharm-Image-2.1-VL4B-FP8\n---\n\n"
        "# Sharm Image 2.1 Sprites 128 LoRA\n\nBuilt with Qwen. Private research checkpoint.\n\n"
        "Trained on RGBA sprite latents at 512px using the frozen Sharm 4B FP8 conditioner and projector. "
        "128px samples use nearest-neighbor reduction. Check sharm_lora_config.json for projector compatibility.\n",
        encoding="utf-8")
    base_license = WEIGHTS / "sharm-image-2.1-vl4b-fp8/LICENSE"
    (lora_dir / "LICENSE").write_bytes(base_license.read_bytes())
    (lora_dir / "NOTICE").write_bytes((base_license.parent / "NOTICE").read_bytes())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
