#!/usr/bin/env python3
"""Run the shipped loader in a clean process using the verified HF mirrors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    # No import from the development package: exercise exactly the shipped code.
    sys.path.insert(0, str(args.base.resolve()))
    import torch
    import numpy as np
    from PIL import Image
    from sharm_image21 import pipeline as shipped

    if not Path(shipped.__file__).resolve().is_relative_to(args.base.resolve()):
        raise RuntimeError("Le loader utilisé ne provient pas du modèle publié.")
    case = json.loads((args.lora / "comparisons/lora/comparison.json").read_text())["pairs"][0]
    pipeline = shipped.SharmImage21Pipeline.from_pretrained(args.base)
    image = pipeline(case["prompt"], lora_dir=args.lora, width=512, height=512,
                     output_resolution=512, num_inference_steps=30, true_cfg_scale=1.0,
                     generator=torch.Generator(device="cuda").manual_seed(case["seed"])).images[0]
    pixels = np.asarray(image)
    if (image.mode != "RGBA" or image.size != (512, 512) or pixels[..., :3].std() < 1
            or image.getchannel("A").getextrema()[1] == 0):
        raise RuntimeError("La génération après relecture HF est vide ou invalide.")
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "published-generation-512.png"
    image.save(output)
    image.resize((128, 128), Image.Resampling.NEAREST).save(args.output / "published-generation-128.png")
    report = {"passed": True, "prompt": case["prompt"], "seed": case["seed"],
              "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "base": str(args.base), "lora": str(args.lora), "loader": str(shipped.__file__),
              "seconds": time.monotonic() - started, "gpu": torch.cuda.get_device_name(),
              "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "scope": "shipped loader and verified private HF mirrors, 4B FP8, projector and LoRA"}
    (args.output / "published-inference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
