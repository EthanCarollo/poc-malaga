#!/usr/bin/env python3
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import torch
from sharm_image21.core import ARTIFACTS, CACHE, WEIGHTS, FeatureCache, Projector, atomic_json, read_jsonl
from sharm_image21.pipeline import SharmImage21Pipeline


@torch.inference_mode()
def main():
    base = WEIGHTS / "sharm-image-2.1-vl4b-fp8"
    pipeline = SharmImage21Pipeline.from_pretrained(base)
    row = next(row for row in read_jsonl(ARTIFACTS / "corpus.jsonl") if row["split"] == "validation")
    expected_source = FeatureCache(CACHE / "student").get(row["id"])
    projector = Projector.load(base / "projector", "cuda").eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = projector(expected_source.cuda()).to(torch.bfloat16).cpu()
    del projector
    torch.cuda.empty_cache()
    actual, mask = pipeline.encode_prompt(row["text"])
    torch.testing.assert_close(actual[0], expected, rtol=0.02, atol=0.1)
    if not mask.all() or actual.shape[-1] != 4096:
        raise ValueError("Embeddings du modèle exporté incompatibles.")
    atomic_json(ARTIFACTS / "export-validation.json", {"passed": True, "private": True,
                "base": str(base), "projector_sha256": pipeline.config["projector_sha256"],
                "scope": "exported 4B FP8 and projector reload, no 8B required"})
    print("Export Sharm rechargé et vérifié sans encodeur 8B.", flush=True)


if __name__ == "__main__":
    main()
