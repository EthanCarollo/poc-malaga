#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from sharm_image21.core import ARTIFACTS, CACHE, ORIGINAL, ROOT, atomic_json, digest, read_jsonl

DATASET_REVISION = "fb620fbe49fa4420e0734bd9c0df11f51176b61f"


def synthetic_prompts(count, seed):
    rng = random.Random(seed)
    objects = ["dragon", "robot", "cat", "castle", "tree", "bicycle", "glass bottle", "owl",
               "fox", "wooden chair", "train", "sailing boat", "mushroom", "flower", "book"]
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "white", "black", "silver"]
    styles = ["pixel art", "watercolor illustration", "studio photograph", "pencil drawing",
              "clay sculpture", "flat vector illustration", "oil painting", "game sprite"]
    texts = set()
    while len(texts) < count:
        obj, other = rng.sample(objects, 2)
        color, other_color = rng.sample(colors, 2)
        style = rng.choice(styles)
        n = rng.choice(["two", "three", "four"])
        templates = [
            f"A {color} {obj} to the left of a {other_color} {other}, {style}, clear composition.",
            f"Exactly {n} objects: {color} {obj}s arranged in a row on a white table, {style}.",
            f"This is an RGBA image with transparency. A single {color} {obj}, {style}. The image has alpha channel and the background is transparent.",
            f'A {color} sign above a {other}, with the exact text "{rng.choice(["HELLO", "SHARM", "OPEN", "BONJOUR", "2026"])}", legible typography, {style}.',
            f"Une illustration de {obj}, {style}, au premier plan devant {other}, lumière douce, composition nette.",
            f"A close view of a {color} {obj} beside a {other_color} {other}, warm sunset light, detailed texture, {style}.",
        ]
        texts.add(rng.choice(templates))
    return sorted(texts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "corpus.jsonl")
    args = parser.parse_args()
    if args.output.exists():
        print(f"Corpus déjà préparé : {args.output}", flush=True)
        return
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    source = Path(hf_hub_download("poloclub/diffusiondb", "metadata.parquet", repo_type="dataset",
                                  revision=DATASET_REVISION, local_dir=CACHE / "sources"))
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL / "processor", local_files_only=True)
    rng = random.Random(args.seed)
    sample, seen, accepted = [], set(), 0
    capacity = args.limit * 2
    for batch in pq.ParquetFile(source).iter_batches(batch_size=8192, columns=["prompt", "prompt_nsfw"]):
        values = batch.to_pydict()
        for text, score in zip(values["prompt"], values["prompt_nsfw"]):
            text = re.sub(r"\s+", " ", text or "").strip()
            key = text.casefold()
            if not 35 <= len(text) <= 1000 or (score is not None and score > 0.3) or key in seen:
                continue
            seen.add(key)
            accepted += 1
            if len(sample) < capacity:
                sample.append(text)
            else:
                index = rng.randrange(accepted)
                if index < capacity:
                    sample[index] = text
    rng.shuffle(sample)
    local = synthetic_prompts(min(1500, args.limit // 10), args.seed)
    sprite_metadata = ROOT / "dataset/train_512/metadata.jsonl"
    sprites = sorted({row["text"] for row in read_jsonl(sprite_metadata)}) if sprite_metadata.exists() else []
    rows, used = [], set()
    candidates = [(text, "sharm-authored-templates") for text in local]
    candidates += [(text, "local-sprite-captions") for text in sprites]
    candidates += [(text, "poloclub/diffusiondb") for text in sample]
    for start in range(0, len(candidates), 256):
        selected = candidates[start:start+256]
        ids = tokenizer([text for text, _ in selected], add_special_tokens=False)["input_ids"]
        for (text, source_name), tokens in zip(selected, ids):
            if not 8 <= len(tokens) <= args.max_tokens or text.casefold() in used:
                continue
            used.add(text.casefold())
            sample_id = digest(text)
            # Group close caption prefixes together, instead of splitting near-identical variants.
            group = re.sub(r"\W+", " ", text.casefold()).strip()[:96]
            split = "validation" if int(digest(group)[:8], 16) % 10 == 0 else "train"
            rows.append({"id": sample_id, "text": text, "split": split, "source": source_name,
                         "tokens_unwrapped": len(tokens)})
            if len(rows) == args.limit:
                break
        if len(rows) == args.limit:
            break
    if len(rows) != args.limit:
        raise RuntimeError(f"Corpus insuffisant : {len(rows)} / {args.limit}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
    report = {"count": len(rows), "sha256": digest(rows), "seed": args.seed,
              "max_tokens": args.max_tokens, "dataset_revision": DATASET_REVISION,
              "dataset_license": "CC0-1.0", "source_url": "https://huggingface.co/datasets/poloclub/diffusiondb",
              "splits": {split: sum(row["split"] == split for row in rows) for split in ("train", "validation")},
              "sources": {name: sum(row["source"] == name for row in rows) for name in sorted({r["source"] for r in rows})}}
    atomic_json(args.output.with_suffix(".manifest.json"), report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
