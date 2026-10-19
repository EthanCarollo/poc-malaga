#!/usr/bin/env python3
"""Build paired RGBA compositions with measured boxes and held-out object pairs."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from sharm_image21.spatial_data import (RELATIONS, compose, fingerprint, read_rows,
                                       relation_holds, sha256, write_json)
from sharm_image21.image_io import checkerboard_preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--assets", type=Path, help="JSONL : id, group_id, label, file_name ; chemins relatifs au manifeste.")
    source.add_argument("--sprites", type=Path, help="metadata.jsonl produit par 02-sprite-lora/prepare_dataset.py.")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/spatial/dataset")
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < 30 or args.resolution < 64 or args.resolution % 32:
        raise ValueError("Au moins 30 scènes ; résolution multiple de 32, au moins 64.")
    if args.output.exists():
        raise FileExistsError(f"Choisir un nouveau dossier de dataset : {args.output}")
    source_path = args.assets or args.sprites
    asset_root = source_path.resolve().parent
    assets = read_rows(source_path)
    if args.sprites:
        with (asset_root.parent / "manifest.csv").open(encoding="utf-8") as stream:
            species = {Path(r["file"]).name: r["species_id"] for r in csv.DictReader(stream)}
        assets = [{"id": r["file_name"], "group_id": species[r["file_name"]],
                   "label": r.get("label") or r["text"].split(";")[0].replace("pkm_ow_sprite, ", ""),
                   "file_name": r["file_name"]} for r in assets]
    by_group = {}
    for asset in assets:
        if not all(asset.get(k) for k in ("id", "group_id", "label", "file_name")):
            raise ValueError("Chaque asset requiert id, group_id, label et file_name.")
        asset["sha256"] = sha256(asset_root / asset["file_name"])
        by_group.setdefault(str(asset["group_id"]), []).append(asset)
    pairs = list(itertools.combinations(sorted(by_group), 2))
    if len(pairs) < 3 or len({a["id"] for a in assets}) != len(assets):
        raise ValueError("Au moins trois groupes distincts et des identifiants uniques sont requis.")
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    holdout = max(1, len(pairs)//10)
    groups = {"test": pairs[:holdout], "validation": pairs[holdout:2*holdout], "train": pairs[2*holdout:]}
    totals = {"test": max(1, args.count//10), "validation": max(1, args.count//10)}
    totals["train"] = args.count-totals["test"]-totals["validation"]
    args.output.mkdir(parents=True)
    (args.output / "images").mkdir()
    (args.output / "previews").mkdir()
    rows = []
    for split, total in totals.items():
        for index in range(total):
            group_a, group_b = rng.choice(groups[split])
            a, b = (rng.choice(by_group[g]) for g in (group_a, group_b))
            relation = RELATIONS[index % len(RELATIONS)]
            small, large = rng.choice((0.23, 0.28)), rng.choice((0.30, 0.34))
            widths = (small, large) if index % 2 else (large, small)
            centers = [(0.25, 0.5), (0.75, 0.5)] if relation in ("left", "right") else [(0.5, 0.25), (0.5, 0.75)]
            if relation in ("right", "below"):
                centers.reverse()
            scene_assets = (a, b)
            if index % 5 == 4:
                centers = ([(0.17, 0.5), (0.5, 0.5), (0.83, 0.5)] if relation in ("left", "right")
                           else [(0.5, 0.17), (0.5, 0.5), (0.5, 0.83)])
                if relation in ("right", "below"):
                    centers.reverse()
                scene_assets, widths = (a, b, a), (0.20, 0.24, 0.20)
            # Vary exact coordinates while retaining the requested relation.
            jitter = 0.015 if len(scene_assets) == 3 else 0.05
            centers = [(cx+rng.uniform(-jitter, jitter), cy+rng.uniform(-jitter, jitter)) for cx, cy in centers]
            objects = [{**asset, "box": [cx-w/2, cy-w/2, cx+w/2, cy+w/2]}
                       for asset, (cx, cy), w in zip(scene_assets, centers, widths)]
            image, objects = compose(args.resolution, objects, asset_root)
            if not relation_holds(objects[0]["box"], objects[1]["box"], relation):
                raise RuntimeError("Composition incompatible avec sa relation.")
            en = {"left": "to the left of", "right": "to the right of", "above": "above", "below": "below"}
            fr = {"left": "à gauche de", "right": "à droite de", "above": "au-dessus de", "below": "en dessous de"}
            french = index % 3 == 0
            text = (f"Exactement deux objets distincts. Objet 1 : {a['label']}. Objet 2 : {b['label']}. "
                    f"L'objet 1 est {fr[relation]} l'objet 2. Aucun chevauchement. Fond transparent."
                    if french else
                    f"Exactly two distinct objects. Object 1: {a['label']}. Object 2: {b['label']}. "
                    f"Object 1 is {en[relation]} object 2. No overlap. Transparent background.")
            if len(objects) == 3:
                text = (text.replace("Exactement deux", "Exactement trois") + f" Objet 3 : {a['label']}."
                        if french else text.replace("Exactly two", "Exactly three") + f" Object 3: {a['label']}.")
            plain_text = text
            # Include the requested allocations in text too, with matching order.
            text += " Normalized object boxes xyxy: " + json.dumps([o["box"] for o in objects], separators=(",", ":")) + "."
            scene_id = fingerprint({"split": split, "index": index, "seed": args.seed, "objects": objects})
            name = f"images/{scene_id}.png"
            image.save(args.output / name)
            if index < 4:
                checkerboard_preview(image).save(args.output / f"previews/{split}-{index}.png")
            rows.append({"id": scene_id, "caption_id": fingerprint(text), "text": text,
                         "caption_plain_id": fingerprint(plain_text), "text_plain": plain_text, "split": split,
                         "pair_group": sorted((group_a, group_b)), "relation": relation,
                         "language": "fr" if french else "en", "objects": objects,
                         "file_name": name, "image_sha256": sha256(args.output / name)})
    captions = [{"id": r[key], "text": r[text_key], "split": r["split"]}
                for r in rows for key, text_key in (("caption_id", "text"), ("caption_plain_id", "text_plain"))]
    corpus = list({r["id"]: r for r in captions}.values())
    for name, values in (("scenes.jsonl", rows), ("corpus.jsonl", corpus)):
        (args.output / name).write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in values), encoding="utf-8")
    write_json(args.output / "manifest.json", {"schema": 1, "seed": args.seed, "resolution": args.resolution,
               "dataset_sha256": fingerprint(rows), "corpus_sha256": fingerprint(corpus), "counts": totals,
               "split_unit": "unordered object group pair; variants and inverse relations remain together",
               "asset_manifest_sha256": sha256(source_path), "assets": assets})
    print(json.dumps({"dataset": str(args.output), "scenes": len(rows), "splits": totals}), flush=True)


if __name__ == "__main__":
    main()
