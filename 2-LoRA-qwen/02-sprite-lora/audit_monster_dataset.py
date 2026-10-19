#!/usr/bin/env python3
"""Audit prepared data, source identities, RGBA pixels and species holdout."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_monster_dataset import HERE, Reference, bbox, decode_label, dump, legacy_cells, sha


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=HERE / "dataset/raw")
    parser.add_argument("--reference-dir", type=Path, default=HERE / "dataset/reference")
    args = parser.parse_args()
    root = args.dataset
    report = json.loads((root / "dataset.json").read_text())
    rows = read_jsonl(root / "manifest.jsonl")
    sources = json.loads((root / "sources.json").read_text())
    catalog = sources["source_catalog"]
    unresolved = json.loads((root / "unresolved.json").read_text())
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    require(sha((root / "manifest.jsonl").read_bytes()) == report["manifest_sha256"], "Manifest hash")
    require(len(rows) == report["unique_images"], "Manifest count")
    require(len({r["pixel_sha256"] for r in rows}) == len(rows), "Duplicate target RGBA pixels")
    require(len({r["sample_id"] for r in rows}) == len(rows), "Duplicate sample IDs")
    train = {r["species_id"] for r in rows if r["split"] == "train"}
    validation = {r["species_id"] for r in rows if r["split"] == "validation"}
    require(not train & validation and bool(train) and bool(validation), "Species leakage / empty split")
    raw = {}
    for entry in sources["sources"]:
        path = args.raw_dir / entry["file"]
        require(sha(path.read_bytes()) == entry["sha256"], f"Changed raw file: {path.name}")
        if "duplicate_of" not in entry:
            with Image.open(path) as image:
                raw[path.name] = np.array(image.convert("RGBA"))

    refs = Reference(args.reference_dir)
    accounted = set()
    expected = set()
    for group in catalog["battle_groups"]:
        for name in group["sheets"]:
            a = raw[name]
            for r in range(a.shape[0] // 103):
                for c in range(8):
                    if a[r*103:r*103+96,c*96:c*96+96,3].any():
                        expected.add((name,r,c))
    for name, (generation, _) in catalog["legacy_sheets"].items():
        for i, _ in enumerate(legacy_cells(generation)):
            r,c = divmod(i,10)
            expected.add((name,r,c))
    for row in rows:
        d = row["definition"]
        require(d["species_id"] == row["species_id"], f"Definition species: {row['sample_id']}")
        require(d["name_en"] == refs.names[row["species_id"], "9"]["name"], f"Canonical name: {row['sample_id']}")
        require(d["form_identifier"].split("-")[0] == refs.species[row["species_id"]]["identifier"].split("-")[0], f"Form identity: {row['sample_id']}")
        if d["form_name_en"].lower() in ("male", "female"):
            require(d["form_name_en"].lower() == d["gender"], f"Contradictory sex/form: {row['sample_id']}")
        for origin in row["source_cells"]:
            key = (origin["sheet"],origin["row"],origin["column"])
            require(key not in accounted, f"Cell used twice: {key}")
            accounted.add(key)
            if origin["sheet"] in catalog["legacy_sheets"]:
                generation = catalog["legacy_sheets"][origin["sheet"]][0]
                species, _ = legacy_cells(generation)[origin["row"]*10+origin["column"]]
                require(species == row["species_id"], f"Legacy cell species: {key}")
            else:
                r,c = origin["row"],origin["column"]
                label = decode_label(raw[origin["sheet"]][r*103+96:r*103+103,c*96:c*96+96,3])
                require(int(label[:3]) == row["species_id"], f"Printed species: {key}")
    for record in unresolved:
        key = (record["sheet"],record["row"],record["column"])
        require(key not in accounted, f"Quarantined and trained: {key}")
        accounted.add(key)
        require((root / record["original_cell_file"]).is_file(), f"Missing quarantine cell: {key}")
    require(accounted == expected, f"Source-cell accounting: missing={len(expected-accounted)}, extra={len(accounted-expected)}")
    require(len(expected) == report["verified_source_cells"] + report["unresolved_cells"], "Cell totals")
    inventory_path = root / "source_cells.jsonl"
    if inventory_path.exists():
        inventory = read_jsonl(inventory_path)
        require(len(inventory) == len(expected), "Source inventory coverage")
        form_numbers = {}
        for item in inventory:
            label = item.get("printed_label")
            if not label or item["status"] == "quarantine":
                continue
            d = item["definition"]
            require(int(label[:3]) == d["species_id"], f"Inventory printed species: {label}")
            suffix = label.split("-")[-1]
            if suffix in ("F","M"):
                require(d["gender"] == {"F":"female","M":"male"}[suffix], f"Printed sex: {label}")
            elif "-" in label and suffix.isdigit():
                key = (d["species_id"],int(suffix))
                if key in form_numbers:
                    require(form_numbers[key] == d["form_id"], f"Cross-palette/view form conflict: {label}")
                form_numbers[key] = d["form_id"]

    def audit_image(row):
        name = row["file_name"]
        path = root / "images_128" / name
        findings = []
        with Image.open(path) as image:
            if image.mode != "RGBA" or image.size != (128,128):
                return [f"Invalid RGBA/size: {name}"]
            a = np.array(image)
        if sha(path.read_bytes()) != row["image_sha256"] or sha(a.tobytes()) != row["pixel_sha256"]:
            findings.append(f"Changed image: {name}")
        if set(np.unique(a[...,3])) != {0,255} or np.any(a[a[...,3]==0,:3]):
            findings.append(f"Alpha/hidden RGB: {name}")
        box = bbox(a)
        if list(box or []) != row["target_bbox_xyxy"] or min(box[0],box[1],128-box[2],128-box[3]) < 4:
            findings.append(f"Clipping/margin: {name}")
        x0,y0,x1,y1 = box
        scale = row["integer_scale_from_native"]
        crop = a[y0:y1,x0:x1]
        native = crop[::scale,::scale]
        if not np.array_equal(crop,np.repeat(np.repeat(native,scale,0),scale,1)):
            findings.append(f"Fractional resize/smoothing: {name}")
        origin = row["source_cells"][0]
        cx,cy,cw,ch = origin["crop_xywh"]
        source = raw[origin["sheet"]][cy:cy+ch,cx:cx+cw]
        nx0,ny0,nx1,ny1 = row["native_bbox_xyxy"]
        original = source[ny0:ny1,nx0:nx1]
        mask = native[...,3] > 0
        if original.shape != native.shape or not np.array_equal(original[mask,:3],native[mask,:3]):
            findings.append(f"Source foreground pixels changed: {name}")
        for size in (256,512):
            with Image.open(root / f"train_{size}" / name) as im:
                enlarged = np.array(im)
                if im.mode != "RGBA" or not np.array_equal(enlarged,np.repeat(np.repeat(a,size//128,0),size//128,1)):
                    findings.append(f"Training resize {size}: {name}")
        return findings

    with ThreadPoolExecutor(max_workers=8) as pool:
        for count, findings in enumerate(pool.map(audit_image,rows),1):
            errors.extend(findings)
            if count % 1000 == 0:
                print(f"Audited pixels {count}/{len(rows)}",flush=True)
    for size in (128,256,512):
        folder = root / ("images_128" if size==128 else f"train_{size}")
        require({p.name for p in folder.glob("*.png")} == {r["file_name"] for r in rows}, f"Unlisted or missing PNG files {size}")
        metadata = read_jsonl(folder / "metadata.jsonl")
        require(len(metadata)==len(rows), f"Metadata count {size}")
        require({r["file_name"] for r in metadata} == {r["file_name"] for r in rows}, f"Metadata files {size}")
        by_name = {r["file_name"]:r for r in rows}
        for item in metadata:
            original = by_name[item["file_name"]]
            require(item["text"]==original["text"] and item["split"]==original["split"] and item["group_id"]==original["group_id"], f"Metadata disagreement {size}: {item['file_name']}")
        for split in ("train","validation"):
            require({r["file_name"] for r in read_jsonl(folder / f"{split}.jsonl")} == {r["file_name"] for r in rows if r["split"]==split}, f"Split list {size}/{split}")
    corpus = read_jsonl(root / "caption-corpus.jsonl")
    require({r["text"] for r in corpus} == {r["text"] for r in rows}, "Caption corpus coverage")
    require(all(r["id"]==sha(json.dumps(r["text"],sort_keys=True,ensure_ascii=False).encode()) for r in corpus), "Caption cache digest")
    result = {"passed": not errors, "errors": errors, "unique_images_checked":len(rows),
              "png_files_checked":len(rows)*3,"source_cells_accounted":len(expected),
              "species":len(train|validation),"train_species":len(train),"validation_species":len(validation),
              "verification_levels":dict(Counter(r["verification"] for r in rows)),
              "manifest_sha256":report["manifest_sha256"],"human_review":False,
              "note":"Integrity/geometry/metadata audit, not a guarantee of generated image quality."}
    dump(root / "audit.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if errors:
        raise SystemExit(1)


if __name__=="__main__":
    main()
