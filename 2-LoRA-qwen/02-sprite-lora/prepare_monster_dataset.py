#!/usr/bin/env python3
"""Reconcile supplied battle sheets with pinned references and build RGBA sprites."""
from __future__ import annotations

import argparse
import colorsys
import csv
import hashlib
import html
import importlib.util
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fetch_monster_references import SPRITES_REVISION, download, git_blob_hash

HERE = Path(__file__).resolve().parent
VERSION = "monster-rgba-v1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def decode_label(alpha):
    """Exact bitmap decoding of the sheet's font; no probabilistic OCR."""
    glyphs = json.loads((HERE / "monster_label_font.json").read_text())["glyphs"]
    occupied = np.where((alpha > 0).any(axis=0))[0]
    chunks = np.split(occupied, np.where(np.diff(occupied) > 1)[0] + 1)
    text = ""
    for chunk in chunks:
        if not len(chunk):
            continue
        key = "/".join("".join("1" if v else "0" for v in line)
                       for line in alpha[:, chunk.min():chunk.max() + 1] > 0)
        if key not in glyphs:
            raise ValueError("Unknown source-label glyph")
        text += glyphs[key]
    return text


def bbox(a):
    y, x = np.nonzero(a[..., 3])
    return (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1) if len(y) else None


def signature(a):
    box = bbox(a)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    v = a[y0:y1, x0:x1].copy()
    v[v[..., 3] == 0] = 0
    v[..., :3] >>= 3  # 5-bit channels: sheet v<<3 versus reference round(v*255/31).
    return f"{v.shape[1]}x{v.shape[0]}:" + sha(v.tobytes())


def read_csv(root, name):
    with (root / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class Reference:
    def __init__(self, root: Path):
        self.root = root
        self.lock = json.loads((root / "reference-lock.json").read_text())
        if self.lock["sprites_revision"] != SPRITES_REVISION:
            raise RuntimeError("Unexpected reference revision")
        for name, expected in self.lock["csv_sha256"].items():
            if sha((root / f"{name}.csv").read_bytes()) != expected:
                raise RuntimeError(f"Modified reference CSV: {name}")
        self.pokemon = {int(r["id"]): r for r in read_csv(root, "pokemon")}
        self.species = {int(r["id"]): r for r in read_csv(root, "pokemon_species")}
        self.names = {(int(r["pokemon_species_id"]), r["local_language_id"]): r
                      for r in read_csv(root, "pokemon_species_names")}
        self.forms = {r["identifier"]: r for r in read_csv(root, "pokemon_forms")}
        self.default_forms = {int(r["pokemon_id"]): r for r in self.forms.values() if r["is_default"] == "1"}
        self.form_names = {int(r["pokemon_form_id"]): r for r in read_csv(root, "pokemon_form_names")
                           if r["local_language_id"] == "9"}
        self.form_names_fr = {int(r["pokemon_form_id"]): r for r in read_csv(root, "pokemon_form_names")
                              if r["local_language_id"] == "5"}
        self.shapes = {int(r["id"]): r["identifier"] for r in read_csv(root, "pokemon_shapes")}
        self.shape_descriptions = {int(r["pokemon_shape_id"]): r["description"]
                                  for r in read_csv(root, "pokemon_shape_prose") if r["local_language_id"] == "9"}
        self.types = {int(r["id"]): r["identifier"] for r in read_csv(root, "types")}
        self.mon_types = defaultdict(list)
        for r in read_csv(root, "pokemon_types"):
            self.mon_types[int(r["pokemon_id"])].append(self.types[int(r["type_id"])])
        self.tree = json.loads((root / "sprite-tree.json").read_text())
        self.tree_entries = {r["path"]: r for r in self.tree["tree"] if r["type"] == "blob"}
        self.frames_cache = {}
        self.local_paths = {p.relative_to(root / "sprites").as_posix() for p in (root / "sprites").rglob("*") if p.is_file()}
        self.emerald_entries = {r["path"]: r for r in json.loads((root / "emerald-tree.json").read_text())["tree"]
                                if r["type"] == "blob"}

    @cache
    def entity(self, path):
        stem = Path(path).stem
        identifier, _, suffix = stem.partition("-")
        mon = self.pokemon[int(identifier)]
        species_id = int(mon["species_id"])
        form = self.forms.get(f"{self.species[species_id]['identifier']}-{suffix}") if suffix else None
        form = form or self.default_forms.get(int(identifier))
        form_id = int(form["id"]) if form else species_id
        sex = "female" if "female" in Path(path).parts or suffix == "female" else None
        if sex is None and self.species[species_id]["has_gender_differences"] == "1":
            sex = "male"
        # Frillish/Jellicent have explicit form IDs in addition to female/ paths.
        sex_form = self.forms.get(f"{self.species[species_id]['identifier']}-{sex}") if sex else None
        if sex_form:
            form_id = int(sex_form["id"])
        return (species_id, form_id, sex)

    def frames(self, family, path):
        key = (family, path)
        if key not in self.frames_cache:
            entries = self.tree_entries if family == "sprites" else self.emerald_entries
            if git_blob_hash((self.root / family / path).read_bytes()) != entries[path]["sha"]:
                raise RuntimeError(f"Modified reference sprite: {family}/{path}")
            with Image.open(self.root / family / path) as image:
                seen, frames = set(), []
                for index in range(getattr(image, "n_frames", 1)):
                    image.seek(index)
                    a = np.array(image.convert("RGBA"))
                    key_hash = sha(a.tobytes())
                    if key_hash not in seen:
                        frames.append((index, a))
                        seen.add(key_hash)
                self.frames_cache[key] = frames
        return self.frames_cache[key]

    def index(self):
        cache = self.root / "match-index-v1.json"
        identity = sha(json.dumps(self.lock, sort_keys=True).encode())
        if cache.is_file():
            data = json.loads(cache.read_text())
            if data.get("identity") == identity:
                return data["index"]
        jobs = []
        for family in ("sprites", "emerald"):
            for f in sorted((self.root / family).rglob("*.png")):
                rel = f.relative_to(self.root / family).as_posix()
                if family == "emerald" and "back" in f.parts:
                    continue
                jobs.append((family, rel))

        def process(job):
            family, path = job
            # All files must match their Git blob, including warm-cache builds.
            tree = self.tree_entries if family == "sprites" else self.emerald_entries
            if git_blob_hash((self.root / family / path).read_bytes()) != tree[path]["sha"]:
                raise RuntimeError(f"Modified reference sprite: {family}/{path}")
            results = []
            for frame, a in self.frames(family, path):
                sig = signature(a)
                if sig:
                    results.append((sig, {"family": family, "path": path, "frame": frame}))
            return results

        self.emerald_entries = {r["path"]: r for r in json.loads((self.root / "emerald-tree.json").read_text())["tree"]
                                if r["type"] == "blob"}
        index = defaultdict(list)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for count, result in enumerate(pool.map(process, jobs), 1):
                for key, match in result:
                    index[key].append(match)
                if count % 500 == 0:
                    print(f"Index references {count}/{len(jobs)}", flush=True)
        dump(cache, {"identity": identity, "index": dict(index)})
        return dict(index)

    def animated_candidates(self, entities, view, shiny):
        candidates = []
        for path, entry in self.tree_entries.items():
            if not path.startswith("animated/") or not path.endswith(".gif"):
                continue
            if not Path(path).stem.split("-", 1)[0].isdigit():
                continue
            if ("back" in Path(path).parts) != (view == "back") or ("shiny" in Path(path).parts) != shiny:
                continue
            if self.entity(path) not in entities:
                continue
            target = self.root / "sprites" / path
            if not target.is_file() or git_blob_hash(target.read_bytes()) != entry["sha"]:
                data = download(f"https://raw.githubusercontent.com/PokeAPI/sprites/{SPRITES_REVISION}/"
                                f"sprites/pokemon/versions/generation-v/black-white/{path}")
                if git_blob_hash(data) != entry["sha"]:
                    raise RuntimeError(f"Bad animated reference: {path}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            candidates.extend({"family": "sprites", "path": path, "frame": frame}
                              for frame, _ in self.frames("sprites", path))
        return candidates


def match_alpha(tile, reference, matte):
    """Require every composited RGB5 pixel to agree; restore opaque bg-colored details."""
    bg = np.array(matte, dtype=np.uint8)
    cut = tile.copy()
    cut[np.all(cut[..., :3] == bg, axis=-1)] = 0
    candidate = reference.copy()
    candidate[np.all((candidate[..., :3] >> 3) == (bg >> 3), axis=-1)] = 0
    if signature(cut) != signature(candidate) or bbox(cut) is None:
        return None
    x0, y0, _, _ = bbox(cut)
    x1, y1, _, _ = bbox(candidate)
    dx, dy = x0 - x1, y0 - y1
    aligned = np.zeros_like(tile)
    yy, xx = np.nonzero(reference[..., 3])
    ty, tx = yy + dy, xx + dx
    if (ty < 0).any() or (tx < 0).any() or (ty >= tile.shape[0]).any() or (tx >= tile.shape[1]).any():
        return None
    aligned[ty, tx] = reference[yy, xx]
    composite = aligned[..., :3].copy()
    composite[aligned[..., 3] == 0] = bg
    if not np.array_equal(composite >> 3, tile[..., :3] >> 3):
        return None
    result = tile.copy()
    result[..., 3] = aligned[..., 3]
    result[result[..., 3] == 0] = 0
    return result


def legacy_cells(generation):
    spec = importlib.util.spec_from_file_location("legacy_sprite_cutter", HERE / "prepare_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sheet_cells(generation)


def get_tiles(raw, catalog):
    tiles, sources = [], []
    known = set(catalog["duplicate_files"])
    for group in catalog["battle_groups"]:
        for name, (view, shiny) in group["sheets"].items():
            known.add(name)
            with Image.open(raw / name) as image:
                a = np.array(image.convert("RGBA"))
            if a.shape[1] != 768 or a.shape[0] % 103:
                raise ValueError(f"Unexpected grid: {name}")
            for r in range(a.shape[0] // 103):
                # The following seven pixels contain tiny dex-number/form labels;
                # they are outside every sprite crop (black text on transparency).
                for c in range(8):
                    tile = a[r * 103:r * 103 + 96, c * 96:c * 96 + 96].copy()
                    if not tile[..., 3].any():
                        continue
                    source_label = decode_label(a[r * 103 + 96:r * 103 + 103, c * 96:c * 96 + 96, 3])
                    if not source_label[:3].isdigit():
                        raise ValueError(f"Invalid source label: {name}:{r}:{c}: {source_label}")
                    tiles.append({"sheet": name, "row": r, "column": c, "view": view, "shiny": shiny,
                                  "region": group["region"], "family": "sprites", "tile": tile,
                                  "matte_rgb": list(Counter(map(tuple, np.concatenate((tile[0,:,:3], tile[-1,:,:3], tile[:,0,:3], tile[:,-1,:3])).tolist())).most_common(1)[0][0]),
                                  "source_label": source_label, "source_species_id": int(source_label[:3]),
                                  "species_range": group["species_range"], "crop_xywh": [c * 96, r * 103, 96, 96]})
            sources.append({"file": name, "sha256": sha((raw / name).read_bytes()), "size": [a.shape[1], a.shape[0]],
                            "view": view, "shiny": shiny, "region": group["region"]})
    for name, (generation, shiny) in catalog["legacy_sheets"].items():
        known.add(name)
        with Image.open(raw / name) as image:
            a = np.array(image.convert("RGBA"))
        for i, (species_id, form) in enumerate(legacy_cells(generation)):
            r, c = divmod(i, 10)
            tile = a[5 + r * 69:69 + r * 69, 5 + c * 69:69 + c * 69].copy()
            tiles.append({"sheet": name, "row": r, "column": c, "view": "front", "shiny": shiny,
                          "region": generation.split(" /")[0], "family": "emerald", "tile": tile,
                          "matte_rgb": [165, 235, 255],
                          "expected_species": species_id, "expected_form": form, "pose_variant": i % 2,
                          "crop_xywh": [5 + c * 69, 5 + r * 69, 64, 64]})
        sources.append({"file": name, "sha256": sha((raw / name).read_bytes()), "size": [a.shape[1], a.shape[0]],
                        "view": "front", "shiny": shiny, "region": generation.split(" /")[0]})
    for duplicate, canonical in catalog["duplicate_files"].items():
        if sha((raw / duplicate).read_bytes()) != sha((raw / canonical).read_bytes()):
            raise ValueError(f"Declared duplicate changed: {duplicate}")
        sources.append({"file": duplicate, "duplicate_of": canonical, "sha256": sha((raw / duplicate).read_bytes())})
    unexpected = sorted(p.name for p in raw.glob("*") if p.is_file() and p.name not in known and not p.name.endswith(":Zone.Identifier"))
    if unexpected:
        raise ValueError(f"Uncatalogued raw files: {unexpected}")
    return tiles, sources


def palette(a):
    counts = Counter(map(tuple, a[a[..., 3] > 0, :3].tolist()))
    colors = Counter()
    for rgb, count in counts.items():
        h, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in rgb))
        if v < 0.18:
            color = "black"
        elif s < 0.16:
            color = "white" if v > 0.85 else "gray"
        else:
            hue = h * 360
            if hue < 15 or hue >= 345:
                color = "pink" if v > 0.7 and s < 0.65 else "red"
            elif hue < 45:
                color = "brown" if v < 0.72 else "orange"
            elif hue < 70:
                color = "yellow"
            elif hue < 165:
                color = "green"
            elif hue < 195:
                color = "cyan"
            elif hue < 260:
                color = "blue"
            elif hue < 290:
                color = "purple"
            else:
                color = "pink"
        colors[color] += count
    total = sum(counts.values())
    return [{"rgb": list(rgb), "hex": "#" + "".join(f"{c:02X}" for c in rgb), "pixels": n,
             "fraction": round(n / total, 6)} for rgb, n in counts.most_common()], [c for c, n in colors.most_common() if n / total >= .08][:3]


def definition(ref, entity, tile):
    species_id, form_id, gender = entity
    species = ref.species[species_id]
    name = ref.names[species_id, "9"]["name"]
    french = ref.names[species_id, "5"]["name"]
    form = next((f for f in ref.forms.values() if int(f["id"]) == form_id), None)
    form_name = ref.form_names.get(form_id, {}).get("form_name") or (form["form_identifier"].replace("-", " ") if form else "")
    if species_id == 351 and not form_name:
        form_name = "Normal Form"
    types = ref.mon_types[int(form["pokemon_id"]) if form else species_id]
    if species_id == 493 and form:
        types = [form["form_identifier"]]
    return {"species_id": species_id, "name_en": name, "name_fr": french,
            "genus_en": ref.names[species_id, "9"]["genus"], "form_id": form_id,
            "form_identifier": form["identifier"] if form else species["identifier"], "form_name_en": form_name,
            "form_name_fr": ref.form_names_fr.get(form_id, {}).get("form_name") or form_name,
            "gender": gender or ("genderless" if species["gender_rate"] == "-1" else "unspecified"),
            "palette": "shiny" if tile["shiny"] else "normal", "view": tile["view"],
            "sprite_style": "emerald" if tile["family"] == "emerald" else "black-white",
            "body_shape_taxonomy": ref.shapes[int(species["shape_id"])],
            "body_shape_taxonomy_description": ref.shape_descriptions[int(species["shape_id"])],
            "types_taxonomy": types,
            "taxonomy_is_not_a_visual_annotation": True}


def caption(d):
    parts = ["sharm_monster, pixel art monster sprite", d["name_en"]]
    if d["form_name_en"] and d["form_name_en"].lower() not in ("male", "female"):
        parts.append(d["form_name_en"])
    if d["gender"] in ("male", "female"):
        parts.append(d["gender"])
    parts.extend([f"{d['palette']} palette", "front view" if d["view"] == "front" else "back view",
                  "Emerald-style battle sprite" if d["sprite_style"] == "emerald" else "Black-and-White-style battle sprite"])
    return ", ".join(parts) + ". Single complete character, crisp pixel edges, transparent background."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=HERE / "dataset/raw")
    parser.add_argument("--reference-dir", type=Path, default=HERE / "dataset/reference")
    parser.add_argument("--output-dir", type=Path, default=HERE / "dataset/prepared/monster-clean-v2")
    parser.add_argument("--catalog", type=Path, default=HERE / "monster_sources.json")
    args = parser.parse_args()
    output = args.output_dir
    if (output / "dataset.json").exists():
        raise FileExistsError("Use a new version directory; existing prepared datasets are immutable.")
    if output.resolve() == args.raw_dir.resolve() or args.raw_dir.resolve() in output.resolve().parents:
        raise ValueError("Output must be outside raw sources")
    catalog = json.loads(args.catalog.read_text())
    reference = Reference(args.reference_dir)
    index = reference.index()
    tiles, sources = get_tiles(args.raw_dir, catalog)
    print(f"Source cells: {len(tiles)}", flush=True)
    for tile in tiles:
        cut = tile["tile"].copy()
        cut[np.all(cut[..., :3] == tile["matte_rgb"], axis=-1)] = 0
        matches = index.get(signature(cut), [])
        matches = [m for m in matches if m["family"] == tile["family"]
                   and ("shiny" in Path(m["path"]).parts) == tile["shiny"]
                   and ("back" in Path(m["path"]).parts) == (tile["view"] == "back")]
        if "source_species_id" in tile:
            if any(reference.entity(m["path"])[0] != tile["source_species_id"] for m in matches):
                raise ValueError(f"Pixel match contradicts printed label: {tile['sheet']}:{tile['source_label']}")
            suffix = tile["source_label"].split("-")[-1]
            if suffix in ("F", "M"):
                gender = {"F": "female", "M": "male"}[suffix]
                matches = [m for m in matches if reference.entity(m["path"])[2] == gender]
        if "expected_species" in tile:
            matches = [m for m in matches if reference.entity(m["path"])[0] == tile["expected_species"]]
        else:
            lo, hi = tile["species_range"]
            matches = [m for m in matches if lo <= reference.entity(m["path"])[0] <= hi]
        tile["matches"] = matches

    # Concordance between front/back and normal/shiny at the same source cell.
    groups = defaultdict(list)
    for tile in tiles:
        if tile["family"] == "sprites":
            groups[tile["region"], tile["row"], tile["column"]].append(tile)
    for key, group in groups.items():
        sets = [{reference.entity(m["path"]) for m in t["matches"]} for t in group if t["matches"]]
        entities = set.intersection(*sets) if sets else set()
        if not entities and sets:
            raise ValueError(f"Conflicting cross-sheet identities: {key}: {sets}")
        fallback = catalog["fallback_identities"].get(":".join(map(str, key)))
        labels = {t["source_label"] for t in group}
        if len(labels) != 1:
            raise ValueError(f"Different printed labels in a paired group: {key}: {labels}")
        label = next(iter(labels))
        suffix = label.split("-")[-1]
        type_order = catalog.get("form_label_type_order", {}).get(label[:3].lstrip("0"))
        if type_order is not None and suffix.isdigit() and "-" in label:
            type_name = reference.types[type_order[int(suffix)]]
            canonical_form = reference.forms[f"{reference.species[int(label[:3])]['identifier']}-{type_name}"]
            entities = {e for e in entities if e[1] == int(canonical_form["id"])}
            if sets and not entities:
                raise ValueError(f"Printed type form contradicts reference matches: {key}: {label}")
        if not entities and ("-" not in label or suffix in ("F", "M")):
            sid = int(label[:3])
            entities = {(sid, int(reference.default_forms[sid]["id"]), {"F":"female", "M":"male"}.get(suffix))}
        if not entities and fallback:
            sid = fallback["species_id"]
            forms = reference.default_forms[sid]
            sex = fallback.get("gender")
            entities = {(sid, int(forms["id"]), sex)}
        if len(entities) != 1:
            raise ValueError(f"Unresolved source identity: {key}: {label}: {entities}")
        for tile in group:
            tile["entities"] = entities

    accepted, deferred = [], []
    for count, tile in enumerate(tiles, 1):
        candidates = tile["matches"]
        entities = tile.get("entities") or {reference.entity(m["path"]) for m in candidates}
        if entities:
            candidates = [m for m in candidates if reference.entity(m["path"]) in entities]
        if not candidates and tile["family"] == "sprites" and entities:
            # First recheck static references: a foreground color can equal the cell background.
            for path in reference.tree_entries:
                if path.startswith("animated/") or not path.endswith(".png"):
                    continue
                if ("back" in Path(path).parts) != (tile["view"] == "back") or ("shiny" in Path(path).parts) != tile["shiny"]:
                    continue
                if path in reference.local_paths and reference.entity(path) in entities:
                    candidates.append({"family": "sprites", "path": path, "frame": 0})
        match, clean = None, None

        def attempt(possible):
            for item in possible:
                for frame, a in reference.frames(item["family"], item["path"]):
                    if frame != item["frame"]:
                        continue
                    result = match_alpha(tile["tile"], a, tile["matte_rgb"])
                    if result is not None:
                        return item, result
            return None, None

        match, clean = attempt(candidates)
        if match is None and tile["family"] == "sprites" and entities:
            match, clean = attempt(reference.animated_candidates(entities, tile["view"], tile["shiny"]))
        # Unverified cells are visible in the review, never silently accepted as clean.
        if match is None:
            tile["entities"] = entities
            deferred.append(tile)
        else:
            entity = reference.entity(match["path"])
            if tile["family"] == "emerald":
                entity = (entity[0], entity[1], None)  # Gen III artwork has no male/female sprite pair.
            tile.update({"clean": clean, "match": match, "entity": entity, "verification": "exact_rgb5_composite"})
            accepted.append(tile)
        if count % 250 == 0 or count == len(tiles):
            print(f"Reconciled {count}/{len(tiles)}; verified {len(accepted)}; pending {len(deferred)}", flush=True)

    # A second palette can verify the exact opacity mask of the same pose. If
    # that is unavailable, use a key color only when absent from the reference
    # foreground palette. Never erase a color that may be part of the monster.
    peers = defaultdict(list)
    for tile in accepted:
        peers[tile["family"], tile["region"], tile["row"], tile["column"], tile["view"]].append(tile)
    rejected = []
    for tile in deferred:
        raw = tile["tile"]
        bg = np.array(tile["matte_rgb"], dtype=np.uint8)
        observed = np.any(raw[..., :3] != bg, axis=-1) & (raw[..., 3] > 0)
        clean, entity, proof = None, None, None
        key = (tile["family"], tile["region"], tile["row"], tile["column"], tile["view"])
        for peer in peers[key]:
            mask = peer["clean"][..., 3] > 0
            if not (observed & ~mask).any():
                clean = raw.copy()
                clean[..., 3] = peer["clean"][..., 3]
                entity = peer["entity"]
                proof = {"paired_source": peer["sheet"], "paired_verification": peer["verification"],
                         "paired_reference": peer["match"]}
                tile["verification"] = "paired_palette_exact_alpha"
                break
        if clean is None and tile["family"] == "emerald":
            # The cyan matte is not a representable v<<3 color; all foreground
            # channels must be multiples of eight. The audited legacy table
            # supplies the identity, independently of modern sprite animation.
            if np.all(raw[observed, :3] % 8 == 0) and np.any(bg % 8 != 0):
                sid = tile["expected_species"]
                form_word = tile["expected_form"].split(" ")[0]
                identifier = f"castform-{form_word}" if sid == 351 and form_word not in ("", "normal") else None
                form = reference.forms[identifier] if identifier else reference.default_forms[sid]
                entity = (sid, int(form["id"]), None)
                clean = raw.copy()
                clean[~observed] = 0
                proof = {"identity": "audited legacy layout", "alpha": "non-15bit cyan matte; all foreground channels v<<3"}
                tile["verification"] = "audited_legacy_grid_unique_matte"
        if clean is None and tile["family"] == "sprites":
            entities = tile.get("entities", set())
            palette_colors = set()
            for (family, path), frames in list(reference.frames_cache.items()):
                if family != "sprites":
                    continue
                if ("back" in Path(path).parts) != (tile["view"] == "back") or ("shiny" in Path(path).parts) != tile["shiny"]:
                    continue
                if reference.entity(path) not in entities:
                    continue
                for _, a in frames:
                    palette_colors.update(map(tuple, (a[a[..., 3] > 0, :3] >> 3).tolist()))
            actual = set(map(tuple, (raw[observed, :3] >> 3).tolist()))
            if entities and palette_colors and tuple(bg >> 3) not in palette_colors:
                clean = raw.copy()
                clean[~observed] = 0
                entity = sorted(entities, key=str)[0]
                proof = {"identity": "printed dex/form label + canonical taxonomy; cross-sheet concordance when available", "alpha": "matte absent from canonical foreground palette",
                         "foreground_palette_rgb5": sorted(palette_colors),
                         "source_colors_not_in_reference": sorted(actual - palette_colors),
                         "pixel_geometry_is_not_an_exact_reference_match": True}
                tile["verification"] = "source_label_reference_palette_unique_matte"
        if clean is not None:
            clean[clean[..., 3] == 0] = 0
            tile.update({"clean": clean, "entity": entity, "match": proof})
            accepted.append(tile)
        else:
            record = {k: v for k, v in tile.items() if k not in ("tile", "matches", "entities")}
            record["reason"] = "opacity not sufficiently verified; never used for training"
            if tile.get("entities"):
                record["proposed_definition"] = definition(reference, sorted(tile["entities"], key=str)[0], tile)
            record["_tile"] = raw
            rejected.append(record)

    dump(args.reference_dir / "reconciliation.json", {"accepted": len(accepted), "rejected": [{k:v for k,v in r.items() if k != "_tile"} for r in rejected]})
    build_output(output, args, reference, sources, accepted, rejected)


def build_output(output, args, ref, sources, accepted, rejected):
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("images_128", "train_256", "train_512"):
        (output / folder).mkdir(exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=8)
    writes = []

    def save_images(image, filename, data):
        (output / "images_128" / filename).write_bytes(data)
        for size in (256, 512):
            image.resize((size, size), Image.Resampling.NEAREST).save(output / f"train_{size}" / filename)
    source_hashes = {r["file"]: r["sha256"] for r in sources}
    manifest, duplicates, by_pixels, inventory = [], [], {}, []
    for tile in accepted:
        clean = tile["clean"]
        # Integer fitting with a four-pixel margin preserves the source grid and
        # aspect ratio, without fractional resizing or smoothing.
        box = bbox(clean)
        x0, y0, x1, y1 = box
        content = clean[y0:y1, x0:x1].copy()
        scale = min(4, 120 // content.shape[1], 120 // content.shape[0])
        content = np.repeat(np.repeat(content, scale, axis=0), scale, axis=1)
        normalized = np.zeros((128, 128, 4), dtype=np.uint8)
        ox, oy = (128 - content.shape[1]) // 2, (128 - content.shape[0]) // 2
        normalized[oy:oy + content.shape[0], ox:ox + content.shape[1]] = content
        pixel_sha = sha(normalized.tobytes())
        d = definition(ref, tile["entity"], tile)
        variant = {k:d[k] for k in ("species_id","form_id","form_identifier","form_name_en","form_name_fr","gender","palette","view","sprite_style")}
        origin = {"sheet": tile["sheet"], "row": tile["row"], "column": tile["column"],
                  "crop_xywh": tile["crop_xywh"], "source_sha256": source_hashes[tile["sheet"]],
                  "printed_label": tile.get("source_label"), "matte_rgb": tile["matte_rgb"],
                  "verification":tile["verification"], "reference": tile["match"]}
        item = {**origin, "species_id":d["species_id"],"definition":variant,"status":"included"}
        if pixel_sha in by_pixels:
            existing = by_pixels[pixel_sha]
            if existing["species_id"] != d["species_id"]:
                raise ValueError("Cross-species pixel duplicate: requires explicit grouping review")
            existing["source_cells"].append(origin)
            if variant not in existing["equivalent_variants"]:
                existing["equivalent_variants"].append(variant)
            inventory.append({**item,"sample_id":existing["sample_id"],"status":"deduplicated"})
            duplicates.append({"sheet": tile["sheet"], "row": tile["row"], "column": tile["column"],
                               "duplicate_of": existing["sample_id"], "equivalent_variant":variant,
                               "reason": "identical centered RGBA pixels"})
            continue
        sid = f"{d['species_id']:04d}-{d['form_identifier']}-{d['gender']}-{d['palette']}-{d['view']}-{d['sprite_style']}-{pixel_sha[:10]}"
        image = Image.fromarray(normalized)
        folder = output / "images_128"
        folder.mkdir(exist_ok=True)
        filename = sid + ".png"
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        writes.append(pool.submit(save_images, image, filename, data))
        colors, color_names = palette(clean)
        d.update({"palette_rgb": colors, "palette_color_families_measured": color_names,
                  "description_en": caption(d), "description_fr":
                  f"Sprite de {d['name_fr']}, vue {'de face' if d['view']=='front' else 'de dos'}, "
                  f"forme {d['form_name_fr'] or 'standard'}, "
                  f"palette {'shiny' if d['palette']=='shiny' else 'normale'}, "
                  f"sexe {dict(male='mâle', female='femelle', genderless='asexué', unspecified='non spécifié')[d['gender']]}, "
                  f"style {d['sprite_style']}, personnage entier sur fond transparent.",
                  "description_provenance": "canonical identity + exact image/reference reconciliation; no invented lore"})
        split_group = str(d["species_id"])
        # Same existing species split rule, with the canonical JSON-string digest.
        group_hash = sha(json.dumps(split_group, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
        split = "validation" if int(group_hash[:8], 16) % 10 == 0 else "train"
        row = {"sample_id": sid, "file_name": filename, "file": f"images_128/{filename}",
               "text": d["description_en"], "species_id": d["species_id"], "species": d["name_en"],
               "group_id": split_group, "split": split, "label": d["name_en"], "definition": d,
               "pixel_sha256": pixel_sha, "image_sha256": sha(data),
               "native_cell_size": tile["tile"].shape[0], "native_bbox_xyxy": list(box),
               "target_bbox_xyxy": [ox, oy, ox + content.shape[1], oy + content.shape[0]],
               "integer_scale_from_native": scale, "foreground_pixels": int((normalized[..., 3] > 0).sum()),
               "verification": tile["verification"], "source_cells": [origin],"equivalent_variants":[variant]}
        manifest.append(row)
        inventory.append({**item,"sample_id":sid})
        by_pixels[pixel_sha] = row
        if len(manifest) % 500 == 0:
            print(f"Prepared unique images: {len(manifest)}", flush=True)
    for future in writes:
        future.result()
    pool.shutdown()
    jsonl(output / "manifest.jsonl", manifest)
    for size in (128, 256, 512):
        folder = output / ("images_128" if size == 128 else f"train_{size}")
        jsonl(folder / "metadata.jsonl", [{"file_name": r["file_name"], "text": r["text"],
              "group_id": r["group_id"], "species_id": r["species_id"], "split": r["split"],
              "sample_id": r["sample_id"], "label": r["label"]} for r in manifest])
        for split in ("train", "validation"):
            jsonl(folder / f"{split}.jsonl", [{"file_name": r["file_name"], "text": r["text"],
                  "group_id": r["group_id"], "species_id": r["species_id"]} for r in manifest if r["split"] == split])
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["file", "species_id", "species", "split", "caption", "shiny", "form", "view", "style"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in manifest:
            d = row["definition"]
            writer.writerow({"file": row["file"], "species_id": row["species_id"], "species": row["species"],
                             "split": row["split"], "caption": row["text"], "shiny": d["palette"] == "shiny",
                             "form": d["form_identifier"], "view": d["view"], "style": d["sprite_style"]})
    captions = sorted(set(r["text"] for r in manifest))
    jsonl(output / "caption-corpus.jsonl", [{"id": sha(json.dumps(text, ensure_ascii=False, sort_keys=True,
          separators=(",", ":")).encode()), "text": text} for text in captions])
    definitions = {}
    for row in manifest:
        d = row["definition"]
        species = definitions.setdefault(str(row["species_id"]), {"name_en": d["name_en"],
                    "name_fr": d["name_fr"], "genus_en":d["genus_en"],"species_id":row["species_id"],
                    "body_shape_taxonomy":d["body_shape_taxonomy"],
                    "taxonomy_is_not_a_visual_annotation":True,"forms":{},"samples":[]})
        species["samples"].append(row["sample_id"])
        for variant in row["equivalent_variants"]:
            form = species["forms"].setdefault(variant["form_identifier"],
                {"id":variant["form_id"],"name_en":variant["form_name_en"],"name_fr":variant["form_name_fr"],"samples":[]})
            form["samples"].append(row["sample_id"])
    dump(output / "definitions.json", definitions)
    dump(output / "sources.json", {"sources": sources, "references": ref.lock, "source_catalog": json.loads(args.catalog.read_text())})
    dump(output / "duplicates.json", duplicates)
    quarantine = output / "quarantine"
    quarantine.mkdir(exist_ok=True)
    for record in rejected:
        pixels = record.pop("_tile")
        name = f"{Path(record['sheet']).stem}-r{record['row']:02d}-c{record['column']:02d}.png"
        Image.fromarray(pixels).save(quarantine / name)
        record["original_cell_file"] = f"quarantine/{name}"
        inventory.append({**record,"status":"quarantine"})
    dump(output / "unresolved.json", rejected)
    jsonl(output / "source_cells.jsonl", inventory)
    report = {"schema_version": 1, "dataset_version": output.name, "preparation_version":VERSION,"raw_files": len(sources),
              "unique_source_sheets": len(sources) - sum("duplicate_of" in r for r in sources),
              "verified_source_cells": len(accepted), "unique_images": len(manifest),
              "duplicate_cells": len(duplicates), "unresolved_cells": len(rejected),
              "species": len(definitions), "by_style": dict(Counter(r["definition"]["sprite_style"] for r in manifest)),
              "by_view": dict(Counter(r["definition"]["view"] for r in manifest)),
              "by_palette": dict(Counter(r["definition"]["palette"] for r in manifest)),
              "by_split": dict(Counter(r["split"] for r in manifest)),
              "by_verification": dict(Counter(r["verification"] for r in manifest)),
              "manifest_sha256": sha((output / "manifest.jsonl").read_bytes()),
              "preparation_script_sha256": sha(Path(__file__).read_bytes()),
              "training_started": False, "human_review": False, "reference_matching_is_not_human_review": True}
    make_review(output, manifest)
    dump(output / "dataset.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def make_review(output, manifest):
    review = output / "review"
    review.mkdir(exist_ok=True)
    links = []
    for start in range(0, len(manifest), 80):
        page = start // 80 + 1
        rows = manifest[start:start + 80]
        cards = []
        canvas = Image.new("RGB", (8 * 160, 10 * 160), "#DDDDE2")
        draw = ImageDraw.Draw(canvas)
        for i, r in enumerate(rows):
            x, y = (i % 8) * 160, (i // 8) * 160
            for gy in range(0, 128, 8):
                for gx in range(0, 128, 8):
                    draw.rectangle((x + 16 + gx, y + gy, x + 23 + gx, y + gy + 7),
                                   fill="#CCCCCC" if (gx // 8 + gy // 8) % 2 else "#EEEEEE")
            with Image.open(output / r["file"]) as im:
                canvas.paste(im, (x + 16, y), im)
            d = r["definition"]
            draw.text((x + 3, y + 129), f"{r['species_id']} {r['species']}", fill="black")
            draw.text((x + 3, y + 143), f"{d['view']} {d['palette']} {d['gender'][:1]}", fill="black")
            cards.append(f"<article><img loading='lazy' src='../{html.escape(r['file'])}'>"
                         f"<b>{html.escape(r['sample_id'])}</b><p>{html.escape(r['text'])}</p>"
                         f"<details><summary>Definition / provenance</summary><pre>{html.escape(json.dumps(r, ensure_ascii=False, indent=2))}</pre></details></article>")
        canvas.save(review / f"contact-{page:03d}.png")
        (review / f"page-{page:03d}.html").write_text("<!doctype html><meta charset='utf-8'><style>"
            "body{font:14px system-ui;background:#eee}main{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}"
            "article{background:white;padding:12px;overflow-wrap:anywhere}img{width:256px;height:256px;image-rendering:pixelated;"
            "background:repeating-conic-gradient(#ddd 0% 25%,#fff 0% 50%) 0/16px 16px}pre{white-space:pre-wrap}</style>"
            f"<h1>Monster dataset — page {page}</h1><a href='index.html'>Index</a><main>{''.join(cards)}</main>", encoding="utf-8")
        links.append(f"<li><a href='page-{page:03d}.html'>Page {page}</a> — <a href='contact-{page:03d}.png'>Planche contact</a></li>")
    (review / "index.html").write_text("<!doctype html><meta charset='utf-8'><h1>Sharm Image Monster Pixel Art</h1>"
        "<p>Galerie de contrôle, sans annotation humaine inventée. Images à 128px natifs, affichage nearest-neighbor.</p>"
        f"<ul>{''.join(links)}</ul>", encoding="utf-8")


if __name__ == "__main__":
    main()
