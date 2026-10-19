#!/usr/bin/env python3
"""Download a pinned PokéAPI reference, never training images from this cache."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

SPRITES_REVISION = "7a7d66b1620e646f69afd48dc1273da9033ede9c"
DATA_REVISION = "6fd69dfe90465ba96534f6816fc817737d1f97f6"
SPRITE_PREFIX = "sprites/pokemon/versions/generation-v/black-white"
CSV_FILES = ("pokemon_species", "pokemon_species_names", "pokemon", "pokemon_forms",
             "pokemon_form_names", "pokemon_shapes", "pokemon_shape_prose", "pokemon_colors",
             "pokemon_types", "types", "type_names")


def download(url: str) -> bytes:
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": "SharmMonsterDataset/1.0"}), timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def git_blob_hash(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=Path(__file__).parent / "dataset/reference")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root = args.reference_dir
    root.mkdir(parents=True, exist_ok=True)
    lock = {"sprites_revision": SPRITES_REVISION, "pokeapi_revision": DATA_REVISION}
    for name in CSV_FILES:
        url = f"https://raw.githubusercontent.com/PokeAPI/pokeapi/{DATA_REVISION}/data/v2/csv/{name}.csv"
        data = download(url)
        (root / f"{name}.csv").write_bytes(data)
        lock.setdefault("csv_sha256", {})[name] = hashlib.sha256(data).hexdigest()
    pokemon = {row["id"]: row for row in csv.DictReader((root / "pokemon.csv").open(encoding="utf-8"))}
    allowed = {str(i) for i in range(1, 650)}
    for row in csv.DictReader((root / "pokemon_forms.csv").open(encoding="utf-8")):
        mon = pokemon[row["pokemon_id"]]
        if int(mon["species_id"]) <= 649 and int(row["introduced_in_version_group_id"] or 999) <= 14:
            allowed.add(row["pokemon_id"])
    entries = []
    for prefix, folder, tree_name in ((SPRITE_PREFIX, "sprites", "sprite-tree.json"),
            ("sprites/pokemon/versions/generation-iii/emerald", "emerald", "emerald-tree.json")):
        tree = json.loads(download(f"https://api.github.com/repos/PokeAPI/sprites/git/trees/"
                                   f"{SPRITES_REVISION}:{prefix}?recursive=1"))
        if tree.get("truncated"):
            raise RuntimeError("Incomplete GitHub reference tree")
        (root / tree_name).write_text(json.dumps(tree), encoding="utf-8")
        for entry in tree["tree"]:
            path = entry["path"]
            stem = Path(path).stem.split("-", 1)[0]
            if entry["type"] != "blob" or not path.endswith(".png"):
                continue
            if folder == "sprites":
                keep = not path.startswith("animated/") and stem in allowed
            else:
                keep = path.startswith("animated/") and stem.isdigit() and (int(stem) <= 386 or stem in {"10013", "10014", "10015"})
            if keep:
                entries.append({**entry, "prefix": prefix, "folder": folder})

    def fetch(entry):
        target = root / entry["folder"] / entry["path"]
        if target.is_file() and git_blob_hash(target.read_bytes()) == entry["sha"]:
            return
        data = download(f"https://raw.githubusercontent.com/PokeAPI/sprites/{SPRITES_REVISION}/"
                        f"{entry['prefix']}/{entry['path']}")
        if git_blob_hash(data) != entry["sha"]:
            raise RuntimeError(f"Reference checksum mismatch: {entry['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, entry) for entry in entries]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 250 == 0 or index == len(futures):
                print(f"References: {index}/{len(futures)}", flush=True)
    lock["sprite_files"] = len(entries)
    (root / "reference-lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
