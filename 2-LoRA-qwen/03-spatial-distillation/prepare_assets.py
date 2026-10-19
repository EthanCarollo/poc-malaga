#!/usr/bin/env python3
"""Download labelled, reproducible RGBA assets from PokeAPI's sprite repository."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import io
import json
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from PIL import Image
from sharm_image21.spatial_data import sha256, write_json


def fetch(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "Sharm-spatial-assets/1.0"}), timeout=30) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "cache/spatial/assets-pokeapi")
    parser.add_argument("--species", type=int, default=386)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 3 <= args.species <= 386 or not 1 <= args.workers <= 8:
        raise ValueError("De 3 à 386 espèces et de 1 à 8 téléchargements parallèles requis.")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = args.output / "source.json"
    if provenance.exists():
        source = json.loads(provenance.read_text(encoding="utf-8"))
        if source["species"] != args.species:
            raise ValueError("Un autre corpus existe ici ; choisir un autre dossier.")
    else:
        commit = json.loads(fetch("https://api.github.com/repos/PokeAPI/sprites/commits/master"))["sha"]
        if not re.fullmatch(r"[a-f0-9]{40}", commit):
            raise ValueError("Révision GitHub invalide.")
        source = {"repository": "https://github.com/PokeAPI/sprites", "revision": commit, "species": args.species,
                  "variants": ["default", "shiny"], "style": "front battle sprites; distinct from the original overworld sheets"}
        write_json(provenance, source)
    spec = importlib.util.spec_from_file_location("sprite_traits", ROOT / "02-sprite-lora/prepare_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def prepare(species_id):
        trait_path = args.output / f"species-{species_id}.json"
        traits = json.loads(trait_path.read_text(encoding="utf-8")) if trait_path.exists() else module.fetch_species_trait(species_id)
        if not trait_path.exists():
            write_json(trait_path, traits)
        records = []
        for variant in source["variants"]:
            name = f"pokemon-{species_id}-{variant}.png"
            target = args.output / name
            stamp = target.with_suffix(".source.json")
            url = (f"https://raw.githubusercontent.com/PokeAPI/sprites/{source['revision']}/sprites/pokemon/"
                   f"{'shiny/' if variant == 'shiny' else ''}{species_id}.png")
            if target.exists() and stamp.exists():
                metadata = json.loads(stamp.read_text())
                if metadata["url"] != url or metadata["image_sha256"] != sha256(target):
                    raise ValueError(f"Asset modifié : {target}")
            else:
                content = fetch(url)
                with Image.open(io.BytesIO(content)) as raw:
                    image = raw.convert("RGBA")
                alpha = image.getchannel("A")
                if alpha.getbbox() is None or alpha.getextrema()[0] != 0:
                    raise ValueError(f"Sprite vide ou sans alpha : {url}")
                image.save(target)
                write_json(stamp, {"url": url, "source_sha256": hashlib.sha256(content).hexdigest(),
                                   "image_sha256": sha256(target)})
            # Shiny coloration may differ from the canonical species color.
            label = f"{traits['name']}, the {traits['genus']}, {variant} pixel-art sprite"
            records.append({"id": f"pokemon-{species_id}-{variant}", "group_id": f"pokemon-{species_id}",
                            "label": label, "file_name": name, "source_url": url, "image_sha256": sha256(target)})
        return records

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prepare, species_id) for species_id in range(1, args.species+1)]
        for index, future in enumerate(as_completed(futures)):
            rows.extend(future.result())
            if (index+1) % 20 == 0 or index+1 == args.species:
                print(f"Assets PokéAPI : {index+1}/{args.species} espèces", flush=True)
    rows.sort(key=lambda r: r["id"])
    path = args.output / "assets.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
    print(f"Manifeste RGBA : {path}", flush=True)


if __name__ == "__main__":
    main()
