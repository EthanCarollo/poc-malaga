#!/usr/bin/env python3
"""Cut the local Pokémon overworld sheets into 64px RGBA training images."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parents[1]
SHEETS = (
    ("21228.png", "Hoenn / Gen 3", 252, 135),
    ("21229.png", "Johto / Gen 2", 152, 100),
    ("21230.png", "Kanto / Gen 1", 1, 151),
)
CELL_SIZE = 64
PITCH = 69
ORIGIN = 5
BACKGROUND = (165, 235, 255)  # #A5EBFF, the flat blue used behind every cell.
POKEAPI_BASE = "https://pokeapi.co/api/v2/pokemon-species/"
DISPLAY_NAME_OVERRIDES = {
    "farfetchd": "Farfetch'd",
    "ho-oh": "Ho-Oh",
    "mr-mime": "Mr. Mime",
    "mime-jr": "Mime Jr.",
    "nidoran-f": "Nidoran♀",
    "nidoran-m": "Nidoran♂",
}


def parse_color(value: str) -> tuple[int, int, int]:
    value = value.removeprefix("#")
    if len(value) != 6:
        raise argparse.ArgumentTypeError("La couleur doit être au format RRGGBB ou #RRGGBB.")
    try:
        return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]
    except ValueError as error:
        raise argparse.ArgumentTypeError("La couleur contient un chiffre hexadécimal invalide.") from error


def remove_background_color(tile: Image.Image, color: tuple[int, int, int]) -> int:
    """Make every exact match of the flat sheet background transparent."""
    # `Image.convert("RGBA")` returns a separate image even when `tile` is
    # already RGBA. Keep the caller's image in that case so the alpha edits
    # are present when the caller saves the tile.
    if tile.mode != "RGBA":
        tile = tile.convert("RGBA")
    pixels = tile.load()
    width, height = tile.size
    for y in range(height):
        for x in range(width):
            if pixels[x, y][:3] == color:
                pixels[x, y] = (0, 0, 0, 0)

    return tile.getchannel("A").tobytes().count(255)


def normalize_flavor_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value.replace("POKéMON", "Pokémon")).strip()
    if len(value) > 200:
        value = value[:197].rsplit(" ", 1)[0] + "…"
    return value


def fetch_species_trait(species_id: int) -> dict[str, str]:
    request = Request(
        f"{POKEAPI_BASE}{species_id}/",
        headers={"User-Agent": "pokemon-overworld-lora-dataset/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.load(response)
            genus = next(
                entry["genus"]
                for entry in payload["genera"]
                if entry["language"]["name"] == "en"
            )
            flavor_text = next(
                entry["flavor_text"]
                for entry in payload["flavor_text_entries"]
                if entry["language"]["name"] == "en"
            )
            identifier = payload["name"]
            return {
                "name": DISPLAY_NAME_OVERRIDES.get(identifier, identifier.replace("-", " ").title()),
                "genus": genus,
                "color": payload["color"]["name"],
                "shape": payload["shape"]["name"].replace("-", " "),
                "pokedex_note": normalize_flavor_text(flavor_text),
            }
        except (URLError, TimeoutError, KeyError, StopIteration, json.JSONDecodeError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Impossible de récupérer les traits PokéAPI du Pokémon #{species_id}: {last_error}")


def load_species_traits(output_dir: Path) -> dict[str, dict[str, str]]:
    """Cache English Pokédex genus/color/shape and one short Pokédex note."""
    cache_path = output_dir / "pokemon_traits.json"
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if all(str(species_id) in cached for species_id in range(1, 387)):
                return cached
        except (OSError, json.JSONDecodeError):
            pass

    traits: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch_species_trait, species_id): species_id
            for species_id in range(1, 387)
        }
        for future in as_completed(futures):
            species_id = futures[future]
            traits[str(species_id)] = future.result()
    cache_path.write_text(json.dumps(traits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return traits


def make_caption(species: dict[str, str]) -> str:
    return (
        f"pkm_ow_sprite, {species['name']}, the {species['genus']}; "
        f"{species['color']} coloration, {species['shape']} body. "
        f"Pokédex note: {species['pokedex_note']} "
        "Classic handheld-game overworld pixel-art sprite, single character, transparent background."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "dataset",
        help="Dossier de sortie (par défaut : 2-LoRA-qwen/dataset).",
    )
    parser.add_argument(
        "--training-size",
        type=int,
        default=1024,
        help="Taille carrée des copies d’entraînement ; doit être multiple de 32.",
    )
    parser.add_argument(
        "--background",
        type=parse_color,
        default=BACKGROUND,
        help="Couleur de fond à retirer, en RRGGBB (par défaut : A5EBFF).",
    )
    args = parser.parse_args()
    if args.training_size < CELL_SIZE or args.training_size % 32:
        parser.error("--training-size doit être au moins 64 et être un multiple de 32.")

    output_dir = args.output_dir.expanduser().resolve()
    tiles_dir = output_dir / "sprites_64"
    training_dir = output_dir / f"train_{args.training_size}"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    traits = load_species_traits(output_dir)
    records: list[dict[str, object]] = []
    total_cells = 0
    total_expected_sprites = sum(species_count * 2 for _, _, _, species_count in SHEETS)
    empty_cells = 0
    for filename, generation, first_species_id, species_count in SHEETS:
        sheet_path = SCRIPT_DIR / filename
        if not sheet_path.is_file():
            raise FileNotFoundError(f"Planche locale introuvable : {sheet_path}")
        with Image.open(sheet_path) as source:
            source = source.convert("RGB")
            if source.getpixel((ORIGIN, ORIGIN)) != args.background:
                raise ValueError(
                    f"Le fond attendu {args.background} n’est pas présent en ({ORIGIN}, {ORIGIN}) dans {filename}. "
                    "Vérifie la planche avant de changer la couleur avec --background."
                )
            columns = (source.width - ORIGIN - CELL_SIZE) // PITCH + 1
            rows = (source.height - ORIGIN - CELL_SIZE) // PITCH + 1
            total_cells += rows * columns
            kept = 0
            for row in range(rows):
                for column in range(columns):
                    species_offset = row * (columns // 2) + column // 2
                    if species_offset >= species_count:
                        continue  # Ignore footer artwork/credits after the last Kanto Pokémon.
                    left = ORIGIN + column * PITCH
                    top = ORIGIN + row * PITCH
                    tile = source.crop((left, top, left + CELL_SIZE, top + CELL_SIZE)).convert("RGBA")
                    colors = tile.getcolors(maxcolors=2)
                    if colors and len(colors) == 1 and colors[0][1] == (255, 255, 255, 255):
                        empty_cells += 1
                        continue  # Some source-sheet slots are blank white placeholders.
                    occupied_pixels = remove_background_color(tile, args.background)
                    if occupied_pixels < 12:
                        continue

                    species_id = first_species_id + species_offset
                    species = traits[str(species_id)]
                    caption = make_caption(species)
                    stem = f"{Path(filename).stem}_r{row:02d}_c{column:02d}"
                    tile_path = tiles_dir / f"{stem}.png"
                    training_path = training_dir / f"{stem}.png"
                    tile.save(tile_path)
                    tile.resize((args.training_size, args.training_size), Image.Resampling.NEAREST).save(training_path)
                    records.append(
                        {
                            "file": training_path.relative_to(output_dir).as_posix(),
                            "preview_64": tile_path.relative_to(output_dir).as_posix(),
                            "sheet": filename,
                            "generation": generation,
                            "species_id": species_id,
                            "species": species["name"],
                            "caption": caption,
                            "row": row,
                            "column": column,
                            "occupied_pixels": occupied_pixels,
                        }
                    )
                    kept += 1
            print(
                f"{filename}: {source.width}×{source.height}, grille {columns}×{rows}, "
                f"cellules Pokédex utilisées {species_count * 2}, sprites extraits {kept}."
            )

    if not records:
        raise RuntimeError("Aucune tuile non vide détectée ; aucun dataset n’a été produit.")

    # Remove only stale files bearing this script's generated sheet/row/column
    # naming pattern (for example footer cells left by an older run).
    expected_names = {
        Path(str(record[key])).name
        for record in records
        for key in ("file", "preview_64")
    }
    generated_prefixes = tuple(f"{Path(filename).stem}_r" for filename, *_ in SHEETS)
    removed_stale = 0
    for folder in (tiles_dir, training_dir):
        for image_path in folder.glob("*.png"):
            if image_path.name.startswith(generated_prefixes) and image_path.name not in expected_names:
                image_path.unlink()
                removed_stale += 1

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    metadata_path = training_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for record in records:
            metadata_file.write(
                json.dumps(
                    {
                        "file_name": Path(str(record["file"])).name,
                        "text": record["caption"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"Cellules examinées : {total_cells} ; sprites Pokédex attendus : "
        f"{total_expected_sprites} ; images conservées : {len(records)}."
    )
    if empty_cells:
        print(f"Cases blanches vides ignorées : {empty_cells}.")
    if removed_stale:
        print(f"Anciennes images générées supprimées : {removed_stale}.")
    print(f"Aperçus 64×64 : {tiles_dir}")
    print(f"Images d’entraînement {args.training_size}×{args.training_size} : {training_dir}")
    print(f"Légendes image par image : {metadata_path}")
    print(f"Inventaire : {manifest_path}")


if __name__ == "__main__":
    main()
