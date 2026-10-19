"""Regressions for the six audited normal/shiny sprite sheets."""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sprite_preparation", ROOT / "02-sprite-lora/prepare_dataset.py")
sprites = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sprites)


class SpriteDatasetChecks(unittest.TestCase):
    def test_missing_species_and_extra_castform_cells(self):
        kanto = sprites.sheet_cells("Kanto / Gen 1")
        johto = sprites.sheet_cells("Johto / Gen 2")
        hoenn = sprites.sheet_cells("Hoenn / Gen 3")
        self.assertEqual((len(kanto), len(johto), len(hoenn)), (302, 198, 270))
        self.assertEqual(kanto[-2:], [(151, ""), (151, "")])
        self.assertEqual(johto[96:100], [(200, ""), (200, ""), (202, ""), (202, "")])
        self.assertEqual(johto[-2:], [(251, ""), (251, "")])
        self.assertEqual(hoenn[148:152], [(326, ""), (326, ""), (328, ""), (328, "")])
        self.assertEqual(hoenn[190], (348, ""))
        self.assertEqual(hoenn[196:200], [(351, form) for form in
                         ("normal form", "sunny form", "rainy form", "snowy form")])
        self.assertEqual(hoenn[200], (352, ""))
        self.assertEqual(hoenn[-2:], [(386, ""), (386, "")])
        ids = {species for cells in (kanto, johto, hoenn) for species, _ in cells}
        self.assertEqual(ids, set(range(1, 387)) - {201, 327})
        self.assertEqual(len(sprites.SHEETS), 6)

    def test_shiny_and_weather_captions_do_not_use_normal_color(self):
        species = {"name": "Castform", "genus": "Weather Pokémon", "color": "gray",
                   "shape": "ball", "pokedex_note": "Its form changes with the weather."}
        self.assertIn("gray coloration", sprites.make_caption(species))
        for shiny in (False, True):
            caption = sprites.make_caption(species, shiny, "snowy form")
            self.assertNotIn("gray coloration", caption)
            self.assertIn("snowy form", caption)
            self.assertEqual("shiny Castform" in caption, shiny)
        self.assertNotIn("gray coloration", sprites.make_caption(species, True))

    def test_alpha_removes_enclosed_background_and_preserves_nearby_color(self):
        tile = Image.new("RGBA", (16, 16), (*sprites.BACKGROUND, 255))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((2, 2, 13, 13), fill=(200, 10, 20, 255))
        draw.rectangle((6, 6, 9, 9), fill=(*sprites.BACKGROUND, 255))
        tile.putpixel((0, 0), (165, 235, 254, 255))
        self.assertEqual(sprites.remove_background_color(tile, sprites.BACKGROUND), 129)
        self.assertEqual(tile.getpixel((7, 7)), (0, 0, 0, 0))
        self.assertEqual(tile.getpixel((15, 15)), (0, 0, 0, 0))
        self.assertEqual(tile.getpixel((0, 0)), (165, 235, 254, 255))
        self.assertEqual(tile.getpixel((3, 3)), (200, 10, 20, 255))

    def test_cli_writes_correct_labels_groups_and_source_hashes_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "dataset"
            output.mkdir()
            traits = {str(i): {"name": f"Species {i}", "genus": "Test Pokémon", "color": "gray",
                              "shape": "ball", "pokedex_note": "Fixture."} for i in range(1, 387)}
            (output / "pokemon_traits.json").write_text(json.dumps(traits), encoding="utf-8")
            marked = {"21229.png": 98, "21228.png": 196, "21650.png": 199}
            for name, generation, _ in sprites.SHEETS:
                count = len(sprites.sheet_cells(generation))
                sheet = Image.new("RGB", (695, 5+69*((count+9)//10)), sprites.BACKGROUND)
                if name in marked:
                    row, col = divmod(marked[name], 10)
                    x, y = 5+col*69, 5+row*69
                    ImageDraw.Draw(sheet).rectangle((x+10, y+10, x+15, y+15), fill="red")
                sheet.save(root / name)
            with patch.object(sys, "argv", ["prepare_dataset.py", "--sheet-dir", str(root),
                                           "--output-dir", str(output), "--training-size", "64"]), \
                    patch.object(sprites, "urlopen", side_effect=AssertionError("Unexpected network")), \
                    contextlib.redirect_stdout(io.StringIO()):
                sprites.main()
            with (output / "manifest.csv").open(encoding="utf-8") as stream:
                rows = {row["sheet"]: row for row in csv.DictReader(stream)}
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows["21229.png"]["species_id"], "202")
            self.assertEqual(rows["21650.png"]["label"], "shiny Species 351, snowy form")
            self.assertEqual(rows["21228.png"]["species_id"], "351")
            self.assertTrue(all(len(row["source_sha256"]) == 64 for row in rows.values()))
            metadata = [json.loads(line) for line in (output / "train_64/metadata.jsonl").read_text().splitlines()]
            self.assertEqual(sorted(row["group_id"] for row in metadata), ["202", "351", "351"])
            sources = json.loads((output / "sources.json").read_text())
            self.assertEqual(len(sources["sheets"]), 6)
            self.assertEqual(sources["sprite_count"], 3)


if __name__ == "__main__":
    unittest.main()
