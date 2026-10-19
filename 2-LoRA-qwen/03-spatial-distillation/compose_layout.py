#!/usr/bin/env python3
"""Place existing PNG assets deterministically inside explicit boxes."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from sharm_image21.spatial_data import compose, write_json
from sharm_image21.image_io import save_png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True, help="JSON : resolution, objects [{label,file_name,box}].")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    image, records = compose(layout["resolution"], layout["objects"], args.layout.parent)
    save_png(image, args.output, preview_path=args.output.with_name(args.output.stem+"-preview.png"), require_transparency=True)
    write_json(args.output.with_suffix(".layout.json"), {"resolution": layout["resolution"], "objects": records})


if __name__ == "__main__":
    main()
