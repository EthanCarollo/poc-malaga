#!/usr/bin/env python3
"""Compatibility entry point for the sprite dataset preparation CLI."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "02-sprite-lora/prepare_dataset.py"), run_name="__main__")
