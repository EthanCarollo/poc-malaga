#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from huggingface_hub import snapshot_download
from sharm_image21.core import ORIGINAL, STUDENT, STUDENT_REVISION, TEACHER_REVISION


def main():
    parser = argparse.ArgumentParser(description="Download pinned models, optionally in disk-budgeted stages.")
    parser.add_argument("--stage", choices=("all", "teacher", "inference"), default="all")
    args = parser.parse_args()
    original_patterns = ["model_index.json", "processor/*", "LICENSE", "README.md",
                         "text_encoder/*.json", "transformer/*.json", "vae/*.json", "scheduler/*"]
    if args.stage in ("all", "teacher"):
        original_patterns.append("text_encoder/*")
    if args.stage in ("all", "inference"):
        original_patterns.extend(["transformer/*", "vae/*"])
    snapshot_download("Qwen/Qwen-Image-2.1", revision=TEACHER_REVISION, local_dir=ORIGINAL, max_workers=2,
                       allow_patterns=original_patterns)
    student_patterns = ["*.json", "*.txt", "*.model", "*.jinja"]
    if args.stage in ("all", "inference"):
        student_patterns.append("*.safetensors")
    snapshot_download("Qwen/Qwen3-VL-4B-Instruct-FP8", revision=STUDENT_REVISION, local_dir=STUDENT, max_workers=2,
                       allow_patterns=student_patterns)
    print("Poids source vérifiés aux révisions figées.", flush=True)


if __name__ == "__main__":
    main()
