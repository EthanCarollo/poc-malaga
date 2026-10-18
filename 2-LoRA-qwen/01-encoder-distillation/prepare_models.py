#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from huggingface_hub import snapshot_download
from sharm_image21.core import ORIGINAL, STUDENT, STUDENT_REVISION, TEACHER_REVISION


def main():
    snapshot_download("Qwen/Qwen-Image-2.1", revision=TEACHER_REVISION, local_dir=ORIGINAL, max_workers=2,
                       allow_patterns=["model_index.json", "transformer/*", "vae/*", "scheduler/*",
                                       "text_encoder/*", "processor/*", "LICENSE", "README.md"])
    snapshot_download("Qwen/Qwen3-VL-4B-Instruct-FP8", revision=STUDENT_REVISION, local_dir=STUDENT, max_workers=2,
                       allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    print("Poids source vérifiés aux révisions figées.", flush=True)


if __name__ == "__main__":
    main()
