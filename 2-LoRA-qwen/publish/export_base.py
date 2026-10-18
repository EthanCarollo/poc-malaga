#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from sharm_image21.core import (ARTIFACTS, ORIGINAL, ROOT, STUDENT, WEIGHTS,
                                atomic_json, file_hash, model_identity)
from sharm_image21.model_cards import base_card, measure_footprint, package_comparisons


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_presentation(destination, validation, fingerprint):
    source = ARTIFACTS / "base-comparisons"
    if validation.get("comparison_sha256") != file_hash(source / "comparison.json"):
        raise ValueError("La galerie VL ne correspond pas au rapport de validation.")
    comparison = package_comparisons(source, destination / "comparisons/encoder", "encoder", fingerprint)
    footprint = measure_footprint(ORIGINAL, STUDENT, ARTIFACTS / "projector")
    atomic_json(destination / "footprint.json", footprint)
    atomic_json(destination / "base-validation.json", validation)
    (destination / "README.md").write_text(base_card(comparison, footprint, validation), encoding="utf-8")


def write_checksums(destination):
    checksums = {str(path.relative_to(destination)): file_hash(path) for path in destination.rglob("*")
                 if path.is_file() and path != destination / "checksums.json"}
    atomic_json(destination / "checksums.json", checksums)


def main():
    destination = WEIGHTS / "sharm-image-2.1-vl4b-fp8"
    projector_dir = ARTIFACTS / "projector"
    validation = json.loads((ARTIFACTS / "base-validation.json").read_text())
    fingerprint = file_hash(projector_dir / "model.safetensors")
    if not validation.get("passed") or validation["projector_sha256"] != fingerprint:
        raise ValueError("Export d'une base non validée refusé.")
    if (destination / "sharm_config.json").exists():
        previous = json.loads((destination / "sharm_config.json").read_text())
        if previous["projector_sha256"] != fingerprint:
            raise ValueError("Une base d'une autre révision existe dans le dossier de sortie.")
        write_presentation(destination, validation, fingerprint)
        write_checksums(destination)
        print(f"Base déjà exportée, fiche et comparaisons actualisées : {destination}", flush=True)
        return
    staging = destination.with_name(destination.name + ".staging")
    staging.mkdir(mode=0o700, parents=True, exist_ok=True)
    for component in ("transformer", "vae", "scheduler"):
        shutil.copytree(ORIGINAL / component, staging / component, copy_function=link_or_copy, dirs_exist_ok=True)
    for path in STUDENT.iterdir():
        if not path.is_file() or path.name == "README.md":
            continue
        if path.suffix == ".safetensors" or path.name in ("config.json", "generation_config.json", "model.safetensors.index.json"):
            target = staging / "text_encoder" / path.name
        elif path.name.startswith(("tokenizer", "preprocessor", "video_preprocessor", "chat_template", "vocab", "merges")):
            target = staging / "processor" / path.name
        else:
            continue
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            link_or_copy(path, target)
    shutil.copytree(projector_dir, staging / "projector", copy_function=shutil.copy2, dirs_exist_ok=True)
    shutil.copytree(ROOT / "common/sharm_image21", staging / "sharm_image21", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ORIGINAL / "LICENSE", staging / "LICENSE")
    (staging / "NOTICE").write_text(
        "Built with Qwen.\nQwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, "
        "Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.\n"
        "Sharm modifies the conditioning pipeline and adds a trained projector.\n"
        "Qwen3-VL-4B-Instruct-FP8: Qwen Team, Apache-2.0.\n", encoding="utf-8")
    shutil.copy2(ROOT / "licenses/Apache-2.0.txt", staging / "LICENSE.encoder")
    config = {"name": "Sharm Image 2.1 VL4B FP8", "encoder_format": "fp8-e4m3", "encoder_frozen": True,
              "student_identity": model_identity("student"), "projector_sha256": fingerprint,
              "base_validation_passed": True, "base_revision": model_identity("teacher")["revision"],
              "scope": "experimental-text-to-image", "image_editing_validated": False,
              "hf_repo_intent": "ethanzxv/Sharm-Image-2.1-VL4B-FP8", "private": True, "published": False}
    atomic_json(staging / "sharm_config.json", config)
    atomic_json(staging / "model_index.json", {"_class_name": "SharmImage21Pipeline", "_module": "sharm_image21.pipeline"})
    atomic_json(staging / "training_manifest.json", json.loads((projector_dir / "training-complete.json").read_text()))
    write_presentation(staging, validation, fingerprint)
    write_checksums(staging)
    staging.replace(destination)
    print(f"Base Sharm exportée localement, publication différée : {destination}", flush=True)


if __name__ == "__main__":
    main()
