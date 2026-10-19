#!/usr/bin/env python3
"""Resume the spatial experiment remotely with a staged model download budget."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts/spatial"
sys.path.insert(0, str(ROOT / "common"))
from sharm_image21.core import (FeatureCache, ORIGINAL, atomic_json, digest, file_hash,
                                model_identity, read_jsonl)


def status(stage, state="running", **extra):
    atomic_json(ARTIFACTS / "remote-status.json", {"stage": stage, "state": state, "pid": os.getpid(),
                "updated_at": time.time(), "free_disk_gib": shutil.disk_usage(PROJECT).free/2**30, **extra})


def execute(stage, script, *arguments, offline=False):
    status(stage)
    environment = dict(os.environ)
    if offline:
        environment["HF_HUB_OFFLINE"] = "1"
    else:
        environment.pop("HF_HUB_OFFLINE", None)
    subprocess.run([sys.executable, "-u", str(script), *map(str, arguments)], cwd=PROJECT,
                   env=environment, check=True)


def release_teacher_weights():
    """Delete only this new workspace's temporary teacher shards after a full cache."""
    marker = json.loads((ARTIFACTS / "remote-workspace.json").read_text(encoding="utf-8"))
    if marker.get("root") != str(PROJECT.resolve()) or marker.get("temporary_teacher_weights") is not True:
        raise ValueError("Le nettoyage du professeur exige le marqueur de ce workspace isolé.")
    corpus = read_jsonl(ARTIFACTS / "dataset/corpus.jsonl")
    teacher = FeatureCache(ROOT / "cache/spatial/teacher",
                           {"encoder": model_identity("teacher"), "corpus": digest(corpus)})
    teacher.require(corpus)
    # Validate every indexed shard before deleting any regenerable source weights.
    from safetensors import safe_open
    for name in {r["shard"] for r in teacher.index["entries"].values()}:
        path = teacher.directory / name
        if path.parent.resolve() != teacher.directory.resolve() or path.is_symlink() or not path.is_file():
            raise ValueError("Shard du professeur manquant ou externe.")
        with safe_open(path, framework="pt", device="cpu") as values:
            for key, entry in teacher.index["entries"].items():
                if entry["shard"] == name and list(values.get_slice(key).get_shape()) != entry["shape"]:
                    raise ValueError("Dimensions du cache du professeur incorrectes.")
    directory = ORIGINAL / "text_encoder"
    if directory.is_symlink() or not directory.resolve().is_relative_to(PROJECT.resolve()):
        raise ValueError("Le professeur doit appartenir au workspace isolé.")
    directory = directory.resolve()
    index = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))
    targets = []
    for name in set(index["weight_map"].values()):
        path = directory / name
        if Path(name).name != name or not name.endswith(".safetensors") or path.is_symlink():
            raise ValueError("Chemin de poids temporaire inattendu.")
        targets.append(path)
    report = {"teacher_index_sha256": file_hash(teacher.path), "captions": len(corpus),
              "released_bytes": sum(p.stat().st_size for p in targets if p.exists()),
              "files": [p.name for p in targets], "scope": "Owned temporary teacher shards; configs retained"}
    for path in targets:
        path.unlink(missing_ok=True)
    report_path = ARTIFACTS / "temporary-teacher-release.json"
    if report["released_bytes"] or not report_path.exists():
        atomic_json(report_path, report)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    lock = (ARTIFACTS / "remote.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    stage = "initializing"
    try:
        corpus = read_jsonl(ARTIFACTS / "dataset/corpus.jsonl")
        directory = ROOT / "cache/spatial/teacher"
        cache = FeatureCache(directory) if (directory / "index.json").exists() else None
        teacher_complete = cache is not None and all(r["id"] in cache for r in corpus)
        if not teacher_complete:
            stage = "download_teacher"
            execute(stage, ROOT / "01-encoder-distillation/prepare_models.py", "--stage", "teacher")
            stage = "features-teacher"
            execute(stage, ROOT / "01-encoder-distillation/cache_features.py", "teacher", "--corpus",
                    ARTIFACTS / "dataset/corpus.jsonl", "--cache-root", ROOT / "cache/spatial", "--batch-size", 4,
                    offline=True)
        stage = "release_temporary_teacher"
        status(stage)
        release_teacher_weights()
        stage = "download_inference"
        execute(stage, ROOT / "01-encoder-distillation/prepare_models.py", "--stage", "inference")
        stage = "training_chain"
        execute(stage, ROOT / "03-spatial-distillation/run.py", "--profile",
                ROOT / "03-spatial-distillation/configs/remote16gb-256.json", offline=True)
        status("finished", "complete", quality="candidate_requires_spatial_review")
    except BaseException as error:
        status(stage, "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
