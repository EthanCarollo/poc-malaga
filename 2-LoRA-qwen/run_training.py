#!/usr/bin/env python3
"""Run the two private Sharm training stages sequentially, with resumable caches."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "common"))
from sharm_image21.core import ARTIFACTS, PROJECT, atomic_json, model_identity


def status():
    for name in ("run-status.json", "cache-teacher-status.json", "cache-student-status.json",
                 "projector-status.json", "lora-status.json", "publication-request.json", "publication-status.json"):
        path = ARTIFACTS / name
        if path.exists():
            print(name, path.read_text(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--lora-steps", type=int, default=500)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    args = parser.parse_args()
    if min(args.lora_steps, args.teacher_batch_size) < 1:
        raise ValueError("Le nombre de pas LoRA et la taille des lots doivent être positifs.")
    os.umask(0o077)
    ARTIFACTS.mkdir(mode=0o700, parents=True, exist_ok=True)
    logs = ARTIFACTS / "logs"
    logs.mkdir(mode=0o700, exist_ok=True)
    if args.status:
        status()
        return
    if args.detach:
        old = ARTIFACTS / "run-status.json"
        if old.exists():
            previous = json.loads(old.read_text())
            try:
                os.kill(previous["pid"], 0)
            except (ProcessLookupError, KeyError):
                pass
            else:
                if previous.get("state") == "running":
                    print(f"Chaîne déjà active, PID {previous['pid']}", flush=True)
                    return
        with (logs / "runner.log").open("a") as output:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                      "--lora-steps", str(args.lora_steps),
                                      "--teacher-batch-size", str(args.teacher_batch_size)], cwd=PROJECT,
                                     stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(f"Chaîne Sharm démarrée, PID {child.pid}. Journaux : {logs}", flush=True)
        return
    lock = (ARTIFACTS / "run.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Une chaîne Sharm est déjà active.")
    steps = [
        ("prepare_models", "01-encoder-distillation/prepare_models.py", []),
        ("prepare_sprites", "02-sprite-lora/prepare_dataset.py", ["--training-size", "512"]),
        ("prepare_corpus", "01-encoder-distillation/prepare_corpus.py", []),
        ("audit_fp8", "01-encoder-distillation/audit_runtime.py", []),
        ("cache_teacher", "01-encoder-distillation/cache_features.py", ["teacher", "--batch-size", str(args.teacher_batch_size)]),
        ("cache_student", "01-encoder-distillation/cache_features.py", ["student", "--batch-size", "4"]),
        ("train_projector", "01-encoder-distillation/train_projector.py", []),
        ("validate_base", "01-encoder-distillation/validate_base.py", []),
        ("export_base", "publish/export_base.py", []),
        ("verify_export", "publish/verify_export.py", []),
        ("cache_sprites", "02-sprite-lora/prepare_training_cache.py", []),
        ("train_lora", "02-sprite-lora/train_lora.py", ["--steps", str(args.lora_steps)]),
        ("validate_sprites", "02-sprite-lora/evaluate_sprites.py", []),
    ]
    versions = {}
    for package in ("torch", "transformers", "diffusers", "accelerate", "bitsandbytes", "kernels", "safetensors", "pyarrow"):
        versions[package] = importlib.metadata.version(package)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    request_path = ARTIFACTS / "publication-request.json"
    request = json.loads(request_path.read_text()) if request_path.exists() else {}
    publish_requested = request.get("authorized") is True and request.get("private") is True and request.get("account") == "ethanzxv"
    state = {"pid": os.getpid(), "started_at": time.time(), "state": "running", "stage": "initializing",
             "private": True, "publish": publish_requested,
             "publication": "awaiting_training_validation" if publish_requested else "deferred_by_user",
             "cache_batch_sizes": {"teacher": args.teacher_batch_size, "student": 1},
             "source_commit": source_commit, "versions": versions, "stages": [name for name, _, _ in steps]}
    child = None

    def interrupted(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        state.update(state="interrupted", updated_at=time.time())
        atomic_json(ARTIFACTS / "run-status.json", state)
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for name, script, arguments in steps:
            state.update(stage=name, updated_at=time.time())
            atomic_json(ARTIFACTS / "run-status.json", state)
            if name == "audit_fp8" and (ARTIFACTS / "runtime-audit.json").exists():
                audit = json.loads((ARTIFACTS / "runtime-audit.json").read_text())
                if audit.get("passed") and audit["encoder_identity"] == model_identity("student"):
                    print("Audit FP8 compatible déjà effectué.", flush=True)
                    continue
            print(f"Démarrage de {name}", flush=True)
            with (logs / f"{name}.log").open("a") as output:
                child = subprocess.Popen([sys.executable, "-u", str(ROOT / script), *arguments],
                                         cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
                state["child_pid"] = child.pid
                atomic_json(ARTIFACTS / "run-status.json", state)
                returncode = child.wait()
            if returncode != 0:
                raise RuntimeError(f"{name} a échoué (code {returncode}). Journal : {logs / (name+'.log')}")
        state.update(state="complete", stage="finished", updated_at=time.time(),
                     publication="awaiting_review_and_upload" if publish_requested else "deferred_by_user")
        atomic_json(ARTIFACTS / "run-status.json", state)
        print("Deux entraînements terminés localement. " +
              ("Publication HF privée autorisée après revue." if publish_requested else "Publication HF privée différée."), flush=True)
    except Exception as error:
        state.update(state="failed", error=str(error), updated_at=time.time())
        atomic_json(ARTIFACTS / "run-status.json", state)
        raise


if __name__ == "__main__":
    main()
