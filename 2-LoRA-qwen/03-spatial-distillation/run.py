#!/usr/bin/env python3
"""Run the spatial experiment in isolated folders; never publish a candidate."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "common"))
from sharm_image21.spatial_data import sha256, write_json


def preflight():
    modules = {k: importlib.util.find_spec(k) is not None for k in
               ("torch", "torchvision", "diffusers", "transformers", "bitsandbytes", "peft", "safetensors", "PIL", "numpy")}
    weights = ROOT.parent / "1-local-generation/weights"
    paths = [weights / "qwen-image-2.1/transformer/config.json", weights / "qwen-image-2.1/vae/config.json",
             weights / "qwen-image-2.1/text_encoder/model.safetensors.index.json",
             weights / "sharm-vl4b-fp8/model.safetensors.index.json"]
    gpu = None
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if modules["torch"]:
        import torch
        if torch.cuda.is_available():
            gpu = {"name": torch.cuda.get_device_name(), "total_gib": torch.cuda.get_device_properties(0).total_memory/2**30}
    return {"ready": all(modules.values()) and all(p.exists() for p in paths) and gpu is not None and compiler is not None,
            "modules": modules, "gpu": gpu, "c_compiler": compiler,
            "missing_model_paths": [str(p) for p in paths if not p.exists()]}


def flags(config):
    return [item for key, value in config.items() for item in ("--"+key, str(value))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--assets", type=Path)
    source.add_argument("--sprites", type=Path)
    source.add_argument("--pokeapi", action="store_true", help="Préparer automatiquement les sprites publics de face.")
    parser.add_argument("--profile", type=Path, default=HERE / "configs/8gb.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/spatial/dataset")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache/spatial")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/spatial/run")
    parser.add_argument("--stage", choices=("all", "data", "features", "latents", "trajectories", "warmup", "flow", "generation"), default="all")
    args = parser.parse_args()
    if args.check:
        report = preflight()
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["ready"] else 2)
    if args.stage != "data":
        report = preflight()
        if not report["ready"]:
            print(json.dumps(report, indent=2))
            raise SystemExit("Préparer le runtime avec setup_sharm.sh et les modèles avec prepare_models.py.")
    profile = json.loads(args.profile.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    # Linux/WSL is the supported GPU runtime. Data generation also works on Windows.
    lock = (args.output / "run.lock").open("a")
    if sys.platform != "win32":
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    stages = []
    if args.pokeapi:
        args.assets = args.cache / "assets-pokeapi/assets.jsonl"
        if args.stage in ("all", "data"):
            stages.append(("assets", HERE / "prepare_assets.py", ["--output", str(args.assets.parent)]))
    if args.stage in ("all", "data"):
        if args.dataset.exists():
            manifest = json.loads((args.dataset / "manifest.json").read_text())
            if any(manifest[k] != profile["dataset"][k] for k in ("resolution", "seed")) or sum(manifest["counts"].values()) != profile["dataset"]["count"]:
                raise ValueError("Le profil ne correspond pas au dataset existant ; choisir un autre dossier.")
            if (args.assets or args.sprites) and sha256(args.assets or args.sprites) != manifest["asset_manifest_sha256"]:
                raise ValueError("Le manifeste d'assets a changé.")
        else:
            if not (args.assets or args.sprites):
                raise ValueError("Fournir --assets ou --sprites pour préparer les compositions.")
            stages.append(("data", HERE / "prepare_dataset.py", ["--assets" if args.assets else "--sprites", str(args.assets or args.sprites), "--output", str(args.dataset), *flags(profile["dataset"])]))
    if args.stage in ("all", "features"):
        stages.append(("audit_fp8", ROOT / "01-encoder-distillation/audit_runtime.py", []))
        for kind, batch in (("teacher", "4"), ("student", "1")):
            stages.append((f"features-{kind}", ROOT / "01-encoder-distillation/cache_features.py",
                           [kind, "--corpus", str(args.dataset / "corpus.jsonl"), "--cache-root", str(args.cache), "--batch-size", batch]))
    if args.stage in ("all", "flow"):
        stages.append(("audit_backward", HERE / "verify_runtime.py", []))
    for kind in ("latents", "trajectories"):
        if args.stage in ("all", kind):
            stages.append((kind, HERE / "prepare_cache.py", [kind, "--dataset", str(args.dataset), "--cache", str(args.cache),
                           *(flags(profile["trajectories"]) if kind == "trajectories" else [])]))
    for phase in ("warmup", "flow"):
        if args.stage in ("all", phase):
            init = ROOT / "artifacts/projector" if phase == "warmup" else args.output / "warmup/best/projector"
            initialize = ["--init-projector", str(init)] if init.exists() or phase == "flow" else []
            stages.append((phase, HERE / "train.py", [phase, "--dataset", str(args.dataset), "--cache", str(args.cache),
                           "--output", str(args.output / phase), *initialize, *flags(profile[phase])]))
    if args.stage in ("all", "generation"):
        predictions = args.output / "validation/predictions.json"
        stages.append(("generation", HERE / "generate.py", ["--candidate", str(args.output / "flow/best"),
                       "--dataset", str(args.dataset), "--cache", str(args.cache), "--output", str(predictions.parent),
                       "--include-teacher", *flags(profile["generation"])]))
        stages.append(("review", HERE / "review.py", ["--predictions", str(predictions)]))
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    for name, script, arguments in stages:
        write_json(args.output / "run-status.json", {"stage": name, "state": "running"})
        print(f"Démarrage : {name}", flush=True)
        child = None
        try:
            child = subprocess.Popen([sys.executable, "-u", str(script), *arguments], cwd=ROOT.parent)
            code = child.wait()
            if code:
                raise subprocess.CalledProcessError(code, child.args)
        except BaseException as error:
            write_json(args.output / "run-status.json", {"stage": name, "state": "failed", "error": str(error)})
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
    write_json(args.output / "run-status.json", {"state": "complete", "stage": args.stage,
               "quality": "candidate_requires_spatial_review", "profile_sha256": sha256(args.profile)})


if __name__ == "__main__":
    main()
