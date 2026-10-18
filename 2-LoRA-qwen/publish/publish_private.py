#!/usr/bin/env python3
"""Publish validated Sharm releases privately and verify the exact HF revisions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from sharm_image21.core import ARTIFACTS, ROOT, WEIGHTS, atomic_json, file_hash
from sharm_image21.model_cards import load_comparisons


REPOS = {"base": "ethanzxv/Sharm-Image-2.1-VL4B-FP8",
         "lora": "ethanzxv/Sharm-Image-2.1-Sprites-128-LoRA"}
SOURCES = {"base": WEIGHTS / "sharm-image-2.1-vl4b-fp8",
           "lora": WEIGHTS / "sharm-image-2.1-sprites-128-lora"}


def status(state, **fields):
    path = ARTIFACTS / "publication-status.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    atomic_json(path, {**previous, "state": state, "updated_at": time.time(), **fields})
    print(json.dumps({"state": state, **fields}, ensure_ascii=False), flush=True)


def hardlink_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prerequisites():
    request = json.loads((ARTIFACTS / "publication-request.json").read_text())
    if request.get("authorized") is not True or request.get("private") is not True or request.get("account") != "ethanzxv":
        raise RuntimeError("Une demande de publication privée pour ethanzxv est requise.")
    run = json.loads((ARTIFACTS / "run-status.json").read_text())
    if run.get("state") != "complete":
        raise RuntimeError("Les deux entraînements et leurs évaluations doivent être terminés.")
    base, lora = SOURCES["base"], SOURCES["lora"]
    config = json.loads((base / "sharm_config.json").read_text())
    fingerprint = config["projector_sha256"]
    if file_hash(base / "projector/model.safetensors") != fingerprint:
        raise ValueError("Le projecteur de l’export a changé.")
    for report in ("base-validation.json", "export-validation.json"):
        value = json.loads((ARTIFACTS / report).read_text())
        if value.get("passed") is not True or value["projector_sha256"] != fingerprint:
            raise ValueError(f"Validation incompatible ou manquante : {report}")
    for path in (base / "projector/training-complete.json", lora / "training-complete.json"):
        if not path.is_file():
            raise RuntimeError(f"Entraînement non terminé : {path}")
    lora_config = json.loads((lora / "sharm_lora_config.json").read_text())
    lora_sha = file_hash(lora / "pytorch_lora_weights.safetensors")
    if lora_config["projector_sha256"] != fingerprint or lora_config["lora_sha256"] != lora_sha:
        raise ValueError("Les poids LoRA et leur déclaration sont incompatibles.")
    load_comparisons(base / "comparisons/encoder", "encoder", fingerprint)
    load_comparisons(lora / "comparisons/encoder", "encoder", fingerprint)
    load_comparisons(lora / "comparisons/lora", "lora", fingerprint, lora_sha)
    if file_hash(base / "comparisons/encoder/comparison.json") != file_hash(lora / "comparisons/encoder/comparison.json"):
        raise ValueError("Les deux fiches ne montrent pas la même référence VL.")
    review = json.loads((ARTIFACTS / "visual-review.json").read_text())
    expected = {"encoder_comparison_sha256": file_hash(base / "comparisons/encoder/comparison.json"),
                "lora_comparison_sha256": file_hash(lora / "comparisons/lora/comparison.json")}
    if review.get("passed") is not True or any(review.get(key) != value for key, value in expected.items()):
        raise ValueError("La revue visuelle doit porter sur les images exactes de cette publication.")
    return config, lora_sha, review


def release_files(kind):
    root = SOURCES[kind]
    names = ["README.md", "LICENSE", "LICENSE.encoder", "NOTICE", "footprint.json"]
    if kind == "base":
        names += ["sharm_config.json", "model_index.json", "training_manifest.json", "base-validation.json"]
        patterns = ("transformer/*.json", "transformer/*.safetensors", "vae/*.json", "vae/*.safetensors",
                    "scheduler/*.json", "text_encoder/*.json", "text_encoder/*.safetensors", "processor/*",
                    "projector/config.json", "projector/model.safetensors", "projector/training-complete.json",
                    "sharm_image21/*.py", "comparisons/encoder/*.png", "comparisons/encoder/comparison.json")
    else:
        names += ["pytorch_lora_weights.safetensors", "sharm_lora_config.json", "training-complete.json", "evaluation.json"]
        patterns = ("comparisons/encoder/*.png", "comparisons/encoder/comparison.json",
                    "comparisons/lora/*.png", "comparisons/lora/comparison.json")
    paths = {root / name for name in names}
    for pattern in patterns:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    for path in paths:
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Fichier de publication manquant ou externe : {path.name}")
        if path.suffix in (".pt", ".pth", ".log") or path.name.startswith("."):
            raise ValueError(f"Fichier privé hors export : {path.name}")
    return sorted(paths)


def build_release(kind, config, lora_sha, review, base_hf_revision=None):
    source = SOURCES[kind]
    destination = ARTIFACTS / "hf-releases" / kind
    if not destination.resolve().is_relative_to(ARTIFACTS.resolve()):
        raise ValueError("Le dossier temporaire HF doit rester dans artifacts.")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    for path in release_files(kind):
        hardlink_or_copy(path, destination / path.relative_to(source))
    # Configs are replaced atomically so the hardlinked local export stays intact.
    filename = "sharm_config.json" if kind == "base" else "sharm_lora_config.json"
    metadata = json.loads((destination / filename).read_text())
    metadata.pop("published", None)
    metadata.update(private=True, hf_repo_id=REPOS[kind])
    if kind == "lora":
        if not base_hf_revision:
            raise ValueError("La révision HF exacte de la base est requise pour publier le LoRA.")
        metadata.update(base_hf_revision=base_hf_revision, base_diffusion_revision=config["base_revision"])
    atomic_json(destination / filename, metadata)
    shutil.copy2(ROOT / "requirements-inference.txt", destination / "requirements-inference.txt")
    atomic_json(destination / "visual-review.json", review)
    atomic_json(destination / "release_manifest.json", {
        "target_repo": REPOS[kind], "private": True, "projector_sha256": config["projector_sha256"],
        "original_base_revision": config["base_revision"], "student_identity": config["student_identity"],
        "lora_sha256": lora_sha if kind == "lora" else None, "base_hf_revision": base_hf_revision,
        "scope": "experimental text-to-image, image editing not validated", "prepared_at": time.time(),
        "source_export": source.name, "source_config_sha256": file_hash(source / filename),
    })
    if kind == "lora":
        card_path = destination / "README.md"
        # Replace a hardlink before editing the release-specific card.
        card = card_path.read_text(encoding="utf-8")
        card_path.unlink()
        card_path.write_text(card + f"\nRévision HF exacte de la base : `{base_hf_revision}`.\n", encoding="utf-8")
    checksums = {path.relative_to(destination).as_posix(): file_hash(path) for path in destination.rglob("*")
                 if path.is_file() and ".cache" not in path.relative_to(destination).parts
                 and path.name != "checksums.json"}
    atomic_json(destination / "checksums.json", checksums)
    return destination


def ensure_private(api, repo):
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    info = api.model_info(repo)
    if info.private is not True:
        api.update_repo_settings(repo, private=True, repo_type="model")
    if api.model_info(repo).private is not True:
        raise RuntimeError(f"Confidentialité non confirmée pour {repo} ; envoi refusé.")


def git_blob_hash(path):
    result = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_remote(api, repo, revision, directory):
    info = api.model_info(repo, revision=revision, files_metadata=True)
    if info.private is not True or info.sha != revision:
        raise RuntimeError("La révision distante ou sa confidentialité ne correspond pas.")
    local = json.loads((directory / "checksums.json").read_text())
    local["checksums.json"] = file_hash(directory / "checksums.json")
    remote = {entry.rfilename: entry for entry in info.siblings}
    unexpected = set(remote) - set(local) - {".gitattributes"}
    if unexpected:
        raise RuntimeError(f"Fichiers inattendus dans le dépôt HF : {sorted(unexpected)}")
    for filename, checksum in local.items():
        entry = remote.get(filename)
        path = directory / filename
        if entry is None or entry.size != path.stat().st_size:
            raise RuntimeError(f"Taille distante incorrecte : {filename}")
        if entry.lfs is not None:
            if entry.lfs.sha256 != checksum:
                raise RuntimeError(f"Empreinte distante incorrecte : {filename}")
        elif entry.blob_id != git_blob_hash(path):
            raise RuntimeError(f"Contenu Git distant incorrect : {filename}")
    # Fetch metadata from the authenticated endpoint at the immutable revision.
    for filename in ("README.md", "checksums.json", "release_manifest.json"):
        fetched = Path(hf_hub_download(repo, filename, revision=revision, token=True))
        if file_hash(fetched) != local[filename]:
            raise RuntimeError(f"Relecture HF incohérente : {filename}")
    try:
        HfApi(token=False).model_info(repo)
    except HfHubHTTPError as error:
        if error.response.status_code not in (401, 403, 404):
            raise
    else:
        raise RuntimeError("Le dépôt est accessible sans authentification ; publication interrompue.")
    return {"repo_id": repo, "revision": revision, "private": True, "verified_files": len(local),
            "bytes": sum((directory / name).stat().st_size for name in local), "anonymous_access": False}


def upload(api, kind, directory):
    repo = REPOS[kind]
    ensure_private(api, repo)
    status(f"uploading_{kind}", current_repo=repo)
    commit = api.upload_folder(repo_id=repo, repo_type="model", folder_path=directory,
                               ignore_patterns=[".cache/**", "**/__pycache__/**", "*.pyc"],
                               commit_message=f"Publish validated private Sharm {kind} with paired comparisons")
    status(f"verifying_{kind}", current_repo=repo, current_revision=commit.oid)
    return verify_remote(api, repo, commit.oid, directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-inference", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    api = HfApi()
    if api.whoami().get("name") != "ethanzxv":
        raise RuntimeError("Authentification HF du compte ethanzxv requise.")
    config, lora_sha, review = prerequisites()
    status("preparing", private=True, account="ethanzxv")
    base_dir = build_release("base", config, lora_sha, review)
    base = upload(api, "base", base_dir)
    status("base_published", base=base)
    lora_dir = build_release("lora", config, lora_sha, review, base_hf_revision=base["revision"])
    lora = upload(api, "lora", lora_dir)
    status("uploads_verified", base=base, lora=lora)
    if args.verify_inference:
        status("verifying_published_inference")
        # HF checks server etags and can reuse identical local weight files.
        snapshot_download(REPOS["base"], revision=base["revision"], token=True, local_dir=base_dir)
        snapshot_download(REPOS["lora"], revision=lora["revision"], token=True, local_dir=lora_dir)
        subprocess.run([sys.executable, str(Path(__file__).with_name("verify_published.py")),
                        "--base", str(base_dir), "--lora", str(lora_dir), "--output", str(ARTIFACTS)], check=True)
    status("complete", base=base, lora=lora, inference_verified=args.verify_inference)
    path = ARTIFACTS / "run-status.json"
    run = json.loads(path.read_text())
    atomic_json(path, {**run, "publish": True, "publication": "published_private", "published_revisions": {
        "base": base["revision"], "lora": lora["revision"]}, "updated_at": time.time()})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error=str(error))
        raise
