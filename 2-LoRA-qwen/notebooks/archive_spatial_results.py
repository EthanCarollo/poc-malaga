#!/usr/bin/env python3
"""Archive reviewed spatial results for Git after verified HF publication; no weights."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

from notebook_utils import PROJECT, contained, read_json, sha256


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def select(value, keys):
    return {k: value[k] for k in keys if k in value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=PROJECT / "artifacts/spatial")
    args = parser.parse_args()
    source = args.artifacts.resolve()
    publication = read_json(source / "publication-status.json", {})
    if publication.get("state") != "complete" or not re.fullmatch(r"[a-f0-9]{40}", publication.get("revision", "")):
        raise ValueError("Publication HF vérifiée et révision immuable requises avant archivage final.")
    if publication.get("repo_id") != "ethanzxv/Sharm-Image-2.1-VL4B-FP8":
        raise ValueError("La publication ne concerne pas Sharm Image Base.")
    run = source / "run"
    if read_json(run / "run-status.json", {}).get("state") != "complete":
        raise ValueError("La chaîne de génération doit être terminée.")
    warmup = read_json(run / "warmup/training-complete.json")
    flow = read_json(run / "flow/training-complete.json")
    if not warmup or not flow:
        raise ValueError("Rapports des deux phases requis.")
    predictions_path = run / "validation/predictions.json"
    predictions = read_json(predictions_path)
    release = source / "release"
    # Reuse the publisher's integrity checks, including the shipped loader report.
    spec = importlib.util.spec_from_file_location("spatial_publisher", PROJECT / "publish/publish_spatial.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    manifest, _ = publisher.verified_payload(release)
    if publication.get("projector_sha256") != manifest["projector_sha256"]:
        raise ValueError("Publication et export ne décrivent pas le même projecteur.")
    review = read_json(release / "visual-review.json")
    candidate_ids = {r["id"] for r in predictions["rows"] if r["method"].startswith("candidate_")}
    if (review.get("passed") is not True or review.get("method") not in ("agent_visual_review", "human_visual_review")
            or not review.get("reviewer") or review.get("projector_sha256") != manifest["projector_sha256"]
            or review.get("prediction_signature") != predictions["signature"]
            or not candidate_ids <= set(review.get("reviewed_ids", []))):
        raise ValueError("Revue complète liée à cette galerie et à ce projecteur requise.")
    if predictions != read_json(release / "comparisons/spatial/predictions.json"):
        raise ValueError("La galerie locale diffère de la galerie publiée.")
    methods = Counter(r["method"] for r in predictions["rows"])
    expected = len(predictions["config"]["rows"]) * len(predictions["config"]["seeds"])
    if not candidate_ids or set(methods) != set(predictions["config"]["methods"]) or any(n != expected for n in methods.values()):
        raise ValueError("Galerie incomplète.")
    copies = {}
    for row in predictions["rows"]:
        image = contained(run / "validation", row["file_name"])
        if sha256(image) != row["image_sha256"]:
            raise ValueError("Une génération a changé depuis son manifeste.")
        copies["run/validation/" + row["file_name"]] = image
        preview = contained(run / "validation", row["preview"])
        if not preview.is_file():
            raise FileNotFoundError(preview)
        if sha256(preview) != sha256(contained(release / "comparisons/spatial", row["preview"])):
            raise ValueError("Un aperçu diffère de la galerie publiée.")
        copies["run/validation/" + row["preview"]] = preview
    verification = read_json(release / "export-validation.json")
    for name in verification["images"]:
        copies["release/" + name] = contained(release, name)
    # Only named JSON reports and PNGs enter Git. No recursive artifact copy.
    copies.update({"run/validation/predictions.json": predictions_path,
                   "visual-review.json": release / "visual-review.json",
                   "release/export-validation.json": release / "export-validation.json",
                   "release/release_manifest.json": release / "release_manifest.json"})
    for phase in ("warmup", "flow"):
        copies[f"run/{phase}/history.jsonl"] = run / phase / "history.jsonl"
    for name, path in copies.items():
        if not path.is_file() or path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError(f"Fichier absent ou trop volumineux pour le rapport : {name}")
    profile = read_json(PROJECT / "03-spatial-distillation/configs/remote16gb-256.json")
    config = flow.get("config", {})
    # Fail on a changed local recipe instead of attributing it to an older run.
    for key, expected_value in profile["flow"].items():
        if config.get(key.replace("-", "_")) != expected_value:
            raise ValueError("Le profil local ne correspond plus au run ; archiver son profil exact.")
    report_id = "spatial-" + publication["revision"][:12]
    destination = PROJECT / "reports" / report_id
    if destination.exists():
        raise FileExistsError("Le rapport est immuable ; consulter son manifeste, ne pas l'écraser.")
    staging = source / "notebook-report-staging" / report_id
    if staging.exists():
        raise FileExistsError("Un brouillon d'archivage existe déjà ; l'inspecter avant reprise.")
    staging.mkdir(parents=True)
    for name, path in copies.items():
        target = contained(staging, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    completion_keys = ("signature", "steps_completed", "best_numerical_validation_loss", "spatial_quality_validated", "reason")
    for phase, value in (("warmup", warmup), ("flow", flow)):
        write(staging / f"run/{phase}/training-complete.json", select(value, completion_keys))
    write(staging / "publication-status.json", select(publication, ("state", "repo_id", "revision", "private", "files", "projector_sha256")))
    write(staging / "profile.json", profile)
    code_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    write(staging / "summary.json", {
        "schema": 1, "model": "Sharm Image Base", "hf_repo": publication["repo_id"],
        "hf_revision": publication["revision"], "projector_sha256": manifest["projector_sha256"],
        "dataset_sha256": manifest["dataset_sha256"], "generation_counts": dict(methods),
        "visual_review_method": review["method"], "spatial_quality_validated": False,
        "scope": "Reviewed experimental candidate; no statistical human spatial validation claimed",
        "archived_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "archive_code_revision": code_revision,
        "training_identity": flow.get("identity", {}),
    })
    files = {path.relative_to(staging).as_posix(): sha256(path) for path in sorted(staging.rglob("*")) if path.is_file()}
    write(staging / "manifest.json", {"schema": 1, "files": files})
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staging, destination)
    write(destination.parent / "current.json", {"schema": 1, "report": report_id, "hf_revision": publication["revision"],
                                                "manifest_sha256": sha256(destination / "manifest.json")})
    print(json.dumps({"report": report_id, "files": len(files), "next": "Review the report, commit with Angular convention and push origin/main"}))


if __name__ == "__main__":
    main()
