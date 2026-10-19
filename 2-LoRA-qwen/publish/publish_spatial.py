#!/usr/bin/env python3
"""Package and publish the completed spatial candidate to its authorized HF repo."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts/spatial"
REPO = "ethanzxv/Sharm-Image-2.1-VL4B-FP8"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def contained(root, name, suffix=None):
    if not isinstance(name, str) or not name:
        raise ValueError("Chemin relatif non vide requis.")
    relative = PurePosixPath(name)
    if ("\\" in name or ":" in name or relative.is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError(f"Chemin relatif invalide : {name}")
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()) or (suffix and path.suffix != suffix):
        raise ValueError(f"Chemin externe ou format incorrect : {name}")
    return path


def payload_path_allowed(name):
    """Only model artifacts, shipped Python and auditable comparison images."""
    path = PurePosixPath(name)
    if name in {"README.md", "LICENSE", "LICENSE.encoder", "NOTICE", "requirements-inference.txt",
                "model_index.json", "sharm_config.json", "training_manifest.json", "visual-review.json"}:
        return True
    if len(path.parts) == 2:
        component = path.parts[0]
        if component in ("transformer", "vae", "scheduler", "text_encoder"):
            return path.suffix in (".json", ".safetensors")
        if component == "processor":
            return path.suffix in (".json", ".txt", ".jinja") and path.name.startswith(
                ("tokenizer", "preprocessor", "video_preprocessor", "chat_template", "vocab", "merges"))
        if component == "projector":
            return path.name in ("config.json", "model.safetensors")
        if component == "lora":
            return path.name == "pytorch_lora_weights.safetensors"
        if component == "sharm_image21":
            return path.suffix == ".py"
    return (name == "comparisons/spatial/predictions.json"
            or len(path.parts) == 4 and path.parts[:2] == ("comparisons", "spatial")
            and path.parts[2] in ("images", "previews") and path.suffix == ".png")


def link_file(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(source) != sha(destination):
            raise ValueError(f"Fichier différent déjà présent : {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        if shutil.disk_usage(destination.parent).free < source.stat().st_size+512*2**20:
            raise RuntimeError("Espace insuffisant pour copier les poids ; utiliser un volume avec hardlinks.")
        shutil.copy2(source, destination)


def prerequisites(run, dataset):
    candidate_dir = run / "flow/best"
    candidate = read(candidate_dir / "candidate.json")
    complete = read(run / "flow/training-complete.json")
    warmup = read(run / "warmup/training-complete.json")
    if read(run / "run-status.json").get("state") != "complete":
        raise ValueError("La chaîne d'entraînement et sa galerie doivent être terminées.")
    if complete["signature"] != candidate["signature"] or complete["steps_completed"] < candidate["step"]:
        raise ValueError("Le candidat ne correspond pas à l'entraînement terminé.")
    if candidate["config"]["phase"] != "flow" or not math.isfinite(candidate["numerical_validation_loss"]):
        raise ValueError("Candidat flow avec validation numérique finie requis.")
    if sha(candidate_dir / "projector/model.safetensors") != candidate["projector_sha256"]:
        raise ValueError("Projecteur modifié depuis la validation.")
    if candidate["lora_sha256"] and sha(candidate_dir / "lora/pytorch_lora_weights.safetensors") != candidate["lora_sha256"]:
        raise ValueError("LoRA modifié depuis la validation.")
    manifest = read(dataset / "manifest.json")
    if manifest["dataset_sha256"] != candidate["identity"]["dataset_sha256"]:
        raise ValueError("Dataset incompatible avec le candidat.")
    predictions = read(run / "validation/predictions.json")
    if predictions["config"]["candidate_sha256"] != sha(candidate_dir / "candidate.json"):
        raise ValueError("Galerie d'une autre révision.")
    review = read(ARTIFACTS / "visual-review.json")
    if (review.get("passed") is not True or not review.get("reviewer")
            or review.get("method") not in ("agent_visual_review", "human_visual_review")
            or review.get("prediction_signature") != predictions["signature"]
            or review.get("projector_sha256") != candidate["projector_sha256"]):
        raise ValueError("Revue visuelle de ce candidat et de sa galerie requise.")
    reviewed_ids = set(review.get("reviewed_ids", []))
    required = {row["id"] for row in predictions["rows"] if row["method"].startswith("candidate")}
    if not required or not required <= reviewed_ids:
        raise ValueError("La revue doit couvrir toutes les générations du candidat de cette galerie.")
    return candidate_dir, candidate, complete, warmup, manifest, predictions, review


def card(config, manifest, predictions):
    text = f"""---
license: other
license_name: qwen-research
license_link: LICENSE
pipeline_tag: text-to-image
base_model:
- Qwen/Qwen-Image-2.1
- Qwen/Qwen3-VL-4B-Instruct-FP8
language: [fr, en]
---

# Sharm Image 2.1 VL4B FP8 — spatial

Built with Qwen. Encodeur Qwen3-VL 4B FP8 gelé, projecteur contextuel BF16 et contrôle optionnel par boîtes `xyxy` normalisées. Le diffuseur est chargé en NF4 avec calcul BF16. Les PNG téléchargés conservent leur alpha ; le damier est un aperçu séparé.

## Entraînement et portée

Le corpus local comprend {sum(manifest['counts'].values())} compositions RGBA à {manifest['resolution']} px, à partir de 1 540 sprites et 384 espèces, palettes normales et shiny. Les variantes d'une paire d'espèces restent dans le même split. Le test porte sur de nouvelles combinaisons d'assets connus. Les sources PNG et caches d'entraînement ne sont pas distribués dans ce dépôt.

Les paramètres et résultats numériques figurent dans `training_manifest.json`. Le meilleur checkpoint numérique a été retenu sur validation, puis ses générations ont reçu la revue décrite dans `visual-review.json`. Une revue par agent est identifiée comme telle ; elle ne constitue pas le protocole humain avec intervalle de Wilson. **La parité générale avec le 8B et des positions parfaites ne sont pas établies.**

## Utilisation

Installer `requirements-inference.txt` dans un environnement CUDA compatible, avec un compilateur C pour Triton. Télécharger ce dépôt, puis ajouter son dossier au `PYTHONPATH`. Le kernel officiel `kernels-community/finegrained-fp8` version 4 doit être disponible dans le cache HF ; son téléchargement initial nécessite Internet.

```python
import torch
from sharm_image21.pipeline import SharmImage21Pipeline

model = SharmImage21Pipeline.from_pretrained("{REPO}")
model.generate_png(
    "Exactly two distinct objects. Bulbasaur to the left of Charmander. Transparent background.",
    "result.png", preview_path="result-preview.png",
    layout=[{{"label": "Bulbasaur", "box": [0.1, 0.3, 0.4, 0.7]}},
            {{"label": "Charmander", "box": [0.6, 0.3, 0.9, 0.7]}}],
    width={manifest['resolution']}, height={manifest['resolution']}, output_resolution={manifest['resolution']},
    num_inference_steps=30, true_cfg_scale=1.0, use_kv_cache=False,
    generator=torch.Generator(device="cuda").manual_seed(1000),
)
```

Les labels et boîtes suivent l'ordre des objets. Les boîtes sont un contrôle appris, sans contrainte mathématique sur les pixels générés. Pour placer des PNG existants dans des allocations imposées, utiliser `sharm_image21.spatial_data.compose`, qui conserve le ratio et retourne les boîtes peintes. L'édition à partir d'images d'entrée n'est pas validée.

## Comparaisons

Même prompt et mêmes graines pour le 8B et le candidat, avec le même diffuseur NF4. Les modes naturels et les modes avec coordonnées sont séparés dans `comparisons/spatial/predictions.json`. Les aperçus ci-dessous sont les premiers exemples sélectionnés avant leur génération.

"""
    for method in ("teacher_text", "candidate_text", "teacher_layout", "candidate_layout"):
        row = next((r for r in predictions["rows"] if r["method"] == method), None)
        if row:
            text += f"### {method}\n\n![{method}](comparisons/spatial/{row['preview']})\n\n[PNG RGBA](comparisons/spatial/{row['file_name']})\n\n"
    text += "## Licence et vérification\n\nLa licence Qwen Research et les notices sont conservées ; l'encodeur conserve ses éléments Apache-2.0. Les empreintes des fichiers sont dans `release_manifest.json`. `export-validation.json` vérifie le loader livré et une génération RGBA réelle ; sa portée est technique.\n"
    return text


def prepare(run, dataset, output):
    candidate_dir, candidate, complete, warmup, manifest, predictions, review = prerequisites(run, dataset)
    original = PROJECT / "1-local-generation/weights/qwen-image-2.1"
    student = PROJECT / "1-local-generation/weights/sharm-vl4b-fp8"
    if (output / "sharm_config.json").exists() and read(output / "sharm_config.json")["projector_sha256"] != candidate["projector_sha256"]:
        raise ValueError("Un autre candidat occupe le dossier d'export.")
    for component in ("transformer", "vae", "scheduler"):
        for source in (original / component).iterdir():
            if source.is_file() and source.suffix in (".json", ".safetensors"):
                link_file(source, output / component / source.name)
    for source in student.iterdir():
        if source.suffix == ".safetensors" or source.name in ("config.json", "generation_config.json", "model.safetensors.index.json"):
            link_file(source, output / "text_encoder" / source.name)
        elif source.is_file() and source.name.startswith(("tokenizer", "preprocessor", "video_preprocessor", "chat_template", "vocab", "merges")):
            link_file(source, output / "processor" / source.name)
    link_file(candidate_dir / "projector/model.safetensors", output / "projector/model.safetensors")
    settings = read(candidate_dir / "projector/config.json")
    keys = ("input_dim", "output_dim", "bottleneck", "context_layers", "context_heads", "layout_enabled", "convention")
    write(output / "projector/config.json", {k: settings[k] for k in keys if k in settings})
    if candidate["lora_sha256"]:
        link_file(candidate_dir / "lora/pytorch_lora_weights.safetensors", output / "lora/pytorch_lora_weights.safetensors")
    shutil.copytree(ROOT / "common/sharm_image21", output / "sharm_image21", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    config = {"name": "Sharm Image 2.1 VL4B FP8 spatial", "pipeline_format": "spatial-v1",
              "encoder_format": "fp8-e4m3", "encoder_frozen": True, "training_complete": True,
              "student_identity": candidate["identity"]["student"]["encoder"],
              "base_revision": candidate["identity"]["teacher"]["encoder"]["revision"],
              "projector_sha256": candidate["projector_sha256"], "spatial_lora_sha256": candidate["lora_sha256"],
              "base_validation_passed": False, "spatial_quality_validated": False,
              "scope": "experimental-text-to-image-with-optional-layout", "image_editing_validated": False}
    write(output / "sharm_config.json", config)
    write(output / "model_index.json", {"_class_name": "SharmImage21Pipeline", "_module": "sharm_image21.pipeline"})
    write(output / "visual-review.json", review)
    def summarize(record):
        return {k: record[k] for k in ("signature", "steps_completed", "best_numerical_validation_loss", "reason")}
    write(output / "training_manifest.json", {"warmup": summarize(warmup), "flow": summarize(complete),
          "selected_step": candidate["step"], "selected_validation_loss": candidate["numerical_validation_loss"],
          "identity": candidate["identity"], "dataset": {k: manifest[k] for k in ("dataset_sha256", "counts", "resolution", "seed")}})
    for name, source in (("LICENSE", original / "LICENSE"), ("LICENSE.encoder", ROOT / "licenses/Apache-2.0.txt"),
                         ("requirements-inference.txt", ROOT / "requirements-inference.txt")):
        shutil.copy2(source, output / name)
    (output / "NOTICE").write_text("Built with Qwen. Qwen Research license applies to Qwen-Image-2.1.\n"
                                   "Qwen3-VL-4B-Instruct-FP8: Qwen Team, Apache-2.0.\n"
                                   "Sharm adds a trained contextual projector, optional layout tokens and optional diffuser LoRA.\n", encoding="utf-8")
    gallery = output / "comparisons/spatial"
    for row in predictions["rows"]:
        for key in ("file_name", "preview"):
            link_file(contained(run / "validation", row[key], ".png"), contained(gallery, row[key], ".png"))
        if sha(gallery / row["file_name"]) != row["image_sha256"]:
            raise ValueError("Image de comparaison modifiée.")
    write(gallery / "predictions.json", predictions)
    (output / "README.md").write_text(card(config, manifest, predictions), encoding="utf-8")
    files = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()
             and payload_path_allowed(p.relative_to(output).as_posix())}
    write(output / "release_manifest.json", {"schema": 1, "repo_id": REPO, "projector_sha256": candidate["projector_sha256"],
          "dataset_sha256": manifest["dataset_sha256"], "files": files})
    print(json.dumps({"prepared": str(output), "files": len(files)}))


def verified_payload(output):
    manifest, verification = read(output / "release_manifest.json"), read(output / "export-validation.json")
    if manifest.get("repo_id") != REPO:
        raise ValueError("Le manifeste vise un autre dépôt.")
    if verification.get("passed") is not True or verification["release_manifest_sha256"] != sha(output / "release_manifest.json"):
        raise ValueError("Vérification du loader exporté requise pour ces fichiers exacts.")
    if verification.get("projector_sha256") != manifest["projector_sha256"]:
        raise ValueError("Vérification d'un autre projecteur.")
    for name in manifest["files"]:
        if not payload_path_allowed(name):
            raise ValueError(f"Fichier hors périmètre de publication : {name}")
    allowed_images = {f"validation/{method}{suffix}.png" for method in ("candidate_text", "candidate_layout")
                      for suffix in ("", "-preview")}
    if set(verification["images"]) != allowed_images:
        raise ValueError("Les deux vérifications RGBA et leurs aperçus sont requis.")
    files = {**manifest["files"], "release_manifest.json": sha(output / "release_manifest.json"),
             "export-validation.json": sha(output / "export-validation.json"), **verification["images"]}
    for name, expected in files.items():
        path = contained(output, name)
        if sha(path) != expected:
            raise ValueError(f"Export modifié ou chemin externe : {name}")
    return manifest, files


def push(output):
    from huggingface_hub import HfApi, hf_hub_download
    request = read(ARTIFACTS / "publication-request.json")
    if not request.get("authorized") or request.get("account") != "ethanzxv" or request.get("repo_id") != REPO:
        raise ValueError("Autorisation de publication manquante pour ce dépôt précis.")
    manifest, files = verified_payload(output)
    api = HfApi()
    if api.whoami()["name"] != "ethanzxv":
        raise ValueError("Compte Hugging Face incorrect.")
    info = api.model_info(REPO)
    expected_private = request["visibility_at_authorization"] == "private"
    if info.private != expected_private:
        raise ValueError("La visibilité du dépôt a changé depuis la demande ; vérifier avant publication.")
    if info.sha != request["base_hf_revision"]:
        # A retried upload can verify an already published, identical release.
        remote = read(hf_hub_download(REPO, "release_manifest.json", revision=info.sha))
        if remote != manifest:
            raise ValueError("Le dépôt a reçu d'autres modifications depuis l'autorisation.")
        revision = info.sha
    else:
        commit = api.upload_folder(repo_id=REPO, repo_type="model", folder_path=output,
                allow_patterns=list(files), parent_commit=info.sha,
                commit_message="Publish trained spatial projector, RGBA loader and paired validation gallery")
        revision = commit.oid
    # Verify the returned immutable revision, rather than trusting the upload response.
    remote = read(hf_hub_download(REPO, "release_manifest.json", revision=revision))
    if remote != manifest:
        raise ValueError("Le manifeste distant diffère de l'export local.")
    info = api.model_info(REPO, revision=revision, files_metadata=True)
    remote_files = {r.rfilename: r for r in info.siblings}
    for name, expected in files.items():
        remote_file = remote_files.get(name)
        if remote_file is None or remote_file.size != (output / name).stat().st_size:
            raise ValueError(f"Fichier distant absent ou tronqué : {name}")
        if remote_file.lfs and remote_file.lfs.sha256 != expected:
            raise ValueError(f"Empreinte distante incorrecte : {name}")
        if not remote_file.lfs and sha(hf_hub_download(REPO, name, revision=revision)) != expected:
            raise ValueError(f"Contenu distant incorrect : {name}")
    write(ARTIFACTS / "publication-status.json", {"state": "complete", "repo_id": REPO,
          "revision": revision, "private": info.private, "files": len(files), "projector_sha256": manifest["projector_sha256"]})
    print(json.dumps({"published": REPO, "revision": revision}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "push"))
    parser.add_argument("--run", type=Path, default=ARTIFACTS / "run")
    parser.add_argument("--dataset", type=Path, default=ARTIFACTS / "dataset")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "release")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.run, args.dataset, args.output)
    else:
        push(args.output)


if __name__ == "__main__":
    main()
