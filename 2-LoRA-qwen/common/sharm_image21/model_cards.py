"""Package real paired generations and reproducible Hugging Face model cards."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import time
from urllib.parse import quote

from .image_io import checkerboard_preview


COMPARISON_SCHEMA = 1
LABELS = {
    "encoder": ("VL 8B d’origine — BF16", "Sharm VL 4B gelé — FP8 + projecteur BF16"),
    "lora": ("Base Sharm sans LoRA", "Même base Sharm avec LoRA sprites"),
}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def generation_kwargs(steps: int) -> dict:
    """One parameter set is passed unchanged to both members of every pair."""
    if steps < 1:
        raise ValueError("Le nombre de pas de comparaison doit être positif.")
    return dict(width=512, height=512, output_resolution=512, num_inference_steps=steps,
                true_cfg_scale=1.0, negative_prompt=None, num_images_per_prompt=1,
                use_kv_cache=True, output_type="pil")


def comparison_manifest(kind: str, pipeline, *, steps: int, projector_sha256: str,
                        controls: dict, selection: dict, script: Path, project: Path) -> dict:
    import diffusers
    import torch

    if kind not in LABELS:
        raise ValueError(kind)
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "diffusers", "transformers", "bitsandbytes", "safetensors", "Pillow", "kernels")}
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
    diffusers_root = Path(diffusers.__file__).resolve().parents[2]
    diffusers_commit = None
    if (diffusers_root / ".git").exists():
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=diffusers_root,
                                  text=True, capture_output=True, check=False)
        if revision.returncode == 0:
            diffusers_commit = revision.stdout.strip()
    scheduler = json.loads(json.dumps(dict(pipeline.scheduler.config), default=str))
    return {
        "schema_version": COMPARISON_SCHEMA, "kind": kind,
        "projector_sha256": projector_sha256,
        "settings": {**generation_kwargs(steps), "scheduler_class": type(pipeline.scheduler).__name__,
                     "scheduler_config": scheduler, "generator_device": "cuda",
                     "initial_noise": "fresh generator reset to the pair seed for each member",
                     "diffusion_precision": "NF4, double quantization, BF16 compute",
                     "vae_precision": "BF16", "postprocess": "none" if kind == "encoder" else "512 to 128, nearest-neighbor"},
        "controls": controls, "selection": selection,
        "provenance": {"created_at": time.time(), "source_commit": source_commit,
                       "script_sha256": sha256(script), "model_cards_sha256": sha256(Path(__file__)),
                       "versions": versions, "cuda_version": torch.version.cuda,
                       "diffusers_source_commit": diffusers_commit,
                       "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability())},
        "pairs": [],
    }


def save_member(image, directory: Path, stem: str, label: str, final_size: int | None = None) -> dict:
    """Keep alpha in downloads; composite a separate checkerboard for the card."""
    from PIL import Image

    if image.mode != "RGBA" or image.size != (512, 512):
        raise ValueError("La comparaison attend une génération RGBA 512 × 512.")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", stem):
        raise ValueError("Nom de fichier de comparaison invalide.")
    directory.mkdir(parents=True, exist_ok=True)

    def save_asset(value, filename):
        path = directory / filename
        value.save(path)
        return {"file": filename, "sha256": sha256(path), "width": value.width,
                "height": value.height, "mode": value.mode}

    native = save_asset(image, f"{stem}-512.png")
    if final_size is not None:
        if final_size != 128:
            raise ValueError("Seule la réduction documentée 512 vers 128 est prise en charge.")
        image = image.resize((final_size, final_size), Image.Resampling.NEAREST)
        final = save_asset(image, f"{stem}-128.png")
    else:
        final = native
    preview = checkerboard_preview(image, tile_size=8)
    return {"label": label, "image": final, "source": native,
            "preview": save_asset(preview, f"{stem}-preview.png")}


def _asset_path(directory: Path, filename: str) -> Path:
    relative = PurePosixPath(filename)
    if not filename or relative.is_absolute() or ".." in relative.parts or "\\" in filename:
        raise ValueError("Chemin d’image de comparaison invalide.")
    path = directory.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(directory.resolve()) or path.suffix != ".png":
        raise ValueError("Image située hors du dossier de comparaison.")
    return path


def load_comparisons(directory: Path, kind: str, projector_sha256: str,
                     lora_sha256: str | None = None) -> dict:
    """Refuse incomplete, stale or unbound image galleries during export."""
    from PIL import Image

    manifest = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    if (manifest["schema_version"] != COMPARISON_SCHEMA or manifest["kind"] != kind
            or manifest["projector_sha256"] != projector_sha256):
        raise ValueError("Galerie incompatible avec cette base Sharm.")
    if lora_sha256 is not None and manifest["controls"].get("lora_sha256") != lora_sha256:
        raise ValueError("Galerie incompatible avec ces poids LoRA.")
    settings = manifest["settings"]
    if any(settings.get(key) != value for key, value in generation_kwargs(settings["num_inference_steps"]).items()):
        raise ValueError("Paramètres de génération de comparaison incompatibles.")
    if not manifest["pairs"]:
        raise ValueError("Une galerie vide ne peut pas être publiée.")
    seen = set()
    for pair in manifest["pairs"]:
        if pair["id"] in seen or not pair["prompt"].strip() or not isinstance(pair["seed"], int):
            raise ValueError("Paire de comparaison vide ou dupliquée.")
        seen.add(pair["id"])
        for side, label in zip(("left", "right"), LABELS[kind]):
            member = pair[side]
            if member["label"] != label:
                raise ValueError("Les côtés de la comparaison sont ambigus.")
            for key in ("image", "source", "preview"):
                asset = member[key]
                path = _asset_path(directory, asset["file"])
                if sha256(path) != asset["sha256"]:
                    raise ValueError(f"Image de comparaison modifiée : {path.name}")
                expected_size = (128, 128) if kind == "lora" and key != "source" else (512, 512)
                with Image.open(path) as image:
                    if (image.size != expected_size or image.size != (asset["width"], asset["height"])
                            or image.mode != asset["mode"] or image.mode != ("RGB" if key == "preview" else "RGBA")):
                        raise ValueError("Format ou taille d’image de comparaison incorrects.")
    return manifest


def package_comparisons(source: Path, destination: Path, kind: str, projector_sha256: str,
                        lora_sha256: str | None = None) -> dict:
    manifest = load_comparisons(source, kind, projector_sha256, lora_sha256)
    destination.mkdir(parents=True, exist_ok=True)
    filenames = {asset["file"] for pair in manifest["pairs"] for side in ("left", "right")
                 for asset in (pair[side]["image"], pair[side]["source"], pair[side]["preview"])}
    for filename in sorted(filenames):
        target = _asset_path(destination, filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_asset_path(source, filename), target)
    shutil.copy2(source / "comparison.json", destination / "comparison.json")
    return load_comparisons(destination, kind, projector_sha256, lora_sha256)


def measure_footprint(original: Path, student: Path, projector: Path) -> dict:
    def tensor_bytes(directory):
        paths = list(directory.glob("*.safetensors"))
        if not paths:
            raise ValueError(f"Aucun poids safetensors dans {directory}")
        return sum(path.stat().st_size for path in paths)

    teacher_bytes = tensor_bytes(original / "text_encoder")
    student_bytes = tensor_bytes(student)
    projector_bytes = (projector / "model.safetensors").stat().st_size
    shared_bytes = tensor_bytes(original / "transformer") + tensor_bytes(original / "vae")
    return {"measurement": "sum of checkpoint safetensors file sizes, headers included; excludes configs, caches and optimizer states",
            "original_encoder_bytes": teacher_bytes, "student_encoder_bytes": student_bytes,
            "projector_bytes": projector_bytes, "shared_diffusion_vae_bytes": shared_bytes,
            "original_total_bytes": teacher_bytes + shared_bytes,
            "sharm_total_bytes": student_bytes + projector_bytes + shared_bytes,
            "encoder_reduction_percent": 100 * (1 - (student_bytes + projector_bytes) / teacher_bytes),
            "total_reduction_percent": 100 * (1 - (student_bytes + projector_bytes + shared_bytes) / (teacher_bytes + shared_bytes))}


def _prompt_block(prompt: str) -> str:
    fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", prompt)), default=0))
    return f"{fence}text\n{prompt}\n{fence}"


def _hf_file_url(repo_id: str, prefix: str, filename: str, *, download: bool = False) -> str:
    path = "/".join(quote(part, safe="-_.") for part in (*prefix.split("/"), *filename.split("/")))
    url = f"https://huggingface.co/{quote(repo_id, safe='/-_.')}/resolve/main/{path}"
    return f"{url}?download=true" if download else url


def gallery_markdown(manifest: dict, prefix: str, repo_id: str) -> str:
    lines = []
    left, right = LABELS[manifest["kind"]]
    for index, pair in enumerate(manifest["pairs"], 1):
        lines.extend((f"### Exemple {index} — graine `{pair['seed']}`", "", _prompt_block(pair["prompt"]), "",
                      f"| {left} | {right} |", "| --- | --- |"))
        cells, downloads = [], []
        for side, label in zip(("left", "right"), (left, right)):
            member = pair[side]
            preview_url = _hf_file_url(repo_id, prefix, member["preview"]["file"])
            cells.append(f'<img src="{preview_url}" alt="{label}" width="256" />')
            final, native = member["image"], member["source"]
            final_url = _hf_file_url(repo_id, prefix, final["file"], download=True)
            links = f"[Télécharger le PNG RGBA {final['width']} px]({final_url})"
            if native["file"] != final["file"]:
                native_url = _hf_file_url(repo_id, prefix, native["file"], download=True)
                links += f" · [source PNG 512 px]({native_url})"
            downloads.append(links)
        lines.extend(("| " + " | ".join(cells) + " |", "| " + " | ".join(downloads) + " |", ""))
    settings = manifest["settings"]
    lines.extend((f"Réglages communs : {settings['width']} × {settings['height']}, "
                  f"{settings['num_inference_steps']} pas, `{settings['scheduler_class']}`, "
                  f"CFG {settings['true_cfg_scale']:g}, aucun prompt négatif. "
                  "Le générateur CUDA est réinitialisé à la même graine pour chaque côté.", "",
                  f"[Prompts, graines, réglages complets, versions et empreintes des fichiers]({prefix}/comparison.json).", ""))
    return "\n".join(lines) + "\n"


def compact_objective(footprint: dict) -> str:
    def gib(value):
        return f"{value / 2**30:.2f} Gio"

    return (
        "## Objectif : compacter le modèle de base\n\n"
        "Sharm remplace l’encodeur VL d’environ 8B fourni avec Qwen-Image-2.1 par "
        "Qwen3-VL-4B-Instruct-FP8 gelé et un petit projecteur MLP entraîné, de largeur 2560 → 4096. "
        "Le but est de réduire la taille des poids du modèle de base tout en mesurant la qualité conservée. "
        "Le projecteur est exporté en BF16. Le checkpoint 4B utilise FP8 E4M3 pour ses couches quantifiées ; "
        "ses exceptions déclarées, dont la branche vision, conservent leur précision BF16.\n\n"
        "| Poids sur disque | Base d’origine | Base Sharm |\n"
        "| --- | --- | --- |\n"
        f"| Encodeur et éventuel projecteur | {gib(footprint['original_encoder_bytes'])} | "
        f"{gib(footprint['student_encoder_bytes'] + footprint['projector_bytes'])} |\n"
        f"| Diffuseur + VAE, communs aux deux bases | {gib(footprint['shared_diffusion_vae_bytes'])} | "
        f"{gib(footprint['shared_diffusion_vae_bytes'])} |\n"
        f"| Total des poids safetensors | {gib(footprint['original_total_bytes'])} | "
        f"{gib(footprint['sharm_total_bytes'])} |\n\n"
        f"Réduction mesurée : **{footprint['encoder_reduction_percent']:.1f} %** pour l’encodeur avec son "
        f"projecteur, **{footprint['total_reduction_percent']:.1f} %** sur les poids de la base complète. "
        "Ces mesures additionnent les tailles des fichiers safetensors ; configurations, caches, optimiseur "
        "et LoRA sont exclus. [Mesures en octets](footprint.json). La VRAM maximale et la vitesse "
        "d’inférence demandent des mesures séparées.\n\n"
        "Les poids du diffuseur et du VAE exportés sont ceux de la base d’origine. Pour les galeries, "
        "le même diffuseur est chargé en NF4 avec calcul BF16 des deux côtés, et le VAE en BF16. "
        "La comparaison VL mesure donc conjointement le remplacement 8B → 4B, FP8 et le projecteur, "
        "dans ce runtime commun.\n\n"
    )


def loading_example(with_lora: bool = False) -> str:
    setup = (
        "## Charger le modèle\n\n"
        "Le runtime de référence est Linux/WSL, CUDA 13.0 et une RTX 5060 Ti de 16 Gio. "
        "Les versions et la révision Diffusers sont figées dans "
        "[requirements-inference.txt](requirements-inference.txt). Installer ces dépendances dans "
        "un environnement Python séparé, puis authentifier localement un compte ayant accès "
        "aux dépôts privés avec `hf auth login`.\n\n"
        "```python\nimport json\nimport sys\nfrom pathlib import Path\nimport torch\n"
        "from PIL import Image\nfrom huggingface_hub import snapshot_download\n\n"
    )
    if with_lora:
        setup += (
            'lora = Path(snapshot_download("ethanzxv/Sharm-Image-2.1-Sprites-128-LoRA", token=True))\n'
            'metadata = json.loads((lora / "sharm_lora_config.json").read_text())\n'
            'base = Path(snapshot_download(metadata["base_model"],\n'
            '                              revision=metadata.get("base_hf_revision"), token=True))\n'
        )
    else:
        setup += 'base = Path(snapshot_download("ethanzxv/Sharm-Image-2.1-VL4B-FP8", token=True))\n'
    setup += (
        'sys.path.insert(0, str(base))\n'
        'from sharm_image21.pipeline import SharmImage21Pipeline\n\n'
        'pipeline = SharmImage21Pipeline.from_pretrained(base)\n'
    )
    if with_lora:
        setup += (
            'case = json.loads((lora / "comparisons/lora/comparison.json").read_text())["pairs"][0]\n'
            'image = pipeline.generate_png(case["prompt"], "sharm-sprite-512.png",\n'
            '    preview_path="sharm-sprite-512-preview.png", transparent_background=True,\n'
            '    lora_dir=lora, width=512, height=512, output_resolution=512,\n'
            '    num_inference_steps=30, true_cfg_scale=1.0,\n'
            '    attention_kwargs={"scale": metadata.get("recommended_weight", 1.0)},\n'
            '    generator=torch.Generator("cuda").manual_seed(case["seed"]))\n'
            'pipeline.save_png(image.resize((128, 128), Image.Resampling.NEAREST),\n'
            '    "sharm-sprite-128.png", preview_path="sharm-sprite-128-preview.png",\n'
            '    require_transparency=True)\n'
        )
    else:
        setup += (
            'case = json.loads((base / "comparisons/encoder/comparison.json").read_text())["pairs"][0]\n'
            'pipeline.generate_png(case["prompt"], "sharm-image.png",\n'
            '    preview_path="sharm-image-preview.png", transparent_background=True,\n'
            '    width=512, height=512, output_resolution=512, num_inference_steps=20,\n'
            '    true_cfg_scale=1.0,\n'
            '    generator=torch.Generator("cuda").manual_seed(case["seed"]))\n'
        )
    return setup + "```\n\n"


def base_card(manifest: dict, footprint: dict, validation: dict) -> str:
    metrics = validation["embedding_metrics"]
    return (
        "---\nlicense: other\nlicense_name: qwen-research\nlicense_link: LICENSE\n"
        "pipeline_tag: text-to-image\nbase_model:\n- Qwen/Qwen-Image-2.1\n"
        "- Qwen/Qwen3-VL-4B-Instruct-FP8\nlanguage:\n- fr\n- en\n---\n\n"
        "# Sharm Image 2.1 VL4B FP8\n\nBuilt with Qwen. Checkpoint de recherche destiné au dépôt privé "
        "`ethanzxv/Sharm-Image-2.1-VL4B-FP8`.\n\n"
        + compact_objective(footprint)
        + "## Comparaison du VL 8B et du VL 4B FP8 + projecteur\n\n"
        "Même prompt, même graine et mêmes réglages pour chaque paire, sans LoRA. "
        "Les prompts sont issus de la validation du projecteur. Les exemples sont sélectionnés avant "
        "génération, sans sélection selon leur apparence. Les aperçus compositent l’alpha sur un damier ; "
        "les PNG téléchargeables conservent le canal alpha.\n\n"
        + gallery_markdown(manifest, "comparisons/encoder", "ethanzxv/Sharm-Image-2.1-VL4B-FP8")
        + "## Validation et utilisation\n\n"
        f"Validation des embeddings : erreur quadratique relative `{metrics['relative_mse']:.4f}`, "
        f"cosinus `{metrics['cosine']:.4f}`, ratio de normes `{metrics['norm_ratio']:.4f}`. "
        f"Erreur relative moyenne des prédictions du diffuseur : `{validation['denoiser_relative_mse']:.4f}`. "
        "[Rapport complet](base-validation.json). Ces contrôles numériques et les quelques exemples "
        "visuels demandent une revue humaine pour juger le suivi des consignes et la qualité.\n\n"
        "Utiliser le loader fourni `sharm_image21.pipeline.SharmImage21Pipeline.from_pretrained`. "
        "Il encode le texte avec le 4B FP8 et le projecteur, puis libère l’encodeur avant de charger "
        "le diffuseur en NF4. L’encodeur 8B sert à la distillation et aux références en cache. "
        "Cette version est expérimentale en texte vers image ; l’édition par image reste à valider.\n\n"
        + loading_example()
        + "## Sources et licence\n\n"
        "Base : [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1). "
        "Encodeur : [Qwen3-VL-4B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-FP8). "
        "Le corpus d’alignement comprend des prompts "
        "[DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb) sous CC0-1.0, "
        "des compositions rédigées par templates et des légendes sprites. "
        f"[Code d’entraînement et d’évaluation](https://github.com/EthanCarollo/poc-malaga/tree/"
        f"{manifest['provenance']['source_commit']}/2-LoRA-qwen). "
        "Les révisions sources et le projecteur sont identifiés dans `sharm_config.json` "
        "et `training_manifest.json`. Qwen Research s’applique au modèle combiné ; "
        "les notices et la licence Apache-2.0 de l’encodeur sont conservées.\n"
    )


def lora_card(lora: dict, encoder: dict, footprint: dict) -> str:
    training = lora["controls"]["training"]
    return (
        "---\nlicense: other\nlicense_name: qwen-research\nlicense_link: LICENSE\n"
        "pipeline_tag: text-to-image\nbase_model: ethanzxv/Sharm-Image-2.1-VL4B-FP8\nbase_model_relation: adapter\n"
        "tags:\n- lora\n- pixel-art\nlanguage:\n- fr\n- en\n---\n\n"
        "# Sharm Image 2.1 Sprites 128 LoRA\n\nBuilt with Qwen. Checkpoint de recherche destiné au dépôt privé "
        "`ethanzxv/Sharm-Image-2.1-Sprites-128-LoRA`.\n\n"
        "La base Sharm vise à compacter Qwen-Image-2.1 grâce à un VL 4B FP8 gelé et son projecteur. "
        "Ce LoRA adapte le diffuseur de la base Sharm aux sprites de monstres. "
        "Le 4B FP8 et le projecteur restent gelés pendant cet entraînement. "
        "Les latents d’apprentissage sont en 512 × 512 RGBA ; le livrable sprite est un PNG "
        "128 × 128 après réduction nearest-neighbor.\n\n"
        f"Entraînement : {training['steps']} pas d’optimiseur, rang/alpha {training['rank']}, "
        f"batch 1 et accumulation {training['accumulation']}, taux d’apprentissage `{training['learning_rate']:g}`. "
        f"{lora['controls']['training_image_count']} images sont utilisées pour l’apprentissage du LoRA "
        f"et {lora['controls']['validation_image_count']} pour sa validation.\n\n"
        "## Avant / après LoRA — même prompt\n\n"
        "Chaque paire utilise exactement le même prompt, la même graine, les mêmes embeddings "
        "du VL 4B FP8 + projecteur et les mêmes réglages du diffuseur. Seul l’adaptateur LoRA "
        f"change, avec un poids de {lora['controls']['lora_weight']:g}. Les espèces montrées sont réservées à la validation du LoRA ; "
        "leurs légendes peuvent être présentes dans le corpus d’alignement du projecteur. "
        "Les six premières espèces distinctes de validation sont retenues avant génération.\n\n"
        "Les deux côtés subissent la même réduction 512 → 128 en nearest-neighbor, sans détourage "
        "ni retouche après génération. Le damier des aperçus permet de voir l’alpha ; les liens "
        "fournissent les PNG RGBA 128 px et leurs sources 512 px.\n\n"
        + gallery_markdown(lora, "comparisons/lora", "ethanzxv/Sharm-Image-2.1-Sprites-128-LoRA")
        + compact_objective(footprint)
        + "## VL 8B d’origine / VL 4B FP8 + projecteur — même prompt\n\n"
        "Cette seconde comparaison concerne la compaction de la base avant le LoRA sprites. "
        "Le LoRA est désactivé des deux côtés. Le même diffuseur NF4, le même VAE BF16, "
        "le prompt, la graine et les réglages sont conservés ; le conditionnement VL change. "
        "Ces images sont les références de validation de la base Sharm, également présentes "
        "sur sa fiche.\n\n"
        + gallery_markdown(encoder, "comparisons/encoder", "ethanzxv/Sharm-Image-2.1-Sprites-128-LoRA")
        + "## Compatibilité et portée\n\n"
        "Charger ce LoRA avec la base `ethanzxv/Sharm-Image-2.1-VL4B-FP8` et son loader fourni. "
        "`sharm_lora_config.json` lie les poids LoRA à l’empreinte du projecteur exact. "
        "Les versions du runtime, révisions du code, réglages et empreintes des images sont "
        "enregistrés dans les manifestes de comparaison. La reproductibilité dépend aussi du matériel "
        "et des kernels utilisés. La revue visuelle de l’identité, du pixel art et de la transparence "
        "reste nécessaire ; ces exemples ne constituent pas une garantie de qualité générale.\n\n"
        + loading_example(with_lora=True)
        + "Le dataset local comprend 770 sprites RGBA issus de 385 espèces. Le fond exact "
        "`#A5EBFF` a été retiré du dataset, y compris les îlots déconnectés. Les deux variantes "
        "d’une espèce restent dans le même ensemble. Qwen Research et les attributions de la base "
        "sont conservées dans `LICENSE`, `LICENSE.encoder` et `NOTICE`.\n"
        f"[Code d’entraînement et d’évaluation](https://github.com/EthanCarollo/poc-malaga/tree/"
        f"{lora['provenance']['source_commit']}/2-LoRA-qwen).\n"
    )
