"""Small, CPU-only readers and explicit launchers shared by the Sharm notebooks."""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def find_project(start=None):
    start = Path(start or Path.cwd()).resolve()
    for parent in (start, *start.parents):
        for candidate in (parent, parent / "2-LoRA-qwen"):
            if (candidate / "03-spatial-distillation/run.py").is_file():
                return candidate
    raise FileNotFoundError("Ouvrir le notebook depuis le dépôt poc ou 2-LoRA-qwen.")


PROJECT = find_project(Path(__file__).parent)
REPO = PROJECT.parent
ARTIFACTS = PROJECT / "artifacts"


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Le fichier doit rester dans le dossier demandé.")
    return path


def current_dataset():
    root = PROJECT / "02-sprite-lora/dataset"
    pointer = read_json(root / "current.json")
    if not pointer:
        raise FileNotFoundError("Préparer et sélectionner une version du dataset Monster.")
    return contained(root, pointer["dataset"]), pointer


def archived_report():
    root = PROJECT / "reports"
    pointer = read_json(root / "current.json")
    if not pointer:
        return None
    directory = contained(root, pointer["report"])
    if sha256(directory / "manifest.json") != pointer["manifest_sha256"]:
        raise ValueError("Le manifeste du rapport versionné a changé.")
    return directory


def result_file(name):
    """Prefer a versioned final report; otherwise use actual local artifacts."""
    archive = archived_report()
    if archive and (archive / name).is_file():
        path = contained(archive, name)
        expected = read_json(archive / "manifest.json")["files"].get(name)
        if not expected or sha256(path) != expected:
            raise ValueError("Un fichier du rapport versionné a changé.")
        return path
    return ARTIFACTS / "spatial" / name


def show_json(path):
    data = read_json(path)
    if data is None:
        print(f"Rapport indisponible : {Path(path).name}. Aucun résultat n'est supposé.")
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    return data


def table(rows, columns):
    from IPython.display import HTML, display
    rows = list(rows)
    if not rows:
        print("Aucune mesure disponible.")
        return
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns) + "</tr>" for row in rows)
    display(HTML(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"))


def show_image(path, label=""):
    """Compose a viewing checkerboard without modifying the source PNG."""
    from PIL import Image, ImageDraw
    from IPython.display import display
    path = Path(path)
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "#eeeeee")
        draw = ImageDraw.Draw(background)
        for y in range(0, rgba.height, 8):
            for x in range(0, rgba.width, 8):
                if (x // 8 + y // 8) % 2:
                    draw.rectangle((x, y, x + 7, y + 7), fill="#cccccc")
        background.alpha_composite(rgba)
        print(label or path.name, "|", source.mode, source.size)
        display(background.convert("RGB"))


def _active_chain():
    if sys.platform == "win32":
        return False  # GPU actions below are unavailable on Windows.
    result = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True, check=True)
    names = ("run_remote.py", "03-spatial-distillation/run.py", "03-spatial-distillation/train.py", "train_lora.py")
    return any(any(name in line for name in names) for line in result.stdout.splitlines())


def preview_script(script, arguments=()):
    """Show a future recipe without offering a launch switch."""
    path = contained(PROJECT, script)
    print(shlex.join([sys.executable, str(path), *map(str, arguments)]))
    print("Recette préparée uniquement. Aucun processus lancé.")


def run_script(script, arguments=(), *, enabled=False, gpu=False):
    """Print first; only launch when the notebook explicitly enables this action."""
    path = contained(PROJECT, script)
    command = [sys.executable, str(path), *map(str, arguments)]
    print(shlex.join(command))
    if not enabled:
        print("Aperçu uniquement. Passer le drapeau de cette cellule à True pour exécuter.")
        return
    if not path.is_file():
        raise FileNotFoundError(path)
    if gpu:
        if sys.platform == "win32":
            raise RuntimeError("Utiliser le kernel CUDA du serveur Linux ; aucun entraînement sur le laptop.")
        if os.environ.get("SHARM_NOTEBOOK_GPU") != "1":
            raise RuntimeError("Activer SHARM_NOTEBOOK_GPU=1 sur le serveur après avoir vérifié le GPU libre.")
        if _active_chain():
            raise RuntimeError("Une chaîne est déjà active. Lire son état et attendre sa fin.")
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if processes:
            raise RuntimeError("Le GPU exécute déjà un calcul. Attendre sa libération avant cette action.")
    subprocess.run(command, cwd=REPO, check=True)
