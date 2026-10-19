#!/usr/bin/env python3
"""Validate the shipped package and execute real natural/layout RGBA inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=ROOT / "artifacts/spatial/release")
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/spatial/run")
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/spatial/dataset")
    args = parser.parse_args()
    if json.loads((args.run / "run-status.json").read_text())["state"] != "complete":
        raise ValueError("Terminer la chaîne avant de mobiliser le GPU pour l'export.")
    lock = (args.run / "run.lock").open("a")
    if sys.platform != "win32":
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    directory = args.release.resolve()
    manifest_path = directory / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or sha(path) != expected:
            raise ValueError(f"Export modifié : {name}")
    # Import the exact code copied into the release, with no workspace package cached.
    sys.path.insert(0, str(directory))
    import torch
    import sharm_image21.pipeline as shipped
    from sharm_image21.core import Projector
    from sharm_image21.image_io import save_png, transparent_fraction
    if not Path(shipped.__file__).resolve().is_relative_to(directory):
        raise ValueError("Le loader importé ne provient pas de l'export.")
    candidate = Projector.load(args.run / "flow/best/projector")
    exported = Projector.load(directory / "projector")
    for name, value in candidate.state_dict().items():
        torch.testing.assert_close(value, exported.state_dict()[name], rtol=0, atol=0)
    del candidate, exported
    model = shipped.SharmImage21Pipeline.from_pretrained(directory)
    predictions = json.loads((args.run / "validation/predictions.json").read_text())
    scenes = {r["id"]: r for r in (json.loads(line) for line in (args.dataset / "scenes.jsonl").read_text(encoding="utf-8").splitlines())}
    images, results = {}, []
    for method in ("candidate_text", "candidate_layout"):
        example = next(r for r in predictions["rows"] if r["method"] == method)
        scene = scenes[example["scene_id"]]
        layout = [{"label": o["label"], "box": o["box"]} for o in scene["objects"]] if method.endswith("layout") else None
        resolution = predictions["config"]["resolution"]
        output = model(scene["text_plain"], layout=layout, width=resolution, height=resolution, output_resolution=resolution,
                       num_inference_steps=predictions["config"]["steps"], true_cfg_scale=1.0, use_kv_cache=False,
                       generator=torch.Generator(device="cuda").manual_seed(example["seed"]))
        image = output.images[0]
        if image.mode != "RGBA":
            raise ValueError("RGBA natif requis à la sortie du modèle exporté.")
        name, preview = f"validation/{method}.png", f"validation/{method}-preview.png"
        save_png(image, directory / name, preview_path=directory / preview, require_transparency=True)
        images.update({name: sha(directory / name), preview: sha(directory / preview)})
        results.append({"method": method, "scene_id": scene["id"], "seed": example["seed"],
                        "transparent_fraction": transparent_fraction(image)})
    report = {"passed": True, "release_manifest_sha256": sha(manifest_path), "images": images,
              "projector_sha256": manifest["projector_sha256"], "projector_roundtrip_exact": True,
              "loader_from_release": True, "cases": results,
              "scope": "Export integrity and real natural/layout RGBA inference; spatial quality requires separate review."}
    (directory / "export-validation.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
