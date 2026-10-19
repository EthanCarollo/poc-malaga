#!/usr/bin/env python3
"""Assemble the exact remotely verified release using identical local base weights."""
from __future__ import annotations

import argparse
from pathlib import Path

from publish_spatial import (ARTIFACTS, PROJECT, REPO, contained, link_file,
                             payload_path_allowed, read, sha, verified_payload, write)


def base_source(name, project=PROJECT):
    """Only public, frozen weight components can be reused from the local machine."""
    contained(project, name)
    parts = name.split("/")
    if len(parts) != 2 or not name.endswith(".safetensors"):
        return None
    if parts[0] == "text_encoder":
        return contained(project / "1-local-generation/weights/sharm-vl4b-fp8", parts[1], ".safetensors")
    if parts[0] in ("transformer", "vae"):
        return contained(project / "1-local-generation/weights/qwen-image-2.1", name, ".safetensors")
    return None


def collect(incoming, output, project=PROJECT):
    incoming, output = Path(incoming).resolve(), Path(output).resolve()
    if incoming == output or incoming.is_relative_to(output) or output.is_relative_to(incoming):
        raise ValueError("Les dossiers de réception et d'export doivent être distincts.")
    manifest = read(incoming / "release_manifest.json")
    verification = read(incoming / "export-validation.json")
    if (manifest.get("repo_id") != REPO or verification.get("passed") is not True
            or verification.get("release_manifest_sha256") != sha(incoming / "release_manifest.json")
            or verification.get("projector_sha256") != manifest["projector_sha256"]):
        raise ValueError("Un export vérifié sur le serveur pour ce dépôt précis est requis.")
    expected_images = {f"validation/{method}{suffix}.png" for method in ("candidate_text", "candidate_layout")
                       for suffix in ("", "-preview")}
    if set(verification.get("images", {})) != expected_images:
        raise ValueError("Les deux générations RGBA vérifiées et leurs aperçus sont requis.")
    files = {}
    for name, expected in manifest["files"].items():
        if not payload_path_allowed(name):
            raise ValueError(f"Fichier hors périmètre de publication : {name}")
        source = base_source(name, project) or contained(incoming, name)
        if sha(source) != expected:
            raise ValueError(f"Fichier local incompatible avec l'export distant : {name}")
        files[name] = source
    for name, expected in verification["images"].items():
        source = contained(incoming, name, ".png")
        if sha(source) != expected:
            raise ValueError(f"Image de vérification modifiée : {name}")
        files[name] = source
    for name in ("release_manifest.json", "export-validation.json"):
        files[name] = contained(incoming, name)
    # Validate the complete source set before populating the final release.
    for name, source in files.items():
        link_file(source, contained(output, name))
    _, payload = verified_payload(output)
    report = {"passed": True, "files": len(payload), "release_manifest_sha256": sha(output / "release_manifest.json"),
              "projector_sha256": manifest["projector_sha256"], "reused_base_files": sum(base_source(n, project) is not None for n in files),
              "scope": "Exact remote release assembled locally; no inference or remote publication performed"}
    write(output.parent / "remote-collection.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incoming", type=Path, default=ARTIFACTS / "remote-release")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "release")
    args = parser.parse_args()
    import json
    print(json.dumps(collect(args.incoming, args.output)))


if __name__ == "__main__":
    main()
