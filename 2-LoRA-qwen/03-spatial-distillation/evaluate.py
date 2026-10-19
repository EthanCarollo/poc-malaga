#!/usr/bin/env python3
"""Score identity-checked object annotations; missing reviews never disappear."""
from __future__ import annotations

import argparse
import json
import itertools
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from PIL import Image
from sharm_image21.spatial_data import box_iou, center_error, fingerprint, relation_holds, sha256, wilson, write_json


def score_record(record, annotation, root, min_iou, max_center_error, require_placement=True):
    path = root / record["file_name"]
    if sha256(path) != record["image_sha256"]:
        raise ValueError("Génération modifiée après son manifeste.")
    if not annotation or annotation.get("image_sha256") != record["image_sha256"] or not annotation.get("reviewed"):
        return {"passed": False, "reason": "missing_or_stale_review"}
    if not annotation.get("reviewer") or annotation.get("method") != "human_identity_and_boxes":
        raise ValueError("Revue humaine identifiée requise pour ce protocole.")
    found = annotation.get("objects", [])
    expected_ids = {o["object_id"] for o in record["objects"]}
    if len(found) != len(expected_ids) or {o.get("object_id") for o in found} != expected_ids:
        return {"passed": False, "reason": "object_count_or_identity"}
    if annotation.get("extra_objects", 0) != 0 or not annotation.get("distinct") or not all(o.get("identity_ok") for o in found):
        return {"passed": False, "reason": "fusion_extra_or_wrong_identity"}
    boxes = {o["object_id"]: o["box"] for o in found}
    ious = [box_iou(o["box"], boxes[o["object_id"]]) for o in record["objects"]]
    errors = [center_error(o["box"], boxes[o["object_id"]]) for o in record["objects"]]
    relation_ok = relation_holds(boxes["1"], boxes["2"], record["relation"])
    separation_ok = all(box_iou(a, b) == 0 for a, b in itertools.combinations(boxes.values(), 2))
    with Image.open(path) as image:
        alpha_ok = False
        if image.mode == "RGBA":
            histogram = image.getchannel("A").histogram()
            alpha_ok = (sum(histogram[:17])/(image.width*image.height) >= 0.005
                        and sum(histogram[240:])/(image.width*image.height) >= 0.01)
    placement_ok = min(ious) >= min_iou and max(errors) <= max_center_error
    passed = relation_ok and separation_ok and alpha_ok and (placement_ok or not require_placement)
    return {"passed": passed, "reason": "ok" if passed else "geometry_or_alpha", "relation_ok": relation_ok,
            "alpha_ok": alpha_ok, "separation_ok": separation_ok, "placement_ok": placement_ok,
            "min_iou": min(ious), "max_center_error": max(errors)}


def evaluate(manifest, annotations, root, methods, min_iou=0.5, max_center_error=0.05, min_groups=300, lower_bound=0.95):
    if len({a["id"] for a in annotations}) != len(annotations):
        raise ValueError("Annotations dupliquées.")
    index = {a["id"]: a for a in annotations}
    if len({r["id"] for r in manifest["rows"]}) != len(manifest["rows"]):
        raise ValueError("Générations dupliquées.")
    reports = {}
    for method in methods:
        records = [r for r in manifest["rows"] if r["method"] == method]
        if not records:
            raise ValueError(f"Aucune génération pour {method}.")
        expected = len(manifest["config"]["rows"])*len(manifest["config"]["seeds"])
        if len(records) != expected:
            raise ValueError("Génération incomplète : tous les prompts et graines sont requis.")
        expected_pairs = {(scene, seed) for scene in manifest["config"]["rows"] for seed in manifest["config"]["seeds"]}
        if {(r["scene_id"], r["seed"]) for r in records} != expected_pairs:
            raise ValueError("Les scènes/graines ne correspondent pas au protocole.")
        groups, results, relations = defaultdict(list), [], defaultdict(list)
        for record in records:
            outcome = score_record(record, index.get(record["id"]), root, min_iou, max_center_error,
                                   require_placement=method.endswith("layout"))
            groups[fingerprint(record["pair_group"])].append(outcome["passed"])
            relations[record["relation"]].append(outcome["passed"])
            results.append({"id": record["id"], **outcome})
        successes = sum(all(values) for values in groups.values())
        interval = wilson(successes, len(groups))
        reports[method] = {"images": len(records), "success_rate_per_image": sum(r["passed"] for r in results)/len(results),
            "pair_groups": len(groups), "successful_pair_groups": successes, "success_rate": successes/len(groups),
            "require_exact_placement": method.endswith("layout"),
            "wilson_95": interval, "statistical_unit": "object pair group; every scene and seed must pass",
            "per_relation": {k: sum(v)/len(v) for k, v in relations.items()}, "results": results,
            "passed": len(groups) >= min_groups and interval[0] >= lower_bound}
    return {"prediction_signature": manifest["signature"], "methods": reports,
            "criteria": {"min_iou": min_iou, "max_center_error": max_center_error, "min_groups": min_groups,
                         "required_wilson_lower_bound": lower_bound},
            "scope": "Two/three RGBA objects from the asset vocabulary; held-out pairs. Wilson assumes independent pair groups."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["candidate_text", "candidate_layout"])
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument("--max-center-error", type=float, default=0.05)
    parser.add_argument("--min-groups", type=int, default=300)
    parser.add_argument("--lower-bound", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.min_iou <= 1 or not 0 <= args.max_center_error <= 1 or args.min_groups < 1 or not 0 <= args.lower_bound <= 1:
        raise ValueError("Seuils d'évaluation invalides.")
    manifest = json.loads(args.predictions.read_text(encoding="utf-8"))
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    if annotations["prediction_signature"] != manifest["signature"]:
        raise ValueError("Annotations d'une autre génération.")
    report = evaluate(manifest, annotations["rows"], args.predictions.parent, args.methods,
                      args.min_iou, args.max_center_error, args.min_groups, args.lower_bound)
    report["annotations_sha256"] = sha256(args.annotations)
    write_json(args.output, report)
    print(json.dumps({k: {a: b for a, b in v.items() if a != "results"} for k, v in report["methods"].items()}, indent=2))
    if not all(r["passed"] for r in report["methods"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
