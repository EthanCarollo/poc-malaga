"""CPU checks for ordered conditioning, RGBA ground truth and honest scoring."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
import torch
from PIL import Image, ImageDraw
from sharm_image21.core import FeatureCache, Projector, atomic_json, digest
from sharm_image21.spatial_data import compose, fingerprint, read_rows, relation_holds, sha256, wilson

spec = importlib.util.spec_from_file_location("spatial_evaluation", ROOT / "03-spatial-distillation/evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class ProjectorChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)

    def test_legacy_arithmetic_and_serialization(self):
        model = Projector(8, 16, 8)
        source = torch.randn(7, 8)
        expected = model.skip(source)+model.down(torch.nn.functional.gelu(model.up(model.norm(source))))
        torch.testing.assert_close(model(source), expected, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as folder:
            model.save(Path(folder))
            loaded = Projector.load(Path(folder))
            torch.testing.assert_close(loaded(source), expected, rtol=0, atol=0)
            self.assertNotIn("context_layers", loaded.settings)

    def test_context_mask_order_layout_and_bf16_roundtrip(self):
        model = Projector(8, 16, 8, context_layers=1, context_heads=2, layout_enabled=True).eval()
        torch.nn.init.normal_(model.down.weight, std=0.1)
        source = torch.randn(5, 8)
        padding = torch.cat((source, torch.randn(3, 8)*100))[None]
        with torch.no_grad():
            result = model(source)
            padded = model(padding, attention_mask=torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]))
            torch.testing.assert_close(result, padded[0, :5], atol=1e-5, rtol=1e-5)
            permutation = torch.tensor([4, 3, 2, 1, 0])
            self.assertFalse(torch.allclose(model(source[permutation]), result[permutation], atol=1e-5))
        boxes = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.6, 0.2, 0.9, 0.8]])
        output = model(source, layout=boxes)
        self.assertEqual(output.shape, (7, 16))
        output.square().mean().backward()
        for prefix in ("context.", "layout_box.", "layout_object."):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if n.startswith(prefix)))
        with tempfile.TemporaryDirectory() as folder:
            model.save(Path(folder), storage_dtype=torch.bfloat16)
            loaded = Projector.load(Path(folder)).float()
            with torch.no_grad():
                expected = model(source, layout=boxes)
                torch.testing.assert_close(loaded(source, layout=boxes), expected, rtol=0.02, atol=0.01)
        with self.assertRaises(ValueError):
            model(source, layout=torch.tensor([[0.8, 0.1, 0.2, 0.4]]))
        with self.assertRaises(ValueError):
            Projector(8, 16, 8)(source, layout=boxes)

    def test_gradient_through_frozen_surrogate(self):
        # This checks graph construction, not the real NF4 CUDA kernel.
        model = Projector(8, 16, 8, context_layers=1, context_heads=2, layout_enabled=True)
        decoder = torch.nn.Linear(16, 4).requires_grad_(False)
        source, target = torch.randn(6, 8), torch.randn(4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        for _ in range(2):
            optimizer.zero_grad()
            output = decoder(model(source).mean(0))
            torch.nn.functional.mse_loss(output, target).backward()
            self.assertTrue(model.skip.weight.grad.abs().sum() > 0)
            self.assertTrue(all(p.grad is None for p in decoder.parameters()))
            optimizer.step()


class DataAndScoringChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.assets = []
        for i, color in enumerate(("red", "blue", "green", "yellow", "purple", "orange")):
            image = Image.new("RGBA", (48, 64), (0, 0, 0, 0))
            ImageDraw.Draw(image).rectangle((6, 8, 40, 56), fill=color)
            name = f"{i}.png"
            image.save(self.root / name)
            self.assets.append({"id": str(i), "group_id": str(i), "label": color+" rectangle", "file_name": name})
        self.source = self.root / "assets.jsonl"
        self.source.write_text("".join(json.dumps(a)+"\n" for a in self.assets), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_dataset_reproducibility_and_pair_holdout(self):
        paths = [self.root / "dataset-a", self.root / "dataset-b"]
        for output in paths:
            subprocess.run([sys.executable, str(ROOT / "03-spatial-distillation/prepare_dataset.py"),
                            "--assets", str(self.source), "--output", str(output), "--count", "120", "--resolution", "128"],
                           check=True, capture_output=True, text=True)
        a, b = (read_rows(p / "scenes.jsonl") for p in paths)
        self.assertEqual(a, b)
        splits = {}
        self.assertEqual({len(r["objects"]) for r in a}, {2, 3})
        for row in a:
            self.assertNotIn("Normalized object boxes", row["text_plain"])
            self.assertIn("Normalized object boxes", row["text"])
            pair = tuple(row["pair_group"])
            self.assertEqual(splits.setdefault(pair, row["split"]), row["split"])
            self.assertEqual(sha256(paths[0] / row["file_name"]), row["image_sha256"])
            self.assertTrue(relation_holds(row["objects"][0]["box"], row["objects"][1]["box"], row["relation"]))
            with Image.open(paths[0] / row["file_name"]) as image:
                self.assertEqual(image.mode, "RGBA")
                self.assertEqual(image.getchannel("A").getextrema(), (0, 255))
        manifest = json.loads((paths[0] / "manifest.json").read_text())
        self.assertEqual(manifest["dataset_sha256"], fingerprint(a))

    def fixture(self):
        objects = [{**self.assets[0], "box": [0.1, 0.3, 0.35, 0.7]},
                   {**self.assets[1], "box": [0.65, 0.3, 0.9, 0.7]}]
        image, objects = compose(128, objects, self.root)
        image.save(self.root / "result.png")
        record = {"id": "prediction", "scene_id": "scene", "seed": 1, "method": "candidate_layout",
                  "pair_group": ["a", "b"], "relation": "left", "file_name": "result.png",
                  "image_sha256": sha256(self.root / "result.png"), "objects": objects}
        annotation = {"id": record["id"], "image_sha256": record["image_sha256"], "reviewed": True,
                      "reviewer": "unit-check", "method": "human_identity_and_boxes", "distinct": True, "extra_objects": 0,
                      "objects": [{"object_id": o["object_id"], "box": o["box"], "identity_ok": True} for o in objects]}
        return record, annotation

    def test_honest_success_and_failure_scores(self):
        record, annotation = self.fixture()
        score = lambda a: evaluation.score_record(record, a, self.root, 0.5, 0.05)
        self.assertTrue(score(annotation)["passed"])
        self.assertFalse(score(None)["passed"])
        for field, value in (("distinct", False), ("extra_objects", 1), ("reviewed", False), ("image_sha256", "stale")):
            changed = copy.deepcopy(annotation)
            changed[field] = value
            self.assertFalse(score(changed)["passed"])
        changed = copy.deepcopy(annotation)
        changed["objects"][0]["identity_ok"] = False
        self.assertFalse(score(changed)["passed"])
        shifted = copy.deepcopy(annotation)
        for obj in shifted["objects"]:
            obj["box"] = [v-0.1 if i in (1, 3) else v for i, v in enumerate(obj["box"])]
        self.assertFalse(score(shifted)["passed"])
        self.assertTrue(evaluation.score_record(record, shifted, self.root, 0.5, 0.05, require_placement=False)["passed"])
        changed = copy.deepcopy(annotation)
        changed["objects"][0]["box"], changed["objects"][1]["box"] = changed["objects"][1]["box"], changed["objects"][0]["box"]
        self.assertFalse(score(changed)["passed"])
        manifest = {"signature": "test", "config": {"rows": ["scene"], "seeds": [1]}, "rows": [record]}
        report = evaluation.evaluate(manifest, [annotation], self.root, ["candidate_layout"], min_groups=1, lower_bound=0)
        self.assertTrue(report["methods"]["candidate_layout"]["passed"])
        self.assertFalse(evaluation.evaluate(manifest, [], self.root, ["candidate_layout"], min_groups=1, lower_bound=0.9)["methods"]["candidate_layout"]["passed"])
        manifest["config"]["seeds"].append(2)
        with self.assertRaises(ValueError):
            evaluation.evaluate(manifest, [annotation], self.root, ["candidate_layout"])

    def test_composition_rejects_overlap_and_invalid_boxes(self):
        objects = [{**self.assets[0], "box": [0.1, 0.1, 0.8, 0.8]}, {**self.assets[1], "box": [0.1, 0.1, 0.8, 0.8]}]
        with self.assertRaises(ValueError):
            compose(128, objects, self.root)
        self.assertLess(wilson(4, 4)[0], 0.6)
        self.assertGreater(wilson(300, 300)[0], 0.98)

    def test_data_runner_reuse_and_profile_conflict(self):
        profile = self.root / "profile.json"
        profile.write_text(json.dumps({"dataset": {"count": 30, "resolution": 128, "seed": 42}}), encoding="utf-8")
        command = [sys.executable, str(ROOT / "03-spatial-distillation/run.py"), "--stage", "data",
                   "--assets", str(self.source), "--dataset", str(self.root / "dataset"),
                   "--output", str(self.root / "run"), "--profile", str(profile)]
        for _ in range(2):
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        profile.write_text(json.dumps({"dataset": {"count": 31, "resolution": 128, "seed": 42}}), encoding="utf-8")
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ne correspond pas", result.stderr)


class TrainingChecks(unittest.TestCase):
    def test_cpu_warmup_checkpoint_and_configuration_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset, cache, initial, output = (root / k for k in ("data", "cache", "initial", "run"))
            dataset.mkdir()
            rows = [{"id": str(i), "caption_id": str(i), "split": "validation" if i >= 4 else "train", "objects": []}
                    for i in range(6)]
            corpus = [{"id": str(i), "text": f"scene {i}", "split": r["split"]} for i, r in enumerate(rows)]
            for name, values in (("scenes.jsonl", rows), ("corpus.jsonl", corpus)):
                (dataset / name).write_text("".join(json.dumps(r)+"\n" for r in values), encoding="utf-8")
            atomic_json(dataset / "manifest.json", {"dataset_sha256": digest(rows)})
            matrix = torch.randn(8, 16)
            student = FeatureCache(cache / "student", {"corpus": digest(corpus), "encoder": {"kind": "synthetic-student"}})
            teacher = FeatureCache(cache / "teacher", {"corpus": digest(corpus), "encoder": {"kind": "synthetic-teacher"}})
            for row in rows:
                source = torch.randn(5, 8)
                student.write([(row["id"], source, list(range(5)))])
                teacher.write([(row["id"], source @ matrix, list(range(5)))])
            Projector(8, 16, 8).save(initial)
            command = [sys.executable, str(ROOT / "03-spatial-distillation/train.py"), "warmup", "--device", "cpu",
                       "--dataset", str(dataset), "--cache", str(cache), "--output", str(output),
                       "--init-projector", str(initial), "--steps", "4", "--accumulation", "1", "--eval-every", "2",
                       "--context-layers", "1", "--context-heads", "2"]
            environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            result = subprocess.run(command, capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            complete = json.loads((output / "training-complete.json").read_text())
            self.assertEqual(complete["steps_completed"], 4)
            self.assertFalse(complete["spatial_quality_validated"])
            checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["step"], 4)
            self.assertIn("optimizer", checkpoint)
            self.assertIn("python_rng", checkpoint)
            self.assertIn("torch_rng", checkpoint)
            weights = output / "best/projector/model.safetensors"
            original_hash = sha256(weights)
            Projector.load(weights.parent)
            result = subprocess.run(command, capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(sha256(weights), original_hash)
            result = subprocess.run(command+["--learning-rate", "0.02"], capture_output=True, text=True, env=environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("autre expérience", result.stderr)


if __name__ == "__main__":
    unittest.main()
