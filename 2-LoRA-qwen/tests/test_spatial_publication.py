"""CPU regression checks for spatial release guards; no GPU or network calls."""
from __future__ import annotations

from contextlib import nullcontext
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from sharm_image21.core import file_hash
from sharm_image21.pipeline import SharmImage21Pipeline
from sharm_image21.spatial_data import layout_prompt, read_rows

spec = importlib.util.spec_from_file_location("publish_spatial", ROOT / "publish/publish_spatial.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


class LayoutPromptChecks(unittest.TestCase):
    objects = [{"label": "Bulbasaur", "box": [0.1, 0.2, 0.4, 0.8]},
               {"label": "shiny Castform, rainy form", "box": [0.6, 0.2, 0.9, 0.8]}]

    def test_bindings_english_french_and_added_labels(self):
        for prompt in ("Object 1: Bulbasaur. Object 2: shiny Castform, rainy form.",
                       "Objet 1 : Bulbasaur. Objet 2 : shiny Castform, rainy form.",
                       "Bulbasaur to the left of shiny Castform, rainy form."):
            result, boxes = layout_prompt(prompt, self.objects)
            self.assertTrue(result.startswith(prompt))
            self.assertEqual(boxes, [o["box"] for o in self.objects])
            self.assertEqual(result.count("Normalized object boxes xyxy:"), 1)
            if prompt.startswith(("Object 1", "Objet 1")):
                self.assertEqual(result.count("Bulbasaur"), 1)
            else:
                self.assertIn("Object 1: Bulbasaur.", result)

    def test_mismatched_numbering_and_duplicate_coordinates_rejected(self):
        for prompt in ("Object 1: Charmander. Object 2: Bulbasaur.",
                       "Object 1: Bulbasaur.", "Normalized object boxes xyxy: []",
                       "Object 1: Bulbasaur. Object 2: shiny Castform, rainy form. Object 3: Charmander.",
                       "Object 1: Bulbasaur. Object 1: Bulbasaur. Object 2: shiny Castform, rainy form."):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                layout_prompt(prompt, self.objects)

    def test_invalid_objects_and_boxes_rejected(self):
        for objects in ([], self.objects*9, [{"label": "", "box": [0, 0, 1, 1]}],
                        [{"label": "a", "box": [0.5, 0, 0.2, 1]}],
                        [{"label": "a", "box": [0, 0, 1.1, 1]}],
                        [{"label": "a", "box": [0, 0, float("nan"), 1]}],
                        [{"label": "a", "box": None}]):
            with self.subTest(objects=objects), self.assertRaises(ValueError):
                layout_prompt("test", objects)

    def test_exact_training_prompt_convention(self):
        dataset = ROOT / "artifacts/spatial/dataset/scenes.jsonl"
        if not dataset.exists():
            self.skipTest("Local training dataset absent")
        for row in read_rows(dataset):
            text, boxes = layout_prompt(row["text_plain"], row["objects"])
            self.assertEqual(text, row["text"])
            self.assertEqual(boxes, [o["box"] for o in row["objects"]])


class PipelineReleaseChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        (self.directory / "projector").mkdir()
        (self.directory / "projector/model.safetensors").write_bytes(b"fixture")
        self.config = {"encoder_format": "fp8-e4m3", "encoder_frozen": True,
                       "projector_sha256": file_hash(self.directory / "projector/model.safetensors"),
                       "pipeline_format": "spatial-v1", "training_complete": True,
                       "student_identity": {"kind": "fixture"}, "spatial_lora_sha256": None}

    def pipeline(self):
        (self.directory / "sharm_config.json").write_text(json.dumps(self.config), encoding="utf-8")
        return SharmImage21Pipeline(self.directory)

    def test_spatial_completion_legacy_validation_and_weight_guards(self):
        self.pipeline()
        self.config["training_complete"] = False
        with self.assertRaisesRegex(ValueError, "incomplet"):
            self.pipeline()
        self.config.pop("pipeline_format")
        with self.assertRaisesRegex(ValueError, "non valid"):
            self.pipeline()
        self.config["base_validation_passed"] = True
        self.pipeline()
        self.config["projector_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Empreinte"):
            self.pipeline()

    def test_spatial_lora_integrity(self):
        folder = self.directory / "lora"
        folder.mkdir()
        (folder / "pytorch_lora_weights.safetensors").write_bytes(b"adapter")
        self.config["spatial_lora_sha256"] = file_hash(folder / "pytorch_lora_weights.safetensors")
        self.pipeline()
        (folder / "pytorch_lora_weights.safetensors").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "LoRA spatial"):
            self.pipeline()

    def test_sequential_encoding_frees_encoder_before_projector(self):
        model = self.pipeline()
        events, prompts = [], []

        class Encoder:
            identity = {"kind": "fixture"}

            def __init__(self, *args, **kwargs):
                events.append("encoder")

            def encode(self, texts):
                prompts.extend(texts)
                return [(torch.zeros(i+4, 8), []) for i in range(len(texts))]

            def __del__(self):
                events.append("freed")

        class Projector:
            def eval(self):
                return self

            def __call__(self, values, layout=None):
                return torch.ones(len(values), 16)

        def load(*args):
            self.assertEqual(events, ["encoder", "freed"])
            events.append("projector")
            return Projector()

        prefix = "sharm_image21.pipeline."
        with patch(prefix+"FrozenEncoder", Encoder), patch(prefix+"Projector.load", load), \
                patch(prefix+"torch.autocast", lambda *a, **k: nullcontext()), \
                patch.object(torch.Tensor, "cuda", lambda value: value):
            embeddings, mask = model.encode_prompt(["a", "b"])
        self.assertEqual(prompts, ["a", "b"])
        self.assertEqual(embeddings.shape, (2, 5, 16))
        self.assertEqual(embeddings.dtype, torch.bfloat16)
        self.assertEqual(mask.sum(1).tolist(), [4, 5])
        with self.assertRaises(ValueError):
            model.encode_prompt(["a", "b"], layout=LayoutPromptChecks.objects)

    def test_call_forwards_layout_and_loads_matching_spatial_lora(self):
        (self.directory / "lora").mkdir()
        lora = self.directory / "lora/pytorch_lora_weights.safetensors"
        lora.write_bytes(b"lora")
        self.config["spatial_lora_sha256"] = file_hash(lora)
        model = self.pipeline()
        calls = []

        class Diffuser:
            def load_lora_weights(self, path, **kwargs):
                calls.append((path, kwargs))

            def __call__(self, **kwargs):
                self.last = kwargs
                return SimpleNamespace(images=["output"])

        with patch.object(model, "encode_prompt", return_value=(torch.zeros(1, 4, 16), torch.ones(1, 4))) as encode, \
                patch("sharm_image21.pipeline.load_diffuser_pipeline", return_value=Diffuser()), \
                patch.object(torch.Tensor, "cuda", lambda value: value):
            result = model("prompt", layout=LayoutPromptChecks.objects, width=256)
            self.assertEqual(result.images, ["output"])
            encode.assert_called_once_with("prompt", layout=LayoutPromptChecks.objects)
            self.assertEqual(calls, [(self.directory / "lora", {"adapter_name": "spatial"})])


class PublicationGuardChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run, self.dataset = self.root / "run", self.root / "dataset"
        self.weights = self.run / "flow/best/projector/model.safetensors"
        self.weights.parent.mkdir(parents=True)
        self.weights.write_bytes(b"trained")
        self.candidate = {"signature": "training", "step": 10, "config": {"phase": "flow"},
                          "numerical_validation_loss": 0.1, "projector_sha256": publisher.sha(self.weights),
                          "lora_sha256": None, "identity": {"dataset_sha256": "dataset"}}
        publisher.write(self.run / "flow/best/candidate.json", self.candidate)
        complete = {"signature": "training", "steps_completed": 10}
        publisher.write(self.run / "flow/training-complete.json", complete)
        publisher.write(self.run / "warmup/training-complete.json", complete)
        publisher.write(self.run / "run-status.json", {"state": "complete"})
        publisher.write(self.dataset / "manifest.json", {"dataset_sha256": "dataset"})
        self.predictions = {"signature": "gallery", "config": {"candidate_sha256": publisher.sha(self.run / "flow/best/candidate.json")},
                            "rows": [{"id": "one", "method": "candidate_text"}, {"id": "two", "method": "candidate_layout"}]}
        publisher.write(self.run / "validation/predictions.json", self.predictions)
        self.review = {"passed": True, "method": "agent_visual_review", "reviewer": "test-fixture",
                       "prediction_signature": "gallery", "projector_sha256": publisher.sha(self.weights),
                       "reviewed_ids": ["one", "two"]}
        publisher.write(self.root / "visual-review.json", self.review)
        self.patch = patch.object(publisher, "ARTIFACTS", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_completed_matching_and_fully_reviewed_candidate_accepted(self):
        self.assertEqual(publisher.prerequisites(self.run, self.dataset)[1], self.candidate)

    def test_unfinished_stale_gallery_and_altered_weights_rejected(self):
        publisher.write(self.run / "run-status.json", {"state": "running"})
        with self.assertRaisesRegex(ValueError, "termin"):
            publisher.prerequisites(self.run, self.dataset)
        publisher.write(self.run / "run-status.json", {"state": "complete"})
        self.predictions["config"]["candidate_sha256"] = "stale"
        publisher.write(self.run / "validation/predictions.json", self.predictions)
        with self.assertRaisesRegex(ValueError, "autre révision"):
            publisher.prerequisites(self.run, self.dataset)
        self.weights.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Projecteur modifi"):
            publisher.prerequisites(self.run, self.dataset)

    def test_review_requires_exact_gallery_all_images_and_identified_reviewer(self):
        for updates in ({"passed": False}, {"reviewer": ""}, {"method": "automated_mse"},
                        {"prediction_signature": "stale"}, {"projector_sha256": "stale"},
                        {"reviewed_ids": ["one"]}):
            with self.subTest(updates=updates):
                publisher.write(self.root / "visual-review.json", {**self.review, **updates})
                with self.assertRaises(ValueError):
                    publisher.prerequisites(self.run, self.dataset)

    def test_path_and_payload_scope(self):
        self.assertEqual(publisher.contained(self.root, "images/a.png", ".png"), self.root / "images/a.png")
        for name in ("../outside.png", "C:/outside.png", "/outside.png", "images/../../a.png", "images\\a.png", "images//a.png"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                publisher.contained(self.root, name, ".png")
        for name in (".env", "token", "publication-request.json", "checkpoint.pt", "cache/a.safetensors",
                     "dataset/images/a.png", "sharm_image21/__pycache__/a.pyc", "processor/tokenizer.secret"):
            self.assertFalse(publisher.payload_path_allowed(name), name)
        for name in ("README.md", "projector/model.safetensors", "text_encoder/model-00001-of-00002.safetensors",
                     "sharm_image21/pipeline.py", "comparisons/spatial/images/a.png"):
            self.assertTrue(publisher.payload_path_allowed(name), name)

    def test_verified_payload_detects_changes_and_excludes_unlisted_files(self):
        output = self.root / "release"
        output.mkdir()
        (output / "README.md").write_text("fixture", encoding="utf-8")
        (output / ".env").write_text("must never upload", encoding="utf-8")
        manifest = {"repo_id": publisher.REPO, "projector_sha256": "projector", "files": {"README.md": publisher.sha(output / "README.md")}}
        publisher.write(output / "release_manifest.json", manifest)
        images = {}
        for method in ("candidate_text", "candidate_layout"):
            for suffix in ("", "-preview"):
                name = f"validation/{method}{suffix}.png"
                path = output / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"fixture")
                images[name] = publisher.sha(path)
        verification = {"passed": True, "release_manifest_sha256": publisher.sha(output / "release_manifest.json"),
                        "projector_sha256": "projector", "images": images}
        publisher.write(output / "export-validation.json", verification)
        _, files = publisher.verified_payload(output)
        self.assertNotIn(".env", files)
        self.assertIn("validation/candidate_layout.png", files)
        (output / "README.md").write_text("altered", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Export modifi"):
            publisher.verified_payload(output)
        changed = copy.deepcopy(manifest)
        changed["files"][".env"] = publisher.sha(output / ".env")
        publisher.write(output / "release_manifest.json", changed)
        verification["release_manifest_sha256"] = publisher.sha(output / "release_manifest.json")
        publisher.write(output / "export-validation.json", verification)
        with self.assertRaisesRegex(ValueError, "hors périmètre"):
            publisher.verified_payload(output)


if __name__ == "__main__":
    unittest.main()
