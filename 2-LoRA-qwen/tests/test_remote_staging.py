"""CPU guards for staged downloads and releasing owned temporary teacher weights."""
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
from sharm_image21.core import FeatureCache, atomic_json, digest


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


with patch.dict(sys.modules, {} if sys.platform != "win32" else {"fcntl": SimpleNamespace()}):
    remote = module("remote_staging", ROOT / "03-spatial-distillation/run_remote.py")
models = module("staged_models", ROOT / "01-encoder-distillation/prepare_models.py")


class DownloadStageChecks(unittest.TestCase):
    def patterns(self, stage):
        with patch.object(sys, "argv", ["prepare_models", "--stage", stage]), patch.object(models, "snapshot_download") as download:
            models.main()
        return [call.kwargs["allow_patterns"] for call in download.call_args_list]

    def test_teacher_download_avoids_diffuser_and_student_weight_files(self):
        original, student = self.patterns("teacher")
        self.assertIn("text_encoder/*", original)
        self.assertNotIn("transformer/*", original)
        self.assertNotIn("vae/*", original)
        self.assertNotIn("*.safetensors", student)

    def test_inference_does_not_redownload_released_teacher_weights(self):
        original, student = self.patterns("inference")
        self.assertNotIn("text_encoder/*", original)
        self.assertIn("text_encoder/*.json", original)
        self.assertIn("transformer/*", original)
        self.assertIn("*.safetensors", student)

    def test_same_experiment_with_reduced_activation_recomputation(self):
        small = json.loads((ROOT / "03-spatial-distillation/configs/8gb.json").read_text())
        server = json.loads((ROOT / "03-spatial-distillation/configs/remote16gb-256.json").read_text())
        self.assertEqual(server["flow"].pop("checkpointing-mode"), "alternating")
        small["flow"].pop("checkpointing-mode")
        self.assertEqual(server, small)


class TemporaryTeacherChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.root = self.project / "2-LoRA-qwen"
        self.artifacts = self.root / "artifacts/spatial"
        self.original = self.project / "1-local-generation/weights/qwen-image-2.1"
        self.directory = self.original / "text_encoder"
        self.directory.mkdir(parents=True)
        self.weight = self.directory / "model-00001-of-00001.safetensors"
        self.weight.write_bytes(b"temporary downloaded teacher")
        atomic_json(self.directory / "model.safetensors.index.json", {"weight_map": {"a": self.weight.name}})
        atomic_json(self.artifacts / "remote-workspace.json", {"root": str(self.project.resolve()), "temporary_teacher_weights": True})
        self.rows = [{"id": "a", "text": "fixture", "split": "train"}]
        corpus = self.artifacts / "dataset/corpus.jsonl"
        corpus.parent.mkdir(parents=True)
        corpus.write_text(json.dumps(self.rows[0])+"\n", encoding="utf-8")
        self.cache = FeatureCache(self.root / "cache/spatial/teacher", {"encoder": {"fixture": True}, "corpus": digest(self.rows)})
        self.cache.write([("a", torch.ones(3, 4, dtype=torch.bfloat16), [1, 2, 3])])
        for name, value in {"ROOT": self.root, "PROJECT": self.project, "ARTIFACTS": self.artifacts,
                            "ORIGINAL": self.original, "model_identity": lambda kind: {"fixture": True}}.items():
            active = patch.object(remote, name, value)
            active.start()
            self.addCleanup(active.stop)

    def test_only_owned_weights_released_after_complete_valid_cache(self):
        remote.release_teacher_weights()
        self.assertFalse(self.weight.exists())
        self.assertTrue((self.directory / "model.safetensors.index.json").exists())
        self.assertTrue((self.cache.directory / "shard-000000.safetensors").exists())
        report = json.loads((self.artifacts / "temporary-teacher-release.json").read_text())
        self.assertEqual(report["captions"], 1)
        remote.release_teacher_weights()  # Resuming does not need the removed weights.

    def test_unmarked_workspace_and_incomplete_cache_reject_deletion(self):
        atomic_json(self.artifacts / "remote-workspace.json", {"root": "wrong", "temporary_teacher_weights": True})
        with self.assertRaises(ValueError):
            remote.release_teacher_weights()
        self.assertTrue(self.weight.exists())
        atomic_json(self.artifacts / "remote-workspace.json", {"root": str(self.project.resolve()), "temporary_teacher_weights": True})
        self.cache.index["entries"] = {}
        atomic_json(self.cache.path, self.cache.index)
        with self.assertRaises(ValueError):
            remote.release_teacher_weights()
        self.assertTrue(self.weight.exists())

    def test_truncated_cache_and_external_weight_paths_rejected(self):
        shard = self.cache.directory / "shard-000000.safetensors"
        original = shard.read_bytes()
        shard.write_bytes(b"truncated")
        with self.assertRaises(Exception):
            remote.release_teacher_weights()
        self.assertTrue(self.weight.exists())
        shard.write_bytes(original)
        atomic_json(self.directory / "model.safetensors.index.json", {"weight_map": {"a": "../outside.safetensors"}})
        with self.assertRaises(ValueError):
            remote.release_teacher_weights()
        self.assertTrue(self.weight.exists())


if __name__ == "__main__":
    unittest.main()
