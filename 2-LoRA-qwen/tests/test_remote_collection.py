"""CPU checks for exact release collection without downloading duplicate base weights."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "publish"))
import publish_spatial as publisher

spec = importlib.util.spec_from_file_location("collect_remote", ROOT / "publish/collect_remote_release.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class CollectionChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.incoming = self.root / "incoming"
        self.output = self.root / "release"
        self.incoming.mkdir()
        self.base = self.root / "1-local-generation/weights/sharm-vl4b-fp8/model.safetensors"
        self.base.parent.mkdir(parents=True)
        self.base.write_bytes(b"public frozen weights")
        self.projector = self.incoming / "projector/model.safetensors"
        self.projector.parent.mkdir()
        self.projector.write_bytes(b"trained projector")
        (self.incoming / "README.md").write_bytes(b"Exact remote card\n")
        self.manifest = {"repo_id": publisher.REPO, "projector_sha256": publisher.sha(self.projector),
                         "files": {"README.md": publisher.sha(self.incoming / "README.md"),
                                   "projector/model.safetensors": publisher.sha(self.projector),
                                   "text_encoder/model.safetensors": publisher.sha(self.base)}}
        publisher.write(self.incoming / "release_manifest.json", self.manifest)
        images = {}
        for method in ("candidate_text", "candidate_layout"):
            for suffix in ("", "-preview"):
                name = f"validation/{method}{suffix}.png"
                path = self.incoming / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"checked image fixture")
                images[name] = publisher.sha(path)
        self.verification = {"passed": True, "release_manifest_sha256": publisher.sha(self.incoming / "release_manifest.json"),
                             "projector_sha256": self.manifest["projector_sha256"], "images": images}
        publisher.write(self.incoming / "export-validation.json", self.verification)

    def test_reuses_identical_local_base_and_preserves_exact_remote_bytes(self):
        report = collector.collect(self.incoming, self.output, self.root)
        self.assertTrue(report["passed"])
        self.assertEqual(report["reused_base_files"], 1)
        self.assertTrue((self.output / "text_encoder/model.safetensors").samefile(self.base))
        for name in ("README.md", "release_manifest.json", "export-validation.json", "projector/model.safetensors"):
            self.assertEqual((self.output / name).read_bytes(), (self.incoming / name).read_bytes())
        publisher.verified_payload(self.output)

    def test_altered_local_base_rejected_before_output_created(self):
        self.base.write_bytes(b"wrong revision")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            collector.collect(self.incoming, self.output, self.root)
        self.assertFalse(self.output.exists())

    def test_altered_report_unverified_export_and_unsafe_paths_rejected(self):
        self.verification["passed"] = False
        publisher.write(self.incoming / "export-validation.json", self.verification)
        with self.assertRaises(ValueError):
            collector.collect(self.incoming, self.output, self.root)
        self.verification["passed"] = True
        publisher.write(self.incoming / "export-validation.json", self.verification)
        with self.assertRaises(ValueError):
            collector.collect(self.incoming, self.incoming / "nested", self.root)
        with self.assertRaises(ValueError):
            collector.base_source("text_encoder/../outside.safetensors", self.root)

    def test_trained_weights_never_substituted_with_public_base(self):
        self.assertIsNone(collector.base_source("projector/model.safetensors", self.root))
        self.assertIsNone(collector.base_source("lora/pytorch_lora_weights.safetensors", self.root))
        self.projector.write_bytes(b"altered")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            collector.collect(self.incoming, self.output, self.root)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
