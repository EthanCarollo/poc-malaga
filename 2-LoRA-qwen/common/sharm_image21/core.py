from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent
WEIGHTS = PROJECT / "1-local-generation" / "weights"
ORIGINAL = WEIGHTS / "qwen-image-2.1"
STUDENT = WEIGHTS / "sharm-vl4b-fp8"
ARTIFACTS = ROOT / "artifacts"
CACHE = ROOT / "cache"
FEATURE_CONVENTION = "qwenimage21-raw-template-leftpad-pre-final-rmsnorm-v1"
TEACHER_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"
STUDENT_REVISION = "fefbb44cbcce8d1bb7e20b920b94f77432b3446d"
SYS_PROMPT = "Comprehend and analyze the provided prompt."
TEMPLATE = (
    f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def model_identity(kind: str, model_dir: Path | None = None) -> dict:
    model_dir = model_dir or (STUDENT if kind == "student" else ORIGINAL / "text_encoder")
    return {
        "kind": kind,
        "revision": STUDENT_REVISION if kind == "student" else TEACHER_REVISION,
        "config_sha256": file_hash(model_dir / "config.json"),
        "index_sha256": file_hash(model_dir / "model.safetensors.index.json"),
        "convention": FEATURE_CONVENTION,
        "prefill_policy": "single-prompt-fp8" if kind == "student" else "left-padded-bf16",
    }


class Projector(nn.Module):
    def __init__(self, input_dim: int = 2560, output_dim: int = 4096, bottleneck: int = 1024):
        super().__init__()
        self.settings = dict(input_dim=input_dim, output_dim=output_dim, bottleneck=bottleneck)
        self.skip = nn.Linear(input_dim, output_dim)
        self.norm = nn.RMSNorm(input_dim, eps=1e-6)
        self.up = nn.Linear(input_dim, bottleneck)
        self.down = nn.Linear(bottleneck, output_dim)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

    def forward(self, hidden):
        return self.skip(hidden) + self.down(F.gelu(self.up(self.norm(hidden))))

    def save(self, path: Path, metadata: dict | None = None):
        path.mkdir(parents=True, exist_ok=True)
        atomic_json(path / "config.json", {**self.settings, "convention": FEATURE_CONVENTION, **(metadata or {})})
        temporary = path / "model.tmp.safetensors"
        save_file({key: val.detach().cpu().contiguous() for key, val in self.state_dict().items()}, temporary)
        os.replace(temporary, path / "model.safetensors")

    @classmethod
    def load(cls, path: Path, device="cpu"):
        config = json.loads((path / "config.json").read_text())
        if config.get("convention") != FEATURE_CONVENTION:
            raise ValueError("Convention d'embeddings du projecteur incompatible.")
        model = cls(**{key: config[key] for key in ("input_dim", "output_dim", "bottleneck")})
        model.load_state_dict(load_file(path / "model.safetensors"))
        return model.to(device)


class FrozenEncoder:
    """Extract the norm INPUT without LM logits, past-key-values or all-layer captures."""

    def __init__(self, kind: str, gpu_memory: str = "11GiB", model_dir: Path | None = None):
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        if kind not in ("teacher", "student"):
            raise ValueError(kind)
        self.kind = kind
        self.model_dir = model_dir or (STUDENT if kind == "student" else ORIGINAL / "text_encoder")
        processor_dir = self.model_dir if kind == "student" else ORIGINAL / "processor"
        self.processor = Qwen3VLProcessor.from_pretrained(processor_dir, local_files_only=True)
        self.identity = model_identity(kind, self.model_dir)
        sys_message = [{"role": "system", "content": [{"type": "text", "text": SYS_PROMPT}]}]
        self.drop_idx = len(self.processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)[0])
        kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
        if kind == "student":
            os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")
            kwargs["device_map"] = "cuda"
        else:
            kwargs.update(device_map="auto", max_memory={0: gpu_memory, "cpu": "9GiB"})
        started = time.monotonic()
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_dir, **kwargs)
        self.model.eval().requires_grad_(False)
        self.hidden_size = self.model.config.text_config.hidden_size
        self.device = torch.device("cuda:0")
        self.quantized_linears = 0
        if kind == "student":
            for name, module in self.model.named_modules():
                if type(module).__name__ == "FP8Linear":
                    if module.weight.dtype != torch.float8_e4m3fn:
                        raise RuntimeError(f"Déquantification inattendue : {name} / {module.weight.dtype}")
                    self.quantized_linears += 1
            if not self.quantized_linears or getattr(self.model.config.quantization_config, "dequantize", False):
                raise RuntimeError("Le décodeur étudiant n'est pas chargé en FP8 effectif.")
        print(json.dumps({"encoder": kind, "load_seconds": round(time.monotonic()-started, 2),
                          "hidden_size": self.hidden_size, "fp8_linears": self.quantized_linears,
                          "gpu_allocated_gib": torch.cuda.memory_allocated()/2**30}), flush=True)

    @torch.inference_mode()
    def encode(self, prompts: list[str]) -> list[tuple[torch.Tensor, list[int]]]:
        # On SM120, mixed-length FP8 prefill changed these features by 8.4% relative
        # to single-prompt inference. Keep the same prefill in cache and deployment.
        if self.kind == "student" and len(prompts) > 1:
            return [self.encode([text])[0] for text in prompts]
        inputs = self.processor(text=[TEMPLATE.format(text or " ") for text in prompts],
                                padding=True, padding_side="left", return_tensors="pt").to(self.device)
        captured = []

        def take_norm_input(module, args):
            captured.append(args[0].detach())

        handle = self.model.model.language_model.norm.register_forward_pre_hook(take_norm_input)
        try:
            self.model.model(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Impossible d'isoler la dernière représentation pré-normalisation.")
        hidden = captured.pop()
        result = []
        for row, mask, ids in zip(hidden, inputs.attention_mask.bool(), inputs.input_ids):
            values = row[mask][self.drop_idx:].to(device="cpu", dtype=torch.bfloat16).contiguous()
            token_ids = ids[mask][self.drop_idx:].cpu().tolist()
            if not len(values) or values.shape[-1] != self.hidden_size or not torch.isfinite(values).all():
                raise RuntimeError("Embeddings vides, de mauvaise taille ou non finis.")
            result.append((values, token_ids))
        return result


class FeatureCache:
    def __init__(self, directory: Path, expected: dict | None = None):
        self.directory = directory
        self.path = directory / "index.json"
        directory.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.index = json.loads(self.path.read_text())
            if expected is not None and self.index["identity"] != expected:
                raise ValueError(f"Cache incompatible : {directory}. Choisir un nouveau dossier.")
        else:
            if expected is None:
                raise FileNotFoundError(self.path)
            self.index = {"identity": expected, "entries": {}, "shards": 0}
            atomic_json(self.path, self.index)

    def __contains__(self, sample_id):
        return sample_id in self.index["entries"]

    def write(self, rows: list[tuple[str, torch.Tensor, list[int]]]):
        if not rows:
            return
        shard_name = f"shard-{self.index['shards']:06d}.safetensors"
        target = self.directory / shard_name
        temporary = target.with_suffix(".tmp")
        save_file({sample_id: tensor.cpu().contiguous() for sample_id, tensor, _ in rows}, temporary)
        os.replace(temporary, target)
        for sample_id, tensor, token_ids in rows:
            self.index["entries"][sample_id] = {"shard": shard_name, "shape": list(tensor.shape),
                                                   "token_ids": token_ids}
        self.index["shards"] += 1
        atomic_json(self.path, self.index)

    def get(self, sample_id: str) -> torch.Tensor:
        entry = self.index["entries"][sample_id]
        with safe_open(self.directory / entry["shard"], framework="pt", device="cpu") as handle:
            return handle.get_tensor(sample_id)

    def require(self, rows: list[dict]):
        missing = [row["id"] for row in rows if row["id"] not in self]
        if missing:
            raise ValueError(f"Cache incomplet : {self.directory}, {len(missing)} entrées manquantes.")


def alignment_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred, gold = predicted.float(), target.float()
    return {
        "relative_mse": ((pred-gold).square().mean()/gold.square().mean().clamp_min(1e-8)).item(),
        "cosine": F.cosine_similarity(pred, gold, dim=-1).mean().item(),
        "norm_ratio": (pred.norm(dim=-1).mean()/gold.norm(dim=-1).mean().clamp_min(1e-8)).item(),
    }
