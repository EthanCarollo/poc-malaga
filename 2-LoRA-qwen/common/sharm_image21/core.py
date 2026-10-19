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
    def __init__(self, input_dim: int = 2560, output_dim: int = 4096, bottleneck: int = 1024,
                 context_layers: int = 0, context_heads: int = 8, layout_enabled: bool = False):
        super().__init__()
        self.settings = dict(input_dim=input_dim, output_dim=output_dim, bottleneck=bottleneck)
        if context_layers < 0 or context_heads < 1 or (context_layers and bottleneck % context_heads):
            raise ValueError("Largeur d'attention divisible par le nombre de têtes requise.")
        # Preserve the v1 config and state keys when no spatial extension is used.
        if context_layers or layout_enabled:
            self.settings.update(context_layers=context_layers, context_heads=context_heads,
                                 layout_enabled=layout_enabled)
        self.skip = nn.Linear(input_dim, output_dim)
        self.norm = nn.RMSNorm(input_dim, eps=1e-6)
        self.up = nn.Linear(input_dim, bottleneck)
        self.down = nn.Linear(bottleneck, output_dim)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        self.context = nn.ModuleList([
            nn.TransformerEncoderLayer(bottleneck, context_heads, 4*bottleneck, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(context_layers)])
        if layout_enabled:
            self.layout_box = nn.Sequential(nn.Linear(4, bottleneck), nn.GELU(), nn.Linear(bottleneck, bottleneck))
            self.layout_object = nn.Embedding(16, bottleneck)

    def forward(self, hidden, attention_mask=None, layout=None):
        """Project ordered tokens; optional normalized xyxy boxes follow object order.

        The legacy token-wise path keeps identical arithmetic. In the contextual
        path, valid tokens must stay in order (no random token subsampling).
        """
        if not self.context and layout is None:
            return self.skip(hidden) + self.down(F.gelu(self.up(self.norm(hidden))))
        unbatched = hidden.ndim == 2
        hidden = hidden.unsqueeze(0) if unbatched else hidden
        if hidden.ndim != 3:
            raise ValueError("États attendus : [tokens, dimensions] ou [lot, tokens, dimensions].")
        values = self.up(self.norm(hidden))
        text_length = values.shape[1]
        padding = None
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask
            if mask.shape != hidden.shape[:2] or not mask.bool().any(dim=1).all():
                raise ValueError("Masque de tokens invalide ou séquence entièrement masquée.")
            padding = ~mask.to(device=hidden.device, dtype=torch.bool)
        if layout is not None:
            if not self.settings.get("layout_enabled"):
                raise ValueError("Ce projecteur n'a pas été entraîné avec des boîtes spatiales.")
            boxes = layout.unsqueeze(0) if layout.ndim == 2 else layout
            if boxes.ndim != 3 or boxes.shape[0] != hidden.shape[0] or boxes.shape[-1] != 4 or not 1 <= boxes.shape[1] <= 16:
                raise ValueError("De 1 à 16 boîtes xyxy par prompt sont requises.")
            if not torch.isfinite(boxes).all() or (boxes < 0).any() or (boxes > 1).any() or (boxes[..., 2:] <= boxes[..., :2]).any():
                raise ValueError("Boîtes xyxy normalisées non vides requises.")
            slots = torch.arange(boxes.shape[1], device=hidden.device)
            layout_values = self.layout_box(boxes.to(device=hidden.device, dtype=values.dtype)) + self.layout_object(slots)[None].to(values.dtype)
            values = torch.cat((values, layout_values), dim=1)
            if padding is not None:
                padding = F.pad(padding, (0, boxes.shape[1]), value=False)
        width = values.shape[-1]
        positions = torch.arange(values.shape[1], device=values.device).float()[:, None]
        frequencies = torch.exp(torch.arange(0, width, 2, device=values.device).float()*(-9.210340372/width))
        angles = positions*frequencies
        positional = torch.zeros(values.shape[1], width, device=values.device)
        positional[:, 0::2], positional[:, 1::2] = angles.sin(), angles[:, :width//2].cos()
        values = values + positional.to(values.dtype)
        for layer in self.context:
            values = layer(values, src_key_padding_mask=padding)
        text = self.skip(hidden) + self.down(F.gelu(values[:, :text_length]))
        if values.shape[1] > text_length:
            text = torch.cat((text, self.down(F.gelu(values[:, text_length:]))), dim=1)
        return text[0] if unbatched else text

    def save(self, path: Path, metadata: dict | None = None, storage_dtype=None):
        path.mkdir(parents=True, exist_ok=True)
        atomic_json(path / "config.json", {**self.settings, "convention": FEATURE_CONVENTION, **(metadata or {})})
        temporary = path / "model.tmp.safetensors"
        save_file({key: val.detach().to(device="cpu", dtype=storage_dtype or val.dtype).contiguous()
                   for key, val in self.state_dict().items()}, temporary)
        os.replace(temporary, path / "model.safetensors")

    @classmethod
    def load(cls, path: Path, device="cpu"):
        config = json.loads((path / "config.json").read_text())
        if config.get("convention") != FEATURE_CONVENTION:
            raise ValueError("Convention d'embeddings du projecteur incompatible.")
        state = load_file(path / "model.safetensors")
        keys = ("input_dim", "output_dim", "bottleneck", "context_layers", "context_heads", "layout_enabled")
        model = cls(**{key: config[key] for key in keys if key in config}).to(state["skip.weight"].dtype)
        model.load_state_dict(state)
        return model.to(device)


class FrozenEncoder:
    """Extract the norm INPUT without LM logits, past-key-values or all-layer captures."""

    def __init__(self, kind: str, gpu_memory: str = "auto", model_dir: Path | None = None,
                 processor_dir: Path | None = None):
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        if kind not in ("teacher", "student"):
            raise ValueError(kind)
        self.kind = kind
        self.model_dir = model_dir or (STUDENT if kind == "student" else ORIGINAL / "text_encoder")
        processor_dir = processor_dir or (self.model_dir if kind == "student" else ORIGINAL / "processor")
        self.processor = Qwen3VLProcessor.from_pretrained(processor_dir, local_files_only=True)
        self.identity = model_identity(kind, self.model_dir)
        sys_message = [{"role": "system", "content": [{"type": "text", "text": SYS_PROMPT}]}]
        self.drop_idx = len(self.processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)[0])
        kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
        if kind == "student":
            os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")
            # This extractor calls model.model with text only: visual and lm_head
            # are never executed. Keep their weights in RAM to leave room for
            # FP8 prefill on an 8 GB GPU; all language layers stay on CUDA.
            kwargs["device_map"] = {"model.language_model": "cuda", "model.visual": "cpu", "lm_head": "cpu"}
        else:
            import psutil
            free_gpu, _ = torch.cuda.mem_get_info()
            gpu_budget = max(512*2**20, free_gpu-2*2**30) if gpu_memory == "auto" else gpu_memory
            cpu_budget = max(1*2**30, psutil.virtual_memory().available-2*2**30)
            offload = CACHE / "teacher-offload"
            offload.mkdir(parents=True, exist_ok=True)
            kwargs.update(device_map="auto", max_memory={0: gpu_budget, "cpu": cpu_budget},
                          offload_folder=str(offload), offload_state_dict=True)
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
                    if name.startswith("model.language_model.") and module.weight.device.type != "cuda":
                        raise RuntimeError(f"Couche FP8 textuelle hors CUDA : {name}")
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
            # With CPU/disk offload, the final hidden states can live on a
            # different device from the token IDs and their attention mask.
            values = row[mask.to(row.device)][self.drop_idx:].to(device="cpu", dtype=torch.bfloat16).contiguous()
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
