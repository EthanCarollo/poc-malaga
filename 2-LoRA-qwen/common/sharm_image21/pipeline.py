from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from .core import FrozenEncoder, Projector, file_hash
from .diffusion import load_diffuser_pipeline
from .image_io import save_png as _save_png
from .spatial_data import layout_prompt


class SharmImage21Pipeline:
    """Sequential FP8 conditioning and NF4 diffusion, with optional spatial control.

    This first training run validates text-to-image only. Reference-image editing
    must receive its own multimodal alignment and evaluation before enabling it.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self.config = json.loads((directory / "sharm_config.json").read_text())
        if self.config["encoder_format"] != "fp8-e4m3" or not self.config["encoder_frozen"]:
            raise ValueError("Encodeur Sharm gelé en FP8 requis.")
        if file_hash(directory / "projector/model.safetensors") != self.config["projector_sha256"]:
            raise ValueError("Empreinte du projecteur incorrecte.")
        if self.config.get("pipeline_format") == "spatial-v1":
            if not self.config.get("training_complete"):
                raise ValueError("Entraînement spatial incomplet.")
            lora_sha = self.config.get("spatial_lora_sha256")
            if lora_sha and file_hash(directory / "lora/pytorch_lora_weights.safetensors") != lora_sha:
                raise ValueError("Empreinte du LoRA spatial incorrecte.")
        elif not self.config.get("base_validation_passed"):
            raise ValueError("Base Sharm non validée.")

    @classmethod
    def from_pretrained(cls, path_or_repo, revision=None, token=None):
        directory = Path(path_or_repo)
        if not directory.is_dir():
            from huggingface_hub import snapshot_download
            directory = Path(snapshot_download(str(path_or_repo), revision=revision, token=token))
        return cls(directory)

    @torch.inference_mode()
    def encode_prompt(self, prompt, layout=None):
        prompts = [prompt] if isinstance(prompt, str) else prompt
        if not isinstance(prompts, (list, tuple)) or not prompts or not all(isinstance(p, str) for p in prompts):
            raise ValueError("Un prompt ou une liste non vide de prompts est requis.")
        boxes = None
        if layout is not None:
            if len(prompts) != 1:
                raise ValueError("Le contrôle par boîtes attend un seul prompt.")
            text, boxes = layout_prompt(prompts[0], layout)
            prompts = [text]
        encoder = FrozenEncoder("student", model_dir=self.directory / "text_encoder",
                                processor_dir=self.directory / "processor")
        if encoder.identity != self.config["student_identity"]:
            raise ValueError("Encodeur différent de celui utilisé pour entraîner le projecteur.")
        features = encoder.encode(prompts)
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        model = Projector.load(self.directory / "projector", "cuda").eval()
        if boxes is not None:
            boxes = torch.tensor(boxes, device="cuda", dtype=torch.bfloat16)
        output = []
        for hidden, _ in features:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output.append(model(hidden.cuda(), layout=boxes).to(torch.bfloat16).cpu())
        del model, features
        gc.collect()
        torch.cuda.empty_cache()
        embeddings = pad_sequence(output, batch_first=True)
        mask = torch.arange(embeddings.shape[1])[None, :] < torch.tensor([len(x) for x in output])[:, None]
        return embeddings, mask

    def __call__(self, prompt, *, layout=None, lora_dir=None, image=None, **kwargs):
        if image is not None:
            raise ValueError("L'édition par image n'est pas validée pour cette première base Sharm.")
        embeddings, mask = self.encode_prompt(prompt, layout=layout)
        negative = kwargs.pop("negative_prompt", None)
        if negative is not None and kwargs.get("true_cfg_scale", 1.0) > 1:
            negative_embeddings, negative_mask = self.encode_prompt(negative)
            kwargs.update(negative_prompt_embeds=negative_embeddings.cuda(), negative_prompt_embeds_mask=negative_mask.cuda())
        pipeline = load_diffuser_pipeline(self.directory)
        if self.config.get("pipeline_format") == "spatial-v1" and self.config.get("spatial_lora_sha256"):
            if lora_dir is not None:
                raise ValueError("Le candidat spatial utilise déjà son LoRA entraîné.")
            pipeline.load_lora_weights(self.directory / "lora", adapter_name="spatial")
        elif lora_dir is not None:
            lora_dir = Path(lora_dir)
            metadata = json.loads((lora_dir / "sharm_lora_config.json").read_text())
            if metadata["projector_sha256"] != self.config["projector_sha256"]:
                raise ValueError("Ce LoRA a été entraîné avec un autre projecteur.")
            if file_hash(lora_dir / "pytorch_lora_weights.safetensors") != metadata["lora_sha256"]:
                raise ValueError("Empreinte des poids LoRA incorrecte.")
            if metadata.get("base_diffusion_revision", self.config["base_revision"]) != self.config["base_revision"]:
                raise ValueError("Ce LoRA vise une autre révision du diffuseur.")
            pipeline.load_lora_weights(lora_dir, adapter_name="sharm_sprites")
        try:
            return pipeline(prompt_embeds=embeddings.cuda(), prompt_embeds_mask=mask.cuda(), **kwargs)
        finally:
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()

    def generate_png(self, prompt, path, *, preview_path=None, transparent_background=True,
                     **kwargs):
        """Generate one RGBA PNG, optionally with a checkerboard preview beside it.

        Transparent output is requested in the prompt and checked before the PNG
        is written. The regular ``__call__`` API remains available for batched or
        non-PNG inference.
        """
        if kwargs.get("output_type", "pil") != "pil":
            raise ValueError("generate_png requiert output_type='pil'.")
        if kwargs.get("num_images_per_prompt", 1) != 1:
            raise ValueError("generate_png attend une seule image par prompt.")
        if transparent_background:
            instruction = (
                "The background is fully transparent; only the foreground subject "
                "is visible, with transparent pixels outside the subject."
            )
            if isinstance(prompt, str):
                prompt = f"{prompt.rstrip()} {instruction}"
            elif isinstance(prompt, (list, tuple)) and all(isinstance(item, str) for item in prompt):
                if len(prompt) != 1:
                    raise ValueError("generate_png attend un seul prompt.")
                prompt = [f"{item.rstrip()} {instruction}" for item in prompt]
            else:
                raise TypeError("Le prompt doit être une chaîne ou une liste de chaînes.")
        elif isinstance(prompt, (list, tuple)) and len(prompt) != 1:
            raise ValueError("generate_png attend un seul prompt.")
        elif not isinstance(prompt, str) and not isinstance(prompt, (list, tuple)):
            raise TypeError("Le prompt doit être une chaîne ou une liste de chaînes.")

        output = self(prompt, **kwargs)
        images = getattr(output, "images", None)
        if images is None or len(images) != 1:
            raise ValueError("generate_png attend une seule image PIL (output_type='pil').")
        return _save_png(images[0], path, preview_path=preview_path,
                         require_transparency=transparent_background)

    @staticmethod
    def save_png(image, path, *, preview_path=None, require_transparency=False):
        """Save an existing result as PNG and optionally create its card preview."""
        return _save_png(image, path, preview_path=preview_path,
                         require_transparency=require_transparency)
