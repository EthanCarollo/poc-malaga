from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from .core import FrozenEncoder, Projector, file_hash
from .diffusion import load_diffuser_pipeline


class SharmImage21Pipeline:
    """Sequential FP8 conditioning and NF4 diffusion for a 16 GiB GPU.

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
        if not self.config.get("base_validation_passed"):
            raise ValueError("Base Sharm non validée.")

    @classmethod
    def from_pretrained(cls, path_or_repo, revision=None, token=None):
        directory = Path(path_or_repo)
        if not directory.is_dir():
            from huggingface_hub import snapshot_download
            directory = Path(snapshot_download(str(path_or_repo), revision=revision, token=token))
        return cls(directory)

    @torch.inference_mode()
    def encode_prompt(self, prompt):
        prompts = [prompt] if isinstance(prompt, str) else prompt
        encoder = FrozenEncoder("student", model_dir=self.directory / "text_encoder",
                                processor_dir=self.directory / "processor")
        if encoder.identity != self.config["student_identity"]:
            raise ValueError("Encodeur différent de celui utilisé pour entraîner le projecteur.")
        model = Projector.load(self.directory / "projector", "cuda").eval()
        output = []
        for hidden, _ in encoder.encode(prompts):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output.append(model(hidden.cuda()).to(torch.bfloat16).cpu())
        del model, encoder
        gc.collect()
        torch.cuda.empty_cache()
        embeddings = pad_sequence(output, batch_first=True)
        mask = torch.arange(embeddings.shape[1])[None, :] < torch.tensor([len(x) for x in output])[:, None]
        return embeddings, mask

    def __call__(self, prompt, *, lora_dir=None, image=None, **kwargs):
        if image is not None:
            raise ValueError("L'édition par image n'est pas validée pour cette première base Sharm.")
        embeddings, mask = self.encode_prompt(prompt)
        negative = kwargs.pop("negative_prompt", None)
        if negative is not None and kwargs.get("true_cfg_scale", 1.0) > 1:
            negative_embeddings, negative_mask = self.encode_prompt(negative)
            kwargs.update(negative_prompt_embeds=negative_embeddings.cuda(), negative_prompt_embeds_mask=negative_mask.cuda())
        pipeline = load_diffuser_pipeline(self.directory)
        if lora_dir is not None:
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
