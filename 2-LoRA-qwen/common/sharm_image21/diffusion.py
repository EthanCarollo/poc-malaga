"""Diffuser-only loading: never constructs the 8B text encoder."""
from pathlib import Path

import torch
from .core import ORIGINAL, STUDENT


def load_transformer(base: Path = ORIGINAL, training: bool = False):
    from diffusers import BitsAndBytesConfig, QwenImage21Transformer2DModel
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = QwenImage21Transformer2DModel.from_pretrained(
        base, subfolder="transformer", quantization_config=quantization,
        torch_dtype=torch.bfloat16, local_files_only=True)
    model.requires_grad_(False)
    model.to("cuda")
    if training:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model.enable_gradient_checkpointing()
    else:
        model.eval()
    return model


def load_diffuser_pipeline(base: Path = ORIGINAL, transformer=None):
    from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
    from transformers import Qwen3VLProcessor
    if transformer is None:
        transformer = load_transformer(base)
    processor_path = base / "processor" if (base / "sharm_config.json").exists() else STUDENT
    processor = Qwen3VLProcessor.from_pretrained(processor_path, local_files_only=True)
    vae = AutoencoderKLQwenImage21.from_pretrained(base, subfolder="vae", torch_dtype=torch.bfloat16,
                                                  local_files_only=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base, subfolder="scheduler", local_files_only=True)
    pipeline = QwenImage21Pipeline(vae=vae, text_encoder=None, processor=processor,
                                  transformer=transformer, scheduler=scheduler)
    pipeline.to("cuda")
    return pipeline


def predict_flow(transformer, latents, embeddings, timestep):
    from diffusers import QwenImage21Pipeline
    batch, channels, frames, height, width = latents.shape
    if frames != 1 or embeddings.shape[0] != batch:
        raise ValueError("Batch ou nombre de frames incompatible avec le trainer sprites.")
    packed = QwenImage21Pipeline._pack_latents(latents, batch, channels, height, width)
    image_mask = torch.cat([
        torch.zeros(batch, embeddings.shape[1], dtype=torch.bool, device=latents.device),
        torch.ones(batch, height*width//4, dtype=torch.bool, device=latents.device)], dim=1)
    result = transformer(hidden_states=packed, encoder_hidden_states=embeddings,
                         encoder_hidden_states_mask=None, timestep=timestep,
                         img_shapes=[[(1, height, width)]]*batch, img_mask=image_mask, return_dict=False)[0]
    result = result[:, -packed.shape[1]:]
    return QwenImage21Pipeline._unpack_latents(result, height*16, width*16, 16)
