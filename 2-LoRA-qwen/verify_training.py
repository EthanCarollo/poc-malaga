#!/usr/bin/env python3
"""Small real-GPU training checks before dispatching the long jobs."""
from pathlib import Path
import argparse
import gc
import json
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
import torch
from sharm_image21.core import ARTIFACTS, FeatureCache, Projector, atomic_json, seed_all


def verify_projector():
    seed_all()
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        model = Projector(8, 16, 4).cuda()
        source = torch.randn(256, 8, device="cuda")
        target = source @ torch.randn(8, 16, device="cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
        initial = torch.nn.functional.mse_loss(model(source), target).item()
        for _ in range(150):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(source), target)
            loss.backward()
            optimizer.step()
        final = torch.nn.functional.mse_loss(model(source), target).item()
        if final > initial*0.02:
            raise RuntimeError(f"Le projecteur n'apprend pas la correspondance synthétique : {initial} -> {final}")
        model.save(folder / "projector")
        reloaded = Projector.load(folder / "projector", "cuda")
        torch.testing.assert_close(model(source), reloaded(source), rtol=0, atol=0)
        model.save(folder / "projector-bf16", storage_dtype=torch.bfloat16)
        reloaded_bf16 = Projector.load(folder / "projector-bf16", "cuda")
        reference_bf16 = model.to(torch.bfloat16)
        torch.testing.assert_close(reference_bf16(source.to(torch.bfloat16)),
                                   reloaded_bf16(source.to(torch.bfloat16)), rtol=0, atol=0)
        cache = FeatureCache(folder / "cache", {"model": "one"})
        cache.write([("sample", source[:4].cpu(), [1, 2, 3, 4])])
        torch.testing.assert_close(cache.get("sample"), source[:4].cpu())
        try:
            FeatureCache(folder / "cache", {"model": "two"})
        except ValueError:
            pass
        else:
            raise RuntimeError("Le cache accepte une identité incompatible.")
        print(json.dumps({"projector_smoke": True, "initial_loss": initial, "final_loss": final}), flush=True)


def verify_diffusion_training():
    from sharm_image21.diffusion import load_transformer, predict_flow
    from peft import LoraConfig
    model = load_transformer(training=True)
    model.add_adapter(LoraConfig(r=8, lora_alpha=8, init_lora_weights="gaussian",
                                 target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    model.train()
    params = [param for param in model.parameters() if param.requires_grad]
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(params, lr=1e-4)
    seed_all()
    clean = torch.randn(1, 64, 1, 32, 32, device="cuda", dtype=torch.bfloat16)
    embeddings = torch.randn(1, 32, 4096, device="cuda", dtype=torch.bfloat16)
    losses = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        noise = torch.randn_like(clean)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = predict_flow(model, (noise+clean)/2, embeddings,
                                      torch.tensor([0.5], device="cuda", dtype=torch.bfloat16))
        if prediction.shape != clean.shape:
            raise RuntimeError("Forme de prédiction incorrecte.")
        loss = torch.nn.functional.mse_loss(prediction.float(), (noise-clean).float())
        if not torch.isfinite(loss):
            raise RuntimeError("Perte du diffuseur non finie.")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        if norm == 0:
            raise RuntimeError("Aucun gradient LoRA.")
        optimizer.step()
        losses.append(loss.item())
    report = {"passed": True, "lora_backward": True, "losses": losses,
              "trainable_parameters": sum(param.numel() for param in params),
              "gpu_peak_gib": torch.cuda.max_memory_allocated()/2**30}
    with tempfile.TemporaryDirectory() as folder:
        from diffusers import QwenImage21Pipeline
        from diffusers.training_utils import _collate_lora_metadata
        from peft.utils import get_peft_model_state_dict
        from peft import set_peft_model_state_dict
        folder = Path(folder)
        QwenImage21Pipeline.save_lora_weights(folder, transformer_lora_layers=get_peft_model_state_dict(model),
                                              **_collate_lora_metadata({"transformer": model}))
        if not (folder / "pytorch_lora_weights.safetensors").exists():
            raise RuntimeError("Adaptateur non sérialisé.")
        torch.save({"adapter": get_peft_model_state_dict(model), "optimizer": optimizer.state_dict()}, folder / "state.pt")
        state = torch.load(folder / "state.pt", weights_only=True, map_location="cpu")
        set_peft_model_state_dict(model, state["adapter"], adapter_name="default")
        optimizer.load_state_dict(state["optimizer"])
        report["checkpoint_roundtrip"] = True
    atomic_json(ARTIFACTS / "diffusion-training-audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion", action="store_true")
    args = parser.parse_args()
    verify_projector()
    gc.collect()
    torch.cuda.empty_cache()
    if args.diffusion:
        verify_diffusion_training()


if __name__ == "__main__":
    main()
