#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import torch
from torch.nn import functional as F
from sharm_image21.core import ARTIFACTS, ORIGINAL, WEIGHTS, FeatureCache, atomic_json, digest, seed_all
from sharm_image21.diffusion import load_transformer, predict_flow


def save_adapter(model, folder):
    from diffusers import QwenImage21Pipeline
    from diffusers.training_utils import _collate_lora_metadata
    from peft.utils import get_peft_model_state_dict
    QwenImage21Pipeline.save_lora_weights(save_directory=folder,
        transformer_lora_layers=get_peft_model_state_dict(model), **_collate_lora_metadata({"transformer": model}))


def get_pair(row, texts, latents):
    return latents.get(row["id"]).unsqueeze(0).cuda(), texts.get(row["caption_id"]).unsqueeze(0).cuda()


@torch.no_grad()
def evaluate(model, rows, texts, latents):
    model.eval()
    losses = []
    for index, row in enumerate(rows[:16]):
        clean, embeddings = get_pair(row, texts, latents)
        noise = torch.randn(clean.shape, device="cuda", dtype=clean.dtype,
                            generator=torch.Generator(device="cuda").manual_seed(8400+index))
        noisy = 0.5*clean+0.5*noise
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = predict_flow(model, noisy, embeddings, torch.tensor([0.5], device="cuda", dtype=torch.bfloat16))
        losses.append(F.mse_loss(predicted.float(), (noise-clean).float()).item())
    return sum(losses)/len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--checkpointing-mode", choices=("all", "alternating", "none"), default="alternating")
    parser.add_argument("--output", type=Path, default=WEIGHTS / "sharm-image-2.1-sprites-128-lora")
    args = parser.parse_args()
    if args.gradient_checkpointing:
        args.checkpointing_mode = "all"
    if min(args.steps, args.accumulation, args.rank, args.checkpoint_every) < 1 or args.learning_rate <= 0:
        raise ValueError("Pas, accumulation, rang et fréquence de checkpoint doivent être positifs.")
    seed_all(42)
    from diffusers import FlowMatchEulerDiscreteScheduler
    from peft import LoraConfig
    from peft import set_peft_model_state_dict
    import bitsandbytes as bnb
    from safetensors.torch import load_file

    data = json.loads((ARTIFACTS / f"sprite-training-{args.resolution}.json").read_text())
    validation = json.loads((ARTIFACTS / "base-validation.json").read_text())
    if not validation.get("passed") or validation["projector_sha256"] != data["identity"]["projector_sha256"]:
        raise RuntimeError("Base validée et cache sprites de révisions différentes.")
    rows = data["rows"]
    training_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    if not training_rows or not validation_rows:
        raise ValueError("Un holdout par espèce est requis.")
    assert not ({r['species_id'] for r in training_rows} & {r['species_id'] for r in validation_rows})
    texts, latents = FeatureCache(Path(data["text_cache"])), FeatureCache(Path(data["latent_cache"]))
    run_config = {**vars(args), "output": str(args.output), "cache_identity": data["identity"]}
    # Activation recomputation changes memory/time, not the training examples or
    # objective. Keep checkpoints compatible when toggling this runtime option.
    run_signature = digest({key: value for key, value in run_config.items()
                            if key not in ("gradient_checkpointing", "checkpointing_mode")})
    complete_path = args.output / "training-complete.json"
    if complete_path.exists():
        previous = json.loads(complete_path.read_text())
        if previous["run_signature"] != run_signature:
            raise RuntimeError("Un LoRA différent existe ; choisir un autre dossier de sortie.")
        print("LoRA déjà terminé.", flush=True)
        return
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    model = load_transformer(training=True, gradient_checkpointing=args.checkpointing_mode)
    model.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                                 target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(parameters, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ORIGINAL, subfolder="scheduler", local_files_only=True)
    timesteps = scheduler.timesteps.to("cuda")
    sigmas = scheduler.sigmas.to(device="cuda", dtype=torch.bfloat16)
    checkpoint_pointer = args.output / "latest-checkpoint.json"
    first_step = 0
    if checkpoint_pointer.exists():
        folder = args.output / json.loads(checkpoint_pointer.read_text())["folder"]
        state = torch.load(folder / "training-state.pt", weights_only=True, map_location="cpu")
        if state["run_signature"] != run_signature:
            raise ValueError("Checkpoint LoRA incompatible avec cette configuration.")
        set_peft_model_state_dict(model, load_file(folder / "adapter-state.safetensors"), adapter_name="default")
        optimizer.load_state_dict(state["optimizer"])
        first_step = state["step"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        random.setstate(state["python_rng"])
    baseline = evaluate(model, validation_rows, texts, latents)
    started = time.monotonic()
    for step in range(first_step, args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for micro in range(args.accumulation):
            row = random.choice(training_rows)
            clean, embeddings = get_pair(row, texts, latents)
            noise = torch.randn_like(clean)
            index = torch.randint(len(timesteps), (1,), device="cuda")
            sigma = sigmas[index].view(1, 1, 1, 1, 1)
            noisy = ((1-sigma)*clean+sigma*noise).to(torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = predict_flow(model, noisy, embeddings, timesteps[index]/1000)
            loss = F.mse_loss(predicted.float(), (noise-clean).float())
            if not torch.isfinite(loss):
                raise RuntimeError("Perte LoRA non finie.")
            (loss/args.accumulation).backward()
            total_loss += loss.item()/args.accumulation
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        if grad_norm == 0:
            raise RuntimeError("Aucun gradient LoRA ; vérifier le checkpointing et les adaptateurs.")
        optimizer.step()
        report = {"stage": "train_lora", "step": step+1, "steps": args.steps,
                  "loss": total_loss, "seconds": time.monotonic()-started,
                  "updated_at": time.time(), "gpu_peak_gib": torch.cuda.max_memory_allocated()/2**30}
        if (step+1) % 5 == 0 or step == first_step:
            atomic_json(ARTIFACTS / "lora-status.json", report)
            print(json.dumps(report), flush=True)
        if (step+1) % args.checkpoint_every == 0 or step+1 == args.steps:
            report["validation_flow_mse"] = evaluate(model, validation_rows, texts, latents)
            folder = args.output / f"checkpoint-{step+1}-{uuid.uuid4().hex[:8]}"
            staging = folder.with_name(folder.name + ".staging")
            save_adapter(model, staging)
            from peft.utils import get_peft_model_state_dict
            from safetensors.torch import save_file
            save_file({name: value.detach().cpu().contiguous() for name, value in get_peft_model_state_dict(model).items()},
                       staging / "adapter-state.safetensors")
            torch.save({"run_signature": run_signature, "step": step+1, "optimizer": optimizer.state_dict(),
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                        "python_rng": random.getstate()}, staging / "training-state.pt")
            staging.replace(folder)
            atomic_json(checkpoint_pointer, {"folder": folder.name, "step": step+1, "run_signature": run_signature})
            with (args.output / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(report)+"\n")
    save_adapter(model, args.output)
    atomic_json(complete_path, {"run_signature": run_signature, "config": run_config,
                                "baseline_validation_flow_mse": baseline, "finished_at": time.time(),
                                "hf_repo_intent": "ethanzxv/Sharm-Image-2.1-Sprites-128-LoRA",
                                "private": True, "published": False})
    print(f"LoRA terminé : {args.output}", flush=True)


if __name__ == "__main__":
    main()
