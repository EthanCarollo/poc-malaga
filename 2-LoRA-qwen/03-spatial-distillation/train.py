#!/usr/bin/env python3
"""Train a contextual projector, then optional diffuser LoRA, against spatial targets."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
import torch
from torch.nn import functional as F
from sharm_image21.core import (FeatureCache, Projector, atomic_json, digest, file_hash, read_jsonl, seed_all)
from sharm_image21.diffusion import load_transformer, predict_flow


def relative_mse(predicted, target):
    return (predicted.float()-target.float()).square().mean()/target.float().square().mean().clamp_min(1e-8)


def embedding_loss(predicted, target):
    return relative_mse(predicted, target)+0.1*(1-F.cosine_similarity(predicted.float(), target.float(), dim=-1).mean())


def load_adapter_state(model, state):
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, state, adapter_name="default")


def save_candidate(projector, diffuser, destination, metadata):
    projector.save(destination / "projector", metadata, storage_dtype=torch.bfloat16)
    config = {**metadata, "projector_sha256": file_hash(destination / "projector/model.safetensors"),
              "status": "candidate_requires_spatial_review", "lora_sha256": None}
    if diffuser is not None and any(p.requires_grad for p in diffuser.parameters()):
        from diffusers import QwenImage21Pipeline
        from diffusers.training_utils import _collate_lora_metadata
        from peft.utils import get_peft_model_state_dict
        folder = destination / "lora"
        QwenImage21Pipeline.save_lora_weights(folder, transformer_lora_layers=get_peft_model_state_dict(diffuser),
                                              **_collate_lora_metadata({"transformer": diffuser}))
        config["lora_sha256"] = file_hash(folder / "pytorch_lora_weights.safetensors")
    atomic_json(destination / "candidate.json", config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("warmup", "flow"))
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/spatial/dataset")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache/spatial")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-projector", type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--context-layers", type=int, default=2)
    parser.add_argument("--context-heads", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-learning-rate", type=float, default=5e-5)
    parser.add_argument("--distillation-weight", type=float, default=0.25)
    parser.add_argument("--embedding-weight", type=float, default=0.05)
    parser.add_argument("--layout-dropout", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--validation-limit", type=int, default=64)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--checkpointing-mode", choices=("all", "alternating", "none"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="CPU autorisé pour le warmup sur features déjà extraites ; flow exige CUDA.")
    args = parser.parse_args()
    if min(args.steps, args.accumulation, args.eval_every, args.validation_limit, args.patience) < 1:
        raise ValueError("Durées, accumulation et tailles de validation positives requises.")
    if not 0 <= args.layout_dropout <= 1 or min(args.distillation_weight, args.embedding_weight, args.lora_rank) < 0:
        raise ValueError("Poids de pertes/rang positifs et dropout entre 0 et 1 requis.")
    if min(args.learning_rate, args.lora_learning_rate) <= 0 or (args.phase == "warmup" and args.lora_rank):
        raise ValueError("LR positifs ; le LoRA est réservé à la phase flow.")
    if args.device == "cpu" and args.phase == "flow":
        raise ValueError("Le flow NF4 exige CUDA.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requis ; les contrôles CPU sont dans tests/test_spatial.py.")
    device = torch.device(args.device)
    def amp():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    seed_all(args.seed)
    rows = read_jsonl(args.dataset / "scenes.jsonl")
    dataset_manifest = json.loads((args.dataset / "manifest.json").read_text())
    if digest(rows) != dataset_manifest["dataset_sha256"]:
        raise ValueError("Dataset modifié après préparation.")
    train = [r for r in rows if r["split"] == "train"]
    valid = [r for r in rows if r["split"] == "validation"]
    random.Random(args.seed).shuffle(valid)
    valid = valid[:args.validation_limit]
    if not train or not valid:
        raise ValueError("Entraînement et validation distincts requis.")
    student, teacher = (FeatureCache(args.cache / k) for k in ("student", "teacher"))
    corpus_hash = digest(read_jsonl(args.dataset / "corpus.jsonl"))
    for cache in (student, teacher):
        if cache.index["identity"]["corpus"] != corpus_hash:
            raise ValueError("Features d'un autre corpus.")
        cache.require([{"id": key} for r in train+valid for key in (r["caption_id"], r.get("caption_plain_id", r["caption_id"]))])
    for row in train+valid:
        for key in (row["caption_id"], row.get("caption_plain_id", row["caption_id"])):
            if student.index["entries"][key]["token_ids"] != teacher.index["entries"][key]["token_ids"]:
                raise ValueError(f"Tokens désalignés : {key}")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    identity = {"dataset_sha256": digest(rows), "student": student.index["identity"], "teacher": teacher.index["identity"],
                "student_index": file_hash(student.path), "teacher_index": file_hash(teacher.path),
                "init_projector": file_hash(args.init_projector / "model.safetensors") if args.init_projector else None,
                "trainer_sha256": file_hash(Path(__file__)),
                "core_sha256": file_hash(ROOT / "common/sharm_image21/core.py"),
                "diffusion_sha256": file_hash(ROOT / "common/sharm_image21/diffusion.py")}
    diffuser, latents, trajectory_noisy, trajectory_flow = None, None, None, None
    trajectory_by_scene = {}
    trajectory_training_rows = []
    if args.phase == "flow":
        latents = FeatureCache(args.cache / "latents")
        if latents.index["identity"]["dataset_sha256"] != digest(rows):
            raise ValueError("Latents d'un autre dataset.")
        latents.require(train+valid)
        if args.distillation_weight:
            trajectory = json.loads((args.cache / "trajectories.json").read_text())
            if trajectory["identity"]["dataset_sha256"] != digest(rows) or trajectory["identity"]["teacher_index_sha256"] != file_hash(teacher.path):
                raise ValueError("Trajectoires d'une autre révision.")
            trajectory_noisy, trajectory_flow = (FeatureCache(args.cache / k) for k in ("trajectory-latents", "trajectory-flows"))
            for cache in (trajectory_noisy, trajectory_flow):
                if cache.index["identity"] != trajectory["identity"]:
                    raise ValueError("Identité des trajectoires incompatible.")
                cache.require(trajectory["rows"])
            for record in trajectory["rows"]:
                if "timestep" not in record:
                    raise ValueError("Cache de trajectoires incomplet.")
                trajectory_by_scene.setdefault(record["scene_id"], []).append(record)
            trajectory_training_rows = [r for r in train if r["id"] in trajectory_by_scene]
            if not trajectory_training_rows:
                raise ValueError("Aucune trajectoire d'entraînement ; les cibles de validation sont exclues des gradients.")
            identity["trajectory_identity"] = trajectory["identity"]
        identity["latent_identity"] = latents.index["identity"]
    signature = digest({"config": config, "identity": identity})
    args.output.mkdir(parents=True, exist_ok=True)
    complete_path = args.output / "training-complete.json"
    if complete_path.exists():
        if json.loads(complete_path.read_text())["signature"] != signature:
            raise ValueError("Une autre expérience occupe ce dossier.")
        print("Expérience déjà terminée.", flush=True)
        return
    if args.init_projector:
        initial = Projector.load(args.init_projector)
        sizes = {k: initial.settings[k] for k in ("input_dim", "output_dim", "bottleneck")}
        projector = Projector(**sizes, context_layers=args.context_layers, context_heads=args.context_heads, layout_enabled=True)
        result = projector.load_state_dict(initial.float().state_dict(), strict=False)
        if result.unexpected_keys:
            raise ValueError(f"Architecture initiale incompatible : {result.unexpected_keys}")
        del initial
    else:
        projector = Projector(context_layers=args.context_layers, context_heads=args.context_heads, layout_enabled=True)
    projector = projector.float().to(device)
    groups = [{"params": list(projector.parameters()), "lr": args.learning_rate, "initial_lr": args.learning_rate}]
    if args.phase == "flow":
        diffuser = load_transformer(training=True, gradient_checkpointing=args.checkpointing_mode)
        if args.lora_rank:
            from peft import LoraConfig
            diffuser.add_adapter(LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank,
                                            target_modules=["to_q", "to_k", "to_v", "to_out.0"], init_lora_weights="gaussian"))
            groups.append({"params": [p for p in diffuser.parameters() if p.requires_grad],
                           "lr": args.lora_learning_rate, "initial_lr": args.lora_learning_rate})
    if device.type == "cuda":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(groups, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    parameters = [p for group in groups for p in group["params"]]
    checkpoint = args.output / "checkpoint.pt"
    first_step, best, stale = 0, float("inf"), 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Checkpoint incompatible : choisir un autre dossier.")
        projector.load_state_dict(state["projector"])
        if state["lora"]:
            load_adapter_state(diffuser, state["lora"])
        optimizer.load_state_dict(state["optimizer"])
        first_step, best, stale = state["step"], state["best"], state["stale"]
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"])
        random.setstate(state["python_rng"])
        del state

    def features(row, use_layout):
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        key = row["caption_id"] if use_layout else row.get("caption_plain_id", row["caption_id"])
        source = student.get(key).to(device=device, dtype=dtype)
        gold = teacher.get(key).to(device=device, dtype=dtype)
        boxes = torch.tensor([o["box"] for o in row["objects"]], device=device, dtype=dtype) if use_layout else None
        predicted = projector(source, layout=boxes)
        return predicted, gold, len(source)

    @torch.no_grad()
    def evaluate():
        projector.eval()
        if diffuser is not None:
            diffuser.eval()
        scores = []
        for index, row in enumerate(valid):
            with amp():
                if args.phase == "warmup":
                    predicted, gold, length = features(row, False)
                    scores.append(embedding_loss(predicted[:length], gold).item())
                else:
                    # Both inference modes must be monitored; hold noise/t fixed.
                    clean = latents.get(row["id"])[None].cuda()
                    generator = torch.Generator(device="cuda").manual_seed(8000+index)
                    noise = torch.randn(clean.shape, device="cuda", dtype=clean.dtype, generator=generator)
                    t = torch.tensor([0.15+0.7*((index % 7)/6)], device="cuda", dtype=torch.bfloat16)
                    losses = []
                    for use_layout in (False, True):
                        predicted, _, _ = features(row, use_layout)
                        flow = predict_flow(diffuser, (1-t.view(1, 1, 1, 1, 1))*clean+t.view(1, 1, 1, 1, 1)*noise, predicted[None], t)
                        losses.append(F.mse_loss(flow.float(), (noise-clean).float()).item())
                    scores.append(sum(losses)/len(losses))
        return sum(scores)/len(scores)

    started = time.monotonic()
    last_step = first_step
    for step in range(first_step, args.steps if stale < args.patience else first_step):
        projector.train()
        if diffuser is not None:
            diffuser.train()
        optimizer.zero_grad(set_to_none=True)
        scale = min(1, (step+1)/max(1, int(args.steps*0.05)))*0.5*(1+math.cos(math.pi*step/args.steps))
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"]*scale
        totals = {"embedding": 0.0, "supervised": 0.0, "distillation": 0.0}
        for micro in range(args.accumulation):
            row = random.choice(train)
            with amp():
                projected, gold, length = features(row, args.phase == "flow" and random.random() >= args.layout_dropout)
                alignment = embedding_loss(projected[:length], gold)
                if args.phase == "warmup":
                    loss = alignment
                else:
                    clean = latents.get(row["id"])[None].cuda()
                    noise = torch.randn_like(clean)
                    t = torch.rand(1, device="cuda", dtype=torch.float32).clamp(0.01, 0.99).to(torch.bfloat16)
                    sigma = t.view(1, 1, 1, 1, 1)
                    flow = predict_flow(diffuser, (1-sigma)*clean+sigma*noise, projected[None], t)
                    supervised = F.mse_loss(flow.float(), (noise-clean).float())
                    loss = supervised+args.embedding_weight*alignment
                    totals["supervised"] += supervised.item()/args.accumulation
            if not torch.isfinite(loss):
                raise RuntimeError("Perte non finie.")
            (loss/args.accumulation).backward()
            totals["embedding"] += alignment.item()/args.accumulation
            # Separate backwards free the first diffuser graph before distillation.
            if args.phase == "flow" and args.distillation_weight:
                distill_row = random.choice(trajectory_training_rows)
                record = random.choice(trajectory_by_scene[distill_row["id"]])
                with amp():
                    projected, _, _ = features(distill_row, random.random() >= args.layout_dropout)
                    noisy = trajectory_noisy.get(record["id"])[None].cuda()
                    target = trajectory_flow.get(record["id"])[None].cuda()
                    t = torch.tensor([record["timestep"]], device="cuda", dtype=torch.bfloat16)
                    distillation = relative_mse(predict_flow(diffuser, noisy, projected[None], t), target)
                if not torch.isfinite(distillation):
                    raise RuntimeError("Distillation non finie.")
                (args.distillation_weight*distillation/args.accumulation).backward()
                totals["distillation"] += distillation.item()/args.accumulation
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        if norm == 0:
            raise RuntimeError("Aucun gradient : vérifier le passage arrière jusqu'au projecteur.")
        optimizer.step()
        last_step = step+1
        report = {"phase": args.phase, "step": last_step, "steps": args.steps, "losses": totals,
                  "seconds": time.monotonic()-started, "gpu_peak_gib": torch.cuda.max_memory_allocated()/2**30 if device.type == "cuda" else 0}
        if last_step % 5 == 0 or step == first_step:
            atomic_json(args.output / "status.json", report)
            print(json.dumps(report), flush=True)
        if last_step % args.eval_every == 0 or last_step == args.steps:
            score = evaluate()
            if not math.isfinite(score):
                raise RuntimeError("Perte de validation non finie.")
            report["numerical_validation_loss"] = score
            if score < best:
                best, stale = score, 0
                save_candidate(projector, diffuser, args.output / "best", {"signature": signature, "identity": identity,
                               "config": config, "step": last_step, "numerical_validation_loss": score})
            else:
                stale += 1
            lora_state = {}
            if args.lora_rank:
                from peft.utils import get_peft_model_state_dict
                lora_state = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(diffuser).items()}
            state = {"signature": signature, "step": last_step, "best": best, "stale": stale,
                     "projector": {k: v.detach().cpu() for k, v in projector.state_dict().items()},
                     "lora": lora_state,
                     "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None, "python_rng": random.getstate()}
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(checkpoint)
            with (args.output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report)+"\n")
            if stale >= args.patience:
                break
    atomic_json(complete_path, {"signature": signature, "config": config, "identity": identity,
                "steps_completed": last_step, "best_numerical_validation_loss": best,
                "spatial_quality_validated": False, "reason": "early_stopping" if stale >= args.patience else "steps_completed"})
    print(f"Candidat enregistré : {args.output / 'best'}. Évaluation spatiale requise.", flush=True)


if __name__ == "__main__":
    main()
