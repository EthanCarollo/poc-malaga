#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import torch
from torch.nn import functional as F
from sharm_image21.core import (ARTIFACTS, CACHE, FeatureCache, Projector, alignment_metrics,
                                atomic_json, digest, read_jsonl, seed_all)


def paired_features(row, student, teacher):
    key = row["id"]
    if student.index["entries"][key]["token_ids"] != teacher.index["entries"][key]["token_ids"]:
        raise ValueError(f"Tokens désalignés pour {key}")
    source, target = student.get(key), teacher.get(key)
    if source.shape[0] != target.shape[0]:
        raise ValueError(f"Longueurs désalignées pour {key}")
    return source, target


@torch.no_grad()
def evaluate(model, rows, student, teacher):
    model.eval()
    scores = []
    for row in rows:
        source, target = paired_features(row, student, teacher)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(source.cuda())
        scores.append(alignment_metrics(prediction, target.cuda()))
    return {name: sum(score[name] for score in scores)/len(scores) for name in scores[0]}


def train_batch(rows, student, teacher, token_limit):
    inputs, targets = [], []
    for row in rows:
        source, target = paired_features(row, student, teacher)
        if len(source) > token_limit:
            selected = torch.randperm(len(source))[:token_limit]
            source, target = source[selected], target[selected]
        inputs.append(source)
        targets.append(target)
    return torch.cat(inputs).cuda(), torch.cat(targets).cuda()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=ARTIFACTS / "corpus.jsonl")
    parser.add_argument("--linear-steps", type=int, default=1000)
    parser.add_argument("--mlp-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tokens-per-caption", type=int, default=96)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    if min(args.linear_steps, args.mlp_steps, args.batch_size, args.tokens_per_caption, args.eval_every) < 1:
        raise ValueError("Les tailles de lots et durées d'entraînement doivent être positives.")
    seed_all(args.seed)
    torch.set_num_threads(4)
    corpus = read_jsonl(args.corpus)
    rows = corpus[:args.limit] if args.limit else corpus
    args.output_root.mkdir(parents=True, exist_ok=True)
    student, teacher = FeatureCache(CACHE / "student"), FeatureCache(CACHE / "teacher")
    student.require(rows)
    teacher.require(rows)
    for cache in (student, teacher):
        if cache.index["identity"]["corpus"] != digest(corpus):
            raise ValueError("Le cache ne correspond pas au corpus d'entraînement.")
    train = [row for row in rows if row["split"] == "train"]
    valid = [row for row in rows if row["split"] == "validation"]
    if not train or not valid:
        raise ValueError("Ensembles entraînement et validation requis.")
    random.Random(args.seed).shuffle(valid)
    monitor = valid[:256]
    metadata = {"teacher": teacher.index["identity"], "student": student.index["identity"],
                "corpus_sha256": digest(corpus), "training_subset_sha256": digest(rows),
                "scope": "experimental text-to-image alignment; editing not validated"}
    destination = args.output_root / "projector"
    linear_destination = args.output_root / "projector-linear"
    if (destination / "training-complete.json").is_file():
        previous = json.loads((destination / "training-complete.json").read_text())
        if (previous["corpus_sha256"] != metadata["corpus_sha256"]
                or previous.get("training_subset_sha256") != metadata["training_subset_sha256"]):
            raise ValueError("Le projecteur existant correspond à un autre corpus.")
        print("Entraînement du projecteur déjà terminé.", flush=True)
        return
    model = Projector().cuda()
    checkpoint = args.output_root / "projector-checkpoint.pt"
    state = torch.load(checkpoint, weights_only=True, map_location="cpu") if checkpoint.exists() else None
    if state and state["metadata"] != metadata:
        raise ValueError("Checkpoint projecteur incompatible.")
    if state:
        model.load_state_dict(state["model"])
    global_best = state["global_best"] if state else float("inf")
    history = args.output_root / "projector-history.jsonl"
    started = time.monotonic()
    for phase, steps in (("linear", args.linear_steps), ("mlp", args.mlp_steps)):
        if state and state["phase"] == "mlp" and phase == "linear":
            continue
        if phase == "mlp" and (linear_destination / "model.safetensors").exists() and not (state and state["phase"] == "mlp"):
            model = Projector.load(linear_destination, "cuda")
        for name, param in model.named_parameters():
            param.requires_grad_(phase == "mlp" or name.startswith("skip."))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01)
        start_step = 0
        if state and state["phase"] == phase:
            optimizer.load_state_dict(state["optimizer"])
            start_step = state["step"]
            torch.set_rng_state(state["torch_rng"])
            random.setstate(state["python_rng"])
        phase_best = state.get("phase_best", float("inf")) if state and state["phase"] == phase else float("inf")
        for step in range(start_step, steps):
            model.train()
            source, target = train_batch(random.sample(train, min(args.batch_size, len(train))), student, teacher,
                                         args.tokens_per_caption)
            lr_scale = min(1.0, (step+1)/max(1, int(steps*0.03)))
            optimizer.param_groups[0]["lr"] = 3e-4*lr_scale
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = model(source)
            pred, gold = predicted.float(), target.float()
            relative_mse = (pred-gold).square().mean()/gold.square().mean().clamp_min(1e-8)
            cosine_loss = 1-F.cosine_similarity(pred, gold, dim=-1).mean()
            loss = relative_mse + 0.1*cosine_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Perte projecteur non finie ; arrêt avant le LoRA.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if (step+1) % 25 == 0:
                print(f"projector {phase} {step+1}/{steps}: loss={loss.item():.5f}", flush=True)
            if (step+1) % args.eval_every == 0 or step+1 == steps:
                metrics = evaluate(model, monitor, student, teacher)
                report = {"stage": "train_projector", "phase": phase, "step": step+1, "steps": steps,
                          "metrics": metrics, "seconds": time.monotonic()-started, "updated_at": time.time()}
                with history.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(report)+"\n")
                atomic_json(args.output_root / "projector-status.json", report)
                if metrics["relative_mse"] < phase_best:
                    phase_best = metrics["relative_mse"]
                    model.save(linear_destination if phase == "linear" else args.output_root / "projector-mlp", metadata)
                if metrics["relative_mse"] < global_best:
                    global_best = metrics["relative_mse"]
                    model.save(destination, {**metadata, "selected_phase": phase, "validation": metrics})
                temporary = checkpoint.with_suffix(".tmp")
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "phase": phase, "step": step+1, "global_best": global_best, "phase_best": phase_best,
                            "metadata": metadata, "torch_rng": torch.get_rng_state(), "python_rng": random.getstate()}, temporary)
                temporary.replace(checkpoint)
                print(json.dumps(report), flush=True)
        state = None
    best = Projector.load(destination, "cuda")
    selected_config = json.loads((destination / "config.json").read_text())
    best.save(destination, selected_config, storage_dtype=torch.bfloat16)
    best = Projector.load(destination, "cuda")
    final_scores = evaluate(best, valid, student, teacher)
    linear = Projector.load(linear_destination, "cuda")
    baseline_scores = evaluate(linear, valid, student, teacher)
    complete = {**metadata, "validation": final_scores, "linear_baseline": baseline_scores,
                "finished_at": time.time(), "config": {key: str(value) if isinstance(value, Path) else value
                                                         for key, value in vars(args).items()}}
    atomic_json(destination / "training-complete.json", complete)
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == "__main__":
    main()
