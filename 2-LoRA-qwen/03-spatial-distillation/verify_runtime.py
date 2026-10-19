#!/usr/bin/env python3
"""Exercise the real NF4 backward into contextual and layout projector weights."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
import torch
from sharm_image21.core import Projector, atomic_json
from sharm_image21.diffusion import load_transformer, predict_flow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--context-layers", type=int, default=2)
    args = parser.parse_args()
    if args.resolution < 64 or args.resolution % 32:
        raise ValueError("Résolution multiple de 32 requise.")
    if not 1 <= args.tokens <= 4096 or args.context_layers < 1:
        raise ValueError("De 1 à 4096 tokens et au moins un bloc contextuel requis.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requis.")
    import bitsandbytes as bnb
    torch.cuda.reset_peak_memory_stats()
    diffuser = load_transformer(training=True, gradient_checkpointing="all").train()
    projector = Projector(context_layers=args.context_layers, layout_enabled=True).cuda().train()
    optimizer = bnb.optim.AdamW8bit(projector.parameters(), lr=1e-4)
    hidden = torch.randn(args.tokens, 2560, device="cuda", dtype=torch.bfloat16)
    boxes = torch.tensor([[0.1, 0.3, 0.35, 0.7], [0.65, 0.3, 0.9, 0.7]], device="cuda", dtype=torch.bfloat16)
    latent = torch.randn(1, 64, 1, args.resolution//16, args.resolution//16, device="cuda", dtype=torch.bfloat16)
    losses = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = predict_flow(diffuser, latent, projector(hidden, layout=boxes)[None],
                                     torch.tensor([0.5], device="cuda", dtype=torch.bfloat16))
        loss = predicted.float().square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Perte NF4 non finie.")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(projector.parameters(), 1, error_if_nonfinite=True)
        if not norm or any(p.grad is not None for p in diffuser.parameters()):
            raise RuntimeError("Gradient absent ou poids du diffuseur dégelés.")
        if step == 1:
            for prefix in ("context.", "layout_box.", "layout_object."):
                if not any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in projector.named_parameters() if n.startswith(prefix)):
                    raise RuntimeError(f"Aucun gradient dans {prefix}")
        optimizer.step()
        losses.append(loss.item())
    report = {"passed": True, "backward_into_projector": True, "diffuser_frozen": True,
              "resolution": args.resolution, "tokens": args.tokens, "context_layers": args.context_layers,
              "optimizer": "AdamW8bit", "losses": losses,
              "peak_gib": torch.cuda.max_memory_allocated()/2**30,
              "scope": "two synthetic NF4 backward steps; generated spatial quality is not evaluated"}
    atomic_json(ROOT / "artifacts/spatial/runtime-backward-audit.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
