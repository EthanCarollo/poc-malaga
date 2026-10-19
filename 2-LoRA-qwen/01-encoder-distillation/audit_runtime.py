#!/usr/bin/env python3
from pathlib import Path
import json
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import torch
from sharm_image21.core import ARTIFACTS, ORIGINAL, STUDENT, FrozenEncoder, atomic_json, digest


@torch.inference_mode()
def verify_fp8_kernel():
    """Compile/execute a small real FP8 operation before loading the 4B."""
    from transformers.integrations.finegrained_fp8 import FP8Linear
    os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")
    torch.manual_seed(42)
    layer = FP8Linear(256, 256, block_size=(128, 128)).cuda().eval().requires_grad_(False)
    layer.weight.copy_((torch.randn(256, 256, device="cuda")/16).to(torch.float8_e4m3fn))
    layer.weight_scale_inv.copy_(torch.rand(2, 2, device="cuda")*0.02+0.005)
    source = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
    result = layer(source)
    scale = layer.weight_scale_inv.repeat_interleave(128, 0).repeat_interleave(128, 1)
    reference = torch.nn.functional.linear(source.float(), layer.weight.float()*scale)
    relative_mse = ((result.float()-reference).square().mean()/reference.square().mean()).item()
    if not torch.isfinite(result).all() or not relative_mse < 0.02:
        raise RuntimeError(f"Calcul FP8 du kernel incohérent : MSE relative synthétique {relative_mse}")
    report = {"passed": True, "synthetic_relative_mse": relative_mse, "shape": [17, 256, 256],
              "scope": "FP8 arithmetic versus dequantized reference; not 4B versus 8B quality"}
    print(json.dumps({"kernel_smoke": report}), flush=True)
    return report


def main():
    teacher = json.loads((ORIGINAL / "processor/tokenizer.json").read_text())
    student = json.loads((STUDENT / "tokenizer.json").read_text())
    teacher_vocab = teacher["model"]["vocab"]
    student_vocab = student["model"]["vocab"]
    def normalized_merges(config):
        return [tuple(pair.split(" ", 1)) if isinstance(pair, str) else tuple(pair)
                for pair in config["model"].get("merges", [])]
    if (teacher_vocab != student_vocab or normalized_merges(teacher) != normalized_merges(student)
            or teacher["added_tokens"] != student["added_tokens"]):
        raise RuntimeError("Tokenizers non alignés : arrêter avant la distillation token par token.")
    kernel_check = verify_fp8_kernel()
    encoder = FrozenEncoder("student")
    prompts = ["A red cube to the left of a blue sphere on a white table.",
               "This is an RGBA image with transparency. A small green dragon pixel-art sprite. The background is transparent."]
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    outputs = encoder.encode(prompts)
    single = encoder.encode(prompts[:1])[0][0]
    batch_delta = (outputs[0][0].float()-single.float()).norm()/single.float().norm()
    if batch_delta > 0.05:
        raise RuntimeError(f"Incohérence entre encodage seul et batch : {batch_delta.item():.4f}")
    report = {"passed": True, "encoder_identity": encoder.identity,
              "kernel_smoke": kernel_check,
              "vocab_sha256": digest(teacher_vocab), "fp8_linears": encoder.quantized_linears,
              "feature_shapes": [list(value.shape) for value, _ in outputs],
              "batch_relative_delta": batch_delta.item(), "seconds": time.monotonic()-started,
              "fp8_prefill_policy": "single-prompt-fp8",
              "gpu_peak_gib": torch.cuda.max_memory_allocated()/2**30,
              "capability": list(torch.cuda.get_device_capability()),
              "scope": "text-to-image; image editing not yet validated"}
    atomic_json(ARTIFACTS / "runtime-audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
