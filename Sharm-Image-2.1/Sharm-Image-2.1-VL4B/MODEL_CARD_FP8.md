---
license: other
license_name: qwen-research
license_link: LICENSE
pipeline_tag: text-to-image
base_model:
- Qwen/Qwen-Image-2.1
- Qwen/Qwen3-VL-4B-Instruct-FP8
language: [fr, en]
---
# Sharm Image 2.1 VL4B FP8

Sharm Image Base: Qwen3-VL 4B text conditioning in FP8, the previously published BF16 alignment projector, and Qwen-Image-2.1 DiT linear weights in scaled FP8 E4M3. On supported NVIDIA GPUs the DiT also quantizes activations per row and uses native FP8 matrix multiplication, accumulating to BF16. Biases, norms, VAE and projector retain their original precision. There is no NF4 DiT in this release.

The projector is byte-identical to revision `1e912f5a1e5df40bdaf927abf3e4220d73781321`, SHA-256 `a414685b8f1c0b305ffeb0dc286afb47b168fc03a5be9d2832ff820fec678883`. The rejected spatial candidate is not included. Optional learned layout conditioning and input-image editing are not provided by this base.

## Run locally

Use Linux, a CUDA GPU with FP8 support (compute capability 8.9 or newer), and the pinned runtime in `requirements-inference.txt`. The export is verified on a 16 GB RTX 5060 Ti; this is not a promise that 8 GB is sufficient. Install PyTorch CUDA 13.0 wheels first, then these requirements. The official `kernels-community/finegrained-fp8` v4 encoder kernel must be available; its first download requires Internet. A C compiler is required by Triton.

```python
import sys
from huggingface_hub import snapshot_download
folder = snapshot_download("ethanzxv/Sharm-Image-2.1-VL4B-FP8")
sys.path.insert(0, folder)
import torch
from sharm_image21.pipeline import SharmImage21Pipeline
model = SharmImage21Pipeline.from_pretrained(folder)
image = model(
    "A single red fox, full body, isolated on a transparent background.",
    negative_prompt="", true_cfg_scale=3.0,
    width=512, height=512, output_resolution=512,
    num_inference_steps=30, use_kv_cache=False,
    generator=torch.Generator(device="cuda").manual_seed(1000),
).images[0]
assert image.mode == "RGBA"
image.save("result.png")
```

PNG alpha is saved unchanged. Checkerboards are separate previews, never painted onto the downloadable RGBA file. `fp8-validation.json` records the exact loader, actual dtypes, native FP8 calls and two real GPU generations. `release_manifest.json` binds files by SHA-256. Quantization can alter image quality; historical embedding MSE is not a position error or proof of 8B parity. The archived encoder comparisons and original base validation refer to the earlier source revision and are not benchmarks of this DiT quantization.

**Known alpha limitation:** the two inspected outputs have recognizable subjects but retain a largely opaque white background despite the transparent-background prompt. Their alpha ranges are 245–255 and 64–255, with no fully transparent pixels. The RGBA format works; clean transparent cutouts are not validated. The original alpha bytes are preserved rather than replaced by an undisclosed matte.

## Reproduce

The standalone Base notebook and all Sharm Image data/reports live in the `Sharm-Image-2.1/` folder of [the code repository](https://github.com/EthanCarollo/poc-malaga). Conversion streams the original BF16 DiT into FP8 shards; generation and publication stay disabled by default in Git.

Built with Qwen. The Qwen Research license, Apache-2.0 encoder license and notices are preserved. Dataset artwork rights are not changed by model or code licenses.
