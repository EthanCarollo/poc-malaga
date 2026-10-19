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

## Tests et comparatifs visuels

**16 paires / 32 générations réelles**, inspectées par agent : 8 prompts × 2 graines, 512 px, 30 pas, CFG 3. Chaque paire partage exactement les embeddings et les latents initiaux sauvegardés. Le même encodeur 4B FP8 et le même projecteur sont utilisés. À gauche : l'ancien loader DiT NF4 (`1e912f5a`). À droite : le DiT FP8 natif (`a0c0e168`). Les deux tournent dans le même runtime figé ; ce n'est pas un nouvel entraînement.

| Mesure sur ce passage | Avant NF4 | Après FP8 |
| --- | ---: | ---: |
| Temps médian génération + décodage, hors première image | 26,43 s | 20,90 s |
| Pic VRAM allouée par PyTorch | 5,73 Gio | 8,95 Gio |
| Images conservées | 16 | 16 |

Les temps excluent chargement et encodage texte ; un passage par variante ne constitue pas une garantie de performance. La MSE RGB moyenne sur blanc vaut **0,037987** et la MSE alpha moyenne **0,046231**, sur [0,1]. Ces écarts entre pixels ne mesurent ni la qualité ni l'erreur de position.

**Régression visible : le dragon, graine 1001, perd son fond transparent en FP8.** Les compositions restent souvent proches, mais certains détails et contenus changent. L'étoile et le cercle sont mieux séparés en FP8 à la graine 1000 ; aucun résultat ne garantit des positions parfaites. Cette revue est `agent_visual_review`, sans annotation humaine ni évaluation humaine statistique.

[Rapport et paramètres complets](comparisons/nf4-vs-fp8-v1/README.md) · [Revue des 16 paires](comparisons/nf4-vs-fp8-v1/REVIEW.md) · [Mesures JSON](comparisons/nf4-vs-fp8-v1/metrics.json) · [SHA-256](comparisons/nf4-vs-fp8-v1/manifest.json). Les PNG originaux gardent leur alpha ; le damier appartient seulement aux planches de prévisualisation.

### Renard

![Avant NF4 à gauche, après FP8 à droite : deux graines](comparisons/nf4-vs-fp8-v1/sheets/pairs-01.png)

### Dragon — perte de transparence à la graine 1001

![Avant NF4 à gauche, après FP8 à droite : deux graines](comparisons/nf4-vs-fp8-v1/sheets/pairs-02.png)

### Fleur

![Avant NF4 à gauche, après FP8 à droite : deux graines](comparisons/nf4-vs-fp8-v1/sheets/pairs-03.png)

### Grenouille

![Avant NF4 à gauche, après FP8 à droite : deux graines](comparisons/nf4-vs-fp8-v1/sheets/pairs-04.png)

### Cube à gauche de la sphère

![Avant NF4 à gauche, après FP8 à droite : deux graines](comparisons/nf4-vs-fp8-v1/sheets/pairs-05.png)

### Étoile au-dessus du cercle

![Avant NF4 à gauche, après FP8 à droite : deux graines](comparisons/nf4-vs-fp8-v1/sheets/pairs-06.png)

### Texte OPEN et livres

![Avant NF4 à gauche, après FP8 à droite : deux graines](comparisons/nf4-vs-fp8-v1/sheets/pairs-07.png)

### Paysage

![Avant NF4 à gauche, après FP8 à droite : deux graines](comparisons/nf4-vs-fp8-v1/sheets/pairs-08.png)

### Comparatif historique des encodeurs : 8B BF16 / 4B FP8

Ces **4 paires historiques** utilisent le même DiT **NF4**, à 512 px, **20 pas, CFG 1**, avec le projecteur `a414685b`. Elles proviennent de `1e912f5a` ; elles sont distinctes des tests de diffusion ci-dessus. Elles montrent des sujets communs mais aussi des différences de fond, de disposition et de déformations, sans démontrer une parité 8B.

[Paramètres historiques exacts](comparisons/encoder/comparison.json).

| Cas | Encodeur 8B BF16 | Encodeur 4B FP8 + projecteur |
| --- | --- | --- |
| Fleur noire | ![8B BF16](comparisons/encoder/00-teacher-preview.png) | ![4B FP8](comparisons/encoder/00-sharm-preview.png) |
| Texte OPEN | ![8B BF16](comparisons/encoder/01-teacher-preview.png) | ![4B FP8](comparisons/encoder/01-sharm-preview.png) |
| Vélo et bateau | ![8B BF16](comparisons/encoder/02-teacher-preview.png) | ![4B FP8](comparisons/encoder/02-sharm-preview.png) |
| Vélo et fleurs | ![8B BF16](comparisons/encoder/03-teacher-preview.png) | ![4B FP8](comparisons/encoder/03-sharm-preview.png) |

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
