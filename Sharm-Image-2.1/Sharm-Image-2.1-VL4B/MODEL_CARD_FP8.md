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

**Sharm Image Base** associe le text encoder Qwen3-VL 4B FP8, le projecteur d'alignement précédemment publié et le DiT Qwen-Image-2.1 en FP8 E4M3 avec échelles par ligne. Le modèle est disponible ; les tests réels ci-dessous montrent aussi ses limites de fidélité, de placement et de transparence. Aucun nouvel entraînement Monster ou Iris n'a été lancé.

| Composant | Modèle Qwen d'origine | Sharm actuel |
| --- | --- | --- |
| Text encoder | Qwen3-VL 8B BF16 | Qwen3-VL 4B FP8, gelé |
| Alignement | Conditionnement natif 8B | Projecteur BF16 `a414685b` |
| DiT | Poids BF16 | 232 projections, 7 115 112 448 poids FP8 E4M3 |
| Calcul DiT quantifié | BF16 | Activations quantifiées par ligne, matmul FP8 natif, sortie BF16 |
| Autres composants | Précision d'origine | Normes, biais, VAE et projecteur gardent leur précision |

L'audit de l'encodeur compte 3 633 315 840 paramètres de langue FP8 ; tous les paramètres du modèle ne sont donc pas en FP8. Les 232 projections du DiT utilisent effectivement les kernels FP8 sur le GPU testé. Ce dépôt ne fournit ni édition depuis une image d'entrée, ni module appris de coordonnées. Les PNG générés conservent leur canal alpha natif.

Le projecteur est identique à celui de `1e912f5a1e5df40bdaf927abf3e4220d73781321`, SHA-256 `a414685b8f1c0b305ffeb0dc286afb47b168fc03a5be9d2832ff820fec678883`. Le candidat spatial entraîné puis refusé n'est pas utilisé. Les sorties comparées de Sharm sont celles des poids publiés à `a0c0e168097024cecd8c43f68004d1bedbab32df` ; la mise à jour de cette carte conserve les poids et le loader.

## Lire les noms et les formats

**B = milliard de paramètres**, c'est-à-dire de valeurs apprises. Cela décrit la taille d'un réseau, pas sa mémoire en Go.

| Modèle | Encodeur qui comprend le texte | DiT qui génère l'image | Sortie |
| --- | --- | --- | --- |
| **Sharm Image 2.1 VL4B FP8** | **Qwen3-VL 4B**, utilisé pour le texte, en FP8 ; projecteur d'alignement BF16 | **Qwen-Image-2.1, environ 7,1B** ; projections principales en FP8 natif | RGBA, avec canal alpha |
| **Sharm Iris PixelArt** | **Qwen3-VL 4B**, en BF16 dans le notebook Iris | **Iris-3B, environ 3B**, chargé en BF16 pour ces tests | RGB ; aucun alpha natif |

Le **4B de Sharm appartient à Qwen3-VL**, son encodeur de texte. Le **3B d'Iris appartient au DiT**, son générateur. Iris possède donc aussi un encodeur Qwen3-VL 4B séparé. « VL » signifie vision et langage ; ces parcours utilisent le texte uniquement.

- **Encodeur / embeddings** : transforme le prompt, la description demandée, en tableaux de nombres qui conditionnent le générateur.
- **Projecteur** : adapte les représentations de l'encodeur Qwen3-VL 4B au format attendu par le générateur Qwen-Image-2.1.
- **DiT** (*Diffusion Transformer*) : réseau qui transforme progressivement le bruit initial en image. Sharm utilise ensuite un **VAE**, le décodeur qui convertit sa représentation comprimée en pixels. Iris calcule directement les pixels, sans VAE.
- **FP8 / BF16 / NF4** : formats de nombres à 8, 16 ou 4 bits. Ils changent la mémoire et parfois le résultat ; la taille en paramètres reste la même. Certains composants gardent une autre précision.
- **LoRA / fine-tuning** : petites matrices apprises / adaptation sur un dataset. Les tests du modèle Iris de base ne chargent aucun LoRA et ne lancent aucun entraînement.
- **Graine / pas de génération / CFG** : départ aléatoire reproductible / nombre d'itérations pour produire l'image / intensité du guidage par le texte. Les pas d'entraînement sont, eux, des mises à jour des poids.
- **RGB / RGBA** : couleurs seules / couleurs plus alpha, le canal de transparence. Un fichier RGBA peut quand même avoir un fond opaque.
- **MSE** : moyenne des différences au carré entre nombres ou pixels ; elle ne mesure pas directement la position correcte d'un objet.
- **Offload / cache / runtime** : transfert de composants entre RAM et GPU / résultats conservés pour éviter un recalcul / environnement logiciel d'exécution.

- **E4M3 / FP8 natif** : variante FP8 utilisée avec des facteurs d'échelle / calcul réellement effectué en FP8 par le GPU pour les projections indiquées, vérifié par audit.
- **Gelé / dataset / validation** : poids non modifiés / ensemble d'images et descriptions / exemples tenus à part pour contrôler le résultat.
- **SHA-256 / révision** : empreinte d'un fichier / identifiant d'une version du dépôt, pour retrouver exactement les octets testés.
- **HF / kernel / lien physique** : Hugging Face, qui héberge les modèles / processus Python qui exécute le notebook / plusieurs noms partageant le même fichier sur disque.

Le style **pixel art** concerne la grille visible, les contours et la palette. Le terme **pixel-space** d'Iris décrit son calcul direct des pixels ; le style pixel art est demandé par les prompts et doit être vérifié dans les sorties.

## 1. Ce que génère réellement Sharm 4B : sujets et scènes simples

Ces **16 images Sharm actuelles** testent des sujets reconnaissables, du pixel art, des objets, du texte et un paysage. Huit prompts fixés, deux graines (1000/1001), 512 × 512, 30 pas, CFG 3, sans LoRA et sans nouvel entraînement. Il s'agit des sorties FP8 du diagnostic présenté en section 3 : elles ne constituent pas seize essais indépendants supplémentaires.

Le renard, le dragon, la grenouille et le paysage sont reconnaissables. Les demandes de détourage, les détails et certaines compositions ne sont pas toujours respectés. La fleur noire et les livres montrent ces limites. Tous les exemples et les deux graines sont conservés.

| Test et observation par agent | Graine 1000 — Sharm actuel | Graine 1001 — Sharm actuel |
| --- | --- | --- |
| **Renard** — Sujet reconnaissable ; le fond reste largement opaque. | ![Renard, 1000](comparisons/nf4-vs-fp8-v1/images/fox-1000-after-fp8.png) | ![Renard, 1001](comparisons/nf4-vs-fp8-v1/images/fox-1001-after-fp8.png) |
| **Dragon pixel art** — Sujet reconnaissable ; à 1001, le fond transparent de la variante NF4 devient opaque en FP8. | ![Dragon pixel art, 1000](comparisons/nf4-vs-fp8-v1/images/dragon-1000-after-fp8.png) | ![Dragon pixel art, 1001](comparisons/nf4-vs-fp8-v1/images/dragon-1001-after-fp8.png) |
| **Fleur noire** — La silhouette est sombre et pauvre en détails. | ![Fleur noire, 1000](comparisons/nf4-vs-fp8-v1/images/flower-1000-after-fp8.png) | ![Fleur noire, 1001](comparisons/nf4-vs-fp8-v1/images/flower-1001-after-fp8.png) |
| **Grenouille pixel art** — Sujet reconnaissable ; les proportions varient entre les graines. | ![Grenouille pixel art, 1000](comparisons/nf4-vs-fp8-v1/images/frog-1000-after-fp8.png) | ![Grenouille pixel art, 1001](comparisons/nf4-vs-fp8-v1/images/frog-1001-after-fp8.png) |
| **Cube à gauche de la sphère** — Les deux objets et leur ordre gauche/droite restent visibles. | ![Cube à gauche de la sphère, 1000](comparisons/nf4-vs-fp8-v1/images/left-right-1000-after-fp8.png) | ![Cube à gauche de la sphère, 1001](comparisons/nf4-vs-fp8-v1/images/left-right-1001-after-fp8.png) |
| **Étoile au-dessus du cercle** — Disposition verticale visible ; à 1001, les objets se touchent presque. | ![Étoile au-dessus du cercle, 1000](comparisons/nf4-vs-fp8-v1/images/above-below-1000-after-fp8.png) | ![Étoile au-dessus du cercle, 1001](comparisons/nf4-vs-fp8-v1/images/above-below-1001-after-fp8.png) |
| **Panneau OPEN et livres** — OPEN est lisible ; à 1001, une bibliothèque supplémentaire remplace la composition simple demandée. | ![Panneau OPEN et livres, 1000](comparisons/nf4-vs-fp8-v1/images/typography-1000-after-fp8.png) | ![Panneau OPEN et livres, 1001](comparisons/nf4-vs-fp8-v1/images/typography-1001-after-fp8.png) |
| **Paysage de montagne** — La scène et les grandes masses sont cohérentes ; les détails varient. | ![Paysage de montagne, 1000](comparisons/nf4-vs-fp8-v1/images/landscape-1000-after-fp8.png) | ![Paysage de montagne, 1001](comparisons/nf4-vs-fp8-v1/images/landscape-1001-after-fp8.png) |

[Prompts et réglages exacts](comparisons/nf4-vs-fp8-v1/protocol.json) · [Revue exhaustive](comparisons/nf4-vs-fp8-v1/REVIEW.md). Les PNG ci-dessus conservent leur alpha natif, sans détourage ajouté. Les comparaisons suivantes détaillent séparément le changement de modèle complet et le changement de quantification.

## 2. Comparatif principal : modèle de base original / Sharm actuel

**16 paires, 32 générations réelles, toutes conservées et examinées par agent.** À gauche : `Qwen/Qwen-Image-2.1` à `d26bb61231c349cf6b7896fa83353113880e1ba3`, avec ses features 8B BF16 et son vrai DiT BF16. À droite : Sharm publié, encodeur 4B FP8 + projecteur + vrai DiT FP8 natif. Aucun LoRA et aucun candidat spatial refusé n'interviennent.

### Protocole reproductible

| Paramètre | Valeur commune aux deux variantes |
| --- | --- |
| Prompts | 8 scènes de validation, première scène par relation/langue dans l'ordre source, choisies avant génération |
| Relations et langues | Gauche, droite, dessus, dessous × français et anglais |
| Objets demandés | 6 prompts à deux objets et 2 prompts à trois objets ; textes exacts conservés |
| Graines | 1000 et 1001 pour chaque prompt |
| Résolution et pas | 512 × 512, 30 pas |
| Guidance | CFG 1, aucun prompt négatif |
| Bruit initial | Même tenseur sauvegardé et même SHA pour les deux images de chaque paire |
| Conditionnement | Features propres à chaque encodeur, volontairement différentes |
| Autres réglages | Même scheduler et runtime figé ; KV cache désactivé |

La référence utilise les **vraies features 8B BF16 pré-calculées**, dont l'identité, les tokens, la convention avant RMSNorm et les empreintes sont contrôlés. L'encodeur 8B n'a pas été relancé pour cette comparaison. Le DiT de référence est entièrement BF16, sans NF4 : il passe sur GPU pendant le débruitage, puis sur CPU pour laisser le VAE décoder. Sharm utilise son loader FP8 natif. **Ce protocole compare les images ; le cache d'encodage et l'offload empêchent une comparaison complète de vitesse ou de VRAM des deux modèles.**

Cette comparaison change à la fois l'encodeur, le projecteur et le format du DiT. Elle mesure leur résultat combiné ; elle ne permet pas d'attribuer chaque défaut au seul FP8. Le diagnostic NF4/FP8 plus bas conserve, lui, le même conditionnement 4B.

[Protocole et prompts exacts](comparisons/base-vs-sharm-v1/protocol.json) · [Sources et SHA](comparisons/base-vs-sharm-v1/sources.json) · [Audit des dtypes](comparisons/base-vs-sharm-v1/loader-audits.json) · [Toutes les mesures](comparisons/base-vs-sharm-v1/metrics.json) · [Revue complète](comparisons/base-vs-sharm-v1/REVIEW.md) · [Manifeste des fichiers](comparisons/base-vs-sharm-v1/manifest.json)

### Résultats observés et limites

**La parité avec le 8B n'est pas démontrée.** Sur plusieurs scènes multi-objets, Sharm produit un seul personnage hybride ou des sujets superposés. La référence conserve plus souvent plusieurs silhouettes distinctes dans cet échantillon ; elle commet aussi des erreurs d'espèces, de nombre et de relation. Elle n'est pas une vérité terrain.

**Les positions fiables ne sont pas validées.** Une séparation verticale apparaît dans certains cas, notamment `above-en`, mais elle ne suffit pas lorsque l'identité des objets n'est pas respectée. Les demandes sans chevauchement échouent aussi dans plusieurs sorties Sharm. Les huit planches montrent les deux graines sans sélection.

**Le canal alpha fonctionne, la qualité du détourage reste variable.** Ici, les 16 PNG Sharm et 11 des 16 PNG de référence contiennent au moins un pixel totalement transparent. Ce critère très faible n'est pas un score de détourage : il ne contrôle ni les contours ni la transparence complète du fond. Certains fonds clairs ou motifs de damier sont générés dans l'image de référence. Le test CFG 3 du dragon plus bas montre aussi une perte de transparence en FP8.

La MSE RGB moyenne sur fond blanc vaut **0.052359** et la MSE alpha **0.279840**. Pour des valeurs normalisées entre 0 et 1, `MSE = moyenne((A - B)²)` ; le RGB est composé sur blanc par `C = alpha × RGB + (1 - alpha)`. La MSE alpha compare directement les deux canaux alpha. Ces chiffres mesurent une différence entre sorties, sans vérité terrain : **ce ne sont ni un pourcentage d'erreur des positions ni une mesure de qualité**. Une faible différence entre deux mauvaises images ne valide pas le résultat.

La revue est identifiée `agent_visual_review`, par Codex. Cet échantillon de 8 prompts et 2 graines n'est pas une évaluation humaine statistique et ne démontre ni parité générale avec le 8B, ni positions parfaites. Les PNG RGBA originaux sont conservés ; les damiers ci-dessous sont uniquement des aperçus.

### Toutes les images : original à gauche, Sharm à droite

#### 1. À gauche — FR

Objets demandés : shiny Pidgeot, shiny Gligar. Graines 1000 (haut) et 1001 (bas).

La référence garde deux silhouettes côte à côte aux deux graines, sans fidélité fiable aux espèces. Sharm fusionne les sujets en une grande créature à la graine 1000 ; à 1001, deux créatures se chevauchent.

![Qwen original à gauche, Sharm actuel à droite — left-fr, deux graines](comparisons/base-vs-sharm-v1/sheets/pairs-01.png)

#### 2. À gauche — EN

Objets demandés : shiny Rhydon, shiny Articuno, shiny Rhydon. Graines 1000 (haut) et 1001 (bas).

Les trois objets restent distincts dans les images de référence, avec des espèces déformées. Sharm produit des créatures superposées et ne restitue pas les trois objets séparés attendus.

![Qwen original à gauche, Sharm actuel à droite — left-en, deux graines](comparisons/base-vs-sharm-v1/sheets/pairs-02.png)

#### 3. À droite — FR

Objets demandés : Mudkip, shiny Camerupt, Mudkip. Graines 1000 (haut) et 1001 (bas).

La référence manque le troisième objet à la graine 1000 et produit trois silhouettes à 1001, avec des erreurs d'identité et de relation. Sharm donne un seul personnage aux deux graines.

![Qwen original à gauche, Sharm actuel à droite — right-fr, deux graines](comparisons/base-vs-sharm-v1/sheets/pairs-03.png)

#### 4. À droite — EN

Objets demandés : shiny Ralts, Solrock. Graines 1000 (haut) et 1001 (bas).

Deux silhouettes restent visibles dans la référence mais leurs identités et l'association à la relation ne sont pas fiables. Sharm remplace les deux objets par un seul personnage aux deux graines.

![Qwen original à gauche, Sharm actuel à droite — right-en, deux graines](comparisons/base-vs-sharm-v1/sheets/pairs-04.png)

#### 5. Au-dessus — FR

Objets demandés : Plusle, Shellder. Graines 1000 (haut) et 1001 (bas).

La référence conserve deux formes séparées verticalement, avec des identités erronées. Sharm ne montre qu'un personnage à 1000 ; à 1001, un petit personnage est posé au-dessus d'un grand et les contours se touchent.

![Qwen original à gauche, Sharm actuel à droite — above-fr, deux graines](comparisons/base-vs-sharm-v1/sheets/pairs-05.png)

#### 6. Au-dessus — EN

Objets demandés : shiny Aerodactyl, shiny Raikou. Graines 1000 (haut) et 1001 (bas).

Les deux variantes gardent deux créatures disposées verticalement aux deux graines. Leur forme, leurs couleurs et leurs identités dérivent ; cette disposition seule ne valide pas les sujets demandés.

![Qwen original à gauche, Sharm actuel à droite — above-en, deux graines](comparisons/base-vs-sharm-v1/sheets/pairs-06.png)

#### 7. En dessous — FR

Objets demandés : shiny Castform, sunny form, Paras. Graines 1000 (haut) et 1001 (bas).

La référence montre deux figures verticales mais ne restitue pas fidèlement Castform/Paras et leur association à la consigne. Sharm n'en conserve qu'une aux deux graines.

![Qwen original à gauche, Sharm actuel à droite — below-fr, deux graines](comparisons/base-vs-sharm-v1/sheets/pairs-07.png)

#### 8. En dessous — EN

Objets demandés : Hoppip, shiny Cacnea. Graines 1000 (haut) et 1001 (bas).

La référence conserve deux silhouettes verticales, sans fidélité suffisante aux espèces. Sharm donne un seul personnage à 1000, puis deux à 1001 ; la relation entre Hoppip et Cacnea n'est pas validée.

![Qwen original à gauche, Sharm actuel à droite — below-en, deux graines](comparisons/base-vs-sharm-v1/sheets/pairs-08.png)

## 3. Diagnostic secondaire : ancien DiT NF4 / DiT FP8 natif

**16 paires / 32 générations réelles**, inspectées par agent : 8 prompts × 2 graines, 512 px, 30 pas, CFG 3. Chaque paire partage exactement les embeddings et les latents initiaux sauvegardés. Le même encodeur 4B FP8 et le même projecteur sont utilisés. À gauche : l'ancien loader DiT NF4 (`1e912f5a`). À droite : le DiT FP8 natif (`a0c0e168`). Les deux tournent dans le même runtime figé ; ce n'est pas un nouvel entraînement.

| Mesure sur ce passage | Avant NF4 | Après FP8 |
| --- | ---: | ---: |
| Temps médian génération + décodage, hors première image | 26,43 s | 20,90 s |
| Pic VRAM allouée par PyTorch | 5,73 Gio | 8,95 Gio |
| Images conservées | 16 | 16 |

Les temps excluent chargement et encodage texte ; un passage par variante ne constitue pas une garantie de performance. La MSE RGB moyenne sur blanc vaut **0,037987** et la MSE alpha moyenne **0,046231**, sur [0,1]. Ces écarts entre pixels ne mesurent ni la qualité ni l'erreur de position.

**Régression visible : le dragon, graine 1001, perd son fond transparent en FP8.** Les compositions restent souvent proches, mais certains détails et contenus changent. L'étoile et le cercle sont mieux séparés en FP8 à la graine 1000 ; aucun résultat ne garantit des positions parfaites. Cette revue est `agent_visual_review`, sans annotation humaine ni évaluation humaine statistique.

[Rapport et paramètres complets](comparisons/nf4-vs-fp8-v1/README.md) · [Revue des 16 paires](comparisons/nf4-vs-fp8-v1/REVIEW.md) · [Mesures JSON](comparisons/nf4-vs-fp8-v1/metrics.json) · [SHA-256](comparisons/nf4-vs-fp8-v1/manifest.json). Les PNG originaux gardent leur alpha ; le damier appartient seulement aux planches de prévisualisation.

<details>
<summary>Voir toutes les 16 paires NF4 / FP8 (8 planches)</summary>

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


</details>

## 4. Comparatif historique des encodeurs : 8B BF16 / 4B FP8

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

**Known alpha limitation of the two exported-loader smoke tests (CFG 3):** these two outputs have recognizable subjects but retain a largely opaque white background despite the transparent-background prompt. Their alpha ranges are 245–255 and 64–255, with no fully transparent pixels. The RGBA format works; clean transparent cutouts are not validated. The original alpha bytes are preserved rather than replaced by an undisclosed matte.

## Reproduce

The standalone Base notebook and all Sharm Image data/reports live in the `Sharm-Image-2.1/` folder of [the code repository](https://github.com/EthanCarollo/poc-malaga). Conversion streams the original BF16 DiT into FP8 shards; generation and publication stay disabled by default in Git.

Built with Qwen. The Qwen Research license, Apache-2.0 encoder license and notices are preserved. Dataset artwork rights are not changed by model or code licenses.
