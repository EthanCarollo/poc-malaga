# Tests

Dépôt de preuves de concept autour de solutions locales en IA. Le dépôt accueillera trois petits projets ; le premier est `1-local-generation`.

## Projets

- `1-local-generation` — application Windows native de génération locale d’images avec FLUX 3 Image Klein en FP8 et Qwen-Image-2.1.
- `Sharm-Image-2.1` — projet Sharm : alignement d’un encodeur Qwen3-VL 4B FP8 gelé avec un projecteur entraînable, puis LoRA de sprites. Les [deux notebooks](Sharm-Image-2.1/README.md) contiennent les étapes et la publication privée sous `ethanzxv`.

La [Base Sharm Image](https://huggingface.co/ethanzxv/Sharm-Image-2.1-VL4B-FP8/tree/812304af8f902f6ab0138b66b4b8b336125780be) est publiée avec **text encoder et DiT FP8**. Le code, les données et les preuves sont regroupés dans [Sharm-Image-2.1/](Sharm-Image-2.1/README.md), qui remplace l'ancien dossier `2-LoRA-qwen`.

Voir les [comparatifs avant / après et leurs galeries complètes](Sharm-Image-2.1/Sharm-Image-2.1-VL4B/reports/README.md#comparatifs-avant--après) : **encodeur Qwen3-VL 8B BF16 + DiT Qwen-Image-2.1 BF16 / encodeur Qwen3-VL 4B FP8 + même architecture DiT en FP8**, puis diagnostic NF4 / FP8 avec conditionnement identique et archive historique des encodeurs.

Le nouveau projet **[Sharm-Iris-PixelArt](https://github.com/EthanCarollo/Sharm-Iris-PixelArt)** possède son dépôt séparé : un notebook de fine-tuning Iris-3B, les 386 fiches `pokemon_traits.json` et 3 444 sprites vérifiés. Le notebook est préparé ; aucun entraînement Iris n'a encore été lancé. [Destination HF des futurs poids](https://huggingface.co/ethanzxv/Sharm-Iris-PixelArt).

### À quoi correspondent les tailles ?

**B signifie milliard de paramètres**, pas Go de mémoire.

| Projet | Encodeur de texte | Générateur d’image (DiT) | Sortie |
| --- | --- | --- | --- |
| Sharm Image 2.1 VL4B FP8 | **Qwen3-VL 4B FP8** : le « 4B » du nom | **Qwen-Image-2.1 ≈7,1B**, projections principales FP8 | RGBA ; fond parfois opaque |
| Sharm Iris PixelArt | **Qwen3-VL 4B BF16**, réseau séparé | **Iris ≈3B** : le « 3B » du modèle source ; poids FP32, calcul partiel BF16 | RGB ; pas d’alpha natif |

Le [glossaire des composants et formats](Sharm-Image-2.1/README.md#lire-les-noms-et-les-formats) explique encodeur, DiT, projecteur, VAE, précisions numériques, mémoire, LoRA et mesures. Les notebooks et les cartes HF reprennent ces définitions.

Chaque projet documentera ses dépendances, son installation et son lancement dans son propre dossier.

Le [projet Sharm](Sharm-Image-2.1/README.md) contient deux notebooks autonomes dans deux dossiers : [Base VL 4B FP8](Sharm-Image-2.1/Sharm-Image-2.1-VL4B/Sharm_Image_2_1_VL4B.ipynb) et [LoRA Monster](Sharm-Image-2.1/Sharm-Image-2.1-VL4B-LoRA-Monster/Sharm_Image_2_1_VL4B_LoRA_Monster.ipynb). Ils contiennent le code d'exécution local, des données jusqu'à la publication ; les résultats finaux revus sont conservés dans des [rapports versionnés](Sharm-Image-2.1/Sharm-Image-2.1-VL4B/reports/README.md).

## Données et workflow Git

Les [PNG sources et le dataset Monster validé](Sharm-Image-2.1/Sharm-Image-2.1-VL4B-LoRA-Monster/dataset/README.md) sont versionnés avec les notebooks, définitions, rapports et galeries. Les poids des modèles, identifiants, environnements, caches et brouillons restent locaux.

Chaque ensemble cohérent de changements est commité selon la convention Angular `type(scope): description`, puis poussé sur `origin/main`. Voir [AGENTS.md](AGENTS.md) pour la consigne permanente.
