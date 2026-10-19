# Tests

Dépôt de preuves de concept autour de solutions locales en IA. Le dépôt accueillera trois petits projets ; le premier est `1-local-generation`.

## Projets

- `1-local-generation` — application Windows native de génération locale d’images avec FLUX 3 Image Klein en FP8 et Qwen-Image-2.1.
- `2-LoRA-qwen` — projet Sharm : alignement d’un encodeur Qwen3-VL 4B FP8 gelé avec un projecteur entraînable, puis LoRA de sprites. Les [deux notebooks](2-LoRA-qwen/README.md) contiennent les étapes et la publication privée sous `ethanzxv`.

Chaque projet documentera ses dépendances, son installation et son lancement dans son propre dossier.

Le [projet Sharm](2-LoRA-qwen/README.md) contient deux notebooks autonomes dans deux dossiers : [Base VL 4B FP8](2-LoRA-qwen/Sharm-Image-2.1-VL4B/Sharm_Image_2_1_VL4B.ipynb) et [LoRA Monster](2-LoRA-qwen/Sharm-Image-2.1-VL4B-LoRA-Monster/Sharm_Image_2_1_VL4B_LoRA_Monster.ipynb). Ils contiennent le code d'exécution local, des données jusqu'à la publication ; les résultats finaux revus sont conservés dans des [rapports versionnés](2-LoRA-qwen/Sharm-Image-2.1-VL4B/reports/README.md).

## Données et workflow Git

Les [PNG sources et le dataset Monster validé](2-LoRA-qwen/Sharm-Image-2.1-VL4B-LoRA-Monster/dataset/README.md) sont versionnés avec les notebooks, définitions, rapports et galeries. Les poids des modèles, identifiants, environnements, caches et brouillons restent locaux.

Chaque ensemble cohérent de changements est commité selon la convention Angular `type(scope): description`, puis poussé sur `origin/main`. Voir [AGENTS.md](AGENTS.md) pour la consigne permanente.
