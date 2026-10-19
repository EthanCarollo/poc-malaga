# Sharm Image 2.1 — deux notebooks

Le projet s’exécute depuis **deux notebooks autonomes**, chacun dans son dossier. Ouvrir le fichier voulu avec un kernel Python sur la machine Linux CUDA où les calculs doivent tourner. Les opérations s’exécutent localement dans ce kernel ; aucun accès SSH n’est intégré aux notebooks.

| Modèle | Notebook | Contenu |
|---|---|---|
| **Sharm Image 2.1 VL 4B FP8 — Base** | [Notebook Base](Sharm-Image-2.1-VL4B/Sharm_Image_2_1_VL4B.ipynb) | Installation, données RGBA, distillation 8B → 4B, caches, entraînement du projecteur, comparaisons, export, inférence, publication et rapports |
| **Sharm Image 2.1 VL 4B LoRA Monster** | [Notebook Monster](Sharm-Image-2.1-VL4B-LoRA-Monster/Sharm_Image_2_1_VL4B_LoRA_Monster.ipynb) | Dataset, définitions, Base retenue, caches, entraînement LoRA, comparaisons avant/après, contrôle RGBA et publication |

```text
2-LoRA-qwen/
├── README.md
├── Sharm-Image-2.1-VL4B/
│   ├── Sharm_Image_2_1_VL4B.ipynb
│   ├── configs/
│   ├── licenses/
│   ├── requirements-*.txt
│   └── reports/
└── Sharm-Image-2.1-VL4B-LoRA-Monster/
    ├── Sharm_Image_2_1_VL4B_LoRA_Monster.ipynb
    ├── monster_sources.json
    ├── monster_label_font.json
    └── dataset/
        ├── raw/
        ├── current.json
        └── prepared/monster-clean-v2/
```

## Exécuter

1. Ouvrir le notebook sur la machine de calcul avec Python 3.12 et CUDA.
2. Lire la cellule d’installation et utiliser le Python du kernel. Les versions et la révision Diffusers sont figées dans le dossier Base.
3. Exécuter les cellules de préparation et choisir les paramètres.
4. Activer uniquement le drapeau `RUN_...` de l’étape à lancer. Les drapeaux sont à `False` dans Git ; leur activation exécute réellement le code.
5. Lire les mesures et examiner les PNG sur damier avant l’export et la publication.

Chaque fichier contient ses fonctions dans une annexe de cellules Python. Le chargeur crée des modules en mémoire à partir de ces cellules et exclut les cellules de lancement des imports. Les calculs ne dépendent d’aucun fichier `.py`, `.sh`, `.ps1` ou d’un troisième notebook du dépôt. Le package Python imposé par le format HF est généré uniquement dans l’artefact d’export.

Enregistrer le notebook après modification du code de l’annexe : le chargeur lit ces cellules dans le fichier sauvegardé. Les paramètres du parcours sont les valeurs courantes du kernel. Les checkpoints historiques conservent l’identité de leurs anciennes sources ; une nouvelle recette doit utiliser un workspace/run neuf ou des identités explicitement compatibles.

Aucun nouvel entraînement LoRA n’est lancé pendant la préparation de ces fichiers. La chaîne Base historique déjà active conserve ses sources dans son workspace jusqu’à sa fin. Les nouveaux notebooks ne sont pas copiés par-dessus les fichiers d’un calcul actif.

## Modèles et données

La Base comprend le générateur Qwen Image 2.1, le VL 4B FP8 gelé et son projecteur. Le 8B BF16 est un enseignant hors ligne. Le LoRA Monster spécialise une révision précise de cette Base ; le SHA du projecteur et la révision HF doivent correspondre.

Les 25 PNG sources et le dataset Monster validé sont dans le dossier Monster : 4 481 images RGBA uniques, 649 espèces, définitions FR/EN, légendes, splits par espèce et rapports. Le déplacement conserve les octets des PNG et des manifestes signés. La provenance de la recette historique reste dans les anciens commits Git.

Une baisse de perte numérique ne donne pas un taux d’erreur des positions. Les notebooks distinguent les mesures, la revue visuelle identifiée et l’évaluation humaine statistique. Les rapports absents sont signalés comme absents ; aucun résultat ni annotation humaine n’est inventé.

Les poids publiés restent sur les destinations HF autorisées : [Base](https://huggingface.co/ethanzxv/Sharm-Image-2.1-VL4B-FP8), [Monster](https://huggingface.co/ethanzxv/Sharm-Image-2.1-Sprites-128-LoRA). Les notebooks vérifient les empreintes et la révision immuable après envoi, en conservant la visibilité prévue.

## Git et provenance

Les deux notebooks, configurations, données et rapports sont commités et poussés sur `main` avec la convention Angular. Nettoyer les sorties des cellules après conservation des preuves dans les rapports. Respecter `.gitattributes` pour les fichiers signés.

Les poids, environnements, caches, journaux et secrets restent hors Git. Les résultats finaux vont dans `Sharm-Image-2.1-VL4B/reports/`. Les anciennes sources exécutables restent consultables dans l’historique, notamment à `45be39a2ebac8105ce508849b8d1f5c32feecc90`. Les annexes conservent leur provenance ; les nouveaux runs calculent une empreinte du code des cellules, indépendante des sorties et compteurs d’exécution. Cette migration ne réattribue pas des entraînements historiques au nouveau code.
