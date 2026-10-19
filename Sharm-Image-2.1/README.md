# Sharm Image 2.1 — deux notebooks

Le projet s’exécute depuis **deux notebooks autonomes**, chacun dans son dossier. Ouvrir le fichier voulu avec un kernel Python sur la machine Linux CUDA où les calculs doivent tourner. Les opérations s’exécutent localement dans ce kernel ; aucun accès SSH n’est intégré aux notebooks.

| Modèle | Notebook | Contenu |
|---|---|---|
| **Sharm Image 2.1 VL 4B FP8 — Base** | [Notebook Base](Sharm-Image-2.1-VL4B/Sharm_Image_2_1_VL4B.ipynb) | Installation, données RGBA, distillation 8B → 4B, caches, entraînement du projecteur, comparaisons, export, inférence, publication et rapports |
| **Sharm Image 2.1 VL 4B LoRA Monster** | [Notebook Monster](Sharm-Image-2.1-VL4B-LoRA-Monster/Sharm_Image_2_1_VL4B_LoRA_Monster.ipynb) | Dataset, définitions, Base retenue, caches, entraînement LoRA, comparaisons avant/après, contrôle RGBA et publication |

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


## Base publiée : text encoder et DiT FP8

Le [README HF détaillé, révision `812304af`](https://huggingface.co/ethanzxv/Sharm-Image-2.1-VL4B-FP8/tree/812304af8f902f6ab0138b66b4b8b336125780be) affiche les sorties Sharm généralistes, toutes les paires original/Sharm et les diagnostics. Cette révision conserve les poids et le loader de `a0c0e168` et renouvelle leur vérification GPU.

La révision [HF `a0c0e168097024cecd8c43f68004d1bedbab32df`](https://huggingface.co/ethanzxv/Sharm-Image-2.1-VL4B-FP8/tree/a0c0e168097024cecd8c43f68004d1bedbab32df) contient le VL 4B FP8, le projecteur précédemment publié et le DiT en FP8 E4M3 avec échelles par ligne. Les 232 couches linéaires éligibles du DiT utilisent aussi un calcul FP8 natif sur la RTX 5060 Ti ; normes, biais, VAE et projecteur conservent leur précision. Les anciens shards BF16 du DiT ont été retirés de cette révision.

Les 52 fichiers ont été vérifiés après envoi et le dépôt reste privé. Le loader exact a produit deux PNG RGBA réels, inspectés par agent, avec un pic d'environ 8,95 Gio de VRAM. Leurs sujets sont reconnaissables, mais les fonds restent largement opaques : les détourages transparents propres ne sont pas validés. Voir les [preuves et limites](Sharm-Image-2.1-VL4B/reports/README.md).

Les [comparatifs avant / après](Sharm-Image-2.1-VL4B/reports/README.md#comparatifs-avant--après) mettent en premier le **modèle original 8B BF16 + DiT BF16 face à Sharm 4B FP8 + DiT FP8** (16 paires, mêmes bruits initiaux). Ils distinguent ensuite le diagnostic NF4 → FP8 à conditionnement identique, les 4 paires historiques d'encodeurs et le candidat spatial refusé. Le notebook Base contient leur génération, galerie avec curseur, mesures et archivage.

Le candidat spatial refusé reste archivé ; il n'est pas utilisé dans cette Base. Le nouveau projet pixel art Iris se trouve dans un dépôt séparé : [Sharm-Iris-PixelArt](https://github.com/EthanCarollo/Sharm-Iris-PixelArt), avec un notebook et les données de traits versionnées. Aucun nouvel entraînement Monster ou Iris n'a été lancé.

```text
Sharm-Image-2.1/
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

Aucun nouvel entraînement LoRA n’est lancé pendant la préparation de ces fichiers. La chaîne Base spatiale historique est terminée et son candidat est refusé. Ses sources, caches et checkpoints restent conservés ; la conversion FP8 publiée reprend le projecteur antérieur validé techniquement.

## Modèles et données

La Base comprend le générateur Qwen Image 2.1 avec DiT FP8, le VL 4B FP8 gelé et son projecteur BF16. Le 8B BF16 est un enseignant hors ligne. Le LoRA Monster historique reste lié à sa révision de Base d'origine ; sa compatibilité avec ce DiT FP8 n'est pas validée. Pour son entraînement, conserver les composants BF16 publics à la révision figée avant préparation NF4.

Les 25 PNG sources et le dataset Monster validé sont dans le dossier Monster : 4 481 images RGBA uniques, 649 espèces, définitions FR/EN, légendes, splits par espèce et rapports. Le déplacement conserve les octets des PNG et des manifestes signés. La provenance de la recette historique reste dans les anciens commits Git.

Une baisse de perte numérique ne donne pas un taux d’erreur des positions. Les notebooks distinguent les mesures, la revue visuelle identifiée et l’évaluation humaine statistique. Les rapports absents sont signalés comme absents ; aucun résultat ni annotation humaine n’est inventé.

Les poids publiés restent sur les destinations HF autorisées : [Base](https://huggingface.co/ethanzxv/Sharm-Image-2.1-VL4B-FP8), [Monster](https://huggingface.co/ethanzxv/Sharm-Image-2.1-Sprites-128-LoRA). Les notebooks vérifient les empreintes et la révision immuable après envoi, en conservant la visibilité prévue.

## Git et provenance

Les deux notebooks, configurations, données et rapports sont commités et poussés sur `main` avec la convention Angular. Nettoyer les sorties des cellules après conservation des preuves dans les rapports. Respecter `.gitattributes` pour les fichiers signés.

Les poids, environnements, caches, journaux et secrets restent hors Git. Les résultats finaux vont dans `Sharm-Image-2.1-VL4B/reports/`. Les anciennes sources exécutables restent consultables dans l’historique, notamment à `45be39a2ebac8105ce508849b8d1f5c32feecc90`. Les annexes conservent leur provenance ; les nouveaux runs calculent une empreinte du code des cellules, indépendante des sorties et compteurs d’exécution. Cette migration ne réattribue pas des entraînements historiques au nouveau code.

## Espace disque sur la machine distante

Le nettoyage des seuls caches d'installation pip/uv a libéré **8,52 Gio**, de 4,30 à 12,81 Gio disponibles au moment de l'opération. Les poids, features enseignant/étudiant, datasets, checkpoints et environnements sont conservés. Les exports de documentation réutilisent les poids identiques par liens physiques. [Rapport exact](Sharm-Image-2.1-VL4B/reports/storage-maintenance.json).

L'étape optionnelle de maintenance est intégrée au notebook Base et désactivée par défaut. Elle vérifie les chemins et l'absence d'installateur actif avant de toucher aux caches ; elle ne supprime aucun modèle HF.
