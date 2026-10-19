# Sharm Image 2.1 VL 4B FP8 — Base

**Point d'entrée unique : [Sharm_Image_2_1_VL4B.ipynb](Sharm_Image_2_1_VL4B.ipynb).**

Le notebook contient l'installation, les paramètres, le corpus spatial RGBA, les caches 8B/4B, l'apprentissage du projecteur, les courbes, la galerie appariée, la revue, l'export, les vraies générations du loader, la publication HF et l'archivage des résultats.

Le VL reste gelé en FP8. Le profil courant entraîne le projecteur, sans LoRA du diffuseur. Le 8B n'est pas inclus dans la Base exportée. Les limites des pertes numériques et de l'évaluation spatiale sont expliquées avant les cellules de revue.

## Publication FP8 disponible

La Base avec **text encoder et DiT FP8** est publiée à [la révision `a0c0e168`](https://huggingface.co/ethanzxv/Sharm-Image-2.1-VL4B-FP8/tree/a0c0e168097024cecd8c43f68004d1bedbab32df). Elle utilise le projecteur antérieur `a414685b`, sans le candidat spatial ci-dessous. Le parcours principal ajouté au notebook convertit, vérifie, publie et archive cet export ; aucune nouvelle distillation n'est requise.

Le loader exact a généré deux PNG RGBA réels sur GPU ; les poids et appels FP8 ont été contrôlés. Pic mesuré : 8,95 Gio sur RTX 5060 Ti. Les fonds restent largement opaques malgré la demande de transparence : le format RGBA fonctionne, pas une garantie de détourage. La [carte du modèle](MODEL_CARD_FP8.md) et les [rapports](reports/README.md) précisent les limites.

## Résultat du 19 octobre 2026

Les 1 000 pas de warmup, les 1 500 pas de flow et les 288 générations sont terminés. **Le candidat flow est refusé après inspection des 144 images du candidat et des 144 homologues 8B.** Il répète des formes pour des prompts différents et déforme les sujets. Les 24 générations de diagnostic du warmup ont également été inspectées : leurs contours sont plus propres, mais la fidélité aux sujets reste insuffisante.

Voir le [compte rendu et les preuves](reports/README.md). Le profil spatial conservé reproduit cette expérience ; il ne constitue pas une recette validée. Ce candidat spatial n'a pas été publié. La publication FP8 ci-dessus reprend le projecteur antérieur. Les notebooks permettent d'archiver aussi un rejet, avec `ARCHIVE_OUTCOME = "rejected"`, sans fabriquer de révision HF.

Utiliser un kernel Linux CUDA sur la machine de calcul. Les drapeaux `RUN_...` sont initialement à `False` ; les activer lance réellement l'étape dans le kernel. La bibliothèque est intégrée à l'annexe du même fichier, sans script source externe.

Les paramètres se trouvent dans `configs/`, les dépendances figées dans `requirements-*.txt`, les licences dans `licenses/`, les rapports finaux dans `reports/`. Les poids et caches restent hors Git. L'artefact HF contient le package de chargement requis par son format, généré depuis les cellules de ce notebook.
