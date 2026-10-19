# Sharm Image 2.1 VL 4B FP8 — Base

**Point d'entrée unique : [Sharm_Image_2_1_VL4B.ipynb](Sharm_Image_2_1_VL4B.ipynb).**

Le notebook contient l'installation, les paramètres, le corpus spatial RGBA, les caches 8B/4B, l'apprentissage du projecteur, les courbes, la galerie appariée, la revue, l'export, les vraies générations du loader, la publication HF et l'archivage des résultats.

Le VL reste gelé en FP8. Le profil courant entraîne le projecteur, sans LoRA du diffuseur. Le 8B n'est pas inclus dans la Base exportée. Les limites des pertes numériques et de l'évaluation spatiale sont expliquées avant les cellules de revue.

Utiliser un kernel Linux CUDA sur la machine de calcul. Les drapeaux `RUN_...` sont initialement à `False` ; les activer lance réellement l'étape dans le kernel. La bibliothèque est intégrée à l'annexe du même fichier, sans script source externe.

Les paramètres se trouvent dans `configs/`, les dépendances figées dans `requirements-*.txt`, les licences dans `licenses/`, les rapports finaux dans `reports/`. Les poids et caches restent hors Git. L'artefact HF contient le package de chargement requis par son format, généré depuis les cellules de ce notebook.
