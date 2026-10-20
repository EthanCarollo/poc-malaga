# Sharm Image 2.1 VL 4B FP8 — Base

**Point d'entrée unique : [Sharm_Image_2_1_VL4B.ipynb](Sharm_Image_2_1_VL4B.ipynb).**

Le notebook contient l'installation, les paramètres, le corpus spatial RGBA, les caches 8B/4B, l'apprentissage du projecteur, les courbes, la galerie appariée, la revue, l'export, les vraies générations du loader, la publication HF et l'archivage des résultats.

Pour utiliser la Base publiée, suivre la cellule **Générer avec la Base FP8 publiée** : elle charge l’encodeur de texte Qwen3-VL 4B FP8, le projecteur antérieur et le DiT Qwen-Image-2.1 FP8. L’encodeur enseignant Qwen3-VL 8B reste hors de l’export. Les sections de distillation reproduisent une expérience historique refusée ; elles ne sont pas nécessaires pour générer une image avec la Base publiée.

## Lire les noms et les formats

**B = milliard de paramètres**, c'est-à-dire de valeurs apprises. Cela décrit la taille d'un réseau, pas sa mémoire en Go.

| Modèle | Encodeur qui comprend le texte | DiT qui génère l'image | Sortie |
| --- | --- | --- | --- |
| **Sharm Image 2.1 VL4B FP8** | **Qwen3-VL 4B**, utilisé pour le texte, en FP8 ; projecteur d'alignement BF16 | **Qwen-Image-2.1, environ 7,1B** ; projections principales en FP8 natif | RGBA, avec canal alpha |
| **Sharm Iris PixelArt** | **Qwen3-VL 4B**, en BF16 dans le notebook Iris | **Iris-3B, environ 3B**, poids FP32 et calcul partiel BF16 | RGB ; aucun alpha natif |

Le **4B de Sharm appartient à Qwen3-VL**, son encodeur de texte. Le **3B d'Iris appartient au DiT**, son générateur. Iris possède donc aussi un encodeur Qwen3-VL 4B séparé. « VL » signifie vision et langage ; ces parcours utilisent le texte uniquement.

- **Encodeur / embeddings** : transforme le prompt, la description demandée, en tableaux de nombres qui conditionnent le générateur.
- **Projecteur** : adapte les représentations de l'encodeur Qwen3-VL 4B au format attendu par le générateur Qwen-Image-2.1.
- **DiT** (*Diffusion Transformer*) : réseau qui transforme progressivement le bruit initial en image. Sharm utilise ensuite un **VAE**, le décodeur qui convertit sa représentation comprimée en pixels. Iris calcule directement les pixels, sans VAE.
- **FP8 / BF16 / FP32 / NF4** : formats de nombres à 8, 16, 32 ou 4 bits. Ils changent la mémoire et parfois le résultat ; la taille en paramètres reste la même. Certains composants gardent une autre précision.
- **LoRA / fine-tuning** : petites matrices apprises / adaptation sur un dataset. Les tests du modèle Iris de base ne chargent aucun LoRA et ne lancent aucun entraînement.
- **Graine / pas de génération / CFG** : départ aléatoire reproductible / nombre d'itérations pour produire l'image / intensité du guidage par le texte. Les pas d'entraînement sont, eux, des mises à jour des poids.
- **RGB / RGBA** : couleurs seules / couleurs plus alpha, le canal de transparence. Un fichier RGBA peut quand même avoir un fond opaque.
- **MSE** : moyenne des différences au carré entre nombres ou pixels ; elle ne mesure pas directement la position correcte d'un objet.
- **Offload / cache / runtime** : transfert de composants entre RAM et GPU / résultats conservés pour éviter un recalcul / environnement logiciel d'exécution.

- **E4M3 / FP8 natif** : variante FP8 utilisée avec des facteurs d'échelle / calcul réellement effectué en FP8 par le GPU pour les projections indiquées, vérifié par audit.
- **Gelé / dataset / validation** : poids non modifiés / ensemble d'images et descriptions / exemples tenus à part pour contrôler le résultat.
- **SHA-256 / révision** : empreinte d'un fichier / identifiant d'une version du dépôt, pour retrouver exactement les octets testés.
- **HF / kernel / lien physique** : Hugging Face, qui héberge les modèles / processus Python qui exécute le notebook / plusieurs noms partageant le même fichier sur disque.

- **Autocast** : choix automatique du format de certaines opérations ; ici, calcul partiel BF16 avec les poids DiT Iris conservés en FP32.
- **RAM / VRAM / swap** : mémoire centrale du CPU / mémoire dédiée au GPU / espace disque où Linux peut déplacer des pages de RAM.

Le style **pixel art** concerne la grille visible, les contours et la palette. Le terme **pixel-space** d'Iris décrit son calcul direct des pixels ; le style pixel art est demandé par les prompts et doit être vérifié dans les sorties.


## Publication FP8 disponible

La Base avec **encodeur de texte Qwen3-VL 4B FP8 et DiT Qwen-Image-2.1 7,1B FP8** est disponible à [la révision `e6ef25ce`](https://huggingface.co/ethanzxv/Sharm-Image-2.1-VL4B-FP8/tree/e6ef25ce358f400781a9cd3c3e0fb8068312b599), avec les tests visibles directement dans son README. Ses poids et son loader sont identiques à `a0c0e168` ; seuls la carte, son manifeste et les nouvelles preuves de contrôle ont changé. Elle utilise le projecteur antérieur `a414685b`, sans le candidat spatial ci-dessous. Le parcours principal ajouté au notebook convertit, vérifie, publie et archive cet export ; aucune nouvelle distillation n'est requise.

Le loader exact a généré deux PNG RGBA réels sur GPU ; les poids et appels FP8 ont été contrôlés. Pic mesuré : 8,95 Gio sur RTX 5060 Ti. Les fonds restent largement opaques malgré la demande de transparence : le format RGBA fonctionne, pas une garantie de détourage. La [carte du modèle](MODEL_CARD_FP8.md) et les [rapports](reports/README.md) précisent les limites.

## Voir les comparatifs

- **[Modèle Qwen original / Sharm publié : 16 paires et 32 PNG](reports/base-vs-sharm-v1/README.md)**. Vraies features 8B BF16 et vrai DiT BF16, face au 4B FP8 + projecteur + DiT FP8. Huit prompts fixés avant génération, quatre relations, FR/EN, deux graines, 512 px, 30 pas, CFG 1 et même bruit initial. [Revue de chaque paire](reports/base-vs-sharm-v1/REVIEW.md).
- [Avant NF4 / après FP8 : 16 paires, 32 PNG et mesures](reports/comparison-nf4-fp8-a0c0e1680970-v1/README.md). Même encodeur 4B FP8, même projecteur, mêmes embeddings et bruit initial ; galerie HTML avec curseur et planches exhaustives.
- [8B BF16 / 4B FP8 : les 4 paires historiques](reports/encoder-8b-4b-1e912f5a/README.md), avec DiT NF4 partagé. Ce petit échantillon est distinct du nouveau comparatif de diffusion.

Le comparatif principal montre des pertes de nombre d'objets et des personnages hybrides avec Sharm ; la référence fait aussi des erreurs d'identité. Il ne valide ni la parité avec le 8B ni des positions parfaites. Les huit planches conservent tous les exemples.

Le code de génération, les contrôles, la galerie et la publication des preuves sont intégrés au notebook Base. Aucun entraînement supplémentaire ne démarre. Les écarts de pixels et les revues visuelles restent distincts d'une mesure humaine de précision spatiale.

## Résultat du 19 octobre 2026

Les 1 000 pas de warmup, les 1 500 pas de flow et les 288 générations sont terminés. **Le candidat flow est refusé après inspection des 144 images du candidat et des 144 homologues 8B.** Il répète des formes pour des prompts différents et déforme les sujets. Les 24 générations de diagnostic du warmup ont également été inspectées : leurs contours sont plus propres, mais la fidélité aux sujets reste insuffisante.

Voir le [compte rendu et les preuves](reports/README.md). Le profil spatial conservé reproduit cette expérience ; il ne constitue pas une recette validée. Ce candidat spatial n'a pas été publié. La publication FP8 ci-dessus reprend le projecteur antérieur. Les notebooks permettent d'archiver aussi un rejet, avec `ARCHIVE_OUTCOME = "rejected"`, sans fabriquer de révision HF.

Utiliser un kernel Linux CUDA sur la machine de calcul. Les drapeaux `RUN_...` sont initialement à `False` ; les activer lance réellement l'étape dans le kernel. La bibliothèque est intégrée à l'annexe du même fichier, sans script source externe.

Les paramètres se trouvent dans `configs/`, les dépendances figées dans `requirements-*.txt`, les licences dans `licenses/`, les rapports finaux dans `reports/`. Les poids et caches restent hors Git. L'artefact HF contient le package de chargement requis par son format, généré depuis les cellules de ce notebook.
