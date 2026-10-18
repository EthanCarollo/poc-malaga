# Sharm : VL 4B FP8 gelé, projecteur et LoRA sprites

Le projet comprend deux entraînements successifs : aligner les représentations du Qwen3-VL 4B FP8 gelé avec le diffuseur Qwen-Image-2.1 grâce à un projecteur entraînable, puis entraîner un LoRA sprites sur cette base figée. Le [plan détaillé](PLAN_SHARM.md) documente les choix et les limites.

## Installer et lancer

Depuis la racine du dépôt, sur WSL avec le runtime Qwen et CUDA :

```bash
./2-LoRA-qwen/setup_sharm.sh
./2-LoRA-qwen/start_training.sh
```

Le lancement se détache du terminal et exécute les étapes dans des processus séparés. Il reprend les shards et checkpoints déjà terminés. Une seconde invocation détecte la chaîne déjà active.

Pour lire son état :

```bash
./1-local-generation/.venv-qwen/bin/python ./2-LoRA-qwen/run_training.py --status
```

Les journaux sont dans `artifacts/logs/`. `artifacts/run-status.json` contient le PID, l’étape active, la révision de code et les versions installées. Les fichiers `cache-*-status.json`, `projector-status.json` et `lora-status.json` donnent la progression de chaque calcul. Garder WSL et la machine en fonctionnement pendant les entraînements.

## Chaîne exécutée

1. Vérifier les poids source aux révisions figées et préparer les sprites locaux.
2. Préparer 20 000 prompts : 18 115 prompts DiffusionDB (CC0-1.0), 1 500 compositions rédigées par templates et 385 légendes sprites. La répartition enregistrée contient 17 951 exemples d’entraînement et 2 049 de validation.
3. Contrôler le 4B FP8 réel. Le checkpoint officiel quantifie le décodeur et garde des exceptions déclarées en BF16, dont la branche vision. L’encodage du 4B se fait un prompt à la fois pour conserver les mêmes conditions de calcul entre cache et génération.
4. Extraire les cibles du VL 8B d’origine et les features du 4B en deux passes distinctes. Le 8B sert uniquement d’enseignant hors ligne. Les états extraits précèdent la dernière RMSNorm et conservent les tokens et masques attendus.
5. Entraîner une projection linéaire pendant 1 000 pas, puis le MLP résiduel pendant 5 000 pas. Seul le projecteur reçoit des gradients. Son candidat retenu est enregistré en BF16 et évalué après cette conversion.
6. Comparer les embeddings, les prédictions du diffuseur et des générations réservées. Si les contrôles échouent, la chaîne s’arrête avant le LoRA et écrit la cause dans son état.
7. Exporter et recharger la base Sharm complète, avec son 4B FP8, son projecteur et les composants de diffusion.
8. Pré-calculer les embeddings projetés et les latents RGBA des sprites, puis entraîner le LoRA pendant 500 pas, batch 1 avec accumulation 4, rang/alpha 8, diffuseur NF4 et optimiseur 8-bit.
9. Produire des échantillons sur des espèces réservées, en 512 px et en 128 px après réduction nearest-neighbor. Les deux variantes d’une espèce appartiennent au même ensemble pour éviter une fuite de validation.

La première base est expérimentale et validée uniquement pour le texte vers image. L’édition par image nécessite encore un alignement et une évaluation multimodaux ; le loader refuse cette fonction dans cette version. Les contrôles automatiques de génération ne remplacent pas la revue visuelle des images et de l’identité des sprites.

## Confidentialité et sorties

Les modèles, caches et données sont hors Git. Cette exécution conserve tous les poids localement : **l’envoi HF est différé à la demande de l’utilisateur**. Les destinations prévues restent privées :

- `ethanzxv/Sharm-Image-2.1-VL4B-FP8` ;
- `ethanzxv/Sharm-Image-2.1-Sprites-128-LoRA`.

Les exports locaux sont `1-local-generation/weights/sharm-image-2.1-vl4b-fp8/` et `1-local-generation/weights/sharm-image-2.1-sprites-128-lora/`. Leur révision de projecteur est enregistrée afin de vérifier la compatibilité du LoRA. Le loader fourni est `sharm_image21.pipeline.SharmImage21Pipeline` ; le chargement stock ne suffit pas pour ajouter le projecteur.

La licence Qwen Research et les attributions d’origine sont conservées. Aucun jeton HF n’est nécessaire pour entraîner depuis les poids publics. Une authentification locale du compte `ethanzxv` sera nécessaire pour envoyer ultérieurement les exports en privé.

## Dataset sprites

Les trois planches locales `21228.png`, `21229.png`, `21230.png` fournissent 770 sprites non vides de 64 × 64 : 385 espèces, avec deux variantes chacune. Les deux cases Celebi sont vides. La préparation enlève tous les pixels exacts `#A5EBFF`, y compris les îlots de fond isolés, et conserve le RGBA. Les copies 512 px sont agrandies en nearest-neighbor.

```bash
./1-local-generation/.venv-qwen/bin/python ./2-LoRA-qwen/prepare_dataset.py --training-size 512
```

Les planches source doivent être présentes localement ; elles sont exclues de Git. Le répertoire `dataset/` et les traits PokéAPI mis en cache sont aussi exclus. Les sources 64 px ne créent pas de détails natifs supplémentaires lors d’un agrandissement vers 128 px.

Pour relancer seulement le LoRA après validation de la base et préparation des caches :

```bash
./2-LoRA-qwen/train_lora.sh
```

Le runtime de référence utilise PyTorch 2.14.1+cu130, Transformers 5.20.1, Diffusers à la révision figée et `kernels==0.17.2`. Le test `verify_training.py --diffusion` vérifie un apprentissage synthétique du projecteur, la sauvegarde/relecture des caches et un vrai passage avant/arrière du diffuseur NF4 avec adaptateurs LoRA.
