# 1 — Génération locale

Application Windows native de génération locale d’images, inspirée de la capture fournie : prompt et modèle à gauche, résultat à droite, avec les pages **Créer**, **Historique** et **Réglages**. L’interface est une fenêtre WPF avec des contrôles Windows ; aucun navigateur n’est lancé. Le moteur Python tourne dans WSL2 et communique avec cette fenêtre sur le loopback de la machine.

## Modèles

- **FLUX 3 Image Klein** — transformer 5,6B, variante FP8, 16 étapes. L’agrandisseur de prompt local est optionnel dans l’interface et désactivé par défaut pour limiter la mémoire.
- **Qwen-Image-2.1** — modèle image 7B, BF16, 40 étapes. Le moteur active le déchargement vers le CPU sur les GPU de moins de 20 Go.

Les moteurs tournent dans deux environnements Python distincts, car les versions de `transformers` demandées par les implémentations officielles diffèrent. L’API locale utilise seulement la bibliothèque standard Python. Les images produites et l’historique restent dans `outputs/`.

## Matériel à connaître

Le worker FLUX charge séparément l’encodeur de texte et le VAE sur CPU, puis ne place le DiT FP8 et ses tenseurs de travail sur le GPU pendant le débruitage. Le VAE revient ensuite sur CPU pour décoder l’image. L’agrandisseur PUP est désactivé dans ce mode mémoire.

Le pipeline FLUX 3 de référence mesurait **26 Go de VRAM à 1024 × 1024 en FP8, sans l’agrandisseur de prompt**, sur le matériel de test B200 de Black Forest Labs. L’interface avertit sous ce seuil et propose un format compact 512 × 512 pour essayer le chargement séquentiel. Cela ne garantit pas que le DiT FP8 et ses activations tiennent dans 8 Go ; la génération peut échouer ou être très lente selon la carte, la RAM système et le pilote. La documentation BFL conseille la variante INT8 dans ComfyUI pour les cartes de 8 Go. Qwen utilise le déchargement CPU fourni par Diffusers.

Les poids FLUX sont sous licence FLUX Kommunity. Il faut accepter les conditions sur Hugging Face avant le téléchargement. Les poids Qwen ont également leurs propres conditions. Ce projet ne télécharge rien automatiquement au lancement.

## Installation sous Windows avec WSL2

L’implémentation officielle FLUX dépend de paquets CUDA, Triton et vLLM conçus pour Linux. Utilise WSL2 avec une distribution Linux et un pilote NVIDIA compatible CUDA.

Dans un terminal WSL, depuis le dossier du projet :

```bash
cd /mnt/c/Users/eth/Documents/GitHub/poc/1-local-generation
chmod +x setup-wsl.sh download-models.sh
./setup-wsl.sh
```

Sur Hugging Face, connecte-toi, accepte les conditions FLUX 3 Image Klein et Qwen-Image-2.1, puis télécharge les poids :

```bash
hf auth login
./download-models.sh
```

Après l’installation des dépendances et le téléchargement des modèles, lance l’application Windows avec `launch-native.bat` (double-clic depuis l’Explorateur). Elle démarre l’API WSL en arrière-plan et l’arrête à la fermeture de la fenêtre :

```text
launch-native.bat
```

Le premier démarrage vérifie WSL2 et Python 3. Si une API POC Malaga est déjà lancée sur le port local 7860, l’application la réutilise sans l’arrêter à la fermeture.

Les premiers téléchargements sont volumineux et le premier lancement de chaque modèle charge ses poids en mémoire. Si le serveur ne trouve pas les interpréteurs, adapte les chemins dans `runtime-config.json` à partir de `runtime-config.example.json`.

## Sources des moteurs

- [Code d’inférence FLUX 3 Image Klein — Black Forest Labs](https://github.com/black-forest-labs/flux-3-image-klein)
- [Carte des poids FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein)
- [Qwen-Image-2.1 — QwenLM](https://github.com/QwenLM/Qwen-Image-2.1)

## Limites de ce POC

- Un moteur est chargé à la fois ; choisir l’autre modèle arrête le worker courant pour libérer le GPU.
- FLUX séquence maintenant encodeur, DiT et VAE, mais conserve le DiT FP8 sur GPU durant les 16 étapes ; ce mode expérimental ne promet pas une exécution sur toutes les cartes 8 Go.
- Génération texte-image et édition par une image de référence sont exposées dans une interface locale.
- Les dimensions disponibles sont volontairement limitées à quatre formats de démonstration.
- L’interface ne démarre pas elle-même les téléchargements des poids : ils sont déclenchés explicitement par `download-models.sh`.
