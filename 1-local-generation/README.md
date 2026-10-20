# Générer des images sur Windows

Cette application ouvre une fenêtre Windows pour écrire un prompt, choisir un modèle, générer une image et la retrouver dans l’historique. Le calcul tourne dans WSL2 ; aucun navigateur n’est requis.

## Modèles

| Modèle | Taille du générateur | Format et génération |
| --- | --- | --- |
| FLUX 3 Image Klein | 5,6 milliards de paramètres | FP8, 16 étapes |
| Qwen-Image-2.1 | Environ 7 milliards de paramètres | BF16, 40 étapes ; peut utiliser la mémoire du CPU |

Un seul modèle utilise le GPU à la fois. Les images et l’historique restent dans le dossier outputs.

## Installation

Dans WSL2, depuis ce dossier :

    chmod +x setup-wsl.sh download-models.sh
    ./setup-wsl.sh

Connectez-vous à Hugging Face et acceptez les licences des modèles, puis téléchargez les poids :

    hf auth login
    ./download-models.sh

Lancez ensuite launch-native.bat depuis Windows. Au premier démarrage, l’application prépare le service local et charge les modèles ; elle ne télécharge pas les poids sans action.

## Matériel et limites

FP8 et BF16 sont des formats de nombres de 8 et 16 bits ; ils ne désignent pas la mémoire nécessaire. Sur le matériel de référence, FLUX demande 26 Go de mémoire graphique en 1024 × 1024. Le mode expérimental 512 × 512 essaie de réduire cette quantité, mais ne garantit pas le fonctionnement sur une carte de 8 Go. Le moteur Qwen peut déplacer des parties du modèle dans la mémoire du CPU ; la génération devient alors plus lente.

L’application peut créer une image ou modifier une image de référence. Les dimensions proposées sont limitées à quatre formats de démonstration.

## Sources

- [Code FLUX 3 Image Klein](https://github.com/black-forest-labs/flux-3-image-klein)
- [Poids FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein)
- [Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1)