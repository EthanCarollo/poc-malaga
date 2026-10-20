# Sharm Image 2.1 VL 4B FP8

**Pour générer une image, ouvrir [le notebook Base](Sharm_Image_2_1_VL4B.ipynb).**

## Quel modèle est-ce ?

- **Qwen3-VL 4B** lit le prompt. C’est le « 4B » du nom.
- **Qwen-Image-2.1, environ 7,1 milliards de paramètres**, génère les pixels.
- Le text encoder et ses principales couches de génération utilisent le format FP8. Certaines autres parties gardent leur format d’origine.
- La sortie est un PNG RGBA. Cela ne garantit pas un fond transparent.

La Base est publiée sur [Hugging Face](https://huggingface.co/ethanzxv/Sharm-Image-2.1-VL4B-FP8). Le README du modèle montre des images réelles et leurs comparatifs.

## Tests visibles

- [Qwen original contre Sharm : 16 paires](reports/base-vs-sharm-v1/README.md). Huit consignes, deux graines par consigne, même bruit de départ. Sharm perd ou mélange parfois des objets ; le modèle original fait aussi des erreurs.
- [Ancien format NF4 contre FP8 : 16 paires](reports/comparison-nf4-fp8-a0c0e1680970-v1/README.md). C’est le même encodeur de texte, le même projecteur et le même bruit.
- [Quatre anciennes paires d’encodeurs](reports/encoder-8b-4b-1e912f5a/README.md).

Les tests ne prouvent ni une parité avec le modèle 8B ni une précision parfaite des positions. Les arrière-plans alpha restent parfois opaques.

## Utiliser le notebook

Le notebook contient ses propres étapes. Ouvrir un kernel Python avec CUDA, lire les réglages, puis activer manuellement l’étape voulue. Les lancements sont désactivés par défaut. La génération avec la Base publiée ne demande pas de nouvel entraînement.