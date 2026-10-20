# Projets d’images IA

Ce dépôt réunit des prototypes locaux et leurs résultats vérifiés.

## Les projets

- **Sharm Image 2.1** — Qwen3-VL 4B comprend le texte ; Qwen-Image-2.1, environ 7,1 milliards de paramètres, génère l’image. Le text encoder et les principales couches du DiT sont en FP8. [Tests avant/après](Sharm-Image-2.1/Sharm-Image-2.1-VL4B/reports/README.md).
- **Sharm Iris PixelArt** — Iris-3B génère l’image et Qwen3-VL 4B comprend le texte. Le modèle de base a 16 tests visibles ; le fine-tuning Monster est suivi dans son [dépôt séparé](https://github.com/EthanCarollo/Sharm-Iris-PixelArt).

Le **4B désigne toujours l’encodeur de texte Qwen3-VL** dans Sharm Image. Pour Iris, le **3B désigne le générateur Iris**.

## Utiliser Sharm

Les deux parcours ont leur propre notebook autonome :

- [Générer et vérifier la Base Sharm](Sharm-Image-2.1/Sharm-Image-2.1-VL4B/Sharm_Image_2_1_VL4B.ipynb)
- [Préparer le LoRA Monster de Sharm Image](Sharm-Image-2.1/Sharm-Image-2.1-VL4B-LoRA-Monster/Sharm_Image_2_1_VL4B_LoRA_Monster.ipynb)

Les calculs s’exécutent sur la machine du kernel ; les opérations GPU et publication sont à déclencher manuellement.

## Modèles et preuves

- [Sharm Image 2.1 VL 4B FP8 sur Hugging Face](https://huggingface.co/ethanzxv/Sharm-Image-2.1-VL4B-FP8)
- [Sharm Iris PixelArt et ses tests](https://huggingface.co/ethanzxv/Sharm-Iris-PixelArt)
- [Données Monster et leurs définitions](Sharm-Image-2.1/Sharm-Image-2.1-VL4B-LoRA-Monster/dataset/README.md)

Les poids, mots de passe et caches restent hors de Git. Les résultats et données versionnés conservent leurs empreintes.