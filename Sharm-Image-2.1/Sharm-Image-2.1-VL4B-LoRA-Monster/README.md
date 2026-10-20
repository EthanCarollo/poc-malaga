# Sharm Image 2.1 LoRA Monster

Ce notebook prépare un petit adaptateur LoRA qui apprend le style pixel art des monstres.

- Le **4B** est l’encodeur Qwen3-VL qui lit le texte.
- Le générateur est **Qwen-Image-2.1, environ 7,1 milliards de paramètres**.
- Le corpus validé contient **4 481 sprites RGBA uniques de 649 espèces**. Voir les [données et définitions](dataset/README.md).
- Le split garde les espèces de validation à part.

Ouvrir le [notebook autonome](Sharm_Image_2_1_VL4B_LoRA_Monster.ipynb) sur la machine de calcul. L’entraînement et les publications restent désactivés dans le fichier versionné et ne partent qu’après activation manuelle.

La recette historique utilise des poids NF4 précis ; elle ne devient pas compatible avec une autre Base simplement en changeant son chemin. Le notebook vérifie la révision et le projecteur avant toute étape.

Ce LoRA Qwen est distinct du projet [Iris PixelArt Monster](https://huggingface.co/ethanzxv/Sharm-Iris-PixelArt-Monster).