# Sharm Image 2.1 VL 4B LoRA Monster

**Point d'entrée unique : [Sharm_Image_2_1_VL4B_LoRA_Monster.ipynb](Sharm_Image_2_1_VL4B_LoRA_Monster.ipynb).**

Le **4B** du nom désigne l’**encodeur de texte Qwen3-VL**, distinct du **DiT Qwen-Image-2.1 d’environ 7,1 milliards de paramètres** qui génère l’image. **LoRA Monster** désigne de petites matrices à entraîner sur ce générateur pour le style des monstres ; ce n’est pas un nouvel encodeur 4B. [Définitions des composants, formats et mesures](../README.md#lire-les-noms-et-les-formats).

La Base publiée utilise l’encodeur et les projections principales du DiT en FP8. La recette Monster historique charge son DiT en NF4 à partir de poids BF16 figés ; sa compatibilité avec le DiT FP8 publié n’est pas encore validée. Les sorties et les comparatifs doivent toujours identifier la révision exacte de leur Base.

Le notebook contient l'installation, l'inspection et la préparation du dataset, le choix d'une Base exacte, la validation de compatibilité, les caches, l'entraînement LoRA, les comparaisons avant/après en RGBA, la revue, la vérification du loader et la publication liée à une révision HF immuable de la Base.

Le dataset versionné contient **25 PNG sources, 4 481 sprites RGBA uniques de 128 × 128 et 649 espèces**. Les définitions, légendes, formes, palettes, vues, styles et provenances sont conservés. Le split par espèce évite de répartir les variantes d'une même espèce entre entraînement et validation.

Les lancements réels restent à `False` dans Git. L'utilisateur peut les activer manuellement dans le kernel de sa machine distante. Aucun entraînement ni pré-calcul GPU Monster n'a été lancé pendant la préparation.

Le code partagé et les algorithmes sont intégrés à l'annexe du fichier ; aucun script source ou autre notebook n'est requis. Les dépendances figées communes se trouvent dans le dossier Base voisin. Les poids de la Base sont sélectionnés explicitement, avec le SHA du projecteur ; l'ancien LoRA ne devient pas compatible par simple changement de chemin.
