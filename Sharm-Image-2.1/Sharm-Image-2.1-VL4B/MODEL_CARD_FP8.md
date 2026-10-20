---
license: other
license_name: qwen-research
license_link: LICENSE
pipeline_tag: text-to-image
base_model:
- Qwen/Qwen-Image-2.1
- Qwen/Qwen3-VL-4B-Instruct-FP8
language: [fr, en]
---
# Sharm Image 2.1 VL 4B FP8

## Le modèle en clair

**B signifie milliard de paramètres appris. Qwen3-VL 4B lit votre description ; Qwen-Image-2.1, environ 7,1 milliards de paramètres, génère l’image.** Le « 4B » désigne l’encodeur de texte, pas le générateur d’images.

FP8, BF16 et NF4 sont des formats de nombres de 8, 16 et 4 bits. L’encodeur de texte et les 232 principales couches du générateur utilisent le calcul FP8. Le projecteur, petit module qui adapte le texte au générateur, et le décodeur d’image gardent leur format d’origine. Les fichiers sont des PNG RGBA, avec un canal de transparence qui ne garantit pas un fond transparent.

## Test 1 : Qwen original face à Sharm

Les images montrent les mêmes consignes, nombres de départ aléatoire (graines) et bruits initiaux. À gauche se trouve Qwen Image 2.1 original : encodeur de texte Qwen3-VL 8B BF16 et générateur Qwen-Image-2.1 BF16. À droite se trouve Sharm : encodeur de texte Qwen3-VL 4B FP8, projecteur et générateur Qwen-Image-2.1 FP8.

Le test complet contient 16 paires, soit 32 images : 8 consignes et 2 graines par consigne. Sharm fusionne ou oublie parfois des objets ; le modèle original fait aussi des erreurs d’identité et de relation. **La précision des positions n’est validée pour aucun des deux.**

![Modèle original à gauche, Sharm à droite : deux consignes et leurs graines](comparisons/base-vs-sharm-v1/sheets/pairs-01.png)

![Comparaison des consignes verticales](comparisons/base-vs-sharm-v1/sheets/pairs-06.png)

[Voir les 16 paires, la galerie et les prompts exacts](comparisons/base-vs-sharm-v1/README.md).

## Test 2 : même Sharm, ancien format NF4 et FP8

Ici, l’encodeur, le projecteur, les consignes et les bruits sont identiques ; seul le format du générateur change. Sur ce passage, la durée médiane mesurée était de 26,43 s en NF4 et 20,90 s en FP8. La mémoire GPU mesurée passait de 5,73 à 8,95 Gio. Un passage ne prédit pas les performances sur une autre machine. Le dragon perd son fond transparent dans l’exemple FP8 ci-dessous.

![Comparaison NF4 et FP8 pour le dragon pixel art](comparisons/nf4-vs-fp8-v1/sheets/pairs-02.png)

[Voir les 16 paires, leurs mesures et leurs réglages](comparisons/nf4-vs-fp8-v1/README.md).

## Vérification du modèle publié

Le chargeur livré avec le modèle a généré deux PNG RGBA réels sur GPU. L’audit confirme le format FP8 des couches concernées. Les sujets sont reconnaissables, mais les fonds de ces deux images restent largement opaques : le détourage transparent n’est pas garanti.

![Génération FP8 du renard](comparisons/nf4-vs-fp8-v1/images/fox-1000-after-fp8.png)

![Génération FP8 du dragon](comparisons/nf4-vs-fp8-v1/images/dragon-1001-after-fp8.png)

[Réglages et vérification GPU](comparisons/nf4-vs-fp8-v1/README.md) · [Revue visuelle identifiée comme revue par agent](comparisons/nf4-vs-fp8-v1/REVIEW.md).

## Sources et limites

La Base reprend le projecteur antérieur validé. Le candidat spatial entraîné puis refusé n’est pas utilisé. Les images du comparatif principal changent à la fois l’encodeur, le projecteur et le générateur ; elles ne permettent pas d’attribuer chaque défaut au seul FP8.

Une revue par agent n’est pas une évaluation humaine statistique. Une différence de pixels n’est pas un taux d’erreur de placement. Voir les rapports complets pour les prompts, mesures, empreintes et limites.