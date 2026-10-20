# Qwen original et Sharm actuel : 16 paires

Chaque planche montre le **même prompt, la même graine et le même bruit de départ**. À gauche : Qwen Image 2.1 original, encodeur de texte 8B BF16 et générateur BF16. À droite : Sharm, encodeur Qwen3-VL 4B FP8, projecteur et générateur FP8.

Le test contient 8 consignes (gauche, droite, dessus, dessous en français et en anglais) et 2 graines par consigne : **16 paires, 32 PNG**, 512 × 512, 30 étapes.

**Résultat observé :** Sharm fusionne ou oublie parfois des objets demandés. Qwen original conserve plus souvent plusieurs silhouettes dans cet échantillon, mais se trompe aussi sur les identités, le nombre et les relations. La précision des positions n’est validée pour aucun modèle.

Ce test change plusieurs composants à la fois ; il ne permet pas d’attribuer l’écart au seul format FP8. La MSE est une différence de pixels, pas un pourcentage d’erreurs de placement. Revue identifiée comme agent ; pas d’évaluation humaine statistique.

[Galerie interactive](gallery.html) · [Prompts et réglages](protocol.json) · [Mesures](metrics.json) · [Revue complète](visual-review.json) · [SHA-256](manifest.json).

## Toutes les paires

![Prompts 1 et 2](sheets/pairs-01.png)

![Prompts 3 et 4](sheets/pairs-02.png)

![Prompts 5 et 6](sheets/pairs-03.png)

![Prompts 7 et 8](sheets/pairs-04.png)

![Prompts 9 et 10](sheets/pairs-05.png)

![Prompts 11 et 12](sheets/pairs-06.png)

![Prompts 13 et 14](sheets/pairs-07.png)

![Prompts 15 et 16](sheets/pairs-08.png)