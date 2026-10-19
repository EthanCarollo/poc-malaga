# Revue visuelle du comparatif NF4 / FP8

**Méthode : `agent_visual_review`, par Codex.** Les 32 images ont été inspectées sur les 8 planches à leur résolution originale ; les aperçus sources y sont à 512 px natifs. Aucun exemple n'a été retiré. Cette revue n'est pas une annotation humaine ni une évaluation statistique de précision spatiale.

Les renards, grenouilles, cubes et paysages gardent généralement leurs sujets et leur composition, avec des écarts de proportions et de détails. Les changements peuvent être plus importants : le panneau et les livres changent d'échelle, et une bibliothèque apparaît à la place d'une pile dans un cas.

**Transparence : régression visible pour le dragon, graine 1001.** Le résultat NF4 possède un fond transparent ; le FP8 conserve un fond blanc opaque. La transparence n'est pas uniformément préservée par ce changement. La fleur reste une silhouette noire peu détaillée dans les deux variantes.

**Positions : résultat variable selon le cas.** L'étoile et le cercle se superposent en NF4 à la graine 1000 ; ils sont séparés verticalement en FP8. À la graine 1001, l'écart devient très faible. Ces exemples ne démontrent pas des positions parfaites.

Les détails pour les 16 paires et les SHA des 32 images sont dans [visual-review.json](visual-review.json). Les mesures sont dans [metrics.json](metrics.json), et toutes les images sont visibles dans [la galerie](gallery.html).

| Mesure sur ce passage | NF4 | FP8 |
| --- | ---: | ---: |
| Temps médian génération + décodage, hors première image | 26,43 s | 20,90 s |
| Pic VRAM allouée par PyTorch | 5,73 Gio | 8,95 Gio |

Ce passage mesure un temps médian environ 21 % plus court, avec environ 3,22 Gio de VRAM allouée supplémentaire. Le chargement des modèles et l'encodage texte sont exclus du temps ; il n'y a qu'un passage séquentiel par variante, sans garantie de vitesse. La MSE RGB moyenne sur blanc vaut 0,037987 et la MSE alpha moyenne 0,046231 sur [0,1] : ce sont des différences entre sorties, sans vérité terrain ni score de qualité.
