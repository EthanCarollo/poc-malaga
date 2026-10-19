# Revue du modèle original et de Sharm actuel

Revue réelle par **Codex (agent)** : 32 images à taille native, deux graines par prompt. Aucune annotation humaine ni précision spatiale statistique.

**La parité avec le 8B n'est pas démontrée.** Sur plusieurs scènes multi-objets, Sharm produit un seul personnage hybride ou des sujets superposés. La référence conserve plus souvent plusieurs silhouettes distinctes dans cet échantillon ; elle commet aussi des erreurs d'espèces, de nombre et de relation. Elle n'est pas une vérité terrain.

**Les positions fiables ne sont pas validées.** Une séparation verticale apparaît dans certains cas, notamment `above-en`, mais elle ne suffit pas lorsque l'identité des objets n'est pas respectée. Les demandes sans chevauchement échouent aussi dans plusieurs sorties Sharm. Les huit planches montrent les deux graines sans sélection.

**Le canal alpha fonctionne, la qualité du détourage reste variable.** Ici, les 16 PNG Sharm et 11 des 16 PNG de référence contiennent au moins un pixel totalement transparent. Ce critère très faible n'est pas un score de détourage : il ne contrôle ni les contours ni la transparence complète du fond. Certains fonds clairs ou motifs de damier sont générés dans l'image de référence. Le test CFG 3 du dragon plus bas montre aussi une perte de transparence en FP8.

## left-fr

La référence garde deux silhouettes côte à côte aux deux graines, sans fidélité fiable aux espèces. Sharm fusionne les sujets en une grande créature à la graine 1000 ; à 1001, deux créatures se chevauchent.

| Graine | Qwen original 8B/BF16 | Sharm actuel 4B/FP8 |
| --- | --- | --- |
| 1000 | Deux figures distinctes, jaune à gauche et gris-bleu à droite ; espèces demandées non fidèlement restituées. | Une grande figure jaune hybride ; les deux sujets distincts demandés sont perdus. |
| 1001 | Deux silhouettes séparées, oiseau jaune à gauche et créature sombre à droite ; un damier bleu est visible malgré un alpha minimal de 221. | Petite créature jaune à gauche et grande créature bleue à droite, avec recouvrement ; identités demandées non fiables. |

## left-en

Les trois objets restent distincts dans les images de référence, avec des espèces déformées. Sharm produit des créatures superposées et ne restitue pas les trois objets séparés attendus.

| Graine | Qwen original 8B/BF16 | Sharm actuel 4B/FP8 |
| --- | --- | --- |
| 1000 | Trois silhouettes : deux petites créatures orange encadrent une figure bleu clair ; les espèces Rhydon/Articuno ne sont pas fidèles. | Grande créature orange et créature ailée cyan se chevauchent fortement ; absence de trois sujets distincts. |
| 1001 | Trois créatures rouge/bleu/rouge côte à côte ; la structure à trois objets est visible, pas une validation des identités. | Grande créature bleue au premier plan et créature ailée au-dessus ; la composition change et les trois objets distincts sont perdus. |

## right-fr

La référence manque le troisième objet à la graine 1000 et produit trois silhouettes à 1001, avec des erreurs d'identité et de relation. Sharm donne un seul personnage aux deux graines.

| Graine | Qwen original 8B/BF16 | Sharm actuel 4B/FP8 |
| --- | --- | --- |
| 1000 | Deux créatures au lieu de trois, cyan à gauche et orange à droite ; l'objet cyan n'est pas à droite comme demandé. | Un seul quadrupède cyan hybride. |
| 1001 | Trois créatures de couleurs différentes, avec déformations ; la relation entre les objets nommés n'est pas établie. | Un seul quadrupède bleu avec une forme verte sur le dos ; aucune paire d'objets distincts à placer. |

## right-en

Deux silhouettes restent visibles dans la référence mais leurs identités et l'association à la relation ne sont pas fiables. Sharm remplace les deux objets par un seul personnage aux deux graines.

| Graine | Qwen original 8B/BF16 | Sharm actuel 4B/FP8 |
| --- | --- | --- |
| 1000 | Rocher bleu à gauche et figure rose à droite ; deux silhouettes, sans fidélité Ralts/Solrock démontrée. | Une seule créature rose à grandes oreilles. |
| 1001 | Figure ailée bleue à gauche et petite forme bleue à droite ; les sujets demandés ne sont pas fidèlement restitués. | Une seule créature bleue épineuse. |

## above-fr

La référence conserve deux formes séparées verticalement, avec des identités erronées. Sharm ne montre qu'un personnage à 1000 ; à 1001, un petit personnage est posé au-dessus d'un grand et les contours se touchent.

| Graine | Qwen original 8B/BF16 | Sharm actuel 4B/FP8 |
| --- | --- | --- |
| 1000 | Forme verte en haut et créature orange en bas, bien séparées ; pas de fidélité fiable à Plusle/Shellder. | Une grande créature jaune hybride ; les deux objets sont perdus. |
| 1001 | Petite forme cyan au-dessus d'une plus grande, séparées ; espèces non fidèles. | Petite créature verte sur une grande tête bleue ; disposition verticale mais contact/recouvrement et sujets non fidèles. |

## above-en

Les deux variantes gardent deux créatures disposées verticalement aux deux graines. Leur forme, leurs couleurs et leurs identités dérivent ; cette disposition seule ne valide pas les sujets demandés.

| Graine | Qwen original 8B/BF16 | Sharm actuel 4B/FP8 |
| --- | --- | --- |
| 1000 | Créature bleue en haut et quadrupède orange en bas, séparés ; identités déformées. | Créature cyan déformée au-dessus d'un personnage orange, sans recouvrement évident ; fidélité aux espèces non établie. |
| 1001 | Créature ailée cyan au-dessus d'un quadrupède blanc/orange, séparés ; identités déformées. | Créature ailée cyan au-dessus d'une créature dorée ; séparation visible, mais apparence du sujet du bas fortement modifiée. |

## below-fr

La référence montre deux figures verticales mais ne restitue pas fidèlement Castform/Paras et leur association à la consigne. Sharm n'en conserve qu'une aux deux graines.

| Graine | Qwen original 8B/BF16 | Sharm actuel 4B/FP8 |
| --- | --- | --- |
| 1000 | Figure jaune en haut et créature blanche en bas ; les identités et la relation demandée ne sont pas validées. | Un seul personnage vert clair/jaune à grandes oreilles. |
| 1001 | Petite créature bleu clair au-dessus d'une grande forme verte ; les sujets demandés ne sont pas fidèles. | Une seule tête/créature cyan épineuse flottante. |

## below-en

La référence conserve deux silhouettes verticales, sans fidélité suffisante aux espèces. Sharm donne un seul personnage à 1000, puis deux à 1001 ; la relation entre Hoppip et Cacnea n'est pas validée.

| Graine | Qwen original 8B/BF16 | Sharm actuel 4B/FP8 |
| --- | --- | --- |
| 1000 | Personnage jaune au-dessus d'une forme orange végétale ; les sujets nommés ne sont pas fidèlement restitués. | Un seul personnage rond jaune/vert ; absence de deux objets distincts. |
| 1001 | Petite créature verte au-dessus d'une grande forme verte/violette ; deux silhouettes, pas de validation des espèces ou de leur ordre. | Petite créature crème au-dessus d'une figure verte feuillue ; deux silhouettes séparées mais identités non fidèles. |

