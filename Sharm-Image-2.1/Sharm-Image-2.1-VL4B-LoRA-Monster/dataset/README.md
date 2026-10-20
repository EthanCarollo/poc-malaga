# Données Monster pour Sharm Image

La version prête à utiliser se trouve dans prepared/monster-clean-v2. Elle contient **4 481 sprites RGBA uniques de 128 × 128**, représentant **649 espèces**. Les 25 planches sources restent inchangées dans raw.

## Répartition

| | Nombre |
| --- | ---: |
| Style Emerald | 1 534 |
| Style Black/White | 2 947 |
| Vue de face / de dos | 3 011 / 1 470 |
| Palette normale / shiny | 2 409 / 2 072 |
| Entraînement | 4 084 images, 583 espèces |
| Validation | 397 images, 66 espèces |

Une espèce entière appartient à un seul split. Les doublons visuels ne sont stockés qu’une fois ; les variantes équivalentes sont gardées dans les métadonnées.

## Fichiers importants

- raw : les 25 planches d’origine, sans modification.
- images_128 : sprites RGBA validés.
- train_256 et train_512 : agrandissements entiers au plus proche voisin, sans lissage.
- manifest.jsonl : identité, forme, sexe, style, vue, palette, provenance et légendes de chaque image.
- definitions.json : noms français/anglais et définitions des espèces et formes.
- sources.json et dataset.json : origines, révisions et empreintes SHA-256.
- audit.json : résultats des contrôles d’images et de métadonnées.
- review : planches de contrôle des découpes.

Les légendes décrivent uniquement ce qui peut être déduit des sprites. Les couleurs shiny ne sont pas remplacées par des couleurs normales attendues. Les histoires et capacités du Pokédex ne servent pas d’étiquettes d’image.

## Préparation et vérifications

Les cellules d’une planche sont isolées, comparées aux références disponibles, puis enregistrées avec leur transparence. Le personnage garde ses proportions et est centré sur la grille 128 × 128. Les contrôles vérifient les dimensions, l’alpha, les marges, les couleurs sources, les empreintes et l’absence d’espèces communes entre train et validation.

Les correspondances ne sont pas toutes des copies pixel par pixel d’une référence indépendante. L’audit conserve la preuve et la méthode par image ; une revue par agent des découpes ne constitue pas une annotation humaine.

Les PNG sources, manifestes et définitions sont versionnés. Le fichier .gitattributes préserve leurs octets et leurs SHA-256 sur Windows et Linux.