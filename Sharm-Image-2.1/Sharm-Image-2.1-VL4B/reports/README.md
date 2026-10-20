# Résultats Sharm Image 2.1

Cette page résume les tests. Les liens ouvrent les images complètes, les réglages et les rapports.

## Tests avant / après

| Comparaison | Résultat |
| --- | --- |
| [Qwen original / Sharm FP8 : 16 paires](base-vs-sharm-v1/README.md) | Sharm mélange ou oublie parfois des objets. Qwen original se trompe aussi. Aucun des deux ne valide des positions parfaites. |
| [Sharm NF4 / Sharm FP8 : 16 paires](comparison-nf4-fp8-a0c0e1680970-v1/README.md) | Même prompt, même encodeur et mêmes bruits. Sur ce seul passage, la médiane mesurée est 26,43 s en NF4 et 20,90 s en FP8 ; la VRAM passe de 5,73 à 8,95 Gio. |
| [Ancien encodeur 8B / 4B FP8 : 4 paires](encoder-8b-4b-1e912f5a/README.md) | Petit test historique ; le générateur y était partagé en NF4. |

Le comparatif principal utilise 8 consignes, 2 graines par consigne, 512 × 512 pixels, 30 étapes et le même bruit initial. Il change l’encodeur, son projecteur et le format du générateur à la fois : on ne peut pas attribuer chaque différence à FP8 seul.

## Base publiée

La [Base FP8 publiée](fp8-a0c0e1680970/summary.json) a été vérifiée sur GPU avec son vrai loader : deux PNG RGBA à 512 × 512, 30 étapes et CFG 3. Le calcul FP8 et les fichiers ont été contrôlés. Les fonds de ces deux images restent largement opaques ; un PNG RGBA ne signifie pas que le détourage est réussi.

- [Images et revue visuelle identifiée](fp8-a0c0e1680970/fp8-visual-review.json)
- [Contrôles des poids et du calcul FP8](fp8-a0c0e1680970/fp8-validation.json)
- [SHA-256 des fichiers de preuve](fp8-a0c0e1680970/manifest.json)

## Essai spatial refusé

L’essai de 2 500 mises à jour a été terminé puis refusé après inspection de ses générations. Il n’est pas dans la Base publiée. [Rapport, limites et toutes les images](rejected-7a39a8cc674e/summary.json).

Les revues par agent sont identifiées comme telles ; elles ne remplacent pas une évaluation humaine statistique. Une baisse de MSE ne prouve pas que les objets sont à la bonne place.