# Comparatif historique des encodeurs : 8B BF16 / 4B FP8

**4 paires, 8 PNG originaux** et leurs aperçus, récupérés avec leurs octets exacts depuis la révision HF `1e912f5a1e5df40bdaf927abf3e4220d73781321`. Aucune nouvelle génération et aucun entraînement. Les SHA et l'origine sont dans [sources.json](sources.json).

Le DiT était exécuté en NF4 dans les deux variantes. L'encodeur 8B BF16 est comparé au 4B FP8 avec son projecteur historique `a414685b`. Ce petit échantillon historique n'est pas une validation de la nouvelle diffusion FP8, ni une statistique de précision spatiale. La galerie du candidat spatial refusé est encore une autre expérience.

[Paramètres et mesures d'origine](base-validation.json), [manifeste de génération d'origine](comparisons/encoder/comparison.json), [PNG et aperçus originaux](comparisons/encoder/). Les planches ci-dessous n'altèrent pas ces PNG.

## Paire 1, graine 1000

This is an RGBA image with transparency. A single black flower, pixel art. The image has alpha channel and the background is transparent.

![8B BF16 et 4B FP8, archive historique](pair-00.png)

## Paire 2, graine 1001

A black sign above a book, with the exact text "OPEN", legible typography, studio photograph.

![8B BF16 et 4B FP8, archive historique](pair-01.png)

## Paire 3, graine 1002

A black bicycle to the left of a purple sailing boat, clay sculpture, clear composition.

![8B BF16 et 4B FP8, archive historique](pair-02.png)

## Paire 4, graine 1003

Une illustration de bicycle, game sprite, au premier plan devant flower, lumière douce, composition nette.

![8B BF16 et 4B FP8, archive historique](pair-03.png)
