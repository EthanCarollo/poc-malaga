# Sharm Image 2.1

Sharm Image assemble deux parties : **Qwen3-VL 4B lit le prompt** et **Qwen-Image-2.1, environ 7,1 milliards de paramètres, dessine l’image**. B signifie milliard de paramètres appris. Le « 4B » du nom parle de l’encodeur de texte, pas du générateur d’images.

## Les deux notebooks

| Notebook | À quoi il sert |
| --- | --- |
| [Base VL 4B FP8](Sharm-Image-2.1-VL4B/Sharm_Image_2_1_VL4B.ipynb) | Générer des images, comparer les variantes, vérifier l’export et publier la Base |
| [LoRA Monster](Sharm-Image-2.1-VL4B-LoRA-Monster/Sharm_Image_2_1_VL4B_LoRA_Monster.ipynb) | Préparer les données et adapter le générateur au style des monstres |

Les étapes sont dans les notebooks. Elles s’exécutent sur la machine du kernel ; les opérations GPU et publication sont à déclencher manuellement.

## Ce que montrent les tests

- [Original Qwen contre Sharm actuel : 16 paires](Sharm-Image-2.1-VL4B/reports/base-vs-sharm-v1/README.md). Même prompt, graine et bruit initial ; les deux modèles peuvent se tromper. La disposition correcte n’est pas garantie.
- [Même Sharm en NF4 puis en FP8 : 16 paires](Sharm-Image-2.1-VL4B/reports/comparison-nf4-fp8-a0c0e1680970-v1/README.md).
- [Tests et limites de la Base publiée](https://huggingface.co/ethanzxv/Sharm-Image-2.1-VL4B-FP8).

Les pages Hugging Face affichent des images de ces tests ; les rapports locaux conservent toutes les images, les réglages et leurs empreintes.

## Quelques mots expliqués

- **Encodeur de texte** : lit la description et la transmet au générateur.
- **DiT** : le réseau qui dessine l’image.
- **Projecteur** : relie les représentations de l’encodeur au générateur.
- **FP8, BF16 et NF4** : formats numériques de 8, 16 et 4 bits. Ici, toutes les parties du modèle ne sont pas en FP8.
- **RGBA** : PNG avec un canal de transparence ; ce canal ne garantit pas que le fond est transparent.
- **Graine** : valeur qui fixe le départ aléatoire d’une génération.
- **MSE** : moyenne des différences au carré entre des valeurs ; pour des images, elle mesure un écart de pixels, pas un taux d’erreur des positions.
- **LoRA** : petites matrices ajoutées au générateur pour apprendre un style sans réentraîner tout le modèle.

La Base publiée reprend le projecteur antérieur validé. L’ancien essai spatial a été refusé et n’est pas inclus dans la Base.