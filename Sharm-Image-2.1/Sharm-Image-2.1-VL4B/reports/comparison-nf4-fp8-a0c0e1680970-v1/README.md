# Sharm Image Base : avant NF4 / après FP8

Comparaison complète de **16 paires / 32 PNG RGBA** : 8 prompts × 2 graines (1000, 1001), 512 × 512, 30 pas, CFG 3, prompt négatif vide, KV cache désactivé. Le protocole est fixé avant les générations ; aucun exemple n'est retiré.

| Élément | Avant | Après |
| --- | --- | --- |
| Révision HF | `1e912f5a1e5df40bdaf927abf3e4220d73781321` | `a0c0e168097024cecd8c43f68004d1bedbab32df` |
| Text encoder | Qwen3-VL 4B FP8 | Identique |
| Projecteur | `a414685b` | Identique |
| DiT stocké | BF16 | FP8 E4M3 |
| DiT exécuté | NF4 via le loader publié | FP8 natif, 232 modules |

Les deux loaders sont rejoués dans **le même runtime figé**. Ce n'est pas une reproduction des performances d'un ancien environnement logiciel. Les embeddings positifs/négatifs et les latents initiaux sont calculés une fois, sauvegardés puis réutilisés avec empreintes identiques dans chaque paire. Les SHA des vrais poids et du code sont dans [sources.json](sources.json). Aucun fine-tuning n'est lancé.

## Consulter

- Ouvrir [gallery.html](gallery.html) localement pour les 16 paires à taille native et le curseur avant/après.
- Les 8 planches ci-dessous montrent **toutes** les paires, sur damier de prévisualisation. Les [PNG originaux](images/) gardent leurs octets RGBA ; aucune suppression de fond n'est appliquée.
- [Protocole exact](protocol.json), [mesures par image et paire](metrics.json), [audits des loaders](loader-audits.json), [revue identifiée](visual-review.json).

## Mesures observées

| Mesure | Avant NF4 | Après FP8 |
| --- | ---: | ---: |
| Images | 16 | 16 |
| Médiane génération + décodage, hors première image (s) | 26.433 | 20.898 |
| Maximum VRAM allouée par PyTorch (Gio) | 5.728 | 8.946 |
| Fraction moyenne de pixels alpha = 0 | 0.053529 | 0.027252 |

La mesure de temps exclut le chargement des modèles et l'encodage texte. Chaque variante n'a qu'un passage séquentiel ; ces temps ne constituent pas une garantie de vitesse ni un benchmark statistique. Les contextes graphiques du bureau restent présents.

MSE RGB moyenne sur fond blanc, normalisée sur [0,1] : **0.037987**. MSE alpha moyenne : **0.046231**. Ces valeurs mesurent la différence entre deux sorties ; NF4 n'est pas une vérité terrain. Elles ne démontrent ni meilleure qualité, ni précision des positions, ni parité avec le 8B. Une revue par agent ne remplace pas une évaluation humaine statistique. Voir les conclusions réellement renseignées dans `visual-review.json`.

## Galerie exhaustive

### Paires 1–2

![Avant et après, mêmes prompts et graines](sheets/pairs-01.png)

### Paires 3–4

![Avant et après, mêmes prompts et graines](sheets/pairs-02.png)

### Paires 5–6

![Avant et après, mêmes prompts et graines](sheets/pairs-03.png)

### Paires 7–8

![Avant et après, mêmes prompts et graines](sheets/pairs-04.png)

### Paires 9–10

![Avant et après, mêmes prompts et graines](sheets/pairs-05.png)

### Paires 11–12

![Avant et après, mêmes prompts et graines](sheets/pairs-06.png)

### Paires 13–14

![Avant et après, mêmes prompts et graines](sheets/pairs-07.png)

### Paires 15–16

![Avant et après, mêmes prompts et graines](sheets/pairs-08.png)
