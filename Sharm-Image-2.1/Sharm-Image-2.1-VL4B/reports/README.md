# Rapports finaux versionnés

Ce dossier conserve les résultats lisibles depuis les notebooks : paramètres, historiques, comparaisons complètes et revue identifiée. Les contrôles du loader et la révision HF immuable sont requis pour une archive publiée. Un candidat refusé reçoit une archive distincte, sans publication ni validation inventée.

## README HF détaillé et vérification renouvelée

**[Révision HF `e6ef25ce358f400781a9cd3c3e0fb8068312b599`](https://huggingface.co/ethanzxv/Sharm-Image-2.1-VL4B-FP8/tree/e6ef25ce358f400781a9cd3c3e0fb8068312b599).** Le README affiche les 16 sorties Sharm sur sujets simples, les 16 paires Qwen original / Sharm et les diagnostics complémentaires. Chaque taille est rattachée au composant : Qwen3-VL 4B pour le texte, DiT Qwen-Image-2.1 7,1B pour la génération. Le lexique explique les formats et réglages.

Les 52 fichiers du payload ont été vérifiés sur cette révision immuable ; les 170 autres fichiers préexistants sont conservés. Les poids et le loader restent identiques à `a0c0e168`. La carte, le manifeste et les preuves de contrôle ont changé. Deux générations réelles ont été rejouées avec le loader exact : leurs PNG sont identiques octet pour octet aux deux contrôles d'origine, avec les mêmes limites de transparence.

[Résumé signé](fp8-e6ef25ce358f/summary.json) · [Publication vérifiée](fp8-e6ef25ce358f/publication-status.json) · [Contrôle GPU](fp8-e6ef25ce358f/fp8-validation.json) · [Revue identifiée](fp8-e6ef25ce358f/fp8-visual-review.json) · [SHA des preuves](fp8-e6ef25ce358f/manifest.json).

## Comparatifs avant / après

| Comparatif | Galerie et paramètres | Portée |
| --- | --- | --- |
| **Modèle Qwen original → Sharm publié** | [16 paires, 32 PNG et protocole](base-vs-sharm-v1/README.md) · [galerie HTML](base-vs-sharm-v1/gallery.html) · [revue exhaustive](base-vs-sharm-v1/REVIEW.md) | Features 8B BF16 + vrai DiT BF16 / 4B FP8 + projecteur + DiT FP8 natif ; 4 relations × FR/EN × 2 graines, 512 px, 30 pas, CFG 1, même bruit initial |
| **Ancien DiT NF4 → nouveau DiT FP8** | [16 paires, 32 PNG, planches et mesures](comparison-nf4-fp8-a0c0e1680970-v1/README.md) · [galerie HTML avec curseur](comparison-nf4-fp8-a0c0e1680970-v1/gallery.html) | Encodeur 4B FP8 et projecteur identiques ; mêmes embeddings et latents initiaux sauvegardés ; 8 prompts × 2 graines, 512 px, 30 pas, CFG 3 |
| **Encodeur 8B BF16 → encodeur 4B FP8 + projecteur** | [4 paires historiques et leurs PNG originaux](encoder-8b-4b-1e912f5a/README.md) | Archive de la révision HF `1e912f5a` ; DiT NF4 partagé, 512 px, 20 pas, CFG 1 ; aucune nouvelle génération |
| **Candidat spatial refusé / enseignant 8B** | [Galerie complète de l'expérience refusée](rejected-7a39a8cc674e/run/validation/review.html) | Expérience d'entraînement distincte ; ce projecteur n'est pas celui de la Base publiée |

Pour le comparatif de diffusion, ouvrir `gallery.html` localement après avoir cloné le dépôt : chaque paire est visible côte à côte à 512 px et en superposition avec un curseur. Les 8 planches du README montrent aussi toutes les paires directement dans GitHub. Aucun PNG n'est retiré du rapport ni détouré après génération. Les paramètres, poids et loaders portent leurs SHA et révisions immuables.

Les mesures MSE RGB/alpha indiquent des différences de pixels entre les variantes comparées. Elles ne donnent pas un taux d'erreur des positions et ne prouvent pas une meilleure qualité. Les revues par agent sont identifiées ; aucune validation humaine statistique n'est inventée. Les comparatifs sont reproductibles depuis les cellules **Avant / après de la Base publiée** et **Archiver et publier les comparatifs réels** du notebook Base, avec tous les lancements désactivés par défaut.

La [revue par agent des quatre paires historiques d'encodeurs](encoder-8b-4b-agent-review.json) constate des sujets communs et du texte lisible, mais aussi des différences de fond, de disposition et des déformations. Cet échantillon ne démontre pas une parité 8B ni des positions parfaites.

La [revue complète NF4 / FP8](comparison-nf4-fp8-a0c0e1680970-v1/REVIEW.md) signale notamment une perte de transparence pour le dragon à la graine 1001, des changements de contenu pour les livres et une meilleure séparation étoile/cercle à la graine 1000. Sur ce seul passage, le temps médian génération + décodage passe de 26,43 s à 20,90 s, avec un pic de VRAM allouée passant de 5,73 à 8,95 Gio. Ces mesures excluent chargement et encodage texte ; elles ne constituent pas une garantie générale de vitesse ou de qualité.

Le premier envoi des preuves NF4/FP8 a ajouté `COMPARISONS.md` et son dossier sur HF sans modifier le payload `a0c0e168` ([révision vérifiée](comparison-publication.json)). Le [comparatif principal](base-comparison-publication.json) ajoute ensuite les 32 images original/Sharm, sans modifier le modèle. La carte détaillée reçoit son propre manifeste et une nouvelle vérification GPU ; elle conserve tous les poids et le loader.

### Ce que montre le comparatif principal

Sharm produit un seul personnage hybride à la place de plusieurs objets dans plusieurs scènes, ou des sujets superposés. La référence conserve plus souvent plusieurs silhouettes dans cet échantillon mais se trompe aussi sur les espèces, le nombre et les relations. Les cas `above-en` conservent une séparation verticale des deux côtés sans établir la fidélité aux sujets. **La parité 8B et la précision spatiale ne sont pas validées.**

La MSE RGB sur blanc est de 0,052359 et la MSE alpha de 0,279840 ; ce sont des écarts entre sorties normalisées, pas des taux d'erreur de placement. Les 16 PNG Sharm et 11/16 références ont au moins un pixel alpha 0 ; ce critère faible ne mesure pas la qualité du détourage. Les paramètres, la revue identifiée et tous les PNG sont signés.

Le 8B n'a pas été réencodé : ses features BF16 réelles sont réutilisées après contrôle des tokens, métadonnées et SHA. Le DiT de référence reste entièrement BF16 et utilise l'offload CPU. Le protocole ne permet donc pas de comparer la vitesse ou la VRAM complète des deux modèles. La [provenance d'exécution](base-vs-sharm-v1/reproducibility.json) conserve le commit et les empreintes du code effectivement employé.

## Publication du 19 octobre 2026 : Base avec text encoder et DiT FP8

**Révision HF vérifiée : [`a0c0e168097024cecd8c43f68004d1bedbab32df`](https://huggingface.co/ethanzxv/Sharm-Image-2.1-VL4B-FP8/tree/a0c0e168097024cecd8c43f68004d1bedbab32df).** Les 52 fichiers ont été contrôlés sur cette révision immuable, avec conservation de la visibilité privée. Le DiT possède 232 couches linéaires quantifiées, soit 7 115 112 448 poids E4M3 ; l'audit de génération observe 13 920 appels FP8 natifs par image avec CFG, sans repli BF16 de ces couches. L'encodeur contient 3 633 315 840 paramètres de langue FP8 ; d'autres paramètres et composants conservent leur précision.

Cette publication reprend le projecteur `a414685b` de la Base antérieure, sans entraînement supplémentaire et sans utiliser le candidat spatial refusé. Les anciens shards BF16 du DiT ont été retirés du dépôt à cette révision. Les normes, biais, VAE et projecteur restent hors quantification FP8.

Deux vraies générations RGBA à 512 px, 30 pas, CFG 3 ont été produites par le loader exporté, avec un pic mesuré de 8,946 Gio de VRAM. Le renard et le dragon pixel art sont reconnaissables. **Les fonds sont largement blancs et opaques : alpha 245–255 et 64–255, sans pixel totalement transparent.** Le format RGBA et le calcul FP8 sont validés techniquement ; la qualité de détourage et la précision spatiale ne le sont pas. La revue est identifiée comme `agent_visual_review`, sans statistique humaine inventée.

- [Résumé publié](fp8-a0c0e1680970/summary.json)
- [Vérification GPU et vrais formats](fp8-a0c0e1680970/fp8-validation.json)
- [Revue et limites](fp8-a0c0e1680970/fp8-visual-review.json)
- [Renard RGBA original](fp8-a0c0e1680970/validation/fp8-1.png)
- [Dragon RGBA original](fp8-a0c0e1680970/validation/fp8-2.png)
- [SHA-256 des preuves archivées](fp8-a0c0e1680970/manifest.json)

L'étape `archive` du parcours FP8 intégré au notebook conserve ces octets exacts, les paramètres de quantification et la révision HF. Aucun poids n'entre dans cette archive Git.

## Expérience antérieure du 19 octobre 2026 : candidat spatial refusé

**L'entraînement est terminé, la qualité attendue n'est pas atteinte.** Le candidat n'a été ni exporté pour publication, ni envoyé sur Hugging Face.

| Étape | Résultat observé |
| --- | --- |
| Warmup | 1 000 / 1 000 pas ; meilleur score numérique 0,092788 |
| Flow | 1 500 / 1 500 pas ; meilleure perte de validation flow 0,099749 |
| Galerie principale | 24 scènes × 3 graines × 4 méthodes = 288 PNG RGBA |
| Revue par agent | 144 images candidat et 144 homologues 8B inspectés ; candidat refusé |
| Diagnostic warmup | 6 scènes × 2 graines × 2 modes = 24 images inspectées ; remplacement refusé |
| Contrôle des fichiers | SHA-256 des 312 PNG de génération conformes aux deux manifestes |

Les sujets du candidat flow sont déformés et des formes très similaires reviennent pour des prompts différents. Des fragments supplémentaires et des motifs de fond apparaissent. Les variantes avec boîtes ne démontrent pas un placement correct des sujets demandés. Le professeur 8B a lui aussi des erreurs : ses images ne sont pas des annotations de référence.

Sur les **24 mêmes prompts de validation sans boîtes**, les embeddings 4B et 8B ont chacun 24 empreintes distinctes. L'erreur relative moyenne d'alignement avec le 8B passe de **0,087146 au warmup à 0,183908 après flow** ; le cosinus moyen passe de **0,951827 à 0,897007**. Ces mesures comparent les tenseurs d'embeddings, pas les coordonnées des objets. Elles diffèrent de la perte flow de 0,099749, qui compare des champs de débruitage. Le score warmup ajoute un terme de cosinus à l'erreur relative.

La dégradation d'alignement après flow est mesurée. Le diagnostic visuel montre aussi des répétitions dès le warmup. Cela ne prouve donc pas qu'un simple changement du poids de perte flow suffirait. La cause unique n'est pas établie ; aucun nouvel entraînement n'a été lancé sur cette hypothèse. La suite doit examiner la conservation de l'information sémantique et le protocole de sélection avant de reprendre le calcul.

### Preuves conservées

- [Résumé et identités](rejected-7a39a8cc674e/summary.json)
- [Revue complète identifiée `agent_visual_review`](rejected-7a39a8cc674e/visual-review.json)
- [Galerie appariée complète](rejected-7a39a8cc674e/run/validation/review.html)
- [Manifeste des 288 générations](rejected-7a39a8cc674e/run/validation/predictions.json)
- [Mesures des embeddings](rejected-7a39a8cc674e/post-training-diagnostics.json)
- [Revue du diagnostic warmup](rejected-7a39a8cc674e/warmup-diagnostic-review.json)
- [SHA-256 de tous les fichiers archivés](rejected-7a39a8cc674e/manifest.json)

Le projecteur refusé porte le SHA-256 `6fc98a1df3b9879a7a79d568c0f6ec054bcbe8c5b15602bc57f7bd8ad228b568`. Ses poids restent hors Git. Les PNG et manifestes sont archivés avec leurs octets exacts. Les deux phases conservent leurs historiques et paramètres ; aucune annotation humaine ni estimation statistique de précision spatiale n'a été créée.

Avant la conversion FP8, l'API HF avait confirmé Base `1e912f5a1e5df40bdaf927abf3e4220d73781321` et ancien Monster `07a21bb50fb7749b8b7378ddf6f67ea3de078d5d`. La Base a depuis reçu la publication décrite ci-dessus ; Monster reste à sa révision antérieure. Leur visibilité reste privée. Aucun nouvel entraînement ou pré-calcul GPU Monster n'a été lancé.

## Archiver un candidat refusé

Dans le notebook Base, choisir `ARCHIVE_OUTCOME = "rejected"`, puis activer manuellement `RUN_ARCHIVE_RESULTS`. Cette étape exige une chaîne terminée, une galerie complète sans doublon, une revue de rejet couvrant toutes les images du candidat, le profil exact et les empreintes liées. Elle conserve aussi une éventuelle galerie de diagnostic, séparément. Elle ne charge aucun poids et ne publie rien.

Le dossier `rejected-<SHA candidat>` est immuable. Son résumé porte `outcome: rejected`, `published: false` et `hf_revision: null`. Le pointeur `current.json` désigne le dernier résultat archivé, qui peut donc être un rejet.

## Archive d'une Base publiée

Après publication HF vérifiée et collecte des petits rapports dans `artifacts/spatial/`, utiliser la cellule **Conserver les vrais résultats et versionner** du notebook Base. Elle exécute la fonction d’archivage intégrée à ce même fichier.

La fonction vérifie la fin des deux phases et de la génération, la revue couvrant toutes les images du candidat, les SHA des PNG et la cohérence de l'export validé avec la publication. Il réutilise les contrôles du publisher. Il ne lance aucun entraînement, ne crée aucune annotation et ne publie aucun poids.

```text
Sharm-Image-2.1-VL4B/reports/
├── current.json                         # pointeur et SHA du manifeste sélectionné
└── spatial-<12 premiers caractères HF>/
    ├── summary.json                     # identités, SHA, révisions et portée des conclusions
    ├── profile.json                     # paramètres correspondant au run
    ├── publication-status.json          # révision distante vérifiée
    ├── visual-review.json               # auteur, méthode, couverture et observations
    ├── manifest.json                    # SHA-256 de chaque fichier de l'archive
    ├── run/
    │   ├── warmup/                      # fin et historique numérique
    │   ├── flow/                        # fin et historique numérique
    │   └── validation/                  # manifeste, tous les PNG et aperçus appariés
    └── release/
        ├── release_manifest.json        # références et empreintes des poids, sans leurs octets
        ├── export-validation.json       # contrôles du loader exact exporté
        └── validation/                  # deux PNG RGBA réels et leurs deux aperçus
```

La liste de fichiers est explicite : les poids, features, latents, optimiseur, journaux et fichiers de connexion ne sont jamais copiés récursivement. Un brouillon d'assemblage reste dans `artifacts/spatial/notebook-report-staging/`, hors Git. Une archive terminée ne s'écrase pas ; une nouvelle révision HF produit un autre dossier.

Les octets des PNG, manifestes de générations et rapports de revue sont préservés. Les résumés de fin sélectionnent les champs numériques et les identités utiles, sans exporter les chemins absolus du serveur. `archive_code_revision` identifie le code au moment de l'archivage ; les SHA de `training_identity` identifient le code et les caches réellement employés par le trainer.

## Portée de l'évaluation

Une revue `agent_visual_review` reste une revue par agent. La validation technique du loader et ses deux générations RGBA n'établissent pas un taux de réussite spatial humain. Le résumé conserve `spatial_quality_validated: false` ; une éventuelle évaluation humaine indépendante doit avoir ses propres annotations, protocole, taille d'échantillon et rapport.

## Finalisation Git

Après l'archivage, examiner les fichiers, nettoyer les sorties des notebooks, puis committer et pousser l'ensemble cohérent sur `origin/main` avec la convention Angular. Conserver le SHA du commit poussé et la révision HF dans le compte rendu final. Les données et rapports signés respectent `.gitattributes` : aucune conversion automatique de fins de ligne.
