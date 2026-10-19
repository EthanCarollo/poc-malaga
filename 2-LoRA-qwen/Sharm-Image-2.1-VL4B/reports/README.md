# Rapports finaux versionnés

Ce dossier conserve les résultats lisibles depuis les notebooks : paramètres, historiques, comparaisons complètes et revue identifiée. Les contrôles du loader et la révision HF immuable sont requis pour une archive publiée. Un candidat refusé reçoit une archive distincte, sans publication ni validation inventée.

## Compte rendu du 19 octobre 2026 : candidat refusé

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

L'API HF authentifiée a confirmé les révisions précédentes, inchangées : Base `1e912f5a1e5df40bdaf927abf3e4220d73781321`, ancien Monster `07a21bb50fb7749b8b7378ddf6f67ea3de078d5d`. Leur visibilité reste privée. Aucun nouvel entraînement ou pré-calcul GPU Monster n'a été lancé.

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
