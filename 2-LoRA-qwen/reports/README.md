# Rapports finaux versionnés

Ce dossier conserve les résultats lisibles depuis les notebooks : paramètres, historiques, comparaisons complètes, revue identifiée, contrôles du loader et révision HF immuable. Aucun résultat final n'est ajouté avant la fin réelle des calculs et des contrôles.

## Archive d'une Base publiée

Après publication HF vérifiée et collecte des petits rapports dans `artifacts/spatial/`, depuis la racine :

```powershell
.\.venv-spatial-check\Scripts\python.exe 2-LoRA-qwen/notebooks/archive_spatial_results.py
```

Le script vérifie la fin des deux phases et de la génération, la revue couvrant toutes les images du candidat, les SHA des PNG et la cohérence de l'export validé avec la publication. Il réutilise les contrôles du publisher. Il ne lance aucun entraînement, ne crée aucune annotation et ne publie aucun poids.

```text
reports/
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
