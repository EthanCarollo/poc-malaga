# Instructions du dépôt

## Git : consigne de l'utilisateur

- Travailler sur `main`, sauf demande explicite d'une autre branche.
- Après chaque ensemble cohérent de changements autorisés, créer un commit puis le pousser sur `origin/main` dans la même intervention. Ne pas laisser de changements finaux de code, documentation ou données validées uniquement en local.
- Utiliser la convention Angular : `type(scope): description`, en anglais, avec une description concise à l'impératif et sans point final. Types usuels : `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
- Inclure tous les fichiers pertinents : code, configuration, tests existants, README, schémas, PNG sources et versions validées du dataset Monster avec leurs manifestes, définitions et rapports.
- Conserver hors Git les secrets et identifiants, environnements, poids de modèles, caches d'exécution et de références, journaux locaux et brouillons de préparation. Les poids ont leur destination Hugging Face autorisée ; un push Git ne doit pas les publier dans le dépôt GitHub public.
- Vérifier l'état de la branche et les changements distants avant de pousser. Préserver le travail existant et résoudre les conflits sans push forcé.
- Les octets des données signées par empreinte doivent être préservés : respecter les règles de `.gitattributes` pour les datasets.

## Notebooks et résultats Sharm

- Pour le moment, préparer uniquement les notebooks du LoRA : ne lancer aucun nouvel entraînement LoRA, ni ses pré-calculs GPU, sans nouvelle demande explicite de l'utilisateur. La chaîne Base spatiale déjà active conserve son périmètre autorisé.
- Documenter le parcours dans `2-LoRA-qwen/notebooks/` : architecture, données, encodeurs et MSE, Base spatiale, générations, LoRA Monster, export et publication.
- Appeler les scripts existants depuis les notebooks ; conserver les actions de calcul et publication désactivées par défaut dans les fichiers versionnés.
- Après la fin réelle de la chaîne, la revue et la publication vérifiée, archiver les résultats dans `2-LoRA-qwen/reports/` avec les paramètres, historiques, galeries complètes, revues identifiées, SHA-256 et révision HF immuable. Voir `reports/README.md` et `notebooks/archive_spatial_results.py`.
- Conserver les sorties de cellules nettoyées, les preuves dans des fichiers explicites et les poids hors Git. Ne pas inventer de résultat ni d'annotation humaine ; distinguer les pertes numériques, la revue par agent et la validation spatiale humaine.
- Finaliser ce rangement, puis committer et pousser les notebooks et rapports sur `main` avant de clôturer le suivi d'entraînement.
