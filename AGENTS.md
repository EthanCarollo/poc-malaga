# Instructions du dépôt

## Git : consigne de l'utilisateur

- Travailler sur `main`, sauf demande explicite d'une autre branche.
- Après chaque ensemble cohérent de changements autorisés, créer un commit puis le pousser sur `origin/main` dans la même intervention. Ne pas laisser de changements finaux de code, documentation ou données validées uniquement en local.
- Utiliser la convention Angular : `type(scope): description`, en anglais, avec une description concise à l'impératif et sans point final. Types usuels : `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
- Inclure tous les fichiers pertinents : code, configuration, tests existants, README, schémas, PNG sources et versions validées du dataset Monster avec leurs manifestes, définitions et rapports.
- Conserver hors Git les secrets et identifiants, environnements, poids de modèles, caches d'exécution et de références, journaux locaux et brouillons de préparation. Les poids ont leur destination Hugging Face autorisée ; un push Git ne doit pas les publier dans le dépôt GitHub public.
- Vérifier l'état de la branche et les changements distants avant de pousser. Préserver le travail existant et résoudre les conflits sans push forcé.
- Les octets des données signées par empreinte doivent être préservés : respecter les règles de `.gitattributes` pour les datasets.
