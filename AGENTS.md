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

- La demande du 19 octobre autorise la publication de la Base avec le text encoder 4B FP8 et le DiT FP8, en conservant le projecteur antérieur `a414685b`. Ne pas publier ni relancer automatiquement le candidat spatial refusé `6fc98a1d`.
- Regrouper le projet Sharm Image dans `Sharm-Image-2.1/`. Le projet Iris est un dépôt GitHub séparé (`EthanCarollo/Sharm-Iris-PixelArt`) avec son propre notebook et ses données ; préparer ce notebook sans lancer son fine-tuning.

- Pour le moment, préparer uniquement les notebooks du LoRA : ne lancer aucun nouvel entraînement LoRA, ni ses pré-calculs GPU, sans nouvelle demande explicite de l'utilisateur. La chaîne Base spatiale déjà active conserve son périmètre autorisé.
- Conserver exactement deux notebooks exécutables : `Sharm-Image-2.1/Sharm-Image-2.1-VL4B/Sharm_Image_2_1_VL4B.ipynb` et `Sharm-Image-2.1/Sharm-Image-2.1-VL4B-LoRA-Monster/Sharm_Image_2_1_VL4B_LoRA_Monster.ipynb`, dans leurs deux dossiers distincts.
- Intégrer le code des étapes et des fonctions partagées dans les cellules de ces deux fichiers. Les calculs tournent localement dans le kernel de la machine choisie, sans script source externe ni troisième notebook. Conserver les lancements de calcul et publication désactivés par défaut dans Git ; ils restent réellement exécutables manuellement.
- Après la fin réelle de la chaîne, la revue et la publication vérifiée, archiver les résultats dans `Sharm-Image-2.1/Sharm-Image-2.1-VL4B/reports/` avec les paramètres, historiques, galeries complètes, revues identifiées, SHA-256 et révision HF immuable. L'étape d'archivage est intégrée au notebook Base.
- Conserver les sorties de cellules nettoyées, les preuves dans des fichiers explicites et les poids hors Git. Ne pas inventer de résultat ni d'annotation humaine ; distinguer les pertes numériques, la revue par agent et la validation spatiale humaine.
- Finaliser ce rangement, puis committer et pousser les notebooks et rapports sur `main` avant de clôturer le suivi d'entraînement.
