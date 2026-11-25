# Compte rendu · TP5 : suivre les entraînements (MLflow) et versionner les données (DVC)

**Groupe :** …  ·  **Membres :** …  ·  **Date :** …

> Remplissez chaque partie dans ce fichier, puis rendez-le avec la pull request du TP5.
> Collez les sorties du terminal entre les lignes ``` (trois accents graves).

---

## 1. Mise en place (dans `TP/TP5`)

Dernière ligne de `python -m pytest` après `pip install -r requirements.txt` et `pip install -e .` :

```
5 failed, 27 passed, 1 warning in 8.80s
```

Quels tests échouent, et pourquoi ?

Les cinq tests échoués sont :

test_training_creates_a_run

test_run_records_metrics

test_run_records_data_version

test_run_stores_the_model

test_options_change_the_run

Ces tests appartiennent à tests/test_tracking.py et concernent le suivi des entraînements avec MLflow.

Lors de leur exécution, une erreur ImportError: DLL load failed while importing _collections_cy apparaît sous Windows. Un diagnostic complémentaire est nécessaire pour identifier la dépendance responsable. Les fonctionnalités de suivi MLflow restent à compléter et à valider.

…

Avant `pip install -e .`, où pointait le package `maintai` installé dans l'environnement ? Et après ?

Après pip install -e ., le package maintai est installé en mode éditable et pointe vers le code source du TP5.

Le chemin utilisé avant cette installation n'a pas été relevé. Il reste donc à vérifier si une autre installation de maintai était présente auparavant.
…

## 2. Sprint 1 : MLflow

Votre fonction `log_run` (`maintai/tracking.py`) :

```python

def log_run(params: dict, metrics: dict, model, X_example, data_path: Path) -> str:
    """
    Enregistre un entraînement de MaintAI dans MLflow.

    Chaque entraînement est identifié par un run unique contenant :
    - les paramètres utilisés pour entraîner le modèle ;
    - l'empreinte MD5 des données pour assurer leur traçabilité ;
    - les métriques de performance obtenues ;
    - le modèle entraîné, afin de pouvoir le réutiliser.

    Args:
        params: Dictionnaire des paramètres d'entraînement.
        metrics: Dictionnaire des métriques de performance.
        model: Modèle de Machine Learning entraîné.
        X_example: Exemple des données d'entrée du modèle.
        data_path: Chemin vers le fichier CSV utilisé pour l'entraînement.

    Returns:
        str: Identifiant unique du run MLflow créé.
    """

    # Étape 1 : sélectionner l'expérience "maintai".
    # Elle permet de regrouper les entraînements du projet dans MLflow.
    mlflow.set_experiment(EXPERIMENT)

    # Étape 2 : ouvrir un nouveau run MLflow.
    # Le gestionnaire "with" assure sa fermeture automatiquement.
    with mlflow.start_run() as run:

        # Étape 3 : enregistrer les paramètres de configuration.
        # Exemples : type de modèle, seuil de décision et valeur de C.
        mlflow.log_params(params)

        # Étape 4 : calculer et enregistrer l'empreinte MD5 des données.
        # Elle permettra d'identifier la version du dataset avec DVC.
        data_md5 = file_md5(data_path)
        mlflow.log_param("data_md5", data_md5)

        # Étape 5 : enregistrer les métriques de performance.
        # Exemples : accuracy, recall et precision.
        mlflow.log_metrics(metrics)

        # Étape 6 : sauvegarder le modèle dans les artefacts MLflow.
        # L'exemple d'entrée décrit le format attendu par le modèle.
        # Les types autorisés facilitent le chargement sécurisé avec skops.
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_example,
            skops_trusted_types=TRUSTED_TYPES,
        )

        # Étape 7 : retourner l'identifiant unique de l'entraînement.
        # Cet identifiant sert à retrouver le run dans MLflow.
        return run.info.run_id

```

Dernière ligne de `python -m pytest` à la fin du sprint 1 :

```
32 passed, 4 warnings in 17.00s
```

Vos runs (au moins quatre), relevés dans `mlflow ui` (mode « Model training ») :

| Commande | recall | precision | accuracy |
|---|---|---|---|
| `python -m maintai.train` |0.750 |0.622 |0.904 |
| `python -m maintai.train --C 0.01` |0.691 |0.653 |0.908 |
| `python -m maintai.train --threshold 0.2` | 0.838|0.543 |0.882 |
| `python -m maintai.train --model forest` | 0.728| 0.562|0.886 |
| python -m maintai.train --threshold 0.15| 0.875|0.500 |0.864 |

Quel run choisiriez-vous pour MaintAI, et pourquoi ? (Rappel : une panne manquée coûte plus cher qu'une fausse alerte.)

Je choisirais le modèle entraîné avec un seuil de décision de **0.15**, correspondant au run `ef2839e8aa794c3596645125d875b8d2`.

Ce modèle possède le meilleur rappel (*recall*) parmi les cinq configurations testées, avec une valeur de **87,5 %**. Il permet donc de détecter une plus grande proportion des pannes réelles.

Sa précision est de 50 % et son accuracy de 86,4 %, ce qui montre qu'il génère davantage de fausses alertes que certaines autres configurations.

Cependant, dans le contexte de la maintenance prédictive, une panne non détectée peut provoquer un arrêt de production et des coûts importants. Nous privilégions donc le rappel plutôt que la précision.

Ce choix reste expérimental et devra être confirmé en fonction du coût réel des fausses alertes et des pannes manquées.

…

Dans `mlflow ui`, ouvrez le modèle d'un run. Quelle version de scikit-learn est indiquée dans son `requirements.txt` ? Quel problème du TP4 cela règle-t-il ?

La version indiquée dans le fichier `requirements.txt` du modèle MLflow est **scikit-learn 1.8.0**.

Cet enregistrement permet de connaître précisément la version de la bibliothèque utilisée lors de l'entraînement.

Cela répond au problème de compatibilité et de reproductibilité abordé au TP4 : lorsqu'un modèle est chargé dans un autre environnement, une version différente de scikit-learn peut provoquer des erreurs ou des comportements incompatibles.

Le fichier `requirements.txt` permet donc de recréer un environnement compatible avec celui utilisé pour entraîner le modèle.
…

Où MLflow enregistre-t-il les runs sur votre disque ? Ces fichiers vont-ils dans Git ? Pourquoi ?


MLflow enregistre les informations des entraînements dans le dossier du projet `TP/TP5/TP5`, principalement dans deux emplacements :

- `mlflow.db` : base de données contenant les expériences, les identifiants des runs, les paramètres et les métriques.
- `mlruns/` : dossier contenant les artefacts des entraînements, notamment les modèles sauvegardés.

Ces fichiers ne doivent pas être ajoutés à Git, car ce sont des fichiers générés lors des expérimentations. Ils peuvent être volumineux et évoluer à chaque entraînement.

Git sert principalement à versionner le code source et les fichiers de configuration. MLflow permet de conserver et de comparer les résultats des expériences.

…

## 3. Sprint 2 : DVC

Contenu de `data/sensors.csv.dvc` après `dvc add data/sensors.csv` :

```
outs:
- md5: 73e0b7cb3c6f795c70c3ddf970b99c15
  size: 126815
  hash: md5
  path: sensors.csv
```

Comparez la valeur `md5` de ce fichier au paramètre `data_md5` de votre dernier run MLflow. Que constatez-vous ?


La valeur MD5 enregistrée par DVC est identique au paramètre `data_md5` du dernier run MLflow :

`73e0b7cb3c6f795c70c3ddf970b99c15`

Cette correspondance confirme que le modèle a été entraîné sur la même version du dataset que celle suivie par DVC.

Elle permet d'assurer la traçabilité entre les données utilisées et les résultats de l'entraînement, ce qui facilite la reproductibilité des expérimentations.
…

Pourquoi `data/sensors.csv` ne va pas dans Git, alors que `data/sensors.csv.dvc` y va ?

Le fichier `data/sensors.csv` contient les données du projet. Il est suivi par DVC afin d'éviter de stocker directement des datasets potentiellement volumineux dans Git.

Le fichier `data/sensors.csv.dvc` est un petit fichier de métadonnées contenant notamment l'empreinte MD5 et le chemin du dataset. Il est versionné dans Git pour permettre d'identifier précisément chaque version des données.

Ainsi, Git conserve l'historique du code et des références aux données, tandis que DVC gère le stockage et les versions des fichiers de données.

…

Après `python -m maintai.dataset --n 8000 --seed 7`, que dit `dvc status` ?

```
(collez ici)
```

| Version | Commit Git (`git log --oneline`) | md5 (8 premiers caractères) | Nombre de lignes |
|---|---|---|---|
| v1 | | | |
| v2 | | | |

Revenez à la v1 (`git checkout <commit> -- data/sensors.csv.dvc`, puis `dvc checkout`). Combien de lignes contient `data/sensors.csv` ? Puis revenez à la v2.

…

Supprimez `data/sensors.csv` et le dossier `.dvc/cache`, puis lancez `dvc pull`. Que se passe-t-il ?

…

## 4. Relier modèle et données

Entraînez un modèle sur la v2. Dans `mlflow ui`, comparez ce run à celui de la v1 avec les mêmes réglages :

| Run | data_md5 (8 caractères) | n_train | recall | precision |
|---|---|---|---|---|
| v1 | | | | |
| v2 | | | | |

Peut-on dire que le modèle v2 est « meilleur » ? Pourquoi faut-il être prudent en comparant deux runs entraînés sur des données différentes ?

…

## 5. Décision : l'ADR 004

Dans `docs/adr/004-choix-du-modele.md` : le modèle et le seuil retenus, à partir de votre tableau MLflow. Indiquez l'identifiant du run et le `data_md5` de la version des données utilisée.

## 6. Questions de compréhension

1. Quelle différence entre Git, DVC et MLflow ? Que versionne chacun ?
   …
2. Pourquoi le stockage DVC est-il ajouté avec `--local` dans ce TP ? Que faudrait-il pour que tout le groupe partage les mêmes données ?
   …
3. Livrer un modèle, c'est trois choses (TP4). Lesquelles, et où sont-elles maintenant ?
   …

## 7. Le rendu Git

- Lien de la pull request : …
- Le tag `tp5` existe sur GitHub : oui / non
- `git status` avant le commit : `mlflow.db`, `mlruns/`, `data/sensors.csv` ou `.dvc/cache` apparaissaient-ils ? oui / non

## 8. Ce qui reste flou

…
