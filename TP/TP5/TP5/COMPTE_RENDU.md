# Compte rendu · TP5 : suivre les entraînements (MLflow) et versionner les données (DVC)

**Groupe :*G1* …  ·  **Membres :*Soumia Abbassi* …  ·  **Date :*25/11/2026* …

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


Avant `pip install -e .`, où pointait le package `maintai` installé dans l'environnement ? Et après ?

Après pip install -e ., le package maintai est installé en mode éditable et pointe vers le code source du TP5.

Le chemin utilisé avant cette installation n'a pas été relevé. Il reste donc à vérifier si une autre installation de maintai était présente auparavant.

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


Dans `mlflow ui`, ouvrez le modèle d'un run. Quelle version de scikit-learn est indiquée dans son `requirements.txt` ? Quel problème du TP4 cela règle-t-il ?

La version indiquée dans le fichier `requirements.txt` du modèle MLflow est **scikit-learn 1.8.0**.

Cet enregistrement permet de connaître précisément la version de la bibliothèque utilisée lors de l'entraînement.

Cela répond au problème de compatibilité et de reproductibilité abordé au TP4 : lorsqu'un modèle est chargé dans un autre environnement, une version différente de scikit-learn peut provoquer des erreurs ou des comportements incompatibles.

Le fichier `requirements.txt` permet donc de recréer un environnement compatible avec celui utilisé pour entraîner le modèle.

Où MLflow enregistre-t-il les runs sur votre disque ? Ces fichiers vont-ils dans Git ? Pourquoi ?


MLflow enregistre les informations des entraînements dans le dossier du projet `TP/TP5/TP5`, principalement dans deux emplacements :

- `mlflow.db` : base de données contenant les expériences, les identifiants des runs, les paramètres et les métriques.
- `mlruns/` : dossier contenant les artefacts des entraînements, notamment les modèles sauvegardés.

Ces fichiers ne doivent pas être ajoutés à Git, car ce sont des fichiers générés lors des expérimentations. Ils peuvent être volumineux et évoluer à chaque entraînement.

Git sert principalement à versionner le code source et les fichiers de configuration. MLflow permet de conserver et de comparer les résultats des expériences.


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

Pourquoi `data/sensors.csv` ne va pas dans Git, alors que `data/sensors.csv.dvc` y va ?

Le fichier `data/sensors.csv` contient les données du projet. Il est suivi par DVC afin d'éviter de stocker directement des datasets potentiellement volumineux dans Git.

Le fichier `data/sensors.csv.dvc` est un petit fichier de métadonnées contenant notamment l'empreinte MD5 et le chemin du dataset. Il est versionné dans Git pour permettre d'identifier précisément chaque version des données.

Ainsi, Git conserve l'historique du code et des références aux données, tandis que DVC gère le stockage et les versions des fichiers de données.


Après `python -m maintai.dataset --n 8000 --seed 7`, que dit `dvc status` ?

```
data\sensors.csv.dvc:
        changed outs:
                modified:           data\sensors.csv
```

| Version | Commit Git (`git log --oneline`) | md5 (8 premiers caractères) | Nombre de lignes |
|---|---|---|---|
| v1 | f1780d3|73e0b7cb |5001 |
| v2 |f4aaf37  |93fdeef2  |8001  |

Revenez à la v1 (`git checkout <commit> -- data/sensors.csv.dvc`, puis `dvc checkout`). Combien de lignes contient `data/sensors.csv` ? Puis revenez à la v2.

Après restauration de la version v1 à l'aide de Git et DVC, le fichier `data/sensors.csv` contient **5 001 lignes**, soit 5 000 observations et une ligne d'en-tête.

Nous avons ensuite restauré la version v2 avec `git restore` et `dvc checkout`. Le fichier contient de nouveau **8 001 lignes**, soit 8 000 observations et une ligne d'en-tête.

Cette expérience confirme que DVC permet de retrouver différentes versions d'un dataset à partir des références conservées dans Git.


Supprimez `data/sensors.csv` et le dossier `.dvc/cache`, puis lancez `dvc pull`. Que se passe-t-il ?

Après avoir supprimé `data/sensors.csv` et le dossier `.dvc/cache`, nous avons exécuté la commande `dvc pull`.

DVC a récupéré les données depuis le stockage local configuré précédemment avec `--local`, puis a reconstitué le cache et restauré le fichier CSV.

Le terminal a affiché :

```text
1 file fetched and 1 file added
```

Après restauration, le fichier contient **8 001 lignes**, soit 8 000 observations et une ligne d'en-tête.

Cette expérience montre que DVC permet de récupérer les données même après leur suppression locale, à condition qu'elles aient été sauvegardées dans un stockage DVC accessible.


## 4. Relier modèle et données

Entraînez un modèle sur la v2. Dans `mlflow ui`, comparez ce run à celui de la v1 avec les mêmes réglages :

| Run | data_md5 (8 caractères) | n_train | recall | precision |
|---|---|---|---|---|
| v1 | 73e0b7cb|4 000 |0.750 |0.622 |
| v2 |93fdeef2 |6 400 |0.741 |0.650 |

Peut-on dire que le modèle v2 est « meilleur » ? Pourquoi faut-il être prudent en comparant deux runs entraînés sur des données différentes ?


On ne peut pas affirmer que le modèle v2 est globalement meilleur.

Avec la version v2, l'accuracy passe de 90,4 % à 91,6 % et la précision de 62,2 % à 65 %. En revanche, le recall diminue légèrement de 75 % à 74,1 %.

Ces résultats montrent que le modèle v2 réalise moins de fausses alertes parmi ses prédictions positives, mais détecte une proportion légèrement plus faible des pannes réelles.

Il faut rester prudent, car les deux modèles ont été entraînés et évalués sur des versions différentes du dataset : v1 contient 5 000 observations et v2 en contient 8 000. Les jeux de test, leur composition et leur difficulté peuvent donc différer.

Pour comparer rigoureusement les deux modèles, il faudrait les évaluer sur un même jeu de test indépendant, avec les mêmes métriques et un protocole identique.


#### Recherche des meilleurs hyperparamètres — Grid Search

**Objectif :**

Rechercher automatiquement une configuration de régression logistique adaptée à la détection des pannes, en privilégiant le rappel (*recall*) et en surveillant la précision des alertes.

**Méthode et code :**

Nous avons développé `maintai/grid_search.py`, utilisant `itertools.product` pour générer toutes les combinaisons et `StratifiedKFold` pour effectuer une validation croisée stratifiée à 5 plis.

Chaque configuration est entraînée et évaluée sur les données d'apprentissage uniquement. Le jeu de test est réservé à l'évaluation finale.


# Générer toutes les combinaisons d'hyperparamètres.
combinations = list(product(
    c_values,
    solvers,
    thresholds,
    class_weights,
    max_iterations,
))

# Parcourir les 30 configurations.

````python
for index, (C, solver, threshold, weight, max_iter) in enumerate(
    combinations, start=1
):
    recalls = []
    precisions = []
    accuracies = []

    # Évaluer chaque configuration avec 5 plis.
    for train_idx, valid_idx in cv.split(X_train, y_train):
        X_fit = X_train.iloc[train_idx]
        X_valid = X_train.iloc[valid_idx]
        y_fit = y_train.iloc[train_idx]
        y_valid = y_train.iloc[valid_idx]

        # Construire et entraîner le modèle.
        model = build_model("logreg", C=C)
        model.set_params(
            clf__solver=solver,
            clf__class_weight=weight,
            clf__max_iter=max_iter,
        )
        model.fit(X_fit, y_fit)

        # Appliquer le seuil de décision testé.
        probabilities = model.predict_proba(X_valid)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        # Calculer les performances du modèle.
        recalls.append(recall_score(y_valid, predictions, zero_division=0))
        precisions.append(precision_score(y_valid, predictions, zero_division=0))
        accuracies.append(accuracy_score(y_valid, predictions))

    # Calculer les moyennes sur les cinq plis.
    result = {
        "configuration": index,
        "C": C,
        "solver": solver,
        "threshold": threshold,
        "class_weight": weight,
        "max_iter": max_iter,
        "recall_cv": round(float(np.mean(recalls)), 4),
        "precision_cv": round(float(np.mean(precisions)), 4),
        "accuracy_cv": round(float(np.mean(accuracies)), 4),
    }

    results.append(result)
````python

**Hyperparamètres et plages testées :**

| Hyperparamètre | Valeurs |
|---|---|
| `C` | 0.01, 0.1, 1.0 |
| `solver` | lbfgs, liblinear |
| `threshold` | 0.10, 0.15, 0.20, 0.25, 0.30 |
| `class_weight` | balanced |
| `max_iter` | 1000 |

Nombre total de configurations : **3 × 2 × 5 × 1 × 1 = 30**.

Le programme réalise 150 entraînements de validation croisée, puis un entraînement final.

**Critère de sélection :**

La première règle envisagée exigeait une précision d'au moins 50 %. Aucune configuration ne respectait cette contrainte : la précision maximale obtenue était de 39,07 %.

Nous avons donc sélectionné, à titre expérimental, la configuration maximisant le F2-score calculé à partir des moyennes de précision et de recall des plis. Le F2-score privilégie le recall, car manquer une panne peut avoir un coût important.

**Meilleure configuration : n° 25**

- Modèle : régression logistique
- `C = 1.0`
- `solver = lbfgs`
- `threshold = 0.30`
- `class_weight = balanced`
- `max_iter = 1000`

**Résultats en validation croisée :**

| Métrique | Résultat |
|---|---:|
| Recall moyen | 0.9499 |
| Precision moyenne | 0.3907 |
| Accuracy moyenne | 0.8039 |
| F2 calculé | 0.7385 |

**Résultats sur le jeu de test indépendant :**

| Métrique | Résultat |
|---|---:|
| Recall | 0.961 |
| Precision | 0.388 |
| Accuracy | 0.801 |

**Traçabilité :**

- Fichier du code : `maintai/grid_search.py`
- Tableau des 30 configurations : `docs/grid_search_results.csv`
- Run MLflow : `1343d56e57b648c68f8936a9e054bef3`
- Version des données : DVC v2, MD5 `93fdeef2cac1893c4f720c16457eea75`

**Conclusion :**

La configuration retenue détecte 96,1 % des pannes du jeu de test, mais sa précision de 38,8 % entraîne de nombreuses fausses alertes.

Cette configuration constitue un résultat expérimental privilégiant le rappel. Elle ne satisfait pas la contrainte de précision initialement envisagée et doit être comparée à d'autres approches avant tout déploiement.

## 5. Décision : l'ADR 004

Dans `docs/adr/004-choix-du-modele.md` : le modèle et le seuil retenus, à partir de votre tableau MLflow. Indiquez l'identifiant du run et le `data_md5` de la version des données utilisée.

L'ADR 004 a été rédigé dans `docs/adr/004-choix-du-modele.md`.

À la suite d'une recherche exhaustive de 30 configurations avec validation croisée à cinq plis, nous avons retenu provisoirement la **régression logistique (configuration n° 25)**.

**Hyperparamètres retenus :**
- `C = 1.0`
- `solver = lbfgs`
- `threshold = 0.30`
- `class_weight = balanced`
- `max_iter = 1000`

Ce modèle obtient un rappel de **96,1 %**, une précision de **38,8 %** et une accuracy de **80,1 %** sur le jeu de test.

Le choix privilégie la détection des pannes, car une panne manquée coûte plus cher qu'une fausse alerte. Toutefois, la précision relativement faible impose de considérer cette décision comme expérimentale.

**Run MLflow :** `1343d56e57b648c68f8936a9e054bef3`

**Version du dataset :** DVC v2 (8 000 observations).

**MD5 :** `93fdeef2cac1893c4f720c16457eea75`.

## 6. Questions de compréhension

1. Quelle différence entre Git, DVC et MLflow ? Que versionne chacun ?

**Git** permet de versionner le code source, les fichiers de configuration et la documentation du projet. Il conserve l'historique des modifications grâce aux commits.

**DVC (Data Version Control)** permet de suivre les différentes versions des datasets et des artefacts volumineux. Git conserve les fichiers `.dvc`, tandis que les données sont stockées dans le cache DVC et éventuellement dans un stockage distant ou local séparé.

**MLflow** permet de suivre les expériences de Machine Learning. Il enregistre les paramètres d'entraînement, les métriques, les versions des données, les modèles et les identifiants des runs.

Ces trois outils sont complémentaires : Git suit le code, DVC suit les données et MLflow suit les expériences et leurs modèles.
2. Pourquoi le stockage DVC est-il ajouté avec `--local` dans ce TP ? Que faudrait-il pour que tout le groupe partage les mêmes données ?

L'option `--local` permet d'enregistrer la configuration du stockage DVC uniquement sur notre ordinateur, dans le fichier `.dvc/config.local`, qui n'est pas versionné dans Git.

Dans notre TP, nous avons configuré un stockage local situé dans `C:\Users\SOUMIA\dvc-storage-maintai`.

Cette configuration évite d'imposer aux autres membres du groupe un chemin propre à notre PC.

Pour permettre à tous les membres de partager les mêmes données, il faudrait configurer un stockage DVC commun et accessible, par exemple un bucket S3 ou un serveur partagé, puis versionner sa configuration non sensible dans Git.

Chaque membre pourrait alors utiliser `dvc pull` pour récupérer les mêmes versions des datasets. Les identifiants d'accès resteraient privés.
3. Livrer un modèle, c'est trois choses (TP4). Lesquelles, et où sont-elles maintenant ?

Livrer un modèle de Machine Learning nécessite trois éléments :

1. **Le modèle entraîné** : il est sauvegardé dans `models/model.joblib` et enregistré comme artefact dans MLflow.

2. **Le contrat d'entrée du modèle** : il est décrit dans `models/model_info.json`. Ce fichier précise notamment les variables attendues, le seuil de décision et la version de scikit-learn utilisée.

3. **Les dépendances logicielles** : elles permettent de reproduire l'environnement d'exécution. Elles sont décrites dans les fichiers `requirements.txt` du projet et du modèle MLflow.

Dans TP5, Git versionne le code et les fichiers de configuration, DVC permet de retrouver la version des données d'entraînement et MLflow conserve les expériences ainsi que les artefacts des modèles.

Ces éléments facilitent la reproductibilité et le déploiement du modèle sur un autre environnement.

## 7. Le rendu Git

- Lien de la pull request : …
- Le tag `tp5` existe sur GitHub : oui / non
- `git status` avant le commit : `mlflow.db`, `mlruns/`, `data/sensors.csv` ou `.dvc/cache` apparaissaient-ils ? oui / non

## 8. Ce qui reste flou

La principale difficulté rencontrée concerne le compromis entre le rappel (*recall*) et la précision (*precision*) lors du choix des hyperparamètres.

Le Grid Search a montré qu'un rappel très élevé peut entraîner de nombreuses fausses alertes. Il reste à approfondir la manière de définir un compromis optimal en fonction du coût réel des pannes manquées et des fausses alertes.

Nous souhaitons également approfondir la gestion d'un stockage DVC partagé entre plusieurs membres afin de faciliter la reproductibilité des expérimentations.