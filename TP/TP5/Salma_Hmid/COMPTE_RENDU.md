# Compte rendu · TP5 : suivre les entraînements (MLflow) et versionner les données (DVC)

**Groupe :** …  ·  **Membres :** …  ·  **Date :** …

> Remplissez chaque partie dans ce fichier, puis rendez-le avec la pull request du TP5.
> Collez les sorties du terminal entre les lignes ``` (trois accents graves).

---

## 1. Mise en place (dans `TP/TP5`)

Dernière ligne de `python -m pytest` après `pip install -r requirements.txt` et `pip install -e .` :

```
5 failed, 27 passed, 2 warnings in 31.64s
```

Quels tests échouent, et pourquoi ?

Les cinq tests du fichier `tests/test_tracking.py` échouent :

- `test_training_creates_a_run`
- `test_run_records_metrics`
- `test_run_records_data_version`
- `test_run_stores_the_model`
- `test_options_change_the_run`

Ces tests échouent car la fonction `log_run()` du fichier `maintai/tracking.py` n'est pas encore implémentée. Elle se termine par `return ""` : aucun entraînement n'est donc enregistré dans MLflow. Chaque test s'arrête sur l'assertion « aucun run MLflow : log_run n'a rien enregistré ».

Avant `pip install -e .`, où pointait le package `maintai` installé dans l'environnement ? Et après ?

Dans le cadre du TP, avant l'installation en mode éditable, le package `maintai` pointait vers le TP4. Dans notre cas, un nouvel environnement virtuel utilisant Python 3.12.10 a été créé pour le TP5.

Après l'exécution de `pip install -e .`, le package `maintai` pointe vers le dossier `TP/TP5/maintai`. Les modifications apportées au code source sont directement prises en compte sans réinstallation du package.

## 2. Sprint 1 : MLflow

Votre fonction `log_run` (`maintai/tracking.py`) :

```python
def log_run(params: dict, metrics: dict, model, X_example, data_path: Path) -> str:
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_param("data_md5", file_md5(data_path))
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_example,
            skops_trusted_types=TRUSTED_TYPES
        )

        return run.info.run_id
```

Dernière ligne de `python -m pytest` à la fin du sprint 1 :

```
32 passed, 4 warnings in 11.22s
```

Vos runs :

| Commande | recall | precision | accuracy |
|---|---|---|---|
| `python -m maintai.train` | 0.750 | 0.622 | 0.904 |
| `python -m maintai.train --C 0.01` | 0.691 | 0.653 | 0.908 |
| `python -m maintai.train --threshold 0.2` | 0.838 | 0.543 | 0.882 |
| `python -m maintai.train --model forest` | 0.728 | 0.562 | 0.886 |
| (votre essai) | — | — | — |

**Quel run choisiriez-vous pour MaintAI, et pourquoi ?**

Nous retenons la régression logistique avec un seuil de décision de 0.2, car elle présente le meilleur recall (83,8 %) parmi les quatre entraînements réalisés.

Dans le contexte de la maintenance prédictive, une panne non détectée coûte plus cher qu'une fausse alerte. Nous acceptons donc une précision plus faible (54,3 %) en échange d'une meilleure détection des pannes.

La forêt aléatoire n'est pas meilleure dans nos essais, car son recall est inférieur (72,8 %).
…

Dans `mlflow ui`, ouvrez le modèle d'un run. Quelle version de scikit-learn est indiquée dans son `requirements.txt` ? Quel problème du TP4 cela règle-t-il ?

…
La version de scikit-learn indiquée dans le fichier `requirements.txt` du modèle est `1.8.0`.

MLflow enregistre automatiquement les versions exactes des bibliothèques utilisées lors de l'entraînement. Cela résout le problème de compatibilité des dépendances rencontré au TP4 et facilite la reproductibilité du modèle dans un autre environnement.

Où MLflow enregistre-t-il les runs sur votre disque ? Ces fichiers vont-ils dans Git ? Pourquoi ?

…
MLflow enregistre les métadonnées des runs dans la base locale `mlflow.db` et les artefacts des modèles dans le dossier `mlruns/`.

Ces fichiers ne sont pas versionnés dans Git, car ils sont générés automatiquement et peuvent être volumineux. Ils sont exclus par le fichier `.gitignore`.

## 3. Sprint 2 : DVC

Contenu de `data/sensors.csv.dvc` après `dvc add data/sensors.csv` :

```
outs:
- md5: b7fec4274774842dc1f48202c6013490
  size: 121814
  hash: md5
  path: sensors.csv
```

Comparez la valeur `md5` de ce fichier au paramètre `data_md5` de votre dernier run MLflow. Que constatez-vous ?

Les empreintes MD5 enregistrées dans DVC et MLflow sont identiques :

`b7fec4274774842dc1f48202c6013490`

Les quatre entraînements ont donc utilisé la même version du dataset V1. Cette correspondance permet de retrouver les données exactes utilisées pour chaque entraînement et garantit la traçabilité entre le modèle et les données.

Pourquoi `data/sensors.csv` ne va pas dans Git, alors que `data/sensors.csv.dvc` y va ?

Le fichier `data/sensors.csv` contient les données réelles et peut être volumineux. Il est donc versionné par DVC afin de ne pas alourdir le dépôt Git.

Le fichier `data/sensors.csv.dvc` contient uniquement les métadonnées du dataset : son empreinte MD5, sa taille et son chemin. Git conserve ce fichier afin de permettre la restauration des différentes versions des données.

Après `python -m maintai.dataset --n 8000 --seed 7`, que dit `dvc status` ?
data/sensors.csv.dvc:
        changed outs:
                modified:           data/sensors.csv

| Run | data_md5 (8 caractères) | n_train | recall | precision |
|---|---|---|---|---|
| v1 | b7fec427 | 4000 | 0.750 | 0.622 |
| v2 | 31ef5ebb | 6400 | 0.741 | 0.650 |

**Peut-on dire que le modèle v2 est « meilleur » ? Pourquoi faut-il être prudent en comparant deux runs entraînés sur des données différentes ?**

Nous ne pouvons pas affirmer que le modèle V2 est meilleur. Bien que sa précision et son accuracy aient augmenté, son recall a légèrement diminué, passant de 75 % à 74,1 %.

De plus, les deux modèles ont été entraînés et évalués sur des ensembles de données différents. Pour effectuer une comparaison fiable, il faudrait les évaluer sur un même jeu de test indépendant, avec les mêmes conditions d'évaluation.
Revenez à la v1 (`git checkout <commit> -- data/sensors.csv.dvc`, puis `dvc checkout`). Combien de lignes contient `data/sensors.csv` ? Puis revenez à la v2.

…
Après avoir récupéré le fichier `sensors.csv.dvc` du commit `76c495a` et exécuté `dvc checkout`, le dataset contient de nouveau 5000 lignes.

Après avoir restauré le fichier `sensors.csv.dvc` depuis HEAD et exécuté une seconde fois `dvc checkout`, le dataset contient 8000 lignes.

Les deux versions ont donc été restaurées avec succès, sans régénérer les données.

Supprimez `data/sensors.csv` et le dossier `.dvc/cache`, puis lancez `dvc pull`. Que se passe-t-il ?
Après avoir supprimé le fichier `data/sensors.csv` et le dossier `.dvc/cache`, la commande `dvc pull` récupère les données depuis le stockage DVC configuré.

La sortie du terminal confirme la restauration :

```
A       data/sensors.csv
1 file fetched and 1 file added
```

Le fichier restauré contient bien 8000 lignes. La commande `dvc status` confirme que les données sont à jour.

DVC permet donc de récupérer les données même lorsque le fichier et le cache local ont été supprimés.

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
# ADR 004 — Choix du modèle de maintenance prédictive

**Date :** 25/11/2026

**Statut :** Accepté

## Contexte

Dans le cadre du projet MaintAI, plusieurs entraînements ont été réalisés et enregistrés avec MLflow afin de comparer les performances de différents modèles et paramètres.

La principale contrainte métier est qu'une panne non détectée coûte plus cher qu'une fausse alerte. Le recall constitue donc notre métrique prioritaire.

## Décision

Nous retenons le modèle de régression logistique avec un seuil de décision de 0.2, entraîné sur le dataset V1.

Les performances obtenues sont :

- Recall : 0.838
- Precision : 0.543
- Accuracy : 0.882

Ce modèle présente le meilleur recall parmi les entraînements réalisés.

## Conséquences

Ce choix permet de détecter davantage de pannes réelles, mais entraîne une augmentation des fausses alertes.

Nous acceptons cette diminution de la précision afin de limiter le risque de pannes non détectées.

Une évaluation sur un jeu de test indépendant reste nécessaire avant une utilisation en production.

## Traçabilité

- Modèle : Logistic Regression
- Seuil : 0.2
- Dataset : V1 — 5000 relevés
- Commit des données : `76c495a`
- Run MLflow : `5bbad30fca884e448bf8f82a98b60b0c`
- data_md5 : `b7fec4274774842dc1f48202c6013490`
- Version de scikit-learn : 1.8.0
La décision concernant le choix du modèle est documentée dans `docs/adr/004-choix-du-modele.md`.

Nous avons retenu la régression logistique avec un seuil de 0.2, entraînée sur le dataset V1, car elle présente le meilleur recall (83,8 %) parmi les essais réalisés.
## 6. Questions de compréhension

1. Quelle différence entre Git, DVC et MLflow ? Que versionne chacun ?
- Git versionne le code source et les fichiers de métadonnées DVC.
- DVC versionne les datasets et permet de sauvegarder et restaurer leurs différentes versions.
- MLflow conserve les entraînements, leurs paramètres, leurs métriques, les modèles et les versions des dépendances.
2. Pourquoi le stockage DVC est-il ajouté avec `--local` dans ce TP ? Que faudrait-il pour que tout le groupe partage les mêmes données ?
   L'option `--local` permet d'enregistrer la configuration du stockage dans le fichier `.dvc/config.local`, qui n'est pas versionné dans Git. Cela évite de partager un chemin spécifique à notre ordinateur.

Pour que tous les membres du groupe puissent accéder aux mêmes données, il faudrait configurer un stockage distant commun, comme un serveur partagé, Google Drive ou Amazon S3, avec les autorisations nécessaires.
3. Livrer un modèle, c'est trois choses (TP4). Lesquelles, et où sont-elles maintenant ?
   1. Le code source : versionné dans Git.
2. Les données d'entraînement : versionnées et sauvegardées avec DVC.
3. Le modèle entraîné : enregistré dans MLflow avec son identifiant de run et les versions exactes des bibliothèques nécessaires à sa reproduction.

## 7. Le rendu Git

- Lien de la pull request : …
- Le tag `tp5` existe sur GitHub : oui / non
- `git status` avant le commit : `mlflow.db`, `mlruns/`, `data/sensors.csv` ou `.dvc/cache` apparaissaient-ils ? oui / non
Non. Ces fichiers et dossiers sont correctement exclus du suivi Git grâce aux fichiers .gitignore.

## 8. Ce qui reste flou

Le fonctionnement local de Git, DVC et MLflow est compris. Je souhaiterais approfondir leur utilisation dans un environnement de production, notamment la configuration d'un stockage distant partagé et l'automatisation du suivi des entraînements.
