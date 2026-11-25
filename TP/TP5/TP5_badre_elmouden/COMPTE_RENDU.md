# Compte rendu · TP5 : suivre les entraînements (MLflow) et versionner les données (DVC)

**Groupe :** 5e IADATA · G1  ·  **Membres :** Badre El Mouden,  ·  **Date :** 25/11/2026

> Remplissez chaque partie dans ce fichier, puis rendez-le avec la pull request du TP5.
> Collez les sorties du terminal entre les lignes ``` (trois accents graves).

---

## 1. Mise en place (dans `TP/TP5`)

Dernière ligne de `python -m pytest` après `pip install -r requirements.txt` et `pip install -e .` :

```
5 failed, 27 passed, 2 warnings
```

Quels tests échouent, et pourquoi ?

Les 5 tests de `tests/test_tracking.py` échouent : `test_training_creates_a_run`, `test_run_records_metrics`, `test_run_records_data_version`, `test_run_stores_the_model` et `test_options_change_the_run`. Dans le starter, `log_run` n'est qu'un TODO qui renvoie `""` : aucun run n'est enregistré, et `mlflow.search_runs` ne trouve rien (« aucun run MLflow : log_run n'a rien enregistré »). Les 27 autres tests, ceux de l'API, du client, de la CLI, des données, de l'entraînement et de l'UI, viennent du TP4 corrigé et passent.

Avant `pip install -e .`, où pointait le package `maintai` installé dans l'environnement ? Et après ?

Avant, l'environnement était celui du TP4 : le package `maintai` installé en mode éditable pointait vers `TP\TP4`, avec sa version et sa commande `maintai-predict`. `test_installed_version_matches_the_code` échoue dans ce cas, parce que la version installée n'est pas celle du code du TP5. Après `pip install -e .` dans `TP\TP5`, le package pointe vers `TP\TP5\maintai` (version 0.3.0), et `version("maintai") == maintai.__version__`.

## 2. Sprint 1 : MLflow

Votre fonction `log_run` (`maintai/tracking.py`) :

```python
def log_run(params: dict, metrics: dict, model, X_example, data_path: Path) -> str:
    """Enregistre un entraînement : paramètres, version des données, métriques et modèle. Renvoie l'id du run."""
    # 1. Toutes les runs de MaintAI sont rangées dans la même expérience
    mlflow.set_experiment(EXPERIMENT)
    # 2. Tout ce qui est enregistré dans le bloc with appartient à ce run
    with mlflow.start_run() as run:
        # 3a. Les réglages de l'entraînement (model, C, threshold, n_train)
        mlflow.log_params(params)
        # 3b. La version des données : même md5 que celui écrit par DVC dans data/sensors.csv.dvc
        mlflow.log_param("data_md5", file_md5(data_path))
        # 3c. Les résultats mesurés sur le jeu de test
        mlflow.log_metrics(metrics)
        # 3d. Le modèle, avec un exemple d'entrée et les versions exactes des bibliothèques
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_example,
            skops_trusted_types=TRUSTED_TYPES,
        )
    # 4. L'identifiant permet de retrouver ce run (et donc ses données) dans mlflow ui
    return run.info.run_id
```

Dernière ligne de `python -m pytest` à la fin du sprint 1 :

```
32 passed, 6 warnings
```

Vos runs (au moins quatre), relevés dans `mlflow ui` (mode « Model training ») :

| Commande | recall | precision | accuracy |
|---|---|---|---|
| `python -m maintai.train` | 0.750 | 0.622 | 0.904 |
| `python -m maintai.train --C 0.01` | 0.691 | 0.653 | 0.908 |
| `python -m maintai.train --threshold 0.2` | 0.838 | 0.543 | 0.882 |
| `python -m maintai.train --model forest` | 0.728 | 0.562 | 0.886 |
| (votre essai) `python -m maintai.train --threshold 0.15` | 0.875 | 0.500 | 0.864 |
| (votre essai) `python -m maintai.train --model forest --threshold 0.2` | 0.831 | 0.479 | 0.854 |

Tous ces runs ont `data_md5 = 73e0b7cb3c6f795c70c3ddf970b99c15` (v1), `n_train = 4000` et `baseline_recall = 0.0`.

Quel run choisiriez-vous pour MaintAI, et pourquoi ? (Rappel : une panne manquée coûte plus cher qu'une fausse alerte.)

Je choisis la **régression logistique avec `--threshold 0.2`**. Le recall est le critère principal : au seuil 0.3, le modèle rate 34 pannes sur 136 ; au seuil 0.2, il n'en rate plus que 22. En échange, il y a 96 fausses alertes au lieu de 62, soit environ 3 inspections inutiles par panne en plus détectée, ce qui reste acceptable. Le seuil 0.15 détecte encore 5 pannes de plus, mais au prix de 23 fausses alertes supplémentaires : une alerte sur deux devient fausse, et l'accuracy (0.864) retombe au niveau de la baseline. Le C = 0.01 fait baisser le recall : il est écarté.

**La forêt aléatoire n'est pas meilleure** : au même seuil, elle a un recall **et** une precision plus faibles que la régression logistique (0.728 / 0.562 contre 0.750 / 0.622 au seuil 0.3 ; 0.831 / 0.479 contre 0.838 / 0.543 au seuil 0.2). En plus, elle est plus lourde (200 arbres). Un modèle plus complexe ne garantit pas un meilleur résultat. Détails dans l'ADR 004.

Dans `mlflow ui`, ouvrez le modèle d'un run. Quelle version de scikit-learn est indiquée dans son `requirements.txt` ? Quel problème du TP4 cela règle-t-il ?

`scikit-learn==1.8.0`, avec `mlflow==3.18.0`, `numpy==2.4.4`, `pandas==3.0.2`, `scipy==1.18.1` et `skops==0.16.0`. Au TP4, un modèle sauvegardé avec une version de scikit-learn pouvait être rechargé avec une autre, ce qui donne un avertissement de version incohérente, voire un modèle faux ou impossible à charger. Il fallait noter la version à la main (`sklearn_version` dans `model_info.json`) et figer `requirements.txt`. Maintenant, MLflow enregistre automatiquement, **avec chaque modèle**, les versions exactes des bibliothèques (`requirements.txt`, `conda.yaml`, `python_env.yaml`) : on sait exactement quel environnement recréer pour recharger ce modèle.

Où MLflow enregistre-t-il les runs sur votre disque ? Ces fichiers vont-ils dans Git ? Pourquoi ?

Dans le dossier courant `TP\TP5` :
- `mlflow.db` : une base SQLite avec les expériences, les runs, les paramètres et les métriques ;
- `mlruns/` : les artefacts, c'est-à-dire chaque modèle (`model.skops`, `MLmodel`, `requirements.txt`, `input_example.json`…).

Ils ne vont **pas** dans Git : ils sont exclus par le `.gitignore` (`mlflow.db`, `mlruns/`). Ce sont des fichiers binaires, lourds, qui grossissent à chaque entraînement et sont propres à chaque machine (chemins locaux). Git doit garder le code. Pour partager les runs dans l'équipe, il faudrait un serveur MLflow commun.

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

Les deux valeurs sont **identiques** : `73e0b7cb3c6f795c70c3ddf970b99c15`. `file_md5` calcule la même empreinte que DVC, sur les octets du fichier. Le run MLflow et le reçu DVC sont donc reliés : à partir d'un run, on lit `data_md5`, on retrouve dans `git log` le commit dont le `.dvc` porte ce md5, puis `git checkout <commit> -- data/sensors.csv.dvc` et `dvc checkout` (ou `dvc pull`) redonnent exactement les données de l'entraînement.

Remarque : le md5 porte sur les octets. Sous Windows, pandas écrit des fins de ligne `\r\n`. Les mêmes 5000 relevés générés sous Linux ou macOS donnent donc un autre md5 (`b7fec427…`, 121814 octets). C'est une raison de plus pour récupérer les données avec `dvc pull` au lieu de les régénérer.

Pourquoi `data/sensors.csv` ne va pas dans Git, alors que `data/sensors.csv.dvc` y va ?

Git est fait pour le code (du texte, petit, avec des différences ligne par ligne). Un fichier de données peut peser des centaines de Mo, et Git garderait chaque version pour toujours dans l'historique : le dépôt deviendrait énorme et lent. Le CSV est donc exclu (`data/*.csv` dans le `.gitignore`, et `git add data/sensors.csv` est refusé). DVC le garde dans son cache (`.dvc/cache`) et dans le stockage. Git ne versionne que le **reçu** `.dvc` : 5 lignes, avec le md5, la taille et le chemin. Ce reçu suffit à désigner une version précise des données, et il voyage avec le code dans chaque commit.

Après `python -m maintai.dataset --n 8000 --seed 7`, que dit `dvc status` ?

```
data/sensors.csv.dvc:
	changed outs:
		modified:           data/sensors.csv
```

| Version | Commit Git (`git log --oneline`) | md5 (8 premiers caractères) | Nombre de lignes |
|---|---|---|---|
| v1 | [À COMPLÉTER : hash] Données v1 | 73e0b7cb | 5000 |
| v2 | [À COMPLÉTER : hash] Données v2 | 93fdeef2 | 8000 |

Revenez à la v1 (`git checkout <commit> -- data/sensors.csv.dvc`, puis `dvc checkout`). Combien de lignes contient `data/sensors.csv` ? Puis revenez à la v2.

Après le retour à la v1, `data/sensors.csv` contient **5000 lignes** (md5 `73e0b7cb…`). `dvc checkout` a remis depuis le cache le fichier qui correspond à l'ancien reçu. `git status` montre `data/sensors.csv.dvc` modifié : c'est l'ancien reçu, on ne le commite pas. Avec `git checkout HEAD -- data/sensors.csv.dvc` puis `dvc checkout`, on revient à la v2 : **8000 lignes**, et `git status` est de nouveau propre.

Supprimez `data/sensors.csv` et le dossier `.dvc/cache`, puis lancez `dvc pull`. Que se passe-t-il ?

```
A       data/sensors.csv
1 file fetched and 1 file added
```

DVC lit le reçu `data/sensors.csv.dvc` (md5 `93fdeef2…`) et ne trouve le fichier ni dans l'espace de travail ni dans le cache, puisqu'il a été supprimé. Il le **télécharge depuis le stockage** (`%USERPROFILE%\dvc-stockage`), recrée `.dvc/cache`, puis remet `data/sensors.csv` à sa place. On retrouve 8000 lignes avec le même md5, et `dvc status` répond « Data and pipelines are up to date ». C'est exactement ce que fait un coéquipier après `git pull` : le code et le reçu viennent de Git, les données viennent du stockage DVC.

## 4. Relier modèle et données

Entraînez un modèle sur la v2. Dans `mlflow ui`, comparez ce run à celui de la v1 avec les mêmes réglages :

| Run | data_md5 (8 caractères) | n_train | recall | precision |
|---|---|---|---|---|
| v1 | 73e0b7cb | 4000 | 0.750 | 0.622 |
| v2 | 93fdeef2 | 6400 | 0.741 | 0.650 |

(Mêmes réglages : régression logistique, C = 1.0, seuil 0.3.)

Peut-on dire que le modèle v2 est « meilleur » ? Pourquoi faut-il être prudent en comparant deux runs entraînés sur des données différentes ?

**Non.** Le run v2 a une meilleure precision (0.650 contre 0.622) mais un recall légèrement plus bas (0.741 contre 0.750), alors que le recall est notre critère principal. Surtout, les deux runs ne sont **pas évalués sur le même examen**. Le jeu de test v1 contient 1000 relevés dont 136 pannes ; celui de la v2 en contient 1600, tirés avec une autre graine, dont 205 pannes (taux de panne 12.8 % contre 13.6 %). Une différence de métrique peut donc venir des données de test autant que du modèle. Les écarts sont aussi minimes : 0.9 point de recall représente environ 1 panne sur 136.

Pour comparer honnêtement, il faut évaluer les deux modèles **sur le même jeu de test** (un jeu de référence fixe, lui-même versionné avec DVC), et idéalement répéter la mesure (validation croisée) pour voir la variabilité. En faisant ce test, le modèle v2 n'est pas meilleur au seuil 0.3. Sur le test v1, il obtient un recall de 0.735 contre 0.750 pour le modèle v1. Sur le test v2, il obtient 0.741 contre 0.761 pour le modèle v1.

## 5. Décision : l'ADR 004

Dans `docs/adr/004-choix-du-modele.md` : le modèle et le seuil retenus, à partir de votre tableau MLflow. Indiquez l'identifiant du run et le `data_md5` de la version des données utilisée.

Fait : régression logistique, C = 1.0, **seuil 0.2**, run `0aad8b4c63c240fba816c0632d1e484b`, `data_md5 = 73e0b7cb3c6f795c70c3ddf970b99c15` (v1). Voir `docs/adr/004-choix-du-modele.md`.

## 5 bis. Recherche par grille (30 runs au maximum)

Au lieu de lancer les réglages un par un, `python -m maintai.search` (nouveau module `maintai/search.py`) essaie toutes les combinaisons d'une grille. Chaque combinaison devient un run MLflow : 30 runs « enfants » rangés sous un run parent `grid-search`, dans l'expérience `maintai-grid-search`. Les runs de l'expérience `maintai` restent ainsi lisibles.

**La grille : 30 combinaisons, c'est-à-dire 30 runs, le maximum autorisé.**

| Modèle | Hyperparamètre essayé | Seuils essayés | Combinaisons |
|---|---|---|---|
| Régression logistique | C ∈ {0.01, 0.1, 1, 10} | {0.15, 0.2, 0.25, 0.3, 0.4} | 4 × 5 = 20 |
| Forêt aléatoire (200 arbres) | max_depth ∈ {None, 8} | les mêmes 5 seuils | 2 × 5 = 10 |
| **Total** | | | **30** |

Le plafond est écrit dans le code (`MAX_RUNS = 30`) : une grille plus grande est refusée avant le moindre entraînement, avec le message « La grille contient N combinaisons, plus que le maximum de 30 runs ». L'option `--max-runs` permet de le baisser.

**Comment chaque combinaison est mesurée :**
- **Validation croisée stratifiée à 5 plis, sur les 4000 relevés d'entraînement seulement.** Le jeu de test ne sert pas à choisir. Sinon, on retiendrait le réglage qui a eu de la chance sur ces 1000 relevés, et sa mesure sur le test serait optimiste.
- **Le seuil ne demande pas de ré-entraîner.** Pour chaque modèle, on calcule une fois la probabilité « hors pli » de chaque relevé (prédite par un modèle qui ne l'a pas vu), puis on la compare aux 5 seuils. Il faut donc 6 modèles × 5 plis = 30 entraînements pour 30 runs.
- **Critère : coût = 4 × pannes manquées + 1 × fausse alerte**, ramené à 1000 relevés. Le rapport 4 reprend l'arbitrage de l'ADR 004 : on y a accepté 2,8 fausses alertes par panne détectée en plus (seuil 0.3 → 0.2) et refusé 4,6 (seuil 0.2 → 0.15).
- **Le gagnant est ré-entraîné** sur les 4000 relevés, puis mesuré **une seule fois** sur le test.

Commande et fin de la sortie :

```
python -m maintai.search
...
30 runs enfants dans l'expérience maintai-grid-search (run parent [À COMPLÉTER : id])
Meilleur réglage (validation croisée) : logreg (C=10.0), seuil 0.2 — run [À COMPLÉTER : id]
Mesure unique sur le jeu de test : {"accuracy": 0.882, "recall": 0.838, "precision": 0.543, "threshold": 0.2}
```

Les 10 meilleures combinaisons, en validation croisée sur les données v1 (`data_md5 = 73e0b7cb…`) :

| Rang | Modèle | Réglage | Seuil | recall CV | precision CV | accuracy CV | pannes manquées / 542 | fausses alertes / 3458 | coût / 1000 relevés |
|---|---|---|---|---|---|---|---|---|---|
| 1 | logreg | C = 10 | 0.2 | 0.832 | 0.581 | 0.896 | 91 | 325 | 172.2 |
| 2 | logreg | C = 1 | 0.2 | 0.832 | 0.579 | 0.895 | 91 | 328 | 173.0 |
| 3 | logreg | C = 0.1 | 0.2 | 0.836 | 0.573 | 0.893 | 89 | 338 | 173.5 |
| 4 | logreg | C = 0.01 | 0.2 | 0.838 | 0.563 | 0.890 | 88 | 352 | 176.0 |
| 5 | logreg | C = 1 | 0.25 | 0.795 | 0.624 | 0.907 | 111 | 260 | 176.0 |
| 6 | logreg | C = 0.1 | 0.25 | 0.793 | 0.624 | 0.907 | 112 | 259 | 176.8 |
| 7 | logreg | C = 10 | 0.25 | 0.793 | 0.623 | 0.907 | 112 | 260 | 177.0 |
| 8 | logreg | C = 10 | 0.15 | 0.865 | 0.522 | 0.875 | 73 | 429 | 180.2 |
| 9 | logreg | C = 1 | 0.15 | 0.865 | 0.519 | 0.873 | 73 | 434 | 181.5 |
| 10 | logreg | C = 0.1 | 0.15 | 0.876 | 0.509 | 0.869 | 67 | 459 | 181.8 |

<details>
<summary>Les 30 combinaisons (aussi dans <code>models/grid_search.csv</code> et dans <code>mlflow ui</code>)</summary>

| Rang | Modèle | Réglage | Seuil | recall CV | precision CV | accuracy CV | pannes manquées / 542 | fausses alertes / 3458 | coût / 1000 relevés |
|---|---|---|---|---|---|---|---|---|---|
| 1 | logreg | C = 10 | 0.2 | 0.832 | 0.581 | 0.896 | 91 | 325 | 172.2 |
| 2 | logreg | C = 1 | 0.2 | 0.832 | 0.579 | 0.895 | 91 | 328 | 173.0 |
| 3 | logreg | C = 0.1 | 0.2 | 0.836 | 0.573 | 0.893 | 89 | 338 | 173.5 |
| 4 | logreg | C = 0.01 | 0.2 | 0.838 | 0.563 | 0.890 | 88 | 352 | 176.0 |
| 5 | logreg | C = 1 | 0.25 | 0.795 | 0.624 | 0.907 | 111 | 260 | 176.0 |
| 6 | logreg | C = 0.1 | 0.25 | 0.793 | 0.624 | 0.907 | 112 | 259 | 176.8 |
| 7 | logreg | C = 10 | 0.25 | 0.793 | 0.623 | 0.907 | 112 | 260 | 177.0 |
| 8 | logreg | C = 10 | 0.15 | 0.865 | 0.522 | 0.875 | 73 | 429 | 180.2 |
| 9 | logreg | C = 1 | 0.15 | 0.865 | 0.519 | 0.873 | 73 | 434 | 181.5 |
| 10 | logreg | C = 0.1 | 0.15 | 0.876 | 0.509 | 0.869 | 67 | 459 | 181.8 |
| 11 | logreg | C = 1 | 0.3 | 0.751 | 0.661 | 0.914 | 135 | 209 | 187.2 |
| 12 | logreg | C = 10 | 0.3 | 0.751 | 0.660 | 0.914 | 135 | 210 | 187.5 |
| 13 | logreg | C = 0.1 | 0.3 | 0.745 | 0.670 | 0.916 | 138 | 199 | 187.8 |
| 14 | logreg | C = 0.01 | 0.25 | 0.762 | 0.633 | 0.908 | 129 | 239 | 188.8 |
| 15 | logreg | C = 0.01 | 0.15 | 0.900 | 0.461 | 0.844 | 54 | 571 | 196.8 |
| 16 | forest | forêt, max_depth = 8 | 0.2 | 0.791 | 0.547 | 0.883 | 113 | 355 | 201.8 |
| 17 | forest | forêt, max_depth = 8 | 0.15 | 0.840 | 0.488 | 0.859 | 87 | 478 | 206.5 |
| 18 | logreg | C = 0.01 | 0.3 | 0.686 | 0.717 | 0.921 | 170 | 147 | 206.8 |
| 19 | forest | forêt, max_depth = 8 | 0.25 | 0.740 | 0.591 | 0.895 | 141 | 278 | 210.5 |
| 20 | forest | forêt, max_depth = None | 0.2 | 0.795 | 0.518 | 0.872 | 111 | 401 | 211.2 |
| 21 | forest | forêt, max_depth = None | 0.25 | 0.756 | 0.557 | 0.885 | 132 | 326 | 213.5 |
| 22 | logreg | C = 10 | 0.4 | 0.670 | 0.723 | 0.920 | 179 | 139 | 213.8 |
| 23 | forest | forêt, max_depth = None | 0.3 | 0.720 | 0.609 | 0.899 | 152 | 250 | 214.5 |
| 24 | logreg | C = 1 | 0.4 | 0.668 | 0.724 | 0.920 | 180 | 138 | 214.5 |
| 25 | forest | forêt, max_depth = None | 0.15 | 0.834 | 0.468 | 0.849 | 90 | 513 | 218.2 |
| 26 | forest | forêt, max_depth = 8 | 0.3 | 0.696 | 0.633 | 0.904 | 165 | 219 | 219.8 |
| 27 | logreg | C = 0.1 | 0.4 | 0.648 | 0.731 | 0.920 | 191 | 129 | 223.2 |
| 28 | forest | forêt, max_depth = None | 0.4 | 0.625 | 0.669 | 0.907 | 203 | 168 | 245.0 |
| 29 | forest | forêt, max_depth = 8 | 0.4 | 0.607 | 0.699 | 0.911 | 213 | 142 | 248.5 |
| 30 | logreg | C = 0.01 | 0.4 | 0.529 | 0.815 | 0.920 | 255 | 65 | 271.2 |

</details>

Que montre la recherche ?

- **Le seuil 0.2 est le meilleur pour chaque valeur de C**, et la régression logistique occupe les 15 premières places. La meilleure forêt (max_depth = 8, seuil 0.2) n'arrive qu'au 16e rang : elle rate 113 pannes au lieu de 91, avec plus de fausses alertes (355 contre 328). Le constat de l'ADR 004 se confirme : un modèle plus complexe ne fait pas mieux ici.
- **Le seuil compte beaucoup plus que C.** Pour un même seuil entre 0.15 et 0.3, C entre 0.1 et 10 change le coût de moins de 1 %. Passer du seuil 0.3 au seuil 0.2 le baisse de 8 %. C = 0.01 régularise trop : au seuil 0.3, le recall tombe à 0.686.
- **Le gagnant (C = 10, seuil 0.2) ne bat pas le réglage de l'ADR 004.** Il n'a que 3 fausses alertes de moins sur 4000 relevés en validation croisée, et sur le jeu de test il donne exactement la même mesure que C = 1.0 au seuil 0.2 : recall 0.838, precision 0.543, accuracy 0.882. On garde donc C = 1.0 (voir l'ADR 005).
- **Le choix dépend du rapport de coût.** Avec un rapport de 4 ou 5, le seuil 0.2 gagne ; avec 3, le seuil 0.25 gagne ; avec 6 ou plus, le seuil 0.15 gagne. Il faut valider ce rapport avec l'équipe de maintenance, puis le passer avec `--cost-fn`.

| Rapport de coût (`--cost-fn`) | 2 | 3 | 4 (défaut) | 5 | 6 | 8 |
|---|---|---|---|---|---|---|
| Réglage gagnant | logreg C = 0.1, seuil 0.3 | logreg C = 1, seuil 0.25 | logreg C = 10, seuil 0.2 | logreg C = 10, seuil 0.2 | logreg C = 0.1, seuil 0.15 | logreg C = 0.1, seuil 0.15 |

Pourquoi limiter la recherche à 30 runs ?

- **Lisibilité** : au-delà de quelques dizaines de runs, la comparaison dans `mlflow ui` devient difficile.
- **Temps de calcul** : chaque valeur ajoutée multiplie la taille de la grille.
- **Risque statistique** : plus on essaie de combinaisons, plus le meilleur score de validation croisée est gonflé par le hasard. C'est déjà visible ici : les trois premiers ne diffèrent que de quelques fausses alertes.

Pour explorer plus large, il faudrait passer à une recherche aléatoire plutôt que d'agrandir la grille.

Tests ajoutés (`tests/test_search.py`) :
- la grille par défaut tient en 30 runs ;
- une grille trop grande est refusée ;
- il y a un run enfant par combinaison, avec le même `data_md5` ;
- le gagnant est celui de plus faible coût en validation croisée, et son seuil est écrit dans `model_info.json` avec `--save`.

Dernière ligne de `python -m pytest` :

```
36 passed
```

Décision correspondante : `docs/adr/005-recherche-par-grille.md`.

## 6. Questions de compréhension

1. Quelle différence entre Git, DVC et MLflow ? Que versionne chacun ?
   - **Git** versionne le **code**, ainsi que les reçus `.dvc` des données (md5, taille, chemin). Commandes : `git add`, `git commit`, `git checkout`.
   - **DVC** versionne les **fichiers de données** eux-mêmes, hors de Git : dans un cache local (`.dvc/cache`) et dans un stockage partagé. Commandes : `dvc add`, `dvc push`, `dvc pull`, `dvc checkout`.
   - **MLflow** garde la trace de **chaque entraînement** (un run) : ses paramètres, l'empreinte des données (`data_md5`), ses métriques et le modèle avec les versions exactes des bibliothèques. Commandes : `log_run` dans le code, `mlflow ui`.
   - En résumé, Git est l'historique du code, DVC l'étiquette des données et MLflow le cahier de manip.
2. Pourquoi le stockage DVC est-il ajouté avec `--local` dans ce TP ? Que faudrait-il pour que tout le groupe partage les mêmes données ?
   - Le stockage est un dossier de **mon** PC (`%USERPROFILE%\dvc-stockage`). Ce chemin n'existe pas chez les coéquipiers. Avec `--local`, il est écrit dans `.dvc/config.local`, qui est ignoré par Git (`.dvc/.gitignore`) : il ne casse pas la configuration des autres.
   - Pour partager les données, il faut un stockage accessible à tous : un bucket S3, Google Drive, un serveur SSH ou un dossier réseau partagé. On l'ajoute **sans** `--local` (par exemple `dvc remote add -d stockage s3://…`) pour que `.dvc/config` soit commité. Chacun met ses identifiants dans son `config.local`, puis fait `git pull` et `dvc pull`.
3. Livrer un modèle, c'est trois choses (TP4). Lesquelles, et où sont-elles maintenant ?
   - **Le code** : dans Git, et dans le wheel du TP4.
   - **Le modèle** : dans MLflow, sous la forme d'un run et de son identifiant (artefact `model`).
   - **Les versions exactes des bibliothèques** : dans MLflow, avec le `requirements.txt` enregistré avec le modèle.
   - Le TP5 ajoute **les données** : le reçu est dans Git, le fichier dans le stockage DVC, et elles sont reliées au run par `data_md5`.

## 7. Le rendu Git

- Lien de la pull request : [À COMPLÉTER : lien GitHub de la PR tp5 → main]
- Le tag `tp5` existe sur GitHub : [À COMPLÉTER : oui / non, après `git tag tp5` et `git push --tags`]
- `git status` avant le commit : `mlflow.db`, `mlruns/`, `data/sensors.csv` ou `.dvc/cache` apparaissaient-ils ? **Non.** Le `.gitignore` de `TP\TP5` exclut `mlflow.db`, `mlruns/`, `models/` et `data/*.csv`, et `.dvc/.gitignore` exclut `/cache`, `/tmp` et `/config.local`. J'ai aussi ajouté `venv/` au `.gitignore`, parce qu'un environnement virtuel `venv\` (et pas seulement `.venv\`) se trouve dans le dossier.

## 8. Ce qui reste flou

Comment comparer proprement deux modèles quand les données changent : faut-il garder un jeu de test de référence fixe, versionné à part avec DVC ? Et comment, en pratique, partager dans le groupe un stockage DVC et un serveur MLflow communs, au lieu d'un dossier et d'une base locaux à chaque PC ?
