# Compte rendu · TP4 : MaintAI devient un package Python

**Groupe :** Badre Elmouden, Salma Hmid, Amal Dalouch, Soumia Abbassi  ·  **Membres :** Soumia Abbassi, …  ·  **Date :** 11/11/2026

> Remplissez chaque partie dans ce fichier, puis rendez-le avec la pull request du TP4.
> Collez les sorties du terminal entre les lignes ``` (trois accents graves).

---

## 1. Mise en place (dans `TP/TP4`, avec son propre `.venv`)

Dernière ligne de `python -m pytest` juste après `pip install -r requirements.txt` :

```
=================== 2 failed, 25 passed, 1 warning in 3.53s ====================
```

Quels sont les deux tests qui échouent, et pourquoi ?

Les deux tests de `tests/test_package.py`. Ils vérifient que `maintai` est **installé** comme package, ce qui n'est pas encore le cas. Python trouve le code (grâce à `pythonpath = .` dans `pytest.ini`), mais pip n'a enregistré aucune fiche (métadonnées) pour `maintai`.

- `test_installed_version_matches_the_code` : `importlib.metadata.PackageNotFoundError: No package metadata was found for maintai`. `version("maintai")` lit les métadonnées d'installation, qui n'existent pas.
- `test_command_is_declared` : `AssertionError: assert 'maintai-predict' in ['normalizer', 'pyproject-build', 'f2py', ...]`. La commande n'est ni déclarée dans `[project.scripts]` ni installée : elle n'apparaît pas parmi les `console_scripts` de l'environnement.

## 2. Sprint 1 : `pyproject.toml` et `pip install -e .`

Vos sections `dependencies` et `[project.scripts]` :

```toml
dependencies = [
    "numpy>=1.26",
    "pandas>=2.2",
    "scikit-learn>=1.5",
    "joblib>=1.4",
    "httpx>=0.27",
]

[project.scripts]
maintai-predict = "maintai.cli:main"
```

Chaque dépendance correspond à un `import` du dossier `maintai/` : `numpy` (dataset.py), `pandas` (dataset.py, data.py, predict.py), `scikit-learn` (data.py, train.py), `joblib` (train.py, predict.py), `httpx` (client.py). Les autres imports (`os`, `json`, `argparse`, `pathlib`, `dataclasses`…) viennent de la bibliothèque standard.

Pourquoi `pytest` et `fastapi` ne sont-ils pas dans `dependencies` ?

`dependencies` liste ce qu'il faut **toujours** installer pour que le package fonctionne. Or :

- `pytest` (et `ruff`) ne servent qu'à développer : les tests ne font pas partie du package. Ils sont dans le groupe `dev`.
- `fastapi` et `uvicorn` ne servent qu'à faire tourner l'API (`maintai/api/`), et `streamlit` qu'à l'interface. Quelqu'un qui veut seulement `maintai-predict` ou le client n'en a pas besoin. Ils sont dans les groupes optionnels `api` et `ui`, installés sur demande : `pip install -e ".[api,dev]"`.

Garder `dependencies` minimal rend le package plus léger et réduit les conflits avec les autres packages de l'environnement.

Sortie de `pip show maintai` (les quatre premières lignes) :

```
Name: maintai
Version: 0.2.0
Summary: Prédiction de panne de machines à partir de relevés de capteurs (modèle, API, client et commande maintai-predict)
Home-page:
```

`pip show` indique aussi `Editable project location: ...\TP\TP4` et `Requires: httpx, joblib, numpy, pandas, scikit-learn`.

`maintai-predict 72 6 85 15000` affiche `Probabilité de panne : 95 % (ALERTE)`.

Dernière ligne de `python -m pytest` à la fin du sprint 1 :

```
======================== 27 passed, 1 warning in 1.95s =========================
```

Placez-vous dans un autre dossier (`cd %TEMP%`) et lancez `maintai-predict 72 6 85 15000`, puis `python -c "import maintai; print(maintai.__file__)"`. Qu'affichent ces deux commandes ? Où Python trouve-t-il le code de `maintai` ?

```
Probabilité de panne : 95 % (ALERTE)
...\TP\TP4\maintai\__init__.py
```

La commande marche depuis n'importe quel dossier. Python trouve le code **dans le dossier du projet `TP\TP4\maintai\`**, et non dans une copie. `pip install -e .` n'a rien copié : il a posé dans `site-packages` un fichier `__editable__.maintai-0.2.0.pth` et un « finder » (`__editable___maintai_0_2_0_finder.py`). Au démarrage de Python, ce fichier `.pth` ajoute un chercheur de modules qui redirige `import maintai` vers `TP\TP4\maintai`. C'est pour cela que `streamlit run ui/app.py` fonctionne maintenant sans `python -m` : `maintai` est trouvé quel que soit le dossier de départ.

Modifiez le message affiché par `maintai/cli.py`, puis relancez `maintai-predict` sans réinstaller. Le changement est-il visible ? Pourquoi ?

Oui, immédiatement. Après avoir ajouté `[MaintAI modifié]` au message :

```
[MaintAI modifié] Probabilité de panne : 95 % (ALERTE)
```

En mode éditable, l'installation est un **lien** vers le dossier du projet : la commande importe `maintai.cli` depuis `TP\TP4\maintai\cli.py` à chaque lancement, donc elle exécute la dernière version enregistrée. Il ne faut réinstaller que si `pyproject.toml` change (nouvelle dépendance, nouvelle commande dans `[project.scripts]`, nouvelle version), parce que ces informations sont lues à l'installation. J'ai ensuite remis le message d'origine.

## 3. Sprint 2 : construire et installer le wheel

Sortie de `dir dist` après `python -m build` :

```
maintai-0.2.0-py3-none-any.whl     11 608 octets
maintai-0.2.0.tar.gz               12 590 octets
```

`python -m build` fabrique deux fichiers : le **wheel** (`.whl`, prêt à installer) et la **distribution source** (`.tar.gz`, les sources à reconstruire).

Contenu du wheel (`tar -tf dist\maintai-0.2.0-py3-none-any.whl`). Le modèle, les tests et `ui/` y sont-ils ? Pourquoi ?

```
maintai/__init__.py
maintai/cli.py
maintai/client.py
maintai/config.py
maintai/data.py
maintai/dataset.py
maintai/predict.py
maintai/train.py
maintai/api/__init__.py
maintai/api/main.py
maintai/api/schemas.py
maintai/api/security.py
maintai-0.2.0.dist-info/METADATA
maintai-0.2.0.dist-info/WHEEL
maintai-0.2.0.dist-info/entry_points.txt
maintai-0.2.0.dist-info/top_level.txt
maintai-0.2.0.dist-info/RECORD
```

**Non**, ni le modèle, ni les tests, ni `ui/`. Le wheel ne contient que le code Python du dossier `maintai/`, parce que `[tool.setuptools.packages.find]` dit `include = ["maintai*"]`. Il contient aussi la fiche `dist-info` : `METADATA` (nom, version, `Requires-Dist`) et `entry_points.txt` (`maintai-predict = maintai.cli:main`).

- `models/` est en dehors de `maintai/` : c'est un artefact produit par l'entraînement, qui change sans que le code change, et qui est exclu de Git. Il se livre à part.
- `tests/` sert à développer, pas à utiliser le package (les tests se retrouvent seulement dans le `.tar.gz` de la distribution source).
- `ui/app.py` est une application qui *utilise* `maintai`, pas une partie de la bibliothèque. Elle demande en plus Streamlit.

Dans un environnement neuf (`%TEMP%\essai`), installez le wheel et lancez `maintai-predict` :

| Étape | Ce qui s'affiche |
|---|---|
| sans `MAINTAI_MODELS_DIR` | `Modèle introuvable : %TEMP%\essai\.venv\Lib\site-packages\models\model.joblib` puis `Entraînez-le (python -m maintai.train) ou indiquez son dossier dans MAINTAI_MODELS_DIR.` (code de sortie 1). Le wheel n'a pas de modèle, et `config.py` le cherche à côté du package installé, dans `site-packages\models`. |
| avec `MAINTAI_MODELS_DIR` vers vos `models` | Trois avertissements `InconsistentVersionWarning: Trying to unpickle estimator StandardScaler / LogisticRegression / Pipeline from version 1.8.0 when using version 1.9.1. This might lead to breaking code or invalid results. Use at your own risk.`, puis `Probabilité de panne : 95 % (ALERTE)`. |
| après `pip install scikit-learn==1.8.0` | `Probabilité de panne : 95 % (ALERTE)`, sans aucun avertissement. |

D'où vient l'avertissement `InconsistentVersionWarning` ? Quelle ligne de `models/model_info.json` permet de le comprendre ?

Le wheel dit `scikit-learn>=1.5` : dans l'environnement neuf, pip a installé **la version la plus récente**, scikit-learn 1.9.1 (et pandas 3.1.1, numpy 2.5.4). Le modèle `model.joblib` est un objet scikit-learn sérialisé (pickle) avec la version 1.8.0. Quand une autre version de scikit-learn le relit, elle prévient que la structure interne des objets a pu changer : la prédiction peut être fausse, ou le chargement peut échouer. La ligne qui l'explique :

```json
"sklearn_version": "1.8.0"
```

Installer exactement `scikit-learn==1.8.0` (la version de `requirements.txt`) supprime l'avertissement. Livrer un modèle, c'est donc livrer trois choses : le code (le wheel), le modèle (`models/`) et les versions exactes (`requirements.txt`).

## 4. Questions de compréhension

1. Quelle différence entre `requirements.txt` et la liste `dependencies` de `pyproject.toml` ?
   - `requirements.txt` décrit **un environnement complet et figé** (`==`) : toutes les bibliothèques du projet, outils compris (pytest, fastapi, streamlit, build), dans les versions exactes avec lesquelles le modèle a été entraîné. Il sert à reproduire l'environnement à l'identique et à relire le modèle sans risque.
   - `dependencies` décrit **ce dont le package a besoin pour être importé**, avec des versions minimales (`>=`). Pip choisit les versions, ce qui permet d'installer `maintai` à côté d'autres packages sans conflit inutile.

   En bref, `pyproject.toml` dit « au moins » et `requirements.txt` dit « exactement ».
2. Quelle différence entre `pip install -e .` et `pip install dist\maintai-0.2.0-py3-none-any.whl` ?
   - `pip install -e .` (éditable) installe un **lien** vers le dossier du projet. Les modifications du code sont visibles tout de suite, sans réinstaller, mais le dossier source doit rester présent. On l'utilise pour **développer**.
   - Installer le wheel **copie** le code dans `site-packages`. C'est une photo figée de la version 0.2.0, indépendante du dossier source : pour livrer une correction, il faut reconstruire le wheel et le réinstaller. On l'utilise pour **livrer** (à un collègue, sur un serveur).
3. Qu'est-ce qu'un « point d'entrée » (`[project.scripts]`) ? Où pip crée-t-il le fichier `maintai-predict.exe` ?
   Un point d'entrée associe le nom d'une commande à une fonction Python (`maintai-predict = "maintai.cli:main"`). À l'installation, pip génère un petit lanceur qui importe `maintai.cli` et exécute `sys.exit(main())` avec le Python de l'environnement : la valeur renvoyée par `main` (0 ou 1) devient le code de sortie de la commande. Sous Windows, pip le crée dans **`.venv\Scripts\maintai-predict.exe`** (sous Linux ou macOS, dans `.venv/bin/maintai-predict`). Ce dossier est dans le `PATH` quand le `.venv` est activé. Comme le lanceur est créé à l'installation, une commande ajoutée après coup demande de relancer `pip install -e .`.

## 5. Le rendu Git

- Lien de la pull request : …
- Le tag `tp4` existe sur GitHub : oui / non
- `git status` avant le commit : `dist/`, `build/` ou `maintai.egg-info/` apparaissaient-ils ? **non** : le `.gitignore` du starter exclut `dist/`, `build/`, `*.egg-info/` (ainsi que `.venv/`, `models/`, `data/*.csv` et `.env`). Le wheel se reconstruit avec `python -m build` : il ne va pas dans Git.

## 6. Ce qui reste flou

- Comment livrer le modèle proprement avec le package : faut-il le publier à part (dossier partagé, registre de modèles) ou l'inclure comme donnée du package, et dans ce cas comment gérer sa taille et ses versions ?
- Comment publier le wheel pour que les collègues l'installent simplement (dépôt interne de type PyPI) plutôt que de s'envoyer le fichier `.whl` ?
