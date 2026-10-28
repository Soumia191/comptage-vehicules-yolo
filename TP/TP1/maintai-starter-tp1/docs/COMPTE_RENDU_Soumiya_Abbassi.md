# Compte rendu · TP1 : une API de prédiction pour MaintAI

**Groupe :** G1  ·  **Membres :** Soumiya Abbassi  ·  **Date :** 28/10/2026

> Remplissez chaque partie dans ce fichier, puis rendez-le avec la pull request du TP1.
> Collez les sorties du terminal entre les lignes ``` (trois accents graves).

---

## 1. Mise en place

Dernière ligne de `python -m pytest`, juste après la mise en place (rien n'est encore écrit) :

```
FAILED tests/test_api.py::test_predict_returns_probability - assert 422 == 200
FAILED tests/test_api.py::test_high_risk_reading_raises_alert - KeyError: 'alert'
FAILED tests/test_api.py::test_503_when_model_missing - assert 422 == 503
========= 3 failed, 10 passed, 1 warning in 5.07s ==========
```

Dans `tests/test_api.py`, 4 tests sur 7 passent déjà alors que rien n'est écrit. Lesquels ? Pour les deux tests qui attendent un code 422, expliquez pourquoi ils passent « pour une mauvaise raison » (indice : regardez `extra="forbid"` dans `schemas.py`).

Sur les 7 tests de l'API, voici ceux qui passent sans qu'on ait rien codé :

| Test | Pourquoi il passe |
|---|---|
| `test_health` | `/health` et le chargement du modèle sont fournis dans le squelette |
| `test_openapi_documents_predict` | la route `/v1/predict` est déjà déclarée, donc documentée |
| `test_out_of_range_value_is_rejected` | 422, mais pour une mauvaise raison |
| `test_missing_field_is_rejected` | 422, mais pour une mauvaise raison |

Le piège vient de `extra="forbid"`. Cette option dit à Pydantic : « refuse toute clé que le modèle ne connaît pas ». Or, au départ, `SensorReading` ne connaît **aucune** clé. Chaque requête est donc rejetée dès son premier champ, avec le message `Extra inputs are not permitted`. Les deux tests obtiennent bien leur 422, mais ce n'est pas parce que 500 °C est trop chaud ou que `load_pct` manque : c'est parce que le schéma refuse tout. La requête correcte de `test_predict_returns_probability`, rejetée elle aussi (`assert 422 == 200`), le confirme.

## 2. Le schéma d'entrée (`maintai/api/schemas.py`)

| Variable | Type | Minimum (`ge`) | Maximum (`le`) | Justification physique (pas le min / max des données) |
|---|---|---|---|---|
| `temperature_c` | `float` | -40.0 | 150.0 | Une machine industrielle peut fonctionner dans un environnement très froid, d'où -40 °C, valeur basse habituelle des plages industrielles. En haut, 150 °C est juste sous la tenue thermique des isolants de moteur (classe F, 155 °C) : au-dessus, la machine ne pourrait plus fonctionner, la mesure serait donc fausse. |
| `vibration_mm_s` | `float` | 0.0 | 50.0 | Il s'agit d'une valeur RMS, donc positive ou nulle. D'après l'ISO 10816, quelques mm/s suffisent déjà à signaler un problème et une dizaine de mm/s correspond à des dommages ; 50 mm/s est une limite extrême au-delà de laquelle le capteur est sûrement défectueux. |
| `load_pct` | `float` | 0.0 | 100.0 | Le taux de charge s'exprime en pourcentage de la capacité nominale : entre 0 % (machine à vide) et 100 % (charge maximale). |
| `runtime_hours` | `float` | 0.0 | 200000.0 | Les heures de fonctionnement s'accumulent depuis 0. La borne de 200 000 h équivaut à plus de 22 ans de marche continue, soit plus que la vie d'une machine. |

Nous avons évité de reprendre les valeurs extrêmes des données d'entraînement : ce seraient des limites statistiques, pas physiques, et l'API refuserait à tort des machines réelles un peu différentes de celles du jeu synthétique.

Après avoir écrit `schemas.py` seul, que donne `python -m pytest`, et quel est le message d'erreur des tests qui échouent ? Pourquoi est-ce un bon signe ?

```
FAILED tests/test_api.py::test_predict_returns_probability - NotImplementedError: À écrire : TP1, partie 3
FAILED tests/test_api.py::test_high_risk_reading_raises_alert - NotImplementedError: À écrire : TP1, partie 3
FAILED tests/test_api.py::test_503_when_model_missing - NotImplementedError: À écrire : TP1, partie 3
=================== 3 failed, 10 passed, 1 warning in 2.70s ====================
```

Le compteur reste à 3 échecs, mais l'erreur a changé de nature : ce n'est plus un 422 renvoyé par la validation, c'est une `NotImplementedError` levée dans `predict()`. C'est un bon signe parce qu'il prouve que notre schéma accepte les relevés corrects (la requête va plus loin qu'avant) et que les tests 422 testent maintenant réellement les bornes et les champs obligatoires. L'échec restant pointe exactement la partie suivante du TP.

## 3. La fonction `predict()` (`maintai/api/main.py`)

Votre code :

```python
@router.post("/predict", response_model=PredictionOut)
def predict(reading: SensorReading) -> PredictionOut:
    predictor = state.get("predictor")
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé : entraînez-le puis redémarrez l'API.",
        )
    result = predictor.predict(reading.model_dump())
    return PredictionOut.model_validate(result)
```

Pourquoi répondre **503** et pas 500 quand le modèle manque ?

Les codes de la famille 5xx indiquent tous une erreur côté serveur, mais ils ne disent pas la même chose. 500 signifie « quelque chose a cassé et on ne sait pas quoi ». 503 signifie « le service n'est pas prêt pour l'instant ». Un modèle non entraîné est un état connu et temporaire, que notre code détecte volontairement : 503 est donc le code honnête. Il évite de faire croire à un bug, il indique au client qu'il peut réessayer, et il oriente directement vers la solution (entraîner le modèle puis redémarrer), en cohérence avec `/health` qui affiche `model_loaded: false`.

Pourquoi le modèle est-il chargé **une seule fois**, au démarrage, et pas à chaque requête ?

Une prédiction est rapide ; ouvrir et désérialiser `model.joblib` ne l'est pas. Recharger le modèle à chaque requête multiplierait ce coût par le nombre d'appels, augmenterait le temps de réponse et la consommation de ressources. En le chargeant dans le `lifespan` au démarrage, on paie ce coût une seule fois, le modèle reste en mémoire pour toutes les requêtes, toutes utilisent la même version, et un problème de fichier apparaît dès le lancement du serveur.

## 4. Les tests

Dernière ligne de `python -m pytest` à la fin du TP :

```
======================== 13 passed, 1 warning in 3.66s =========================
```

## 5. L'API en marche

Après `python -m maintai.dataset` et `python -m maintai.train`, lancez `uvicorn maintai.api.main:app --reload`, puis envoyez ces requêtes depuis `/docs` (ou avec `curl.exe`).

| Requête | Code obtenu | Réponse (le corps JSON) |
|---|---|---|
| `GET /health` | 200 | `{"status":"ok","model_loaded":true}` |
| `POST /v1/predict` avec `{"temperature_c": 72.5, "vibration_mm_s": 6.1, "load_pct": 85, "runtime_hours": 15000}` | 200 | `{"failure_probability":0.9553,"alert":true,"threshold":0.3,"model_version":"2026-10-28T17:41:35Z"}` |
| `POST /v1/predict` avec une température de 500 | 422 | `{"detail":[{"type":"less_than_equal","loc":["body","temperature_c"],"msg":"Input should be less than or equal to 150","input":500,"ctx":{"le":150.0}}]}` |
| `POST /v1/predicte` (faute de frappe dans l'adresse) | 404 | `{"detail":"Not Found"}` |

Dans `/docs`, où voit-on les bornes que vous avez choisies ?

Dans Swagger UI, il suffit de déplier `POST /v1/predict` puis de cliquer sur **Schema** à côté de l'exemple du *Request body* : chaque champ y est listé avec son type, sa description, son `minimum` et son `maximum`. La même information est reprise dans la section **Schemas** en bas de page. Tout cela est généré à partir de nos `Field(ge=..., le=...)`.

## 6. Questions de compréhension

1. REST et FastAPI : quelle est la différence ?
   REST est une convention d'architecture pour les API web : on expose des ressources par des URL, on agit dessus avec les méthodes HTTP, et chaque requête se suffit à elle-même (sans état). FastAPI est un outil Python qui implémente une API HTTP. En résumé, REST est la méthode de conception, FastAPI est le moyen technique de la mettre en œuvre.
2. Le schéma Pydantic sert à trois choses. Lesquelles ?
   - **Valider** : il vérifie la présence, le type et les bornes de chaque champ et renvoie un 422 détaillé en cas de problème ;
   - **Convertir** : il transforme le corps JSON en un objet Python typé (`reading.temperature_c` est un `float`) ;
   - **Documenter** : il génère le schéma OpenAPI, donc la page `/docs`, sans écrire de documentation à la main.
3. Dans `http://localhost:8000/v1/predict`, que désignent `localhost`, `8000` et `/v1/predict` ?
   `localhost` est le nom de notre propre machine (127.0.0.1) ; `8000` est le port sur lequel uvicorn reçoit les connexions ; `/v1/predict` est le chemin de l'endpoint : préfixe de version `/v1` puis la route `/predict`.
4. Pourquoi `/v1` dans l'adresse ?
   Il marque la version du contrat de l'API. Le jour où le format change de manière incompatible, on crée `/v2` en parallèle, et les clients existants continuent à fonctionner avec `/v1` le temps de migrer.
5. Dans la commande `uvicorn maintai.api.main:app`, que désignent `maintai.api.main` et `app` ?
   `maintai.api.main` est le module Python, désigné par son chemin d'import avec des points (fichier `maintai/api/main.py`). `app` est l'objet `FastAPI` créé dans ce module, que uvicorn récupère et sert.

## 7. Le rendu Git

- Lien de la pull request : [à compléter]
- Le tag `tp1` existe sur GitHub : [oui / non]
- Qui a relu la pull request, et quel commentaire a-t-il laissé ?
  [à compléter : nom du relecteur et son commentaire]

## 8. Ce qui reste flou

- Pourquoi la version du modèle (`model_version`) est une date d'entraînement et pas un numéro du type `1.0.0` : comment ferait-on pour revenir à un ancien modèle ?
- Comment l'API se comporterait avec plusieurs utilisateurs en même temps, puisque `predict()` n'est pas une fonction `async`.
