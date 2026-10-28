# Compte rendu · TP1 : une API de prédiction pour MaintAI

**Groupe :** G1  ·  **Membres :** Badre El Mouden  ·  **Date :** 28/10/2026

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

Les 4 tests qui passent sans aucun code écrit sont :

- `test_health` : la route `/health` et le chargement du modèle (`lifespan`) sont déjà fournis ;
- `test_openapi_documents_predict` : la route `/v1/predict` est déclarée, elle apparaît donc dans `/openapi.json` ;
- `test_out_of_range_value_is_rejected` (attend 422) ;
- `test_missing_field_is_rejected` (attend 422).

Les deux derniers passent pour une mauvaise raison. Au départ, `SensorReading` ne déclare **aucun champ** mais il a `extra="forbid"`. Pydantic refuse donc **tout** champ envoyé, quel qu'il soit. Le 422 ne vient ni de la borne sur la température (500 °C), ni du champ `load_pct` manquant : il vient du fait que `temperature_c` est un champ inconnu (`"type": "extra_forbidden"`, `"msg": "Extra inputs are not permitted"`). La preuve : un relevé parfaitement valide est lui aussi refusé avec 422, d'où `assert 422 == 200` dans `test_predict_returns_probability`. Ces deux tests sont « verts » alors que la règle qu'ils doivent vérifier n'existe pas encore.

## 2. Le schéma d'entrée (`maintai/api/schemas.py`)

| Variable | Type | Minimum (`ge`) | Maximum (`le`) | Justification physique (pas le min / max des données) |
|---|---|---|---|---|
| `temperature_c` | `float` | -40.0 | 150.0 | -40 °C correspond à la limite basse de la plage industrielle standard (une machine peut être installée dehors ou dans un local froid). Au-delà de 150 °C, le bobinage d'un moteur dépasse sa classe d'isolation (classe F : 155 °C) : la machine serait détruite, la mesure est donc forcément une erreur de capteur. |
| `vibration_mm_s` | `float` | 0.0 | 50.0 | Une vitesse de vibration RMS est une norme, elle ne peut pas être négative. Selon l'ISO 10816, au-delà d'environ 11 mm/s la machine est déjà en zone « dommages ». 50 mm/s laisse de la marge pour des machines très dégradées, mais une valeur plus haute n'est pas physiquement crédible. |
| `load_pct` | `float` | 0.0 | 100.0 | Un taux de charge est un pourcentage de la charge nominale : 0 % = à vide, 100 % = pleine charge. |
| `runtime_hours` | `float` | 0.0 | 200000.0 | Un compteur d'heures ne peut pas être négatif. 200 000 h ≈ 23 ans de fonctionnement 24 h/24 : c'est au-delà de la durée de vie d'une machine industrielle. |

Les bornes ne sont **pas** le min / max du jeu de données (par exemple la vibration max des données est 22,23 mm/s) : le modèle doit pouvoir recevoir une machine plus abîmée que celles vues à l'entraînement, mais pas une valeur impossible.

Après avoir écrit `schemas.py` seul, que donne `python -m pytest`, et quel est le message d'erreur des tests qui échouent ? Pourquoi est-ce un bon signe ?

```
FAILED tests/test_api.py::test_predict_returns_probability - NotImplementedError: À écrire : TP1, partie 3
FAILED tests/test_api.py::test_high_risk_reading_raises_alert - NotImplementedError: À écrire : TP1, partie 3
FAILED tests/test_api.py::test_503_when_model_missing - NotImplementedError: À écrire : TP1, partie 3
=================== 3 failed, 10 passed, 1 warning in 2.70s ====================
```

Le nombre de tests en échec ne change pas (3), mais **la raison** change : on n'a plus un 422, l'erreur est maintenant `NotImplementedError` levée dans `predict()`. Cela veut dire que la requête valide franchit désormais la validation Pydantic et arrive jusqu'à la fonction. Le schéma fait donc son travail, et les deux tests 422 passent maintenant pour la bonne raison (borne `le=150` et champ obligatoire). Le seul travail restant est clairement localisé : écrire `predict()`.

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

Un 500 (*Internal Server Error*) signifie « bug inattendu dans le code du serveur ». Ici ce n'est pas un bug : le serveur fonctionne, c'est une ressource dont il dépend (le fichier `model.joblib`) qui n'est pas disponible. Le code 503 (*Service Unavailable*) dit exactement cela : « le service est temporairement indisponible, réessayez plus tard ». Le client sait qu'il n'a rien fait de mal et qu'il peut réessayer une fois le modèle entraîné, et la supervision distingue un problème de déploiement d'un vrai plantage. En plus, `/health` renvoie `model_loaded: false`, ce qui rend la panne facile à diagnostiquer.

Pourquoi le modèle est-il chargé **une seule fois**, au démarrage, et pas à chaque requête ?

Charger le modèle, c'est lire un fichier sur le disque et le désérialiser avec `joblib` : c'est lent par rapport à une prédiction (qui ne prend que quelques millisecondes). Le faire à chaque requête ajouterait cette latence à chaque appel et gaspillerait CPU et mémoire. En le chargeant une fois dans `lifespan` et en le gardant dans `state`, toutes les requêtes réutilisent le même objet en mémoire, utilisent toutes la même version du modèle, et un modèle absent est détecté dès le démarrage plutôt qu'au milieu du trafic.

## 4. Les tests

Dernière ligne de `python -m pytest` à la fin du TP :

```
================================================================= 13 passed, 1 warning in 3.12s ==================================================================
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

Dans la section **Schemas** en bas de la page (schéma `SensorReading`), et dans l'onglet **Schema** du *Request body* de `POST /v1/predict` : chaque champ affiche `minimum` et `maximum` (par exemple `temperature_c` : minimum -40, maximum 150) avec sa description. Ces informations sont générées automatiquement à partir des `Field(ge=..., le=...)` et sont aussi visibles dans `/openapi.json`.

## 6. Questions de compréhension

1. REST et FastAPI : quelle est la différence ?
   REST est un **style d'architecture** : des ressources désignées par des URL, manipulées avec les verbes HTTP (GET, POST…), des échanges sans état et des représentations (ici du JSON). FastAPI est un **framework Python** qui permet d'écrire une API HTTP. On utilise FastAPI pour construire une API qui respecte les principes REST, mais l'un est un ensemble de règles, l'autre un outil.
2. Le schéma Pydantic sert à trois choses. Lesquelles ?
   - **Valider** les données reçues (types, champs obligatoires, bornes) et renvoyer automatiquement un 422 sinon ;
   - **Convertir** le JSON en objet Python typé (par exemple `85` devient le `float` `85.0`) ;
   - **Documenter** l'API : le schéma alimente `/openapi.json` et la page `/docs`.
3. Dans `http://localhost:8000/v1/predict`, que désignent `localhost`, `8000` et `/v1/predict` ?
   `localhost` est la machine elle-même (127.0.0.1) : le serveur tourne sur notre propre ordinateur. `8000` est le port sur lequel uvicorn écoute. `/v1/predict` est le chemin de la route (l'endpoint) : version 1 de l'API, ressource « predict ».
4. Pourquoi `/v1` dans l'adresse ?
   Pour versionner l'API. Si un jour on change le contrat (nouvelle variable, format de réponse différent), on publie `/v2/predict` sans casser les clients qui utilisent encore `/v1/predict`.
5. Dans la commande `uvicorn maintai.api.main:app`, que désignent `maintai.api.main` et `app` ?
   `maintai.api.main` est le module Python à importer (le fichier `maintai/api/main.py`, écrit avec des points). `app` est le nom de la variable, dans ce module, qui contient l'objet `FastAPI` (l'application ASGI) qu'uvicorn doit servir.

## 7. Le rendu Git

- Lien de la pull request : [à compléter]
- Le tag `tp1` existe sur GitHub : [oui / non]
- Qui a relu la pull request, et quel commentaire a-t-il laissé ?
  [à compléter : nom du relecteur et son commentaire]

## 8. Ce qui reste flou

- Le fonctionnement exact de `lifespan` avec `@asynccontextmanager` : je comprends que le code avant `yield` s'exécute au démarrage et celui après à l'arrêt, mais pas encore bien le rôle de l'asynchrone ici.
- L'avertissement `StarletteDeprecationWarning` sur `httpx` qui apparaît à chaque exécution des tests : faut-il déjà passer à `httpx2` ?
