# Compte rendu · TP1 : une API de prédiction pour MaintAI

**Groupe :** G1  ·  **Membres :** Amal Dalouch  ·  **Date :** 28/10/2026

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

Avant d'avoir écrit la moindre ligne, ces quatre tests sont déjà au vert :

1. `test_health`, car la route `/health` et le `lifespan` qui charge le modèle font partie du code de départ ;
2. `test_openapi_documents_predict`, car la route `POST /v1/predict` existe déjà, même vide, et FastAPI l'ajoute à la documentation OpenAPI ;
3. `test_out_of_range_value_is_rejected`, qui attend un 422 ;
4. `test_missing_field_is_rejected`, qui attend aussi un 422.

Pour les deux derniers, le code 422 est le bon, mais pas pour la bonne cause. Le modèle `SensorReading` de départ est vide, et l'option `extra="forbid"` interdit tout champ qui n'est pas déclaré. Comme aucun champ n'est déclaré, n'importe quel JSON contenant `temperature_c` est rejeté avec l'erreur `extra_forbidden` (« Extra inputs are not permitted »). Le test croit vérifier « 500 °C est refusé » ou « un champ manquant est refusé », alors qu'en réalité **tout** est refusé, même un relevé correct. C'est d'ailleurs ce qui fait échouer `test_predict_returns_probability` avec `assert 422 == 200`.

## 2. Le schéma d'entrée (`maintai/api/schemas.py`)

| Variable | Type | Minimum (`ge`) | Maximum (`le`) | Justification physique (pas le min / max des données) |
|---|---|---|---|---|
| `temperature_c` | `float` | -40.0 | 150.0 | -40 °C est le bas de la plage de fonctionnement industrielle classique (atelier non chauffé, extérieur l'hiver). 150 °C est un plafond physique : à cette température l'isolation d'un moteur électrique (classe F, 155 °C) est détruite, donc une mesure plus haute signale un capteur défaillant. |
| `vibration_mm_s` | `float` | 0.0 | 50.0 | La vibration est mesurée en valeur efficace (RMS), qui est toujours positive. La norme ISO 10816 classe une machine comme dangereuse dès une dizaine de mm/s ; 50 mm/s est donc une limite haute très large, au-dessus de laquelle la valeur n'a plus de sens physique. |
| `load_pct` | `float` | 0.0 | 100.0 | C'est un pourcentage de la charge nominale : il va par définition de 0 % (aucune charge) à 100 % (charge maximale prévue par le constructeur). |
| `runtime_hours` | `float` | 0.0 | 200000.0 | Un nombre d'heures cumulées est forcément positif. 200 000 h représentent environ 23 ans de marche continue, plus que la durée de vie normale d'un équipement. |

Nous n'avons volontairement pas repris les extrêmes du fichier `sensors.csv` (par exemple 94,5 °C de température max) : une machine réelle peut sortir de ce qu'on a vu à l'entraînement, et l'API doit l'accepter tant que la valeur reste physiquement possible.

Après avoir écrit `schemas.py` seul, que donne `python -m pytest`, et quel est le message d'erreur des tests qui échouent ? Pourquoi est-ce un bon signe ?

```
FAILED tests/test_api.py::test_predict_returns_probability - NotImplementedError: À écrire : TP1, partie 3
FAILED tests/test_api.py::test_high_risk_reading_raises_alert - NotImplementedError: À écrire : TP1, partie 3
FAILED tests/test_api.py::test_503_when_model_missing - NotImplementedError: À écrire : TP1, partie 3
=================== 3 failed, 10 passed, 1 warning in 2.70s ====================
```

On a toujours 3 échecs, mais le message n'est plus le même : avant c'était un 422, maintenant c'est une `NotImplementedError` qui vient de l'intérieur de `predict()`. Autrement dit, les données valides sont acceptées par le schéma et la requête atteint enfin la fonction. C'est encourageant : le schéma est correct, les deux tests 422 vérifient maintenant une vraie règle, et l'erreur nous indique précisément la prochaine étape (écrire le corps de `predict()`).

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

Le code 500 est réservé aux erreurs imprévues : une exception non gérée, un bug. Un modèle absent est au contraire une situation prévue et gérée : l'application tourne, mais elle ne peut pas encore rendre le service demandé. 503 (*Service Unavailable*) décrit précisément cet état temporaire. Pour le client, cela signifie « ta requête est correcte, reviens plus tard », et pour l'équipe, cela indique qu'il faut entraîner ou déployer le modèle, pas corriger du code. La route `/health` confirme le diagnostic avec `"model_loaded": false`.

Pourquoi le modèle est-il chargé **une seule fois**, au démarrage, et pas à chaque requête ?

Parce que le chargement coûte cher : lecture du fichier `model.joblib` sur le disque puis reconstruction de l'objet scikit-learn. Si on le refaisait à chaque appel, chaque prédiction serait ralentie et le serveur consommerait inutilement des ressources, surtout avec beaucoup de requêtes simultanées. Avec le `lifespan`, le modèle est chargé une fois, gardé en mémoire dans `state`, et partagé par toutes les requêtes. Bonus : si le fichier manque, on le sait dès le lancement.

## 4. Les tests

Dernière ligne de `python -m pytest` à la fin du TP :

```
================================================================= 13 passed, 1 warning in 2.94s ==================================================================
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

En ouvrant `POST /v1/predict`, dans le bloc *Request body*, onglet **Schema** : chaque propriété de `SensorReading` affiche son `minimum` et son `maximum`. On les retrouve aussi tout en bas de la page, dans la partie **Schemas**. FastAPI les génère tout seul à partir des paramètres `ge` et `le` de `Field`.

## 6. Questions de compréhension

1. REST et FastAPI : quelle est la différence ?
   REST n'est pas un logiciel : c'est une manière de concevoir une API web (des ressources accessibles par des URL, des méthodes HTTP qui ont un sens, aucune session gardée côté serveur entre deux requêtes). FastAPI, lui, est une bibliothèque Python concrète qui sert à programmer ce type d'API. REST, c'est le « quoi », FastAPI, le « avec quoi ».
2. Le schéma Pydantic sert à trois choses. Lesquelles ?
   - il **contrôle** les entrées : champs présents, bons types, valeurs dans les bornes, sinon 422 ;
   - il **transforme** le JSON brut en objet Python typé, utilisable directement dans le code ;
   - il **décrit** l'API : c'est lui qui produit le schéma OpenAPI affiché dans `/docs`.
3. Dans `http://localhost:8000/v1/predict`, que désignent `localhost`, `8000` et `/v1/predict` ?
   `localhost` désigne l'ordinateur local (adresse 127.0.0.1) ; `8000` est le numéro de port où le serveur uvicorn attend les requêtes ; `/v1/predict` est le chemin qui identifie la route appelée dans l'API.
4. Pourquoi `/v1` dans l'adresse ?
   C'est le numéro de version de l'API. Il permet de faire évoluer le contrat plus tard (une `/v2`) tout en laissant fonctionner les applications qui dépendent de la version 1.
5. Dans la commande `uvicorn maintai.api.main:app`, que désignent `maintai.api.main` et `app` ?
   La partie avant les deux-points, `maintai.api.main`, est le chemin d'import du module (le fichier `maintai/api/main.py`). La partie après, `app`, est l'objet `FastAPI` défini dans ce fichier, c'est-à-dire l'application que uvicorn va exécuter.

## 7. Le rendu Git

- Lien de la pull request : [à compléter]
- Le tag `tp1` existe sur GitHub : [oui / non]
- Qui a relu la pull request, et quel commentaire a-t-il laissé ?
  [à compléter : nom du relecteur et son commentaire]

## 8. Ce qui reste flou

- Je ne vois pas encore comment choisir le seuil d'alerte (`threshold = 0.3`) : pourquoi 0,3 et pas 0,5, et qui doit décider de cette valeur ?
- La différence entre `model_dump()` et `model_validate()` dans Pydantic n'est pas encore très claire pour moi.
