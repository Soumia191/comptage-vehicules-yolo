# Compte rendu · TP1 : une API de prédiction pour MaintAI

**Groupe :** G1  ·  **Membres :** Salma Hmid  ·  **Date :** 28/10/2026

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

Les tests qui réussissent dès le départ sont `test_health`, `test_openapi_documents_predict`, `test_out_of_range_value_is_rejected` et `test_missing_field_is_rejected`. Les deux premiers sont normaux : `/health` est fourni, et la route `/v1/predict` est déjà enregistrée, donc elle figure dans `/openapi.json`.

Les deux tests 422 sont des « faux positifs ». Le schéma `SensorReading` fourni n'a aucun attribut, et `extra="forbid"` demande à Pydantic de rejeter tout attribut non prévu. Résultat : la première clé du JSON (`temperature_c`) est considérée comme inconnue et la requête est refusée avec `extra_forbidden`. Le refus n'a donc rien à voir avec la valeur 500 °C ni avec l'absence de `load_pct`. On le voit bien avec le troisième test en échec : même la requête valide de `test_predict_returns_probability` reçoit un 422 au lieu d'un 200.

## 2. Le schéma d'entrée (`maintai/api/schemas.py`)

| Variable | Type | Minimum (`ge`) | Maximum (`le`) | Justification physique (pas le min / max des données) |
|---|---|---|---|---|
| `temperature_c` | `float` | -40.0 | 150.0 | Borne basse : -40 °C, température minimale de la plage industrielle usuelle pour un équipement installé en extérieur. Borne haute : 150 °C, limite au-delà de laquelle les isolants d'un moteur (classe F, 155 °C) ne tiennent plus ; une lecture supérieure vient d'un capteur en panne, pas de la machine. |
| `vibration_mm_s` | `float` | 0.0 | 50.0 | Une vibration RMS est une amplitude, jamais négative. L'ISO 10816 place la zone de dommages autour de 7 à 11 mm/s selon la taille de la machine ; 50 mm/s est un maximum généreux qui couvre les machines en fin de vie sans accepter des valeurs absurdes. |
| `load_pct` | `float` | 0.0 | 100.0 | Pourcentage de la charge nominale : 0 quand la machine tourne à vide, 100 à pleine charge. On n'accepte pas de surcharge dans cette première version. |
| `runtime_hours` | `float` | 0.0 | 200000.0 | Compteur horaire cumulé, donc positif. 200 000 h (environ 23 ans sans arrêt) est un plafond réaliste pour la durée de vie d'une machine. |

Ces limites décrivent ce qui est **physiquement possible**, pas ce que contient notre jeu de données synthétique (où `runtime_hours` ne dépasse pas 20 000 h, par exemple).

Après avoir écrit `schemas.py` seul, que donne `python -m pytest`, et quel est le message d'erreur des tests qui échouent ? Pourquoi est-ce un bon signe ?

```
FAILED tests/test_api.py::test_predict_returns_probability - NotImplementedError: À écrire : TP1, partie 3
FAILED tests/test_api.py::test_high_risk_reading_raises_alert - NotImplementedError: À écrire : TP1, partie 3
FAILED tests/test_api.py::test_503_when_model_missing - NotImplementedError: À écrire : TP1, partie 3
=================== 3 failed, 10 passed, 1 warning in 2.70s ====================
```

Les trois mêmes tests échouent encore, mais avec `NotImplementedError` au lieu de 422. L'erreur s'est « déplacée » : elle ne vient plus de la validation mais du corps de `predict()`, qui n'est pas encore écrit. C'est bon signe, car cela prouve que le schéma laisse passer les relevés valides et que les tests 422 échouent désormais seulement quand une règle (borne ou champ obligatoire) est vraiment violée. Il ne reste plus qu'une seule chose à coder.

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

Parce que ce n'est pas une erreur de programmation. Un 500 voudrait dire « le serveur a planté de façon inattendue » et pousserait à chercher un bug dans le code. Un 503 dit « le serveur est en vie mais le service n'est pas disponible pour le moment », ce qui est exactement le cas tant que `model.joblib` n'existe pas. Le client comprend qu'il peut réessayer plus tard, et un outil de surveillance ou un équilibreur de charge peut traiter ce cas différemment d'une vraie panne.

Pourquoi le modèle est-il chargé **une seule fois**, au démarrage, et pas à chaque requête ?

Désérialiser le modèle depuis le disque prend bien plus de temps que de calculer une probabilité. Le recharger à chaque requête rendrait l'API lente et gaspillerait de la mémoire et du processeur. Grâce au `lifespan`, le chargement se fait une seule fois quand uvicorn démarre ; toutes les requêtes utilisent ensuite le même objet `Predictor` en mémoire, donc la même version du modèle, et une absence de modèle est repérée immédiatement.

## 4. Les tests

Dernière ligne de `python -m pytest` à la fin du TP :

```
================================================================= 13 passed, 1 warning in 3.68s ==================================================================
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

Elles apparaissent dans la description du schéma `SensorReading` : soit dans l'onglet **Schema** du corps de requête de `POST /v1/predict`, soit dans la rubrique **Schemas** en bas de la page Swagger. Pour chaque champ, on lit `minimum` et `maximum` (par exemple 0 et 100 pour `load_pct`). Ces valeurs viennent directement de `ge` et `le`, via le fichier `/openapi.json`.

## 6. Questions de compréhension

1. REST et FastAPI : quelle est la différence ?
   REST est un ensemble de principes pour concevoir une API sur HTTP (ressources identifiées par des URL, verbes HTTP, communication sans état, données échangées en JSON par exemple). FastAPI est un framework, c'est-à-dire du code Python prêt à l'emploi, qui nous aide à implémenter une API respectant ces principes. On peut faire du REST sans FastAPI (avec Flask, Django…) et on pourrait faire avec FastAPI une API qui ne respecte pas bien REST.
2. Le schéma Pydantic sert à trois choses. Lesquelles ?
   - **validation** : refuser automatiquement (422) une donnée absente, mal typée ou hors bornes ;
   - **conversion** : transformer le JSON reçu en objet Python avec les bons types ;
   - **documentation** : produire automatiquement le contrat OpenAPI visible dans `/docs`.
3. Dans `http://localhost:8000/v1/predict`, que désignent `localhost`, `8000` et `/v1/predict` ?
   `localhost` = l'hôte, ici notre propre machine ; `8000` = le port réseau ouvert par uvicorn ; `/v1/predict` = le chemin de la ressource, qui correspond à la route `predict` du routeur préfixé par `/v1`.
4. Pourquoi `/v1` dans l'adresse ?
   Pour gérer les versions de l'API : quand le contrat d'entrée ou de sortie changera de façon incompatible, on ajoutera `/v2` et les anciens clients continueront d'appeler `/v1` sans être cassés.
5. Dans la commande `uvicorn maintai.api.main:app`, que désignent `maintai.api.main` et `app` ?
   `maintai.api.main` indique à uvicorn quel module importer (`maintai/api/main.py`) ; `app` est la variable de ce module qui contient l'application FastAPI à lancer.

## 7. Le rendu Git

- Lien de la pull request : [à compléter]
- Le tag `tp1` existe sur GitHub : [oui / non]
- Qui a relu la pull request, et quel commentaire a-t-il laissé ?
  [à compléter : nom du relecteur et son commentaire]

## 8. Ce qui reste flou

- Je ne sais pas encore comment l'API devrait se comporter si une machine dépasse légèrement 100 % de charge (surcharge temporaire) : faut-il élargir la borne ou garder le refus ?
- Le rôle de `--reload` : je ne sais pas s'il faut l'enlever quand on déploie l'API pour de vrai.
