# Compte rendu · TP2 : tester, sécuriser et appeler l'API MaintAI

**Groupe :** Badre Elmouden, Salma Hmid, Amal Dalouch, Soumia Abbassi

**Membre :** Elmouden badre

**Date :** 04/11/2026

> Remplissez chaque partie dans ce fichier, puis rendez-le avec la pull request du TP2.
> Collez les sorties du terminal entre les lignes ``` (trois accents graves).

---

## 1. Activité 1 : casser son API (dans `TP/TP1`)

Vos cinq requêtes, le code que vous aviez **prévu**, puis le code **obtenu** dans `/docs` :

| Requête | Ce qui ne va pas | Code prévu | Code obtenu |
|---|---|---|---|
| 1 | champ manquant(load_pct) | 422|422|
| 2 | valeur absurde (temperature_c = 500)|422 | 422|
| 3 | texte au lieu d'un nombre (temperature_c ="abc") |422 |422 |
| 4 | champ en trop (couleur = "rouge") |422 | 422|
| 5 | mauvaise route(/v1/prediction) |404 |404|

Une de vos prévisions était-elle fausse ? Laquelle, et pourquoi ?
Non, toutes nos prévisions étaient correctes. Les quatre premières requêtes ont retourné le code HTTP 422, car les données ne respectaient pas les règles de validation Pydantic. La dernière requête a retourné le code HTTP 404, car la route demandée n'existe pas.
…

Dernière ligne de `python -m pytest` avec `tests/test_casser.py` (dans `TP/TP1`) :

```
5 passed, 1 warning in 0.85s
```

À quoi sert `@pytest.mark.parametrize` ? Combien de tests pytest compte-t-il pour votre fonction, et pourquoi ?
@pytest.mark.parametrize permet d'exécuter une même fonction de test avec plusieurs jeux de données.

Dans notre cas, nous avons défini cinq requêtes invalides. Pytest exécute donc cinq fois la fonction test_requete_refusee, une fois pour chaque cas.

Nous obtenons ainsi cinq tests avec une seule fonction Python.

…

## 2. Ruff

Nous avons utilisé Ruff pour analyser la qualité du code Python dans TP1.

Commande exécutée :

```bash
ruff check .
```

Ruff a initialement signalé des problèmes dans les fichiers `demo_api.py` et `maintai/train.py`, notamment l'organisation des imports (`I001`) et l'utilisation de `datetime.UTC` (`UP017`).

Nous avons corrigé les problèmes avec :

```bash
ruff check demo_api.py maintai/train.py --fix
```

Résultat :

```text
Found 4 errors (4 fixed, 0 remaining).
```

Nous avons ensuite relancé les tests automatisés pour vérifier que les corrections n'avaient pas modifié le fonctionnement de l'application.

```text
18 passed, 1 warning in 0.83s
```

**Quelle différence entre `ruff check .` et `python -m pytest` ?**

`ruff check .` effectue une analyse statique du code Python. Il détecte certains problèmes de qualité, de style et de programmation sans exécuter les tests.

`python -m pytest` exécute les tests automatisés pour vérifier que les fonctions et l'application produisent les résultats attendus.

Les deux outils sont complémentaires : Ruff améliore la qualité du code, tandis que Pytest vérifie son comportement.

## 3. Mise en place du TP2 (dans `TP/TP2`)

Dernière ligne de `python -m pytest` juste après la mise en place :

```text
2 failed, 16 passed, 1 warning in 0.89s
```

Quels sont les deux tests qui échouent, et que reçoivent-ils au lieu de 401 ?

Les deux tests qui échouent sont :

1. `test_missing_key_is_rejected`
2. `test_wrong_key_is_rejected`

Ces deux tests attendent le code HTTP 401 (Unauthorized),
mais reçoivent HTTP 200 (OK).

Cela signifie que l'API accepte les requêtes sans clé API
ou avec une clé incorrecte, car la sécurité n'est pas
encore activée au début du TP2.

Quels sont les deux tests qui échouent, et que reçoivent-ils au lieu de 401 ?

…

## 4. La clé API (sprint 1)

Votre fonction `require_api_key()` (`maintai/api/security.py`) :

```python
def require_api_key(
    x_api_key: str | None = Header(default=None)
) -> None:
    """Vérifie la clé API envoyée par le client."""

    expected_key = get_settings().api_key

    if expected_key is None:
        raise HTTPException(
            status_code=500,
            detail="Clé API non configurée sur le serveur"
        )

    if x_api_key is None or not secrets.compare_digest(
        x_api_key, expected_key
    ):
        raise HTTPException(
            status_code=401,
            detail="Clé API absente ou invalide"
        )
```

Votre ligne `router = ...` dans `maintai/api/main.py` :

```python
router = APIRouter(
    prefix="/v1",
    dependencies=[Depends(require_api_key)]
)
```

Après avoir écrit `security.py` seul (sans toucher au router), combien de tests échouent ? Pourquoi ?
Deux tests continuent d'échouer : test_missing_key_is_rejected et test_wrong_key_is_rejected.

En effet, la fonction require_api_key() est définie, mais elle n'est pas encore appelée par FastAPI.

Pour activer la vérification de la clé API sur toutes les routes /v1, nous avons ajouté Depends(require_api_key) au router.

…

Dernière ligne de `python -m pytest` à la fin du sprint 1 :

```
18 passed, 1 warning in 0.96s
```

Lancez `uvicorn maintai.api.main:app --reload --env-file .env`, puis envoyez ces requêtes (avec `/docs` ou `curl.exe`) :

| Requête | Code obtenu | Réponse (le corps JSON) |
|---|---|---|
| `GET /health`, sans clé | 200|{"status":"ok","model_loaded":true} |
| `POST /v1/predict`, relevé normal, sans clé |401 | {"detail":"Clé API absente ou invalide"}|
| `POST /v1/predict`, relevé normal, fausse clé |401 | {"detail":"Clé API absente ou invalide"}|
| `POST /v1/predict`, relevé normal, bonne clé | 200| Prédiction retournée avec succès (HTTP 200) — valeurs JSON non conservées.|
| `POST /v1/predict`, 500 °C, sans clé | 401| {"detail": "Clé API absente ou invalide"}|
| `POST /v1/predict`, 500 °C, bonne clé |422 | {"detail":[{"type":"less_than_equal","loc":["body","temperature_c"],"msg":"Input should be less than or equal to 150","input":500,"ctx":{"le":150}}]}|

500 °C sans clé et 500 °C avec la clé ne donnent pas le même code. Pourquoi ?

La vérification de la clé API est effectuée avant la validation des données.

Sans clé API, la requête est refusée avec le code HTTP 401 (Unauthorized), même si la température est invalide.

Avec une clé API valide, l'authentification réussit. Cependant, la température de 500 °C dépasse la limite maximale de 150 °C définie dans le schéma Pydantic. L'API retourne donc le code HTTP 422 (Unprocessable Entity).

Cela montre que l'authentification est effectuée avant la validation des données.
…

Relancez uvicorn **sans** `--env-file .env` et envoyez un relevé avec votre clé. Quel code obtenez-vous, et pourquoi ce n'est pas 401 ?

Nous avons relancé Uvicorn sans l'option `--env-file .env`, puis envoyé un relevé avec notre clé API.

**Code obtenu : HTTP 500 — Internal Server Error.**

Réponse JSON :

```json
{
  "detail": "Clé API non configurée sur le serveur"
}
```

Ce résultat s'explique par l'absence de configuration de la clé API sur le serveur. Sans chargement du fichier `.env`, la variable `MAINTAI_API_KEY` n'est pas définie.

La fonction `get_settings()` retourne donc `None` pour la clé attendue, et `require_api_key()` déclenche une erreur HTTP 500.

Il ne s'agit pas d'une erreur 401, car le problème ne vient pas d'une clé incorrecte envoyée par le client, mais d'une mauvaise configuration du serveur.

…

## 5. La page web (sprint 2)

Votre code `fetch` (le contenu du `addEventListener` dans `web/index.html`) :

```javascript
document.getElementById("go").addEventListener("click", async () => {

  const reading = Object.fromEntries(
    FEATURES.map(f => [
      f,
      Number(document.getElementById(f).value)
    ])
  );

  const url = document.getElementById("url").value;
  const cle = document.getElementById("key").value;

  try {
    const resp = await fetch(`${url}/v1/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": cle
      },
      body: JSON.stringify(reading)
    });

    const data = await resp.json();

    if (!resp.ok) {
      show("err", `Erreur ${resp.status}: ${JSON.stringify(data)}`);
      return;
    }

    if (data.alert) {
      show("alert", `Alerte ! Probabilité de panne : ${(data.failure_probability * 100).toFixed(2)}%`);
    } else {
      show("ok", `Pas d'alerte. Probabilité de panne : ${(data.failure_probability * 100).toFixed(2)}%`);
    }

  } catch (e) {
    show("err", `Erreur de connexion : ${e.message}`);
  }

});
```

Ce qu'affiche la page (servie sur le port 5500) :

| Essai | Ce qu'affiche la page |
|---|---|
| relevé normal, bonne clé |Pas d'alerte. Probabilité de panne : 0.27% |
| 500 °C, bonne clé | 500 °C, bonne clé | Erreur 422 : Input should be less than or equal to 150 |
| relevé normal, fausse clé | Erreur 401 : Clé API absente ou invalide|

**Expérience CORS.** Servez la page sur le port 5501. Que dit la page, et que dit la console (F12) ?
Nous avons lancé la page web sur le port 5501 au lieu du port 5500.

La page affiche :

```text
Erreur de connexion : Failed to fetch
```

Dans la console de Google Chrome, nous avons obtenu :

```text
POST http://localhost:8000/v1/predict net::ERR_FAILED
```

Chrome indique également que la requête est bloquée par la politique CORS.

Cela s'explique par le fait que l'origine `http://localhost:5501` ne figure pas dans la liste des origines autorisées par notre API.

En revanche, `http://localhost:5500` est autorisée.

Le navigateur bloque donc la communication depuis le port 5501.

…

Pourquoi `curl.exe` obtient-il une réponse alors que la page du port 5501 est bloquée ?
CORS est un mécanisme de sécurité appliqué par les navigateurs web.

Lorsqu'une page JavaScript effectue une requête vers une autre origine, le navigateur vérifie si cette origine est autorisée par le serveur.

En revanche, `curl` n'applique pas la politique CORS des navigateurs. Il peut donc envoyer une requête HTTP et recevoir la réponse du serveur.

Cependant, l'authentification reste obligatoire : pour accéder à `/v1/predict`, `curl` doit toujours transmettre une clé API valide.
…

## 6. Les tests de l'activité 1 dans le TP2

**Sans `headers=HEADERS`, combien de tests échouent, et avec quel code ? Lequel des cinq cas passe quand même, et pourquoi ?**

Sans l'en-tête `X-API-Key`, quatre tests échouent avec le code HTTP 401 au lieu du code 422 attendu.

Les quatre cas concernés sont :

- Champ manquant.
- Valeur absurde.
- Texte au lieu d'un nombre.
- Champ supplémentaire.

Cela s'explique par le fait que l'authentification est vérifiée avant la validation des données.

Le cinquième test, « mauvaise route », réussit avec le code HTTP 404, car la route `/v1/prediction` n'existe pas.

**Dernière ligne de `python -m pytest` avec `headers=HEADERS` :**

```text
23 passed, 1 warning in 0.83s
```

Après l'ajout de `headers=HEADERS`, les requêtes utilisent une clé API valide et passent l'étape d'authentification.

Les 23 tests réussissent, ce qui confirme le fonctionnement attendu des tests de validation, d'entraînement et de sécurité.

## 7. Questions de compréhension

### 1. Quelle différence entre 401 et 422 ? Et entre 401 et 403 ?

Le code **401 Unauthorized** indique que la requête ne possède pas d'informations d'authentification valides, par exemple lorsqu'une clé API est absente ou incorrecte.

Le code **422 Unprocessable Entity** indique que les données envoyées ne respectent pas les règles de validation définies par le schéma Pydantic.

Le code **403 Forbidden** indique que le serveur refuse l'accès à une ressource, généralement parce que le client ne possède pas les permissions nécessaires.

Dans notre projet MaintAI :

- Sans clé API → HTTP 401.
- Température de 500 °C avec une bonne clé → HTTP 422.
- HTTP 403 concerne l'autorisation, qui n'est pas implémentée dans ce TP.

### 2. Pourquoi la clé n'est-elle ni dans le code ni dans Git ? Que contiennent `.env` et `.env.example`, et lequel est commité ?

La clé API est une information sensible qui doit rester confidentielle.

Si elle est enregistrée directement dans le code et envoyée sur GitHub, une personne non autorisée pourrait la récupérer et accéder à l'API.

Le fichier `.env` contient les variables d'environnement réelles, notamment la clé API utilisée par le serveur.

Le fichier `.env.example` contient des exemples de configuration avec des valeurs fictives.

Le fichier `.env` est exclu de Git grâce à `.gitignore`, tandis que `.env.example` peut être commité pour documenter les variables nécessaires.

### 3. Pourquoi la clé voyage-t-elle dans un en-tête et pas dans l'URL ?

Une URL peut être enregistrée dans l'historique du navigateur, les journaux du serveur ou certains outils de surveillance.

Placer une clé API dans l'URL augmente donc les risques de divulgation.

Dans MaintAI, nous utilisons l'en-tête HTTP `X-API-Key` pour transmettre la clé.

Cette méthode est préférable pour l'authentification. En production, HTTPS doit également être utilisé pour protéger les données transmises.

### 4. Que fait `secrets.compare_digest` de plus qu'un simple `==` ?

La fonction `secrets.compare_digest()` permet de comparer deux valeurs sensibles en limitant les différences de temps d'exécution liées à leur contenu.

Une comparaison classique avec `==` peut présenter des variations de temps exploitables dans certaines conditions.

L'utilisation de `secrets.compare_digest()` réduit les risques d'attaques par analyse du temps de réponse, appelées *timing attacks*.

Nous l'utilisons dans MaintAI pour comparer la clé API reçue avec la clé configurée sur le serveur.

### 5. Qu'est-ce qu'une origine ? `http://localhost:5500` et `http://localhost:8000` sont-elles la même origine ?

Une origine web est définie par trois éléments :

- Le protocole, par exemple `http`.
- Le nom de domaine, par exemple `localhost`.
- Le numéro de port, par exemple `5500`.

Les adresses `http://localhost:5500` et `http://localhost:8000` ne correspondent pas à la même origine, car leurs ports sont différents.

Le mécanisme CORS permet au serveur de définir les origines autorisées à accéder à ses réponses depuis un navigateur.

Dans notre expérience, la page sur le port 5500 pouvait communiquer avec l'API, tandis que celle sur le port 5501 était bloquée par le navigateur.