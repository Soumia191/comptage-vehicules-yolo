# Compte rendu · TP3 : une interface Streamlit pour MaintAI

**Groupe :** Badre Elmouden, Salma Hmid, Amal Dalouch, Soumia Abbassi  ·  **Membres :** Badre El Mouden, …  ·  **Date :** 11/11/2026

> Remplissez chaque partie dans ce fichier, puis rendez-le avec la pull request du TP3.
> Collez les sorties du terminal entre les lignes ``` (trois accents graves).

---

## 1. Mise en place (dans `TP/TP3`)

Dernière ligne de `python -m pytest` juste après la mise en place :

```
=================== 5 failed, 18 passed, 1 warning in 2.58s ====================
```

Quels fichiers de tests échouent, et quelle erreur donnent-ils ?

Deux fichiers, ceux des nouveautés du TP3 ; les 18 tests hérités du TP2 (`test_api.py`, `test_data.py`, `test_security.py`, `test_train.py`) passent.

- `tests/test_client.py` : les 3 tests échouent avec `NotImplementedError: TP3 sprint 2 : MaintaiClient.predict`. Le corps de `predict` n'est pas encore écrit : il lève volontairement cette exception.
- `tests/test_ui.py` : les 2 tests échouent avec `IndexError: list index out of range`. La page ne contient encore aucun `st.metric`, donc `at.metric[0]` (et `at.metric[-1]` après le clic sur « Prédire ») n'existe pas.

## 2. Lancer Streamlit

Lancez d'abord `streamlit run ui/app.py`. Que s'affiche-t-il dans le navigateur ?

```
ModuleNotFoundError: No module named 'maintai'
Traceback:
File "...\TP\TP3\ui\app.py", line 9, in <module>
    from maintai.client import APIError, MaintaiClient
```

Lancez ensuite `python -m streamlit run ui/app.py`. Pourquoi cette commande fonctionne-t-elle et pas la première ?

Python cherche les modules dans les dossiers de `sys.path`, et cette liste dépend de la façon de lancer.

- Avec `streamlit run ui/app.py`, on lance le programme `streamlit.exe` du `.venv`. Streamlit ajoute à `sys.path` le dossier du script (`ui/`), mais pas le dossier courant `TP/TP3`. Or le package `maintai/` se trouve dans `TP/TP3`, pas dans `ui/` : l'import échoue.
- Avec `python -m streamlit ...`, l'option `-m` ajoute le dossier courant (`TP/TP3`) en tête de `sys.path`. `maintai/` y est, donc `from maintai.client import ...` fonctionne.

Le TP4 règle le problème pour de bon : une fois installé comme package (`pip install -e .`), `maintai` est trouvé quel que soit le dossier ou la commande.

## 3. Sprint 1 : l'onglet « Données »

Votre code de l'onglet « Données » (de `c1, c2, c3 = ...` jusqu'au nuage de points) :

```python
    c1, c2, c3 = st.columns(3)
    c1.metric("Relevés", len(df))
    # failure vaut 0 ou 1 : sa moyenne est la part de pannes
    c2.metric("Taux de panne", f"{df.failure.mean():.1%}")
    c3.metric("Température moyenne", f"{df.temperature_c.mean():.1f} °C")

    # 2. Le choix d'une variable
    variable = st.selectbox("Variable", FEATURES)

    # 3. Le taux de panne par tranche de cette variable
    tranches = pd.cut(df[variable], bins=10)
    taux = df.groupby(tranches, observed=True).failure.mean()
    # Le milieu de chaque tranche, gardé en nombre (en texte, les barres seraient mal triées)
    taux.index = [round(i.mid, 2) for i in taux.index]
    taux.index.name = variable
    st.subheader(f"Taux de panne par tranche de {variable}")
    st.bar_chart(taux, x_label=variable, y_label="taux de panne")

    # 4. Le nuage de points température / vibration, coloré selon la panne
    st.subheader("Température et vibration")
    nuage = df.assign(panne=df.failure.map({0: "non", 1: "oui"}))
    st.scatter_chart(nuage, x="temperature_c", y="vibration_mm_s", color="panne")
```

| Indicateur | Valeur affichée |
|---|---|
| Relevés | 5000 |
| Taux de panne | 13.6% |
| Température moyenne | 59.8 °C |

Dernière ligne de `python -m pytest` à la fin du sprint 1 : `4 failed, 19 passed`.

Choisissez `vibration_mm_s` dans la liste. Comment évolue le taux de panne quand la vibration augmente ?

Le taux de panne augmente très fortement et régulièrement avec la vibration :

| Vibration (milieu de tranche, mm/s) | 1.12 | 3.35 | 5.57 | 7.79 | 10.01 | ≥ 12.24 |
|---|---|---|---|---|---|---|
| Taux de panne | 2 % | 10 % | 32 % | 66 % | 84 % | 100 % |

Au-delà d'environ 12 mm/s, toutes les machines du jeu de données sont tombées en panne. Il n'y a que 8 barres et non 10 : avec `observed=True`, les tranches vides (aucune machine entre ~16 et ~20 mm/s) ne sont pas affichées. Les dernières tranches ne comptent que très peu de relevés (21, 5 et 1) : leur 100 % est moins fiable que les autres valeurs.

**Laquelle des deux variables compte le plus ?** La vibration. Pour la température, le taux passe seulement d'environ 0 % (27 °C) à 37 % (84-91 °C), contre 2 % → 100 % pour la vibration. La corrélation avec `failure` le confirme : 0,53 pour la vibration contre 0,23 pour la température. Le nuage de points le montre aussi : les points « oui » se séparent surtout vers le haut (vibration forte), beaucoup moins vers la droite (température forte).

Ajoutez un `print("relance")` en haut de `ui/app.py`, puis déplacez un curseur. Que voyez-vous dans le terminal, et qu'est-ce que cela montre sur le fonctionnement de Streamlit ?

Résultat de l'expérience (nombre de « relance » dans le terminal) :

| Action | « relance » affichés au total |
|---|---|
| ouverture de la page | 1 |
| changement de variable dans la liste déroulante | 2 (+1) |
| passage à l'onglet « Prédiction » | 2 (+0) |
| 5 déplacements d'un curseur du formulaire | 2 (+0) |
| clic sur « Prédire » | 3 (+1) |

Cela montre la règle de Streamlit : chaque interaction avec un widget relance le script entier, de la première à la dernière ligne, avec la nouvelle valeur (un « relance » par interaction). Il y a deux exceptions :

- les curseurs placés dans `st.form("releve")` ne relancent rien : seul le bouton du formulaire déclenche la relance, avec toutes les valeurs d'un coup ;
- les onglets (`st.tabs`) sont gérés dans le navigateur : en changer ne relance pas le script.

À quoi sert `@st.cache_data` sur la fonction `charger` ?

Comme le script est relancé à chaque interaction, `pd.read_csv` serait exécuté à chaque clic, ce qui rend la page lente. `@st.cache_data` garde en mémoire le résultat de `charger(path)` pour un même argument. Le CSV est lu une seule fois ; aux relances suivantes, Streamlit renvoie une copie du DataFrame déjà chargé, sans relire le fichier. Le cache est invalidé si le chemin ou le code de la fonction change.

## 4. Sprint 2 : l'onglet « Prédiction »

Votre méthode `predict` (`maintai/client.py`) :

```python
    def predict(self, reading: dict) -> dict:
        """Envoie un relevé ; lève APIError si l'API refuse, httpx.HTTPError si elle est injoignable."""
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        # self.http et non httpx.post : les tests fournissent leur propre client
        r = self.http.post("/v1/predict", json=reading, headers=headers)
        if r.status_code != 200:
            try:
                detail = r.json().get("detail")
            except ValueError:  # réponse qui n'est pas du JSON (proxy, page d'erreur...)
                detail = r.text
            raise APIError(r.status_code, detail)
        return r.json()
```

Et le code sous `if envoye:` (`ui/app.py`) :

```python
    if envoye:  # True seulement pendant la relance provoquée par le bouton « Prédire »
        api = MaintaiClient(url, api_key=cle)
        try:
            resultat = api.predict(releve)
        except APIError as e:
            # L'API a répondu, mais a refusé : 401 (clé), 422 (relevé invalide), 503 (modèle absent)
            st.warning(f"L'API a refusé la requête (erreur {e.status_code}) : {e.detail}")
        except httpx.HTTPError:
            # Pas de réponse du tout : uvicorn arrêté, mauvaise URL ou mauvais port
            st.error(f"API injoignable à {url} : vérifiez qu'uvicorn est lancé, l'URL et le port.")
        else:
            pct = round(resultat["failure_probability"] * 100)
            st.metric("Probabilité de panne", f"{pct} %")
            seuil = round(resultat["threshold"] * 100)
            if resultat["alert"]:
                st.error(f"ALERTE : risque de panne au-dessus du seuil de {seuil} %. "
                         "Planifiez une intervention.")
            else:
                st.success(f"Fonctionnement normal : risque sous le seuil de {seuil} %.")
            st.caption(f"Modèle entraîné le {resultat['model_version']}")
```

Dernière ligne de `python -m pytest` à la fin du sprint 2 :

```
======================== 23 passed, 1 warning in 2.01s =========================
```

Avec l'API lancée (`uvicorn maintai.api.main:app --reload --env-file .env`) :

| Essai | Ce qu'affiche Streamlit |
|---|---|
| valeurs par défaut, bonne clé | « Probabilité de panne : **95 %** », puis en rouge « ALERTE : risque de panne au-dessus du seuil de 30 %. Planifiez une intervention. » (uvicorn : `POST /v1/predict 200 OK`) |
| valeurs par défaut, fausse clé | En jaune : « L'API a refusé la requête (erreur 401) : Clé API absente ou invalide » (uvicorn : `401 Unauthorized`). Même message si le champ de la clé est vide. |
| API arrêtée (Ctrl+C dans le terminal d'uvicorn) | En rouge : « API injoignable à http://localhost:8000 : vérifiez qu'uvicorn est lancé, l'URL et le port. » (`httpx.ConnectError`, rien dans le terminal d'uvicorn) |

Le port 8501 n'est pas dans `MAINTAI_CORS_ORIGINS`, et pourtant la prédiction fonctionne. Pourquoi ?

CORS est une règle appliquée **par le navigateur** : il bloque la réponse quand du JavaScript chargé depuis une origine (par exemple `http://localhost:5501`) appelle une autre origine (`http://localhost:8000`) qui ne l'autorise pas.

Avec Streamlit, le navigateur n'appelle jamais l'API. Il ne parle qu'au serveur Streamlit (port 8501, même origine que la page, par WebSocket). C'est le script Python, côté serveur, qui envoie la requête à l'API avec `httpx`. Une requête de serveur à serveur n'a pas d'origine de navigateur : `httpx` ne fait aucune vérification CORS. Ce qui a changé par rapport au TP2, c'est donc **qui** envoie la requête.

## 5. Questions de compréhension

1. Dans la page web du TP2, qui envoie la requête à l'API ? Et dans Streamlit ?
   Dans le TP2, c'est le **navigateur** de l'utilisateur, qui exécute le `fetch` JavaScript de la page : la requête part du poste du client, et CORS s'applique. Dans Streamlit, c'est le **serveur Streamlit** (le processus Python qui exécute `ui/app.py`) qui l'envoie avec `httpx` via `MaintaiClient`. Le navigateur ne fait qu'afficher le résultat que Streamlit lui envoie.
2. Pourquoi les tests de `test_client.py` n'ont-ils pas besoin d'uvicorn ?
   La fixture `client` fournit un `TestClient` de FastAPI, qui est un client `httpx`. On le donne à `MaintaiClient(api_key="test-key", http=client)`. Comme `predict` utilise `self.http.post(...)`, la requête est passée directement à l'application FastAPI, en mémoire, dans le même processus : pas de réseau, pas de port, pas de serveur. Le même code de `predict` est testé, seul le transport change (c'est pour cela qu'il faut `self.http` et non `httpx.post`). La fixture entraîne aussi le modèle dans un dossier temporaire et fixe `MAINTAI_API_KEY=test-key`.
3. Page HTML + JavaScript (TP2) ou Streamlit (TP3) : laquelle choisiriez-vous pour une démonstration à un client ? Et pour l'application d'un technicien de maintenance ? Pourquoi ?
   - **Démonstration à un client : Streamlit.** Elle se construit en une séance, en Python seulement, avec des graphiques (barres, nuage) et des indicateurs prêts à l'emploi. Elle montre à la fois les données et le modèle. On renonce à un design personnalisé, ce qui n'est pas gênant pour une démo.
   - **Application du technicien : une page HTML + JavaScript** (ou un vrai front-end). Elle sert tous les jours, souvent sur tablette ou téléphone dans l'atelier. Il faut une interface sur mesure, rapide et simple (gros boutons, pas de graphiques d'exploration). Une page statique est légère, et elle ne relance pas tout un script Python à chaque clic : Streamlit garde une session Python ouverte par utilisateur sur le serveur, ce qui passe mal à l'échelle. Il faudra en revanche gérer CORS et ne pas exposer la clé dans le navigateur (idéalement : une authentification par utilisateur plutôt qu'une clé partagée).
   - Nuance : pour un outil interne avec peu d'utilisateurs et une équipe qui ne fait que du Python, Streamlit reste défendable, d'autant que la clé peut alors rester côté serveur (dans les secrets de Streamlit) sans que le technicien la saisisse.

## 6. Le rendu Git

- Lien de la pull request : …
- Le tag `tp3` existe sur GitHub : oui / non
- `git status` avant le commit : `.env`, `data/` ou `models/` apparaissaient-ils ? **non** : `.gitignore` exclut `.env`, `data/*.csv` et `models/` (vérifié avec `git check-ignore -v`) ; seul `data/.gitkeep` est suivi.

## 7. Ce qui reste flou

- Comment déployer cette interface pour de vrais techniciens : où mettre la clé API (secrets de Streamlit ou variable d'environnement) pour qu'elle ne soit plus saisie à la main ?
- Combien d'utilisateurs simultanés un serveur Streamlit peut-il supporter, puisque chaque session garde son propre script Python ?
