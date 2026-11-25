# ADR 005 : Régler MaintAI par une recherche par grille plafonnée à 30 runs

Date : 25/11/2026 · Auteurs : Badre El Mouden, [À COMPLÉTER : coéquipiers]

## Contexte
Dans l'ADR 004, le modèle et le seuil ont été choisis en comparant six runs lancés à la main. Cette méthode a deux limites :
- peu de combinaisons ont été essayées ;
- le choix a été fait sur le jeu de test, donc la mesure du réglage retenu sur ce même test est optimiste : on a pu choisir le réglage qui a eu de la chance sur ces 1000 relevés.

Il faut une recherche systématique et reproductible, traçable dans MLflow, sans multiplier les runs : au-delà de quelques dizaines, `mlflow ui` devient illisible, et le temps de calcul compte sur nos PC.

## Options étudiées
1. **Continuer à la main** (`python -m maintai.train --C … --threshold …`). C'est simple, mais l'exploration reste partielle, et le choix se fait sur le test. Rejetée.
2. **`GridSearchCV` de scikit-learn**. C'est l'outil standard, mais il ne crée pas un run MLflow par combinaison et n'optimise qu'un seul score. Surtout, le seuil d'alerte n'est pas un paramètre de l'estimateur : il faudrait ré-entraîner le modèle pour chaque seuil, ou écrire un estimateur sur mesure. Rejetée.
3. **Une recherche par grille maison (`python -m maintai.search`)**. Elle crée un run MLflow par combinaison, mesure par validation croisée sur l'entraînement seulement, choisit avec un critère de coût et refuse toute grille de plus de 30 combinaisons. **Retenue.**
4. **Une recherche aléatoire ou bayésienne (Optuna)**. Elle est utile pour de grands espaces, mais inutile pour trois réglages (C, max_depth, seuil), et le résultat est moins lisible pour l'équipe. Gardée pour plus tard.

## Décision
**Les réglages de MaintAI se choisissent avec `python -m maintai.search`, qui suit ces règles :**

- **Grille : 30 combinaisons, donc 30 runs enfants, qui est le maximum.**
  - régression logistique, C ∈ {0.01, 0.1, 1, 10} ;
  - forêt aléatoire de 200 arbres, max_depth ∈ {None, 8} ;
  - seuils ∈ {0.15, 0.2, 0.25, 0.3, 0.4} pour chaque modèle ;
  - total : (4 + 2) × 5 = 30.
- **Plafond** : `MAX_RUNS = 30` dans le code. Une grille plus grande est refusée avant tout entraînement (`--max-runs` pour le baisser).
- **Mesure** : validation croisée stratifiée à 5 plis sur les 4000 relevés d'entraînement (prédictions hors pli). Le seuil est appliqué aux probabilités hors pli : 6 modèles × 5 plis = 30 entraînements pour 30 runs.
- **Critère** : coût = 4 × pannes manquées + 1 × fausse alerte, ramené à 1000 relevés ; en cas d'égalité, le meilleur recall l'emporte. Le rapport 4 traduit l'arbitrage déjà fait dans l'ADR 004, qui a accepté 2,8 fausses alertes par panne détectée en plus (seuil 0.3 → 0.2) et refusé 4,6 (seuil 0.2 → 0.15). Le rapport implicite est donc compris entre 2,8 et 4,6.
- **Test** : le gagnant est ré-entraîné sur tout l'entraînement, puis mesuré **une seule fois** sur le jeu de test.
- **Traçabilité** : tout est rangé dans l'expérience MLflow `maintai-grid-search`. Les 30 runs enfants sont sous un run parent `grid-search`, qui porte la grille, le meilleur réglage, sa mesure sur le test et son modèle.

**Résultat sur les données v1** (`data_md5 = 73e0b7cb3c6f795c70c3ddf970b99c15`) :

| Rang | Réglage | recall CV | precision CV | pannes manquées / 542 | fausses alertes / 3458 | coût / 1000 |
|---|---|---|---|---|---|---|
| 1 | logreg, C = 10, seuil 0.2 | 0.832 | 0.581 | 91 | 325 | 172.2 |
| 2 | logreg, C = 1, seuil 0.2 | 0.832 | 0.579 | 91 | 328 | 173.0 |
| 3 | logreg, C = 0.1, seuil 0.2 | 0.836 | 0.573 | 89 | 338 | 173.5 |
| 5 | logreg, C = 1, seuil 0.25 | 0.795 | 0.624 | 111 | 260 | 176.0 |
| 9 | logreg, C = 1, seuil 0.15 | 0.865 | 0.519 | 73 | 434 | 181.5 |
| 11 | logreg, C = 1, seuil 0.3 | 0.751 | 0.661 | 135 | 209 | 187.2 |
| 16 | meilleure forêt (max_depth = 8, seuil 0.2) | 0.791 | 0.547 | 113 | 355 | 201.8 |

**Modèle retenu : on garde celui de l'ADR 004, c'est-à-dire la régression logistique avec C = 1.0 et un seuil de 0.2.**

- La grille place C = 10 en tête, mais l'écart avec C = 1.0 n'est que de 3 fausses alertes sur 4000 relevés, ce qui reste dans le bruit.
- Les deux réglages donnent exactement la même mesure sur le jeu de test : recall 0.838, precision 0.543, accuracy 0.882.
- C = 1.0 est la valeur par défaut, déjà en service.
- La recherche par grille confirme donc l'ADR 004 au lieu de la remplacer.

- Run parent : [À COMPLÉTER : identifiant affiché par `python -m maintai.search`]
- Run enfant retenu (logreg, C = 1.0, seuil 0.2) : [À COMPLÉTER : identifiant, colonne `run_id` de `models/grid_search.csv`]
- Données : v1, `data_md5 = 73e0b7cb3c6f795c70c3ddf970b99c15`

## Conséquences
- **Ce que cela facilite** :
  - chaque réglage essayé est traçable dans MLflow ;
  - refaire la recherche sur une nouvelle version des données tient en une commande ;
  - le plafond de 30 runs empêche une grille de grossir sans qu'on le décide.
- **Ce que l'on apprend** : le seuil compte beaucoup plus que C. Pour un même seuil entre 0.15 et 0.3, C entre 0.1 et 10 change le coût de moins de 1 %, alors que passer du seuil 0.3 au seuil 0.2 le baisse de 8 %. La forêt n'apparaît qu'au 16e rang. C = 0.01 régularise trop.
- **Ce que l'on accepte** :
  - une recherche ajoute 31 runs (30 enfants et 1 parent) dans `mlflow.db` ;
  - elle a pris une quinzaine de secondes lors de notre essai, la forêt étant le modèle le plus long à entraîner ;
  - le critère dépend d'une hypothèse de coût. Avec un rapport de 3, le seuil 0.25 gagnerait ; avec un rapport de 6 ou plus, le seuil 0.15 gagnerait. Le rapport 4 est à valider avec l'équipe de maintenance (coût réel d'une panne et d'une inspection), puis à passer avec `--cost-fn`.
- **À surveiller** :
  - relancer la recherche à chaque nouvelle version des données DVC, et comparer les modèles sur un même jeu de test ;
  - si l'on ajoute un modèle à la grille, retirer d'autres valeurs pour rester sous 30 runs, ou passer à une recherche aléatoire (option 4).
