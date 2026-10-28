# Compte rendu – Entraînement et comparaison de modèles

**Groupe :*G1* [Elmouden badre] · **Date :** 21/10/2026

## 1. Objectif

Compléter le projet `monprojet` afin d'entraîner plusieurs modèles de classification, de les comparer à une baseline sur un jeu de test, puis de sauvegarder le meilleur selon une métrique choisie dans `config.py`.

## 2. Structure du projet

| Fichier | Rôle |
|---|---|
| `monprojet/config.py` | Paramètres : `TASK`, `TARGET`, `DATA_PATH`, `SPLIT`, `TEST_SIZE`, `SEED`, `MAIN_METRIC`, `MODELS_DIR`… |
| `monprojet/data.py` | `load_data()` et `split()` |
| `monprojet/models.py` | `candidates()` |
| `monprojet/train.py` | `build_preprocessing()` (fournie), `evaluate()` et `main()` |

Lancement depuis la racine du projet :

```bash
python -m monprojet.train
```

## 3. Travail réalisé

### 3.1 `load_data()`
- Lit `config.DATA_PATH` (ou le chemin passé en argument).
- Lève une `ValueError` explicite si la colonne cible `config.TARGET` est absente.
- Supprime les lignes où la cible est manquante.

### 3.2 `split()`
- `X` = colonnes renvoyées par `feature_columns(df)`, `y` = colonne cible.
- Mode `random` : `train_test_split` avec `test_size=config.TEST_SIZE`, `random_state=config.SEED` et stratification sur `y` en classification, pour conserver la même proportion de classes dans le train et le test.
- Mode `chronologique` : tri selon `config.TIME_COLUMN`, les lignes les plus récentes sont gardées pour le test (le modèle apprend le passé et est évalué sur le futur).
- Mode utilisé pour ce rendu : [random / chronologique].

### 3.3 `candidates()`
| Nom | Modèle | Rôle |
|---|---|---|
| `baseline` | `DummyClassifier(strategy="most_frequent")` | Référence : prédit toujours la classe majoritaire |
| `logistic_regression` | `LogisticRegression(max_iter=1000)` | Modèle linéaire, simple et interprétable |
| `random_forest` | `RandomForestClassifier(n_estimators=200)` | Ensemble d'arbres (bagging) |
| `gradient_boosting` | `GradientBoostingClassifier()` | Ensemble d'arbres (boosting) |

Les modèles avec une part d'aléatoire reçoivent `random_state=config.SEED` pour que les résultats soient reproductibles. KNN a été retiré de la liste.

### 3.4 Prétraitement (`build_preprocessing`, fournie)
Chaque modèle est placé dans un `Pipeline` précédé d'un `ColumnTransformer` :
- colonnes numériques : imputation par la médiane puis `StandardScaler` ;
- colonnes catégorielles : imputation par la modalité la plus fréquente puis `OneHotEncoder` ;
- colonne texte éventuelle : `TfidfVectorizer`.

Le prétraitement est appris uniquement sur le train, ce qui évite toute fuite d'information du test vers l'entraînement. Le pipeline complet (prétraitement + modèle) est sauvegardé.

### 3.5 `evaluate()`
- Classification : `accuracy`, `recall`, `precision`, `f1` (moyenne `binary` pour deux classes, `macro` sinon, `zero_division=0`).
- Régression : `mae`, `rmse`, `r2`.
- Valeurs converties en `float` et arrondies à 3 décimales.

### 3.6 `main()`
1. Chargement et découpage des données.
2. Entraînement et évaluation de chaque candidat.
3. Tableau de comparaison trié selon `config.MAIN_METRIC`, dans le sens donné par `HIGHER_IS_BETTER`.
4. Sélection du meilleur modèle.
5. Sauvegarde dans `config.MODELS_DIR` : `model.joblib` (pipeline entraîné) et `metrics.json` (métrique principale, meilleur modèle, scores de tous les modèles).

## 4. Résultats

Découpage : **4 000 lignes d'entraînement**, **1 000 lignes de test**. Métrique principale : **recall**.

| Modèle | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|
| **gradient_boosting** | 0.888 | **0.608** | 0.713 | **0.656** |
| logistic_regression | **0.889** | 0.574 | **0.737** | 0.645 |
| random_forest | 0.881 | 0.557 | 0.705 | 0.622 |
| baseline | 0.824 | 0.000 | 0.000 | 0.000 |

**Meilleur modèle selon le recall : `gradient_boosting` (0.608).**

## 5. Analyse

**Les classes sont déséquilibrées.** La baseline, qui prédit toujours la classe majoritaire, obtient 82,4 % d'accuracy. La classe positive représente donc environ 17,6 % du jeu de test, soit à peu près 176 exemples sur 1 000.

**L'accuracy est trompeuse ici.** La baseline atteint 82,4 % d'accuracy sans détecter un seul cas positif (recall de 0). Un modèle peut donc avoir une bonne accuracy tout en étant inutile, ce qui justifie de choisir une autre métrique principale.

**Pourquoi le recall.** Le recall mesure la part des cas positifs réellement détectés. Le choisir comme métrique principale revient à considérer qu'un cas positif manqué (faux négatif) coûte plus cher qu'une fausse alerte (faux positif).

**Tous les vrais modèles battent la baseline** sur toutes les métriques, ce qui confirme qu'ils apprennent une information utile.

**Gradient boosting.** Avec un recall de 0,608, il détecte environ 107 des 176 cas positifs et en manque environ 69. Sa précision de 0,713 signifie qu'environ 29 % de ses alertes sont des faux positifs (environ 43 sur 150 prédictions positives).

**Écart avec la régression logistique.** La régression logistique est très proche : meilleure accuracy et meilleure précision, recall inférieur de 0,034. Sur environ 176 positifs, cela représente à peine 6 cas détectés en plus pour le gradient boosting. Avec un seul découpage train/test, cet écart reste fragile, et la régression logistique pourrait être préférée si l'interprétabilité compte.

**Random forest** est légèrement en dessous des deux autres sur toutes les métriques.

## 6. Limites et pistes d'amélioration

- **Un seul découpage** : une validation croisée stratifiée donnerait une estimation plus fiable et permettrait de voir si l'écart entre gradient boosting et régression logistique est réel.
- **Seuil de décision** : les modèles utilisent le seuil par défaut de 0,5. Abaisser ce seuil augmenterait le recall, au prix d'une baisse de précision.
- **Déséquilibre des classes** : utiliser `class_weight="balanced"` (régression logistique, random forest) pour pousser les modèles à mieux détecter la classe minoritaire.
- **Hyperparamètres** : les modèles sont utilisés avec des réglages par défaut ; une recherche (`GridSearchCV`, `RandomizedSearchCV`) pourrait améliorer les résultats.

## 7. Conclusion

Le pipeline complet (chargement, découpage, prétraitement, entraînement, évaluation, sélection et sauvegarde) fonctionne. Les trois modèles dépassent nettement la baseline. Selon le recall, le **gradient boosting** est retenu et sauvegardé dans `models/model.joblib`, avec les scores de tous les modèles dans `models/metrics.json`.
