# ADR 004 — Choix du modèle de maintenance prédictive

**Statut :** Décision expérimentale provisoire  
**Date :** 25/11/2026  
**Projet :** MaintAI — TP5

## 1. Contexte

MaintAI vise à prédire les pannes à partir des données des capteurs.

Une panne non détectée peut provoquer un arrêt de production et des coûts importants. Le rappel (*recall*) constitue donc une métrique prioritaire, tout en surveillant la précision afin de limiter les fausses alertes.

Nous avons comparé plusieurs entraînements dans MLflow, puis effectué une recherche exhaustive de 30 configurations d'hyperparamètres sur la version v2 du dataset.

## 2. Méthodologie

La recherche a été implémentée dans `maintai/grid_search.py`, avec une boucle utilisant `itertools.product` et une validation croisée stratifiée à cinq plis.

Les configurations ont été évaluées sur les données d'entraînement. Le jeu de test a été réservé à l'évaluation du modèle finalement sélectionné.

### Hyperparamètres étudiés

| Hyperparamètre | Valeurs |
|---|---|
| `C` | 0.01, 0.1, 1.0 |
| `solver` | lbfgs, liblinear |
| `threshold` | 0.10, 0.15, 0.20, 0.25, 0.30 |
| `class_weight` | balanced |
| `max_iter` | 1000 |

**Nombre total :** 30 configurations, évaluées sur cinq plis, soit 150 ajustements.

Trois hyperparamètres ont varié ; `class_weight` et `max_iter` ont été fixés.

## 3. Critère de sélection

Initialement, une précision minimale de 50 % avait été envisagée. Aucune des 30 configurations ne satisfait cette contrainte.

Une sélection exploratoire a donc été réalisée en maximisant le F2-score calculé à partir des moyennes de précision et de rappel des cinq plis.

Ce critère accorde davantage d'importance au rappel, conformément au besoin métier de détecter les pannes.

## 4. Modèle retenu à titre expérimental

**Configuration n° 25 — Régression logistique**

| Hyperparamètre | Valeur |
|---|---|
| `C` | 1.0 |
| `solver` | lbfgs |
| `threshold` | 0.30 |
| `class_weight` | balanced |
| `max_iter` | 1000 |

### Résultats de validation croisée

| Métrique | Valeur |
|---|---:|
| Recall | 0.9499 |
| Precision | 0.3907 |
| Accuracy | 0.8039 |
| F2 calculé | 0.7385 |

### Résultats sur le jeu de test

| Métrique | Valeur |
|---|---:|
| Recall | 0.961 |
| Precision | 0.388 |
| Accuracy | 0.801 |

## 5. Justification

La configuration n° 25 est retenue comme candidat expérimental parce qu'elle obtient le meilleur F2-score selon notre méthode de sélection.

Son rappel de 96,1 % sur le jeu de test indique qu'elle détecte une proportion importante des pannes réelles.

Cependant, sa précision de 38,8 % signifie qu'une proportion importante des alertes correspond à de fausses alertes.

Elle ne satisfait donc pas l'objectif initial de précision minimale de 50 %.

Le choix reste provisoire : une évaluation tenant compte des coûts métier des fausses alertes et des pannes manquées sera nécessaire avant un éventuel déploiement.

## 6. Traçabilité et reproductibilité

- **Code :** `maintai/grid_search.py`
- **Résultats complets :** `docs/grid_search_results.csv`
- **Run MLflow :** `1343d56e57b648c68f8936a9e054bef3`
- **Dataset :** DVC v2 — 8 000 observations
- **Commit Git v2 :** `f4aaf37`
- **MD5 des données :** `93fdeef2cac1893c4f720c16457eea75`
- **Version de scikit-learn observée :** `1.8.0`

## 7. Limites et perspectives

La recherche a été limitée à 30 configurations, avec une seule pondération des classes et une seule valeur de `max_iter`.

Le F2 utilisé pour la sélection a été calculé à partir des moyennes des métriques des plis, et non comme moyenne directe des F2 individuels.

La recherche devra être élargie, notamment à d'autres pondérations et seuils, afin d'améliorer le compromis entre rappel et précision.

Une validation sur des données indépendantes et représentatives sera nécessaire pour confirmer la généralisation du modèle.

## 8. Décision

**La régression logistique avec `C=1.0`, `solver=lbfgs`, `threshold=0.30`, `class_weight=balanced` et `max_iter=1000` est retenue comme candidat expérimental issu du Grid Search.**

Cette décision ne constitue pas une validation du modèle pour une utilisation en production.