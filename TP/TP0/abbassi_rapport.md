# TP0 : MaintAI — Prédire la fréquentation de la cantine

**Nom :** Soumia Abbassi · **Groupe :** G1 · **Binôme :** Salma Hmid

## Le problème
Le gestionnaire de la cantine doit décider chaque jour du nombre de repas à commander et à préparer. Nous cherchons à prédire ce nombre (`repas_servis`) à partir d'informations connues à l'avance : effectif inscrit, jour, menu, météo, examens, sorties scolaires, veille de vacances et fréquentation de la semaine précédente.

| Élément | Choix |
|---|---|
| Données | `cantine_repas.csv`, 340 jours (09/2024 à 06/2026) |
| Tâche | régression |
| Découpage | chronologique : 272 jours d'entraînement, 68 jours de test (25/02 au 19/06/2026) |
| Métrique principale | MAE (erreur absolue moyenne, en repas) |

Un découpage aléatoire mélangerait des jours futurs dans l'entraînement. Le découpage chronologique reproduit l'usage réel : on apprend sur le passé et on prédit les jours suivants.

## Résultats sur le jeu de test

| Modèle | MAE | RMSE | R² |
|---|---|---|---|
| baseline | 70.271 | 88.416 | -0.151 |
| régression Ridge | 22.623 | 28.545 | 0.880 |
| gradient boosting | 29.852 | 37.727 | 0.790 |
| random forest | 31.825 | 41.647 | 0.745 |

## Mon choix, et pourquoi
Nous choisissons **Ridge**, pour trois raisons :

1. **La meilleure précision.** Avec une MAE de 22,6 repas, l'erreur moyenne est d'environ 4 % de la fréquentation moyenne du test (≈ 525 repas).
2. **Un vrai gain par rapport à la baseline.** On gagne 47,6 repas d'erreur par jour par rapport à la prédiction par la moyenne (70,3).
3. **La simplicité.** C'est un modèle linéaire : il s'entraîne en un instant et ses coefficients sont lisibles. Les modèles d'arbres, plus lourds, sont moins précis ici, probablement à cause du faible volume d'historique.

## Le critère (MAE ≤ 30 repas) est-il atteint ?
**Oui.** Ridge obtient 22,6 repas, soit 7,4 repas sous le seuil. Le gradient boosting y arrive tout juste (29,9). La random forest (31,8) et la baseline (70,3) restent au-dessus.

## Ce qui reste flou
- **Représentativité du test :** 68 jours d'une seule fin d'année scolaire. Les mois d'automne et d'hiver ne sont pas évalués.
- **Variable incomplète :** `repas_semaine_precedente` est vide pour 60 jours (10 dans le test) et remplacée par la médiane.
- **Événements rares :** l'erreur sur les jours d'examen, de sortie ou de veille de vacances n'est pas mesurée à part.
- **Asymétrie des erreurs :** la MAE ne distingue pas un manque de repas d'un surplus, alors que ces deux erreurs n'ont pas le même coût.
- **Pas d'optimisation :** les modèles ont gardé leurs paramètres par défaut.
