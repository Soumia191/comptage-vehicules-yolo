# TP0 : CantineAI — Estimation du nombre de repas à la cantine

**Nom :** Amal Dalouch · **Groupe :** G1 ·

## Le problème
Chaque jour, la cantine doit savoir combien de repas préparer. L'objectif est d'estimer `repas_servis` à partir des inscrits, du jour de la semaine, du menu, de la température et de la pluie, des événements (examens, sortie scolaire, veille de vacances) et des repas servis la semaine précédente. Une estimation trop basse laisse des élèves sans repas ; une estimation trop haute crée du gaspillage.

- Type de problème : **régression**
- Jeu de données : `cantine_repas.csv`, 340 jours de service entre septembre 2024 et juin 2026
- Découpage **chronologique** : les 272 premiers jours servent à l'entraînement, les 68 derniers (25/02 au 19/06/2026) servent au test. Le modèle est donc évalué sur des jours qu'il n'a jamais vus, comme en conditions réelles.
- Métrique principale : **MAE**. Elle donne l'écart moyen en repas, une unité que le responsable de la cantine comprend directement.

## Résultats sur le jeu de test

| Modèle | MAE | RMSE | R² |
|---|---|---|---|
| baseline | 70.271 | 88.416 | -0.151 |
| Ridge | **22.623** | **28.545** | **0.880** |
| gradient boosting | 29.852 | 37.727 | 0.790 |
| random forest | 31.825 | 41.647 | 0.745 |

La baseline prédit simplement la moyenne. Son R² négatif montre qu'elle fait moins bien qu'une constante adaptée à la période de test.

## Mon choix, et pourquoi
Le modèle retenu est la **régression Ridge**.

Sur la période de test, il se trompe en moyenne d'environ 23 repas par jour, pour environ 525 repas servis en moyenne : c'est un écart d'à peu près 4 %. Par rapport à la baseline, l'erreur est divisée par trois (de 70 à 23 repas). Ridge arrive aussi premier sur la RMSE et le R², donc l'avantage ne se limite pas à une seule métrique.

Les modèles d'ensemble (gradient boosting, random forest) sont plus complexes mais moins précis ici. Avec seulement 272 jours d'historique et des variables liées de façon assez linéaire à la cible, un modèle linéaire régularisé suffit. Il est en plus facile à interpréter et à maintenir.

## Le critère (MAE ≤ 30 repas) est-il atteint ?
**Oui**, avec de la marge : 22,6 repas pour un seuil de 30. Parmi les autres modèles, seul le gradient boosting passe sous le seuil, de très peu (29,9). La random forest et la baseline ne l'atteignent pas.

## Ce qui reste flou
- L'évaluation repose sur 68 jours seulement, tous entre février et juin. Rien ne garantit les mêmes performances à la rentrée ou en hiver.
- 60 jours n'ont pas de valeur pour `repas_semaine_precedente` (10 dans le test). L'imputation par la médiane est une approximation.
- Les situations exceptionnelles (examens, sorties, veilles de vacances) sont rares dans les données, et on ne connaît pas l'erreur du modèle sur ces jours-là précisément.
- La MAE pénalise de la même façon un repas en trop et un repas manquant, alors qu'en pratique manquer de repas est plus grave. Il serait utile de mesurer la part des jours sous-estimés.
- Aucun réglage d'hyperparamètres n'a été fait. Une validation croisée temporelle pourrait améliorer les modèles d'arbres et changer le classement.
