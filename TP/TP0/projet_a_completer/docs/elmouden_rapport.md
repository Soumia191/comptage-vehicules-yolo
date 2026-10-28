# TP0 : CantineAI — Prévision des repas de la cantine

**Nom :** Badre El Mouden · **Groupe :** G1 ·

## Le problème
Prédire le nombre de repas servis chaque jour à la cantine à partir du calendrier (jour, examens, sorties, veille de vacances), du menu, de la météo, du nombre d'inscrits et des repas servis la semaine précédente. Si on sous-estime, des élèves n'ont pas à manger ; si on surestime, la nourriture est gaspillée.

- **Tâche :** régression, cible `repas_servis`
- **Données :** `data/cantine_repas.csv`, 340 jours (du 09/09/2024 au 19/06/2026)
- **Découpage chronologique :** entraînement sur 272 jours (jusqu'au 24/02/2026), test sur les 68 jours les plus récents (du 25/02 au 19/06/2026)
- **Métrique principale :** MAE, c'est-à-dire l'erreur moyenne en nombre de repas. Elle se lit directement en repas, ce qui la rend simple à expliquer au gestionnaire de la cantine.

## Résultats sur le jeu de test

| Modèle | MAE (repas) | RMSE | R² |
|---|---|---|---|
| baseline (moyenne) | 70.271 | 88.416 | -0.151 |
| **régression Ridge** | **22.623** | **28.545** | **0.880** |
| gradient boosting | 29.852 | 37.727 | 0.790 |
| random forest | 31.825 | 41.647 | 0.745 |

## Mon choix, et pourquoi
Je retiens la **régression Ridge** :
- C'est le meilleur modèle selon la métrique principale : il se trompe en moyenne de **23 repas par jour**, soit environ 4 % des 525 repas servis en moyenne sur la période de test.
- Il bat nettement la baseline : la MAE passe de 70,3 à 22,6 repas, une baisse de **68 %**.
- Il est aussi le meilleur sur la RMSE et le R² (0,88). Il ne fait donc pas de grosses erreurs ponctuelles qui seraient cachées par la moyenne.
- C'est le modèle le plus simple, le plus rapide et le plus facile à expliquer : chaque coefficient montre l'effet d'une variable. Les modèles d'arbres (random forest, gradient boosting) font moins bien, sans doute parce que 272 jours d'entraînement sont peu pour eux et que les relations dans ces données sont surtout linéaires (inscrits, repas de la semaine précédente).

## Le critère (MAE ≤ 30 repas) est-il atteint ?
**Oui.** La MAE de Ridge vaut 22,6 repas, sous le seuil de 30 repas (environ 6 % des repas servis). La baseline (70,3) ne l'atteint pas. Le gradient boosting (29,9) passe juste sous le seuil, avec très peu de marge.

## Ce qui reste flou
- **Petit jeu de test :** 68 jours, sur une seule période (février à juin). On ne sait pas comment le modèle se comporte en septembre ou en décembre.
- **Valeurs manquantes :** `repas_semaine_precedente` manque pour 60 jours, dont 10 dans le test. Elles sont remplacées par la médiane, ce qui peut dégrader certaines prédictions.
- **Jours rares :** il y a peu d'exemples d'examens, de sorties scolaires et de veilles de vacances. L'erreur sur ces jours-là n'est pas mesurée à part, alors que ce sont probablement les plus difficiles.
- **Sens de l'erreur :** la MAE traite de la même façon un repas manquant et un repas en trop. Or un élève sans repas coûte sans doute plus cher qu'un repas gaspillé. Il faudrait regarder la part des jours sous-estimés.
- **Réglages :** les hyperparamètres n'ont pas été optimisés (par exemple l'alpha de Ridge ou la profondeur des arbres). Le classement pourrait changer après une recherche par validation croisée temporelle.
