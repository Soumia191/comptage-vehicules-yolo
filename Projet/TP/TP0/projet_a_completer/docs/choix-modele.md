# Choix du modèle

Date : 21/10/2026 · Groupe : [Soumia Abbassi,Salma Hmid]

## Contexte

Le projet vise à prédire les pannes à partir des caractéristiques
présentes dans le fichier exemple_pannes.csv. La colonne à prédire
est failure.

La métrique principale est le recall (rappel), car nous souhaitons
détecter un maximum de pannes réelles et limiter les pannes non détectées.

## Données et découpage

- Fichier : data/exemple_pannes.csv.
- Nombre de lignes après suppression des cibles manquantes : [à compléter].
- Variables utilisées : toutes les colonnes sauf failure et l’éventuelle
  colonne de date.
- Les lignes dont la cible est manquante sont supprimées.
- Découpage aléatoire stratifié : 80 % pour l’entraînement et 20 % pour le test.
- Graine aléatoire : 42, pour reproduire le découpage.
- Les valeurs numériques manquantes sont remplacées par la médiane,
  puis les variables numériques sont standardisées.
- Les valeurs catégorielles manquantes sont remplacées par la valeur
  la plus fréquente, puis les catégories sont encodées.

## Résultats sur le jeu de test

| Modèle | Recall (métrique principale) | Accuracy | Precision | F1 |
|---|---:|---:|---:|---:|
| Baseline | 0,000 | 0,824 | 0,000 | 0,000 |
| Régression logistique | 0,574 | 0,889 | 0,737 | 0,645 |
| Arbre de décision | 0,597 | 0,839 | 0,538 | 0,566 |
| Forêt aléatoire | 0,545 | 0,873 | 0,671 | 0,602 |

## Décision

Le modèle retenu est l’arbre de décision (decision_tree), car il obtient
le meilleur recall : 0,597. Il détecte donc environ 59,7 % des pannes
réelles sur le jeu de test.

Il dépasse la baseline de 0,597 en recall, soit 59,7 points de pourcentage.
La baseline ne détecte aucune panne malgré une accuracy de 82,4 %.

L’arbre de décision dépasse la régression logistique de 2,3 points
de recall. En revanche, sa précision est plus faible : 53,8 % contre
73,7 %. Le choix privilégie donc la détection des pannes, au prix
de davantage de fausses alertes.

Critère d’acceptation défini dans l’étude de conception :
[indiquer le seuil de recall attendu].

Le critère est atteint si le seuil attendu est inférieur ou égal
à 59,7 %. Sans ce seuil, nous ne pouvons pas conclure.

Le modèle retenu est sauvegardé dans models/model.joblib.
Les scores sont sauvegardés dans models/metrics.json.

## Limites

- L’évaluation repose sur un seul découpage entraînement/test :
  la stabilité des résultats n’a pas été vérifiée par validation croisée.
- L’arbre de décision manque encore environ 40,3 % des pannes réelles.
- Sa précision de 53,8 % signifie qu’une partie importante des alertes
  ne correspond pas à une panne réelle.
- La taille exacte du jeu de test et le nombre de pannes qu’il contient
  restent à préciser.
- Les performances sur de nouvelles machines ou de futures données
  ne sont pas garanties par ces seuls résultats.
- Le jeu de test ayant servi à comparer et sélectionner les modèles,
  une évaluation finale sur des données indépendantes serait utile.