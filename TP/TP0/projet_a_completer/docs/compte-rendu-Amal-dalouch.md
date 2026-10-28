# Compte rendu individuel — TP0

Nom : Amal Dalouch
Date : 21/10/2026
Projet : Prédiction des pannes et comparaison de modèles

## 1. But du projet

Le travail consiste à compléter une application Python capable
de prédire les pannes de machines à partir d’un fichier CSV.

L’objectif est également de comparer plusieurs méthodes avec
une baseline, puis de justifier le choix du modèle à partir
de résultats mesurés sur un jeu de test.

## 2. Participation personnelle

Dans le cadre du travail collectif et avec une aide guidée,
j’ai participé à l’étude du fonctionnement du projet
et à l’analyse des résultats obtenus.

Je me suis intéressée au rôle de la baseline, à la signification
des métriques et aux limites du modèle retenu.

J’ai également contribué à la réflexion sur la justification
du choix final et à la préparation de la documentation.

## 3. Étapes étudiées

### Préparation des données

Le programme lit le fichier exemple_pannes.csv et vérifie
que la colonne cible failure est présente.

Les observations sans valeur cible sont supprimées.
Les valeurs manquantes des variables d’entrée sont ensuite
traitées dans le prétraitement.

Les variables numériques sont standardisées et les variables
catégorielles sont encodées pour être utilisées par les modèles.

### Séparation des données

Les 5 000 observations disponibles après nettoyage sont réparties
en deux groupes :

- 4 000 pour l’entraînement ;
- 1 000 pour le test.

Le découpage aléatoire est stratifié et utilise la graine 42.
Les modèles sont évalués sur le même jeu de test pour permettre
une comparaison cohérente.

### Entraînement et comparaison

Quatre méthodes sont évaluées : une baseline, une régression
logistique, un arbre de décision et une forêt aléatoire.

Le programme sélectionne le meilleur modèle selon le recall,
puis sauvegarde le modèle et les scores.

## 4. Résultats observés

| Méthode | Accuracy | Recall | Precision | F1 |
|---|---:|---:|---:|---:|
| Baseline | 0,824 | 0,000 | 0,000 | 0,000 |
| Régression logistique | 0,889 | 0,574 | 0,737 | 0,645 |
| Arbre de décision | 0,839 | 0,597 | 0,538 | 0,566 |
| Forêt aléatoire | 0,873 | 0,545 | 0,671 | 0,602 |

L’arbre de décision est sélectionné avec un recall de 59,7 %.
Son avance sur la régression logistique est de 2,3 points
de pourcentage pour cette métrique.

## 5. Analyse du choix

Le recall est prioritaire, car une panne non détectée peut
empêcher une intervention de maintenance nécessaire.

La baseline montre qu’un modèle peut obtenir une accuracy élevée
sans détecter de panne : elle atteint 82,4 % d’accuracy,
mais son recall est nul.

L’arbre de décision détecte davantage de pannes que les autres
candidats sur ce découpage. Cependant, sa précision de 53,8 %
indique qu’une partie importante des alertes est incorrecte.

Le modèle retenu représente donc un compromis : améliorer
la détection des pannes tout en acceptant des inspections
potentiellement inutiles.

## 6. Vérification et livrables

L’exécution des tests fournis donne huit tests réussis.

Les principaux livrables du projet sont :

- le code de chargement, d’entraînement et d’évaluation ;
- le modèle sauvegardé dans models/model.joblib ;
- les résultats dans models/metrics.json ;
- l’étude de conception et la justification du choix du modèle.

## 7. Enseignements personnels

Ce TP m’a permis de mieux comprendre qu’un modèle doit être
évalué selon les conséquences des erreurs qu’il peut produire.

J’ai aussi compris l’intérêt d’une baseline : elle fournit
une référence simple pour juger l’apport des modèles entraînés.

Enfin, les tests automatisés vérifient le fonctionnement attendu
du code, tandis que les métriques servent à examiner la qualité
des prédictions. Ces deux vérifications sont complémentaires.

## 8. Réserves et pistes de progression

Le modèle manque encore environ 40,3 % des pannes réelles
du jeu de test. Les résultats restent donc insuffisants pour
supposer une détection fiable dans toutes les situations.

L’évaluation repose sur un seul découpage. Une validation croisée
permettrait de vérifier si les différences entre les modèles
se maintiennent sur plusieurs répartitions des données.

Après sélection et ajustement, une évaluation finale sur
des données indépendantes serait nécessaire.
