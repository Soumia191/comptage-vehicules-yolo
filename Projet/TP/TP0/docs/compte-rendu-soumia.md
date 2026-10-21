# Compte rendu individuel — TP0

Nom : Soumia Abbassi
Date : 21/10/2026
Projet : Prédiction des pannes de machines

## 1. Objectif du TP

Ce TP consiste à compléter un projet Python pour entraîner
et comparer plusieurs modèles de machine learning.

L’objectif est de prédire la variable failure à partir des
caractéristiques des machines, puis de sélectionner le modèle
qui détecte le mieux les pannes.


## 2. Ma contribution personnelle

Avec une aide guidée, j’ai participé à la complétion des fonctions
load_data() et split() pour charger, nettoyer et découper les données.

J’ai également participé à la mise en place des modèles candidats
et des fonctions evaluate() et main() pour entraîner les modèles,
comparer leurs scores et sauvegarder le meilleur.

J’ai suivi la vérification des huit tests et analysé les résultats
obtenus, notamment le choix de l’arbre de décision selon le recall.

Enfin, j’ai participé à la préparation de l’étude de conception,
du document de choix du modèle et de mon compte rendu individuel.

## 3. Fonctionnement du projet

Le programme suit plusieurs étapes :

1. Charger le fichier CSV et vérifier la présence de la cible.
2. Supprimer les lignes dont la cible est manquante.
3. Séparer les données d’entraînement et de test.
4. Préparer les variables numériques et catégorielles.
5. Entraîner une baseline et trois modèles de classification.
6. Calculer et comparer leurs performances.
7. Sauvegarder le meilleur modèle et ses scores.

## 4. Données utilisées

Le fichier utilisé est data/exemple_pannes.csv.
Il contient 5 000 lignes après nettoyage.

La colonne cible est failure.

Le découpage est aléatoire et stratifié :
- 4 000 lignes pour l’entraînement, soit 80 % ;
- 1 000 lignes pour le test, soit 20 %.

La graine aléatoire est fixée à 42 pour reproduire le découpage.

## 5. Résultats du groupe

Les huit tests automatisés du projet passent.

L’entraînement a produit les résultats suivants :

| Modèle | Recall | Accuracy | Precision | F1 |
|---|---:|---:|---:|---:|
| Baseline | 0,000 | 0,824 | 0,000 | 0,000 |
| Régression logistique | 0,574 | 0,889 | 0,737 | 0,645 |
| Arbre de décision | 0,597 | 0,839 | 0,538 | 0,566 |
| Forêt aléatoire | 0,545 | 0,873 | 0,671 | 0,602 |

Le modèle retenu est l’arbre de décision, car il obtient
le meilleur recall : 59,7 %.

Il dépasse la baseline de 59,7 points de pourcentage en recall.
La baseline ne détecte aucune panne, même si son accuracy
atteint 82,4 %.

## 6. Analyse des résultats

Le recall mesure la proportion de pannes réelles détectées.
Il est prioritaire dans ce projet pour limiter les pannes
qui passent inaperçues.

L’arbre de décision détecte davantage de pannes que les autres
modèles testés. Cependant, sa précision de 53,8 % montre
qu’une partie des alertes est incorrecte.

La régression logistique obtient une meilleure précision
et un meilleur score F1. Le choix du modèle dépend donc
de l’importance accordée aux pannes manquées et aux fausses alertes.

## 7. Difficultés personnelles rencontrées

J’ai eu besoin d’explications pour comprendre le rôle des fonctions
à compléter et l’ordre des étapes du projet.

La distinction entre les données d’entrée X et la cible y,
ainsi que l’interprétation des métriques, ont nécessité
un accompagnement étape par étape.

Les échanges et les exemples m’ont aidée à mieux comprendre
le découpage des données et le choix du modèle selon le recall.

J’ai également appris comment ajouter mon compte rendu
dans le dossier docs du dépôt GitHub depuis mon PC.
## 8. Ce que je retiens

Ce travail illustre l’importance de séparer les données
d’entraînement et de test, de comparer les modèles à une baseline
et de choisir une métrique adaptée au problème.

Une accuracy élevée ne suffit pas à démontrer qu’un modèle
détecte correctement les pannes.

## 9. Limites et améliorations

L’évaluation repose sur un seul découpage des données.
Une validation croisée permettrait de vérifier la stabilité
des résultats.

Le modèle retenu manque encore environ 40,3 % des pannes.
Une évaluation sur des données indépendantes serait nécessaire
avant une utilisation réelle.
