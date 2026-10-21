# Compte rendu individuel — TP0

Nom : Salma Hmid
Date : 21/10/2026
Sujet : Comparaison de modèles pour la détection des pannes

## 1. Présentation du travail

Ce TP porte sur la construction d’une chaîne de machine learning,
depuis la lecture des données jusqu’à la sauvegarde d’un modèle.

Le problème traité est une classification : prédire si une machine
présente une panne à partir des informations disponibles.
La variable cible est failure.

## 2. Contribution et démarche personnelle

Avec l’accompagnement du groupe et une aide guidée, j’ai participé
à la compréhension du programme, au suivi de son exécution
et à l’interprétation des résultats.

J’ai étudié le rôle des principales étapes : nettoyage des données,
séparation entraînement/test, préparation des variables,
entraînement et évaluation.

J’ai également participé à la discussion sur le choix du modèle,
en comparant la détection des pannes et la fiabilité des alertes.

## 3. Organisation de l’expérience

Le fichier exemple_pannes.csv contient 5 000 observations après
suppression des lignes dont la cible est manquante.

Les données sont réparties en deux ensembles :
- 4 000 observations pour entraîner les modèles ;
- 1 000 observations pour mesurer leurs performances.

Le découpage est stratifié afin de conserver des proportions
proches de chaque classe dans les deux ensembles.
La graine 42 permet de reproduire ce découpage.

Le prétraitement et le modèle sont réunis dans une Pipeline.
Le prétraitement est ajusté sur les données d’entraînement,
puis appliqué aux données de test.

## 4. Comparaison des modèles

Une baseline est utilisée comme point de référence.
Elle prédit systématiquement la classe majoritaire.

Trois modèles sont ensuite comparés à cette baseline :
la régression logistique, l’arbre de décision et la forêt aléatoire.

| Modèle | Recall | Precision | F1 | Accuracy |
|---|---:|---:|---:|---:|
| Baseline | 0,000 | 0,000 | 0,000 | 0,824 |
| Régression logistique | 0,574 | 0,737 | 0,645 | 0,889 |
| Arbre de décision | 0,597 | 0,538 | 0,566 | 0,839 |
| Forêt aléatoire | 0,545 | 0,671 | 0,602 | 0,873 |

## 5. Interprétation

La baseline obtient une accuracy de 82,4 %, mais son recall est nul.
Elle ne détecte donc aucune panne. Ce résultat montre pourquoi
l’accuracy seule ne convient pas pour sélectionner le modèle.

L’arbre de décision obtient le meilleur recall, avec 59,7 %.
Il est retenu conformément à la métrique principale configurée.

La régression logistique présente toutefois une précision supérieure :
73,7 %, contre 53,8 % pour l’arbre. Ses alertes sont donc plus souvent
correctes sur ce jeu de test, mais elle détecte une proportion
légèrement plus faible des pannes réelles.

Le choix de l’arbre privilégie la détection des pannes,
en acceptant davantage de fausses alertes.

## 6. Validation du programme

Les huit tests automatisés fournis passent.
Ils vérifient les comportements attendus pour le chargement,
le découpage, les modèles et l’évaluation.

L’entraînement produit :
- model.joblib, qui contient le modèle retenu et son prétraitement ;
- metrics.json, qui contient les scores des modèles.

La réussite des tests confirme le respect des attentes testées.
Elle ne garantit pas les performances sur de nouvelles données.

## 7. Difficultés et apprentissages

La principale difficulté de compréhension était de distinguer
le recall de la precision.

Le recall répond à la question : quelle proportion des pannes
réelles est détectée ?

La precision répond à la question : parmi les alertes émises,
quelle proportion correspond réellement à une panne ?

La comparaison des résultats m’a permis de mieux comprendre
ce compromis et l’importance de choisir une métrique adaptée
à l’objectif du projet.

## 8. Améliorations envisagées

Une validation croisée permettrait d’examiner la stabilité
des performances. Il serait également utile d’ajuster
les paramètres des modèles sur les données d’entraînement.

Enfin, une évaluation sur des données indépendantes serait
nécessaire avant toute utilisation opérationnelle.
La décision d’intervention doit rester sous contrôle humain.
