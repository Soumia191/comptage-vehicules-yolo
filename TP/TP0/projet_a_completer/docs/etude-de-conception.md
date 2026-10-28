# Étude de conception — Prédiction des pannes de machines

Groupe : [Soumia Abbassi,Salma Hmid,Amal Dalouch,Badr Elmouden] · Version : brouillon · Date : 21/10/2026

## 1. Contexte et utilisateurs

L’utilisateur principal est un responsable de maintenance.
Il souhaite identifier les machines susceptibles de tomber en panneÒ
pour prioriser les vérifications et les interventions.

Le système fournit une prédiction à partir des caractéristiques
disponibles pour chaque machine.

## 2. Problème et décision

- Décision : déterminer les machines à inspecter en priorité.
- Fréquence prévue : à chaque nouvelle collecte de données.
- Une panne non détectée peut entraîner un arrêt de production
  et des coûts de réparation.
- Une fausse alerte peut provoquer une inspection inutile
  et mobiliser l’équipe de maintenance.

La prédiction aide le responsable de maintenance.
Elle ne déclenche pas automatiquement une intervention.

## 3. Approche par sous-problème

| Sous-problème | Approche | Justification |
|---|---|---|
| Vérifier la présence de la cible | Règles Python | Une vérification explicite suffit. |
| Traiter les valeurs manquantes et encoder les variables | Prétraitement scikit-learn | Des méthodes standards suffisent. |
| Prédire la présence d’une panne | ML supervisé : classification | Permet de combiner plusieurs variables ; les résultats sont comparés à une baseline simple. |
| Comparer et sélectionner les modèles | Calcul de métriques et règles | Le choix suit la métrique principale définie. |

Le projet compare une régression logistique, un arbre de décision
et une forêt aléatoire. Le deep learning et les LLM ne sont pas
nécessaires pour cette première version sur données tabulaires.

## 4. Données

| Jeu de données | Source et licence | Volume | Données personnelles ? |
|---|---|---|---|
| data/exemple_pannes.csv | Fichier d’exemple fourni avec le projet ; provenance et licence à vérifier | 5 000 lignes après suppression des cibles manquantes | À vérifier dans les colonnes du fichier |

- Cible : failure.
- Variables d’entrée : toutes les colonnes sauf la cible
  et l’éventuelle colonne de date.
- Les lignes dont la cible est manquante sont supprimées.
- Les valeurs numériques manquantes sont remplacées par la médiane.
- Les valeurs catégorielles manquantes sont remplacées
  par la catégorie la plus fréquente.
- Le prétraitement est ajusté uniquement sur les données d’entraînement.
- Découpage aléatoire stratifié avec une graine de 42 :
  4 000 lignes pour l’entraînement et 1 000 pour le test.

Si les données représentent une succession temporelle de mesures,
la pertinence d’un découpage chronologique devra être examinée.

## 5. Métriques, baseline et critères d’acceptation

### Baseline

La baseline est un DummyClassifier qui prédit toujours
la classe majoritaire.

Résultats mesurés sur le jeu de test :
- Accuracy : 82,4 %.
- Recall : 0 %.
- Precision : 0 %.
- F1 : 0 %.

### Métrique principale

La métrique principale est le recall de la classe panne.

Elle mesure la proportion de pannes réelles détectées.
Elle est privilégiée pour limiter les pannes non détectées.

La precision et le F1 sont également suivis pour apprécier
les fausses alertes et l’équilibre des performances.

### Critère d’acceptation proposé, à valider

Obtenir un recall d’au moins 60 % sur les 1 000 lignes du jeu
de test fixé par la graine 42, et dépasser le recall de la baseline.

Ce seuil est proposé après les premiers essais : il ne constitue
donc pas un critère fixé à l’avance.

Le meilleur recall observé est celui de l’arbre de décision :
59,7 %. Il dépasse la baseline, mais n’atteint pas le seuil proposé
de 60 %.

## 6. Architecture prévue

1. Configuration des chemins, de la cible et de la métrique
   dans monprojet/config.py.
2. Lecture et nettoyage du CSV dans monprojet/data.py.
3. Découpage en données d’entraînement et de test.
4. Prétraitement et entraînement de chaque modèle candidat.
5. Évaluation sur le même jeu de test.
6. Comparaison selon le recall.
7. Sauvegarde du modèle retenu dans models/model.joblib.
8. Sauvegarde des scores dans models/metrics.json.

Le programme s’exécute en ligne de commande avec :
python -m monprojet.train

## 7. Choix technologiques

- Python : programmation du traitement et de l’entraînement.
- pandas : lecture et manipulation des données.
- NumPy : calculs numériques.
- scikit-learn : prétraitement, modèles et métriques.
- joblib : sauvegarde du modèle et de son prétraitement.
- pytest : vérification des fonctions du projet.
- Environnement virtuel venv : isolation des dépendances.

## 8. Risques, éthique, confidentialité

- Les pannes non détectées peuvent retarder une intervention.
- Les fausses alertes peuvent entraîner des inspections inutiles.
- Des données peu représentatives peuvent réduire les performances
  sur d’autres machines.
- Des mesures provenant d’une même machine dans les deux jeux
  peuvent rendre l’évaluation trop optimiste.
- La provenance, la licence et la présence éventuelle de données
  personnelles doivent être vérifiées.
- Une décision opérationnelle doit rester sous contrôle humain.

## 9. Limites et hors périmètre

- Les résultats reposent sur un seul découpage des données.
- Le meilleur modèle manque encore environ 40,3 % des pannes.
- La comparaison ne démontre pas les performances en conditions réelles.
- Une validation sur des données indépendantes reste nécessaire.
- Le projet ne prévoit pas de surveillance en temps réel.
- Il ne diagnostique pas la cause exacte d’une panne.
- Il ne déclenche pas automatiquement une réparation.