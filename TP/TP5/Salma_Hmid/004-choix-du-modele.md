# ADR 004 — Choix du modèle

**Date :** 25/11/2026

**Statut :** Accepté

## Contexte

Plusieurs modèles et paramètres ont été comparés à l'aide de MLflow. Dans le contexte de MaintAI, une panne non détectée coûte plus cher qu'une fausse alerte. Le recall est donc la métrique prioritaire.

## Décision

Nous retenons la régression logistique avec un seuil de décision de 0.2, entraînée sur le dataset V1.

- Recall : 0.838
- Precision : 0.543
- Accuracy : 0.882

Ce modèle présente le meilleur recall parmi les essais réalisés.

## Conséquences

Nous acceptons une précision plus faible et davantage de fausses alertes afin de détecter un maximum de pannes réelles.

## Traçabilité

- Modèle : régression logistique (logreg)
- Seuil : 0.2
- Dataset : V1 (5000 lignes)
- Commit Git : `76c495a`
- Run MLflow : `5bbad30fca884e448bf8f82a98b60b0c`
- data_md5 : `b7fec4274774842dc1f48202c6013490`
- scikit-learn : 1.8.0