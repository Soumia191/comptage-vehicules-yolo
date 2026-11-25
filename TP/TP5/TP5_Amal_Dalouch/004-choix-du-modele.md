\# ADR 004 — Choix du modèle de prédiction



\*\*Auteur :\*\* Amal Dalouch  

\*\*Groupe :\*\* G1  

\*\*Date :\*\* 25/11/2026  

\*\*Statut :\*\* Accepté



\## Contexte



MaintAI est une application de maintenance prédictive permettant de détecter les risques de panne des machines. Une panne non détectée coûte plus cher qu'une fausse alerte. Le Recall constitue donc la métrique prioritaire.



\## Options étudiées



| Modèle | Recall | Precision | Accuracy |

|---|---|---|---|

| Régression logistique par défaut | 0.750 | 0.622 | 0.904 |

| Régression logistique C = 0.01 | 0.691 | 0.653 | 0.908 |

| Régression logistique seuil = 0.2 | 0.838 | 0.543 | 0.882 |

| Random Forest | 0.728 | 0.562 | 0.886 |



\## Décision



Nous retenons la régression logistique avec C = 1.0 et un seuil de décision de 0.2, car elle présente le meilleur Recall parmi les quatre expérimentations.



\- Modèle : Logistic Regression

\- Seuil : 0.2

\- Recall : 0.838

\- Precision : 0.543

\- Accuracy : 0.882

\- Nom du run : shivering-cod-873

\- Run ID : à renseigner depuis MLflow

\- Version des données : V1

\- Data MD5 : à renseigner depuis MLflow



\## Conséquences



Le modèle sélectionné permet de détecter 83.8 % des pannes réelles, réduisant ainsi le risque de pannes non détectées.



En contrepartie, la diminution du seuil augmente le nombre de fausses alertes.



MLflow permet de conserver les paramètres, les métriques et le modèle entraîné, tandis que DVC assure le versionnement des données utilisées.

