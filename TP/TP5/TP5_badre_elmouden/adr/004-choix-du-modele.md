# ADR 004 : Choix du modèle et du seuil d'alerte de MaintAI

Date : 25/11/2026 · Auteurs : Badre El Mouden, [À COMPLÉTER : coéquipiers]

## Contexte
MaintAI doit signaler à l'avance les machines qui risquent de tomber en panne. Une panne manquée (faux négatif) coûte bien plus cher qu'une fausse alerte (une inspection inutile). Le critère principal est donc le **recall**. La **precision** passe en second : elle sert à limiter les fausses alertes, pour que les techniciens continuent de croire aux alertes.

Les six candidats ont été entraînés le 25/11/2026 et comparés dans MLflow (expérience `maintai`) sur **la même version des données** : v1, 5000 relevés, `data_md5 = 73e0b7cb3c6f795c70c3ddf970b99c15`. Le découpage est le même pour tous (seed 42, 4000 relevés d'entraînement et 1000 de test, dont 136 pannes).

## Options étudiées

| Run (commande `python -m maintai.train …`) | run_id (8 car.) | recall | precision | accuracy | pannes manquées / 136 | fausses alertes / 864 |
|---|---|---|---|---|---|---|
| logreg, C = 1.0, seuil 0.3 (défaut) | `97e9964f` | 0.750 | 0.622 | 0.904 | 34 | 62 |
| logreg, `--C 0.01`, seuil 0.3 | `df2288ab` | 0.691 | 0.653 | 0.908 | 42 | 50 |
| **logreg, C = 1.0, `--threshold 0.2`** | **`0aad8b4c`** | **0.838** | **0.543** | **0.882** | **22** | **96** |
| logreg, C = 1.0, `--threshold 0.15` | `606527da` | 0.875 | 0.500 | 0.864 | 17 | 119 |
| `--model forest`, seuil 0.3 | `041bf0a0` | 0.728 | 0.562 | 0.886 | 37 | 77 |
| `--model forest --threshold 0.2` | `37f3b34a` | 0.831 | 0.479 | 0.854 | 23 | 123 |
| baseline (toujours « pas de panne ») | – | 0.000 | – | 0.864 | 136 | 0 |

1. **Régression logistique, seuil 0.3 (réglage du TP4)** : le plus équilibré, avec la meilleure accuracy, mais il rate une panne sur quatre (34 sur 136).
2. **Régression logistique plus régularisée (C = 0.01)** : la precision monte un peu, mais le recall baisse (0.691). Ce run va contre la contrainte principale. Rejeté.
3. **Régression logistique, seuil 0.2** : 12 pannes de plus détectées par rapport au seuil 0.3, pour 34 fausses alertes de plus.
4. **Régression logistique, seuil 0.15** : le meilleur recall, mais une alerte sur deux est fausse, et l'accuracy (0.864) tombe au niveau de la baseline. Par rapport au seuil 0.2, on gagne 5 pannes pour 23 fausses alertes de plus : chaque nouvelle détection coûte de plus en plus cher.
5. **Forêt aléatoire (200 arbres)** : au même seuil, elle fait moins bien que la régression logistique sur le recall **et** sur la precision (0.728 / 0.562 contre 0.750 / 0.622 au seuil 0.3 ; 0.831 / 0.479 contre 0.838 / 0.543 au seuil 0.2). En plus, elle est plus lourde et moins lisible. Rejetée : un modèle plus complexe n'est pas meilleur ici.

## Décision
**Régression logistique (StandardScaler + LogisticRegression, C = 1.0) avec un seuil d'alerte de 0.2.**

- Run MLflow retenu : `0aad8b4c63c240fba816c0632d1e484b` (expérience `maintai`, paramètres `model = logreg`, `C = 1.0`, `threshold = 0.2`, `n_train = 4000`)
- Données utilisées : `data_md5 = 73e0b7cb3c6f795c70c3ddf970b99c15` (v1, 5000 lignes, 126815 octets). Sous DVC, c'est la version du commit Git « Données v1 », que l'on retrouve avec `git log --oneline -- data/sensors.csv.dvc`
- Pour reproduire : `git checkout <commit v1> -- data/sensors.csv.dvc`, puis `dvc checkout`, puis `python -m maintai.train --threshold 0.2`
- Environnement : scikit-learn 1.8.0, numpy 2.4.4, pandas 3.0.2, d'après le `requirements.txt` enregistré avec le modèle du run

Raison principale : une panne manquée coûte plus cher qu'une fausse alerte. Le seuil 0.2 fait passer les pannes manquées de 34 à 22 sur 136, avec environ 3 inspections inutiles de plus par panne en plus détectée. C'est le meilleur compromis entre le seuil 0.3, qui rate trop de pannes, et le seuil 0.15, où une alerte sur deux est fausse.

## Conséquences
- **Ce que l'on accepte en échange** : plus de fausses alertes. La precision passe de 0.622 à 0.543 (près d'une alerte sur deux est fausse), et 21 % des machines du jeu de test déclenchent une alerte, contre 16 % au seuil 0.3. L'accuracy baisse de 0.904 à 0.882. Il faudra prévoir plus d'inspections.
- **Ce que cela facilite** : la décision est traçable. L'identifiant du run donne les paramètres, les métriques, le modèle et les versions exactes des bibliothèques. Le `data_md5` donne la version des données : on la retrouve avec le reçu `data/sensors.csv.dvc` dans Git, puis `dvc checkout` ou `dvc pull`.
- **Déploiement** : le seuil est écrit dans `models/model_info.json` par `python -m maintai.train --threshold 0.2`, et l'API le relit à partir de là. La valeur par défaut `MAINTAI_THRESHOLD` (0.3 dans `config.py`) n'est pas modifiée dans ce TP : il faudra la passer à 0.2 ou toujours entraîner avec `--threshold 0.2`.
- **À surveiller** :
  - le recall en service (pannes non signalées) et le taux de fausses alertes remonté par les techniciens ;
  - la dérive des données. La v2 (8000 relevés, `data_md5 = 93fdeef2…`) existe déjà : avant de remplacer le modèle, il faut le ré-entraîner au seuil 0.2 sur la v2, puis comparer les deux modèles **sur un même jeu de test** ;
  - les données sont synthétiques : avec des données réelles, il faudra recalibrer le seuil à partir du coût réel d'une panne et d'une inspection.

## Mise à jour du 25/11/2026 : recherche par grille (ADR 005)
La recherche par grille de l'ADR 005 (`python -m maintai.search` : 30 runs, validation croisée à 5 plis sur l'entraînement, coût = 4 × pannes manquées + fausses alertes) **confirme cette décision** :
- le seuil 0.2 est le meilleur pour toutes les valeurs de C ;
- la meilleure forêt n'arrive qu'au 16e rang sur 30 ;
- le premier de la grille (C = 10, seuil 0.2) ne compte que 3 fausses alertes de moins que C = 1.0 sur 4000 relevés, et il obtient la même mesure sur le jeu de test (recall 0.838, precision 0.543, accuracy 0.882).

Le modèle retenu reste donc la régression logistique avec C = 1.0 et un seuil de 0.2.
