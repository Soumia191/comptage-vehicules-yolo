
"""
TP5 - Recherche des meilleurs hyperparamètres pour MaintAI.

Objectifs :
1. Tester 30 configurations avec une boucle.
2. Utiliser une validation croisée à 5 plis.
3. Comparer recall, precision et accuracy.
4. Sélectionner la meilleure configuration.
5. Évaluer le modèle final sur les données de test.
6. Enregistrer le résultat dans MLflow.

Exécution :
    python -m maintai.grid_search
"""

import csv
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from maintai.config import SEED, get_settings
from maintai.data import load_data, split
from maintai.train import build_model, evaluate
from maintai.tracking import log_run


def main():
    # 1. Charger les données et conserver le jeu de test séparé.
    # Le jeu de test ne participe pas au choix des hyperparamètres.
    settings = get_settings()
    df = load_data(settings.data_path)
    X_train, X_test, y_train, y_test = split(df)

    # 2. Définir les cinq hyperparamètres et leurs valeurs.
    # Les trois premiers varient pour produire 30 configurations.
    c_values = [0.01, 0.1, 1.0]
    solvers = ["lbfgs", "liblinear"]
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
    class_weights = ["balanced"]
    max_iterations = [1000]

    # 3. Générer les combinaisons automatiquement.
    combinations = list(product(
        c_values,
        solvers,
        thresholds,
        class_weights,
        max_iterations,
    ))

    print(f"Nombre de configurations : {len(combinations)}")

    # 4. Préparer la validation croisée stratifiée.
    # Chaque pli conserve approximativement la proportion de pannes.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    results = []

    # 5. Parcourir toutes les configurations.
    for index, (C, solver, threshold, weight, max_iter) in enumerate(
        combinations, start=1
    ):
        recalls = []
        precisions = []
        accuracies = []

        # 6. Évaluer chaque configuration sur cinq plis.
        for train_idx, valid_idx in cv.split(X_train, y_train):
            X_fit = X_train.iloc[train_idx]
            X_valid = X_train.iloc[valid_idx]
            y_fit = y_train.iloc[train_idx]
            y_valid = y_train.iloc[valid_idx]

            # Construire une nouvelle pipeline pour chaque pli.
            model = build_model("logreg", C=C)
            model.set_params(
                clf__solver=solver,
                clf__class_weight=weight,
                clf__max_iter=max_iter,
            )

            # Entraîner uniquement sur le pli d'apprentissage.
            model.fit(X_fit, y_fit)

            # Convertir les probabilités en alertes avec le seuil testé.
            probabilities = model.predict_proba(X_valid)[:, 1]
            predictions = (probabilities >= threshold).astype(int)

            # Mesurer les performances sur le pli de validation.
            recalls.append(recall_score(
                y_valid, predictions, zero_division=0
            ))
            precisions.append(precision_score(
                y_valid, predictions, zero_division=0
            ))
            accuracies.append(accuracy_score(
                y_valid, predictions
            ))

        # 7. Calculer les moyennes de validation croisée.
        result = {
            "configuration": index,
            "C": C,
            "solver": solver,
            "threshold": threshold,
            "class_weight": weight,
            "max_iter": max_iter,
            "recall_cv": round(float(np.mean(recalls)), 4),
            "precision_cv": round(float(np.mean(precisions)), 4),
            "accuracy_cv": round(float(np.mean(accuracies)), 4),
        }

        results.append(result)

        print(
            f"{index:02d}/30 | C={C} | solver={solver} "
            f"| seuil={threshold} "
            f"| recall={result['recall_cv']:.4f} "
            f"| precision={result['precision_cv']:.4f} "
            f"| accuracy={result['accuracy_cv']:.4f}"
        )


    # 8. Calculer le F2-score de chaque configuration.
    # Le F2-score donne davantage d'importance au recall
    # qu'à la précision, car manquer une panne est coûteux.
    for result in results:
        precision = result["precision_cv"]
        recall = result["recall_cv"]

        # Formule du F2-score : 5 * P * R / (4 * P + R).
        denominator = 4 * precision + recall

        result["f2_cv"] = (
            round(5 * precision * recall / denominator, 4)
            if denominator > 0 else 0.0
        )

    # 9. Sélectionner la configuration avec le meilleur F2-score.
    # La précision puis l'accuracy départagent les égalités.
    best = max(
        results,
        key=lambda r: (
            r["f2_cv"],
            r["precision_cv"],
            r["accuracy_cv"],
        ),
    )

    print("\nMEILLEURE CONFIGURATION EN VALIDATION CROISÉE")
    print(best)

    # La contrainte initiale de précision de 50 % n'est
    # satisfaite par aucune configuration de cette grille.
    # Le modèle choisi est donc une sélection expérimentale,
    # et non une solution répondant à cette contrainte.


    # 10. Entraîner le modèle retenu sur tout le jeu d'entraînement.
    best_model = build_model("logreg", C=best["C"])
    best_model.set_params(
        clf__solver=best["solver"],
        clf__class_weight=best["class_weight"],
        clf__max_iter=best["max_iter"],
    )
    best_model.fit(X_train, y_train)

    # 11. Évaluer une seule fois sur le jeu de test indépendant.
    final_metrics = evaluate(
        best_model,
        X_test,
        y_test,
        best["threshold"],
    )

    print("\nRÉSULTATS SUR LE JEU DE TEST")
    print(final_metrics)

    # 12. Enregistrer la configuration finale dans MLflow.
    # Le MD5 du dataset est ajouté automatiquement par log_run.
    run_id = log_run(
        params={
            "model": "logreg",
            "search": "grid_search_30",
            "C": best["C"],
            "solver": best["solver"],
            "threshold": best["threshold"],
            "class_weight": best["class_weight"],
            "max_iter": best["max_iter"],
            "n_train": len(X_train),
        },
        metrics={
            "recall": final_metrics["recall"],
            "precision": final_metrics["precision"],
            "accuracy": final_metrics["accuracy"],
            "recall_cv": best["recall_cv"],
            "precision_cv": best["precision_cv"],
            "accuracy_cv": best["accuracy_cv"],
        },
        model=best_model,
        X_example=X_train.head(3),
        data_path=settings.data_path,
    )

    print(f"\nRun MLflow : {run_id}")

    # 13. Sauvegarder le tableau complet des 30 configurations.
    # Ce fichier servira de preuve dans le compte rendu.
    output = Path("docs/grid_search_results.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"Résultats sauvegardés : {output}")


if __name__ == "__main__":
    main()
