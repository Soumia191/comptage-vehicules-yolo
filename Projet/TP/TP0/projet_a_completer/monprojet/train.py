"""Entraîne, évalue et compare les modèles candidats.

Usage : python -m monprojet.train
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from monprojet import config
from monprojet.data import load_data, split
from monprojet.models import candidates


HIGHER_IS_BETTER = {
    "accuracy": True,
    "recall": True,
    "precision": True,
    "f1": True,
    "mae": False,
    "rmse": False,
    "r2": True,
}


def build_preprocessing(X: pd.DataFrame) -> ColumnTransformer:
    """Prépare les colonnes numériques, catégorielles et textuelles."""
    text = [config.TEXT_COLUMN] if config.TEXT_COLUMN else []

    numeric = [
        c for c in X.columns
        if c not in text and pd.api.types.is_numeric_dtype(X[c])
    ]

    categorical = [
        c for c in X.columns
        if c not in text and c not in numeric
    ]

    steps = []

    if numeric:
        steps.append((
            "nombres",
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
            ),
            numeric,
        ))

    if categorical:
        steps.append((
            "categories",
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore"),
            ),
            categorical,
        ))

    if text:
        steps.append((
            "texte",
            TfidfVectorizer(),
            config.TEXT_COLUMN,
        ))

    return ColumnTransformer(steps, sparse_threshold=0)


def evaluate(model, X_test, y_test) -> dict:
    """Prédit sur le jeu de test et calcule les métriques."""
    predictions = model.predict(X_test)

    if config.TASK == "classification":
        classes = getattr(model, "classes_", None)

        if classes is None:
            classes = pd.unique(
                np.concatenate([
                    np.asarray(y_test),
                    np.asarray(predictions),
                ])
            )

        average = "binary" if len(classes) == 2 else "macro"

        scores = {
            "accuracy": accuracy_score(y_test, predictions),
            "recall": recall_score(
                y_test,
                predictions,
                average=average,
                zero_division=0,
            ),
            "precision": precision_score(
                y_test,
                predictions,
                average=average,
                zero_division=0,
            ),
            "f1": f1_score(
                y_test,
                predictions,
                average=average,
                zero_division=0,
            ),
        }

    elif config.TASK == "regression":
        scores = {
            "mae": mean_absolute_error(y_test, predictions),
            "rmse": np.sqrt(
                mean_squared_error(y_test, predictions)
            ),
            "r2": r2_score(y_test, predictions),
        }

    else:
        raise ValueError(
            f"Type de problème inconnu : {config.TASK}"
        )

    return {
        name: round(float(value), 3)
        for name, value in scores.items()
    }


def main() -> None:
    """Entraîne les candidats et sauvegarde le meilleur modèle."""
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    results = {}
    trained_models = {}

    for name, estimator in candidates().items():
        model = Pipeline([
            ("preparation", build_preprocessing(X_train)),
            ("modele", estimator),
        ])

        model.fit(X_train, y_train)

        results[name] = evaluate(model, X_test, y_test)
        trained_models[name] = model

    if not results:
        raise ValueError("Aucun modèle candidat à entraîner.")

    metric = config.MAIN_METRIC

    if metric not in HIGHER_IS_BETTER:
        raise ValueError(
            f"Métrique principale inconnue : {metric}"
        )

    if any(
        metric not in scores
        for scores in results.values()
    ):
        raise ValueError(
            f"La métrique '{metric}' ne correspond pas "
            f"à la tâche '{config.TASK}'."
        )

    higher_is_better = HIGHER_IS_BETTER[metric]

    comparison = pd.DataFrame(results).T
    comparison = comparison.sort_values(
        by=metric,
        ascending=not higher_is_better,
        kind="stable",
    )

    print(comparison)

    if higher_is_better:
        best_name = max(
            results,
            key=lambda name: results[name][metric],
        )
    else:
        best_name = min(
            results,
            key=lambda name: results[name][metric],
        )

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        trained_models[best_name],
        config.MODELS_DIR / "model.joblib",
    )

    metrics = {
        "main_metric": metric,
        "best": best_name,
        "results": results,
    }

    with (config.MODELS_DIR / "metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nMeilleur modèle selon {metric} : {best_name}"
    )


if __name__ == "__main__":
    main()