"""Les modèles à comparer."""

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from monprojet import config


def candidates() -> dict:
    """Renvoie les modèles non entraînés, avec la baseline en premier."""
    if config.TASK == "classification":
        return {
            "baseline": DummyClassifier(
                strategy="most_frequent",
                random_state=config.SEED,
            ),
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                random_state=config.SEED,
            ),
            "decision_tree": DecisionTreeClassifier(
                random_state=config.SEED,
            ),
            "random_forest": RandomForestClassifier(
                random_state=config.SEED,
            ),
        }

    if config.TASK == "regression":
        return {
            "baseline": DummyRegressor(strategy="mean"),
            "linear_regression": LinearRegression(),
            "decision_tree": DecisionTreeRegressor(
                random_state=config.SEED,
            ),
            "random_forest": RandomForestRegressor(
                random_state=config.SEED,
            ),
        }

    raise ValueError(f"Type de problème inconnu : {config.TASK}")