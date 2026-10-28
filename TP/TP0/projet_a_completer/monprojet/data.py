"""Chargement et découpage des données."""

import pandas as pd
from sklearn.model_selection import train_test_split

from monprojet import config


def load_data(path=None) -> pd.DataFrame:
    """Lit le CSV et retire les lignes dont la cible est manquante."""
    df = pd.read_csv(config.DATA_PATH if path is None else path)

    if config.TARGET not in df.columns:
        raise ValueError(
            f"La colonne cible '{config.TARGET}' est absente du fichier CSV."
        )

    return df.dropna(subset=[config.TARGET])


def feature_columns(df: pd.DataFrame) -> list:
    """Renvoie les colonnes d'entrée du modèle."""
    if config.FEATURES is not None:
        return list(config.FEATURES)

    excluded = {config.TARGET, config.TIME_COLUMN}
    return [c for c in df.columns if c not in excluded]


def split(df: pd.DataFrame):
    """Renvoie X_train, X_test, y_train, y_test."""
    X = df[feature_columns(df)]
    y = df[config.TARGET]

    if config.SPLIT == "random":
        return train_test_split(
            X,
            y,
            test_size=config.TEST_SIZE,
            random_state=config.SEED,
            stratify=y if config.TASK == "classification" else None,
        )

    if config.SPLIT == "chronologique":
        if config.TIME_COLUMN is None:
            raise ValueError(
                "TIME_COLUMN est obligatoire pour un découpage chronologique."
            )

        if config.TIME_COLUMN not in df.columns:
            raise ValueError(
                f"La colonne de date '{config.TIME_COLUMN}' est absente."
            )

        order = pd.to_datetime(df[config.TIME_COLUMN]).argsort()
        X = X.iloc[order]
        y = y.iloc[order]

        return train_test_split(
            X,
            y,
            test_size=config.TEST_SIZE,
            shuffle=False,
        )

    raise ValueError(f"Découpage inconnu : {config.SPLIT}")