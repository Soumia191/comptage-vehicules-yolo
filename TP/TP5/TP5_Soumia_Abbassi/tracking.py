"""Suivi des entraînements avec MLflow : chaque entraînement devient un « run » qu'on peut comparer.

Les runs sont enregistrés dans mlflow.db (dossier courant) ; on les regarde avec : mlflow ui
"""
import hashlib
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # coupe un message d'information de MLflow

import mlflow
import mlflow.sklearn

EXPERIMENT = "maintai"
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]  # nécessaire pour --model forest


def file_md5(path: Path) -> str:
    """Empreinte du fichier de données : la même que celle que DVC écrit dans le fichier .dvc."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()



def log_run(params: dict, metrics: dict, model, X_example, data_path: Path) -> str:
    """
    Enregistre un entraînement de MaintAI dans MLflow.

    Chaque entraînement est identifié par un run unique contenant :
    - les paramètres utilisés pour entraîner le modèle ;
    - l'empreinte MD5 des données pour assurer leur traçabilité ;
    - les métriques de performance obtenues ;
    - le modèle entraîné, afin de pouvoir le réutiliser.

    Args:
        params: Dictionnaire des paramètres d'entraînement.
        metrics: Dictionnaire des métriques de performance.
        model: Modèle de Machine Learning entraîné.
        X_example: Exemple des données d'entrée du modèle.
        data_path: Chemin vers le fichier CSV utilisé pour l'entraînement.

    Returns:
        str: Identifiant unique du run MLflow créé.
    """

    # Étape 1 : sélectionner l'expérience "maintai".
    # Elle permet de regrouper les entraînements du projet dans MLflow.
    mlflow.set_experiment(EXPERIMENT)

    # Étape 2 : ouvrir un nouveau run MLflow.
    # Le gestionnaire "with" assure sa fermeture automatiquement.
    with mlflow.start_run() as run:

        # Étape 3 : enregistrer les paramètres de configuration.
        # Exemples : type de modèle, seuil de décision et valeur de C.
        mlflow.log_params(params)

        # Étape 4 : calculer et enregistrer l'empreinte MD5 des données.
        # Elle permettra d'identifier la version du dataset avec DVC.
        data_md5 = file_md5(data_path)
        mlflow.log_param("data_md5", data_md5)

        # Étape 5 : enregistrer les métriques de performance.
        # Exemples : accuracy, recall et precision.
        mlflow.log_metrics(metrics)

        # Étape 6 : sauvegarder le modèle dans les artefacts MLflow.
        # L'exemple d'entrée décrit le format attendu par le modèle.
        # Les types autorisés facilitent le chargement sécurisé avec skops.
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_example,
            skops_trusted_types=TRUSTED_TYPES,
        )

        # Étape 7 : retourner l'identifiant unique de l'entraînement.
        # Cet identifiant sert à retrouver le run dans MLflow.
        return run.info.run_id

