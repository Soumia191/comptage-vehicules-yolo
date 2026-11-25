"""Recherche par grille : 30 runs au maximum, un run MLflow par combinaison, choix sur la validation croisée."""
import json

import mlflow
import pytest

from maintai.search import EXPERIMENT, GRID, MAX_RUNS, THRESHOLDS, combinations, run_search

SMALL_GRID = {"logreg": {"C": [0.1, 1.0]}, "forest": {"n_estimators": [20], "max_depth": [4]}}


def test_default_grid_fits_in_30_runs():
    assert MAX_RUNS == 30
    assert len(combinations(GRID, THRESHOLDS)) <= MAX_RUNS


def test_grid_larger_than_the_maximum_is_refused(settings_env):
    with pytest.raises(ValueError, match="maximum"):
        run_search(grid={"logreg": {"C": [0.1, 1.0, 10.0]}}, thresholds=[0.2, 0.3], max_runs=5)


def test_one_child_run_per_combination(settings_env, tmp_path, monkeypatch):
    monkeypatch.setenv("MAINTAI_MODELS_DIR", str(tmp_path / "models"))
    res = run_search(grid=SMALL_GRID, thresholds=[0.2, 0.3], folds=3, save_best=True)
    assert res["n_runs"] == 6
    children = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string=f"tags.mlflow.parentRunId = '{res['parent_run_id']}'",
    )
    assert len(children) == 6
    assert set(children["params.data_md5"]) == {res["data_md5"]}


def test_winner_is_chosen_on_cross_validation(settings_env, tmp_path, monkeypatch):
    monkeypatch.setenv("MAINTAI_MODELS_DIR", str(tmp_path / "models"))
    res = run_search(grid=SMALL_GRID, thresholds=[0.2, 0.3], folds=3, save_best=True)
    # le gagnant a le coût de validation croisée le plus faible…
    assert res["cv"]["cv_cost_per_1000"] == res["table"]["cv_cost_per_1000"].min()
    # …et le modèle écrit dans models/ porte son seuil
    info = json.loads((tmp_path / "models" / "model_info.json").read_text(encoding="utf-8"))
    assert info["threshold"] == res["best"]["threshold"]
    metrics = json.loads((tmp_path / "models" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["search"]["best_run_id"] == res["best_run_id"]
