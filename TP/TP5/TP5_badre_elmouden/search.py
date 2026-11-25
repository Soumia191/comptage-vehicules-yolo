"""Recherche par grille (grid search) des réglages de MaintAI, plafonnée à 30 runs MLflow.

Principe :
1. chaque combinaison (modèle, hyperparamètre, seuil d'alerte) est évaluée par validation croisée
   stratifiée sur le jeu d'ENTRAÎNEMENT seulement (prédictions hors pli) : le jeu de test ne sert
   pas à choisir, sinon sa mesure serait optimiste ;
2. chaque combinaison devient un run MLflow « enfant », rangé sous un run parent « grid-search »
   dans l'expérience maintai-grid-search ;
3. la combinaison au coût le plus faible (coût = cost_fn × pannes manquées + cost_fp × fausses
   alertes) est ré-entraînée sur tout l'entraînement puis mesurée UNE seule fois sur le test.
   Son modèle est enregistré dans le run parent (et dans models/ avec --save).

Le seuil ne demande pas de ré-entraîner : pour un modèle donné, on calcule une fois les
probabilités hors pli, puis on les compare à chaque seuil de la grille.

Usage : python -m maintai.search
        python -m maintai.search --folds 5 --cost-fn 4 --max-runs 30
        python -m maintai.search --save          (remplace models/ par le meilleur modèle)
"""
import argparse
import itertools
import json
import os
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from maintai.config import SEED, get_settings
from maintai.data import load_data, split
from maintai.tracking import TRUSTED_TYPES, file_md5
from maintai.train import build_model, evaluate, evaluate_baseline, save

EXPERIMENT = "maintai-grid-search"
MAX_RUNS = 30

# 4 valeurs de C × 5 seuils = 20 runs ; 2 forêts × 5 seuils = 10 runs ; total = 30 runs
GRID: dict[str, dict[str, list]] = {
    "logreg": {"C": [0.01, 0.1, 1.0, 10.0]},
    "forest": {"n_estimators": [200], "max_depth": [None, 8]},
}
THRESHOLDS = [0.15, 0.2, 0.25, 0.3, 0.4]
CV_METRICS = ["cv_recall", "cv_precision", "cv_accuracy", "cv_missed", "cv_false_alerts", "cv_cost_per_1000"]


def combinations(grid: dict, thresholds: list[float]) -> list[dict]:
    """Toutes les combinaisons de la grille : une par run MLflow."""
    combos = []
    for kind, params in grid.items():
        names = list(params)
        for values in itertools.product(*(params[n] for n in names)):
            for t in thresholds:
                combos.append({"model": kind, **dict(zip(names, values)), "threshold": t})
    return combos


def cv_probabilities(kind: str, hyper: dict, X, y, folds: int) -> np.ndarray:
    """Probabilité de panne de chaque relevé d'entraînement, prédite par un modèle qui ne l'a pas vu."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    model = build_model(kind, **hyper)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def score(y, proba: np.ndarray, threshold: float, cost_fn: float, cost_fp: float) -> dict:
    """Métriques d'une combinaison, à partir des prédictions hors pli."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = len(y)
    return {
        "cv_recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "cv_precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "cv_accuracy": round((tp + tn) / n, 4),
        "cv_missed": int(fn),
        "cv_false_alerts": int(fp),
        # coût pour 1000 relevés : comparable quelle que soit la taille des données
        "cv_cost_per_1000": round(1000 * (cost_fn * fn + cost_fp * fp) / n, 2),
    }


def hyperparams(combo: dict) -> dict:
    return {k: v for k, v in combo.items() if k not in ("model", "threshold")}


def label(combo: dict) -> str:
    h = ", ".join(f"{k}={v}" for k, v in hyperparams(combo).items())
    return f"{combo['model']} ({h}), seuil {combo['threshold']}"


def run_search(grid: dict | None = None, thresholds: list[float] | None = None, folds: int = 5,
               cost_fn: float = 4.0, cost_fp: float = 1.0, max_runs: int = MAX_RUNS,
               save_best: bool = False) -> dict:
    grid = grid or GRID
    thresholds = thresholds or THRESHOLDS
    combos = combinations(grid, thresholds)
    if len(combos) > max_runs:
        raise ValueError(f"La grille contient {len(combos)} combinaisons, plus que le maximum de "
                         f"{max_runs} runs : réduisez la grille ou augmentez --max-runs.")

    settings = get_settings()
    df = load_data(settings.data_path)
    X_train, X_test, y_train, y_test = split(df)
    data_md5 = file_md5(settings.data_path)

    mlflow.set_experiment(EXPERIMENT)
    rows = []
    with mlflow.start_run(run_name="grid-search") as parent:
        mlflow.log_params({
            "search": "grid", "n_runs": len(combos), "max_runs": max_runs, "folds": folds,
            "cost_fn": cost_fn, "cost_fp": cost_fp, "thresholds": str(thresholds),
            "grid": json.dumps(grid), "data_md5": data_md5, "n_train": len(X_train),
        })
        # Un seul calcul hors pli par (modèle, hyperparamètres), réutilisé pour tous les seuils
        cache: dict[str, np.ndarray] = {}
        for i, combo in enumerate(combos):
            hyper = hyperparams(combo)
            key = json.dumps({"model": combo["model"], **hyper}, sort_keys=True)
            if key not in cache:
                cache[key] = cv_probabilities(combo["model"], hyper, X_train, y_train, folds)
            metrics = score(y_train, cache[key], combo["threshold"], cost_fn, cost_fp)
            with mlflow.start_run(run_name=label(combo), nested=True) as child:
                mlflow.log_params({**combo, "data_md5": data_md5, "n_train": len(X_train), "folds": folds})
                mlflow.log_metrics(metrics)
            rows.append({"combo": i, **combo, **metrics, "run_id": child.info.run_id})

        # Choix : coût le plus faible, puis meilleur recall, puis meilleure precision
        table = pd.DataFrame(rows).sort_values(
            ["cv_cost_per_1000", "cv_recall", "cv_precision"], ascending=[True, False, False]
        ).reset_index(drop=True)
        best = table.iloc[0]
        best_combo = combos[int(best["combo"])]
        hyper = hyperparams(best_combo)

        # Le meilleur réglage est ré-entraîné sur tout l'entraînement, puis mesuré une seule fois sur le test
        model = build_model(best_combo["model"], **hyper).fit(X_train, y_train)
        test_metrics = evaluate(model, X_test, y_test, best_combo["threshold"])
        baseline = evaluate_baseline(X_train, y_train, X_test, y_test)
        mlflow.log_params({f"best_{k}": v for k, v in best_combo.items()})
        mlflow.log_param("best_run_id", best["run_id"])
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if k != "threshold"}
                           | {"best_cv_cost_per_1000": float(best["cv_cost_per_1000"]),
                              "baseline_recall": baseline["recall"]})
        mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(3),
                                 skops_trusted_types=TRUSTED_TYPES)

    result = {
        "parent_run_id": parent.info.run_id,
        "best_run_id": best["run_id"],
        "best": best_combo,
        "cv": {k: float(best[k]) for k in CV_METRICS},
        "test": test_metrics,
        "baseline": baseline,
        "n_runs": len(combos),
        "data_md5": data_md5,
        "table": table,
    }
    if save_best:
        save(model, {"model": test_metrics, "baseline": baseline, "n_train": len(X_train),
                     "n_test": len(X_test), "search": {k: result[k] for k in
                                                       ("parent_run_id", "best_run_id", "best", "n_runs")}},
             settings, best_combo["threshold"])
    return result


def markdown_table(table: pd.DataFrame, top: int | None = None) -> str:
    """Tableau prêt à coller dans le compte rendu."""
    t = table if top is None else table.head(top)
    lines = ["| # | Modèle | Hyperparamètres | Seuil | recall CV | precision CV | accuracy CV | "
             "pannes manquées | fausses alertes | coût / 1000 | run_id (8 car.) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in t.reset_index(drop=True).iterrows():
        hyper = "C = " + str(r["C"]) if r["model"] == "logreg" else (
            f"{int(r['n_estimators'])} arbres, max_depth = "
            f"{'None' if pd.isna(r['max_depth']) else int(r['max_depth'])}")
        lines.append(f"| {i + 1} | {r['model']} | {hyper} | {r['threshold']} | {r['cv_recall']:.3f} | "
                     f"{r['cv_precision']:.3f} | {r['cv_accuracy']:.3f} | {r['cv_missed']} | "
                     f"{r['cv_false_alerts']} | {r['cv_cost_per_1000']:.1f} | `{r['run_id'][:8]}` |")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m maintai.search")
    p.add_argument("--folds", type=int, default=5, help="plis de la validation croisée (défaut : 5)")
    p.add_argument("--cost-fn", type=float, default=4.0,
                   help="coût d'une panne manquée, en nombre d'inspections inutiles (défaut : 4)")
    p.add_argument("--cost-fp", type=float, default=1.0, help="coût d'une fausse alerte (défaut : 1)")
    p.add_argument("--max-runs", type=int, default=MAX_RUNS, help="nombre maximal de runs (défaut : 30)")
    p.add_argument("--save", action="store_true", help="écrire le meilleur modèle dans models/")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv or [])
    res = run_search(folds=args.folds, cost_fn=args.cost_fn, cost_fp=args.cost_fp,
                     max_runs=args.max_runs, save_best=args.save)
    table = res["table"]
    out = get_settings().models_dir
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "grid_search.csv", index=False)
    print(markdown_table(table))
    print(f"\n{res['n_runs']} runs enfants dans l'expérience {EXPERIMENT} (run parent {res['parent_run_id']})")
    print(f"Meilleur réglage (validation croisée) : {label(res['best'])} — run {res['best_run_id']}")
    print("Mesure unique sur le jeu de test : " + json.dumps(res["test"]))
    print(f"Tableau complet : {out / 'grid_search.csv'}")
    if args.save:
        print(f"Meilleur modèle écrit dans {get_settings().model_path}")
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
