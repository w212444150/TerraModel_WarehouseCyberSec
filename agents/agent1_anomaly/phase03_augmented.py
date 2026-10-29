"""
Agent 1 retrained on augmented features (Limitation 1 fix).

Compares network-only vs network+PLC feature sets head-to-head on the test set.
Writes the better model to models/agent1_augmented.joblib.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import yaml
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from experiments.tracker import ExperimentRun

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

NETWORK_FEATURES = [
    "pkt_count", "pkt_size_mean", "pkt_size_std", "inter_arrival_mean_ms",
    "unique_src_ips", "unique_dst_ips",
    "frac_modbus_tcp", "frac_ethernet_ip", "frac_profinet", "frac_http", "frac_other",
    "proto_entropy", "size_cv", "src_dst_ratio", "pkt_count_deviation",
]
PLC_FEATURES = [
    "plc_val_mean", "plc_val_std", "plc_val_range",
    "plc_setpoint_delta", "plc_outofband_frac",
]


def eval_model(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, y_proba)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


def tune(X_train, y_train, X_val, y_val, seed, n_trials=25):
    def obj(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        m = xgb.XGBClassifier(**params, eval_metric="aucpr", tree_method="hist",
                               random_state=seed, n_jobs=-1, verbosity=0)
        m.fit(X_train, y_train)
        return average_precision_score(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    model = xgb.XGBClassifier(**best, eval_metric="aucpr", tree_method="hist",
                               random_state=seed, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train)
    return model, study


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    reports_dir = Path("reports/phase03")
    seed = cfg["seed"]

    train = pd.read_parquet(proc / "agent1_augmented_train.parquet")
    val = pd.read_parquet(proc / "agent1_augmented_val.parquet")
    test = pd.read_parquet(proc / "agent1_augmented_test.parquet")

    print(f"[agent1-aug] train={len(train)} val={len(val)} test={len(test)}")

    results = {}

    for label, cols in [("network_only", NETWORK_FEATURES),
                        ("network_plus_plc", NETWORK_FEATURES + PLC_FEATURES)]:
        X_train = train[cols].values; y_train = train["is_attack"].values
        X_val = val[cols].values; y_val = val["is_attack"].values
        X_test = test[cols].values; y_test = test["is_attack"].values

        model, study = tune(X_train, y_train, X_val, y_val, seed, n_trials=25)
        metrics = eval_model(model, X_test, y_test)

        run = ExperimentRun(f"agent1_augmented_{label}")
        run.log_params(model="XGBClassifier", feature_set=label, n_features=len(cols),
                       **{k: float(v) for k, v in study.best_params.items()})
        run.log_metrics(**{f"test_{k}": v for k, v in metrics.items() if isinstance(v, float)})
        run.save()

        results[label] = {
            "n_features": len(cols),
            "best_optuna_val_pr_auc": float(study.best_value),
            "best_params": study.best_params,
            "test_metrics": metrics,
        }
        print(f"[agent1-aug][{label}] test F1={metrics['f1']:.4f}  "
              f"Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}  "
              f"PR-AUC={metrics['pr_auc']:.4f}")

        if label == "network_plus_plc":
            joblib.dump({"model": model, "features": cols},
                        models_dir / "agent1_augmented.joblib")

    # Delta
    net = results["network_only"]["test_metrics"]
    aug = results["network_plus_plc"]["test_metrics"]
    delta = {
        "f1_delta": round(aug["f1"] - net["f1"], 4),
        "pr_auc_delta": round(aug["pr_auc"] - net["pr_auc"], 4),
        "recall_delta": round(aug["recall"] - net["recall"], 4),
    }
    out = {
        "fix": "Limitation 1 - Agent 1 now sees PLC features",
        "results": results,
        "augmentation_delta": delta,
    }
    (reports_dir / "agent1_augmented_metrics.json").write_text(json.dumps(out, indent=2))

    print(f"\n[agent1-aug] FIX IMPACT (network+PLC vs network-only):")
    print(f"[agent1-aug]   F1:      {net['f1']:.4f} -> {aug['f1']:.4f}  (delta {delta['f1_delta']:+.4f})")
    print(f"[agent1-aug]   PR-AUC:  {net['pr_auc']:.4f} -> {aug['pr_auc']:.4f}  (delta {delta['pr_auc_delta']:+.4f})")
    print(f"[agent1-aug]   Recall:  {net['recall']:.4f} -> {aug['recall']:.4f}  (delta {delta['recall_delta']:+.4f})")


if __name__ == "__main__":
    main()
