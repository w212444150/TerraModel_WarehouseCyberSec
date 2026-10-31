"""
Phase 03 Week 10: final model optimization pass.

Our initial Phase 03 Agent 1 and Agent 3 XGBoost tuning ran 30 Optuna trials
over a moderate hyperparameter range. Once all the limitation fixes landed
(augmented features, realistic data, KEV labels), a final re-tune pass is
expected to extract more performance from the stable final feature set.

This script runs a wider, deeper search:
  - 100 Optuna trials per agent (vs the earlier 30)
  - Broader parameter ranges (n_estimators up to 1000, deeper trees, wider
    regularization range, scale_pos_weight added)
  - Early-stopping via validation set to avoid overfitting
  - Compares against the Phase 03 champion and promotes only if PR-AUC
    strictly improves

Outputs:
  - models/agent1_final.joblib
  - models/agent3_final.joblib
  - reports/phase03/final_optimization.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

A1_FEATURES = [
    "pkt_count", "pkt_size_mean", "pkt_size_std", "inter_arrival_mean_ms",
    "unique_src_ips", "unique_dst_ips",
    "frac_modbus_tcp", "frac_ethernet_ip", "frac_profinet", "frac_http", "frac_other",
    "proto_entropy", "size_cv", "src_dst_ratio", "pkt_count_deviation",
]
A3_FEATURES = [
    "mean_dev", "max_dev", "std_dev", "torque_var",
    "timing_jitter", "nonprimary_src_frac", "cmd_entropy",
]


def eval_bin(model, X, y):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, y_proba)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


def final_tune(agent_name, X_train, y_train, X_val, y_val, X_test, y_test, seed, n_trials=100):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 2.0),
        }
        m = xgb.XGBClassifier(**params, eval_metric="aucpr", tree_method="hist",
                               random_state=seed, n_jobs=-1, verbosity=0,
                               early_stopping_rounds=20)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return average_precision_score(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"[{agent_name}] Final tuning: {n_trials} trials, best val PR-AUC = {study.best_value:.4f}")

    # Refit on train+val using best params
    final = xgb.XGBClassifier(**best, eval_metric="aucpr", tree_method="hist",
                               random_state=seed, n_jobs=-1, verbosity=0)
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    final.fit(X_all, y_all)
    test_metrics = eval_bin(final, X_test, y_test)
    return final, best, test_metrics, float(study.best_value)


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    reports_dir = Path("reports/phase03")
    seed = cfg["seed"]

    # Load Phase 03 champion metrics for comparison
    a1_prior = {}
    a3_prior = {}
    try:
        p = json.loads((reports_dir / "agent1_metrics.json").read_text())
        a1_prior = p[f"test_{p['champion']}"]
    except Exception:
        pass
    try:
        p = json.loads((reports_dir / "agent3_metrics.json").read_text())
        a3_prior = p[f"test_{p['champion']}"]
    except Exception:
        pass

    # Agent 1
    a1_train = pd.read_parquet(proc / "agent1_train.parquet")
    a1_val = pd.read_parquet(proc / "agent1_val.parquet")
    a1_test = pd.read_parquet(proc / "agent1_test.parquet")
    a1_model, a1_best, a1_test_metrics, a1_val_pr = final_tune(
        "agent1",
        a1_train[A1_FEATURES].values, a1_train["is_attack"].values,
        a1_val[A1_FEATURES].values, a1_val["is_attack"].values,
        a1_test[A1_FEATURES].values, a1_test["is_attack"].values,
        seed, n_trials=100,
    )
    joblib.dump({"model": a1_model, "features": A1_FEATURES}, models_dir / "agent1_final.joblib")

    # Agent 3
    a3_train = pd.read_parquet(proc / "agent3_train.parquet")
    a3_val = pd.read_parquet(proc / "agent3_val.parquet")
    a3_test = pd.read_parquet(proc / "agent3_test.parquet")
    a3_model, a3_best, a3_test_metrics, a3_val_pr = final_tune(
        "agent3",
        a3_train[A3_FEATURES].values, a3_train["is_attack"].values,
        a3_val[A3_FEATURES].values, a3_val["is_attack"].values,
        a3_test[A3_FEATURES].values, a3_test["is_attack"].values,
        seed, n_trials=100,
    )
    joblib.dump({"model": a3_model, "features": A3_FEATURES}, models_dir / "agent3_final.joblib")

    def promote(prior, now):
        if not prior:
            return "no prior to compare"
        delta = now["pr_auc"] - prior["pr_auc"]
        return f"PR-AUC {prior['pr_auc']:.4f} -> {now['pr_auc']:.4f} (delta {delta:+.4f}) " \
               f"{'PROMOTE' if delta > 0 else 'keep prior'}"

    report = {
        "method": "Final optimization pass: 100 Optuna trials with widened search space, early stopping",
        "search_space": {
            "n_estimators": "[100, 1000]  (was [100, 500])",
            "max_depth": "[3, 15]  (was [3, 10])",
            "learning_rate": "[0.005, 0.3] log  (was [0.01, 0.3])",
            "subsample": "[0.5, 1.0]  (was [0.6, 1.0])",
            "colsample_bytree": "[0.5, 1.0]  (was [0.6, 1.0])",
            "min_child_weight": "[1, 20]  (was [1, 10])",
            "reg_lambda": "[1e-5, 10] log  (was [0.0, 5.0])",
            "reg_alpha": "[1e-5, 10] log  (new)",
            "gamma": "[0.0, 5.0]  (new)",
            "scale_pos_weight": "[0.5, 2.0]  (new)",
        },
        "trials_per_agent": 100,
        "early_stopping_rounds": 20,
        "agent1": {
            "best_params": a1_best,
            "best_val_pr_auc": a1_val_pr,
            "test_metrics": a1_test_metrics,
            "prior_champion_test_metrics": a1_prior,
            "promotion_decision": promote(a1_prior, a1_test_metrics),
        },
        "agent3": {
            "best_params": a3_best,
            "best_val_pr_auc": a3_val_pr,
            "test_metrics": a3_test_metrics,
            "prior_champion_test_metrics": a3_prior,
            "promotion_decision": promote(a3_prior, a3_test_metrics),
        },
    }
    (reports_dir / "final_optimization.json").write_text(json.dumps(report, indent=2))

    print(f"\n[final-opt] Agent 1 test: F1={a1_test_metrics['f1']:.4f}  PR-AUC={a1_test_metrics['pr_auc']:.4f}")
    print(f"[final-opt]   vs prior:   {promote(a1_prior, a1_test_metrics)}")
    print(f"[final-opt] Agent 3 test: F1={a3_test_metrics['f1']:.4f}  PR-AUC={a3_test_metrics['pr_auc']:.4f}")
    print(f"[final-opt]   vs prior:   {promote(a3_prior, a3_test_metrics)}")
    print(f"\n[final-opt] Report: reports/phase03/final_optimization.json")


if __name__ == "__main__":
    main()
