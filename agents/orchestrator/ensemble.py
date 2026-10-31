"""
Phase 03 Week 10 refinement: real ensemble classifier.

The orchestrator is already a "soft ensemble" because it correlates alerts
across three independent agents. This module adds a formal ML ensemble
inside each per-agent model: a VotingClassifier (soft voting) that
combines Random Forest and XGBoost predictions for Agent 1 and Agent 3.

Why both:
  - Random Forest is strong on small, mixed-type feature sets with low overfit risk
  - XGBoost is strong when boundaries between classes are subtle
  - Averaging their probability outputs (soft vote) typically reduces variance
    and often improves PR-AUC by 1 to 3 percentage points on tabular data

If the ensemble beats both individual champions on the test set, it becomes
the production model. If not, we keep the single-model champion and record
the ensemble result for transparency.

Outputs:
  - models/agent1_ensemble.joblib / agent3_ensemble.joblib
  - reports/phase03/ensemble_metrics.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

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


def eval_binary(model, X, y):
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


def build_ensemble(agent_name: str, train_df, test_df, feature_cols, seed) -> dict:
    X_train = train_df[feature_cols].values
    y_train = train_df["is_attack"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["is_attack"].values

    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                 class_weight="balanced", n_jobs=-1, random_state=seed)
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                   eval_metric="aucpr", tree_method="hist",
                                   random_state=seed, n_jobs=-1, verbosity=0)
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb_model)],
        voting="soft",
        n_jobs=1,
    )

    rf.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    ensemble.fit(X_train, y_train)

    rf_metrics = eval_binary(rf, X_test, y_test)
    xgb_metrics = eval_binary(xgb_model, X_test, y_test)
    ens_metrics = eval_binary(ensemble, X_test, y_test)

    # Pick production model: ensemble wins only if PR-AUC strictly beats both components
    best_single_pr = max(rf_metrics["pr_auc"], xgb_metrics["pr_auc"])
    ensemble_wins = ens_metrics["pr_auc"] > best_single_pr

    print(f"\n[{agent_name}] Model comparison on test set:")
    print(f"[{agent_name}]   Random Forest:  F1={rf_metrics['f1']:.4f}  PR-AUC={rf_metrics['pr_auc']:.4f}")
    print(f"[{agent_name}]   XGBoost:        F1={xgb_metrics['f1']:.4f}  PR-AUC={xgb_metrics['pr_auc']:.4f}")
    print(f"[{agent_name}]   Soft Ensemble:  F1={ens_metrics['f1']:.4f}  PR-AUC={ens_metrics['pr_auc']:.4f}")
    print(f"[{agent_name}]   Production pick: {'ENSEMBLE' if ensemble_wins else 'single-model champion'}")

    return {
        "agent": agent_name,
        "random_forest": rf_metrics,
        "xgboost": xgb_metrics,
        "soft_ensemble": ens_metrics,
        "ensemble_pr_auc_delta_vs_best_single": round(ens_metrics["pr_auc"] - best_single_pr, 4),
        "ensemble_wins": ensemble_wins,
        "production_pick": "soft_ensemble" if ensemble_wins else "best_single",
        "ensemble_model": ensemble,
    }


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    reports_dir = Path("reports/phase03")
    seed = cfg["seed"]

    a1_train = pd.read_parquet(proc / "agent1_train.parquet")
    a1_test = pd.read_parquet(proc / "agent1_test.parquet")
    a3_train = pd.read_parquet(proc / "agent3_train.parquet")
    a3_test = pd.read_parquet(proc / "agent3_test.parquet")

    a1 = build_ensemble("agent1", a1_train, a1_test, A1_FEATURES, seed)
    a3 = build_ensemble("agent3", a3_train, a3_test, A3_FEATURES, seed)

    joblib.dump({"model": a1["ensemble_model"], "features": A1_FEATURES},
                models_dir / "agent1_ensemble.joblib")
    joblib.dump({"model": a3["ensemble_model"], "features": A3_FEATURES},
                models_dir / "agent3_ensemble.joblib")

    # Strip model object before JSON
    for a in (a1, a3):
        a.pop("ensemble_model", None)

    report = {
        "method": "VotingClassifier (soft voting, RF + XGBoost)",
        "rationale": ("Random Forest reduces variance via bagging; XGBoost reduces "
                      "bias via boosting. Averaging their probabilities tends to lift "
                      "PR-AUC by 1-3 pp on tabular data when the individual models "
                      "have different error patterns."),
        "agent1": a1,
        "agent3": a3,
    }
    (reports_dir / "ensemble_metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\n[ensemble] Wrote reports/phase03/ensemble_metrics.json")


if __name__ == "__main__":
    main()
