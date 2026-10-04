"""
Agent 1 baseline: Isolation Forest on Agent 1 network-window features.

Trains on NORMAL windows from the training set only (unsupervised anomaly detection).
Scores on the test set. Reports F1 (binary normal vs attack), precision, recall,
and PR-AUC.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "pkt_count", "pkt_size_mean", "pkt_size_std", "inter_arrival_mean_ms",
    "unique_src_ips", "unique_dst_ips",
    "frac_modbus_tcp", "frac_ethernet_ip", "frac_profinet", "frac_http", "frac_other",
    "proto_entropy", "size_cv", "src_dst_ratio", "pkt_count_deviation",
]


def main() -> None:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(proc / "agent1_train.parquet")
    test = pd.read_parquet(proc / "agent1_test.parquet")

    # Train on NORMAL rows only (unsupervised anomaly detection)
    train_normal = train[train["attack_class"] == "normal"]
    X_train = train_normal[FEATURE_COLS].values
    X_test = test[FEATURE_COLS].values
    y_test = test["is_attack"].values

    scaler = StandardScaler().fit(X_train)
    Xs_train = scaler.transform(X_train)
    Xs_test = scaler.transform(X_test)

    agent1_cfg = cfg["agent1"]
    model = IsolationForest(
        n_estimators=agent1_cfg["n_estimators"],
        contamination=agent1_cfg["contamination"],
        random_state=cfg["seed"],
    )
    model.fit(Xs_train)

    # score_samples: higher = more normal. We negate so higher = more anomalous.
    anomaly_score = -model.score_samples(Xs_test)
    y_pred = (model.predict(Xs_test) == -1).astype(int)  # -1 means anomaly

    metrics = {
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, anomaly_score)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_test": int(len(y_test)),
        "n_attack_test": int(y_test.sum()),
    }

    joblib.dump({"model": model, "scaler": scaler, "features": FEATURE_COLS},
                models_dir / "agent1_isolation_forest.joblib")
    (reports_dir / "agent1_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("[agent1] Isolation Forest baseline on test set:")
    print(f"[agent1]   F1        = {metrics['f1']:.4f}")
    print(f"[agent1]   Precision = {metrics['precision']:.4f}")
    print(f"[agent1]   Recall    = {metrics['recall']:.4f}")
    print(f"[agent1]   PR-AUC    = {metrics['pr_auc']:.4f}")
    print(f"[agent1]   Confusion matrix (rows=true, cols=pred, order=[normal, attack]):")
    cm = metrics["confusion_matrix"]
    print(f"[agent1]     [[{cm[0][0]:4d}, {cm[0][1]:4d}],")
    print(f"[agent1]      [{cm[1][0]:4d}, {cm[1][1]:4d}]]")


if __name__ == "__main__":
    main()
