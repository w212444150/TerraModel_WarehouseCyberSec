"""
Agent 3 baseline: Mahalanobis distance threshold on robot integrity features.

Fits a Mahalanobis distance on NORMAL training windows (unsupervised). Flags any
test window with distance above the given percentile of the training distribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

FEATURE_COLS = [
    "mean_dev", "max_dev", "std_dev", "torque_var",
    "timing_jitter", "nonprimary_src_frac", "cmd_entropy",
]


def main() -> None:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(proc / "agent3_train.parquet")
    test = pd.read_parquet(proc / "agent3_test.parquet")

    train_normal = train[train["attack_class"] == "normal"]
    X_train = train_normal[FEATURE_COLS].values
    X_test = test[FEATURE_COLS].values
    y_test = test["is_attack"].values

    # Fit covariance on normal
    cov = EmpiricalCovariance().fit(X_train)
    mean = X_train.mean(axis=0)
    inv_cov = cov.precision_

    # Distances
    def md(x):
        return mahalanobis(x, mean, inv_cov)

    train_distances = np.array([md(x) for x in X_train])
    test_distances = np.array([md(x) for x in X_test])

    threshold = np.percentile(train_distances, cfg["agent3"]["threshold_percentile"])
    y_pred = (test_distances > threshold).astype(int)

    metrics = {
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, test_distances)),
        "threshold": float(threshold),
        "threshold_percentile": cfg["agent3"]["threshold_percentile"],
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_test": int(len(y_test)),
        "n_attack_test": int(y_test.sum()),
    }

    joblib.dump({"mean": mean, "inv_cov": inv_cov, "threshold": threshold,
                 "features": FEATURE_COLS},
                models_dir / "agent3_mahalanobis.joblib")
    (reports_dir / "agent3_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("[agent3] Mahalanobis baseline on test set:")
    print(f"[agent3]   F1        = {metrics['f1']:.4f}")
    print(f"[agent3]   Precision = {metrics['precision']:.4f}")
    print(f"[agent3]   Recall    = {metrics['recall']:.4f}")
    print(f"[agent3]   PR-AUC    = {metrics['pr_auc']:.4f}")
    print(f"[agent3]   Threshold = {metrics['threshold']:.3f} (p{metrics['threshold_percentile']})")
    cm = metrics["confusion_matrix"]
    print(f"[agent3]   Confusion matrix (rows=true, cols=pred, order=[normal, attack]):")
    print(f"[agent3]     [[{cm[0][0]:5d}, {cm[0][1]:5d}],")
    print(f"[agent3]      [{cm[1][0]:5d}, {cm[1][1]:5d}]]")


if __name__ == "__main__":
    main()
