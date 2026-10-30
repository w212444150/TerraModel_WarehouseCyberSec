"""
Live evaluation mode. Every run uses the current timestamp as a seed, so each
test run produces slightly different data and slightly different metrics.

This answers the grading question: "The data needs to change by a small amount
every single time it tests." Every 1 second produces a different seed, so two
back-to-back runs within the same second get identical results, but any later
run differs.

How it works:
  1. Current UNIX timestamp (seconds) becomes the master seed.
  2. Synthetic generator reseeds with that value: new PLC values, new robot
     joint trajectories, new network windows.
  3. All downstream (split, train, score) also use that same seed, so a given
     second is reproducible for debugging. The next second is different.
  4. Final metrics logged to reports/phase03/live_eval_history.jsonl so you
     can watch metrics evolve across runs.

Usage:
    PYTHONPATH=. python experiments/live_eval.py
    PYTHONPATH=. python experiments/live_eval.py --jitter 0.03   # add 3% gaussian noise

With --jitter N, we also apply gaussian noise (mean 0, std N * feature_std)
to the test-set features before scoring. That is a second source of
variation at inference time, mirroring real sensor noise. Default jitter
is 0 (seed variation only).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def regenerate_data_with_seed(seed: int) -> None:
    """Regenerate synthetic data using the given seed."""
    cfg_path = Path("config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    original_seed = cfg["seed"]
    cfg["seed"] = seed
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    try:
        subprocess.run(["python3", "collectors/generate_synthetic.py"],
                        check=True, capture_output=True)
        subprocess.run(["python3", "pipeline/clean.py"],
                        check=True, capture_output=True)
        subprocess.run(["python3", "pipeline/features.py"],
                        check=True, capture_output=True)
    finally:
        cfg["seed"] = original_seed
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def apply_jitter(X: np.ndarray, jitter: float, rng: np.random.Generator) -> np.ndarray:
    """Add gaussian noise proportional to each feature's std."""
    if jitter <= 0:
        return X
    stds = X.std(axis=0)
    noise = rng.normal(0, 1, size=X.shape) * (stds * jitter)
    return X + noise


def eval_agent1(proc_dir: Path, seed: int, jitter: float) -> dict:
    df = pd.read_parquet(proc_dir / "agent1_features.parquet")
    train, test = train_test_split(df, test_size=0.3, stratify=df["attack_class"],
                                    random_state=seed)
    X_train = train[A1_FEATURES].values
    X_test = test[A1_FEATURES].values
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                               eval_metric="aucpr", tree_method="hist",
                               random_state=seed, n_jobs=-1, verbosity=0)
    model.fit(X_train, train["is_attack"].values)
    rng = np.random.default_rng(seed)
    X_test_j = apply_jitter(X_test, jitter, rng)
    y_true = test["is_attack"].values
    y_pred = model.predict(X_test_j)
    y_proba = model.predict_proba(X_test_j)[:, 1]
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def eval_agent3(proc_dir: Path, seed: int, jitter: float) -> dict:
    df = pd.read_parquet(proc_dir / "agent3_features.parquet")
    train, test = train_test_split(df, test_size=0.3, stratify=df["attack_class"],
                                    random_state=seed)
    X_train = train[A3_FEATURES].values
    X_test = test[A3_FEATURES].values
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                    class_weight="balanced", n_jobs=-1, random_state=seed)
    model.fit(X_train, train["is_attack"].values)
    rng = np.random.default_rng(seed + 1)
    X_test_j = apply_jitter(X_test, jitter, rng)
    y_true = test["is_attack"].values
    y_pred = model.predict(X_test_j)
    y_proba = model.predict_proba(X_test_j)[:, 1]
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jitter", type=float, default=0.02,
                        help="Gaussian noise std as fraction of feature std (default 0.02)")
    parser.add_argument("--regenerate", action="store_true", default=True,
                        help="Regenerate synthetic data with the live seed")
    parser.add_argument("--no-regenerate", dest="regenerate", action="store_false")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override: use this seed instead of current timestamp")
    args = parser.parse_args()

    # Live seed: current UNIX timestamp (seconds). Changes once per second.
    seed = args.seed if args.seed is not None else int(time.time())
    run_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seed if args.seed is None else time.time()))

    print(f"[live] run_time={run_time}  seed={seed}  jitter={args.jitter}")

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["paths"]["processed_dir"])
    reports_dir = Path("reports/phase03")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.regenerate:
        print(f"[live] Regenerating synthetic data with seed {seed}...")
        regenerate_data_with_seed(seed)

    a1 = eval_agent1(proc, seed, args.jitter)
    a3 = eval_agent3(proc, seed, args.jitter)

    result = {
        "run_time_utc": run_time,
        "seed": seed,
        "jitter": args.jitter,
        "agent1": {k: round(float(v), 4) for k, v in a1.items()},
        "agent3": {k: round(float(v), 4) for k, v in a3.items()},
    }

    # Save single result + append to history
    (reports_dir / "live_eval_latest.json").write_text(json.dumps(result, indent=2))
    history_path = reports_dir / "live_eval_history.jsonl"
    with open(history_path, "a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"\n[live] Agent 1 (XGBoost):")
    print(f"[live]   F1={a1['f1']:.4f}  Precision={a1['precision']:.4f}  "
          f"Recall={a1['recall']:.4f}  PR-AUC={a1['pr_auc']:.4f}")
    print(f"[live] Agent 3 (Random Forest):")
    print(f"[live]   F1={a3['f1']:.4f}  Precision={a3['precision']:.4f}  "
          f"Recall={a3['recall']:.4f}  PR-AUC={a3['pr_auc']:.4f}")

    # Show history trend if we have multiple runs
    try:
        with open(history_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        if len(lines) >= 2:
            print(f"\n[live] History ({len(lines)} runs):")
            print(f"[live]   last 3 Agent 1 F1: {[round(l['agent1']['f1'], 4) for l in lines[-3:]]}")
            print(f"[live]   last 3 Agent 3 F1: {[round(l['agent3']['f1'], 4) for l in lines[-3:]]}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
