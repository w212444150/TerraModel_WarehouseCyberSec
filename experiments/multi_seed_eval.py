"""
Multi-seed evaluation: train and score each agent on N different random seeds,
report mean +/- std per metric.

Why this exists. Our main pipeline pins random_state to 42 so that any grader
can reproduce the exact numbers in the Phase 03 report. That is good for
auditability but bad for improvement detection: if we change a feature or
model, the single-seed metric cannot tell us whether the change truly helped
or whether seed 42 just happens to be favorable.

This script runs the full Phase 03 Agent 1 (XGBoost), Agent 2 (Logistic on KEV),
and Agent 3 (Random Forest) training N times with N different seeds for the
train/val/test split AND the model init. It reports mean metrics and
standard deviations.

Usage:
    PYTHONPATH=. python experiments/multi_seed_eval.py --n-seeds 10
    PYTHONPATH=. python experiments/multi_seed_eval.py --n-seeds 10 --fresh-data

With --fresh-data, the synthetic generator also re-seeds, producing a fresh
dataset per fold. Without it, the dataset stays fixed and only split + model
seeds vary (standard CV behavior).

Output: reports/phase03/multi_seed_summary.json
"""
from __future__ import annotations

import argparse
import json
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


def regenerate_data(seed: int) -> None:
    """Re-run the synthetic generator with a new seed, then clean+features+split."""
    import yaml
    cfg_path = Path("config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg_original_seed = cfg["seed"]
    cfg["seed"] = seed
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    try:
        import subprocess
        subprocess.run(["python3", "collectors/generate_synthetic.py"], check=True,
                        capture_output=True)
        subprocess.run(["python3", "pipeline/clean.py"], check=True, capture_output=True)
        subprocess.run(["python3", "pipeline/features.py"], check=True, capture_output=True)
    finally:
        cfg["seed"] = cfg_original_seed
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def eval_agent1(proc_dir: Path, seed: int) -> dict:
    df = pd.read_parquet(proc_dir / "agent1_features.parquet")
    train, test = train_test_split(df, test_size=0.3, stratify=df["attack_class"],
                                    random_state=seed)
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                               eval_metric="aucpr", tree_method="hist",
                               random_state=seed, n_jobs=-1, verbosity=0)
    model.fit(train[A1_FEATURES].values, train["is_attack"].values)
    y_true = test["is_attack"].values
    y_pred = model.predict(test[A1_FEATURES].values)
    y_proba = model.predict_proba(test[A1_FEATURES].values)[:, 1]
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def eval_agent3(proc_dir: Path, seed: int) -> dict:
    df = pd.read_parquet(proc_dir / "agent3_features.parquet")
    train, test = train_test_split(df, test_size=0.3, stratify=df["attack_class"],
                                    random_state=seed)
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                    class_weight="balanced", n_jobs=-1, random_state=seed)
    model.fit(train[A3_FEATURES].values, train["is_attack"].values)
    y_true = test["is_attack"].values
    y_pred = model.predict(test[A3_FEATURES].values)
    y_proba = model.predict_proba(test[A3_FEATURES].values)[:, 1]
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def eval_agent2(proc_dir: Path, kev_path: Path, seed: int) -> dict:
    from agents.agent2_assessment.phase03_kev import join_advisories_to_kev, FEATURE_COLS
    adv = pd.read_csv(proc_dir / "agent2_features.csv")
    kev = pd.read_csv(kev_path)
    joined = join_advisories_to_kev(adv, kev)
    X = joined[FEATURE_COLS].values
    y = joined["is_kev_match"].values
    if y.sum() < 2 or (len(y) - y.sum()) < 2:
        return {"precision_at_10": float(joined["is_kev_match"].head(10).mean())}
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    model.fit(Xs, y)
    joined["score"] = model.predict_proba(Xs)[:, 1]
    top10 = joined.nlargest(10, "score")
    return {"precision_at_10": float(top10["is_kev_match"].mean())}


def summarize(runs: list[dict]) -> dict:
    """Compute mean and std per metric key across runs."""
    keys = set()
    for r in runs:
        keys.update(r.keys())
    out = {}
    for k in sorted(keys):
        vals = [r[k] for r in runs if k in r]
        out[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
            "n": len(vals),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Number of seeds to evaluate (default 10)")
    parser.add_argument("--fresh-data", action="store_true",
                        help="Regenerate synthetic data per seed (slow but true variance)")
    parser.add_argument("--base-seed", type=int, default=100,
                        help="Starting seed, varied by iteration (default 100)")
    args = parser.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["paths"]["processed_dir"])
    reports_dir = Path("reports/phase03")
    reports_dir.mkdir(parents=True, exist_ok=True)
    kev_path = Path(cfg["paths"]["threat_dir"]) / "kev" / "kev_flat.csv"

    print(f"[multi-seed] Running {args.n_seeds} seeds "
          f"({'fresh data per seed' if args.fresh_data else 'fixed data, varied splits + models'})")

    a1_runs, a2_runs, a3_runs = [], [], []
    for i in range(args.n_seeds):
        seed = args.base_seed + i
        if args.fresh_data:
            print(f"[multi-seed] Seed {seed}: regenerating data...")
            regenerate_data(seed)

        a1 = eval_agent1(proc, seed)
        a3 = eval_agent3(proc, seed)
        a2 = eval_agent2(proc, kev_path, seed) if kev_path.exists() else {}
        a1_runs.append(a1); a2_runs.append(a2); a3_runs.append(a3)

        print(f"[multi-seed] seed={seed}  A1 F1={a1['f1']:.4f}  A3 F1={a3['f1']:.4f}  "
              f"A2 P@10={a2.get('precision_at_10', 'n/a'):.4f}")

    summary = {
        "n_seeds": args.n_seeds,
        "base_seed": args.base_seed,
        "fresh_data_per_seed": args.fresh_data,
        "agent1": summarize(a1_runs),
        "agent2": summarize(a2_runs),
        "agent3": summarize(a3_runs),
    }
    out_path = reports_dir / "multi_seed_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[multi-seed] Wrote {out_path}")

    print(f"\n[multi-seed] AGENT 1 (XGBoost, {args.n_seeds} seeds):")
    for metric, stats in summary["agent1"].items():
        print(f"             {metric:10s}: {stats['mean']:.4f} +/- {stats['std']:.4f}  "
              f"(min {stats['min']:.4f}, max {stats['max']:.4f})")
    print(f"\n[multi-seed] AGENT 3 (Random Forest, {args.n_seeds} seeds):")
    for metric, stats in summary["agent3"].items():
        print(f"             {metric:10s}: {stats['mean']:.4f} +/- {stats['std']:.4f}  "
              f"(min {stats['min']:.4f}, max {stats['max']:.4f})")
    if summary["agent2"]:
        print(f"\n[multi-seed] AGENT 2 (Logistic on KEV, {args.n_seeds} seeds):")
        for metric, stats in summary["agent2"].items():
            print(f"             {metric:16s}: {stats['mean']:.4f} +/- {stats['std']:.4f}")


if __name__ == "__main__":
    main()
