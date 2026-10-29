"""
Lightweight experiment tracker. One JSON per run under reports/phase03/experiments/.

MLflow is overkill for a capstone and adds a 100 MB dependency. This gives us:
  - params + metrics logged per run
  - timestamped files for an audit trail
  - a consolidated runs_summary.json for quick comparison

Compatible with CI: no DB, no server, just files.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class ExperimentRun:
    def __init__(self, name: str, out_dir: str | Path = "reports/phase03/experiments"):
        self.name = name
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = time.time()
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self.artifacts: list[str] = []

    def log_params(self, **kwargs) -> None:
        self.params.update(kwargs)

    def log_metrics(self, **kwargs) -> None:
        self.metrics.update({k: float(v) if hasattr(v, "__float__") else v for k, v in kwargs.items()})

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(str(path))

    def save(self) -> Path:
        duration = time.time() - self.started_at
        out = {
            "name": self.name,
            "started_at": self.started_at,
            "duration_s": round(duration, 3),
            "params": self.params,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }
        path = self.out_dir / f"{int(self.started_at)}_{self.name}.json"
        path.write_text(json.dumps(out, indent=2, default=str))
        return path


def consolidate(out_dir: str | Path = "reports/phase03/experiments") -> Path:
    """Walk every run JSON and emit a flat runs_summary.json."""
    out_dir = Path(out_dir)
    runs = []
    for p in sorted(out_dir.glob("*.json")):
        if p.name == "runs_summary.json":
            continue
        try:
            runs.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    summary_path = out_dir / "runs_summary.json"
    summary_path.write_text(json.dumps(runs, indent=2, default=str))
    return summary_path
