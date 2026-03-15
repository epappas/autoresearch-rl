from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Protocol


class ExperimentTracker(Protocol):
    """Protocol for experiment tracking backends."""

    def log_params(self, params: dict[str, object]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...
    def log_artifact(self, path: str, name: str) -> None: ...
    def set_status(self, status: str) -> None: ...


class LocalFileTracker:
    """File-based experiment tracker that writes to a structured directory.

    Directory layout:
        base_dir/run_id/
            params.json      -- merged parameter snapshots
            metrics.jsonl     -- one JSON object per step
            artifacts/        -- copied artifact files
            status.json       -- current run status
    """

    def __init__(self, base_dir: str, run_id: str) -> None:
        self._run_dir = Path(base_dir) / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts_dir = self._run_dir / "artifacts"
        self._artifacts_dir.mkdir(exist_ok=True)
        self._params_path = self._run_dir / "params.json"
        self._metrics_path = self._run_dir / "metrics.jsonl"
        self._status_path = self._run_dir / "status.json"

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def log_params(self, params: dict[str, object]) -> None:
        """Merge *params* into the existing params.json (create if absent)."""
        existing: dict[str, object] = {}
        if self._params_path.exists():
            existing = json.loads(
                self._params_path.read_text(encoding="utf-8")
            )
        existing.update(params)
        self._params_path.write_text(
            json.dumps(existing, indent=2), encoding="utf-8"
        )

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Append a timestamped metrics record to metrics.jsonl."""
        record = {"step": step, "ts": time.time(), **metrics}
        with self._metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def log_artifact(self, path: str, name: str) -> None:
        """Copy the file at *path* into the artifacts directory as *name*."""
        src = Path(path)
        if not src.is_file():
            raise FileNotFoundError(f"artifact source not found: {path}")
        dest = self._artifacts_dir / name
        shutil.copy2(str(src), str(dest))

    def set_status(self, status: str) -> None:
        """Write (overwrite) the current run status."""
        payload = {"status": status, "ts": time.time()}
        self._status_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
