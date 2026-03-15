from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass


@dataclass
class LoopCheckpoint:
    episode_id: str
    iteration: int
    best_score: float
    best_value: float | None
    no_improve_streak: int
    history: list[dict]
    recent_statuses: list[str]
    policy_state: dict
    elapsed_s: float
    timestamp: float


def save_checkpoint(path: str, checkpoint: LoopCheckpoint) -> None:
    """Save loop state to JSON file atomically (write to temp, then rename)."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(asdict(checkpoint), f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def save_policy_snapshot(base_dir: str, version: int, weights: dict) -> str:
    """Save policy weights as a versioned snapshot. Returns snapshot path."""
    path = os.path.join(base_dir, f"policy_v{version:04d}.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": version, "weights": weights}, f)
    return path


def load_policy_snapshot(base_dir: str, version: int) -> dict | None:
    """Load policy snapshot by version. Returns None if not found."""
    path = os.path.join(base_dir, f"policy_v{version:04d}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["weights"]


def get_latest_snapshot_version(base_dir: str) -> int:
    """Get the highest version number of saved snapshots. Returns -1 if none."""
    if not os.path.isdir(base_dir):
        return -1
    versions = []
    for name in os.listdir(base_dir):
        if name.startswith("policy_v") and name.endswith(".json"):
            try:
                v = int(name[8:12])
                versions.append(v)
            except ValueError:
                pass
    return max(versions) if versions else -1


def load_checkpoint(path: str) -> LoopCheckpoint | None:
    """Load checkpoint from file. Returns None if not found."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return LoopCheckpoint(
        episode_id=data["episode_id"],
        iteration=data["iteration"],
        best_score=data["best_score"],
        best_value=data["best_value"],
        no_improve_streak=data["no_improve_streak"],
        history=data["history"],
        recent_statuses=data["recent_statuses"],
        policy_state=data["policy_state"],
        elapsed_s=data["elapsed_s"],
        timestamp=data["timestamp"],
    )
