from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def write_manifest(base_dir: str, payload: dict) -> Path:
    run_id = payload.get("run_id") or new_run_id()
    out = Path(base_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "ts": int(time.time()),
        "episode_id": payload.get("episode_id", run_id),
        **payload,
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
