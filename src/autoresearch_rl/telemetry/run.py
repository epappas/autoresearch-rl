from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from autoresearch_rl.telemetry.comparability import hardware_fingerprint


def _git_info(cwd: str | None = None) -> dict[str, str]:
    def _run(args: list[str]) -> str:
        cp = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
        return (cp.stdout or "").strip()

    return {
        "commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "status": _run(["git", "status", "--porcelain"]),
        "diff": _run(["git", "diff"]),
    }


def write_run_manifest(path: str, config: dict, run_id: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "ts": int(time.time()),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
            "AR_SEED": os.environ.get("AR_SEED", ""),
        },
        "hardware_fingerprint": hardware_fingerprint(),
        "git": _git_info(),
        "config": config,
    }

    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p
