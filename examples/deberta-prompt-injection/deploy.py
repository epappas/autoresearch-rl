#!/usr/bin/env python3
"""Deploy deberta-prompt-injection to Basilica GPU cloud."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
CONFIG = DIR / "config.yaml"


def main() -> int:
    p = argparse.ArgumentParser(description="Deploy deberta-prompt-injection to Basilica")
    p.add_argument(
        "--policy",
        choices=["hybrid", "llm_diff", "llm", "grid"],
        default=None,
        help="Override policy type (default: hybrid from config.yaml)",
    )
    p.add_argument("overrides", nargs="*", help="Extra --override key=value pairs")
    args = p.parse_args()

    if not os.environ.get("BASILICA_API_TOKEN"):
        print("ERROR: BASILICA_API_TOKEN not set")
        print("  export BASILICA_API_TOKEN='your-token'")
        return 1

    cmd = ["uv", "run", "autoresearch-rl", "--config", str(CONFIG)]
    cmd += ["--override", "target.type=basilica"]
    cmd += ["--override", "target.train_cmd=[\"python3\",\"/app/train.py\","
            "\"--train-file\",\"/app/data/train.jsonl\","
            "\"--val-file\",\"/app/data/val.jsonl\","
            "\"--output-dir\",\"/tmp/deberta-out\"]"]
    cmd += ["--override", "target.timeout_s=1200"]
    cmd += ["--override", "controller.max_wall_time_s=7200"]
    cmd += ["--override", "comparability.expected_budget_s=7200"]
    cmd += ["--override", "target.basilica.ttl_seconds=1200"]
    if args.policy:
        cmd += ["--override", f"policy.type={args.policy}"]
    for ov in args.overrides:
        cmd += ["--override", ov]

    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
