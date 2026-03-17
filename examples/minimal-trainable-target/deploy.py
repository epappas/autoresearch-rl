#!/usr/bin/env python3
"""Deploy minimal-trainable-target to Basilica GPU cloud."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
CONFIG = DIR / "config.yaml"
DOCKERFILE = DIR / "Dockerfile"


def main() -> int:
    p = argparse.ArgumentParser(description="Deploy minimal-trainable-target to Basilica")
    p.add_argument("--image-tag", default=None, help="Docker image tag to build and push")
    p.add_argument("--skip-build", action="store_true", help="Skip docker build/push")
    p.add_argument("overrides", nargs="*", help="Extra --override key=value pairs")
    args = p.parse_args()

    if not os.environ.get("BASILICA_API_TOKEN"):
        print("ERROR: BASILICA_API_TOKEN not set")
        print("  export BASILICA_API_TOKEN='your-token'")
        return 1

    if args.image_tag and not args.skip_build:
        print(f"Building Docker image: {args.image_tag}")
        ret = subprocess.run(
            ["docker", "build", "-t", args.image_tag, "-f", str(DOCKERFILE), str(DIR)],
            check=False,
        )
        if ret.returncode != 0:
            return ret.returncode

        print(f"Pushing Docker image: {args.image_tag}")
        ret = subprocess.run(["docker", "push", args.image_tag], check=False)
        if ret.returncode != 0:
            return ret.returncode

    cmd = ["uv", "run", "autoresearch-rl", "--config", str(CONFIG)]
    cmd += ["--override", "target.type=basilica"]
    cmd += ["--override", 'target.train_cmd=["python3","/app/train.py"]']
    if args.image_tag:
        cmd += ["--override", f"target.basilica.image={args.image_tag}"]
    for ov in args.overrides:
        cmd += ["--override", ov]

    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
