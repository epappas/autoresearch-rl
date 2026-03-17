#!/usr/bin/env python3
"""Build and push a Docker image, then run basilica-grpo on Basilica cloud."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
CONFIG = DIR / "config.yaml"
DOCKERFILE = DIR / "Dockerfile"
DEFAULT_IMAGE = "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"


def main() -> int:
    p = argparse.ArgumentParser(description="Deploy basilica-grpo to Basilica cloud")
    p.add_argument(
        "--policy",
        choices=["llm", "grid"],
        default=None,
        help="Override policy type (default: llm from config.yaml)",
    )
    p.add_argument("--image-tag", default=None, help="Custom Docker image tag to build and push")
    p.add_argument("--skip-build", action="store_true", help="Skip docker build/push")
    p.add_argument("overrides", nargs="*", help="Extra --override key=value pairs")
    args = p.parse_args()

    if not os.environ.get("BASILICA_API_TOKEN"):
        print("ERROR: BASILICA_API_TOKEN not set")
        print("  export BASILICA_API_TOKEN='your-token'")
        return 1

    image_tag = args.image_tag or DEFAULT_IMAGE

    if args.image_tag and not args.skip_build:
        print(f"Building Docker image: {image_tag}")
        ret = subprocess.run(
            ["docker", "build", "-t", image_tag, "-f", str(DOCKERFILE), str(DIR)],
            check=False,
        )
        if ret.returncode != 0:
            return ret.returncode

        print(f"Pushing Docker image: {image_tag}")
        ret = subprocess.run(["docker", "push", image_tag], check=False)
        if ret.returncode != 0:
            return ret.returncode

    cmd = ["uv", "run", "autoresearch-rl", "--config", str(CONFIG)]
    if args.image_tag:
        cmd += ["--override", f"target.basilica.image={image_tag}"]
    if args.policy:
        cmd += ["--override", f"policy.type={args.policy}"]
    for ov in args.overrides:
        cmd += ["--override", ov]

    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
