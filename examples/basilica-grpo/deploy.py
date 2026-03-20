#!/usr/bin/env python3
"""Build and push a Docker image, then run basilica-grpo on Basilica cloud."""
from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
CONFIG = DIR / "config.yaml"
DOCKERFILE = DIR / "Dockerfile"
DEFAULT_IMAGE = "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime"

# Files to inject into the Basilica container via setup_cmd
INJECT_FILES = {
    "/app/train.py": DIR / "train.py",
    "/app/prepare.py": DIR / "prepare.py",
}


def _build_file_injection_cmd() -> str:
    """Generate a shell command that writes local files into the container."""
    parts: list[str] = ["mkdir -p /app"]
    for dest, src in INJECT_FILES.items():
        content = src.read_text(encoding="utf-8")
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        parts.append(
            f"python3 -c \"import base64; "
            f"open('{dest}','w').write(base64.b64decode('{encoded}').decode('utf-8'))\""
        )
    return " && ".join(parts)


def _build_setup_cmd() -> str:
    """Build the full setup command: inject files, install deps, cache model."""
    file_inject = _build_file_injection_cmd()
    pip_install = (
        "pip install --no-cache-dir "
        "transformers==4.47.1 datasets==3.2.0 trl==0.14.0 accelerate==0.34.2 scipy"
    )
    model_cache = (
        "python3 -c \""
        "from transformers import AutoModelForCausalLM, AutoTokenizer; "
        "AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); "
        "AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')\""
    )
    return f"{file_inject} && {pip_install} && {model_cache}"


def main() -> int:
    p = argparse.ArgumentParser(description="Deploy basilica-grpo to Basilica cloud")
    p.add_argument(
        "--policy",
        choices=["llm", "hybrid", "grid"],
        default=None,
        help="Override policy type (default: hybrid from config.yaml)",
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

    # Build setup_cmd that injects files + installs deps + caches model
    setup_cmd = _build_setup_cmd()

    cmd = ["uv", "run", "autoresearch-rl", "--config", str(CONFIG)]
    if args.image_tag:
        cmd += ["--override", f"target.basilica.image={image_tag}"]
    else:
        # When not using a custom image, inject files via setup_cmd
        cmd += ["--override", f"target.basilica.setup_cmd={setup_cmd}"]
    if args.policy:
        cmd += ["--override", f"policy.type={args.policy}"]
    for ov in args.overrides:
        cmd += ["--override", ov]

    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
