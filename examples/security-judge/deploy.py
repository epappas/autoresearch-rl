#!/usr/bin/env python3
"""Deploy security-judge to Basilica GPU cloud."""
from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
CONFIG = DIR / "config.yaml"
REPO_ROOT = DIR.parent.parent
DEFAULT_IMAGE = "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"

INJECT_FILES = {
    "/app/train.py": DIR / "train.py",
    "/app/prepare.py": DIR / "prepare.py",
}


def _build_file_injection_cmd() -> str:
    """Inject only the scripts. Data is downloaded by prepare.py from HuggingFace."""
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
    file_inject = _build_file_injection_cmd()
    pip_install = (
        "pip install --no-cache-dir "
        "transformers==4.47.1 datasets==3.2.0 accelerate==0.34.2 peft==0.13.2 scipy"
    )
    model_cache = (
        "python3 -c \""
        "from transformers import AutoModelForCausalLM, AutoTokenizer; "
        "AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); "
        "AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')\""
    )
    return f"{file_inject} && {pip_install} && {model_cache}"


def main() -> int:
    p = argparse.ArgumentParser(description="Deploy security-judge to Basilica")
    p.add_argument(
        "--policy", choices=["hybrid", "llm", "llm_diff", "grid"],
        default=None, help="Override policy type",
    )
    p.add_argument("overrides", nargs="*", help="Extra --override key=value pairs")
    args = p.parse_args()

    if not os.environ.get("BASILICA_API_TOKEN"):
        print("ERROR: BASILICA_API_TOKEN not set")
        return 1

    setup_cmd = _build_setup_cmd()

    cmd = ["uv", "run", "autoresearch-rl", "run", str(CONFIG)]
    cmd += ["--override", f"target.basilica.setup_cmd={setup_cmd}"]
    if args.policy:
        cmd += ["--override", f"policy.type={args.policy}"]
    for ov in args.overrides:
        cmd += ["--override", ov]

    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
