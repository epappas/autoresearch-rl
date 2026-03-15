#!/usr/bin/env python3
"""Run autoresearch-rl with Basilica GRPO target.

This script demonstrates the full integration:
1. Loads config from example.yaml
2. Creates a BasilicaTarget that deploys training jobs on Basilica GPU cloud
3. Runs the autoresearch-rl loop with grid search over hyperparameters
4. Each iteration: deploys Qwen2.5-0.5B GRPO training on a GPU, parses metrics

Prerequisites:
    export BASILICA_API_TOKEN="your-token"
    export HF_TOKEN="your-hf-token"

Usage:
    python3 examples/basilica-grpo/run.py
    # or via CLI:
    uv run autoresearch-rl --config examples/basilica-grpo/example.yaml
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from autoresearch_rl.config import RunConfig


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if not os.environ.get("BASILICA_API_TOKEN"):
        print("ERROR: BASILICA_API_TOKEN not set")
        print("  export BASILICA_API_TOKEN='your-token'")
        sys.exit(1)

    config_path = Path(__file__).parent / "example.yaml"
    import yaml
    raw = yaml.safe_load(config_path.read_text())
    cfg = RunConfig.model_validate(raw)

    from autoresearch_rl.controller.continuous import run_continuous
    from autoresearch_rl.target.registry import build_target
    from autoresearch_rl.policy.search import (
        GridPolicy, RandomPolicy, StaticPolicy,
    )

    target = build_target(cfg.target)

    if cfg.policy.type == "grid":
        policy = GridPolicy(cfg.policy.params)
    elif cfg.policy.type == "random":
        policy = RandomPolicy(cfg.policy.params, seed=cfg.policy.seed)
    else:
        policy = StaticPolicy()

    print("=" * 60)
    print("AutoResearch-RL x Basilica GRPO")
    print("=" * 60)
    print(f"Model:     Qwen/Qwen2.5-0.5B-Instruct")
    print(f"Task:      GSM8K (math reasoning)")
    print(f"Policy:    {cfg.policy.type}")
    print(f"GPU:       {cfg.target.basilica.gpu_count}x {cfg.target.basilica.gpu_models}")
    print(f"Timeout:   {cfg.target.timeout_s}s per iteration")
    print(f"Wall time: {cfg.controller.max_wall_time_s}s total")
    print("=" * 60)

    result = run_continuous(
        target=target,
        objective=cfg.objective,
        controller=cfg.controller,
        telemetry=cfg.telemetry,
        policy_cfg=cfg.policy,
        comparability_cfg=cfg.comparability,
    )

    print()
    print("=" * 60)
    print(f"Done: {result.iterations} iterations")
    print(f"Best val_bpb: {result.best_value}")
    print(f"Best score:   {result.best_score}")
    print("=" * 60)


if __name__ == "__main__":
    main()
