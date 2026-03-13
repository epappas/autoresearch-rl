from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from autoresearch_rl.config import RunConfig
from autoresearch_rl.controller.continuous import run_continuous
from autoresearch_rl.target.registry import build_target

app = typer.Typer(add_completion=False)


def _run(config: str) -> None:
    cfg_path = Path(config)
    cfg_data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = RunConfig.model_validate(cfg_data)

    target = build_target(cfg.target)
    result = run_continuous(
        target=target,
        objective=cfg.objective,
        controller=cfg.controller,
        telemetry=cfg.telemetry,
        policy_cfg=cfg.policy,
    )

    typer.echo(json.dumps({"iterations": result.iterations, "best_score": result.best_score}, indent=2))


@app.command()
def run(config: str = typer.Option(..., "--config")) -> None:
    """Continuous autoresearch RL run (always on)."""
    _run(config)


@app.callback(invoke_without_command=True)
def main(config: str = typer.Option(None, "--config")) -> None:
    """Default command: run continuously when --config is provided."""
    if config:
        _run(config)


if __name__ == "__main__":
    app()
