from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import yaml

from autoresearch_rl.config import RunConfig
from autoresearch_rl.controller.continuous import run_continuous
from autoresearch_rl.target.registry import build_target

app = typer.Typer(add_completion=False)


def _apply_override(cfg: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Invalid override: {override}")
    key, raw = override.split("=", 1)
    parts = key.split(".")
    cursor = cfg
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    cursor[parts[-1]] = value


def _load_config(config: str, overrides: list[str]) -> RunConfig:
    cfg_path = Path(config)
    cfg_data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    for ov in overrides:
        _apply_override(cfg_data, ov)
    return RunConfig.model_validate(cfg_data)


def _run(config: str, overrides: list[str], seed: int | None) -> None:
    cfg = _load_config(config, overrides)
    if seed is not None:
        cfg.controller.seed = seed

    target = build_target(cfg.target)
    result = run_continuous(
        target=target,
        objective=cfg.objective,
        controller=cfg.controller,
        telemetry=cfg.telemetry,
        policy_cfg=cfg.policy,
        comparability_cfg=cfg.comparability,
    )

    typer.echo(
        json.dumps(
            {"iterations": result.iterations, "best_value": result.best_value, "best_score": result.best_score},
            indent=2,
        )
    )


@app.command()
def run(
    config: str = typer.Option(..., "--config"),
    override: list[str] = typer.Option([], "--override"),
    seed: int | None = typer.Option(None, "--seed"),
) -> None:
    """Continuous autoresearch RL run (always on)."""
    _run(config, override, seed)


@app.command()
def validate(config: str = typer.Option(..., "--config")) -> None:
    cfg = _load_config(config, [])
    build_target(cfg.target)
    typer.echo("OK")


@app.command()
def print_config(config: str = typer.Option(..., "--config")) -> None:
    cfg = _load_config(config, [])
    typer.echo(cfg.model_dump_json(indent=2))


@app.callback(invoke_without_command=True)
def main(
    config: str = typer.Option(None, "--config"),
    override: list[str] = typer.Option([], "--override"),
    seed: int | None = typer.Option(None, "--seed"),
) -> None:
    """Default command: run continuously when --config is provided."""
    if config:
        _run(config, override, seed)


if __name__ == "__main__":
    app()
