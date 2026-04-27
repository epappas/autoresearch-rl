from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer
import yaml

from autoresearch_rl.config import RunConfig
from autoresearch_rl.config_validate import has_blocking_errors, validate_runtime
from autoresearch_rl.controller.continuous import (
    _diff_extractor,
    _param_extractor,
    run_continuous,
)
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
    try:
        cfg_path = Path(config)
        cfg_data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise typer.BadParameter(f"Invalid config: {exc}")
    for ov in overrides:
        _apply_override(cfg_data, ov)
    return RunConfig.model_validate(cfg_data)


def _emit_validation_errors(errors: list, *, raise_on_block: bool = True) -> None:
    for err in errors:
        typer.echo(err.format(), err=(err.severity == "error"))
    if raise_on_block and has_blocking_errors(errors):
        raise typer.Exit(code=2)


def _run(config: str, overrides: list[str], seed: int | None) -> None:
    cfg = _load_config(config, overrides)
    _emit_validation_errors(validate_runtime(cfg))
    if seed is not None:
        cfg.controller.seed = seed

    program = ""
    if cfg.program_path:
        program = Path(cfg.program_path).read_text(encoding="utf-8")

    target = build_target(cfg.target)
    result = run_continuous(
        target=target,
        objective=cfg.objective,
        controller=cfg.controller,
        telemetry=cfg.telemetry,
        policy_cfg=cfg.policy,
        comparability_cfg=cfg.comparability,
        program=program,
    )

    typer.echo(
        json.dumps(
            {"iterations": result.iterations, "best_value": result.best_value, "best_score": result.best_score},
            indent=2,
        )
    )


@app.command()
def run(
    config: str = typer.Argument(..., help="Path to config.yaml"),
    override: list[str] = typer.Option([], "--override"),
    seed: int | None = typer.Option(None, "--seed"),
) -> None:
    """Continuous autoresearch RL run (always on)."""
    _run(config, override, seed)


@app.command()
def validate(config: str = typer.Argument(..., help="Path to config.yaml")) -> None:
    cfg = _load_config(config, [])
    errors = validate_runtime(cfg)
    _emit_validation_errors(errors, raise_on_block=False)
    if has_blocking_errors(errors):
        raise typer.Exit(code=2)
    build_target(cfg.target)
    typer.echo("OK")


@app.command()
def print_config(config: str = typer.Argument(..., help="Path to config.yaml")) -> None:
    cfg = _load_config(config, [])
    typer.echo(cfg.model_dump_json(indent=2))


@app.command()
def status(
    config: str = typer.Argument(..., help="Path to config.yaml"),
    last: int = typer.Option(10, "--last", help="Number of recent history entries to show"),
) -> None:
    """Show current experiment state (designed for agent use)."""
    cfg = _load_config(config, [])
    out: dict[str, Any] = {
        "best_value": None,
        "best_score": None,
        "iterations_done": 0,
        "no_improve_streak": 0,
        "recent_history": [],
    }
    if cfg.controller.checkpoint_path:
        from autoresearch_rl.checkpoint import load_checkpoint

        ckpt = load_checkpoint(cfg.controller.checkpoint_path)
        if ckpt is not None:
            out["best_value"] = ckpt.best_value
            out["best_score"] = ckpt.best_score if ckpt.best_score < float("inf") else None
            out["iterations_done"] = ckpt.iteration + 1
            out["no_improve_streak"] = ckpt.no_improve_streak
            out["recent_history"] = ckpt.history[-last:]
    typer.echo(json.dumps(out, indent=2))


@app.command("run-one")
def run_one(
    config: str = typer.Argument(..., help="Path to config.yaml"),
    override: list[str] = typer.Option([], "--override"),
    params: str | None = typer.Option(None, "--params", help="JSON dict of hyperparameters"),
    diff_file: str | None = typer.Option(None, "--diff", help="Path to unified diff file"),
) -> None:
    """Run exactly one experiment iteration with an optional pre-supplied proposal.

    If --params or --diff is given, that proposal is injected directly.
    If neither is given, the configured policy proposes one iteration.
    """
    from autoresearch_rl.controller.engine import run_experiment
    from autoresearch_rl.controller.executor import MetricEvaluator, TargetExecutor
    from autoresearch_rl.controller.one_shot import OneTimePolicy
    from autoresearch_rl.policy.interface import DiffProposal, ParamProposal

    if params is not None and diff_file is not None:
        raise typer.BadParameter("--params and --diff are mutually exclusive")

    cfg = _load_config(config, override)
    cfg.controller.max_iterations = 1

    program = ""
    if cfg.program_path:
        program = Path(cfg.program_path).read_text(encoding="utf-8")

    target = build_target(cfg.target)
    manifest_cfg = {"policy": cfg.policy.model_dump()}

    if params is not None:
        proposal: Any = ParamProposal(params=json.loads(params), rationale="agent")
        executor: Any = TargetExecutor(target)
        extractor = _param_extractor
        label = "run-one-param"
    elif diff_file is not None:
        if not cfg.policy.mutable_file:
            raise typer.BadParameter("policy.mutable_file is required for --diff proposals")
        from autoresearch_rl.controller.diff_executor import DiffExecutor

        diff_text = Path(diff_file).read_text(encoding="utf-8")
        proposal = DiffProposal(diff=diff_text, rationale="agent")
        executor = DiffExecutor(target, cfg.policy.mutable_file)
        extractor = _diff_extractor
        label = "run-one-diff"
    else:
        # No proposal supplied: use configured policy for one iteration
        result = run_continuous(
            target=target,
            objective=cfg.objective,
            controller=cfg.controller,
            telemetry=cfg.telemetry,
            policy_cfg=cfg.policy,
            comparability_cfg=cfg.comparability,
            program=program,
        )
        typer.echo(json.dumps({
            "iterations": result.iterations,
            "best_value": result.best_value,
            "best_score": result.best_score,
        }, indent=2))
        return

    result = run_experiment(
        executor=executor,
        evaluator=MetricEvaluator(),
        policy=OneTimePolicy(proposal),
        objective=cfg.objective,
        controller=cfg.controller,
        telemetry=cfg.telemetry,
        comparability_cfg=cfg.comparability,
        proposal_state_builder=lambda h, p: {"history": h, "program": p},
        proposal_params_extractor=extractor,
        program=program,
        description_label=label,
        manifest_config=manifest_cfg,
    )
    typer.echo(json.dumps({
        "iterations": result.iterations,
        "best_value": result.best_value,
        "best_score": result.best_score,
    }, indent=2))


@app.command()
def upload(
    config: str = typer.Argument(..., help="Path to config.yaml"),
    repo: str = typer.Option(..., "--repo", help="HuggingFace Hub repo (e.g. user/model-name)"),
    private: bool = typer.Option(False, "--private", help="Create private repo"),
    token_env: str = typer.Option("HF_TOKEN", "--token-env", help="Env var with HF token"),
) -> None:
    """Upload the best model version to HuggingFace Hub."""
    cfg = _load_config(config, [])

    versions_dir = Path(cfg.telemetry.versions_dir)
    if not versions_dir.exists():
        raise typer.BadParameter(f"No versions directory: {versions_dir}")

    # Find the best version by scanning version.json files
    best_version: dict | None = None
    best_score = float("inf")
    for vdir in sorted(versions_dir.iterdir()):
        meta_path = vdir / "version.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics = meta.get("metrics", {})
        metric_val = metrics.get(cfg.objective.metric)
        if metric_val is None:
            continue
        score = float(metric_val) if cfg.objective.direction == "min" else -float(metric_val)
        if score < best_score:
            best_score = score
            best_version = meta

    if best_version is None:
        raise typer.BadParameter("No versions with objective metric found")

    model_dir = best_version.get("model_dir")
    if not model_dir or not Path(model_dir).exists():
        raise typer.BadParameter(
            f"Best version (iter {best_version['iter']}) has no model_dir "
            f"or path does not exist: {model_dir}"
        )

    token = os.environ.get(token_env)
    if not token:
        raise typer.BadParameter(f"{token_env} not set")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise typer.BadParameter("pip install huggingface-hub required for upload")

    api = HfApi(token=token)
    api.create_repo(repo, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo,
        commit_message=(
            f"autoresearch-rl best model: "
            f"{cfg.objective.metric}={best_version['metrics'].get(cfg.objective.metric)} "
            f"(iter {best_version['iter']})"
        ),
    )

    # Upload version metadata alongside
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(best_version, f, indent=2)
        f.flush()
        api.upload_file(
            path_or_fileobj=f.name,
            path_in_repo="autoresearch_version.json",
            repo_id=repo,
            commit_message="autoresearch-rl version metadata",
        )
        Path(f.name).unlink(missing_ok=True)

    typer.echo(json.dumps({
        "repo": repo,
        "iter": best_version["iter"],
        "metrics": best_version["metrics"],
        "model_dir": model_dir,
    }, indent=2))


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: str = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
    override: list[str] = typer.Option([], "--override"),
    seed: int | None = typer.Option(None, "--seed"),
) -> None:
    """Autonomous ML experiment loop."""
    if ctx.invoked_subcommand is None and config:
        _run(config, override, seed)


if __name__ == "__main__":
    app()
