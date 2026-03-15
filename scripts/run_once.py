import typer
import yaml

from autoresearch_rl.config import RunConfig
from autoresearch_rl.controller.loop import run_loop
from autoresearch_rl.telemetry.comparability import ComparabilityPolicy

app = typer.Typer()


@app.command()
def main(config: str = "configs/example.yaml", iterations: int | None = None) -> None:
    raw = yaml.safe_load(open(config, "r", encoding="utf-8"))
    validated = RunConfig.model_validate(raw)

    # max_iterations and continuous are not modeled in RunConfig; fall back to raw YAML
    iters = iterations or int(raw.get("controller", {}).get("max_iterations", 1))
    continuous = bool(raw.get("controller", {}).get("continuous", False))

    # experiment.contract and experiment.max_wall_seconds are not in RunConfig
    experiment = raw.get("experiment", {})
    contract = experiment.get("contract", {})
    max_wall_s = int(experiment.get("max_wall_seconds", 30))

    comparability_policy = ComparabilityPolicy(
        budget_mode=validated.comparability.budget_mode,
        expected_budget_s=validated.comparability.expected_budget_s,
        expected_hardware_fingerprint=validated.comparability.expected_hardware_fingerprint,
        strict=validated.comparability.strict,
    )

    result = run_loop(
        max_iterations=iters,
        trace_path=validated.telemetry.trace_path,
        ledger_path=validated.telemetry.ledger_path,
        mutable_file=contract.get("mutable_file", "train.py"),
        frozen_file=contract.get("frozen_file", "prepare.py"),
        program_path=contract.get("program_file", "programs/default.md"),
        contract_strict=bool(contract.get("strict", True)),
        trial_timeout_s=max_wall_s,
        comparability_policy=comparability_policy,
        continuous=continuous,
        max_wall_time_s=validated.controller.max_wall_time_s,
        no_improve_limit=validated.controller.no_improve_limit,
        failure_rate_limit=validated.controller.failure_rate_limit,
        failure_window=validated.controller.failure_window,
    )
    print({"ok": True, "iterations": result.iterations, "best_score": result.best_score})


if __name__ == "__main__":
    app()
