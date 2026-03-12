import typer
import yaml

from autoresearch_rl.controller.loop import run_loop
from autoresearch_rl.telemetry.comparability import ComparabilityPolicy

app = typer.Typer()


@app.command()
def main(config: str = "configs/example.yaml", iterations: int | None = None) -> None:
    cfg = yaml.safe_load(open(config, "r", encoding="utf-8"))
    iters = iterations or int(cfg.get("controller", {}).get("max_iterations", 1))
    trace_path = cfg.get("telemetry", {}).get("trace_path", "traces/events.jsonl")
    ledger_path = cfg.get("telemetry", {}).get("ledger_path", "results.tsv")
    experiment = cfg.get("experiment", {})
    contract = experiment.get("contract", {})
    comparability_cfg = experiment.get("comparability", {})
    max_wall_s = int(experiment.get("max_wall_seconds", 30))

    comparability_policy = ComparabilityPolicy(
        budget_mode=str(comparability_cfg.get("budget_mode", "fixed_wallclock")),
        expected_budget_s=int(comparability_cfg.get("expected_budget_s", max_wall_s)),
        expected_hardware_fingerprint=comparability_cfg.get("expected_hardware_fingerprint"),
        strict=bool(comparability_cfg.get("strict", True)),
    )

    result = run_loop(
        max_iterations=min(iters, 3),
        trace_path=trace_path,
        ledger_path=ledger_path,
        mutable_file=contract.get("mutable_file", "train.py"),
        frozen_file=contract.get("frozen_file", "prepare.py"),
        program_path=contract.get("program_file", "programs/default.md"),
        contract_strict=bool(contract.get("strict", True)),
        trial_timeout_s=max_wall_s,
        comparability_policy=comparability_policy,
    )
    print({"ok": True, "iterations": result.iterations, "best_score": result.best_score})


if __name__ == "__main__":
    app()
