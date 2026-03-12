import typer
import yaml

from autoresearch_rl.controller.loop import run_loop

app = typer.Typer()


@app.command()
def main(config: str = "configs/example.yaml", iterations: int | None = None) -> None:
    cfg = yaml.safe_load(open(config, "r", encoding="utf-8"))
    iters = iterations or int(cfg.get("controller", {}).get("max_iterations", 1))
    trace_path = cfg.get("telemetry", {}).get("trace_path", "traces/events.jsonl")
    ledger_path = cfg.get("telemetry", {}).get("ledger_path", "results.tsv")
    contract = cfg.get("experiment", {}).get("contract", {})

    result = run_loop(
        max_iterations=min(iters, 3),
        trace_path=trace_path,
        ledger_path=ledger_path,
        mutable_file=contract.get("mutable_file", "train.py"),
        frozen_file=contract.get("frozen_file", "prepare.py"),
        program_path=contract.get("program_file", "programs/default.md"),
    )
    print({"ok": True, "iterations": result.iterations, "best_score": result.best_score})


if __name__ == "__main__":
    app()
