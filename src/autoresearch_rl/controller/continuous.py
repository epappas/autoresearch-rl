from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from autoresearch_rl.config import ControllerConfig, ObjectiveConfig, TelemetryConfig
from autoresearch_rl.policy.search import GridPolicy, ParamPolicy, RandomPolicy, StaticPolicy
from autoresearch_rl.target.interface import RunOutcome, TargetAdapter
from autoresearch_rl.telemetry.events import emit
from autoresearch_rl.telemetry.ledger import append_result_row, ensure_results_tsv
from autoresearch_rl.telemetry.manifest import new_run_id, write_manifest


@dataclass
class LoopResult:
    best_score: float
    iterations: int


def _objective_value(metrics: dict[str, float], objective: ObjectiveConfig) -> float | None:
    if objective.metric not in metrics:
        return None
    return float(metrics[objective.metric])


def _score(value: float, objective: ObjectiveConfig) -> float:
    return value if objective.direction == "min" else -value


def _policy_from_config(policy_cfg) -> ParamPolicy:
    if policy_cfg.type == "grid":
        return GridPolicy(policy_cfg.params)
    if policy_cfg.type == "random":
        return RandomPolicy(policy_cfg.params)
    return StaticPolicy()


def _save_version(versions_dir: str, iter_idx: int, outcome: RunOutcome, params: dict[str, object]) -> str:
    target_dir = Path(versions_dir) / f"v{iter_idx:04d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "iter": iter_idx,
        "metrics": outcome.metrics,
        "params": params,
        "status": outcome.status,
        "run_dir": outcome.run_dir,
    }
    (target_dir / "version.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return str(target_dir)


def run_continuous(
    *,
    target: TargetAdapter,
    objective: ObjectiveConfig,
    controller: ControllerConfig,
    telemetry: TelemetryConfig,
    policy_cfg,
) -> LoopResult:
    ensure_results_tsv(telemetry.ledger_path)
    Path(telemetry.artifacts_dir).mkdir(parents=True, exist_ok=True)
    Path(telemetry.versions_dir).mkdir(parents=True, exist_ok=True)
    Path(telemetry.trace_path).parent.mkdir(parents=True, exist_ok=True)
    episode_id = new_run_id()
    history: list[dict] = []
    best_score = float("inf")
    best_value = None
    no_improve_streak = 0
    recent_statuses: list[str] = []
    iter_idx = 0
    start_ts = time.monotonic()

    policy = _policy_from_config(policy_cfg)

    while True:
        if controller.max_wall_time_s is not None and (time.monotonic() - start_ts) >= controller.max_wall_time_s:
            break

        proposal = policy.next(history=history)
        run_dir = str(Path(telemetry.artifacts_dir) / f"run-{iter_idx:04d}")
        Path(run_dir).mkdir(parents=True, exist_ok=True)

        emit(telemetry.trace_path, {"type": "proposal", "episode_id": episode_id, "iter": iter_idx, "params": proposal.params})
        train_out = target.run(run_dir=run_dir, params=proposal.params)
        eval_out = target.eval(run_dir=run_dir, params=proposal.params)

        outcome = eval_out
        value = _objective_value(outcome.metrics, objective)
        status = outcome.status if value is not None else "failed"

        decision = "discard"
        if value is not None:
            score = _score(value, objective)
            if score < best_score:
                best_score = score
                best_value = value
                decision = "keep"
                no_improve_streak = 0
                _save_version(telemetry.versions_dir, iter_idx, outcome, proposal.params)
            else:
                no_improve_streak += 1
        else:
            no_improve_streak += 1

        emit(
            telemetry.trace_path,
            {
                "type": "iteration",
                "episode_id": episode_id,
                "iter": iter_idx,
                "status": status,
                "decision": decision,
                "metrics": outcome.metrics,
                "params": proposal.params,
                "elapsed_s": outcome.elapsed_s,
            },
        )

        write_manifest(
            telemetry.artifacts_dir,
            {
                "episode_id": episode_id,
                "iter": iter_idx,
                "status": status,
                "decision": decision,
                "metrics": outcome.metrics,
                "params": proposal.params,
                "stdout": outcome.stdout,
                "stderr": outcome.stderr,
                "run_dir": outcome.run_dir,
            },
        )

        append_result_row(
            path=telemetry.ledger_path,
            commit="continuous",
            val_bpb=float(value if value is not None else 0.0),
            memory_gb=0.0,
            status=decision,
            description="continuous",
            episode_id=str(episode_id),
            iter_idx=int(iter_idx),
            score=float(best_score),
            budget_mode="continuous",
            budget_s=controller.max_wall_time_s or 0,
            hardware_fingerprint="",
            comparable=True,
            non_comparable_reason="",
        )

        history.append(
            {
                "iter": iter_idx,
                "status": status,
                "decision": decision,
                "metrics": outcome.metrics,
                "params": proposal.params,
            }
        )

        recent_statuses.append(status)
        if len(recent_statuses) > max(1, controller.failure_window):
            recent_statuses.pop(0)

        if controller.no_improve_limit is not None and no_improve_streak >= controller.no_improve_limit:
            break

        if controller.failure_rate_limit is not None and len(recent_statuses) >= max(1, controller.failure_window):
            fails = sum(1 for s in recent_statuses if s != "ok")
            if (fails / len(recent_statuses)) >= controller.failure_rate_limit:
                break

        iter_idx += 1

    return LoopResult(best_score=best_score, iterations=iter_idx)
