from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from autoresearch_rl.config import ObjectiveConfig
from autoresearch_rl.controller.contract import ContractConfig, validate_diff_against_contract
from autoresearch_rl.eval.judge import judge_next_state
from autoresearch_rl.eval.metrics import ParsedMetrics, parse_metrics
from autoresearch_rl.eval.scoring import TrialSignals, score_from_signals
from autoresearch_rl.policy.interface import DiffProposal, ParamProposal, Proposal
from autoresearch_rl.sandbox.runner import EarlyStopConfig, TrialResult, run_trial
from autoresearch_rl.target.interface import TargetAdapter


@dataclass
class Outcome:
    status: str
    metrics: dict[str, float]
    stdout: str
    stderr: str
    elapsed_s: float
    run_dir: str
    judge_signals: dict | None = None


class Executor(Protocol):
    def execute(self, proposal: Proposal, run_dir: str) -> Outcome: ...


class Evaluator(Protocol):
    def score(self, outcome: Outcome, objective: ObjectiveConfig) -> float | None: ...


class TargetExecutor:
    """Wraps a TargetAdapter for param-based proposals."""

    def __init__(self, target: TargetAdapter) -> None:
        self._target = target

    def execute(self, proposal: Proposal, run_dir: str) -> Outcome:
        assert isinstance(proposal, ParamProposal)
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        try:
            train_out = self._target.run(run_dir=run_dir, params=proposal.params)
            if train_out.status != "ok":
                outcome = train_out
            else:
                outcome = self._target.eval(run_dir=run_dir, params=proposal.params)
        except Exception as exc:
            return Outcome(
                status="failed", metrics={}, stdout="",
                stderr=str(exc), elapsed_s=0.0, run_dir=run_dir,
            )
        return Outcome(
            status=outcome.status,
            metrics=outcome.metrics,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            elapsed_s=outcome.elapsed_s,
            run_dir=outcome.run_dir,
        )


@dataclass
class SandboxExecutorConfig:
    workdir: str = "."
    trial_timeout_s: int = 30
    trial_command: list[str] = field(default_factory=list)
    early_stop: EarlyStopConfig = field(default_factory=lambda: EarlyStopConfig(enabled=False))
    contract: ContractConfig | None = None


class SandboxExecutor:
    """Wraps sandbox/runner for diff-based proposals."""

    def __init__(self, config: SandboxExecutorConfig) -> None:
        self._cfg = config
        self._previous_trial: TrialResult | None = None
        self._previous_parsed: ParsedMetrics | None = None

    def execute(self, proposal: Proposal, run_dir: str) -> Outcome:
        assert isinstance(proposal, DiffProposal)
        diff = proposal.diff

        if self._cfg.contract is not None:
            ok_contract, contract_reason = validate_diff_against_contract(
                diff, self._cfg.contract
            )
            if self._cfg.contract.strict and not ok_contract:
                return Outcome(
                    status="rejected", metrics={}, stdout="",
                    stderr=contract_reason, elapsed_s=0.0, run_dir=run_dir,
                )

        command = self._cfg.trial_command or [sys.executable, "train.py"]
        trial = run_trial(
            diff=diff,
            timeout_s=self._cfg.trial_timeout_s,
            command=command,
            workdir=self._cfg.workdir,
            apply_patch=True,
            rollback_patch=True,
            early_stop=self._cfg.early_stop,
            use_worktree=True,
        )

        parsed = parse_metrics(trial.stdout)
        metrics: dict[str, float] = {}
        if parsed.val_bpb is not None:
            metrics["val_bpb"] = parsed.val_bpb
        if parsed.loss is not None:
            metrics["loss"] = parsed.loss

        judge_signals: dict | None = None
        if self._previous_trial is not None:
            judge = judge_next_state(
                prev_status=self._previous_trial.status,
                next_status=trial.status,
                next_stdout=trial.stdout,
                next_stderr=trial.stderr,
                vote_count=3,
            )
            judge_signals = {
                "eval_score": judge.eval_score,
                "hint": judge.hint,
                "prev_status": self._previous_trial.status,
                "prev_val_bpb": self._previous_parsed.val_bpb if self._previous_parsed else None,
                "prev_loss": self._previous_parsed.loss if self._previous_parsed else None,
                "prev_diff": getattr(self, "_previous_diff", ""),
            }

        self._previous_trial = trial
        self._previous_parsed = parsed
        self._previous_diff = diff

        return Outcome(
            status=trial.status,
            metrics=metrics,
            stdout=trial.stdout,
            stderr=trial.stderr,
            elapsed_s=trial.elapsed_s,
            run_dir=run_dir,
            judge_signals=judge_signals,
        )


class MetricEvaluator:
    """Extracts objective metric and normalizes direction."""

    def score(self, outcome: Outcome, objective: ObjectiveConfig) -> float | None:
        if objective.metric not in outcome.metrics:
            return None
        value = float(outcome.metrics[objective.metric])
        return value if objective.direction == "min" else -value


class JudgeEvaluator:
    """Scores using judge_next_state + score_from_signals."""

    def score(self, outcome: Outcome, objective: ObjectiveConfig) -> float | None:
        if outcome.judge_signals is None:
            return None
        signals = TrialSignals(
            status=outcome.judge_signals.get("prev_status", outcome.status),
            val_bpb=outcome.judge_signals.get("prev_val_bpb"),
            loss=outcome.judge_signals.get("prev_loss"),
            eval_score=outcome.judge_signals.get("eval_score", 0.0),
            hint=outcome.judge_signals.get("hint", ""),
        )
        return score_from_signals(signals)
