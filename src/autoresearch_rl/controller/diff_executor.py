"""Diff-based executor for code modification proposals.

Validates diffs (safety + contract), applies them in-memory via a temp
git repo, and passes the modified source as a base64 env var to the
target adapter.
"""
from __future__ import annotations

import base64
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from autoresearch_rl.controller.contract import ContractConfig, validate_diff_against_contract
from autoresearch_rl.controller.executor import Outcome, TargetExecutor
from autoresearch_rl.policy.interface import DiffProposal, ParamProposal, Proposal
from autoresearch_rl.sandbox.validator import validate_diff, validate_required_calls
from autoresearch_rl.target.interface import TargetAdapter

logger = logging.getLogger(__name__)

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "autoresearch",
    "GIT_AUTHOR_EMAIL": "ar@local",
    "GIT_COMMITTER_NAME": "autoresearch",
    "GIT_COMMITTER_EMAIL": "ar@local",
}


def _apply_diff_in_memory(source: str, diff: str, filename: str) -> str | None:
    """Apply a unified diff to source in an ephemeral git repo.

    Returns the modified source string, or None on failure.
    """
    with tempfile.TemporaryDirectory(prefix="ar-diff-") as tmpdir:
        src_path = Path(tmpdir) / filename
        src_path.write_text(source, encoding="utf-8")

        env = {**os.environ, **_GIT_ENV}
        git = ["git", "-C", tmpdir]

        init = subprocess.run(git + ["init"], capture_output=True, check=False, env=env)
        if init.returncode != 0:
            logger.warning("git init failed: %s", init.stderr.strip())
            return None

        subprocess.run(git + ["add", filename], capture_output=True, check=False, env=env)
        subprocess.run(
            git + ["commit", "-m", "base"],
            capture_output=True, check=False, env=env,
        )

        apply = subprocess.run(
            git + ["apply", "-"],
            input=diff, text=True, capture_output=True,
        )
        if apply.returncode != 0:
            logger.warning("git apply failed: %s", apply.stderr.strip())
            return None

        return src_path.read_text(encoding="utf-8")


def _persist_diff(mutable_file: str, diff: str) -> bool:
    """Apply a diff permanently to the mutable file on disk.

    Returns True on success, False on failure.
    """
    source = Path(mutable_file).read_text(encoding="utf-8")
    filename = os.path.basename(mutable_file)
    modified = _apply_diff_in_memory(source, diff, filename)
    if modified is None:
        logger.warning("Failed to persist diff to %s", mutable_file)
        return False
    Path(mutable_file).write_text(modified, encoding="utf-8")
    logger.info("Persisted diff to %s", mutable_file)
    return True


def _rejected(reason: str, run_dir: str) -> Outcome:
    return Outcome(
        status="rejected", metrics={}, stdout="",
        stderr=reason, elapsed_s=0.0, run_dir=run_dir,
    )


class DiffExecutor:
    """Validates a DiffProposal, applies it in-memory, and delegates to target."""

    def __init__(
        self,
        target: TargetAdapter,
        mutable_file: str,
        contract: ContractConfig | None = None,
        required_calls: list[str] | None = None,
    ) -> None:
        self._target = target
        self._mutable_file = mutable_file
        self._contract = contract
        self._required_calls = list(required_calls or [])
        self._filename = os.path.basename(mutable_file)

    def execute(self, proposal: Proposal, run_dir: str) -> Outcome:
        assert isinstance(proposal, DiffProposal)
        diff = proposal.diff

        if not diff.strip():
            return _rejected("empty diff", run_dir)

        validation = validate_diff(diff)
        if not validation.ok:
            return _rejected(validation.reason, run_dir)

        if self._contract:
            ok, reason = validate_diff_against_contract(diff, self._contract)
            if self._contract.strict and not ok:
                return _rejected(reason, run_dir)

        source = Path(self._mutable_file).read_text(encoding="utf-8")
        modified = _apply_diff_in_memory(source, diff, self._filename)
        if modified is None:
            return _rejected("diff apply failed", run_dir)

        if self._required_calls:
            req = validate_required_calls(source, modified, self._required_calls)
            if not req.ok:
                return _rejected(req.reason, run_dir)

        encoded = base64.b64encode(modified.encode("utf-8")).decode("ascii")
        params: dict[str, object] = {
            "AR_MODIFIED_SOURCE": encoded,
            "AR_MODIFIED_TARGET": self._filename,
        }

        Path(run_dir).mkdir(parents=True, exist_ok=True)
        # Write modified source to disk so local CommandTarget runs it;
        # Basilica targets receive it via AR_MODIFIED_SOURCE bootstrap.
        Path(self._mutable_file).write_text(modified, encoding="utf-8")
        try:
            train_out = self._target.run(run_dir=run_dir, params=params)
            outcome = train_out
            if train_out.status == "ok":
                outcome = self._target.eval(run_dir=run_dir, params=params)
        except Exception as exc:
            return Outcome(
                status="failed", metrics={}, stdout="",
                stderr=str(exc), elapsed_s=0.0, run_dir=run_dir,
            )
        finally:
            # Restore original; on_keep callback persists the diff if accepted.
            Path(self._mutable_file).write_text(source, encoding="utf-8")
        return Outcome(
            status=outcome.status,
            metrics=outcome.metrics,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            elapsed_s=outcome.elapsed_s,
            run_dir=outcome.run_dir,
        )


class HybridExecutor:
    """Dispatches ParamProposal to TargetExecutor, DiffProposal to DiffExecutor."""

    def __init__(
        self,
        target_executor: TargetExecutor,
        diff_executor: DiffExecutor,
    ) -> None:
        self._target_executor = target_executor
        self._diff_executor = diff_executor

    def execute(self, proposal: Proposal, run_dir: str) -> Outcome:
        if isinstance(proposal, DiffProposal):
            return self._diff_executor.execute(proposal, run_dir)
        assert isinstance(proposal, ParamProposal)
        return self._target_executor.execute(proposal, run_dir)
