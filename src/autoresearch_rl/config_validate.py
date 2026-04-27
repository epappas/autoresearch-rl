"""Two-phase runtime config validation.

Phase 1 (pydantic): structural validation in config.py via model_validator.
Phase 2 (this module): semantic checks against the filesystem and environment.

Pure function: returns errors, never mutates. Called from cli.py before any
deployment or expensive setup. Errors with severity='error' refuse to start;
'warn' is printed and the run continues.

Adopted from RLix's two-phase validate-then-mutate pattern.
"""
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from autoresearch_rl.config import RunConfig

Severity = Literal["error", "warn"]

_RESERVED_PARAM_PREFIXES = ("AR_",)


@dataclass(frozen=True)
class ValidationError:
    severity: Severity
    code: str
    message: str
    field: str = ""

    def format(self) -> str:
        prefix = f"[{self.severity.upper()}]"
        loc = f" ({self.field})" if self.field else ""
        return f"{prefix} {self.code}{loc}: {self.message}"


def validate_runtime(cfg: RunConfig) -> list[ValidationError]:
    """Run all semantic checks. Returns ordered list of errors and warnings."""
    errors: list[ValidationError] = []
    errors.extend(_check_param_keys(cfg))
    errors.extend(_check_policy_files(cfg))
    errors.extend(_check_basilica_target(cfg))
    errors.extend(_check_llm_credentials(cfg))
    errors.extend(_check_checkpoint_dir(cfg))
    errors.extend(_check_model_output_dir(cfg))
    errors.extend(_check_budget_alignment(cfg))
    errors.extend(_check_required_calls_for_cancel(cfg))
    return errors


def has_blocking_errors(errors: list[ValidationError]) -> bool:
    return any(e.severity == "error" for e in errors)


# ---------------------------------------------------------------- individual checks


def _check_param_keys(cfg: RunConfig) -> list[ValidationError]:
    out: list[ValidationError] = []
    for key in cfg.policy.params:
        for reserved in _RESERVED_PARAM_PREFIXES:
            if key.upper().startswith(reserved):
                out.append(ValidationError(
                    severity="error",
                    code="reserved_param_key",
                    field=f"policy.params.{key}",
                    message=(
                        f"parameter name collides with reserved env var prefix "
                        f"'{reserved}' (the controller injects AR_PARAM_<NAME>)"
                    ),
                ))
    return out


def _check_policy_files(cfg: RunConfig) -> list[ValidationError]:
    out: list[ValidationError] = []
    for label, path in (
        ("policy.mutable_file", cfg.policy.mutable_file),
        ("policy.frozen_file", cfg.policy.frozen_file),
        ("policy.program_file", cfg.policy.program_file),
    ):
        if path and not Path(path).is_file():
            out.append(ValidationError(
                severity="error",
                code="missing_file",
                field=label,
                message=f"file does not exist: {path}",
            ))
    return out


def _check_basilica_target(cfg: RunConfig) -> list[ValidationError]:
    if cfg.target.type != "basilica":
        return []
    out: list[ValidationError] = []
    if not os.environ.get("BASILICA_API_KEY"):
        out.append(ValidationError(
            severity="error",
            code="missing_env",
            field="env.BASILICA_API_KEY",
            message="BASILICA_API_KEY is not set; basilica target requires it",
        ))
    if not cfg.target.basilica.gpu_models:
        out.append(ValidationError(
            severity="error",
            code="empty_gpu_models",
            field="target.basilica.gpu_models",
            message="must list at least one GPU model (e.g. ['A100','H100'])",
        ))
    if cfg.target.basilica.gpu_count < 1:
        out.append(ValidationError(
            severity="error",
            code="invalid_gpu_count",
            field="target.basilica.gpu_count",
            message=f"must be >= 1, got {cfg.target.basilica.gpu_count}",
        ))
    return out


def _check_llm_credentials(cfg: RunConfig) -> list[ValidationError]:
    if cfg.policy.type not in {"llm", "llm_diff", "hybrid"}:
        return []
    env_var = cfg.policy.llm_api_key_env
    if not os.environ.get(env_var):
        return [ValidationError(
            severity="error",
            code="missing_env",
            field=f"env.{env_var}",
            message=(
                f"{env_var} (configured via policy.llm_api_key_env) is not set; "
                f"required when policy.type is '{cfg.policy.type}'"
            ),
        )]
    return []


def _check_checkpoint_dir(cfg: RunConfig) -> list[ValidationError]:
    if not cfg.controller.checkpoint_path:
        return []
    parent = Path(cfg.controller.checkpoint_path).parent
    if parent and not parent.exists():
        # Try to create — checkpoint dir is allowed to not exist yet.
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return [ValidationError(
                severity="error",
                code="unwritable_dir",
                field="controller.checkpoint_path",
                message=f"cannot create parent dir {parent}: {exc}",
            )]
    if not os.access(parent, os.W_OK):
        return [ValidationError(
            severity="error",
            code="unwritable_dir",
            field="controller.checkpoint_path",
            message=f"parent dir not writable: {parent}",
        )]
    return []


def _check_model_output_dir(cfg: RunConfig) -> list[ValidationError]:
    if not cfg.telemetry.model_output_dir:
        return []
    parent = Path(cfg.telemetry.model_output_dir)
    if parent.exists() and not os.access(parent, os.W_OK):
        return [ValidationError(
            severity="error",
            code="unwritable_dir",
            field="telemetry.model_output_dir",
            message=f"dir not writable: {parent}",
        )]
    return []


def _check_budget_alignment(cfg: RunConfig) -> list[ValidationError]:
    expected = cfg.comparability.expected_budget_s
    wall = cfg.controller.max_wall_time_s
    if wall is not None and expected > wall:
        return [ValidationError(
            severity="warn",
            code="budget_exceeds_wall",
            field="comparability.expected_budget_s",
            message=(
                f"expected_budget_s={expected} exceeds controller.max_wall_time_s={wall}; "
                f"runs will be cut short before budget is reached"
            ),
        )]
    return []


def _check_required_calls_for_cancel(cfg: RunConfig) -> list[ValidationError]:
    """R3.e: positive-presence guardrail.

    If intra-iteration cancel is configured, the mutable file MUST contain at
    least one emit_progress(...) call — otherwise no progress signal can fire
    cancellation and the feature is dead.

    Currently `controller.intra_iteration_cancel` doesn't exist yet (Phase 2),
    so this is a placeholder that activates once the field is added.
    """
    cancel_enabled = getattr(
        getattr(cfg.controller, "intra_iteration_cancel", None),
        "enabled",
        False,
    )
    if not cancel_enabled:
        return []
    if not cfg.policy.mutable_file:
        return [ValidationError(
            severity="error",
            code="cancel_without_mutable",
            field="controller.intra_iteration_cancel",
            message="intra_iteration_cancel.enabled requires policy.mutable_file",
        )]
    src_path = Path(cfg.policy.mutable_file)
    if not src_path.is_file():
        return []  # already reported by _check_policy_files
    src = src_path.read_text(encoding="utf-8")
    if not _has_emit_progress_call(src):
        return [ValidationError(
            severity="error",
            code="missing_emit_progress",
            field="policy.mutable_file",
            message=(
                f"intra_iteration_cancel.enabled but {src_path} contains no "
                f"emit_progress(...) calls; cancellation can never fire"
            ),
        )]
    return []


def _has_emit_progress_call(src: str) -> bool:
    """Walk AST and return True if any call to emit_progress(...) exists."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # Fall back to regex if source is not valid Python (e.g. mid-edit).
        return bool(re.search(r"\bemit_progress\s*\(", src))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "emit_progress":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "emit_progress":
                return True
    return False
