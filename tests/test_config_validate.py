from __future__ import annotations

from pathlib import Path

from autoresearch_rl.config import (
    BasilicaConfig,
    ComparabilityConfig,
    ControllerConfig,
    PolicyConfig,
    RunConfig,
    TargetConfig,
    TelemetryConfig,
)
from autoresearch_rl.config_validate import (
    ValidationError,
    has_blocking_errors,
    validate_runtime,
)


def _base_cfg(**overrides) -> RunConfig:
    base = dict(
        target=TargetConfig(type="command", train_cmd=["python3", "train.py"]),
        policy=PolicyConfig(type="static"),
        controller=ControllerConfig(),
        telemetry=TelemetryConfig(),
        comparability=ComparabilityConfig(strict=False),
    )
    base.update(overrides)
    return RunConfig(**base)


def _by_code(errs: list[ValidationError], code: str) -> list[ValidationError]:
    return [e for e in errs if e.code == code]


# ---------------------------------------------------------------- reserved param keys


def test_reserved_param_key_blocks() -> None:
    cfg = _base_cfg(policy=PolicyConfig(
        type="random",
        params={"AR_LEARNING_RATE": [1e-3]},
    ))
    errs = validate_runtime(cfg)
    assert _by_code(errs, "reserved_param_key"), errs
    assert has_blocking_errors(errs)


def test_normal_param_key_passes() -> None:
    cfg = _base_cfg(policy=PolicyConfig(
        type="random",
        params={"learning_rate": [1e-3, 1e-4]},
    ))
    errs = validate_runtime(cfg)
    assert not _by_code(errs, "reserved_param_key")


# ---------------------------------------------------------------- policy file existence


def test_missing_mutable_file_blocks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    nonexistent = tmp_path / "nope.py"
    cfg = _base_cfg(policy=PolicyConfig(
        type="llm_diff",
        mutable_file=str(nonexistent),
        llm_api_url="http://stub",
        llm_model="stub",
    ))
    errs = validate_runtime(cfg)
    misses = _by_code(errs, "missing_file")
    assert any("policy.mutable_file" in e.field for e in misses), errs


def test_existing_files_pass(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    src = tmp_path / "train.py"
    src.write_text("pass\n")
    cfg = _base_cfg(policy=PolicyConfig(
        type="llm_diff",
        mutable_file=str(src),
        llm_api_url="http://stub",
        llm_model="stub",
    ))
    errs = validate_runtime(cfg)
    assert not _by_code(errs, "missing_file")


# ---------------------------------------------------------------- basilica target


def test_basilica_without_api_key_blocks(monkeypatch) -> None:
    monkeypatch.delenv("BASILICA_API_KEY", raising=False)
    cfg = _base_cfg(target=TargetConfig(type="basilica", train_cmd=["python3", "train.py"]))
    errs = validate_runtime(cfg)
    assert _by_code(errs, "missing_env"), errs
    assert has_blocking_errors(errs)


def test_basilica_with_api_key_passes(monkeypatch) -> None:
    monkeypatch.setenv("BASILICA_API_KEY", "stub")
    cfg = _base_cfg(target=TargetConfig(type="basilica", train_cmd=["python3", "train.py"]))
    errs = validate_runtime(cfg)
    api_errs = [e for e in errs if e.field == "env.BASILICA_API_KEY"]
    assert not api_errs


def test_basilica_empty_gpu_models_blocks(monkeypatch) -> None:
    monkeypatch.setenv("BASILICA_API_KEY", "stub")
    cfg = _base_cfg(target=TargetConfig(
        type="basilica", train_cmd=["python3", "train.py"],
        basilica=BasilicaConfig(gpu_models=[]),
    ))
    errs = validate_runtime(cfg)
    assert _by_code(errs, "empty_gpu_models"), errs


# ---------------------------------------------------------------- LLM credentials


def test_llm_policy_without_api_key_blocks(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = _base_cfg(policy=PolicyConfig(
        type="llm",
        llm_api_url="http://stub",
        llm_model="stub",
        params={"learning_rate": [1e-3]},
    ))
    errs = validate_runtime(cfg)
    miss = _by_code(errs, "missing_env")
    assert any("OPENAI_API_KEY" in e.field for e in miss), errs


def test_static_policy_skips_llm_check(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = _base_cfg(policy=PolicyConfig(type="static"))
    errs = validate_runtime(cfg)
    assert not _by_code(errs, "missing_env")


# ---------------------------------------------------------------- budget alignment


def test_budget_exceeds_wall_warns_only() -> None:
    cfg = _base_cfg(
        comparability=ComparabilityConfig(expected_budget_s=600, strict=False),
        controller=ControllerConfig(max_wall_time_s=300),
    )
    errs = validate_runtime(cfg)
    warns = _by_code(errs, "budget_exceeds_wall")
    assert warns
    assert all(e.severity == "warn" for e in warns)
    # warn is non-blocking
    assert not has_blocking_errors([e for e in errs if e.code == "budget_exceeds_wall"])


# ---------------------------------------------------------------- aggregate behavior


def test_clean_config_has_no_errors() -> None:
    cfg = _base_cfg()
    errs = validate_runtime(cfg)
    assert errs == []


def test_validation_error_format_includes_severity_and_code() -> None:
    err = ValidationError(severity="error", code="missing_file",
                          message="not found", field="policy.mutable_file")
    formatted = err.format()
    assert "[ERROR]" in formatted
    assert "missing_file" in formatted
    assert "policy.mutable_file" in formatted
