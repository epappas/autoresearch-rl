from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ObjectiveConfig(BaseModel):
    metric: str = "val_bpb"
    direction: Literal["min", "max"] = "min"


class BasilicaConfig(BaseModel):
    image: str = "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"
    gpu_count: int = 1
    gpu_models: list[str] = Field(default_factory=lambda: ["A100", "H100"])
    memory: str = "32Gi"
    cpu: str = "8"
    storage: str | None = "/data"
    ttl_seconds: int = 7200
    min_gpu_memory_gb: int | None = None


class TargetConfig(BaseModel):
    type: Literal["command", "http", "basilica"] = "command"
    train_cmd: list[str] | None = None
    eval_cmd: list[str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    timeout_s: int = 3600
    workdir: str = "."
    basilica: BasilicaConfig = Field(default_factory=BasilicaConfig)


class PolicyConfig(BaseModel):
    type: Literal["grid", "random", "static", "learned"] = "static"
    params: dict[str, list[float] | list[int] | list[str] | list[bool]] = Field(default_factory=dict)
    seed: int = 7


class ComparabilityConfig(BaseModel):
    budget_mode: Literal["fixed_wallclock"] = "fixed_wallclock"
    expected_budget_s: int = 300
    expected_hardware_fingerprint: str | None = None
    strict: bool = True


class ControllerConfig(BaseModel):
    seed: int | None = None
    max_wall_time_s: int | None = None
    no_improve_limit: int | None = None
    failure_rate_limit: float | None = None
    failure_window: int = 10
    checkpoint_path: str | None = None


class ScoringConfig(BaseModel):
    val_bpb: float = 1.0
    loss: float = 0.15
    fail_penalty: float = 0.8
    timeout_penalty: float = 1.2
    neutral_penalty: float = 0.05
    directional_bonus: float = 0.2
    early_stop_penalty: float = 0.4
    eval_score_weight: float = 0.25


class TelemetryConfig(BaseModel):
    trace_path: str = "traces/events.jsonl"
    ledger_path: str = "artifacts/results.tsv"
    artifacts_dir: str = "artifacts/runs"
    versions_dir: str = "artifacts/versions"
    max_file_size_bytes: int = 50 * 1024 * 1024  # 50MB
    max_rotated_files: int = 5


class RunConfig(BaseModel):
    name: str = "autoresearch-run"
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    controller: ControllerConfig = Field(default_factory=ControllerConfig)
    comparability: ComparabilityConfig = Field(default_factory=ComparabilityConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
