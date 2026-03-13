from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ObjectiveConfig(BaseModel):
    metric: str = "val_bpb"
    direction: Literal["min", "max"] = "min"


class TargetConfig(BaseModel):
    type: Literal["command", "http"] = "command"
    train_cmd: list[str] | None = None
    eval_cmd: list[str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    timeout_s: int = 3600
    workdir: str = "."


class PolicyConfig(BaseModel):
    type: Literal["grid", "random", "static"] = "static"
    params: dict[str, list[float] | list[int] | list[str] | list[bool]] = Field(default_factory=dict)
    seed: int = 7


class ComparabilityConfig(BaseModel):
    budget_mode: Literal["fixed_wallclock"] = "fixed_wallclock"
    expected_budget_s: int = 300
    expected_hardware_fingerprint: str | None = None
    strict: bool = True


class ControllerConfig(BaseModel):
    max_wall_time_s: int | None = None
    no_improve_limit: int | None = None
    failure_rate_limit: float | None = None
    failure_window: int = 10
    seed: int | None = None


class TelemetryConfig(BaseModel):
    trace_path: str = "traces/events.jsonl"
    ledger_path: str = "artifacts/results.tsv"
    artifacts_dir: str = "artifacts/runs"
    versions_dir: str = "artifacts/versions"


class RunConfig(BaseModel):
    name: str = "autoresearch-run"
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    controller: ControllerConfig = Field(default_factory=ControllerConfig)
    comparability: ComparabilityConfig = Field(default_factory=ComparabilityConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
