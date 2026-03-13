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
    params: dict[str, list[float] | list[int] | list[str]] = Field(default_factory=dict)


class ControllerConfig(BaseModel):
    max_wall_time_s: int | None = None
    no_improve_limit: int | None = None
    failure_rate_limit: float | None = None
    failure_window: int = 10


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
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
