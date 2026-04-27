from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


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
    setup_cmd: str | None = None


class TargetConfig(BaseModel):
    type: Literal["command", "http", "basilica"] = "command"
    prepare_cmd: list[str] | None = None
    train_cmd: list[str] | None = None
    eval_cmd: list[str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    timeout_s: int = 3600
    workdir: str = "."
    basilica: BasilicaConfig = Field(default_factory=BasilicaConfig)


class PolicyConfig(BaseModel):
    type: Literal[
        "grid", "random", "static", "learned", "llm", "llm_diff", "hybrid"
    ] = "static"
    params: dict[str, list[float] | list[int] | list[str] | list[bool]] = Field(
        default_factory=dict
    )
    seed: int = 7
    llm_api_url: str | None = None
    llm_model: str | None = None
    llm_api_key_env: str = "OPENAI_API_KEY"
    llm_timeout_s: int = 30
    # Diff mode fields
    mutable_file: str | None = None
    frozen_file: str | None = None
    program_file: str | None = None
    contract_strict: bool = True
    required_calls: list[str] = Field(default_factory=lambda: ["emit_progress"])
    # Hybrid mode fields
    hybrid_param_explore_iters: int = 5
    hybrid_stall_threshold: int = 3
    hybrid_diff_failure_limit: int = 3

    @model_validator(mode="after")
    def _validate_llm_fields(self) -> "PolicyConfig":
        if self.type in ("llm", "llm_diff", "hybrid"):
            if not self.llm_api_url:
                raise ValueError(
                    f"llm_api_url is required when policy type is '{self.type}'"
                )
            if not self.llm_model:
                raise ValueError(
                    f"llm_model is required when policy type is '{self.type}'"
                )
        if self.type in ("llm_diff", "hybrid"):
            if not self.mutable_file:
                raise ValueError(
                    f"mutable_file is required when policy type is '{self.type}'"
                )
        return self


class ComparabilityConfig(BaseModel):
    budget_mode: Literal["fixed_wallclock"] = "fixed_wallclock"
    expected_budget_s: int = 300
    expected_hardware_fingerprint: str | None = None
    strict: bool = True


class IntraIterationCancelConfig(BaseModel):
    enabled: bool = False
    min_steps: int = 5
    poll_interval_s: float = 5.0
    min_reports_before_decide: int = 5


class ControllerConfig(BaseModel):
    seed: int | None = None
    max_wall_time_s: int | None = None
    max_iterations: int | None = None
    no_improve_limit: int | None = None
    failure_rate_limit: float | None = None
    failure_window: int = 10
    checkpoint_path: str | None = None
    intra_iteration_cancel: IntraIterationCancelConfig = Field(
        default_factory=IntraIterationCancelConfig
    )


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
    model_output_dir: str | None = None
    timeline_path: str | None = None
    max_file_size_bytes: int = 50 * 1024 * 1024  # 50MB
    max_rotated_files: int = 5


class RunConfig(BaseModel):
    name: str = "autoresearch-run"
    program_path: str | None = None
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    controller: ControllerConfig = Field(default_factory=ControllerConfig)
    comparability: ComparabilityConfig = Field(default_factory=ComparabilityConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
