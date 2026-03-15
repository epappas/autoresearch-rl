from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, Field


class DistillSample(BaseModel):
    episode_id: str
    iteration: int
    status: str
    diff: str
    eval_score: float
    hint: str
    reward: float | None = None
    kl_divergence: float | None = None
    teacher_version: str | None = None
    timestamp: float = Field(default_factory=time.time)


def append_distill_sample(
    path: str,
    payload: DistillSample | dict,
) -> None:
    if isinstance(payload, dict):
        sample = DistillSample.model_validate(payload)
    else:
        sample = payload

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(sample.model_dump_json() + "\n")
