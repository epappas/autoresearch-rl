from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Proposal:
    """Base proposal type shared by all policy families."""

    rationale: str = ""


@dataclass
class ParamProposal(Proposal):
    """Proposal carrying hyperparameter overrides (continuous loop)."""

    params: dict[str, object] = field(default_factory=dict)


@dataclass
class DiffProposal(Proposal):
    """Proposal carrying a unified diff string (legacy loop)."""

    diff: str = ""


class Policy(Protocol):
    """Unified policy protocol for both param and diff proposals."""

    def propose(self, state: dict) -> Proposal: ...


@runtime_checkable
class Learnable(Protocol):
    """Policy that can receive reward feedback after each proposal."""

    def record_reward(self, reward: float) -> None: ...
