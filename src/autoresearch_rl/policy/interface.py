from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


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
