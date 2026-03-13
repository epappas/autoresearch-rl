from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True)
class Proposal:
    diff: str
    rationale: str = ""


class ProposalPolicy(Protocol):
    def propose(self, state: Mapping[str, object]) -> Proposal: ...

    def propose_diff(self, state: Mapping[str, object]) -> str: ...
