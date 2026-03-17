from __future__ import annotations

from autoresearch_rl.policy.interface import Proposal


class OneTimePolicy:
    """Returns a fixed proposal on every propose() call.

    Used by the run-one CLI command to inject a caller-supplied
    param or diff proposal into the experiment engine.
    """

    def __init__(self, proposal: Proposal) -> None:
        self._proposal = proposal

    def propose(self, state: dict) -> Proposal:
        return self._proposal
