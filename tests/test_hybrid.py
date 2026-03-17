from __future__ import annotations

from unittest.mock import MagicMock

from autoresearch_rl.policy.hybrid import HybridPolicy
from autoresearch_rl.policy.interface import DiffProposal, ParamProposal


SAMPLE_SOURCE = """\
import torch
LEARNING_RATE = 0.0026
"""

SAMPLE_DIFF = """\
--- a/train.py
+++ b/train.py
@@ -1,2 +1,2 @@
 import torch
-LEARNING_RATE = 0.0026
+LEARNING_RATE = 0.0020
"""


def _make_entry(
    i: int,
    status: str = "ok",
    decision: str = "discard",
) -> dict:
    return {
        "iter": i,
        "status": status,
        "decision": decision,
        "metrics": {"val_bpb": 1.5},
        "params": {},
    }


def _make_param_policy():
    policy = MagicMock()
    policy.propose.return_value = ParamProposal(
        params={"lr": 0.001}, rationale="llm",
    )
    return policy


def _make_diff_policy():
    policy = MagicMock()
    policy.propose.return_value = DiffProposal(
        diff=SAMPLE_DIFF, rationale="llm-diff",
    )
    return policy


class TestHybridPolicySelection:

    def test_starts_with_param_mode(self):
        hybrid = HybridPolicy(
            _make_param_policy(), _make_diff_policy(),
            param_explore_iters=5,
        )
        state = {"history": [], "source": SAMPLE_SOURCE}
        proposal = hybrid.propose(state)

        assert isinstance(proposal, ParamProposal)
        assert hybrid.active_mode == "param"

    def test_stays_param_during_explore_phase(self):
        hybrid = HybridPolicy(
            _make_param_policy(), _make_diff_policy(),
            param_explore_iters=5,
        )
        history = [_make_entry(i) for i in range(4)]
        state = {"history": history, "source": SAMPLE_SOURCE}
        proposal = hybrid.propose(state)

        assert isinstance(proposal, ParamProposal)
        assert hybrid.active_mode == "param"

    def test_switches_to_diff_after_stall(self):
        hybrid = HybridPolicy(
            _make_param_policy(), _make_diff_policy(),
            param_explore_iters=3,
            stall_threshold=3,
        )
        # 5 iterations, none kept -> stall of 5
        history = [_make_entry(i, decision="discard") for i in range(5)]
        state = {"history": history, "source": SAMPLE_SOURCE}
        proposal = hybrid.propose(state)

        assert isinstance(proposal, DiffProposal)
        assert hybrid.active_mode == "diff"

    def test_stays_param_when_improving(self):
        hybrid = HybridPolicy(
            _make_param_policy(), _make_diff_policy(),
            param_explore_iters=3,
            stall_threshold=3,
        )
        # Recent keep breaks the stall
        history = [_make_entry(i, decision="discard") for i in range(5)]
        history.append(_make_entry(5, decision="keep"))
        history.append(_make_entry(6, decision="discard"))
        state = {"history": history, "source": SAMPLE_SOURCE}
        proposal = hybrid.propose(state)

        assert isinstance(proposal, ParamProposal)

    def test_falls_back_to_param_after_diff_failures(self):
        hybrid = HybridPolicy(
            _make_param_policy(), _make_diff_policy(),
            param_explore_iters=3,
            stall_threshold=3,
            diff_failure_limit=2,
        )
        # Stall triggers diff mode
        history = [_make_entry(i, decision="discard") for i in range(6)]
        state = {"history": history, "source": SAMPLE_SOURCE}

        # First diff proposal - simulate failures
        hybrid.propose(state)
        assert hybrid.active_mode == "diff"
        hybrid.record_reward(-0.1)  # failure
        hybrid.record_reward(-0.1)  # failure

        # Now should fall back to param
        proposal = hybrid.propose(state)
        assert isinstance(proposal, ParamProposal)
        assert hybrid.active_mode == "param"


class TestHybridPolicyRewardRouting:

    def test_param_mode_reward_does_not_affect_diff_counter(self):
        hybrid = HybridPolicy(
            _make_param_policy(), _make_diff_policy(),
            param_explore_iters=5,
        )
        state = {"history": [], "source": SAMPLE_SOURCE}
        hybrid.propose(state)
        assert hybrid.active_mode == "param"

        hybrid.record_reward(-0.1)
        assert hybrid._diff_consecutive_failures == 0

    def test_diff_mode_success_resets_failure_counter(self):
        hybrid = HybridPolicy(
            _make_param_policy(), _make_diff_policy(),
            param_explore_iters=3,
            stall_threshold=3,
            diff_failure_limit=5,
        )
        history = [_make_entry(i, decision="discard") for i in range(6)]
        state = {"history": history, "source": SAMPLE_SOURCE}

        hybrid.propose(state)
        assert hybrid.active_mode == "diff"

        hybrid.record_reward(-0.1)  # failure
        assert hybrid._diff_consecutive_failures == 1

        hybrid.propose(state)
        hybrid.record_reward(1.0)  # success
        assert hybrid._diff_consecutive_failures == 0
