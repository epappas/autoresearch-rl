from __future__ import annotations

import numpy as np

from autoresearch_rl.policy.learned_search import (
    STATE_DIM,
    LearnedParamPolicy,
    LearnedSearchConfig,
)
from autoresearch_rl.policy.ppo import PPOConfig
from autoresearch_rl.policy.interface import ParamProposal


def _param_space() -> dict[str, list]:
    return {"lr": [0.001, 0.01, 0.1], "batch_size": [16, 32]}


def _sample_history(n: int = 5) -> list[dict]:
    history: list[dict] = []
    for i in range(n):
        history.append({
            "iter": i,
            "status": "ok" if i % 3 != 0 else "failed",
            "decision": "keep" if i % 4 == 0 else "discard",
            "metrics": {"val_bpb": 1.5 - i * 0.1},
            "params": {"lr": 0.01, "batch_size": 32},
        })
    return history


def test_state_feature_extraction_shape() -> None:
    policy = LearnedParamPolicy(_param_space())
    features = policy._extract_state_features(_sample_history(10))
    assert features.shape == (STATE_DIM,)
    assert features.dtype == np.float64


def test_state_feature_extraction_empty_history() -> None:
    policy = LearnedParamPolicy(_param_space())
    features = policy._extract_state_features([])
    assert features.shape == (STATE_DIM,)
    assert np.allclose(features, 0.0)


def test_state_feature_extraction_values() -> None:
    policy = LearnedParamPolicy(_param_space())
    history = [
        {"iter": 0, "status": "ok", "decision": "keep",
         "metrics": {"val_bpb": 1.0}, "params": {}},
        {"iter": 1, "status": "ok", "decision": "keep",
         "metrics": {"val_bpb": 0.9}, "params": {}},
    ]
    features = policy._extract_state_features(history)
    assert features[0] == 1.0
    assert features[1] == 0.9
    assert features[8] == 2.0  # streak
    assert features[9] == 0.0  # fail count
    assert abs(features[10] - 0.02) < 1e-9  # iter norm


def test_policy_returns_valid_proposal() -> None:
    policy = LearnedParamPolicy(_param_space())
    proposal = policy.propose({"history": _sample_history()})
    assert isinstance(proposal, ParamProposal)
    assert "lr" in proposal.params
    assert "batch_size" in proposal.params
    assert proposal.params["lr"] in [0.001, 0.01, 0.1]
    assert proposal.params["batch_size"] in [16, 32]
    assert proposal.rationale == "learned"


def test_record_reward_stores_transitions() -> None:
    policy = LearnedParamPolicy(
        _param_space(),
        LearnedSearchConfig(update_every=100),
    )
    policy.propose({"history": _sample_history()})
    assert policy._pending is not None
    policy.record_reward(1.0)
    assert policy._pending is None
    assert policy.buffer_size == 1


def test_record_reward_no_pending() -> None:
    policy = LearnedParamPolicy(_param_space())
    policy.record_reward(1.0)
    assert policy.buffer_size == 0


def test_ppo_update_triggers() -> None:
    cfg = LearnedSearchConfig(
        ppo=PPOConfig(epochs=1, hidden_dim=4, n_layers=1, batch_size=4),
        update_every=4,
        snapshot_every=1000,
    )
    policy = LearnedParamPolicy(_param_space(), cfg)

    for i in range(4):
        policy.propose({"history": _sample_history(i)})
        policy.record_reward(1.0 if i % 2 == 0 else -0.1)

    assert policy.buffer_size == 0
    assert policy._update_count == 1


def test_action_space_covers_all_combinations() -> None:
    space = {"lr": [0.001, 0.01], "batch_size": [16, 32, 64]}
    policy = LearnedParamPolicy(space)
    assert policy.action_dim == 6
    assert len(policy._combos) == 6
