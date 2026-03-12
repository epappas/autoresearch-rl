from autoresearch_rl.policy.baselines import GreedyLLMPolicy, RandomPolicy


def test_random_policy_returns_diff():
    p = RandomPolicy(seed=1)
    d = p.propose_diff({"iter": 0})
    assert d.startswith("diff --git a/train.py b/train.py")


def test_greedy_policy_bootstrap_and_threshold():
    p = GreedyLLMPolicy(improve_threshold=1.3)
    d0 = p.propose_diff({"best_score": None})
    d1 = p.propose_diff({"best_score": 1.5})
    d2 = p.propose_diff({"best_score": 1.2})

    assert "use_qk_norm" in d0
    assert "use_qk_norm" in d1
    assert "grad_clip" in d2
