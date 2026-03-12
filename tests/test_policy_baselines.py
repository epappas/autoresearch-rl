from pathlib import Path

from autoresearch_rl.policy.baselines import GreedyLLMPolicy, RandomPolicy


def _state(tmp_path: Path) -> dict:
    p = tmp_path / "train.py"
    p.write_text("LEARNING_RATE = 0.0026\n", encoding="utf-8")
    return {"workdir": str(tmp_path), "mutable_file": "train.py", "best_score": None}


def test_random_policy_returns_diff(tmp_path: Path):
    s = _state(tmp_path)
    p = RandomPolicy(seed=1)
    d = p.propose_diff(s)
    assert d.startswith("--- a/train.py")


def test_greedy_policy_bootstrap_and_threshold(tmp_path: Path):
    s = _state(tmp_path)
    p = GreedyLLMPolicy(improve_threshold=1.3)

    d0 = p.propose_diff({**s, "best_score": None})
    d1 = p.propose_diff({**s, "best_score": 1.5})
    d2 = p.propose_diff({**s, "best_score": 1.2})

    assert "use_qk_norm" in d0
    assert "use_qk_norm" in d1
    assert "GRAD_CLIP" in d2
