from pathlib import Path

from autoresearch_rl.policy.baselines import GreedyLLMPolicy, RandomDiffPolicy


def _state(tmp_path: Path) -> dict:
    p = tmp_path / "train.py"
    p.write_text("LEARNING_RATE = 0.0026\n", encoding="utf-8")
    return {"workdir": str(tmp_path), "mutable_file": "train.py", "best_score": None}


def test_random_policy_returns_diff(tmp_path: Path):
    s = _state(tmp_path)
    p = RandomDiffPolicy(seed=1)
    d = p.propose(s).diff
    assert d.startswith("--- a/train.py")


def test_greedy_policy_bootstrap_and_threshold(tmp_path: Path):
    s = _state(tmp_path)
    p = GreedyLLMPolicy(improve_threshold=1.3)

    d0 = p.propose({**s, "best_score": None}).diff
    d1 = p.propose({**s, "best_score": 1.5}).diff
    d2 = p.propose({**s, "best_score": 1.2}).diff
    d3 = p.propose({**s, "best_score": 1.2, "no_improve_streak": 3, "history": [{"status": "failed"}, {"status": "timeout"}]}).diff

    assert "use_qk_norm" in d0
    assert "use_qk_norm" in d1
    assert "GRAD_CLIP" in d2
    assert "LEARNING_RATE = 0.0020" in d3
