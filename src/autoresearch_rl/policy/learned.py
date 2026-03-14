from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from autoresearch_rl.policy.interface import Proposal
from autoresearch_rl.sandbox.diff_utils import extract_touched_files_from_diff


def _diff_features(diff: str) -> list[float]:
    lines = diff.splitlines()
    adds = sum(1 for l in lines if l.startswith("+"))
    dels = sum(1 for l in lines if l.startswith("-"))
    files = len(extract_touched_files_from_diff(diff))
    length = len(diff)
    return [1.0, float(files), float(adds), float(dels), float(length)]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


@dataclass
class LearnedDiffPolicy:
    base_policy: object
    weights_path: str
    lr: float = 0.01

    def _load_weights(self) -> list[float]:
        p = Path(self.weights_path)
        if not p.exists():
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        return json.loads(p.read_text(encoding="utf-8"))

    def _save_weights(self, w: list[float]) -> None:
        Path(self.weights_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.weights_path).write_text(json.dumps(w), encoding="utf-8")

    def propose(self, state: dict, pool_size: int = 4) -> Proposal:
        candidates: list[Proposal] = []
        for _ in range(pool_size):
            p = self.base_policy.propose(state)
            candidates.append(p)
        weights = self._load_weights()
        scores = [_dot(_diff_features(p.diff), weights) for p in candidates]
        probs = _softmax(scores)
        idx = max(range(len(candidates)), key=lambda i: probs[i])
        chosen = candidates[idx]
        chosen.rationale += f"|learned_prob={probs[idx]:.4f}"
        return chosen

    def propose_diff(self, state: dict) -> str:
        return self.propose(state).diff

    def logp(self, diff: str) -> float:
        weights = self._load_weights()
        return _dot(_diff_features(diff), weights)

    def update(self, samples: Iterable[dict]) -> None:
        weights = self._load_weights()
        for s in samples:
            diff = s.get("diff", "")
            reward = float(s.get("reward", 0.0))
            old_logp = float(s.get("logp", 0.0))
            feats = _diff_features(diff)
            score = _dot(feats, weights)
            logp = score  # unnormalized logit
            ratio = math.exp(logp - old_logp)
            clipped = max(0.8, min(1.2, ratio))
            grad_scale = -min(ratio, clipped) * reward
            weights = [w - self.lr * grad_scale * f for w, f in zip(weights, feats)]
        self._save_weights(weights)
