from __future__ import annotations

import difflib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from autoresearch_rl.policy.interface import Proposal


def _build_patch(path: Path, new_text: str) -> str:
    old_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    udiff = "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
            lineterm="\n",
        )
    )
    return udiff if udiff.strip() else f"diff --git a/{path.name} b/{path.name}\n"


def _target_path(state: Mapping[str, object]) -> Path:
    workdir = str(state.get("workdir", "."))
    mutable = str(state.get("mutable_file", "train.py"))
    p = Path(mutable)
    if p.is_absolute():
        return p
    return Path(workdir) / p.name


def _recent_statuses(state: Mapping[str, object]) -> list[str]:
    h = state.get("history", [])
    if not isinstance(h, list):
        return []
    out: list[str] = []
    for item in h[-8:]:
        if isinstance(item, dict):
            s = item.get("status")
            if isinstance(s, str):
                out.append(s)
    return out


@dataclass
class RandomPolicy:
    seed: int = 7

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def propose(self, state: Mapping[str, object]) -> Proposal:
        path = _target_path(state)
        text = path.read_text(encoding="utf-8")
        lr = self._rng.choice(["0.0020", "0.0023", "0.0026", "0.0029"])

        if "LEARNING_RATE =" in text:
            new_text = text.replace("LEARNING_RATE = 0.0026", f"LEARNING_RATE = {lr}")
            if new_text == text:
                new_text = text + f"\n# learning_rate_candidate={lr}\n"
        else:
            new_text = text + f"\n# learning_rate_candidate={lr}\n"

        return Proposal(diff=_build_patch(path, new_text), rationale="random_lr_choice")

    def propose_diff(self, state: Mapping[str, object]) -> str:
        return self.propose(state).diff


@dataclass
class GreedyLLMPolicy:
    improve_threshold: float = 1.3

    def propose(self, state: Mapping[str, object]) -> Proposal:
        path = _target_path(state)
        text = path.read_text(encoding="utf-8")

        best = state.get("best_score")
        try:
            best_f = float(best) if best is not None else float("inf")
        except (TypeError, ValueError):
            best_f = float("inf")

        no_improve = int(state.get("no_improve_streak", 0) or 0)
        recent = _recent_statuses(state)
        recent_failures = sum(1 for s in recent if s in {"failed", "timeout", "rejected"})

        if no_improve >= 3 or recent_failures >= 2:
            # back off: try lower LR for stability
            if "LEARNING_RATE =" in text:
                new_text = text.replace("LEARNING_RATE = 0.0026", "LEARNING_RATE = 0.0020")
                if new_text == text:
                    new_text = text + "\nLEARNING_RATE = 0.0020\n"
            else:
                new_text = text + "\nLEARNING_RATE = 0.0020\n"
            rationale = "backoff_after_failures"
        elif best_f > self.improve_threshold:
            if "use_qk_norm = True" not in text:
                new_text = text + "\nuse_qk_norm = True\n"
            else:
                new_text = text + "\n# keep_qk_norm\n"
            rationale = "improve_stability_before_fine_tuning"
        else:
            if "GRAD_CLIP =" in text:
                new_text = text.replace("GRAD_CLIP = 1.0", "GRAD_CLIP = 0.8")
                if new_text == text:
                    new_text = text + "\nGRAD_CLIP = 0.8\n"
            else:
                new_text = text + "\nGRAD_CLIP = 0.8\n"
            rationale = "tighten_optimization"

        return Proposal(diff=_build_patch(path, new_text), rationale=rationale)

    def propose_diff(self, state: Mapping[str, object]) -> str:
        return self.propose(state).diff
