"""Determinism check for examples/parallel-cancel-showcase.

Two distinct contracts:

A. config-deterministic.yaml (cancel DISABLED): two runs with the same
   seed must produce identical params, decisions, kept versions, AND
   best_value. The parallel engine itself is deterministic when cancel
   timing is removed.

B. config.yaml (cancel ENABLED): two runs produce identical params and
   identical best_value, but the cancellation set is allowed to differ
   because cancel timing depends on filesystem-polling jitter. The
   forecaster's decision can flip on a trial whose early-step loss
   straddles the current best.

Allowed to differ in BOTH contracts:
  - exact metric value at the moment a cancelled trial exits
  - episode_id (uuid per run)
  - elapsed_s timings
  - timeline span timestamps

If the strict (cancel-disabled) test ever flakes, that's a real
determinism regression in the parallel engine. The main showcase test
makes weaker but still useful assertions.
"""
from __future__ import annotations

import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = REPO_ROOT / "examples" / "parallel-cancel-showcase"
ARTIFACTS = SHOWCASE / "artifacts"


def _params_per_iter(artifacts_dir: Path) -> dict[int, dict]:
    """Read per-iter manifest.json files and return {iter: params}."""
    out: dict[int, dict] = {}
    for mf in glob.glob(str(artifacts_dir / "runs" / "*" / "manifest.json")):
        data = json.loads(Path(mf).read_text())
        if "iter" in data:
            out[int(data["iter"])] = dict(data.get("params", {}))
    return out


def _decisions_per_iter(artifacts_dir: Path) -> dict[int, str]:
    """Read results.tsv and return {iter: decision}."""
    out: dict[int, str] = {}
    rows = (artifacts_dir / "results.tsv").read_text().splitlines()
    header = rows[0].split("\t")
    iter_col = header.index("iter")
    status_col = header.index("status")
    for row in rows[1:]:
        cols = row.split("\t")
        out[int(cols[iter_col])] = cols[status_col]
    return out


def _kept_versions(artifacts_dir: Path) -> set[str]:
    return {p.name for p in (artifacts_dir / "versions").iterdir() if p.is_dir()}


def _best_value(artifacts_dir: Path) -> float | None:
    """Min metric across kept versions."""
    bests = []
    for vd in (artifacts_dir / "versions").iterdir():
        if not vd.is_dir():
            continue
        meta = json.loads((vd / "version.json").read_text())
        m = meta.get("metrics", {}).get("val_loss")
        if m is not None:
            bests.append(float(m))
    return min(bests) if bests else None


def _run_showcase(label: str, config_name: str = "config.yaml") -> Path:
    """Run the showcase from the repo root, copy artifacts to /tmp/<label>."""
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)
    cp = subprocess.run(
        [
            sys.executable, "-m", "autoresearch_rl.cli", "run",
            str(SHOWCASE / config_name),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=120,
    )
    if cp.returncode != 0:
        pytest.fail(f"showcase {label} failed: {cp.stderr[-500:]}")
    dest = Path(f"/tmp/showcase_det_{label}")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(ARTIFACTS, dest)
    return dest


def test_parallel_engine_is_strictly_deterministic_without_cancel() -> None:
    """Cancel disabled -> params, decisions, versions, best_value all bit-identical."""
    if not SHOWCASE.exists():
        pytest.skip("showcase example not present")

    a = _run_showcase("strict_a", config_name="config-deterministic.yaml")
    b = _run_showcase("strict_b", config_name="config-deterministic.yaml")

    params_a = _params_per_iter(a)
    params_b = _params_per_iter(b)
    assert params_a == params_b, (
        f"params diverged between runs: A keys={sorted(params_a)} "
        f"B keys={sorted(params_b)}"
    )

    decisions_a = _decisions_per_iter(a)
    decisions_b = _decisions_per_iter(b)
    assert decisions_a == decisions_b, (
        f"decisions diverged: A={decisions_a} B={decisions_b}"
    )

    versions_a = _kept_versions(a)
    versions_b = _kept_versions(b)
    assert versions_a == versions_b, (
        f"kept versions diverged: A={versions_a} B={versions_b}"
    )

    best_a = _best_value(a)
    best_b = _best_value(b)
    assert best_a == best_b, f"best_value diverged: A={best_a} B={best_b}"


def test_showcase_with_cancel_keeps_params_and_best_value_stable() -> None:
    """Cancel enabled -> params + best_value still identical across runs;
    cancellation set is allowed to differ (cancel timing is racy)."""
    if not SHOWCASE.exists():
        pytest.skip("showcase example not present")

    a = _run_showcase("a")
    b = _run_showcase("b")

    params_a = _params_per_iter(a)
    params_b = _params_per_iter(b)
    assert params_a == params_b, (
        f"params diverged between runs: A keys={sorted(params_a)} "
        f"B keys={sorted(params_b)}"
    )

    # best_value across runs: must be identical because the same set of
    # params is sampled, and any iter that survives cancellation produces
    # a deterministic loss (no randomness in the synthetic train.py).
    best_a = _best_value(a)
    best_b = _best_value(b)
    assert best_a == best_b, (
        f"best_value diverged: A={best_a} B={best_b} — "
        f"either cancel killed the optimum in one run only "
        f"(tune intra_iteration_cancel.min_steps to fix), or randomness "
        f"leaked into train.py"
    )
