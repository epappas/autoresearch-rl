"""End-to-end smoke tests for in-tree examples.

Two tiers:

Tier 1 — full run (~5-10 s each)
    Examples that work CPU-only with stub credentials. The LLM policies
    fall back to greedy/random when CHUTES_API_KEY is invalid. The full
    keep/discard plumbing runs.

Tier 2 — validate only (~1 s each)
    Examples that need GPU (basilica target) or heavy ML deps (datasets,
    transformers, real model downloads). For these we run
    `autoresearch-rl validate` which exercises config_validate plus
    target construction up to but not including the run loop. That's
    enough to catch:
      - the contract path-comparison class (validator runs on diff)
      - config schema regressions
      - tracked-path overwrite warnings
      - BasilicaTarget construction failures (e.g., SDK API changes)

Defends against the class of bug surfaced in commit fef66d1: every
unit test passed because fixtures used basename-only contract paths,
but every real example silently rejected every diff because actual
configs use workdir-prefixed paths. The full-run tier asserts
best_value != None so this cannot regress.

If you add a new example, add it here. The cost is small; the bug
class is expensive.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _isolate(example_dir: str, tmp_path: Path) -> tuple[Path, Path]:
    """Copy an example to tmp_path and rewrite workdir-prefixed paths to
    be relative to the sandbox. Subprocess(cwd=sandbox_root) then keeps
    the repo's tree clean — important because greedy LLM-fallback diffs
    append text to the trial source.
    """
    src = REPO_ROOT / example_dir
    dst_dir = tmp_path / "example"
    shutil.copytree(src, dst_dir)
    cfg_text = (dst_dir / "config.yaml").read_text()
    cfg_text = cfg_text.replace(f"workdir: {example_dir}", "workdir: .")
    cfg_text = cfg_text.replace(f"{example_dir}/", "")
    cfg_text = cfg_text.replace(
        f"program_path: {example_dir}/program.md",
        "program_path: program.md",
    )
    (dst_dir / "config.yaml").write_text(cfg_text)
    return dst_dir, dst_dir / "config.yaml"


def _cli(
    sandbox_root: Path, *cli_args: str,
    extra_env: dict | None = None, timeout: int = 120,
) -> subprocess.CompletedProcess:
    cmd_env = dict(os.environ)
    cmd_env["PYTHONPATH"] = str(REPO_ROOT / "src")
    if extra_env:
        cmd_env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "autoresearch_rl.cli", *cli_args],
        cwd=str(sandbox_root),
        capture_output=True, text=True, timeout=timeout, env=cmd_env,
    )


def _parse_run_json(stdout: str) -> dict:
    start = stdout.rfind("{")
    end = stdout.rfind("}")
    assert start >= 0 and end > start, f"no JSON in stdout:\n{stdout[-500:]}"
    return json.loads(stdout[start : end + 1])


# ---------------------------------------------------------------- Tier 1: full run


# (example_dir, runs_to_completion_in_ci_with_stub_credentials)
TIER1_FULL_RUN = [
    "examples/minimal-trainable-target",
    "examples/autoresearch-like",
]


@pytest.mark.parametrize("example_dir", TIER1_FULL_RUN)
def test_llm_diff_example_produces_real_best_value(
    example_dir: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Real CLI run must return best_value != None.

    Defends against commit fef66d1: the contract validator silently
    rejected every diff with out_of_scope_mutation_blocked, every
    campaign returned best_value=None.

    Uses a stub API key so LLMDiffPolicy attempts the call and falls
    back to greedy on auth failure — that path exercises the contract
    validator + diff-apply + run-trial chain end-to-end.
    """
    monkeypatch.setenv("CHUTES_API_KEY", "stub")

    sandbox_root, config_path = _isolate(example_dir, tmp_path)
    cp = _cli(
        sandbox_root, "run", str(config_path),
        "--override", "controller.max_iterations=2",
    )
    if cp.returncode != 0:
        pytest.fail(
            f"{example_dir} exited non-zero ({cp.returncode}):\n"
            f"--- stdout ---\n{cp.stdout[-1500:]}\n"
            f"--- stderr ---\n{cp.stderr[-1500:]}"
        )
    result = _parse_run_json(cp.stdout)
    assert result["iterations"] == 2, result
    assert result["best_value"] is not None, (
        f"{example_dir} returned best_value=None — the contract validator "
        f"may be silently rejecting all diffs again. Result: {result}"
    )


# ---------------------------------------------------------------- Tier 2: validate only


# Examples that need real GPU / heavy ML deps to actually run, but whose
# config + target-construction path we can still smoke-test via 'validate'.
TIER2_VALIDATE_ONLY = [
    "examples/basilica-grpo",
    "examples/security-judge",
    "examples/deberta-prompt-injection",
]


@pytest.mark.parametrize("example_dir", TIER2_VALIDATE_ONLY)
def test_example_validates_cleanly_with_stub_credentials(
    example_dir: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """`autoresearch-rl validate` exits 0 with stub credentials.

    Catches: config schema regressions, target-construction failures
    (e.g., basilica-sdk import failures, BasilicaConfig validator
    issues), and any new blocking ValidationError that lands in
    config_validate. Doesn't run the loop — that needs a real GPU /
    real LLM / real datasets that aren't appropriate for CI.

    Stub credentials are sufficient because config_validate only checks
    env-var presence, not validity. Real auth happens at run-time.
    """
    monkeypatch.setenv("CHUTES_API_KEY", "stub")
    monkeypatch.setenv("BASILICA_API_KEY", "stub")

    sandbox_root, config_path = _isolate(example_dir, tmp_path)
    cp = _cli(sandbox_root, "validate", str(config_path), timeout=30)

    # Validate exits 0 on success, 2 on blocking error. We tolerate
    # warnings (severity=warn) — those are advisory and shouldn't fail
    # the smoke. But ANY blocking error is a real regression.
    assert cp.returncode == 0, (
        f"{example_dir} validate failed ({cp.returncode}):\n"
        f"--- stdout ---\n{cp.stdout}\n--- stderr ---\n{cp.stderr}"
    )
    # The CLI prints "OK" on success.
    assert "OK" in cp.stdout, f"validate exited 0 but no 'OK' marker:\n{cp.stdout}"
