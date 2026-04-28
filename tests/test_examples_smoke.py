"""End-to-end smoke tests for the in-tree CPU examples.

Defends against the class of bug that hid the contract path-comparison
regression for weeks: every unit test passed because the contract
fixtures used basename-only paths, but every real example
(workdir-prefixed paths) silently rejected every diff. The campaigns
ran to max_iter and returned best_value=None — looked fine in logs;
totally broken in practice.

These tests run the actual CLI against the actual example configs and
assert the loop produces a real best_value (not None). They run with
no API key — the LLM policies fall back to greedy, which is enough to
exercise the contract validator and the full keep/discard plumbing.

Each example takes ~5-10 s. If you add a new example that uses an LLM
policy, please add it here — the cost is small and the bug class is
expensive.
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
    """Copy an example to tmp_path. Returns (sandbox_root, sandbox_config_path).

    The sandbox uses workdir-relative paths so subprocess(cwd=sandbox_root)
    produces artifacts inside the sandbox. The repo's example dir is never
    modified — important because greedy LLM-fallback diffs append text to
    the trial source.
    """
    src = REPO_ROOT / example_dir
    dst_dir = tmp_path / "example"
    shutil.copytree(src, dst_dir)
    # The shipped config uses workdir 'examples/<name>'. Rewrite to '.'.
    cfg_text = (dst_dir / "config.yaml").read_text()
    cfg_text = cfg_text.replace(f"workdir: {example_dir}", "workdir: .")
    cfg_text = cfg_text.replace(f"{example_dir}/", "")
    cfg_text = cfg_text.replace(
        f"program_path: {example_dir}/program.md",
        "program_path: program.md",
    )
    (dst_dir / "config.yaml").write_text(cfg_text)
    return dst_dir, dst_dir / "config.yaml"


def _run_example(
    sandbox_root: Path, config_path: Path, *, max_iter: int = 2,
    extra_env: dict | None = None,
) -> dict:
    """Run autoresearch-rl from sandbox_root; return parsed JSON output."""
    cmd_env = dict(os.environ)  # inherit caller env (incl. monkeypatch'd keys)
    cmd_env["PYTHONPATH"] = str(REPO_ROOT / "src")
    if extra_env:
        cmd_env.update(extra_env)
    cp = subprocess.run(
        [
            sys.executable, "-m", "autoresearch_rl.cli", "run",
            str(config_path),
            "--override", f"controller.max_iterations={max_iter}",
        ],
        cwd=str(sandbox_root),
        capture_output=True, text=True, timeout=120, env=cmd_env,
    )
    if cp.returncode != 0:
        pytest.fail(
            f"example {config_path} exited non-zero ({cp.returncode}):\n"
            f"--- stdout (last 30 lines) ---\n{chr(10).join(cp.stdout.splitlines()[-30:])}\n"
            f"--- stderr (last 30 lines) ---\n{chr(10).join(cp.stderr.splitlines()[-30:])}"
        )
    text = cp.stdout
    start = text.rfind("{")
    end = text.rfind("}")
    assert start >= 0 and end > start, f"no JSON in stdout:\n{text[-500:]}"
    return json.loads(text[start : end + 1])


@pytest.mark.parametrize("example_dir", [
    "examples/minimal-trainable-target",
])
def test_llm_diff_example_produces_real_best_value(
    example_dir: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Real CLI run of an llm_diff example must return best_value != None.

    The bug this defends against (commit fef66d1): every llm_diff example
    silently produced best_value=None because the contract validator
    rejected every basename-vs-prefix diff with out_of_scope_mutation_blocked.
    Fixing the contract was the immediate fix; this test ensures the same
    class of regression cannot land again without CI noticing.
    """
    # Fake API key so LLMDiffPolicy attempts the call (then falls back to
    # greedy on auth failure). Without ANY key the policy short-circuits to
    # random fallback; with a stub key it goes through the full chat-API
    # error path which is more representative.
    monkeypatch.setenv("CHUTES_API_KEY", "stub")

    sandbox_root, config_path = _isolate(example_dir, tmp_path)
    result = _run_example(sandbox_root, config_path, max_iter=2)

    assert result["iterations"] == 2, result
    assert result["best_value"] is not None, (
        f"best_value is None — the contract validator may be silently "
        f"rejecting all diffs again. Result: {result}"
    )
