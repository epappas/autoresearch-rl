from __future__ import annotations

import subprocess
from pathlib import Path

from autoresearch_rl.sandbox.runner import EarlyStopConfig, run_trial


def test_rejects_forbidden_diff():
    r = run_trial("import socket", timeout_s=1)
    assert r.status == "rejected"


def test_rejects_missing_command():
    r = run_trial("diff --git a/train.py b/train.py\n+ x=1", timeout_s=1)
    assert r.status == "rejected"
    assert "command_required" in r.stderr


def test_early_stop_triggers_on_bad_threshold():
    cmd = [
        "python3",
        "-c",
        "import time; print('val_bpb=9.0', flush=True); time.sleep(2)",
    ]
    r = run_trial(
        "diff --git a/train.py b/train.py\n+ # noop",
        timeout_s=10,
        command=cmd,
        early_stop=EarlyStopConfig(enabled=True, check_every_s=0.1, min_runtime_s=0.0, val_bpb_threshold=2.0),
    )
    assert r.status == "early_stopped"


def test_auto_inits_git_when_missing(tmp_path: Path):
    repo = tmp_path / "nogit"
    repo.mkdir(parents=True)
    f = repo / "train.py"
    f.write_text("x = 1\n", encoding="utf-8")

    diff = (
        "diff --git a/train.py b/train.py\n"
        "index 1f206b1..b8443d8 100644\n"
        "--- a/train.py\n"
        "+++ b/train.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )

    r = run_trial(
        diff=diff,
        timeout_s=5,
        command=["python3", "-c", "print('val_bpb=1.0')"],
        workdir=str(repo),
        apply_patch=True,
        rollback_patch=True,
        auto_init_git=True,
    )

    assert r.status == "ok"
    assert r.patch_applied is True
    assert (repo / ".git").exists()
    assert f.read_text(encoding="utf-8") == "x = 1\n"


def test_apply_patch_and_rollback(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    f = repo / "train.py"
    untouched = repo / "notes.txt"
    f.write_text("x = 1\n", encoding="utf-8")
    untouched.write_text("base\n", encoding="utf-8")

    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@x"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "add", "train.py", "notes.txt"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True, capture_output=True)

    # Unrelated dirty change that must survive rollback.
    untouched.write_text("dirty-unrelated-change\n", encoding="utf-8")

    diff = (
        "diff --git a/train.py b/train.py\n"
        "index 1f206b1..b8443d8 100644\n"
        "--- a/train.py\n"
        "+++ b/train.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )

    r = run_trial(
        diff=diff,
        timeout_s=5,
        command=["python3", "-c", "print('val_bpb=1.0')"],
        workdir=str(repo),
        apply_patch=True,
        rollback_patch=True,
    )

    assert r.status == "ok"
    assert r.patch_applied is True
    # ensure rollback restored touched file only
    assert f.read_text(encoding="utf-8") == "x = 1\n"
    # ensure unrelated local modification was NOT wiped
    assert untouched.read_text(encoding="utf-8") == "dirty-unrelated-change\n"
