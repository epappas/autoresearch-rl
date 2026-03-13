from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from autoresearch_rl.eval.metrics import parse_metrics
from autoresearch_rl.target.interface import RunOutcome, TargetAdapter


@dataclass
class CommandTarget(TargetAdapter):
    train_cmd: list[str]
    eval_cmd: list[str] | None
    workdir: str
    timeout_s: int

    def _run(self, *, cmd: list[str], run_dir: str, params: dict[str, object]) -> RunOutcome:
        env = os.environ.copy()
        env["AR_RUN_DIR"] = run_dir
        env["AR_PARAMS_JSON"] = json.dumps(params)
        for k, v in params.items():
            env[f"AR_PARAM_{str(k).upper()}"] = str(v)

        start = time.monotonic()
        cp = subprocess.run(
            cmd,
            cwd=self.workdir,
            capture_output=True,
            text=True,
            env=env,
            timeout=self.timeout_s,
            check=False,
        )
        elapsed = time.monotonic() - start
        stdout = cp.stdout or ""
        stderr = cp.stderr or ""
        metrics = parse_metrics(stdout + "\n" + stderr)
        metrics_dict = {k: v for k, v in vars(metrics).items() if v is not None}
        status = "ok" if cp.returncode == 0 else "failed"
        return RunOutcome(status=status, metrics=metrics_dict, stdout=stdout, stderr=stderr, elapsed_s=elapsed, run_dir=run_dir)

    def run(self, *, run_dir: str, params: dict[str, object]) -> RunOutcome:
        return self._run(cmd=self.train_cmd, run_dir=run_dir, params=params)

    def eval(self, *, run_dir: str, params: dict[str, object]) -> RunOutcome:
        if not self.eval_cmd:
            return self.run(run_dir=run_dir, params=params)
        return self._run(cmd=self.eval_cmd, run_dir=run_dir, params=params)


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
