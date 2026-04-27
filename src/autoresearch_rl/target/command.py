from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from autoresearch_rl.eval.metrics import parse_metrics
from autoresearch_rl.target.interface import RunOutcome, TargetAdapter
from autoresearch_rl.target.progress import CONTROL_ENV, PROGRESS_ENV
from autoresearch_rl.target.progress_reader import ProgressReader

logger = logging.getLogger(__name__)


@dataclass
class CommandTarget(TargetAdapter):
    train_cmd: list[str]
    eval_cmd: list[str] | None
    workdir: str
    timeout_s: int
    prepare_cmd: list[str] | None = None
    _prepared: bool = field(default=False, init=False, repr=False)

    def _ensure_prepared(self) -> None:
        if self._prepared or not self.prepare_cmd:
            return
        logger.info("Running prepare_cmd: %s", self.prepare_cmd)
        cp = subprocess.run(
            self.prepare_cmd,
            cwd=self.workdir,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
        )
        if cp.returncode != 0:
            logger.error("prepare_cmd failed (rc=%d): %s", cp.returncode, cp.stderr[:500])
            raise RuntimeError(f"prepare_cmd failed: {cp.stderr[:200]}")
        logger.info("prepare_cmd completed")
        self._prepared = True

    def _run(self, *, cmd: list[str], run_dir: str, params: dict[str, object]) -> RunOutcome:
        env = os.environ.copy()
        env["AR_RUN_DIR"] = run_dir
        if "AR_SEED" in os.environ:
            env["AR_SEED"] = os.environ["AR_SEED"]
        if "PYTHONHASHSEED" in os.environ:
            env["PYTHONHASHSEED"] = os.environ["PYTHONHASHSEED"]
        env["AR_PARAMS_JSON"] = json.dumps(params)
        for k, v in params.items():
            env[f"AR_PARAM_{str(k).upper()}"] = str(v)
        if "AR_MODEL_DIR" in params:
            env["AR_MODEL_DIR"] = str(params["AR_MODEL_DIR"])

        # Per-worker progress + control paths. Convention: each iter owns
        # $run_dir/progress.jsonl and $run_dir/control.json. Worker threads
        # never share env state through os.environ, so the parallel engine
        # is race-free (each subprocess inherits its own per-call env dict).
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        progress_path = env.get(PROGRESS_ENV) or str(Path(run_dir) / "progress.jsonl")
        control_path = env.get(CONTROL_ENV) or str(Path(run_dir) / "control.json")
        env[PROGRESS_ENV] = progress_path
        env[CONTROL_ENV] = control_path
        reader = ProgressReader(progress_path, poll_interval_s=0.5)
        reader.start()

        start = time.monotonic()
        try:
            cp = subprocess.run(
                cmd,
                cwd=self.workdir,
                capture_output=True,
                text=True,
                env=env,
                timeout=self.timeout_s,
                check=False,
            )
        finally:
            reader.stop()
        elapsed = time.monotonic() - start
        stdout = cp.stdout or ""
        stderr = cp.stderr or ""
        metrics = parse_metrics(stdout + "\n" + stderr)
        metrics_dict = {k: v for k, v in vars(metrics).items() if v is not None}
        if not metrics_dict:
            for line in (stdout + "\n" + stderr).splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    try:
                        metrics_dict[k] = float(v)
                    except ValueError:
                        continue

        # Backfill metrics from latest progress report if stdout/stderr were silent.
        latest = reader.latest()
        if latest is not None and latest.metrics:
            for k, v in latest.metrics.items():
                metrics_dict.setdefault(k, float(v))

        status = "ok" if cp.returncode == 0 else "failed"
        return RunOutcome(
            status=status, metrics=metrics_dict, stdout=stdout,
            stderr=stderr, elapsed_s=elapsed, run_dir=run_dir,
        )

    def run(self, *, run_dir: str, params: dict[str, object]) -> RunOutcome:
        self._ensure_prepared()
        return self._run(cmd=self.train_cmd, run_dir=run_dir, params=params)

    def eval(self, *, run_dir: str, params: dict[str, object]) -> RunOutcome:
        if not self.eval_cmd:
            return self.run(run_dir=run_dir, params=params)
        return self._run(cmd=self.eval_cmd, run_dir=run_dir, params=params)


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
