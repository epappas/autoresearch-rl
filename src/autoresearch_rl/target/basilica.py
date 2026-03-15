"""Basilica GPU cloud target adapter.

Deploys training jobs on Basilica infrastructure, waits for completion,
and extracts metrics from deployment logs.

The adapter wraps the user's training command in a bootstrap script that:
1. Starts a health-check HTTP server (Basilica requires this to mark "ready")
2. Runs the training command as a subprocess
3. Keeps alive until training completes and logs are flushed
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid

from autoresearch_rl.config import TargetConfig
from autoresearch_rl.target.interface import RunOutcome

logger = logging.getLogger(__name__)

HEALTH_PORT = 8080

# Bootstrap script injected into every Basilica deployment.
# Starts a health-check server, then runs the user command via subprocess.
_BOOTSTRAP = r"""
import subprocess, sys, threading, time
from http.server import HTTPServer, BaseHTTPRequestHandler

class _H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")
    def log_message(self, *a):
        pass

threading.Thread(
    target=lambda: HTTPServer(("", {port}), _H).serve_forever(),
    daemon=True,
).start()

rc = subprocess.call({cmd}, env=dict(**__import__("os").environ))
sys.stdout.flush()
sys.stderr.flush()
time.sleep(15)
sys.exit(rc)
"""


class BasilicaTarget:
    """Run training iterations on Basilica GPU cloud."""

    def __init__(self, cfg: TargetConfig) -> None:
        try:
            from basilica import BasilicaClient
        except ImportError as e:
            raise ImportError(
                "basilica-sdk required. Install: pip install basilica-sdk"
            ) from e

        self._client = BasilicaClient()
        self._cfg = cfg
        self._bcfg = cfg.basilica
        self._last_train_outcome: RunOutcome | None = None

    def run(self, *, run_dir: str, params: dict[str, object]) -> RunOutcome:
        outcome = self._deploy_and_collect(
            params=params, run_dir=run_dir,
            cmd=self._cfg.train_cmd, phase="train",
        )
        self._last_train_outcome = outcome
        return outcome

    def eval(self, *, run_dir: str, params: dict[str, object]) -> RunOutcome:
        if not self._cfg.eval_cmd:
            if self._last_train_outcome is not None:
                return self._last_train_outcome
            return RunOutcome(
                status="ok", metrics={}, stdout="", stderr="",
                elapsed_s=0.0, run_dir=run_dir,
            )
        return self._deploy_and_collect(
            params=params, run_dir=run_dir,
            cmd=self._cfg.eval_cmd, phase="eval",
        )

    def _build_bootstrap_cmd(self, user_cmd: list[str]) -> str:
        """Wrap user command in bootstrap that starts health server."""
        setup = self._bcfg.setup_cmd
        setup_block = ""
        if setup:
            setup_block = (
                f"\nimport subprocess as _sp\n"
                f"_sp.check_call({repr(setup)}, shell=True)\n"
            )
        cmd_repr = repr(user_cmd)
        script = _BOOTSTRAP.format(port=HEALTH_PORT, cmd=cmd_repr)
        # Insert setup block after the health server start
        marker = "threading.Thread("
        idx = script.index(marker)
        # Find the end of the threading block (after .start())
        start_idx = script.index(".start()", idx) + len(".start()\n")
        script = script[:start_idx] + setup_block + script[start_idx:]
        return script

    def _deploy_and_collect(
        self,
        params: dict[str, object],
        run_dir: str,
        cmd: list[str] | None,
        phase: str,
    ) -> RunOutcome:
        from basilica import (
            Deployment, HealthCheckConfig, ProbeConfig,
        )

        tag = uuid.uuid4().hex[:8]
        name = f"ar-{phase}-{tag}"
        t0 = time.monotonic()

        env: dict[str, str] = {
            "AR_PARAMS_JSON": json.dumps(params, default=str),
        }
        for k, v in params.items():
            env[f"AR_PARAM_{str(k).upper()}"] = str(v)

        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            env["HF_TOKEN"] = hf_token

        user_cmd = cmd or ["python3", "train.py"]
        bootstrap = self._build_bootstrap_cmd(user_cmd)

        health_check = HealthCheckConfig(
            liveness=ProbeConfig(
                path="/", port=HEALTH_PORT,
                initial_delay_seconds=5,
                period_seconds=30,
                failure_threshold=30,
            ),
            startup=ProbeConfig(
                path="/", port=HEALTH_PORT,
                initial_delay_seconds=3,
                period_seconds=5,
                failure_threshold=120,
            ),
        )

        logger.info(
            "Deploying %s: image=%s gpu=%dx%s cmd=%s",
            name, self._bcfg.image,
            self._bcfg.gpu_count, self._bcfg.gpu_models,
            user_cmd,
        )

        try:
            response = self._client.create_deployment(
                instance_name=name,
                image=self._bcfg.image,
                command=["python3", "-uc", bootstrap],
                port=HEALTH_PORT,
                env=env,
                gpu_count=self._bcfg.gpu_count,
                gpu_models=self._bcfg.gpu_models,
                memory=self._bcfg.memory,
                cpu=self._bcfg.cpu,
                storage=self._bcfg.storage,
                ttl_seconds=self._bcfg.ttl_seconds,
                min_gpu_memory_gb=self._bcfg.min_gpu_memory_gb,
                public=True,
                health_check=health_check,
            )
            deployment = Deployment._from_response(
                self._client, response
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.error("Create deployment %s failed: %s", name, exc)
            return RunOutcome(
                status="failed", metrics={}, stdout="",
                stderr=str(exc), elapsed_s=elapsed, run_dir=run_dir,
            )

        return self._wait_and_collect(deployment, name, t0, run_dir)

    def _wait_and_collect(
        self,
        deployment: object,
        name: str,
        t0: float,
        run_dir: str,
    ) -> RunOutcome:
        timeout = self._cfg.timeout_s
        poll_interval = 15

        try:
            # Phase 1: wait for deployment to be ready
            ready = False
            waited = 0
            while waited < min(timeout, 600):
                status = deployment.status()
                if status.is_ready:
                    ready = True
                    break
                if status.is_failed:
                    break
                time.sleep(poll_interval)
                waited += poll_interval

            if not ready:
                # Check logs even if not ready -- training may
                # have completed and container exited
                return self._collect_from_logs(
                    deployment, name, t0, run_dir, "not_ready"
                )

            # Phase 2: poll for training completion (metrics in logs)
            remaining = timeout - waited
            return self._poll_for_metrics(
                deployment, name, t0, run_dir, remaining
            )

        except Exception as exc:
            elapsed_s = time.monotonic() - t0
            self._cleanup(deployment, name)
            return RunOutcome(
                status="failed", metrics={}, stdout="",
                stderr=str(exc), elapsed_s=elapsed_s, run_dir=run_dir,
            )

    def _poll_for_metrics(
        self,
        deployment: object,
        name: str,
        t0: float,
        run_dir: str,
        remaining: float,
    ) -> RunOutcome:
        """Poll logs until training metrics appear or timeout."""
        poll_interval = 20
        waited = 0

        while waited < remaining:
            time.sleep(poll_interval)
            waited += poll_interval

            logs = self._extract_messages(self._safe_logs(deployment))
            metrics = self._parse_metrics(logs)

            if metrics:
                elapsed_s = time.monotonic() - t0
                logger.info(
                    "%s found %d metrics after %ds",
                    name, len(metrics), int(elapsed_s),
                )
                self._cleanup(deployment, name)
                return RunOutcome(
                    status="ok", metrics=metrics, stdout=logs,
                    stderr="", elapsed_s=elapsed_s, run_dir=run_dir,
                )

            # Check if deployment died
            try:
                status = deployment.status()
                if status.is_failed:
                    return self._collect_from_logs(
                        deployment, name, t0, run_dir, "failed"
                    )
            except Exception:
                pass

        # Timeout
        return self._collect_from_logs(
            deployment, name, t0, run_dir, "timeout"
        )

    def _collect_from_logs(
        self,
        deployment: object,
        name: str,
        t0: float,
        run_dir: str,
        reason: str,
    ) -> RunOutcome:
        """Extract whatever metrics exist from logs, then cleanup."""
        logs = self._extract_messages(self._safe_logs(deployment))
        metrics = self._parse_metrics(logs)
        elapsed_s = time.monotonic() - t0

        logger.info(
            "%s %s, found %d metrics in logs",
            name, reason, len(metrics),
        )
        self._cleanup(deployment, name)

        if metrics:
            return RunOutcome(
                status="ok", metrics=metrics, stdout=logs,
                stderr="", elapsed_s=elapsed_s, run_dir=run_dir,
            )
        return RunOutcome(
            status="failed", metrics={}, stdout=logs,
            stderr=reason, elapsed_s=elapsed_s, run_dir=run_dir,
        )

    def _extract_messages(self, raw_logs: str) -> str:
        """Extract message text from Basilica SSE JSON log lines."""
        lines: list[str] = []
        for line in raw_logs.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("data: "):
                line = line[6:]
            try:
                parsed = json.loads(line)
                msg = parsed.get("message", "")
                if msg:
                    lines.append(msg)
            except (json.JSONDecodeError, TypeError):
                lines.append(line)
        return "\n".join(lines)

    def _safe_logs(self, deployment: object) -> str:
        try:
            return deployment.logs(tail=500)
        except Exception:
            return ""

    def _cleanup(self, deployment: object, name: str) -> None:
        try:
            deployment.delete()
            logger.info("Cleaned up %s", name)
        except Exception as exc:
            logger.warning("Cleanup failed for %s: %s", name, exc)

    def _parse_metrics(self, logs: str) -> dict[str, float]:
        """Extract key=value metrics from log text."""
        metrics: dict[str, float] = {}
        for match in re.finditer(r"(\w+)=([\d.eE+-]+)", logs):
            key = match.group(1).lower()
            try:
                metrics[key] = float(match.group(2))
            except ValueError:
                continue
        return metrics
