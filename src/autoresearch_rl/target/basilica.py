"""Basilica GPU cloud target adapter.

Deploys training jobs on Basilica infrastructure, waits for completion,
and extracts metrics from deployment logs.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid

from autoresearch_rl.config import TargetConfig
from autoresearch_rl.target.interface import RunOutcome

logger = logging.getLogger(__name__)


class BasilicaTarget:
    """Run training iterations on Basilica GPU cloud.

    Each iteration deploys a container with AR_PARAMS_JSON,
    waits for it to complete, parses metrics from logs.
    """

    def __init__(self, cfg: TargetConfig) -> None:
        try:
            from basilica import BasilicaClient
        except ImportError as e:
            raise ImportError(
                "basilica-sdk is required for basilica target. "
                "Install with: pip install basilica-sdk"
            ) from e

        self._client = BasilicaClient()
        self._cfg = cfg
        self._bcfg = cfg.basilica
        self._last_train_outcome: RunOutcome | None = None

    def run(self, *, run_dir: str, params: dict[str, object]) -> RunOutcome:
        """Deploy training job and wait for completion."""
        outcome = self._deploy_and_collect(
            params=params,
            run_dir=run_dir,
            cmd=self._cfg.train_cmd,
            phase="train",
        )
        self._last_train_outcome = outcome
        return outcome

    def eval(self, *, run_dir: str, params: dict[str, object]) -> RunOutcome:
        """Deploy eval job or return train metrics if no separate eval."""
        if not self._cfg.eval_cmd:
            if self._last_train_outcome is not None:
                return self._last_train_outcome
            return RunOutcome(
                status="ok", metrics={}, stdout="", stderr="",
                elapsed_s=0.0, run_dir=run_dir,
            )
        return self._deploy_and_collect(
            params=params,
            run_dir=run_dir,
            cmd=self._cfg.eval_cmd,
            phase="eval",
        )

    def _deploy_and_collect(
        self,
        params: dict[str, object],
        run_dir: str,
        cmd: list[str] | None,
        phase: str,
    ) -> RunOutcome:
        from basilica import Deployment

        tag = uuid.uuid4().hex[:8]
        name = f"ar-{phase}-{tag}"
        t0 = time.monotonic()

        env = {
            "AR_PARAMS_JSON": json.dumps(params, default=str),
        }
        for k, v in params.items():
            env[f"AR_PARAM_{str(k).upper()}"] = str(v)

        import os
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            env["HF_TOKEN"] = hf_token

        command = cmd or ["python3", "train.py"]

        logger.info(
            "Deploying %s on Basilica: image=%s gpu=%dx%s",
            name, self._bcfg.image,
            self._bcfg.gpu_count, self._bcfg.gpu_models,
        )

        try:
            response = self._client.create_deployment(
                instance_name=name,
                image=self._bcfg.image,
                command=["bash"],
                args=["-c", " ".join(command)],
                env=env,
                gpu_count=self._bcfg.gpu_count,
                gpu_models=self._bcfg.gpu_models,
                memory=self._bcfg.memory,
                cpu=self._bcfg.cpu,
                storage=self._bcfg.storage,
                ttl_seconds=self._bcfg.ttl_seconds,
                min_gpu_memory_gb=self._bcfg.min_gpu_memory_gb,
                public=False,
            )
            deployment = Deployment._from_response(self._client, response)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.error("Failed to create deployment %s: %s", name, exc)
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
        """Poll deployment until done, collect logs, parse metrics."""
        timeout = self._cfg.timeout_s
        poll_interval = 10
        elapsed = 0

        try:
            while elapsed < timeout:
                status = deployment.status()

                if status.is_ready:
                    break

                if status.is_failed:
                    # Short-lived jobs exit and show as "failed" but may
                    # have completed successfully. Check logs for metrics.
                    raw_logs = self._safe_logs(deployment)
                    logs = self._extract_log_messages(raw_logs)
                    metrics = self._parse_metrics(logs)
                    elapsed_s = time.monotonic() - t0
                    logger.info(
                        "%s phase=failed, found %d metrics in logs",
                        name, len(metrics),
                    )
                    self._cleanup(deployment, name)
                    if metrics:
                        return RunOutcome(
                            status="ok", metrics=metrics,
                            stdout=logs, stderr="",
                            elapsed_s=elapsed_s, run_dir=run_dir,
                        )
                    return RunOutcome(
                        status="failed", metrics={},
                        stdout=logs, stderr=status.message or "",
                        elapsed_s=elapsed_s, run_dir=run_dir,
                    )

                time.sleep(poll_interval)
                elapsed += poll_interval

            logs = self._extract_log_messages(
                self._safe_logs(deployment)
            )
            elapsed_s = time.monotonic() - t0

            if elapsed >= timeout:
                self._cleanup(deployment, name)
                return RunOutcome(
                    status="timeout", metrics={},
                    stdout=logs, stderr="deployment timed out",
                    elapsed_s=elapsed_s, run_dir=run_dir,
                )

            time.sleep(5)
            logs = self._extract_log_messages(
                self._safe_logs(deployment)
            )

            metrics = self._parse_metrics(logs)
            self._cleanup(deployment, name)

            return RunOutcome(
                status="ok", metrics=metrics,
                stdout=logs, stderr="",
                elapsed_s=elapsed_s, run_dir=run_dir,
            )

        except Exception as exc:
            elapsed_s = time.monotonic() - t0
            self._cleanup(deployment, name)
            return RunOutcome(
                status="failed", metrics={},
                stdout="", stderr=str(exc),
                elapsed_s=elapsed_s, run_dir=run_dir,
            )

    def _extract_log_messages(self, raw_logs: str) -> str:
        """Extract message content from Basilica JSON log lines."""
        lines = []
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
            logger.info("Cleaned up deployment %s", name)
        except Exception as exc:
            logger.warning("Failed to delete deployment %s: %s", name, exc)

    def _parse_metrics(self, logs: str) -> dict[str, float]:
        """Extract key=value metrics from deployment logs."""
        metrics: dict[str, float] = {}
        patterns = [
            r"(\w+)\s*=\s*([\d.eE+-]+)",
            r"(\w+)\s*:\s*([\d.eE+-]+)",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, logs):
                key = match.group(1).lower()
                try:
                    metrics[key] = float(match.group(2))
                except ValueError:
                    continue
        return metrics
