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
import urllib.request
import uuid
from pathlib import Path

from autoresearch_rl.config import TargetConfig
from autoresearch_rl.target.interface import RunOutcome

logger = logging.getLogger(__name__)

HEALTH_PORT = 8080

# Bootstrap script injected into every Basilica deployment.
# Starts a health-check server, then runs the user command via subprocess.
_BOOTSTRAP = r"""
import subprocess, sys, threading, time, json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path as _Path

_model_dir = __import__("os").environ.get("AR_MODEL_DIR", "")

class _H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        elif self.path == "/model/files":
            self._serve_model_listing()
        elif self.path.startswith("/model/download/"):
            self._serve_model_file(self.path[len("/model/download/"):])
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_model_listing(self):
        if not _model_dir or not _Path(_model_dir).exists():
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({{"files": [], "model_dir": _model_dir}}).encode())
            return
        files = []
        base = _Path(_model_dir)
        for f in sorted(base.rglob("*")):
            if f.is_file():
                files.append({{"path": str(f.relative_to(base)), "size": f.stat().st_size}})
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({{"files": files, "model_dir": _model_dir}}).encode())

    def _serve_model_file(self, rel_path):
        if not _model_dir:
            self.send_response(404)
            self.end_headers()
            return
        fpath = _Path(_model_dir) / rel_path
        if not fpath.exists() or not fpath.is_file():
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(fpath.stat().st_size))
        self.end_headers()
        with open(fpath, "rb") as fp:
            while chunk := fp.read(65536):
                self.wfile.write(chunk)

    def log_message(self, *a):
        pass

threading.Thread(
    target=lambda: HTTPServer(("", {port}), _H).serve_forever(),
    daemon=True,
).start()

# Diff mode: decode and write modified source if AR_MODIFIED_SOURCE is set
import os as _os, base64 as _b64
_src = _os.environ.get("AR_MODIFIED_SOURCE", "")
if _src:
    _tgt = _os.environ.get("AR_MODIFIED_TARGET", "train.py")
    with open(_tgt, "w") as _f:
        _f.write(_b64.b64decode(_src).decode("utf-8"))
    print(f"[ar] wrote modified source to {{_tgt}} ({{len(_src)}} b64 chars)")

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

    @staticmethod
    def _build_bootstrap_cmd(user_cmd: list[str], setup_cmd: str | None = None) -> str:
        """Wrap user command in bootstrap that starts health server."""
        setup = setup_cmd
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
        if "AR_MODEL_DIR" in params:
            env["AR_MODEL_DIR"] = str(params["AR_MODEL_DIR"])

        # Diff mode: pass modified source directly as env vars
        for diff_key in ("AR_MODIFIED_SOURCE", "AR_MODIFIED_TARGET"):
            if diff_key in params:
                env[diff_key] = str(params[diff_key])

        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            env["HF_TOKEN"] = hf_token

        user_cmd = cmd or ["python3", "train.py"]
        # Chain: setup_cmd (pip install etc) -> prepare_cmd -> train_cmd
        setup = self._bcfg.setup_cmd or ""
        if self._cfg.prepare_cmd:
            prepare = " ".join(self._cfg.prepare_cmd)
            setup = f"{setup} && {prepare}" if setup else prepare
        bootstrap = self._build_bootstrap_cmd(user_cmd, setup_cmd=setup or None)

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
                # Download model before cleanup destroys the container
                model_local = self._download_model(deployment, run_dir)
                if model_local:
                    metrics["_model_dir"] = model_local  # type: ignore[assignment]
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

    @staticmethod
    def _extract_messages(raw_logs: str) -> str:
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

    def _download_model(self, deployment: object, run_dir: str) -> str | None:
        """Download model files from container's /model/ endpoint to run_dir."""
        try:
            base_url = deployment.url.rstrip("/")
        except Exception:
            return None

        model_local = str(Path(run_dir) / "model")

        try:
            # List files
            listing_url = f"{base_url}/model/files"
            req = urllib.request.Request(listing_url, method="GET")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            files = data.get("files", [])
            if not files:
                return None

            Path(model_local).mkdir(parents=True, exist_ok=True)
            for entry in files:
                rel = entry["path"]
                dl_url = f"{base_url}/model/download/{rel}"
                dest = Path(model_local) / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                req = urllib.request.Request(dl_url, method="GET")
                with urllib.request.urlopen(req, timeout=120) as resp:
                    dest.write_bytes(resp.read())

            logger.info(
                "Downloaded %d model files to %s", len(files), model_local,
            )
            return model_local
        except Exception as exc:
            logger.warning("Model download failed: %s", exc)
            return None

    def _cleanup(self, deployment: object, name: str) -> None:
        try:
            deployment.delete()
            logger.info("Cleaned up %s", name)
        except Exception as exc:
            logger.warning("Cleanup failed for %s: %s", name, exc)

    # Metric keys that indicate training has completed and real results are available.
    # The adapter only returns "ok" when at least one of these is found in the logs.
    _KNOWN_METRIC_KEYS = frozenset({
        "eval_score", "val_bpb", "loss", "accuracy", "f1",
        "training_seconds", "improvement", "reward",
    })

    @staticmethod
    def _parse_metrics(logs: str) -> dict[str, float]:
        """Extract key=value metrics from log text.

        Only returns metrics if at least one key matches _KNOWN_METRIC_KEYS,
        to avoid false positives from library warnings or preparation output.
        """
        raw: dict[str, float] = {}
        for match in re.finditer(r"(\w+)=([\d.eE+-]+)", logs):
            key = match.group(1).lower()
            try:
                raw[key] = float(match.group(2))
            except ValueError:
                continue
        if not raw.keys() & BasilicaTarget._KNOWN_METRIC_KEYS:
            return {}
        return raw
