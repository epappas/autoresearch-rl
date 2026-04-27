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
import string
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Protocol

from autoresearch_rl.config import TargetConfig
from autoresearch_rl.target.interface import RunOutcome
from autoresearch_rl.telemetry.timeline import global_span

logger = logging.getLogger(__name__)


class _DeploymentStatus(Protocol):
    is_ready: bool
    is_failed: bool


class _Deployment(Protocol):
    """Subset of basilica.Deployment we use. Lets mypy check call sites
    without the SDK installed at type-check time."""

    url: str

    def status(self) -> _DeploymentStatus: ...
    def logs(self, *, tail: int = ...) -> str: ...
    def delete(self) -> None: ...

HEALTH_PORT = 8080

# Bootstrap script injected into every Basilica deployment.
# Starts a health-check server, then runs the user command via subprocess.
# Uses string.Template ($port, $cmd) so dict / f-string braces stay literal.
_BOOTSTRAP_TEMPLATE = string.Template(r"""
import subprocess, sys, threading, time, json, os as _os
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path as _Path

_model_dir = _os.environ.get("AR_MODEL_DIR", "")
_progress_path = _os.environ.get("AR_PROGRESS_FILE", "/tmp/ar_progress.jsonl")
_control_path = _os.environ.get("AR_CONTROL_FILE", "/tmp/ar_control.json")
_os.environ.setdefault("AR_PROGRESS_FILE", _progress_path)
_os.environ.setdefault("AR_CONTROL_FILE", _control_path)

class _H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        elif self.path == "/progress":
            self._serve_progress()
        elif self.path == "/model/files":
            self._serve_model_listing()
        elif self.path.startswith("/model/download/"):
            self._serve_model_file(self.path[len("/model/download/"):])
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/control":
            self._accept_control()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_progress(self):
        try:
            with open(_progress_path, "rb") as fp:
                data = fp.read()
        except OSError:
            data = b""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        self.wfile.write(data)

    def _accept_control(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            with open(_control_path, "wb") as fp:
                fp.write(body)
        except OSError:
            self.send_response(500)
            self.end_headers()
            return
        self.send_response(204)
        self.end_headers()

    def _serve_model_listing(self):
        if not _model_dir or not _Path(_model_dir).exists():
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"files": [], "model_dir": _model_dir}).encode())
            return
        files = []
        base = _Path(_model_dir)
        for f in sorted(base.rglob("*")):
            if f.is_file():
                files.append({"path": str(f.relative_to(base)), "size": f.stat().st_size})
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"files": files, "model_dir": _model_dir}).encode())

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
    target=lambda: HTTPServer(("", $port), _H).serve_forever(),
    daemon=True,
).start()

# Diff mode: decode and write modified source if AR_MODIFIED_SOURCE is set
import os as _os, base64 as _b64
_src = _os.environ.get("AR_MODIFIED_SOURCE", "")
if _src:
    _tgt = _os.environ.get("AR_MODIFIED_TARGET", "train.py")
    with open(_tgt, "w") as _f:
        _f.write(_b64.b64decode(_src).decode("utf-8"))
    print(f"[ar] wrote modified source to {_tgt} ({len(_src)} b64 chars)")

rc = subprocess.call($cmd, env=dict(**__import__("os").environ))
sys.stdout.flush()
sys.stderr.flush()
time.sleep(15)
sys.exit(rc)
""")


class BasilicaTarget:
    """Run training iterations on Basilica GPU cloud."""

    def __init__(self, cfg: TargetConfig) -> None:
        try:
            from basilica import BasilicaClient  # type: ignore[import-not-found]
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
        script = _BOOTSTRAP_TEMPLATE.substitute(port=HEALTH_PORT, cmd=cmd_repr)
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
        from basilica import (  # type: ignore[import-not-found]
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
            with global_span(
                "basilica.create_deployment",
                category="basilica",
                args={"name": name, "image": self._bcfg.image,
                      "gpu_count": self._bcfg.gpu_count},
            ):
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
        deployment: _Deployment,
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
            with global_span(
                "basilica.wait_ready",
                category="basilica",
                args={"name": name, "timeout_s": min(timeout, 600)},
            ) as wait_args:
                while waited < min(timeout, 600):
                    status = deployment.status()
                    if status.is_ready:
                        ready = True
                        break
                    if status.is_failed:
                        break
                    time.sleep(poll_interval)
                    waited += poll_interval
                wait_args["ready"] = ready
                wait_args["waited_s"] = waited

            if not ready:
                # Check logs even if not ready -- training may
                # have completed and container exited
                return self._collect_from_logs(
                    deployment, name, t0, run_dir, "not_ready"
                )

            # Phase 2: poll for training completion (metrics in logs)
            remaining = timeout - waited
            with global_span(
                "basilica.poll_for_metrics",
                category="basilica",
                args={"name": name, "remaining_s": remaining},
            ):
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
        deployment: _Deployment,
        name: str,
        t0: float,
        run_dir: str,
        remaining: float,
    ) -> RunOutcome:
        """Poll logs until training metrics appear or timeout.

        Adaptive interval: drops to 5s when /progress shows live activity,
        backs off to 20s when stalled. Persists progress reports to
        run_dir/progress.jsonl for the controller / IntraIterationGuard.
        """
        progress_path = Path(run_dir) / "progress.jsonl"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        last_progress_size = 0
        poll_interval = 20
        waited = 0.0

        while waited < remaining:
            time.sleep(poll_interval)
            waited += poll_interval

            # Pull /progress snapshot from container if available.
            new_size = self._fetch_progress(deployment, progress_path)
            if new_size > last_progress_size:
                last_progress_size = new_size
                poll_interval = 5  # active — poll faster
            else:
                poll_interval = min(20, poll_interval + 5)  # idle — back off

            # If the local IntraIterationGuard wrote a cancel control file,
            # propagate it to the running container so emit_progress() inside
            # the trial sees the cancel signal on its next call (Phase 2
            # cooperative cancel for Basilica targets).
            self._propagate_control(deployment, run_dir)

            logs = self._extract_messages(self._safe_logs(deployment))
            metrics = self._parse_metrics(logs)

            if metrics:
                elapsed_s = time.monotonic() - t0
                logger.info(
                    "%s found %d metrics after %ds",
                    name, len(metrics), int(elapsed_s),
                )
                # Download model before cleanup destroys the container
                with global_span(
                    "basilica.download_model",
                    category="basilica",
                    args={"name": name, "run_dir": run_dir},
                ) as dl_args:
                    model_local = self._download_model(deployment, run_dir)
                    dl_args["downloaded"] = bool(model_local)
                if model_local:
                    metrics["_model_dir"] = model_local  # type: ignore[assignment]
                with global_span(
                    "basilica.cleanup",
                    category="basilica",
                    args={"name": name},
                ):
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

    def _propagate_control(self, deployment: _Deployment, run_dir: str) -> None:
        """POST run_dir/control.json contents to deployment.url + /control.

        Best-effort: a missing control file is the common case (no cancel
        requested). A failed POST is logged at debug — the next poll will
        retry. The bootstrap server overwrites its container-side control
        file on each POST, so duplicate uploads are idempotent.
        """
        control_local = Path(run_dir) / "control.json"
        if not control_local.exists():
            return
        try:
            data = control_local.read_bytes()
        except OSError:
            return
        if not data.strip():
            return
        # Skip uploads we've already pushed (size-cached so we don't spam
        # the deployment server every 5s of polling).
        last = getattr(self, "_last_control_pushed", {}).get(run_dir)
        if last == len(data):
            return
        try:
            base_url = deployment.url.rstrip("/")
        except Exception:
            return
        try:
            req = urllib.request.Request(
                f"{base_url}/control",
                data=data,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
        except Exception as exc:
            logger.debug("control upload failed for %s: %s", run_dir, exc)
            return
        cache = getattr(self, "_last_control_pushed", None)
        if cache is None:
            cache = {}
            self._last_control_pushed = cache  # type: ignore[attr-defined]
        cache[run_dir] = len(data)
        logger.info("propagated cancel control to %s (%d bytes)", base_url, len(data))

    def _fetch_progress(self, deployment: _Deployment, progress_path: Path) -> int:
        """Append /progress snapshot to local progress.jsonl. Returns new file size.

        Best-effort. Network errors are silent; the existing log-poll path is
        the source of truth for final metrics, /progress only enriches signal.
        """
        try:
            base_url = deployment.url.rstrip("/")
        except Exception:
            return progress_path.stat().st_size if progress_path.exists() else 0
        try:
            req = urllib.request.Request(f"{base_url}/progress", method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
        except Exception:
            return progress_path.stat().st_size if progress_path.exists() else 0
        # /progress returns the entire JSONL each time; rewrite atomically.
        tmp = progress_path.with_suffix(".jsonl.tmp")
        try:
            tmp.write_bytes(data)
            tmp.replace(progress_path)
        except OSError:
            return progress_path.stat().st_size if progress_path.exists() else 0
        return len(data)

    def _collect_from_logs(
        self,
        deployment: _Deployment,
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

    def _safe_logs(self, deployment: _Deployment) -> str:
        try:
            return deployment.logs(tail=500)
        except Exception:
            return ""

    def _download_model(self, deployment: _Deployment, run_dir: str) -> str | None:
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

    def _cleanup(self, deployment: _Deployment, name: str) -> None:
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
