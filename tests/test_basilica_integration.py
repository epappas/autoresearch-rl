"""E2E integration test: fine-tune DeBERTa on Basilica GPU cloud.

Requires:
- BASILICA_API_TOKEN env var set
- Docker image pushed to ghcr.io/epappas/ar-deberta-e2e:latest

Run: uv run pytest tests/test_basilica_integration.py -v -m integration
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid

import pytest

logger = logging.getLogger(__name__)

BASILICA_TOKEN = os.environ.get("BASILICA_API_TOKEN", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

skip_no_token = pytest.mark.skipif(
    not BASILICA_TOKEN,
    reason="BASILICA_API_TOKEN not set",
)

IMAGE = "ghcr.io/epappas/ar-deberta-e2e:latest"
TRAIN_CMD = [
    "python3", "/app/train.py",
    "--train-file", "/app/data/train.jsonl",
    "--val-file", "/app/data/val.jsonl",
    "--output-dir", "/tmp/deberta-out",
    "--epochs", "1",
    "--batch-size", "4",
]
GPU_MODELS = ["A100", "H100", "L40S", "RTX-4090", "RTX-A6000"]
TIMEOUT_S = 900
HEALTH_PORT = 8080


@pytest.mark.integration
@pytest.mark.basilica
@skip_no_token
def test_deberta_e2e_on_basilica() -> None:
    """Deploy DeBERTa fine-tuning on Basilica, verify real metrics come back."""
    from basilica import BasilicaClient, Deployment, HealthCheckConfig, ProbeConfig

    from autoresearch_rl.target.basilica import BasilicaTarget

    client = BasilicaClient()
    tag = uuid.uuid4().hex[:8]
    name = f"ar-e2e-deberta-{tag}"
    deployment = None

    try:
        bootstrap = BasilicaTarget._build_bootstrap_cmd(TRAIN_CMD)

        env: dict[str, str] = {
            "AR_PARAMS_JSON": json.dumps({"epochs": 1, "batch_size": 4}),
        }
        if HF_TOKEN:
            env["HF_TOKEN"] = HF_TOKEN

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

        logger.info("Creating deployment %s with image %s", name, IMAGE)
        response = client.create_deployment(
            instance_name=name,
            image=IMAGE,
            command=["python3", "-uc", bootstrap],
            port=HEALTH_PORT,
            env=env,
            gpu_count=1,
            gpu_models=GPU_MODELS,
            memory="32Gi",
            cpu="8",
            ttl_seconds=TIMEOUT_S,
            public=True,
            health_check=health_check,
        )
        deployment = Deployment._from_response(client, response)
        logger.info("Deployment %s created, waiting for ready...", name)

        # Phase 1: wait for ready (up to 10 min)
        waited = 0
        poll_interval = 15
        while waited < 600:
            status = deployment.status()
            if status.is_ready:
                logger.info("Deployment %s is ready after %ds", name, waited)
                break
            if status.is_failed:
                logs = _safe_logs(deployment)
                pytest.fail(f"Deployment {name} failed during startup. Logs:\n{logs}")
            time.sleep(poll_interval)
            waited += poll_interval

        # Phase 2: poll for metrics in logs (up to remaining time)
        remaining = TIMEOUT_S - waited
        metrics: dict[str, float] = {}
        logs_text = ""
        poll_interval = 20
        poll_waited = 0

        while poll_waited < remaining:
            time.sleep(poll_interval)
            poll_waited += poll_interval

            raw = _safe_logs(deployment)
            logs_text = BasilicaTarget._extract_messages(raw)
            metrics = BasilicaTarget._parse_metrics(logs_text)

            if metrics:
                logger.info(
                    "Found %d metrics after %ds total: %s",
                    len(metrics), waited + poll_waited, metrics,
                )
                break

            # Check if deployment died
            try:
                status = deployment.status()
                if status.is_failed:
                    logs_text = BasilicaTarget._extract_messages(_safe_logs(deployment))
                    metrics = BasilicaTarget._parse_metrics(logs_text)
                    break
            except Exception:
                pass

        # Assertions
        assert metrics, (
            f"No metrics found in logs after {waited + poll_waited}s. "
            f"Last logs:\n{logs_text[-2000:]}"
        )
        assert "val_bpb" in metrics, f"Missing val_bpb in metrics: {metrics}"
        assert "loss" in metrics, f"Missing loss in metrics: {metrics}"
        assert metrics["val_bpb"] < 1.0, f"val_bpb={metrics['val_bpb']} not < 1.0"
        assert metrics["loss"] > 0.0, f"loss={metrics['loss']} not > 0.0"

        logger.info(
            "E2E PASSED: val_bpb=%.4f loss=%.4f f1=%.4f",
            metrics.get("val_bpb", -1),
            metrics.get("loss", -1),
            metrics.get("f1", -1),
        )

    finally:
        if deployment is not None:
            try:
                deployment.delete()
                logger.info("Cleaned up deployment %s", name)
            except Exception as exc:
                logger.warning("Cleanup failed for %s: %s", name, exc)


def _safe_logs(deployment: object) -> str:
    try:
        return deployment.logs(tail=500)  # type: ignore[attr-defined]
    except Exception:
        return ""
