#!/usr/bin/env bash
# Phase A.3 trajectory-aware ablation runner — DESIGN v2.
#
# Re-designs after the v1 sweep (max_iter=4, K=4) revealed a dead
# methodology: with iters == K, the parallel engine makes exactly one
# propose_batch() call with empty history, so render_progress_series
# returns "" for both arms regardless of AR_DISABLE_PROGRESS_SERIES.
# Arms produced identical proposed params; observed eval_score deltas
# were pure training-time stochasticity (train.py is unseeded).
#
# v2: max_iterations=8, K=4. Two batches: the SECOND batch sees the
# 4 completed iters of the first batch, so progress_series is rendered
# for it in Arm A and suppressed in Arm B.
#
# Cost: ~16 GPU-min wall per run, ~$16/run (4 GPU-trial × ~4-8 min each).
# n=3 paired runs = 6 cells = ~$96. Combined with v1 sunk cost (~$40)
# the total Phase A.3 spend is about $136, under the $150 hard kill.
#
# Usage: scripts/research/run_ablation_v2_one.sh <arm: A|B> <seed: int>

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <arm: A|B> <seed: int>" >&2
    exit 2
fi

ARM="$1"
SEED="$2"

if [[ "${ARM}" != "A" && "${ARM}" != "B" ]]; then
    echo "ARM must be A or B, got '${ARM}'" >&2
    exit 2
fi

REPO="/home/epappas/workspace/spacejar/autoresearch-rl/.claude/worktrees/ar-phase-1"
cd "${REPO}"

ENV_FILE=""
if [[ -f "${REPO}/.env" ]]; then
    ENV_FILE="${REPO}/.env"
elif [[ -f "/home/epappas/workspace/spacejar/autoresearch-rl/.env" ]]; then
    ENV_FILE="/home/epappas/workspace/spacejar/autoresearch-rl/.env"
fi
if [[ -z "${ENV_FILE}" ]]; then
    echo "ERROR: no .env found" >&2
    exit 2
fi
set -a
. "${ENV_FILE}"
set +a

LABEL="${ARM}_seed${SEED}"
OUT_DIR="docs/research/data/ablation-2026-04/v2/${LABEL}"
LOG="${OUT_DIR}/launch.log"

if [[ -f "${OUT_DIR}/results.tsv" ]]; then
    if grep -q "^[a-f0-9]" "${OUT_DIR}/results.tsv"; then
        echo "[skip] ${LABEL} already has populated results.tsv"
        exit 0
    fi
    echo "[retry] ${LABEL} results.tsv exists but is empty/header-only; re-running"
    rm -rf "${OUT_DIR}"
fi

mkdir -p "${OUT_DIR}"
echo "[start] ${LABEL} at $(date -Iseconds) commit=$(git rev-parse HEAD)" | tee "${LOG}"

if [[ "${ARM}" == "B" ]]; then
    export AR_DISABLE_PROGRESS_SERIES=1
else
    unset AR_DISABLE_PROGRESS_SERIES || true
fi

set +e
python3 examples/security-judge/deploy.py --policy llm \
    controller.parallel.enabled=true \
    controller.parallel.max_concurrency=4 \
    'controller.parallel.resources={"gpu":4}' \
    controller.max_iterations=8 \
    controller.max_wall_time_s=5400 \
    "controller.seed=${SEED}" \
    "policy.seed=${SEED}" \
    target.basilica.ready_timeout_s=1500 \
    target.timeout_s=2400 \
    "telemetry.trace_path=${OUT_DIR}/events.jsonl" \
    "telemetry.ledger_path=${OUT_DIR}/results.tsv" \
    "telemetry.artifacts_dir=${OUT_DIR}/runs" \
    "telemetry.versions_dir=${OUT_DIR}/versions" \
    "telemetry.model_output_dir=/data/models/ablation-v2-2026-04/${LABEL}" \
    "controller.checkpoint_path=${OUT_DIR}/checkpoint.json" \
    "telemetry.timeline_path=${OUT_DIR}/timeline.json" \
    >> "${LOG}" 2>&1
RC=$?
set -e

echo "[end] ${LABEL} at $(date -Iseconds) rc=${RC}" | tee -a "${LOG}"
exit "${RC}"
