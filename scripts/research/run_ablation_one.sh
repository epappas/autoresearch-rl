#!/usr/bin/env bash
# Run a single (arm, seed) cell of the Phase A.3 ablation.
#
# Usage: scripts/research/run_ablation_one.sh <arm> <seed>
#   arm:  A (progress_series enabled) | B (progress_series disabled via env var)
#   seed: integer seed for controller.seed and policy.seed
#
# Each cell takes ~20-25 min wall on Basilica K=4 A100 + 4 GRPO trials.
#
# Idempotent: existing results.tsv at the output path skips the run.
#
# LLM choice: uses the security-judge default (Chutes/DeepSeek-V3-0324)
# from examples/security-judge/config.yaml — MOONSHOT_API_KEY is not
# present in the .env at this commit. If Chutes 429s and the policy
# falls back to seeded random, both arms experience that fallback
# equally (same seed -> same proposals when LLM unavailable), and
# the disclosed-in-doc effect is "we measured A vs B under whatever
# proposer signal we got, including 429-fallback noise".

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

# .env lives at the actual git root (not the worktree). Fall back to local .env if present.
ENV_FILE=""
if [[ -f "${REPO}/.env" ]]; then
    ENV_FILE="${REPO}/.env"
elif [[ -f "/home/epappas/workspace/spacejar/autoresearch-rl/.env" ]]; then
    ENV_FILE="/home/epappas/workspace/spacejar/autoresearch-rl/.env"
fi
if [[ -z "${ENV_FILE}" ]]; then
    echo "ERROR: no .env found in worktree or repo root" >&2
    exit 2
fi
set -a
. "${ENV_FILE}"
set +a

LABEL="${ARM}_seed${SEED}"
OUT_DIR="docs/research/data/ablation-2026-04/runs/${LABEL}"
LOG="${OUT_DIR}/launch.log"

if [[ -f "${OUT_DIR}/results.tsv" ]]; then
    echo "[skip] ${LABEL} already has results.tsv"
    exit 0
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
    controller.max_iterations=4 \
    controller.max_wall_time_s=3600 \
    "controller.seed=${SEED}" \
    "policy.seed=${SEED}" \
    target.basilica.ready_timeout_s=1500 \
    target.timeout_s=2400 \
    "telemetry.trace_path=${OUT_DIR}/events.jsonl" \
    "telemetry.ledger_path=${OUT_DIR}/results.tsv" \
    "telemetry.artifacts_dir=${OUT_DIR}/runs" \
    "telemetry.versions_dir=${OUT_DIR}/versions" \
    "telemetry.model_output_dir=/data/models/ablation-2026-04/${LABEL}" \
    "controller.checkpoint_path=${OUT_DIR}/checkpoint.json" \
    "telemetry.timeline_path=${OUT_DIR}/timeline.json" \
    >> "${LOG}" 2>&1
RC=$?
set -e

echo "[end] ${LABEL} at $(date -Iseconds) rc=${RC}" | tee -a "${LOG}"
exit "${RC}"
