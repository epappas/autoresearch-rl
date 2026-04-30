#!/usr/bin/env bash
# Phase A.3 trajectory-aware ablation runner — issue #31.
#
# 5 paired runs of security-judge K=4 parallel on Basilica.
# Arm A: full LLM prompt with progress_series.
# Arm B: identical prompt with progress_series field removed
#        (via AR_DISABLE_PROGRESS_SERIES=1).
#
# Pair ordering: A1, B1, A2, B2, ... (alternated to control for
# infrastructure load drift across the ~3.7-hour run).
#
# Same-seed pairing: each pair uses controller.seed=$SEED so the
# random-fallback path is identical when the LLM 429s out.
#
# Output: docs/research/data/ablation-2026-04/runs/{arm}_seed{N}/.
#
# Re-runs: a per-pair done-marker (results.tsv presence) lets you
# Ctrl+C and resume by re-running the script.

set -euo pipefail

REPO="/home/epappas/workspace/spacejar/autoresearch-rl/.claude/worktrees/ar-phase-1"
cd "$REPO"

# Source ~/.env (BASILICA_API_TOKEN, MOONSHOT_API_KEY, CHUTES_API_KEY, HF_TOKEN).
set -a
. "$REPO/.env"
set +a

SEEDS=(1 2 3 4 5)
DATA_DIR="docs/research/data/ablation-2026-04"

# Use Moonshot/Kimi for the LLM proposer — lower 429 rate than Chutes
# during EU evening hours, validated by tests/eval/test_real_llm.py.
LLM_API_URL="https://api.moonshot.ai/v1"
LLM_MODEL="kimi-k2.6"
LLM_KEY_ENV="MOONSHOT_API_KEY"

run_one() {
    local arm="$1"   # A or B
    local seed="$2"
    local label="${arm}_seed${seed}"
    local out_dir="${DATA_DIR}/runs/${label}"
    local log="${out_dir}/launch.log"

    if [[ -f "${out_dir}/results.tsv" ]]; then
        echo "[skip] ${label} already has results.tsv"
        return 0
    fi

    mkdir -p "${out_dir}"
    echo "[start] ${label} at $(date -Iseconds)" | tee "${log}"

    # Arm B sets AR_DISABLE_PROGRESS_SERIES=1; Arm A leaves it unset.
    if [[ "${arm}" == "B" ]]; then
        export AR_DISABLE_PROGRESS_SERIES=1
    else
        unset AR_DISABLE_PROGRESS_SERIES
    fi

    # Run via deploy.py so the file-injection setup_cmd is canonical.
    local rc=0
    python3 examples/security-judge/deploy.py --policy llm \
        controller.parallel.enabled=true \
        controller.parallel.max_concurrency=4 \
        'controller.parallel.resources={"gpu":4}' \
        controller.max_iterations=4 \
        controller.max_wall_time_s=3600 \
        "controller.seed=${seed}" \
        "policy.seed=${seed}" \
        target.basilica.ready_timeout_s=1500 \
        target.timeout_s=2400 \
        "policy.llm_api_url=${LLM_API_URL}" \
        "policy.llm_model=${LLM_MODEL}" \
        "policy.llm_api_key_env=${LLM_KEY_ENV}" \
        "telemetry.trace_path=${out_dir}/events.jsonl" \
        "telemetry.ledger_path=${out_dir}/results.tsv" \
        "telemetry.artifacts_dir=${out_dir}/runs" \
        "telemetry.versions_dir=${out_dir}/versions" \
        "telemetry.model_output_dir=/data/models/ablation-2026-04/${label}" \
        "controller.checkpoint_path=${out_dir}/checkpoint.json" \
        "telemetry.timeline_path=${out_dir}/timeline.json" \
        >> "${log}" 2>&1 || rc=$?

    echo "[end] ${label} at $(date -Iseconds) rc=${rc}" | tee -a "${log}"
    unset AR_DISABLE_PROGRESS_SERIES
    return 0
}

echo "=== A.3 ablation start at $(date -Iseconds) ==="
for seed in "${SEEDS[@]}"; do
    run_one A "${seed}"
    run_one B "${seed}"
done
echo "=== A.3 ablation end at $(date -Iseconds) ==="
