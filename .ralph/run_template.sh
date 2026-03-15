#!/usr/bin/env bash
# RALPH loop runner template
# Usage: .ralph/<task>/run.sh
set -euo pipefail
TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_NAME="$(basename "$TASK_DIR")"
PROJECT_ROOT="$(cd "$TASK_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR=".ralph/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${TASK_NAME}.log"

echo "[$(date -Iseconds)] Starting RALPH loop for $TASK_NAME" | tee -a "$LOG_FILE"

iteration=0
while :; do
  iteration=$((iteration + 1))
  echo "[$(date -Iseconds)] Iteration $iteration" | tee -a "$LOG_FILE"

  # Check for FIX_TASK.md first
  if [ -f "$TASK_DIR/FIX_TASK.md" ]; then
    echo "[$(date -Iseconds)] Found FIX_TASK.md, applying fix" | tee -a "$LOG_FILE"
    claude --print < "$TASK_DIR/FIX_TASK.md" 2>&1 | tee -a "$LOG_FILE" > "$TASK_DIR/RESULTS.md"
    rm "$TASK_DIR/FIX_TASK.md"
    continue
  fi

  # Run main prompt
  claude --print < "$TASK_DIR/PROMPT.md" 2>&1 | tee -a "$LOG_FILE" > "$TASK_DIR/RESULTS.md"

  # Check for completion signal
  if grep -q '<promise>COMPLETE</promise>' "$TASK_DIR/RESULTS.md" 2>/dev/null; then
    echo "[$(date -Iseconds)] Task $TASK_NAME completed!" | tee -a "$LOG_FILE"
    echo '{"status": "complete", "task": "'"$TASK_NAME"'", "completed_at": "'"$(date -Iseconds)"'"}' > "$TASK_DIR/status.json"
    break
  fi

  sleep 2
done

echo "[$(date -Iseconds)] RALPH loop for $TASK_NAME finished" | tee -a "$LOG_FILE"
