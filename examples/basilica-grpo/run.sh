#!/usr/bin/env bash
set -euo pipefail

if [ -z "${BASILICA_API_TOKEN:-}" ]; then
    echo "ERROR: BASILICA_API_TOKEN not set"
    echo "  export BASILICA_API_TOKEN='your-token'"
    exit 1
fi

DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$DIR/config.yaml"

exec uv run autoresearch-rl --config "$CONFIG" "$@"
