#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$DIR/config.yaml"

exec uv run autoresearch-rl --config "$CONFIG" "$@"
