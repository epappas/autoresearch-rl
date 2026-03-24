#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run autoresearch-rl run examples/security-judge/config.yaml "$@"
