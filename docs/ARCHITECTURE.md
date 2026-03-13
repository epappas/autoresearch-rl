# AutoResearch-RL Architecture (continuous CLI)

## Goal
Provide a continuous RL loop that can run against any training target (local command, Docker wrapper, or remote HTTP endpoint) with keep/discard decisions and versioned artifacts.

## Runtime path
`autoresearch-rl run` → `controller/continuous.py` → `target/*` → `telemetry/*`

## Core modules
- `cli.py`: CLI entrypoint (always-continuous)
- `controller/continuous.py`: loop orchestration + stop guards
- `target/command.py`: local/Docker command targets
- `target/http.py`: remote endpoint targets (vLLM/sglang)
- `policy/search.py`: grid/random/static param proposals
- `telemetry/{events,ledger,manifest}.py`: trace + artifacts

## Keep/discard + versioning
Each iteration produces metrics. If better per objective, it is **kept** and a version record is written to `artifacts/versions/v####/version.json`.

## Stop guards
- `max_wall_time_s`
- `no_improve_limit`
- `failure_rate_limit` + `failure_window`

## Contract/sandbox (legacy)
The earlier contract/sandbox loop remains in the repo but is **not** used by the continuous CLI.
If you want to remove or integrate it, do so explicitly.
