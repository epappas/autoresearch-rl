# AutoResearch-RL

Continuous autoresearch RL runner for long-lived model training loops.

## Goals
- Always-on continuous runs (with safety stop guards)
- Modular targets (local command, Docker/remote via HTTP)
- Keep/discard decisions with versioned artifacts
- Trace + ledger output for auditability

## Layout (core)
- `src/autoresearch_rl/cli.py` – CLI entrypoint
- `src/autoresearch_rl/controller/continuous.py` – continuous loop
- `src/autoresearch_rl/target/` – target adapters (command/http)
- `src/autoresearch_rl/policy/` – parameter search policies
- `src/autoresearch_rl/telemetry/` – trace + ledger

## Install
```bash
uv sync --extra dev
pip install -e .
```

## Quickstart (continuous)
```bash
cp configs/example.yaml configs/local.yaml
autoresearch-rl --config configs/local.yaml
# or
autoresearch-rl run --config configs/local.yaml
```

## Targets
You can point the runner at any training entrypoint via `command` or `http` target adapters.

### Command target (local/Docker wrapper)
```yaml
target:
  type: command
  train_cmd: ["python3", "train.py"]
  eval_cmd: ["python3", "eval.py"]
```

**Parameter injection:** the command target passes params via environment variables:
- `AR_PARAMS_JSON` (full dict)
- `AR_PARAM_<NAME>` (uppercased keys)

### HTTP target (remote/vLLM/sglang)
```yaml
target:
  type: http
  url: "http://localhost:8080/train"
  headers:
    Authorization: "Bearer ..."
```

## Output
Each iteration emits:
- `traces/events.jsonl`
- `artifacts/results.tsv`
- `artifacts/runs/` (stdout/stderr + manifest)
- `artifacts/versions/` (only for `keep` decisions)

## Examples

See `examples/README.md`.

## CLI helpers

```bash
autoresearch-rl validate --config configs/example.yaml
autoresearch-rl print-config --config configs/example.yaml
autoresearch-rl --config configs/example.yaml --override controller.max_wall_time_s=10
```
