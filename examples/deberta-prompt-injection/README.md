# deberta-prompt-injection (continuous CLI)

This example is wired for the continuous CLI. You must install requirements
and point the target commands to your environment.

## Run
```bash
pip install -r examples/deberta-prompt-injection/requirements.txt
uv run autoresearch-rl --config examples/deberta-prompt-injection/example.yaml
```

## Notes
Params are injected via env vars (AR_PARAMS_JSON / AR_PARAM_*). Update
`example.yaml` if your training script expects different flags or envs.
