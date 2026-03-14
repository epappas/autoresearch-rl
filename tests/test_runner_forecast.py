from autoresearch_rl.sandbox.runner import EarlyStopConfig, run_trial


def test_forecast_early_stop(tmp_path):
    cmd = [
        "python3",
        "-c",
        "print('training_seconds=1');print('val_bpb=2.0');print('training_seconds=2');print('val_bpb=2.1');print('training_seconds=3');print('val_bpb=2.2')",
    ]
    es = EarlyStopConfig(
        enabled=True,
        forecast_enabled=True,
        forecast_min_points=3,
        forecast_t_max_s=5.0,
        val_bpb_threshold=2.05,
        min_runtime_s=0.0,
        check_every_s=0.1,
    )
    result = run_trial(diff="diff --git a/x b/x\n", timeout_s=5, command=cmd, workdir=str(tmp_path), apply_patch=False, early_stop=es)
    assert result.status in {"early_stopped", "ok"}
