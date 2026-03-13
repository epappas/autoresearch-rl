from pathlib import Path

from autoresearch_rl.telemetry.ledger import append_result_row, ensure_results_tsv


def test_ensure_results_tsv_header(tmp_path: Path):
    p = tmp_path / "results.tsv"
    ensure_results_tsv(str(p))
    txt = p.read_text(encoding="utf-8").splitlines()
    assert txt
    assert txt[0].startswith("commit\tmetric_name\tmetric_value\tmemory_gb\tstatus")


def test_append_result_row(tmp_path: Path):
    p = tmp_path / "results.tsv"
    append_result_row(
        path=str(p),
        commit="abc1234",
        metric_name="val_bpb",
        metric_value=1.23456,
        memory_gb=12.34,
        status="keep",
        description="baseline",
        episode_id="ep1",
        iter_idx=0,
        score=1.11111,
        budget_mode="fixed_wallclock",
        budget_s=300,
        hardware_fingerprint="abcdef0123456789",
        comparable=True,
        non_comparable_reason="",
    )
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "abc1234" in lines[1]
    assert "baseline" in lines[1]
