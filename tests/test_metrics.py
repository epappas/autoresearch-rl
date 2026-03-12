from autoresearch_rl.eval.metrics import parse_metrics


def test_parse_metrics():
    p = parse_metrics("x\nloss=2.0\nval_bpb=1.19\n")
    assert p.loss == 2.0
    assert p.val_bpb == 1.19


def test_parse_metrics_uses_last_values():
    p = parse_metrics("loss=3.0\nval_bpb=1.5\nloss=2.1\nval_bpb=1.22\n")
    assert p.loss == 2.1
    assert p.val_bpb == 1.22


def test_parse_metrics_scientific_notation():
    p = parse_metrics("loss=2.1e+00\nval_bpb=1.10e0\n")
    assert p.loss == 2.1
    assert p.val_bpb == 1.1
