from __future__ import annotations

from autoresearch_rl.eval.metrics import parse_metrics
from autoresearch_rl.sandbox.runner import run_trial


def main() -> None:
    diff = (
        "diff --git a/examples/deberta-prompt-injection/train.py "
        "b/examples/deberta-prompt-injection/train.py\n"
        "+ # candidate change"
    )

    cmd = [
        "python3",
        "examples/deberta-prompt-injection/train.py",
        "--epochs",
        "1",
        "--batch-size",
        "4",
    ]

    result = run_trial(diff=diff, timeout_s=600, command=cmd)
    parsed = parse_metrics(result.stdout)

    print(
        {
            "status": result.status,
            "elapsed_s": round(result.elapsed_s, 3),
            "loss": parsed.loss,
            "val_bpb": parsed.val_bpb,
        }
    )


if __name__ == "__main__":
    main()
