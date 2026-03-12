from __future__ import annotations

from autoresearch_rl.eval.metrics import parse_metrics
from autoresearch_rl.sandbox.runner import run_trial


def main() -> None:
    # Placeholder diff to show how a candidate change is carried through the system.
    diff = "diff --git a/examples/minimal-trainable-target/train.py b/examples/minimal-trainable-target/train.py\n+ # candidate change"

    cmd = [
        "python3",
        "examples/minimal-trainable-target/train.py",
        "--learning-rate",
        "0.0026",
        "--use-qk-norm",
        "--grad-clip",
        "0.85",
    ]

    result = run_trial(diff=diff, timeout_s=20, command=cmd)
    parsed = parse_metrics(result.stdout)

    print({
        "status": result.status,
        "elapsed_s": round(result.elapsed_s, 3),
        "loss": parsed.loss,
        "val_bpb": parsed.val_bpb,
    })


if __name__ == "__main__":
    main()
