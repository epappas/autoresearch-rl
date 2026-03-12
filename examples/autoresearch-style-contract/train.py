"""Mutable target file (autoresearch-style contract demo)."""

LEARNING_RATE = 0.0026


def report() -> str:
    # deterministic toy output in expected parser format
    return "loss=2.1700\nval_bpb=1.3000\n"


if __name__ == "__main__":
    print(report(), end="")
