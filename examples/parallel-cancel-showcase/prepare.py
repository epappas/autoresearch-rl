"""Frozen prepare step. Produces the (deterministic) data the trial needs.

In a real workload, this would tokenize, slice train/val, write tensors to
disk, etc. Here it just touches a sentinel file so train.py can confirm
prepare was actually run before each iteration.

Per the autoresearch-rl contract, this file is the trust boundary: the LLM
cannot modify it. Anything that defines what 'correct' means lives here.
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    # prepare.py runs ONCE per campaign in the workdir (no AR_RUN_DIR is
    # set for the prepare step), so writing relative to cwd places the
    # sentinel where every train.py invocation can find it.
    artifacts = Path("data")
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "ready.json").write_text(json.dumps({
        "schema": "v1",
        "n_examples": 1024,
        "metric": "val_loss",
    }))
    print("prepared")


if __name__ == "__main__":
    main()
