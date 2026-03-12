from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path

import yaml

from autoresearch_rl.eval.metrics import parse_metrics


def _run_one(config: dict, lr: float, wd: float, bs: int, epochs: int) -> dict:
    model = config["model"]["name"]
    train_file = config["model"]["train_file"]
    val_file = config["model"]["val_file"]

    cmd = [
        "python3",
        config["experiment"]["target_script"],
        "--model-name",
        str(model),
        "--train-file",
        str(train_file),
        "--val-file",
        str(val_file),
        "--learning-rate",
        str(lr),
        "--weight-decay",
        str(wd),
        "--batch-size",
        str(bs),
        "--epochs",
        str(epochs),
    ]

    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    parsed = parse_metrics(p.stdout)

    return {
        "command": cmd,
        "returncode": p.returncode,
        "loss": parsed.loss,
        "val_bpb": parsed.val_bpb,
        "stdout": p.stdout,
        "stderr": p.stderr,
        "params": {
            "learning_rate": lr,
            "weight_decay": wd,
            "batch_size": bs,
            "epochs": epochs,
        },
    }


def main(config_path: str = "configs/deberta-example.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    sweep = cfg["benchmark"]["sweep"]
    lrs = sweep.get("learning_rate", [2e-5])
    wds = sweep.get("weight_decay", [0.01])
    bss = sweep.get("batch_size", [4])
    eps = sweep.get("epochs", [1])

    results = []
    for lr, wd, bs, ep in itertools.product(lrs, wds, bss, eps):
        print(f"[benchmark] running lr={lr} wd={wd} bs={bs} epochs={ep}", flush=True)
        result = _run_one(cfg, lr=lr, wd=wd, bs=bs, epochs=ep)
        results.append(result)

    # lower is better
    valid = [r for r in results if r["returncode"] == 0 and r["val_bpb"] is not None]
    best = min(valid, key=lambda x: x["val_bpb"]) if valid else None

    out = {
        "config": config_path,
        "num_runs": len(results),
        "best": best,
        "results": [
            {
                "returncode": r["returncode"],
                "loss": r["loss"],
                "val_bpb": r["val_bpb"],
                "params": r["params"],
            }
            for r in results
        ],
    }

    output_path = Path(cfg["benchmark"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "num_runs": len(results),
        "best_params": best["params"] if best else None,
        "best_val_bpb": best["val_bpb"] if best else None,
        "output": str(output_path),
    }))


if __name__ == "__main__":
    main()
