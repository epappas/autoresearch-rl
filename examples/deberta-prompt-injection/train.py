from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune DeBERTa for prompt-injection classification")
    p.add_argument("--model-name", default="protectai/deberta-v3-base-prompt-injection-v2")
    p.add_argument("--train-file", default="examples/deberta-prompt-injection/data/train.jsonl")
    p.add_argument("--val-file", default="examples/deberta-prompt-injection/data/val.jsonl")
    p.add_argument("--output-dir", default="artifacts/deberta-example")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--use-qk-norm", action="store_true")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if "AR_PARAMS_JSON" in os.environ:
        try:
            params = json.loads(os.environ["AR_PARAMS_JSON"])
            if "learning_rate" in params:
                args.learning_rate = float(params["learning_rate"])
            if "grad_clip" in params:
                args.grad_clip = float(params["grad_clip"])
            if "use_qk_norm" in params:
                args.use_qk_norm = bool(params["use_qk_norm"])
            if "epochs" in params:
                args.epochs = int(params["epochs"])
            if "batch_size" in params:
                args.batch_size = int(params["batch_size"])
            if "weight_decay" in params:
                args.weight_decay = float(params["weight_decay"])
        except json.JSONDecodeError:
            pass
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    ds = ds.map(tok, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1 = f1_score(labels, preds)
        acc = accuracy_score(labels, preds)
        # Keep compatibility with scaffold parser (lower is better)
        val_bpb = 1.0 - f1
        return {"f1": f1, "accuracy": acc, "val_bpb": val_bpb}

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_steps=10,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    f1 = float(metrics.get("eval_f1", 0.0))
    acc = float(metrics.get("eval_accuracy", 0.0))
    val_bpb = float(metrics.get("eval_val_bpb", 1.0))
    loss = float(metrics.get("eval_loss", 1.0))

    print(f"loss={loss:.6f}")
    print(f"val_bpb={val_bpb:.6f}")
    print(f"f1={f1:.6f}")
    print(f"accuracy={acc:.6f}")

    summary = {
        "loss": loss,
        "val_bpb": val_bpb,
        "f1": f1,
        "accuracy": acc,
        "model_name": args.model_name,
    }
    Path(args.output_dir, "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
