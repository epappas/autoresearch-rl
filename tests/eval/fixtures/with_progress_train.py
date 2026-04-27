import torch

from autoresearch_rl.target.progress import emit_progress

LEARNING_RATE = 1e-3
EPOCHS = 10


def train() -> None:
    for epoch in range(EPOCHS):
        loss = 1.0 / (epoch + 1)
        emit_progress(step=epoch, step_target=EPOCHS, metrics={"loss": loss})
        print(f"loss={loss:.6f}")


if __name__ == "__main__":
    train()
