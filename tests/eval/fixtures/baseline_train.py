import torch

LEARNING_RATE = 1e-3
EPOCHS = 10


def train() -> None:
    for epoch in range(EPOCHS):
        loss = 1.0 / (epoch + 1)
        print(f"loss={loss:.6f}")


if __name__ == "__main__":
    train()
