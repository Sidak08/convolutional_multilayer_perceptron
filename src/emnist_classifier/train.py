"""Command-line training entry point."""

import argparse
import random
from pathlib import Path

import torch
from torch import nn

from .data import get_device, make_loaders
from .model import ConvMLPClassifier


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute batch accuracy as a percentage."""
    return (logits.argmax(dim=1) == labels).float().mean().item() * 100


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[float, float]:
    """Return mean cross-entropy loss and accuracy for a data loader."""
    criterion = nn.CrossEntropyLoss()
    loss_total = 0.0
    accuracy_total = 0.0
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_total += criterion(logits, labels).item()
            accuracy_total += accuracy(logits, labels)
    return loss_total / len(loader), accuracy_total / len(loader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EMNIST Conv-MLP classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = get_device()
    train_loader, test_loader = make_loaders(args.data_dir, args.batch_size, args.num_workers)
    model = ConvMLPClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = 0.0

    print(f"Training on {device} with {sum(p.numel() for p in model.parameters()):,} parameters")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_accuracy += accuracy(logits, labels)

        test_loss, test_accuracy = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss / len(train_loader):.4f} | "
            f"train accuracy {train_accuracy / len(train_loader):.2f}% | "
            f"test loss {test_loss:.4f} | test accuracy {test_accuracy:.2f}%"
        )
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model.config.to_dict(),
                    "epoch": epoch,
                    "test_accuracy": test_accuracy,
                },
                args.output_dir / "best_model.pt",
            )
    print(f"Best checkpoint: {args.output_dir / 'best_model.pt'} ({best_accuracy:.2f}% test accuracy)")


if __name__ == "__main__":
    main()

