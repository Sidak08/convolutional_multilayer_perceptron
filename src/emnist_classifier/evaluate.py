"""Command-line evaluation entry point for a saved checkpoint."""

import argparse
from pathlib import Path

import torch

from .data import get_device, make_loaders
from .model import ConvMLPClassifier, ModelConfig
from .train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an EMNIST Conv-MLP checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = ConvMLPClassifier(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, test_loader = make_loaders(args.data_dir, args.batch_size, args.num_workers)
    test_loss, test_accuracy = evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()

