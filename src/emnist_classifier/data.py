"""Dataset and device helpers."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    """Use CUDA when available, otherwise run on CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Download EMNIST ByClass if needed and return train/test data loaders."""
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "torchvision could not be imported. Install a torchvision build compatible "
            "with your installed PyTorch version."
        ) from exc

    transform = transforms.ToTensor()
    train_data = datasets.EMNIST(
        root=data_dir, split="byclass", train=True, download=True, transform=transform
    )
    test_data = datasets.EMNIST(
        root=data_dir, split="byclass", train=False, download=True, transform=transform
    )
    pin_memory = torch.cuda.is_available()
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
    )

