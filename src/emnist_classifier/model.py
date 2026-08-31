"""Neural-network architecture for EMNIST character classification."""

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    """Configuration saved with every model checkpoint."""

    num_classes: int = 62
    feature_channels: int = 64
    hidden_features: int = 128
    dropout: float = 0.25

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class ConvMLPClassifier(nn.Module):
    """A convolutional feature extractor followed by a two-layer MLP head."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        half_channels = self.config.feature_channels // 2

        self.features = nn.Sequential(
            nn.Conv2d(1, half_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(half_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(half_channels, self.config.feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.config.feature_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.config.feature_channels, self.config.hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_features, self.config.num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one unnormalised score (logit) per class for each image."""
        return self.classifier(self.features(images))

