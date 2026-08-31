"""Original multiclass blob-classification exercise."""

import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch import nn


class MultiBlobClassification(nn.Module):
    def __init__(self, features: int = 2, hidden: int = 32, classes: int = 5) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


features, labels = make_blobs(n_samples=1_000, n_features=2, centers=5, cluster_std=0.5)
features = torch.from_numpy(features).float()
labels = torch.from_numpy(labels).long()
train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.2)
model = MultiBlobClassification()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
loss_fn = nn.CrossEntropyLoss()

for _ in range(500):
    logits = model(train_x)
    loss = loss_fn(logits, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print((model(test_x).argmax(dim=1) == test_y).float().mean().item())

