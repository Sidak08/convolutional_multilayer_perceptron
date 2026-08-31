"""Original binary circle-classification exercise."""

import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn


class ClassifyCircle(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 12), nn.ReLU(), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(dim=1)


features, labels = make_circles(5_000, noise=0.05)
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)
train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClassifyCircle().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
loss_fn = nn.BCEWithLogitsLoss()

for _ in range(1_000):
    logits = model(train_x.to(device))
    loss = loss_fn(logits, train_y.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.inference_mode():
    prediction = torch.sigmoid(model(test_x.to(device))).round()
    accuracy = (prediction == test_y.to(device)).float().mean().item() * 100
print(f"Test accuracy: {accuracy:.2f}%")

