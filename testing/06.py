"""Original EMNIST deep multilayer-perceptron experiment."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class TinyVGG(nn.Module):
    """The legacy name was retained even though this version is an MLP."""

    def __init__(self, hidden_neurons: int = 32, classes: int = 62) -> None:
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden_neurons), nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons), nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons), nn.ReLU(),
            nn.Linear(hidden_neurons, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)


train_data = datasets.EMNIST("data", split="byclass", train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyVGG().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(50):
    for images, labels in train_loader:
        logits = model(images.to(device))
        loss = loss_fn(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Finished epoch {epoch + 1}")

Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/TinyVGG.pth")

