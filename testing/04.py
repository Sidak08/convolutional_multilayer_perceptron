"""Original FashionMNIST multilayer-perceptron experiment."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class FashionClass(nn.Module):
    def __init__(self, input_size: int = 784, hidden_size: int = 32, output_size: int = 10) -> None:
        super().__init__()
        layers = [nn.Flatten(), nn.Linear(input_size, hidden_size)]
        for _ in range(8):
            layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.Linear(hidden_size, output_size))
        self.layer_stack = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)


train_data = datasets.FashionMNIST("data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST("data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionClass().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    for images, labels in train_loader:
        logits = model(images.to(device))
        loss = loss_fn(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Finished epoch {epoch + 1}")

Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/clothClassification1.pth")

