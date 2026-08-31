"""Original linear-regression learning exercise."""

from pathlib import Path

import torch
from torch import nn


class LinRegModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


weight, bias = 12, 7
inputs = torch.arange(1, 100).unsqueeze(dim=1).float()
labels = inputs * weight + bias
train_x, train_y = inputs[:80], labels[:80]

torch.manual_seed(6996)
model = LinRegModel()
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for _ in range(100):
    prediction = model(train_x)
    loss = loss_fn(prediction, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/model00.pth")

