"""Plotting helpers used by the early classification notebooks/scripts."""

import matplotlib.pyplot as plt
import torch


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None) -> None:
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend()


def plot_decision_boundary(model, x, y) -> None:
    model.eval()
    device = next(model.parameters()).device
    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    xx, yy = torch.meshgrid(
        torch.linspace(x_min, x_max, 101), torch.linspace(y_min, y_max, 101), indexing="ij"
    )
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(device)
    with torch.inference_mode():
        logits = model(grid)
        predictions = logits.argmax(dim=1) if logits.ndim > 1 else torch.sigmoid(logits).round()
    plt.contourf(xx, yy, predictions.cpu().reshape(xx.shape), cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)

