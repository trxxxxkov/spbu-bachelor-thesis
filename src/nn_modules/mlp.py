"""Implementation of a baseline MLP model that is trained offline to server as a
reference in continual learning experiments."""

import torch
from torch import nn


class BaselineMLP(torch.nn.Module):
    """Multi-Layer Perceptron model used as a reference for performance comparison

    This model is trained offline and serves as a baseline for evaluating the
    performance of proposed models trained in an online learning setting.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 400,
        output_dim: int = 200,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.classifier(x)
