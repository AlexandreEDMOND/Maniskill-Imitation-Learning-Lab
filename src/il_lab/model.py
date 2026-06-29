from __future__ import annotations

import torch
from torch import nn


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, hidden_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = obs_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)
