"""
2. architecture.py
"""

import torch
import torch.nn as nn


class PacmanNetwork(nn.Module):
    """
    MLP for predicting Pacman actions from game state
    32 → [128 → 64 → 32] → 5
    H layers: Linear → BatchNorm → GELU → Dropout
    """

    def __init__(
        self,
        input_features=32,
        num_actions=5,
        hidden_dims=[128, 64, 32],
        activation=nn.GELU(),
        dropout=0.3
    ):
        super().__init__()

        layers = []

        # Linear: z = x·W^T + b
        layers.append(nn.Linear(input_features, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))

        # GELU
        layers.append(activation)

        # Dropout: 30%
        layers.append(nn.Dropout(dropout))

        # Hidden layers: 128 → 64 → 32
        for i in range(len(hidden_dims) - 1):

            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dims[2], num_actions))

        self.net = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def cross_entropy_loss(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy loss in training

        Arguments:
            features: 2D matrice (batch_size, 32)
            actions: 1D vector (batch_size) of indexed expert action

        Returns:
            Scalar cross-entropy loss value
        """
        pred_actions = self.forward(features)
        return self.criterion(pred_actions, actions)
