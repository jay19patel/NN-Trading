import torch
import torch.nn as nn

class SimpleShortNet(nn.Module):
    """
    A simple multi-layer perceptron for binary classification (Sell vs Neutral).
    """
    def __init__(self, input_dim: int):
        super(SimpleShortNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
