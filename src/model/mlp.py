import torch
from torch import nn

class MLP(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim=20, output_dim=2):
        super().__init__()
        self.layers = nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(hidden_dim, output_dim),
                                    torch.nn.Softmax(dim=1))

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)
