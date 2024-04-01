from torch import nn
import torch
from torch.nn import functional as F

# TODO: Look at hidden states. Update lstm to not pass tuple. Use DL homework.

class HidsModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=50):
        super().__init__()

        self.l_mods = nn.Sequential(
            nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        )

        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for mod in self.l_mods:
            x, _ = mod(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
