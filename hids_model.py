from torch import nn
import torch
from torch.nn import functional as F

# TODO: Look at hidden states. Update lstm to not pass tuple. Use DL homework.

class HidsModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.mods = nn.Sequential(nn.LSTM(input_dim, hidden_dim),
                                     nn.LSTM(hidden_dim, hidden_dim),
                                     nn.LSTM(hidden_dim, hidden_dim),
                                     nn.LSTM(hidden_dim, hidden_dim),
                                     nn.LSTM(hidden_dim, hidden_dim),
                                     nn.LSTM(hidden_dim, hidden_dim),
                                     nn.LSTM(hidden_dim, hidden_dim),
                                     nn.LSTM(hidden_dim, hidden_dim),
                                     nn.LSTM(hidden_dim, hidden_dim),
                                     nn.Linear(hidden_dim, output_dim),
                                     nn.Sigmoid()
                                     )

    def forward(self, x):
        x = self.mods(x)
        return x
